#!/usr/bin/env python
"""Run the whole matrix of one dataset on one GPU, phase by phase, skipping finished steps.
    python pipeline.py --dataset taco --gpu 4 --parallel 6
Phase 1  D0 (3 seeds), D1 seed 0 at lambda_struct in {0.3, 1, 3} (sweep), S0 (3 seeds); references GT / PERSIST
         -> select lambda_struct on the validation set (D1 sweep)
Phase 2  D2 seed 0 at lambda_aux in {0.3, 1} (sweep), D1 seeds 1-2, S1 (3 seeds)
         -> select lambda_aux (D2 sweep)
Phase 3  D2 seeds 1-2, S2 (3 seeds)
Every run: train -> generate (A, B on test; A on val for the sweep runs) -> evaluate. The seed-0 D1 / D2 at the chosen lambdas
are the sweep runs themselves (symlinked to the final names). Commands are logged to <OUT>/logs/commands_<ds>.txt.
Several instances (different GPUs) can run the same dataset at once: every step is guarded by a lock file next to its output
(<output>.lock holding the owner pid), so each run is trained / generated / evaluated exactly once and the instances wait for
each other where needed; CUDA out-of-memory failures are retried.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import sat_common as S

PY = "/home/uhnam/miniconda3/envs/dexmachina/bin/python"
HERE = Path(__file__).resolve().parent
LOG = S.OUT / "logs"


def _alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def acquire(lock, gpu):
    """Atomic lock creation; several pipeline instances (different GPUs) share the work. A lock whose owner is dead is stale."""
    lock = Path(lock)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, f"{os.getpid()} {gpu} {time.strftime('%H:%M:%S')}".encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(lock.read_text().split()[0])
        except Exception:  # noqa: BLE001
            return False
        if not _alive(pid):
            lock.unlink(missing_ok=True); return acquire(lock, gpu)
        return False


def wait_lock(lock, done):
    """Wait until the lock disappears, its owner dies or the output exists."""
    lock = Path(lock)
    while lock.exists():
        try:
            pid = int(lock.read_text().split()[0])
        except Exception:  # noqa: BLE001
            pid = -1
        if done() or (pid > 0 and not _alive(pid)) or pid == -1:
            lock.unlink(missing_ok=True); return
        time.sleep(30)


DEFER = -2
MIN_FREE_MIB = 5000


def gpu_free_mib(gpu):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", str(gpu)], capture_output=True, text=True, timeout=30).stdout
        used, total = (int(x) for x in out.strip().split(","))
        return total - used
    except Exception:  # noqa: BLE001
        return 10 ** 6


def locked_step(lock, done, fn, gpu):
    """Run fn() under the lock unless the output exists. If another live instance holds the lock, return DEFER at once (the
    phase loop retries later); before launching, wait until the instance's GPU has MIN_FREE_MIB free (several instances and
    other users share the GPUs)."""
    if done():
        return 0
    if gpu_free_mib(gpu) < MIN_FREE_MIB:            # no room on this GPU right now: leave the step to another instance / later
        return DEFER
    if not acquire(lock, gpu):
        return DEFER
    try:
        if done():
            return 0
        return fn()
    finally:
        Path(lock).unlink(missing_ok=True)


def sh(cmd, log, env, retries=3):
    """Run one step; a CUDA out-of-memory failure (several runs share the GPU) is retried after a pause, up to `retries` times."""
    for attempt in range(retries + 1):
        with open(log, "a") as f:
            f.write(f"\n$ {cmd}   (attempt {attempt + 1})\n"); f.flush()
            r = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=HERE)
        if r.returncode == 0:
            return 0
        tail = Path(log).read_text().rsplit("(attempt ", 1)[-1]            # this attempt's output only
        if "out of memory" in tail or "CUDA error" in tail or "CUBLAS_STATUS_ALLOC_FAILED" in tail:
            with open(log, "a") as f:
                f.write(f"\n[pipeline] OOM / CUDA failure, retry {attempt + 1} of {retries} after 120 s\n")
            time.sleep(120)
            continue
        return r.returncode
    return r.returncode


class Runner:
    def __init__(self, ds, gpu, parallel):
        self.ds, self.gpu, self.parallel = ds, gpu, parallel
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(S.REPO))
        LOG.mkdir(parents=True, exist_ok=True)
        self.cmdlog = LOG / f"commands_{ds}.txt"

    def record(self, cmd):
        with open(self.cmdlog, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] CUDA_VISIBLE_DEVICES={self.gpu} {cmd}\n")

    def run_model(self, model, seed, lam_s=None, lam_a=None, sweep=False):
        name = S.run_name(model, seed, lam_s if (sweep and S.MODELS[model]["struct"]) else None, lam_a if (sweep and S.MODELS[model]["aux"]) else None)
        log = LOG / f"{self.ds}_{name}.log"
        ck = S.ckpt_path(self.ds, name); ck.parent.mkdir(parents=True, exist_ok=True)
        cmd = f"{PY} train.py --dataset {self.ds} --model {model} --seed {seed}" + (f" --lambda-struct {lam_s}" if lam_s is not None else "") + \
              (f" --lambda-aux {lam_a}" if lam_a is not None else "") + (" --sweep" if sweep else "")
        rc = locked_step(str(ck) + ".lock", ck.exists, lambda: (self.record(cmd), sh(cmd, log, self.env))[1], self.gpu)
        if rc == DEFER:
            return f"deferred {name}"
        if rc != 0:
            return f"FAILED train {name}"
        for prot, part in (("A", "test"), ("B", "test")) + ((("A", "val"),) if sweep else ()):
            suffix = prot + ("" if part == "test" else "_val")
            pp, mp = S.preds_path(self.ds, name, suffix), S.metrics_path(self.ds, name, suffix); pp.parent.mkdir(parents=True, exist_ok=True); mp.parent.mkdir(parents=True, exist_ok=True)
            cmd = f"{PY} generate.py --dataset {self.ds} --name {name} --protocol {prot} --part {part}"
            rc = locked_step(str(pp) + ".lock", pp.exists, lambda: (self.record(cmd), sh(cmd, log, self.env))[1], self.gpu)
            if rc == DEFER:
                return f"deferred {name}"
            if rc != 0:
                return f"FAILED generate {name} {suffix}"
            cmd = f"{PY} evaluate.py --dataset {self.ds} --name {name} --protocol {prot} --part {part}"
            rc = locked_step(str(mp) + ".lock", mp.exists, lambda: (self.record(cmd), sh(cmd, log, self.env))[1], self.gpu)
            if rc == DEFER:
                return f"deferred {name}"
            if rc != 0:
                return f"FAILED evaluate {name} {suffix}"
        return f"ok {name}"

    def run_reference(self, name):
        log = LOG / f"{self.ds}_{name}.log"
        for prot, part in (("A", "test"), ("B", "test"), ("A", "val")):
            if name == "GT" and prot == "B":
                continue
            suffix = prot + ("" if part == "test" else "_val")
            mp = S.metrics_path(self.ds, name, suffix); mp.parent.mkdir(parents=True, exist_ok=True)
            cmd = f"{PY} evaluate.py --dataset {self.ds} --name {name} --protocol {prot} --part {part}"
            rc = locked_step(str(mp) + ".lock", mp.exists, lambda: (self.record(cmd), sh(cmd, log, self.env))[1], self.gpu)
            if rc == DEFER:
                return f"deferred {name}"
            if rc != 0:
                return f"FAILED evaluate {name} {suffix}"
        return f"ok {name}"

    def phase(self, jobs):
        """Run the jobs with `parallel` workers; jobs deferred by another instance's lock are retried every minute until done."""
        pending, done = list(jobs), []
        while pending:
            with ThreadPoolExecutor(self.parallel) as ex:
                res = list(ex.map(lambda j: j(), pending))
            done += [r for r in res if not r.startswith("deferred")]
            pending = [j for j, r in zip(pending, res) if r.startswith("deferred")]
            if pending:
                time.sleep(60)
        bad = [r for r in done if r.startswith("FAILED")]
        print(f"[{time.strftime('%H:%M:%S')}] {self.ds} (gpu {self.gpu}): {done}", flush=True)
        if bad:
            raise SystemExit(f"{self.ds}: {bad}")

    def select(self, stage):
        cmd = f"{PY} select_lambda.py --dataset {self.ds} --stage {stage}"; self.record(cmd)
        if sh(cmd, LOG / f"{self.ds}_select_{stage}.log", self.env) != 0:
            raise SystemExit(f"{self.ds}: selection {stage} failed")
        return S.chosen_lambdas(self.ds)

    def link_final(self, model, seed, lam_s, lam_a):
        src = S.run_name(model, seed, lam_s, lam_a if S.MODELS[model]["aux"] else None); dst = S.run_name(model, seed)
        for a, b in ((S.ckpt_path(self.ds, src), S.ckpt_path(self.ds, dst)),) + tuple((S.preds_path(self.ds, src, p), S.preds_path(self.ds, dst, p)) for p in ("A", "B")) + \
                tuple((S.metrics_path(self.ds, src, p), S.metrics_path(self.ds, dst, p)) for p in ("A", "B")) + \
                tuple((S.metrics_path(self.ds, src, p).with_name(S.metrics_path(self.ds, src, p).stem + "_events.csv"), S.metrics_path(self.ds, dst, p).with_name(S.metrics_path(self.ds, dst, p).stem + "_events.csv")) for p in ("A", "B")):
            if a.exists() and not b.exists():
                try:
                    b.symlink_to(a.name)
                except FileExistsError:                       # another instance linked it first
                    pass


def main():
    global MIN_FREE_MIB
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--gpu", required=True); ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--phases", default="123"); ap.add_argument("--min-free", type=int, default=MIN_FREE_MIB, help="MiB that must be free on the GPU before a step starts")
    a = ap.parse_args()
    MIN_FREE_MIB = a.min_free
    R = Runner(a.dataset, a.gpu, a.parallel)
    if "1" in a.phases:
        jobs = [lambda n=n: R.run_reference(n) for n in ("PERSIST", "GT")]
        jobs += [lambda s=s: R.run_model("D0", s) for s in S.SEEDS]
        jobs += [lambda l=l: R.run_model("D1", 0, lam_s=l, sweep=True) for l in S.LAMBDA_STRUCT_GRID]
        jobs += [lambda s=s: R.run_model("S0", s) for s in S.SEEDS]
        R.phase(jobs)
        ch = R.select("struct"); R.link_final("D1", 0, ch["lambda_struct"], None)
    ch = S.chosen_lambdas(a.dataset)
    if "2" in a.phases:
        ls = ch["lambda_struct"]
        jobs = [lambda l=l: R.run_model("D2", 0, lam_s=ls, lam_a=l, sweep=True) for l in S.LAMBDA_AUX_GRID]
        jobs += [lambda s=s: R.run_model("D1", s, lam_s=ls) for s in S.SEEDS[1:]]
        jobs += [lambda s=s: R.run_model("S1", s, lam_s=ls) for s in S.SEEDS]
        R.phase(jobs)
        ch = R.select("aux"); R.link_final("D2", 0, ch["lambda_struct"], ch["lambda_aux"])
    if "3" in a.phases:
        ls, la = ch["lambda_struct"], ch["lambda_aux"]
        jobs = [lambda s=s: R.run_model("D2", s, lam_s=ls, lam_a=la) for s in S.SEEDS[1:]]
        jobs += [lambda s=s: R.run_model("S2", s, lam_s=ls, lam_a=la) for s in S.SEEDS]
        R.phase(jobs)
    print(f"[{time.strftime('%H:%M:%S')}] {a.dataset}: pipeline finished", flush=True)


if __name__ == "__main__":
    main()
