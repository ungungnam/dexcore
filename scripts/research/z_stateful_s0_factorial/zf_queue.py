#!/usr/bin/env python
"""GPU-aware training queue for shared GPUs (the previous follow-up's queue, with a preferred GPU set).    nohup setsid python zf_queue.py > OUT/logs/queue_stdout.log 2>&1 &
Reads OUT/logs/queue.txt (one job per line: "<dataset> <model> [seed] [tag|-] [extra zf_train.py args]"; lines can be appended while the
queue runs) and starts each pending job, in file order, as soon as an allowed GPU has enough FREE memory for it:
  * GPUs in PRIMARY (4, 5: the user's instruction) are used first; a GPU in FALLBACK (free space next to other users' jobs) only when no primary GPU
    can take the job; 0 / 1 are never used; other users' processes are never touched;
  * a job is started only if  free memory - (headroom my running jobs on that GPU may still claim) >= cap + context + MARGIN;  the cap is
    enforced inside the process (--mem-cap-mib), so a job fails with its own OOM instead of squeezing a neighbour; an OOM / CUDA failure
    puts the job back in the queue with --resume;
  * among the GPUs that fit, the one with the most spare compute per job, (100 - other users' utilisation) / (my jobs there + 1), the
    utilisation being a running average of the readings taken while none of these jobs is on that GPU;
  * at most MAX_PER_GPU of these jobs per GPU and MAX_TOTAL at once; one scheduler instance (lock file); a run name is never started twice;
  * every GPU is watched: when one has been completely empty for STABLE seconds and one of these jobs shares its GPU (with another of these
    jobs or with another user's process), that job is moved onto it right after its next checkpoint (stop, resume).
State: OUT/logs/queue_status.json (adopted on restart: a job whose recorded pid is still a zf_train.py process keeps running).
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import zf_common as Z

PRIMARY = (4, 5)
FALLBACK = (7, 6, 0, 1, 3)                    # GPU 2 is left to other users (user's instruction 10-04 16:35: "2번은 다른 사람 양보"); 0-3 only once empty
STABLE = 60                                   # s a GPU must have been empty before it is used (user's instruction 10-04: "empty for 1 minute or more"; was 600)
ALLOWED = PRIMARY + FALLBACK
MAX_PER_GPU = 2                               # whole-sequence runs per GPU (see the placement rule below)
MAX_PER_FALLBACK = 2
MAX_TOTAL = 6                                 # of these jobs at once over all GPUs
CTX = 500                                     # MiB CUDA context per process on top of the allocator cap
MARGIN = 1000                                 # MiB left free beyond everything my jobs can claim
CAP = {"M01": 9500, "M10": 34000, "M11": 34000}   # MiB allocator cap per process (smoke runs: 8.4 GiB reserved for M01; 28.9 GiB for a stateful model without decoder checkpointing)
EXTRA = {"M10": ["--no-dec-ckpt"], "M11": ["--no-dec-ckpt"]}   # a stateful run has a GPU to itself, so it can trade memory for speed (1.33 -> 1.20 s/step; same function)
STATEFUL = lambda j: Z.MODELS[j["model"]]["stateful"]
# Placement rule (measured 10-04): the 63-step rollout is thousands of tiny GPU operations; next to ANY other process on the same GPU it is time-sliced
# to ~6 s/step instead of 1.35.  So a stateful run gets a GPU to itself; whole-sequence runs (few large operations) may share a GPU two by two.
POLL, SETTLE, COOLDOWN = 30, 120, 300         # s between polls / after a launch before the next placement / after a failed attempt
OOM_MARKS = ("OutOfMemoryError", "CUDA error", "CUBLAS_STATUS", "out of memory")
LOGS = Z.OUT / "logs"
QUEUE, STATUS, QLOG, LOCK = LOGS / "queue.txt", LOGS / "queue_status.json", LOGS / "queue.log", LOGS / "queue.lock"
PY = sys.executable
HERE = Path(__file__).resolve().parent


def say(msg):
    with open(QLOG, "a") as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} {msg}\n")


def smi(args):
    return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()


def gpus():
    """{index: (free MiB, utilisation %)}: minimum free memory and mean utilisation of three readings; unreadable lines are skipped."""
    acc = {}
    for _ in range(3):
        for line in smi(["--query-gpu=index,memory.free,utilization.gpu"]):
            try:
                i, free, util = (int(x) for x in line.split(","))
            except ValueError:
                continue
            acc.setdefault(i, []).append((free, util))
        time.sleep(1)
    return {i: (min(f for f, _ in v), sum(u for _, u in v) / len(v)) for i, v in acc.items() if len(v) == 3}


def used_by_pid():
    out = {}
    for line in smi(["--query-compute-apps=pid,used_memory"]):
        try:
            pid, used = (int(x) for x in line.split(","))
        except ValueError:
            continue
        out[pid] = used
    return out


def foreign_gpus():
    """Indices of the GPUs that hold a compute process of ANOTHER user (a pid whose /proc entry is not owned by this user)."""
    idx = {}
    for line in smi(["--query-gpu=index,uuid"]):
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0].isdigit():
            idx[p[1]] = int(p[0])
    out = set()
    for line in smi(["--query-compute-apps=gpu_uuid,pid"]):
        p = [x.strip() for x in line.split(",")]
        if len(p) != 2 or p[0] not in idx or not p[1].isdigit():
            continue
        try:
            mine = os.stat(f"/proc/{p[1]}").st_uid == os.getuid()
        except OSError:
            mine = False
        if not mine:
            out.add(idx[p[0]])
    return out


def read_jobs():
    jobs = []
    for line in QUEUE.read_text().splitlines():
        p = line.split("#")[0].split()
        if len(p) < 2 or p[0] not in Z.DATASETS or p[1] not in CAP:
            continue
        try:
            seed = int(p[2]) if len(p) > 2 else Z.SEED
        except ValueError:
            continue
        tag = p[3] if len(p) > 3 and p[3] != "-" else ""
        jobs.append(dict(key=f"{p[0]}/{Z.run_name(p[1], seed, tag)}", ds=p[0], model=p[1], seed=seed, tag=tag, extra=p[4:]))
    return jobs


def trainer_pids():
    """{pid: cmdline} of every zf_train.py process of this user."""
    out = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                c = (d / "cmdline").read_text().split("\0")
            except OSError:
                continue
            if any(x.endswith("zf_train.py") for x in c):
                out[int(d.name)] = c
    return out


def same_run_alive(j):
    for c in trainer_pids().values():
        arg = lambda k, default="": c[c.index(k) + 1] if k in c else default
        if arg("--dataset") == j["ds"] and arg("--model") == j["model"] and int(arg("--seed", str(Z.SEED))) == j["seed"] and arg("--tag") == j["tag"]:
            return True
    return False


def attempt_text(log_path, offset):
    with open(log_path, "rb") as f:
        f.seek(offset)
        return f.read().decode("utf-8", "replace")


def loop(st, procs, state, env_base):
    jobs = read_jobs()
    for j in jobs:
        st.setdefault(j["key"], dict(state="pending", attempts=0, fails=0, gpu=None, pid=None, not_before=0.0))
    # ---- running jobs: finished?
    for j in jobs:
        s = st[j["key"]]
        if s["state"] != "running":
            continue
        p = procs.get(j["key"])
        rc = p.poll() if p is not None else (None if s["pid"] in trainer_pids() else -999)
        if rc is None:
            continue
        name = Z.run_name(j["model"], j["seed"], j["tag"]); txt = attempt_text(LOGS / f"train_{j['ds']}_{name}.log", s.get("log_offset", 0))
        if f"done {name}:" in txt:
            s.update(state="done", finished=time.strftime("%m-%d %H:%M:%S")); say(f"DONE {j['key']} (gpu {s['gpu']}, attempt {s['attempts']})")
        elif any(m in txt for m in OOM_MARKS):
            s.update(state="pending", not_before=time.time() + COOLDOWN); say(f"OOM / CUDA failure {j['key']} on gpu {s['gpu']} (attempt {s['attempts']}): back in the queue with --resume")
        else:
            s["fails"] += 1
            if s["fails"] >= 2:
                s["state"] = "failed"; say(f"FAILED {j['key']} (exit {rc}); giving up after {s['fails']} non-OOM failures")
            else:
                s.update(state="pending", not_before=time.time() + COOLDOWN); say(f"exit {rc} of {j['key']} without OOM marker; one retry with --resume")
        procs.pop(j["key"], None)
    # ---- place the first pending job (file order)
    running = [j for j in jobs if st[j["key"]]["state"] == "running"]
    mine = {}
    for j in running:
        mine.setdefault(st[j["key"]]["gpu"], []).append(j)
    G = gpus()
    foreign = foreign_gpus(); now = time.time()
    state["free_since"] = {g: state["free_since"].get(g, now) for g in ALLOWED if g in G and g not in foreign}
    for g in ALLOWED:                                                       # other users' utilisation: running average while none of these jobs is on the GPU
        if g in G and not mine.get(g):
            state["util"][g] = G[g][1] if g not in state["util"] else 0.8 * state["util"][g] + 0.2 * G[g][1]
        elif g in G and g not in foreign:
            state["util"][g] = 0.0                                          # only my own processes are on this GPU
        elif g in G and g not in state["util"]:
            state["util"][g] = 100.0                                        # shared with another user and never read alone: treat as busy
    pending = [j for j in jobs if st[j["key"]]["state"] == "pending" and time.time() >= st[j["key"]]["not_before"]]
    if pending and time.time() - state["last_launch"] >= SETTLE and len(running) < MAX_TOTAL:
        j = pending[0]; s = st[j["key"]]; need = CAP[j["model"]] + CTX + MARGIN
        used = used_by_pid()
        headroom = {g: sum(max(0, CAP[x["model"]] + CTX - used.get(st[x["key"]]["pid"], 0)) for x in js) for g, js in mine.items()}   # what my running jobs may still claim
        fit = lambda g: G[g][0] - headroom.get(g, 0)
        def ok(g):
            js = mine.get(g, [])
            if g not in G or fit(g) < need or (g not in PRIMARY and (g in foreign or now - state["free_since"].get(g, now) < STABLE)):
                return False                                                  # a non-primary GPU only after it has been free of other users for STABLE seconds
            if STATEFUL(j):
                return len(js) == 0 and (g not in foreign)                    # a GPU to itself
            return len(js) < 2 and not any(STATEFUL(x) for x in js)
        # the GPU with the most spare compute per job; ties go to 4 / 5.  A GPU another user is working on scores low through its utilisation.
        cand = [(-(100 - state["util"].get(g, 100.0)) / (len(mine.get(g, [])) + 1), 0 if g in PRIMARY else 1, -fit(g), g) for g in ALLOWED if ok(g)]
        if same_run_alive(j):
            s["not_before"] = time.time() + 20                                # its previous process (e.g. one being moved) has not exited yet: try again shortly
        elif cand:
            g = sorted(cand)[0][3]; name = Z.run_name(j["model"], j["seed"], j["tag"]); log_path = LOGS / f"train_{j['ds']}_{name}.log"
            cmd = [PY, str(HERE / "zf_train.py"), "--dataset", j["ds"], "--model", j["model"], "--seed", str(j["seed"]), "--wandb", "--mem-cap-mib", str(CAP[j["model"]])]
            cmd += (["--tag", j["tag"]] if j["tag"] else []) + EXTRA.get(j["model"], []) + list(j["extra"]) + (["--resume"] if s["attempts"] > 0 else [])
            with open(log_path, "a") as f:
                f.write(f"=== attempt {s['attempts'] + 1} on gpu {g} ({time.strftime('%m-%d %H:%M:%S')}, free {G[g][0]} MiB, other users' utilisation {state['util'].get(g, float('nan')):.0f} %)\n")
                f.flush(); off = f.tell()
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=dict(env_base, CUDA_VISIBLE_DEVICES=str(g)), cwd=str(HERE), start_new_session=True)
            procs[j["key"]] = p; state["last_launch"] = time.time()
            s.update(state="running", attempts=s["attempts"] + 1, gpu=g, pid=p.pid, log_offset=off, started=time.strftime("%m-%d %H:%M:%S"))
            STATUS.write_text(json.dumps(st, indent=1))
            say(f"START {j['key']} on gpu {g} (pid {p.pid}, attempt {s['attempts']}, free {G[g][0]} MiB, claimable by my running jobs {headroom.get(g, 0)}, need {need}, others' util {state['util'].get(g, float('nan')):.0f} %)")
        elif time.time() - state["last_wait_note"] >= 1800:
            state["last_wait_note"] = time.time()
            say(f"waiting: {j['key']} needs {need} MiB; usable now {{{', '.join(f'{g}: {fit(g)}' for g in ALLOWED if g in G)}}} (running {len(running)}/{MAX_TOTAL})")
    # ---- rebalance: when a GPU has been completely empty for STABLE seconds, move one job from a GPU that holds two of mine onto it,
    #      right after that job's next checkpoint (so that at most one poll interval of training is lost)
    now = time.time(); need_max = max(CAP.values()) + CTX + MARGIN
    empty = [g for g in ALLOWED if g in G and not mine.get(g) and g not in foreign and G[g][0] >= need_max]
    state["empty_since"] = {g: state["empty_since"].get(g, now) for g in empty}
    stable = [g for g in empty if now - state["empty_since"][g] >= STABLE]
    mig = state.get("migrate")
    if mig is None and stable and now - state["last_launch"] >= SETTLE and not any(st[x["key"]]["state"] == "pending" for x in jobs):   # not in the poll of a launch: that job is not on its GPU yet
        # candidates: my jobs that share their GPU, with another of my jobs or with another user's process (the M01 pair next to a neighbour's server
        # ran at 2-3 s/step instead of 0.7).  The job that gains most first: stateful before whole-sequence, then the most crowded GPU.
        cands = [(not Z.MODELS[x["model"]]["stateful"], -len(js) - (g in foreign), x["key"], g, x) for g, js in mine.items() for x in js if len(js) >= 2 or g in foreign]
        if cands:
            _, _, _, src, j = sorted(cands, key=lambda c: c[:3])[0]; tgt = sorted(stable, key=lambda g: (g not in PRIMARY, g))[0]
            lp = Z.ckpt_path(j["ds"], Z.run_name(j["model"], j["seed"], j["tag"]), "last")
            state["migrate"] = dict(key=j["key"], target=tgt, mtime=lp.stat().st_mtime if lp.exists() else 0.0, path=str(lp))
            say(f"MOVE planned: {j['key']} from gpu {src} to the empty gpu {tgt} after its next checkpoint")
    elif mig is not None:
        s = st.get(mig["key"]); lp = Path(mig["path"])
        if s is None or s["state"] != "running" or mig["target"] not in stable:
            say(f"MOVE cancelled: {mig['key']} (gpu {mig['target']} is no longer empty, or the job ended)"); state["migrate"] = None
        elif lp.exists() and lp.stat().st_mtime > mig["mtime"]:
            try:
                os.kill(s["pid"], 15)
            except OSError:
                pass
            procs.pop(mig["key"], None)
            s.update(state="pending", not_before=now + 30, gpu=None)
            with open(LOGS / f"train_{mig['key'].replace('/', '_')}.log", "a") as f:
                f.write(f"=== attempt above stopped by the queue ({time.strftime('%m-%d %H:%M:%S')}) right after a checkpoint, to move the run to the empty gpu {mig['target']}; it resumes from that checkpoint\n")
            say(f"MOVE {mig['key']}: stopped after its checkpoint, will resume on gpu {mig['target']}"); state["migrate"] = None; state["last_launch"] = 0.0
    STATUS.write_text(json.dumps(st, indent=1))
    return bool(jobs) and all(st[j["key"]]["state"] in ("done", "failed") for j in jobs)


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    lock = open(LOCK, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another zf_queue.py holds the lock; exiting"); return
    st = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    procs, state = {}, dict(util={}, last_launch=0.0, last_wait_note=0.0, empty_since={}, free_since={}, migrate=None)
    env_base = dict(os.environ, PYTHONPATH="/home/uhnam/workspace/dexcore")
    key_file = Path.home() / "api_keys" / "wandb_key.txt"
    if key_file.exists():
        env_base["WANDB_API_KEY"] = key_file.read_text().strip()         # passed through the environment only; never logged
    env_base["WANDB_DIR"] = str(Z.OUT / "wandb"); (Z.OUT / "wandb").mkdir(parents=True, exist_ok=True)
    say(f"queue started (primary GPUs {PRIMARY}, fallback {FALLBACK}, caps {CAP}, context {CTX}, margin {MARGIN} MiB)")
    while True:
        try:
            all_done = loop(st, procs, state, env_base)
            if all_done and not (LOGS / "queue.keep").exists():
                say("QUEUE_EMPTY: all listed jobs finished; exiting"); return
        except Exception:                                                  # noqa: BLE001  (a bad nvidia-smi reading must not stop the queue)
            say("scheduler error (continuing): " + traceback.format_exc().strip().replace("\n", " | "))
        time.sleep(POLL)


if __name__ == "__main__":
    main()
