#!/usr/bin/env python
"""Generate the trajectories of one model trained in this study under the fixed-s_0 protocol (one deterministic trajectory per example).
    CUDA_VISIBLE_DEVICES=5 python zj_generate.py --dataset taco --model M2 [--part val] [--state state_total] [--tag _4blk]
Writes <ds>/preds/<name>[_seltotal]_A[_val].npz in the Stage-2 format: pred (B, 1, 64, 512) float16 raw contact (frame 0 = s_0, never
generated), example, s0, and for the z models z_hat (B, 63, dz; standardised teacher coordinates).  --state state_total writes the
state selected by the full objective (the Stage-2 B1 criterion) under the name suffix "_seltotal".
The reused models (M0 = B0, M1 = B1) keep their Stage-2 prediction files; they are not regenerated.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import zj_common as Z
import zj_infer as I
import zj_models as MD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_generate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", required=True, choices=list(Z.TRAINED_HERE))
    ap.add_argument("--seed", type=int, default=Z.SEED); ap.add_argument("--tag", default=""); ap.add_argument("--part", default="test", choices=["test", "val"])
    ap.add_argument("--state", default="state", choices=["state", "state_total"])
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 7000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    data = I.load_data(ds, dev, a.model, seed=a.seed, tag=a.tag)
    net, ck = MD.load(ds, a.model, data, dev, seed=a.seed, key=a.state, tag=a.tag)
    te = data.idx[a.part]; n_all = torch.from_numpy(te).to(dev)
    P = I.generate(net, data, n_all)
    C_hat = P["C_hat"].cpu().numpy(); gt = data.C[n_all].cpu().numpy()
    err = np.linalg.norm(C_hat[:, 1:] - gt[:, 1:], axis=-1).mean(-1)
    extra = {}
    if "z_hat" in P:
        zs = data.teacher_std(n_all).cpu().numpy(); zh = P["z_hat"].cpu().numpy()
        extra.update(z_err_rms=float(np.sqrt(((zh - zs) ** 2).mean())), z_hat=zh.astype(np.float32))
    name = Z.run_name(a.model, a.seed, a.tag) + ("_seltotal" if a.state == "state_total" else "")
    out = Z.preds_path(ds, name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    step = ck["best_step"] if a.state == "state" else ck["best_step_total"]
    np.savez_compressed(out, pred=C_hat[:, None].astype(np.float16), example=te, s0=gt[:, 0].astype(np.float32), protocol=Z.PROTOCOL, part=a.part, model=ck["model"], seed=ck["seed"], best_step=step,
                        state=a.state, raw_min=float(C_hat.min()), raw_max=float(C_hat.max()), frac_below_0=float((C_hat < 0).mean()), frac_above_1=float((C_hat > 1).mean()),
                        n_nonfinite=int((~np.isfinite(C_hat)).sum()), frame0_equals_s0=bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0), **extra)
    log.info("%s %s %s (step %d): %d examples, E_C(1..63) %.3f%s, raw range [%.3f, %.3f], frame0 == s0 %s (%.0f s) -> %s", ds, name, a.part, step, len(te), err.mean(),
             "".join(f", {k} {v:.3f}" for k, v in extra.items() if isinstance(v, float)), C_hat.min(), C_hat.max(), bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0), time.time() - t0, out)


if __name__ == "__main__":
    main()
