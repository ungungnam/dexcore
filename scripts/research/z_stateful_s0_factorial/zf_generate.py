#!/usr/bin/env python
"""Generate the trajectories of one model trained in this study (fixed GT s_0; one deterministic trajectory per example).
    CUDA_VISIBLE_DEVICES=4 python zf_generate.py --dataset taco --model M11 [--part val]
Writes <ds>/preds/<name>_A[_val].npz in the Stage-2 format: pred (B, 1, 64, 512) float16 raw contact (frame 0 = s_0, never generated), example, s0,
z_hat (B, 63, dz) and, for the stateful models, z0_hat (B, dz).  M00 and D0 keep their previous prediction files."""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import zf_common as Z
import zf_infer as I
import zf_models as MD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zf_generate")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", required=True, choices=list(Z.TRAINED_HERE)); ap.add_argument("--part", default="test", choices=["test", "val"])
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    data = I.load_data(ds, dev, a.model); net, ck = MD.load(ds, a.model, data, dev)
    te = data.idx[a.part]; n_all = torch.from_numpy(te).to(dev)
    P = I.generate(net, data, n_all)
    C_hat = P["C_hat"].cpu().numpy(); gt = data.C[n_all].cpu().numpy(); zh = P["z_hat"].cpu().numpy(); zs = data.teacher_std(n_all).cpu().numpy()
    extra = dict(z_hat=zh.astype(np.float32), z_err_rms=float(np.sqrt(((zh - zs) ** 2).mean())))
    if P["z0_hat"] is not None:
        z0 = P["z0_hat"].cpu().numpy(); z0s = data.z_standardise(data.z_star[n_all, 0]).cpu().numpy()
        extra.update(z0_hat=z0.astype(np.float32), z0_err_rms=float(np.sqrt(((z0 - z0s) ** 2).mean())))
    name = Z.run_name(a.model); out = Z.preds_path(ds, name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    err = np.linalg.norm(C_hat[:, 1:] - gt[:, 1:], axis=-1).mean(-1)
    np.savez_compressed(out, pred=C_hat[:, None].astype(np.float16), example=te, s0=gt[:, 0].astype(np.float32), protocol=Z.PROTOCOL, part=a.part, model=ck["model"], seed=ck["seed"], best_step=ck["best_step"],
                        raw_min=float(C_hat.min()), raw_max=float(C_hat.max()), frac_below_0=float((C_hat < 0).mean()), frac_above_1=float((C_hat > 1).mean()), n_nonfinite=int((~np.isfinite(C_hat)).sum()),
                        frame0_equals_s0=bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0), **extra)
    log.info("%s %s %s (step %d): %d examples, E_C(1..63) %.3f%s, raw range [%.3f, %.3f] (%.0f s) -> %s", ds, name, a.part, ck["best_step"], len(te), err.mean(),
             "".join(f", {k} {v:.3f}" for k, v in extra.items() if isinstance(v, float)), C_hat.min(), C_hat.max(), time.time() - t0, out)


if __name__ == "__main__":
    main()
