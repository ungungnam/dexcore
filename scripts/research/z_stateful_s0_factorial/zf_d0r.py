#!/usr/bin/env python
"""Sensitivity row "D0r": the direct model retrained under the 100 k protocol in the previous follow-up, STOPPED at the user's request at about
26-28 k steps (not finished).  Its best validation state so far is evaluated here because, on ARCTIC, it was already below the reused D0's best
validation loss.  It is a partial run and is reported as such; D0 (Stage-2 B0, finished) stays the reference.
    CUDA_VISIBLE_DEVICES=4 python zf_d0r.py --dataset taco        -> <ds>/preds/D0r_partial_A.npz (+ d0r_partial.json); then zf_evaluate.py --name D0r_partial
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import zf_common as Z
import zj_infer as ZI
import zj_models as ZJM
from s2_data import S2Data

ZJ = Z.ZJ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS); a = ap.parse_args(); ds = a.dataset; dev = torch.device("cuda")
    last = ZJ.ckpt_path(ds, ZJ.run_name("M0r"), "last"); st = torch.load(last, map_location="cpu", weights_only=False)
    ref = torch.load(Z.model_ckpt(ds, "M00"), map_location="cpu", weights_only=False)
    data = S2Data(ds, dev, stats=ref["stats"], need_masks=False, teacher=True)
    net = ZJM.build("M0r", data).to(dev); net.load_state_dict(st["best_state"]["sel"]); net.eval()
    te = data.idx["test"]; n_all = torch.from_numpy(te).to(dev)
    C_hat = ZI.generate(net, data, n_all)["C_hat"].cpu().numpy(); gt = data.C[n_all].cpu().numpy()
    out = Z.preds_path(ds, "D0r_partial"); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pred=C_hat[:, None].astype(np.float16), example=te, s0=gt[:, 0].astype(np.float32), protocol=Z.PROTOCOL, part="test", model="M0r (partial)", seed=0, best_step=int(st["best_step"]["sel"]),
                        raw_min=float(C_hat.min()), raw_max=float(C_hat.max()), frac_below_0=float((C_hat < 0).mean()), frac_above_1=float((C_hat > 1).mean()), n_nonfinite=int((~np.isfinite(C_hat)).sum()),
                        frame0_equals_s0=bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0))
    info = dict(dataset=ds, source=str(last), md5=Z.md5(last), validated_steps=int(st["step"]), best_step=int(st["best_step"]["sel"]), best_val_L_C=float(st["best"]["sel"]), schedule_steps=100000,
                status="stopped at the user's request before the run finished (2026-10-04); partial", test_E_C=float(np.linalg.norm(C_hat[:, 1:] - gt[:, 1:], axis=-1).mean()))
    Z.write_json(Z.ds_out(ds) / "d0r_partial.json", info); print(info)


if __name__ == "__main__":
    main()
