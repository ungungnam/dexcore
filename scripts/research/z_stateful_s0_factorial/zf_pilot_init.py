#!/usr/bin/env python
"""Pilot for ONE design decision of the stateful models (run before any model of the study was trained; no test data).
    CUDA_VISIBLE_DEVICES=4 python zf_pilot_init.py --dataset taco --aug 0|3
Question: can the initial-state module I(s_0, G) -> z_hat_0 be learned from the initial frames alone (1 312 / 1 005 training maps), as the
literal L_z0 = ||z_hat_0 - z*_0||^2 implies, or does it need the teacher latents of other training frames as well?
Trains I alone (the module of zf_models, from scratch) for 4 000 steps on  frame 0 of the training sequences  (--aug 0)  or on  frame 0 plus
`aug` random later frames of the same sequences  (--aug 3), and reports the RMSE of z_hat_0 against z*_0 on the VALIDATION sequences' frame 0.
Rule (zf_common.INIT_AUG): the augmentation is adopted iff RMSE(aug 0) > 1.5 x RMSE(aug 3).   Writes OUT/<ds>/pilot_init_aug<k>.json.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

import zf_common as Z
import zf_models as MD
from s2_data import S2Data


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--aug", type=int, required=True); ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device("cuda"); t0 = time.time(); torch.manual_seed(0)
    data = S2Data(ds, dev, need_masks=False, teacher=True)
    net = MD.FactorialModel(True, False, data.d_traj(), data.d_static(), data.d_geo(), data.z_mu, data.z_sd).to(dev)
    params = list(net.init.parameters()); opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    tr = torch.from_numpy(data.idx["train"]).to(dev); va = torch.from_numpy(data.idx["val"]).to(dev)
    gen = torch.Generator(device=dev); gen.manual_seed(0)

    @torch.no_grad()
    def rmse0(idx):
        net.eval(); se = 0.0
        for i in range(0, len(idx), 64):
            n = idx[i:i + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = net.init_state(data.s0_input(data.C[n, 0]), data.geo_tokens(n, torch.zeros_like(n)), data.fd.S[n]).float()
            se += float((z - data.z_standardise(data.z_star[n, 0])).pow(2).sum())
        net.train(); return (se / (len(idx) * net.dz)) ** 0.5
    hist = []
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = 1e-4 * min(1.0, step / 500)
        n = tr[torch.randint(0, len(tr), (128,), device=dev, generator=gen)]
        t = torch.cat([torch.zeros(128, 1, dtype=torch.long, device=dev), torch.randint(1, Z.T, (128, a.aug), device=dev, generator=gen)], 1)      # frame 0 + aug later frames
        nn_ = n[:, None].expand(-1, 1 + a.aug).reshape(-1); tt = t.reshape(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = net.init_state(data.s0_input(data.C[nn_, tt]), data.geo_tokens(nn_, tt), data.fd.S[nn_]).float()
        loss = F.mse_loss(z, data.z_standardise(data.z_star[nn_, tt]))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 500 == 0:
            hist.append(dict(step=step, train_loss=float(loss), val_rmse_z0=rmse0(va), train_rmse_z0=rmse0(tr[:256])))
            print(ds, "aug", a.aug, hist[-1], flush=True)
    best = min(h["val_rmse_z0"] for h in hist)
    Z.write_json(Z.ds_out(ds) / f"pilot_init_aug{a.aug}.json", dict(dataset=ds, aug=a.aug, steps=a.steps, history=hist, best_val_rmse_z0=best, final_val_rmse_z0=hist[-1]["val_rmse_z0"],
                                                                    final_train_rmse_z0=hist[-1]["train_rmse_z0"], seconds=time.time() - t0))


if __name__ == "__main__":
    main()
