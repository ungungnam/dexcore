#!/usr/bin/env python
"""Bottleneck inspection of the latent-mediated models (analysis only; the teacher is used for analysis, never as a model input at inference).
    CUDA_VISIBLE_DEVICES=5 python s2_diag_bottleneck.py --dataset taco
For B1 / B2 on the test sequences:
  E_C_pred        dense error of the generated trajectory (predicted z_hat [and r_hat])                           = the reported E_C
  E_C_oracle_z    the same fine-tuned decoder(s) fed with the TEACHER z*_t of the GT frame (B2: with its predicted r_hat): what the z
                  bottleneck alone costs when the latent trajectory is perfect
  E_C_recon_s0    ||D_z(z*_0) - s_0||: how much of the initial map's dense detail the z path cannot carry (frame 0, never generated)
  Stage-1 references on the same frames: the A3 autoencoder's own z-only and z + r reconstruction errors (GT encoders, Stage-1 decoders).
Writes <ds>/bottleneck.csv (per model, with horizon bins) and <ds>/bottleneck_curves.npz; python s2_diag_bottleneck.py --merge -> OUT/bottleneck_metrics.csv.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import s2_common as S
from s2_generate import load_data, load_model


@torch.no_grad()
def oracle_decode(net, data, n_all, bs=16):
    errs, e0 = [], []
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; B = len(n)
        s0 = data.C[n, 0]; Sg = data.fd.S[n]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            r = None
            if net.use_r:
                r = net.latents(data.s0_input(s0), data.tau(n), Sg)["r"].float().reshape(B * S.TF, -1)
            t = torch.arange(1, S.T, device=data.device)[None].expand(B, -1)
            z_star = data.teacher_std(n).reshape(B * S.TF, -1)
            geo = data.geo_tokens(n.repeat_interleave(S.TF), t.reshape(-1))
            C_bar, delta = net.decode_maps(z_star, r, geo, Sg.repeat_interleave(S.TF, 0))
            C = data.std_to_raw((C_bar if delta is None else C_bar + delta).float()).view(B, S.TF, 512)
            # frame 0: the teacher latent of s_0 through the z path only
            z0 = data.z_standardise(data.z_star[n, 0]); geo0 = data.geo_tokens(n, torch.zeros_like(n))
            C0 = data.std_to_raw(net._decoder(net.D_z, geo0, torch.cat([Sg, z0 * net.z_sd + net.z_mu], -1)).float())
        errs.append((C - data.C[n, 1:]).norm(dim=-1).cpu().numpy()); e0.append((C0 - s0).norm(dim=-1).cpu().numpy())
    return np.concatenate(errs), np.concatenate(e0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=S.DATASETS); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    if a.merge:
        pd.concat([pd.read_csv(S.ds_out(ds) / "bottleneck.csv") for ds in S.DATASETS if (S.ds_out(ds) / "bottleneck.csv").exists()], ignore_index=True).to_csv(S.OUT / "bottleneck_metrics.csv", index=False); return
    ds = a.dataset; dev = torch.device("cuda"); rows, curves = [], {}
    s1 = np.load(S.CF.latents_path(ds, S.TEACHER_NAME, "test"))
    for m in ("B1", "B2"):
        name = S.run_name(m); data = load_data(ds, dev, name); net, ck = load_model(ds, name, data, dev)
        te = data.idx["test"]; n_all = torch.from_numpy(te).to(dev); take = data.take[te]
        assert np.array_equal(s1["n"], te)
        pred = np.load(S.preds_path(ds, name))["pred"][:, 0].astype(np.float32); gt = data.C[n_all].cpu().numpy()
        e_pred = np.linalg.norm(pred[:, 1:] - gt[:, 1:], axis=-1)                                    # (B, 63)
        e_orc, e0 = oracle_decode(net, data, n_all)
        ci = lambda v: S.cluster_bootstrap(v, take, n_boot=S.N_BOOT)
        row = dict(dataset=ds, model=m)
        for k, v in (("E_C_pred", e_pred.mean(1)), ("E_C_oracle_z", e_orc.mean(1)), ("E_C_recon_s0", e0), ("gap_pred_minus_oracle", e_pred.mean(1) - e_orc.mean(1)),
                     ("stage1_E_zonly", s1["e_zonly"][:, 1:].mean(1)), ("stage1_E_full", s1["e_full"][:, 1:].mean(1))):
            mu, lo, hi = ci(v.astype(np.float64)); row.update({k: mu, k + "_lo": lo, k + "_hi": hi})
        for lo, hi in S.HORIZON_BINS:
            row[f"pred_h{lo}_{hi}"] = float(e_pred[:, lo - 1:hi].mean()); row[f"oracle_h{lo}_{hi}"] = float(e_orc[:, lo - 1:hi].mean())
        row["share_of_error_from_bottleneck"] = float((e_orc.mean() ** 2) / (e_pred.mean() ** 2))
        rows.append(row); curves[f"{m}|pred"] = e_pred.mean(0); curves[f"{m}|oracle_z"] = e_orc.mean(0)
        print(ds, m, {k: round(v, 3) for k, v in row.items() if isinstance(v, float) and not k.endswith(("_lo", "_hi"))})
        del net, data
    pd.DataFrame(rows).to_csv(S.ds_out(ds) / "bottleneck.csv", index=False); np.savez(S.ds_out(ds) / "bottleneck_curves.npz", **curves)


if __name__ == "__main__":
    main()
