#!/usr/bin/env python
"""Local probe vs the saved Stage-2 B1 open-loop latent predictions (nothing is retrained).    python zt_stage2.py
Stage 2 (B1):  (s_0, G, tau_1:63) -> z_hat_f for every frame f = 1..63   — the initial MAP is known, the initial latent is not given.
Local probe:   (z_t^GT, G, tau_t, tau_{t+h}) -> z_hat_{t+h}                — the latent h frames earlier is known.
Two matched comparisons on identical test sequences and target frames (standardised z, take-cluster bootstrap):
  from_start   target frame f = h: Stage-2 error at frame h vs the probe started from the GT z_0 (both know only the initial state);
  all_frames   every target frame f >= h: Stage-2 error at f vs the probe started from the GT z_{f-h} (what a state h frames old buys).
rescue = 1 - RMSE_probe / RMSE_stage2 (positive: access to a GT latent state makes the prediction easier).
Writes OUT/stage2_comparison.csv and <ds>/stage2_curves.npz (per-frame RMSE of Stage 2, of holding z*_0, of the probes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import zt_common as Z
from zt_data import Cache


def main():
    rows = []
    for ds in Z.DATASETS:
        c = Cache(ds); te = c.rows("test"); take = c.take[te]
        s2 = np.load(Z.stage2_preds_path(ds)); assert np.array_equal(s2["example"], c.seq[te]), "Stage-2 predictions are not on the same test sequences"
        ck = torch.load(Z.S2.ckpt_path(ds, Z.S2.run_name("B1")), map_location="cpu", weights_only=False)
        assert np.allclose(ck["stats"]["z_mu"], c.z_mu, atol=1e-4) and np.allclose(ck["stats"]["z_sd"], c.z_sd, atol=1e-4), "different z standardisation"
        zs = c.zs[te]; zh = s2["z_hat"].astype(np.float64)                         # (B, 64, dz), (B, 63, dz): frames 1..63
        sse_s2 = ((zh - zs[:, 1:]) ** 2).sum(-1)                                     # (B, 63) frame f = 1..63 at column f - 1
        sse_hold = ((zs[:, :1] - zs[:, 1:]) ** 2).sum(-1)
        sse_mean = (zs[:, 1:] ** 2).sum(-1)                                          # train-mean predictor (standardised mean = 0)
        curves = {"stage2": np.sqrt(sse_s2.mean(0) / c.dz), "hold_z0": np.sqrt(sse_hold.mean(0) / c.dz), "train_mean": np.sqrt(sse_mean.mean(0) / c.dz)}
        keep = {}
        row_of = {int(r): i for i, r in enumerate(te)}
        for h in Z.HORIZONS:
            p = np.load(Z.preds_path(ds, "probe", h)); r, t = p["test_r"], p["test_t"]
            sse_p = ((p["test_z_hat"] - c.zs[r, t + h]) ** 2).sum(1); sse_pers = ((c.zs[r, t] - c.zs[r, t + h]) ** 2).sum(1)
            P = np.full((len(te), Z.T), np.nan); Pp = np.full((len(te), Z.T), np.nan)          # indexed by (sequence, target frame f = t + h)
            ii = np.array([row_of[int(x)] for x in r]); P[ii, t + h] = sse_p; Pp[ii, t + h] = sse_pers
            curves[f"probe_h{h}_by_target_frame"] = np.sqrt(np.nanmean(P, 0) / c.dz); curves[f"persistence_h{h}_by_target_frame"] = np.sqrt(np.nanmean(Pp, 0) / c.dz)
            for mode, frames in (("from_start", [h]), ("all_frames", list(range(h, Z.T)))):
                f = np.array(frames); a = sse_s2[:, f - 1].sum(1); b = P[:, f].sum(1); pe = Pp[:, f].sum(1); ho = sse_hold[:, f - 1].sum(1); n = np.full(len(te), len(f) * c.dz, float)
                zb = zs[:, f].reshape(-1, c.dz).mean(0); st = ((zs[:, f] - zb) ** 2).sum((1, 2))          # SST around the mean of the SCORED target frames
                if mode == "from_start":
                    keep[h] = (a, b)
                cs = Z.cluster_sums([a, b, pe, ho, st, n], take)
                row = dict(dataset=ds, h=h, mode=mode, n_sequences=len(te), n_target_frames=len(f))
                for k, fn in (("rmse_stage2", lambda s: np.sqrt(s[0] / s[5])), ("rmse_probe", lambda s: np.sqrt(s[1] / s[5])), ("rmse_persistence", lambda s: np.sqrt(s[2] / s[5])), ("rmse_hold_z0", lambda s: np.sqrt(s[3] / s[5])),
                              ("rescue", lambda s: 1 - np.sqrt(s[1] / s[0])), ("r2_stage2", lambda s: 1 - s[0] / s[4]), ("r2_probe", lambda s: 1 - s[1] / s[4]), ("r2_persistence", lambda s: 1 - s[2] / s[4]),
                              ("stage2_gain_over_persistence", lambda s: 1 - np.sqrt(s[0] / s[2]))):
                    row[k], row[k + "_lo"], row[k + "_hi"] = Z.boot_stat(cs, fn)
                rows.append(row)
        # the rule's rescue: mean over h = 4, 8 of the from-start rescue, with a joint take bootstrap
        cs = Z.cluster_sums([keep[4][0], keep[4][1], keep[8][0], keep[8][1]], take)
        v, lo, hi = Z.boot_stat(cs, lambda s: 0.5 * ((1 - np.sqrt(s[1] / s[0])) + (1 - np.sqrt(s[3] / s[2]))))
        rows.append(dict(dataset=ds, h=0, mode="rescue_mean_h4_h8", n_sequences=len(te), n_target_frames=2, rescue=v, rescue_lo=lo, rescue_hi=hi))
        # context: the whole Stage-2 trajectory
        zb = zs[:, 1:].reshape(-1, c.dz).mean(0); sst = ((zs[:, 1:] - zb) ** 2).sum(-1)
        cs = Z.cluster_sums([sse_s2.sum(1), sse_hold.sum(1), sst.sum(1), np.full(len(te), 63 * c.dz, float)], take)
        ctx = dict(dataset=ds, h=0, mode="stage2_all_63_frames", n_sequences=len(te), n_target_frames=63)
        for k, fn in (("rmse_stage2", lambda s: np.sqrt(s[0] / s[3])), ("rmse_hold_z0", lambda s: np.sqrt(s[1] / s[3])), ("r2_stage2", lambda s: 1 - s[0] / s[2])):
            ctx[k], ctx[k + "_lo"], ctx[k + "_hi"] = Z.boot_stat(cs, fn)
        rows.append(ctx)
        np.savez(Z.ds_out(ds) / "stage2_curves.npz", **curves)
        print(ds, pd.DataFrame([r for r in rows if r["dataset"] == ds])[["h", "mode", "rmse_stage2", "rmse_probe", "rmse_persistence", "rmse_hold_z0", "rescue", "r2_stage2", "r2_probe"]].round(3).to_string(index=False))
    pd.DataFrame(rows).to_csv(Z.OUT / "stage2_comparison.csv", index=False)


if __name__ == "__main__":
    main()
