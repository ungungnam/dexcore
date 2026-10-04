#!/usr/bin/env python
"""Post-hoc splits, added after the review of the first report. NOT part of the decision rule; no training, CPU only.    python zt_posthoc.py
Everything is recomputed from the saved predictions (<ds>/preds, <ds>/local_arrays.npz, <ds>/geometry_arrays.npz) and the frame cache:
  posthoc_splits.csv           gain over persistence of probe / probe_notau / linear / ar1 on the TEST pairs by window position crossed with
                               (ARCTIC) subject seen / not seen in training, and of the probes on the VALIDATION pairs by window position;
  trajectory_contribution.csv  paired take-cluster bootstrap of gain(probe) - gain(probe without trajectory), per horizon and position;
  quadrant_B_kinds.csv         h = 4: the kinds of low-dC / high-dz test transitions (weak-contact start / structural step / neither, the
                               definitions of Figure 5) with their latent step relative to the 'high' threshold and their contact norm.
All intervals: take-cluster bootstrap, 1000 replicates (zt_common.boot_stat).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import zt_common as Z
from zt_data import Cache

POS = (("all", 0, 63), ("t0", 0, 0), ("t1_7", 1, 7), ("t0_7", 0, 7), ("t8", 8, 63))
TEST_METHODS = ("probe", "probe_notau", "linear", "ar1")
gain = lambda s: 1 - np.sqrt(s[0] / s[1])


def subject_masks(ds, c, take):
    """{label: mask}. ARCTIC takes are '<sequence>/<subject>'; a subject is 'seen' if any training take belongs to it."""
    out = {"all": np.ones(len(take), bool)}
    if ds == "arctic":
        subj = np.array([x.split("/")[-1] for x in take]); seen = sorted(set(x.split("/")[-1] for x in c.take[c.rows("train")]))
        out["seen"] = np.isin(subj, seen); out["unseen"] = ~out["seen"]
    return out


def main():
    srows, trows, krows = [], [], []
    for ds in Z.DATASETS:
        c = Cache(ds); la = np.load(Z.ds_out(ds) / "local_arrays.npz")
        for h in Z.HORIZONS:
            # ---- test pairs: per-pair squared errors saved by zt_eval_local.py
            r, t = la[f"h{h}|r"], la[f"h{h}|t"]; take = c.take[r]; pers = la[f"h{h}|persistence|sse"].astype(np.float64); sm = subject_masks(ds, c, take)
            sse = {m: la[f"h{h}|{m}|sse"].astype(np.float64) for m in TEST_METHODS}
            for sl, smask in sm.items():
                for pl, lo, hi in POS:
                    sel = smask & (t >= lo) & (t <= hi)
                    if not sel.any():
                        continue
                    for m in TEST_METHODS:
                        g, glo, ghi = Z.boot_stat(Z.cluster_sums([sse[m][sel], pers[sel]], take[sel]), gain)
                        srows.append(dict(dataset=ds, h=h, eval_split="test", method=m, subject=sl, position=pl, n_pairs=int(sel.sum()), n_takes=int(len(set(take[sel]))), gain_rmse=g, gain_rmse_lo=glo, gain_rmse_hi=ghi,
                                          rmse_persistence=float(np.sqrt(pers[sel].mean() / c.dz)), share_of_persistence_sse=float(pers[sel].sum() / pers.sum())))
                    # paired contrast: what the object trajectory adds to the probe (same pairs, same bootstrap resamples)
                    d, dlo, dhi = Z.boot_stat(Z.cluster_sums([sse["probe"][sel], sse["probe_notau"][sel], pers[sel]], take[sel]), lambda s: np.sqrt(s[1] / s[2]) - np.sqrt(s[0] / s[2]))
                    trows.append(dict(dataset=ds, h=h, subject=sl, position=pl, n_pairs=int(sel.sum()), gain_probe_minus_notau=d, lo=dlo, hi=dhi))
            # ---- validation pairs: saved validation predictions of the two probes (all validation takes are from training subjects on ARCTIC)
            for m in ("probe", "probe_notau"):
                p = np.load(Z.preds_path(ds, m, h)); vr, vt = p["val_r"], p["val_t"]; vtake = c.take[vr]
                vs = ((p["val_z_hat"] - c.zs[vr, vt + h]) ** 2).sum(1).astype(np.float64); vp = ((c.zs[vr, vt] - c.zs[vr, vt + h]) ** 2).sum(1).astype(np.float64)
                for pl, lo, hi in POS:
                    sel = (vt >= lo) & (vt <= hi)
                    g, glo, ghi = Z.boot_stat(Z.cluster_sums([vs[sel], vp[sel]], vtake[sel]), gain)
                    srows.append(dict(dataset=ds, h=h, eval_split="val", method=m, subject="all", position=pl, n_pairs=int(sel.sum()), n_takes=int(len(set(vtake[sel]))), gain_rmse=g, gain_rmse_lo=glo, gain_rmse_hi=ghi,
                                      rmse_persistence=float(np.sqrt(vp[sel].mean() / c.dz)), share_of_persistence_sse=float(vp[sel].sum() / vp.sum())))
        # ---- kinds of low-dC / high-dz transitions at the horizon of Figure 5 (same definitions as zt_figures.fig5)
        h = 4; ga = np.load(Z.ds_out(ds) / "geometry_arrays.npz"); dC, dz, dT, fl, nC = (ga[f"h{h}|{k}"] for k in ("dC", "dz", "dT", "flips", "normC")); thr = ga[f"h{h}|thr"]; wthr = float(ga[f"h{h}|weak_thr"][0])
        weak = nC <= wthr; struct = (fl > 0) | (dT >= thr[5]); inB = (dC <= thr[0]) & (dz >= thr[3])
        for kind, mask in (("all", inB), ("weak", inB & weak), ("structural", inB & ~weak & struct), ("neither", inB & ~weak & ~struct)):
            if mask.any():
                q = dz[mask] / thr[3]
                krows.append(dict(dataset=ds, h=h, kind=kind, n=int(mask.sum()), n_B=int(inB.sum()), dz_over_high_thr_min=float(q.min()), dz_over_high_thr_median=float(np.median(q)), dz_over_high_thr_max=float(q.max()),
                                  normC_min=float(nC[mask].min()), normC_median=float(np.median(nC[mask])), test_normC_median=float(np.median(nC)), weak_thr=wthr, dz_high_thr=float(thr[3]), dT_high_thr=float(thr[5]),
                                  dT_median=float(np.median(dT[mask]))))
    S = pd.DataFrame(srows); S.to_csv(Z.OUT / "posthoc_splits.csv", index=False); T = pd.DataFrame(trows); T.to_csv(Z.OUT / "trajectory_contribution.csv", index=False)
    K = pd.DataFrame(krows); K.to_csv(Z.OUT / "quadrant_B_kinds.csv", index=False)
    pd.set_option("display.width", 250)
    print(S[(S.method == "probe") & (S.dataset == "arctic")].round(3).to_string(index=False))
    print(S[(S.method == "probe") & (S.dataset == "taco")].round(3).to_string(index=False))
    print(T[T.position == "all"].round(4).to_string(index=False)); print(K.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
