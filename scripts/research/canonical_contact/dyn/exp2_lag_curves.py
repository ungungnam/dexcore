#!/usr/bin/env python
"""Experiment 2: temporal structure of a contact map -- lag curves and frame-to-frame changes.

Frames: the full stride-1 cache (30 fps, every window). Within one take, for lag L (original
frames) every pair (t, t+L) is scored with the Experiment-1 distances; the per-take mean over t is
the unit of analysis and takes are aggregated with a sequence bootstrap (median and mean over
takes, 95% CI). Two frame sets:
  touching   both frames non-zero (mass >= ZERO_MASS or a vertex within 1 cm)
  all        every frame
Frame-to-frame change delta_t = ||X_t - X_{t-1}|| at 30 fps (plus mass / pattern / centroid
versions). Concentration = share of sum(delta) and sum(delta^2) over a take carried by its top
5 / 10 / 20 % frames, against the iid half-normal null of the same length.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import dc_common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("exp2")
LAGS = [1, 2, 4, 8, 16, 24, 32, 48, 64]
TOPS = [0.05, 0.10, 0.20]
METRICS = ["raw", "mass_abs", "mass_log", "pattern_L2", "centroid"]
RNG = np.random.default_rng(0)


def dists(X, m, Pn, mu, a, b):
    d = X[a] - X[b]
    dp = Pn[a] - Pn[b]
    return dict(raw=np.linalg.norm(d, axis=1), mass_abs=np.abs(m[a] - m[b]),
                mass_log=np.abs(np.log1p(m[a]) - np.log1p(m[b])),
                pattern_L2=np.linalg.norm(dp, axis=1), centroid=np.linalg.norm(mu[a] - mu[b], axis=1))


def top_share(v, frac):
    k = max(1, int(np.ceil(frac * len(v))))
    s = np.sort(v)[::-1]
    return float(s[:k].sum() / max(s.sum(), 1e-12))


def null_share(n, frac, reps=200):
    z = np.abs(RNG.standard_normal((reps, n)))
    k = max(1, int(np.ceil(frac * n)))
    s = -np.sort(-z, axis=1)
    return float((s[:, :k].sum(1) / s.sum(1)).mean())


def main():
    thr = C.zero_threshold()
    curves, per_seq, deltas_q = [], [], []
    for cat, role, hand in C.GROUPS:
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, stride=1)
        m = C.mass(X); Pn = C.pattern(X); mu = Pn @ P
        nz = ~C.is_zero(m, M.n_hard.values, thr)
        hard = M.n_hard.values > 0
        frames = M.frame.values
        rows_lag = {(L, var, k): [] for L in LAGS for var in ("touching", "all", "hard") for k in METRICS}
        seq_of = {(L, var): [] for L in LAGS for var in ("touching", "all", "hard")}
        all_delta = {k: [] for k in METRICS}
        for sid, idx in M.groupby("sequence_id").indices.items():
            idx = idx[np.argsort(frames[idx])]
            f = frames[idx]
            pos = {fr: i for fr, i in zip(f, idx)}
            # frame-to-frame deltas (consecutive frames only)
            cons = np.array([(pos[fr], pos[fr + 1]) for fr in f if fr + 1 in pos])
            rec = dict(group=g, category=cat, role=role, hand=hand, sequence_id=sid, mesh_id=M.mesh_id.values[idx[0]],
                       verb=M.verb.values[idx[0]], n_frames=len(idx), frac_zero=float((~nz[idx]).mean()),
                       mass_mean=float(m[idx].mean()), mass_min=float(m[idx].min()), mass_max=float(m[idx].max()),
                       n_on_off=int(np.abs(np.diff(nz[idx].astype(int))).sum()))
            if len(cons) >= 8:
                dd = dists(X, m, Pn, mu, cons[:, 0], cons[:, 1])
                both = nz[cons[:, 0]] & nz[cons[:, 1]]
                for k in METRICS:
                    v = dd[k][both] if k in ("pattern_L2", "centroid") else dd[k]
                    if len(v) < 8:
                        continue
                    all_delta[k].append(v)
                    rec[f"delta_{k}_mean"] = float(v.mean()); rec[f"delta_{k}_max"] = float(v.max())
                    rec[f"delta_{k}_p90"] = float(np.quantile(v, 0.9))
                    for fr_ in TOPS:
                        rec[f"share_{k}_top{int(fr_*100)}"] = top_share(v, fr_)
                        rec[f"share2_{k}_top{int(fr_*100)}"] = top_share(v ** 2, fr_)
                        rec[f"null_top{int(fr_*100)}_n{len(v)}"] = np.nan   # placeholder, filled below
                    rec[f"gini_{k}"] = float(np.abs(v[:, None] - v[None, :]).mean() / (2 * v.mean() + 1e-12))
                    # frames whose change exceeds 3x / 5x the take median and what they carry
                    med = np.median(v)
                    for kk in (3, 5):
                        big = v > kk * med
                        rec[f"frac_big{kk}_{k}"] = float(big.mean())
                        rec[f"share_big{kk}_{k}"] = float(v[big].sum() / max(v.sum(), 1e-12))
                # centroid travel: whole-take path length and net displacement (touching frames)
                t_idx = idx[nz[idx]]
                if len(t_idx) >= 2:
                    rec["centroid_path"] = float(np.linalg.norm(np.diff(mu[t_idx], axis=0), axis=1).sum())
                    rec["centroid_net"] = float(np.linalg.norm(mu[t_idx[-1]] - mu[t_idx[0]]))
                    rec["centroid_span"] = float(np.linalg.norm(mu[t_idx] - mu[t_idx].mean(0), axis=1).max())
            for L in LAGS:
                pr = np.array([(pos[fr], pos[fr + L]) for fr in f if fr + L in pos])
                if len(pr) < 4:
                    continue
                dd = dists(X, m, Pn, mu, pr[:, 0], pr[:, 1])
                both = nz[pr[:, 0]] & nz[pr[:, 1]]
                bhard = hard[pr[:, 0]] & hard[pr[:, 1]]
                for var, sel in (("all", np.ones(len(pr), bool)), ("touching", both), ("hard", bhard)):
                    if sel.sum() < 4:
                        continue
                    seq_of[(L, var)].append(sid)
                    for k in METRICS:
                        v = dd[k][sel & both] if k in ("pattern_L2", "centroid") else dd[k][sel]
                        rows_lag[(L, var, k)].append(float(v.mean()) if len(v) else np.nan)
                    if var == "touching" and L in (8, 32):
                        rec[f"D{L}_raw"] = float(dd["raw"][sel].mean()); rec[f"D{L}_pattern"] = float(np.nanmean(dd["pattern_L2"][both])) if both.any() else np.nan
                        rec[f"D{L}_mass"] = float(dd["mass_abs"][sel].mean())
            per_seq.append(rec)
        for (L, var, k), vals in rows_lag.items():
            vals = np.asarray(vals); seqs = np.asarray(seq_of[(L, var)])
            ok = ~np.isnan(vals)
            if ok.sum() < 5:
                continue
            for stat, sname in ((np.mean, "mean"), (np.median, "median")):
                pt, lo, hi = C.seq_bootstrap(vals[ok], seqs[ok], None, n_boot=500, stat=stat)
                curves.append(dict(group=g, category=cat, role=role, hand=hand, lag_frames=L, lag_sampled=L / 2,
                                   lag_sec=L / C.FPS, variant=var, metric=k, stat=sname, value=pt, lo=lo, hi=hi, n_seq=int(ok.sum())))
        for k, vs in all_delta.items():
            v = np.concatenate(vs)
            deltas_q.append(dict(group=g, metric=k, n=len(v), mean=v.mean(), **{f"q{int(q*100)}": np.quantile(v, q) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
                                 kurtosis=float(((v - v.mean()) ** 4).mean() / (v.var() ** 2 + 1e-12)),
                                 cv=float(v.std() / (v.mean() + 1e-12))))
        log.info("%s done (%d takes)", g, M.sequence_id.nunique())
    PS = pd.DataFrame(per_seq)
    # null shares for the observed lengths
    for fr_ in TOPS:
        col = f"null_top{int(fr_*100)}"
        cache = {}
        PS[col] = [cache.setdefault(n, null_share(int(n), fr_)) if n == n else np.nan for n in PS.n_frames - 1]
    PS = PS[[c for c in PS.columns if not c.startswith("null_top") or "_n" not in c]]
    PS.to_csv(C.OUT / "exp2" / "per_sequence.csv", index=False)
    CU = pd.DataFrame(curves); CU.to_csv(C.OUT / "lag_curves.csv", index=False)
    pd.DataFrame(deltas_q).to_csv(C.OUT / "exp2" / "delta_quantiles.csv", index=False)
    # transition stats aggregated over takes (sequence bootstrap on per-take values)
    ts = []
    for g, gp in PS.groupby("group"):
        for k in METRICS:
            for col in [f"share_{k}_top5", f"share_{k}_top10", f"share_{k}_top20", f"share2_{k}_top5", f"share2_{k}_top10", f"share2_{k}_top20",
                        f"gini_{k}", f"frac_big3_{k}", f"share_big3_{k}", f"frac_big5_{k}", f"share_big5_{k}", f"delta_{k}_mean"]:
                if col not in gp:
                    continue
                v = gp[col].values; ok = ~np.isnan(v)
                if ok.sum() < 5:
                    continue
                for stat, sname in ((np.mean, "mean"), (np.median, "median")):
                    pt, lo, hi = C.seq_bootstrap(v[ok], gp.sequence_id.values[ok], None, n_boot=500, stat=stat)
                    ts.append(dict(group=g, metric=k, quantity=col, stat=sname, value=pt, lo=lo, hi=hi, n_seq=int(ok.sum())))
        for fr_ in TOPS:
            ts.append(dict(group=g, metric="null", quantity=f"null_top{int(fr_*100)}", stat="mean", value=float(gp[f"null_top{int(fr_*100)}"].mean()), lo=np.nan, hi=np.nan, n_seq=len(gp)))
    pd.DataFrame(ts).to_csv(C.OUT / "transition_stats.csv", index=False)
    # smoothness exponent: log-log slope of the mean curve between lag 1 and 8, touching frames
    sl = []
    for (g, k), gp in CU[(CU.variant == "touching") & (CU.stat == "mean")].groupby(["group", "metric"]):
        gp = gp.set_index("lag_frames").value
        if 1 in gp and 8 in gp and 32 in gp and 64 in gp:
            sl.append(dict(group=g, metric=k, slope_1_8=float(np.log(gp[8] / gp[1]) / np.log(8)),
                           slope_8_32=float(np.log(gp[32] / gp[8]) / np.log(4)), ratio_64_1=float(gp[64] / gp[1]), ratio_64_8=float(gp[64] / gp[8])))
    pd.DataFrame(sl).to_csv(C.OUT / "exp2" / "lag_slopes.csv", index=False)
    pd.set_option("display.width", 250)
    print(CU[(CU.variant == "touching") & (CU.stat == "mean")].pivot_table(index=["group", "metric"], columns="lag_frames", values="value").round(3).to_string())
    print(pd.DataFrame(sl).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
