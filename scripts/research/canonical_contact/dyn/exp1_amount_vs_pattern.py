#!/usr/bin/env python
"""Experiment 1: what changes along each axis -- contact AMOUNT or contact PATTERN?

Pairs are drawn exactly as in four_axes.py (same frames: every 2nd frame of the touch>=0.2
windows; same conditions), but every pair is scored with several distances:
  raw         ||X_a - X_b||_2                      raw_L1 = ||X_a - X_b||_1
  mass_abs    |m_a - m_b|                          mass_log = |log(m_a+1) - log(m_b+1)|
  pattern_L2  ||P_a - P_b||_2                      pattern_L1 = 0.5 * sum |P_a - P_b|
  pattern_cos 1 - <P_a, P_b> / (|P_a||P_b|)
  centroid    ||mu_a - mu_b||                       (canonical units)
Pattern/centroid distances are only defined when neither map is 'zero' (see exp1_threshold.py);
pairs are classified both_zero / one_zero / both_nonzero and pattern metrics are reported on the
both_nonzero class only.

Conditions (category, role, hand fixed; hands compared like-for-like):
  floor   same take, |dphase| < 0.05                   two frames of one moment
  time    same take, |dphase| > 0.5                    early vs late in the same take
  take    same mesh, same verb, other take, |dphase| < TOL
  mesh    other mesh, same verb, other take, |dphase| < TOL
  action  same mesh, other verb, other take, |dphase| < TOL
Phase = (frame - lo) / (hi - lo), lo/hi the first/last frame of the take's touching windows.
TOL is 0.10 for the main table and {0.05, 0.10, 0.20} for the sensitivity table.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import dc_common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("exp1")
CAP = 4000
TOLS = [0.05, 0.10, 0.20]
METRICS = ["raw", "raw_L1", "mass_abs", "mass_log", "pattern_L2", "pattern_L1", "pattern_cos", "centroid"]
PATTERN_METRICS = ["pattern_L2", "pattern_L1", "pattern_cos", "centroid"]


def pair_metrics(X, m, Pn, mu, zero, a, b):
    d = X[a] - X[b]
    both_nz = ~zero[a] & ~zero[b]
    cls = np.where(zero[a] & zero[b], "both_zero", np.where(both_nz, "both_nonzero", "one_zero"))
    out = dict(raw=np.linalg.norm(d, axis=1), raw_L1=np.abs(d).sum(1),
               mass_abs=np.abs(m[a] - m[b]), mass_log=np.abs(np.log1p(m[a]) - np.log1p(m[b])))
    dp = Pn[a] - Pn[b]
    out["pattern_L2"] = np.linalg.norm(dp, axis=1)
    out["pattern_L1"] = 0.5 * np.abs(dp).sum(1)
    na, nb = np.linalg.norm(Pn[a], axis=1), np.linalg.norm(Pn[b], axis=1)
    out["pattern_cos"] = 1 - (Pn[a] * Pn[b]).sum(1) / np.maximum(na * nb, 1e-12)
    out["centroid"] = np.linalg.norm(mu[a] - mu[b], axis=1)
    for k in PATTERN_METRICS:
        out[k] = np.where(both_nz, out[k], np.nan)
    out["cls"] = cls
    return out


def sample_cross(rng, ph, take, mesh, verb, subj, ax, tol, cap=CAP):
    """Pairs of different takes satisfying the axis spec (see dc_common._axis)."""
    n = len(ph)
    A, B = [], []
    for _ in range(100):
        a, b = rng.integers(0, n, cap * 4), rng.integers(0, n, cap * 4)
        ok = take[a] != take[b]
        for key, arr in (("mesh", mesh), ("verb", verb), ("subject", subj)):
            if ax[key] is True:
                ok &= arr[a] == arr[b]
            elif ax[key] is False:
                ok &= arr[a] != arr[b]
        if tol is not None:
            ok &= np.abs(ph[a] - ph[b]) < tol
        A.append(a[ok]); B.append(b[ok])
        if sum(len(x) for x in A) >= cap:
            break
    a, b = np.concatenate(A)[:cap], np.concatenate(B)[:cap]
    return a, b


def sample_within(rng, ph, take, lo, hi, per_take=80):
    idx = pd.Series(range(len(ph))).groupby(take).indices
    A, B = [], []
    for t, i in idx.items():
        if len(i) < 4:
            continue
        a = rng.choice(i, min(len(i), per_take), replace=True)
        b = rng.choice(i, min(len(i), per_take), replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        gap = np.abs(ph[a] - ph[b])
        k = (gap >= lo) & (gap < hi)
        A.append(a[k]); B.append(b[k])
    return np.concatenate(A), np.concatenate(B)


def summarise(vals, sa, sb, stat):
    v = np.asarray(vals, float)
    ok = ~np.isnan(v)
    if ok.sum() < 10:
        return np.nan, np.nan, np.nan, int(ok.sum())
    pt, lo, hi = C.seq_bootstrap(v[ok], sa[ok], sb[ok], n_boot=500, stat=stat)
    return pt, lo, hi, int(ok.sum())


def main(touching_only=True, tag="", backend="normalized"):
    thr = C.zero_threshold()
    rng = np.random.default_rng(0)
    summary, sens, zero_share, pairs_store = [], [], [], {}
    for cat, role, hand in C.GROUPS:
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, backend=backend, stride=C.STRIDE_PAIRS, touching_windows_only=touching_only)
        ok = ~np.isnan(M.phase.values)
        X, M = X[ok], M[ok].reset_index(drop=True)
        m = C.mass(X); Pn = C.pattern(X); mu = Pn @ P
        zero = C.is_zero(m, M.n_hard.values, thr)
        ph = M.phase.values.astype(float)
        take, mesh, verb, subj = M.sequence_id.values, M.mesh_id.values, M.verb.values, M.subject.values
        med_mass = float(np.median(m[~zero]))
        conds = {}
        for ax in C.AXES:
            if ax["kind"] == "within":
                conds[(ax["name"], 0.05)] = sample_within(rng, ph, take, ax["lo"], ax["hi"])
            else:
                for tol in TOLS:
                    conds[(ax["name"], tol)] = sample_cross(rng, ph, take, mesh, verb, subj, ax, tol)
        for (axis, tol), (a, b) in conds.items():
            if len(a) == 0:
                continue
            pm = pair_metrics(X, m, Pn, mu, zero, a, b)
            sa, sb = take[a], take[b]
            main_tol = axis in ("floor", "time") or tol == 0.10
            if main_tol:
                pairs_store[f"{g}__{axis}"] = dict(a=a, b=b, seq_a=sa, seq_b=sb, **pm)
                cls = pd.Series(pm["cls"])
                share = cls.value_counts(normalize=True).to_dict()
                d2 = pm["raw"] ** 2
                zero_share.append(dict(group=g, category=cat, hand=hand, axis=axis, n=len(a),
                                       **{f"frac_{k}": share.get(k, 0.0) for k in ("both_zero", "one_zero", "both_nonzero")},
                                       **{f"d2share_{k}": float(d2[cls.values == k].sum() / d2.sum()) for k in ("both_zero", "one_zero", "both_nonzero")},
                                       raw_mean_all=float(pm["raw"].mean()),
                                       raw_mean_both_nonzero=float(pm["raw"][cls.values == "both_nonzero"].mean()) if (cls == "both_nonzero").any() else np.nan))
            for metric in METRICS:
                for variant in (["all", "both_nonzero", "high_contact", "both_hard"] if main_tol else ["all"]):
                    v = pm[metric].copy()
                    if variant == "both_nonzero":
                        v[pm["cls"] != "both_nonzero"] = np.nan
                    elif variant == "both_hard":            # physical rule: a vertex within 1 cm on both frames
                        v[~((M.n_hard.values[a] > 0) & (M.n_hard.values[b] > 0))] = np.nan
                    elif variant == "high_contact":
                        v[~((m[a] >= med_mass) & (m[b] >= med_mass))] = np.nan
                    for stat, sname in ((np.mean, "mean"), (np.median, "median")):
                        pt, lo, hi, n = summarise(v, sa, sb, stat)
                        row = dict(group=g, category=cat, role=role, hand=hand, axis=axis, tol=tol,
                                   metric=metric, variant=variant, stat=sname, value=pt, lo=lo, hi=hi, n=n)
                        (summary if main_tol else sens).append(row)
                        if main_tol and variant == "all":
                            sens.append(row)
        log.info("%s: %d frames, %d takes, %d meshes, %d subjects, zero frames %.1f%%  pairs: %s", g, len(X),
                 M.sequence_id.nunique(), M.mesh_id.nunique(), M.subject.nunique(), 100 * zero.mean(),
                 {k: len(v[0]) for k, v in conds.items() if k[1] in (0.05, 0.10)})
    S = pd.DataFrame(summary); Z = pd.DataFrame(zero_share); SE = pd.DataFrame(sens)
    out = C.OUT / "exp1"
    S.to_csv(out / f"four_axes_long{tag}.csv", index=False)
    Z.to_csv(out / f"zero_transition_share{tag}.csv", index=False)
    SE.to_csv(out / f"phase_sensitivity{tag}.csv", index=False)
    np.savez_compressed(out / f"pairs{tag}.npz", **{f"{k}__{kk}": vv for k, d in pairs_store.items() for kk, vv in d.items()})

    # ---- main table: rows = axis, columns = metric (mean over pairs, tol 0.10, variant all), per group
    main = S[(S.variant == "all") & (S.stat == "mean")].pivot_table(index=["group", "category", "role", "hand", "axis"],
                                                                     columns="metric", values="value").reset_index()
    # ratios with paired sequence bootstrap: time/take, time/mesh, mesh/take, action/take, per metric
    ratios = []
    for g in main.group.unique():
        for metric in METRICS:
            def get(axis):
                d = pairs_store[f"{g}__{axis}"]
                v = d[metric]; ok = ~np.isnan(v)
                return v[ok], np.stack([d["seq_a"][ok], d["seq_b"][ok]], 1)
            for num, den in C.RATIOS:
                if f"{g}__{num}" not in pairs_store or f"{g}__{den}" not in pairs_store:
                    continue
                vn, sn = get(num); vd, sd = get(den)
                if len(vn) < 10 or len(vd) < 10:
                    continue
                for stat, sname in ((np.mean, "mean"), (np.median, "median")):
                    r, lo, hi = C.paired_ratio_bootstrap(vn, vd, sn, sd, n_boot=500, stat=stat)
                    ratios.append(dict(group=g, metric=metric, ratio=f"{num}/{den}", stat=sname, value=r, lo=lo, hi=hi))
    R = pd.DataFrame(ratios)
    R.to_csv(out / f"ratios{tag}.csv", index=False)
    # incremental RMS estimates (secondary), on means
    inc = []
    for g, gm in main.groupby("group"):
        gm = gm.set_index("axis")
        q = lambda a, b: float(np.sqrt(max(a ** 2 - b ** 2, 0.0)))
        for metric in METRICS:
            if not {"floor", "time", "take"} <= set(gm.index):
                continue
            v = gm[metric]
            row = dict(group=g, metric=metric, time=q(v["time"], v["floor"]), take=q(v["take"], v["floor"]))
            for axn in C.CROSS_AXES:
                if axn != "take" and axn in gm.index:
                    row[axn] = q(v[axn], v["take"])
            inc.append(row)
    pd.DataFrame(inc).to_csv(out / f"incremental_rms{tag}.csv", index=False)
    if not tag:
        main.to_csv(C.OUT / "metrics_four_axes.csv", index=False)
    else:
        main.to_csv(out / f"metrics_four_axes{tag}.csv", index=False)
    pd.set_option("display.width", 250)
    print(main.round(3).to_string(index=False))
    print(Z.round(3).to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--all-windows":
        main(touching_only=False, tag="_allwindows")
    elif len(sys.argv) > 2 and sys.argv[1] == "--backend":
        main(tag=f"_{sys.argv[2]}", backend=sys.argv[2])
    else:
        main()
