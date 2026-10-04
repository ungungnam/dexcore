#!/usr/bin/env python
"""Experiment 4: representation ablation -- static / dense / factorized / sparse-state.

Two questions are kept apart:
  (a) SUFFICIENCY (oracle fits): given a take's own dense contact Y (T,512), how well can each
      representation family reproduce it, at what complexity?  No split is involved.
        static_seq_mean      C(x)            = mean_t Y_t                         512 numbers
        static_window_mean   per 64-frame window mean                             512 / window
        factor_rank1_mass    S(x) a(t),  S = normalised sequence-mean pattern, a = mass_t
        factor_rank1_lsq     best rank-1 (SVD)                                    512 + T
        factor_rankK         truncated SVD, K = 2..4                              K (512 + T)
        sparse_hold_K        optimal K-segment piecewise-constant fit (DP), held  K (512 + 1)
        sparse_interp_K      same segments, linear interpolation between segment means
        sparse_rule_k        change-point rule from Exp 2: new state when delta_t > k x take median
        dense                the frames themselves                                 512 T
  (b) PREDICTION (held-out take / held-out mesh, Experiment-3 folds): the same families built
      from what a model could actually produce from geometry + trajectory:
        pred_static_A        MLP-A (geometry only)                 constant per mesh
        pred_dense_B/_C/_D   MLP-B (current state) / MLP-C (local context) / MLP-D (full window)
        pred_factor_AxC/AxD  S = pattern(MLP-A),  a(t) = mass(MLP-C or MLP-D)
        pred_factor_CxC      S = pattern of the take-mean of MLP-C (trajectory-conditioned static map),
                             a(t) = mass(MLP-C)  -- the representation recommended in the report
        pred_sparse_K        MLP-C output collapsed to K held states (DP); pred_sparseD_K from MLP-D
        base_mesh_mean, base_prev_frame, oracle_seq_mean as references
Metrics per frame (raw L2, mass, pattern, centroid, temporal = error on frame differences),
averaged per take, sequence-bootstrapped per group.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import dc_common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("exp4")
KS = [1, 2, 3, 4, 6, 8, 12, 16]
RANKS = [1, 2, 3, 4]
RULE_K = [2, 3, 5]
OUT = C.OUT / "exp4"
PRED_DIR = C.OUT / "exp3" / "preds"


def segment_dp_all(Y, Kmax):
    """Optimal piecewise-constant segmentations of Y (T,d) into k = 1..Kmax contiguous segments
    (min SSE), all from ONE dynamic-programming table. Returns {k: [(start, end), ...]}."""
    T = len(Y)
    Kmax = min(Kmax, T)
    cs = np.vstack([np.zeros((1, Y.shape[1])), np.cumsum(Y, 0)])            # (T+1, d)
    cs2 = np.concatenate([[0.0], np.cumsum((Y ** 2).sum(1))])               # (T+1,)
    # SSE of frames i..j-1 around their mean, for every i < j, vectorised
    G = cs @ cs.T                                                            # (T+1, T+1)
    i, j = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing="ij")
    n = np.maximum(j - i, 1)
    costs = (cs2[j] - cs2[i]) - (np.diag(G)[j] - 2 * G + np.diag(G)[i]) / n   # |S_j - S_i|^2 / n
    costs = np.where(j > i, costs, np.inf)
    Cm = np.full((Kmax + 1, T + 1), np.inf); Cm[0, 0] = 0.0
    arg = np.zeros((Kmax + 1, T + 1), int)
    for k in range(1, Kmax + 1):
        cand = Cm[k - 1][:, None] + costs                                    # (i, j)
        arg[k] = np.argmin(cand, axis=0); Cm[k] = cand[arg[k], np.arange(T + 1)]
    out = {}
    for k in range(1, Kmax + 1):
        segs, jj = [], T
        for kk in range(k, 0, -1):
            ii = arg[kk, jj]; segs.append((ii, jj)); jj = ii
        out[k] = segs[::-1]
    return out


def segment_dp(Y, K):
    return segment_dp_all(Y, K)[min(K, len(Y))]


def hold(Y, segs):
    R = np.empty_like(Y)
    for i, j in segs:
        R[i:j] = Y[i:j].mean(0)
    return R


def interp(Y, segs):
    mids = np.array([(i + j - 1) / 2 for i, j in segs]); vals = np.stack([Y[i:j].mean(0) for i, j in segs])
    t = np.arange(len(Y))
    if len(segs) == 1:
        return np.tile(vals[0], (len(Y), 1))
    R = np.empty_like(Y)
    for d in range(Y.shape[1]):
        R[:, d] = np.interp(t, mids, vals[:, d])
    return R


def rule_segments(Y, k):
    d = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    cuts = np.where(d > k * np.median(d))[0] + 1
    b = [0] + cuts.tolist() + [len(Y)]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def metrics(R, Y, P, zero_t):
    Rc = np.clip(R, 0, None)
    Pt, Pr = C.pattern(Y), C.pattern(Rc)
    out = dict(raw=np.linalg.norm(R - Y, axis=1), mass_abs=np.abs(C.mass(Rc) - C.mass(Y)),
               pattern_L2=np.where(zero_t, np.nan, np.linalg.norm(Pr - Pt, axis=1)),
               centroid=np.where(zero_t, np.nan, np.linalg.norm(Pr @ P - Pt @ P, axis=1)))
    dt = np.linalg.norm(np.diff(R, axis=0) - np.diff(Y, axis=0), axis=1)
    out["temporal"] = np.concatenate([[np.nan], dt])
    return {k: float(np.nanmean(v)) for k, v in out.items()}


def main():
    thr = C.zero_threshold()
    OUT.mkdir(exist_ok=True)
    rows = []
    for cat, role, hand in C.GROUPS:
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, stride=C.STRIDE_PAIRS)
        zero = C.is_zero(C.mass(X), M.n_hard.values, thr)
        # ---------------- (a) oracle sufficiency, every take
        for sid, idx in M.groupby("sequence_id").indices.items():
            idx = idx[np.argsort(M.frame.values[idx])]
            Y = X[idx]; T = len(Y); zt = zero[idx]
            win = M.window.values[idx]
            recs = {}
            recs["static_seq_mean"] = (np.tile(Y.mean(0), (T, 1)), 512)
            Rw = np.empty_like(Y)
            for w in np.unique(win):
                Rw[win == w] = Y[win == w].mean(0)
            recs["static_window_mean"] = (Rw, 512 * len(np.unique(win)))
            S = C.pattern(Y.mean(0)); m = C.mass(Y)
            recs["factor_rank1_mass"] = (m[:, None] * S[None], 512 + T)
            U, s, Vt = np.linalg.svd(Y, full_matrices=False)
            for r in RANKS:
                recs[f"factor_rank{r}_lsq" if r > 1 else "factor_rank1_lsq"] = ((U[:, :r] * s[:r]) @ Vt[:r], r * (512 + T))
            allsegs = segment_dp_all(Y, max(KS))
            for K in KS:
                segs = allsegs[min(K, T)]
                recs[f"sparse_hold_K{K}"] = (hold(Y, segs), len(segs) * 513)
                recs[f"sparse_interp_K{K}"] = (interp(Y, segs), len(segs) * 513)
            for k in RULE_K:
                segs = rule_segments(Y, k)
                recs[f"sparse_rule_k{k}"] = (hold(Y, segs), len(segs) * 513)
            recs["dense"] = (Y.copy(), 512 * T)
            for name, (R, params) in recs.items():
                n_states = params // 513 if name.startswith("sparse") else np.nan
                rows.append(dict(group=g, category=cat, role=role, hand=hand, kind="oracle", split="none", fold=-1,
                                 sequence_id=sid, mesh_id=M.mesh_id.values[idx[0]], representation=name, T=T,
                                 params=params, compression=params / (512 * T), n_states=n_states, **metrics(R, Y, P, zt)))
        # ---------------- (b) predicted, per Experiment-3 fold
        for split in ("take", C.SPLIT2):
            for k in range(3):
                p = PRED_DIR / f"{g}__{split}{k}.npz"
                if not p.exists():
                    log.warning("missing %s", p); continue
                z = np.load(p, allow_pickle=True)
                key = {(s, int(f)): i for i, (s, f) in enumerate(zip(z["sequence_id"].astype(str), z["frame"]))}
                pa, pb, pc, pd_ = (z[n].astype(np.float32) for n in ("mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_D_full_window"))
                test_seqs = np.unique(z["sequence_id"].astype(str))
                # training-set mesh means (same folds as exp3: everything not in the test set)
                tr = ~np.isin(M.sequence_id.values, test_seqs)
                cat_mean = X[tr].mean(0)
                mm = {mesh: X[tr & (M.mesh_id.values == mesh)].mean(0) for mesh in np.unique(M.mesh_id.values[tr])}
                for sid in test_seqs:
                    idx = M.index[M.sequence_id.values == sid].values
                    idx = idx[np.argsort(M.frame.values[idx])]
                    j = np.array([key[(sid, int(f))] for f in M.frame.values[idx]])
                    Y = X[idx]; T = len(Y); zt = zero[idx]
                    A, B, Cc, D = pa[j], pb[j], pc[j], pd_[j]
                    S_A = C.pattern(np.clip(A, 0, None).mean(0))
                    recs = {"pred_static_A": (A, 512), "pred_dense_B": (B, 512 * T), "pred_dense_C": (Cc, 512 * T), "pred_dense_D": (D, 512 * T),
                            "pred_factor_AxC": (C.mass(np.clip(Cc, 0, None))[:, None] * S_A[None], 512 + T),
                            "pred_factor_CxC": (C.mass(np.clip(Cc, 0, None))[:, None] * C.pattern(np.clip(Cc, 0, None).mean(0))[None], 512 + T),
                            "pred_factor_AxD": (C.mass(np.clip(D, 0, None))[:, None] * S_A[None], 512 + T),
                            "base_mesh_mean": (np.tile(mm.get(M.mesh_id.values[idx[0]], cat_mean), (T, 1)), 512),
                            "base_prev_frame": (np.vstack([Y[:1], Y[:-1]]), 512 * T),
                            "oracle_seq_mean": (np.tile(Y.mean(0), (T, 1)), 512)}
                    allsegs = segment_dp_all(Cc, 8); allsegsD = segment_dp_all(D, 8)
                    for K in (1, 2, 4, 8):
                        segs = allsegs[min(K, T)]
                        recs[f"pred_sparse_K{K}"] = (hold(Cc, segs), len(segs) * 513)
                        segs = allsegsD[min(K, T)]
                        recs[f"pred_sparseD_K{K}"] = (hold(D, segs), len(segs) * 513)
                    for name, (R, params) in recs.items():
                        rows.append(dict(group=g, category=cat, role=role, hand=hand, kind="predicted", split=split, fold=k,
                                         sequence_id=sid, mesh_id=M.mesh_id.values[idx[0]], representation=name, T=T,
                                         params=params, compression=params / (512 * T),
                                         n_states=params // 513 if name.startswith("pred_sparse") else np.nan, **metrics(R, Y, P, zt)))
        log.info("%s done", g)
    PS = pd.DataFrame(rows)
    PS.to_csv(OUT / "per_sequence.csv", index=False)
    agg = []
    for (g, kind, split, rep), gp in PS.groupby(["group", "kind", "split", "representation"]):
        row = dict(group=g, kind=kind, split=split, representation=rep, n_seq=len(gp),
                   params_mean=float(gp.params.mean()), compression=float(gp.compression.mean()),
                   n_states=float(gp.n_states.mean()) if gp.n_states.notna().any() else np.nan)
        for met in ("raw", "mass_abs", "pattern_L2", "centroid", "temporal"):
            v = gp[met].values; ok = ~np.isnan(v)
            pt, lo, hi = C.seq_bootstrap(v[ok], gp.sequence_id.values[ok], None, n_boot=300)
            row[met], row[f"{met}_lo"], row[f"{met}_hi"] = pt, lo, hi
        agg.append(row)
    A = pd.DataFrame(agg)
    A.to_csv(C.OUT / "representation_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(A[A.kind == "oracle"].pivot_table(index="representation", columns="group", values="raw").round(3).to_string())
    print(A[A.kind == "predicted"].pivot_table(index=["representation"], columns=["split"], values="raw", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
