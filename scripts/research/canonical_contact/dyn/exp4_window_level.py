#!/usr/bin/env python
"""Experiment 4 at WINDOW granularity: the oracle fits of exp4 applied to every 64-frame window
(32 sampled frames) instead of the whole take, relative to the window's own mean. Takes differ in
length across datasets (TACO 1-4 windows, ARCTIC 8-17), so this is the granularity-free view of
how much a static map / rank-K factorization / K held states explain within 2.1 s.
Writes exp4/window_level.csv (macro over groups, mean over windows) and exp4/window_level_per_window.csv."""
import numpy as np
import pandas as pd

import dc_common as C
from exp4_representation import segment_dp_all, hold

REPS = ["static_window_mean", "factor_rank1_lsq", "factor_rank2_lsq", "factor_rank3_lsq"] + [f"sparse_hold_K{k}" for k in (2, 3, 4, 8)]


def main():
    rows = []
    for cat, role, hand in C.GROUPS:
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, stride=C.STRIDE_PAIRS)
        for (sid, w), idx in M.groupby(["sequence_id", "window"]).indices.items():
            idx = idx[np.argsort(M.frame.values[idx])]
            Y = X[idx]; T = len(Y)
            if T < 16:
                continue
            recs = {"static_window_mean": np.tile(Y.mean(0), (T, 1))}
            U, s, Vt = np.linalg.svd(Y, full_matrices=False)
            for r in (1, 2, 3):
                recs[f"factor_rank{r}_lsq"] = (U[:, :r] * s[:r]) @ Vt[:r]
            segs = segment_dp_all(Y, 8)
            for k in (2, 3, 4, 8):
                recs[f"sparse_hold_K{k}"] = hold(Y, segs[min(k, T)])
            for name, R in recs.items():
                rows.append(dict(group=g, sequence_id=sid, window=w, T=T, representation=name,
                                 raw=float(np.linalg.norm(R - Y, axis=1).mean()),
                                 temporal=float(np.linalg.norm(np.diff(R, axis=0) - np.diff(Y, axis=0), axis=1).mean())))
    D = pd.DataFrame(rows)
    D.to_csv(C.OUT / "exp4" / "window_level_per_window.csv", index=False)
    ref = D[D.representation == "static_window_mean"].set_index(["group", "sequence_id", "window"])
    D = D.join(ref[["raw", "temporal"]].rename(columns={"raw": "raw_ref", "temporal": "temporal_ref"}), on=["group", "sequence_id", "window"])
    D["raw_rel"] = D.raw / D.raw_ref; D["temporal_rel"] = D.temporal / D.temporal_ref
    A = D.groupby(["group", "representation"])[["raw", "raw_rel", "temporal_rel"]].mean().reset_index()
    A.to_csv(C.OUT / "exp4" / "window_level.csv", index=False)
    pd.set_option("display.width", 200)
    print(A.groupby("representation")[["raw", "raw_rel", "temporal_rel"]].mean().reindex(REPS).round(3))


if __name__ == "__main__":
    main()
