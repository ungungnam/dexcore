#!/usr/bin/env python
"""Does the phase-matching rule decide the answer?

The ladder matches a pair when |phase_a - phase_b| < 0.1, phase being the frame's position in the
sequence's touching span, 0 to 1. Two assumptions ride on that: the tolerance, and normalising the
span (which assumes takes are time-warped versions of one another). Both are varied here, plus an
alignment that ignores the span entirely and keys on the moment of peak contact instead.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/result/uhnam/dexcore/canonical_contact/time_decomp")
RNG = np.random.default_rng(0)
CAP = 4000
TOLS = [0.05, 0.10, 0.20]


def pairs(X, ph, take, mesh, verb, same_mesh, tol, n_want=CAP):
    n = len(ph)
    acc = []
    for _ in range(60):
        a, b = RNG.integers(0, n, CAP * 4), RNG.integers(0, n, CAP * 4)
        ok = (take[a] != take[b]) & (verb[a] == verb[b])
        ok &= (mesh[a] == mesh[b]) if same_mesh else (mesh[a] != mesh[b])
        if tol is not None:
            ok &= np.abs(ph[a] - ph[b]) < tol
        if ok.any():
            acc.append(np.linalg.norm(X[a[ok]] - X[b[ok]], axis=1))
        if sum(len(x) for x in acc) >= n_want:
            break
    v = np.concatenate(acc)[:n_want] if acc else np.array([np.nan])
    return float(v.mean()), len(v)


rows = []
for p in sorted(ROOT.glob("*__*__*.npz")):
    cat, role, hand = p.stem.split("__")
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
    take, mesh, verb = M.sequence_id.astype(str).values, M.mesh_id.values, M.verb.values
    mass = X.sum(1)
    # alternative clock: frames from the take's peak-contact frame, in frames (30 fps), rescaled
    # to the same 0-1 range so one tolerance means the same thing for both clocks
    peak = np.zeros(len(M))
    for t, i in M.groupby(M.sequence_id.astype(str)).indices.items():
        peak[i] = M.frame.values[i] - M.frame.values[i][int(np.argmax(mass[i]))]
    peak = peak / max(np.abs(peak).max(), 1)
    for clock, ph in (("span", M.phase.values.astype(float)), ("peak", peak)):
        for tol in TOLS:
            for name, sm in (("other_take", True), ("other_mesh", False)):
                d, k = pairs(X, ph, take, mesh, verb, sm, tol)
                rows.append(dict(category=cat, hand=hand, clock=clock, tol=tol, pair=name, dist=d, n=k))
        for name, sm in (("other_take", True), ("other_mesh", False)):
            d, k = pairs(X, ph, take, mesh, verb, sm, None)
            rows.append(dict(category=cat, hand=hand, clock=clock, tol=np.nan, pair=name, dist=d, n=k))
D = pd.DataFrame(rows)
D.to_csv(ROOT / "phase_sensitivity.csv", index=False)
W = D.pivot_table(index=["category", "hand", "clock", "tol"], columns="pair", values="dist", dropna=False)
W["GP"] = W.other_mesh / W.other_take
pd.set_option("display.width", 200)
print("=== mean L2 by matching rule (tol NaN = phase ignored) ===")
print(W.round(3).to_string())
print("\n=== geometry penalty (other_mesh / other_take) by clock and tolerance ===")
print(W.reset_index().pivot_table(index=["category", "hand"], columns=["clock", "tol"],
                                  values="GP", dropna=False).round(3).to_string())
