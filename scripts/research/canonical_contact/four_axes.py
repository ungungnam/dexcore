#!/usr/bin/env python
"""All four axes of a TACO contact map on one scale: time, take, mesh, action.

Every pair holds the category, role and hand fixed and is measured as L2 between frame-level
canonical vectors. Mesh and action are both increments over the SAME baseline (another take of the
same mesh doing the same verb, at a matched phase), so the two are directly comparable:

  floor   same mesh, same take, |dphase| < 0.05          two frames of one moment
  time    same mesh, same take, |dphase| > 0.5           start of a take against its end
  take    same mesh, other take, same verb, phase matched
  mesh    other mesh, same verb, phase matched           (increment over take)
  action  same mesh, other verb, phase matched           (increment over take)

Distances do not add, so each pure component is sqrt of the difference of squares.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/result/uhnam/dexcore/canonical_contact/time_decomp")
RNG = np.random.default_rng(0)
CAP, TOL = 4000, 0.1


def sample(X, ph, take, mesh, verb, same_mesh, same_verb, matched=True):
    n = len(ph)
    acc = []
    for _ in range(80):
        a, b = RNG.integers(0, n, CAP * 4), RNG.integers(0, n, CAP * 4)
        ok = take[a] != take[b]
        ok &= (mesh[a] == mesh[b]) if same_mesh else (mesh[a] != mesh[b])
        ok &= (verb[a] == verb[b]) if same_verb else (verb[a] != verb[b])
        if matched:
            ok &= np.abs(ph[a] - ph[b]) < TOL
        if ok.any():
            acc.append(np.linalg.norm(X[a[ok]] - X[b[ok]], axis=1))
        if sum(len(x) for x in acc) >= CAP:
            break
    v = np.concatenate(acc)[:CAP] if acc else np.array([])
    return (float(v.mean()) if len(v) else np.nan), len(v)


def within_take(X, ph, take, lo, hi):
    idx = pd.Series(range(len(ph))).groupby(take).indices
    out = []
    for t, i in idx.items():
        if len(i) < 4:
            continue
        a = RNG.choice(i, min(len(i), 80), replace=True)
        b = RNG.choice(i, min(len(i), 80), replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        gap = np.abs(ph[a] - ph[b])
        m = (gap >= lo) & (gap < hi)
        if m.any():
            out.append(np.linalg.norm(X[a[m]] - X[b[m]], axis=1))
    v = np.concatenate(out) if out else np.array([])
    return (float(v.mean()) if len(v) else np.nan), len(v)


rows = []
for p in sorted(ROOT.glob("*__*__*.npz")):
    cat, role, hand = p.stem.split("__")
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
    ph = M.phase.values.astype(float)
    take, mesh, verb = M.sequence_id.astype(str).values, M.mesh_id.values, M.verb.values
    # how much same-mesh multi-verb material exists at all
    nv = M.groupby("mesh_id").verb.nunique()
    d = {}
    d["floor"], d["n_floor"] = within_take(X, ph, take, 0.0, 0.05)
    d["time_raw"], d["n_time"] = within_take(X, ph, take, 0.5, 1.01)
    d["take_raw"], d["n_take"] = sample(X, ph, take, mesh, verb, True, True)
    d["mesh_raw"], d["n_mesh"] = sample(X, ph, take, mesh, verb, False, True)
    d["action_raw"], d["n_action"] = sample(X, ph, take, mesh, verb, True, False)
    q = lambda a, b: float(np.sqrt(max(a ** 2 - b ** 2, 0.0)))
    rows.append(dict(category=cat, role=role, hand=hand, n_verbs=int(M.verb.nunique()),
                     meshes_multiverb=int((nv >= 2).sum()), n_mesh_total=int(M.mesh_id.nunique()),
                     **d,
                     time=q(d["time_raw"], d["floor"]), take=q(d["take_raw"], d["floor"]),
                     mesh=q(d["mesh_raw"], d["take_raw"]), action=q(d["action_raw"], d["take_raw"])))
D = pd.DataFrame(rows)
D["action_over_mesh"] = D.action / D.mesh
D["time_over_mesh"] = D.time / D.mesh
D.round(4).to_csv(ROOT / "four_axes.csv", index=False)
pd.set_option("display.width", 240)
print("=== raw pair distances (mean L2) ===")
print(D[["category", "hand", "n_verbs", "meshes_multiverb", "floor", "time_raw", "take_raw",
         "mesh_raw", "action_raw", "n_action"]].round(3).to_string(index=False))
print("\n=== pure components (RMS) ===")
print(D[["category", "hand", "time", "take", "mesh", "action", "action_over_mesh",
         "time_over_mesh"]].round(3).to_string(index=False))
print("\naction/mesh: median %.2f  range %.2f-%.2f" % (D.action_over_mesh.median(),
      D.action_over_mesh.min(), D.action_over_mesh.max()))
