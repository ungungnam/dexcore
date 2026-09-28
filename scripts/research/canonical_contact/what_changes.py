#!/usr/bin/env python
"""What physically changes along each axis: contact mass and contact centroid by phase.

The three components answer different 'what changed' questions, so this reports, per phase bin,
how much contact there is (mass = sum of the canonical vector) and where it sits (centroid of the
canonical points weighted by the vector), plus how far that centroid travels over a take compared
with how far it moves between meshes at the same phase.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/result/uhnam/dexcore/canonical_contact/time_decomp")
CP = Path("/result/uhnam/dexcore/canonical_contact/canonical_contact_cache/normalized")
BINS = 5
rows, prof = [], []
for p in sorted(ROOT.glob("*__*__*.npz")):
    cat, role, hand = p.stem.split("__")
    z = np.load(p, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
    P = np.load(CP / f"{cat}__{role}.npz", allow_pickle=True)["canonical_points"]   # (K,3)
    b = np.clip((M.phase.values * BINS).astype(int), 0, BINS - 1)
    cen, mass = [], []
    for k in range(BINS):
        m = b == k
        xm = X[m].mean(0)
        mass.append(float(xm.sum()))
        cen.append((xm[:, None] * P).sum(0) / max(xm.sum(), 1e-9))
        prof.append(dict(category=cat, role=role, hand=hand, bin=k, mass=mass[-1],
                         cx=cen[-1][0], cy=cen[-1][1], cz=cen[-1][2], n=int(m.sum())))
    cen = np.stack(cen)
    # Travel has to be measured WITHIN a mesh: pooling meshes first averages the motion away.
    per_mesh_travel, per_mesh_span = [], []
    for mesh, idx in M.groupby("mesh_id").indices.items():
        bm, Xm = b[idx], X[idx]
        c = [(Xm[bm == k].mean(0)[:, None] * P).sum(0) / max(Xm[bm == k].mean(0).sum(), 1e-9)
             for k in range(BINS) if (bm == k).sum() >= 20]
        if len(c) < 3:
            continue
        c = np.stack(c)
        per_mesh_travel.append(np.linalg.norm(c[-1] - c[0]))
        per_mesh_span.append(max(np.linalg.norm(c[i] - c[j]) for i in range(len(c)) for j in range(len(c))))
    travel = float(np.mean(per_mesh_travel))          # within one mesh, start -> end of the take
    span = float(np.mean(per_mesh_span))              # within one mesh, worst pair of phases
    travel_pooled = float(np.linalg.norm(cen[-1] - cen[0]))   # meshes averaged first (understates)
    # between-mesh centroid spread at MATCHED phase (middle bin), same measure
    mid = b == BINS // 2
    per_mesh = []
    for mesh, idx in M[mid].groupby("mesh_id").indices.items():
        xm = X[mid][idx].mean(0)
        per_mesh.append((xm[:, None] * P).sum(0) / max(xm.sum(), 1e-9))
    per_mesh = np.stack(per_mesh)
    mesh_spread = float(np.linalg.norm(per_mesh - per_mesh.mean(0), axis=1).mean())
    rows.append(dict(category=cat, hand=hand, travel_pooled=travel_pooled,
                     mass_start=mass[0], mass_mid=mass[BINS // 2],
                     mass_end=mass[-1], mass_ratio=mass[-1] / max(mass[0], 1e-9),
                     centroid_travel=travel, centroid_span=span,
                     mesh_centroid_spread=mesh_spread, travel_over_mesh=travel / max(mesh_spread, 1e-9)))
D = pd.DataFrame(rows); F = pd.DataFrame(prof)
D.to_csv(ROOT / "what_changes.csv", index=False); F.to_csv(ROOT / "phase_profile.csv", index=False)
pd.set_option("display.width", 230)
print("=== contact mass (sum of the canonical vector) and centroid travel, canonical units ===")
print(D[["category","hand","mass_start","mass_mid","mass_end","mass_ratio","centroid_travel","centroid_span","travel_pooled","mesh_centroid_spread","travel_over_mesh"]].round(3).to_string(index=False))
print("\n=== mass by phase bin (0 = start of the touching span, 4 = end) ===")
print(F.pivot_table(index=["category", "hand"], columns="bin", values="mass").round(1).to_string())
