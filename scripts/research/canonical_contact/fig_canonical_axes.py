#!/usr/bin/env python
"""The same two axes in the space the model compares: the category's 512 canonical points.

figF draws each instance in its own frame, where every spatula's grip looks like 'the handle'.
Here the same means are drawn on the SHARED canonical points, which is where a cross-mesh
distance is actually taken: time moves the value up and down in place, different meshes light up
different points.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
CACHE = ROOT / "canonical_contact_cache/normalized"
FRAMES = ROOT / "time_decomp"
CAT, ROLE, HAND = "spatula", "tool", "R"
BINS, VMAX = 5, 0.55

z = np.load(FRAMES / f"{CAT}__{ROLE}__{HAND}.npz", allow_pickle=True)
X = z["X"].astype(np.float32)
M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
P = np.load(CACHE / f"{CAT}__{ROLE}.npz", allow_pickle=True)["canonical_points"]
b = np.clip((M.phase.values * BINS).astype(int), 0, BINS - 1)

# 2D view of the canonical points: the two directions with the most spread
C = P - P.mean(0)
U, S, Vt = np.linalg.svd(C, full_matrices=False)
xy = C @ Vt[:2].T
order = np.argsort(np.abs(xy[:, 1]))          # draw near-axis points last so they stay visible

counts = M.groupby("mesh_id").sequence_id.nunique().sort_values(ascending=False)
meshes = counts.index[:BINS].tolist()
main = meshes[0]

fig, axes = plt.subplots(2, BINS, figsize=(16.5, 4.6))
for k in range(BINS):
    v = X[(b == k) & (M.mesh_id.values == main)].mean(0)
    ax = axes[0, k]
    ax.scatter(xy[order, 0], xy[order, 1], c=np.clip(v[order] / VMAX, 0, 1), cmap="YlOrRd",
               vmin=0, vmax=1, s=26, linewidths=0)
    ax.set_title(f"phase {k/BINS:.1f}-{(k+1)/BINS:.1f}   sum {v.sum():.0f}", fontsize=11)
mid = BINS // 2
cen = []
for k, m in enumerate(meshes):
    v = X[(b == mid) & (M.mesh_id.values == m)].mean(0)
    cen.append((xy * v[:, None]).sum(0) / max(v.sum(), 1e-9))
    ax = axes[1, k]
    ax.scatter(xy[order, 0], xy[order, 1], c=np.clip(v[order] / VMAX, 0, 1), cmap="YlOrRd",
               vmin=0, vmax=1, s=26, linewidths=0)
    ax.set_title(f"mesh {m}   sum {v.sum():.0f}", fontsize=11)
cen = np.stack(cen)
for k in range(BINS):                                    # mark each row's contact centroid
    vt = X[(b == k) & (M.mesh_id.values == main)].mean(0)
    ct = (xy * vt[:, None]).sum(0) / max(vt.sum(), 1e-9)
    axes[0, k].plot(*ct, "kx", ms=9, mew=2)
    axes[1, k].plot(*cen[k], "kx", ms=9, mew=2)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for s_ in ax.spines.values():
        s_.set_color("#cccccc")
axes[0, 0].set_ylabel(f"time\n(mesh {main} fixed)", fontsize=11)
axes[1, 0].set_ylabel("mesh\n(phase 0.4-0.6 fixed)", fontsize=11)
W = pd.read_csv(ROOT / "time_decomp/what_changes.csv")
w = W[(W.category == CAT) & (W.hand == HAND)].iloc[0]
fig.suptitle(f"{CAT}, on the shared canonical points: time raises and lowers the values in place, "
             f"the mesh moves which points light up\n"
             f"x = contact centroid  ·  travel over a take within one mesh {w.centroid_travel:.3f}  vs  "
             f"spread between meshes at the same phase {w.mesh_centroid_spread:.3f}  (3-D canonical units)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.88])
p = ROOT / "figures/figG_canonical_axes.png"
fig.savefig(p, dpi=110)
print("saved", p, "| meshes:", meshes)
print("mesh centroid spread (2D canonical units):", float(np.linalg.norm(cen - cen.mean(0), axis=1).mean()))
tc = []
for k in range(BINS):
    v = X[(b == k) & (M.mesh_id.values == main)].mean(0)
    tc.append((xy * v[:, None]).sum(0) / max(v.sum(), 1e-9))
tc = np.stack(tc)
print("time centroid travel start->end:", float(np.linalg.norm(tc[-1] - tc[0])),
      "| max over bins:", float(max(np.linalg.norm(tc[i]-tc[j]) for i in range(BINS) for j in range(BINS))))
