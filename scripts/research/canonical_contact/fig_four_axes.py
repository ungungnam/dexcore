#!/usr/bin/env python
"""One picture of the four axes of a TACO contact map, on a single scale."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
D = pd.read_csv(ROOT / "time_decomp/four_axes.csv")
D["grp"] = D.category + " " + D.hand
D = D.sort_values("action", ascending=False).reset_index(drop=True)

SERIES = [("time", "#2f6fd0", "time  (same mesh, same take, start vs end)"),
          ("take", "#12355b", "take  (same mesh, same verb, other take, phase matched)"),
          ("mesh", "#e8663c", "mesh  (other mesh, same verb, phase matched)"),
          ("action", "#7a3fa8", "action  (same mesh, other verb, phase matched)")]
fig, ax = plt.subplots(figsize=(11.2, 4.9))
w, xs = 0.2, np.arange(len(D))
for k, (col, c, lab) in enumerate(SERIES):
    ax.bar(xs + (k - 1.5) * w, D[col], w, color=c, label=lab, zorder=3)
floor = D.floor.mean()
ax.axhline(floor, ls="--", lw=1.1, color="#777777", zorder=2)
ax.text(len(D) - 0.42, floor + 0.30, f"same-moment floor {D.floor.min():.2f}-{D.floor.max():.2f}",
        fontsize=9, color="#555555", ha="right")
for i, r in D.iterrows():
    if r.action < 0.02:
        ax.text(i + 1.5 * w, 0.06, "0", fontsize=9, ha="center", color="#7a3fa8")
ax.set_xticks(xs); ax.set_xticklabels(D.grp, fontsize=11)
ax.set_ylabel("pure component, L2 between frame-level\ncanonical contact vectors (RMS)", fontsize=10)
ax.set_title("What moves a TACO contact map: time and geometry do, the action barely does\n"
             "mesh and action are both increments over the same take-level baseline, so they compare directly",
             fontsize=12.5, loc="left")
ax.legend(fontsize=9.5, frameon=False, ncol=2, loc="upper right")
ax.set_ylim(0, 5.2); ax.grid(axis="y", color="#e6e6e6", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
p = ROOT / "figures/figH_four_axes.png"
fig.savefig(p, dpi=150)
print("saved", p)
print(D[["grp", "time", "take", "mesh", "action", "n_verbs", "meshes_multiverb"]].round(3).to_string(index=False))
