"""Figure D: distributions of pair distances by pair type (A same verb+same mesh, B same verb+
different mesh, C different verb) for five representations, from contact_pair_distances.csv.
Panels: baselineA (chamfer_scalednorm, own log axis), normalized, random_perm, aligned, dino
(L2 on X_soft, shared axis). Each panel annotated with mean(B)/mean(A) and mean(B)/mean(C).
Outputs figures/figD_<role>_<hand>.png (all categories pooled) and figD_<cat>_<role>_<hand>.png
for the six study categories.
Run: python scripts/figD_pair_distances.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import FIG, ROOT, SIX

REPS = [("baselineA", "chamfer_scalednorm", "Baseline A: Chamfer (scaled-norm, log)"),
        ("normalized", "normalized_l2", "normalized: L2 on X_soft"),
        ("random_perm", "random_perm_l2", "random_perm: L2 on X_soft"),
        ("aligned", "aligned_l2", "aligned: L2 on X_soft"),
        ("dino", "dino_l2", "dino: L2 on X_soft")]
PT = [("A", "A: same verb, same mesh", "#4c72b0"), ("B", "B: same verb, diff. mesh", "#dd8452"),
      ("C", "C: different verb", "#55a868")]


def panel(ax, d, col, log):
    data, means = [], {}
    for k, _lab, _c in PT:
        v = d.loc[d.pair_type == k, col].dropna().values
        if log:
            v = v[v > 0]
        data.append(v)
        means[k] = float(np.mean(v)) if len(v) else np.nan
    pos = [3, 2, 1]
    ok = [i for i, v in enumerate(data) if len(v) >= 2]
    if ok:
        parts = ax.violinplot([np.log10(data[i]) if log else data[i] for i in ok], positions=[pos[i] for i in ok],
                              vert=False, showmeans=True, showextrema=False, widths=0.8)
        for body, i in zip(parts["bodies"], ok):
            body.set_facecolor(PT[i][2]); body.set_edgecolor("none"); body.set_alpha(0.75)
        parts["cmeans"].set_color("#222")
    ax.set_yticks(pos); ax.set_yticklabels([f"{k} (n={len(v)})" for (k, _, _), v in zip(PT, data)], fontsize=8)
    ax.set_ylim(0.4, 3.6)
    if log:
        lo, hi = ax.get_xlim()
        ticks = np.arange(np.floor(lo), np.ceil(hi) + 1)
        ax.set_xticks(ticks); ax.set_xticklabels([f"1e{int(t)}" for t in ticks], fontsize=8)
        ax.set_xlabel("Chamfer distance (scaled-norm units), log10", fontsize=8)
    else:
        ax.set_xlabel("L2 distance on X_soft", fontsize=8)
    ba = means["B"] / means["A"] if means["A"] else np.nan
    bc = means["B"] / means["C"] if means["C"] else np.nan
    ax.text(0.98, 0.04, f"mean B/A = {ba:.2f}\nmean B/C = {bc:.2f}", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="#999", alpha=0.9))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#ddd", lw=0.6)
    ax.set_axisbelow(True)
    return ba, bc


def figure(d, title, out):
    fig, axes = plt.subplots(1, len(REPS), figsize=(4.0 * len(REPS), 3.6))
    res = {}
    xmax = np.nanpercentile(pd.concat([d[c] for _, c, _ in REPS[1:]]), 99.5)
    for ax, (name, col, lab) in zip(axes, REPS):
        res[name] = panel(ax, d, col, log=(name == "baselineA"))
        ax.set_title(lab, fontsize=9)
        if name != "baselineA":
            ax.set_xlim(0, xmax * 1.05)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return res


def main():
    cols = ["category", "role", "hand", "pair_type"] + [c for _, c, _ in REPS]
    d = pd.read_csv(ROOT / "contact_pair_distances.csv", usecols=cols)
    rows = []
    for (role, hand), g in d.groupby(["role", "hand"]):
        cats = g.category.nunique()
        res = figure(g, f"Figure D: pair distances, all {cats} categories pooled, role={role}, hand={hand} "
                        f"(n pairs A/B/C = {(g.pair_type=='A').sum()}/{(g.pair_type=='B').sum()}/{(g.pair_type=='C').sum()})",
                     FIG / f"figD_{role}_{hand}.png")
        rows += [dict(category="ALL", role=role, hand=hand, rep=k, mean_B_over_A=v[0], mean_B_over_C=v[1]) for k, v in res.items()]
    for cat, role, hand in SIX:
        g = d[(d.category == cat) & (d.role == role) & (d.hand == hand)]
        res = figure(g, f"Figure D: pair distances, {cat}/{role}, hand={hand} "
                        f"(n pairs A/B/C = {(g.pair_type=='A').sum()}/{(g.pair_type=='B').sum()}/{(g.pair_type=='C').sum()})",
                     FIG / f"figD_{cat}_{role}_{hand}.png")
        rows += [dict(category=cat, role=role, hand=hand, rep=k, mean_B_over_A=v[0], mean_B_over_C=v[1]) for k, v in res.items()]
    t = pd.DataFrame(rows)
    t.to_csv(FIG / "figD_ratios.csv", index=False)
    print(t.pivot_table(index=["category", "role", "hand"], columns="rep", values=["mean_B_over_A", "mean_B_over_C"]).round(2).to_string())


if __name__ == "__main__":
    main()
