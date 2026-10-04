#!/usr/bin/env python
"""Figures 1-5 of the dynamic-contact report, all from the CSVs in time_decomp/dynamic_contact/.
  fig1_four_axes.png        Exp 1: time/take/mesh/action under raw, mass, pattern, centroid
                            (per group, each metric normalised by the group's cross-take level)
  fig2_lag_curves.png       Exp 2: distance vs lag, per group, four metrics (normalised by the
                            phase-matched cross-take level, which is drawn as the horizontal line)
  fig3_transitions.png      Exp 2: frame-to-frame change distribution + top-k% share
  fig4_prediction.png       Exp 3: input ablation, held-out take vs held-out mesh
  fig5_representation.png   Exp 4: error vs complexity for static / factorized / sparse / dense
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dc_common as C

FIG = C.OUT / "figures"
GROUPS = [C.gname(*g) for g in C.GROUPS]
SHORT = {g: g.replace("__target__", "/").replace("__tool__", "/").replace("__obj__", "/") for g in GROUPS}
AXES = ["time"] + C.CROSS_AXES
AXCOL = {"time": "#c0392b", "take": "#7f8c8d", "mesh": "#2980b9", "action": "#27ae60", "floor": "#bdc3c7",
         "subject": "#2980b9", "take_same_subject": "#8e44ad"}
AX2 = C.AX2
SPLIT2 = C.SPLIT2
METRICS4 = [("raw", "raw L2"), ("mass_abs", "amount |Δmass|"), ("pattern_L2", "normalised pattern L2"), ("centroid", "centroid shift")]


def fig1():
    L = pd.read_csv(C.OUT / "exp1" / "four_axes_long.csv")
    L = L[(L.variant == "all") & (L.stat == "mean")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=False)
    for ax, (met, title) in zip(axes, METRICS4):
        d = L[L.metric == met]
        labelled = set()
        for i, g in enumerate(GROUPS):
            gd = d[d.group == g].set_index("axis")
            ref = gd.loc["take", "value"]
            for j, axn in enumerate(AXES):
                if axn not in gd.index:
                    continue
                v, lo, hi = gd.loc[axn, ["value", "lo", "hi"]] / ref
                ax.errorbar(i + (j - (len(AXES) - 1) / 2) * (0.7 / len(AXES)), v, yerr=[[v - lo], [hi - v]], fmt="o", ms=4,
                            color=AXCOL.get(axn, "#555"), ecolor=AXCOL.get(axn, "#555"), capsize=2,
                            label=axn if axn not in labelled else None)
                labelled.add(axn)
        ax.axhline(1, color="0.5", lw=0.8, ls="--")
        ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([SHORT[g] for g in GROUPS], rotation=45, ha="right", fontsize=8)
        ax.set_title(title); ax.set_ylabel("distance / cross-take distance (same metric)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Fig. 1  {' / '.join(AXES)} pair distances, per metric, normalised by the phase-matched cross-take level "
                 "(mean over pairs, 95% sequence-bootstrap CI)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_four_axes.png", dpi=130); plt.close(fig)


def fig2():
    CU = pd.read_csv(C.OUT / "lag_curves.csv")
    L = pd.read_csv(C.OUT / "exp1" / "four_axes_long.csv")
    L = L[(L.variant == "all") & (L.stat == "mean") & (L.axis.isin(["take", AX2, "floor"]))]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    cmap = plt.get_cmap("tab20" if len(GROUPS) > 10 else "tab10")
    for row, var in enumerate(["touching", "all"]):
        for ax, (met, title) in zip(axes[row], METRICS4):
            d = CU[(CU.variant == var) & (CU.metric == met) & (CU.stat == "mean")]
            for i, g in enumerate(GROUPS):
                gd = d[d.group == g].sort_values("lag_frames")
                refv = L[(L.group == g) & (L.axis == "take") & (L.metric == met)].value
                if len(gd) == 0 or len(refv) == 0:
                    continue
                ref = refv.iloc[0]
                ax.plot(gd.lag_frames, gd.value / ref, "-o", ms=3, color=cmap(i), label=SHORT[g])
                ax.fill_between(gd.lag_frames, gd.lo / ref, gd.hi / ref, color=cmap(i), alpha=0.12)
            ax.axhline(1, color="k", lw=0.8, ls="--")
            refs = []
            for g in GROUPS:          # groups without the second axis (e.g. a single instance) are skipped
                a2 = L[(L.group == g) & (L.axis == AX2) & (L.metric == met)].value
                tk = L[(L.group == g) & (L.axis == "take") & (L.metric == met)].value
                if len(a2) and len(tk):
                    refs.append(a2.iloc[0] / tk.iloc[0])
            if refs:
                ax.axhline(float(np.mean(refs)), color="k", lw=0.8, ls=":")
            ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
            ax.set_xlabel("lag [original frames at 30 fps]"); ax.set_ylabel(f"{title} / cross-take level")
            ax.set_title(f"{title}  ({var} frames)", fontsize=10); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Fig. 2  Distance between frames t and t+lag within one take, normalised by the phase-matched cross-take distance "
                 f"(dashed = cross-take, dotted = macro cross-{AX2} level)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig2_lag_curves.png", dpi=130); plt.close(fig)


def fig3():
    PS = pd.read_csv(C.OUT / "exp2" / "per_sequence.csv")
    TS = pd.read_csv(C.OUT / "transition_stats.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    cmap = plt.get_cmap("tab20" if len(GROUPS) > 10 else "tab10")
    # (a) distribution of per-take normalised delta: pool frames? we only have per-take summaries; use per-take mean/max/p90
    ax = axes[0]
    for i, g in enumerate(GROUPS):
        d = PS[PS.group == g]
        ax.hist(d.delta_raw_max / d.delta_raw_mean, bins=np.linspace(1, 12, 45), histtype="step", color=cmap(i), label=SHORT[g], density=True)
    ax.set_xlabel("max frame-to-frame change / take mean"); ax.set_ylabel("density over takes"); ax.set_title("(a) largest jump relative to the take's mean change")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    # (b) share of sum(delta) carried by top-k% frames vs iid null
    ax = axes[1]
    for k, mk in zip((5, 10, 20), ("o", "s", "^")):
        obs = [TS[(TS.group == g) & (TS.quantity == f"share_raw_top{k}") & (TS.stat == "mean")].value.iloc[0] for g in GROUPS]
        lo = [TS[(TS.group == g) & (TS.quantity == f"share_raw_top{k}") & (TS.stat == "mean")].lo.iloc[0] for g in GROUPS]
        hi = [TS[(TS.group == g) & (TS.quantity == f"share_raw_top{k}") & (TS.stat == "mean")].hi.iloc[0] for g in GROUPS]
        nul = [TS[(TS.group == g) & (TS.quantity == f"null_top{k}")].value.iloc[0] for g in GROUPS]
        x = np.arange(len(GROUPS))
        ax.errorbar(x, obs, yerr=[np.array(obs) - lo, np.array(hi) - obs], fmt=mk + "-", color=AXCOL["time"], ms=4, capsize=2, label=f"observed top {k}%")
        ax.plot(x, nul, mk + "--", color="0.4", ms=4, label=f"iid half-normal null top {k}%")
        ax.axhline(k / 100, color="0.8", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[g] for g in GROUPS], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of Σ|ΔX| over the take"); ax.set_title("(b) how concentrated is the temporal change?"); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    # (c) same for pattern and mass
    ax = axes[2]
    for met, col in (("raw", AXCOL["time"]), ("mass_abs", AXCOL["mesh"]), ("pattern_L2", AXCOL["action"])):
        obs = [TS[(TS.group == g) & (TS.quantity == f"share_{met}_top10") & (TS.stat == "mean")].value.iloc[0] for g in GROUPS]
        ax.plot(np.arange(len(GROUPS)), obs, "o-", color=col, ms=4, label=f"{met}: top 10% share")
    nul = [TS[(TS.group == g) & (TS.quantity == "null_top10")].value.iloc[0] for g in GROUPS]
    ax.plot(np.arange(len(GROUPS)), nul, "s--", color="0.4", ms=4, label="iid null top 10%")
    ax.set_xticks(np.arange(len(GROUPS))); ax.set_xticklabels([SHORT[g] for g in GROUPS], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("share of Σ|Δ| carried by the top 10% frames"); ax.set_title("(c) concentration by metric"); ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.suptitle("Fig. 3  Frame-to-frame contact change at 30 fps: is the temporal variation spread out or concentrated in a few transition frames?", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig3_transitions.png", dpi=130); plt.close(fig)


def fig4():
    A = pd.read_csv(C.OUT / "prediction_ablation.csv")
    models = [("base_cat_mean", "category mean"), ("base_mesh_mean", "mesh mean"), ("mlp_A_geometry", "A geometry"),
              ("mlp_B_current_state", "B +current state"), ("mlp_C_local_context", "C +local ±8f"),
              ("mlp_Dc_coarse_window", "Dc +window (8 states)"), ("mlp_D_full_window", "D +window (32 states)"),
              ("base_prev_frame", "previous frame (oracle)"), ("oracle_seq_mean", "take mean (oracle)")]
    mets = [("raw", "raw L2"), ("mass_abs", "amount |Δmass|"), ("pattern_L2", "pattern L2"), ("zero_f1", "zero-contact F1")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    cmap = plt.get_cmap("tab20" if len(GROUPS) > 10 else "tab10")
    for row, split in enumerate(["take", SPLIT2]):
        for ax, (met, title) in zip(axes[row], mets):
            for i, g in enumerate(GROUPS):
                vals, los, his = [], [], []
                if len(A[(A.group == g) & (A.split == split)]) == 0:      # e.g. no second split for a single-instance group
                    continue
                for m, _ in models:
                    r = A[(A.group == g) & (A.split == split) & (A.model == m)]
                    ref = A[(A.group == g) & (A.split == split) & (A.model == "base_cat_mean")]
                    if len(r) == 0 or len(ref) == 0:
                        vals.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                    if met == "zero_f1":
                        vals.append(r[met].iloc[0]); los.append(np.nan); his.append(np.nan)
                    else:
                        vals.append(r[met].iloc[0] / ref[met].iloc[0]); los.append(r[f"{met}_lo"].iloc[0] / ref[met].iloc[0]); his.append(r[f"{met}_hi"].iloc[0] / ref[met].iloc[0])
                ax.plot(range(len(models)), vals, "-o", ms=3, color=cmap(i), label=SHORT[g])
            ax.set_xticks(range(len(models))); ax.set_xticklabels([n for _, n in models], rotation=40, ha="right", fontsize=7)
            ax.set_title(f"{title} — held-out {split}", fontsize=10); ax.grid(alpha=0.3)
            ax.set_ylabel("error / category-mean error" if met != "zero_f1" else "F1 (zero class)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Fig. 4  Predicting the per-frame contact vector from geometry + object trajectory (same MLP, inputs differ); "
                 "errors relative to the category-mean baseline", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig4_prediction.png", dpi=130); plt.close(fig)


def fig5():
    A = pd.read_csv(C.OUT / "representation_ablation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cmap = plt.get_cmap("tab20" if len(GROUPS) > 10 else "tab10")
    # (a) oracle: error vs compression, per family, macro over groups
    ax = axes[0]
    O = A[A.kind == "oracle"]
    fam = {"static": ["static_seq_mean", "static_window_mean"], "factorized": ["factor_rank1_mass", "factor_rank1_lsq", "factor_rank2_lsq", "factor_rank3_lsq", "factor_rank4_lsq"],
           "sparse (hold)": [f"sparse_hold_K{k}" for k in (1, 2, 3, 4, 6, 8, 12, 16)], "sparse (interp)": [f"sparse_interp_K{k}" for k in (1, 2, 3, 4, 6, 8, 12, 16)],
           "dense": ["dense"]}
    ref = O[O.representation == "static_seq_mean"].set_index("group").raw
    for (name, reps), mk in zip(fam.items(), "osd^*"):
        xs, ys = [], []
        for rep in reps:
            d = O[O.representation == rep].set_index("group")
            if len(d) == 0:
                continue
            xs.append(d.compression.mean()); ys.append((d.raw / ref.loc[d.index]).mean())
        ax.plot(xs, ys, "-" + mk, label=name, ms=6)
    ax.set_xscale("log"); ax.set_xlabel("parameters / (512 × T)  (compression ratio)"); ax.set_ylabel("raw L2 / take-mean error (macro over groups)")
    ax.set_title("(a) oracle fits: error vs complexity, macro over groups"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # (b) sparse: error vs K per group (hold)
    ax = axes[1]
    for i, g in enumerate(GROUPS):
        d = O[(O.group == g) & O.representation.str.startswith("sparse_hold_K")].copy()
        d["K"] = d.representation.str.extract(r"K(\d+)").astype(int); d = d.sort_values("K")
        ax.plot(d.K, d.raw / ref.loc[g], "-o", ms=3, color=cmap(i), label=SHORT[g])
        r1 = O[(O.group == g) & (O.representation == "factor_rank1_lsq")].raw.iloc[0] / ref.loc[g]
        ax.axhline(r1, color=cmap(i), lw=0.6, ls=":")
    Tps = pd.read_csv(C.OUT / "exp4" / "per_sequence.csv"); Tps = Tps[Tps.representation == "dense"]["T"]
    ax.set_xlabel(f"number of held states K (per take of {int(Tps.min())}-{int(Tps.max())} sampled frames, median {int(Tps.median())})"); ax.set_ylabel("raw L2 / take-mean error"); ax.set_title("(b) sparse states: error vs K (dotted = rank-1 factorized)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    # (c) predicted representations, held-out take vs mesh (macro over groups), relative to mesh-mean baseline
    ax = axes[2]
    P = A[A.kind == "predicted"]
    reps = [("base_mesh_mean", "mesh mean"), ("pred_static_A", "static (A)"), ("pred_sparse_K1", "static (take mean of C)"), ("pred_factor_AxC", "factor S(A)·a(C)"), ("pred_factor_CxC", "factor S(C̄)·a(C)"),
            ("pred_sparse_K2", "sparse K=2 (C)"), ("pred_sparse_K4", "sparse K=4 (C)"), ("pred_sparse_K8", "sparse K=8 (C)"),
            ("pred_dense_B", "dense (B)"), ("pred_dense_C", "dense (C)"), ("pred_dense_D", "dense (D)"), ("oracle_seq_mean", "take mean (oracle)")]
    for split, mk in (("take", "o"), (SPLIT2, "s")):
        ys, lo, hi = [], [], []
        for rep, _ in reps:
            d = P[(P.split == split) & (P.representation == rep)].set_index("group")
            r = P[(P.split == split) & (P.representation == "base_mesh_mean")].set_index("group").raw
            d = d[d.index.isin(r.index)]
            if len(d) == 0:
                ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            ys.append((d.raw / r.loc[d.index]).mean()); lo.append((d.raw_lo / r.loc[d.index]).mean()); hi.append((d.raw_hi / r.loc[d.index]).mean())
        ax.errorbar(range(len(reps)), ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt=mk + "-", capsize=2, label=f"held-out {split}")
    ax.set_xticks(range(len(reps))); ax.set_xticklabels([n for _, n in reps], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("raw L2 / mesh-mean error (macro over groups)"); ax.set_title("(c) predicted representations"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Fig. 5  Representation trade-off: static / factorized / sparse-state / dense", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig5_representation.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    import sys
    FIG.mkdir(exist_ok=True)
    which = sys.argv[1:] or ["1", "2", "3", "4", "5"]
    for w in which:
        {"1": fig1, "2": fig2, "3": fig3, "4": fig4, "5": fig5}[w]()
        print("fig", w, "done")
