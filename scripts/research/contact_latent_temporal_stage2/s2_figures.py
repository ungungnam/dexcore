#!/usr/bin/env python
"""Figures 1-5 of the report (PNG under OUT/figures).    CUDA_VISIBLE_DEVICES=5 python s2_figures.py
Figure 5 (qualitative trajectories) needs the per-dataset loaders and is drawn in one subprocess per dataset (--fig5 <ds>).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

import s2_common as S

FIG = S.OUT / "figures"
COL = {"B0": "#4C72B0", "B1": "#DD8452", "B2": "#55A868", "B2_zonly": "#8FD19E", "D0_prev": "#9E9E9E", "PERSIST": "#C44E52", "GT": "black", "hold_z0": "#B07AA1", "train_mean": "#BBBBBB"}
LAB = {"B0": "B0 direct dense", "B1": "B1 z → C", "B2": "B2 (z, r) → C", "B2_zonly": "B2 z-only path", "D0_prev": "previous D0 (25.7M, 3 seeds)", "PERSIST": "persistence C_t = s_0", "GT": "GT (grid floor)"}
MET_LAB = {"E_C": "dense error E_C", "E_C_last16": "E_C, last 16 frames", "part_hamming": "participation Hamming", "part_macro_f1": "participation macro F1 (↑)", "amount_l1": "amount L1 (norm.)",
           "centroid": "centroid error / l", "normal": "normal angle (°)", "q_rel_l1": "wrench rel. L1", "q_cos": "wrench cosine (↑)", "jitter_dense": "dense frame change", "jitter_r2": "R2 frame change"}


def get(acc, ds, model, metric):
    r = acc[(acc.dataset == ds) & (acc.model == model) & (acc.metric == metric)]
    return None if r.empty else r.iloc[0]


def fig1():
    fig, ax = plt.subplots(figsize=(11, 4.6)); ax.axis("off")
    def box(x, y, w, h, text, fc="#F3F3F3", ec="#555555", fs=8.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    rows = [("B0", 0.72, "direct dense"), ("B1", 0.42, "z-mediated"), ("B2", 0.10, "z + r mediated")]
    for m, y, sub in rows:
        box(0.01, y, 0.12, 0.17, f"{m}\n{sub}", fc="#FFFFFF", bold=True)
        box(0.16, y, 0.15, 0.17, "inputs\ns₀, G, τ₁:₆₃")
        arrow(0.31, y + 0.085, 0.345, y + 0.085)
        box(0.345, y, 0.19, 0.17, "shared temporal backbone\n768 × 8 blocks, 12 heads\nadaLN-zero FiLM on (G, s₀)\n68 M params", fc="#E8EEF7")
        arrow(0.535, y + 0.085, 0.57, y + 0.085)
        if m == "B0":
            box(0.57, y, 0.16, 0.17, "MLP head\n768-2048-2048-512\n(6.8 M)", fc="#E3F0E3")
            arrow(0.73, y + 0.085, 0.77, y + 0.085)
            box(0.77, y, 0.22, 0.17, "ρ̂₁:₆₃ → Ĉ_t = s₀ + σ_r ρ̂_t\nloss L_C", fc="#FFF4E0")
        elif m == "B1":
            box(0.57, y, 0.16, 0.17, "linear head\nẑ₁:₆₃ (64-D)\nloss L_z vs teacher z*", fc="#E3F0E3")
            arrow(0.73, y + 0.085, 0.77, y + 0.085)
            box(0.77, y, 0.22, 0.17, "D_z(ẑ_t, G_t) → Ĉ_t\nStage-1 decoder, fine-tuned\nloss L_C", fc="#FFF4E0")
        else:
            box(0.57, y, 0.16, 0.17, "two linear heads\nẑ₁:₆₃ (64), r̂₁:₆₃ (64)\nL_z on ẑ; r̂ unsupervised", fc="#E3F0E3")
            arrow(0.73, y + 0.085, 0.77, y + 0.085)
            box(0.77, y, 0.22, 0.17, "C̄_t = D_z(ẑ_t), ΔC_t = D_r(ẑ_t, r̂_t)\nĈ_t = C̄_t + ΔC_t\nL_full + ½ L_zonly + L_z + 0.01 L_res", fc="#FFF4E0")
    ax.text(0.5, 0.97, "Teacher (training only): z*_t = E_z^Stage-1(C_t^GT, G_t), frozen A3 encoder; never used at inference", ha="center", va="top", fontsize=8.5, color="#333333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Figure 1 — The three models share one temporal backbone and the same inputs; only the output pathway differs", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig1_schematic.png", dpi=130); plt.close(fig)


def fig2(acc):
    metrics = ["E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1"]
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.0 * len(metrics), 6.2))
    for i, ds in enumerate(S.DATASETS):
        for j, met in enumerate(metrics):
            ax = axes[i, j]; models = ["B0", "B1", "B2"]
            for k, m in enumerate(models):
                r = get(acc, ds, m, met)
                if r is None:
                    continue
                ax.bar(k, r.value, color=COL[m], yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], capsize=3, width=0.7, label=LAB[m] if (i == 0 and j == 0) else None)
                ax.text(k, r.ci_hi, f"{r.value:.3f}" if met != "normal" else f"{r.value:.1f}", ha="center", va="bottom", fontsize=7)
            lo = min(get(acc, ds, m, met).ci_lo for m in models if get(acc, ds, m, met) is not None); hi = max(get(acc, ds, m, met).ci_hi for m in models if get(acc, ds, m, met) is not None)
            y0, y1 = lo - 0.42 * (hi - lo) - 1e-6, hi + 0.35 * (hi - lo) + 1e-6
            refs = {ref: get(acc, ds, ref, met) for ref in ("D0_prev", "GT", "PERSIST")}
            if refs["D0_prev"] is not None and y0 < refs["D0_prev"].value < y1:
                ax.axhline(refs["D0_prev"].value, color=COL["D0_prev"], ls="--", lw=1.0, label=LAB["D0_prev"] if (i == 0 and j == 0) else None)
            fm = (lambda v: f"{v:.1f}") if met == "normal" else (lambda v: f"{v:.3f}")
            ax.text(0.5, 0.02, "  ·  ".join(f"{lab} {fm(refs[k].value)}" for k, lab in (("D0_prev", "prev. D0"), ("GT", "GT floor"), ("PERSIST", "persist.")) if refs[k] is not None and not (k == "GT" and met == "E_C")),
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=6.3, color="#444444")
            ax.set_xticks(range(3)); ax.set_xticklabels(models, fontsize=8); ax.set_title(f"{S.LABEL[ds]}: {MET_LAB[met]}", fontsize=8.5)
            ax.set_ylim(y0, y1)
    h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Figure 2 — Fixed-s₀ test performance of B0 / B1 / B2 (mean ± 95 % take-cluster bootstrap; lower is better; zoomed axes — the reference values of the previous D0, the GT grid-extraction floor\n"
                 "and the persistence predictor are printed at the bottom of each panel; dashed line: previous D0 where it falls inside the panel)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(FIG / "fig2_main_metrics.png", dpi=130); plt.close(fig)


def fig3(curves):
    panels = [("dense", "dense error ‖Ĉ_t − C_t‖"), ("z_err", "z error RMSE (standardised)"), ("part", "participation Hamming"), ("centroid", "centroid error / l"), ("wrench", "wrench rel. L1")]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 6.0))
    for i, ds in enumerate(S.DATASETS):
        for j, (c, lab) in enumerate(panels):
            ax = axes[i, j]
            models = ["B1", "B2", "hold_z0", "train_mean"] if c == "z_err" else ["B0", "B1", "B2", "PERSIST", "GT"]
            for m in models:
                k = f"{ds}|{m}|{c}"
                if k not in curves:
                    continue
                y = curves[k]; t = np.arange(1, len(y) + 1) if c == "z_err" else np.arange(len(y))
                if c != "z_err":
                    y, t = y[1:], t[1:]
                if m == "PERSIST" and c == "dense" and np.nanmax(y) > 3 * max(np.nanmax(curves.get(f"{ds}|{mm}|{c}", [0])[1:]) for mm in ("B0", "B1", "B2") if f"{ds}|{mm}|{c}" in curves):
                    continue
                ax.plot(t, y, color=COL[m], lw=1.4 if m in ("B0", "B1", "B2") else 1.0, ls="-" if m in ("B0", "B1", "B2") else "--", label={**LAB, "hold_z0": "hold z*₀", "train_mean": "train-mean z"}[m])
            if c == "dense" and (S.ds_out(ds) / "bottleneck_curves.npz").exists():
                bc = np.load(S.ds_out(ds) / "bottleneck_curves.npz")
                ax.plot(np.arange(1, S.T), bc["B1|oracle_z"], color=COL["B1"], lw=1.0, ls=":", label="B1 decoder with teacher z* (bottleneck floor)")
            ax.set_title(f"{S.LABEL[ds]}: {lab}", fontsize=8.5); ax.set_xlabel("frame t", fontsize=8); ax.grid(alpha=0.3)
            if c == "dense":
                top = max(np.nanmax(curves[f"{ds}|{mm}|dense"][1:]) for mm in ("B0", "B1", "B2") if f"{ds}|{mm}|dense" in curves)
                ax.set_ylim(-0.05, 1.3 * top)                                   # the persistence curve leaves the panel (TACO: up to 4.4)
            if i == 0 and j in (0, 1):
                ax.legend(fontsize=6, loc="lower right")
    fig.suptitle("Figure 3 — Error vs horizon (test, fixed s₀): dense error, z error of the latent-mediated models (with the hold-z*₀ and train-mean references) and structural errors", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig3_horizon.png", dpi=130); plt.close(fig)


def fig4(curves, acc):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.0))
    for i, ds in enumerate(S.DATASETS):
        for j, (c, lab) in enumerate([("jitter", "dense frame-to-frame change ‖Ĉ_{t+1} − Ĉ_t‖"), ("jitter_r2", "R2 frame-to-frame change (z-scored)")]):
            ax = axes[i, j]
            for m in ("GT", "B0", "B1", "B2"):
                k = f"{ds}|{m}|{c}"
                if k in curves:
                    ax.plot(np.arange(1, len(curves[k]) + 1), curves[k], color=COL[m], lw=1.6 if m == "GT" else 1.2, label=LAB[m])
            ax.set_title(f"{S.LABEL[ds]}: {lab}", fontsize=8.5); ax.set_xlabel("frame", fontsize=8); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
        for j, (met, lab) in enumerate([("jitter_dense", "mean dense frame change"), ("jitter_dev_r2", "|R2 frame change − GT| (per sequence)")]):
            ax = axes[i, 2 + j]
            for k, m in enumerate(["B0", "B1", "B2"]):
                r = get(acc, ds, m, met)
                if r is not None:
                    ax.bar(k, r.value, color=COL[m], yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], capsize=3, width=0.7); ax.text(k, r.ci_hi, f"{r.value:.3f}", ha="center", va="bottom", fontsize=7)
            g = get(acc, ds, "GT", met)
            if g is not None and met == "jitter_dense":
                ax.axhline(g.value, color="black", ls=":", label="GT")
            ax.set_xticks(range(3)); ax.set_xticklabels(["B0", "B1", "B2"]); ax.set_title(f"{S.LABEL[ds]}: {lab}", fontsize=8.5)
    fig.suptitle("Figure 4 — Temporal stability: predicted vs GT frame-to-frame change of the dense map and of the exact R2 vector (curves: mean over test sequences; bars: mean ± 95 % CI)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig4_stability.png", dpi=130); plt.close(fig)


def fig6():
    """Validation curves: the dense term of the three models and the z term of the latent models vs training step."""
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    ck = pd.read_csv(S.OUT / "training_summary.csv")
    for i, ds in enumerate(S.DATASETS):
        logs = {m: pd.read_csv(S.train_log_path(ds, S.run_name(m))) for m in S.MODELS if S.train_log_path(ds, S.run_name(m)).exists()}
        for j, (key, lab) in enumerate([("dense", "validation dense term (MSE of ρ)"), ("E_C", "validation dense error E_C"), ("L_z", "validation L_z (latent models)")]):
            ax = axes[i, j]
            for m, lg in logs.items():
                col = {"dense": "val_L_full" if "val_L_full" in lg else "val_L_C", "E_C": "val_E_C", "L_z": "val_L_z"}[key]
                if col not in lg:
                    continue
                ax.plot(lg.step, lg[col], color=COL[m], lw=1.4, label=LAB[m])
                r = ck[(ck.dataset == ds) & (ck.model == m)]
                if len(r):
                    bs = int(r.iloc[0].best_step); v = lg.loc[lg.step == bs, col]
                    if len(v):
                        ax.scatter([bs], [v.iloc[0]], color=COL[m], s=45, zorder=5, edgecolor="k", linewidth=0.6)
            ax.set_xscale("log"); ax.set_xlabel("training step (log)", fontsize=8); ax.set_title(f"{S.LABEL[ds]}: {lab}", fontsize=8.5); ax.grid(alpha=0.3, which="both")
            if key != "L_z":
                lo = min(lg[{"dense": "val_L_full" if "val_L_full" in lg else "val_L_C", "E_C": "val_E_C"}[key]].min() for lg in logs.values())
                ax.set_ylim(lo * 0.97, lo * 1.25)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure 6 — Validation curves (EMA weights, all 63 frames decoded). Circles: the selected step of each run (minimum of its full validation objective).\n"
                 "B0 keeps improving for tens of thousands of steps; the latent models reach their optimum at 4 k steps and plateau.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "fig6_validation_curves.png", dpi=130); plt.close(fig)


def fig5(ds):
    import torch
    from s2_data import S2Data
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = S2Data(ds, dev, need_masks=False, teacher=True)
    M = {m: np.load(S.metrics_path(ds, S.run_name(m)), allow_pickle=True) for m in ("B0", "B1", "B2")}
    P = {m: np.load(S.preds_path(ds, S.run_name(m)))["pred"][:, 0].astype(np.float32) for m in ("B0", "B1", "B2")}
    ex = M["B0"]["example"]; E = {m: M[m]["E_C"][:, 0] for m in M}
    comp = lambda z: np.nanmean(np.stack([z["part_hamming"][:, 0], z["amount_l1"][:, 0], z["centroid"][:, 0], z["normal"][:, 0] / 90.0]), 0)
    cases = {"B1 improves most over B0 (structural composite)": int(np.nanargmax(comp(M["B0"]) - comp(M["B1"]))),
             "r improves B2 most over B1 (dense error)": int(np.nanargmax(E["B1"] - E["B2"])),
             "failure case (largest dense error of the best model)": int(np.nanargmax(np.minimum(np.minimum(E["B0"], E["B1"]), E["B2"])))}
    frames = [0, 4, 8, 16, 32, 48, 63]; sel_rows = []
    for ci, (case, i) in enumerate(cases.items()):
        n = int(ex[i]); Pt = (data.geo.X_top[data.geo.geo_index[n]] + data.geo.X_bot[data.geo.geo_index[n]]).cpu().numpy()
        Pc = Pt - Pt.mean(0); _, _, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
        rows = [("GT", data.C[n].cpu().numpy())] + [(m, P[m][i]) for m in ("B0", "B1", "B2")]
        fig, axes = plt.subplots(len(rows) + 1, len(frames), figsize=(2.3 * len(frames), 2.1 * (len(rows) + 1)))
        for r_, (lab, Cm) in enumerate(rows):
            for c_, t in enumerate(frames):
                ax = axes[r_, c_]; ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(Cm[t], 0, 1), cmap="viridis", s=5, vmin=0, vmax=1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if c_ == 0:
                    ax.set_ylabel(lab if lab == "GT" else f"{lab}\nE_C {E[lab][i]:.2f}", fontsize=8)
                if r_ == 0:
                    ax.set_title(f"t = {t}", fontsize=8)
                if r_ > 0:
                    ax.text(0.02, 0.02, f"‖err‖ {np.linalg.norm(Cm[t] - rows[0][1][t]):.2f}", transform=ax.transAxes, fontsize=6, color="white")
        ax = axes[len(rows), 0]; a_true = M["B0"]["a_true"][i]
        ax.imshow(np.concatenate([a_true] + [M[m]["r2_a"][i, 0] for m in ("B0", "B1", "B2")], 1).T, aspect="auto", cmap="Greys", interpolation="nearest"); ax.set_title("participation: GT | B0 | B1 | B2", fontsize=7); ax.set_yticks([]); ax.set_xlabel("frame", fontsize=7)
        ax = axes[len(rows), 1]
        for m in ("B0", "B1", "B2"):
            ax.plot(M[m]["curve_dense"][i, 0], color=COL[m], lw=1.0, label=m)
        ax.set_title("dense error vs t", fontsize=7); ax.legend(fontsize=6)
        ax = axes[len(rows), 2]; mt = data.m[n].cpu().numpy().sum(1); ax.plot(mt, "k", label="GT")
        for m in ("B0", "B1", "B2"):
            ax.plot(M[m]["r2_m"][i, 0].astype(np.float32).sum(1), color=COL[m], lw=1.0, label=m)
        ax.set_title("total contact amount", fontsize=7)
        ax = axes[len(rows), 3]
        for m in ("B0", "B1", "B2"):
            ax.plot(M[m]["curve_wrench"][i, 0], color=COL[m], lw=1.0)
        ax.set_title("wrench rel. L1 vs t", fontsize=7)
        ax = axes[len(rows), 4]
        for m in ("B0", "B1", "B2"):
            ax.plot(M[m]["curve_jitter"][i, 0], color=COL[m], lw=1.0)
        ax.plot(np.linalg.norm(np.diff(rows[0][1], axis=0), axis=-1), "k", lw=1.0); ax.set_title("frame-to-frame change (black: GT)", fontsize=7)
        for c_ in range(5, len(frames)):
            axes[len(rows), c_].axis("off")
        meta = data.meta.iloc[n]
        fig.suptitle(f"Figure 5{'abc'[ci]} ({S.LABEL[ds]}): {case} — test example {n} ({getattr(meta, 'category', '')}, {getattr(meta, 'hand', '')}); E_C B0 {E['B0'][i]:.2f} / B1 {E['B1'][i]:.2f} / B2 {E['B2'][i]:.2f}", fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); name = f"fig5{'abc'[ci]}_{ds}.png"; fig.savefig(FIG / name, dpi=110); plt.close(fig)
        sel_rows.append(dict(dataset=ds, case=case, example=n, E_C_B0=float(E["B0"][i]), E_C_B1=float(E["B1"][i]), E_C_B2=float(E["B2"][i]), figure=name))
    pd.DataFrame(sel_rows).to_csv(FIG / f"fig5_selection_{ds}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig5", choices=S.DATASETS, help="draw Figure 5 of one dataset (internal)"); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.fig5:
        fig5(a.fig5); return
    acc = pd.read_csv(S.OUT / "main_metrics.csv"); curves = dict(np.load(S.OUT / "temporal_curves.npz"))
    fig1(); fig2(acc); fig3(curves); fig4(curves, acc); fig6()
    for ds in S.DATASETS:
        subprocess.run([sys.executable, __file__, "--fig5", ds], check=True)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
