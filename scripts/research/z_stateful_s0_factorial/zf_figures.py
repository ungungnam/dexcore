#!/usr/bin/env python
"""Figures of the report (PNG under OUT/figures).    python zf_figures.py        (CPU only; the qualitative figures load the canonical geometry per dataset)
  fig1 2 x 2 architecture | fig2 main 2 x 2 results | fig3 contact error vs horizon | fig4 z error vs horizon | fig5 z generalisation gap |
  fig6 early frames | fig7 qualitative TACO | fig8 qualitative ARCTIC | fig9 training curves | fig10 stateful check (transition from true states)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

import zf_common as Z

FIG = Z.OUT / "figures"
COL = {"M00": "#DD8452", "M01": "#C44E52", "M10": "#55A868", "M11": "#8172B3", "D0": "#4C72B0", "PERSIST": "#999999", "GT": "black"}
LAB = {"M00": "M00 whole z + absolute", "M01": "M01 whole z + s₀-residual", "M10": "M10 stateful z + absolute", "M11": "M11 stateful z + s₀-residual", "D0": "D0 direct dense", "PERSIST": "persistence C_t = s₀"}
MET_LAB = {"E_C": "dense error E_C", "part_hamming": "participation Hamming", "centroid": "centroid error / l", "normal": "normal angle (°)", "q_rel_l1": "wrench rel. L1"}
ALL = ["M00", "M01", "M10", "M11", "D0"]


def acc(ds):
    p = Z.OUT / f"{ds}_metrics.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["dataset", "model", "metric", "value", "ci_lo", "ci_hi"])


def row(A, m, metric):
    r = A[(A.model == m) & (A.metric == metric)]
    return None if r.empty else r.iloc[0]


def fig1():
    fig, ax = plt.subplots(figsize=(13, 6.6)); ax.axis("off")
    def box(x, y, w, h, text, fc="#F3F3F3", ec="#555555", fs=8.3, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=fc, ec=ec, lw=1.0, ls=ls)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)
    ax.text(0.59, 0.955, "contact decoding", ha="center", fontsize=10, fontweight="bold"); ax.text(0.40, 0.905, "B0  absolute:  Ĉ_t = D_abs(ẑ_t, G_t)", ha="center", fontsize=9)
    ax.text(0.79, 0.905, "B1  s₀-preserving:  Ĉ_t = s₀ + D_Δ(s₀, ẑ_t, G_t)", ha="center", fontsize=9)
    ax.text(0.015, 0.50, "latent temporal formulation", rotation=90, va="center", fontsize=10, fontweight="bold")
    rows = [(0.50, "A0  whole-sequence z\n(s₀, G, τ) → ẑ₁:₆₃\nin one pass\n768 × 8-block backbone\n68.0 M", ("M00", "M01")), (0.10, "A1  stateful z\nẑ₀ = I(s₀, G)\nẑ_t+1 = ẑ_t + F(ẑ_t, G, τ_t..t+4)\nrolled out 63 steps\non its own output\nI: 6.1 M, F: 55.7 M", ("M10", "M11"))]
    for y, lab, (ma, mb) in rows:
        box(0.045, y, 0.185, 0.34, lab, fc="#E8EEF7", fs=8.2)
        box(0.25, y, 0.30, 0.34, f"{ma}\n\nẑ_t and the frame's geometry tokens\n→ 6-block point decoder (8.0 M)\n→ the whole map Ĉ_t", fc="#FFF4E0" if ma == "M00" else "#EAF4EA", bold=False, fs=9)
        box(0.64, y, 0.30, 0.34, f"{mb}\n\nẑ_t, geometry tokens and s₀ at every point\n→ the same decoder, output starts at 0\n→ the change ΔĈ_t;  Ĉ_t = s₀ + ΔĈ_t", fc="#FBE9E7" if mb == "M01" else "#EFEAF6", fs=9)
    ax.text(0.5, 0.035, "M00 = the previous follow-up's model (reused).  Losses: L_C + L_z  (+ L_z0 on ẑ₀ for the stateful row).  Teacher latents z* = frozen Stage-1 encoder on GT frames: training targets only.\n"
                        "D0 (external reference): the direct dense model, backbone → MLP head → Ĉ_t = s₀ + σ_r ρ̂_t.", ha="center", va="center", fontsize=8.5, color="#333333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Figure 1 — The 2 × 2 design: how z evolves through time × how z is turned back into contact", fontsize=10.5, y=0.995)
    fig.tight_layout(); fig.savefig(FIG / "fig1_design.png", dpi=130); plt.close(fig)


def fig2():
    mets = ["E_C", "part_hamming", "centroid", "normal", "q_rel_l1"]
    fig, axes = plt.subplots(2, len(mets), figsize=(16, 6.4))
    for i, ds in enumerate(Z.DATASETS):
        A = acc(ds); ms = [m for m in ALL if row(A, m, "E_C") is not None]
        for j, met in enumerate(mets):
            ax = axes[i, j]; vals = [row(A, m, met) for m in ms]
            for k, (m, r) in enumerate(zip(ms, vals)):
                ax.bar(k, r.value, color=COL[m], width=0.74); ax.errorbar(k, r.value, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], color="k", capsize=3, lw=1)
                ax.text(k, r.ci_hi, f"{r.value:.3f}" if r.value < 10 else f"{r.value:.1f}", ha="center", va="bottom", fontsize=7)
            if vals:
                lo = min(v.ci_lo for v in vals); hi = max(v.ci_hi for v in vals); ax.set_ylim(max(0, lo - 0.9 * (hi - lo)), hi + 0.25 * (hi - lo))
            ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, fontsize=8); ax.set_title(f"{Z.LABEL[ds]}: {MET_LAB[met]}", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Figure 2 — Main 2 × 2 results and the direct model D0 (test; lower is better; whiskers: 95 % take-cluster bootstrap; y axes do not start at 0)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig2_main_2x2.png", dpi=130); plt.close(fig)


def fig3(curves):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]
        for m in ALL:
            if f"{ds}|{m}|dense" in curves:
                ax.plot(np.arange(1, Z.T), curves[f"{ds}|{m}|dense"][1:], color=COL[m], lw=1.7 if m != "D0" else 2.0, ls="--" if m == "D0" else "-", label=LAB[m])
        for lo, hi in Z.HORIZON_BINS[:-1]:
            ax.axvline(hi + 0.5, color="#DDDDDD", lw=0.8)
        ax.set_xlabel("frame t (horizon from s₀)"); ax.set_ylabel("dense error ‖Ĉ_t − C_t‖"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Figure 3 — Contact error versus horizon (vertical lines: the bins 1–16, 17–32, 33–48, 49–63)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig3_contact_vs_horizon.png", dpi=130); plt.close(fig)


def fig4(curves):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0)); x = np.arange(1, Z.T)
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]
        for m in Z.Z_MODELS:
            if f"{ds}|{m}|z_err" in curves:
                ax.plot(x, curves[f"{ds}|{m}|z_err"], color=COL[m], lw=1.7, label=f"{m} test")
            if f"{ds}|{m}|z_err_true_z0" in curves:
                ax.plot(x, curves[f"{ds}|{m}|z_err_true_z0"], color=COL[m], lw=1.0, ls=":", label=f"{m} rolled out from the true z*₀ (analysis)")
        if f"{ds}|hold_z0|z_err" in curves:
            ax.plot(x, curves[f"{ds}|hold_z0|z_err"], color="#999999", lw=1.0, ls="--", label="hold z*₀")
        ax.set_xlabel("frame t"); ax.set_ylabel("RMSE of ẑ_t − z*_t (standardised)"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(alpha=0.3); ax.set_ylim(0, None)
        ax.legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle("Figure 4 — Latent error versus horizon: whole-sequence (M00, M01) and stateful (M10, M11) prediction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig4_z_vs_horizon.png", dpi=130); plt.close(fig)


def fig5(lat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); w = 0.26
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; L = lat[lat.dataset == ds]; ms = [m for m in Z.Z_MODELS if (L.model == m).any()]
        for k, m in enumerate(ms):
            r = L[L.model == m].iloc[0]
            for q, (col, lab, c) in enumerate((("rmse_train", "train", "#9ECAE1"), ("rmse_val", "validation", "#4292C6"), ("rmse_test", "test", "#084594"))):
                ax.bar(k + (q - 1) * w, r[col], width=w * 0.94, color=c, label=lab if k == 0 else None); ax.text(k + (q - 1) * w, r[col], f"{r[col]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"{m}\ngap {L[L.model == m].iloc[0]['gap']:+.2f}" for m in ms], fontsize=8.5); ax.set_title(f"{Z.LABEL[ds]}: RMSE of ẑ − z* (frames 1–63)", fontsize=9.5); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, ax.get_ylim()[1] * 1.18); ax.legend(fontsize=8, ncol=3, loc="upper left")
    fig.suptitle("Figure 5 — Latent generalisation: the same weights on training, validation and test sequences (gap = test − train)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig5_generalisation_gap.png", dpi=130); plt.close(fig)


def fig6(E):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2)); w = 0.16
    for i, ds in enumerate(Z.DATASETS):
        e = E[E.dataset == ds]
        for j, (q, yl, ttl) in enumerate((("E_C", "dense error ‖Ĉ_t − C_t‖", "contact error at early frames"), ("drift_mean_ratio", "mean ‖Ĉ_t − s₀‖ / mean ‖C_t − s₀‖", "movement away from s₀ relative to the data (1 = as much as the data)"))):
            ax = axes[j, i]; d = e[e.quantity == q]; ms = [m for m in ALL if (d.model == m).any()]
            for k, m in enumerate(ms):
                for a, t in enumerate(Z.EARLY_FRAMES):
                    r = d[(d.model == m) & (d.frame == t)]
                    if r.empty:
                        continue
                    r = r.iloc[0]; x = a + (k - (len(ms) - 1) / 2) * w
                    ax.bar(x, r.value, width=w * 0.94, color=COL[m], label=LAB[m] if a == 0 else None)
                    if np.isfinite(r.ci_lo):
                        ax.errorbar(x, r.value, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], color="k", capsize=2, lw=0.8)
            if q == "drift_mean_ratio":
                ax.axhline(1.0, color="k", lw=0.8, ls=":")
            ax.set_xticks(range(len(Z.EARLY_FRAMES))); ax.set_xticklabels([f"t = {t}" for t in Z.EARLY_FRAMES]); ax.set_ylabel(yl, fontsize=8.5); ax.set_title(f"{Z.LABEL[ds]}: {ttl}", fontsize=9); ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                hl = ax.get_legend_handles_labels()
    fig.legend(*hl, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.955), frameon=False)          # one legend for the four panels, clear of the bars
    fig.suptitle("Figure 6 — Early frames: absolute (M00, M10) versus s₀-preserving (M01, M11) decoding, with the direct model D0", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(FIG / "fig6_early_frames.png", dpi=130); plt.close(fig)


def fig9():
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.8)); T = pd.read_csv(Z.OUT / "training_summary.csv")
    panels = [("val_L_C", "validation dense loss L_C"), ("val_L_z", "validation L_z (frames 1–63)"), ("train_L_z", "training L_z (batch, at validation steps)"), ("val_L_z0_frame0", "validation L_z0 (stateful: ẑ₀ vs z*₀)")]
    for i, ds in enumerate(Z.DATASETS):
        logs = {m: pd.read_csv(Z.model_train_log(ds, m)) for m in ALL if Z.model_train_log(ds, m).exists() and len(T[(T.dataset == ds) & (T.model == m)])}
        for j, (col, lab) in enumerate(panels):
            ax = axes[i, j]
            for m, lg in logs.items():
                if col not in lg:
                    continue
                ax.plot(lg.step, lg[col], color=COL[m], lw=1.4, ls="--" if m == "D0" else "-", label=LAB[m])
                bs = int(T[(T.dataset == ds) & (T.model == m)].iloc[0].best_step); v = lg.loc[lg.step == bs, col]
                if len(v):
                    ax.scatter([bs], [v.iloc[0]], color=COL[m], s=42, zorder=5, edgecolor="k", linewidth=0.6)
            ax.set_xscale("log"); ax.set_xlabel("training step (log)", fontsize=8); ax.set_title(f"{Z.LABEL[ds]}: {lab}", fontsize=8.5); ax.grid(alpha=0.3, which="both")
            if col == "val_L_C" and logs:
                lo = min(lg[col].min() for lg in logs.values() if col in lg); ax.set_ylim(lo * 0.97, lo * 1.3)
            if col in ("train_L_z", "val_L_z0_frame0"):
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure 9 — Training dynamics (EMA weights at validation, all 63 frames decoded). Circles: the selected step of each run.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig9_training_curves.png", dpi=130); plt.close(fig)


def fig10(lat):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2)); w = 0.26
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; L = lat[(lat.dataset == ds) & lat.model.isin(["M10", "M11"])]
        if L.empty or "rmse_from_true_h1" not in L:
            continue
        for a, h in enumerate((1, 4, 8)):
            ax.bar(a - w, L.iloc[0][f"rmse_hold_h{h}"], width=w * 0.94, color="#999999", label="hold z*_t (persistence)" if a == 0 else None); ax.text(a - w, L.iloc[0][f"rmse_hold_h{h}"], f"{L.iloc[0][f'rmse_hold_h{h}']:.2f}", ha="center", va="bottom", fontsize=7)
            for k, r in enumerate(L.itertuples()):
                v = getattr(r, f"rmse_from_true_h{h}"); ax.bar(a + k * w, v, width=w * 0.94, color=COL[r.model], label=f"{r.model} transition rolled h steps from the true z*_t" if a == 0 else None); ax.text(a + k * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3)); ax.set_xticklabels(["h = 1", "h = 4", "h = 8"]); ax.set_ylabel("RMSE of the latent at t + h (standardised)"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7.5)
    fig.suptitle("Figure 10 — Stateful check: the learned transition started from TRUE latents (analysis only; generation never sees them)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig10_transition_from_true_states.png", dpi=130); plt.close(fig)


def qualitative(ds):
    """Selection rules (fixed in code, applied to the per-sequence test errors):
       stateful   the largest mean gain of stateful over whole-sequence prediction, 0.5 [(M00 - M10) + (M01 - M11)]
       decoder    the largest mean gain of s_0-preserving over absolute decoding, 0.5 [(M00 - M01) + (M10 - M11)]
       failure    the largest loss of the full model against the direct model, M11 - D0"""
    import torch
    from s2_data import S2Data
    data = S2Data(ds, torch.device("cpu"), need_masks=False, teacher=False)
    mods = [m for m in ("D0", "M00", "M01", "M10", "M11") if Z.model_metrics(ds, m).exists()]
    M = {m: np.load(Z.model_metrics(ds, m), allow_pickle=True) for m in mods}; P = {m: np.load(Z.model_preds(ds, m))["pred"][:, 0].astype(np.float32) for m in mods}
    ex = M["D0"]["example"]; E = {m: M[m]["E_C"][:, 0] for m in M}
    cases = {"stateful": ("stateful z helps most", int(np.nanargmax(0.5 * ((E["M00"] - E["M10"]) + (E["M01"] - E["M11"]))))),
             "decoder": ("s₀-preserving decoding helps most", int(np.nanargmax(0.5 * ((E["M00"] - E["M01"]) + (E["M10"] - E["M11"]))))),
             "failure": ("M11 loses most against D0", int(np.nanargmax(E["M11"] - E["D0"])))}
    frames = [0, 1, 4, 8, 16, 32, 63]; sel_rows = []
    for key, (case, i) in cases.items():
        n = int(ex[i]); Pt = (data.geo.X_top[data.geo.geo_index[n]] + data.geo.X_bot[data.geo.geo_index[n]]).cpu().numpy()
        Pc = Pt - Pt.mean(0); _, _, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
        rows = [("GT", data.C[n].cpu().numpy())] + [(m, P[m][i]) for m in mods]
        fig, axes = plt.subplots(len(rows) + 1, len(frames), figsize=(2.2 * len(frames), 1.95 * (len(rows) + 1)))
        for r_, (lab, Cm) in enumerate(rows):
            for c_, t in enumerate(frames):
                ax = axes[r_, c_]; ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(Cm[t], 0, 1), cmap="viridis", s=4, vmin=0, vmax=1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if c_ == 0:
                    ax.set_ylabel(lab if lab == "GT" else f"{lab}\nE_C {E[lab][i]:.2f}", fontsize=8)
                if r_ == 0:
                    ax.set_title(f"t = {t}" + (" (s₀, given)" if t == 0 else ""), fontsize=8)
                if r_ > 0 and t > 0:
                    ax.text(0.02, 0.02, f"‖err‖ {np.linalg.norm(Cm[t] - rows[0][1][t]):.2f}", transform=ax.transAxes, fontsize=6, color="white")
        ax = axes[len(rows), 0]
        for m in mods:
            ax.plot(M[m]["curve_dense"][i, 0], color=COL[m], lw=1.0, label=m, ls="--" if m == "D0" else "-")
        ax.set_title("dense error vs t", fontsize=7); ax.legend(fontsize=5.5)
        ax = axes[len(rows), 1]; ax.plot(data.m[n].cpu().numpy().sum(1), "k")
        for m in mods:
            ax.plot(M[m]["r2_m"][i, 0].astype(np.float32).sum(1), color=COL[m], lw=1.0, ls="--" if m == "D0" else "-")
        ax.set_title("total contact amount (black: GT)", fontsize=7)
        ax = axes[len(rows), 2]
        for m in mods:
            ax.plot(M[m]["curve_wrench"][i, 0], color=COL[m], lw=1.0, ls="--" if m == "D0" else "-")
        ax.set_title("wrench rel. L1 vs t", fontsize=7)
        ax = axes[len(rows), 3]
        for m in mods:
            ax.plot(M[m]["curve_jitter"][i, 0], color=COL[m], lw=1.0, ls="--" if m == "D0" else "-")
        ax.plot(np.linalg.norm(np.diff(rows[0][1], axis=0), axis=-1), "k", lw=1.0); ax.set_title("frame-to-frame change (black: GT)", fontsize=7)
        for c_ in range(4, len(frames)):
            axes[len(rows), c_].axis("off")
        meta = data.meta.iloc[n]; num = 7 if ds == "taco" else 8
        fig.suptitle(f"Figure {num} ({Z.LABEL[ds]}, {key}): {case} — test example {n} ({getattr(meta, 'category', '')}, {getattr(meta, 'hand', '')}); E_C " + " / ".join(f"{m} {E[m][i]:.2f}" for m in mods), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); name = f"fig{num}_{ds}_{key}.png"; fig.savefig(FIG / name, dpi=105); plt.close(fig)
        cd = lambda m: M[m]["curve_dense"][i, 0]
        sel_rows.append(dict(dataset=ds, key=key, case=case, example=n, take=str(data.take[n]), **{f"E_C_{m}": float(E[m][i]) for m in mods}, **{f"e1_{m}": float(cd(m)[1]) for m in mods},
                             **{f"e16_{m}": float(np.mean(cd(m)[1:17])) for m in mods}, **{f"late_{m}": float(np.mean(cd(m)[33:])) for m in mods}, figure=name))
    pd.DataFrame(sel_rows).to_csv(FIG / f"qualitative_selection_{ds}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--qual", choices=Z.DATASETS); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.qual:
        qualitative(a.qual); return
    curves = dict(np.load(Z.OUT / "temporal_curves.npz")); fig1(); fig2(); fig3(curves); fig9()
    lp = Z.OUT / "latent_metrics.csv"
    try:
        lat = pd.read_csv(lp)
    except Exception:                                                              # noqa: BLE001
        lat = pd.DataFrame()
    if len(lat):
        fig4(curves); fig5(lat); fig10(lat)
    try:
        E = pd.read_csv(Z.OUT / "early_frame_metrics.csv")
    except Exception:                                                              # noqa: BLE001
        E = pd.DataFrame()
    if len(E):
        fig6(E)
    for ds in Z.DATASETS:
        if all(Z.model_metrics(ds, m).exists() for m in ("M00", "M01", "M10", "M11")):
            subprocess.run([sys.executable, __file__, "--qual", ds], check=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
