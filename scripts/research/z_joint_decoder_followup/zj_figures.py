#!/usr/bin/env python
"""Figures of the report (PNG under OUT/figures).    python zj_figures.py        (CPU only; Figure 6 loads the canonical geometry per dataset)
  fig1 architecture | fig2 main comparison | fig3 dense error vs horizon | fig4 z prediction error vs horizon | fig5 decoder mismatch |
  fig6a-c qualitative trajectories (selection rule in fig6()) | fig7 validation / training curves | fig8 what the decoder met in training
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

import zj_common as Z

S2 = Z.S2
FIG = Z.OUT / "figures"
COL = {"M0": "#4C72B0", "M1": "#DD8452", "M2": "#55A868", "M3": "#8172B3", "M0r": "#8FAADC", "PERSIST": "#C44E52", "GT": "black", "hold_z0": "#B07AA1", "train_mean": "#BBBBBB"}
LAB = {"M0": "M0 direct dense", "M1": "M1 previous z → C", "M2": "M2 joint, larger decoder", "M3": "M3 dual-input decoder", "M0r": "M0 retrained (100 k protocol)", "PERSIST": "persistence C_t = s_0"}
MET_LAB = {"E_C": "dense error E_C", "E_C_last16": "E_C, last 16 frames", "part_hamming": "participation Hamming", "amount_l1": "amount L1 (norm.)", "centroid": "centroid error / l",
           "normal": "normal angle (°)", "q_rel_l1": "wrench rel. L1"}
INP_LAB = {"oracle": "teacher z*\n(oracle)", "pred": "predicted ẑ\n(the model)", "noisy_gauss": "z* + matched\nGaussian noise", "noisy_perm": "z* + another\nsequence's error"}
INP_COL = {"oracle": "#7FB77E", "pred": "#333333", "noisy_gauss": "#E0A458", "noisy_perm": "#C97C5D"}


def present(acc, ds, models):
    return [m for m in models if len(acc[(acc.dataset == ds) & (acc.model == m)])]


def get(acc, ds, model, metric):
    r = acc[(acc.dataset == ds) & (acc.model == model) & (acc.metric == metric)]
    return None if r.empty else r.iloc[0]


def fig1():
    fig, ax = plt.subplots(figsize=(12.5, 6.0)); ax.axis("off")
    def box(x, y, w, h, text, fc="#F3F3F3", ec="#555555", fs=8.3, bold=False, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.0, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None)
    def arrow(x0, y0, x1, y1, ls="-", col="#333333"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.1, color=col, ls=ls))
    y = 0.50; h = 0.20
    box(0.01, y, 0.11, h, "inputs\ns₀, G, τ₁:₆₃", fc="#FFFFFF", bold=True)
    arrow(0.12, y + h / 2, 0.15, y + h / 2)
    box(0.15, y, 0.20, h, "temporal network F_θ\none backbone: 768 × 8 blocks,\n12 heads, FiLM on (s₀, G)\n68 M params (same as M0 / M1)", fc="#E8EEF7")
    arrow(0.35, y + h / 2, 0.38, y + h / 2)
    box(0.38, y, 0.13, h, "ẑ₁:₆₃ (64-D)\nall frames at once\n(no rollout)", fc="#E3F0E3")
    arrow(0.51, y + h / 2, 0.54, y + h / 2)
    box(0.54, y, 0.24, h, "contact decoder D_φ(ẑ_t, G_t)\nStage-1 D_z (4 blocks, loaded)\n+ 2 identity-initialised blocks\n8.0 M params, all trained", fc="#FFF4E0")
    arrow(0.78, y + h / 2, 0.81, y + h / 2)
    box(0.81, y, 0.18, h, "Ĉ₁:₆₃ (512-D maps)\nloss L_C vs GT C\n(B0's residual units)", fc="#FFFFFF", bold=True)
    # teacher (training only)
    ty = 0.14
    box(0.15, ty, 0.20, 0.15, "GT contact C_t (training set)", fc="#FFFFFF", ls="--")
    arrow(0.35, ty + 0.075, 0.38, ty + 0.075, ls="--", col="#888888")
    box(0.38, ty, 0.20, 0.15, "frozen Stage-1 encoder E_z\n(A3; never in the model)", fc="#EEEEEE", ls="--")
    arrow(0.58, ty + 0.075, 0.61, ty + 0.075, ls="--", col="#888888")
    box(0.61, ty, 0.17, 0.15, "teacher z*_t\n(cached, standardised\nwith TRAIN statistics)", fc="#EEEEEE", ls="--")
    arrow(0.64, ty + 0.15, 0.46, y, ls="--", col="#888888"); ax.text(0.30, 0.385, "target of ẑ:  L_z = ‖ẑ_t − z*_t‖²  (λ_z = 1)", fontsize=8.3, color="#555555")
    arrow(0.74, ty + 0.15, 0.70, y, ls="--", col="#8172B3"); ax.text(0.735, 0.385, "M3 only: also decoded,  + 0.25 · L_C(D_φ(z*_t))", fontsize=8.3, color="#8172B3")
    ax.text(0.5, 0.955, "M2 (and M3): one network end to end.  Gradient of L_C flows through D_φ and ẑ into F_θ; nothing is frozen.", ha="center", fontsize=9.5, fontweight="bold")
    ax.text(0.5, 0.895, "Dashed = training-only supervision.  M1 (Stage-2 B1) has the same path with the 4-block decoder, a shorter budget and selection on L_C + L_z.\n"
                        "M0 (Stage-2 B0) replaces  ẑ → D_φ  by an MLP head that predicts the dense residual directly.", ha="center", va="top", fontsize=8.5, color="#333333")
    ax.set_xlim(0, 1); ax.set_ylim(0.08, 1)
    fig.suptitle("Figure 1 — Architecture of the jointly adapted z-mediated model", fontsize=10.5, y=0.995)
    fig.tight_layout(); fig.savefig(FIG / "fig1_architecture.png", dpi=130); plt.close(fig)


def fig2(acc):
    mets = ["E_C", "part_hamming", "centroid", "normal", "q_rel_l1"]
    fig, axes = plt.subplots(2, len(mets), figsize=(15, 6.2))
    for i, ds in enumerate(Z.DATASETS):
        ms = present(acc, ds, ["M0", "M1", "M2", "M3"])
        for j, met in enumerate(mets):
            ax = axes[i, j]
            for k, m in enumerate(ms):
                r = get(acc, ds, m, met)
                ax.bar(k, r.value, color=COL[m], width=0.72)
                ax.errorbar(k, r.value, yerr=[[r.value - r.ci_lo], [r.ci_hi - r.value]], color="k", capsize=3, lw=1)
                ax.text(k, r.ci_hi, f"{r.value:.3f}" if r.value < 10 else f"{r.value:.1f}", ha="center", va="bottom", fontsize=7.5)
            vals = [get(acc, ds, m, met) for m in ms]
            lo = min(v.ci_lo for v in vals); hi = max(v.ci_hi for v in vals)
            ax.set_ylim(max(0, lo - 0.9 * (hi - lo)), hi + 0.25 * (hi - lo))
            ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, fontsize=8); ax.set_title(f"{Z.LABEL[ds]}: {MET_LAB[met]}", fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Figure 2 — Main comparison on the test sequences (lower is better; whiskers: 95 % take-cluster bootstrap; y axes do not start at 0)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig2_main_comparison.png", dpi=130); plt.close(fig)


def fig3(curves, acc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]
        for m in present(acc, ds, ["M0", "M1", "M2", "M3"]):
            c = curves[f"{ds}|{m}|dense"]; ax.plot(np.arange(1, Z.T), c[1:], color=COL[m], lw=1.6, label=LAB[m])
        if f"{ds}|PERSIST|dense" in curves:
            ax.plot(np.arange(1, Z.T), curves[f"{ds}|PERSIST|dense"][1:], color=COL["PERSIST"], lw=1.0, ls=":", label=LAB["PERSIST"])
        for lo, hi in Z.HORIZON_BINS[:-1]:
            ax.axvline(hi + 0.5, color="#DDDDDD", lw=0.8)
        ax.set_xlabel("frame t (horizon from s₀)"); ax.set_ylabel("dense error ‖Ĉ_t − C_t‖ (mean over test sequences)"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("Figure 3 — Dense contact error versus horizon (vertical lines: the horizon bins of Table 7)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig3_error_vs_horizon.png", dpi=130); plt.close(fig)


def fig4(curves):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; x = np.arange(1, Z.T)
        for m in ("M1", "M2", "M3"):
            if f"{ds}|{m}|z_err" in curves:
                ax.plot(x, curves[f"{ds}|{m}|z_err"], color=COL[m], lw=1.7, label=f"{m} test")
                ax.plot(x, curves[f"{ds}|{m}|z_err_train"], color=COL[m], lw=1.0, ls="--", label=f"{m} train (same weights)")
        if f"{ds}|hold_z0|z_err" in curves:
            ax.plot(x, curves[f"{ds}|hold_z0|z_err"], color=COL["hold_z0"], lw=1.0, ls=":", label="hold z*₀ (not available to the models)")
            ax.plot(x, curves[f"{ds}|train_mean|z_err"], color=COL["train_mean"], lw=1.0, ls=":", label="train-mean z")
        ax.set_xlabel("frame t"); ax.set_ylabel("RMSE of ẑ_t − z*_t (standardised teacher coordinates)"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(alpha=0.3); ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False); ax.set_ylim(0, None)
    fig.suptitle("Figure 4 — z prediction error versus horizon: test (solid) and the training sequences (dashed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig4_z_prediction.png", dpi=130); plt.close(fig)


def fig5(dd):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6)); inputs = ["oracle", "pred", "noisy_gauss", "noisy_perm"]
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; d = dd[(dd.dataset == ds) & (dd.split == "test")]; decs = [m for m in ("M1", "M2", "M3") if (d.decoder == m).any()]
        w = 0.19
        for k, m in enumerate(decs):
            for q, inp in enumerate(inputs):
                r = d[(d.decoder == m) & (d.input == inp)]
                if r.empty:
                    continue
                r = r.iloc[0]; x = k + (q - 1.5) * w
                ax.bar(x, r.E_C, width=w * 0.94, color=INP_COL[inp], label=INP_LAB[inp].replace("\n", " ") if k == 0 else None)
                ax.errorbar(x, r.E_C, yerr=[[r.E_C - r.ci_lo], [r.ci_hi - r.E_C]], color="k", capsize=2, lw=0.9)
                ax.text(x, r.ci_hi, f"{r.E_C:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(decs))); ax.set_xticklabels([f"decoder of {m}" for m in decs]); ax.set_ylabel("dense error E_C (test)"); ax.set_title(Z.LABEL[ds], fontsize=10); ax.grid(axis="y", alpha=0.3)
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.suptitle("Figure 5 — Decoder-mismatch diagnostic: the same trained decoder fed with four latent inputs (perturbations matched to the model's own test z error)", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95)); fig.savefig(FIG / "fig5_decoder_mismatch.png", dpi=130); plt.close(fig)


def fig7():
    """Validation curves of the z models and M0: dense term, z term, the decoder's teacher-z floor, and the training z error."""
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.6)); T = pd.read_csv(Z.OUT / "training_summary.csv")
    panels = [("val_L_C", "validation dense loss L_C (predicted z)"), ("val_L_z", "validation L_z"), ("train_L_z", "training L_z (EMA of the batch loss)"), ("val_E_C_gt", "validation E_C, teacher z* → decoder")]
    for i, ds in enumerate(Z.DATASETS):
        logs = {m: pd.read_csv(Z.model_train_log(ds, m)) for m in ("M0", "M1", "M2", "M3", "M0r") if Z.model_train_log(ds, m).exists() and Z.model_ckpt(ds, m).exists()}
        for j, (col, lab) in enumerate(panels):
            ax = axes[i, j]
            for m, lg in logs.items():
                if col == "train_L_z":
                    p = Z.model_train_log(ds, m).with_name(Z.model_train_log(ds, m).stem + "_train.csv")
                    if not p.exists():
                        continue
                    tl = pd.read_csv(p)
                    if "L_z" not in tl:
                        continue
                    y = tl["L_z"].rolling(10, min_periods=1).mean(); ax.plot(tl.step, y, color=COL[m], lw=1.2, label=LAB[m]); continue
                if col not in lg:
                    continue
                ax.plot(lg.step, lg[col], color=COL[m], lw=1.4, label=LAB[m])
                r = T[(T.dataset == ds) & (T.model == m)]
                if len(r):
                    bs = int(r.iloc[0].best_step); v = lg.loc[lg.step == bs, col]
                    if len(v):
                        ax.scatter([bs], [v.iloc[0]], color=COL[m], s=45, zorder=5, edgecolor="k", linewidth=0.6)
            ax.set_xscale("log"); ax.set_xlabel("training step (log)", fontsize=8); ax.set_title(f"{Z.LABEL[ds]}: {lab}", fontsize=8.5); ax.grid(alpha=0.3, which="both")
            if col == "val_L_C":
                lo = min(lg[col].min() for lg in logs.values() if col in lg); ax.set_ylim(lo * 0.97, lo * 1.25)
            if col == "train_L_z":
                ax.set_yscale("log")
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure 7 — Training dynamics (EMA weights at validation, all 63 frames decoded). Circles: the selected step of each run.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig7_training_curves.png", dpi=130); plt.close(fig)


def fig8(zm, dd):
    """What the decoder met in training: z error and dense error on training sequences vs held-out sequences."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for i, ds in enumerate(Z.DATASETS):
        z = zm[zm.dataset == ds]; ms = [m for m in ("M1", "M2", "M3") if (z.model == m).any()]
        ax = axes[2 * i]; w = 0.26
        for k, m in enumerate(ms):
            r = z[z.model == m].iloc[0]
            for q, (col, lab, c) in enumerate((("rmse_train", "train", "#9ECAE1"), ("rmse_val", "validation", "#4292C6"), ("rmse_test", "test", "#084594"))):
                ax.bar(k + (q - 1) * w, r[col], width=w * 0.94, color=c, label=lab if k == 0 else None); ax.text(k + (q - 1) * w, r[col], f"{r[col]:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms); ax.set_title(f"{Z.LABEL[ds]}: RMSE of ẑ − z*", fontsize=9); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=7)
        ax = axes[2 * i + 1]; d = dd[dd.dataset == ds]; w = 0.2
        for k, m in enumerate(ms):
            vals = [("train", "pred", "#9ECAE1", "train: predicted ẑ"), ("train", "oracle", "#C7E9C0", "train: teacher z*"), ("test", "pred", "#084594", "test: predicted ẑ"), ("test", "oracle", "#238B45", "test: teacher z*")]
            for q, (sp, inp, c, lab) in enumerate(vals):
                r = d[(d.decoder == m) & (d.input == inp) & (d.split.str.startswith(sp))]
                if r.empty:
                    continue
                v = r.iloc[0].E_C; ax.bar(k + (q - 1.5) * w, v, width=w * 0.94, color=c, label=lab if k == 0 else None); ax.text(k + (q - 1.5) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels([f"dec. {m}" for m in ms]); ax.set_title(f"{Z.LABEL[ds]}: dense error E_C by decoder input", fontsize=9); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=6.5)
    fig.suptitle("Figure 8 — What the decoder met in training: on the training sequences the predicted ẑ is close to the teacher z*; on held-out sequences it is not", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FIG / "fig8_train_vs_heldout.png", dpi=130); plt.close(fig)


def fig6(ds):
    """Qualitative trajectories.  Selection rule (fixed in code, applied to the per-sequence test errors):
       helps   the test sequence with the largest E_C(M1) - E_C(M2)                 (where the jointly adapted model gains most over the previous one)
       fails   the test sequence with the largest E_C(M2) - E_C(M0)                 (where z mediation loses most against direct prediction)
       typical the test sequence at the 90th percentile of E_C(M2)                  (a representative large-error case, not the single worst)"""
    import torch
    from s2_data import S2Data
    data = S2Data(ds, torch.device("cpu"), need_masks=False, teacher=False)
    mods = [m for m in ("M0", "M1", "M2") if Z.model_metrics(ds, m).exists()]
    M = {m: np.load(Z.model_metrics(ds, m), allow_pickle=True) for m in mods}
    P = {m: np.load(Z.model_preds(ds, m))["pred"][:, 0].astype(np.float32) for m in mods}
    ex = M["M0"]["example"]; E = {m: M[m]["E_C"][:, 0] for m in M}
    order = np.argsort(E["M2"]); cases = {"helps": ("M2 gains most over M1", int(np.nanargmax(E["M1"] - E["M2"]))), "fails": ("M2 loses most against M0", int(np.nanargmax(E["M2"] - E["M0"]))),
                                           "typical": ("90th percentile of M2's dense error", int(order[int(round(0.9 * (len(order) - 1)))]))}
    frames = [0, 4, 8, 16, 32, 48, 63]; sel_rows = []
    for key, (case, i) in cases.items():
        n = int(ex[i]); Pt = (data.geo.X_top[data.geo.geo_index[n]] + data.geo.X_bot[data.geo.geo_index[n]]).cpu().numpy()
        Pc = Pt - Pt.mean(0); _, _, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
        rows = [("GT", data.C[n].cpu().numpy())] + [(m, P[m][i]) for m in mods]
        fig, axes = plt.subplots(len(rows) + 1, len(frames), figsize=(2.3 * len(frames), 2.1 * (len(rows) + 1)))
        for r_, (lab, Cm) in enumerate(rows):
            for c_, t in enumerate(frames):
                ax = axes[r_, c_]; ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(Cm[t], 0, 1), cmap="viridis", s=5, vmin=0, vmax=1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                if c_ == 0:
                    ax.set_ylabel(lab if lab == "GT" else f"{lab}\nE_C {E[lab][i]:.2f}", fontsize=8)
                if r_ == 0:
                    ax.set_title(f"t = {t}" + (" (s₀, given)" if t == 0 else ""), fontsize=8)
                if r_ > 0 and t > 0:
                    ax.text(0.02, 0.02, f"‖err‖ {np.linalg.norm(Cm[t] - rows[0][1][t]):.2f}", transform=ax.transAxes, fontsize=6, color="white")
        ax = axes[len(rows), 0]; a_true = M["M0"]["a_true"][i]
        ax.imshow(np.concatenate([a_true] + [M[m]["r2_a"][i, 0] for m in mods], 1).T, aspect="auto", cmap="Greys", interpolation="nearest"); ax.set_title("participation: GT | " + " | ".join(mods), fontsize=7); ax.set_yticks([]); ax.set_xlabel("frame", fontsize=7)
        ax = axes[len(rows), 1]
        for m in mods:
            ax.plot(M[m]["curve_dense"][i, 0], color=COL[m], lw=1.0, label=m)
        ax.set_title("dense error vs t", fontsize=7); ax.legend(fontsize=6)
        ax = axes[len(rows), 2]; ax.plot(data.m[n].cpu().numpy().sum(1), "k", label="GT")
        for m in mods:
            ax.plot(M[m]["r2_m"][i, 0].astype(np.float32).sum(1), color=COL[m], lw=1.0)
        ax.set_title("total contact amount (black: GT)", fontsize=7)
        ax = axes[len(rows), 3]
        for m in mods:
            ax.plot(M[m]["curve_wrench"][i, 0], color=COL[m], lw=1.0)
        ax.set_title("wrench rel. L1 vs t", fontsize=7)
        ax = axes[len(rows), 4]
        for m in mods:
            ax.plot(M[m]["curve_jitter"][i, 0], color=COL[m], lw=1.0)
        ax.plot(np.linalg.norm(np.diff(rows[0][1], axis=0), axis=-1), "k", lw=1.0); ax.set_title("frame-to-frame change (black: GT)", fontsize=7)
        for c_ in range(5, len(frames)):
            axes[len(rows), c_].axis("off")
        meta = data.meta.iloc[n]
        fig.suptitle(f"Figure 6 ({Z.LABEL[ds]}, {key}): {case} — test example {n} ({getattr(meta, 'category', '')}, {getattr(meta, 'hand', '')}); E_C " + " / ".join(f"{m} {E[m][i]:.2f}" for m in mods), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.97)); name = f"fig6_{ds}_{key}.png"; fig.savefig(FIG / name, dpi=110); plt.close(fig)
        amt_gt = data.m[n].cpu().numpy().sum(1); amt = {m: M[m]["r2_m"][i, 0].astype(np.float32).sum(1) for m in mods}
        sel_rows.append(dict(dataset=ds, key=key, case=case, example=n, take=str(data.take[n]), **{f"E_C_{m}": float(E[m][i]) for m in mods}, figure=name,
                             **{f"e1_{m}": float(M[m]["curve_dense"][i, 0][1]) for m in mods}, amt_peak_gt=float(amt_gt[1:].max()), amt_end_gt=float(amt_gt[48:].mean()),
                             **{f"amt_peak_{m}": float(amt[m][1:].max()) for m in mods}, **{f"amt_end_{m}": float(amt[m][48:].mean()) for m in mods}))
    pd.DataFrame(sel_rows).to_csv(FIG / f"fig6_selection_{ds}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig6", choices=Z.DATASETS, help="draw Figure 6 of one dataset (internal: one dataset loader per process)"); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.fig6:
        fig6(a.fig6); return
    acc = pd.read_csv(Z.OUT / "main_metrics.csv"); curves = dict(np.load(Z.OUT / "temporal_curves.npz"))
    fig1(); fig2(acc); fig3(curves, acc); fig7()
    if (Z.OUT / "z_metrics.csv").exists() and len(pd.read_csv(Z.OUT / "z_metrics.csv")):
        zm = pd.read_csv(Z.OUT / "z_metrics.csv"); dd = pd.read_csv(Z.OUT / "decoder_diagnostic.csv")
        fig4(curves); fig5(dd); fig8(zm, dd)
    for ds in Z.DATASETS:
        subprocess.run([sys.executable, __file__, "--fig6", ds], check=True, env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=""))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
