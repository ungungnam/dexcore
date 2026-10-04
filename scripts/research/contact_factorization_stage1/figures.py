#!/usr/bin/env python
"""Figures 1-8 of the report from the root tables and the per-dataset example caches (no GPU, no data loading).
    python figures.py        -> <out>/figures/fig1_architecture.png ... fig8_residual.png
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import cf_common as S

FIG = S.OUT / "figures"
CODES = {"A3:r": ("A3 r (realisation)", "tab:red"), "A3:z": ("A3 z (structure)", "tab:blue"), "A2:r": ("A2 r (no relational loss)", "salmon"), "A0:h": ("A0 h (plain AE)", "tab:gray"),
         "A3:resid_z": ("dense residual C − $\\bar{C}$ (A3)", "tab:purple"), "dense_C": ("dense map C", "black"), "teacher": ("R2 + wrench teacher", "tab:green")}
MODEL_COL = {"A0": "tab:gray", "A1": "tab:green", "A2": "tab:orange", "A3": "tab:red"}


def proj2d(x, valid):
    v = x[valid] if valid.sum() > 3 else x
    c = v.mean(0); _, _, Vh = np.linalg.svd(v - c, full_matrices=False)
    return (x - c) @ Vh[:2].T


def scatter_map(ax, x, valid, c, vmin=0, vmax=1, cmap="viridis", title="", s=7):
    uv = proj2d(x, valid); m = valid.astype(bool)
    sc = ax.scatter(uv[m, 0], uv[m, 1], c=c[m], s=s, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0)
    ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=7)
    return sc


def load(name):
    p = S.OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def fig1():
    fig, ax = plt.subplots(figsize=(11.5, 5)); ax.axis("off"); ax.set_xlim(0, 11.5); ax.set_ylim(-0.2, 4.6)
    def box(x, y, w, h, text, fc="#eef3fb", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="0.3", lw=1)); ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    def arrow(x0, y0, x1, y1, text="", dy=0.12):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", lw=1.2, color="0.25"))
        if text:
            ax.text((x0 + x1) / 2 + 0.45, (y0 + y1) / 2 + dy, text, ha="center", fontsize=7.5, color="0.25")
    box(0.2, 3.1, 1.3, 0.8, "C (512)\ndense contact", "#f6f6f6"); box(0.2, 1.9, 1.3, 0.8, "G\ngeometry / descriptor", "#f6f6f6")
    box(2.3, 3.1, 1.6, 0.8, "$E_z(C, G)$\npoint transformer"); box(4.6, 3.1, 0.9, 0.8, "z (64)", "#dbe8ff"); box(6.2, 3.1, 1.6, 0.8, "$D_z(z, G)$"); box(8.6, 3.1, 1.3, 0.8, "$\\bar{C}$ (512)\nz-only map", "#e8f5e9")
    box(2.3, 1.2, 1.6, 0.8, "$E_r(C, z, G)$\n$[C\\,|\\,C - \\bar{C}]$"); box(4.6, 1.2, 0.9, 0.8, "r (64)", "#ffe0e0"); box(6.2, 1.2, 1.6, 0.8, "$D_r(z, r, G)$\nzero-init output"); box(8.6, 1.2, 1.3, 0.8, "$\\Delta C$ (512)\nresidual", "#fff3e0")
    box(10.15, 2.15, 1.25, 0.8, "$\\hat{C} = \\bar{C} + \\Delta C$", "#e8f5e9")
    arrow(1.5, 3.5, 2.3, 3.5); arrow(1.5, 2.3, 2.3, 3.3); arrow(1.5, 2.3, 2.3, 1.7); arrow(1.5, 3.5, 2.3, 1.75)
    arrow(3.9, 3.5, 4.6, 3.5); arrow(5.5, 3.5, 6.2, 3.5); arrow(7.8, 3.5, 8.6, 3.5); arrow(5.05, 3.1, 5.05, 2.0, "stop-grad", 0.0)
    arrow(3.9, 1.6, 4.6, 1.6); arrow(5.5, 1.6, 6.2, 1.6); arrow(7.8, 1.6, 8.6, 1.6); arrow(9.9, 3.2, 10.6, 2.95); arrow(9.9, 1.9, 10.6, 2.15)
    ax.text(0.2, 0.75, "Losses:  $L_{coarse} = MSE(\\bar{C}, C)$    $L_{full} = MSE(\\hat{C}, C)$    $L_{rel} = $ SmoothL1($d_z(i,j)/\\overline{d_z}$, $d_{teacher}(i,j)/\\overline{d_{teacher}}$), "
            "$d_{teacher} = \\frac{1}{2} d_{R2} + \\frac{1}{2} d_{wrench}$    $\\lambda_\\delta \\cdot$ mean$|\\Delta C|$ (0.05)\n"
            "No independence loss, no R2 output. Phase A trains $E_z$ / $D_z$ ($L_{coarse} + L_{rel}$); phase B adds $E_r$ / $D_r$ (z branch at 0.1 × lr for 2k steps, then joint).\n"
            "A0: one latent h (128) through $E_z$ / $D_z$, reconstruction only.   A1: z only + $L_{rel}$.   A2: z / r without $L_{rel}$.   A3: z / r + $L_{rel}$ (proposed).", fontsize=7.8, va="top")
    fig.suptitle("Figure 1 — Stage-1 factorisation: structure-oriented z, complementary realisation code r", fontsize=10)
    fig.savefig(FIG / "fig1_architecture.png", dpi=160, bbox_inches="tight"); plt.close(fig)


def fig2():
    rows = []
    for ds in S.DATASETS:
        p = S.ds_out(ds) / "residual_maps.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        for j in range(3):
            if f"A3_ex{j}_C" in z:
                rows.append((ds, j, z))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (ds, j, z) in enumerate(rows):
        k = f"A3_ex{j}_"; x, valid = z[k + "x"], z[k + "valid"]; C, Cb, Ch, d = z[k + "C"], z[k + "C_bar"], z[k + "C_hat"], z[k + "delta"]; meta = z[k + "meta"]
        scatter_map(axes[i, 0], x, valid, C, title=f"{S.LABEL[ds]} mesh {meta[0]} (seq {meta[1]}, t = {meta[2]})\nGT C")
        scatter_map(axes[i, 1], x, valid, Cb, title=f"z-only $\\bar{{C}}$   ‖$\\bar{{C}}$ − C‖ = {np.linalg.norm(Cb - C):.2f}")
        scatter_map(axes[i, 2], x, valid, d, vmin=-0.3, vmax=0.3, cmap="RdBu_r", title=f"ΔC = D_r(z, r, G)   mean|ΔC| = {np.abs(d).mean():.3f}")
        scatter_map(axes[i, 3], x, valid, Ch, title=f"full $\\hat{{C}} = \\bar{{C}} + \\Delta C$   ‖$\\hat{{C}}$ − C‖ = {np.linalg.norm(Ch - C):.2f}")
    fig.suptitle("Figure 2 — Reconstruction decomposition of A3 (test frames with the median z-residual energy of the three most frequent meshes; canonical points, PCA view)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig2_reconstruction_examples.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig3():
    rec = load("reconstruction_metrics.csv")
    if rec.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, ds in zip(axes, S.DATASETS):
        r = rec[rec.dataset == ds]
        if r.empty:
            continue
        xs = np.arange(4); w = 0.38
        for i, m in enumerate(S.MODELS):
            q = r[r.model == m]
            if q.empty:
                continue
            q = q.iloc[0]
            ax.bar(i - w / 2, q.E_zonly, w, color=MODEL_COL[m], alpha=0.45, hatch="//", yerr=[[q.E_zonly - q.E_zonly_lo], [q.E_zonly_hi - q.E_zonly]], capsize=2, label="z-only $\\bar{C}$ (A0: h)" if i == 0 else None)
            ax.bar(i + w / 2, q.E_full, w, color=MODEL_COL[m], yerr=[[q.E_full - q.E_full_lo], [q.E_full_hi - q.E_full]], capsize=2, label="full $\\hat{C}$" if i == 0 else None)
            ax.text(i, max(q.E_zonly, q.E_full) * 1.03, f"R²={q.R2_full:.3f}", ha="center", fontsize=7)
        for b, ls, lab in (("mean", ":", "train mean"), ("mesh_mean", "-.", "per-mesh mean"), ("pca64", "--", "PCA-64 (linear)"), ("pca128", "-", "PCA-128 (linear)")):
            q = r[r.model == b]
            if not q.empty:
                ax.axhline(q.iloc[0].E_full, color="0.4", ls=ls, lw=1, label=lab)
        ax.set_xticks(xs); ax.set_xticklabels(list(S.MODELS)); ax.set_ylabel("‖$\\hat{C}$ − C‖ (raw contact units, test)"); ax.set_title(S.LABEL[ds]); ax.set_yscale("log"); ax.legend(fontsize=6.5, ncol=2)
    fig.suptitle("Figure 3 — Reconstruction: z-only vs full error per model (95 % take-bootstrap CI) with linear baselines", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig3_reconstruction.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig4():
    pr = load("probe_metrics.csv")
    if pr.empty:
        return
    inputs = ["A3:z", "A3:r", "A3:zr", "A2:z", "A2:r", "A0:h", "A1:z", "C_pca64", "C", "trivial"]
    cols = {"A3:z": "tab:blue", "A3:r": "tab:red", "A3:zr": "tab:purple", "A2:z": "lightsteelblue", "A2:r": "salmon", "A0:h": "tab:gray", "A1:z": "tab:green", "C_pca64": "0.6", "C": "black", "trivial": "0.85"}
    quant = [("part_auprc", "participation macro AUPRC ↑"), ("amount_r2", "amount R² ↑"), ("centroid", "centroid error (l) ↓"), ("normal", "normal angle (°) ↓"), ("wrench_ev", "wrench explained variance ↑"), ("structure_retention", "structure retention (mean of 5) ↑")]
    fig, axes = plt.subplots(2, 6, figsize=(16, 6))
    for i, ds in enumerate(S.DATASETS):
        p = pr[(pr.dataset == ds) & ((pr.probe == "mlp") | (pr.probe == "constant"))]
        for j, (k, lab) in enumerate(quant):
            ax = axes[i, j]
            for n_, inp in enumerate(inputs):
                q = p[p.input == inp]
                if q.empty:
                    continue
                q = q.iloc[0]; err = None
                if f"{k}_lo" in q and not np.isnan(q.get(f"{k}_lo", np.nan)):
                    err = [[max(q[k] - q[f"{k}_lo"], 0)], [max(q[f"{k}_hi"] - q[k], 0)]]
                ax.bar(n_, q[k], color=cols[inp], yerr=err, capsize=2)
            ax.set_xticks(range(len(inputs))); ax.set_xticklabels(inputs, rotation=70, fontsize=6.5); ax.set_title(f"{S.LABEL[ds]}: {lab}", fontsize=8)
    fig.suptitle("Figure 4 — Where is the structural / functional information? MLP probes from the frozen codes (test; C = dense map, trivial = train mean)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig4_information.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig5():
    panels = []
    for ds in S.DATASETS:
        p = S.ds_out(ds) / "neighbor_examples.npz"
        if p.exists():
            panels.append((ds, np.load(p, allow_pickle=True)))
    if not panels:
        return
    fig, axes = plt.subplots(4 * len(panels), 4, figsize=(8.5, 2.1 * 4 * len(panels)))
    for pi, (ds, z) in enumerate(panels):
        e = lambda k: z[f"ex0__{k}"]
        x, valid = e("x"), e("valid")
        for r_, key, lab in zip(range(4), ("A3:z", "A3:r", "C", "teacher"), ("z neighbours (A3)", "r neighbours (A3)", "dense-C neighbours", "teacher (R2 + wrench) neighbours")):
            row = 4 * pi + r_
            scatter_map(axes[row, 0], x, valid, e("anchor_C"), title=f"{S.LABEL[ds]} anchor (mesh {e('mesh')})\n{lab}")
            for c_ in range(3):
                scatter_map(axes[row, c_ + 1], x, valid, e(f"{key}_C")[c_], title=f"#{c_ + 1}: teacher d = {e(f'{key}_dt')[c_]:.2f}, dense d = {e(f'{key}_dc')[c_]:.2f}")
    fig.suptitle("Figure 5 — Nearest neighbours of one test anchor per dataset in z, r, dense-C and teacher space (same mesh, other takes)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "fig5_neighbors.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig6():
    rows = []
    for ds in S.DATASETS:
        p = S.ds_out(ds) / "swap_examples.npz"; q = S.ds_out(ds) / "swap_pairs_A3.csv"
        if not p.exists() or not q.exists():
            continue
        z = np.load(p, allow_pickle=True); pairs = pd.read_csv(q)
        for k in range(3):
            if f"A3__ex{k}__C_A" in z:
                rows.append((ds, k, z, pairs))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), 4, figsize=(10, 2.4 * len(rows)))
    axes = np.atleast_2d(axes)
    for i, (ds, k, z, pairs) in enumerate(rows):
        e = lambda s: z[f"A3__ex{k}__{s}"]; x, valid = e("x"), e("valid")
        ab = pairs[(pairs.pair == k) & (pairs.swap == "AB")].iloc[0]; ba = pairs[(pairs.pair == k) & (pairs.swap == "BA")].iloc[0]
        scatter_map(axes[i, 0], x, valid, e("C_A"), title=f"{S.LABEL[ds]} pair {k} (mesh {e('mesh')}): A\nteacher d(A,B) = {ab.d_teacher_sel:.2f}, dense = {ab.d_dense_sel:.2f}")
        scatter_map(axes[i, 1], x, valid, e("C_B"), title="B (similar R2 / wrench, different dense map)")
        scatter_map(axes[i, 2], x, valid, e("AB"), title=f"z_A + r_B: struct d→A {ab.struct_to_zdonor:.2f} / →B {ab.struct_to_rdonor:.2f}\ndetail cos →B {ab.detail_cos_rdonor:.2f} / →A {ab.detail_cos_zdonor:.2f}; moved {ab.dense_change:.2f}")
        scatter_map(axes[i, 3], x, valid, e("BA"), title=f"z_B + r_A: struct d→B {ba.struct_to_zdonor:.2f} / →A {ba.struct_to_rdonor:.2f}\ndetail cos →A {ba.detail_cos_rdonor:.2f} / →B {ba.detail_cos_zdonor:.2f}; moved {ba.dense_change:.2f}")
    fig.suptitle("Figure 6 — Latent swap test (A3): structure should follow the z donor, fine realisation the r donor", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "fig6_swap.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig7():
    tp = load("temporal_persistence.csv")
    if tp.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for i, ds in enumerate(S.DATASETS):
        p = S.ds_out(ds) / "temporal_curves.npz"; t = tp[tp.dataset == ds]
        if not p.exists() or t.empty:
            continue
        cv = np.load(p)
        for code, (lab, col) in CODES.items():
            if f"{code}__cos" in cv:
                axes[i, 0].plot(cv[f"{code}__cos"], color=col, label=lab, lw=1.3 if code in ("A3:r", "A3:z") else 1); axes[i, 1].plot(cv[f"{code}__acf"][:33], color=col, label=lab, lw=1.3 if code in ("A3:r", "A3:z") else 1)
        axes[i, 0].set_xlabel("frame t"); axes[i, 0].set_ylabel("cos(x_t − μ, x_0 − μ)"); axes[i, 0].set_title(f"{S.LABEL[ds]}: similarity to the initial frame"); axes[i, 0].legend(fontsize=6.5)
        axes[i, 1].set_xlabel("lag h (frames)"); axes[i, 1].set_ylabel("autocorrelation"); axes[i, 1].set_title(f"{S.LABEL[ds]}: autocorrelation"); axes[i, 1].set_ylim(0, 1.02)
        hs = [1, 4, 8, 16]; codes = [c for c in CODES if ((t.code == c) & (t.metric == "displacement_ratio")).any()]
        for j, code in enumerate(codes):
            v = [t[(t.code == code) & (t.metric == "displacement_ratio") & (t.h == h)].value.iloc[0] for h in hs]
            axes[i, 2].bar(np.arange(len(hs)) + (j - len(codes) / 2) * 0.8 / len(codes), v, 0.8 / len(codes), color=CODES[code][1], label=CODES[code][0])
        axes[i, 2].set_xticks(range(len(hs))); axes[i, 2].set_xticklabels([f"h = {h}" for h in hs]); axes[i, 2].set_ylabel("‖x_{t+h} − x_t‖ / random-pair distance"); axes[i, 2].set_title(f"{S.LABEL[ds]}: displacement ratio"); axes[i, 2].legend(fontsize=6)
    fig.suptitle("Figure 7 — Temporal persistence of the frame-wise codes on GT test trajectories (no temporal training)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig7_temporal.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig8():
    loc = load("residual_localisation.csv")
    panels = [(ds, np.load(S.ds_out(ds) / "residual_maps.npz", allow_pickle=True)) for ds in S.DATASETS if (S.ds_out(ds) / "residual_maps.npz").exists()]
    if not panels or loc.empty:
        return
    fig, axes = plt.subplots(len(panels), 4, figsize=(14, 3.4 * len(panels)))
    axes = np.atleast_2d(axes); edges = np.linspace(0, 0.5, 51); mid = (edges[:-1] + edges[1:]) / 2
    for i, (ds, z) in enumerate(panels):
        if "A3_delta_abs_hist" not in z:
            continue
        axes[i, 0].plot(mid, z["A3_delta_abs_hist"] / z["A3_delta_abs_hist"].sum(), color="tab:red", label="|ΔC| (A3)"); axes[i, 0].plot(mid, z["A3_resid_z_abs_hist"] / z["A3_resid_z_abs_hist"].sum(), color="tab:purple", label="|C − $\\bar{C}$| (z residual)")
        axes[i, 0].set_yscale("log"); axes[i, 0].set_xlabel("magnitude (raw contact units)"); axes[i, 0].set_ylabel("fraction of points"); axes[i, 0].set_title(f"{S.LABEL[ds]}: distribution over test points"); axes[i, 0].legend(fontsize=7)
        l = loc[(loc.dataset == ds) & (loc.model == "A3") & (loc.bin != "all")]
        xs = np.arange(len(l)); w = 0.27
        axes[i, 1].bar(xs - w, l.frac_points, w, color="0.7", label="fraction of points"); axes[i, 1].bar(xs, l.resid_z_energy_share, w, color="tab:purple", label="share of z-residual energy"); axes[i, 1].bar(xs + w, l.delta_energy_share, w, color="tab:red", label="share of ΔC energy")
        for x_, fr in zip(xs, l.frac_resid_z_energy_removed):
            axes[i, 1].text(x_, max(l.resid_z_energy_share.max(), l.delta_energy_share.max()) * 1.02, f"{100 * fr:.0f} % removed", ha="center", fontsize=7)
        axes[i, 1].set_xticks(xs); axes[i, 1].set_xticklabels([f"{b}\n(GT C {lo:.2f}–{hi:.2f})" for b, lo, hi in [("off", 0, 0.05), ("soft", 0.05, 0.45), ("hard", 0.45, 1.0)]], fontsize=7); axes[i, 1].set_title(f"{S.LABEL[ds]}: where ΔC acts (by GT contact bin)"); axes[i, 1].legend(fontsize=6.5)
        meshes = [k.split("_")[1] for k in z.files if k.startswith("A3_") and k.endswith("_delta_abs") and "hist" not in k]
        if meshes:
            m = meshes[0]; x, valid = z[f"x_{m}"], z[f"valid_{m}"]
            vmax = float(np.percentile(z[f"A3_{m}_delta_abs"], 99))
            scatter_map(axes[i, 2], x, valid, z[f"A3_{m}_C_mean"], title=f"{S.LABEL[ds]} mesh {m}: mean GT contact ({int(z[f'A3_{m}_n'])} test frames)", cmap="viridis")
            scatter_map(axes[i, 3], x, valid, z[f"A3_{m}_delta_abs"], vmin=0, vmax=vmax, cmap="magma", title=f"mesh {m}: mean |ΔC| per canonical point (max {vmax:.3f})")
    fig.suptitle("Figure 8 — What r changes: magnitude distribution, localisation relative to the GT contact patches, per-point mean |ΔC|", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "fig8_residual.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f(); print("ok", f.__name__)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc(); print("FAILED", f.__name__, e)


if __name__ == "__main__":
    main()
