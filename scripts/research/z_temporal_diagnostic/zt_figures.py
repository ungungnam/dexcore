#!/usr/bin/env python
"""Figures 1-5 of the diagnostic (PNG under OUT/figures).    CUDA_VISIBLE_DEVICES=5 python zt_figures.py
Figure 5 (representative transitions) needs the canonical point layout of the inherited loader: one subprocess per dataset (--fig5 <ds>).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import zt_common as Z

FIG = Z.OUT / "figures"
COL = {"persistence": "#9E9E9E", "mean": "#D9D9D9", "linear": "#8172B3", "probe_notau": "#64B5CD", "probe": "#DD8452", "stage2": "#4C72B0", "hold": "#C44E52"}
LAB = {"persistence": "persistence z_t", "mean": "train-mean z", "linear": "linear (ridge)", "probe_notau": "probe without trajectory", "probe": "nonlinear probe", "stage2": "Stage-2 B1 (open loop from s₀)",
       "hold": "hold the GT z₀"}
EXAMPLE_H = 4


def fig1(L):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2)); methods = ["persistence", "linear", "probe_notau", "probe"]
    for i, ds in enumerate(Z.DATASETS):
        l = L[L.dataset == ds]
        for j, (key, title, fmt) in enumerate([("rmse", "latent RMSE (standardised z; lower is better)", "{:.3f}"), ("r2", "explained variance R² of z_{t+h} (higher is better)", "{:.2f}"), ("gain_rmse", "gain over persistence, 1 − RMSE / RMSE_persistence", "{:.1%}")]):
            ax = axes[i, j]; ms = methods if key != "gain_rmse" else methods[1:]; w = 0.8 / len(ms)
            for k, m in enumerate(ms):
                for hi, h in enumerate(Z.HORIZONS):
                    r = l[(l.h == h) & (l.method == m)].iloc[0]; x = hi + (k - (len(ms) - 1) / 2) * w
                    ax.bar(x, r[key], width=w * 0.92, color=COL[m], yerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]], capsize=2, label=LAB[m] if hi == 0 else None)
                    ax.text(x, max(r[key + "_hi"], 0) + 0.004 * (1 if key != "r2" else 2), fmt.format(r[key]), ha="center", va="bottom", fontsize=6.5, rotation=90 if key == "gain_rmse" else 0)
            if key == "rmse":
                for hi, h in enumerate(Z.HORIZONS):
                    mr = l[(l.h == h) & (l.method == "mean")].iloc[0].rmse; ax.hlines(mr, hi - 0.45, hi + 0.45, color="k", ls=":", lw=1.0, label="train-mean z" if hi == 0 else None)
            if key == "gain_rmse":
                ax.axhline(Z.DECISION["gain_clear"], color="k", ls="--", lw=0.8); ax.text(-0.45, Z.DECISION["gain_clear"], "'clear' ≥ 10 %", fontsize=7, va="bottom", ha="left"); ax.axhline(0, color="k", lw=0.6)
                lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_, hi_ + 0.22 * (hi_ - lo_))
            ax.set_xticks(range(3)); ax.set_xticklabels([f"h = {h}" for h in Z.HORIZONS]); ax.set_title(f"{Z.LABEL[ds]}: {title}", fontsize=9); ax.grid(axis="y", alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Figure 1 — Local predictability of z from the current GT latent: P_h(z_t, G, τ) → z_{t+h} on the test pairs (mean ± 95 % take-cluster bootstrap)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig1_local_predictability.png", dpi=130); plt.close(fig)


def fig2(S):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 7.6))
    for i, ds in enumerate(Z.DATASETS):
        cv = np.load(Z.ds_out(ds) / "stage2_curves.npz"); f = np.arange(1, Z.T)
        for j, xmax in enumerate((63, 12)):
            ax = axes[i, j]
            ax.plot(f, cv["stage2"], color=COL["stage2"], lw=2.0, label=LAB["stage2"]); ax.plot(f, cv["hold_z0"], color=COL["hold"], lw=1.2, ls="--", label=LAB["hold"])
            for h, mk in zip(Z.HORIZONS, ("o", "s", "D")):
                y = cv[f"probe_h{h}_by_target_frame"]; ok = np.isfinite(y)
                ax.plot(np.arange(Z.T)[ok], y[ok], color=COL["probe"], lw=1.0, alpha=0.75, ls={1: ":", 4: "-.", 8: "-"}[h], label=f"probe from the GT latent {h} frame{'s' if h > 1 else ''} earlier")
                s = S[(S.dataset == ds) & (S.h == h) & (S["mode"] == "from_start")].iloc[0]
                ax.errorbar([h], [s.rmse_probe], yerr=[[s.rmse_probe - s.rmse_probe_lo], [s.rmse_probe_hi - s.rmse_probe]], color="k", marker=mk, ms=6, capsize=3, ls="none", zorder=5,
                            label="probe from the GT z₀ (same information horizon as Stage 2)" if h == 1 else None)
            ax.set_xlim(0.5, xmax + 0.5); ax.set_ylim(0, 1.0 if xmax == 12 else None); ax.set_xlabel("target frame f (frames after the known initial state)", fontsize=8.5); ax.set_ylabel("latent RMSE (standardised)", fontsize=8.5)
            ax.set_title(f"{Z.LABEL[ds]}: {'whole window' if xmax == 63 else 'first 12 frames'}", fontsize=9); ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Figure 2 — Does the current GT latent rescue the prediction?  Stage-2 B1 open-loop error per target frame vs the local probe fed with a GT latent state", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig2_local_vs_stage2.png", dpi=130); plt.close(fig)


def fig3(G):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, ds in enumerate(Z.DATASETS):
        ga = np.load(Z.ds_out(ds) / "geometry_arrays.npz")
        for j, h in enumerate(Z.HORIZONS):
            ax = axes[i, j]; x, y = ga[f"h{h}|dC"], ga[f"h{h}|dz"]; thr = ga[f"h{h}|thr"]; g = G[(G.dataset == ds) & (G.h == h)].iloc[0]
            hb = ax.hexbin(x, y, gridsize=55, bins="log", cmap="viridis", mincnt=1, extent=(0, np.quantile(x, 0.995), 0, np.quantile(y, 0.995)))
            for v in thr[:2]:
                ax.axvline(v, color="w", lw=0.8, ls="--")
            for v in thr[2:4]:
                ax.axhline(v, color="w", lw=0.8, ls="--")
            ax.set_xlim(0, np.quantile(x, 0.995)); ax.set_ylim(0, np.quantile(y, 0.995))
            ax.set_title(f"{Z.LABEL[ds]}, h = {h}: Pearson {g.pearson_C_z:.2f}, Spearman {g.spearman_C_z:.2f}  (n = {int(g.n_test)})", fontsize=9)
            ax.set_xlabel("ΔC = ‖C_t − C_{t+h}‖ / random-pair distance", fontsize=8.5); ax.set_ylabel("Δz = ‖z_t − z_{t+h}‖ / random-pair distance", fontsize=8.5)
            fig.colorbar(hb, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Figure 3 — Temporal geometry: latent step vs dense-contact step of every test transition (log-density hexbin; dashed white lines: train 30 % / 70 % quantiles)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG / "fig3_deltaC_vs_deltaz.png", dpi=130); plt.close(fig)


def fig4(Q):
    keys = [("A_lowC_lowz", "A  low ΔC / low Δz", "#55A868"), ("B_lowC_highz", "B  low ΔC / high Δz", "#C44E52"), ("C_highC_lowz", "C  high ΔC / low Δz", "#8172B3"), ("D_highC_highz", "D  high ΔC / high Δz", "#4C72B0")]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; q = Q[Q.dataset == ds]
        for k, (key, lab, col) in enumerate(keys):
            for hi, h in enumerate(Z.HORIZONS):
                r = q[q.h == h].iloc[0]; x = hi + (k - 1.5) * 0.2
                ax.bar(x, r[key], width=0.18, color=col, yerr=[[r[key] - r[key + "_lo"]], [r[key + "_hi"] - r[key]]], capsize=2, label=lab if hi == 0 else None)
                ax.text(x, r[key + "_hi"] + 0.004, f"{r[key] * 100:.1f}", ha="center", va="bottom", fontsize=7)
        for hi, h in enumerate(Z.HORIZONS):                             # what B and C would hold if ΔC and Δz were independent, at the TEST marginals of that horizon
            r = q[q.h == h].iloc[0]
            for k_, key_ in ((1, "indep_B_at_test_marginals"), (2, "indep_C_at_test_marginals")):
                x = hi + (k_ - 1.5) * 0.2; ax.hlines(r[key_], x - 0.09, x + 0.09, colors="k", linestyles=":", lw=1.3)
        ax.set_xlabel(f"dotted: B and C expected under independence at the test marginals\n(B {q.indep_B_at_test_marginals.min() * 100:.1f}–{q.indep_B_at_test_marginals.max() * 100:.1f} %, "
                      f"C {q.indep_C_at_test_marginals.min() * 100:.1f}–{q.indep_C_at_test_marginals.max() * 100:.1f} %; 9 % on train by construction)", fontsize=8)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"h = {h}" for h in Z.HORIZONS]); ax.set_ylabel("fraction of test transitions"); ax.set_title(f"{Z.LABEL[ds]} (numbers in %)", fontsize=9.5); ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper center", ncol=2)
        ax.set_ylim(0, 0.44)
    fig.suptitle("Figure 4 — Temporal-geometry quadrants (low = below the train 30 % quantile, high = above the train 70 % quantile of each quantity)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "fig4_quadrants.png", dpi=130); plt.close(fig)


def fig6(SP):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6)); pos = list(dict.fromkeys(SP[SP.split_kind == "window position"].split)); cols = ["#C44E52", "#DD8452", "#4C72B0"]
    for i, ds in enumerate(Z.DATASETS):
        ax = axes[i]; s = SP[(SP.dataset == ds) & (SP.method == "probe") & (SP.split_kind == "window position")]
        for k, p in enumerate(pos):
            for hi, h in enumerate(Z.HORIZONS):
                r = s[(s.split == p) & (s.h == h)].iloc[0]; x = hi + (k - 1) * 0.27
                ax.bar(x, r.gain_rmse, width=0.25, color=cols[k], yerr=[[r.gain_rmse - r.gain_rmse_lo], [r.gain_rmse_hi - r.gain_rmse]], capsize=2, label=f"pairs starting at {p}" if hi == 0 else None)
                ax.text(x, max(r.gain_rmse_hi, 0) + 0.006, f"{r.gain_rmse * 100:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0, color="k", lw=0.7); ax.axhline(Z.DECISION["gain_clear"], color="k", ls="--", lw=0.8)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"h = {h}" for h in Z.HORIZONS]); ax.set_ylabel("gain over persistence, 1 − RMSE / RMSE_persistence"); ax.set_title(f"{Z.LABEL[ds]} (numbers in %)", fontsize=9.5); ax.grid(axis="y", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Figure 6 — The probe's gain over persistence by position in the window (windows start at the first firm contact; dashed: the 10 % 'clear' line)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / "fig6_gain_by_window_position.png", dpi=130); plt.close(fig)


def fig5(ds):
    import torch
    from cf_data import FrameData
    from zt_data import Cache
    c = Cache(ds, with_contact=True); ck = torch.load(c.ckpt, map_location="cpu", weights_only=False)
    data = FrameData(ds, torch.device("cuda" if torch.cuda.is_available() else "cpu"), stats=ck["stats"], need_masks=False)
    h = EXAMPLE_H; ga = np.load(Z.ds_out(ds) / "geometry_arrays.npz"); r, t, dC, dz, dT, fl, nC = (ga[f"h{h}|{k}"] for k in ("r", "t", "dC", "dz", "dT", "flips", "normC")); thr = ga[f"h{h}|thr"]
    weak = nC <= float(ga[f"h{h}|weak_thr"][0]); struct = (fl > 0) | (dT >= thr[5])
    inB = (dC <= thr[0]) & (dz >= thr[3]); inA = (dC <= thr[0]) & (dz <= thr[2]); inD = (dC >= thr[1]) & (dz >= thr[3])
    def median_of(mask):                                                # the transition whose latent step is the median of the group (a typical member)
        idx = np.where(mask)[0]
        return [int(idx[np.argsort(dz[idx])[len(idx) // 2]])] if len(idx) else []
    # one example per KIND of exception in the low-ΔC / high-Δz quadrant, each the median-Δz member of its kind
    kinds = [("low ΔC / high Δz — weak-contact start", inB & weak), ("low ΔC / high Δz — structural step (participation flip or large teacher step)", inB & ~weak & struct),
             ("low ΔC / high Δz — neither (not a weak-contact start, same structure)", inB & ~weak & ~struct)]
    ex = [(f"{lab}  [{int(m.sum())} of {int(inB.sum())}]", i) for lab, m in kinds for i in median_of(m)]
    ex += [("low ΔC / low Δz — typical (median), contact present", i) for i in median_of(inA & ~weak)] + [("high ΔC / high Δz — typical (median)", i) for i in median_of(inD)]
    B = np.where(inB)[0]
    fig, axes = plt.subplots(len(ex), 4, figsize=(13.5, 2.75 * len(ex)), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.25])); rows = []
    for k, (kind, i) in enumerate(ex):
        row, tt = int(r[i]), int(t[i]); n = int(c.seq[row]); g = data.geo.geo_index[n]
        Pt = (data.geo.X_top[g] + data.geo.X_bot[g]).cpu().numpy(); Pc = Pt - Pt.mean(0); _, _, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
        C0, C1 = c.C[row, tt], c.C[row, tt + h]
        for j, (img, ttl, cm, vm) in enumerate([(C0, f"C_t  (t = {tt})", "viridis", (0, 1)), (C1, f"C_t+{h}", "viridis", (0, 1)), (C1 - C0, "C_t+h − C_t", "RdBu_r", (-0.3, 0.3))]):
            ax = axes[k, j]; ax.scatter(xy[:, 0], xy[:, 1], c=img, cmap=cm, s=6, vmin=vm[0], vmax=vm[1]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=8.5)
            if j == 0:
                ax.set_ylabel("\n".join(__import__("textwrap").wrap(kind, 34)), fontsize=7)
        a0, a1 = c.a[row, tt], c.a[row, tt + h]; m0, m1 = c.m[row, tt], c.m[row, tt + h]; zs0, zs1 = c.zs[row, tt], c.zs[row, tt + h]
        parts = Z.CF.PARTS; pset = lambda a_: "{" + ", ".join(p for p, v in zip(parts, a_) if v > 0) + "}"
        txt = (f"{Z.LABEL[ds]}  seq {n}, take {c.take[row]}\n\nΔC = {dC[i]:.3f}   (train 30 % / 70 %: {thr[0]:.3f} / {thr[1]:.3f})\nΔz = {dz[i]:.3f}   (train 30 % / 70 %: {thr[2]:.3f} / {thr[3]:.3f})\n"
               f"‖Δz‖ = {np.linalg.norm(zs1 - zs0):.2f} std units over 64 dims\nteacher (R2 + wrench) step = {dT[i]:.3f}   (train 30 % / 70 %: {thr[4]:.3f} / {thr[5]:.3f})\n\n"
               f"R2 change:\n  participation {pset(a0)}\n      → {pset(a1)}   ({int(fl[i])} flips)\n  total amount {m0.sum():.3f} → {m1.sum():.3f}\n  ‖C_t‖ = {np.linalg.norm(C0):.2f}, ‖ΔC‖ = {np.linalg.norm(C1 - C0):.2f} (raw)")
        axes[k, 3].axis("off"); axes[k, 3].text(0.0, 0.98, txt, va="top", ha="left", fontsize=8, family="monospace", transform=axes[k, 3].transAxes)
        rows.append(dict(dataset=ds, kind=kind, sequence=n, take=c.take[row], t=tt, h=h, dC=float(dC[i]), dz=float(dz[i]), dT=float(dT[i]), flips=int(fl[i]), dz_norm_std=float(np.linalg.norm(zs1 - zs0)), amount_before=float(m0.sum()), amount_after=float(m1.sum()),
                         participation_before=pset(a0), participation_after=pset(a1), norm_C_t=float(np.linalg.norm(C0)), norm_dC_raw=float(np.linalg.norm(C1 - C0))))
    fig.suptitle(f"Figure 5 ({Z.LABEL[ds]}) — Representative test transitions at h = {h}: one median example of each kind of low-ΔC / high-Δz exception\n"
                 f"and one typical well-behaved case of each kind   [low-ΔC / high-Δz: {len(B)} of {len(dC)} test transitions]", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.965)); fig.savefig(FIG / f"fig5_transitions_{ds}.png", dpi=115); plt.close(fig)
    pd.DataFrame(rows).to_csv(FIG / f"fig5_selection_{ds}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig5", choices=Z.DATASETS); a = ap.parse_args(); FIG.mkdir(parents=True, exist_ok=True)
    if a.fig5:
        fig5(a.fig5); return
    L = pd.read_csv(Z.OUT / "local_prediction_metrics.csv"); S = pd.read_csv(Z.OUT / "stage2_comparison.csv"); G = pd.read_csv(Z.OUT / "temporal_geometry_metrics.csv"); Q = pd.read_csv(Z.OUT / "quadrant_metrics.csv")
    fig1(L); fig2(S); fig3(G); fig4(Q); fig6(pd.read_csv(Z.OUT / "local_prediction_splits.csv"))
    for ds in Z.DATASETS:
        subprocess.run([sys.executable, __file__, "--fig5", ds], check=True)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
