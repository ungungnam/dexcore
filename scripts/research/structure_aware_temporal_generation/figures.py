#!/usr/bin/env python
"""Figures A-H of the report from the aggregate tables and the prediction / metric files.    CUDA_VISIBLE_DEVICES=4 python figures.py
Writes OUT/figures/fig{A..H}_*.png (+ a small CSV of the qualitative example selection)."""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sat_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("figures")
FIG = S.OUT / "figures"
COL = {"D0": "#7f7f7f", "D1": "#1f77b4", "D2": "#17becf", "S0": "#bcbd22", "S1": "#d62728", "S2": "#ff7f0e", "GT": "black", "PERSIST": "#8c564b"}
MLAB = {"E_C": "dense error E_C", "part_hamming": "participation error (Hamming)", "amount_l1": "amount L1 / train-mean total", "centroid": "centroid distance / l",
        "normal": "normal angle (deg)", "q_rel_l1": "wrench rel. L1", "jitter_dense": "dense frame change", "jitter_r2": "R2 frame change (z)",
        "D_C": "dense diversity D_C", "D_Z": "R2 diversity D_Z", "D_q": "wrench diversity D_q", "D_part": "participation diversity", "D_amount": "amount diversity", "D_centroid": "centroid diversity", "D_normal": "normal diversity (deg)"}
STRUCT6 = ["E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1"]


def get(acc, ds, model, metric, stat="K1", prot="A"):
    r = acc[(acc.dataset == ds) & (acc.model == model) & (acc.metric == metric) & (acc.stat == stat) & (acc.protocol == prot)]
    return (np.nan, np.nan, np.nan, np.nan) if not len(r) else (float(r.iloc[0].value), float(r.iloc[0].ci_lo), float(r.iloc[0].ci_hi), float(r.iloc[0].seed_std))


def fig_a(acc):
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))
    for i, ds in enumerate(S.DATASETS):
        for j, m in enumerate(STRUCT6):
            ax = axes[i, j]
            for k, model in enumerate(S.MODELS):
                v, lo, hi, sd = get(acc, ds, model, m)
                ax.bar(k, v, color=COL[model], yerr=[[v - lo], [hi - v]], capsize=3, label=model)
                if model in S.DIFF:
                    vm = get(acc, ds, model, m, "mean")[0]; ax.plot([k - 0.3, k + 0.3], [vm, vm], color="k", lw=1.2)
            for ref, ls in (("GT", "--"), ("PERSIST", ":")):
                v = get(acc, ds, ref, m)[0]
                if np.isfinite(v):
                    ax.axhline(v, color=COL[ref], ls=ls, lw=1, label=f"{'GT map (floor)' if ref == 'GT' else 'persistence'}")
            ax.set_xticks(range(6)); ax.set_xticklabels(list(S.MODELS)); ax.set_title(f"{S.LABEL[ds]}: {MLAB[m]}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure A. Six models, fixed GT s_0 (protocol A), sample K1 (bars, 95 % take CI; black tick = mean over 10 samples); frames 1..63", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figA_six_models.png", dpi=130); plt.close(fig)


def fig_b(acc):
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))
    for i, ds in enumerate(S.DATASETS):
        for j, m in enumerate(STRUCT6):
            ax = axes[i, j]
            for fam, models, col in (("deterministic", S.DET, "#1f77b4"), ("diffusion (K1)", S.DIFF, "#d62728")):
                vals = [get(acc, ds, mm, m) for mm in models]
                ax.errorbar(range(3), [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], marker="o", color=col, label=fam, capsize=3)
            vals = [get(acc, ds, mm, m, "mean") for mm in S.DIFF]
            ax.plot(range(3), [v[0] for v in vals], marker="s", ls="--", color="#ff7f0e", label="diffusion (mean of 10)")
            ax.set_xticks(range(3)); ax.set_xticklabels(["L0 dense", "L1 + output R2", "L2 + hidden aux"], fontsize=8); ax.set_title(f"{S.LABEL[ds]}: {MLAB[m]}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure B. Structural supervision ablation L0 -> L1 -> L2 (protocol A, 95 % take CI)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figB_loss_ablation.png", dpi=130); plt.close(fig)


def fig_c(div):
    keys = ["D_C", "D_part", "D_amount", "D_centroid", "D_normal", "D_q", "D_Z"]
    fig, axes = plt.subplots(2, len(keys), figsize=(24, 6))
    for i, ds in enumerate(S.DATASETS):
        for j, k in enumerate(keys):
            ax = axes[i, j]
            for x, model in enumerate(S.DIFF):
                r = div[(div.dataset == ds) & (div.model == model) & (div.metric == k) & (div.protocol == "A")]
                if len(r):
                    v, lo, hi = r.iloc[0].value, r.iloc[0].ci_lo, r.iloc[0].ci_hi
                    ax.bar(x, v, color=COL[model], yerr=[[v - lo], [hi - v]], capsize=3)
            ax.set_xticks(range(3)); ax.set_xticklabels(S.DIFF); ax.set_title(f"{S.LABEL[ds]}: {MLAB[k]}", fontsize=10)
    fig.suptitle("Figure C. Fixed-s_0 stochastic diversity (protocol A, 10 futures of the same s_0, G, tau; mean pairwise distance, 95 % take CI)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figC_fixed_s0_diversity.png", dpi=130); plt.close(fig)


def fig_d(div):
    sources = [("D1", "B", "varied s_0 + deterministic future (D1)"), ("S1", "A", "fixed s_0 + stochastic future (S1)"), ("S1", "B", "varied s_0 + stochastic future (S1)"), ("PERSIST", "B", "varied s_0, no evolution")]
    keys = ["D_C", "D_Z", "D_q", "D_part", "D_amount", "D_centroid", "D_normal"]
    fig, axes = plt.subplots(2, len(keys), figsize=(24, 6.5))
    for i, ds in enumerate(S.DATASETS):
        for j, k in enumerate(keys):
            ax = axes[i, j]
            for x, (model, prot, lab) in enumerate(sources):
                r = div[(div.dataset == ds) & (div.model == model) & (div.metric == k) & (div.protocol == prot)]
                if len(r):
                    v, lo, hi = r.iloc[0].value, r.iloc[0].ci_lo, r.iloc[0].ci_hi
                    ax.bar(x, v, color=["#1f77b4", "#d62728", "#9467bd", "#8c564b"][x], yerr=[[v - lo], [hi - v]], capsize=3, label=lab)
            ax.set_xticks(range(len(sources))); ax.set_xticklabels(["init+det", "fixed+stoch", "init+stoch", "init only"], fontsize=8); ax.set_title(f"{S.LABEL[ds]}: {MLAB[k]}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure D. Initial vs future stochasticity: diversity of 10 trajectories per example in the dense, R2 and wrench spaces (95 % take CI)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figD_initial_vs_future.png", dpi=130); plt.close(fig)


def fig_e(acc):
    fig, axes = plt.subplots(2, 6, figsize=(22, 7))
    for i, ds in enumerate(S.DATASETS):
        for j, m in enumerate(STRUCT6):
            ax = axes[i, j]
            for x, model in enumerate(S.DIFF):
                for s, (stat, hatch) in enumerate((("K1", ""), ("mean", "//"), ("best10", "xx"))):
                    v, lo, hi, _ = get(acc, ds, model, m, stat)
                    ax.bar(x * 4 + s, v, color=COL[model], hatch=hatch, yerr=[[v - lo], [hi - v]], capsize=2, label=stat if x == 0 else None, edgecolor="k", lw=0.3)
            for model in S.DET:
                v = get(acc, ds, model, m)[0]
                ax.axhline(v, color=COL[model], lw=1, ls="--", label=model if j == 0 else None)
            ax.set_xticks([1, 5, 9]); ax.set_xticklabels(S.DIFF); ax.set_title(f"{S.LABEL[ds]}: {MLAB[m]}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure E. Stochastic models: K1 (plain) / mean over 10 (//) / best-of-10 per metric (xx) vs the deterministic models (dashed), protocol A", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figE_k1_mean_best.png", dpi=130); plt.close(fig)


def fig_f(curves):
    keys = ["dense", "part", "amount", "centroid", "normal", "wrench", "jitter", "jitter_r2"]
    fig, axes = plt.subplots(2, len(keys), figsize=(26, 6.5))
    for i, ds in enumerate(S.DATASETS):
        for j, k in enumerate(keys):
            ax = axes[i, j]
            for model in list(S.MODELS) + ["GT", "PERSIST"]:
                key = f"{ds}|A|{model}|{k}|K1"
                if key not in curves:
                    continue
                y = curves[key]; t = np.arange(len(y)) + (0.5 if k.startswith("jitter") else 0)
                if model == "GT" and not k.startswith("jitter"):
                    continue
                ax.plot(t, y, color=COL[model], lw=1.6 if model in ("GT",) else 1.1, ls="--" if model in ("GT", "PERSIST") else "-", label=model)
            ax.set_title(f"{S.LABEL[ds]}: {MLAB.get({'dense': 'E_C', 'part': 'part_hamming', 'amount': 'amount_l1', 'wrench': 'q_rel_l1', 'jitter': 'jitter_dense'}.get(k, k), k)}", fontsize=9); ax.set_xlabel("frame t")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Figure F. Error vs time from the initial frame (protocol A, K1, seed-averaged); jitter panels: frame-to-frame change of the generated and GT trajectories", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figF_temporal_curves.png", dpi=130); plt.close(fig)


def fig_g(ev):
    keys = ["dense_post", "part_post", "amount_post", "centroid_post", "normal_post", "wrench_post", "transition_realised", "transition_timing_error", "D_C_window", "D_Z_window"]
    fig, axes = plt.subplots(2, len(keys), figsize=(30, 6.5))
    for i, ds in enumerate(S.DATASETS):
        for j, k in enumerate(keys):
            ax = axes[i, j]
            for x, model in enumerate(S.MODELS):
                for ci, cls in enumerate(S.EVENT_CLASSES):
                    r = ev[(ev.dataset == ds) & (ev.model == model) & (ev.stat == "K1") & (ev.cls == cls) & (ev.metric == k)]
                    if len(r):
                        v, lo, hi = r.iloc[0].value, r.iloc[0].ci_lo, r.iloc[0].ci_hi
                        ax.bar(ci * 7 + x, v, color=COL[model], yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], capsize=1.5, label=model if ci == 0 else None)
            ax.set_xticks([3 + 7 * ci for ci in range(4)]); ax.set_xticklabels([S.EVENT_LABEL[c] for c in S.EVENT_CLASSES], fontsize=7, rotation=15); ax.set_title(f"{S.LABEL[ds]}: {k}", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("Figure G. Event-conditioned metrics at the post-transition frame (protocol A, K1; D_C / D_Z over the event window for the stochastic models)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "figG_events.png", dpi=120); plt.close(fig)


def fig_h(ds):
    """Qualitative examples: canonical points coloured by contact at five frames for the GT, D1 and three S1 samples (same s_0),
    with the R2 trajectories (participation raster, amounts, centroid traces). One dataset per process."""
    import torch
    from sat_data import SeqData
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sel_rows = []
    for ds in [ds]:
        data = SeqData(ds, dev, need_masks=False)
        zD = {m: np.load(S.metrics_path(ds, S.run_name(m, 0), "A"), allow_pickle=True) for m in ("D0", "D1") if S.metrics_path(ds, S.run_name(m, 0), "A").exists()}
        zS = np.load(S.metrics_path(ds, S.run_name("S1", 0), "A"), allow_pickle=True) if S.metrics_path(ds, S.run_name("S1", 0), "A").exists() else None
        if zS is None or "D1" not in zD:
            continue
        pS = np.load(S.preds_path(ds, S.run_name("S1", 0), "A"))["pred"].astype(np.float32); pD = np.load(S.preds_path(ds, S.run_name("D1", 0), "A"))["pred"].astype(np.float32)
        ex = zS["example"]; DC, DZ = zS["D_C"], zS["D_Z"]; EC = np.nanmean(zS["E_C"], 1); jit = np.nanmean(zS["jitter_dense"], 1)
        gt_jit = np.load(S.metrics_path(ds, "GT", "A"))["jitter_dense"][:, 0]
        comp = lambda z: np.nanmean(np.stack([z["part_hamming"][:, 0], z["amount_l1"][:, 0], z["centroid"][:, 0], z["normal"][:, 0] / 90.0]), 0)
        cases = {"dense variation, stable structure": int(np.argmax(np.where(DZ < np.nanpercentile(DZ, 30), DC, -1))),
                 "largest structural branching": int(np.argmax(DZ)),
                 "noisiest diffusion (jitter / GT jitter)": int(np.argmax(jit / np.maximum(gt_jit, 1e-6))),
                 "largest structural-loss improvement (D1 vs D0)": int(np.argmax(comp(zD["D0"]) - comp(zD["D1"]))) if "D0" in zD else 0}
        frames = [0, 8, 16, 32, 48, 63]
        for case, i in cases.items():
            n = int(ex[i]); P = data.geo.X_top[data.geo.geo_index[n]] + data.geo.X_bot[data.geo.geo_index[n]]; P = P.cpu().numpy()
            # 2-D view: the two principal axes of the canonical points
            Pc = P - P.mean(0); u, s_, vt = np.linalg.svd(Pc, full_matrices=False); xy = Pc @ vt[:2].T
            rows = [("GT", data.C[n].cpu().numpy()), ("D1", pD[i, 0])] + [(f"S1 sample {k}", pS[i, k]) for k in range(3)]
            fig, axes = plt.subplots(len(rows) + 2, len(frames), figsize=(2.6 * len(frames), 2.3 * (len(rows) + 2)))
            for r, (lab, Cm) in enumerate(rows):
                for c_, t in enumerate(frames):
                    ax = axes[r, c_]; ax.scatter(xy[:, 0], xy[:, 1], c=np.clip(Cm[t], 0, 1), cmap="viridis", s=6, vmin=0, vmax=1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
                    if c_ == 0:
                        ax.set_ylabel(lab, fontsize=9)
                    if r == 0:
                        ax.set_title(f"t = {t}", fontsize=9)
            # R2 trajectories: participation raster (GT / D1 / S1 samples) and amount curves
            a_true = zS["a_true"][i]; aD = zD["D1"]["r2_a"][i, 0]; aS = zS["r2_a"][i, :3]
            ax = axes[len(rows), 0]; ax.imshow(np.concatenate([a_true, aD] + list(aS), 1).T, aspect="auto", cmap="Greys", interpolation="nearest"); ax.set_title("participation: GT | D1 | S1 x3", fontsize=8); ax.set_yticks([]); ax.set_xlabel("frame")
            ax = axes[len(rows), 1]; mD = zD["D1"]["r2_m"][i, 0].astype(np.float32); mS = zS["r2_m"][i, :3].astype(np.float32)
            mt = data.m[n].cpu().numpy()
            ax.plot(mt.sum(1), "k", label="GT exact"); ax.plot(mD.sum(1), color=COL["D1"], label="D1"); [ax.plot(mS[k].sum(1), color=COL["S1"], alpha=0.6, lw=0.8) for k in range(3)]; ax.set_title("total amount", fontsize=8); ax.legend(fontsize=6)
            for k_, part in enumerate(S.PARTS[1:4]):
                ax = axes[len(rows), 2 + k_]; j = 1 + k_
                pt = data.p[n].cpu().numpy()[:, j]; pDj = zD["D1"]["r2_p"][i, 0].astype(np.float32)[:, j]; pSj = zS["r2_p"][i, :3].astype(np.float32)[:, :, j]
                for d_ in range(3):
                    ax.plot(pt[:, d_], color="k", lw=0.8); ax.plot(pDj[:, d_], color=COL["D1"], lw=0.8); [ax.plot(pSj[k][:, d_], color=COL["S1"], lw=0.5, alpha=0.5) for k in range(3)]
                ax.set_title(f"{part} centroid xyz / l", fontsize=8)
            ax = axes[len(rows) + 1, 0]; ax.plot(zS["curve_dense"][i, :3].T, color=COL["S1"], lw=0.8); ax.plot(zD["D1"]["curve_dense"][i, 0], color=COL["D1"]); ax.set_title("dense error vs t", fontsize=8)
            ax = axes[len(rows) + 1, 1]; ax.plot(zS["curve_D_C_t"][i], color="k"); ax.set_title("D_C(t) across 10 samples", fontsize=8)
            ax = axes[len(rows) + 1, 2]; ax.plot(zS["curve_D_Z_t"][i], color="k"); ax.set_title("D_Z(t) across 10 samples", fontsize=8)
            for c_ in range(3, len(frames)):
                axes[len(rows) + 1, c_].axis("off")
            fig.suptitle(f"Figure H ({S.LABEL[ds]}): {case} — example {n} ({data.meta.iloc[n].category}, {data.meta.iloc[n].hand}); D_C {DC[i]:.2f}, D_Z {DZ[i]:.2f}, E_C(S1 mean) {EC[i]:.2f}", fontsize=10, y=0.995)
            fig.tight_layout(rect=(0, 0, 1, 0.98)); name = f"figH_{ds}_{case.split(',')[0].split(' (')[0].replace(' ', '_')}.png"; fig.savefig(FIG / name, dpi=110); plt.close(fig)
            sel_rows.append(dict(dataset=ds, case=case, example=n, category=data.meta.iloc[n].category, D_C=float(DC[i]), D_Z=float(DZ[i]), E_C_S1_mean=float(EC[i]), figure=name))
    pd.DataFrame(sel_rows).to_csv(FIG / f"figH_selection_{ds}.csv", index=False)


def main():
    import argparse, subprocess, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--only-h", choices=S.DATASETS, help="draw figure H of one dataset (internal)")
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    if a.only_h:
        fig_h(a.only_h); return
    acc = pd.concat([pd.read_csv(S.OUT / "fixed_s0_metrics.csv"), pd.read_csv(S.OUT / "end_to_end_metrics.csv")], ignore_index=True)
    div = pd.read_csv(S.OUT / "diversity_metrics.csv"); ev = pd.read_csv(S.OUT / "event_metrics.csv")
    curves = dict(np.load(S.OUT / "temporal_curves.npz"))
    fig_a(acc); fig_b(acc); fig_c(div); fig_d(div); fig_e(acc); fig_f(curves); fig_g(ev)
    for ds in S.DATASETS:                                                     # figure H: one dataset per process (loader constraint)
        r = subprocess.run([sys.executable, __file__, "--only-h", ds], capture_output=True, text=True)
        if r.returncode != 0:
            log.warning("figure H (%s) skipped: %s", ds, r.stderr[-800:])
    sel = [pd.read_csv(FIG / f"figH_selection_{ds}.csv") for ds in S.DATASETS if (FIG / f"figH_selection_{ds}.csv").exists()]
    if sel:
        pd.concat(sel, ignore_index=True).to_csv(FIG / "figH_selection.csv", index=False)
    log.info("figures written to %s", FIG)


if __name__ == "__main__":
    main()
