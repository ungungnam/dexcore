#!/usr/bin/env python
"""Figures of one dataset (after aggregate.py):  DC_DATASET=<ds> python figures.py
  fig1_error_vs_t      reconstruction error vs rollout frame
  fig2_temporal_vs_t   temporal error vs frame + predicted / true frame-to-frame change magnitude
  fig3_error_vs_K      E_C (and S0 error) vs K for best-of-K, with the mean-over-samples and the B0/B1 levels
  fig4_gtinit_vs_sampled   per-group E_C of B0 / B1 / B2 (K=1, best-of-10) + per-example scatter B1 vs B2
  fig5_G_vs_GT         per-example E_C and S0 error of the two samplers (best-of-10), per-group paired differences
  fig6_qual_<kind>     contact-map evolution: GT, B1, B2 (best S0), B3 rows at 6 frames; stable / difficult / high-drift
  fig7_sampler_modes   10 S0 samples of one geometry against the GT C0 of the test takes of the same mesh
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hc_common as H
from hc_common import C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("fig")
FIG = H.OUT / "figures"
RES = H.OUT / "results"
STYLE = {("static_gt", 0): ("B0 static GT-init", "k", "--"), ("gtinit_vf", 0): ("B1 GT-init + VF", "C0", "-"),
         ("samplerG_vf", 1): ("B2 p(S0|G)+VF K=1", "C1", ":"), ("samplerG_vf", 10): ("B2 p(S0|G)+VF best-of-10", "C1", "-"),
         ("samplerGT_vf", 1): ("B3 p(S0|G,τ)+VF K=1", "C2", ":"), ("samplerGT_vf", 10): ("B3 p(S0|G,τ)+VF best-of-10", "C2", "-"),
         ("samplerG_static", 10): ("p(S0|G) static best-of-10", "C1", "--")}
QUAL_T = [0, 8, 16, 32, 48, 63]


def curves(split):
    return pd.read_csv(RES / "curves.csv").query("split == @split")


def bimart_curves():
    """Mean per-t curves of the BimArt rows, or {} when the comparison was not run."""
    p = H.OUT / "eval" / "fixed0" / "bimart_curves.npz"
    if not p.exists():
        return {}
    z = np.load(p); out = {}
    for key in z.files:
        if key == "example":
            continue
        name, Kk, met = key.split("__"); out[(name, int(Kk[1:]), met)] = np.nanmean(z[key], 0)
    return out


BIMART_STYLE = {("bimart", 1): ("B4 BimArt K=1", "C5", ":"), ("bimart", 10): ("B4 BimArt best-of-10", "C5", "-")}


def fig1_2(split):
    cv = curves(split)
    bc = bimart_curves()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (m, K), (lab, col, ls) in STYLE.items():
        g = cv[(cv.model == m) & (cv.K == K) & (cv.metric == "err")]
        if len(g):
            ax[0].plot(g.t, g.value, color=col, ls=ls, label=lab)
        g = cv[(cv.model == m) & (cv.K == K) & (cv.metric == "derr")]
        if len(g) and m != "static_gt":
            ax[1].plot(g.t, g.value, color=col, ls=ls, label=lab)
    for (m, K), (lab, col, ls) in BIMART_STYLE.items():
        if (m, K, "err") in bc:
            ax[0].plot(np.arange(len(bc[(m, K, "err")])), bc[(m, K, "err")], color=col, ls=ls, label=lab + " (vs full GT)")
            ax[1].plot(np.arange(len(bc[(m, K, "derr")])), bc[(m, K, "derr")], color=col, ls=ls, label=lab)
    ax[0].set(xlabel="frame t after the first firm contact", ylabel="||Ĉ_t − C_t||  (raw)", title=f"{H.DATASET}: reconstruction error vs rollout time ({split})")
    ax[1].set(xlabel="frame t", ylabel="||ΔĈ_t − ΔC_t||  (raw)", title="temporal error vs rollout time")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"fig1_error_vs_t_{split}.png", dpi=130); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for (m, K), (lab, col, ls) in STYLE.items():
        g = cv[(cv.model == m) & (cv.K == K) & (cv.metric == "derr")]
        if len(g) and m != "static_gt":
            ax[0].plot(g.t, g.value, color=col, ls=ls, label=lab)
        g = cv[(cv.model == m) & (cv.K == K) & (cv.metric == "dmag")]
        if len(g) and m != "static_gt":
            ax[1].plot(g.t, g.value, color=col, ls=ls, label=lab)
    A = pd.read_csv(RES / "aggregate.csv").query("split == @split")
    true = A[(A.model == "static_gt")].dmag_true.iloc[0]
    # the true magnitude curve = the temporal error of the static model (Ĉ constant)
    g = cv[(cv.model == "static_gt") & (cv.K == 0) & (cv.metric == "derr")]
    ax[1].plot(g.t, g.value, color="k", ls="--", label="true ||ΔC_t||")
    ax[0].set(xlabel="frame t", ylabel="||ΔĈ_t − ΔC_t||", title=f"{H.DATASET}: temporal error ({split})")
    ax[1].set(xlabel="frame t", ylabel="||ΔĈ_t||", title=f"predicted vs true frame-to-frame change (true mean {true:.3f})")
    ax[0].legend(fontsize=7); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"fig2_temporal_vs_t_{split}.png", dpi=130); plt.close(fig)


def fig3(split):
    A = pd.read_csv(RES / "aggregate.csv").query("split == @split").set_index(["model", "K"])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for m, col, lab in (("samplerG_vf", "C1", "B2 p(S0|G)+VF"), ("samplerGT_vf", "C2", "B3 p(S0|G,τ)+VF")):
        Ks = [k for k in (1, 5, 10) if (m, k) in A.index]
        v = [A.loc[(m, k), "E_C"] for k in Ks]; lo = [A.loc[(m, k), "E_C_lo"] for k in Ks]; hi = [A.loc[(m, k), "E_C_hi"] for k in Ks]
        ax[0].errorbar(Ks, v, yerr=[np.array(v) - lo, np.array(hi) - v], color=col, marker="o", label=f"{lab} best-of-K")
        if (m, -1) in A.index:
            ax[0].axhline(A.loc[(m, -1), "E_C"], color=col, ls=":", label=f"{lab} mean over samples")
        ms = m.replace("_vf", "_static")
        Ks2 = [k for k in (1, 5, 10) if (ms, k) in A.index]
        ax[1].plot(Ks2, [A.loc[(ms, k), "s0_err"] for k in Ks2], color=col, marker="s", label=f"{lab.split('+')[0]} S0 error, best-of-K (by E_C)")
    rp = RES / "reference_baselines_summary.csv"
    if rp.exists():
        Rf = pd.read_csv(rp).set_index(["model", "K"])
        Ks3 = [k for k in (1, 10) if ("ref_take_static", k) in Rf.index]
        ax[1].plot(Ks3, [Rf.loc[("ref_take_static", k), "s0_err"] for k in Ks3], color="0.4", marker="^", ls="--", label="random training take of the same mesh (empirical prior)")
        ax[1].axhline(Rf.loc[("ref_mean_static", 0), "s0_err"], color="0.4", ls=":", label="mean training map of the mesh")
        ax[1].axhline(Rf.loc[("ref_nearest_static", 0), "s0_err"], color="0.4", ls="-.", label="nearest training map (retrieval oracle)")
        ax[0].plot(Ks3, [Rf.loc[("ref_take_static", k), "E_C"] for k in Ks3], color="0.4", marker="^", ls="--", label="random training take held static (K=1, best-of-10)")
    bp = RES / "bimart_comparison.csv"
    if bp.exists():
        Bm = pd.read_csv(bp).set_index(["model", "K"])
        Ks4 = [k for k in (1, 5, 10) if ("bimart", k) in Bm.index]
        ax[0].plot(Ks4, [Bm.loc[("bimart", k), "E_C"] for k in Ks4], color="C5", marker="D", label="B4 BimArt contact stage (vs full GT)")
        ax[0].plot(Ks4, [Bm.loc[("bimart_vs_gtr", k), "E_C"] for k in Ks4], color="C5", marker="D", ls="--", label="B4 BimArt on its own BPS support")
        ax[0].axhline(Bm.loc[("static_gtr_vs_gtr", 0), "E_C"], color="C5", ls=":", label="B0 on BimArt's support")
    ax[0].axhline(A.loc[("gtinit_vf", 0), "E_C"], color="C0", label="B1 GT-init + VF")
    ax[0].axhline(A.loc[("static_gt", 0), "E_C"], color="k", ls="--", label="B0 static GT-init")
    ax[0].set(xlabel="K", ylabel="E_C (raw)", title=f"{H.DATASET}: best-of-K rollout error ({split})", xticks=[1, 5, 10]); ax[0].legend(fontsize=7)
    ax[1].set(xlabel="K", ylabel="||S0 − C0||  (raw)", title="initial-contact error of the selected sample", xticks=[1, 5, 10]); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / f"fig3_error_vs_K_{split}.png", dpi=130); plt.close(fig)


def per_example(split):
    import glob
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(H.OUT / "eval" / f"{split}*" / "per_example.csv")))], ignore_index=True)


def fig4(split):
    G = pd.read_csv(RES / "per_group.csv").query("split == @split")
    R = per_example(split)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), gridspec_kw=dict(width_ratios=[2, 1]))
    groups = sorted(G.group.unique()); x = np.arange(len(groups)); w = 0.2
    for i, ((m, K), (lab, col, _)) in enumerate([(k, STYLE[k]) for k in (("static_gt", 0), ("gtinit_vf", 0), ("samplerG_vf", 1), ("samplerG_vf", 10))]):
        v = G[(G.model == m) & (G.K == K)].set_index("group").reindex(groups).E_C
        ax[0].bar(x + (i - 1.5) * w, v, w, color=col, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(groups, rotation=90, fontsize=6 if len(groups) > 12 else 8); ax[0].set(ylabel="E_C (raw)", title=f"{H.DATASET}: per-group error, GT-init vs sampled init ({split})"); ax[0].legend(fontsize=7)
    a = R[(R.model == "gtinit_vf") & (R.K == 0)].set_index("example").E_C; b = R[(R.model == "samplerG_vf") & (R.K == 10)].set_index("example").E_C
    common = a.index.intersection(b.index); m = max(a.max(), b.max())
    ax[1].scatter(a[common], b[common], s=4, alpha=0.4); ax[1].plot([0, m], [0, m], "k--", lw=0.8)
    ax[1].set(xlabel="B1 GT-init + VF  E_C", ylabel="B2 p(S0|G)+VF best-of-10  E_C", title="per sequence")
    fig.tight_layout(); fig.savefig(FIG / f"fig4_gtinit_vs_sampled_{split}.png", dpi=130); plt.close(fig)


def fig5(split):
    R = per_example(split)
    P = pd.read_csv(RES / "paired_differences.csv").query("split == @split")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    for j, (met, K, mG, mGT, lab) in enumerate([("E_C", 10, "samplerG_vf", "samplerGT_vf", "rollout E_C, best-of-10"), ("s0_err", 10, "samplerG_static", "samplerGT_static", "S0 error, best-of-10 (by E_C)")]):
        a = R[(R.model == mG) & (R.K == K)].set_index("example")[met]; b = R[(R.model == mGT) & (R.K == K)].set_index("example")[met]
        common = a.index.intersection(b.index); m = max(a.max(), b.max())
        ax[j].scatter(a[common], b[common], s=4, alpha=0.4); ax[j].plot([0, m], [0, m], "k--", lw=0.8)
        ax[j].set(xlabel=f"p(S0|G): {lab}", ylabel=f"p(S0|G,τ_local): {lab}", title=f"{H.DATASET} ({split}): {lab}")
    G = pd.read_csv(RES / "per_group.csv").query("split == @split")
    groups = sorted(G.group.unique())
    d = (G[(G.model == "samplerGT_vf") & (G.K == 10)].set_index("group").reindex(groups).E_C - G[(G.model == "samplerG_vf") & (G.K == 10)].set_index("group").reindex(groups).E_C)
    ax[2].barh(np.arange(len(groups)), d.values, color=np.where(d.values < 0, "C2", "C1")); ax[2].set_yticks(np.arange(len(groups))); ax[2].set_yticklabels(groups, fontsize=6 if len(groups) > 12 else 8)
    ax[2].axvline(0, color="k", lw=0.8); ax[2].set(xlabel="E_C(B3) − E_C(B2), best-of-10  (< 0: trajectory context helps)", title="per group")
    fig.tight_layout(); fig.savefig(FIG / f"fig5_G_vs_GT_{split}.png", dpi=130); plt.close(fig)


def project(P):
    Pc = P - P.mean(0); U, s, Vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ Vt[:2].T


def fig6(split):
    """Qualitative examples from fold 0 of the split: stable (lowest true change among sequences with
    above-median mass), difficult (highest B1 error), high-drift (largest ||C_63 - C_0||)."""
    import glob
    z, meta = H.load_sequences()
    d = sorted(glob.glob(str(H.OUT / "eval" / f"{split}0")))[0]
    pr = np.load(d + "/preds.npz"); R = pd.read_csv(d + "/per_example.csv")
    ex = pr["example"]; gt = pr["gt"].astype(np.float32)
    b1 = R[(R.model == "gtinit_vf") & (R.K == 0)].set_index("example")
    mass = gt.sum(2).mean(1); change = np.linalg.norm(gt[:, 1:] - gt[:, :-1], axis=2).mean(1); drift = np.linalg.norm(gt[:, -1] - gt[:, 0], axis=1)
    cand = {"stable": int(np.argmin(np.where(mass >= np.median(mass), change, np.inf))),
            "difficult": int(np.argmax(b1.reindex(ex).E_C.values)), "high_drift": int(np.argmax(drift))}
    for kind, i in cand.items():
        n = ex[i]; row = meta.iloc[n]
        P2 = project(z["canonical_points"][list(z["canonical_cats"]).index(row.category)])
        rows = [("GT", gt[i]), ("B1 GT-init + VF", pr["gtinit_vf"][i].astype(np.float32)),
                ("B2 p(S0|G)+VF (best of 10)", pr["samplerG_vf__K10"][i].astype(np.float32)), ("B3 p(S0|G,τ)+VF (best of 10)", pr["samplerGT_vf__K10"][i].astype(np.float32))]
        vmax = max(gt[i].max(), 1e-3)
        fig, axes = plt.subplots(len(rows), len(QUAL_T) + 1, figsize=(2.1 * (len(QUAL_T) + 1), 1.35 * len(rows) + 0.5))
        for r, (lab, seq) in enumerate(rows):
            for c, t in enumerate(QUAL_T):
                ax = axes[r, c + 1]; ax.scatter(P2[:, 0], P2[:, 1], c=np.clip(seq[t], 0, vmax), s=6, cmap="inferno", vmin=0, vmax=vmax); ax.set_aspect("equal"); ax.axis("off")
                if r == 0:
                    ax.set_title(f"t = {t}", fontsize=9)
            ax = axes[r, 0]; ax.axis("off"); ax.text(0.0, 0.5, lab, fontsize=8, va="center")
        # the sampled S0 (sample 0 and the best-of-10 rollout's own S0) in the label column of rows 2-3
        for r, key, k_model in ((2, "S0_G", "samplerG_vf"), (3, "S0_GT", "samplerGT_vf")):
            kk = int(R[(R.model == k_model) & (R.K == 10) & (R.example == n)].k_star.iloc[0])
            ax = axes[r, 0]; ax.cla(); ax.axis("off")
            ax.scatter(P2[:, 0], P2[:, 1], c=np.clip(pr[key][i, kk].astype(np.float32), 0, vmax), s=4, cmap="inferno", vmin=0, vmax=vmax); ax.set_aspect("equal")
            ax.set_title(f"{rows[r][0].split(' (')[0]}\nS0 sample k*={kk}", fontsize=7)
        e = R[(R.example == n) & (R.K.isin([0, 10]))].set_index("model").E_C
        fig.suptitle(f"{H.DATASET} {kind}: {row.group}, take {str(row.sequence_id)[:40]}, onset frame {row.t0}, mass {mass[i]:.0f}, true Δ {change[i]:.3f} | "
                     f"E_C B0 {e.get('static_gt', np.nan):.2f} B1 {e.get('gtinit_vf', np.nan):.2f} B2 {e.get('samplerG_vf', np.nan):.2f} B3 {e.get('samplerGT_vf', np.nan):.2f}", fontsize=8)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.02, hspace=0.05); fig.savefig(FIG / f"fig6_qual_{kind}_{split}.png", dpi=120); plt.close(fig)
        log.info("qualitative %s: example %d (%s)", kind, n, row.group)


def fig7(split):
    import glob
    z, meta = H.load_sequences()
    d = sorted(glob.glob(str(H.OUT / "eval" / f"{split}0")))[0]
    pr = np.load(d + "/preds.npz"); ex = pr["example"]
    sub = meta.iloc[ex]
    mesh = sub.groupby(["group", "mesh_id"]).size().idxmax()          # the (group, mesh) with most test takes
    sel = np.where((sub.group == mesh[0]).values & (sub.mesh_id == mesh[1]).values)[0][:10]
    P2 = project(z["canonical_points"][list(z["canonical_cats"]).index(sub.iloc[sel[0]].category)])
    gt = pr["gt"][sel, 0].astype(np.float32); s0 = pr["S0_G"][sel[0]].astype(np.float32); s0t = pr["S0_GT"][sel[0]].astype(np.float32)
    vmax = max(gt.max(), 1e-3)
    fig, axes = plt.subplots(3, 10, figsize=(20, 6.5))
    for c in range(10):
        for r, (arr, lab) in enumerate(((gt, "GT C0 of test take"), (s0, "p(S0|G) sample"), (s0t, "p(S0|G,τ) sample"))):
            ax = axes[r, c]; ax.axis("off")
            if c < len(arr):
                ax.scatter(P2[:, 0], P2[:, 1], c=np.clip(arr[c], 0, vmax), s=5, cmap="inferno", vmin=0, vmax=vmax); ax.set_aspect("equal")
                ax.set_title(f"{lab} {c}", fontsize=7)
    fig.suptitle(f"{H.DATASET}: first-contact maps of {len(sel)} test takes of {mesh[0]} / mesh {mesh[1]} (top) vs 10 samples of each sampler for take 0 (geometry only, and + local trajectory)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / f"fig7_sampler_modes_{split}.png", dpi=110); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    A = pd.read_csv(RES / "aggregate.csv")
    for split in sorted(A.split.unique()):
        fig1_2(split); fig3(split); fig4(split); fig5(split)
        try:
            fig6(split); fig7(split)
        except Exception as e:  # noqa: BLE001
            log.warning("qualitative figures failed for %s: %s", split, e)
    log.info("figures written to %s", FIG)


if __name__ == "__main__":
    main()
