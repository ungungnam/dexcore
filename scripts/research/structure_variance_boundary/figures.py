#!/usr/bin/env python
"""Figures A–G of the structure / variance boundary report (from the aggregate tables and the caches).
    python figures.py [--only A B ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sv_common as S
import wc_common as W

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("figures")
FIG = S.OUT / "figures"
ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "CONTACT_FULL", "FULL"]
SHORT = {"R0": "R0", "R1": "R1", "R2": "R2", "R3": "R3", "R4": "R4", "R5": "R5", "CONTACT_FULL": "C_full", "FULL": "FULL"}
COL = {"taco": "tab:blue", "arctic": "tab:orange"}
HCOL = {1: "tab:green", 4: "tab:blue", 8: "tab:red"}


def xt(ax):
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels([f"{SHORT[r]}\n({S.REP_DIM[r]})" for r in ORDER], fontsize=8)


def fig_a():
    fp = pd.read_csv(S.OUT / "functional_probe.csv")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for c, ds in enumerate(S.DATASETS):
        d = fp[(fp.dataset == ds) & (fp.probe == "mlpG_seedavg") & (fp.target == "q")].set_index("rep")
        r = fp[(fp.dataset == ds) & (fp.probe == "ridgeG") & (fp.target == "q")].set_index("rep")
        st = fp[(fp.dataset == ds) & (fp.probe == "mlpG") & (fp.target == "q_strict")].set_index("rep")
        for row, (m, lab) in enumerate((("rel_l1", "relative L1 error of the 76-D profile"), ("cosine", "profile cosine"))):
            a = ax[row, c]
            y = [d.loc[rep, m] if rep in d.index else np.nan for rep in ORDER]
            lo = [d.loc[rep, f"{m}_lo"] if rep in d.index else np.nan for rep in ORDER]; hi = [d.loc[rep, f"{m}_hi"] if rep in d.index else np.nan for rep in ORDER]
            a.errorbar(range(len(ORDER)), y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], marker="o", color=COL[ds], lw=2, capsize=3, label="MLP probe + G (3 seeds, take-bootstrap 95 % CI)")
            if len(r):
                a.plot(range(len(ORDER)), [r.loc[rep, m] if rep in r.index else np.nan for rep in ORDER], marker="s", ls="--", color="0.4", label="ridge probe + G")
            if "MEAN" in d.index:
                a.axhline(d.loc["MEAN", m], color="k", lw=1, ls="-.", label="train-mean profile")
            xt(a); a.set_ylabel(lab); a.grid(alpha=0.3); a.set_title(f"{S.LABEL[ds]}: representation -> wrench profile q_t (test frames in contact)", fontsize=10)
            h1, l1 = a.get_legend_handles_labels(); h2, l2 = [], []
            if len(st):                                                  # strict profile on its own (right) axis: different scale
                a2 = a.twinx()
                a2.plot(range(len(ORDER)), [st.loc[rep, m] if rep in st.index else np.nan for rep in ORDER], marker="^", ls=":", color="tab:purple", label="MLP probe + G, strict LP profile (right axis)")
                a2.set_ylabel(f"{lab} — strict profile", color="tab:purple", fontsize=8); a2.tick_params(axis="y", colors="tab:purple")
                h2, l2 = a2.get_legend_handles_labels()
            if row == 0:
                a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")
    fig.tight_layout(); fig.savefig(FIG / "figA_functional_ladder.png", dpi=130); plt.close(fig)


def fig_b():
    tp = pd.read_csv(S.OUT / "temporal_prediction.csv")
    panels = [("T1", "auprc_macro", "T1 future participation: macro AUPRC (higher = better)"), ("T2", "T2_rel_l1", "T2 future amount: normalised L1 (÷ train-mean total amount)"),
              ("T3", "T3_cent", "T3 future centroid: distance / l (active parts)"), ("T3", "T3_ang", "T3 future normal: angular error [deg]"), ("T4", "T4_rel_l1", "T4 future wrench profile: relative L1")]
    fig, ax = plt.subplots(len(panels), 2, figsize=(13, 3.1 * len(panels)))
    for c, ds in enumerate(S.DATASETS):
        for row, (tg, m, lab) in enumerate(panels):
            a = ax[row, c]
            for h in S.HORIZONS:
                d = tp[(tp.dataset == ds) & (tp.metric == m) & (tp.h == h)].set_index("rep")
                y = [d.loc[rep, "value"] if rep in d.index else np.nan for rep in ORDER]
                lo = [d.loc[rep, "lo"] if rep in d.index else np.nan for rep in ORDER]; hi = [d.loc[rep, "hi"] if rep in d.index else np.nan for rep in ORDER]
                a.errorbar(range(len(ORDER)), y, yerr=[np.nan_to_num(np.array(y) - np.array(lo)), np.nan_to_num(np.array(hi) - np.array(y))], marker="o", color=HCOL[h], lw=1.8, capsize=2, label=f"h = {h}")
                if "PERSIST" in d.index:
                    a.axhline(d.loc["PERSIST", "value"], color=HCOL[h], lw=1, ls=":", alpha=0.8)
                if "MEAN" in d.index:
                    a.axhline(d.loc["MEAN", "value"], color=HCOL[h], lw=1, ls="-.", alpha=0.5)
            xt(a); a.set_ylabel(lab, fontsize=8); a.grid(alpha=0.3)
            if row == 0:
                a.set_title(f"{S.LABEL[ds]}: causal 8-frame window of each representation + G + tau_local -> targets at t + h", fontsize=9); a.legend(fontsize=7, title="dotted: persistence, dash-dot: train mean", title_fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figB_temporal_ladder.png", dpi=130); plt.close(fig)


def fig_c():
    sat = pd.read_csv(S.OUT / "saturation_summary.csv"); ch = pd.read_csv(S.OUT / "saturation_choice.csv")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for c, ds in enumerate(S.DATASETS):
        d = sat[sat.dataset == ds].set_index("rep"); a = ax[c]
        x = [S.REP_DIM[r] for r in ORDER]
        fg = [d.loc[r, "functional_gap"] if r in d.index else np.nan for r in ORDER]; tg = [d.loc[r, "temporal_gap"] if r in d.index else np.nan for r in ORDER]
        a.errorbar(x, fg, yerr=[[d.loc[r, "functional_gap"] - d.loc[r, "functional_gap_lo"] if r in d.index else 0 for r in ORDER], [d.loc[r, "functional_gap_hi"] - d.loc[r, "functional_gap"] if r in d.index else 0 for r in ORDER]],
                   marker="o", color="tab:red", lw=2, capsize=3, label="functional: rel. L1 of q vs FULL (gap)")
        a.errorbar(x, tg, yerr=[[d.loc[r, "temporal_gap"] - d.loc[r, "temporal_gap_lo"] if r in d.index else 0 for r in ORDER], [d.loc[r, "temporal_gap_hi"] - d.loc[r, "temporal_gap"] if r in d.index else 0 for r in ORDER]],
                   marker="s", color="tab:blue", lw=2, capsize=3, label="temporal: composite T1–T4 error vs FULL (gap, mean over h)")
        if "functional_gap_best" in d.columns:
            a.plot(x, [d.loc[r, "functional_gap_best"] if r in d.index else np.nan for r in ORDER], color="tab:red", lw=1, ls="--", alpha=0.7, label=f"functional gap to the best ({d.best_functional.iloc[0]})")
            a.plot(x, [d.loc[r, "temporal_gap_best"] if r in d.index else np.nan for r in ORDER], color="tab:blue", lw=1, ls="--", alpha=0.7, label=f"temporal gap to the best ({d.best_temporal.iloc[0]})")
        for thr, ls in zip(S.SATURATION_THRESHOLDS, (":", "--", "-.")):
            a.axhline(thr, color="0.5", lw=1, ls=ls, label=f"{thr * 100:g} % threshold")
        a.axhline(0, color="k", lw=0.8)
        for r in ORDER:
            a.annotate(SHORT[r], (S.REP_DIM[r], (fg[ORDER.index(r)] if np.isfinite(fg[ORDER.index(r)]) else 0)), fontsize=8, xytext=(3, 4), textcoords="offset points")
        for thr in S.SATURATION_THRESHOLDS:
            best = ch[(ch.dataset == ds) & (ch.threshold == thr) & (ch.criterion == "both") & (ch.reference == "FULL")].smallest_saturated.iloc[0]
            if best != "none":
                a.axvline(S.REP_DIM[best], color="tab:green", lw=1.5, alpha=0.6)
                a.text(S.REP_DIM[best], a.get_ylim()[1] * 0.95 if a.get_ylim()[1] > 0 else 0.5, f"saturated @{thr * 100:g} %: {best}", rotation=90, fontsize=7, va="top", ha="right", color="tab:green")
        a.set_xscale("log"); a.set_xlabel("representation dimension (per frame)"); a.set_ylabel("relative error gap to FULL"); a.grid(alpha=0.3)
        a.set_title(f"{S.LABEL[ds]}: Pareto / saturation", fontsize=10); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figC_pareto_saturation.png", dpi=130); plt.close(fig)


def fig_d():
    E = pd.read_csv(S.OUT / "event_representation_change.csv")
    E = E[(E.split == "test") & E.cls.isin(["persistent_spatial", "persistent_mixed"])]
    levels = ["R0", "R2", "R5"]
    fig, ax = plt.subplots(2, len(levels), figsize=(5 * len(levels), 9))
    for r, ds in enumerate(S.DATASETS):
        d = E[E.dataset == ds]
        for c, lv in enumerate(levels):
            a = ax[r, c]
            for cls, mk in (("persistent_spatial", "o"), ("persistent_mixed", "^")):
                dd = d[d.cls == cls]
                sc = a.scatter(dd.d_C, dd[f"d_{lv}"], c=dd.d_q, cmap="viridis", vmin=0, vmax=0.6, s=22, marker=mk, edgecolor="k", lw=0.3, label=cls.replace("_", " "))
            a.set_xlabel("dense contact change ||C_post - C_pre||"); a.set_ylabel(f"structured change d_{lv} (RMS z-score, {S.REP_DIM[lv]}-D)")
            a.set_title(f"{S.LABEL[ds]}: {S.REP_LABEL[lv]}", fontsize=9); a.grid(alpha=0.3)
            if c == 0:
                a.legend(fontsize=7)
            fig.colorbar(sc, ax=a, label="wrench-profile change d_q (rel. L1)")
    fig.suptitle("Event-level dense change vs structured change, coloured by the wrench-profile change (test persistent events)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "figD_dense_vs_structure.png", dpi=130); plt.close(fig)


def fig_e():
    inc = pd.read_csv(S.OUT / "incremental_gain.csv")
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c, ds in enumerate(S.DATASETS):
        d = inc[inc.dataset == ds]; a = ax[c]
        labels = [f"{r.frm}->{r.to}\n+{r.added}" for r in d.itertuples()]
        x = np.arange(len(d)); w = 0.25
        a.bar(x - w, 1 - d.functional_ratio, w, yerr=[np.nan_to_num(d.functional_ratio - d.functional_lo), np.nan_to_num(d.functional_hi - d.functional_ratio)], color="tab:red", capsize=2, label="functional: 1 - rel.L1(to)/rel.L1(from)")
        a.bar(x, 1 - d.temporal_ratio, w, yerr=[np.nan_to_num(d.temporal_ratio - d.temporal_lo), np.nan_to_num(d.temporal_hi - d.temporal_ratio)], color="tab:blue", capsize=2, label="temporal T1–T4 composite: 1 - err(to)/err(from)")
        if "dense_E_C_ratio" in d:
            a.bar(x + w, 1 - d.dense_E_C_ratio, w, yerr=[np.nan_to_num(d.dense_E_C_ratio - d.dense_E_C_lo), np.nan_to_num(d.dense_E_C_hi - d.dense_E_C_ratio)], color="0.6", capsize=2, label="dense diagnostic T5: 1 - E_C(to)/E_C(from)")
        a.axhline(0, color="k", lw=0.8); a.set_xticks(x); a.set_xticklabels(labels, fontsize=7); a.set_ylabel("relative error reduction from the added block"); a.grid(alpha=0.3, axis="y")
        a.set_title(f"{S.LABEL[ds]}: incremental information gain (take-bootstrap 95 % CI)", fontsize=10); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figE_feature_contribution.png", dpi=130); plt.close(fig)


def fig_f():
    if (S.OUT / "residual_prediction.csv").stat().st_size < 10:
        log.warning("figure F skipped: no residual results yet"); return
    rp = pd.read_csv(S.OUT / "residual_prediction.csv"); tr = pd.read_csv(S.OUT / "temporal_ratios.csv"); tp = pd.read_csv(S.OUT / "temporal_prediction.csv")
    ch = pd.read_csv(S.OUT / "saturation_choice.csv")
    models = ["zero", "last", "condmean", "history", "history_delta", "oracle", "oracle_delta"]
    labels = {"zero": "zero", "last": "last residual", "condmean": "mean | Z*_t", "history": "FULL history", "history_delta": "FULL + residual history, Δ on r_t",
              "oracle": "oracle: + GT Z*_{t+h} (non-causal)", "oracle_delta": "oracle Δ on r_t (non-causal)"}
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    for c, ds in enumerate(S.DATASETS):
        zs = sorted(rp[rp.dataset == ds].zstar.unique())
        if len(zs) == 0:
            continue
        pick = ch[(ch.dataset == ds) & (ch.threshold == 0.05) & (ch.criterion == "both") & (ch.reference == "FULL")].smallest_saturated.iloc[0]
        zstar = pick if pick in zs else zs[0]                                # the rule's Z* when Experiment D ran for it
        for r, blk in enumerate(("C", "H")):
            a = ax[r, c]; x = np.arange(len(S.HORIZONS)); w = 0.8 / len(models)
            for i, m in enumerate(models):
                d = rp[(rp.dataset == ds) & (rp.zstar == zstar) & (rp.model == m) & (rp.block == blk) & (rp.metric == "mse_ratio_to_zero")].set_index("h")
                y = [1 - d.loc[h, "value"] if h in d.index else np.nan for h in S.HORIZONS]
                lo = [1 - d.loc[h, "hi"] if h in d.index else np.nan for h in S.HORIZONS]; hi = [1 - d.loc[h, "lo"] if h in d.index else np.nan for h in S.HORIZONS]
                a.bar(x + (i - (len(models) - 1) / 2) * w, y, w, yerr=[np.nan_to_num(np.array(y) - np.array(lo)), np.nan_to_num(np.array(hi) - np.array(y))], capsize=2, label=f"residual: {labels[m]}")
            # structured-state predictability of FULL on the Z* blocks: R^2 vs the train-mean predictor (standardised MSE)
            sr = []
            for h in S.HORIZONS:
                d = tp[(tp.dataset == ds) & (tp.rep == "FULL") & (tp.h == h) & (tp.metric.isin(["mse_m", "mse_geom", "mse_q"]))]
                sr.append(1 - d.value.mean() if len(d) else np.nan)
            a.plot(x, sr, marker="D", color="k", lw=2, label="structured state (m, p/n, q) from FULL history: R² vs train mean")
            a.axhline(0, color="k", lw=0.8); a.set_xticks(x); a.set_xticklabels([f"h = {h}" for h in S.HORIZONS]); a.set_ylabel(f"R² of the future {'contact' if blk == 'C' else 'hand'} residual (1 - MSE / residual energy)")
            a.set_title(f"{S.LABEL[ds]}: Z* = {zstar}; residual of the {'512-D contact' if blk == 'C' else '300-D hand'} block", fontsize=10); a.grid(alpha=0.3, axis="y")
            if r == 0:
                a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figF_residual_predictability.png", dpi=130); plt.close(fig)


def fig_g(n_per=1):
    """Qualitative examples: (1) large dense change, small structured (Z*) change, small wrench change; (2) large dense,
    large structured, large wrench change; (3) small dense change but a large wrench change (if any). Drawn with the
    wrench report's pre / post renderer plus the per-part descriptors and the wrench profiles."""
    sys.path.insert(0, str(S.REPO / "scripts/research/wrench_counterfactual"))
    import sanity_viz as SV
    import wrench_analysis as WA
    E = pd.read_csv(S.OUT / "event_representation_change.csv"); ch = pd.read_csv(S.OUT / "saturation_choice.csv")
    sheet = []
    for ds in S.DATASETS:
        zstar = ch[(ch.dataset == ds) & (ch.threshold == 0.05) & (ch.criterion == "both") & (ch.reference == "FULL")].smallest_saturated.iloc[0]
        if zstar == "none":
            zstar = "R2"
        d = E[(E.dataset == ds) & (E.split == "test") & E.cls.isin(["persistent_spatial", "persistent_mixed"])].copy()
        trn = E[(E.dataset == ds) & (E.split == "train") & E.cls.isin(["persistent_spatial", "persistent_mixed"])]
        thr_C, thr_Z, thr_q = trn.d_C.median(), trn[f"d_{zstar}"].median(), trn.d_q.median()
        cases = {"variation": d[(d.d_C > thr_C) & (d[f"d_{zstar}"] <= thr_Z) & (d.d_q <= thr_q)].sort_values("d_C", ascending=False),
                 "structural": d[(d.d_C > thr_C) & (d[f"d_{zstar}"] > thr_Z) & (d.d_q > thr_q)].sort_values("d_q", ascending=False),
                 "hidden": d[(d.d_C <= thr_C) & (d.d_q > thr_q)].sort_values("d_q", ascending=False)}
        F = S.load_features(ds); blocks = S.block_arrays(F)
        Wev = W.load_selected(ds).set_index("event_id")
        for kind, sub in cases.items():
            for ev in sub.head(n_per).itertuples():
                wev = Wev.loc[ev.event_id]
                try:
                    g = WA.load_geometry(ds, ev.event_id); mesh = WA.mesh_of(ds, wev)
                except Exception as ex:  # noqa: BLE001
                    log.warning("no geometry for %s %s: %s", ds, ev.event_id, ex); continue
                lab = W.mano_labels()
                p_pre = W.build_patches(WA.geom_dict(g, "pre"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
                p_post = W.build_patches(WA.geom_dict(g, "post"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
                fig = plt.figure(figsize=(17, 14)); gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.7, 0.7])
                ax0 = [fig.add_subplot(gs[0, k]) for k in range(3)]; ax1 = [fig.add_subplot(gs[1, k]) for k in range(3)]
                SV.draw_frame(ax0, g, "pre", mesh, p_pre, lab, f"PRE frame {g['frames']['pre']}: {len(p_pre)} patches")
                SV.draw_frame(ax1, g, "post", mesh, p_post, lab, f"POST frame {g['frames']['post']}: {len(p_post)} patches")
                n, pre, post = int(ev.example), int(ev.pre_frame), int(ev.post_frame)
                a = fig.add_subplot(gs[2, :2])
                qp, qq = F["q"][n, pre], F["q"][n, post]; order = np.argsort(-qq)
                a.plot(qq[order], color="tab:blue", lw=2, label=f"post h(u), mean {qq.mean():.2f}"); a.plot(qp[order], color="tab:red", lw=2, label=f"pre h(u), mean {qp.mean():.2f}")
                a.set_xlabel("direction (sorted by the post capability)"); a.set_ylabel("directional capacity h(u)"); a.legend(fontsize=8)
                a.set_title(f"wrench profiles: d_q = {ev.d_q:.2f}, retention {ev.retention:.2f}, rel. Q change {ev.rel_change_Q:+.2f}", fontsize=9)
                a = fig.add_subplot(gs[2, 2])
                vals = [ev.d_C / max(thr_C, 1e-9), ev.d_H, ev.d_R0, ev.d_R1, ev.d_R2, ev.d_R3, ev.d_R4, ev.d_R5, ev.d_CONTACT_FULL, ev.d_FULL]
                names = ["d_C/median", "d_H", "d_R0", "d_R1", "d_R2", "d_R3", "d_R4", "d_R5", "d_Cfull", "d_FULL"]
                a.barh(names, vals, color=["0.3", "0.5"] + ["tab:green"] * 6 + ["0.6", "0.6"]); a.set_title("change magnitudes (z-scored RMS; d_C relative to the train median)", fontsize=8); a.tick_params(labelsize=7)
                a = fig.add_subplot(gs[3, :])
                rows = []
                for k in range(6):
                    for fr_, nm in ((pre, "pre"), (post, "post")):
                        i = fr_ + S.PAD
                        rows.append([W.PARTS[k], nm, int(blocks["part"][n, i, k]), f"{blocks['amount'][n, i, k]:.4f}", "(" + ", ".join(f"{v:+.2f}" for v in F["p"][n, i, k]) + ")" if blocks["part"][n, i, k] else "-",
                                     "(" + ", ".join(f"{v:+.2f}" for v in F["nrm"][n, i, k]) + ")" if blocks["part"][n, i, k] else "-", f"{F['s'][n, i, k]:.3f}" if blocks["part"][n, i, k] else "-", int(F["c"][n, i, k])])
                a.axis("off"); tb = a.table(cellText=rows, colLabels=["part", "frame", "a_k", "m_k", "p_k / l", "n_k", "s_k / l", "c_k"], loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 0.9)
                fig.suptitle(f"{S.LABEL[ds]} — {kind}: {ev.cls} event {ev.event_id} ({wev.sequence_id}, {wev.group}); ||dC|| = {ev.d_C:.2f} (train median {thr_C:.2f}), d_{zstar} = {getattr(ev, 'd_' + zstar):.2f} (median {thr_Z:.2f}), "
                             f"d_q = {ev.d_q:.2f} (median {thr_q:.2f}); fingers: {ev.finger_kind} (+{ev.n_appeared} / -{ev.n_disappeared} / slid {ev.n_slid})", fontsize=9)
                fig.tight_layout(); name = f"figG_examples_{ds}_{kind}_{ev.event_id}.png"; fig.savefig(FIG / name, dpi=110); plt.close(fig)
                sheet.append(dict(dataset=ds, kind=kind, zstar=zstar, event_id=ev.event_id, cls=ev.cls, d_C=ev.d_C, d_Z=getattr(ev, "d_" + zstar), d_q=ev.d_q, retention=ev.retention, finger_kind=ev.finger_kind, figure=name,
                                  n_candidates=len(sub)))
    pd.DataFrame(sheet).to_csv(FIG / "figG_examples.csv", index=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*", default=None); a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    for name, fn in (("A", fig_a), ("B", fig_b), ("C", fig_c), ("D", fig_d), ("E", fig_e), ("F", fig_f), ("G", fig_g)):
        if a.only and name not in a.only:
            continue
        try:
            fn(); log.info("figure %s done", name)
        except Exception as ex:  # noqa: BLE001
            log.exception("figure %s failed: %s", name, ex)


if __name__ == "__main__":
    main()
