#!/usr/bin/env python
"""Assemble structure_variance_report.md from the aggregate tables and the narrative sections (narrative/*.md).
    python report.py
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

import sv_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("report")
NARR = S.OUT / "narrative"
ORDER = ["R0", "R1", "R2", "R3", "R4", "R5", "CONTACT_FULL", "FULL"]
RES_MODELS = ["zero", "last", "condmean", "history", "history_delta", "oracle", "oracle_delta"]
RES_LABEL = {"zero": "zero (r^ = 0)", "last": "last (r^ = r_t)", "condmean": "mean given Z\\*\\_t (+ G, τ)", "history": "FULL history + G + τ",
             "history_delta": "FULL history + residual history, change on r_t", "oracle": "FULL history + GT Z\\*\\_{t+h} (non-causal)",
             "oracle_delta": "… + residual history, change on r_t (non-causal)"}


def narrative(name):
    p = NARR / f"{name}.md"
    return p.read_text() if p.exists() else f"*({name}: to be written)*\n"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def ci(v, lo, hi, nd=2):
    return f"{f(v, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def table_1():
    d = json.load(open(S.OUT / "representation_definitions.json"))
    return md_table(["representation", "blocks", "dim / frame", "window dim (8 frames)", "components"], [[r["representation"], r["blocks"], r["dim"], r["window_dim"], r["components"]] for r in d["table"]])


def table_2():
    fp = pd.read_csv(S.OUT / "functional_probe.csv"); rows = []
    for ds in S.DATASETS:
        d = fp[(fp.dataset == ds) & (fp.probe == "mlpG_seedavg") & (fp.target == "q")].set_index("rep")
        rt = fp[(fp.dataset == ds) & (fp.probe == "ratio_to_FULL") & (fp.target == "q") & (fp.metric == "rel_l1")].set_index("rep")
        sd = fp[(fp.dataset == ds) & (fp.probe == "mlpG") & (fp.target == "q")].set_index("rep")
        so = fp[(fp.dataset == ds) & (fp.probe == "mlp") & (fp.target == "q")].set_index("rep")
        rg = fp[(fp.dataset == ds) & (fp.probe == "ridgeG") & (fp.target == "q")].set_index("rep")
        st = fp[(fp.dataset == ds) & (fp.probe == "mlpG") & (fp.target == "q_strict")].set_index("rep")
        for rep in ["MEAN"] + ORDER:
            if rep not in d.index:
                continue
            r = d.loc[rep]
            rows.append([S.LABEL[ds], rep, S.REP_DIM.get(rep, 0), ci(r.rel_l1, r.rel_l1_lo, r.rel_l1_hi, 3), f(sd.loc[rep, "rel_l1_seed_sd"], 3) if rep in sd.index else "–", ci(r.cosine, r.cosine_lo, r.cosine_hi, 3),
                         f(r.nrmse, 3), f(sd.loc[rep, "r2"], 3) if rep in sd.index else "–", ci(r.Q_err, r.Q_err_lo, r.Q_err_hi, 3), f(r.overlap, 3),
                         ci(rt.loc[rep, "ratio"], rt.loc[rep, "ratio_lo"], rt.loc[rep, "ratio_hi"], 3) if rep in rt.index else "–",
                         f(so.loc[rep, "rel_l1"], 3) if rep in so.index else "–", f(rg.loc[rep, "rel_l1"], 3) if rep in rg.index else "–", f(st.loc[rep, "rel_l1"], 3) if rep in st.index else "–"])
    return md_table(["dataset", "representation", "dim", "rel. L1 [CI]", "seed sd", "cosine [CI]", "nRMSE", "R²", "Q error [CI]", "overlap", "rel. L1 ratio to FULL [CI]", "rel. L1 without G", "ridge + G rel. L1", "strict-profile rel. L1"], rows)


def table_3():
    tp = pd.read_csv(S.OUT / "temporal_prediction.csv"); rows = []
    mets = [("T1", "auprc_macro", 3), ("T1", "f1_macro", 3), ("T2", "T2_rel_l1", 3), ("T3", "T3_cent", 3), ("T3", "T3_ang", 1), ("T4", "T4_rel_l1", 3), ("T4", "T4_cos", 3), ("T4", "T4_Qerr", 3), ("T5C", "E_C", 3), ("T5H", "E_H", 4)]
    for ds in S.DATASETS:
        for h in S.HORIZONS:
            for rep in ["PERSIST", "MEAN"] + ORDER:
                d = tp[(tp.dataset == ds) & (tp.h == h) & (tp.rep == rep)]
                if not len(d):
                    continue
                cells = []
                for tg, m, nd in mets:
                    x = d[d.metric == m]
                    cells.append(ci(x.value.iloc[0], x.lo.iloc[0], x.hi.iloc[0], nd) if len(x) and np.isfinite(x.lo.iloc[0]) else (f(x.value.iloc[0], nd) if len(x) else "–"))
                rows.append([S.LABEL[ds], h, rep] + cells)
    return md_table(["dataset", "h", "representation", "T1 AUPRC macro", "T1 F1 macro", "T2 norm. L1", "T3 centroid / l", "T3 normal [deg]", "T4 rel. L1", "T4 cosine", "T4 Q error", "T5 ‖ΔC‖ error", "T5 hand RMS [m]"], rows)


def table_3b():
    tr = pd.read_csv(S.OUT / "temporal_ratios.csv"); rows = []
    for ds in S.DATASETS:
        for rep in ORDER:
            d = tr[(tr.dataset == ds) & (tr.rep == rep) & (tr.vs == "FULL")]
            if not len(d):
                continue
            cells = []
            for m in ("composite_T1_T4", "T1_ratio", "T2_ratio", "T3_ratio", "T4_ratio"):
                for h in ((0,) if m == "composite_T1_T4" else ()) + tuple(S.HORIZONS):
                    x = d[(d.metric == m) & (d.h == h)]
                    cells.append(ci(x.ratio.iloc[0], x.lo.iloc[0], x.hi.iloc[0], 3) if len(x) else "–")
            rows.append([S.LABEL[ds], rep] + cells)
    hdr = ["dataset", "representation", "composite (mean over h)"] + [f"composite h={h}" for h in S.HORIZONS] + [f"{t} h={h}" for t in ("T1", "T2", "T3", "T4") for h in S.HORIZONS]
    return md_table(hdr, rows)


def table_4():
    inc = pd.read_csv(S.OUT / "incremental_gain.csv"); rows = []
    for r in inc.itertuples():
        rows.append([S.LABEL[r.dataset], f"{r.frm} → {r.to}", r.added, f"{r.dim_from} → {r.dim_to}", ci(r.functional_ratio, r.functional_lo, r.functional_hi, 3) if hasattr(r, "functional_ratio") else "–",
                     ci(r.temporal_ratio, r.temporal_lo, r.temporal_hi, 3) if hasattr(r, "temporal_ratio") else "–"] + [f(getattr(r, f"temporal_ratio_h{h}", np.nan), 3) for h in S.HORIZONS]
                    + [ci(getattr(r, "dense_E_C_ratio", np.nan), getattr(r, "dense_E_C_lo", np.nan), getattr(r, "dense_E_C_hi", np.nan), 3), ci(getattr(r, "dense_E_H_ratio", np.nan), getattr(r, "dense_E_H_lo", np.nan), getattr(r, "dense_E_H_hi", np.nan), 3)])
    return md_table(["dataset", "step", "added block", "dim", "functional error ratio to/from [CI]", "temporal composite ratio [CI]"] + [f"composite h={h}" for h in S.HORIZONS] + ["dense T5 ‖ΔC‖ ratio [CI]", "dense T5 hand ratio [CI]"], rows)


def table_5():
    co = pd.read_csv(S.OUT / "event_correlations.csv"); pr = pd.read_csv(S.OUT / "event_probes.csv"); qd = pd.read_csv(S.OUT / "event_quadrants.csv")
    parts = []
    rows = []
    for ds in S.DATASETS:
        for subset in ("persistent_spatial", "all_persistent"):
            d = co[(co.dataset == ds) & (co.subset == subset)]
            for col in ["d_part", "d_amount", "d_geom", "d_topo", "d_hand", "d_H"] + [f"d_{lv}" for lv in ORDER]:
                a = d[(d.x == "d_C") & (d.y == col)]; b = d[(d.x == col) & (d.y == "d_q")]; c = d[(d.x == col) & (d.y == "one_minus_R")]
                rows.append([S.LABEL[ds], subset.replace("_", " "), col, ci(a.rho.iloc[0], a.lo.iloc[0], a.hi.iloc[0]) if len(a) else "–", ci(b.rho.iloc[0], b.lo.iloc[0], b.hi.iloc[0]) if len(b) else "–", ci(c.rho.iloc[0], c.lo.iloc[0], c.hi.iloc[0]) if len(c) else "–"])
            a = d[(d.x == "d_C") & (d.y == "d_q")]
            if len(a):
                rows.append([S.LABEL[ds], subset.replace("_", " "), "d_C itself", "–", ci(a.rho.iloc[0], a.lo.iloc[0], a.hi.iloc[0]), "–"])
    parts.append("**Table 5a — Spearman correlations on the test events** (take-cluster bootstrap 95 % CI): of the dense change with each block / level change, and of each change with the wrench-profile change d_q and with 1 − retention.\n\n"
                 + md_table(["dataset", "events", "change", "ρ(d_C, change)", "ρ(change, d_q)", "ρ(change, 1 − R)"], rows))
    rows = []
    for ds in S.DATASETS:
        for subset in ("persistent_spatial", "all_persistent"):
            d = pr[(pr.dataset == ds) & (pr.subset == subset) & (pr.wrench == "d_q")]
            for r in d.itertuples():
                rows.append([S.LABEL[ds], subset.replace("_", " "), r.level, r.features, r.n_train, r.n_test, ci(r.ridge_r2, r.ridge_r2_lo, r.ridge_r2_hi, 3), f(r.logit_auroc, 3), f(r.logit_auprc, 3), f(r.positive_rate, 2)])
    parts.append("\n**Table 5b — probes fitted on the TRAIN events, evaluated on the TEST events**: ridge R² for the wrench-profile change d_q from the cumulative block changes (log1p, train-standardised), and a logistic probe for 'd_q above the train median' (AUROC / AUPRC).\n\n"
                 + md_table(["dataset", "events", "level", "features", "n train", "n test", "ridge R² [CI]", "AUROC", "AUPRC", "positive rate"], rows))
    rows = []
    for ds in S.DATASETS:
        for subset in ("persistent_spatial", "all_persistent"):
            d = qd[(qd.dataset == ds) & (qd.subset == subset) & (qd.wrench == "d_q")]
            for lv in ORDER + ["H"]:
                r = d[d.change == f"d_{lv}"]
                if not len(r):
                    continue
                r = r.iloc[0]
                rows.append([S.LABEL[ds], subset.replace("_", " "), f"d_{lv}", r.n_large_C, ci(r.frac_variation_candidates, r.lo1, r.hi1), r.n_large_w, ci(r.frac_large_w_missed, r.lo2, r.hi2)])
    parts.append("\n**Table 5c — train-median quadrants (secondary view)**: among test events with a large dense change (d_C above the train median), the fraction with a small change of the level AND a small wrench change (candidate within-structure variation); among events with a large wrench change (d_q above the train median), the fraction the level misses (small d_level).\n\n"
                 + md_table(["dataset", "events", "level", "n large d_C", "P(small d_level & small d_q | large d_C) [CI]", "n large d_q", "P(small d_level | large d_q) [CI]"], rows))
    return "\n".join(parts)


def table_6():
    if (S.OUT / "residual_prediction.csv").stat().st_size < 10:
        return "*(Experiment D results not available yet)*\n"
    rp = pd.read_csv(S.OUT / "residual_prediction.csv"); rows = []
    for ds in S.DATASETS:
        d = rp[rp.dataset == ds]
        for zstar in sorted(d.zstar.unique()):
            dz = d[d.zstar == zstar]
            for blk in ("C", "H"):
                dec = dz[(dz.model == "decoder") & (dz.block == blk)]
                for model in RES_MODELS:
                    cells = []
                    for metric in ("mse_ratio_to_zero", "mse_ratio_to_last"):
                        for h in S.HORIZONS:
                            x = dz[(dz.model == model) & (dz.block == blk) & (dz.h == h) & (dz.metric == metric)]
                            cells.append(ci(x.value.iloc[0], x.lo.iloc[0], x.hi.iloc[0], 3) if len(x) else "–")
                    if all(c == "–" for c in cells):
                        continue
                    rows.append([S.LABEL[ds], zstar, blk, f(dec.value.iloc[0], 3) if len(dec) else "–", RES_LABEL[model]] + cells)
    return md_table(["dataset", "Z*", "block", "decoder R² (test)", "predictor of r_{t+h}"] + [f"MSE / residual energy, h={h} [CI]" for h in S.HORIZONS] + [f"MSE / last-residual MSE, h={h} [CI]" for h in S.HORIZONS], rows)


def table_7():
    ch = pd.read_csv(S.OUT / "saturation_choice.csv"); sat = pd.read_csv(S.OUT / "saturation_summary.csv"); rows = []
    for ds in S.DATASETS:
        for thr in S.SATURATION_THRESHOLDS:
            c = ch[(ch.dataset == ds) & (ch.threshold == thr) & (ch.reference == "FULL")].set_index("criterion")
            cb = ch[(ch.dataset == ds) & (ch.threshold == thr) & (ch.reference == "best")].set_index("criterion")
            rows.append([S.LABEL[ds], f"{thr * 100:g} %", c.loc["functional", "smallest_saturated"], c.loc["temporal", "smallest_saturated"], c.loc["both", "smallest_saturated"],
                         cb.loc["functional", "smallest_saturated"], cb.loc["temporal", "smallest_saturated"], cb.loc["both", "smallest_saturated"]])
    t = md_table(["dataset", "threshold", "smallest wrench-sufficient (vs FULL)", "smallest temporally sufficient (vs FULL)", "smallest saturated on both (vs FULL)",
                  "wrench-sufficient (vs best)", "temporally sufficient (vs best)", "both (vs best)"], rows)
    rows = []
    for ds in S.DATASETS:
        d = sat[sat.dataset == ds].set_index("rep")
        for rep in ORDER:
            if rep in d.index:
                r = d.loc[rep]
                rows.append([S.LABEL[ds], rep, S.REP_DIM[rep], ci(r.functional_gap, r.functional_gap_lo, r.functional_gap_hi, 3), ci(r.temporal_gap, r.temporal_gap_lo, r.temporal_gap_hi, 3),
                             " / ".join("yes" if r[f"saturated_{thr}"] else "no" for thr in S.SATURATION_THRESHOLDS),
                             ci(r.functional_gap_best, r.functional_gap_best_lo, r.functional_gap_best_hi, 3), ci(r.temporal_gap_best, r.temporal_gap_best_lo, r.temporal_gap_best_hi, 3),
                             " / ".join("yes" if r[f"saturated_{thr}_best"] else "no" for thr in S.SATURATION_THRESHOLDS)])
    return t + "\n" + md_table(["dataset", "representation", "dim", "functional gap to FULL [CI]", "temporal composite gap to FULL [CI]", "saturated vs FULL at 2.5 / 5 / 10 %",
                                  "functional gap to the best [CI]", "temporal gap to the best [CI]", "saturated vs best at 2.5 / 5 / 10 %"], rows)


def sanity_table():
    s = json.load(open(S.OUT / "sanity" / "sanity_summary.json")); rows = []
    for ds in S.DATASETS:
        x = s[ds]; b = x["build"]
        rows.append([S.LABEL[ds], b["n_sequences"], b["n_errors"], f"{b['min_d_check_max']:.1e}", f"{b['C_check_max']:.1e}", f"{100 * b['clamped_frac']:.1f} %", f"{x['centroid_to_nearest_mesh_vertex_mm']['median']:.2f} / {x['centroid_to_nearest_mesh_vertex_mm']['p95']:.2f}",
                     ", ".join(f"{k}: {v['mean']:+.2f} (n={v['n']})" for k, v in x["palm_normal_dot_dorsal"].items()), f"{x['inactive_parts_geom_max_abs']:.0e} / {x['inactive_parts_topo_max_abs']:.0e}",
                     f"{x['events_test']['n_matched']} / {x['events_test']['n_wrench']}, pre/post {'equal' if x['events_test']['pre_post_equal'] else 'DIFFER'}, dC diff {x['events_test']['dC_max_abs_diff']:.1e}",
                     f"{x['mean_active_parts']:.2f}", f"{100 * x['q_zero_frac']:.1f} % / {100 * x['q_strict_zero_frac']:.1f} %"])
    return md_table(["dataset", "sequences", "errors", "max |min-dist recomputed − cached|", "max |C padded − sequences|", "clamped frames", "centroid → mesh vertex [mm] median / p95", "palm normal · dorsal axis (by hand)",
                     "inactive parts: max |geom| / |topo|", "test events vs wrench report", "active parts / frame", "frames with q = 0 (support / strict)"], rows)


def main():
    parts = ["# Where is the boundary between structured hand–contact state and within-structure variation? A representation-sufficiency diagnostic on TACO and ARCTIC\n",
             f"Scripts: `scripts/research/structure_variance_boundary/`; outputs: `{S.OUT}`. Diagnostic only: the same sequences, split, contact maps, hand states, events and wrench model as the previous studies; "
             "small probes and the previous temporal recipe with the input representation as the only change; no generative model, no rollout.\n",
             "## Decision table\n", narrative("decision_table"),
             "## 1. Research question\n", narrative("question"),
             "## 2. Existing evidence / motivation\n", narrative("motivation"),
             "## 3. Representation definitions\n", narrative("definitions"), "\n**Table 1 — representation ladder.**\n\n", table_1(),
             "## 4. Functional sufficiency (Experiment A)\n", narrative("functional"), "\n**Table 2 — reconstruction of the 76-D support-function wrench profile q_t from each representation** (MLP probe on the representation and the object descriptor G, regularisation chosen on validation, 3 seeds; test frames in contact; take-cluster bootstrap 95 % CI; the ratio column compares the relative-L1 error sum to FULL's; sensitivities: the same MLP without G, a ridge probe with G, and the strict exact-direction profile as target).\n\n", table_2(),
             "\n![Figure A](figures/figA_functional_ladder.png)\n",
             "## 5. Temporal predictive sufficiency (Experiment B)\n", narrative("temporal"),
             "\n**Table 3 — future targets at h = 1 / 4 / 8 from the causal 8-frame window of each representation + G + τ_local** (means over the valid test pairs, seed-averaged, take-cluster bootstrap 95 % CI; PERSIST = last value, MEAN = train mean).\n\n", table_3(),
             "\n**Table 3b — error ratios to FULL** (paired take bootstrap; T1 enters through the per-frame participation-probability error, T3 as the mean of the centroid and normal ratios; composite = mean of the T1–T4 ratios).\n\n", table_3b(),
             "\n![Figure B](figures/figB_temporal_ladder.png)\n\n![Figure C](figures/figC_pareto_saturation.png)\n",
             "\n**Table 4 — incremental information gain** (error ratio after / before adding a block; < 1 = the block helps; take-cluster bootstrap 95 % CI).\n\n", table_4(),
             "\n![Figure E](figures/figE_feature_contribution.png)\n",
             "## 6. Event-level analysis (Experiment C)\n", narrative("events"), "\n", table_5(), "\n![Figure D](figures/figD_dense_vs_structure.png)\n",
             "## 7. Residual predictability (Experiment D)\n", narrative("residual"), "\n**Table 6 — predictability of the residual discarded by Z\\*** (MSE of the predicted future residual over the residual's own energy, per block; < 1 = better than the zero residual; 1 − value = R² against zero).\n\n", table_6(),
             "\n![Figure F](figures/figF_residual_predictability.png)\n",
             "## 8. Dataset differences\n", narrative("datasets"),
             "\n**Table 7 — boundary decision.**\n\n", table_7(),
             "## 9. Sanity checks\n", narrative("sanity"), "\n", sanity_table(),
             "## 10. Limitations\n", narrative("limitations"),
             "## 11. Final answer\n", narrative("answer"),
             "\n![Figure G examples](figures/figG_examples_taco_variation_placeholder.png)\n" if False else "",
             "## Files\n", narrative("files")]
    text = "\n".join(parts)
    (S.OUT / "structure_variance_report.md").write_text(text)
    log.info("report written: %s (%d lines)", S.OUT / "structure_variance_report.md", text.count("\n"))


if __name__ == "__main__":
    main()
