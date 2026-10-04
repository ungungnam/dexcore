#!/usr/bin/env python
"""Step 5: the Markdown report.   python report.py
Assembles OUT/wrench_counterfactual_report.md from the tables (events.csv, summary_by_event_type.csv,
sensitivity_*.csv, finger_transition_summary.csv, quadrant.csv, thresholds.json, sanity_summary.json,
dataset_physics_availability.json) and the narrative sections in OUT/narrative/*.md.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import wc_common as W

NARR = W.OUT / "narrative"
CLS = ["persistent_spatial", "persistent_mixed", "onset", "release", "transient", "amount"]


def narrative(name):
    p = NARR / f"{name}.md"
    return p.read_text().strip() + "\n" if p.exists() else f"_({name}: to be written)_\n"


def md_table(header, rows):
    esc = lambda s: str(s).replace("|", "\\|")
    return "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n" + "\n".join("| " + " | ".join(esc(c) for c in r) + " |" for r in rows) + "\n"


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(df_row, m, stat="median", nd=2):
    return f"{f(df_row[f'{m}_{stat}'], nd)} [{f(df_row[f'{m}_{stat}_lo'], nd)}, {f(df_row[f'{m}_{stat}_hi'], nd)}]"


def table_summary():
    S = pd.read_csv(W.OUT / "summary_by_event_type.csv")
    rows = []
    for ds in W.DATASETS:
        for c in CLS:
            d = S[(S.dataset == ds) & (S.class_report == c)]
            if not len(d):
                continue
            r = d.iloc[0]
            rows.append([W.LABEL[ds], W.CLASS_LABEL[c], f"{int(r.n_events)} / {int(r.n_takes)}", f(r.dC_median), f(r.Q_pre_median), f(r.Q_post_median),
                         ci(r, "rel_change_Q"), ci(r, "R_pre_to_post"), ci(r, "cosine"), ci(r, "l1_rel_distance"),
                         f"{100 * r['frac_retention_ge_0.8']:.0f} %", f"{100 * r['frac_abs_rel_change_le_0.25']:.0f} %", f"{100 * r['frac_rel_change_gt_0.25']:.0f} %"])
    return md_table(["dataset", "class", "events / takes", "‖ΔC‖ med.", "Q_pre med.", "Q_post med.", "rel. change (Q_post−Q_pre)/Q_pre, med. [CI]", "retention R pre→post, med. [CI]",
                     "profile cosine, med. [CI]", "rel. L1 distance, med. [CI]", "R ≥ 0.8", "|rel. change| ≤ 0.25", "rel. change > +0.25"], rows)


def table_strict():
    S = pd.read_csv(W.OUT / "summary_by_event_type.csv")
    rows = []
    for ds in W.DATASETS:
        for c in CLS:
            d = S[(S.dataset == ds) & (S.class_report == c)]
            if not len(d):
                continue
            r = d.iloc[0]
            rows.append([W.LABEL[ds], W.CLASS_LABEL[c], f(r.strict_coverage_pre_mean), f(r.strict_coverage_post_mean), f(r.strict_Q_pre_median, 3), f(r.strict_Q_post_median, 3),
                         ci(r, "strict_rel_change_Q") if "strict_rel_change_Q_median" in r else "–", ci(r, "strict_R_pre_to_post") if "strict_R_pre_to_post_median" in r else "–",
                         f(r.n_patches_pre_mean, 1), f(r.n_patches_post_mean, 1), f(r.n_fingers_pre_mean, 1), f(r.n_fingers_post_mean, 1)])
    return md_table(["dataset", "class", "strict coverage pre (mean)", "post", "strict Q_pre med.", "strict Q_post med.", "strict rel. change med. [CI]", "strict retention med. [CI]",
                     "patches pre (mean)", "post", "fingers pre (mean)", "post"], rows)


def table_B():
    S = pd.read_csv(W.OUT / "summary_by_event_type.csv")
    keys = [("force_K8", "force ∥ Δv, K=8"), ("torque_K8", "torque ∥ Δω, K=8"), ("force_K16", "force ∥ Δv, K=16"), ("torque_K16", "torque ∥ Δω, K=16"), ("support", "support (anti-gravity)")]
    rows = []
    for ds in W.DATASETS:
        for c in ("persistent_spatial", "persistent_mixed", "onset", "release"):
            d = S[(S.dataset == ds) & (S.class_report == c)]
            if not len(d):
                continue
            r = d.iloc[0]
            cells = []
            for k, _ in keys:
                if f"B_{k}_ratio_median" in r and not np.isnan(r[f"B_{k}_ratio_median"]):
                    cells.append(f"{f(r[f'B_{k}_pre_median'])} → {f(r[f'B_{k}_post_median'])}; ratio {ci(r, f'B_{k}_ratio')}; R {f(r[f'B_{k}_retention_median'])}")
                else:
                    cells.append("–")
            rows.append([W.LABEL[ds], W.CLASS_LABEL[c]] + cells)
    return md_table(["dataset", "class"] + [f"{lab}: h_pre → h_post (med.); post/pre med. [CI]; retention" for _, lab in keys], rows)


def table_quadrant():
    Q = pd.read_csv(W.OUT / "quadrant.csv"); thr = json.load(open(W.OUT / "thresholds.json"))
    rows = []
    for ds in W.DATASETS:
        for (dq, rq), g in Q[Q.dataset == ds].groupby(["dC_threshold", "dQ_threshold"]):
            g = g.set_index("quadrant")
            rows.append([W.LABEL[ds], f"{dq} = {g.dC_value.iloc[0]:.2f}", f"{rq} = {g.dQ_value.iloc[0]:.2f}",
                         f"{int(g.loc['large_dC_small_dQ', 'n'])} ({100 * g.loc['large_dC_small_dQ', 'frac']:.0f} % [{100 * g.loc['large_dC_small_dQ', 'lo']:.0f}, {100 * g.loc['large_dC_small_dQ', 'hi']:.0f}])",
                         f"{int(g.loc['large_dC_large_dQ', 'n'])} ({100 * g.loc['large_dC_large_dQ', 'frac']:.0f} %)", f"{int(g.loc['small_dC_small_dQ', 'n'])} ({100 * g.loc['small_dC_small_dQ', 'frac']:.0f} %)",
                         f"{int(g.loc['small_dC_large_dQ', 'n'])} ({100 * g.loc['small_dC_large_dQ', 'frac']:.0f} %)",
                         f"{100 * g.loc['small_dQ_given_large_dC', 'frac']:.0f} % [{100 * g.loc['small_dQ_given_large_dC', 'lo']:.0f}, {100 * g.loc['small_dQ_given_large_dC', 'hi']:.0f}]" if "small_dQ_given_large_dC" in g.index else "–"])
    return md_table(["dataset", "‖ΔC‖ threshold (train quantile)", "|rel. change| threshold (train quantile)", "large ΔC & small ΔQ: n (% [CI])", "large ΔC & large ΔQ", "small ΔC & small ΔQ", "small ΔC & large ΔQ", "P(small ΔQ | large ΔC) [CI]"], rows), thr


def table_sens(name, label):
    S = pd.read_csv(W.OUT / f"sensitivity_{name}.csv")
    rows = []
    for ds in W.DATASETS:
        for c in ("persistent_spatial", "persistent_mixed", "release"):
            d = S[(S.dataset == ds) & (S.class_report == c)]
            for _, r in d.iterrows():
                rows.append([W.LABEL[ds], W.CLASS_LABEL[c], r.setting, f"μ={r.mu}, thr={100 * r.thr:.1f} cm, merge={100 * r.merge:.0f} cm, {r.budget}", int(r.n_events),
                             ci(r, "rel_change_Q"), ci(r, "R_pre_to_post"), ci(r, "cosine"), f"{100 * r['frac_retention_ge_0.8']:.0f} %"])
    return md_table(["dataset", "class", "setting", "parameters", "n", "rel. change med. [CI]", "retention med. [CI]", "cosine med. [CI]", "R ≥ 0.8"], rows)


def table_fingers():
    Fg = pd.read_csv(W.OUT / "finger_transition_summary.csv")
    rows = []
    for ds in W.DATASETS:
        for c in ("persistent_spatial", "persistent_mixed", "onset", "release"):
            d = Fg[(Fg.dataset == ds) & (Fg.class_report == c)]
            if not len(d):
                continue
            r = d.iloc[0]
            rows.append([W.LABEL[ds], W.CLASS_LABEL[c], int(r.n_events), f"{r.n_fingers_pre_mean:.1f} → {r.n_fingers_post_mean:.1f}", f"{100 * r.frac_same_finger_set:.0f} %",
                         f"{r.n_slid_mean:.2f} / {r.n_appeared_mean:.2f} / {r.n_disappeared_mean:.2f}"] +
                        [f"{100 * r[f'{p}_stable']:.0f} / {100 * r[f'{p}_slid']:.0f} / {100 * r[f'{p}_appeared']:.0f} / {100 * r[f'{p}_disappeared']:.0f}" for p in W.PARTS] +
                        [f"{int(r.n_events_changed0)}: {f(r.median_retention_changed0)} | {int(r.n_events_changed1)}: {f(r.median_retention_changed1)} | {int(r.n_events_changed2plus)}: {f(r.median_retention_changed2plus)}"])
    return md_table(["dataset", "class", "n", "fingers pre → post (mean)", "same finger set", "fingers slid / appeared / disappeared per event (mean)"] +
                    [f"{p}: stable / slid / appeared / disappeared (%)" for p in W.PARTS] + ["n events with 0 | 1 | ≥2 reconfigured fingers: median retention"], rows)


def table_sanity():
    s = json.load(open(W.OUT / "sanity_summary.json"))
    rows = []
    for ds in W.DATASETS:
        d = s[ds]
        rows.append([W.LABEL[ds], d["n_events"], d["n_errors"], f"{d['min_d_agreement_max']:.1e}", f"{100 * d['hand_outward_mean']:.0f} %", f"{100 * d['normal_flipped_toward_hand_frac_mean']:.0f} %",
                     f"{100 * d['finger_label_agreement_778_vs_fps100']:.1f} %", f"{d['contact_vertices_median_pre_post'][0]:.0f} / {d['contact_vertices_median_pre_post'][1]:.0f}",
                     f"{d['patches_per_frame_mean']:.1f}", f"{d['vertices_per_patch_median']:.0f}", d["persistent_spatial_figures"]])
    syn = s["synthetic_grasp_map"]
    return md_table(["dataset", "events", "errors", "max |min-dist recomputed − cached|", "hand on the mesh's outward side", "contact normals flipped toward the hand (sampled events)",
                     "finger label: 778-vertex vs 100-point FPS agreement", "contact vertices pre / post (median)", "patches per frame (mean)", "vertices per patch (median)", "pre/post figures"], rows), syn


def main():
    phys = json.load(open(W.OUT / "dataset_physics_availability.json")) if (W.OUT / "dataset_physics_availability.json").exists() else {}
    tq, thr = table_quadrant(); ts, syn = table_sanity()
    parts = ["# Are post-grasp spatial contact reconfigurations associated with meaningful changes in manipulation wrench capability? A counterfactual grasp-wrench analysis on TACO and ARCTIC\n",
             f"Scripts: `scripts/research/wrench_counterfactual/`; outputs: `{W.OUT}`. Analysis only: GT hands, GT objects, the existing event labels, the existing fixed test split; no model was trained, no physics was rolled out.\n",
             "## Decision table\n", narrative("decision_table"),
             "## 1. Research question\n", narrative("question"),
             "## 2. Motivation from the previous findings\n", narrative("motivation"),
             "## 3. Wrench / contact-patch formulation\n", narrative("formulation"),
             "## 4. Dataset support and exclusions\n", narrative("data"),
             "\n**Physical quantities available per dataset** (`dataset_physics_availability.json`).\n\n", md_table(["quantity", "TACO", "ARCTIC", "used as"],
             [[k, str(phys.get("taco", {}).get(k, "–")), str(phys.get("arctic", {}).get(k, "–")), str(phys.get("used_as", {}).get(k, "–"))] for k in phys.get("used_as", {})]),
             "## 5. Sanity checks\n", narrative("sanity"), "\n", ts,
             "\n**Synthetic grasp-map check** (" + ("passed" if syn["passed"] else "FAILED") + "): " + "; ".join(f"{c['name']}: {c['got']:.3f} (expected {c['expected']:.3f})" for c in syn["cases"]) + ".\n",
             "## 6. Persistent-spatial main results\n", narrative("main"),
             "\n**Table 1 — pre / post wrench capability by event class** (primary setting: contact < 1 cm, merge radius 1 cm, unit finger budgets, μ = 0.5; support-function capacity over 76 shared directions; medians with take-cluster bootstrap 95 % CI).\n\n", table_summary(),
             "\n**Table 1b — strict exact-direction capacity and grasp structure.**\n\n", table_strict(),
             "\n**Table 2 — large contact change vs small wrench change, persistent spatial events** (thresholds = quantiles of the TRAIN persistent-spatial events: " +
             "; ".join(f"{W.LABEL[d]}: n = {t['n_train']}, ‖ΔC‖ median {t['dC_q50']:.2f} / Q75 {t['dC_q75']:.2f}, |rel. change| median {t['abs_rel_change_Q_q50']:.2f} / Q25 {t['abs_rel_change_Q_q25']:.2f}" for d, t in thr.items()) + ").\n\n", tq,
             "\n![Figure A](figures/figA_contact_change_vs_wrench_change.png)\n\n![Figure B](figures/figB_retention_distribution.png)\n",
             "## 7. Mixed / onset / release controls\n", narrative("controls"), "\n![Figure C](figures/figC_by_event_class.png)\n",
             "## 8. Manipulation-conditioned directions\n", narrative("metric_b"), "\n**Table 3 — directional capacity along the future object motion** (h = support-function capacity; pre grasp transported to the post frame; ratio = post / pre; retention = min(pre/post, 1)).\n\n", table_B(),
             "\n![Figure D](figures/figD_manipulation_directions.png)\n",
             "## 9. Friction / patch / force-budget sensitivity\n", narrative("sensitivity"),
             "\n**Table 4a — friction coefficient.**\n\n", table_sens("mu", "mu"), "\n**Table 4b — contact threshold and patch merge radius.**\n\n", table_sens("patch", "patch"),
             "\n**Table 4c — force-budget scheme.**\n\n", table_sens("budget", "budget"), "\n![Figure E](figures/figE_sensitivity.png)\n",
             "## 10. Finger-level analysis and qualitative examples\n", narrative("fingers"), "\n**Table 5 — finger-level transitions** (primary setting).\n\n", table_fingers(),
             "\n", narrative("qualitative"),
             "## 11. Interpretation\n", narrative("interpretation"),
             "## 12. Limitations\n", narrative("limitations"),
             "## 13. Final answer\n", narrative("answer"),
             "## Files\n", narrative("files")]
    (W.OUT / "wrench_counterfactual_report.md").write_text("\n".join(parts))
    print("report written:", W.OUT / "wrench_counterfactual_report.md")


if __name__ == "__main__":
    main()
