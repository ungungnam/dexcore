#!/usr/bin/env python
"""Step 6: the Markdown report.  python report.py
Assembles OUT/hand_contact_predictive_info_report.md from the result tables of both datasets and
the narrative sections in OUT/narrative/*.md (written after the numbers were read).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import hp_common as P

NARR = P.OUT / "narrative"
CLS_T2 = ["persistent_spatial", "persistent_mixed", "onset", "release", "transient", "non_spike"]


def load(name):
    return pd.concat([pd.read_csv(P.ds_out(ds) / "results" / f"{name}.csv") for ds in P.DATASETS if (P.ds_out(ds) / "results" / f"{name}.csv").exists()], ignore_index=True)


def narrative(name):
    p = NARR / f"{name}.md"
    return p.read_text().strip() + "\n" if p.exists() else f"_({name}: to be written)_\n"


def md_table(header, rows):
    esc = lambda s: str(s).replace("|", "\\|")
    return "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n" + "\n".join("| " + " | ".join(esc(c) for c in r) + " |" for r in rows) + "\n"


def f(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def ci(pt, lo, hi, nd=3):
    return f"{f(pt, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def table1():
    R = load("overall_delta_results")
    rows = []
    for ds in P.DATASETS:
        for c in ("F0", "F1", "F2k8", "F2rel", "F2shuf", "Ffut1", "Ffut"):
            d = R[(R.dataset == ds) & (R.cond == c)].set_index("h")
            if not len(d):
                continue
            rows.append([P.LABEL[ds], P.CONDITIONS[c]["label"] + ("" if P.CONDITIONS[c]["fut"] == 0 else " ⚠ non-causal")] +
                        [ci(d.loc[h, "E"], d.loc[h, "E_lo"], d.loc[h, "E_hi"]) for h in P.HORIZONS] +
                        [f"{100 * d.loc[h, 'improvement']:.1f} % [{100 * d.loc[h, 'improvement_lo']:.1f}, {100 * d.loc[h, 'improvement_hi']:.1f}]" for h in P.HORIZONS] +
                        [f"{100 * d.loc[h, 'improvement_sq']:.1f} %" for h in (4, 8)] +
                        [ci(d.loc[h, "cos"], d.loc[h, "cos_lo"], d.loc[h, "cos_hi"], 2) for h in (4, 8)] + [f"{d.loc[8, 'E_seed_std']:.3f}"])
        d0 = R[(R.dataset == ds) & (R.cond == "F0")].set_index("h")
        rows.append([P.LABEL[ds], "zero change (reference)"] + [f(d0.loc[h, "E0"]) for h in P.HORIZONS] + ["0 %"] * 3 + ["0 %"] * 2 + ["–", "–", "–"])
    return md_table(["dataset", "condition", "E_Δ h=1", "E_Δ h=4", "E_Δ h=8", "impr. h=1", "impr. h=4", "impr. h=8", "impr. (squared) h=4", "impr. (squared) h=8", "cos h=4", "cos h=8", "seed sd (h=8)"], rows)


def table2(h):
    R = load("event_conditioned_results")
    rows = []
    for ds in P.DATASETS:
        for c in ("F0", "F1", "F2k8", "F2rel", "F2shuf", "Ffut1", "Ffut"):
            d = R[(R.dataset == ds) & (R.cond == c) & (R.h == h)].set_index("frame_class")
            if not len(d):
                continue
            rows.append([P.LABEL[ds], P.CONDITIONS[c]["label"] + ("" if P.CONDITIONS[c]["fut"] == 0 else " ⚠")] + [f(d.loc[k, "E"]) if k in d.index else "–" for k in CLS_T2])
        d0 = R[(R.dataset == ds) & (R.cond == "F0") & (R.h == h)].set_index("frame_class")
        rows.append([P.LABEL[ds], "zero change"] + [f(d0.loc[k, "E0"]) if k in d0.index else "–" for k in CLS_T2])
        rows.append([P.LABEL[ds], "n frames"] + [str(int(d0.loc[k, "n_frames"])) if k in d0.index else "–" for k in CLS_T2])
    return md_table(["dataset", f"condition (h = {h})"] + [P.CLASS_LABEL[k] for k in CLS_T2], rows)


def table3():
    R = load("transition_prediction_results")
    if not len(R):
        return "_no event-prediction results_\n"
    rows = []
    for ds in P.DATASETS:
        for c in ("F0", "F1", "F2k8", "F2rel", "F2shuf", "Ffut1", "Ffut"):
            d = R[(R.dataset == ds) & (R.cond == c)].set_index("h")
            if not len(d):
                continue
            rows.append([P.LABEL[ds], P.CONDITIONS[c]["label"] + ("" if P.CONDITIONS[c]["fut"] == 0 else " ⚠ non-causal")] +
                        sum([[ci(d.loc[h, "auprc"], d.loc[h, "auprc_lo"], d.loc[h, "auprc_hi"]), ci(d.loc[h, "auroc"], d.loc[h, "auroc_lo"], d.loc[h, "auroc_hi"]),
                              f(d.loc[h, "auprc_spatial_mixed"]), f(d.loc[h, "auprc_release_regrasp"]), f(d.loc[h, "recall_at_p50"], 2)] for h in P.EVENT_HORIZONS], []))
        d0 = R[(R.dataset == ds) & (R.cond == "F0")].set_index("h")
        rows.append([P.LABEL[ds], "chance (prevalence)"] + sum([[f(d0.loc[h, "prevalence"]), "0.5", "–", "–", "–"] for h in P.EVENT_HORIZONS], []))
    return md_table(["dataset", "condition", "AUPRC h=4", "AUROC h=4", "AUPRC spatial/mixed h=4", "AUPRC release/regrasp h=4", "R@P50 h=4",
                     "AUPRC h=8", "AUROC h=8", "AUPRC spatial/mixed h=8", "AUPRC release/regrasp h=8", "R@P50 h=8"], rows)


def table4(h):
    I = load("incremental_information")
    rows = []
    for ds in P.DATASETS:
        for cls in ["all", "non_spike", "spike", "persistent_spatial", "persistent_mixed", "onset", "release", "transient"]:
            d = I[(I.dataset == ds) & (I.h == h) & (I.frame_class == cls)].set_index("pair")
            if not len(d):
                continue
            cell = lambda p: (ci(d.loc[p, "diff"], d.loc[p, "diff_lo"], d.loc[p, "diff_hi"]) + (" *" if d.loc[p, "significant"] else "") + f" ({d.loc[p, 'rel_to_F0_pct']:+.1f} %)") if p in d.index else "–"
            rows.append([P.LABEL[ds], P.CLASS_LABEL[cls], str(int(d.n_frames.iloc[0])), cell("F1-F0"), cell("F2k8-F1"), cell("F2k8-F0"), cell("F2shuf-F0"), cell("F2k8-F2shuf"), cell("F2rel-F2shuf"), cell("Ffut-F2rel")])
    return md_table(["dataset", f"frame class (h = {h})", "n", "F1 − F0", "F2 − F1", "F2 − F0", "F2shuf − F0 (capacity control)", "F2 − F2shuf (information)", "F2rel − F2shuf (motion information)", "Ffut − F2rel (non-causal)"], rows)


def table_auprc_inc():
    I = load("transition_incremental")
    if not len(I):
        return ""
    rows = []
    for ds in P.DATASETS:
        for h in P.EVENT_HORIZONS:
            d = I[(I.dataset == ds) & (I.h == h)].set_index("pair")
            cell = lambda p: (ci(d.loc[p, "diff"], d.loc[p, "diff_lo"], d.loc[p, "diff_hi"]) + (" *" if d.loc[p, "significant"] else "")) if p in d.index else "–"
            rows.append([P.LABEL[ds], str(h), cell("F1-F0"), cell("F2k8-F1"), cell("F2k8-F0"), cell("F2rel-F0"), cell("F2shuf-F0"), cell("F2k8-F2shuf"), cell("F2rel-F2shuf"), cell("Ffut-F2rel")])
    return md_table(["dataset", "h", "ΔAUPRC F1 − F0", "F2 − F1", "F2 − F0", "F2rel − F0", "F2shuf − F0 (control)", "F2 − F2shuf", "F2rel − F2shuf", "Ffut − F2rel (non-causal)"], rows)


def table5():
    A = load("history_length_ablation")
    rows = []
    order = ["F0", "F1", "F2k1", "F2k4", "F2k8", "F2rel", "F2shuf", "Ffut1", "Ffut"]
    for ds in P.DATASETS:
        for h in (4, 8):
            for cls in ("all", "persistent_spatial", "persistent_mixed", "release"):
                d = A[(A.dataset == ds) & (A.h == h) & (A.frame_class == cls)].set_index("cond")
                if not len(d) or "F0" not in d.index:
                    continue
                base = d.loc["F0", "E"]
                rows.append([P.LABEL[ds], str(h), P.CLASS_LABEL[cls], str(int(d.n_frames.iloc[0]))] + [f"{100 * (d.loc[c, 'E'] / base - 1):+.1f} %" if c in d.index else "–" for c in order[1:]])
    return md_table(["dataset", "h", "frame class", "n", "H_t only", "k=1", "k=4", "k=8", "motion only (k=8)", "shuffled (control)", "oracle +1 frame ⚠", "oracle +8 frames ⚠"], rows)


def table_sanity():
    rows = []
    for ds in P.DATASETS:
        s = json.load(open(P.ds_out(ds) / "sanity" / "hand_alignment.json"))
        rep = json.load(open(P.ds_out(ds) / "cache" / "hand_representation.json"))
        al = s["all"]; ev = s["event_labels"]; ag = s["agreement_with_previous_test_table"]
        rows.append([P.LABEL[ds], f"{al['n_sequences']} seq / {al['n_frames']} frames", f"{al['pearson']:.3f}", f"{al['median_abs_diff_mm']:.1f} mm", f"{100 * al['hard_frac_hand100_within_2cm']:.0f} %",
                     f"{s.get('tool_wrong_6d_layout', {}).get('pearson', np.nan):.3f}" if "tool_wrong_6d_layout" in s else "–",
                     f"{ev['n_events_test']} = {ev['n_events_test_previous']}, identical: {ev['test_events_identical_to_previous']}", f"{ag['firm_state']:.3f} / {ag['spike']:.3f}",
                     f"{s['spearman_d_vs_hand_speed_test']['rho']:.3f} [{s['spearman_d_vs_hand_speed_test']['lo']:.2f}, {s['spearman_d_vs_hand_speed_test']['hi']:.2f}]"
                     + (f" (wrong 6D layout: {s['spearman_d_vs_hand_speed_test_with_wrong_6d_layout']['rho']:.3f})" if "spearman_d_vs_hand_speed_test_with_wrong_6d_layout" in s else "")])
    return md_table(["dataset", "checked", "Pearson(hand-100 min dist, cached contact min)", "median |diff|", "hard-contact frames with hand within 2 cm", "same, wrong 6D layout",
                     "test events vs earlier analysis", "firm / spike agreement", "Spearman(d_t, hand speed) test"], rows)


def table_balance():
    rows = []
    for ds in P.DATASETS:
        s = json.load(open(P.ds_out(ds) / "sanity" / "hand_alignment.json"))["event_label_balance"]
        for h in P.EVENT_HORIZONS:
            rows.append([P.LABEL[ds], str(h)] + [f"{s[f'h{h}_{p}']['n_pos']} / {s[f'h{h}_{p}']['n_valid']} ({100 * s[f'h{h}_{p}']['prevalence']:.1f} %)" for p in ("train", "val", "test")])
    return md_table(["dataset", "h", "train positives / valid frames", "val", "test"], rows)


def table_params():
    rows = []
    for ds in P.DATASETS:
        p = P.ds_out(ds) / "results" / "training_summary.json"
        if not p.exists():
            continue
        s = json.load(open(p))
        for c in P.CONDITIONS:
            k = f"delta_{c}_seed0"; e = f"event_{c}_seed0"
            if k in s:
                steps = [s[f"delta_{c}_seed{sd}"]["best_step"] for sd in P.SEEDS if f"delta_{c}_seed{sd}" in s]
                rows.append([P.LABEL[ds], P.CONDITIONS[c]["label"], str(P.hand_dim(c)), f"{s[k]['n_params'] / 1e6:.2f} M", f"{s[k]['n_hand_params'] / 1e6:.2f} M", f"{s[k]['best_val']:.3f}",
                             "/".join(str(x) for x in steps), str(s[e]["n_params"]) if e in s else "–"])
    return md_table(["dataset", "condition", "hand input dim", "Δ-model params", "of which hand encoder", "best val loss (seed 0)", "best step (seeds 0/1/2)", "event MLP params"], rows)


def table_vf():
    V = load("f0_vs_existing_vf")
    if not len(V):
        return ""
    rows = []
    for ds in P.DATASETS:
        for cls in ("all", "non_spike", "spike", "persistent_spatial", "persistent_mixed"):
            d = V[(V.dataset == ds) & (V.frame_class == cls)].set_index("existing_vf")
            if not len(d):
                continue
            rows.append([P.LABEL[ds], P.CLASS_LABEL[cls], str(int(d.n_frames.iloc[0])), f(d.E1_zero.iloc[0]), f(d.loc["vf", "E1_existing_vf"]) if "vf" in d.index else "–",
                         f(d.loc["vf_unroll8", "E1_existing_vf"]) if "vf_unroll8" in d.index else "–", f(d.E1_F0.iloc[0])])
    return md_table(["dataset", "frame class", "n", "zero change", "existing VF (one-step recipe), teacher-forced", "existing VF (rollout recipe), teacher-forced", "F0 (this study), h = 1"], rows)


def main():
    cfg = {ds: json.load(open(P.ds_out(ds) / "results" / "evaluation_config.json")) for ds in P.DATASETS if (P.ds_out(ds) / "results" / "evaluation_config.json").exists()}
    parts = ["# Does causal hand state / history carry predictive information about future contact evolution? An oracle diagnostic on TACO and ARCTIC\n",
             f"Scripts: `scripts/research/hand_contact_predictive_info/`; outputs: `{P.OUT}` (per dataset: `cache/`, `preds/`, `results/`, `figures/`, `sanity/`). "
             "Everything is inherited from the hierarchical contact-generation study (sequences, contact vectors, geometry, object trajectories, fixed split) and the temporal contact-event analysis (event labels); nothing was retrained or re-split.\n",
             "## Summary and decision\n", narrative("decision"),
             "## 1. Research question\n", narrative("question"),
             "## 2. Why this experiment is necessary\n", narrative("why"),
             "## 3. Data and hand representation\n", narrative("data"), "\n**Sanity checks (coordinate transforms, alignment, label reproduction).**\n\n", table_sanity(),
             "\n**Class balance of the event-prediction labels** (y_t^(h) = 1 iff a persistent contact-mode transition starts in (t, t+h]).\n\n", table_balance(),
             "\n**Models and capacity** (one Δ-C predictor and one event MLP per condition and seed; only the hand input differs).\n\n", table_params(),
             "\n**F0 against the existing contact-only vector field on the one-step target** (teacher-forced, raw units).\n\n", table_vf(),
             "## 4. Anti-leakage / causal protocol\n", narrative("protocol"),
             "## 5. Δ-contact prediction results\n", narrative("delta"), "\n**Table 1 — overall Δ-contact prediction** (test frames; mean with take-level cluster-bootstrap 95 % CI; seed mean over 3 seeds; improvement = 1 − ΣE/ΣE0 over the zero-change predictor; cosine on frames whose ‖Δ_h C‖ exceeds the train median).\n\n", table1(),
             "\n![Figure A](figures/figA_delta_error.png)\n\n![Figure C](figures/figC_amount_vs_spatial.png)\n",
             "## 6. Persistent-event results\n", narrative("events"), "\n**Table 2 — event-conditioned Δ-contact error, h = 4** (frame class = the dominant event among the transitions inside Δ_4; E_Δ means).\n\n", table2(4),
             "\n**Table 2b — the same at h = 8.**\n\n", table2(8), "\n**Table 2c — the same at h = 1.**\n\n", table2(1),
             "\n![Figure B](figures/figB_event_conditioned.png)\n",
             "## 7. Transition-prediction results\n", narrative("transition"), "\n**Table 3 — event prediction** (seed-mean AUPRC / AUROC with take-bootstrap 95 % CI; subtype columns: one-vs-rest AUPRC with the other positives removed; R@P50 = recall at precision ≥ 0.5).\n\n", table3(),
             "\n**Paired take-bootstrap differences of the seed-mean AUPRC.**\n\n", table_auprc_inc(), "\n![Figure D](figures/figD_event_prediction.png)\n",
             "## 8. Hand-history-length analysis and incremental information\n", narrative("history"),
             "\n**Table 4 — incremental information, h = 4** (paired per-frame differences of E_Δ, take-cluster bootstrap 95 % CI; * = interval excludes 0; in brackets the difference relative to F0's error).\n\n", table4(4),
             "\n**Table 4b — the same at h = 8.**\n\n", table4(8), "\n**Table 4c — the same at h = 1.**\n\n", table4(1),
             "\n**Table 5 — history length, representation, control and oracle** (E_Δ relative to F0, negative = better).\n\n", table5(), "\n![Figure E](figures/figE_history_length.png)\n",
             "## 9. Non-causal future-hand oracle (upper bound, not a usable model)\n", narrative("oracle"),
             "## 10. Qualitative examples\n", narrative("qualitative"),
             "## 11. Failure cases\n", narrative("failures"),
             "## 12. Limitations\n", narrative("limitations"),
             "## 13. Go / No-Go decision for hand-contact co-evolution\n", narrative("gonogo"),
             "## Answers to Q1–Q6\n", narrative("answers"),
             "## Files\n", narrative("files")]
    (P.OUT / "hand_contact_predictive_info_report.md").write_text("\n".join(parts))
    print("report written:", P.OUT / "hand_contact_predictive_info_report.md")


if __name__ == "__main__":
    main()
