#!/usr/bin/env python
"""Assemble report.md: the Stage-2 decision table first, then the 18 sections (narrative/<nn>_<name>.md) with Tables 1-10 and the
figures inserted at the referenced places ({{table1}} ... {{table10}}, {{sanity}}, {{fig1}} ...).    python s2_report.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import s2_common as S

NARR = S.OUT / "narrative"
SECTIONS = ["research_question", "background", "stage1_finding", "hypotheses", "dataset_conditioning", "models", "losses", "training_setup", "main_results", "b1_vs_b0",
            "b2_vs_b1", "horizon", "temporal_stability", "dataset_differences", "qualitative", "sanity_checks", "limitations", "conclusion"]
TITLES = ["Research question", "Background from the previous temporal experiment", "Stage-1 finding", "Stage-2 hypotheses", "Dataset / conditioning", "Three model definitions", "Losses",
          "Training setup", "Main B0 vs B1 vs B2 results", "B1 vs B0 — does z help?", "B2 vs B1 — does r help?", "Horizon analysis", "Temporal stability", "Dataset differences",
          "Qualitative analysis", "Sanity checks", "Limitations", "Final conclusion"]
MODEL_LAB = {"B0": "B0 direct dense", "B1": "B1 z → C", "B2": "B2 (z, r) → C", "B2_zonly": "B2, z-only path C̄", "D0_prev": "previous D0 (25.7 M, 3 seeds)", "PERSIST": "persistence C_t = s₀", "GT": "GT (grid-extraction floor)"}
T2_METRICS = [("E_C", "E_C"), ("E_C_last16", "E_C last 16"), ("part_hamming", "part. Hamming"), ("part_macro_f1", "macro F1 ↑"), ("part_macro_auprc", "AUPRC ↑"), ("amount_l1", "amount L1"),
              ("centroid", "centroid / l"), ("normal", "normal °"), ("q_rel_l1", "wrench rel L1"), ("q_cos", "wrench cos ↑"), ("jitter_dense", "dense Δ"), ("jitter_r2", "R2 Δ")]
PAIR_METRICS = ["E_C", "E_C_last16", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos", "jitter_dense", "jitter_r2", "jitter_dev_dense", "jitter_dev_r2", "L_z"]


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def load(name):
    p = S.OUT / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def get(acc, ds, m, met):
    r = acc[(acc.dataset == ds) & (acc.model == m) & (acc.metric == met)]
    return None if r.empty else r.iloc[0]


def table1(mt, ts):
    rows = []
    for r in mt.itertuples():
        tot = {ds: ts[(ts.dataset == ds) & (ts.model == r.model)].n_params_total.iloc[0] for ds in S.DATASETS if len(ts[(ts.dataset == ds) & (ts.model == r.model)])}
        bb = {ds: ts[(ts.dataset == ds) & (ts.model == r.model)].n_params_backbone.iloc[0] for ds in S.DATASETS if len(ts[(ts.dataset == ds) & (ts.model == r.model)])}
        rows.append([r.model, r.intermediate, r.output_head, r.decoder, r.losses, " / ".join(f"{bb[ds] / 1e6:.1f} M" for ds in bb), " / ".join(f"{tot[ds] / 1e6:.1f} M" for ds in tot)])
    return md_table(["model", "intermediate", "output head", "decoder", "losses", "backbone params (TACO / ARCTIC)", "total params (TACO / ARCTIC)"], rows)


def table2(acc):
    out = ""
    for ds in S.DATASETS:
        rows = []
        for m in ["B0", "B1", "B2", "B2_zonly", "D0_prev", "PERSIST", "GT"]:
            cells = [MODEL_LAB[m]]
            for met, _ in T2_METRICS:
                r = get(acc, ds, m, met)
                if r is None:
                    cells.append("–"); continue
                d = 1 if met == "normal" else 3
                cells.append(f"**{f(r.value, d)}** [{f(r.ci_lo, d)}, {f(r.ci_hi, d)}]" if m in ("B0", "B1", "B2") else f(r.value, d))
            rows.append(cells)
        out += f"\n**{S.LABEL[ds]}** (test, fixed GT s₀; mean over sequences, 95 % take-cluster bootstrap CI for the three Stage-2 models; ↑ = higher is better, all others lower is better)\n\n" + md_table(["model"] + [l for _, l in T2_METRICS], rows)
    return out


def pair_table(P, a, b):
    out = ""
    for ds in S.DATASETS:
        rows = []
        for met in PAIR_METRICS:
            r = P[(P.dataset == ds) & (P.a == a) & (P.b == b) & (P.metric == met)]
            if r.empty:
                continue
            r = r.iloc[0]; d = 1 if met == "normal" else 3
            rows.append([met, f(r.value_b, d), f(r.value_a, d), f"{r['diff']:+.{d}f} [{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}]", f"{r.rel_improvement * 100:+.1f} %", r.verdict])
        out += f"\n**{S.LABEL[ds]}**\n\n" + md_table(["metric", b, a, f"diff {a} − {b} [95 % CI]", f"improvement of {a}", "verdict"], rows)
    return out


def table5(zm, rm):
    rows = []
    for r in zm.itertuples():
        if r.model == "B2_vs_B1":
            rows.append([S.LABEL[r.dataset], "B2 − B1", f"{r.L_z_test:+.3f} [{r.L_z_lo:+.3f}, {r.L_z_hi:+.3f}]", f"ratio {r.ratio_B2_B1:.2f}", "", "", "", "", "", ""]); continue
        rows.append([S.LABEL[r.dataset], r.model, f"{f(r.L_z_test)} [{f(r.L_z_lo)}, {f(r.L_z_hi)}]", f(r.L_z_hold_z0), f(r.L_z_train_mean), f(r.rmse_test), f"{f(r.rmse_h1_8)} / {f(r.rmse_h9_16)} / {f(r.rmse_h17_32)} / {f(r.rmse_h33_48)} / {f(r.rmse_h49_63)}",
                     f"{f(r.rmse_hold_h1_8)} / {f(r.rmse_hold_h9_16)} / {f(r.rmse_hold_h17_32)} / {f(r.rmse_hold_h33_48)} / {f(r.rmse_hold_h49_63)}", f"{f(r.r2_per_dim_mean, 2)} [{f(r.r2_per_dim_min, 2)}, {f(r.r2_per_dim_max, 2)}]", f(r.corr_Lz_EC, 2)])
    t = md_table(["dataset", "model", "test L_z (mean sq. std. error) [CI]", "hold z*₀", "train mean", "RMSE", "RMSE by horizon 1–8 / 9–16 / 17–32 / 33–48 / 49–63", "hold-z*₀ RMSE by horizon", "R² per dim mean [min, max]", "corr(L_z, E_C)"], rows)
    if len(rm):
        rows = [[S.LABEL[r.dataset], f(r.E_C_zonly), f(r.E_C_full), f"{r.E_C_gain_from_r:+.3f} [{r.gain_lo:+.3f}, {r.gain_hi:+.3f}]", f"{r.frac_sequences_improved * 100:.0f} %", f(r.delta_norm_mean), f(r.bar_minus_s0_norm_mean), f(r.delta_over_bar_ratio, 2),
                 f(r.r_std_mean, 2), f(r.r_eff_rank, 1), f(r.r_acf_h8, 2)] for r in rm.itertuples()]
        t += "\n**Table 5b — The r pathway of B2 (test)**\n\n" + md_table(["dataset", "E_C of C̄ (z-only)", "E_C of Ĉ (full)", "gain from ΔC [CI]", "sequences improved", "mean ‖ΔC_t‖", "mean ‖C̄_t − s₀‖", "ratio", "std of r̂ (mean over dims)", "eff. rank of r̂", "acf of r̂ at h = 8"], rows)
    return t


def table6(acc):
    rows = []
    for ds in S.DATASETS:
        for m in ["GT", "B0", "B1", "B2", "D0_prev", "PERSIST"]:
            cells = [S.LABEL[ds], MODEL_LAB[m]]
            for met in ["jitter_dense", "jitter_r2", "tv_dense", "jitter_dev_dense", "jitter_dev_r2", "jitter_ratio_dense", "jitter_ratio_r2"]:
                r = get(acc, ds, m, met)
                cells.append("–" if r is None else (f"{f(r.value)} [{f(r.ci_lo)}, {f(r.ci_hi)}]" if m in ("B0", "B1", "B2") else f(r.value)))
            rows.append(cells)
    return md_table(["dataset", "model", "dense frame change", "R2 frame change", "dense total variation", "|dense change − GT|", "|R2 change − GT|", "dense change / GT", "R2 change / GT"], rows)


def table7(D):
    qs = [q for q in D.question.unique() if q != "case"]
    rows = []
    for q in qs:
        cells = [q]
        for ds in S.DATASETS:
            r = D[(D.dataset == ds) & (D.question == q)]
            cells.append("–" if r.empty else f"**{r.iloc[0].answer}**")
        ev = " / ".join(f"{S.LABEL[ds]}: {D[(D.dataset == ds) & (D.question == q)].iloc[0].evidence}" for ds in S.DATASETS if not D[(D.dataset == ds) & (D.question == q)].empty)
        cells.append(ev); rows.append(cells)
    cases = {ds: D[(D.dataset == ds) & (D.question == "case")].iloc[0].answer for ds in S.DATASETS if not D[(D.dataset == ds) & (D.question == "case")].empty}
    rows.append(["Result case (Section 20 of the plan)"] + [f"**Case {cases.get(ds, '–')}**" for ds in S.DATASETS] + ["A strong factorised; B z useful, r not needed; C z structure + r dense; D no benefit from z; E z hurts"])
    return md_table(["Question", "TACO", "ARCTIC", "Evidence"], rows)


def table8(ts):
    rows = []
    for r in ts.itertuples():
        lg = pd.read_csv(S.train_log_path(r.dataset, S.run_name(r.model)))
        rows.append([S.LABEL[r.dataset], r.model, f"{r.n_params_total / 1e6:.1f} M", int(r.steps), int(r.best_step), r.stopped_by, f(r.val_init, 3), f(r.best_val, 4), f(r.val_at_20k, 4) if not np.isnan(r.val_at_20k) else "–", f(r.final_val, 4),
                     f(r.val_dense_at_best, 4), f"{f(r.min_val_dense, 4)} ({int(r.min_val_dense_step) // 1000} k)", f(float(lg.train.iloc[-1]), 4), f"{r.hours:.1f} h", f(getattr(r, "gnorm_ratio_max_min_1000", np.nan), 1), f(getattr(r, "gnorm_ratio_max_min_5000", np.nan), 1)])
    return md_table(["dataset", "model", "params", "steps", "selected step", "stopped by", "val objective at init", "at the selected step", "at 20 k", "at the end", "val dense term at the selected step", "its minimum over the run (step)",
                     "training objective at the end", "wall time", "grad-norm ratio max / min of the weighted terms (step 1000)", "(step 5000)"], rows)


def table9(H):
    out = ""
    for ds in S.DATASETS:
        rows = []
        for c, lab in [("dense", "dense error"), ("part", "part. Hamming"), ("centroid", "centroid / l"), ("wrench", "wrench rel L1")]:
            for m in ["B0", "B1", "B2", "GT"]:
                cells = [lab, MODEL_LAB[m]]
                for lo, hi in S.HORIZON_BINS:
                    r = H[(H.dataset == ds) & (H.model == m) & (H.curve == c) & (H.h_lo == lo)]
                    cells.append("–" if r.empty else f(r.iloc[0].value))
                rows.append(cells)
        out += f"\n**{S.LABEL[ds]}**\n\n" + md_table(["curve", "model"] + [f"t = {lo}–{hi}" for lo, hi in S.HORIZON_BINS], rows)
    return out


def table10(P):
    return ("**B2 vs B0**\n" + pair_table(P, "B2", "B0") + "\n**B0 (this study, 74.7 M, 80 k-step budget) vs the previous D0 (25.7 M, per-sequence mean of 3 seeds)**\n" + pair_table(P, "B0", "D0_prev")
            + "\n**B1 vs the previous D0**\n" + pair_table(P, "B1", "D0_prev") + "\n**B2's z-only path vs B1** (same decoder family, different training)\n" + pair_table(P, "B2_zonly", "B1"))


def table11(B):
    if not len(B):
        return "(bottleneck diagnostic not available)\n"
    rows = [[S.LABEL[r.dataset], r.model, f"{f(r.E_C_pred)} [{f(r.E_C_pred_lo)}, {f(r.E_C_pred_hi)}]", f"{f(r.E_C_oracle_z)} [{f(r.E_C_oracle_z_lo)}, {f(r.E_C_oracle_z_hi)}]", f"{f(r.gap_pred_minus_oracle)} [{f(r.gap_pred_minus_oracle_lo)}, {f(r.gap_pred_minus_oracle_hi)}]",
             f(r.E_C_recon_s0), f(r.stage1_E_zonly), f(r.stage1_E_full), " / ".join(f(getattr(r, f"oracle_h{lo}_{hi}")) for lo, hi in S.HORIZON_BINS), " / ".join(f(getattr(r, f"pred_h{lo}_{hi}")) for lo, hi in S.HORIZON_BINS)] for r in B.itertuples()]
    return md_table(["dataset", "model", "E_C, predicted latents [CI]", "E_C, teacher z* through the same decoder [CI]", "gap (prediction error of the latent)", "‖D_z(z*₀) − s₀‖ (frame 0)", "Stage-1 A3 autoencoder, z-only reconstruction of the same frames (1–63 of the test windows)", "Stage-1 A3, z + r reconstruction of the same frames",
                     "teacher-z* error by horizon 1–8 / 9–16 / 17–32 / 33–48 / 49–63", "predicted error by horizon"], rows)


def sanity_table():
    p = S.OUT / "sanity_summary.json"
    if not p.exists():
        return "(sanity summary not available)\n"
    J = json.loads(p.read_text()); rows = []
    keys = sorted({k for ds in J for k in J[ds]})
    for k in keys:
        cells = [k.split("_", 1)[0], k.split("_", 1)[1].replace("_", " ")]
        for ds in S.DATASETS:
            cells.append(J.get(ds, {}).get(k, {}).get("status", "–"))
        cells.append(next((J[ds][k].get("how", "") for ds in J if k in J[ds]), ""))
        rows.append(cells)
    return md_table(["#", "check", "TACO", "ARCTIC", "how"], rows)


def main():
    acc, P, zm, rm, ts, mt, D, H = load("main_metrics.csv"), load("paired_comparisons.csv"), load("z_metrics.csv"), load("r_metrics.csv"), load("training_summary.csv"), load("model_table.csv"), load("decision_summary.csv"), load("horizon_metrics.csv")
    cap = lambda n, title, body: f"**Table {n} — {title}**\n\n{body}"
    note = "Paired differences on identical test sequences; *improvement* is the relative change in the better direction (positive = the first model is better); verdict by the pre-registered rule (interval excludes 0 and |change| ≥ 2 %)."
    tables = {"table11": cap(11, "Bottleneck inspection of the latent-mediated models (test; analysis only)", table11(load("bottleneck_metrics.csv"))),
              "table1": cap(1, "Model definitions and parameter counts", table1(mt, ts)), "table2": cap(2, "Main fixed-s₀ results", table2(acc)),
              "table3": cap(3, "B1 vs B0, paired comparison. " + note, pair_table(P, "B1", "B0")), "table4": cap(4, "B2 vs B1, paired comparison. " + note, pair_table(P, "B2", "B1")),
              "table5": cap(5, "z prediction diagnostics (test; standardised teacher coordinates)", table5(zm, rm)), "table6": cap(6, "Temporal stability (test)", table6(acc)), "table7": table7(D),
              "table8": cap(8, "Training summary", table8(ts)), "table9": cap(9, "Error by horizon bin (test means)", table9(H)), "table10": cap(10, "Further paired comparisons. " + note, table10(P)),
              "sanity": cap(12, "Sanity checks", sanity_table())}
    figs = {f"fig{i}": f"![Figure {i}](figures/{n})" for i, n in [(1, "fig1_schematic.png"), (2, "fig2_main_metrics.png"), (3, "fig3_horizon.png"), (4, "fig4_stability.png"), (6, "fig6_validation_curves.png")]}
    figs["fig5"] = "\n".join(f"![Figure 5{c} {S.LABEL[ds]}](figures/fig5{c}_{ds}.png)" for ds in S.DATASETS for c in "abc" if (S.OUT / "figures" / f"fig5{c}_{ds}.png").exists())
    head = (NARR / "00_head.md").read_text() if (NARR / "00_head.md").exists() else "# Contact latent temporal generation — Stage 2\n"
    body = head + "\n## Decision table (Table 7)\n\n" + tables["table7"] + "\n"
    for i, (sec, title) in enumerate(zip(SECTIONS, TITLES), 1):
        p = NARR / f"{i:02d}_{sec}.md"
        txt = p.read_text() if p.exists() else "_(section not written)_\n"
        for k, v in {**tables, **figs}.items():
            txt = txt.replace("{{" + k + "}}", v)
        body += f"\n## {i}. {title}\n\n{txt}\n"
    (S.OUT / "report.md").write_text(body)
    print("report.md:", len(body.splitlines()), "lines; unresolved placeholders:", body.count("{{"))


if __name__ == "__main__":
    main()
