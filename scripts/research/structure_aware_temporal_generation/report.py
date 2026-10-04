#!/usr/bin/env python
"""Assemble OUT/report.md: the decision table first, then the 17 sections of the request, each = narrative/<section>.md (written
by hand from the numbers) + the generated tables and figure links.    python report.py"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sat_common as S

NARR = S.OUT / "narrative"
SECTIONS = [("1", "Research question", "research_question"), ("2", "Motivation from the previous diagnostics", "motivation"), ("3", "Hypotheses", "hypotheses"),
            ("4", "Datasets and splits", "datasets"), ("5", "Model definitions", "models"), ("6", "Structural-loss implementation", "structural_loss"),
            ("7", "Fixed-s_0 experiment (protocol A)", "fixed_s0"), ("8", "End-to-end sampled-s_0 experiment (protocol B)", "end_to_end"),
            ("9", "Structural-loss ablation", "ablation"), ("10", "Deterministic vs stochastic comparison", "det_vs_stoch"),
            ("11", "Initial vs future stochasticity", "initial_vs_future"), ("12", "Accuracy vs diversity", "accuracy_vs_diversity"),
            ("13", "Event-conditioned analysis", "events"), ("14", "Dataset differences", "dataset_differences"), ("15", "Sanity checks", "sanity"),
            ("16", "Limitations", "limitations"), ("17", "Final answer", "final_answer")]
TABLE_AFTER = {"5": ["table1"], "7": ["table2"], "8": ["table3"], "9": ["table4"], "10": ["table5"], "11": ["table6"], "13": ["table7"]}
FIG_AFTER = {"7": ["figA_six_models.png", "figF_temporal_curves.png"], "9": ["figB_loss_ablation.png"], "10": ["figE_k1_mean_best.png"], "11": ["figC_fixed_s0_diversity.png", "figD_initial_vs_future.png"],
             "12": ["figE_k1_mean_best.png"], "13": ["figG_events.png"]}
MCOL = [("E_C", "dense E_C"), ("part_hamming", "part. Hamming"), ("part_macro_f1", "macro F1"), ("part_macro_auprc", "macro AUPRC"), ("amount_l1", "amount L1"),
        ("amount_rmse", "amount RMSE"), ("centroid", "centroid / l"), ("normal", "normal (deg)"), ("q_rel_l1", "wrench rel L1"), ("q_cos", "wrench cos"), ("q_Qerr", "Q err"),
        ("jitter_dense", "dense jitter"), ("jitter_r2", "R2 jitter")]


def narrative(key):
    p = NARR / f"{key}.md"
    return p.read_text().strip() if p.exists() else f"_({key}: to be written)_"


def f3(v, lo=None, hi=None, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    s = f"{v:.{nd}f}"
    if lo is not None and np.isfinite(lo):
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


def md_table(df, cols=None, index=False):
    df = df if cols is None else df[cols]
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(x) for x in r.values) + " |")
    return "\n".join(lines)


def table1():
    mm = pd.read_csv(S.OUT / "model_matrix.csv"); cfg = pd.read_csv(S.OUT / "run_config.csv")
    rows = []
    for r in mm.itertuples():
        lam = {ds: cfg[(cfg.dataset == ds) & (cfg.model == r.model)] for ds in S.DATASETS}
        ls = " / ".join(f"{lam[ds].lambda_struct.iloc[0]:g}" if len(lam[ds]) and pd.notna(lam[ds].lambda_struct.iloc[0]) else "–" for ds in S.DATASETS)
        la = " / ".join(f"{lam[ds].lambda_aux.iloc[0]:g}" if len(lam[ds]) and pd.notna(lam[ds].lambda_aux.iloc[0]) else "–" for ds in S.DATASETS)
        steps = " / ".join(f"{int(lam[ds].best_step.mean())}" if len(lam[ds]) else "–" for ds in S.DATASETS)
        rows.append({"model": r.model, "family": r.family, "base loss": r.base_loss, "output R2 loss": "yes" if r.output_structural_loss else "no", "hidden aux loss": "yes" if r.hidden_aux_loss else "no",
                     "λ_struct (TACO / ARCTIC)": ls, "λ_aux": la, "params": f"{int(r.n_params) / 1e6:.1f} M" if pd.notna(r.n_params) else "–", "best step (mean of seeds)": steps, "inference": r.inference})
    return "**Table 1. Model definitions.** One backbone (63 frame tokens, width 512, 6 adaLN-FiLM attention blocks, conditioned on G, s_0 [and the diffusion step]); the deterministic head predicts the residual r_t = (C_t − s_0)/σ_r, the diffusion family diffuses the same residual sequence (v-prediction, cosine ᾱ, 1000 steps, DDIM 100). λ chosen once on the validation set (Section 9).\n\n" + md_table(pd.DataFrame(rows))


def acc_table(prot, title, extra_cols=()):
    acc = pd.read_csv(S.OUT / ("fixed_s0_metrics.csv" if prot == "A" else "end_to_end_metrics.csv"))
    cols = list(MCOL) + list(extra_cols)
    out = []
    for ds in S.DATASETS:
        rows = []
        for model in (["GT", "PERSIST"] if prot == "A" else ["PERSIST"]) + list(S.MODELS):
            stats = ["K1"] if model in S.DET or model == "GT" else (["K1", "mean", "best10"] if model in S.DIFF else (["K1", "mean", "best10"] if prot == "B" else ["K1"]))
            for stat in stats:
                r = {"model": {"GT": "GT map (grid floor)", "PERSIST": "persistence C_t = s_0"}.get(model, model), "stat": stat}
                for m, lab in cols:
                    q = acc[(acc.dataset == ds) & (acc.model == model) & (acc.metric == m) & (acc.stat == stat) & (acc.protocol == prot)]
                    r[lab] = f3(q.iloc[0].value, q.iloc[0].ci_lo, q.iloc[0].ci_hi) if len(q) else "–"
                rows.append(r)
        out.append(f"*{S.LABEL[ds]}*\n\n" + md_table(pd.DataFrame(rows)))
    return f"**{title}** Mean over the test sequences of the seed-averaged per-example value [95 % take-cluster bootstrap]; K1 = sample 0, mean = over 10 samples, best10 = per-metric minimum over the 10 samples (oracle selection). Lower is better except macro F1 / AUPRC / wrench cos.\n\n" + "\n\n".join(out)


def pair_table(pairs_sel, title, stats=("K1",)):
    P = pd.read_csv(S.OUT / "paired_comparisons.csv")
    metrics = [("E_C", "dense E_C"), ("part_hamming", "part. Hamming"), ("amount_l1", "amount L1"), ("centroid", "centroid"), ("normal", "normal"), ("q_rel_l1", "wrench rel L1"), ("jitter_dense", "dense jitter")]
    out = []
    for ds in S.DATASETS:
        rows = []
        for a, b in pairs_sel:
            for stat in stats:
                r = {"comparison": f"{b} vs {a}", "stat": stat}
                for m, lab in metrics:
                    q = P[(P.dataset == ds) & (P.a == a) & (P.b == b) & (P.metric == m) & (P.stat == stat) & (P.protocol == "A")]
                    if len(q):
                        x = q.iloc[0]; r[lab] = f"{x.rel_change_pct:+.1f} % [{x.rel_lo:+.1f}, {x.rel_hi:+.1f}]" + ("" if x.seed_consistent else " ~")
                    else:
                        r[lab] = "–"
                rows.append(r)
        out.append(f"*{S.LABEL[ds]}*\n\n" + md_table(pd.DataFrame(rows)))
    return f"**{title}** Relative change of the test mean (b vs a, negative = b better; paired per example, 95 % take bootstrap of the ratio; `~` = the sign differs between seeds). Protocol A.\n\n" + "\n\n".join(out)


def table6():
    D = pd.read_csv(S.OUT / "diversity_metrics.csv")
    out = []
    for ds in S.DATASETS:
        rows = []
        for src in D[D.dataset == ds].source.unique():
            for model in D[(D.dataset == ds) & (D.source == src)].model.unique():
                r = {"source": src, "model": model}
                for m in ("D_C", "D_part", "D_amount", "D_centroid", "D_normal", "D_q", "D_Z"):
                    q = D[(D.dataset == ds) & (D.source == src) & (D.model == model) & (D.metric == m)]
                    r[m] = f3(q.iloc[0].value, q.iloc[0].ci_lo, q.iloc[0].ci_hi) if len(q) else "–"
                rows.append(r)
        out.append(f"*{S.LABEL[ds]}*\n\n" + md_table(pd.DataFrame(rows)))
    return "**Table 6. Diversity decomposition.** Mean pairwise distance between the 10 trajectories of one example (dense ‖·‖, participation Hamming, amount L1 / train-mean total, centroid / l and normal angle over the parts active in both, wrench symmetric relative L1, z-scored R2 vector); initial = 10 sampled s_0 with one future each, future = fixed GT s_0 with 10 stochastic futures.\n\n" + "\n\n".join(out)


def table7():
    E = pd.read_csv(S.OUT / "event_metrics.csv")
    metrics = [("dense_post", "dense @post"), ("part_post", "part. @post"), ("amount_post", "amount @post"), ("centroid_post", "centroid @post"), ("normal_post", "normal @post"), ("wrench_post", "wrench @post"),
               ("transition_realised", "transition realised"), ("transition_timing_error", "timing error (frames)"), ("D_C_window", "D_C window"), ("D_Z_window", "D_Z window")]
    out = []
    for ds in S.DATASETS:
        for cls in S.EVENT_CLASSES:
            rows = []
            n = E[(E.dataset == ds) & (E.cls == cls) & (E.metric == "n_events_total") & (E.stat == "K1")]
            for model in ["PERSIST"] + list(S.MODELS):
                r = {"model": model}
                for m, lab in metrics:
                    q = E[(E.dataset == ds) & (E.model == model) & (E.cls == cls) & (E.metric == m) & (E.stat == "K1")]
                    r[lab] = f3(q.iloc[0].value, q.iloc[0].ci_lo, q.iloc[0].ci_hi, 2) if len(q) else "–"
                rows.append(r)
            out.append(f"*{S.LABEL[ds]} — {S.EVENT_LABEL[cls]} ({int(n.iloc[0].value) if len(n) else 0} events)*\n\n" + md_table(pd.DataFrame(rows)))
    return "**Table 7. Event-conditioned metrics** (protocol A, K1, seed-averaged per event, 95 % take bootstrap): errors at the post-transition frame, the fraction of participation-changing events whose post set is realised within [pre − 4, post + 8] frames and the signed timing error of its first realisation, and the within-window sample diversity of the stochastic models.\n\n" + "\n\n".join(out)


def table8():
    D = pd.read_csv(S.OUT / "decision_summary.csv")
    rows = []
    for q in D.question.unique():
        r = {"Question": q}
        for ds in S.DATASETS:
            x = D[(D.question == q) & (D.dataset == ds)]
            r[S.LABEL[ds]] = x.iloc[0].answer if len(x) else "–"
        r["Evidence"] = " ‖ ".join(f"{S.LABEL[ds]}: {D[(D.question == q) & (D.dataset == ds)].iloc[0].evidence}" for ds in S.DATASETS if len(D[(D.question == q) & (D.dataset == ds)]))
        rows.append(r)
    return md_table(pd.DataFrame(rows))


def sanity_table():
    p = S.OUT / "sanity_summary.json"
    if not p.exists():
        return "_(sanity_summary.json missing)_"
    J = json.loads(p.read_text()); rows = []
    for key in J["taco"]:
        rows.append({"check": key, **{S.LABEL[ds]: J[ds][key]["status"] for ds in S.DATASETS}, "how": J["taco"][key]["how"]})
    return md_table(pd.DataFrame(rows))


def main():
    T = {"table1": table1(), "table2": acc_table("A", "Table 2. Fixed GT s_0 (protocol A), frames 1..63."),
         "table3": acc_table("B", "Table 3. End-to-end with the 10 sampled s_0 (protocol B), frames 0..63.", extra_cols=[("s0_err", "s_0 error")]),
         "table4": pair_table([("D0", "D1"), ("D1", "D2"), ("S0", "S1"), ("S1", "S2")], "Table 4. Structural-loss ablation.", stats=("K1", "mean")),
         "table5": pair_table([("D0", "S0"), ("D1", "S1"), ("D2", "S2")], "Table 5. Deterministic vs stochastic (matched supervision).", stats=("K1", "mean", "best10")),
         "table6": table6(), "table7": table7()}
    parts = ["# Structure-aware temporal contact generation: does R2 supervision help, and must the future stay stochastic?\n",
             f"_{S.OUT}_ · 2026-10-01/02 · TACO and ARCTIC, six models × 3 seeds, protocols A (fixed GT s_0) and B (10 sampled s_0)\n",
             "## Final decision table (Table 8)\n", table8(), "\n" + narrative("decision_table")]
    for num, title, key in SECTIONS:
        parts.append(f"\n## {num}. {title}\n")
        parts.append(narrative(key))
        for t in TABLE_AFTER.get(num, []):
            parts.append("\n" + T[t])
        if num == "15":
            parts.append("\n" + sanity_table())
        for f in FIG_AFTER.get(num, []):
            parts.append(f"\n![{f}](figures/{f})")
        if num == "12":
            sel = S.OUT / "figures" / "figH_selection.csv"
            if sel.exists():
                for r in pd.read_csv(sel).itertuples():
                    parts.append(f"\n![{r.case}](figures/{r.figure})")
    parts.append("\n## Files and reproduction\n" + narrative("files"))
    (S.OUT / "report.md").write_text("\n".join(parts))
    print("report written:", S.OUT / "report.md", "| placeholders:", sum("to be written" in p for p in parts))


if __name__ == "__main__":
    main()
