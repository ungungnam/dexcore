#!/usr/bin/env python
"""Assemble the cross-dataset report from the per-dataset results directories.
  python report.py            -> /result/uhnam/dexcore/reports/hier_contact_gen_report.md
Narrative sections are read from /result/uhnam/dexcore/reports/hier_narrative/{verdict,taco,arctic,
oakink2,control,deviations}.md when present; every number in the auto tables cites its file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOTS = {"taco": Path("/result/uhnam/dexcore/taco/50_hier_contact_gen"), "arctic": Path("/result/uhnam/dexcore/arctic/40_hier_contact_gen"),
         "oakink2": Path("/result/uhnam/dexcore/oakink2/20_hier_contact_gen")}
LABEL = {"taco": "TACO", "arctic": "ARCTIC", "oakink2": "OakInk2"}
REP = Path("/result/uhnam/dexcore/reports")
NARR = REP / "hier_narrative"
MAIN = [("static_gt", 0, "B0 static GT-init"), ("gtinit_vf", 0, "B1 GT-init + VF"), ("samplerG_vf", 1, "B2 p(S0\\|G) + VF, K=1"),
        ("samplerG_vf", 10, "B2 p(S0\\|G) + VF, best-of-10"), ("samplerGT_vf", 1, "B3 p(S0\\|G,τ_local) + VF, K=1"),
        ("samplerGT_vf", 10, "B3 p(S0\\|G,τ_local) + VF, best-of-10")]


def narr(name):
    p = NARR / f"{name}.md"
    return p.read_text().strip() + "\n" if p.exists() else f"_(narrative {name} not written)_\n"


def load(ds):
    r = ROOTS[ds]
    if not (r / "results" / "aggregate.csv").exists():
        return None
    return dict(A=pd.read_csv(r / "results" / "aggregate.csv"), P=pd.read_csv(r / "results" / "paired_differences.csv"),
                Cv=pd.read_csv(r / "results" / "curves.csv"), Tr=pd.read_csv(r / "results" / "training_summary.csv"),
                cfg=json.load(open(r / "config" / "sequence_construction.json")), G=pd.read_csv(r / "results" / "per_group.csv"))


def config_table(D):
    rows = ["| | " + " | ".join(LABEL[d].split(" (")[0] for d in D) + " |", "|---|" + "---|" * len(D)]
    def add(label, f):
        rows.append(f"| {label} | " + " | ".join(str(f(D[d])) for d in D) + " |")
    add("sequence length T (frames) / rate / stride", lambda x: f"{x['cfg']['T']} / {x['cfg']['fps']} Hz / {x['cfg']['stride']}")
    add("firm-contact rule", lambda x: f"{x['cfg']['firm_rule']} (zero_mass {x['cfg']['zero_mass']}, hard 1 cm), {x['cfg']['firm_persist_frames']} consecutive frames")
    add("episode gap (new onset after ≥ N non-firm frames)", lambda x: x['cfg']['episode_gap_frames'])
    add("contact source", lambda x: x['cfg']['contact_source'])
    add("(take, group) pairs / with firm contact / onsets", lambda x: f"{x['cfg']['n_take_group_pairs']} / {x['cfg']['n_with_firm_contact']} / {x['cfg']['n_onsets']}")
    add("sequences kept (T frames after an onset) / of which first onset of the take", lambda x: f"{x['cfg']['n_sequences']} / {x['cfg']['n_first_onset_sequences']}")
    add("distinct split units (takes; OakInk2: recordings)", lambda x: x["cfg"]["n_takes"])
    add("object-state dimension", lambda x: x['cfg']['state_dim'])
    add("local context", lambda x: f"O_t at offsets {x['cfg']['local_context_offsets'][0]}..{x['cfg']['local_context_offsets'][-1]} step 2 (9 states), clamped at take bounds")
    add("split (take level)", lambda x: f"{x['cfg']['split']}: {x['cfg']['n_train_seq']} / {x['cfg']['n_val_seq']} / {x['cfg']['n_test_seq']} sequences = {x['cfg']['n_train_takes']} / {x['cfg']['n_val_takes']} / {x['cfg']['n_test_takes']} takes")
    return "\n".join(rows)


def training_table(D):
    out = ["| dataset | model | params | best validation value | best step | stopped at | stop rule |", "|---|---|---|---|---|---|---|"]
    params = {"sampler_G": "12.4 M", "sampler_GT": "12.5 M", "vf": "9.8 M", "vf_unroll8": "9.8 M", "vf_reg": "9.8 M", "vf_regnoise": "9.8 M", "vf_sel": "9.8 M", "vf_w512": "3.7 M", "vf_noise": "9.8 M", "dense": "9.5 M"}
    for d, x in D.items():
        for m, g in x["Tr"].groupby("model"):
            out.append(f"| {LABEL[d]} | {m} | {params.get(m, '')} | {g.best_val.mean():.4f} | {g.best_step.mean():.0f} | {g.steps.mean():.0f} | {', '.join(g.stopped_by.astype(str))} |")
    return "\n".join(out)


def main_table(ds, x, split, extended=False):
    S = x["A"][x["A"].split == split].set_index(["model", "K"])
    rows = MAIN if not extended else [("samplerG_vf", 5, "B2, best-of-5"), ("samplerG_vf", -1, "B2, mean over the 10 samples"), ("samplerGT_vf", 5, "B3, best-of-5"),
                                      ("samplerGT_vf", -1, "B3, mean over the 10 samples"), ("samplerG_static", 1, "p(S0\\|G) held static, K=1"),
                                      ("samplerG_static", 10, "p(S0\\|G) held static, best-of-10"), ("samplerGT_static", 1, "p(S0\\|G,τ) held static, K=1"),
                                      ("samplerGT_static", 10, "p(S0\\|G,τ) held static, best-of-10")]
    n = S.n_examples.max(); nt = S.n_takes.max()
    out = [f"**{LABEL[ds]}** — test set of the fixed split ({int(n)} sequences from {int(nt)} takes; "
           f"file `{ROOTS[ds].relative_to('/result/uhnam/dexcore')}/results/aggregate.csv`)", "",
           "| variant | E_C (raw) [95 % CI] | E_ΔC | mean pred Δ | mean true Δ | pattern L2 | mass abs | S0 err | E_C macro | E_C first-onset only |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, K, lab in rows:
        if (m, K) not in S.index:
            continue
        r = S.loc[(m, K)]
        out.append(f"| {lab} | {r.E_C:.3f} [{r.E_C_lo:.3f}, {r.E_C_hi:.3f}] | {r.E_dC:.3f} | {r.dmag_pred:.3f} | {r.dmag_true:.3f} | {r.pattern_L2:.4f} | {r.mass_abs:.2f} | {r.s0_err:.3f} | {r.E_C_macro:.3f} | {r.E_C_first:.3f} |")
    return "\n".join(out)


REF_ROWS = [("ref_mean_static", 0, "mean first-contact map of the training takes of the same mesh, held static"),
            ("ref_take_static", 1, "first-contact map of a random training take of the same mesh, held static (K=1)"),
            ("ref_take_static", 10, "… best-of-10 random training takes (empirical best-of-K)"),
            ("ref_take_static", -1, "… mean over the 10 training takes"),
            ("ref_nearest_static", 0, "training first-contact map nearest to the test C_0, held static (retrieval oracle)")]


def ref_table(ds):
    p = ROOTS[ds] / "results" / "reference_baselines_summary.csv"
    if not p.exists():
        return ""
    S = pd.read_csv(p).set_index(["model", "K"])
    out = [f"Data-only reference points for the initial map (`results/reference_baselines_summary.csv`, same test sequences, held static over the {64} frames):", "",
           "| reference | E_C (raw) [95 % CI] | S0 err [95 % CI] | E_C macro |", "|---|---|---|---|"]
    for m, K, lab in REF_ROWS:
        if (m, K) in S.index:
            r = S.loc[(m, K)]
            out.append(f"| {lab} | {r.E_C:.3f} [{r.E_C_lo:.3f}, {r.E_C_hi:.3f}] | {r.s0_err:.3f} [{r.s0_err_lo:.3f}, {r.s0_err_hi:.3f}] | {r.E_C_macro:.3f} |")
    return "\n".join(out)


def gt_window_max(ds):
    z = np.load(ROOTS[ds] / "eval" / "fixed0" / "preds.npz")
    return float(np.abs(z["gt"].astype(np.float32)).max((1, 2)).mean())


def ref(ds, m, K, c="s0_err"):
    p = ROOTS[ds] / "results" / "reference_baselines_summary.csv"
    if not p.exists():
        return "n/a"
    S = pd.read_csv(p).set_index(["model", "K"])
    return f"{S.loc[(m, K), c]:.3f}" if (m, K) in S.index else "n/a"


B4_ROWS = [("bimart", 1, "B4 BimArt contact stage, K=1"), ("bimart", 10, "B4 BimArt contact stage, best-of-10"), ("bimart", -1, "B4, mean over the 10 samples"),
           ("gt_restricted", 0, "GT on BimArt's BPS support (representation gap; K=0)"), ("static_gtr", 0, "GT_r(0) held static, vs full GT"),
           ("bimart_vs_gtr", 1, "B4 vs GT on its own support, K=1"), ("bimart_vs_gtr", 10, "B4 vs GT on its own support, best-of-10"),
           ("static_gtr_vs_gtr", 0, "B0 on BimArt's support: GT_r(0) held static, vs GT_r")]


def b4_tables(ds):
    """The BimArt block: rows from results/bimart_comparison.csv + the paired comparisons."""
    p = ROOTS[ds] / "results" / "bimart_comparison.csv"
    if not p.exists():
        return ""
    S = pd.read_csv(p).set_index(["model", "K"]); P = pd.read_csv(ROOTS[ds] / "results" / "bimart_paired.csv").set_index("comparison")
    n = int(S.n.max())
    out = [f"**B4 — BimArt's contact stage on the same {n} test sequences** (`results/bimart_comparison.csv`; TACO: the gen3 scene-scale contact model trained on the same train split; "
           f"ARCTIC: BimArt's pretrained contact model). BimArt predicts hand-surface distances at 512 BPS basis points per object part, each gathering one mesh vertex per frame; "
           f"prediction and GT are splatted onto the canonical support with the study's operator restricted to those vertices (X_r). "
           f"Rows 'vs full GT' are on the same target as B0–B3 but carry the representation gap of the BPS support (the 'GT on BimArt's BPS support' row); "
           f"rows 'on its own support' compare X_r(BimArt) with X_r(GT), with B0 recomputed on that support.", "",
           "| variant | E_C (raw) [95 % CI] | E_ΔC | mean pred Δ | mean true Δ | pattern L2 | mass abs | S0 err | E_C macro |", "|---|---|---|---|---|---|---|---|---|"]
    for m, K, lab in B4_ROWS:
        if (m, K) in S.index:
            r = S.loc[(m, K)]
            out.append(f"| {lab} | {r.E_C:.3f} [{r.E_C_lo:.3f}, {r.E_C_hi:.3f}] | {r.E_dC:.3f} | {r.dmag_pred:.3f} | {r.dmag_true:.3f} | {r.pattern_L2:.4f} | {r.mass_abs:.2f} | {r.s0_err:.3f} | {r.E_C_macro:.3f} |")
    out += ["", "Paired differences on the same sequences (E_C, 95 % take-bootstrap CI; < 0: BimArt better), `results/bimart_paired.csv`:", "", "| comparison | ΔE_C | fraction where BimArt is better |", "|---|---|---|"]
    for c in P.index:
        r = P.loc[c]
        out.append(f"| {c} | {r.E_C:+.3f} [{r.E_C_lo:+.3f}, {r.E_C_hi:+.3f}] | {100 * r.E_C_frac_better:.0f} % |")
    return "\n".join(out)


def q_table(ds, x, split="fixed"):
    A = x["A"][x["A"].split == split].set_index(["model", "K"]); P = x["P"][x["P"].split == split].set_index("comparison")
    g = lambda m, K, c="E_C": float(A.loc[(m, K), c]) if (m, K) in A.index else np.nan
    p = lambda c, met="E_C": (float(P.loc[c, met]), float(P.loc[c, f"{met}_lo"]), float(P.loc[c, f"{met}_hi"]), float(P.loc[c, f"{met}_frac_better"])) if c in P.index else (np.nan,) * 4
    b0, b1, b2_1, b2_10, b3_1, b3_10, b4 = g("static_gt", 0), g("gtinit_vf", 0), g("samplerG_vf", 1), g("samplerG_vf", 10), g("samplerGT_vf", 1), g("samplerGT_vf", 10), g("dense", 0)
    cv = x["Cv"][(x["Cv"].split == split) & (x["Cv"].metric == "err")]
    curve = lambda m, K, t: float(cv[(cv.model == m) & (cv.K == K) & (cv.t == t)].value.iloc[0]) if len(cv[(cv.model == m) & (cv.K == K) & (cv.t == t)]) else np.nan
    q2 = p("Q2 VF - static (GT init)"); q3a = p("Q3 sampled(G,K=1) - GT init"); q3b = p("Q3 sampled(G,best10) - GT init")
    q4a = p("Q4 GT-sampler - G-sampler (K=1)"); q4b = p("Q4 GT-sampler - G-sampler (best10)"); q4s = p("Q4 S0 only: GT - G (best10)", "s0_err"); q4s1 = p("Q4 S0 only: GT - G (K=1)", "s0_err")
    dvf = p("VF vs static, sampled G best10"); dvf1 = p("VF vs static, sampled G K=1"); nvf = p("noise VF - VF (GT init)"); db1 = p("dense - B1"); db2 = p("dense - B2 best10")
    rows = [f"| Q1 | static GT-init E_C | {b0:.3f}; true mean frame change {g('static_gt', 0, 'dmag_true'):.3f}, error at t = 0/16/32/63: {curve('static_gt', 0, 0):.2f}/{curve('static_gt', 0, 16):.2f}/{curve('static_gt', 0, 32):.2f}/{curve('static_gt', 0, 63):.2f} |",
            f"| Q2 | B1 − B0 (paired, take-bootstrap CI; fraction of sequences where the VF is better) | {q2[0]:+.3f} [{q2[1]:+.3f}, {q2[2]:+.3f}], better in {100 * q2[3]:.0f} % → VF removes {100 * (b0 - b1) / b0:.1f} % of the static error; B1 E_C at t = 16/32/63: {curve('gtinit_vf', 0, 16):.2f}/{curve('gtinit_vf', 0, 32):.2f}/{curve('gtinit_vf', 0, 63):.2f} |",
            f"| Q3 | B2 − B1: K = 1 / best-of-10 | {q3a[0]:+.3f} [{q3a[1]:+.3f}, {q3a[2]:+.3f}] / {q3b[0]:+.3f} [{q3b[1]:+.3f}, {q3b[2]:+.3f}] (B1 {b1:.3f} → B2 {b2_1:.3f} / {b2_10:.3f}) |",
            f"| Q4 | B3 − B2: K = 1 / best-of-10 (< 0: local trajectory helps) | {q4a[0]:+.3f} [{q4a[1]:+.3f}, {q4a[2]:+.3f}] / {q4b[0]:+.3f} [{q4b[1]:+.3f}, {q4b[2]:+.3f}]; S0 error ‖Ŝ0 − C0‖ of the sample, GT − G: K=1 {q4s1[0]:+.3f} [{q4s1[1]:+.3f}, {q4s1[2]:+.3f}], best-of-10 (k* by the held-static E_C) {q4s[0]:+.3f} [{q4s[1]:+.3f}, {q4s[2]:+.3f}] |",
            f"| Q5 | E_C vs K for B2: K = 1 / 5 / 10 / mean-of-10; gap to B1 closed by best-of-10 | {b2_1:.3f} / {g('samplerG_vf', 5):.3f} / {b2_10:.3f} / {g('samplerG_vf', -1):.3f}; " + (f"{100 * (b2_1 - b2_10) / (b2_1 - b1):.0f} % of (B2(K=1) − B1)" if b2_1 - b1 > 1e-6 else "n/a (B2 at K=1 is not worse than B1)") + " |",
            f"| Q5 | S0 error of the sampled map vs the empirical prior: p(S0\\|G) K=1 / best-of-10 vs a random training take of the same mesh K=1 / best-of-10; train mean map; nearest training map | {g('samplerG_static', 1, 's0_err'):.3f} / {g('samplerG_static', 10, 's0_err'):.3f} vs {ref(ds, 'ref_take_static', 1)} / {ref(ds, 'ref_take_static', 10)}; {ref(ds, 'ref_mean_static', 0)}; {ref(ds, 'ref_nearest_static', 0)} |",
            f"| Q6 | rollout stability: B1 E_C over the last 16 frames vs overall; predicted / true frame change (B1, B2 best-10); mean per-sequence largest predicted value (B1) vs the same statistic of the GT window; fraction of predicted entries below −5 % of the C_0 maximum | {g('gtinit_vf', 0, 'E_C_last16'):.3f} vs {b1:.3f}; {g('gtinit_vf', 0, 'dmag_pred'):.3f} & {g('samplerG_vf', 10, 'dmag_pred'):.3f} vs {g('static_gt', 0, 'dmag_true'):.3f}; {g('gtinit_vf', 0, 'max_abs'):.2f} vs {gt_window_max(ds):.2f}; {100 * g('gtinit_vf', 0, 'neg_frac'):.2f} % |",
            f"| VF vs holding the sampled S0 | B2 best-of-10 rollout − same S0 held static / K=1 | {dvf[0]:+.3f} [{dvf[1]:+.3f}, {dvf[2]:+.3f}] / {dvf1[0]:+.3f} [{dvf1[1]:+.3f}, {dvf1[2]:+.3f}] |",
            ]
    return "\n".join(["| question | quantity | value |", "|---|---|---|"] + rows)


def cross_table(D):
    out = ["| quantity | " + " | ".join(LABEL[d].split(" (")[0] for d in D) + " |", "|---|" + "---|" * len(D)]
    def add(label, f):
        out.append(f"| {label} | " + " | ".join(f(D[d]) for d in D) + " |")
    A = lambda x, m, K, c="E_C": float(x["A"][(x["A"].split == "fixed") & (x["A"].model == m) & (x["A"].K == K)][c].iloc[0])
    add("true mean frame-to-frame change ‖ΔC‖", lambda x: f"{A(x, 'static_gt', 0, 'dmag_true'):.3f}")
    add("B0 static GT-init E_C", lambda x: f"{A(x, 'static_gt', 0):.3f}")
    add("B1 GT-init + VF E_C", lambda x: f"{A(x, 'gtinit_vf', 0):.3f}")
    add("VF removes … of the static error", lambda x: f"{100 * (A(x, 'static_gt', 0) - A(x, 'gtinit_vf', 0)) / A(x, 'static_gt', 0):.0f} %")
    add("B2 p(S0\\|G)+VF, K=1 / best-of-10", lambda x: f"{A(x, 'samplerG_vf', 1):.3f} / {A(x, 'samplerG_vf', 10):.3f}")
    add("B3 p(S0\\|G,τ)+VF, K=1 / best-of-10", lambda x: f"{A(x, 'samplerGT_vf', 1):.3f} / {A(x, 'samplerGT_vf', 10):.3f}")
    add("S0 error, p(S0\\|G) K=1 / best-of-10 (k* by the held-static E_C)", lambda x: f"{A(x, 'samplerG_static', 1, 's0_err'):.3f} / {A(x, 'samplerG_static', 10, 's0_err'):.3f}")
    def b4(x, m, K, c="E_C"):
        p = x["root"] / "results" / "bimart_comparison.csv"
        if not p.exists():
            return "—"
        S = pd.read_csv(p).set_index(["model", "K"])
        return f"{S.loc[(m, K), c]:.3f}" if (m, K) in S.index else "—"
    add("B4 BimArt vs full GT, K=1 / best-of-10", lambda x: f"{b4(x, 'bimart', 1)} / {b4(x, 'bimart', 10)}")
    add("B4 BimArt on its own support, K=1 / best-of-10 (B0 there)", lambda x: f"{b4(x, 'bimart_vs_gtr', 1)} / {b4(x, 'bimart_vs_gtr', 10)} ({b4(x, 'static_gtr_vs_gtr', 0)})")
    add("representation gap of the BPS support, E_C(X_r(GT), GT)", lambda x: b4(x, 'gt_restricted', 0))
    add("B1 predicted / true ‖ΔC‖", lambda x: f"{A(x, 'gtinit_vf', 0, 'dmag_pred'):.3f} / {A(x, 'static_gt', 0, 'dmag_true'):.3f}")
    add("B1 temporal error E_ΔC vs static", lambda x: f"{A(x, 'gtinit_vf', 0, 'E_dC'):.3f} vs {A(x, 'static_gt', 0, 'E_dC'):.3f}")
    def leaves(x):
        m = pd.read_csv(x["root"] / "sequences_meta.csv")
        return f"{100 * m[m.split_set == 'test'].leaves_contact.mean():.0f} %"
    add("test sequences leaving firm contact within T", leaves)
    return "\n".join(out)


def main():
    D = {}
    for ds in ROOTS:
        x = load(ds)
        if x is not None:
            x["root"] = ROOTS[ds]; D[ds] = x
    main_ds = list(D)
    L = ["# Hierarchical contact generation: initial contact-mode sampler + deterministic contact-evolution vector field", "",
         "TACO, ARCTIC and OakInk2, each trained and evaluated on its own. Scripts: `scripts/research/hier_contact_gen/` (repo). "
         "Per-dataset experiment directories: " + ", ".join(f"`{ROOTS[d].relative_to('/result/uhnam/dexcore')}`" for d in D) + ".", "",
         "## Verdict", "", narr("verdict"),
         "## 1. Implementation summary", "", narr("implementation"),
         "## 2. Exact configuration per dataset", "", config_table(D), "",
         "Model sizes, training (AdamW 3e-4, EMA 0.999, early stopping on the validation criterion; `results/training_summary.csv`). The 'best validation value' is the v-prediction MSE for the samplers, the one-step L2 for `vf` (the discarded one-step recipe) and the free-rollout E_C in standardised units for `vf_unroll8` (the VF of every table) and the ARCTIC sweep rows, so the values are not comparable across rows:", "", training_table(D), "",
         "## 3. Main tables", ""]
    for ds in D:
        L += [main_table(ds, D[ds], "fixed"), "", b4_tables(ds), "", "<details><summary>extended rows</summary>", "", main_table(ds, D[ds], "fixed", extended=True), "", ref_table(ds), "", "</details>", ""]
    L += ["## 4. Q1–Q6 quantities (test set; paired per-sequence differences from `results/paired_differences.csv`)", ""]
    for ds in D:
        L += [f"### {LABEL[ds]}", "", q_table(ds, D[ds]), ""]
        bo = ROOTS[ds] / "results" / "by_orig_split.csv"
        if bo.exists():
            B = pd.read_csv(bo); B = B[B.model.isin(["static_gt", "gtinit_vf", "samplerG_vf", "samplerGT_vf"]) & B.K.isin([0, 1, 10])]
            B["subset"] = B.orig_split + np.where(B.group_seen_in_train, "", " (groups absent from train)")
            piv = B.pivot_table(index=["model", "K"], columns="subset", values="E_C").round(3).reset_index()
            ns = B[B.model == "static_gt"].groupby("subset").n.sum()
            L += [f"E_C on the dataset's own labelled test subsets, same models (`results/by_orig_split.csv`; test_1 is the main test set; n: " + ", ".join(f"{k} {int(v)}" for k, v in ns.items()) + "):", "",
                  "| " + " | ".join(map(str, piv.columns)) + " |", "|" + "---|" * len(piv.columns)] + ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in piv.values.tolist()] + [""]
    L += ["### Q7. Cross-dataset summary (test sets)", "", cross_table(D), ""]
    L += ["## 5. Figures", ""]
    for ds in D:
        figs = sorted((ROOTS[ds] / "figures").glob("*.png")) if (ROOTS[ds] / "figures").exists() else []
        L += [f"**{LABEL[ds]}** (`{ROOTS[ds].relative_to('/result/uhnam/dexcore')}/figures/`): " + ", ".join(f.name for f in figs), ""]
    L += ["## 6. Interpretation (Q1–Q7)", ""]
    for ds in main_ds:
        L += [f"### {LABEL[ds]}", "", narr(ds), ""]
    L += ["### B4 — the existing dense predictor (BimArt's contact stage)", "", narr("bimart"), ""]
    L += ["## 7. Failures and instabilities encountered", "", narr("failures"), "## 8. Deviations from the specification", "", narr("deviations"),
          "## 9. File index", "", narr("files")]
    (REP / "hier_contact_gen_report.md").write_text("\n".join(L))
    print("written", REP / "hier_contact_gen_report.md", "datasets:", list(D))


if __name__ == "__main__":
    main()
