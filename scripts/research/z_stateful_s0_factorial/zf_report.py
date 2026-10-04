#!/usr/bin/env python
"""Assemble report.md from the result tables and the narrative files.    python zf_report.py
Every number of the prose is injected from the result tables: the narrative files (OUT/narrative/*.md, mirrored in narrative_templates/)
use <<key:format>> placeholders resolved from report_values.json (written here from the CSVs) and {{tableN}} / {{figN}} blocks rendered here.
A placeholder without a value is printed as ??key?? and listed at the end.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import zf_common as Z

S2, ZJ = Z.S2, Z.ZJ
NARR = Z.OUT / "narrative"
SECTIONS = [("research_question", "Research question"), ("background", "Background"), ("hypotheses", "Remaining hypotheses"), ("taco_arctic_prior", "Why TACO and ARCTIC may differ"), ("design", "2×2 experimental design"),
            ("architecture", "Model architecture"), ("training", "Training"), ("taco", "TACO results"), ("arctic", "ARCTIC results"), ("stateful", "Stateful z effect"), ("decoder", "s_0-preserving decoder effect"),
            ("interaction", "Interaction"), ("generalisation", "z generalization analysis"), ("accumulation", "Error accumulation"), ("early", "Early-frame analysis"), ("direct", "Best z model vs direct D0"),
            ("qualitative", "Qualitative examples"), ("failures", "Failure cases"), ("limitations", "Limitations"), ("interpretation", "Final interpretation"), ("next_step", "Recommended next step")]
ROWS = ["M00", "M01", "M10", "M11", "D0", "D0r", "PERSIST", "GT"]
ROW_LAB = {"M00": "M00 whole z + absolute (reused)", "M01": "M01 whole z + s₀-residual", "M10": "M10 stateful z + absolute", "M11": "M11 stateful z + s₀-residual", "D0": "**D0 direct dense (reference)**",
           "D0r": "D0r direct, 100 k protocol, stopped early (partial)", "PERSIST": "persistence C_t = s_0", "GT": "GT maps (extraction floor)"}
MET_LAB = {"E_C": "dense E_C", "E_C_last16": "E_C last 16", "E_C_final": "E_C final frame", "part_hamming": "participation", "amount_l1": "amount", "centroid": "centroid", "normal": "normal (°)", "q_rel_l1": "wrench rel. L1",
           "q_cos": "wrench cos ↑", "jitter_dev_dense": "|Δ change| dense", "jitter_dev_r2": "|Δ change| R2", "L_z": "test L_z"}
key = lambda *p: re.sub(r"[^A-Za-z0-9]+", "_", "_".join(str(x) for x in p)).strip("_")
BINS = [f"E_C_h{lo}_{hi}" for lo, hi in Z.HORIZON_BINS]


def ci(v, lo, hi, d=3):
    return "—" if not np.isfinite(v) else (f"{v:.{d}f}" if not np.isfinite(lo) else f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]")


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load(n):
    p = Z.OUT / n
    try:
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def has(df, **kw):
    if df.empty:
        return df
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m]


def pcell(P, ds, a, b, metric, d=3):
    r = has(P, dataset=ds, a=a, b=b, metric=metric)
    if r.empty:
        return "—"
    r = r.iloc[0]
    return f"{r['diff']:+.{d}f} [{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}] ({r.rel_improvement * 100:+.1f} %) **{r.verdict}**"


def values(A, P, FX, LAT, EF, DC, D, T, SAN):
    V = {}
    for ds, a in A.items():
        for r in a.itertuples():
            V[key("m", ds, r.model, r.metric)] = r.value; V[key("m", ds, r.model, r.metric, "lo")] = r.ci_lo; V[key("m", ds, r.model, r.metric, "hi")] = r.ci_hi
    for r in P.itertuples():
        b = key("p", r.dataset, r.a, "vs", r.b, r.metric)
        V.update({b + "_diff": r.diff, b + "_lo": r.ci_lo, b + "_hi": r.ci_hi, b + "_rel": r.rel_improvement, b + "_relchange": r.rel_change, b + "_verdict": r.verdict, b + "_a": r.value_a, b + "_b": r.value_b})
    for r in FX.itertuples():
        b = key("fx", r.dataset, r.metric, r.effect)
        V.update({b: r.value, b + "_lo": r.ci_lo, b + "_hi": r.ci_hi, b + "_rel": r.rel_to_M00, b + "_verdict": r.verdict})
    for r in LAT.to_dict("records"):
        for k, v in r.items():
            if k not in ("dataset", "model"):
                V[key("lat", r["dataset"], r["model"], k)] = v
    for r in EF.itertuples():
        b = key("ef", r.dataset, r.model, r.quantity, "t" + str(r.frame)); V[b] = r.value; V[b + "_lo"] = r.ci_lo; V[b + "_hi"] = r.ci_hi
    for r in DC.to_dict("records"):
        b = key("dc", r["dataset"], r["reference"], r["metric"])
        V.update({b + "_best": r["value_best_z"], b + "_ref": r["value_reference"], b + "_diff": r["diff"], b + "_lo": r["ci_lo"], b + "_hi": r["ci_hi"], b + "_rel": r["rel_improvement"], b + "_verdict": r["verdict"]})
        V[key("best", r["dataset"])] = r["best_z"]; V[key("val_E_C", r["dataset"])] = r["val_E_C"]
    qk = {"Does stateful z improve generalization?": "q1", "Does stateful z improve final contact?": "q2", "Does s_0-preserving decoding improve contact?": "q3", "Does s_0 preservation specifically fix early frames?": "q4",
          "Are the two changes complementary?": "q5", "Does the best z model beat direct D0?": "q6", "Which architecture is best supported?": "q7", "case": "case", "case D (stateful helps TACO only)?": "caseD",
          "extra seed needed (a decisive dense comparison ambiguous)?": "extra_seed"}
    for r in D.itertuples():
        V[key("dec", r.dataset, qk.get(r.question, r.question))] = r.answer
    for r in T.to_dict("records"):
        for k, v in r.items():
            if k not in ("dataset", "model"):
                V[key("t", r["dataset"], r["model"], k)] = v
    for ds, checks in SAN.items():
        for k, c in checks.items():
            V[key("san", ds, k)] = c["status"]
        for k in ("n_train", "n_val", "n_test", "n_test_takes"):
            if k in checks.get("06_identical_split_and_conditioning", {}):
                V[key("split", ds, k)] = checks["06_identical_split_and_conditioning"][k]
    a, b = V.get(key("t", "taco", "M00", "n_params_total")), V.get(key("t", "taco", "M10", "n_params_total"))
    if a and b:
        V["cap_gap"] = 1 - b / a
    V["san_n"] = sum(len(c) for c in SAN.values()); V["san_n_pass"] = sum(v["status"] == "pass" for c in SAN.values() for v in c.values())
    V["san_not_pass"] = "; ".join(f"{ds} {k} ({v['status']})" for ds, c in SAN.items() for k, v in c.items() if v["status"] != "pass") or "none"
    for ds in Z.DATASETS:
        qp = Z.OUT / "figures" / f"qualitative_selection_{ds}.csv"
        if qp.exists():
            for r in pd.read_csv(qp).to_dict("records"):
                for k, v in r.items():
                    if k not in ("dataset", "key"):
                        V[key("q", ds, r["key"], k)] = v
        dj = Z.ds_out(ds) / "d0r_partial.json"
        if dj.exists():
            for k, v in Z.read_json(dj).items():
                V[key("d0r", ds, k)] = v
    pj = Z.OUT / "init_aug_decision.json"
    if pj.exists():
        pd_ = Z.read_json(pj)
        for ds in Z.DATASETS:
            V[key("pilot", ds, "ratio")] = pd_["ratio_frame0_only_over_augmented_at_step_2000"][ds]
            for a in (0, 3):
                V[key("pilot", ds, f"aug{a}", "val_rmse")] = pd_["readings"][f"{ds}/aug{a}"][3]["val_rmse_z0"]
    pj = Z.OUT / "logs" / "pilot_shared_init" / "init_aug_decision_shared_init.json"      # the superseded first pilot (initial weights shared with the Stage-1 encoder)
    if pj.exists():
        pd_ = Z.read_json(pj)
        for ds in Z.DATASETS:
            V[key("pilot_shared", ds, "ratio")] = pd_["ratio_frame0_only_over_augmented_at_step_2000"][ds]
            for a in (0, 3):
                V[key("pilot_shared", ds, f"aug{a}", "val_rmse")] = pd_["readings"][f"{ds}/aug{a}"][3]["val_rmse_z0"]
    try:                                                                           # earlier studies, read from their own value files
        V.update({"zj_" + k: v for k, v in json.loads((ZJ.OUT / "report_values.json").read_text()).items() if isinstance(v, (int, float, str))})
        V.update({"zt_" + k: v for k, v in json.loads((S2.OUT.parent / "z_temporal_diagnostic" / "report_values.json").read_text()).items() if isinstance(v, (int, float, str))})
    except Exception as e:                                                         # noqa: BLE001
        V["earlier_studies_error"] = str(e)
    return V


def table_models(MT):
    rows = []
    for r in MT.to_dict("records"):
        f = lambda k: " / ".join(f"{r.get(f'n_params_{ds}_{k}', float('nan')) / 1e6:.1f} M" for ds in Z.DATASETS)
        rows.append([r["model"], r["temporal"], r["decoder"], r["loss"], r["trained"], f("temporal"), f("decoder"), f("total")])
    return md_table(["Model", "Temporal part", "Contact decoding", "Loss", "Trained", "Temporal params (TACO / ARCTIC)", "Decoder / head params", "Total params"], rows)


def table_main(A, P, ds):
    mets = ["E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos"]; a = A[ds]; rows = []
    for m in ROWS:
        if has(a, model=m, metric="E_C").empty:
            continue
        rows.append([ROW_LAB[m]] + [(lambda r: ci(r.value, r.ci_lo, r.ci_hi, 2 if met == "normal" else 3))(has(a, model=m, metric=met).iloc[0]) for met in mets])
    for m in ("M00", "M01", "M10", "M11"):
        if not has(P, dataset=ds, a=m, b="D0", metric="E_C").empty:
            rows.append([f"*{m} vs D0 (relative, verdict)*"] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {r.verdict}")(has(P, dataset=ds, a=m, b="D0", metric=met).iloc[0]) for met in mets])
    return md_table(["Model"] + [MET_LAB[m] for m in mets], rows)


def table_pairs(P, pairs, title_metrics=("E_C", "E_C_h1_16", "E_C_h17_32", "E_C_h33_48", "E_C_h49_63", "E_C_final", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "jitter_dev_dense", "L_z")):
    lab = {**MET_LAB, **{f"E_C_h{lo}_{hi}": f"E_C frames {lo}–{hi}" for lo, hi in Z.HORIZON_BINS}}; rows = []
    for ds in Z.DATASETS:
        for met in title_metrics:
            cells = [pcell(P, ds, a, b, met, 2 if met == "normal" else 3) for a, b in pairs]
            if any(c != "—" for c in cells):
                rows.append([Z.LABEL[ds], lab.get(met, met)] + cells)
    return md_table(["Dataset", "Metric"] + [f"{a} − {b}" for a, b in pairs], rows)


def table_latent(LAT, P):
    rows = []
    for r in LAT.itertuples():
        z0 = f"{r.z0_rmse_train:.3f} / {r.z0_rmse_val:.3f} / {r.z0_rmse_test:.3f}" if hasattr(r, "z0_rmse_test") and np.isfinite(getattr(r, "z0_rmse_test", np.nan)) else "—"
        rows.append([Z.LABEL[r.dataset], r.model, f"{r.rmse_train:.3f}", f"{r.rmse_val:.3f}", f"{r.rmse_test:.3f}", f"{r.gap:+.3f} [{r.gap_lo:+.3f}, {r.gap_hi:+.3f}]", f"{r.test_over_train_rmse:.1f}×", ci(r.L_z_test, r.L_z_lo, r.L_z_hi),
                     f"{r.r2_pooled:.3f}", z0, " / ".join(f"{getattr(r, f'rmse_h{lo}_{hi}'):.2f}" for lo, hi in Z.HORIZON_BINS)])
    bins = " / ".join(f"{lo}–{hi}" for lo, hi in Z.HORIZON_BINS)
    return md_table(["Dataset", "Model", "z RMSE train", "z RMSE validation", "z RMSE test", "Generalisation gap (test − train) [CI of the test part]", "test / train", "Test L_z (MSE)", "Explained variance (R², pooled)",
                     "ẑ₀ RMSE train / val / test", f"Test RMSE by horizon {bins}"], rows)


def table_horizon(A, P):
    mets = BINS + ["E_C_final"]; rows = []
    for ds in Z.DATASETS:
        a = A[ds]
        for m in ["M00", "M01", "M10", "M11", "D0", "PERSIST"]:
            if has(a, model=m, metric="E_C").empty:
                continue
            rows.append([Z.LABEL[ds], ROW_LAB[m]] + [(lambda r: ci(r.value, r.ci_lo, r.ci_hi, 2))(has(a, model=m, metric=met).iloc[0]) for met in mets])
        for m in ("M00", "M01", "M10", "M11"):
            if not has(P, dataset=ds, a=m, b="D0", metric="E_C").empty:
                rows.append([Z.LABEL[ds], f"*{m} vs D0*"] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {r.verdict}")(has(P, dataset=ds, a=m, b="D0", metric=met).iloc[0]) for met in mets])
    return md_table(["Dataset", "Model"] + [f"frames {lo}–{hi}" for lo, hi in Z.HORIZON_BINS] + ["final frame (63)"], rows)


def table_early(EF, P):
    rows = []
    for ds in Z.DATASETS:
        e = has(EF, dataset=ds)
        for m in ["M00", "M01", "M10", "M11", "D0"]:
            x = has(e, model=m, quantity="E_C")
            if x.empty:
                continue
            dr = has(e, model=m, quantity="drift_mean_ratio")
            rows.append([Z.LABEL[ds], ROW_LAB[m]] + [(lambda r: ci(r.value, r.ci_lo, r.ci_hi, 2))(has(x, frame=t).iloc[0]) for t in Z.EARLY_FRAMES] + [" / ".join(f"{has(dr, frame=t).iloc[0].value:.2f}" for t in Z.EARLY_FRAMES) if len(dr) else "—"])
        for a, b in (("M01", "M00"), ("M11", "M10"), ("M01", "D0"), ("M11", "D0")):
            if not has(P, dataset=ds, a=a, b=b, metric="E_C_t1").empty:
                rows.append([Z.LABEL[ds], f"*{a} vs {b}*"] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {r.verdict}")(has(P, dataset=ds, a=a, b=b, metric=f"E_C_t{t}").iloc[0]) for t in Z.EARLY_FRAMES] + [""])
    return md_table(["Dataset", "Model"] + [f"E_C at t = {t}" for t in Z.EARLY_FRAMES] + ["movement from s₀ relative to the data at t = 1 / 4 / 8 / 16"], rows)


def table_direct(DC):
    rows = []
    for r in DC[DC.reference == "D0"].itertuples() if len(DC) else []:
        rows.append([Z.LABEL[r.dataset], r.best_z, MET_LAB.get(r.metric, r.metric.replace("E_C_h", "E_C frames ").replace("_", "–", 1) if r.metric.startswith("E_C_h") else r.metric), f"{r.value_best_z:.3f}", f"{r.value_reference:.3f}",
                     f"{r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", f"{r.rel_improvement * 100:+.1f} %", f"**{r.verdict}**"])
    return md_table(["Dataset", "Best z model (by validation E_C)", "Metric", "Best z", "D0", "Difference [95 % CI]", "Relative (+ = z better)", "Verdict"], rows)


def table_decision(D):
    qs = ["Does stateful z improve generalization?", "Does stateful z improve final contact?", "Does s_0-preserving decoding improve contact?", "Does s_0 preservation specifically fix early frames?",
          "Are the two changes complementary?", "Does the best z model beat direct D0?", "Which architecture is best supported?"]
    rows = []
    for q in qs:
        cells, ev = [], []
        for ds in Z.DATASETS:
            r = has(D, dataset=ds, question=q); cells.append(f"**{r.iloc[0].answer}**" if len(r) else "—"); ev.append(f"{Z.LABEL[ds]}: {r.iloc[0].evidence}" if len(r) else "")
        rows.append([q] + cells + ["<br>".join(e for e in ev if e)])
    c = [has(D, dataset=ds, question="case") for ds in Z.DATASETS]
    rows.append(["**Result case (Section 23 of the plan)**"] + [f"**Case {x.iloc[0].answer}**" if len(x) else "—" for x in c] + ["<br>".join(f"{Z.LABEL[ds]}: {x.iloc[0].evidence}" for ds, x in zip(Z.DATASETS, c) if len(x))])
    return md_table(["Question", "TACO", "ARCTIC", "Evidence"], rows)


def table_effects(FX):
    mets = ["E_C"] + BINS + ["E_C_final", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "jitter_dev_dense", "L_z"]; rows = []
    lab = {**MET_LAB, **{f"E_C_h{lo}_{hi}": f"E_C frames {lo}–{hi}" for lo, hi in Z.HORIZON_BINS}}
    for ds in Z.DATASETS:
        for met in mets:
            cells = []
            for eff in ("stateful", "decoder", "interaction"):
                r = has(FX, dataset=ds, metric=met, effect=eff)
                cells.append("—" if r.empty else (lambda r: f"{r.value:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.rel_to_M00 * 100:+.1f} %) **{r.verdict}**")(r.iloc[0]))
            if any(c != "—" for c in cells):
                rows.append([Z.LABEL[ds], lab.get(met, met)] + cells)
    return md_table(["Dataset", "Metric", "Stateful main effect: mean(M10 − M00, M11 − M01)", "Decoder main effect: mean(M01 − M00, M11 − M10)", "Interaction: (M11 − M10) − (M01 − M00)"], rows)


def table_training(T):
    rows = []
    for r in T.itertuples():
        f = lambda v, d=4: "—" if not np.isfinite(v) else f"{v:.{d}f}"
        rows.append([Z.LABEL[r.dataset], r.model, r.source, r.steps, r.stopped_by, r.best_step, r.best_step_objective, f(r.val_L_C_at_best), f(r.val_L_z_at_best, 3), f(r.val_L_z0_at_best, 3), f(r.val_E_C_at_best, 3),
                     f(r.val_E_C_teacher_z_at_best, 3), (f"{r.grad_norm_median:.2f} ({r.grad_norm_share_clipped * 100:.0f} %)" if np.isfinite(getattr(r, "grad_norm_median", np.nan)) else "—"),
                     f"{r.hours:.1f}", f"{r.s_per_step:.2f}", r.gpus])
    return md_table(["Dataset", "Model", "Source", "Steps run", "Stopped by", "Selected step", "Step of min. objective", "val L_C at selected", "val L_z", "val L_z0", "val E_C", "val E_C, teacher z*",
                     "Gradient norm before clipping, median after warm-up (share of logged steps above the clip of 1.0)", "Hours", "s / step", "GPU"], rows)


def table_statecheck(LAT):
    rows = []
    for r in LAT.itertuples():
        if not (hasattr(r, "rmse_from_true_h1") and np.isfinite(getattr(r, "rmse_from_true_h1", np.nan))):
            continue
        rows.append([Z.LABEL[r.dataset], r.model] + [f"{getattr(r, f'rmse_from_true_h{h}'):.3f} vs {getattr(r, f'rmse_hold_h{h}'):.3f} ({getattr(r, f'gain_over_hold_h{h}') * 100:+.1f} %)" for h in (1, 4, 8)]
                    + [f"{r.z0_rmse_test:.3f}", f"{r.rmse_test:.3f}", f"{r.rmse_true_z0_rollout:.3f}", f"{r.E_C_test:.3f}", f"{r.E_C_true_z0_rollout:.3f}", f"{r.E_C_teacher_z:.3f}"])
    return md_table(["Dataset", "Model", "h = 1: transition from true z*_t vs holding z*_t (gain)", "h = 4", "h = 8", "ẑ₀ RMSE (test)", "63-step rollout RMSE, own ẑ₀", "63-step rollout RMSE, true z*₀",
                     "E_C, own rollout", "E_C, rollout from true z*₀", "E_C, teacher latents through the decoder"], rows)


def table_zuse(LAT):
    rows = []
    for r in LAT.itertuples():
        if not np.isfinite(getattr(r, "E_C_z_other", np.nan)):
            continue
        c = lambda t: f"{getattr(r, f'E_C_z_{t}'):.3f} ({getattr(r, f'E_C_z_{t}_rel') * 100:+.1f} %; {getattr(r, f'E_C_z_{t}_minus_own'):+.3f} [{getattr(r, f'E_C_z_{t}_minus_own_lo'):+.3f}, {getattr(r, f'E_C_z_{t}_minus_own_hi'):+.3f}])"
        rows.append([Z.LABEL[r.dataset], r.model, f"{r.E_C_test:.3f}", c("other"), c("mean"), f"{r.E_C_z_other_rel_h1_16 * 100:+.1f} % / {r.E_C_z_other_rel_h49_63 * 100:+.1f} %", f"{r.E_C_teacher_z:.3f}"])
    return md_table(["Dataset", "Model", "E_C, the model's own ẑ", "E_C, ẑ of another test sequence (change; difference [CI])", "E_C, mean latent (change; difference [CI])",
                     "Change with another sequence's ẑ, frames 1–16 / 49–63", "E_C, teacher latents"], rows)


def table_sanity(SAN):
    names = sorted({k for c in SAN.values() for k in c}); rows = []
    for k in names:
        rows.append([k.replace("_", " ")] + [SAN[ds][k]["status"] if k in SAN.get(ds, {}) else "—" for ds in Z.DATASETS] + [next((SAN[ds][k]["how"] for ds in Z.DATASETS if k in SAN.get(ds, {})), "")])
    return md_table(["Check", "TACO", "ARCTIC", "How it was checked"], rows)


def table_ckpt():
    p = Z.OUT / "experiment_config.json"
    if not p.exists():
        return ""
    cfg = json.loads(p.read_text()); rows = []
    for k, r in cfg["runs"].items():
        ds, m = k.split("/"); rows.append([Z.LABEL[ds], m, f"`{r['ckpt']}`", f"`{r['md5']}`", f"{r['best_step']:,}", f"{r['steps']:,}", r["stopped_by"]])
    for ds, t in cfg["teacher"].items():
        rows.append([Z.LABEL[ds], "teacher (Stage-1 A3, frozen)", f"`{t['ckpt']}`", f"`{t['md5']}`", f"{t['best_step']:,}", "—", "—"])
    return md_table(["Dataset", "Model", "Checkpoint", "md5 (first 12)", "Selected step", "Steps run", "Stopped by"], rows)


def main():
    A = {ds: load(f"{ds}_metrics.csv") for ds in Z.DATASETS}
    P, FX, LAT, EF, DC, D, T, MT = (load(n) for n in ("paired_comparisons.csv", "factorial_effects.csv", "latent_metrics.csv", "early_frame_metrics.csv", "direct_comparison.csv", "decision_summary.csv", "training_summary.csv", "model_table.csv"))
    SAN = json.loads((Z.OUT / "sanity_summary.json").read_text()) if (Z.OUT / "sanity_summary.json").exists() else {}
    V = values(A, P, FX, LAT, EF, DC, D, T, SAN); Z.write_json(Z.OUT / "report_values.json", V)
    cap = lambda n, title, body: f"**Table {n} — {title}**\n\n{body}"
    tables = {"table1": cap(1, "Model definitions and parameter counts", table_models(MT)),
              "table2": cap(2, "Main TACO results (test; mean over sequences, 95 % take-cluster bootstrap)", table_main(A, P, "taco")) if len(A["taco"]) else "",
              "table3": cap(3, "Main ARCTIC results (test)", table_main(A, P, "arctic")) if len(A["arctic"]) else "",
              "table4": cap(4, "Decoder effect: s₀-preserving minus absolute decoding, paired (negative difference = lower error; relative in parentheses, + = the first model is better)", table_pairs(P, [("M01", "M00"), ("M11", "M10")])),
              "table5": cap(5, "Stateful effect: stateful minus whole-sequence z, paired", table_pairs(P, [("M10", "M00"), ("M11", "M01")])),
              "table6": cap(6, "Latent generalisation: the same weights on training, validation and test sequences (standardised teacher coordinates, frames 1–63)", table_latent(LAT, P)) if len(LAT) else "",
              "table7": cap(7, "Horizon analysis: dense error by horizon bin and at the final frame (test)", table_horizon(A, P)),
              "table8": cap(8, "Early frames: dense error at t = 1, 4, 8, 16 and movement away from s₀", table_early(EF, P)) if len(EF) else "",
              "table9": cap(9, "Best z model against the direct model D0 (paired, test)", table_direct(DC)) if len(DC) else "",
              "table10": cap(10, "Final decision", table_decision(D)) if len(D) else "",
              "table11": cap(11, "Factorial contrasts on per-sequence values (95 % take-cluster bootstrap; relative to the M00 mean)", table_effects(FX)) if len(FX) else "",
              "table12": cap(12, "Training summary", table_training(T)) if len(T) else "",
              "table13": cap(13, "Stateful check: the transition from true latents, and the rollout from the true initial latent (analysis only)", table_statecheck(LAT)) if len(LAT) else "",
              "table14": cap(14, "Sanity checks", table_sanity(SAN)) if SAN else "",
              "table15": cap(15, "Does the decoder use the latent? The same decoder and the same s₀ and G, with other latents (test; analysis only)", table_zuse(LAT)) if len(LAT) and "E_C_z_other" in LAT else "",
              "ckpt_table": table_ckpt()}
    fig = lambda n, name, alt: f"![Figure {n} — {alt}](figures/{name})"
    figs = {"fig1": fig(1, "fig1_design.png", "2 × 2 design"), "fig2": fig(2, "fig2_main_2x2.png", "main results"), "fig3": fig(3, "fig3_contact_vs_horizon.png", "contact error vs horizon"),
            "fig4": fig(4, "fig4_z_vs_horizon.png", "z error vs horizon"), "fig5": fig(5, "fig5_generalisation_gap.png", "generalisation gap"), "fig6": fig(6, "fig6_early_frames.png", "early frames"),
            "fig9": fig(9, "fig9_training_curves.png", "training curves"), "fig10": fig(10, "fig10_transition_from_true_states.png", "transition from true states")}
    for ds, num in (("taco", 7), ("arctic", 8)):
        for k in ("stateful", "decoder", "failure"):
            figs[f"fig{num}_{k}"] = fig(num, f"fig{num}_{ds}_{k}.png", f"qualitative {Z.LABEL[ds]} {k}")
    missing = set()

    def fill(txt):
        for k, v in {**tables, **figs}.items():
            txt = txt.replace("{{" + k + "}}", v)

        def sub(m):
            k, fmt = m.group(1), m.group(2)
            if k not in V:
                missing.add(k); return f"??{k}??"
            try:
                return format(V[k], fmt) if fmt else str(V[k])
            except (ValueError, TypeError):
                missing.add(k + ":" + str(fmt)); return f"??{k}??"
        return re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", sub, txt)
    body = fill((NARR / "00_head.md").read_text()) + "\n" if (NARR / "00_head.md").exists() else "# z stateful × s_0 factorial\n\n" + tables["table10"] + "\n"
    for i, (sec, title) in enumerate(SECTIONS, 1):
        p = NARR / f"{i:02d}_{sec}.md"
        body += f"\n## {i}. {title}\n\n{fill(p.read_text()) if p.exists() else '_(section not written)_'}\n"
    for name, title in (("90_sanity.md", "Appendix A. Sanity checks"), ("91_questions.md", "Appendix B. The eight questions, answered"), ("92_reproduction.md", "Appendix C. Files, checkpoints and reproduction")):
        if (NARR / name).exists():
            body += f"\n## {title}\n\n{fill((NARR / name).read_text())}\n"
    (Z.OUT / "report.md").write_text(body)
    print("report.md:", len(body.splitlines()), "lines; unresolved blocks:", body.count("{{"), "; missing values:", sorted(missing))


if __name__ == "__main__":
    main()
