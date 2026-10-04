#!/usr/bin/env python
"""Assemble report.md from the result tables and the narrative files.    python zj_report.py
Every number of the prose is injected from the result tables: the narrative files (OUT/narrative/*.md, mirrored in
narrative_templates/) use <<key:format>> placeholders resolved from report_values.json (written here from the CSVs) and
{{tableN}} / {{figN}} blocks rendered here.  A placeholder without a value is printed as ??key?? and listed at the end.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import zj_common as Z

S2 = Z.S2
NARR = Z.OUT / "narrative"
SECTIONS = [("research_question", "Research question"), ("why_followup", "Why this follow-up was necessary"), ("stage1", "Evidence from Stage 1"), ("stage2", "Evidence from Stage 2"),
            ("diagnostic", "Evidence from the latest z temporal diagnostic"), ("architecture", "Model architecture"), ("losses", "Losses"), ("training", "Training setup"),
            ("m2_vs_m1", "M2 vs previous M1"), ("m2_vs_m0", "M2 vs direct M0"), ("z_prediction", "z prediction analysis"), ("decoder", "Decoder mismatch / robustness analysis"),
            ("horizon", "Horizon analysis"), ("taco_vs_arctic", "TACO vs ARCTIC"), ("qualitative", "Qualitative results"), ("failures", "Failure cases"), ("limitations", "Limitations"),
            ("interpretation", "Final interpretation"), ("next_step", "Recommended next step")]
MODELS = ["M0", "M1", "M2", "M3"]
EXTRA = ["M2_seltotal", "M3_seltotal", "M2_4blk", "M2_s1", "M0r"]
ROW_LAB = {"M0": "M0 direct dense (Stage-2 B0)", "M1": "M1 previous z → C (Stage-2 B1)", "M2": "**M2 joint, 6-block decoder**", "M3": "M3 dual-input decoder", "M2_seltotal": "M2, state selected on L_C + L_z",
           "M3_seltotal": "M3, state selected on its full objective", "M2_4blk": "M2 with the 4-block decoder (control)", "M2_s1": "M2, seed 1", "M0r": "M0 retrained (100 k protocol)", "PERSIST": "persistence C_t = s_0",
           "GT": "GT maps (extraction floor)"}
MET_LAB = {"E_C": "dense E_C", "E_C_last16": "E_C last 16", "part_hamming": "participation Hamming", "part_macro_f1": "participation macro F1 ↑", "amount_l1": "amount L1", "centroid": "centroid / l",
           "normal": "normal (°)", "q_rel_l1": "wrench rel. L1", "q_cos": "wrench cosine ↑", "jitter_dev_dense": "|Δ frame change| dense", "jitter_dev_r2": "|Δ frame change| R2", "L_z": "test L_z"}
INP_LAB = {"oracle": "A. teacher z* (oracle)", "pred": "B. predicted ẑ (own)", "noisy_gauss": "C. z* + matched Gaussian noise", "noisy_perm": "C′. z* + another sequence's error"}
MARK = {"better": "better", "worse": "worse", "similar": "similar"}
key = lambda *p: re.sub(r"[^A-Za-z0-9]+", "_", "_".join(str(x) for x in p)).strip("_")


def f(v, d=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


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


def values(A, P, T, ZM, DD, H, D, PR, SAN):
    """Flat {key: value} of everything the narrative may cite."""
    V = {}
    for r in A.itertuples():
        V[key("m", r.dataset, r.model, r.metric)] = r.value; V[key("m", r.dataset, r.model, r.metric, "lo")] = r.ci_lo; V[key("m", r.dataset, r.model, r.metric, "hi")] = r.ci_hi
    for r in P.itertuples():
        b = key("p", r.dataset, r.a, "vs", r.b, r.metric)
        V.update({b + "_diff": r.diff, b + "_lo": r.ci_lo, b + "_hi": r.ci_hi, b + "_rel": r.rel_improvement, b + "_relchange": r.rel_change, b + "_verdict": r.verdict, b + "_a": r.value_a, b + "_b": r.value_b})
    for r in T.to_dict("records"):
        for k, v in r.items():
            if k not in ("dataset", "model"):
                V[key("t", r["dataset"], r["model"], k)] = v
    for r in ZM.to_dict("records"):
        for k, v in r.items():
            if k not in ("dataset", "model"):
                V[key("z", r["dataset"], r["model"], k)] = v
    for r in DD.to_dict("records"):
        b = key("d" if r["split"] == "test" else "dtr", r["dataset"], r["decoder"], r["input"])
        V[b] = r["E_C"]; V[b + "_lo"] = r["ci_lo"]; V[b + "_hi"] = r["ci_hi"]
        for lo, hi in Z.HORIZON_BINS:
            V[b + f"_h{lo}_{hi}"] = r[f"h{lo}_{hi}"]
    for r in H.itertuples():
        V[key("h", r.dataset, r.model, r.curve, r.h_lo, r.h_hi)] = r.value
    qk = {"Does joint decoder adaptation improve over previous M1?": "q1", "Does M2 beat direct M0 in dense contact?": "q2", "Does M2 beat M0 in structural / wrench quality?": "q3", "Does M2 improve long-horizon behavior?": "q4",
          "Is z prediction itself improved?": "q5", "Is the decoder more robust to predicted-z error?": "q6", "Is z mediation justified for the final generator?": "q7", "case": "case",
          "extra seed needed (M2 vs M0 ambiguous)?": "extra_seed", "4-block control needed (M2 better than M1)?": "control_4blk"}
    for r in D.itertuples():
        V[key("dec", r.dataset, qk.get(r.question, r.question))] = r.answer
    for r in PR.to_dict("records"):
        for k, v in r.items():
            if k not in ("dataset", "model"):
                V[key("pr", r["dataset"], r["model"], k)] = v
    for ds, checks in SAN.items():
        V[key("san", ds, "n")] = len(checks); V[key("san", ds, "n_pass")] = sum(c["status"] == "pass" for c in checks.values())
        for k, c in checks.items():
            V[key("san", ds, k)] = c["status"]
    V["san_n"] = sum(len(c) for c in SAN.values()); V["san_n_pass"] = sum(v["status"] == "pass" for c in SAN.values() for v in c.values())
    V["san_not_pass"] = "; ".join(f"{ds} {k} ({v['status']})" for ds, c in SAN.items() for k, v in c.items() if v["status"] != "pass") or "none"
    # derived quantities the prose cites
    g = lambda k: V.get(k, float("nan"))
    for ds in Z.DATASETS:
        for m in ("M1", "M2", "M3"):
            if key("d", ds, m, "pred") not in V:
                continue
            pred, orc = g(key("d", ds, m, "pred")), g(key("d", ds, m, "oracle"))
            V[key("share", ds, m, "gauss")] = g(key("d", ds, m, "noisy_gauss")) / pred; V[key("share", ds, m, "perm")] = g(key("d", ds, m, "noisy_perm")) / pred
            V[key("gap", ds, m)] = pred - orc; V[key("gapshare", ds, m)] = (pred - orc) / pred
            V[key("zratio", ds, m)] = g(key("z", ds, m, "rmse_val")) / g(key("z", ds, m, "rmse_train"))
            V[key("ztest_over_train", ds, m)] = g(key("z", ds, m, "rmse_test")) / g(key("z", ds, m, "rmse_train"))
        if key("p", ds, "M2_seltotal", "vs", "M1", "E_C", "rel") in V:
            V[key("attr", ds, "capacity")] = g(key("p", ds, "M2_seltotal", "vs", "M1", "E_C", "rel")); V[key("attr", ds, "selection")] = -g(key("p", ds, "M2_seltotal", "vs", "M2", "E_C", "rel"))
        for m in ("M1", "M2", "M3"):
            for ref in ("M0",):
                a, b = key("h", ds, m, "dense", 1, 8), key("h", ds, ref, "dense", 1, 8)
                if a in V and b in V:
                    V[key("early_gap", ds, m)] = V[a] - V[b]
    pert = P[(P.question == "same perturbed teacher latent through both decoders") & P.a.astype(str).str.startswith("D_M2(") & P.b.astype(str).str.startswith("D_M1(")] if len(P) else P
    if len(pert):
        V["pert_min"], V["pert_max"] = float(pert.rel_improvement.min()), float(pert.rel_improvement.max())
        V["pert_n"], V["pert_n_better"] = int(len(pert)), int((pert.verdict == "better").sum())
    for ds, checks in SAN.items():
        for m, d in checks.get("03_decoder_updated", {}).get("change_vs_stage1_at_selected_state", {}).items():
            V[key("decchg", ds, m)] = d["rel_change"]; V[key("decextra", ds, m)] = d["extra_blocks_film_norm"]
    for ds, checks in SAN.items():
        for k in ("n_train", "n_val", "n_test", "n_test_takes"):
            if k in checks.get("07_identical_split", {}):
                V[key("split", ds, k)] = checks["07_identical_split"][k]
    for ds in Z.DATASETS:
        qp = Z.OUT / "figures" / f"fig6_selection_{ds}.csv"
        if qp.exists():
            for r in pd.read_csv(qp).to_dict("records"):
                for k, v in r.items():
                    if k not in ("dataset", "key"):
                        V[key("q", ds, r["key"], k)] = v
    for ds in Z.DATASETS:                                                          # training-batch losses of the runs trained here at a few steps
        for m in Z.TRAINED_HERE:
            tp = Z.train_log_path(ds, Z.run_name(m) + "_train")
            if tp.exists():
                tl = pd.read_csv(tp)
                for st in (500, 1000, 2000, 3000, 4000):
                    r = tl[tl.step == st]
                    if len(r) and "L_z" in tl:
                        V[key("trainLz", ds, m, st)] = float(r.L_z.iloc[0])
    V.update(generated_text(V, A, P, T))
    # earlier studies (read from their own tables)
    try:
        s1 = S2.CF.OUT
        rec = pd.read_csv(s1 / "reconstruction_metrics.csv"); dsum = pd.read_csv(s1 / "decision_summary.csv")
        for r in rec.itertuples():
            if r.model in ("A3", "pca64"):
                V[key("s1", r.dataset, r.model, "E_zonly")] = r.E_zonly; V[key("s1", r.dataset, r.model, "E_full")] = r.E_full
        for r in dsum[dsum.model == "A3"].itertuples():
            m = re.search(r"keeps (\d+) % of", r.evidence)
            if m:
                V[key("s1", r.dataset, "keep_pct")] = int(m.group(1))
            m = re.search(r"structure follows the z donor in (\d+) %", r.evidence)
            if m:
                V[key("s1", r.dataset, "swap_pct")] = int(m.group(1))
        b = pd.read_csv(S2.OUT / "bottleneck_metrics.csv")
        for r in b.to_dict("records"):
            for k in ("E_C_pred", "E_C_oracle_z", "stage1_E_zonly", "stage1_E_full", "pred_h1_8", "share_of_error_from_bottleneck"):
                V[key("s2b", r["dataset"], r["model"], k)] = r[k]
        zt = json.loads((S2.OUT.parent / "z_temporal_diagnostic" / "report_values.json").read_text())
        V.update({"zt_" + k: v for k, v in zt.items() if isinstance(v, (int, float, str))})
    except Exception as e:                                                         # noqa: BLE001
        V["earlier_studies_error"] = str(e)
    return V


def generated_text(V, A, P, T):
    """Paragraphs about the optional M3 and the budget-parity M0r: written from the tables when the runs exist, a status line otherwise."""
    L = Z.LABEL; out = {}
    have = lambda ds, m: key("m", ds, m, "E_C") in V
    sp = Z.OUT / "logs" / "stopped_runs.json"; STOP = json.loads(sp.read_text()) if sp.exists() else {}
    def stopped(ds, m, ref, ref_lab):
        r = STOP.get(f"{ds}/{m}")
        if r is None:
            return None
        rv = V.get(key("t", ds, ref, "val_L_C_at_best"), float("nan"))
        return (f"{L[ds]}: the {m} run was stopped at the user's request after {r['last_validated_step']:,} validated steps, before it finished; no test evaluation was made. "
                f"Its best validation dense loss was {r['best_val_L_C']:.4f} (step {r['best_step']:,}), against {rv:.4f} for {ref_lab}.")
    def pair(ds, a, b, metric="E_C"):
        k = key("p", ds, a, "vs", b, metric)
        return f"{V[k + '_diff']:+.3f} [{V[k + '_lo']:+.3f}, {V[k + '_hi']:+.3f}], {V[k + '_rel'] * 100:+.1f} %, **{V[k + '_verdict']}**"
    def struct(ds, a, b):
        vs = [(m, V[key("p", ds, a, "vs", b, m) + "_verdict"]) for m in Z.STRUCT_DECISION]
        return f"{sum(v == 'better' for _, v in vs)} better / {sum(v == 'worse' for _, v in vs)} worse of 5"
    # M3
    a, b = [], []
    for ds in Z.DATASETS:
        if have(ds, "M3"):
            r = T[(T.dataset == ds) & (T.model == "M3")].iloc[0]
            a.append(f"{L[ds]}: dense error {V[key('m', ds, 'M3', 'E_C')]:.3f} (selected step {int(r.best_step):,} of {int(r.steps):,}, stopped by {r.stopped_by}); against M2 {pair(ds, 'M3', 'M2')}; against M1 {pair(ds, 'M3', 'M1')}; "
                     f"test L_z {V.get(key('z', ds, 'M3', 'L_z_test'), float('nan')):.3f}; teacher latent through its decoder {V.get(key('d', ds, 'M3', 'oracle'), float('nan')):.3f} (M2: {V.get(key('d', ds, 'M2', 'oracle'), float('nan')):.3f}).")
            b.append(f"{L[ds]}: against M0 {pair(ds, 'M3', 'M0')}; structure and wrench {struct(ds, 'M3', 'M0')}.")
        else:
            a.append(stopped(ds, "M3", "M2", "M2") or f"{L[ds]}: the M3 run was still training when this report was built."); b.append(f"{L[ds]}: no test result for M3.")
    out["m3_vs_m2_text"] = ("**M3, the optional dual-input decoder.** The same model with the decoder also fed the teacher latent (weight 0.25, on one of the four sampled frames).\n\n" + "\n".join(f"- {x}" for x in a))
    out["m3_vs_m0_text"] = "**M3 against M0.**\n\n" + "\n".join(f"- {x}" for x in b)
    # M0r
    c = []
    for ds in Z.DATASETS:
        if have(ds, "M0r"):
            r = T[(T.dataset == ds) & (T.model == "M0r")].iloc[0]
            c.append(f"{L[ds]}: the retrained M0 reaches {V[key('m', ds, 'M0r', 'E_C')]:.3f} (selected step {int(r.best_step):,} of {int(r.steps):,}, stopped by {r.stopped_by}); against the reused M0 {pair(ds, 'M0r', 'M0')}. "
                     f"M2 against the retrained M0: dense {pair(ds, 'M2', 'M0r')}; structure and wrench {struct(ds, 'M2', 'M0r')}.")
        else:
            c.append((stopped(ds, "M0r", "M0", "the reused M0 at its selected step") or f"{L[ds]}: the retraining was still running when this report was built.") + " The verdicts above use the reused M0.")
    out["m0r_text"] = ("**Budget parity.** The reused M0 had an 80 000-step schedule. As a check, the same model was retrained under this study's protocol (100 000-step schedule, no stop before 25 000 steps).\n\n" + "\n".join(f"- {x}" for x in c))
    ko = []
    for ds in Z.DATASETS:
        if have(ds, "M3"):
            k2, k0 = key("p", ds, "M3", "vs", "M2", "E_C"), key("p", ds, "M3", "vs", "M0", "E_C")
            ko.append(f"{L[ds]} M3(선택, 이중 입력 디코더): E_C {V[key('m', ds, 'M3', 'E_C')]:.3f}, M2 대비 {V[k2 + '_rel'] * 100:+.1f} % ({V[k2 + '_verdict']}), M0 대비 {V[k0 + '_rel'] * 100:+.1f} % ({V[k0 + '_verdict']})")
        elif f"{ds}/M3" in STOP:
            r = STOP[f"{ds}/M3"]; ko.append(f"{L[ds]} M3(선택)는 사용자 요청으로 {r['last_validated_step']:,}스텝에서 중단했고 테스트 평가는 하지 않았습니다(검증 밀집 손실 최저 {r['best_val_L_C']:.4f}, M2는 {V.get(key('t', ds, 'M2', 'val_L_C_at_best'), float('nan')):.4f})")
        else:
            ko.append(f"{L[ds]} M3(선택)는 이 페이지를 만들 때 아직 학습 중이었습니다")
        if have(ds, "M0r"):
            k0, k2 = key("p", ds, "M0r", "vs", "M0", "E_C"), key("p", ds, "M2", "vs", "M0r", "E_C")
            ko.append(f"{L[ds]} M0 재학습(100,000스텝 일정): E_C {V[key('m', ds, 'M0r', 'E_C')]:.3f}, 기존 M0 대비 {V[k0 + '_rel'] * 100:+.1f} % ({V[k0 + '_verdict']}), M2의 재학습 M0 대비 {V[k2 + '_rel'] * 100:+.1f} % ({V[k2 + '_verdict']})")
        elif f"{ds}/M0r" in STOP:
            r = STOP[f"{ds}/M0r"]; ko.append(f"{L[ds]} M0 재학습(예산 동등성 점검)도 {r['last_validated_step']:,}스텝에서 중단했습니다(검증 밀집 손실 최저 {r['best_val_L_C']:.4f}, 기존 M0는 {V.get(key('t', ds, 'M0', 'val_L_C_at_best'), float('nan')):.4f})")
        else:
            ko.append(f"{L[ds]} M0 재학습(예산 동등성 점검)은 아직 학습 중이었습니다")
    out["ko_optional_status"] = "선택 실행과 점검 실행. " + ". ".join(ko) + ". 판정 단어는 better(더 좋음), similar(비슷함), worse(더 나쁨)입니다."
    done = [ds for ds in Z.DATASETS if have(ds, "M0r")]
    out["m0r_limit_text"] = (("The retraining of M0 under the 100 000-step protocol (Section 10) covers this for " + " and ".join(L[d] for d in done) + ".") if done else
                             ("A retraining of M0 under the 100 000-step protocol was stopped at the user's request before it finished (Section 10); at 25 000 steps its validation loss on ARCTIC was already below the reused M0's best, "
                              "so the reused M0 may understate the direct model there." if STOP else "A retraining of M0 under the 100 000-step protocol was still running when this report was built."))
    return out


def paired_cell(P, ds, a, b, metric, d=3):
    r = has(P, dataset=ds, a=a, b=b, metric=metric)
    if r.empty:
        return "—"
    r = r.iloc[0]
    return f"{r['diff']:+.{d}f} [{r.ci_lo:+.{d}f}, {r.ci_hi:+.{d}f}] ({r.rel_improvement * 100:+.2f} %) **{MARK[r.verdict]}**"


def table_models(MT, T):
    rows = []
    for r in MT.to_dict("records"):
        tot = " / ".join(f"{r.get(f'n_params_{ds}_total', np.nan) / 1e6:.1f} M" for ds in Z.DATASETS); dec = " / ".join(f"{r.get(f'n_params_{ds}_decoder_or_head', np.nan) / 1e6:.2f} M" for ds in Z.DATASETS)
        rows.append([r["model"], r["trained"], r["intermediate"], r["decoder"], r["loss"], r["selection"], f"{r.get('n_params_taco_backbone', np.nan) / 1e6:.1f} M", dec, tot])
    return md_table(["Model", "Trained", "Intermediate", "Decoder", "Loss", "Checkpoint selection", "Backbone", "Decoder / head (TACO / ARCTIC)", "Total (TACO / ARCTIC)"], rows)


def table_training(T):
    rows = []
    for r in T.itertuples():
        rows.append([Z.LABEL[r.dataset], r.model, f"{r.max_steps // 1000} k / {r.min_steps // 1000} k / {r.patience}", r.steps, r.stopped_by, r.best_step, r.best_step_objective, f(r.val_L_C_at_best, 4), f(r.min_val_L_C, 4) + f" ({r.min_val_L_C_step})",
                     f(r.val_L_z_at_best, 3), f(r.train_L_z_before_best, 4), f(r.val_E_C_at_best, 3), f(r.val_E_C_teacher_z_at_best, 3), f(r.decoder_rel_change_at_best, 3), f"{r.hours:.1f}", f"{r.s_per_step:.2f}", r.gpus])
    return md_table(["Dataset", "Model", "Budget max / min / patience", "Steps run", "Stopped by", "Selected step", "Step of min. objective", "val L_C at selected", "min val L_C (step)", "val L_z at selected", "train L_z (1 k steps before)",
                     "val E_C at selected", "val E_C, teacher z*", "Decoder change vs its initialisation (all blocks, EMA weights)", "Hours", "s / step", "GPU"], rows)


def table_main(A, P):
    rows = []
    for ds in Z.DATASETS:
        for m in MODELS + EXTRA + ["PERSIST"]:
            e = has(A, dataset=ds, model=m, metric="E_C")
            if e.empty:
                continue
            e = e.iloc[0]; l = has(A, dataset=ds, model=m, metric="E_C_last16").iloc[0]
            rows.append([Z.LABEL[ds], ROW_LAB[m], ci(e.value, e.ci_lo, e.ci_hi), ci(l.value, l.ci_lo, l.ci_hi), paired_cell(P, ds, m, "M1", "E_C") if m not in ("M1", "M0") else "—", paired_cell(P, ds, m, "M0", "E_C") if m != "M0" else "—"])
    return md_table(["Dataset", "Model", "Dense E_C (frames 1–63)", "E_C, last 16 frames", "E_C difference vs M1 (paired)", "E_C difference vs M0 (paired)"], rows)


def table_struct(A, P):
    mets = ["part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos"]; rows = []
    for ds in Z.DATASETS:
        for m in MODELS + ["M0r", "PERSIST", "GT"]:
            if has(A, dataset=ds, model=m, metric="E_C").empty:
                continue
            rows.append([Z.LABEL[ds], ROW_LAB[m]] + [(lambda r: ci(r.value, r.ci_lo, r.ci_hi, 2 if met == "normal" else 3))(has(A, dataset=ds, model=m, metric=met).iloc[0]) for met in mets])
        for a, b in (("M2", "M1"), ("M2", "M0"), ("M3", "M0")):
            if has(P, dataset=ds, a=a, b=b, metric="E_C").empty:
                continue
            rows.append([Z.LABEL[ds], f"*{a} vs {b} (relative, verdict)*"] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {MARK[r.verdict]}")(has(P, dataset=ds, a=a, b=b, metric=met).iloc[0]) for met in mets])
    return md_table(["Dataset", "Model"] + [MET_LAB[m] for m in mets], rows)


def table_stability(A, P):
    mets = ["jitter_dense", "jitter_r2", "jitter_dev_dense", "jitter_dev_r2", "jitter_ratio_dense"]; lab = ["dense frame-to-frame change", "R2 frame-to-frame change", "|change − GT change| dense", "|change − GT change| R2", "dense change / GT change"]
    rows = []
    for ds in Z.DATASETS:
        for m in MODELS + ["GT"]:
            if has(A, dataset=ds, model=m, metric="jitter_dense").empty:
                continue
            rows.append([Z.LABEL[ds], ROW_LAB[m]] + [(lambda r: ci(r.value, r.ci_lo, r.ci_hi) if len(r) else "—")(has(A, dataset=ds, model=m, metric=met)) if False else
                                                     (ci(*has(A, dataset=ds, model=m, metric=met).iloc[0][["value", "ci_lo", "ci_hi"]]) if len(has(A, dataset=ds, model=m, metric=met)) else "—") for met in mets])
        for a, b in (("M2", "M1"), ("M2", "M0")):
            if has(P, dataset=ds, a=a, b=b, metric="jitter_dev_dense").empty:
                continue
            rows.append([Z.LABEL[ds], f"*{a} vs {b}*", "", ""] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {MARK[r.verdict]}")(has(P, dataset=ds, a=a, b=b, metric=met).iloc[0]) for met in ("jitter_dev_dense", "jitter_dev_r2")] + [""])
    return md_table(["Dataset", "Model"] + lab, rows)


def table_z(ZM, P):
    rows = []
    for ds in Z.DATASETS:
        for r in has(ZM, dataset=ds).itertuples():
            rows.append([Z.LABEL[ds], r.model, ci(r.L_z_test, r.L_z_lo, r.L_z_hi), f(r.rmse_test), f(r.r2_pooled), f"{r.r2_per_dim_mean:.3f} ({r.r2_per_dim_min:.2f} … {r.r2_per_dim_max:.2f})",
                         " / ".join(f(getattr(r, f"rmse_h{lo}_{hi}"), 2) for lo, hi in Z.HORIZON_BINS), f(r.rmse_train), f(r.rmse_val), f(r.rmse_hold_z0), f(r.corr_Lz_EC, 2)])
        for a, b in (("M2", "M1"), ("M3", "M2")):
            if not has(P, dataset=ds, a=a, b=b, metric="L_z").empty:
                rows.append([Z.LABEL[ds], f"*{a} vs {b}*", paired_cell(P, ds, a, b, "L_z"), "", "", "", "", "", "", "", ""])
    bins = " / ".join(f"{lo}–{hi}" for lo, hi in Z.HORIZON_BINS)
    return md_table(["Dataset", "Model", "Test L_z (MSE, standardised)", "Test RMSE", "R² (pooled, vs train-mean z)", "R² per dimension: mean (min … max)", f"Test RMSE by horizon {bins}", "RMSE on TRAIN sequences", "RMSE on VALIDATION",
                     "RMSE of holding z*₀", "corr(L_z, E_C) per sequence"], rows)


def table_decoder(DD, P):
    rows = []
    for ds in Z.DATASETS:
        d = has(DD, dataset=ds)
        for m in [x for x in ("M1", "M2", "M3") if not has(d, decoder=x).empty]:
            for inp in ["oracle", "pred", "noisy_gauss", "noisy_perm"] + [f"zhat_{x}" for x in ("M1", "M2", "M3") if x != m]:
                r = has(d, decoder=m, input=inp, split="test")
                if r.empty:
                    continue
                r = r.iloc[0]; tr = has(d, decoder=m, input=inp); tr = tr[tr.split != "test"]
                lab = INP_LAB.get(inp, f"S. predicted ẑ of {inp[5:]} (swapped in)")
                rows.append([Z.LABEL[ds], f"decoder of {m}", lab, ci(r.E_C, r.ci_lo, r.ci_hi), " / ".join(f(r[f"h{lo}_{hi}"], 2) for lo, hi in Z.HORIZON_BINS), f(tr.iloc[0].E_C) if len(tr) else "—"])
    bins = " / ".join(f"{lo}–{hi}" for lo, hi in Z.HORIZON_BINS)
    return md_table(["Dataset", "Decoder", "Latent input", "Test dense E_C", f"by horizon {bins}", "same on 160 TRAIN sequences"], rows)


def table_decoder_pairs(P):
    rows = []
    for ds in Z.DATASETS:
        p = has(P, dataset=ds, metric="E_C"); p = p[p.a.astype(str).str.startswith("D_")] if len(p) else p
        for r in p.itertuples():
            rows.append([Z.LABEL[ds], f"{r.a} − {r.b}", f"{r.value_a:.3f} vs {r.value_b:.3f}", f"{r.diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]", f"{-r.rel_change * 100:+.1f} %", MARK[r.verdict], r.question])
    return md_table(["Dataset", "Comparison (a − b)", "E_C a vs b", "Difference [95 % CI]", "a relative to b (+ = a lower)", "Verdict for a", "What it isolates"], rows)


def table_horizon(H, P):
    rows = []
    for ds in Z.DATASETS:
        for m in MODELS + ["PERSIST"]:
            h = has(H, dataset=ds, model=m, curve="dense")
            if h.empty:
                continue
            rows.append([Z.LABEL[ds], ROW_LAB[m]] + [ci(*has(h, h_lo=lo, h_hi=hi).iloc[0][["value", "ci_lo", "ci_hi"]], d=2) for lo, hi in Z.HORIZON_BINS])
        for a, b in (("M2", "M1"), ("M2", "M0"), ("M1", "M0"), ("M3", "M0")):
            if has(P, dataset=ds, a=a, b=b, metric="E_C_h1_8").empty:
                continue
            rows.append([Z.LABEL[ds], f"*{a} vs {b}*"] + [(lambda r: f"{r.rel_improvement * 100:+.1f} % {MARK[r.verdict]}")(has(P, dataset=ds, a=a, b=b, metric=f"E_C_h{lo}_{hi}").iloc[0]) for lo, hi in Z.HORIZON_BINS])
    return md_table(["Dataset", "Model"] + [f"frames {lo}–{hi}" for lo, hi in Z.HORIZON_BINS], rows)


def table_struct_horizon(P):
    """Structural / wrench curves by horizon bin: paired relative difference (+ = first model better) and verdict."""
    curves = [("part", "participation"), ("amount", "amount"), ("centroid", "centroid"), ("normal", "normal"), ("wrench", "wrench")]; rows = []
    for ds in Z.DATASETS:
        for a, b in (("M2", "M0"), ("M2", "M1")):
            for c, lab in curves:
                cells = []
                for lo, hi in Z.HORIZON_BINS:
                    r = has(P, dataset=ds, a=a, b=b, metric=f"{c}_h{lo}_{hi}")
                    cells.append("—" if r.empty else f"{r.iloc[0].rel_improvement * 100:+.1f} % {MARK[r.iloc[0].verdict]}")
                if any(x != "—" for x in cells):
                    rows.append([Z.LABEL[ds], f"{a} vs {b}", lab] + cells)
    return md_table(["Dataset", "Pair", "Metric"] + [f"frames {lo}–{hi}" for lo, hi in Z.HORIZON_BINS], rows)


def table_decision(D):
    qs = ["Does joint decoder adaptation improve over previous M1?", "Does M2 beat direct M0 in dense contact?", "Does M2 beat M0 in structural / wrench quality?", "Does M2 improve long-horizon behavior?",
          "Is z prediction itself improved?", "Is the decoder more robust to predicted-z error?", "Is z mediation justified for the final generator?"]
    rows = []
    for q in qs:
        cells = []; ev = []
        for ds in Z.DATASETS:
            r = has(D, dataset=ds, question=q)
            cells.append(f"**{r.iloc[0].answer}**" if len(r) else "—"); ev.append(f"{Z.LABEL[ds]}: {r.iloc[0].evidence}" if len(r) else "")
        rows.append([q] + cells + ["<br>".join(e for e in ev if e)])
    c = [has(D, dataset=ds, question="case") for ds in Z.DATASETS]
    rows.append(["**Result case (Section 17 of the plan; all that apply)**"] + [f"**Case {x.iloc[0].answer}**" if len(x) else "—" for x in c] + ["<br>".join(f"{Z.LABEL[ds]}: {x.iloc[0].evidence}" for ds, x in zip(Z.DATASETS, c) if len(x))])
    return md_table(["Question", "TACO", "ARCTIC", "Evidence"], rows)


def table_probe(PR):
    rows = []
    for r in PR.itertuples():
        rows.append([Z.LABEL[r.dataset], r.model, f"{r.n_val} ({r.n_val_takes} takes)", f"{r.oof_val_E_C_unadapted:.3f} → {r.oof_val_E_C_best:.3f} ({r.oof_val_gain_rel * 100:+.1f} %)", r.s_star,
                     f"{r.test_E_C_unadapted:.3f} → {r.test_E_C_adapted:.3f}", f"{r.test_diff:+.3f} [{r.test_lo:+.3f}, {r.test_hi:+.3f}] ({r.test_gain_rel * 100:+.1f} %) {r.test_verdict}",
                     f"{r.vs_M0_diff:+.3f} [{r.vs_M0_lo:+.3f}, {r.vs_M0_hi:+.3f}] {r.vs_M0_verdict}",
                     (f"{r.shrink_factors}" if "shrink_factors" in PR else "—"), (f"{r.shrink_val_E_C_unadapted:.3f} → {r.shrink_val_E_C:.3f}" if "shrink_val_E_C" in PR else "—"),
                     (f"{r.test_E_C_unadapted:.3f} → {r.shrink_test_E_C:.3f} ({r.shrink_test_diff:+.3f} [{r.shrink_test_lo:+.3f}, {r.shrink_test_hi:+.3f}]) {r.shrink_test_verdict}" if "shrink_test_E_C" in PR else "—")])
    return md_table(["Dataset", "Decoder of", "Validation sequences", "A. decoder fine-tuned on validation latents: out-of-fold validation E_C (un-adapted → best)", "A. chosen steps s*", "A. test E_C (un-adapted → adapted)",
                     "A. adapted − un-adapted (paired)", "A. adapted − M0 (paired)", "B. shrinkage factor of ẑ per horizon bin (fitted on validation)", "B. validation E_C (before → after)", "B. test E_C (before → after, paired difference)"], rows)


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


def table_sanity(SAN):
    names = sorted({k for c in SAN.values() for k in c}); rows = []
    for k in names:
        rows.append([k.replace("_", " ")] + [SAN[ds][k]["status"] if k in SAN.get(ds, {}) else "—" for ds in Z.DATASETS] + [next((SAN[ds][k]["how"] for ds in Z.DATASETS if k in SAN.get(ds, {})), "")])
    return md_table(["Check", "TACO", "ARCTIC", "How it was checked"], rows)


def main():
    A, P, T, ZM, DD, H, D, MT, PR = (load(n) for n in ("main_metrics.csv", "paired_comparisons.csv", "training_summary.csv", "z_metrics.csv", "decoder_diagnostic.csv", "horizon_metrics.csv", "decision_summary.csv",
                                                       "model_table.csv", "heldout_probe.csv"))
    SAN = json.loads((Z.OUT / "sanity_summary.json").read_text()) if (Z.OUT / "sanity_summary.json").exists() else {}
    V = values(A, P, T, ZM, DD, H, D, PR, SAN)
    Z.write_json(Z.OUT / "report_values.json", V)
    cap = lambda n, title, body: f"**Table {n} — {title}**\n\n{body}"
    tables = {"table1": cap(1, "Model definitions and parameter counts", table_models(MT, T)), "table2": cap(2, "Training statistics", table_training(T)),
              "table3": cap(3, "Main contact metrics (test; mean over sequences, 95 % take-cluster bootstrap; paired differences by the pre-registered rule: ≥ 2 % and CI excluding 0)", table_main(A, P)),
              "table4": cap(4, "Structural and wrench metrics (test)", table_struct(A, P)),
              "table5": cap(5, "z prediction diagnostics (standardised teacher coordinates)", table_z(ZM, P)) if len(ZM) else "",
              "table6": cap(6, "Decoder robustness / oracle-z diagnostic: one trained decoder, different latent inputs (test)", table_decoder(DD, P)) if len(DD) else "",
              "table6b": cap("6b", "Paired readings of the decoder diagnostic", table_decoder_pairs(P)) if len(DD) else "",
              "table7": cap(7, "Horizon analysis: dense error by horizon bin (test)", table_horizon(H, P)),
              "table7b": cap("7b", "Structural and wrench errors by horizon bin: paired relative difference (+ = the first model is better) and verdict", table_struct_horizon(P)),
              "table8": cap(8, "Final decision", table_decision(D)) if len(D) else "",
              "table9": cap(9, "Temporal stability (test)", table_stability(A, P)),
              "table10": cap(10, "Held-out adaptation probes (post-hoc; the temporal network is frozen and the decoder side is fitted to its VALIDATION latents, then scored on test)", table_probe(PR)) if len(PR) else "",
              "table11": cap(11, "Sanity checks", table_sanity(SAN)) if SAN else "", "ckpt_table": table_ckpt()}
    fig = lambda n, name, alt: f"![Figure {n} — {alt}](figures/{name})"
    figs = {"fig1": fig(1, "fig1_architecture.png", "architecture"), "fig2": fig(2, "fig2_main_comparison.png", "main comparison"), "fig3": fig(3, "fig3_error_vs_horizon.png", "error vs horizon"),
            "fig4": fig(4, "fig4_z_prediction.png", "z prediction"), "fig5": fig(5, "fig5_decoder_mismatch.png", "decoder mismatch"), "fig7": fig(7, "fig7_training_curves.png", "training curves"),
            "fig8": fig(8, "fig8_train_vs_heldout.png", "train vs held-out latent error")}
    for ds in Z.DATASETS:
        for k in ("helps", "fails", "typical"):
            figs[f"fig6_{ds}_{k}"] = fig(6, f"fig6_{ds}_{k}.png", f"qualitative {Z.LABEL[ds]} {k}")
    missing = set()

    def fill(txt):
        for k, v in {**tables, **figs}.items():
            txt = txt.replace("{{" + k + "}}", v)

        def sub(m):
            k, fmt = m.group(1), m.group(2)
            if k not in V:
                missing.add(k); return f"??{k}??"
            v = V[k]
            try:
                return format(v, fmt) if fmt else str(v)
            except (ValueError, TypeError):
                missing.add(k + ":" + str(fmt)); return f"??{k}??"
        return re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", sub, txt)
    body = fill((NARR / "00_head.md").read_text()) + "\n" if (NARR / "00_head.md").exists() else "# z joint-decoder follow-up\n\n"
    for i, (sec, title) in enumerate(SECTIONS, 1):
        p = NARR / f"{i:02d}_{sec}.md"
        body += f"\n## {i}. {title}\n\n{fill(p.read_text()) if p.exists() else '_(section not written)_'}\n"
    for name, title in (("90_sanity.md", "Appendix A. Sanity checks"), ("91_questions.md", "Appendix B. The six questions, answered"), ("92_reproduction.md", "Appendix C. Files, checkpoints and reproduction")):
        if (NARR / name).exists():
            body += f"\n## {title}\n\n{fill((NARR / name).read_text())}\n"
    (Z.OUT / "report.md").write_text(body)
    print("report.md:", len(body.splitlines()), "lines; unresolved blocks:", body.count("{{"), "; missing values:", sorted(missing))


if __name__ == "__main__":
    main()
