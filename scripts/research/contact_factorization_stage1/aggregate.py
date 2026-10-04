#!/usr/bin/env python
"""Root tables and the Stage-1 decision.    python aggregate.py
Reads the per-dataset outputs of eval_*.py and the training logs; writes under the report root:
  model_table.csv, reconstruction_metrics.csv, probe_metrics.csv, neighborhood_metrics.csv, swap_metrics.csv,
  temporal_persistence.csv, latent_statistics.csv, residual_localisation.csv, training_summary.csv, decision_summary.csv,
  experiment_config.json.
Decision rule (pre-registered thresholds in cf_common.DECISION): each Table-8 question is answered yes / partial / no per
dataset from the named metrics; Case A = every question yes and A3's full reconstruction within 10 % of A0; Case C = A3
reconstructs clearly worse than A0 (> 30 %) or z does not keep structure or structure is not concentrated in z or the swap
fails or r has no persistence; Case B otherwise.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

import cf_common as S

D = S.DECISION


def load_all(fname):
    dfs = []
    for ds in S.DATASETS:
        p = S.ds_out(ds) / fname
        if p.exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def training_summary():
    rows = []
    for ds in S.DATASETS:
        for m in S.ALL_MODELS:
            name = S.run_name(m); p = S.ckpt_path(ds, name)
            if not p.exists():
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False); lg = pd.read_csv(S.train_log_path(ds, name))
            tr = lg[lg.kind == "train"]; ev = lg[lg.kind == "eval"]; last_phase = tr.phase.iloc[-1]; tp = tr[tr.phase == last_phase]
            rows.append(dict(dataset=ds, model=m, n_params=ck["n_params"], steps=ck["steps"], best_step=ck["best_step"], phases=str(ck["phases"]),
                             train_loss_last=float(tp.total.iloc[-5:].mean()), train_coarse_last=float(tp.coarse.iloc[-5:].mean()),
                             train_full_last=float(tp.full.iloc[-5:].mean()) if "full" in tp and tp.full.notna().any() else np.nan,
                             train_rel_last=float(tp.rel.iloc[-5:].mean()) if "rel" in tp and tp.rel.notna().any() else np.nan,
                             grad_z_last=float(tp.grad_z.iloc[-5:].mean()), grad_r_last=float(tp.grad_r.iloc[-5:].mean()),
                             grad_z_phaseB_first=float(tr[tr.phase == "B"].grad_z.iloc[:5].mean()) if (tr.phase == "B").any() else np.nan,
                             grad_r_phaseB_first=float(tr[tr.phase == "B"].grad_r.iloc[:5].mean()) if (tr.phase == "B").any() else np.nan,
                             val_E_full_best=ck["best_val"]["E_full"], val_E_zonly_best=ck["best_val"]["E_zonly"], train_E_full_best=ck["best_train"]["E_full"],
                             val_z_std=ck["best_val"]["z_std"], val_r_std=ck["best_val"]["r_std"], val_delta_abs=ck["best_val"]["delta_abs"],
                             val_z_eff_rank=ck["best_val"]["z_eff_rank"], val_r_eff_rank=ck["best_val"]["r_eff_rank"],
                             best_is_last_eval=bool(ck["best_step"] == ev.step.iloc[-1]), hours=float(lg.elapsed.iloc[-1]) / 3600))
    return pd.DataFrame(rows)


def get(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def verdict(ok, partial):
    return "yes" if ok else ("partial" if partial else "no")


def decide(ds, rec, prob, knn, swp, tmp, M="A3"):
    """Table-8 answers for the factorised model M (primary A3; also the bottleneck variant A3_z16)."""
    rows = []
    a0, a3 = get(rec, dataset=ds, model="A0"), get(rec, dataset=ds, model=M)
    pz, pr, pc = get(prob, dataset=ds, input=f"{M}:z", probe="mlp"), get(prob, dataset=ds, input=f"{M}:r", probe="mlp"), get(prob, dataset=ds, input="C", probe="mlp")
    ph = get(prob, dataset=ds, input="A0:h", probe="mlp")
    kz, kr, kc, krand = get(knn, dataset=ds, space=f"{M}:z"), get(knn, dataset=ds, space=f"{M}:r"), get(knn, dataset=ds, space="C"), get(knn, dataset=ds, space="random")
    sw = get(swp, dataset=ds, model=M)
    def t(code, metric, h):
        r_ = get(tmp, dataset=ds, code=code.replace("A3:", f"{M}:"), metric=metric, h=h)
        return r_ if r_ is not None else get(tmp, dataset=ds, code=code, metric=metric, h=h)      # variant without its own residual rows: A3's
    if any(x is None for x in (a0, a3, pz, pr, pc, ph, kz, kr, kc, krand, sw)):
        return [], None
    # Q1 z preserves structure
    ret_z = float(pz.structure_retention); q1 = verdict(ret_z >= D["z_keeps_min"], ret_z >= 0.5)
    rows.append(dict(dataset=ds, model=M, question="Does z preserve structural / functional information?", answer=q1,
                     evidence=f"{M} z MLP probe keeps {100 * ret_z:.0f} % of the dense-C probe's gain (AUPRC {pz.part_auprc:.3f} vs C {pc.part_auprc:.3f}; amount R² {pz.amount_r2:.2f} vs {pc.amount_r2:.2f}; centroid {pz.centroid:.3f} vs {pc.centroid:.3f}; normal {pz.normal:.1f}° vs {pc.normal:.1f}°; wrench EV {pz.wrench_ev:.2f} vs {pc.wrench_ev:.2f}); threshold {D['z_keeps_min']:.2f}"))
    # Q2 r improves dense reconstruction
    gain = float(a3.gain_r); q2 = verdict(gain >= D["gain_r_min"] and a3.E_zonly_minus_full_lo > 0, gain >= 0.1 and a3.E_zonly_minus_full_lo > 0)
    rows.append(dict(dataset=ds, model=M, question="Does r significantly improve dense reconstruction?", answer=q2,
                     evidence=f"{M} E_zonly {a3.E_zonly:.3f} → E_full {a3.E_full:.3f} (raw L2), Gain_r {gain:.2f} [{a3.gain_r_lo:.2f}, {a3.gain_r_hi:.2f}], explained_by_r {a3.explained_by_r:.2f}; A0 E_full {a0.E_full:.3f}; threshold Gain_r ≥ {D['gain_r_min']:.2f}"))
    # Q3 structure concentrated in z
    ret_r = float(pr.structure_retention); margin = ret_z - ret_r; q3 = verdict(margin >= D["z_over_r_margin"], margin >= 0.05)
    rows.append(dict(dataset=ds, model=M, question="Is structure more concentrated in z than r?", answer=q3,
                     evidence=f"structure retention z {ret_z:.2f} vs r {ret_r:.2f} (margin {margin:+.2f}; threshold ≥ {D['z_over_r_margin']:.2f}); A0 h {float(ph.structure_retention):.2f}; r probe: AUPRC {pr.part_auprc:.3f}, wrench EV {pr.wrench_ev:.2f}"))
    # Q4 neighbourhoods
    q4 = verdict(kz.teacher_rel <= D["knn_z_vs_random_max"] and kz.teacher < kc.teacher and kz.recall > kc.recall, kz.teacher_rel <= 0.85 or (kz.teacher < kc.teacher and kz.recall > kc.recall))
    rows.append(dict(dataset=ds, model=M, question="Do z neighbourhoods preserve R2 / wrench similarity?", answer=q4,
                     evidence=f"teacher distance of the 10 z neighbours {kz.teacher:.3f} ({100 * kz.teacher_rel:.0f} % of random {krand.teacher:.3f}) vs dense-C neighbours {kc.teacher:.3f} and r neighbours {kr.teacher:.3f}; recall@10 vs teacher: z {kz.recall:.2f}, C {kc.recall:.2f}, r {kr.recall:.2f}; threshold ≤ {100 * D['knn_z_vs_random_max']:.0f} % of random and better than C"))
    # Q5 swap
    q5 = verdict(sw.frac_struct_follows_z >= D["swap_struct_frac_min"] and sw.frac_detail_follows_r >= D["swap_detail_frac_min"], sw.frac_struct_follows_z >= D["swap_struct_frac_min"] or sw.frac_detail_follows_r >= D["swap_detail_frac_min"])
    rows.append(dict(dataset=ds, model=M, question="Does swapping r alter the realisation without destroying the z-defined structure?", answer=q5,
                     evidence=f"{int(sw.n_swaps)} swaps: structure follows the z donor in {100 * sw.frac_struct_follows_z:.0f} % (teacher distance to z donor {sw.struct_to_zdonor:.3f} vs r donor {sw.struct_to_rdonor:.3f}, floor {sw.struct_floor:.3f}); detail follows the r donor in {100 * sw.frac_detail_follows_r:.0f} % (cos {sw.detail_cos_rdonor:.2f} vs {sw.detail_cos_zdonor:.2f}); swapping r moves the map by {sw.dense_change:.3f} (pair distance {sw.dense_AB:.3f}); thresholds {100 * D['swap_struct_frac_min']:.0f} % / {100 * D['swap_detail_frac_min']:.0f} %"))
    # Q6 persistence of r
    acf8 = float(t("A3:r", "autocorrelation", 8).value); gain8 = float(t("A3:r", "r2_gain_past_beyond_z3", 8).value)
    disp_r, disp_res = float(t("A3:r", "displacement_ratio", 8).value), float(t("A3:resid_z", "displacement_ratio", 8).value); disp_C = float(t("dense_C", "displacement_ratio", 8).value)
    acf_res = float(t("A3:resid_z", "autocorrelation", 8).value); r2_self = float(t("A3:r", "r2_from_past_self", 8).value); r2_z = float(t("A3:r", "r2_from_current_z3", 8).value)
    disp_z = float(t("A3:z", "displacement_ratio", 8).value); acf_h = float(t("A0:h", "autocorrelation", 8).value)
    ok6 = acf8 >= D["r_acf_h8_min"] and acf8 >= acf_h - D["r_acf_vs_h_margin"] and gain8 >= D["r_init_gain_min"] and disp_r / disp_res <= D["r_slower_than_dense"]
    # 'partial' = r is persistent in the absolute sense (acf floor) but misses the reference-relative properties; 'no' = no persistence
    q6 = verdict(ok6, acf8 >= D["r_acf_h8_min"])
    rows.append(dict(dataset=ds, model=M, question="Is r temporally persistent?", answer=q6,
                     evidence=f"r autocorrelation h=8 {acf8:.2f} (z {float(t('A3:z', 'autocorrelation', 8).value):.2f}, h {acf_h:.2f}, dense residual {acf_res:.2f}, dense C {float(t('dense_C', 'autocorrelation', 8).value):.2f}); displacement ratio h=8: r {disp_r:.2f}, z {disp_z:.2f}, dense residual {disp_res:.2f}, dense C {disp_C:.2f}; R² of r_(t+8): from r_t {r2_self:.2f}, from z_(t+8) {r2_z:.2f}, gain of r_t beyond z_(t+8) {gain8:.2f}; thresholds acf ≥ {D['r_acf_h8_min']:.2f} and ≥ acf(h) − {D['r_acf_vs_h_margin']:.2f}, gain ≥ {D['r_init_gain_min']:.2f}, r not faster than the dense residual"))
    # Q7 overall
    close = a3.E_full <= (1 + D["recon_close_to_a0"]) * a0.E_full; far = a3.E_full > 1.3 * a0.E_full
    answers = [q1, q2, q3, q4, q5, q6]
    if all(x == "yes" for x in answers) and close:
        case = "A"
    elif far or q1 == "no" or q3 == "no" or q5 == "no" or q6 == "no":
        case = "C"
    else:
        case = "B"
    missing = [q for q, x in zip(("z structure", "r dense gain", "concentration", "neighbourhoods", "swap", "r persistence"), answers) if x != "yes"]
    rows.append(dict(dataset=ds, model=M, question="Is Stage 1 sufficiently supported to proceed to Stage 2?", answer={"A": "yes (Case A)", "B": "partial (Case B)", "C": "no (Case C)"}[case],
                     evidence=f"{M} E_full {a3.E_full:.3f} vs A0 {a0.E_full:.3f} ({100 * (a3.E_full / a0.E_full - 1):+.0f} %, close = within {100 * D['recon_close_to_a0']:.0f} %: {close}); " + ("all six properties hold" if not missing else "not yet: " + ", ".join(missing))))
    return rows, case


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    rec, prob, knn, swp, tmp, lat, loc = (load_all(f) for f in ("reconstruction.csv", "probe_metrics.csv", "neighborhood_metrics.csv", "swap_metrics.csv", "temporal_persistence.csv", "latent_statistics.csv", "residual_localisation.csv"))
    ts = training_summary()
    counts = {m: int(ts[ts.model == m].n_params.iloc[0]) for m in S.ALL_MODELS if (ts.model == m).any()}
    S.model_table(counts).to_csv(S.OUT / "model_table.csv", index=False)
    for name, df in (("reconstruction_metrics", rec), ("probe_metrics", prob), ("neighborhood_metrics", knn), ("swap_metrics", swp), ("temporal_persistence", tmp), ("latent_statistics", lat), ("residual_localisation", loc), ("training_summary", ts)):
        df.to_csv(S.OUT / f"{name}.csv", index=False)
    rows, cases, cases_variant = [], {}, {}
    for ds in S.DATASETS:
        if not ((rec.dataset == ds).any() and (prob.dataset == ds).any() and (knn.dataset == ds).any() and (swp.dataset == ds).any() and (tmp.dataset == ds).any()):
            print("incomplete", ds); continue
        r, cases[ds] = decide(ds, rec, prob, knn, swp, tmp, "A3"); rows += r
        r, cv = decide(ds, rec, prob, knn, swp, tmp, "A3_z16"); rows += r
        if cv:
            cases_variant[ds] = cv
    dec = pd.DataFrame(rows); dec.to_csv(S.OUT / "decision_summary.csv", index=False)
    cfg = dict(models=S.MODELS, variants=S.VARIANTS, arch=S.ARCH, train=S.TRAIN, teacher=S.TEACHER, probe=S.PROBE, knn_k=S.KNN_K, n_boot=S.N_BOOT, decision_thresholds=S.DECISION, seed=S.SEED,
               datasets=list(S.DATASETS), cases=cases, cases_bottleneck_variant=cases_variant, data_root={ds: str(S.SAT.HP.ROOTS[ds]) for ds in S.DATASETS}, feature_cache={ds: str(S.SAT.SV.feature_path(ds)) for ds in S.DATASETS},
               mask_cache={ds: str(S.SAT.mask_path(ds)) for ds in S.DATASETS}, checkpoints=str(S.CKPT), scripts="scripts/research/contact_factorization_stage1")
    S.write_json(S.OUT / "experiment_config.json", cfg)
    print(dec.to_string()); print(cases)


if __name__ == "__main__":
    main()
