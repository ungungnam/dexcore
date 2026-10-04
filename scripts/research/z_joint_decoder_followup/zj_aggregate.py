#!/usr/bin/env python
"""Aggregate the per-example metrics of M0 / M1 / M2 (/ M3 and the sensitivity rows) into the report tables.    python zj_aggregate.py
Every number is a mean over the test sequences (one trajectory per sequence, fixed GT s_0) with a 95 % take-cluster bootstrap interval
(1000 reps); paired comparisons are differences of per-sequence values on identical test sequences with the same bootstrap and the
pre-registered rule of zj_common.DECISION.  M0 / M1 rows are read from the Stage-2 metric files (same evaluation code; zj_evaluate.py
--recheck verifies it).  Writes to OUT: model_table.csv, training_summary.csv, main_metrics.csv, structural_metrics.csv,
wrench_metrics.csv, temporal_stability.csv, z_metrics.csv, decoder_diagnostic.csv, horizon_metrics.csv, temporal_curves.npz,
paired_comparisons.csv, decision_summary.csv, heldout_probe.csv, experiment_config.json.
"""
from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd
import torch

import zj_common as Z
from s2_aggregate import CURVES, METRICS, STRUCT_TABLE, WRENCH_TABLE, ci_row, per_example, sat_aggregate_module, verdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_aggregate")
S2 = Z.S2
MODEL_ORDER = ["M0", "M1", "M2", "M3", "M2_seltotal", "M3_seltotal", "M2_4blk", "M2_s1", "M0r", "PERSIST", "GT"]
PAIRS = [("M2", "M1", "does the jointly adapted decoder improve on the previous z-mediated model?"),
         ("M2", "M0", "does z mediation beat direct dense prediction once the decoder is adapted?"),
         ("M1", "M0", "the previous Stage-2 comparison (B1 vs B0), reproduced from the stored metrics"),
         ("M3", "M2", "does also feeding the teacher z to the decoder help?"), ("M3", "M1", "dual-input model vs the previous z-mediated model"), ("M3", "M0", "dual-input model vs direct"),
         ("M2_seltotal", "M2", "selection sensitivity: the state at the minimum full objective vs the state at the minimum validation L_C"),
         ("M2_seltotal", "M1", "same selection criterion as M1: what the larger decoder and the longer schedule change"),
         ("M3_seltotal", "M3", "selection sensitivity (M3)"),
         ("M2_4blk", "M1", "control: 4-block decoder under the M2 protocol vs M1"), ("M2", "M2_4blk", "decoder capacity: 6 vs 4 blocks under the same protocol"),
         ("M2_s1", "M2", "seed 1 vs seed 0"), ("M2_s1", "M0", "extra seed vs direct"),
         ("M0r", "M0", "budget parity: M0 retrained under this protocol vs the reused B0"), ("M2", "M0r", "M2 vs the direct model trained under the identical protocol"), ("M3", "M0r", "M3 vs the direct model trained under the identical protocol")]
_CACHE = {}


def metrics_file(ds, key):
    special = {"GT": S2.metrics_path(ds, "GT"), "PERSIST": S2.metrics_path(ds, "PERSIST"), "M2_seltotal": Z.metrics_path(ds, Z.run_name("M2") + "_seltotal"),
               "M3_seltotal": Z.metrics_path(ds, Z.run_name("M3") + "_seltotal"), "M2_4blk": Z.metrics_path(ds, Z.run_name("M2", tag="_4blk")), "M2_s1": Z.metrics_path(ds, Z.run_name("M2", Z.EXTRA_SEED))}
    return special[key] if key in special else Z.model_metrics(ds, key)


def load_metrics(ds, key):
    if (ds, key) not in _CACHE:
        p = metrics_file(ds, key)
        _CACHE[(ds, key)] = None if not p.exists() else {k: v for k, v in np.load(p, allow_pickle=True).items()}
    return _CACHE[(ds, key)]


def gpus_of(log_path):
    """GPU ids of the attempts of the reported run (the launch that was stopped before step 500 and restarted from step 1 is not counted)."""
    if not log_path.exists():
        return ""
    return ",".join(sorted(set(re.findall(r"=== attempt \d+ on gpu (\d+)", log_path.read_text().split("the run restarts from step 1")[-1]))))


def training_row(ds, model):
    ck_p = Z.model_ckpt(ds, model)
    if not ck_p.exists():
        return None, None
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    lg = pd.read_csv(Z.model_train_log(ds, model)); reused = Z.MODELS[model]["reuse"] is not None
    tl_p = Z.model_train_log(ds, model).with_name(Z.model_train_log(ds, model).stem + "_train.csv"); tl = pd.read_csv(tl_p) if tl_p.exists() else pd.DataFrame()
    at = lambda step, col: float(lg.loc[lg.step == step, col].iloc[0]) if (col in lg and (lg.step == step).any()) else np.nan
    bs = int(ck["best_step"]); bst = int(ck.get("best_step_total", ck["best_step"]))
    win = tl[(tl.step > bs - 1000) & (tl.step <= bs)] if len(tl) else tl
    pc = ck["n_params"]
    row = dict(dataset=ds, model=model, source="Stage-2 checkpoint (reused)" if reused else "trained in this study", n_params_total=pc["total"], n_params_backbone=pc["backbone"], n_params_decoder=pc.get("D_z", pc.get("head")),
               decoder_blocks=(0 if "D_z" not in pc else 4 + int(ck["cfg"].get("extra_blocks", 0))), max_steps=int(ck["cfg"]["max_steps"]), min_steps=int(ck["cfg"]["min_steps"]), patience=int(ck["cfg"]["patience"]),
               steps=int(ck["steps"]), stopped_by=ck["stopped_by"], selection=ck.get("selection", "validation objective (L_C + lambda_z L_z)" if "L_z" in "".join(lg.columns) else "validation L_C"),
               best_step=bs, best_step_objective=bst, hours=ck["seconds"] / 3600, s_per_step=ck["seconds"] / max(int(ck["steps"]), 1),
               val_L_C_at_best=at(bs, "val_L_C"), val_L_z_at_best=at(bs, "val_L_z"), val_E_C_at_best=at(bs, "val_E_C"), val_E_C_teacher_z_at_best=at(bs, "val_E_C_gt"),
               val_objective_at_best=at(bs, "val"), min_val_L_C=float(lg["val_L_C"].min()), min_val_L_C_step=int(lg.loc[lg["val_L_C"].idxmin(), "step"]),
               min_val_objective_step=int(lg.loc[lg["val"].idxmin(), "step"]), val_L_C_last=float(lg["val_L_C"].iloc[-1]), val_L_z_last=float(lg["val_L_z"].iloc[-1]) if "val_L_z" in lg else np.nan,
               train_L_z_before_best=float(win["L_z"].mean()) if ("L_z" in win and len(win)) else np.nan, train_L_C_before_best=float(win["L_C"].mean()) if ("L_C" in win and len(win)) else np.nan,
               decoder_rel_change_at_best=at(bs, "dec_rel_change"), decoder_extra_ada_norm_at_best=at(bs, "dec_extra_ada_norm"),
               gpus=gpus_of(Z.OUT / "logs" / f"train_{ds}_{Z.run_name(model)}.log") if not reused else gpus_of(S2.OUT / "logs" / f"train_{ds}_{Z.reused_run(model)}.log"))
    cfg = dict(ckpt=str(ck_p), md5=Z.md5(ck_p), best_step=bs, best_step_objective=bst, steps=int(ck["steps"]), stopped_by=ck["stopped_by"], teacher=ck["teacher"], n_params=pc, cfg=ck["cfg"])
    return row, cfg


def block(get, a, b, metrics):
    """Stage-2 block rule: yes (>= 3 better, none worse) / partial (>= 1 better, <= 1 worse) / worse (more worse than better) / no."""
    vs = {m: get(a, b, m) for m in metrics}; vs = {m: v for m, v in vs.items() if v is not None}
    nb = sum(v.verdict == "better" for v in vs.values()); nw = sum(v.verdict == "worse" for v in vs.values())
    ans = "worse" if nw > nb else ("yes" if (nb >= Z.DECISION["struct_majority"] and nw == 0) else ("partial" if (nb >= 1 and nw <= 1) else "no"))
    return ans, nb, nw, vs


def decide(ds, get, zpair, swap):
    """The pre-registered rule of zj_common.DECISION applied to the paired verdicts -> rows of decision_summary.csv."""
    rows = []
    fmt = lambda r: f"{r.a} {r.value_a:.3f} vs {r.b} {r.value_b:.3f}, diff {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.rel_improvement * 100:+.2f} %)"
    sfmt = lambda vs: "; ".join(f"{m} {v.verdict} ({v.rel_improvement * 100:+.1f} %)" for m, v in vs.items())
    e1, e0 = get("M2", "M1", "E_C"), get("M2", "M0", "E_C")
    s1, nb1, nw1, vs1 = block(get, "M2", "M1", Z.STRUCT_DECISION); s0, nb0, nw0, vs0 = block(get, "M2", "M0", Z.STRUCT_DECISION)
    # axes
    adapt = "improves" if e1.verdict == "better" else ("worse" if e1.verdict == "worse" else ("structure_only" if s1 == "yes" else "none"))
    direct = "beats" if e0.verdict == "better" else ("worse" if e0.verdict == "worse" else ("structure_only" if s0 == "yes" else "similar"))
    q1 = {"improves": "yes", "structure_only": "structure only", "none": "no (similar)", "worse": "no (worse)"}[adapt]
    rows.append(dict(dataset=ds, question="Does joint decoder adaptation improve over previous M1?", answer=q1, evidence=f"E_C: {fmt(e1)}; structure / wrench {nb1} better / {nw1} worse of {len(vs1)} ({sfmt(vs1)})"))
    q2 = {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[e0.verdict]
    rows.append(dict(dataset=ds, question="Does M2 beat direct M0 in dense contact?", answer=q2, evidence=f"E_C: {fmt(e0)}"))
    rows.append(dict(dataset=ds, question="Does M2 beat M0 in structural / wrench quality?", answer=s0, evidence=f"{nb0} better / {nw0} worse of {len(vs0)}: {sfmt(vs0)}"))
    l1, l0 = get("M2", "M1", "E_C_last16"), get("M2", "M0", "E_C_last16")
    lh = lambda r: {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[r.verdict]
    stab = "; ".join(f"{m} vs M0 {get('M2', 'M0', m).verdict}, vs M1 {get('M2', 'M1', m).verdict}" for m in Z.LONG_HORIZON[1:] if get("M2", "M0", m) is not None)
    rows.append(dict(dataset=ds, question="Does M2 improve long-horizon behavior?", answer=f"vs M1: {lh(l1)}; vs M0: {lh(l0)}", evidence=f"last 16 frames: {fmt(l1)}; {fmt(l0)}; stability: {stab}"))
    if zpair is not None:
        q5 = {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[zpair["verdict"]]
        rows.append(dict(dataset=ds, question="Is z prediction itself improved?", answer=q5,
                         evidence=f"test L_z: M2 {zpair['value_a']:.3f} vs M1 {zpair['value_b']:.3f}, diff {zpair['diff']:+.3f} [{zpair['ci_lo']:+.3f}, {zpair['ci_hi']:+.3f}] ({zpair['rel_improvement'] * 100:+.1f} %)"))
    if swap:
        nb = sum(v["verdict"] == "better" for v in swap.values())
        q6 = "yes" if nb == len(swap) else ("partial" if nb >= 1 else ("no (worse)" if all(v["verdict"] == "worse" for v in swap.values()) else "no"))
        rows.append(dict(dataset=ds, question="Is the decoder more robust to predicted-z error?", answer=q6,
                         evidence="same predicted z into both decoders: " + "; ".join(f"on {k}: D_M2 {v['value_a']:.3f} vs D_M1 {v['value_b']:.3f} ({v['rel_improvement'] * 100:+.1f} %, {v['verdict']})" for k, v in swap.items())))
    q7 = {"beats": "yes", "structure_only": "for structure only (dense not better than M0)", "similar": "no (no final-task advantage)", "worse": "no (worse than direct)"}[direct]
    rows.append(dict(dataset=ds, question="Is z mediation justified for the final generator?", answer=q7, evidence=f"M2 vs M0: dense {e0.verdict}, structure / wrench {s0} ({nb0} better / {nw0} worse)"))
    cases = []
    if adapt == "improves" and direct == "beats":
        cases.append("A")
    if adapt in ("improves", "structure_only") and direct == "similar":
        cases.append("B")
    if direct == "structure_only":
        cases.append("C")
    if adapt in ("none", "worse"):
        cases.append("D")
    if direct == "worse":
        cases.append("E")
    if not cases:
        cases.append("A-direct" if direct == "beats" else "unclassified")
    amb = (abs(e0.rel_change) >= Z.DECISION["rel_min"]) and (e0.ci_lo <= 0 <= e0.ci_hi)
    rows.append(dict(dataset=ds, question="case", answer="+".join(cases), evidence=f"adaptation axis (M2 vs M1): {adapt}; direct axis (M2 vs M0): {direct}"))
    rows.append(dict(dataset=ds, question="extra seed needed (M2 vs M0 ambiguous)?", answer="yes" if amb else "no", evidence=f"|relative difference| {abs(e0.rel_change) * 100:.1f} % (threshold 2 %), CI [{e0.ci_lo:+.3f}, {e0.ci_hi:+.3f}]"))
    rows.append(dict(dataset=ds, question="4-block control needed (M2 better than M1)?", answer="yes" if adapt == "improves" else "no", evidence=f"M2 vs M1 dense: {e1.verdict}"))
    return rows


def main():
    Z.OUT.mkdir(parents=True, exist_ok=True)
    AG = sat_aggregate_module()
    acc_rows, pair_rows, zrows, drows, trows, hrows, dec_rows, curves, cfg_runs = [], [], [], [], [], [], [], {}, {}
    for ds in Z.DATASETS:
        for model in Z.MODELS:
            row, cfg = training_row(ds, model)
            if row is not None:
                trows.append(row); cfg_runs[f"{ds}/{model}"] = cfg
        gt = load_metrics(ds, "GT")
        take = gt["take"].astype(str); ex = gt["example"]
        M = {m: load_metrics(ds, m) for m in MODEL_ORDER}; M = {m: z for m, z in M.items() if z is not None}
        for m, z in M.items():
            assert np.array_equal(z["example"], ex), (ds, m, "example mismatch")
            for metric in METRICS:
                if metric in z:
                    acc_rows.append(ci_row(per_example(z, metric), take, dataset=ds, model=m, metric=metric))
            for r in AG.macro_scores([(0, z)], take, "K1"):
                acc_rows.append(dict(dataset=ds, model=m, metric=r["metric"], value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_examples=len(take)))
            for key in ("jitter_dense", "jitter_r2"):
                acc_rows.append(ci_row(np.abs(per_example(z, key) - per_example(gt, key)), take, dataset=ds, model=m, metric=f"jitter_dev_{key[7:]}"))
                acc_rows.append(ci_row(per_example(z, key) / np.maximum(per_example(gt, key), 1e-9), take, dataset=ds, model=m, metric=f"jitter_ratio_{key[7:]}"))
            for c in CURVES:
                k = f"curve_{c}"
                if k in z:
                    cur = z[k].astype(np.float64); cur = cur[:, 0] if cur.ndim == 3 else cur
                    curves[f"{ds}|{m}|{c}"] = np.nanmean(cur, 0)
                    if c in ("dense", "part", "amount", "centroid", "normal", "wrench"):
                        for lo, hi in Z.HORIZON_BINS:
                            hrows.append(ci_row(np.nanmean(cur[:, lo:hi + 1], 1), take, dataset=ds, model=m, curve=c, h_lo=lo, h_hi=hi))
        # ------------------------------------------------------------- paired comparisons (metrics and horizon bins of the dense curve)
        for a, b, q in PAIRS:
            if a not in M or b not in M:
                continue
            for metric in METRICS + ["jitter_dev_dense", "jitter_dev_r2"]:
                if metric.startswith("jitter_dev"):
                    key = "jitter_" + metric[11:]
                    va = np.abs(per_example(M[a], key) - per_example(gt, key)); vb = np.abs(per_example(M[b], key) - per_example(gt, key))
                elif metric in M[a] and metric in M[b]:
                    va, vb = per_example(M[a], metric), per_example(M[b], metric)
                else:
                    continue
                pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=metric, **verdict(metric, va, vb, take)))
            ca, cb = M[a]["curve_dense"].astype(np.float64), M[b]["curve_dense"].astype(np.float64)
            ca = ca[:, 0] if ca.ndim == 3 else ca; cb = cb[:, 0] if cb.ndim == 3 else cb
            for lo, hi in Z.HORIZON_BINS:
                pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=f"E_C_h{lo}_{hi}", **verdict("E_C", np.nanmean(ca[:, lo:hi + 1], 1), np.nanmean(cb[:, lo:hi + 1], 1), take)))
            for c in ("part", "amount", "centroid", "normal", "wrench"):                    # structural / wrench curves by horizon bin (all lower-is-better)
                if f"curve_{c}" not in M[a] or f"curve_{c}" not in M[b]:
                    continue
                xa, xb = M[a][f"curve_{c}"].astype(np.float64), M[b][f"curve_{c}"].astype(np.float64)
                xa = xa[:, 0] if xa.ndim == 3 else xa; xb = xb[:, 0] if xb.ndim == 3 else xb
                with np.errstate(all="ignore"):
                    for lo, hi in Z.HORIZON_BINS:
                        pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=f"{c}_h{lo}_{hi}", **verdict("E_C", np.nanmean(xa[:, lo:hi + 1], 1), np.nanmean(xb[:, lo:hi + 1], 1), take)))
        # ------------------------------------------------------------- z diagnostics and the decoder-mismatch diagnostic (zj_diag.py)
        dp = Z.ds_out(ds) / "diag" / "diag.npz"; zpair, swap = None, {}
        if dp.exists():
            D = dict(np.load(dp, allow_pickle=True)); assert np.array_equal(D["example"], ex)
            zmodels = [m for m in Z.Z_MODELS if f"{m}|zerr2|test" in D]
            tz2 = D["teacher|z2|test"].astype(np.float64); hold = D["hold_z0|zerr2|test"].astype(np.float64); var_dim = D["teacher|var_dim|test"].astype(np.float64)
            for m in zmodels:
                e = D[f"{m}|zerr2|test"].astype(np.float64); r2d = 1 - D[f"{m}|zerr2_dim|test"].astype(np.float64) / np.maximum(var_dim, 1e-9)
                tr, va = D[f"{m}|zerr2|train"].astype(np.float64), D[f"{m}|zerr2|val"].astype(np.float64)
                lz = Z.cluster_bootstrap(e.mean(1), take, n_boot=Z.N_BOOT)
                zrows.append(dict(dataset=ds, model=m, L_z_test=lz[0], L_z_lo=lz[1], L_z_hi=lz[2], rmse_test=float(np.sqrt(e.mean())), r2_pooled=float(1 - e.mean() / tz2.mean()),
                                  r2_per_dim_mean=float(r2d.mean()), r2_per_dim_min=float(r2d.min()), r2_per_dim_max=float(r2d.max()),
                                  rmse_train=float(np.sqrt(tr.mean())), rmse_val=float(np.sqrt(va.mean())), train_over_val_rmse=float(np.sqrt(tr.mean() / va.mean())),
                                  rmse_hold_z0=float(np.sqrt(hold.mean())), rmse_train_mean=float(np.sqrt(tz2.mean())),
                                  **{f"rmse_h{lo}_{hi}": float(np.sqrt(e[:, lo - 1:hi].mean())) for lo, hi in Z.HORIZON_BINS},
                                  **{f"rmse_hold_h{lo}_{hi}": float(np.sqrt(hold[:, lo - 1:hi].mean())) for lo, hi in Z.HORIZON_BINS},
                                  corr_Lz_EC=float(np.corrcoef(e.mean(1), D[f"{m}|pred"].astype(np.float64).mean(1))[0, 1])))
                curves[f"{ds}|{m}|z_err"] = np.sqrt(e.mean(0)); curves[f"{ds}|{m}|z_err_train"] = np.sqrt(tr.mean(0)); curves[f"{ds}|{m}|z_err_val"] = np.sqrt(va.mean(0))
                # decoder inputs
                for inp in ["oracle", "pred", "noisy_gauss", "noisy_perm"] + [f"zhat_{m2}" for m2 in zmodels if m2 != m] + ["pred|train", "oracle|train"]:
                    k = f"{m}|{inp}"
                    if k not in D:
                        continue
                    v = D[k].astype(np.float64)
                    if inp.endswith("|train"):
                        drows.append(dict(dataset=ds, decoder=m, input=inp.replace("|train", ""), split="train (160 sequences)", E_C=float(v.mean()), ci_lo=np.nan, ci_hi=np.nan,
                                          **{f"h{lo}_{hi}": float(v[:, lo - 1:hi].mean()) for lo, hi in Z.HORIZON_BINS}))
                        continue
                    mu, lo_, hi_ = Z.cluster_bootstrap(v.mean(1), take, n_boot=Z.N_BOOT)
                    drows.append(dict(dataset=ds, decoder=m, input=inp, split="test", E_C=mu, ci_lo=lo_, ci_hi=hi_, **{f"h{lo}_{hi}": float(v[:, lo - 1:hi].mean()) for lo, hi in Z.HORIZON_BINS}))
                    curves[f"{ds}|{m}|dec_{inp}"] = v.mean(0)
                pe = lambda k: D[k].astype(np.float64).mean(1)
                for a_in, b_in, q in (("pred", "oracle", "cost of the predicted latent: predicted z vs teacher z through the same decoder"),
                                      ("noisy_gauss", "pred", "matched Gaussian perturbation of the teacher z vs the predicted z (same decoder, same error mean and covariance)"),
                                      ("noisy_perm", "pred", "another sequence's error trajectory added to the teacher z vs the predicted z (same decoder)"),
                                      ("noisy_gauss", "oracle", "matched Gaussian perturbation vs the clean teacher z")):
                    pair_rows.append(dict(dataset=ds, a=f"D_{m}({a_in})", b=f"D_{m}({b_in})", question=q, metric="E_C", **verdict("E_C", pe(f"{m}|{a_in}"), pe(f"{m}|{b_in}"), take)))
            curves[f"{ds}|hold_z0|z_err"] = np.sqrt(hold.mean(0)); curves[f"{ds}|train_mean|z_err"] = np.sqrt(tz2.mean(0))
            for a, b in (("M2", "M1"), ("M3", "M2"), ("M3", "M1")):
                if a in zmodels and b in zmodels:
                    v = verdict("L_z", D[f"{a}|zerr2|test"].astype(np.float64).mean(1), D[f"{b}|zerr2|test"].astype(np.float64).mean(1), take)
                    pair_rows.append(dict(dataset=ds, a=a, b=b, question="z prediction (test L_z, standardised teacher coordinates)", metric="L_z", **v))
                    if (a, b) == ("M2", "M1"):
                        zpair = v
                    # the two decoders on the SAME teacher latent and on the SAME perturbed teacher latents (perturbation matched to the error of zsrc)
                    v = verdict("E_C", D[f"{a}|oracle"].astype(np.float64).mean(1), D[f"{b}|oracle"].astype(np.float64).mean(1), take)
                    pair_rows.append(dict(dataset=ds, a=f"D_{a}(oracle)", b=f"D_{b}(oracle)", question="same teacher latent through both decoders", metric="E_C", **v))
                    for kind in ("noisy_gauss", "noisy_perm"):
                        for zsrc in (b, a):
                            ka = f"{a}|{kind}" if zsrc == a else f"{a}|{kind}_{zsrc}"; kb = f"{b}|{kind}" if zsrc == b else f"{b}|{kind}_{zsrc}"
                            if ka in D and kb in D:
                                v = verdict("E_C", D[ka].astype(np.float64).mean(1), D[kb].astype(np.float64).mean(1), take)
                                pair_rows.append(dict(dataset=ds, a=f"D_{a}({kind} of {zsrc})", b=f"D_{b}({kind} of {zsrc})", question="same perturbed teacher latent through both decoders", metric="E_C", **v))
                    # decoders on the SAME predicted z (swap): robustness of the decoder, separated from the quality of the latent
                    for zsrc in (b, a):
                        ka = f"{a}|pred" if zsrc == a else f"{a}|zhat_{zsrc}"; kb = f"{b}|pred" if zsrc == b else f"{b}|zhat_{zsrc}"
                        v = verdict("E_C", D[ka].astype(np.float64).mean(1), D[kb].astype(np.float64).mean(1), take)
                        pair_rows.append(dict(dataset=ds, a=f"D_{a}(zhat_{zsrc})", b=f"D_{b}(zhat_{zsrc})", question="same predicted z through both decoders", metric="E_C", **v))
                        if (a, b) == ("M2", "M1"):
                            swap[f"zhat_{zsrc}"] = v
        # ------------------------------------------------------------- decision
        P = pd.DataFrame(pair_rows); P = P[P.dataset == ds] if len(P) else P
        def get(a, b, metric, P=P):
            r = P[(P.a == a) & (P.b == b) & (P.metric == metric)] if len(P) else P
            return r.iloc[0] if len(r) else None
        if get("M2", "M1", "E_C") is not None and get("M2", "M0", "E_C") is not None:
            rows_ = decide(ds, get, zpair, swap)
            pert = P[(P.question == "same perturbed teacher latent through both decoders") & P.a.astype(str).str.startswith("D_M2(")] if len(P) else P
            if len(pert):                                                 # reported next to the pre-registered swap test, not part of it
                for r_ in rows_:
                    if r_["question"] == "Is the decoder more robust to predicted-z error?":
                        r_["evidence"] += (f"; not part of the pre-registered test: on identical randomly perturbed teacher latents D_M2 is {pert.rel_improvement.min() * 100:.1f} to {pert.rel_improvement.max() * 100:.1f} % lower than D_M1 "
                                           f"({int((pert.verdict == 'better').sum())} of {len(pert)} comparisons better by the rule)")
            dec_rows.extend(rows_)
    # ------------------------------------------------------------- held-out adaptation probe (zj_probe.py; post-hoc)
    prows = []
    for ds in Z.DATASETS:
        for m in Z.Z_MODELS:
            pp = Z.ds_out(ds) / "probe" / f"probe_{m}.npz"
            if not pp.exists():
                continue
            z = np.load(pp, allow_pickle=True); js = Z.read_json(pp.with_suffix(".json")); take = z["take"].astype(str)
            assert np.array_equal(z["example"], load_metrics(ds, "GT")["example"])
            ad, un = z["E_C_adapted"].astype(np.float64), z["E_C_unadapted"].astype(np.float64)
            v = verdict("E_C", ad, un, take); row = dict(dataset=ds, model=m, **{k: js[k] for k in ("n_val", "n_val_takes", "s_star", "oof_val_E_C_unadapted", "oof_val_E_C_best", "oof_val_gain_rel", "test_E_C_unadapted",
                                                                                                    "test_E_C_adapted", "test_gain_rel", "test_E_C_teacher_z_unadapted", "test_E_C_teacher_z_adapted")},
                                                    test_diff=v["diff"], test_lo=v["ci_lo"], test_hi=v["ci_hi"], test_verdict=v["verdict"],
                                                    oof_val_E_C_step100=float(z["oof_curve"][list(z["oof_steps"]).index(100)]), oof_val_E_C_last=float(z["oof_curve"][-1]),
                                                    fit_fold_sizes=" / ".join(str(int((z["val_fold"] != k).sum())) for k in (0, 1)))
            for ref in ("M0", "M1"):
                vr = verdict("E_C", ad, per_example(load_metrics(ds, ref), "E_C"), take)
                row.update({f"vs_{ref}_diff": vr["diff"], f"vs_{ref}_lo": vr["ci_lo"], f"vs_{ref}_hi": vr["ci_hi"], f"vs_{ref}_verdict": vr["verdict"], f"vs_{ref}_rel": vr["rel_improvement"]})
                pair_rows.append(dict(dataset=ds, a=f"PROBE_{m}", b=ref, question="held-out adaptation probe (decoder fine-tuned on validation latents) vs a compared model", metric="E_C", **vr))
            pair_rows.append(dict(dataset=ds, a=f"PROBE_{m}", b=m, question="held-out adaptation probe vs the same model un-adapted", metric="E_C", **v))
            sp = pp.with_name(f"shrink_{m}.npz")
            if sp.exists():                                               # five-parameter variant: one shrinkage factor of z_hat per horizon bin
                zs_ = np.load(sp, allow_pickle=True); sj = Z.read_json(sp.with_suffix(".json")); sh = zs_["E_C_shrunk"].astype(np.float64)
                vs_ = verdict("E_C", sh, zs_["E_C_unadapted"].astype(np.float64), take); v0 = verdict("E_C", sh, per_example(load_metrics(ds, "M0"), "E_C"), take)
                row.update(shrink_factors=" / ".join(f"{sj['factors'][f'{lo}_{hi}']:.1f}" for lo, hi in Z.HORIZON_BINS), shrink_val_E_C_unadapted=sj["val_E_C_unadapted"], shrink_val_E_C=sj["val_E_C_shrunk"],
                           shrink_test_E_C=sj["test_E_C_shrunk"], shrink_test_diff=vs_["diff"], shrink_test_lo=vs_["ci_lo"], shrink_test_hi=vs_["ci_hi"], shrink_test_rel=vs_["rel_improvement"], shrink_test_verdict=vs_["verdict"],
                           shrink_vs_M0_diff=v0["diff"], shrink_vs_M0_lo=v0["ci_lo"], shrink_vs_M0_hi=v0["ci_hi"], shrink_vs_M0_verdict=v0["verdict"])
                pair_rows.append(dict(dataset=ds, a=f"SHRINK_{m}", b=m, question="latent shrinkage fitted on validation (one factor per horizon bin) vs the same model", metric="E_C", **vs_))
                pair_rows.append(dict(dataset=ds, a=f"SHRINK_{m}", b="M0", question="latent shrinkage fitted on validation vs the direct model", metric="E_C", **v0))
            prows.append(row)
    pd.DataFrame(prows).to_csv(Z.OUT / "heldout_probe.csv", index=False)
    # ------------------------------------------------------------- write
    T = pd.DataFrame(trows); T.to_csv(Z.OUT / "training_summary.csv", index=False)
    mt = []
    for m, c in Z.MODELS.items():
        r = dict(model=m, role=c["label"], intermediate="none (dense rho)" if not c["joint"] else "z (64)", trained="reused Stage-2 checkpoint" if c["reuse"] else "this study",
                 output_head="MLP 768-2048-2048-512 (zero-initialised: starts as persistence)" if not c["joint"] else "linear 768-64 (zero-initialised: starts at the train-mean z)",
                 decoder="—" if not c["joint"] else ("Stage-1 D_z, 4 point-token blocks, width 256 (loaded, fine-tuned)" if m == "M1" else "Stage-1 D_z (loaded) + 2 identity-initialised blocks = 6 blocks, width 256 (all trained)"),
                 loss="L_C" if not c["joint"] else ("L_C(D(z_hat)) + 1.0 L_z" if not c["dual"] else "L_C(D(z_hat)) + 0.25 L_C(D(z*)) + 1.0 L_z"),
                 selection="validation L_C" if (not c["joint"] or m != "M1") else "validation L_C + L_z")
        for ds in Z.DATASETS:
            t = T[(T.dataset == ds) & (T.model == m)] if len(T) else T
            if len(t):
                r.update({f"n_params_{ds}_total": int(t.n_params_total.iloc[0]), f"n_params_{ds}_backbone": int(t.n_params_backbone.iloc[0]), f"n_params_{ds}_decoder_or_head": int(t.n_params_decoder.iloc[0])})
        if any(k.startswith("n_params") for k in r):
            mt.append(r)
    pd.DataFrame(mt).to_csv(Z.OUT / "model_table.csv", index=False)
    acc = pd.DataFrame(acc_rows); acc.to_csv(Z.OUT / "main_metrics.csv", index=False)
    acc[acc.metric.isin(STRUCT_TABLE)].to_csv(Z.OUT / "structural_metrics.csv", index=False)
    acc[acc.metric.isin(WRENCH_TABLE)].to_csv(Z.OUT / "wrench_metrics.csv", index=False)
    acc[acc.metric.isin(S2.TEMPORAL + ["jitter_dev_dense", "jitter_dev_r2", "jitter_ratio_dense", "jitter_ratio_r2"])].to_csv(Z.OUT / "temporal_stability.csv", index=False)
    pd.DataFrame(zrows).to_csv(Z.OUT / "z_metrics.csv", index=False); pd.DataFrame(drows).to_csv(Z.OUT / "decoder_diagnostic.csv", index=False)
    pd.DataFrame(hrows).to_csv(Z.OUT / "horizon_metrics.csv", index=False); pd.DataFrame(pair_rows).to_csv(Z.OUT / "paired_comparisons.csv", index=False)
    Dd = pd.DataFrame(dec_rows); Dd.to_csv(Z.OUT / "decision_summary.csv", index=False)
    np.savez_compressed(Z.OUT / "temporal_curves.npz", **curves)
    teacher = {ds: S2.read_json(S2.teacher_cache_path(ds).with_suffix(".json")) for ds in Z.DATASETS}
    Z.write_json(Z.OUT / "experiment_config.json", dict(
        question="Did adapting the decoder (jointly, on predicted z, with more capacity and time) rescue z-mediated temporal generation?",
        models={k: v for k, v in Z.MODELS.items()}, decoder=Z.DECODER, backbone=Z.BACKBONE, train=Z.TRAIN, m0_reuse=Z.M0_REUSE, decision_rule=Z.DECISION, noise=Z.NOISE, protocol=Z.PROTOCOL,
        n_boot=Z.N_BOOT, horizon_bins=Z.HORIZON_BINS, seed=Z.SEED, runs=cfg_runs, teacher=teacher, cases={r["dataset"]: r["answer"] for r in dec_rows if r["question"] == "case"},
        sources=dict(stage2=str(S2.OUT), stage1=str(S2.CF.OUT), z_temporal_diagnostic="/result/uhnam/dexcore/reports/z_temporal_diagnostic"),
        code_dir=str(Z.HERE), code_md5={p.name: Z.md5(p) for p in sorted(Z.HERE.glob("*.py")) + sorted(Z.HERE.glob("*.sh"))}))
    log.info("tables written: %d metric rows, %d paired rows, %d z rows, %d decoder rows, %d decision rows", len(acc_rows), len(pair_rows), len(zrows), len(drows), len(dec_rows))
    if len(Dd):
        log.info("\n%s", Dd[["dataset", "question", "answer"]].to_string())


if __name__ == "__main__":
    main()
