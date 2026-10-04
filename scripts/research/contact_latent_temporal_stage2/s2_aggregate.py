#!/usr/bin/env python
"""Aggregate the per-example metrics of the six runs (and the references) into the report tables.    python s2_aggregate.py
Every number is a mean over the test sequences (one trajectory per sequence under the fixed-s_0 protocol) with a 95 % take-cluster
bootstrap interval (1000 reps); paired comparisons are differences of per-sequence values on identical test sequences with the same
bootstrap; the previous study's D0 (three seeds) enters as the per-sequence seed average.  Writes to OUT: model_table.csv,
training_summary.csv, main_metrics.csv, structural_metrics.csv, wrench_metrics.csv, z_metrics.csv, r_metrics.csv, temporal_stability.csv,
horizon_metrics.csv, temporal_curves.npz, paired_comparisons.csv, decision_summary.csv, experiment_config.json.
"""
from __future__ import annotations

import importlib.util
import json
import logging

import numpy as np
import pandas as pd
import torch

import s2_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2_aggregate")
METRICS = ["E_C", "E_C_last16", "part_hamming", "amount_l1", "amount_rmse", "centroid", "normal", "centroid_both", "normal_both", "both_frac",
           "q_rel_l1", "q_cos", "q_Qerr", "q_rel_l1_vs_grid", "part_hamming_vs_grid", "amount_l1_vs_grid", "centroid_vs_grid", "normal_vs_grid",
           "jitter_dense", "tv_dense", "jitter_r2"]
STRUCT_TABLE = ["part_hamming", "part_macro_f1", "part_macro_auprc", "amount_l1", "amount_rmse", "centroid", "normal", "centroid_both", "normal_both", "both_frac",
                "part_hamming_vs_grid", "amount_l1_vs_grid", "centroid_vs_grid", "normal_vs_grid"]
WRENCH_TABLE = ["q_rel_l1", "q_cos", "q_Qerr", "q_rel_l1_vs_grid"]
CURVES = ["dense", "part", "amount", "centroid", "normal", "wrench", "jitter", "jitter_r2"]
MODEL_ORDER = ["B0", "B1", "B2", "B2_zonly", "D0_prev", "PERSIST", "GT"]
_CACHE = {}


def sat_aggregate_module():
    spec = importlib.util.spec_from_file_location("sat_aggregate", S.SAT_DIR / "aggregate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_metrics(ds, model):
    """Per-example metric arrays of one model (K = 1) or of the previous D0 (seed-averaged per example)."""
    key = (ds, model)
    if key in _CACHE:
        return _CACHE[key]
    if model == "D0_prev":
        zs = [np.load(S.sat_metrics_path(ds, S.SAT.run_name("D0", s)), allow_pickle=True) for s in S.SAT.SEEDS]
        zs = [{k: z[k] for k in z.files} for z in zs]
        out = {k: np.nanmean(np.stack([z[k].astype(np.float64) for z in zs]), 0) for k in zs[0] if isinstance(zs[0][k], np.ndarray) and zs[0][k].dtype.kind in "fiu" and zs[0][k].ndim >= 2 and k not in ("r2_a", "a_true")}
        out.update(example=zs[0]["example"], take=zs[0]["take"], K=1, protocol="A", r2_a=zs[0]["r2_a"], a_true=zs[0]["a_true"], tp=np.mean([z["tp"] for z in zs], 0),
                   fp=np.mean([z["fp"] for z in zs], 0), fn=np.mean([z["fn"] for z in zs], 0), r2_a_soft=np.mean([z["r2_a_soft"].astype(np.float32) for z in zs], 0), n_seeds=len(zs), _seeds=zs)
        _CACHE[key] = out; return out
    name = {"GT": "GT", "PERSIST": "PERSIST", "B2_zonly": S.run_name("B2") + "_zonly"}.get(model, S.run_name(model))
    p = S.metrics_path(ds, name)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True); out = {k: z[k] for k in z.files}; out["n_seeds"] = 1
    _CACHE[key] = out; return out


def per_example(z, metric):
    v = z[metric].astype(np.float64)
    return v[:, 0] if v.ndim == 2 else v


def ci_row(v, take, **extra):
    m, lo, hi = S.cluster_bootstrap(v, take, n_boot=S.N_BOOT)
    return dict(**extra, value=m, ci_lo=lo, ci_hi=hi, n_examples=int(np.isfinite(v).sum()))


def verdict(metric, va, vb, take):
    """Paired take-cluster bootstrap of mean(a - b) on identical sequences; 'better' / 'worse' / 'similar' by the pre-registered rule."""
    ok = np.isfinite(va) & np.isfinite(vb)
    d, lo, hi = S.paired_cluster_bootstrap(va[ok], vb[ok], take[ok], n_boot=S.N_BOOT)
    mb = float(np.mean(vb[ok])); rel = d / max(abs(mb), 1e-9)
    higher = metric in S.HIGHER_BETTER
    improvement = rel if higher else -rel                                    # > 0: a better than b
    if abs(rel) >= S.DECISION["rel_min"] and (lo > 0 or hi < 0):
        v = "better" if ((d > 0) == higher) else "worse"
    else:
        v = "similar"
    return dict(value_a=float(np.mean(va[ok])), value_b=mb, diff=d, ci_lo=lo, ci_hi=hi, rel_change=rel, rel_improvement=improvement, verdict=v, n=int(ok.sum()))


def z_diagnostics(ds, model, stats, zcache):
    """Test L_z and its horizon profile of a z-mediated model; references: hold z*_0, train mean."""
    p = S.preds_path(ds, S.run_name(model))
    if not p.exists():
        return None
    z = np.load(p); ex = z["example"]
    mu, sd = stats["z_mu"], stats["z_sd"]
    zs = (zcache["z"][ex] - mu) / sd                                           # (B, 64, dz) standardised teacher
    zh = z["z_hat"].astype(np.float64)                                          # (B, 63, dz)
    err = ((zh - zs[:, 1:]) ** 2).mean(-1)                                      # (B, 63)
    hold = ((zs[:, :1] - zs[:, 1:]) ** 2).mean(-1); mean_ref = (zs[:, 1:] ** 2).mean(-1)
    var_d = zs[:, 1:].reshape(-1, zs.shape[-1]).var(0)
    r2_dim = 1 - ((zh - zs[:, 1:]) ** 2).reshape(-1, zs.shape[-1]).mean(0) / np.maximum(var_d, 1e-9)
    return dict(example=ex, err=err, hold=hold, mean_ref=mean_ref, r2_dim=r2_dim, zh=zh, zs=zs[:, 1:])


def r_diagnostics(ds, stats):
    p = S.preds_path(ds, S.run_name("B2"))
    if not p.exists():
        return None
    z = np.load(p); s0 = z["s0"].astype(np.float64)
    delta = z["delta"].astype(np.float64); C_bar = z["C_bar"][:, 0, 1:].astype(np.float64); pred = z["pred"][:, 0, 1:].astype(np.float64)
    r = z["r_hat"].astype(np.float64)
    dn = np.linalg.norm(delta, axis=-1)                                        # (B, 63)
    bn = np.linalg.norm(C_bar - s0[:, None], axis=-1)
    rr = r.reshape(-1, r.shape[-1]); cov = np.cov(rr.T); ev = np.linalg.eigvalsh(cov); ev = np.clip(ev, 0, None)
    eff_rank = float(np.exp(-(ev / ev.sum() * np.log(np.maximum(ev / ev.sum(), 1e-12))).sum())) if ev.sum() > 0 else 0.0
    # frame-to-frame persistence of r_hat (autocorrelation at h = 8 over the predicted trajectories)
    rc = r - rr.mean(0); num = (rc[:, 8:] * rc[:, :-8]).sum(); den = np.sqrt((rc[:, 8:] ** 2).sum() * (rc[:, :-8] ** 2).sum())
    return dict(example=z["example"], delta_norm=dn.mean(1), bar_norm=bn.mean(1), ratio=(dn / np.maximum(bn, 1e-9)).mean(1), r_std=rr.std(0), eff_rank=eff_rank, acf8=float(num / max(den, 1e-12)),
                delta_abs_mean=float(np.abs(delta).mean()), delta_curve=dn.mean(0), consistent=bool(np.abs(pred - (C_bar + delta)).max() < 5e-3))


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    AG = sat_aggregate_module()
    acc_rows, pair_rows, zrows, rrows, trows, hrows, stab_rows, dec_rows, curves, pcounts, cfg_json = [], [], [], [], [], [], [], [], {}, {}, {}
    for ds in S.DATASETS:
        # ------------------------------------------------------------- training summary / parameter counts
        for model in S.MODELS:
            ck_p = S.ckpt_path(ds, S.run_name(model))
            if not ck_p.exists():
                continue
            ck = torch.load(ck_p, map_location="cpu", weights_only=False)
            pcounts.setdefault(model, {}).update({f"{ds}_{k}": v for k, v in ck["n_params"].items()})
            lg = pd.read_csv(S.train_log_path(ds, S.run_name(model)))
            at_best = lg[lg.step == ck["best_step"]].iloc[0].to_dict() if (lg.step == ck["best_step"]).any() else {}
            insp = pd.DataFrame(ck.get("inspect", []))
            gn = {}
            if len(insp):
                for st in (1000, 5000):
                    r = insp[insp.step == st]
                    if len(r):
                        r = r.iloc[0]; g = {c[6:]: float(r[c]) for c in insp.columns if c.startswith("gnorm_")}
                        gn[f"gnorm_ratio_max_min_{st}"] = max(g.values()) / max(min(g.values()), 1e-12); gn[f"gnorms_{st}"] = json.dumps({k: round(v, 4) for k, v in g.items()})
            v20 = lg[lg.step == 20000]["val"].iloc[0] if (lg.step == 20000).any() else np.nan
            # the selection criterion is the full objective; record where the dense validation term alone would have selected (review item)
            dense_col = "val_L_full" if "val_L_full" in lg else "val_L_C"
            dense_sel = dict(val_dense_at_best=float(lg.loc[lg.step == ck["best_step"], dense_col].iloc[0]) if (lg.step == ck["best_step"]).any() else np.nan,
                             min_val_dense=float(lg[dense_col].min()), min_val_dense_step=int(lg.loc[lg[dense_col].idxmin(), "step"]), val_E_C_at_best=float(lg.loc[lg.step == ck["best_step"], "val_E_C"].iloc[0]) if (lg.step == ck["best_step"]).any() else np.nan,
                             min_val_E_C=float(lg["val_E_C"].min()))
            trows.append(dict(dataset=ds, model=model, n_params_total=ck["n_params"]["total"], n_params_backbone=ck["n_params"]["backbone"], steps=ck["steps"], best_step=ck["best_step"], best_val=ck["best_val"],
                              val_init=ck["val_init"], val_at_20k=v20, final_val=lg["val"].iloc[-1], stopped_by=ck["stopped_by"], converged=ck["converged"], seconds=ck["seconds"], hours=ck["seconds"] / 3600,
                              lambda_z=ck["lambdas"]["z"], lambda_c=ck["lambdas"]["c"], lambda_res=ck["lambdas"]["res"], batch=ck["cfg"]["batch"], n_dec_frames=ck["cfg"]["n_dec_frames"],
                              **{f"best_{k}": v for k, v in at_best.items() if k.startswith("val_")}, **dense_sel, **gn))
            cfg_json.setdefault("runs", {})[f"{ds}/{model}"] = dict(ckpt=str(ck_p), md5=S.md5(ck_p), best_step=int(ck["best_step"]), steps=int(ck["steps"]), stopped_by=ck["stopped_by"],
                                                                    teacher=ck["teacher"], n_params=ck["n_params"], cfg=ck["cfg"], lambdas=ck["lambdas"])
        # ------------------------------------------------------------- per-model metrics
        gt = load_metrics(ds, "GT")
        if gt is None:
            log.warning("%s: no GT metrics yet", ds); continue
        take = gt["take"].astype(str); ex = gt["example"]
        Z = {m: load_metrics(ds, m) for m in MODEL_ORDER}
        Z = {m: z for m, z in Z.items() if z is not None}
        for m, z in Z.items():
            assert np.array_equal(z["example"], ex), (ds, m, "example mismatch")
            for metric in METRICS:
                if metric in z:
                    acc_rows.append(ci_row(per_example(z, metric), take, dataset=ds, model=m, metric=metric))
            for r in AG.macro_scores(list(enumerate(z["_seeds"])) if "_seeds" in z else [(0, z)], take, "K1"):      # previous D0: mean of the per-seed scores
                acc_rows.append(dict(dataset=ds, model=m, metric=r["metric"], value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_examples=len(take)))
            # temporal stability vs the GT's own frame-to-frame change (paired per sequence)
            for key in ("jitter_dense", "jitter_r2"):
                dev = np.abs(per_example(z, key) - per_example(gt, key))
                acc_rows.append(ci_row(dev, take, dataset=ds, model=m, metric=f"jitter_dev_{key[7:]}"))
                acc_rows.append(ci_row(per_example(z, key) / np.maximum(per_example(gt, key), 1e-9), take, dataset=ds, model=m, metric=f"jitter_ratio_{key[7:]}"))
            # curves (mean over examples per frame) and horizon bins
            for c in CURVES:
                k = f"curve_{c}"
                if k in z:
                    cur = z[k].astype(np.float64); cur = cur[:, 0] if cur.ndim == 3 else cur
                    curves[f"{ds}|{m}|{c}"] = np.nanmean(cur, 0)
                    if c in ("dense", "part", "amount", "centroid", "normal", "wrench"):
                        for lo, hi in S.HORIZON_BINS:
                            v = np.nanmean(cur[:, lo:hi + 1], 1); r = ci_row(v, take, dataset=ds, model=m, curve=c, h_lo=lo, h_hi=hi); hrows.append(r)
        # ------------------------------------------------------------- paired comparisons
        for a, b, q in S.PAIRS + [("B0", "D0_prev", "scaled direct model vs the previous D0 (3-seed average)"), ("B1", "D0_prev", "z-mediated vs the previous (early-stopped, 25 M) direct model"),
                                  ("B2", "D0_prev", "z + r mediated vs the previous direct model"), ("B2_zonly", "B1", "B2's z-only path vs B1"), ("B2", "B2_zonly", "the r correction of B2")]:
            if a not in Z or b not in Z:
                continue
            for metric in METRICS + ["jitter_dev_dense", "jitter_dev_r2"]:
                if metric.startswith("jitter_dev"):
                    key = "jitter_" + metric[11:]
                    va = np.abs(per_example(Z[a], key) - per_example(gt, key)); vb = np.abs(per_example(Z[b], key) - per_example(gt, key))
                elif metric in Z[a] and metric in Z[b]:
                    va, vb = per_example(Z[a], metric), per_example(Z[b], metric)
                else:
                    continue
                pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=metric, **verdict(metric, va, vb, take)))
        # ------------------------------------------------------------- z diagnostics
        zc = np.load(S.teacher_cache_path(ds), allow_pickle=True)
        zd = {}
        for m in ("B1", "B2"):
            ck_p = S.ckpt_path(ds, S.run_name(m))
            if not ck_p.exists() or not S.preds_path(ds, S.run_name(m)).exists():
                continue
            st = torch.load(ck_p, map_location="cpu", weights_only=False)["stats"]
            d = z_diagnostics(ds, m, st, zc); zd[m] = d
            assert np.array_equal(d["example"], ex)
            row = dict(dataset=ds, model=m, **{k: v for k, v in zip(("L_z_test", "L_z_lo", "L_z_hi"), S.cluster_bootstrap(d["err"].mean(1), take, n_boot=S.N_BOOT))},
                       L_z_hold_z0=float(d["hold"].mean()), L_z_train_mean=float(d["mean_ref"].mean()), rmse_test=float(np.sqrt(d["err"].mean())),
                       r2_per_dim_mean=float(d["r2_dim"].mean()), r2_per_dim_min=float(d["r2_dim"].min()), r2_per_dim_max=float(d["r2_dim"].max()),
                       **{f"rmse_h{lo}_{hi}": float(np.sqrt(d["err"][:, lo - 1:hi].mean())) for lo, hi in S.HORIZON_BINS},
                       **{f"rmse_hold_h{lo}_{hi}": float(np.sqrt(d["hold"][:, lo - 1:hi].mean())) for lo, hi in S.HORIZON_BINS},
                       corr_Lz_EC=float(np.corrcoef(d["err"].mean(1), per_example(Z[m], "E_C"))[0, 1]) if m in Z else np.nan)
            zrows.append(row)
            curves[f"{ds}|{m}|z_err"] = np.sqrt(d["err"].mean(0)); curves[f"{ds}|hold_z0|z_err"] = np.sqrt(d["hold"].mean(0)); curves[f"{ds}|train_mean|z_err"] = np.sqrt(d["mean_ref"].mean(0))
        if "B1" in zd and "B2" in zd:
            v = verdict("L_z", zd["B2"]["err"].mean(1), zd["B1"]["err"].mean(1), take)
            pair_rows.append(dict(dataset=ds, a="B2", b="B1", question="z quality of B2 vs B1 (test L_z)", metric="L_z", **v))
            zrows.append(dict(dataset=ds, model="B2_vs_B1", L_z_test=v["diff"], L_z_lo=v["ci_lo"], L_z_hi=v["ci_hi"], rmse_test=np.nan, ratio_B2_B1=v["value_a"] / max(v["value_b"], 1e-9)))
        # ------------------------------------------------------------- r diagnostics (B2)
        if "B2" in Z and "B2_zonly" in Z:
            rd = r_diagnostics(ds, None)
            gain = per_example(Z["B2_zonly"], "E_C") - per_example(Z["B2"], "E_C")
            g, glo, ghi = S.cluster_bootstrap(gain, take, n_boot=S.N_BOOT)
            rrows.append(dict(dataset=ds, model="B2", E_C_full=float(np.mean(per_example(Z["B2"], "E_C"))), E_C_zonly=float(np.mean(per_example(Z["B2_zonly"], "E_C"))), E_C_gain_from_r=g, gain_lo=glo, gain_hi=ghi,
                              frac_sequences_improved=float((gain > 0).mean()), delta_norm_mean=float(rd["delta_norm"].mean()), bar_minus_s0_norm_mean=float(rd["bar_norm"].mean()),
                              delta_over_bar_ratio=float(rd["ratio"].mean()), delta_abs_mean_raw=rd["delta_abs_mean"], r_std_mean=float(rd["r_std"].mean()), r_std_min=float(rd["r_std"].min()), r_std_max=float(rd["r_std"].max()),
                              r_eff_rank=rd["eff_rank"], r_acf_h8=rd["acf8"]))
            curves[f"{ds}|B2|delta_norm"] = rd["delta_curve"]
        # ------------------------------------------------------------- decision
        P = pd.DataFrame(pair_rows); P = P[P.dataset == ds]
        get = lambda a, b, metric: P[(P.a == a) & (P.b == b) & (P.metric == metric)].iloc[0] if len(P[(P.a == a) & (P.b == b) & (P.metric == metric)]) else None
        if get("B1", "B0", "E_C") is not None:
            dec_rows.extend(decide(ds, get, zd))
    # ------------------------------------------------------------- write
    mt = S.model_table(pcounts); mt.to_csv(S.OUT / "model_table.csv", index=False)
    pd.DataFrame(trows).to_csv(S.OUT / "training_summary.csv", index=False)
    acc = pd.DataFrame(acc_rows); acc.to_csv(S.OUT / "main_metrics.csv", index=False)
    acc[acc.metric.isin(STRUCT_TABLE)].to_csv(S.OUT / "structural_metrics.csv", index=False)
    acc[acc.metric.isin(WRENCH_TABLE)].to_csv(S.OUT / "wrench_metrics.csv", index=False)
    acc[acc.metric.isin(S.TEMPORAL + ["jitter_dev_dense", "jitter_dev_r2", "jitter_ratio_dense", "jitter_ratio_r2"])].to_csv(S.OUT / "temporal_stability.csv", index=False)
    pd.DataFrame(zrows).to_csv(S.OUT / "z_metrics.csv", index=False); pd.DataFrame(rrows).to_csv(S.OUT / "r_metrics.csv", index=False)
    pd.DataFrame(hrows).to_csv(S.OUT / "horizon_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(S.OUT / "paired_comparisons.csv", index=False)
    D = pd.DataFrame(dec_rows); D.to_csv(S.OUT / "decision_summary.csv", index=False)
    np.savez_compressed(S.OUT / "temporal_curves.npz", **curves)
    cfg_json.update(models=S.MODELS, backbone=S.BACKBONE, train=S.TRAIN, decision_rule=S.DECISION, pairs=S.PAIRS, protocol=S.PROTOCOL, n_boot=S.N_BOOT, horizon_bins=S.HORIZON_BINS,
                    teacher={ds: S.read_json(S.teacher_cache_path(ds).with_suffix(".json")) for ds in S.DATASETS if S.teacher_cache_path(ds).with_suffix(".json").exists()},
                    cases={r["dataset"]: r["answer"] for r in dec_rows if r["question"] == "case"}, metric_sources=dict(previous_D0=str(S.SAT.OUT), stage1=str(S.CF.OUT)))
    S.write_json(S.OUT / "experiment_config.json", cfg_json)
    log.info("tables written: %d metric rows, %d paired rows, %d decision rows", len(acc_rows), len(pair_rows), len(dec_rows))
    if len(D):
        log.info("\n%s", D[["dataset", "question", "answer"]].to_string())


def decide(ds, get, zd):
    """The pre-registered decision rule (s2_common.DECISION) applied to the paired verdicts."""
    R = S.DECISION; rows = []
    fmt = lambda r: f"{r.a} {r.value_a:.3f} vs {r.b} {r.value_b:.3f}, diff {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.rel_improvement * 100:+.1f} %)"
    def block(a, b, metrics):
        vs = {m: get(a, b, m) for m in metrics}; vs = {m: v for m, v in vs.items() if v is not None}
        nb = sum(v.verdict == "better" for v in vs.values()); nw = sum(v.verdict == "worse" for v in vs.values())
        if nw > nb:
            ans = "worse"
        elif nb >= R["struct_majority"] and nw == 0:
            ans = "yes"
        elif nb >= 1 and nw <= 1:
            ans = "partial"
        else:
            ans = "no"
        return ans, nb, nw, vs
    # Q1 dense
    e = get("B1", "B0", "E_C"); q1 = {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[e.verdict]
    rows.append(dict(dataset=ds, question="Does z mediation improve dense temporal generation?", answer=q1, evidence=f"E_C: {fmt(e)}"))
    # Q2 structure / wrench
    q2, nb2, nw2, vs2 = block("B1", "B0", S.STRUCT_DECISION)
    rows.append(dict(dataset=ds, question="Does z mediation improve structural / wrench quality?", answer=q2, evidence=f"{nb2} better / {nw2} worse of {len(vs2)}: " + "; ".join(f"{m} {v.verdict} ({v.rel_improvement * 100:+.1f} %)" for m, v in vs2.items())))
    # Q3 long-horizon stability
    stab_metrics = list(R["stability_metrics"])
    q3, nb3, nw3, vs3 = block("B1", "B0", stab_metrics)
    q3 = {"yes": "yes", "partial": "partial", "no": "no", "worse": "worse"}[q3] if nb3 + nw3 > 0 else "no"
    if q3 == "yes" and nb3 < len(vs3) and nb3 < 2:
        q3 = "partial"
    rows.append(dict(dataset=ds, question="Does z improve long-horizon stability?", answer=q3, evidence="; ".join(f"{m} {v.verdict} ({v.rel_improvement * 100:+.1f} %)" for m, v in vs3.items())))
    # Q4 r dense
    e4 = get("B2", "B1", "E_C"); q4 = {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[e4.verdict]
    rows.append(dict(dataset=ds, question="Does r improve dense reconstruction beyond z?", answer=q4, evidence=f"E_C: {fmt(e4)}"))
    # Q5 r preserves z structure
    _, nb5, nw5, vs5 = block("B2", "B1", S.STRUCT_DECISION)
    lz = get("B2", "B1", "L_z"); z_ok = (lz is not None) and (lz.value_a <= (1 + R["z_quality_tol"]) * lz.value_b)
    struct_ok = nw5 == 0
    q5 = "yes" if (struct_ok and z_ok) else ("partial" if (struct_ok or z_ok) else "no")
    rows.append(dict(dataset=ds, question="Does r preserve z-defined structure?", answer=q5,
                     evidence=f"structural metrics: {nb5} better / {nw5} worse of {len(vs5)}; test L_z B2 {lz.value_a:.3f} vs B1 {lz.value_b:.3f} (ratio {lz.value_a / max(lz.value_b, 1e-9):.2f}, tol 1.10)" if lz is not None else "no z diagnostics"))
    # case and best-supported model
    b1_better = (q1 == "yes") or (q2 == "yes" and e.verdict != "worse") or (q3 == "yes" and e.verdict != "worse" and q2 != "worse")
    b1_worse = (e.verdict == "worse" and q2 != "yes") or (q2 == "worse")
    b2_better = (q4 == "yes") and (q5 != "no")
    if q1 == "yes" and q4 == "yes" and q5 == "yes":
        case = "A"
    elif (q2 == "yes" or q3 == "yes") and q4 == "yes" and struct_ok:
        case = "C"
    elif b1_better and q4 != "yes":
        case = "B"
    elif b1_worse:
        case = "E"
    elif not b1_better:
        case = "D"
    else:
        case = "A" if b2_better else "B"
    e20 = get("B2", "B0", "E_C"); _, nb20, nw20, _ = block("B2", "B0", S.STRUCT_DECISION)
    if b2_better and (e20 is not None and e20.verdict != "worse") and nw20 == 0:
        best = "B2"
    elif b1_better:
        best = "B1"
    elif b2_better:
        best = "B2 (r helps dense generation; z mediation alone does not beat B0)"
    else:
        best = "B0"
    rows.append(dict(dataset=ds, question="Which model is best supported: B0 / B1 / B2?", answer=best,
                     evidence=f"case {case}: B1 vs B0 {'better' if b1_better else ('worse' if b1_worse else 'similar')}; B2 vs B1 dense {q4}, structure preserved {q5}; B2 vs B0 E_C {e20.verdict if e20 is not None else '-'}, structure {nb20} better / {nw20} worse"))
    rows.append(dict(dataset=ds, question="case", answer=case, evidence="A strong factorised; B z useful, r not needed; C z structure + r dense; D no benefit from z; E z hurts"))
    return rows


if __name__ == "__main__":
    main()
