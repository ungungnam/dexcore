#!/usr/bin/env python
"""Aggregate the 2 x 2 design into the report tables.    python zf_aggregate.py
Every number is a mean over the test sequences (one trajectory per sequence, fixed GT s_0) with a 95 % take-cluster bootstrap interval
(1000 reps).  Paired comparisons and the factorial contrasts are computed on per-sequence values of identical test sequences:
    stateful main effect  = mean(M10 - M00, M11 - M01)      decoder main effect = mean(M01 - M00, M11 - M10)      interaction = (M11 - M10) - (M01 - M00)
and read by the pre-registered rule of zf_common.DECISION (>= 2 % of the M00 mean and CI excluding 0).  M00 and D0 rows come from the
previous studies' metric files (same evaluation code).  Writes to OUT: model_table.csv, training_summary.csv, taco_metrics.csv,
arctic_metrics.csv, paired_comparisons.csv, factorial_effects.csv, latent_metrics.csv, horizon_metrics.csv, early_frame_metrics.csv,
direct_comparison.csv, decision_summary.csv, temporal_curves.npz, experiment_config.json.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import torch

import zf_common as Z
from s2_aggregate import METRICS, ci_row, per_example, sat_aggregate_module, verdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zf_aggregate")
S2, ZJ = Z.S2, Z.ZJ
MODEL_ORDER = ["M00", "M01", "M10", "M11", "D0", "D0r", "PERSIST", "GT"]
PAIRS = [("M01", "M00", "decoder effect, whole-sequence z"), ("M11", "M10", "decoder effect, stateful z"), ("M10", "M00", "stateful effect, absolute decoder"),
         ("M11", "M01", "stateful effect, s_0-preserving decoder"), ("M11", "M00", "both changes against the reference z model"),
         ("M00", "D0", "z model vs direct"), ("M01", "D0", "z model vs direct"), ("M10", "D0", "z model vs direct"), ("M11", "D0", "z model vs direct"),
         ("D0r", "D0", "direct model retrained under the 100 k protocol (stopped at the user's request, partial) vs the reused direct model"),
         ("M00", "D0r", "z model vs the partially retrained direct model"), ("M01", "D0r", "z model vs the partially retrained direct model"),
         ("M10", "D0r", "z model vs the partially retrained direct model"), ("M11", "D0r", "z model vs the partially retrained direct model")]
STRUCT_CURVES = ("part", "amount", "centroid", "normal", "wrench")
EXTRA_DENSE = [f"E_C_h{lo}_{hi}" for lo, hi in Z.HORIZON_BINS] + [f"E_C_t{t}" for t in Z.EARLY_FRAMES] + ["E_C_final"]
_CACHE = {}


def metrics_file(ds, key):
    special = {"GT": S2.metrics_path(ds, "GT"), "PERSIST": S2.metrics_path(ds, "PERSIST"), "D0r": Z.metrics_path(ds, "D0r_partial")}
    return special[key] if key in special else Z.model_metrics(ds, key)


def load_metrics(ds, key):
    if (ds, key) not in _CACHE:
        p = metrics_file(ds, key)
        _CACHE[(ds, key)] = None if not p.exists() else {k: v for k, v in np.load(p, allow_pickle=True).items()}
    return _CACHE[(ds, key)]


def curve(z, name):
    c = z[f"curve_{name}"].astype(np.float64)
    return c[:, 0] if c.ndim == 3 else c                                           # (B, 64)


def value(z, gt, metric):
    """Per-sequence values of a metric name used in this study (standard metrics, dense error by bin / frame, deviations of the frame-to-frame change)."""
    if metric.startswith("jitter_dev_"):
        k = "jitter_" + metric[11:]
        return np.abs(per_example(z, k) - per_example(gt, k))
    m = re.fullmatch(r"(E_C|part|amount|centroid|normal|wrench)_h(\d+)_(\d+)", metric)
    if m:
        with np.errstate(all="ignore"):
            return np.nanmean(curve(z, "dense" if m.group(1) == "E_C" else m.group(1))[:, int(m.group(2)):int(m.group(3)) + 1], 1)
    m = re.fullmatch(r"E_C_t(\d+)", metric)
    if m:
        return curve(z, "dense")[:, int(m.group(1))]
    if metric == "E_C_final":
        return curve(z, "dense")[:, Z.T - 1]
    return per_example(z, metric) if metric in z else None


def gpus_of(log_path):
    return ",".join(sorted(set(re.findall(r"=== attempt \d+ on gpu (\d+)", log_path.read_text())))) if log_path.exists() else ""


def training_row(ds, model):
    ck_p = Z.model_ckpt(ds, model) if model != "D0" else S2.ckpt_path(ds, S2.run_name("B0"))
    if not ck_p.exists():
        return None, None
    ck = torch.load(ck_p, map_location="cpu", weights_only=False); lg = pd.read_csv(Z.model_train_log(ds, model))
    at = lambda step, col: float(lg.loc[lg.step == step, col].iloc[0]) if (col in lg and (lg.step == step).any()) else np.nan
    bs = int(ck["best_step"]); pc = ck["n_params"]; reused = model in ("M00", "D0")
    tlog = {"M00": ZJ.OUT / "logs" / "train_{}_M2_seed0.log".format(ds), "D0": S2.OUT / "logs" / f"train_{ds}_B0_seed0.log"}.get(model, Z.OUT / "logs" / f"train_{ds}_{Z.run_name(model)}.log")
    txt = tlog.read_text().split("the run restarts from step 1")[-1] if tlog.exists() else ""
    row = dict(dataset=ds, model=model, source="reused checkpoint" if reused else "trained in this study", n_params_total=pc["total"], n_params_temporal=pc.get("temporal", pc.get("backbone", 0) + (pc.get("head", 0) if model != "D0" else 0)),
               n_params_init=pc.get("init", 0), n_params_transition=pc.get("transition", 0), n_params_decoder=pc.get("D_z", pc.get("head", 0) if model == "D0" else 0),
               max_steps=int(ck["cfg"]["max_steps"]), min_steps=int(ck["cfg"]["min_steps"]), patience=int(ck["cfg"]["patience"]), steps=int(ck["steps"]), stopped_by=ck["stopped_by"], best_step=bs,
               best_step_objective=int(ck.get("best_step_total", bs)), hours=ck["seconds"] / 3600, s_per_step=ck["seconds"] / max(int(ck["steps"]), 1),
               val_L_C_at_best=at(bs, "val_L_C"), val_L_z_at_best=at(bs, "val_L_z"), val_L_z0_at_best=at(bs, "val_L_z0_frame0"), val_E_C_at_best=at(bs, "val_E_C"), val_E_C_teacher_z_at_best=at(bs, "val_E_C_gt"),
               min_val_L_C=float(lg["val_L_C"].min()), val_L_C_last=float(lg["val_L_C"].iloc[-1]), train_L_z_at_best=at(bs, "train_L_z"), gpus=",".join(sorted(set(re.findall(r"=== attempt \d+ on gpu (\d+)", txt)))))
    tp = Z.model_train_log(ds, model); tp = tp.with_name(tp.stem + "_train.csv")                # per-100-step training rows: the gradient norm BEFORE clipping
    if tp.exists():
        try:
            tr = pd.read_csv(tp)
        except pd.errors.EmptyDataError:                                                       # a run shorter than one logging interval
            tr = pd.DataFrame()
        if "grad_norm" in tr and len(tr):
            late = tr[tr.step > int(ck["cfg"].get("warmup", 5000))] if (tr.step > int(ck["cfg"].get("warmup", 5000))).any() else tr
            row.update(grad_norm_median=float(late.grad_norm.median()), grad_norm_share_clipped=float((late.grad_norm > float(ck["cfg"].get("grad_clip", 1.0))).mean()))
    return row, dict(ckpt=str(ck_p), md5=Z.md5(ck_p), best_step=bs, steps=int(ck["steps"]), stopped_by=ck["stopped_by"], n_params=pc, cfg={k: v for k, v in ck["cfg"].items() if k != "grad_inspect_steps"})


def contrast_row(ds, metric, name, v, ref_mean, take, higher=False):
    """Bootstrap of a per-sequence contrast (negative = lower error); verdict by the rule against the M00 mean of the metric."""
    ok = np.isfinite(v); mu, lo, hi = Z.cluster_bootstrap(v[ok], take[ok], n_boot=Z.N_BOOT); rel = mu / max(abs(ref_mean), 1e-9)
    good = (mu > 0) == higher
    ver = ("better" if good else "worse") if (abs(rel) >= Z.DECISION["rel_min"] and (lo > 0 or hi < 0)) else "similar"
    if name == "interaction":
        ver = ("non-zero" if (lo > 0 or hi < 0) else "zero within CI")
    return dict(dataset=ds, metric=metric, effect=name, value=mu, ci_lo=lo, ci_hi=hi, rel_to_M00=rel, verdict=ver, n=int(ok.sum()))


def block(get, a, b, metrics):
    vs = {m: get(a, b, m) for m in metrics}; vs = {m: v for m, v in vs.items() if v is not None}
    nb = sum(v.verdict == "better" for v in vs.values()); nw = sum(v.verdict == "worse" for v in vs.values())
    ans = "worse" if nw > nb else ("yes" if (nb >= Z.DECISION["struct_majority"] and nw == 0) else ("partial" if (nb >= 1 and nw <= 1) else "no"))
    return ans, nb, nw, vs


def two_pair(get, pairs, metric):
    """Reading of two paired comparisons on one metric: yes (both better) / partial (one better, none worse) / worse (any worse, none better) / mixed / no."""
    rs = [get(a, b, metric) for a, b in pairs]
    if any(r is None for r in rs):
        return None, rs
    nb = sum(r.verdict == "better" for r in rs); nw = sum(r.verdict == "worse" for r in rs)
    return ("yes" if nb == 2 else ("partial" if (nb == 1 and nw == 0) else ("worse" if (nw >= 1 and nb == 0) else ("mixed" if (nb and nw) else "no")))), rs


def decide(ds, get, val_ec, lat):
    rows = []; fmt = lambda r: f"{r.a} {r.value_a:.3f} vs {r.b} {r.value_b:.3f}, {r['diff']:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] ({r.rel_improvement * 100:+.2f} %, {r.verdict})"
    st_pairs, dec_pairs = [("M10", "M00"), ("M11", "M01")], [("M01", "M00"), ("M11", "M10")]
    q1, r1 = two_pair(get, st_pairs, "L_z")
    gap = "; ".join(f"{m}: train {lat[m]['rmse_train']:.3f} / test {lat[m]['rmse_test']:.3f} (gap {lat[m]['gap']:+.3f})" for m in Z.Z_MODELS if m in lat)
    rows.append(dict(dataset=ds, question="Does stateful z improve generalization?", answer=q1, evidence="test L_z: " + "; ".join(fmt(r) for r in r1) + ". z RMSE " + gap))
    q2, r2 = two_pair(get, st_pairs, "E_C")
    rows.append(dict(dataset=ds, question="Does stateful z improve final contact?", answer=q2, evidence="E_C: " + "; ".join(fmt(r) for r in r2)))
    q3, r3 = two_pair(get, dec_pairs, "E_C")
    rows.append(dict(dataset=ds, question="Does s_0-preserving decoding improve contact?", answer=q3, evidence="E_C: " + "; ".join(fmt(r) for r in r3)))
    q4a, r4 = two_pair(get, dec_pairs, "E_C_h1_16"); d4 = [get(m, "D0", "E_C_h1_16") for m in ("M01", "M11")]
    not_worse = any(r is not None and r.verdict != "worse" for r in d4)
    q4 = {"yes": "yes" if not_worse else "partial (better than absolute decoding, still worse than D0)", "partial": "partial", "no": "no", "worse": "worse", "mixed": "mixed"}[q4a]
    rows.append(dict(dataset=ds, question="Does s_0 preservation specifically fix early frames?", answer=q4,
                     evidence="frames 1-16: " + "; ".join(fmt(r) for r in r4) + "; against D0: " + "; ".join(fmt(r) for r in d4 if r is not None)))
    q5, r5 = two_pair(get, [("M11", "M01"), ("M11", "M10")], "E_C")
    rows.append(dict(dataset=ds, question="Are the two changes complementary?", answer=q5, evidence="E_C: " + "; ".join(fmt(r) for r in r5)))
    best = min(val_ec, key=val_ec.get); e = get(best, "D0", "E_C"); s, nb, nw, vs = block(get, best, "D0", Z.STRUCT_DECISION)
    q6 = {"better": "yes", "similar": "no (similar)", "worse": "no (worse)"}[e.verdict]
    rows.append(dict(dataset=ds, question="Does the best z model beat direct D0?", answer=q6,
                     evidence=f"best z model by VALIDATION E_C: {best} ({', '.join(f'{m} {v:.3f}' for m, v in val_ec.items())}); test E_C: {fmt(e)}; structure / wrench {nb} better / {nw} worse of {len(vs)} "
                              f"({'; '.join(f'{m} {v.verdict} ({v.rel_improvement * 100:+.1f} %)' for m, v in vs.items())})"))
    form = {"M00": "whole-sequence z", "M01": "s_0-preserving z (whole-sequence)", "M10": "stateful z", "M11": "stateful + s_0-preserving z"}[best]
    if e.verdict == "better":
        q7 = form
    elif e.verdict == "similar" and s == "yes":
        q7 = f"{form} for structure; direct dense D0 equal in dense error"
    elif e.verdict == "similar":
        q7 = f"direct dense D0 (the best z model, {form}, is similar and not structurally better)"
    else:
        q7 = "direct dense D0"
    rows.append(dict(dataset=ds, question="Which architecture is best supported?", answer=q7, evidence=f"best z by validation: {best}; vs D0 dense {e.verdict}, structure / wrench {s}"))
    beats_m00 = [m for m in ("M01", "M10", "M11") if get(m, "M00", "E_C") is not None and get(m, "M00", "E_C").verdict == "better"] if get("M11", "M00", "E_C") is not None else []
    r10, r01 = get("M10", "M00", "E_C"), get("M01", "M00", "E_C")
    m11_best = all(get("M11", b, "E_C").verdict == "better" for b in ("M00", "M01", "M10"))
    cases = []
    if m11_best:
        cases.append("C")
    if q2 == "yes" and q3 != "yes" and not m11_best:
        cases.append("A")
    if q3 == "yes" and q2 != "yes" and not m11_best:
        cases.append("B")
    if not (set(beats_m00) | ({"M10"} if r10.verdict == "better" else set()) | ({"M01"} if r01.verdict == "better" else set())) and e.verdict != "better":
        cases.append("E")
    rows.append(dict(dataset=ds, question="case", answer="+".join(cases) if cases else "mixed", evidence=f"stateful effect on contact: {q2}; decoder effect on contact: {q3}; M11 better than M00, M01 and M10: {m11_best}; best z vs D0: {e.verdict}"))
    amb = [f"{r.a} vs {r.b}" for r in list(r2) + list(r3) + [e] if abs(r.rel_change) >= Z.DECISION["rel_min"] and r.ci_lo <= 0 <= r.ci_hi]
    rows.append(dict(dataset=ds, question="extra seed needed (a decisive dense comparison ambiguous)?", answer="yes: " + ", ".join(amb) if amb else "no", evidence="comparisons with |relative difference| >= 2 % whose CI includes 0"))
    return rows, q2


def main():
    Z.OUT.mkdir(parents=True, exist_ok=True)
    AG = sat_aggregate_module()
    pair_rows, fx_rows, lat_rows, hrows, erows, dec_rows, drows, trows, curves, cfg_runs, st_ans = [], [], [], [], [], [], [], [], {}, {}, {}
    for ds in Z.DATASETS:
        for model in list(Z.Z_MODELS) + ["D0"]:
            row, cfg = training_row(ds, model)
            if row is not None:
                trows.append(row); cfg_runs[f"{ds}/{model}"] = cfg
        gt = load_metrics(ds, "GT"); take = gt["take"].astype(str); ex = gt["example"]
        M = {m: load_metrics(ds, m) for m in MODEL_ORDER}; M = {m: z for m, z in M.items() if z is not None}
        acc = []
        for m, z in M.items():
            assert np.array_equal(z["example"], ex), (ds, m, "example mismatch")
            for metric in METRICS + ["jitter_dev_dense", "jitter_dev_r2"] + EXTRA_DENSE:
                v = value(z, gt, metric)
                if v is not None:
                    acc.append(ci_row(v, take, dataset=ds, model=m, metric=metric))
            for r in AG.macro_scores([(0, z)], take, "K1"):
                acc.append(dict(dataset=ds, model=m, metric=r["metric"], value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n_examples=len(take)))
            for key in ("jitter_dense", "jitter_r2"):
                acc.append(ci_row(per_example(z, key) / np.maximum(per_example(gt, key), 1e-9), take, dataset=ds, model=m, metric=f"jitter_ratio_{key[7:]}"))
            for c in ("dense",) + STRUCT_CURVES + ("jitter", "jitter_r2"):
                if f"curve_{c}" in z:
                    cur = curve(z, c); curves[f"{ds}|{m}|{c}"] = np.nanmean(cur, 0)
                    if c in ("dense",) + STRUCT_CURVES:
                        for lo, hi in Z.HORIZON_BINS:
                            with np.errstate(all="ignore"):
                                hrows.append(ci_row(np.nanmean(cur[:, lo:hi + 1], 1), take, dataset=ds, model=m, curve=c, h_lo=lo, h_hi=hi))
            for t in Z.EARLY_FRAMES:
                erows.append(ci_row(curve(z, "dense")[:, t], take, dataset=ds, model=m, frame=t, quantity="E_C"))
        pd.DataFrame(acc).to_csv(Z.OUT / f"{ds}_metrics.csv", index=False)
        # ------------------------------------------------------------- paired comparisons
        mets = METRICS + ["jitter_dev_dense", "jitter_dev_r2"] + EXTRA_DENSE + [f"{c}_h{lo}_{hi}" for c in STRUCT_CURVES for lo, hi in Z.HORIZON_BINS]
        for a, b, q in PAIRS:
            if a not in M or b not in M:
                continue
            for metric in mets:
                va, vb = value(M[a], gt, metric), value(M[b], gt, metric)
                if va is None or vb is None:
                    continue
                with np.errstate(all="ignore"):
                    pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=metric, **verdict(metric if metric in S2.HIGHER_BETTER else "E_C", va, vb, take)))
        # ------------------------------------------------------------- factorial contrasts
        if all(m in M for m in Z.Z_MODELS):
            for metric in ["E_C", "E_C_last16", "E_C_final"] + [f"E_C_h{lo}_{hi}" for lo, hi in Z.HORIZON_BINS] + [f"E_C_t{t}" for t in Z.EARLY_FRAMES] + list(Z.STRUCT_DECISION) + ["q_cos", "jitter_dev_dense", "jitter_dev_r2"]:
                v = {m: value(M[m], gt, metric) for m in Z.Z_MODELS}; ref = float(np.nanmean(v["M00"])); hb = metric in S2.HIGHER_BETTER
                fx_rows.append(contrast_row(ds, metric, "stateful", 0.5 * ((v["M10"] - v["M00"]) + (v["M11"] - v["M01"])), ref, take, hb))
                fx_rows.append(contrast_row(ds, metric, "decoder", 0.5 * ((v["M01"] - v["M00"]) + (v["M11"] - v["M10"])), ref, take, hb))
                fx_rows.append(contrast_row(ds, metric, "interaction", (v["M11"] - v["M10"]) - (v["M01"] - v["M00"]), ref, take, hb))
        # ------------------------------------------------------------- latent diagnostics (zf_diag.py)
        dp = Z.ds_out(ds) / "diag" / "diag.npz"; lat = {}
        if dp.exists():
            D = dict(np.load(dp, allow_pickle=True)); assert np.array_equal(D["example"], ex)
            g = lambda k: D[k].astype(np.float64)
            tz2, hold, var_dim = g("teacher|z2|test"), g("hold_z0|zerr2|test"), g("teacher|var_dim|test")
            zmodels = [m for m in Z.Z_MODELS if f"{m}|zerr2|test" in D]
            for m in zmodels:
                e, tr, va = g(f"{m}|zerr2|test"), g(f"{m}|zerr2|train"), g(f"{m}|zerr2|val")
                lz = Z.cluster_bootstrap(e.mean(1), take, n_boot=Z.N_BOOT); r2d = 1 - g(f"{m}|zerr2_dim|test") / np.maximum(var_dim, 1e-9)
                rm = lambda x: float(np.sqrt(x.mean()))
                gap_b = Z.cluster_bootstrap(e.mean(1), take, n_boot=Z.N_BOOT)                                    # CI of the test part; the train part is a fixed number
                row = dict(dataset=ds, model=m, L_z_test=lz[0], L_z_lo=lz[1], L_z_hi=lz[2], L_z_train=float(tr.mean()), L_z_val=float(va.mean()), rmse_train=rm(tr), rmse_val=rm(va), rmse_test=rm(e),
                           gap=rm(e) - rm(tr), gap_lo=float(np.sqrt(gap_b[1])) - rm(tr), gap_hi=float(np.sqrt(gap_b[2])) - rm(tr), gap_L_z=float(e.mean() - tr.mean()), test_over_train_rmse=rm(e) / max(rm(tr), 1e-9),
                           r2_pooled=float(1 - e.mean() / tz2.mean()), r2_per_dim_mean=float(r2d.mean()), rmse_hold_z0=rm(hold), rmse_train_mean=rm(tz2),
                           **{f"rmse_h{lo}_{hi}": rm(e[:, lo - 1:hi]) for lo, hi in Z.HORIZON_BINS}, **{f"rmse_t{t}": rm(e[:, t - 1]) for t in Z.EARLY_FRAMES + (32, 63)},
                           E_C_teacher_z=float(g(f"{m}|oracle").mean()), E_C_train=float(g(f"{m}|pred|train").mean()), E_C_teacher_z_train=float(g(f"{m}|oracle|train").mean()), E_C_test=float(g(f"{m}|pred").mean()))
                if f"{m}|pred_z_other" in D:                                                                     # does the decoder use the latent?  own z_hat vs another sequence's z_hat vs the mean latent
                    own = g(f"{m}|pred").mean(1)
                    for tag in ("other", "mean"):
                        alt = g(f"{m}|pred_z_{tag}").mean(1); b = Z.cluster_bootstrap(alt - own, take, n_boot=Z.N_BOOT)
                        row.update({f"E_C_z_{tag}": float(alt.mean()), f"E_C_z_{tag}_minus_own": b[0], f"E_C_z_{tag}_minus_own_lo": b[1], f"E_C_z_{tag}_minus_own_hi": b[2], f"E_C_z_{tag}_rel": float(alt.mean() / own.mean() - 1)})
                        for lo, hi in ((1, 16), (49, 63)):
                            row[f"E_C_z_{tag}_rel_h{lo}_{hi}"] = float(g(f"{m}|pred_z_{tag}")[:, lo - 1:hi].mean() / g(f"{m}|pred")[:, lo - 1:hi].mean() - 1)
                if f"{m}|z0err2|test" in D:
                    z0 = g(f"{m}|z0err2|test"); b0 = Z.cluster_bootstrap(z0, take, n_boot=Z.N_BOOT)
                    row.update(z0_rmse_train=rm(g(f"{m}|z0err2|train")), z0_rmse_val=rm(g(f"{m}|z0err2|val")), z0_rmse_test=rm(z0), z0_L_test=b0[0], z0_L_lo=b0[1], z0_L_hi=b0[2],
                               rmse_true_z0_rollout=rm(g(f"{m}|zerr2_true_z0")), E_C_true_z0_rollout=float(g(f"{m}|pred_true_z0").mean()),
                               **{f"rmse_from_true_h{h}": rm(g(f"{m}|zerr2_h{h}")) for h in (1, 4, 8)}, **{f"rmse_hold_h{h}": rm(g(f"persist|zerr2_h{h}")) for h in (1, 4, 8)},
                               **{f"gain_over_hold_h{h}": 1 - rm(g(f"{m}|zerr2_h{h}")) / rm(g(f"persist|zerr2_h{h}")) for h in (1, 4, 8)})
                    curves[f"{ds}|{m}|z_err_true_z0"] = np.sqrt(g(f"{m}|zerr2_true_z0").mean(0))
                lat_rows.append(row); lat[m] = row
                curves[f"{ds}|{m}|z_err"] = np.sqrt(e.mean(0)); curves[f"{ds}|{m}|z_err_train"] = np.sqrt(tr.mean(0)); curves[f"{ds}|{m}|z_err_val"] = np.sqrt(va.mean(0))
                curves[f"{ds}|{m}|drift"] = g(f"{m}|drift").mean(0)
                for t in Z.EARLY_FRAMES:
                    erows.append(ci_row(g(f"{m}|drift")[:, t - 1] / np.maximum(g("gt|drift")[:, t - 1], 1e-6), take, dataset=ds, model=m, frame=t, quantity="drift_ratio"))
                    erows.append(dict(dataset=ds, model=m, frame=t, quantity="drift_mean_ratio", value=float(g(f"{m}|drift")[:, t - 1].mean() / g("gt|drift")[:, t - 1].mean()), ci_lo=np.nan, ci_hi=np.nan, n_examples=len(take)))
            curves[f"{ds}|hold_z0|z_err"] = np.sqrt(hold.mean(0)); curves[f"{ds}|gt|drift"] = g("gt|drift").mean(0); curves[f"{ds}|D0|drift"] = g("D0|drift").mean(0)
            for t in Z.EARLY_FRAMES:
                erows.append(dict(dataset=ds, model="D0", frame=t, quantity="drift_mean_ratio", value=float(g("D0|drift")[:, t - 1].mean() / g("gt|drift")[:, t - 1].mean()), ci_lo=np.nan, ci_hi=np.nan, n_examples=len(take)))
            for a, b, q in PAIRS[:5]:
                if a in zmodels and b in zmodels:
                    pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric="L_z", **verdict("L_z", g(f"{a}|zerr2|test").mean(1), g(f"{b}|zerr2|test").mean(1), take)))
                    for lo, hi in Z.HORIZON_BINS:
                        pair_rows.append(dict(dataset=ds, a=a, b=b, question=q, metric=f"L_z_h{lo}_{hi}", **verdict("L_z", g(f"{a}|zerr2|test")[:, lo - 1:hi].mean(1), g(f"{b}|zerr2|test")[:, lo - 1:hi].mean(1), take)))
            if all(m in zmodels for m in Z.Z_MODELS):
                v = {m: g(f"{m}|zerr2|test").mean(1) for m in Z.Z_MODELS}; ref = float(v["M00"].mean())
                fx_rows.append(contrast_row(ds, "L_z", "stateful", 0.5 * ((v["M10"] - v["M00"]) + (v["M11"] - v["M01"])), ref, take))
                fx_rows.append(contrast_row(ds, "L_z", "decoder", 0.5 * ((v["M01"] - v["M00"]) + (v["M11"] - v["M10"])), ref, take))
                fx_rows.append(contrast_row(ds, "L_z", "interaction", (v["M11"] - v["M10"]) - (v["M01"] - v["M00"]), ref, take))
        # ------------------------------------------------------------- decision and the comparison with the direct model
        P = pd.DataFrame(pair_rows); P = P[P.dataset == ds] if len(P) else P
        def get(a, b, metric, P=P):
            r = P[(P.a == a) & (P.b == b) & (P.metric == metric)] if len(P) else P
            return r.iloc[0] if len(r) else None
        val_ec = {}
        for m in Z.Z_MODELS:
            vp = Z.model_metrics(ds, m, "val")
            if vp.exists():
                val_ec[m] = float(np.nanmean(np.load(vp, allow_pickle=True)["E_C"]))
        if all(m in M for m in Z.Z_MODELS) and len(val_ec) == 4 and all(m in lat for m in Z.Z_MODELS):
            rows_, st_ans[ds] = decide(ds, get, val_ec, lat); dec_rows.extend(rows_)
            best = min(val_ec, key=val_ec.get)
            for metric in ["E_C", "E_C_last16", "E_C_final"] + [f"E_C_h{lo}_{hi}" for lo, hi in Z.HORIZON_BINS] + [f"E_C_t{t}" for t in Z.EARLY_FRAMES] + list(Z.STRUCT_DECISION) + ["q_cos", "jitter_dev_dense", "jitter_dev_r2"]:
                for ref in ("D0", "D0r"):
                    r = get(best, ref, metric)
                    if r is not None:
                        drows.append(dict(dataset=ds, best_z=best, reference=ref, metric=metric, value_best_z=r.value_a, value_reference=r.value_b, diff=r["diff"], ci_lo=r.ci_lo, ci_hi=r.ci_hi, rel_improvement=r.rel_improvement, verdict=r.verdict,
                                          val_E_C=", ".join(f"{m} {v:.3f}" for m, v in val_ec.items())))
    if len(st_ans) == 2:                                                             # Case D is a statement about the two datasets together
        d = st_ans.get("taco") == "yes" and st_ans.get("arctic") in ("no", "worse", "mixed")
        for ds in Z.DATASETS:
            dec_rows.append(dict(dataset=ds, question="case D (stateful helps TACO only)?", answer="yes" if d else "no", evidence=f"stateful effect on contact: TACO {st_ans.get('taco')}, ARCTIC {st_ans.get('arctic')}"))
    # ------------------------------------------------------------- write
    T = pd.DataFrame(trows); T.to_csv(Z.OUT / "training_summary.csv", index=False)
    mt = []
    for m in list(Z.Z_MODELS) + ["D0"]:
        c = Z.MODELS.get(m, dict(stateful=False, residual=False, label="direct dense model: backbone -> MLP head -> rho_hat_1:63, C_t = s_0 + sigma_r rho_hat_t (Stage-2 B0, reused)", reuse="s2:B0"))
        r = dict(model=m, description=c["label"], temporal="none (dense output)" if m == "D0" else ("stateful: I(s_0, G) -> z_0, residual-MLP transition rolled out 63 steps" if c["stateful"] else "whole-sequence backbone -> z_1:63 in one pass"),
                 decoder="MLP head on the backbone (residual on s_0)" if m == "D0" else ("s_0-preserving point decoder: C_t = s_0 + D_delta(s_0, z_t, G_t)" if c["residual"] else "absolute point decoder: C_t = D_abs(z_t, G_t)"),
                 loss="L_C" if m == "D0" else ("L_C + L_z + L_z0" if c["stateful"] else "L_C + L_z"), trained="reused" if c["reuse"] else "this study")
        for ds in Z.DATASETS:
            t = T[(T.dataset == ds) & (T.model == m)] if len(T) else T
            if len(t):
                r.update({f"n_params_{ds}_total": int(t.n_params_total.iloc[0]), f"n_params_{ds}_temporal": int(t.n_params_temporal.iloc[0]), f"n_params_{ds}_decoder": int(t.n_params_decoder.iloc[0]),
                          f"n_params_{ds}_init": int(t.n_params_init.iloc[0]), f"n_params_{ds}_transition": int(t.n_params_transition.iloc[0])})
        if any(k.startswith("n_params") for k in r):
            mt.append(r)
    pd.DataFrame(mt).to_csv(Z.OUT / "model_table.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(Z.OUT / "paired_comparisons.csv", index=False); pd.DataFrame(fx_rows).to_csv(Z.OUT / "factorial_effects.csv", index=False)
    pd.DataFrame(lat_rows).to_csv(Z.OUT / "latent_metrics.csv", index=False); pd.DataFrame(hrows).to_csv(Z.OUT / "horizon_metrics.csv", index=False)
    pd.DataFrame(erows).to_csv(Z.OUT / "early_frame_metrics.csv", index=False); pd.DataFrame(drows).to_csv(Z.OUT / "direct_comparison.csv", index=False)
    Dd = pd.DataFrame(dec_rows); Dd.to_csv(Z.OUT / "decision_summary.csv", index=False)
    np.savez_compressed(Z.OUT / "temporal_curves.npz", **curves)
    dj = Z.OUT / "init_aug_decision.json"
    Z.write_json(Z.OUT / "experiment_config.json", dict(
        question="What was wrong with the previous z-mediated generator: open-loop z prediction, the loss of s_0 in decoding, both, or neither?",
        models=Z.MODELS, direct="D0 = Stage-2 B0 (reused prediction and metric files)", backbone=Z.BACKBONE, state=Z.STATE, transition_choice=Z.TRANSITION_CHOICE, decoder=Z.DECODER, train=Z.TRAIN,
        init_supervision=Z.read_json(dj) if dj.exists() else None, decision_rule=Z.DECISION, horizon_bins=Z.HORIZON_BINS, early_frames=Z.EARLY_FRAMES, n_boot=Z.N_BOOT, seed=Z.SEED, protocol=Z.PROTOCOL, runs=cfg_runs,
        teacher={ds: S2.read_json(S2.teacher_cache_path(ds).with_suffix(".json")) for ds in Z.DATASETS}, cases={r["dataset"]: r["answer"] for r in dec_rows if r["question"] == "case"},
        sources=dict(stage2=str(S2.OUT), joint_decoder_followup=str(ZJ.OUT), stage1=str(S2.CF.OUT)), code_dir=str(Z.HERE), code_md5={p.name: Z.md5(p) for p in sorted(Z.HERE.glob("*.py")) + sorted(Z.HERE.glob("*.sh"))}))
    log.info("tables written: %d paired rows, %d factorial rows, %d latent rows, %d decision rows", len(pair_rows), len(fx_rows), len(lat_rows), len(dec_rows))
    if len(Dd):
        log.info("\n%s", Dd[["dataset", "question", "answer"]].to_string())


if __name__ == "__main__":
    main()
