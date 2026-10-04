#!/usr/bin/env python
"""Experiment B aggregation: per-frame test metrics of every (representation, head, seed) -> temporal_prediction.csv
(means over the valid test frames, seed spread, take-cluster bootstrap 95 % CI; persistence and train-mean baselines)
and temporal_paired.csv (the ratio of every representation's error sum to FULL's, jointly bootstrapped over takes,
plus the composite T1–T4 gap used by the saturation rule).
    python evaluate_temporal.py --dataset taco
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score

import sv_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval_temporal")
STRUCT_METRICS = {"T2": ["T2_rel_l1", "mse_m"], "T3": ["T3_cent", "T3_ang", "mse_geom"], "T4": ["T4_rel_l1", "T4_cos", "T4_Qerr", "mse_q"]}
DENSE_METRICS = {"T5C": ["E_C", "cos_C", "mse_C"], "T5H": ["E_H", "mse_H"]}
PERSIST = {"T2_rel_l1": "T2_rel_l1_persist", "T3_cent": "T3_cent_persist", "T3_ang": "T3_ang_persist", "T4_rel_l1": "T4_rel_l1_persist",
           "mse_m": "mse_m_persist", "mse_geom": "mse_geom_persist", "mse_q": "mse_q_persist", "E_C": "E0_C", "E_H": "E0_H", "mse_C": "mse_C_persist", "mse_H": "mse_H_persist"}
# metrics entering the composite structured error (lower = better), by target
COMPOSITE = {"T1": "one_minus_auprc", "T2": "T2_rel_l1", "T3": ["T3_cent", "T3_ang"], "T4": "T4_rel_l1"}
N_BOOT_AUPRC = 300


def load_runs(ds, head):
    runs = {}
    for p in sorted((S.ds_out(ds) / "temporal" / "metrics").glob(f"*_{head}_seed*.npz")):
        rep, _, seed = p.stem.rsplit("_", 2)
        if rep not in S.REPS:
            continue
        z = np.load(p, allow_pickle=True)
        runs.setdefault(rep, {})[int(seed[4:])] = {k: z[k] for k in z.files}
    return runs


def auprc_f1(prob, true, valid):
    """macro / micro AUPRC and F1 (threshold 0.5) over the valid frames."""
    out = {}
    ap_parts, f1_parts = [], []
    P = prob[valid]; Y = true[valid]
    for k in range(6):
        if Y[:, k].min() == Y[:, k].max():
            continue
        ap_parts.append(average_precision_score(Y[:, k], P[:, k])); f1_parts.append(f1_score(Y[:, k], P[:, k] >= 0.5))
    out["auprc_macro"] = float(np.mean(ap_parts)); out["f1_macro"] = float(np.mean(f1_parts))
    out["auprc_micro"] = float(average_precision_score(Y.ravel(), P.ravel())); out["f1_micro"] = float(f1_score(Y.ravel(), P.ravel() >= 0.5))
    out["prevalence"] = float(Y.mean())
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); ds = a.dataset
    rows, paired = [], []
    F = S.load_features(ds)
    D = S.HP.load_raw(ds); man = D["man"]; tr = man["train"]
    # train-mean baselines (raw units)
    m_tr = F["m"][tr][:, S.PAD:S.PAD + S.T].reshape(-1, 6); m_bar = m_tr.mean(0)
    q_tr = F["q"][tr].reshape(-1, S.N_DIR); q_bar = q_tr[q_tr.sum(1) > 1e-6].mean(0)
    for head, METRICS in (("structured", STRUCT_METRICS), ("dense", DENSE_METRICS)):
        runs = load_runs(ds, head)
        if not runs:
            continue
        ref = next(iter(next(iter(runs.values())).values()))
        take = ref["take"].astype(str); ex = ref["example"]; tt = ref["t"]; valid_all = ref["valid"] > 0
        nh = len(S.HORIZONS)
        # per-frame targets for the mean baselines
        m_true = F["m"][ex[:, None], S.PAD + np.minimum(tt[:, None] + np.array(S.HORIZONS)[None], S.T - 1)]           # (N, 3, 6)
        q_true = F["q"][ex[:, None], np.minimum(tt[:, None] + np.array(S.HORIZONS)[None], S.T - 1)]                    # (N, 3, 76)
        seed_avg = {}
        for rep, seeds in runs.items():
            keys = [k for k in seeds[min(seeds)] if k not in ("example", "t", "take", "horizons")]
            seed_avg[rep] = {k: np.nanmean(np.stack([seeds[s][k].astype(np.float32) for s in seeds]), 0) for k in keys}
            seed_avg[rep]["_seeds"] = sorted(seeds)
            for tg, mets in METRICS.items():
                for met in mets:
                    for j, h in enumerate(S.HORIZONS):
                        vals = seed_avg[rep][met][:, j]; ok = valid_all[:, j] & np.isfinite(vals)
                        pt, lo, hi = S.cluster_bootstrap(vals[ok], take[ok], n_boot=a.n_boot)
                        per_seed = [float(np.nanmean(seeds[s][met][:, j][ok])) for s in seeds]
                        rows.append(dict(dataset=ds, head=head, rep=rep, target=tg, metric=met, h=h, value=pt, lo=lo, hi=hi, seed_sd=float(np.std(per_seed)),
                                         n_seeds=len(seeds), n_frames=int(ok.sum()), n_takes=len(np.unique(take[ok]))))
                        if met in PERSIST and rep == "FULL":
                            pv = seeds[min(seeds)][PERSIST[met]][:, j]; okp = valid_all[:, j] & np.isfinite(pv)
                            pt, lo, hi = S.cluster_bootstrap(pv[okp], take[okp], n_boot=a.n_boot)
                            rows.append(dict(dataset=ds, head=head, rep="PERSIST", target=tg, metric=met, h=h, value=pt, lo=lo, hi=hi, seed_sd=0.0, n_seeds=0,
                                             n_frames=int(okp.sum()), n_takes=len(np.unique(take[okp]))))
            if head == "structured":
                for j, h in enumerate(S.HORIZONS):
                    ok = valid_all[:, j]
                    r = auprc_f1(seed_avg[rep]["a_prob"][:, j], seed_avg[rep]["a_true"][:, j] > 0.5, ok)
                    per_seed = [auprc_f1(seeds[s]["a_prob"][:, j].astype(np.float32), seeds[s]["a_true"][:, j] > 0.5, ok) for s in seeds]
                    # take bootstrap of the macro AUPRC / F1
                    u, inv = np.unique(take[ok], return_inverse=True); rng = np.random.default_rng(0); reps_ap, reps_f1 = [], []
                    P = seed_avg[rep]["a_prob"][:, j][ok]; Y = seed_avg[rep]["a_true"][:, j][ok] > 0.5
                    for _ in range(N_BOOT_AUPRC):
                        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)); sel = np.repeat(np.arange(len(P)), w[inv])
                        rr = auprc_f1(P[sel], Y[sel], np.ones(len(sel), bool)); reps_ap.append(rr["auprc_macro"]); reps_f1.append(rr["f1_macro"])
                    for met, per, boot in (("auprc_macro", [p["auprc_macro"] for p in per_seed], reps_ap), ("f1_macro", [p["f1_macro"] for p in per_seed], reps_f1),
                                           ("auprc_micro", [p["auprc_micro"] for p in per_seed], None), ("f1_micro", [p["f1_micro"] for p in per_seed], None)):
                        rows.append(dict(dataset=ds, head=head, rep=rep, target="T1", metric=met, h=h, value=r[met], lo=np.percentile(boot, 2.5) if boot else np.nan,
                                         hi=np.percentile(boot, 97.5) if boot else np.nan, seed_sd=float(np.std(per)), n_seeds=len(seeds), n_frames=int(ok.sum()), n_takes=len(u)))
                    if rep == "FULL":
                        rp = auprc_f1(seed_avg[rep]["a_now"][:, j].astype(np.float32), seed_avg[rep]["a_true"][:, j] > 0.5, ok)
                        for met in ("auprc_macro", "f1_macro", "auprc_micro", "f1_micro"):
                            rows.append(dict(dataset=ds, head=head, rep="PERSIST", target="T1", metric=met, h=h, value=rp[met], lo=np.nan, hi=np.nan, seed_sd=0.0, n_seeds=0, n_frames=int(ok.sum()), n_takes=len(u)))
                        rows.append(dict(dataset=ds, head=head, rep="MEAN", target="T1", metric="auprc_macro", h=h, value=r["prevalence"], lo=np.nan, hi=np.nan, seed_sd=0.0, n_seeds=0, n_frames=int(ok.sum()), n_takes=len(u)))
                        mt = m_true[:, j][ok]; rel = np.abs(m_bar[None] - mt).sum(1) / m_bar.sum()          # normalised by the train-mean total amount
                        pt, lo, hi = S.cluster_bootstrap(rel, take[ok], n_boot=a.n_boot)
                        rows.append(dict(dataset=ds, head=head, rep="MEAN", target="T2", metric="T2_rel_l1", h=h, value=pt, lo=lo, hi=hi, seed_sd=0.0, n_seeds=0, n_frames=int(ok.sum()), n_takes=len(u)))
                        qt = q_true[:, j][ok]; okq = qt.sum(1) > 1e-6; rel = np.abs(q_bar[None] - qt[okq]).sum(1) / qt[okq].sum(1)
                        pt, lo, hi = S.cluster_bootstrap(rel, take[ok][okq], n_boot=a.n_boot)
                        rows.append(dict(dataset=ds, head=head, rep="MEAN", target="T4", metric="T4_rel_l1", h=h, value=pt, lo=lo, hi=hi, seed_sd=0.0, n_seeds=0, n_frames=int(okq.sum()), n_takes=len(u)))
        # ---------------------------------------------------------------- paired gaps to FULL (joint take bootstrap)
        if "FULL" not in seed_avg:
            continue
        u, inv = np.unique(take, return_inverse=True); rng = np.random.default_rng(1)
        sum_metrics = [m for mets in METRICS.values() for m in mets if not m.startswith("mse") and m not in ("T4_cos", "cos_C")]
        stats = {}
        for rep in seed_avg:
            for met in sum_metrics:
                for j in range(nh):
                    v = seed_avg[rep][met][:, j]; ok = valid_all[:, j] & np.isfinite(v)
                    stats[(rep, met, j)] = (np.bincount(inv[ok], weights=v[ok], minlength=len(u)), np.bincount(inv[ok], minlength=len(u)).astype(float))
        boot_w = [np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float) for _ in range(a.n_boot)]
        ratios = {}
        for rep in seed_avg:
            if rep == "FULL":
                continue
            for met in sum_metrics:
                for j, h in enumerate(S.HORIZONS):
                    sn, cn = stats[(rep, met, j)]; sf, cf = stats[("FULL", met, j)]
                    pt = (sn.sum() / cn.sum()) / (sf.sum() / cf.sum())
                    reps_ = np.array([((w * sn).sum() / (w * cn).sum()) / ((w * sf).sum() / (w * cf).sum()) for w in boot_w])
                    ratios[(rep, met, j)] = reps_
                    paired.append(dict(dataset=ds, head=head, rep=rep, metric=met, h=h, ratio_to_full=pt, lo=float(np.percentile(reps_, 2.5)), hi=float(np.percentile(reps_, 97.5))))
        if head == "structured":
            # T1: 1 - macro AUPRC ratio with a take bootstrap (fewer replicates), then the composite T1-T4 gap
            ap_boot = {}
            for rep in seed_avg:
                for j in range(nh):
                    ok = valid_all[:, j]; P = seed_avg[rep]["a_prob"][:, j][ok]; Y = seed_avg[rep]["a_true"][:, j][ok] > 0.5; inv_ok = inv[ok]
                    pts = [1 - auprc_f1(P, Y, np.ones(len(P), bool))["auprc_macro"]]
                    rng2 = np.random.default_rng(2)
                    for _ in range(N_BOOT_AUPRC):
                        w = np.bincount(rng2.integers(0, len(u), len(u)), minlength=len(u)); sel = np.repeat(np.arange(len(P)), w[inv_ok])
                        pts.append(1 - auprc_f1(P[sel], Y[sel], np.ones(len(sel), bool))["auprc_macro"])
                    ap_boot[(rep, j)] = np.array(pts)
            for rep in seed_avg:
                if rep == "FULL":
                    continue
                comp_pt, comp_boot = [], []
                for j, h in enumerate(S.HORIZONS):
                    r1 = ap_boot[(rep, j)] / ap_boot[("FULL", j)]
                    paired.append(dict(dataset=ds, head=head, rep=rep, metric="one_minus_auprc", h=h, ratio_to_full=float(r1[0]), lo=float(np.percentile(r1[1:], 2.5)), hi=float(np.percentile(r1[1:], 97.5))))
                    parts = [r1[0], ratios[(rep, "T2_rel_l1", j)].mean() * 0 + (stats[(rep, "T2_rel_l1", j)][0].sum() / stats[(rep, "T2_rel_l1", j)][1].sum()) / (stats[("FULL", "T2_rel_l1", j)][0].sum() / stats[("FULL", "T2_rel_l1", j)][1].sum())]
                    t3 = 0.5 * ((stats[(rep, "T3_cent", j)][0].sum() / stats[(rep, "T3_cent", j)][1].sum()) / (stats[("FULL", "T3_cent", j)][0].sum() / stats[("FULL", "T3_cent", j)][1].sum())
                                + (stats[(rep, "T3_ang", j)][0].sum() / stats[(rep, "T3_ang", j)][1].sum()) / (stats[("FULL", "T3_ang", j)][0].sum() / stats[("FULL", "T3_ang", j)][1].sum()))
                    t4 = (stats[(rep, "T4_rel_l1", j)][0].sum() / stats[(rep, "T4_rel_l1", j)][1].sum()) / (stats[("FULL", "T4_rel_l1", j)][0].sum() / stats[("FULL", "T4_rel_l1", j)][1].sum())
                    comp = np.mean(parts + [t3, t4]); comp_pt.append(comp)
                    # bootstrap replicates of the composite (T1 replicates are fewer: cycle them)
                    b = np.array([r1[1 + (k % N_BOOT_AUPRC)] for k in range(a.n_boot)])
                    b = (b + ratios[(rep, "T2_rel_l1", j)] + 0.5 * (ratios[(rep, "T3_cent", j)] + ratios[(rep, "T3_ang", j)]) + ratios[(rep, "T4_rel_l1", j)]) / 4.0
                    comp_boot.append(b)
                    paired.append(dict(dataset=ds, head=head, rep=rep, metric="composite_T1_T4", h=h, ratio_to_full=float(comp), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
                b = np.mean(comp_boot, 0)
                paired.append(dict(dataset=ds, head=head, rep=rep, metric="composite_T1_T4", h=0, ratio_to_full=float(np.mean(comp_pt)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5))))
    out = S.ds_out(ds) / "temporal"
    pd.DataFrame(rows).to_csv(out / "temporal_prediction.csv", index=False)
    pd.DataFrame(paired).to_csv(out / "temporal_paired.csv", index=False)
    log.info("%s: %d metric rows, %d paired rows", ds, len(rows), len(paired))


if __name__ == "__main__":
    main()
