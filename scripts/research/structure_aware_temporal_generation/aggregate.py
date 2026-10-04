#!/usr/bin/env python
"""Aggregate the per-example metrics of every run into the report tables.    python aggregate.py
Every number is a mean over the test sequences of the SEED-AVERAGED per-example value (3 seeds), with a 95 % take-cluster
bootstrap interval (1000 reps) and the spread (std) of the three per-seed test means. Statistics: K1 (sample 0), mean over
the K samples, best-of-10 (the per-example minimum over the samples of THAT metric; a per-metric oracle selection).
Writes to OUT: model_matrix.csv, fixed_s0_metrics.csv (protocol A), end_to_end_metrics.csv (protocol B), structural_metrics.csv,
wrench_metrics.csv, diversity_metrics.csv, event_metrics.csv, paired_comparisons.csv, accuracy_vs_diversity.csv,
temporal_curves.npz, decision_summary.csv, experiment_config.json.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

import sat_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aggregate")
STATS = ("K1", "mean", "best10")
METRICS = ["E_C", "E_C_last16", "s0_err", "part_hamming", "amount_l1", "amount_rmse", "centroid", "normal", "centroid_both", "normal_both", "both_frac",
           "q_rel_l1", "q_cos", "q_Qerr", "q_rel_l1_vs_grid", "part_hamming_vs_grid", "amount_l1_vs_grid", "centroid_vs_grid", "normal_vs_grid",
           "jitter_dense", "tv_dense", "jitter_r2"]
HIGHER_BETTER = {"q_cos", "both_frac"}
NO_BEST = {"both_frac", "jitter_dense", "tv_dense", "jitter_r2", "s0_err"}
DIVERSITY = ["D_C", "D_part", "D_amount", "D_centroid", "D_normal", "D_q", "D_Z"]
STRUCT_KEYS = ["part_hamming", "amount_l1", "centroid", "normal"]
PAIRS = [("D0", "D1", "output structural supervision (det)"), ("D1", "D2", "hidden aux supervision (det)"), ("S0", "S1", "output structural supervision (diff)"),
         ("S1", "S2", "hidden aux supervision (diff)"), ("D0", "S0", "stochastic vs deterministic (L0)"), ("D1", "S1", "stochastic vs deterministic (L1)"),
         ("D2", "S2", "stochastic vs deterministic (L2)")]
MAIN_METRICS = ["E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "jitter_dense", "jitter_r2"]
_CACHE = {}


def load(ds, name, prot):
    key = (ds, name, prot)
    if key not in _CACHE:
        p = S.metrics_path(ds, name, prot)
        if not p.exists():
            return None
        z = np.load(p, allow_pickle=True)
        _CACHE[key] = {k: z[k] for k in z.files}
    return _CACHE[key]


def runs(ds, model, prot):
    out = [(s, load(ds, S.run_name(model, s), prot)) for s in S.SEEDS]
    return [(s, z) for s, z in out if z is not None]


def per_example(z, metric, stat):
    v = z[metric].astype(np.float64)                                                # (B, K)
    if stat == "K1":
        return v[:, 0]
    if stat == "mean":
        return np.nanmean(v, 1)
    if metric in NO_BEST:
        return np.full(len(v), np.nan)
    return np.nanmax(v, 1) if metric in HIGHER_BETTER else np.nanmin(v, 1)


def seed_rows(zs, take, metric, stat, extra):
    per = [per_example(z, metric, stat) for _, z in zs]
    if not per or np.all(np.isnan(per[0])):
        return None
    v = np.nanmean(np.stack(per), 0)
    m, lo, hi = S.cluster_bootstrap(v, take)
    return dict(**extra, metric=metric, stat=stat, value=m, ci_lo=lo, ci_hi=hi, seed_std=float(np.std([np.nanmean(p) for p in per])), n_seeds=len(per), n_examples=int(np.isfinite(v).sum()))


def macro_scores(zs, take, stat, n_boot=200):
    """Dataset-level macro F1 (from summed TP / FP / FN) and macro AUPRC (soft scores), seed-averaged, take bootstrap."""
    rows = []
    for key in ("part_macro_f1", "part_macro_auprc"):
        vals, reps = [], []
        for si, (_, z) in enumerate(zs):
            K = z["tp"].shape[1]; ks = [0] if stat == "K1" else list(range(K))
            if key == "part_macro_f1":
                f = lambda sel: np.mean([np.mean([2 * z["tp"][sel, k, j].sum() / max(2 * z["tp"][sel, k, j].sum() + z["fp"][sel, k, j].sum() + z["fn"][sel, k, j].sum(), 1) for j in range(6)]) for k in ks])
            else:
                sc = z["r2_a_soft"].astype(np.float32); tr = z["a_true"]; ev = slice(1, S.T) if z["protocol"] == "A" else slice(0, S.T)
                def f(sel, sc=sc, tr=tr, ev=ev, ks=ks):
                    out = []
                    for k in ks:
                        y = tr[sel][:, ev].reshape(-1, 6) > 0.5; s = sc[sel][:, k, ev].reshape(-1, 6)
                        out.append(np.mean([average_precision_score(y[:, j], s[:, j]) for j in range(6) if y[:, j].any() and (~y[:, j]).any()]))
                    return np.mean(out)
            vals.append(f(np.arange(len(take))))
            if si == 0:
                u, inv = np.unique(take, return_inverse=True); groups = [np.where(inv == g)[0] for g in range(len(u))]
                rng = np.random.default_rng(0)
                for _ in range(n_boot):
                    sel = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))]); reps.append(f(sel))
        rows.append(dict(metric=key, stat=stat, value=float(np.mean(vals)), ci_lo=float(np.percentile(reps, 2.5)), ci_hi=float(np.percentile(reps, 97.5)), seed_std=float(np.std(vals)), n_seeds=len(vals), n_examples=len(take)))
    return rows


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    mm = S.model_matrix()
    cfg_rows, acc_rows, div_rows, pair_rows, ev_rows, avd_rows, curves = [], [], [], [], [], [], {}
    for ds in S.DATASETS:
        ch = S.chosen_lambdas(ds) or {}
        for model in S.MODELS:
            for s, z in runs(ds, model, "A"):
                ck = torch.load(S.ckpt_path(ds, S.run_name(model, s)), map_location="cpu", weights_only=False)
                cfg_rows.append(dict(dataset=ds, model=model, seed=s, lambda_struct=ck["lambda_struct"], lambda_aux=ck["lambda_aux"], n_params=ck["n_params"], best_step=ck["best_step"],
                                     steps=ck["steps"], stopped_by=ck["stopped_by"], converged=ck["converged"], best_val=ck["best_val"], val_init=ck["val_init"]))
        gt = load(ds, "GT", "A"); take = gt["take"].astype(str)
        for prot in S.PROTOCOLS:
            refs = [("GT", load(ds, "GT", prot))] if prot == "A" else []
            refs.append(("PERSIST", load(ds, "PERSIST", prot)))
            groups = [(name, [(0, z)]) for name, z in refs if z is not None] + [(model, runs(ds, model, prot)) for model in S.MODELS]
            for name, zs in groups:
                if not zs:
                    continue
                extra = dict(dataset=ds, protocol=prot, model=name, family=("reference" if name in ("GT", "PERSIST") else S.MODELS[name]["family"]))
                for stat in STATS:
                    for metric in METRICS + (DIVERSITY if zs[0][1]["K"] > 1 else []):
                        if metric in DIVERSITY and stat != "K1":
                            continue
                        if metric not in zs[0][1]:
                            continue
                        if metric in DIVERSITY:
                            per = [z[metric].astype(np.float64) for _, z in zs]; v = np.nanmean(np.stack(per), 0); m, lo, hi = S.cluster_bootstrap(v, take)
                            r = dict(**extra, metric=metric, stat="samples", value=m, ci_lo=lo, ci_hi=hi, seed_std=float(np.std([np.nanmean(p) for p in per])), n_seeds=len(per), n_examples=len(v))
                        else:
                            r = seed_rows(zs, take, metric, stat, extra)
                        if r is not None:
                            acc_rows.append(r)
                    if stat != "best10":
                        for r in macro_scores(zs, take, stat):
                            acc_rows.append(dict(**extra, **r))
                # curves (K1 and mean over samples), seed-averaged
                for key in [k for k in zs[0][1] if k.startswith("curve_")]:
                    c = np.stack([z[key].astype(np.float64) for _, z in zs])                                    # (seeds, B, K, T) or (seeds, B, T)
                    if c.ndim == 4:
                        curves[f"{ds}|{prot}|{name}|{key[6:]}|K1"] = np.nanmean(c[:, :, 0], (0, 1)); curves[f"{ds}|{prot}|{name}|{key[6:]}|mean"] = np.nanmean(c, (0, 1, 2))
                    else:
                        curves[f"{ds}|{prot}|{name}|{key[6:]}|samples"] = np.nanmean(c, (0, 1))
                # diversity rows (Table 6)
                if zs[0][1]["K"] > 1:
                    source = {"A": "future (fixed s_0, stochastic futures)", "B": "initial (sampled s_0, one future each)"}[prot]
                    if name == "PERSIST":
                        source = "initial, no evolution (sampled s_0 held)"
                    elif S.MODELS[name]["family"] == "diff" and prot == "B":
                        source = "initial + future (sampled s_0, one stochastic future each)"
                    for metric in DIVERSITY:
                        per = [z[metric].astype(np.float64) for _, z in zs]; v = np.nanmean(np.stack(per), 0); m, lo, hi = S.cluster_bootstrap(v, take)
                        div_rows.append(dict(dataset=ds, source=source, protocol=prot, model=name, metric=metric, value=m, ci_lo=lo, ci_hi=hi, seed_std=float(np.std([np.nanmean(p) for p in per])), n_seeds=len(per)))
                # accuracy vs diversity (stochastic models, protocol A)
                if name in S.DIFF and prot == "A":
                    for s, z in zs:
                        row = dict(dataset=ds, model=name, seed=s)
                        for metric in ["E_C"] + STRUCT_KEYS + ["q_rel_l1"]:
                            for stat in STATS:
                                row[f"{metric}_{stat}"] = float(np.nanmean(per_example(z, metric, stat)))
                        for metric in DIVERSITY:
                            row[metric] = float(np.nanmean(z[metric]))
                        avd_rows.append(row)
            # paired comparisons
            for a_, b_, what in PAIRS:
                za, zb = runs(ds, a_, prot), runs(ds, b_, prot)
                if not za or not zb:
                    continue
                for stat in STATS:
                    for metric in MAIN_METRICS + ["centroid_both", "normal_both", "q_cos", "E_C_last16"]:
                        pa = [per_example(z, metric, stat) for _, z in za]; pb = [per_example(z, metric, stat) for _, z in zb]
                        if np.all(np.isnan(pa[0])):
                            continue
                        va, vb = np.nanmean(np.stack(pa), 0), np.nanmean(np.stack(pb), 0)
                        d, dlo, dhi = S.paired_cluster_bootstrap(vb, va, take)
                        r, rlo, rhi = S.ratio_cluster_bootstrap(vb, va, take)
                        seed_d = [float(np.nanmean(q) - np.nanmean(p)) for p, q in zip(pa, pb)]
                        pair_rows.append(dict(dataset=ds, protocol=prot, a=a_, b=b_, comparison=what, metric=metric, stat=stat, mean_a=float(np.nanmean(va)), mean_b=float(np.nanmean(vb)),
                                              diff=d, diff_lo=dlo, diff_hi=dhi, rel_change_pct=100 * (r - 1), rel_lo=100 * (rlo - 1), rel_hi=100 * (rhi - 1),
                                              seed_diffs=json.dumps([round(x, 5) for x in seed_d]), seed_consistent=bool(all(np.sign(x) == np.sign(d) for x in seed_d)), n_seeds=min(len(pa), len(pb))))
        # events (protocol A, test events of the four classes)
        for name in list(S.MODELS) + ["PERSIST"]:
            zs = runs(ds, name, "A") if name in S.MODELS else [(0, load(ds, "PERSIST", "A"))]
            dfs = []
            for s, _ in zs:
                p = S.metrics_path(ds, S.run_name(name, s) if name in S.MODELS else name, "A"); p = p.with_name(p.stem + "_events.csv")
                if p.exists():
                    d = pd.read_csv(p); d["seed"] = s; dfs.append(d)
            if not dfs:
                continue
            E = pd.concat(dfs, ignore_index=True)
            for stat in ("K1", "mean"):
                Es = (E[E.k == 0] if stat == "K1" else E).copy()
                cols = [c for c in Es.columns if c.endswith(("_post", "_window", "_pre")) or c in ("transition_realised", "transition_timing_error", "post_set_correct_at_post")]
                for c in cols:
                    Es[c] = pd.to_numeric(Es[c].map({True: 1.0, False: 0.0}).fillna(Es[c]) if Es[c].dtype == object else Es[c], errors="coerce")
                A = Es.groupby(["event_id", "cls", "take_key"], as_index=False)[cols].mean()
                for cls in S.EVENT_CLASSES:
                    sub = A[A.cls == cls]
                    if not len(sub):
                        continue
                    for c in cols:
                        v = sub[c].values.astype(float)
                        if np.isfinite(v).sum() == 0:
                            continue
                        m, lo, hi = S.cluster_bootstrap(v, sub.take_key.values.astype(str))
                        ev_rows.append(dict(dataset=ds, model=name, stat=stat, cls=cls, metric=c, value=m, ci_lo=lo, ci_hi=hi, n_events=int(np.isfinite(v).sum())))
                    ev_rows.append(dict(dataset=ds, model=name, stat=stat, cls=cls, metric="n_events_total", value=len(sub), ci_lo=np.nan, ci_hi=np.nan, n_events=len(sub)))
    acc = pd.DataFrame(acc_rows)
    acc[acc.protocol == "A"].to_csv(S.OUT / "fixed_s0_metrics.csv", index=False)
    acc[acc.protocol == "B"].to_csv(S.OUT / "end_to_end_metrics.csv", index=False)
    acc[acc.metric.isin(STRUCT_KEYS + ["part_macro_f1", "part_macro_auprc", "amount_rmse", "centroid_both", "normal_both", "both_frac"])].to_csv(S.OUT / "structural_metrics.csv", index=False)
    acc[acc.metric.str.startswith("q_")].to_csv(S.OUT / "wrench_metrics.csv", index=False)
    pd.DataFrame(div_rows).to_csv(S.OUT / "diversity_metrics.csv", index=False)
    pd.DataFrame(ev_rows).to_csv(S.OUT / "event_metrics.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(S.OUT / "paired_comparisons.csv", index=False)
    pd.DataFrame(avd_rows).to_csv(S.OUT / "accuracy_vs_diversity.csv", index=False)
    np.savez_compressed(S.OUT / "temporal_curves.npz", **curves)
    cfg = pd.DataFrame(cfg_rows); cfg.to_csv(S.OUT / "run_config.csv", index=False)
    mm2 = mm.merge(cfg.groupby("model").agg(n_params=("n_params", "first"), lambda_struct=("lambda_struct", "first"), lambda_aux=("lambda_aux", "first")).reset_index(), on="model", how="left") if len(cfg) else mm
    mm2.to_csv(S.OUT / "model_matrix.csv", index=False)
    decision(acc, pd.DataFrame(div_rows), pd.DataFrame(pair_rows))
    S.write_json(S.OUT / "experiment_config.json", dict(datasets=list(S.DATASETS), T=S.T, generated_frames="1..63 (frame 0 = s_0, never generated)", K=S.K_SAMPLES, seeds=list(S.SEEDS),
                                                       models={m: c["label"] for m, c in S.MODELS.items()}, lambda_grid=dict(struct=list(S.LAMBDA_STRUCT_GRID), aux=list(S.LAMBDA_AUX_GRID)),
                                                       chosen_lambdas={ds: S.chosen_lambdas(ds) for ds in S.DATASETS}, calibration={ds: {k: v for k, v in S.load_calibration(ds).items() if k in ("c_hard", "n_min", "beta", "tau_c", "tau_n", "sigma_r", "m_bar_train")} for ds in S.DATASETS},
                                                       soft_temperatures=dict(tau_c=S.TAU_C, tau_n=S.TAU_N), protocols=dict(A="fixed GT s_0; D: 1 trajectory, S: K = 10", B="10 sampled s_0 (frozen sampler_G), one trajectory each"),
                                                       statistics="take-cluster bootstrap, 1000 reps, 95 % CI; 3 seeds averaged per example; seed spread = std of per-seed test means",
                                                       runs=cfg.to_dict("records")))
    log.info("tables written: %d accuracy rows, %d diversity rows, %d paired rows, %d event rows", len(acc), len(div_rows), len(pair_rows), len(ev_rows))


def decision(acc, div, pairs):
    """Table 8: rule-based reading of the numbers (the report interprets them)."""
    rows = []
    if not len(pairs) or not len(div):
        pd.DataFrame([dict(question="(no model runs aggregated yet)", dataset=ds, answer="n/a", evidence="") for ds in S.DATASETS]).to_csv(S.OUT / "decision_summary.csv", index=False)
        return
    def pair(ds, a, b, metric, stat="K1", prot="A"):
        r = pairs[(pairs.dataset == ds) & (pairs.a == a) & (pairs.b == b) & (pairs.metric == metric) & (pairs.stat == stat) & (pairs.protocol == prot)]
        return None if not len(r) else r.iloc[0]
    def val(ds, model, metric, stat="K1", prot="A"):
        r = acc[(acc.dataset == ds) & (acc.model == model) & (acc.metric == metric) & (acc.stat == stat) & (acc.protocol == prot)]
        return np.nan if not len(r) else float(r.iloc[0].value)
    def dv(ds, model, metric, prot):
        r = div[(div.dataset == ds) & (div.model == model) & (div.metric == metric) & (div.protocol == prot)]
        return np.nan if not len(r) else float(r.iloc[0].value)
    def fmt(r):
        return "n/a" if r is None else f"{r.rel_change_pct:+.1f} % [{r.rel_lo:+.1f}, {r.rel_hi:+.1f}]"
    def verdict(rs, thr=-5.0):
        """rs: list of paired rows (b vs a); improvement = negative relative change."""
        rs = [r for r in rs if r is not None]
        if not rs:
            return "n/a"
        sig = [r for r in rs if r.rel_hi < 0]; big = [r for r in sig if r.rel_change_pct <= thr]
        worse = [r for r in rs if r.rel_lo > 0 and r.rel_change_pct >= -thr]
        if len(big) >= max(1, len(rs) // 2):
            return "mixed (" + ", ".join(f"{r.metric} {r.rel_change_pct:+.0f} %" for r in big) + " better; " + ", ".join(f"{r.metric} {r.rel_change_pct:+.0f} %" for r in worse) + " worse)" if worse else "yes"
        if len(worse) > len(sig):
            return "no (worse)" if any(r.rel_change_pct >= 5 for r in worse) else "no (slightly worse)"
        if sig:
            return "marginal (< 5 %)"
        return "no"
    questions = []
    for ds in S.DATASETS:
        struct_pairs = lambda a, b, stat="K1", prot="A": [pair(ds, a, b, m, stat, prot) for m in STRUCT_KEYS + ["q_rel_l1", "E_C"]]
        ev = lambda a, b, stat="K1", prot="A": "; ".join(f"{m}: {fmt(pair(ds, a, b, m, stat, prot))}" for m in STRUCT_KEYS + ["q_rel_l1", "E_C"])
        q = {}
        q["Does output structural supervision help?"] = (verdict(struct_pairs("D0", "D1")) + " (det) / " + verdict(struct_pairs("S0", "S1")) + " (diff)", "D1 vs D0: " + ev("D0", "D1") + " | S1 vs S0: " + ev("S0", "S1"))
        q["Does hidden auxiliary supervision help beyond it?"] = (verdict(struct_pairs("D1", "D2")) + " (det) / " + verdict(struct_pairs("S1", "S2")) + " (diff)", "D2 vs D1: " + ev("D1", "D2") + " | S2 vs S1: " + ev("S1", "S2"))
        q["Does the stochastic future improve K1 structural accuracy?"] = (verdict(struct_pairs("D1", "S1")), "S1 vs D1 (K1): " + ev("D1", "S1") + " | S0 vs D0 (K1): " + ev("D0", "S0"))
        q["Does the stochastic future improve only best-of-K?"] = (f"best-of-10: {verdict(struct_pairs('D1', 'S1', 'best10'))}; mean sample: {verdict(struct_pairs('D1', 'S1', 'mean'))}",
                                                                   "S1 vs D1 best-of-10: " + ev("D1", "S1", "best10") + " | mean: " + ev("D1", "S1", "mean"))
        dc, dz, dq = (np.nanmean([dv(ds, m, k, "A") for m in S.DIFF]) for k in ("D_C", "D_Z", "D_q"))
        ec = np.nanmean([val(ds, m, "E_C") for m in S.DIFF]); ham = np.nanmean([val(ds, m, "part_hamming") for m in S.DIFF]); qe = np.nanmean([val(ds, m, "q_rel_l1") for m in S.DIFF])
        dpart = np.nanmean([dv(ds, m, "D_part", "A") for m in S.DIFF])
        q["Where does fixed-s0 diversity live?"] = ("dense" if (dpart / max(ham, 1e-9) < 0.5 and dq / max(qe, 1e-9) < 0.5) else "dense + structural",
                                                   f"S models, protocol A: D_C {dc:.3f} vs E_C {ec:.3f} ({dc / max(ec, 1e-9):.2f}); D_part {dpart:.3f} vs Hamming error {ham:.3f} ({dpart / max(ham, 1e-9):.2f}); D_q {dq:.3f} vs wrench rel L1 {qe:.3f} ({dq / max(qe, 1e-9):.2f}); D_Z {dz:.3f}")
        dzi = np.nanmean([dv(ds, m, "D_Z", "B") for m in S.DET]); dqi = np.nanmean([dv(ds, m, "D_q", "B") for m in S.DET]); dci = np.nanmean([dv(ds, m, "D_C", "B") for m in S.DET])
        q["Does initial sampling create most structural diversity?"] = ("yes" if dzi > 1.5 * dz else ("comparable" if dzi > 0.67 * dz else "no"),
                                                                        f"initial (D models, sampled s_0): D_Z {dzi:.3f}, D_q {dqi:.3f}, D_C {dci:.3f} | future (S models, fixed s_0): D_Z {dz:.3f}, D_q {dq:.3f}, D_C {dc:.3f}")
        k1 = verdict(struct_pairs("D1", "S1")); best = verdict(struct_pairs("D1", "S1", "best10"))
        jit = pair(ds, "D1", "S1", "jitter_dense"); worse_k1 = any(r is not None and r.rel_lo > 0 and r.rel_change_pct >= 5 for r in struct_pairs("D1", "S1"))
        noisy = jit is not None and jit.rel_change_pct > 100
        if k1 == "yes" and dz > 0.67 * dzi:
            case = "Case B (future stochasticity necessary)"
        elif noisy and (worse_k1 or k1.startswith("no")):
            case = "Case A (initial stochasticity sufficient) with the Case-C observation that the stochastic future adds dense noise"
        else:
            case = "Case A (initial stochasticity sufficient)"
        q["Is future temporal diffusion justified?"] = (case, f"K1 structural S1 vs D1: {k1}; best-of-10: {best}; jitter S1 vs D1: {fmt(jit)}; D_Z future / initial = {dz / max(dzi, 1e-9):.2f}")
        for qq, (ans, evidence) in q.items():
            rows.append(dict(question=qq, dataset=ds, answer=ans, evidence=evidence))
    pd.DataFrame(rows).to_csv(S.OUT / "decision_summary.csv", index=False)


if __name__ == "__main__":
    main()
