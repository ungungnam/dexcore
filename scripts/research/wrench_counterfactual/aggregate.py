#!/usr/bin/env python
"""Step 3: tables with take-cluster bootstrap intervals.   python aggregate.py
Reads <ds>/events_metrics.csv (event x setting) and <ds>/events_metrics_train.csv (train persistent
spatial, primary setting; thresholds only). Writes at the root:
  events.csv                     one row per test event, primary setting, all metadata and metrics
  summary_by_event_type.csv      dataset x class: medians / means with 95 % take-bootstrap CIs
  sensitivity_mu.csv, sensitivity_patch.csv, sensitivity_budget.csv   the same summaries per setting
  finger_transition_summary.csv  finger-level participation and per-finger change frequencies
  quadrant.csv                   fraction of persistent-spatial events with large contact change and small
                                 wrench change, thresholds = TRAIN quantiles (with sensitivity)
  thresholds.json                the train quantiles used
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import wc_common as W

METRICS = ["dC", "Q_pre", "Q_post", "delta_Q", "rel_change_Q", "ratio_Q", "R_pre_to_post", "R_post_to_pre", "cosine", "l1_rel_distance",
           "coverage_pre", "coverage_post", "strict_Q_pre", "strict_Q_post", "strict_rel_change_Q", "strict_R_pre_to_post", "strict_coverage_pre", "strict_coverage_post",
           "n_patches_pre", "n_patches_post", "n_fingers_pre", "n_fingers_post", "n_contact_pre", "n_contact_post"]
B_KEYS = ["force_K8", "torque_K8", "force_K16", "torque_K16", "support"]
N_BOOT = 1000


def load_all(suffix=""):
    return pd.concat([pd.read_csv(W.ds_out(ds) / f"events_metrics{suffix}.csv", dtype={"event_id": str, "sequence_id": str, "take_key": str}) for ds in W.DATASETS
                      if (W.ds_out(ds) / f"events_metrics{suffix}.csv").exists()], ignore_index=True)


def ci_cols(vals, clusters, stat):
    if stat == "median":
        return W.cluster_bootstrap_median(vals, clusters, n_boot=N_BOOT)
    return W.cluster_bootstrap(vals, clusters, n_boot=N_BOOT)


def summarize(df, keys):
    """Per (keys) group: n, and for each metric the median [CI] and mean [CI]."""
    rows = []
    for key, g in df.groupby(keys, sort=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,))); row["n_events"] = len(g); row["n_takes"] = g.take_key.nunique()
        for m in METRICS + [f"B_{k}_{s}" for k in B_KEYS for s in ("pre", "post")] + [f"B_{k}_ratio" for k in B_KEYS] + [f"B_{k}_diff" for k in B_KEYS] + [f"B_{k}_retention" for k in B_KEYS]:
            if m not in g or g[m].notna().sum() == 0:
                continue
            v = g[m].values.astype(float); cl = g.take_key.values
            for stat in ("median", "mean"):
                pt, lo, hi = ci_cols(v, cl, stat)
                row[f"{m}_{stat}"], row[f"{m}_{stat}_lo"], row[f"{m}_{stat}_hi"] = pt, lo, hi
        # paired fractions
        row["frac_Q_post_gt_pre"] = float((g.Q_post > g.Q_pre).mean())
        row["frac_retention_ge_0.8"] = float((g.R_pre_to_post >= 0.8).mean()); row["frac_retention_ge_0.9"] = float((g.R_pre_to_post >= 0.9).mean())
        row["frac_abs_rel_change_le_0.1"] = float((g.rel_change_Q.abs() <= 0.1).mean()); row["frac_abs_rel_change_le_0.25"] = float((g.rel_change_Q.abs() <= 0.25).mean())
        row["frac_rel_change_gt_0.25"] = float((g.rel_change_Q > 0.25).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def add_B(df):
    for k in B_KEYS:
        if f"B_{k}_pre" in df:
            df[f"B_{k}_ratio"] = (df[f"B_{k}_post"] + 1e-6) / (df[f"B_{k}_pre"] + 1e-6)
            df[f"B_{k}_diff"] = df[f"B_{k}_post"] - df[f"B_{k}_pre"]
            df[f"B_{k}_retention"] = np.minimum(df[f"B_{k}_pre"] / (df[f"B_{k}_post"] + 1e-6), 1.0).where(np.maximum(df[f"B_{k}_pre"], df[f"B_{k}_post"]) > 1e-6)
    return df


def main():
    M = add_B(load_all()); Mt = add_B(load_all("_train"))
    P = M[M.setting == "primary"].copy()
    P["class_report"] = np.where(P.primary, "persistent_spatial", np.where(P.secondary, "persistent_mixed", np.where(P.cls.isin(["onset", "release", "transient", "amount"]), P.cls, "other")))
    P.to_csv(W.OUT / "events.csv", index=False)
    # ---- summaries by class (primary setting)
    S = summarize(P[P.class_report != "other"], ["dataset", "class_report"])
    S.to_csv(W.OUT / "summary_by_event_type.csv", index=False)
    # ---- sensitivity: the persistent-spatial (primary) and control classes under every setting
    sens = {}
    for name, sel in (("mu", M.setting.str.startswith("mu") | (M.setting == "primary")), ("patch", M.setting.str.startswith("merge") | M.setting.str.startswith("thr") | (M.setting == "primary")),
                      ("budget", M.setting.str.startswith("budget") | (M.setting == "primary"))):
        d = M[sel].copy(); d["class_report"] = np.where(d.primary, "persistent_spatial", np.where(d.secondary, "persistent_mixed", d.cls))
        d = d[d.class_report.isin(["persistent_spatial", "persistent_mixed", "onset", "release"])]
        sens[name] = summarize(d, ["dataset", "class_report", "setting", "mu", "thr", "merge", "budget"])
        sens[name].to_csv(W.OUT / f"sensitivity_{name}.csv", index=False)
    # ---- finger transitions (primary setting, primary + secondary classes + controls)
    rows = []
    for (ds, cls), g in P[P.class_report != "other"].groupby(["dataset", "class_report"]):
        row = dict(dataset=ds, class_report=cls, n_events=len(g), n_fingers_pre_mean=g.n_fingers_pre.mean(), n_fingers_post_mean=g.n_fingers_post.mean(),
                   n_patches_pre_mean=g.n_patches_pre.mean(), n_patches_post_mean=g.n_patches_post.mean(),
                   frac_same_finger_set=float((g.fingers_pre == g.fingers_post).mean()), n_slid_mean=g.n_slid.mean(), n_appeared_mean=g.n_appeared.mean(), n_disappeared_mean=g.n_disappeared.mean())
        for p in W.PARTS:
            for state in ("stable", "slid", "appeared", "disappeared", "absent"):
                row[f"{p}_{state}"] = float((g[f"fc_{p}"] == state).mean())
            row[f"{p}_present_pre"] = float(g.fingers_pre.fillna("").str.contains(p).mean()); row[f"{p}_present_post"] = float(g.fingers_post.fillna("").str.contains(p).mean())
        # wrench change conditional on the number of reconfigured fingers
        for k in (0, 1, 2):
            m = (g.n_slid + g.n_appeared + g.n_disappeared) == k if k < 2 else (g.n_slid + g.n_appeared + g.n_disappeared) >= 2
            row[f"n_events_changed{k}{'plus' if k == 2 else ''}"] = int(m.sum())
            row[f"median_abs_rel_change_Q_changed{k}{'plus' if k == 2 else ''}"] = float(g.rel_change_Q[m].abs().median()) if m.any() else np.nan
            row[f"median_retention_changed{k}{'plus' if k == 2 else ''}"] = float(g.R_pre_to_post[m].median()) if m.any() else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_csv(W.OUT / "finger_transition_summary.csv", index=False)
    # ---- quadrant analysis with TRAIN thresholds (persistent spatial only)
    thr, quad = {}, []
    for ds in W.DATASETS:
        t = Mt[(Mt.dataset == ds) & (Mt.setting == "primary")]
        te = P[(P.dataset == ds) & (P.class_report == "persistent_spatial")]
        if not len(t) or not len(te):
            continue
        thr[ds] = dict(n_train=int(len(t)), dC_q50=float(t.dC.quantile(0.5)), dC_q75=float(t.dC.quantile(0.75)), abs_rel_change_Q_q50=float(t.rel_change_Q.abs().quantile(0.5)),
                       abs_rel_change_Q_q25=float(t.rel_change_Q.abs().quantile(0.25)), retention_q50=float(t.R_pre_to_post.quantile(0.5)), l1_q50=float(t.l1_rel_distance.quantile(0.5)))
        for dC_q, rq in (("dC_q50", "abs_rel_change_Q_q50"), ("dC_q75", "abs_rel_change_Q_q50"), ("dC_q50", "abs_rel_change_Q_q25"), ("dC_q75", "abs_rel_change_Q_q25")):
            big = te.dC >= thr[ds][dC_q]; small = te.rel_change_Q.abs() <= thr[ds][rq]
            for name, flag in (("large_dC_small_dQ", big & small), ("large_dC_large_dQ", big & ~small), ("small_dC_small_dQ", ~big & small), ("small_dC_large_dQ", ~big & ~small)):
                pt, lo, hi = W.cluster_bootstrap_frac(flag.values, te.take_key.values, n_boot=N_BOOT)
                quad.append(dict(dataset=ds, dC_threshold=dC_q, dQ_threshold=rq, dC_value=thr[ds][dC_q], dQ_value=thr[ds][rq], quadrant=name, n=int(flag.sum()), frac=pt, lo=lo, hi=hi))
            # among the large-change events: fraction with small wrench change
            if big.any():
                pt, lo, hi = W.cluster_bootstrap_frac(small[big].values, te.take_key.values[big.values], n_boot=N_BOOT)
                quad.append(dict(dataset=ds, dC_threshold=dC_q, dQ_threshold=rq, dC_value=thr[ds][dC_q], dQ_value=thr[ds][rq], quadrant="small_dQ_given_large_dC", n=int((big & small).sum()), frac=pt, lo=lo, hi=hi))
    pd.DataFrame(quad).to_csv(W.OUT / "quadrant.csv", index=False)
    W.write_json(W.OUT / "thresholds.json", thr)
    # ---- correlation of contact change with wrench change (continuous, primary result)
    corr = []
    from scipy.stats import spearmanr
    for (ds, cls), g in P[P.class_report.isin(["persistent_spatial", "persistent_mixed"])].groupby(["dataset", "class_report"]):
        for m in ("rel_change_Q", "R_pre_to_post", "l1_rel_distance", "cosine", "strict_rel_change_Q"):
            ok = g[m].notna()
            if ok.sum() > 5:
                rho = spearmanr(g.dC[ok], g[m][ok]).correlation
                corr.append(dict(dataset=ds, class_report=cls, metric=m, spearman_with_dC=float(rho), n=int(ok.sum())))
    pd.DataFrame(corr).to_csv(W.OUT / "correlation_dC_vs_wrench.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["dataset", "class_report", "n_events", "dC_median", "Q_pre_median", "Q_post_median", "rel_change_Q_median", "rel_change_Q_median_lo", "rel_change_Q_median_hi",
            "R_pre_to_post_median", "R_pre_to_post_median_lo", "R_pre_to_post_median_hi", "cosine_median", "frac_retention_ge_0.8", "frac_abs_rel_change_le_0.25", "B_force_K8_ratio_median", "B_torque_K8_ratio_median", "B_support_ratio_median"]
    print(S[[c for c in cols if c in S]].round(3).to_string())
    print(pd.DataFrame(quad)[lambda d: d.quadrant == "small_dQ_given_large_dC"].round(3).to_string())
    print(pd.DataFrame(corr).round(3).to_string())


if __name__ == "__main__":
    main()
