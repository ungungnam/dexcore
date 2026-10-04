#!/usr/bin/env python
"""Aggregate the test-set evaluation of one dataset:  DC_DATASET=<ds> python aggregate.py
  results/aggregate.csv        model x K: pooled mean + 95 % take-bootstrap CI of every metric,
                               macro (mean of group means), first-onset-only subset, counts
  results/per_group.csv        group x split x model x K means
  results/curves.csv           split x model x K x t: mean per-t curves (err, derr, dmag)
  results/main_table_<split>.md   the report tables (main rows + extended rows)
  results/training_summary.csv    best validation loss / steps per fold and model
"""
from __future__ import annotations

import glob
import json
import logging

import numpy as np
import pandas as pd
import torch

import hc_common as H

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("agg")
METRICS = ["E_C", "E_dC", "dmag_pred", "dmag_true", "pattern_L2", "centroid", "mass_abs", "s0_err", "E_C_last16", "max_abs", "neg_frac"]
MAIN_ROWS = [("static_gt", 0, "B0 static GT-init"), ("gtinit_vf", 0, "B1 GT-init + VF"),
             ("samplerG_vf", 1, "B2 p(S0\\|G) + VF, K=1"), ("samplerG_vf", 10, "B2 p(S0\\|G) + VF, best-of-10"),
             ("samplerGT_vf", 1, "B3 p(S0\\|G,tau) + VF, K=1"), ("samplerGT_vf", 10, "B3 p(S0\\|G,tau) + VF, best-of-10")]
EXTRA_ROWS = [("samplerG_vf", 5, "B2, best-of-5"), ("samplerG_vf", -1, "B2, mean over 10 samples"),
              ("samplerGT_vf", 5, "B3, best-of-5"), ("samplerGT_vf", -1, "B3, mean over 10 samples"),
              ("samplerG_static", 1, "p(S0\\|G) held static, K=1"), ("samplerG_static", 10, "p(S0\\|G) held static, best-of-10"),
              ("samplerGT_static", 1, "p(S0\\|G,tau) held static, K=1"), ("samplerGT_static", 10, "p(S0\\|G,tau) held static, best-of-10"),
              ("gtinit_vfnoise", 0, "B1 with the noise-trained VF"), ("samplerG_vfnoise", 1, "B2 with the noise-trained VF, K=1"),
              ("samplerG_vfnoise", 10, "B2 with the noise-trained VF, best-of-10"), ("samplerGT_vfnoise", 10, "B3 with the noise-trained VF, best-of-10")]


def json_idx(part):
    return json.load(open(H.OUT / "manifests" / f"{H.SPLIT}{H.FOLD}.json")).get(part, [])


def main():
    res = H.OUT / "results"; res.mkdir(exist_ok=True)
    files = [str(H.OUT / "eval" / f"{H.SPLIT}{H.FOLD}" / "per_example.csv")]
    R = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    log.info("%s: %d fold files, %d rows, splits %s", H.DATASET, len(files), len(R), sorted(R.split.unique()))
    rows, grows = [], []
    for (split, model, K), g in R.groupby(["split", "model", "K"]):
        row = dict(split=split, model=model, K=K, n_examples=len(g), n_takes=g.take_key.nunique(), n_folds=g.fold.nunique(), n_groups=g.group.nunique())
        for m in METRICS:
            pt, lo, hi = H.cluster_bootstrap(g[m].values, g.take_key.values, n_boot=500)
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = pt, lo, hi
            row[f"{m}_macro"] = float(g.groupby("group")[m].mean().mean())
            row[f"{m}_first"] = float(g[g.first_onset == 1][m].mean())
            row[f"{m}_median"] = float(g[m].median())
        rows.append(row)
        gg = g.groupby("group")[METRICS].mean(); gg["n_examples"] = g.groupby("group").size()
        gg = gg.reset_index(); gg["split"] = split; gg["model"] = model; gg["K"] = K
        grows.append(gg)
    A = pd.DataFrame(rows); A.to_csv(res / "aggregate.csv", index=False)
    # the labelled extra test set (TACO test_2..4), evaluated with the same models, reported apart
    extra = H.OUT / "eval" / f"{H.SPLIT}{H.FOLD}_extra" / "per_example.csv"
    if extra.exists():
        meta = pd.read_csv(H.OUT / "sequences_meta.csv", usecols=["orig_split", "group"])
        E = pd.concat([R, pd.read_csv(extra)], ignore_index=True)
        E["orig_split"] = meta.orig_split.values[E.example.values]
        E["group_seen_in_train"] = E.group.isin(set(meta.group.values[np.asarray(json_idx("train"))]))
        (E.groupby(["orig_split", "group_seen_in_train", "model", "K"])[METRICS].mean()
         .join(E.groupby(["orig_split", "group_seen_in_train", "model", "K"]).size().rename("n")).reset_index().to_csv(res / "by_orig_split.csv", index=False))
    pd.concat(grows).to_csv(res / "per_group.csv", index=False)
    # paired per-example differences of the key comparisons (same examples, take bootstrap)
    pairs = [("Q2 VF - static (GT init)", "gtinit_vf", 0, "static_gt", 0), ("Q3 sampled(G,K=1) - GT init", "samplerG_vf", 1, "gtinit_vf", 0),
             ("Q3 sampled(G,best10) - GT init", "samplerG_vf", 10, "gtinit_vf", 0), ("Q4 GT-sampler - G-sampler (K=1)", "samplerGT_vf", 1, "samplerG_vf", 1),
             ("Q4 GT-sampler - G-sampler (best10)", "samplerGT_vf", 10, "samplerG_vf", 10), ("Q4 S0 only: GT - G (K=1)", "samplerGT_static", 1, "samplerG_static", 1),
             ("Q4 S0 only: GT - G (best10)", "samplerGT_static", 10, "samplerG_static", 10), ("VF vs static, sampled G best10", "samplerG_vf", 10, "samplerG_static", 10),
             ("VF vs static, sampled G K=1", "samplerG_vf", 1, "samplerG_static", 1), ("dense - B2 best10", "dense", 0, "samplerG_vf", 10),
             ("noise VF - VF (GT init)", "gtinit_vfnoise", 0, "gtinit_vf", 0), ("dense - B1", "dense", 0, "gtinit_vf", 0)]
    prow = []
    for split in R.split.unique():
        for label, m1, k1, m2, k2 in pairs:
            a = R[(R.split == split) & (R.model == m1) & (R.K == k1)].set_index("example"); b = R[(R.split == split) & (R.model == m2) & (R.K == k2)].set_index("example")
            common = a.index.intersection(b.index)
            if len(common) == 0:
                continue
            d = dict(split=split, comparison=label, a=f"{m1} K{k1}", b=f"{m2} K{k2}", n=len(common))
            for met in ("E_C", "E_dC", "pattern_L2", "mass_abs", "s0_err"):
                diff = a.loc[common, met].values - b.loc[common, met].values
                pt, lo, hi = H.cluster_bootstrap(diff, a.loc[common, "take_key"].values, n_boot=500)
                d[met], d[f"{met}_lo"], d[f"{met}_hi"] = pt, lo, hi
                d[f"{met}_frac_better"] = float(np.nanmean(diff < 0))
            prow.append(d)
    pd.DataFrame(prow).to_csv(res / "paired_differences.csv", index=False)
    # curves
    crow = []
    for d in [str(H.OUT / "eval" / f"{H.SPLIT}{H.FOLD}")]:
        split = H.SPLIT
        z = np.load(d + "/curves.npz")
        for key in z.files:
            if key == "example":
                continue
            name, Kk, met = key.split("__"); v = z[key]
            crow.append(pd.DataFrame(dict(split=split, model=name, K=int(Kk[1:]), metric=met, t=np.arange(v.shape[1]), value=np.nanmean(v, 0), n=len(v), fold=d[-1])))
    Cv = pd.concat(crow, ignore_index=True)
    Cv = Cv.groupby(["split", "model", "K", "metric", "t"]).apply(lambda g: pd.Series(dict(value=float(np.average(g.value, weights=g.n)), n=int(g.n.sum())))).reset_index()
    Cv.to_csv(res / "curves.csv", index=False)
    # training summary
    trow = []
    for f in sorted(glob.glob(str(H.CKPT / f"{H.SPLIT}{H.FOLD}" / "*.pt"))):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        trow.append(dict(fold=f.split("/")[-2], model=ck["model"] + ck.get("tag", ""), best_val=ck["best_val"], best_step=ck["best_step"], steps=ck["steps"], noise_sd=ck["noise_sd"],
                         stopped_by=ck.get("stopped_by", "?"), converged=ck.get("converged", False)))
    pd.DataFrame(trow).to_csv(res / "training_summary.csv", index=False)
    # markdown tables
    for split in sorted(A.split.unique()):
        S = A[A.split == split].set_index(["model", "K"])
        lines = [f"### {H.DATASET} — fixed split, test set ({int(S.n_examples.max())} test sequences, {int(S.n_takes.max())} takes)", "",
                 "| variant | E_C (raw) | 95 % CI | E_ΔC | mean pred Δ | mean true Δ | pattern L2 | mass abs | S0 err | E_C macro |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        def fmt(model, K, label):
            if (model, K) not in S.index:
                return None
            r = S.loc[(model, K)]
            return (f"| {label} | {r.E_C:.3f} | [{r.E_C_lo:.3f}, {r.E_C_hi:.3f}] | {r.E_dC:.3f} | {r.dmag_pred:.3f} | {r.dmag_true:.3f} | "
                    f"{r.pattern_L2:.4f} | {r.mass_abs:.2f} | {r.s0_err:.3f} | {r.E_C_macro:.3f} |")
        lines += [l for l in (fmt(*r) for r in MAIN_ROWS) if l]
        lines += ["", "Extended rows:", "", lines[2], lines[3]] + [l for l in (fmt(*r) for r in EXTRA_ROWS) if l]
        (res / f"main_table_{split}.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines[:11]))
    log.info("written %s", res)


if __name__ == "__main__":
    main()
