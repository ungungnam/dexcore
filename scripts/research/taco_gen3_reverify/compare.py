#!/usr/bin/env python
"""R1 / R5 / R6: the window-level comparisons, original vs FPS vs scene-scale run.

Everything is computed on the SAME windows in every run (the probe was run over the same window
grid; for the train split the scene probe covered all 81194 windows, so it is restricted to the
640-window subsample the original probe used). Uncertainty is a sequence-clustered bootstrap.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

G = Path("/result/uhnam/dexcore/bimart_taco_scene/contact_probe/gen/reverify")
B = "/result/uhnam/dexcore/bimart_taco/contact_probe/"
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)

w = pd.read_pickle(G / "window_level.pkl")
bk = pd.read_pickle(G / "bucket_level.pkl")

# ---- like-for-like window sets -------------------------------------------------------------
keys = {r: set(map(tuple, g[["split", "sequence_id", "start"]].drop_duplicates().values))
        for r, g in w.groupby("run")}
common = keys["orig"] & keys["scene"]
w["k"] = list(map(tuple, w[["split", "sequence_id", "start"]].values))
bk["k"] = list(map(tuple, bk[["split", "sequence_id", "start"]].values))
print("windows per run:", {r: len(v) for r, v in keys.items()}, "| orig&scene common:", len(common))
w = w[w.k.isin(common)].copy()
bk = bk[bk.k.isin(common)].copy()
print(w.groupby(["run", "split"]).size().unstack().to_string())

rng = np.random.default_rng(0)
NB = 2000


def boot_paired(a: pd.DataFrame, b: pd.DataFrame, col: str, cluster="sequence_id"):
    """Mean(b) - mean(a) with a sequence-clustered paired bootstrap; a and b share the window keys."""
    m = a.merge(b, on=["split", "sequence_id", "start", "hand"], suffixes=("_a", "_b"))
    d = (m[f"{col}_b"] - m[f"{col}_a"]).values
    cl = pd.factorize(m[cluster])[0]
    S = cl.max() + 1
    idx = [np.where(cl == s)[0] for s in range(S)]
    est = np.empty(NB)
    for i in range(NB):
        pick = rng.integers(0, S, S)
        est[i] = d[np.concatenate([idx[p] for p in pick])].mean()
    return float(d.mean()), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), len(m), S


# ============================= R1: headline + stages =======================================
print("\n" + "=" * 110)
print("R1 (a,b): per split x hand, all three runs")
t = w.groupby(["split", "hand", "run"])[["motion_mm", "contact_mm", "total_mm", "cos",
                                          "contact_map_err_mm", "touch_iou", "gt_touch_frac"]].mean()
print(t.round(2).unstack("run").to_string())
t.round(4).to_csv(G / "r1_split_hand_run.csv")

print("\nR1: pooled over hands (share_contact = contact/(contact+motion) in the vector sense is in the attribution file)")
p = w.groupby(["split", "run"])[["motion_mm", "contact_mm", "total_mm", "cos"]].mean().round(2)
print(p.unstack("run").to_string())

print("\nR1 (paired orig -> scene, sequence-clustered bootstrap):")
rows = []
for split in sorted(w.split.unique()):
    a = w[(w.run == "orig") & (w.split == split)]
    b = w[(w.run == "scene") & (w.split == split)]
    if not len(b):
        continue
    for col in ("motion_mm", "contact_mm", "total_mm", "contact_map_err_mm", "touch_iou"):
        d, lo, hi, n, S = boot_paired(a, b, col)
        rows.append(dict(split=split, metric=col, orig=a[col].mean(), scene=b[col].mean(),
                         delta=d, lo=lo, hi=hi, n_window_hands=n, n_seq=S))
r1 = pd.DataFrame(rows)
print(r1.round(3).to_string(index=False))
r1.to_csv(G / "r1_paired_deltas.csv", index=False)

print("\nR1 (c): contact and motion R - L per split and run")
rl = w.pivot_table(index=["run", "split"], columns="hand", values=["contact_mm", "motion_mm"], aggfunc="mean")
rl["contact_R_minus_L"] = rl[("contact_mm", "R")] - rl[("contact_mm", "L")]
rl["motion_R_minus_L"] = rl[("motion_mm", "R")] - rl[("motion_mm", "L")]
print(rl[["contact_R_minus_L", "motion_R_minus_L"]].round(2).to_string())
rl.round(4).to_csv(G / "r1_hand_asymmetry.csv")

print("\nR1 (d): train vs test_1 gap (pooled hands)")
gap = w[w.split.isin(["train", "test_1"])].groupby(["run", "split"])[["motion_mm", "contact_mm", "total_mm"]].mean().unstack("split")
for c in ("motion_mm", "contact_mm", "total_mm"):
    gap[(c, "ratio")] = gap[(c, "test_1")] / gap[(c, "train")]
print(gap.round(2).to_string())

# ============================= R5: novelty / bowl / H1 =====================================
nov = pd.read_csv(B + "novelty_tags.csv")
ms = pd.read_csv(B + "mesh_size.csv").set_index("mesh")
w2 = w.merge(nov.drop(columns=[c for c in ("split", "verb", "tool_cat", "target_cat", "tool_mesh",
                                           "target_mesh", "triplet") if c in nov.columns]),
             on="sequence_id", how="left")
print("\n" + "=" * 110)
print("R5 (a): error by novelty class (touching windows only, gt_touch_frac>0)")
tw = w2[w2.gt_touch_frac > 0].copy()


def cls(r):
    if r.target_cat == "bowl" and r.split in ("test_3", "test_4"):
        return "bowl_target"
    if r.split == "train":
        return "train"
    if not r.tool_cat_seen:
        return "new_tool_cat"
    if not r.target_mesh_seen and r.target_cat_seen:
        return "new_target_mesh_seen_cat"
    if r.mesh_pair_seen:
        return "both_meshes_seen"
    return "other_seen"


tw["cls"] = tw.apply(cls, axis=1)
cl = tw.groupby(["cls", "run"])[["contact_mm", "motion_mm"]].mean().unstack("run")
cnt = tw.groupby("cls").size().rename("n_window_hands")
print(pd.concat([cl.round(1), cnt], axis=1).to_string())
cl.round(3).to_csv(G / "r5_novelty_classes.csv")


def ols_cluster(y, X, cluster):
    X = np.asarray(X, float); y = np.asarray(y, float); n, k = X.shape
    XtX = np.linalg.pinv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    g = pd.factorize(cluster)[0]; Gn = g.max() + 1
    meat = np.zeros((k, k))
    for j in range(Gn):
        m = g == j; s = X[m].T @ e[m]; meat += np.outer(s, s)
    V = XtX @ meat @ XtX * (Gn / (Gn - 1)) * ((n - 1) / (n - k))
    se = np.sqrt(np.diag(V))
    return b, se, 2 * stats.t.sf(np.abs(b / se), Gn - 1), Gn


print("\nR5 (b): bowl-target coefficient (mm), sequence-clustered SE, one model per run")
rows = []
for run, g in tw.groupby("run"):
    g = g.copy()
    g["tgt_size"] = ms.reindex(g.target_mesh).radius_m.values
    g["tool_size"] = ms.reindex(g.tool_mesh).radius_m.values
    g["novel_tgt"] = (~g.target_mesh_seen.astype(bool)).astype(float)
    g["novel_tool"] = (~g.tool_mesh_seen.astype(bool)).astype(float)
    g["handR"] = (g.hand == "R").astype(float)
    g["bowl"] = (g.target_cat == "bowl").astype(float)
    g = g.dropna(subset=["tgt_size", "tool_size"])
    for ycol in ("contact_mm", "motion_mm"):
        for name, verb in (("no_verb_dummies", False), ("with_verb_dummies", True)):
            X = pd.DataFrame({"const": 1.0, "bowl": g.bowl, "novel_tgt": g.novel_tgt,
                              "novel_tool": g.novel_tool, "handR": g.handR,
                              "touch": g.gt_touch_frac, "tgt_size": g.tgt_size,
                              "tool_size": g.tool_size}, index=g.index)
            if verb:
                X = pd.concat([X, pd.get_dummies(g.verb, prefix="v", drop_first=True).astype(float)], axis=1)
            X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
            b, se, p, Gn = ols_cluster(g[ycol], X, g.sequence_id)
            d = dict(zip(X.columns, b)); s = dict(zip(X.columns, se)); pv = dict(zip(X.columns, p))
            rows.append(dict(run=run, y=ycol, model=name, n=len(g), n_seq=Gn,
                             bowl=d["bowl"], se=s["bowl"], p=pv["bowl"],
                             novel_tgt=d["novel_tgt"], novel_tool=d["novel_tool"],
                             tgt_size=d["tgt_size"], handR=d["handR"]))
r5 = pd.DataFrame(rows)
print(r5.round(2).to_string(index=False))
r5.to_csv(G / "r5_bowl_regression.csv", index=False)

print("\nR5 (d): frame-bucket profile, touching windows, contact error; gap test_3 - test_1")
bkt = bk.merge(w[["run", "split", "sequence_id", "start", "hand", "gt_touch_frac"]]
               .rename(columns={"gt_touch_frac": "wtouch"}),
               on=["run", "split", "sequence_id", "start", "hand"])
bkt = bkt[bkt.wtouch > 0]
bkt["first"] = bkt.start.eq(8)
for sub, name in ((bkt, "all"), (bkt[~bkt["first"]], "nonfirst"), (bkt[bkt["first"]], "first")):
    piv = sub.groupby(["run", "bucket", "split"]).contact_mm.mean().unstack("split")
    if "test_3" in piv and "test_1" in piv:
        piv["gap31"] = piv.test_3 - piv.test_1
    print(f"--- {name}"); print(piv.round(1).to_string())
    piv.round(3).to_csv(G / f"r5_frame_profile_{name}.csv")

# ============================= R6: G2 support, G4 subject ==================================
print("\n" + "=" * 110)
print("R6 (a) G2: test_1/train gap ratio by support of the verb / triplet / mesh pair")
sup = tw[tw.split.isin(["train", "test_1"])].copy()
rows = []
for col in ("n_train_seq_same_verb", "n_train_seq_same_triplet", "n_train_seq_same_tool_mesh"):
    for run, g in sup.groupby("run"):
        g = g.dropna(subset=[col]).copy()
        g["q"] = pd.qcut(g[col].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
        for q, gg in g.groupby("q", observed=True):
            a = gg[gg.split == "train"]; b = gg[gg.split == "test_1"]
            if len(a) and len(b):
                rows.append(dict(support=col, run=run, q=q, support_median=float(gg[col].median()),
                                 train_contact=a.contact_mm.mean(), test1_contact=b.contact_mm.mean(),
                                 ratio_contact=b.contact_mm.mean() / a.contact_mm.mean(),
                                 train_motion=a.motion_mm.mean(), test1_motion=b.motion_mm.mean(),
                                 ratio_motion=b.motion_mm.mean() / a.motion_mm.mean(),
                                 n_train=len(a), n_test=len(b)))
r6 = pd.DataFrame(rows)
print(r6.round(2).to_string(index=False))
r6.to_csv(G / "r6_g2_support.csv", index=False)

print("\nR6 (b) G4: matched (subject, triplet, date) cells, test_1 vs train")
sj = pd.read_csv(B + "gen/betas_per_sequence_subjects.csv")[["sequence_id", "subject", "date", "take"]]
sw = tw.merge(sj, on="sequence_id", how="left")
print("subject coverage:", sw.subject.notna().mean().round(3))
rows = []
for run, g in sw.groupby("run"):
    gg = g[g.split.isin(["train", "test_1"])].dropna(subset=["subject"])
    if gg.split.nunique() < 2:            # the FPS run has no train probe
        print(f"  {run}: only {list(gg.split.unique())} -- skipped")
        continue
    seq = gg.groupby(["split", "subject", "triplet", "date", "sequence_id"])[["contact_mm", "motion_mm"]].mean().reset_index()
    cell = seq.groupby(["subject", "triplet", "date", "split"])[["contact_mm", "motion_mm"]].mean().unstack("split")
    if ("contact_mm", "train") not in cell.columns or ("contact_mm", "test_1") not in cell.columns:
        print(f"  {run}: matched cells need both splits -- skipped")
        continue
    both = cell.dropna()
    rows.append(dict(run=run, n_cells=len(both),
                     train_contact=both[("contact_mm", "train")].mean(),
                     test1_contact=both[("contact_mm", "test_1")].mean(),
                     train_motion=both[("motion_mm", "train")].mean(),
                     test1_motion=both[("motion_mm", "test_1")].mean(),
                     frac_test_gt_train_contact=float((both[("contact_mm", "test_1")] > both[("contact_mm", "train")]).mean()),
                     frac_test_gt_train_motion=float((both[("motion_mm", "test_1")] > both[("motion_mm", "train")]).mean())))
    both.round(3).to_csv(G / f"r6_g4_cells_{run}.csv")
r6b = pd.DataFrame(rows)
print(r6b.round(2).to_string(index=False))
r6b.to_csv(G / "r6_g4_matched_cells.csv", index=False)
print("\n->", G)
