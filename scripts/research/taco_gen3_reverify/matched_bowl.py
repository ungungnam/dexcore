#!/usr/bin/env python
"""R5(c) matched pairs + R6(a) support: the two analyses the earlier tables could not reuse.

Matched pairs: same verb and same TOOL MESH, target = bowl (test_3) vs a seen target (train/test_1).
Restricted to windows where the DENSE truth says the right hand really holds the tool, so the old
label defect cannot masquerade as error. Support: how many TRAIN sequences share the verb / triplet /
tool mesh, counted from the sequence index (the novelty tags cover test sequences only).
"""
from pathlib import Path

import numpy as np
import pandas as pd

G = Path("/result/uhnam/dexcore/bimart_taco_scene/contact_probe/gen/reverify")
ROOTS = {"orig": "/result/uhnam/dexcore/bimart_taco/", "scene": "/result/uhnam/dexcore/bimart_taco_scene/"}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)

tr = pd.read_csv(G / "truncation_per_window.csv")
w = pd.read_pickle(G / "window_level.pkl")


def half_err(root, split):
    """Mean |pred - gt| in mm over each quarter of the 2048-long contact map."""
    z = np.load(root + f"contact_probe/{split}_contact_maps.npz")
    e = np.abs(z["pred"].astype(np.float32) - z["gt"].astype(np.float32)) * 1000
    di = pd.read_csv(root + f"contact_probe/{split}_contact_maps_index.csv")
    for name, sl in (("err_L_tool", slice(0, 512)), ("err_L_targ", slice(512, 1024)),
                     ("err_R_tool", slice(1024, 1536)), ("err_R_targ", slice(1536, 2048))):
        di[name] = e[..., sl].mean((1, 2))
    di["split"] = split
    return di[["split", "sequence_id", "start", "err_L_tool", "err_L_targ", "err_R_tool", "err_R_targ"]]


rows = []
for run, root in ROOTS.items():
    H = pd.concat([half_err(root, s) for s in ("train", "test_1", "test_3")])
    pw = (w[w.run == run].pivot_table(index=["split", "sequence_id", "start"], columns="hand",
                                      values=["contact_mm", "motion_mm", "contact_trans_mm",
                                              "contact_resid_mm", "gt_touch_frac"], aggfunc="mean"))
    pw.columns = [f"{a}_{b}" for a, b in pw.columns]
    t = tr[tr.run == run].drop(columns=[c for c in tr.columns
                                        if c.startswith(("contact_mm", "motion_mm",
                                                         "contact_map_err_mm", "touch_iou"))])
    D = (t.merge(H, on=["split", "sequence_id", "start"])
         .merge(pw.reset_index(), on=["split", "sequence_id", "start"]))
    held = D[D.dense_R_tool > 0.5]
    t3 = held[(held.split == "test_3") & (held.target_cat == "bowl")]
    ref = held[(held.split.isin(["train", "test_1"])) & (held.target_cat != "bowl")]
    pairs = sorted(set(zip(t3.verb, t3.tool_mesh)) & set(zip(ref.verb, ref.tool_mesh)))
    cols = ["contact_mm_R", "contact_trans_mm_R", "contact_resid_mm_R", "contact_mm_L",
            "err_R_tool", "err_R_targ", "err_L_tool", "err_L_targ", "motion_mm_R", "motion_mm_L",
            "gath_R_tool", "pred_R_tool", "dense_R_tool"]
    print(f"{run}: held windows {held.groupby('split').size().to_dict()}, matched pairs {len(pairs)}")
    for v, m in pairs:
        a = t3[(t3.verb == v) & (t3.tool_mesh == m)]
        b = ref[(ref.verb == v) & (ref.tool_mesh == m)]
        r = {"run": run, "verb": v, "tool_mesh": m, "n_bowl": len(a), "n_ref": len(b),
             "n_seq_bowl": a.sequence_id.nunique(), "n_seq_ref": b.sequence_id.nunique()}
        for c in cols:
            r[c + "_bowl"] = a[c].mean(); r[c + "_ref"] = b[c].mean()
        rows.append(r)
M = pd.DataFrame(rows)
M.to_csv(G / "r5_matched_pairs.csv", index=False)
print("\n=== matched pairs, pair-weighted means (bowl target vs seen target, same verb+tool mesh) ===")
agg = M.groupby("run")[[c for c in M.columns if c.endswith(("_bowl", "_ref"))]].mean()
show = ["contact_mm_R", "contact_mm_L", "err_R_tool", "err_R_targ", "err_L_tool", "err_L_targ",
        "motion_mm_R", "motion_mm_L", "contact_trans_mm_R", "contact_resid_mm_R", "pred_R_tool", "gath_R_tool"]
print(pd.DataFrame({c: [agg.loc[r, c + "_bowl"] for r in agg.index] for c in show},
                   index=[f"{r}_bowl" for r in agg.index]).round(1).to_string())
print(pd.DataFrame({c: [agg.loc[r, c + "_ref"] for r in agg.index] for c in show},
                   index=[f"{r}_ref" for r in agg.index]).round(1).to_string())
print("\n=== bowl - ref difference per run (pair-weighted; paired bootstrap over pairs) ===")
rng = np.random.default_rng(0)
out = []
for run, g in M.groupby("run"):
    for c in show:
        d = (g[c + "_bowl"] - g[c + "_ref"]).dropna().values
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        out.append(dict(run=run, metric=c, n_pairs=len(d), diff=d.mean(),
                        lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
O = pd.DataFrame(out)
print(O.round(1).pivot(index="metric", columns="run", values=["diff", "lo", "hi"]).to_string())
O.to_csv(G / "r5_matched_pair_diffs.csv", index=False)

# ---------------- support (G2) -------------------------------------------------------------
idx = pd.read_csv(ROOTS["scene"] + "sequence_index.csv")
trn = idx[idx.split == "train"]
sup_verb = trn.verb.value_counts(); sup_trip = trn.triplet.value_counts(); sup_tool = trn.tool_mesh.value_counts()
ww = w[w.split.isin(["train", "test_1"])].copy()
ww["sup_verb"] = ww.verb.map(sup_verb).fillna(0)
ww["sup_triplet"] = ww.triplet.map(sup_trip).fillna(0)
ww["sup_tool_mesh"] = ww.tool_mesh.map(sup_tool).fillna(0)
rows = []
for col in ("sup_verb", "sup_triplet", "sup_tool_mesh"):
    for run, g in ww.groupby("run"):
        if g.split.nunique() < 2:
            continue
        g = g.copy()
        g["q"] = pd.qcut(g[col].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
        for q, gg in g.groupby("q", observed=True):
            a = gg[gg.split == "train"]; b = gg[gg.split == "test_1"]
            if len(a) > 5 and len(b) > 5:
                rows.append(dict(support=col, run=run, q=q, median_support=float(gg[col].median()),
                                 train_contact=a.contact_mm.mean(), test1_contact=b.contact_mm.mean(),
                                 ratio_contact=b.contact_mm.mean() / a.contact_mm.mean(),
                                 train_motion=a.motion_mm.mean(), test1_motion=b.motion_mm.mean(),
                                 ratio_motion=b.motion_mm.mean() / a.motion_mm.mean(),
                                 n_train=len(a), n_test=len(b)))
Sdf = pd.DataFrame(rows)
print("\n=== G2: train->test_1 ratio by training support quartile ===")
print(Sdf.round(2).to_string(index=False))
Sdf.to_csv(G / "r6_g2_support.csv", index=False)
print("\n->", G)
