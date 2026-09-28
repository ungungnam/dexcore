#!/usr/bin/env python
"""R4: is the tool-contact label still truncated, and does the model now predict tool contact?

Per window and run, the right hand's TOOL contact fraction is read three ways:
  dense   every tool vertex                     -- the truth
  gath    the 512 BPS-sampled tool vertices     -- the label the model was trained and scored on
  pred    the contact model's own output        -- what it learned
Each run reads its OWN sequences/*.npz (the BPS indices differ: the scene run scales tool+target by
one factor, so the tool no longer overflows the basis), and its OWN dumped maps. The dense contact
is identical across roots, which is what makes the label comparison like-for-like.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/uhnam/workspace/dexcore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.analysis.bimart import meshes as M

OUT = Path("/result/uhnam/dexcore/bimart_taco_scene/contact_probe/gen/reverify")
ROOTS = {"orig": "/result/uhnam/dexcore/bimart_taco/",
         "scene": "/result/uhnam/dexcore/bimart_taco_scene/"}
PROBE = {"orig": "contact_probe/", "scene": "contact_probe/"}
md = M.load(ROOTS["orig"] + "assets/taco_mesh_dict.npy")


def n_tool(mesh_id):
    for k in (mesh_id, str(mesh_id), f"{int(mesh_id):03d}"):
        if k in md:
            return len(md[k]["verts_original"])
    raise KeyError(mesh_id)


rows = []
for run, root in ROOTS.items():
    idx_all = pd.read_csv(root + "sequence_index.csv").set_index("sequence_id")
    P = root + PROBE[run]
    for split in ("train", "test_1", "test_2", "test_3", "test_4"):
        mp = Path(P + f"{split}_contact_maps.npz")
        if not mp.exists():
            continue
        pred = np.load(mp)["pred"]
        di = pd.read_csv(P + f"{split}_contact_maps_index.csv")
        pw = None
        if Path(P + f"{split}.csv").exists():
            pr = pd.read_csv(P + f"{split}.csv",
                             usecols=["sequence_id", "start", "hand", "contact_mm", "motion_mm",
                                      "contact_map_err_mm", "touch_iou"])
            pw = pr.groupby(["sequence_id", "start", "hand"]).mean().unstack("hand")
        cache = {}
        for i, r in di.iterrows():
            f = idx_all.loc[r.sequence_id, "file"]
            if f not in cache:
                with np.load(root + "sequences/" + f) as zz:
                    cache = {f: (zz["contact_left"], zz["contact_right"], zz["obj_cano_bps_inds"])}
            cl, cr, inds = cache[f]
            s, T = int(r.start), 64
            nt = n_tool(r.tool_mesh)
            cr_w, cl_w, ind_w = cr[s:s + T], cl[s:s + T], inds[s:s + T]
            rt = np.arange(len(cr_w))[:, None]
            p = pred[i].astype(np.float32)
            rec = {"run": run, "split": split, "sequence_id": r.sequence_id, "start": s,
                   "verb": r.verb, "tool_cat": r.tool_cat, "target_cat": r.target_cat,
                   "tool_mesh": r.tool_mesh, "target_mesh": r.target_mesh,
                   "dense_R_tool": float((cr_w[:, :nt].min(1) < 0.01).mean()),
                   "gath_R_tool": float((cr_w[rt, ind_w[:, :512]].min(1) < 0.01).mean()),
                   "pred_R_tool": float((p[:, 1024:1536].min(1) < 0.01).mean()),
                   "dense_L_targ": float((cl_w[:, nt:].min(1) < 0.01).mean()),
                   "gath_L_targ": float((cl_w[rt, ind_w[:, 512:]].min(1) < 0.01).mean()),
                   "pred_L_targ": float((p[:, 512:1024].min(1) < 0.01).mean()),
                   "uniq_tool_verts": int(len(np.unique(ind_w[:, :512])))}
            if pw is not None and (r.sequence_id, s) in pw.index:
                for c in ("contact_mm", "motion_mm", "contact_map_err_mm", "touch_iou"):
                    for h in ("L", "R"):
                        rec[f"{c}_{h}"] = float(pw.loc[(r.sequence_id, s), (c, h)])
            rows.append(rec)
        print(run, split, len(di), "windows", flush=True)

D = pd.DataFrame(rows)
D["gap_R_tool"] = D.dense_R_tool - D.gath_R_tool
D["label_err"] = (D.gath_R_tool - D.dense_R_tool).abs()
D["pred_err"] = (D.pred_R_tool - D.dense_R_tool).abs()
D["pred_err_L"] = (D.pred_L_targ - D.dense_L_targ).abs()
D.to_csv(OUT / "truncation_per_window.csv", index=False)
pd.set_option("display.width", 240)

print("\n=== (a) right-hand TOOL contact fraction: truth / label / prediction, and the label error ===")
print(D.groupby(["split", "run"])[["dense_R_tool", "gath_R_tool", "pred_R_tool", "gap_R_tool",
                                   "label_err", "pred_err", "uniq_tool_verts"]].mean().round(3).to_string())
print("\n=== (a2) label error by target category (test splits) ===")
t = D[D.split != "train"]
print(t.pivot_table(index="target_cat", columns="run", values="label_err", aggfunc="mean").round(3).to_string())
print("\n=== (a3) label error by tool category (test splits) ===")
print(t.pivot_table(index="tool_cat", columns="run", values="label_err", aggfunc="mean").round(3).to_string())
print("\n=== (b) left-hand TARGET half (control: never truncated) ===")
print(D.groupby(["split", "run"])[["dense_L_targ", "gath_L_targ", "pred_L_targ", "pred_err_L"]].mean().round(3).to_string())
print("\n=== (b2) windows where the label says untouched (<0.1) but the truth says held (>0.5) ===")
for (sp, run), g in D.groupby(["split", "run"]):
    n = int(((g.dense_R_tool > 0.5) & (g.gath_R_tool < 0.1)).sum())
    print(f"  {sp:7s} {run:5s}: {n:4d} / {len(g):5d}")
print("\n=== (c) does contact error track the truncation gap?  spearman(gap, contact_mm_R) ===")
for (sp, run), g in D.dropna(subset=["contact_mm_R"]).groupby(["split", "run"]):
    rr = spearmanr(g.gap_R_tool, g.contact_mm_R)
    print(f"  {sp:7s} {run:5s}: {rr.correlation:+.3f}  (p={rr.pvalue:.1e}, n={len(g)})")
print("\n=== (c2) test_3 gap terciles ===")
for run, g in D[(D.split == "test_3")].dropna(subset=["contact_mm_R"]).groupby("run"):
    g = g.copy()
    g["ter"] = pd.qcut(g.gap_R_tool.rank(method="first"), 3, labels=["low", "mid", "high"])
    print(f"  --- {run}")
    print(g.groupby("ter", observed=True)[["gap_R_tool", "contact_mm_R", "contact_mm_L", "motion_mm_R"]].mean().round(2).to_string())
print("\n=== (d) does the prediction track the truth or the label?  within-(tool_mesh,target_cat) partial slopes ===")
for run, g in D[D.split != "train"].groupby("run"):
    g = g.copy()
    g["grp"] = g.tool_mesh.astype(str) + "|" + g.target_cat.astype(str)
    dm = g.groupby("grp")[["pred_R_tool", "dense_R_tool", "gath_R_tool"]].transform(lambda x: x - x.mean())
    X = np.c_[dm.dense_R_tool, dm.gath_R_tool, np.ones(len(dm))]
    beta, *_ = np.linalg.lstsq(X, dm.pred_R_tool, rcond=None)
    print(f"  {run}: pred ~ dense {beta[0]:+.3f} + label {beta[1]:+.3f}  (n={len(g)}, "
          f"corr(dense,label)={np.corrcoef(g.dense_R_tool, g.gath_R_tool)[0, 1]:.3f})")
print("\n-> ", OUT / "truncation_per_window.csv")
