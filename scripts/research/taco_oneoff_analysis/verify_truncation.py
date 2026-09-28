"""Is the 'recorded' tool-contact map truncated when the target is small, and does that drive contact_mm?

Per test window: right-hand tool contact three ways -- DENSE (all tool vertices), GATHERED (the 512
BPS-sampled tool vertices; what the model was trained on and scored against), PREDICTED (the contact
model's output). If predicted ~ dense >> gathered, the 'contact error' is partly a label artefact.
"""
import sys, json, numpy as np, pandas as pd
sys.path.insert(0, "/home/uhnam/workspace/dexcore")
from src.analysis.bimart import meshes as M
R = "/result/uhnam/dexcore/bimart_taco/"; P = R + "contact_probe/"
md = M.load(R + "assets/taco_mesh_dict.npy")
def n_tool(mesh_id):
    for k in (mesh_id, str(mesh_id), f"{int(mesh_id):03d}"):
        if k in md: return len(md[k]["verts_original"])
    raise KeyError(mesh_id)
idx_all = pd.read_csv(R + "sequence_index.csv").set_index("sequence_id")
rows = []
for split in ("test_1", "test_2", "test_3", "test_4", "train"):
    z = np.load(P + f"{split}_contact_maps.npz"); pred = z["pred"]      # [N,64,2048] f16 metres
    di = pd.read_csv(P + f"{split}_contact_maps_index.csv")
    probe = pd.read_csv(P + f"{split}.csv", usecols=["window","hand","contact_mm","motion_mm","contact_trans_mm","gt_touch_frac"])
    pw = probe.groupby(["window","hand"]).mean().unstack("hand")
    cache = {}
    for i, r in di.iterrows():
        f = idx_all.loc[r.sequence_id, "file"]
        if f not in cache:
            zz = np.load(R + "sequences/" + f)
            cache = {f: (zz["contact_left"], zz["contact_right"], zz["obj_cano_bps_inds"])}
        cl, cr, inds = cache[f]
        s, T = int(r.start), 64
        nt = n_tool(r.tool_mesh)
        cr_w, cl_w, ind_w = cr[s:s+T], cl[s:s+T], inds[s:s+T]
        dense_R_tool = (cr_w[:, :nt].min(1) < 0.01).mean()
        dense_L_targ = (cl_w[:, nt:].min(1) < 0.01).mean()
        dense_R_targ = (cr_w[:, nt:].min(1) < 0.01).mean()
        dense_L_tool = (cl_w[:, :nt].min(1) < 0.01).mean()
        rows_t = np.arange(T)[:, None]
        g_R_tool = (cr_w[rows_t, ind_w[:, :512]].min(1) < 0.01).mean()
        g_L_targ = (cl_w[rows_t, ind_w[:, 512:]].min(1) < 0.01).mean()
        uniq_tool = len(np.unique(ind_w[:, :512]))
        p = pred[i].astype(np.float32)
        p_R_tool = (p[:, 1024:1536].min(1) < 0.01).mean(); p_L_targ = (p[:, 512:1024].min(1) < 0.01).mean()
        w = int(r.window)
        rows.append({"split": split, "window": w, "sequence_id": r.sequence_id, "verb": r.verb,
                     "tool_cat": r.tool_cat, "target_cat": r.target_cat, "tool_mesh": r.tool_mesh, "target_mesh": r.target_mesh,
                     "dense_R_tool": dense_R_tool, "gath_R_tool": g_R_tool, "pred_R_tool": p_R_tool,
                     "dense_L_targ": dense_L_targ, "gath_L_targ": g_L_targ, "pred_L_targ": p_L_targ,
                     "dense_R_targ": dense_R_targ, "dense_L_tool": dense_L_tool, "uniq_gathered_tool_verts": uniq_tool,
                     "contact_mm_R": pw.loc[w, ("contact_mm","R")] if w in pw.index else np.nan,
                     "contact_mm_L": pw.loc[w, ("contact_mm","L")] if w in pw.index else np.nan,
                     "motion_mm_R": pw.loc[w, ("motion_mm","R")] if w in pw.index else np.nan})
    print(split, len(di), "windows done", flush=True)
D = pd.DataFrame(rows); D["gap_R_tool"] = D.dense_R_tool - D.gath_R_tool
D.to_csv(P + "truncation_per_window.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== right-hand TOOL contact fraction: dense truth vs gathered 'GT' vs model prediction ===")
print(D.groupby("split")[["dense_R_tool","gath_R_tool","pred_R_tool","uniq_gathered_tool_verts","gap_R_tool"]].mean().round(3).to_string())
print("\n=== same, test_3 and test_4 by target category ===")
print(D[D.split.isin(["test_3","test_4"])].groupby(["split","target_cat"])[["dense_R_tool","gath_R_tool","pred_R_tool","gap_R_tool","uniq_gathered_tool_verts"]].mean().round(3).to_string())
print("\n=== left-hand TARGET contact (should NOT be truncated: target is scaled to fit) ===")
print(D.groupby("split")[["dense_L_targ","gath_L_targ","pred_L_targ"]].mean().round(3).to_string())
print("\n=== is the model's prediction closer to the dense truth than the gathered label is? |pred-dense| vs |gath-dense| ===")
D["pred_vs_dense"] = (D.pred_R_tool - D.dense_R_tool).abs(); D["gath_vs_dense"] = (D.gath_R_tool - D.dense_R_tool).abs()
print(D.groupby("split")[["pred_vs_dense","gath_vs_dense"]].mean().round(3).to_string())
print("\n=== does contact_mm (right hand) track the truncation gap?  spearman per split, and gap terciles ===")
from scipy.stats import spearmanr
for s, g in D.dropna(subset=["contact_mm_R"]).groupby("split"):
    rr = spearmanr(g.gap_R_tool, g.contact_mm_R); print(f"  {s}: spearman(gap, contact_mm_R) = {rr.correlation:+.3f}  n={len(g)}")
T3 = D[(D.split=="test_3")].dropna(subset=["contact_mm_R"]).copy()
T3["gap_ter"] = pd.qcut(T3.gap_R_tool.rank(method="first"), 3, labels=["low","mid","high"])
print(T3.groupby("gap_ter", observed=True)[["gap_R_tool","contact_mm_R","contact_mm_L","motion_mm_R"]].mean().round(1).to_string())
print("\n  test_3 windows with dense_R_tool>0.5 but gath_R_tool<0.1 (label says untouched, truth says held):", int(((T3.dense_R_tool>0.5)&(T3.gath_R_tool<0.1)).sum()), "of", len(T3))
