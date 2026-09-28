"""Does the contact model still under-predict tool contact, now that the label is correct?

The defect made the label say "tool untouched" while the hand held it, and the model reproduced the
label rather than the truth (partial slope 0.83 on the label, -0.02 on the dense truth). With the
label corrected, three things must be checked:
  1. the label now agrees with the dense truth (it should, by construction)
  2. the model's prediction now agrees with the truth
  3. the old regression no longer finds a label-over-truth preference -- which it cannot, because
     label and truth now coincide, so instead we ask whether prediction tracks truth at all.
"""
import sys
sys.path.insert(0, "/home/uhnam/workspace/dexcore")
import numpy as np, pandas as pd
from src.analysis.bimart import fps_contact as F, meshes as M

R = "/result/uhnam/dexcore/bimart_taco/"
md = M.load(R + "assets/taco_mesh_dict.npy")
table = F.load_table(R + "assets/taco_fps_index_fps.npy")
idx_all = pd.read_csv(R + "sequence_index.csv").set_index("sequence_id")
rows = []
for split in ("train", "test_1", "test_2", "test_3"):
    for variant, P in (("old", R + "contact_probe/"), ("new", R + "contact_probe_fps/")):
        z = np.load(P + f"{split}_contact_maps.npz"); pred = z["pred"].astype(np.float32)
        gt = z["gt"].astype(np.float32)
        di = pd.read_csv(P + f"{split}_contact_maps_index.csv")
        for i, r in di.iterrows():
            f = idx_all.loc[r.sequence_id, "file"]
            with np.load(R + "sequences/" + f) as zz:
                cr = zz["contact_right"]; cl = zz["contact_left"]
                bind = zz["obj_cano_bps_inds"]
            s, T = int(r.start), 64
            tk = F.mesh_key(md, r.tool_mesh); n_tool = len(md[tk]["verts_original"])
            w = slice(s, s + T)
            dense_R = float((cr[w][:, :n_tool].min(1) < 0.01).mean())
            dense_L = float((cl[w][:, n_tool:].min(1) < 0.01).mean())
            # the label this variant trained on
            if variant == "old":
                rr = np.arange(T)[:, None]
                lab_R = float((cr[w][rr, bind[w][:, :512]].min(1) < 0.01).mean())
            else:
                inds = F.gather_indices(table, md, r.tool_mesh, r.target_mesh)
                lab_R = float((cr[w][:, inds[:512]].min(1) < 0.01).mean())
            rows.append({"split": split, "variant": variant, "sequence_id": r.sequence_id,
                         "target_cat": r.target_cat, "tool_mesh": r.tool_mesh,
                         "dense_R_tool": dense_R, "label_R_tool": lab_R,
                         "pred_R_tool": float((pred[i][:, 1024:1536].min(1) < 0.01).mean()),
                         "dense_L_targ": dense_L,
                         "pred_L_targ": float((pred[i][:, 512:1024].min(1) < 0.01).mean())})
    print(split, "done", flush=True)
D = pd.DataFrame(rows)
D.to_csv("/result/uhnam/dexcore/bimart_taco/contact_probe_fps/g7_recheck.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== right-hand TOOL contact: dense truth / the label trained on / the model's prediction ===")
print(D.groupby(["split", "variant"])[["dense_R_tool", "label_R_tool", "pred_R_tool"]].mean().round(3).to_string())
print("\n=== absolute error of the prediction against the DENSE TRUTH (what we actually want) ===")
D["pred_err"] = (D.pred_R_tool - D.dense_R_tool).abs()
D["label_err"] = (D.label_R_tool - D.dense_R_tool).abs()
print(D.groupby(["split", "variant"])[["label_err", "pred_err"]].mean().round(3).unstack().to_string())
print("\n=== left-hand TARGET contact (control: the half the defect never touched) ===")
print(D.groupby(["split", "variant"])[["dense_L_targ", "pred_L_targ"]].mean().round(3).to_string())
print("\n=== does the prediction track the truth? within-(tool_mesh,target_cat) partial slopes ===")
for variant, g in D.groupby("variant"):
    gg = g[g.split != "train"].copy()
    gg["grp"] = gg.tool_mesh.astype(str) + "|" + gg.target_cat.astype(str)
    dm = gg.groupby("grp")[["pred_R_tool", "dense_R_tool", "label_R_tool"]].transform(lambda x: x - x.mean())
    X = np.c_[dm.dense_R_tool, dm.label_R_tool, np.ones(len(dm))]
    beta = np.linalg.lstsq(X, dm.pred_R_tool, rcond=None)[0]
    print(f"  {variant}: pred ~ dense {beta[0]:+.3f}  + label {beta[1]:+.3f}   (n={len(gg)} sequences, "
          f"corr(dense,label)={np.corrcoef(gg.dense_R_tool, gg.label_R_tool)[0,1]:.3f})")
