"""Does one scene scale restore the tool's coverage, with upstream's BPS-indexed gathering kept?

Compares, per sequence-hand, the recorded tool-contact fraction read three ways:
  dense       every tool vertex -- the truth
  bps@target  BPS indices with both objects scaled by the TARGET's factor (the defect)
  bps@scene   BPS indices with one scale fitted to BOTH objects (the fix, upstream's structure)
The third keeps the label co-indexed with the BPS input, which the farthest-point label did not.
"""
import sys
sys.path.insert(0, "/home/uhnam/workspace/dexcore")
import numpy as np, pandas as pd
from src.analysis.bimart import meshes as M, features as Fe
from src.analysis.loaders import taco
from src import geometry as G

R = "/result/uhnam/dexcore/bimart_taco/"
md = M.load(R + "assets/taco_mesh_dict.npy")
basis = Fe.load_basis()
idx = pd.read_csv(R + "sequence_index.csv")
refs = {r.sequence_id: r for r in taco.index()}
pick = pd.concat([g.sample(min(3, len(g)), random_state=0) for _, g in idx.groupby("target_cat")])
rows = []
for _, r in pick.iterrows():
    traj = taco.load(refs[r.sequence_id], with_hands=False)
    tool, targ = traj.tool, traj.target
    mt, mo = md[tool.name], md[targ.name]
    T = traj.num_frames
    R_t, p_t = G.wxyz_to_R(tool.quat), tool.pos
    R_g, p_g = G.wxyz_to_R(targ.quat), targ.pos
    tool_w = np.einsum("tij,nj->tni", R_t, mt["verts_original"]) + p_t[:, None, :]
    tool_c = Fe.to_canonical(tool_w, R_g, p_g)
    n_tool = len(mt["verts_original"])
    with np.load(R + "sequences/" + r.file) as z:
        cl, cr = z["contact_left"], z["contact_right"]
    T = min(T, len(cl)); tool_c = tool_c[:T]
    s_targ = mo["scale"]
    r_scene = max(float(np.linalg.norm(tool_c.reshape(-1, 3), axis=1).max()),
                  float(np.linalg.norm(mo["verts_original"], axis=1).max()))
    s_scene = Fe.BASIS_RADIUS / r_scene
    out = {}
    for tag, s in (("target", s_targ), ("scene", s_scene)):
        _, it = Fe.bps_from_points(tool_c * s, basis)
        _, ig = Fe.bps_static(mo["verts_original"] * s, basis, T)
        inds = np.concatenate([it, ig + n_tool], axis=1)
        rr = np.arange(T)[:, None]
        out[tag] = {"L": cl[:T][rr, inds], "R": cr[:T][rr, inds], "uniq": len(np.unique(it))}
    for hand, c in (("L", cl[:T]), ("R", cr[:T])):
        rows.append({"target_cat": r.target_cat, "tool_cat": r.tool_cat, "hand": hand,
                     "seq": r.sequence_id,
                     "dense_tool": float((c[:, :n_tool].min(1) < .01).mean()),
                     "bps_target_tool": float((out["target"][hand][:, :512].min(1) < .01).mean()),
                     "bps_scene_tool": float((out["scene"][hand][:, :512].min(1) < .01).mean()),
                     "dense_targ": float((c[:, n_tool:].min(1) < .01).mean()),
                     "bps_target_targ": float((out["target"][hand][:, 512:].min(1) < .01).mean()),
                     "bps_scene_targ": float((out["scene"][hand][:, 512:].min(1) < .01).mean()),
                     "uniq_target": out["target"]["uniq"], "uniq_scene": out["scene"]["uniq"],
                     "s_targ": float(s_targ), "s_scene": float(s_scene)})
D = pd.DataFrame(rows)
D.to_csv("/tmp/claude-1007/-home-uhnam-workspace-dexcore/329c193d-d467-4058-a77c-1c34e0c894c3/scratchpad/scene_scale_check.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(pick)} sequences, {len(D)} sequence-hands\n")
print("=== TOOL contact fraction: truth / old scaling / scene scaling ===")
print(D.groupby("target_cat")[["dense_tool","bps_target_tool","bps_scene_tool","s_targ","s_scene","uniq_target","uniq_scene"]].mean().round(3).to_string())
print("\n=== TARGET contact fraction (must not regress) ===")
print(D.groupby("target_cat")[["dense_targ","bps_target_targ","bps_scene_targ"]].mean().round(3).to_string())
print("\n=== absolute error against the dense truth ===")
for part, a, b in (("tool","bps_target_tool","bps_scene_tool"), ("targ","bps_target_targ","bps_scene_targ")):
    print(f"  {part}: old scaling {(D[a]-D['dense_'+part]).abs().mean():.3f}   scene scaling {(D[b]-D['dense_'+part]).abs().mean():.3f}")
bad = D[(D.hand=="R") & (D.dense_tool>0.5) & (D.bps_target_tool<0.1)]
print(f"\n=== worst old cases (n={len(bad)}): dense {bad.dense_tool.mean():.2f} | old {bad.bps_target_tool.mean():.2f} | scene {bad.bps_scene_tool.mean():.2f}")
