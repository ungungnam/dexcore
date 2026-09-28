"""Does gathering contact at per-mesh farthest-point samples recover the truth the BPS gathering loses?

Three readings of the same recorded contact, per sequence and hand:
  dense    every object vertex (~8000) -- the truth
  bps      the 512+512 vertices BPS selected, scaled by the TARGET's factor -- what was trained on
  fps      512+512 farthest-point samples of each mesh's own surface -- the proposed label
"""
import sys
sys.path.insert(0, "/home/uhnam/workspace/dexcore")
import numpy as np, pandas as pd
from src.analysis.bimart import meshes as M

R = "/result/uhnam/dexcore/bimart_taco/"
md = M.load(R + "assets/taco_mesh_dict.npy")


def fps(verts, k, seed=0):
    """Farthest point sampling: k indices spread over the surface, deterministic."""
    n = len(verts)
    if n <= k:
        return np.arange(n)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = int(np.argmax(np.linalg.norm(verts - verts.mean(0), axis=1)))  # deterministic start
    d = np.linalg.norm(verts - verts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(verts - verts[sel[i]], axis=1))
    return sel


FPS = {}
def fps_for(name, k=512):
    if name not in FPS:
        FPS[name] = fps(np.asarray(md[name]["verts_original"]), k)
    return FPS[name]


idx = pd.read_csv(R + "sequence_index.csv")
rng = np.random.default_rng(0)
# a spread over target categories, weighted to the ones the defect was worst on
pick = pd.concat([g.sample(min(4, len(g)), random_state=0) for _, g in idx.groupby("target_cat")])
rows = []
for _, r in pick.iterrows():
    z = np.load(R + "sequences/" + r.file)
    cl, cr, bind = z["contact_left"], z["contact_right"], z["obj_cano_bps_inds"]
    n_tool = len(md[str(r.tool_mesh) if str(r.tool_mesh) in md else f"{int(r.tool_mesh):03d}"]["verts_original"]) \
             if not isinstance(r.tool_mesh, str) else len(md[r.tool_mesh]["verts_original"])
    tname = next(k for k in (str(r.tool_mesh), f"{int(r.tool_mesh):03d}") if k in md)
    gname = next(k for k in (str(r.target_mesh), f"{int(r.target_mesh):03d}") if k in md)
    n_tool = len(md[tname]["verts_original"])
    fi = np.concatenate([fps_for(tname), fps_for(gname) + n_tool])
    T = len(cl); rowsT = np.arange(T)[:, None]
    for hand, c in (("L", cl), ("R", cr)):
        dense_tool = (c[:, :n_tool].min(1) < 0.01).mean()
        dense_targ = (c[:, n_tool:].min(1) < 0.01).mean()
        b = c[rowsT, bind]                        # (T,1024) old label
        f = c[:, fi]                              # (T,1024) new label
        bps_tool = (b[:, :512].min(1) < 0.01).mean(); bps_targ = (b[:, 512:].min(1) < 0.01).mean()
        fps_tool = (f[:, :512].min(1) < 0.01).mean(); fps_targ = (f[:, 512:].min(1) < 0.01).mean()
        rows.append({"split": r.split, "target_cat": r.target_cat, "tool_cat": r.tool_cat,
                     "seq": r.sequence_id, "hand": hand, "n_tool": n_tool,
                     "dense_tool": dense_tool, "bps_tool": bps_tool, "fps_tool": fps_tool,
                     "dense_targ": dense_targ, "bps_targ": bps_targ, "fps_targ": fps_targ,
                     "uniq_bps_tool": len(np.unique(bind[:, :512])),
                     "target_scale": md[gname]["scale"], "tool_scale": md[tname]["scale"]})
D = pd.DataFrame(rows)
D.to_csv("/tmp/claude-1007/-home-uhnam-workspace-dexcore/329c193d-d467-4058-a77c-1c34e0c894c3/scratchpad/fps_check.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(pick)} sequences, {len(D)} sequence-hands\n")
print("=== TOOL contact fraction: truth vs old label vs new label (mean over sequence-hands) ===")
print(D.groupby("target_cat")[["dense_tool", "bps_tool", "fps_tool", "uniq_bps_tool", "target_scale"]].mean().round(3).to_string())
print("\n=== TARGET contact fraction (the half that was already fine -- check it is not broken) ===")
print(D.groupby("target_cat")[["dense_targ", "bps_targ", "fps_targ"]].mean().round(3).to_string())
print("\n=== overall absolute error against the dense truth ===")
for part in ("tool", "targ"):
    print(f"  {part}:  old |bps-dense| = {(D[f'bps_{part}']-D[f'dense_{part}']).abs().mean():.3f}"
          f"   new |fps-dense| = {(D[f'fps_{part}']-D[f'dense_{part}']).abs().mean():.3f}")
print("\n=== right hand, worst old cases (dense>0.5 but bps<0.1) ===")
bad = D[(D.hand == "R") & (D.dense_tool > 0.5) & (D.bps_tool < 0.1)]
print(f"  n={len(bad)} of {len(D[D.hand=='R'])} right-hand rows; on those: dense {bad.dense_tool.mean():.2f}"
      f" old {bad.bps_tool.mean():.2f} new {bad.fps_tool.mean():.2f}")
