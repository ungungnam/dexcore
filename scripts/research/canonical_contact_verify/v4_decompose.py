"""(ii)+(iii)+(c): B^2 decomposition into norms vs cross-mesh inner product; relative-rotation / ICP-scale
regression of (aligned - normalized) B distance; category-group pooling; leave-one-category-out; flip cases."""
import numpy as np, pandas as pd, glob, sys
R = "/result/uhnam/dexcore/canonical_contact/"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
t = pd.read_csv(R + "contact_pair_distances.csv")
# ---- paired bootstrap differences from v1
pb = pd.read_csv(R + "verify/gp_bootstrap_pooled.csv")
print("=== paired bootstrap GP differences (same sequence draws), pooled")
print(pb[pb.scope == "pooled_diff"][["role", "hand", "representation", "GP", "GP_lo", "GP_hi", "frac_pos"]].round(3).to_string(index=False))

# ---- load X vectors for the samples used in pairs
def load_X(be):
    out = {}
    for f in sorted(glob.glob(R + f"canonical_contact_cache/{be}/*__*.npz")):
        if f.endswith("__seq.npz"): continue
        z = np.load(f, allow_pickle=False)
        cat, role = str(z["category"]), str(z["role"])
        key = list(zip(z["sequence_id"], z["window"].astype(int), z["hand"]))
        out[(cat, role)] = ({k: n for n, k in enumerate(key)}, z["X_soft"].astype(np.float64), z["coverage"])
    return out
X = {be: load_X(be) for be in ["normalized", "aligned", "dino", "random_perm"]}
def vecs(be, g):
    cat, role, hand = g.name if isinstance(g.name, tuple) else (None, None, None)
    return None
rows = []
for be in X:
    for (cat, role, hand, pt), g in t.groupby(["category", "role", "hand", "pair_type"]):
        if (cat, role) not in X[be]: continue
        idx, Xs, cov = X[be][(cat, role)]
        ii = np.array([idx[(s, w, hand)] for s, w in zip(g.seq_i, g.window_i)])
        jj = np.array([idx[(s, w, hand)] for s, w in zip(g.seq_j, g.window_j)])
        xi, xj = Xs[ii], Xs[jj]
        d2 = ((xi - xj) ** 2).sum(1)
        # check against csv
        col = f"{be}_l2"
        err = np.abs(np.sqrt(d2) - g[col].values).max()
        rows.append(dict(backend=be, category=cat, role=role, hand=hand, pair_type=pt, n=len(g),
                         mean_d2=d2.mean(), mean_sqnorm=0.5 * ((xi ** 2).sum(1) + (xj ** 2).sum(1)).mean(),
                         mean_dot=(xi * xj).sum(1).mean(),
                         mean_cos_sim=((xi * xj).sum(1) / np.maximum(np.linalg.norm(xi, axis=1) * np.linalg.norm(xj, axis=1), 1e-9)).mean(),
                         both_zero_frac=((xi < 1e-6) & (xj < 1e-6)).mean(), max_err_vs_csv=err))
dec = pd.DataFrame(rows); dec.to_csv(R + "verify/b2_decomposition.csv", index=False)
print("\nmax |recomputed L2 - csv| :", dec.max_err_vs_csv.max())
print("\n=== pooled decomposition  mean d^2 = 2*mean_sqnorm - 2*mean_dot   (weighted by pairs)")
for (role, hand) in [("tool", "R"), ("target", "L")]:
    for pt in "ABC":
        sub = dec[(dec.role == role) & (dec.hand == hand) & (dec.pair_type == pt)]
        for be, g in sub.groupby("backend"):
            w = g.n.values
            f = lambda c: (g[c].values * w).sum() / w.sum()
            print(f"  {role:6s}{hand} {pt} {be:12s} n={w.sum():6d} d2={f('mean_d2'):7.3f} sqnorm={f('mean_sqnorm'):7.3f} dot={f('mean_dot'):7.3f} "
                  f"cos_sim={f('mean_cos_sim'):.3f} d={np.sqrt(f('mean_d2')):.3f}")

# ---- relative rotation / scale that `aligned` introduces between the two meshes of a B pair
al = {}
for f in glob.glob(R + "canonical_backend/aligned/*.npz"):
    z = np.load(f, allow_pickle=False)
    ids = [str(i) for i in z["mesh_ids"]]; Rm = z["R"]; s = z["s"]; verts = z["verts_all"]; off = z["verts_offsets"]; c = z["c"]
    rad = [np.linalg.norm(verts[off[n]:off[n + 1]] - c[n], axis=1).max() for n in range(len(ids))]
    al[str(z["category"])] = {i: (Rm[n], s[n] * rad[n]) for n, i in enumerate(ids)}
def ang(Rrel): return np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
B = t[t.pair_type == "B"].copy()
B["rel_rot_deg"] = [ang(al[c][f"{i:03d}"][0].T @ al[c][f"{j:03d}"][0]) for c, i, j in zip(B.category, B.mesh_i, B.mesh_j)]
B["log_scale_ratio"] = [abs(np.log(al[c][f"{i:03d}"][1] / al[c][f"{j:03d}"][1])) for c, i, j in zip(B.category, B.mesh_i, B.mesh_j)]
B["diff_al_no"] = B.aligned_l2 - B.normalized_l2
B["diff_dn_no"] = B.dino_l2 - B.normalized_l2
B["rot_bin"] = pd.cut(B.rel_rot_deg, [-1, 30, 90, 150, 181], labels=["<30", "30-90", "90-150", ">150"])
B["scale_bin"] = pd.cut(B.log_scale_ratio, [-0.01, 0.1, 0.25, 10], labels=["|log s|<0.1", "0.1-0.25", ">0.25"])
print("\n=== B pairs: (aligned_l2 - normalized_l2) by the relative rotation `aligned` introduces between the two meshes")
print(B.groupby("rot_bin", observed=True).agg(n=("diff_al_no", "size"), mean_diff_al_no=("diff_al_no", "mean"),
      mean_diff_dn_no=("diff_dn_no", "mean"), aligned_B=("aligned_l2", "mean"), normalized_B=("normalized_l2", "mean")).round(3).to_string())
print("\n=== ... by |log ICP-scale ratio| between the two meshes, restricted to rel rotation < 30 deg")
print(B[B.rel_rot_deg < 30].groupby("scale_bin", observed=True).agg(n=("diff_al_no", "size"), mean_diff_al_no=("diff_al_no", "mean"),
      aligned_B=("aligned_l2", "mean"), normalized_B=("normalized_l2", "mean")).round(3).to_string())
print("\n=== per category (tool R / target L): share of B pairs with rel rot > 150, and diff by bin")
for (cat, role, hand), g in B.groupby(["category", "role", "hand"]):
    if (role, hand) not in [("tool", "R"), ("target", "L")] or len(g) < 100: continue
    lo = g[g.rel_rot_deg < 30]; hi = g[g.rel_rot_deg > 150]
    print(f"  {cat:12s} {role:6s}{hand} n={len(g):5d}  frac>150={100*len(hi)/len(g):4.0f}%  frac<30={100*len(lo)/len(g):4.0f}%  "
          f"diff(al-no) rot<30: {lo.diff_al_no.mean() if len(lo) else np.nan:6.3f}  rot>150: {hi.diff_al_no.mean() if len(hi) else np.nan:6.3f}   "
          f"diff(dino-no) rot<30: {lo.diff_dn_no.mean() if len(lo) else np.nan:6.3f}  rot>150: {hi.diff_dn_no.mean() if len(hi) else np.nan:6.3f}"
          f"   mean|log s ratio|={g.log_scale_ratio.mean():.2f}")
B[["category", "role", "hand", "mesh_i", "mesh_j", "rel_rot_deg", "log_scale_ratio", "normalized_l2", "aligned_l2", "dino_l2"]].to_csv(R + "verify/B_pairs_relrot.csv", index=False)

# ---- category groups & leave-one-category-out pooled GP (point estimates)
SYM = {"bowl", "plate", "pan", "cup", "helmet", "box", "eraser", "toy", "soap"}
HANDLED = {"spatula", "spoon", "knife", "hammer", "screwdriver", "brush", "ruler", "roller", "kettle", "teapot", "glue gun"}
def gp(g, col):
    A = g[g.pair_type == "A"][col].mean(); Bm = g[g.pair_type == "B"][col].mean(); C = g[g.pair_type == "C"][col].mean()
    return Bm / A, A, Bm, C
print("\n=== pooled GP by category group (point estimates)")
for (role, hand) in [("tool", "R"), ("target", "L")]:
    g0 = t[(t.role == role) & (t.hand == hand)]
    for name, grp in [("ALL", None), ("symmetric-body", SYM), ("handled/elongated", HANDLED)]:
        g = g0 if grp is None else g0[g0.category.isin(grp)]
        s = f"  {role:6s}{hand} {name:18s} nA={int((g.pair_type=='A').sum()):5d} nB={int((g.pair_type=='B').sum()):5d} "
        for be in ["normalized", "aligned", "dino", "random_perm"]:
            v, A, Bm, C = gp(g, f"{be}_l2"); s += f" {be}={v:.3f}(B={Bm:.2f})"
        print(s)
    print("  leave-one-category-out GP (normalized / aligned / dino):")
    for cat in sorted(g0.category.unique()):
        g = g0[g0.category != cat]
        print(f"    drop {cat:12s}: " + "  ".join(f"{be}={gp(g, f'{be}_l2')[0]:.3f}" for be in ["normalized", "aligned", "dino"]))
    print("  macro (unweighted mean of per-category GP over categories with >=30 A and B pairs):")
    vals = {be: [] for be in ["normalized", "aligned", "dino", "random_perm"]}
    for cat, g in g0.groupby("category"):
        if (g.pair_type == "A").sum() < 30 or (g.pair_type == "B").sum() < 30: continue
        for be in vals: vals[be].append(gp(g, f"{be}_l2")[0])
    print("    " + "  ".join(f"{be}={np.mean(v):.3f} (n_cat={len(v)})" for be, v in vals.items()))
print("done")
