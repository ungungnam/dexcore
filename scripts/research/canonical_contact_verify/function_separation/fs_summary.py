import pandas as pd, numpy as np
from pathlib import Path
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
OUT = Path("/result/uhnam/dexcore/canonical_contact/verify/function_separation")
g = pd.read_csv(OUT / "group_metrics.csv")
p = pd.read_csv(OUT / "pooled_metrics.csv")
vp = pd.read_csv(OUT / "verb_pair_metrics.csv")

print("=== (a) POOLED per (backend, variant, role, hand): CII=B/C, CII_dm=B/C_dm, Csm/A, macro AUC(C>B), macro frac C>medB ===")
cols = ["backend", "level", "variant", "role", "hand", "n_groups", "n_B", "n_C", "mean_A", "mean_B", "mean_C", "mean_C_dm", "mean_C_sm",
        "cii", "cii_dm", "csm_over_a", "cii_macro", "cii_dm_macro", "auc_C_gt_B_macro", "frac_C_gt_medB_macro", "csm_over_a_macro"]
for (role, hand) in [("tool", "R"), ("target", "L"), ("tool", "L"), ("target", "R")]:
    s = p[(p.role == role) & (p.hand == hand)].sort_values(["level", "variant", "backend"])
    print(f"\n--- ({hand}, {role}) ---")
    print(s[cols].round(3).to_string(index=False))

print("\n=== (a) per-group soft_l2 window: CII_dm with CI, AUC, frac, C_sm/A with CI (groups with n_C_dm>=100) ===")
s = g[(g.variant == "soft_l2") & (g.level == "win") & (g.n_C_dm >= 100)]
piv = s.pivot_table(index=["category", "role", "hand", "n_samples", "n_verbs"], columns="backend",
                    values=["cii_dm", "auc_Cdm_gt_B", "csm_over_a"])
print(piv.round(3).to_string())
print("\n per-group CI details (aligned, soft_l2):")
s2 = s[s.backend == "aligned"][["category", "role", "hand", "n_samples", "n_sequences", "n_verbs", "n_A", "n_B", "n_C_dm", "n_C_sm",
                                "cii", "cii_lo", "cii_hi", "cii_dm", "cii_dm_lo", "cii_dm_hi", "auc_Cdm_gt_B", "frac_Cdm_gt_medB",
                                "csm_over_a", "csm_over_a_lo", "csm_over_a_hi", "auc_Csm_gt_A"]]
print(s2.round(3).to_string(index=False))
print("\n groups where CII_dm CI excludes 1 (aligned/normalized/dino, soft_l2, win):")
s3 = g[(g.variant == "soft_l2") & (g.level == "win") & (g.backend.isin(["aligned", "normalized", "dino"]))]
sig = s3[(s3.cii_dm_hi < 1) | (s3.cii_dm_lo > 1)]
print(sig[["backend", "category", "role", "hand", "n_samples", "n_C_dm", "cii_dm", "cii_dm_lo", "cii_dm_hi", "auc_Cdm_gt_B"]].round(3).to_string(index=False))
print(f"  n significant below 1 (C_dm > B): {(sig.cii_dm_hi < 1).sum()} / {len(s3)};  above 1: {(sig.cii_dm_lo > 1).sum()}")

print("\n=== (c) controls, per-group aligned, soft vs hard vs nn vs pc1/pc3/zscore/lowmean/mid vs seq: cii_dm and csm_over_a (large groups) ===")
big = g[(g.backend == "aligned") & (g.n_samples >= 100)]
print(big.pivot_table(index=["category", "role", "hand"], columns=["level", "variant"], values="cii_dm").round(3).to_string())
print(big.pivot_table(index=["category", "role", "hand"], columns=["level", "variant"], values="csm_over_a").round(3).to_string())
print(big.pivot_table(index=["category", "role", "hand"], columns=["level", "variant"], values="auc_Cdm_gt_B").round(3).to_string())

print("\n=== (b) verb pairs: soft_l2 window, top/bottom by ratio C_dm/B per backend (n_seq>=5 both, n_C_dm>=50) ===")
v = vp[(vp.variant == "soft_l2") & (vp.level == "win") & (vp.n_seq_i >= 5) & (vp.n_seq_j >= 5) & (vp.n_C_dm >= 50)].copy()
v["pair"] = v.category + "/" + v.role + "/" + v.hand + ": " + v.verb_i + " vs " + v.verb_j
cols = ["pair", "n_seq_i", "n_seq_j", "n_B", "n_C_dm", "mean_B", "mean_C_dm", "ratio_Cdm_over_B", "ratio_lo", "ratio_hi", "auc_Cdm_gt_B", "frac_Cdm_gt_medB", "ratio_Csm_over_A", "ratio_Csm_lo", "ratio_Csm_hi"]
for be in ["aligned", "normalized", "dino", "random_perm"]:
    s = v[v.backend == be].sort_values("ratio_Cdm_over_B", ascending=False)
    print(f"\n--- {be}: n verb pairs = {len(s)}; ratio>1 with CI excluding 1: {(s.ratio_lo > 1).sum()}, ratio<1 CI excl 1: {(s.ratio_hi < 1).sum()}; median ratio {s.ratio_Cdm_over_B.median():.3f}")
    print(" TOP 10 most separated"); print(s.head(10)[cols].round(3).to_string(index=False))
    print(" BOTTOM 10 least separated"); print(s.tail(10)[cols].round(3).to_string(index=False))
print("\n--- named pairs of interest (aligned/normalized/dino) ---")
want = [("knife", "cut", "scrape off"), ("spatula", "put in", "put out"), ("spatula", "skim off", "stir"), ("spatula", "put in", "stir"),
        ("spoon", "put in", "put out"), ("spoon", "cut", "scrape off"), ("brush", "brush", "dust"), ("roller", "brush", "dust"),
        ("kettle", "empty", "pour in some"), ("bowl", "empty", "pour in some"), ("eraser", "brush", "smear"), ("plate", "cut", "scrape off"),
        ("pan", "brush", "dust"), ("pan", "cut", "scrape off"), ("bowl", "cut", "scrape off"), ("bowl", "put in", "put out"), ("teapot", "empty", "pour in some")]
for cat, a, b in want:
    s = vp[(vp.variant == "soft_l2") & (vp.level == "win") & (vp.category == cat) & (vp.verb_i == a) & (vp.verb_j == b)]
    if len(s):
        print(s[["backend", "category", "role", "hand", "n_seq_i", "n_seq_j", "n_C_dm", "ratio_Cdm_over_B", "ratio_lo", "ratio_hi", "auc_Cdm_gt_B", "ratio_Csm_over_A", "ratio_Csm_lo", "ratio_Csm_hi", "auc_Csm_gt_A"]].round(3).to_string(index=False))
print("\n--- verb pairs with other variants (hard/nn/pc1/zscore): count of separated pairs ---")
for va in ["soft_l2", "hard_l2", "nn_l2", "pc1_removed_l2", "zscore_l2"]:
    for lv in ["win", "seq"]:
        s = vp[(vp.variant == va) & (vp.level == lv) & (vp.n_seq_i >= 5) & (vp.n_seq_j >= 5) & (vp.n_C_dm >= 50) & (vp.backend == "aligned")]
        if len(s):
            print(f"{va:16s} {lv}: n={len(s)} median ratio={s.ratio_Cdm_over_B.median():.3f} sep(lo>1)={(s.ratio_lo>1).sum()} sep(lo>1.1)={(s.ratio_lo>1.1).sum()} max={s.ratio_Cdm_over_B.max():.2f} median AUC={s.auc_Cdm_gt_B.median():.3f}")

print("\n=== verb decoding (balanced accuracy - chance), logistic regression ===")
d = pd.read_csv(OUT / "verb_decoding.csv")
print(d.pivot_table(index=["category", "role", "hand", "n", "n_verbs", "n_meshes"], columns=["features", "split", "backend"], values="bacc_minus_chance").round(2).to_string())
print(d.groupby(["backend", "features", "split"]).bacc_minus_chance.agg(["mean", "median", "count", lambda x: (x > 0.05).sum()]).round(3))

print("\n=== (e) instance-space same-mesh test: C_sm vs A with NO canonicalisation ===")
i1 = pd.read_csv(OUT / "instance_same_mesh.csv")
s = i1[(i1.feature == "inst_soft_l2")]
print(s.sort_values("n_A", ascending=False).round(3).to_string(index=False))
for f in ["inst_soft_l2", "inst_hard_l2"]:
    s = i1[i1.feature == f]
    w = s.n_A + s.n_Csm
    print(f"{f}: meshes={len(s)}  pair-weighted ratio C_sm/A = {np.average(s.ratio_Csm_over_A, weights=w):.3f}; median ratio {s.ratio_Csm_over_A.median():.3f}; pair-weighted AUC {np.average(s.auc_Csm_gt_A, weights=w):.3f}; meshes with ratio>1.2: {(s.ratio_Csm_over_A>1.2).sum()}")
i2 = pd.read_csv(OUT / "instance_same_mesh_verbpairs.csv")
i2 = i2[(i2.n_A >= 30) & (i2.n_Csm >= 30)]
print("\n instance-space verb pairs (same mesh), top 15 by ratio and bottom 10:")
i2s = i2.sort_values("ratio", ascending=False)
print(i2s.head(15).round(3).to_string(index=False)); print(i2s.tail(10).round(3).to_string(index=False))
print(" knife cut vs scrape off per mesh:"); print(i2[(i2.category == "knife")].round(3).to_string(index=False))
