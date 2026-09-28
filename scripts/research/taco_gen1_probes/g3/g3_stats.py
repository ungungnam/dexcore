#!/usr/bin/env python
"""G3 verification, step 2: statistics on g3_features.csv joined with the model's per-window errors."""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
R = Path("/result/uhnam/dexcore/bimart_taco/contact_probe"); OUT = R / "gen" / "verify" / "g3"
F = pd.read_csv(OUT / "g3_features.csv"); J = pd.read_csv(R / "knn" / "knn_joined.csv")
keep = ["split", "sequence_id", "start", "motion", "contact", "touch", "motion_L", "motion_R", "contact_L", "contact_R", "touch_L", "touch_R", "retr_hand_mm", "retr_cmap_mm"]
D = F.merge(J[keep], on=["split", "sequence_id", "start"], how="inner")
T = D[D.split == "test_1"].reset_index(drop=True)
print(f"test_1 joined windows {len(T)} (features {int((F.split=='test_1').sum())}, model {int((J.split=='test_1').sum())}); sequences {T.sequence_id.nunique()}")
print("nn_dist reproduction: max |new-old| among test_1 =", float(np.abs(T.nn_dist - J[J.split=='test_1'].set_index(['sequence_id','start']).loc[list(zip(T.sequence_id,T.start)),'nn_dist'].values).max()))
res = {}
def sp(x, y): r, p = stats.spearmanr(x, y); return float(r), float(p)
def pe(x, y): r, p = stats.pearsonr(x, y); return float(r), float(p)

# ---------------- (a) rho, cluster bootstrap by sequence, sequence level
rng = np.random.default_rng(0)
seqs = T.sequence_id.unique(); groups = {s: T.index[T.sequence_id == s].values for s in seqs}
def cluster_boot(x, y, fn=sp, B=2000):
    out = []
    for _ in range(B):
        pick = rng.choice(seqs, len(seqs), replace=True); ii = np.concatenate([groups[s] for s in pick])
        out.append(fn(x.values[ii], y.values[ii])[0])
    return np.percentile(out, [2.5, 50, 97.5])
def jackknife(x, y, fn=sp):
    vals = []
    for s in seqs:
        m = T.sequence_id != s; vals.append(fn(x[m], y[m])[0])
    return float(min(vals)), float(max(vals))
a = {}
for err in ("motion", "contact"):
    r, p = sp(T.nn_dist, T[err]); lo, med, hi = cluster_boot(T.nn_dist, T[err]); jl, jh = jackknife(T.nn_dist, T[err])
    Sq = T.groupby("sequence_id").agg(nn=("nn_dist", "mean"), e=(err, "mean"), n=("start", "size"))
    rs, ps = sp(Sq.nn, Sq.e); rp, pp = pe(Sq.nn, Sq.e)
    # sequence-level, weighting each sequence once: also the bootstrap of the sequence-level rho
    bs = [sp(*Sq.iloc[rng.choice(len(Sq), len(Sq), replace=True)][["nn", "e"]].values.T)[0] for _ in range(2000)]
    # within-sequence (demeaned) correlation for sequences with >=2 windows
    multi = T.groupby("sequence_id").filter(lambda g: len(g) >= 2)
    dm = multi.groupby("sequence_id")[["nn_dist", err]].transform(lambda v: v - v.mean())
    rw, pw = sp(dm.nn_dist, dm[err])
    a[err] = dict(window_rho=r, window_p=p, n_windows=len(T), cluster_boot_ci=[float(lo), float(hi)], cluster_boot_median=float(med),
                  jackknife_seq_range=[jl, jh], seq_rho=rs, seq_p=ps, seq_pearson=rp, n_seq=len(Sq), seq_boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  within_seq_rho=rw, within_seq_p=pw, within_seq_n=len(dm))
    print(f"\n(a) {err}: window rho={r:.3f} (p={p:.1e}, n={len(T)}); cluster-bootstrap 95% CI [{lo:.3f},{hi:.3f}]; jackknife-1-seq range [{jl:.3f},{jh:.3f}]")
    print(f"    sequence-level (n={len(Sq)}): spearman {rs:.3f} (p={ps:.1e}), pearson {rp:.3f}, boot CI [{np.percentile(bs,2.5):.3f},{np.percentile(bs,97.5):.3f}]; within-sequence demeaned rho {rw:.3f} (p={pw:.2f}, n={len(dm)})")
res["a"] = a
# quartiles of nn_dist -> error
q = pd.qcut(T.nn_dist, 4, labels=["Q1", "Q2", "Q3", "Q4"])
print("    quartiles of nn_dist -> mean motion / contact:", T.groupby(q, observed=True)[["motion", "contact"]].mean().round(1).to_dict())
res["a"]["quartiles"] = T.groupby(q, observed=True)[["nn_dist", "motion", "contact"]].mean().round(2).to_dict()

# ---------------- helpers: OLS with cluster-robust SE, partial Spearman
def ols_cluster(y, X, cl):
    X = np.asarray(X, float); y = np.asarray(y, float); n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X); b = XtX_inv @ X.T @ y; u = y - X @ b
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        m = cl == g; s = X[m].T @ u[m]; meat += np.outer(s, s)
    G = len(np.unique(cl)); adj = G / (G - 1) * (n - 1) / (n - k)
    V = adj * XtX_inv @ meat @ XtX_inv; se = np.sqrt(np.diag(V))
    r2 = 1 - (u ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return b, se, r2, u
def design(df, cont, cats):
    cols = [np.ones(len(df))]; names = ["const"]
    for c in cont: v = df[c].values.astype(float); cols.append((v - v.mean()) / v.std()); names.append(c)
    for c in cats:
        d = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float)
        for n in d.columns: cols.append(d[n].values); names.append(n)
    return np.column_stack(cols), names
def partial_spearman(df, x, y, cont, cats):
    Xc, _ = design(df, cont, cats)
    rx = stats.rankdata(df[x]); ry = stats.rankdata(df[y])
    ux = rx - Xc @ np.linalg.lstsq(Xc, rx, rcond=None)[0]; uy = ry - Xc @ np.linalg.lstsq(Xc, ry, rcond=None)[0]
    return pe(ux, uy)

# per-hand long table
L = pd.concat([T.assign(hand="L", err_m=T.motion_L, err_c=T.contact_L, touch_h=T.touch_L),
               T.assign(hand="R", err_m=T.motion_R, err_c=T.contact_R, touch_h=T.touch_R)], ignore_index=True)
L["seq"] = pd.factorize(L.sequence_id)[0]

# ---------------- (b) confounds
b = {}
controls_cont = ["touch_h", "tscale", "toolscale"]; controls_cat = ["hand", "verb", "target_cat"]
for err in ("err_m", "err_c"):
    out = {}
    for name, cont, cats in (("raw", [], []), ("hand+touch", ["touch_h"], ["hand"]), ("+scales", ["touch_h", "tscale", "toolscale"], ["hand"]),
                             ("+verb", ["touch_h", "tscale", "toolscale"], ["hand", "verb"]),
                             ("full (hand,verb,target_cat,touch,tscale,toolscale)", controls_cont, controls_cat),
                             ("full + tool_cat", controls_cont, controls_cat + ["tool_cat"])):
        X, names = design(L, ["nn_dist"] + cont, cats); beta, se, r2, _ = ols_cluster(L[err].values, X, L.seq.values)
        i = names.index("nn_dist"); pr, pp = partial_spearman(L, "nn_dist", err, cont, cats)
        X0, _ = design(L, cont, cats); r2_0 = ols_cluster(L[err].values, X0, L.seq.values)[2]
        out[name] = dict(beta_mm_per_sd=float(beta[i]), se=float(se[i]), t=float(beta[i] / se[i]), r2=float(r2), r2_without_nn=float(r2_0), partial_spearman=pr, partial_p=pp, k=len(names))
        print(f"(b) {err} | {name:60s}: beta={beta[i]:6.2f} mm/SD (cluster SE {se[i]:.2f}, t={beta[i]/se[i]:.2f}); partial Spearman={pr:.3f} (p={pp:.1e}); R2 {r2_0:.3f}->{r2:.3f}")
    b[err] = out
# also target_mesh / tool_mesh fixed effects (identity fully absorbed), and triplet fixed effects
for err in ("err_m", "err_c"):
    for name, cats in (("mesh FE (hand, tool_mesh, target_mesh) + touch", ["hand", "tool_mesh", "target_mesh"]), ("triplet FE (hand, triplet) + touch", ["hand", "triplet"])):
        X, names = design(L, ["nn_dist", "touch_h"], cats); beta, se, r2, _ = ols_cluster(L[err].values, X, L.seq.values); i = names.index("nn_dist")
        pr, pp = partial_spearman(L, "nn_dist", err, ["touch_h"], cats)
        b[err][name] = dict(beta_mm_per_sd=float(beta[i]), se=float(se[i]), t=float(beta[i] / se[i]), r2=float(r2), partial_spearman=pr, partial_p=pp, k=len(names))
        print(f"(b) {err} | {name:60s}: beta={beta[i]:6.2f} mm/SD (cluster SE {se[i]:.2f}, t={beta[i]/se[i]:.2f}); partial Spearman={pr:.3f} (p={pp:.1e}); R2 {r2:.3f} (k={len(names)})")
res["b"] = b
print("    Spearman(nn_dist, tscale)=%.3f, (nn_dist, toolscale)=%.3f, (nn_dist, touch)=%.3f" % (sp(T.nn_dist, T.tscale)[0], sp(T.nn_dist, T.toolscale)[0], sp(T.nn_dist, T.touch)[0]))
print("    verb-mean nn_dist vs verb-mean motion (n verbs=%d): rho=%.3f" % (T.verb.nunique(), sp(*T.groupby('verb')[['nn_dist','motion']].mean().values.T)[0]))

# ---------------- (c) block decomposition
c = {}
blocks = ["bps_tool", "bps_target", "rot", "trans", "tscale", "toolscale"]
d2cols = [f"d2_{k}" for k in blocks]; share = T[d2cols].div(T.nn_dist ** 2, axis=0)
c["share_of_d2_mean"] = share.mean().round(4).to_dict(); c["share_of_d2_mean_train"] = D[D.split == "train"][d2cols].div(D[D.split == "train"].nn_dist ** 2, axis=0).mean().round(4).to_dict()
c["share_mean_half"] = float((T.d2_all_mean / T.nn_dist ** 2).mean())
print("\n(c) share of squared nn_dist by block (test_1):", {k: round(v, 3) for k, v in c["share_of_d2_mean"].items()}, " mean-half share %.3f" % c["share_mean_half"])
print("    same for train queries:", {k: round(v, 3) for k, v in c["share_of_d2_mean_train"].items()})
print("    mean per-dim RMS z-distance to the NN by block (test_1 / train):")
tab = []
for k in blocks + ["all_mean", "all_std"]:
    row = dict(block=k, dims=int(json.load(open(OUT / "g3_features_meta.json"))["blocks"][k]),
               rms_test1=float(T[f"rms_{k}"].mean()), rms_train=float(D[D.split == "train"][f"rms_{k}"].mean()),
               d2_test1=float(T[f"d2_{k}"].mean()), d2_train=float(D[D.split == "train"][f"d2_{k}"].mean()))
    row["rho_d2_motion"], row["p_d2_motion"] = sp(T[f"d2_{k}"], T.motion); row["rho_d2_contact"], _ = sp(T[f"d2_{k}"], T.contact)
    if f"bw_{k}_dist" in T: 
        row["rho_bw_motion"], row["p_bw_motion"] = sp(T[f"bw_{k}_dist"], T.motion); row["rho_bw_contact"], _ = sp(T[f"bw_{k}_dist"], T.contact)
        row["bw_dist_test1"] = float(T[f"bw_{k}_dist"].mean()); row["bw_dist_train"] = float(D[D.split == "train"][f"bw_{k}_dist"].mean())
        row["bw_same_verb"] = float(T[f"bw_{k}_same_verb"].mean())
    row["rho_d2_nn_dist"], _ = sp(T[f"d2_{k}"], T.nn_dist)
    tab.append(row)
    print(f"    {k:10s} dims={row['dims']:5d} rms {row['rms_test1']:.3f}/{row['rms_train']:.3f}  d2 {row['d2_test1']:7.1f}/{row['d2_train']:7.1f}  rho(d2_blk,motion)={row['rho_d2_motion']:+.3f} rho(d2_blk,contact)={row['rho_d2_contact']:+.3f}"
          + (f"  | blockwise-NN: dist {row['bw_dist_test1']:.2f}/{row['bw_dist_train']:.2f} rho(motion)={row['rho_bw_motion']:+.3f} rho(contact)={row['rho_bw_contact']:+.3f} sameverb={row['bw_same_verb']:.2f}" if "rho_bw_motion" in row else ""))
ctab = pd.DataFrame(tab); ctab.to_csv(OUT / "g3_block_table.csv", index=False); c["table"] = tab
# mean vs std halves within each block
for k in blocks:
    print(f"    {k:10s} mean-half rho(motion)={sp(T[f'd2_{k}_mean'], T.motion)[0]:+.3f}  std-half rho(motion)={sp(T[f'd2_{k}_std'], T.motion)[0]:+.3f}")
# multivariate: motion on all 6 block sqrt-distances (z-scored), clustered
Tz = T.copy(); Tz["seq"] = pd.factorize(Tz.sequence_id)[0]
for tgt in ("motion", "contact"):
    cont = [f"d2_{k}" for k in blocks]
    for cc in cont: Tz[cc + "_sqrt"] = np.sqrt(Tz[cc])
    X, names = design(Tz, [cc + "_sqrt" for cc in cont], []); beta, se, r2, _ = ols_cluster(Tz[tgt].values, X, Tz.seq.values)
    c[f"multi_{tgt}"] = {n: dict(beta=float(bb), se=float(s), t=float(bb / s)) for n, bb, s in zip(names[1:], beta[1:], se[1:])}; c[f"multi_{tgt}_r2"] = float(r2)
    print(f"    OLS {tgt} ~ all six block distances (z-scored sqrt d2), R2={r2:.3f}: " + ", ".join(f"{n.replace('d2_','').replace('_sqrt','')} {bb:+.1f}(t={bb/s:+.1f})" for n, bb, s in zip(names[1:], beta[1:], se[1:])))
    # bps_tool vs trans+rot head-to-head
    X, names = design(Tz, ["d2_bps_tool_sqrt", "d2_trans_sqrt", "d2_rot_sqrt"], []); beta, se, r2, _ = ols_cluster(Tz[tgt].values, X, Tz.seq.values)
    print(f"    OLS {tgt} ~ bps_tool + trans + rot, R2={r2:.3f}: " + ", ".join(f"{n} {bb:+.1f}(t={bb/s:+.1f})" for n, bb, s in zip(names[1:], beta[1:], se[1:])))
# raw handling amplitude: window-mean |trans|, |rot|
T["trans_mag"] = np.sqrt(T.w_trans_x ** 2 + T.w_trans_y ** 2 + T.w_trans_z ** 2); T["rot_mag"] = np.sqrt(T.w_rot_x ** 2 + T.w_rot_y ** 2 + T.w_rot_z ** 2)
print("    raw window-mean |target translation| : rho(motion)=%+.3f, rho(nn_dist)=%+.3f ; |rotvec|: rho(motion)=%+.3f, rho(nn_dist)=%+.3f" % (sp(T.trans_mag, T.motion)[0], sp(T.trans_mag, T.nn_dist)[0], sp(T.rot_mag, T.motion)[0], sp(T.rot_mag, T.nn_dist)[0]))
c["raw_trans_mag_rho_motion"] = sp(T.trans_mag, T.motion)[0]; c["raw_rot_mag_rho_motion"] = sp(T.rot_mag, T.motion)[0]
# who is the NN for the bps_target-only search: same target mesh?
res["c"] = c

# ---------------- (d) hand novelty vs conditioning novelty
d = {}
d["rho_hand_nn_motion"], d["p_hand_nn_motion"] = sp(T.hand_nn_dist, T.motion); d["rho_cond_nn_motion"], _ = sp(T.nn_dist, T.motion)
d["rho_hand_nn_contact"], _ = sp(T.hand_nn_dist, T.contact); d["rho_cond_nn_contact"], _ = sp(T.nn_dist, T.contact)
d["rho_hand_vs_cond"], _ = sp(T.hand_nn_dist, T.nn_dist)
d["boot_ci_hand_motion"] = [float(v) for v in cluster_boot(T.hand_nn_dist, T.motion)[[0, 2]]]
# bootstrap the difference of the two rhos
diffs = []
for _ in range(2000):
    pick = rng.choice(seqs, len(seqs), replace=True); ii = np.concatenate([groups[s] for s in pick])
    diffs.append(sp(T.hand_nn_dist.values[ii], T.motion.values[ii])[0] - sp(T.nn_dist.values[ii], T.motion.values[ii])[0])
d["boot_ci_rho_diff_hand_minus_cond_motion"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
print(f"\n(d) Spearman with motion: hand_nn_dist {d['rho_hand_nn_motion']:+.3f} (CI {d['boot_ci_hand_motion']}), nn_dist {d['rho_cond_nn_motion']:+.3f}; diff hand-cond CI {d['boot_ci_rho_diff_hand_minus_cond_motion']}")
print(f"    with contact: hand {d['rho_hand_nn_contact']:+.3f}, cond {d['rho_cond_nn_contact']:+.3f}; rho(hand_nn_dist, nn_dist)={d['rho_hand_vs_cond']:+.3f}")
Tz = T.copy(); Tz["seq"] = pd.factorize(Tz.sequence_id)[0]
for tgt in ("motion", "contact"):
    X, names = design(Tz, ["nn_dist", "hand_nn_dist"], []); beta, se, r2, _ = ols_cluster(Tz[tgt].values, X, Tz.seq.values)
    pr_c = partial_spearman(Tz, "nn_dist", tgt, ["hand_nn_dist"], []); pr_h = partial_spearman(Tz, "hand_nn_dist", tgt, ["nn_dist"], [])
    d[f"joint_{tgt}"] = dict(beta_cond=float(beta[1]), t_cond=float(beta[1] / se[1]), beta_hand=float(beta[2]), t_hand=float(beta[2] / se[2]), r2=float(r2),
                             partial_sp_cond_given_hand=pr_c[0], partial_sp_hand_given_cond=pr_h[0],
                             r2_cond_only=float(ols_cluster(Tz[tgt].values, design(Tz, ["nn_dist"], [])[0], Tz.seq.values)[2]),
                             r2_hand_only=float(ols_cluster(Tz[tgt].values, design(Tz, ["hand_nn_dist"], [])[0], Tz.seq.values)[2]))
    print(f"    OLS {tgt} ~ nn_dist + hand_nn_dist: cond {beta[1]:+.1f} mm/SD (t={beta[1]/se[1]:+.1f}), hand {beta[2]:+.1f} mm/SD (t={beta[2]/se[2]:+.1f}); R2 cond-only {d[f'joint_{tgt}']['r2_cond_only']:.3f}, hand-only {d[f'joint_{tgt}']['r2_hand_only']:.3f}, both {r2:.3f}; partial Spearman cond|hand {pr_c[0]:+.3f}, hand|cond {pr_h[0]:+.3f}")
# full-control model with both
for tgt in ("err_m", "err_c"):
    X, names = design(L, ["nn_dist", "hand_nn_dist"] + controls_cont, controls_cat); beta, se, r2, _ = ols_cluster(L[tgt].values, X, L.seq.values)
    i, j = names.index("nn_dist"), names.index("hand_nn_dist")
    d[f"joint_full_{tgt}"] = dict(beta_cond=float(beta[i]), t_cond=float(beta[i] / se[i]), beta_hand=float(beta[j]), t_hand=float(beta[j] / se[j]), r2=float(r2))
    print(f"    per-hand full controls, {tgt} ~ nn_dist + hand_nn_dist + controls: cond {beta[i]:+.1f} (t={beta[i]/se[i]:+.1f}), hand {beta[j]:+.1f} (t={beta[j]/se[j]:+.1f}), R2={r2:.3f}")
# retrieval quality: hand-space NN hand error vs cond NN hand error (sanity), and do they correlate with model error?
print(f"    retrieval hand error using hand-space NN {T.retr_hand_mm_handnn.mean():.1f} mm vs conditioning NN {T.retr_hand_mm_condnn.mean():.1f} mm; rho(retr_hand_condnn, motion)={sp(T.retr_hand_mm_condnn, T.motion)[0]:+.3f}, rho(retr_hand_handnn, motion)={sp(T.retr_hand_mm_handnn, T.motion)[0]:+.3f}")
d["rho_retr_hand_condnn_motion"] = sp(T.retr_hand_mm_condnn, T.motion)[0]; d["rho_retr_hand_handnn_motion"] = sp(T.retr_hand_mm_handnn, T.motion)[0]
res["d"] = d

# ---------------- (e) coverage numbers
e = {}
g = T.groupby("nn_same_triplet")[["nn_dist", "motion", "contact"]].agg(["mean", "size"]); print("\n(e) by whether the NN is another take of the same triplet:\n", g.round(2))
e["by_same_triplet"] = {str(k): {c: float(T[T.nn_same_triplet == k][c].mean()) for c in ["nn_dist", "motion", "contact"]} | {"n": int((T.nn_same_triplet == k).sum())} for k in (True, False)}
same_all = T.nn_same_triplet & T.nn_same_tool_mesh & T.nn_same_target_mesh
e["by_same_all"] = {str(k): {c: float(T[same_all == k][c].mean()) for c in ["nn_dist", "motion", "contact"]} | {"n": int((same_all == k).sum())} for k in (True, False)}
print("    same triplet AND both meshes:", e["by_same_all"])
# within same-triplet windows, does distance still predict error?
for k in (True, False):
    sub = T[same_all == k]; e[f"rho_within_same_all_{k}"] = sp(sub.nn_dist, sub.motion)[0]
    print(f"    within same_all={k} (n={len(sub)}): rho(nn_dist,motion)={sp(sub.nn_dist, sub.motion)[0]:+.3f}, rho(nn_dist,contact)={sp(sub.nn_dist, sub.contact)[0]:+.3f}")
# number of training sequences of the same triplet (coverage) vs error and vs nn_dist
off = pd.read_csv("/result/uhnam/dexcore/bimart_taco/train_store/offsets.csv"); ntr = off[off.split == "train"].groupby("triplet").size()
T["n_train_takes_triplet"] = T.triplet.map(ntr).fillna(0)
print(f"    rho(n train takes of same triplet, nn_dist)={sp(T.n_train_takes_triplet, T.nn_dist)[0]:+.3f}, (n takes, motion)={sp(T.n_train_takes_triplet, T.motion)[0]:+.3f}, (n takes, contact)={sp(T.n_train_takes_triplet, T.contact)[0]:+.3f}")
e["rho_ntakes_nn_dist"] = sp(T.n_train_takes_triplet, T.nn_dist)[0]; e["rho_ntakes_motion"] = sp(T.n_train_takes_triplet, T.motion)[0]; e["rho_ntakes_contact"] = sp(T.n_train_takes_triplet, T.contact)[0]
# linear fit slope: mm of motion error per unit nn_dist, and the expected error if test nn_dist matched train's mean (19.65)
sl, ic, *_ = stats.linregress(T.nn_dist, T.motion); e["slope_mm_per_unit"] = float(sl); e["pred_motion_at_train_nn_dist"] = float(ic + sl * D[D.split == "train"].nn_dist.mean()); e["pred_motion_at_test_nn_dist"] = float(ic + sl * T.nn_dist.mean())
print(f"    linear: motion = {ic:.1f} + {sl:.2f}*nn_dist ; at train-mean nn_dist ({D[D.split=='train'].nn_dist.mean():.1f}) -> {e['pred_motion_at_train_nn_dist']:.1f} mm ; at test mean ({T.nn_dist.mean():.1f}) -> {e['pred_motion_at_test_nn_dist']:.1f} mm")
res["e"] = e
T.to_csv(OUT / "g3_test1_windows.csv", index=False)
json.dump(res, open(OUT / "g3_summary.json", "w"), indent=1, default=float)
print("\n->", OUT)
