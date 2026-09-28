"""Refutation attempt for claim (2): 'no representation separates function (B ~ C)'.

Exhaustive within-(category, role, hand) pairs recomputed from the canonical cache (not the capped
3000-pair sample), for several representation variants, plus verb-pair granularity, verb decoding,
and an instance-space (same-mesh) data-property test.
Outputs -> <root>/verify/function_separation/*.csv
"""
from __future__ import annotations

import sys, time, json, zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
OUT = ROOT / "verify" / "function_separation"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))
from cc_common import load_cache  # noqa

BACKENDS = ["aligned", "normalized", "random_perm", "dino"]
TOUCH = 0.2
N_BOOT = 1000
SEED = 0
MIN_SEQ_PER_VERB = 3


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- helpers
def pdist_l2(X):
    X = X.astype(np.float64)
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.maximum(D2, 0, out=D2)
    return np.sqrt(D2)


def pdist_cos(X):
    X = X.astype(np.float64)
    n = np.linalg.norm(X, axis=1)
    n[n == 0] = np.nan
    S = (X @ X.T) / (n[:, None] * n[None, :])
    return 1 - S


def seq_agg(D, mask, seq_code, n_seq):
    """Sum and count of D over pairs (i<j) in mask, aggregated per (seq_i, seq_j) -> (S, N) symmetric."""
    ii, jj = np.where(mask)
    d = D[ii, jj]
    fin = np.isfinite(d)
    ii, jj, d = ii[fin], jj[fin], d[fin]
    si, sj = seq_code[ii], seq_code[jj]
    flat = si * n_seq + sj
    S = np.bincount(flat, weights=d, minlength=n_seq * n_seq).reshape(n_seq, n_seq)
    N = np.bincount(flat, minlength=n_seq * n_seq).reshape(n_seq, n_seq).astype(np.float64)
    S = S + S.T; N = N + N.T  # symmetric (pairs are i<j so no diagonal double count issue)
    return S, N, d


def boot_means(SN_list, inset):
    """inset: (n_boot, n_seq) bool. Returns for each (S,N): (n_boot,) bootstrap mean (nan if no pairs)."""
    B = inset.astype(np.float64)
    out = []
    for S, N in SN_list:
        num = ((B @ S) * B).sum(1)
        den = ((B @ N) * B).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(den > 0, num / den, np.nan))
    return out


def sdiv(a, b):
    return float(a) / float(b) if (b is not None and np.isfinite(b) and b != 0 and np.isfinite(a)) else np.nan


def ci(x):
    x = x[np.isfinite(x)]
    if x.size < 10:
        return np.nan, np.nan, int(x.size)
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), int(x.size)


def auc_and_frac(b, c):
    """AUC = P(C > B) (Mann-Whitney), p-value (pairs treated as independent; anti-conservative),
    frac of C pairs farther than the median B pair."""
    if len(b) < 2 or len(c) < 2:
        return np.nan, np.nan, np.nan
    try:
        u, p = mannwhitneyu(c, b, alternative="two-sided")
        auc = u / (len(b) * len(c))
    except ValueError:
        auc, p = np.nan, np.nan
    return float(auc), float(p), float((c > np.median(b)).mean())


def remove_pcs(X, k):
    Xc = X - X.mean(0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc - (Xc @ Vt[:k].T) @ Vt[:k]


# ----------------------------------------------------------------------------- load
def load_all():
    data = {}
    for be in BACKENDS:
        d = ROOT / "canonical_contact_cache" / be
        for p in sorted(d.glob("*.npz")):
            stem = p.stem
            level = "seq" if stem.endswith("__seq") else "win"
            cat_role = stem[:-5] if level == "seq" else stem
            cat, role = cat_role.rsplit("__", 1)
            raw = np.load(p, allow_pickle=True)
            c = load_cache(p, min_touch=TOUCH)
            keep = np.asarray(raw["touch_frac"]).astype(float) >= TOUCH
            c["X_soft_nn"] = np.asarray(raw["X_soft_nn"], np.float32)[keep]
            data[(be, cat, role, level)] = c
    return data


# ----------------------------------------------------------------------------- per-group analysis
def variants_for(c, level, X_seq_centre=None):
    """name -> (matrix X or precomputed D, row-subset mask, metric)"""
    Xs, Xh, Xn = c["X_soft"], c["X_hard"], c["X_soft_nn"]
    v = {"soft_l2": (Xs, None, "l2"),
         "soft_cos": (Xs, None, "cos"),
         "hard_l2": (Xh, None, "l2"),
         "nn_l2": (Xn, None, "l2"),
         "centred_cos": (Xs - Xs.mean(0), None, "cos"),
         "pc1_removed_l2": (remove_pcs(Xs, 1), None, "l2"),
         "pc3_removed_l2": (remove_pcs(Xs, 3), None, "l2"),
         "zscore_l2": ((Xs - Xs.mean(0)) / np.where(Xs.std(0) > 1e-6, Xs.std(0), np.inf), None, "l2"),
         }
    m = Xs.mean(0)
    low = m <= np.median(m)  # the 'non-grip' half of canonical points (low mean contact)
    v["lowmean_half_l2"] = (Xs[:, low], None, "l2")
    v["highmean_half_l2"] = (Xs[:, ~low], None, "l2")
    if level == "win":
        v["soft_l2_mid"] = (Xs, c["meta"].window.values >= 1, "l2")
    return v


def analyse_group(c, hand, be, cat, role, level, rows, verbpair_rows, rng_seed):
    meta = c["meta"]
    sel = np.where(meta.hand.values == hand)[0]
    if len(sel) < 4:
        return
    mt = meta.iloc[sel].reset_index(drop=True)
    verbs = mt.verb.values; mesh = mt.mesh_id.values; seq = mt.sequence_id.values
    useq, seq_code = np.unique(seq, return_inverse=True)
    n_seq = len(useq)
    if len(np.unique(verbs)) < 2:
        return
    rng = np.random.default_rng(rng_seed)
    inset = np.zeros((N_BOOT, n_seq), bool)
    for b in range(N_BOOT):
        inset[b, rng.integers(0, n_seq, n_seq)] = True
    sub = {k: (v[0][sel] if isinstance(v[0], np.ndarray) else v[0], v[1][sel] if v[1] is not None else None, v[2])
           for k, v in variants_for(c, level).items()}
    same_seq = seq_code[:, None] == seq_code[None, :]
    same_verb = verbs[:, None] == verbs[None, :]
    same_mesh = mesh[:, None] == mesh[None, :]
    upper = np.triu(np.ones((len(sel), len(sel)), bool), 1) & ~same_seq
    masks = {"A": upper & same_verb & same_mesh, "B": upper & same_verb & ~same_mesh,
             "C": upper & ~same_verb, "C_sm": upper & ~same_verb & same_mesh,
             "C_dm": upper & ~same_verb & ~same_mesh}
    for vname, (X, rowmask, metric) in sub.items():
        D = pdist_l2(X) if metric == "l2" else pdist_cos(X)
        mm = masks
        if rowmask is not None:
            rm = rowmask[:, None] & rowmask[None, :]
            mm = {k: v & rm for k, v in masks.items()}
        agg = {k: seq_agg(D, m, seq_code, n_seq) for k, m in mm.items()}
        bm = dict(zip(agg, boot_means([(S, N) for S, N, _ in agg.values()], inset)))
        r = dict(backend=be, level=level, variant=vname, category=cat, role=role, hand=hand,
                 n_samples=int(len(sel) if rowmask is None else rowmask.sum()), n_sequences=n_seq,
                 n_verbs=int(len(np.unique(verbs))), n_meshes=int(len(np.unique(mesh))))
        for k, (S, N, d) in agg.items():
            r[f"n_{k}"] = int(d.size); r[f"mean_{k}"] = float(d.mean()) if d.size else np.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            for name, (nu, de) in {"cii": ("B", "C"), "cii_dm": ("B", "C_dm"), "csm_over_a": ("C_sm", "A"),
                                   "gp": ("B", "A")}.items():
                r[name] = sdiv(r[f"mean_{nu}"], r[f"mean_{de}"])
                lo, hi, nv = ci(bm[nu] / bm[de])
                r[f"{name}_lo"], r[f"{name}_hi"] = lo, hi
        b, cc = agg["B"][2], agg["C"][2]
        r["auc_C_gt_B"], r["mwu_p_B_vs_C"], r["frac_C_gt_medB"] = auc_and_frac(b, cc)
        r["auc_Cdm_gt_B"], _, r["frac_Cdm_gt_medB"] = auc_and_frac(b, agg["C_dm"][2])
        r["auc_Csm_gt_A"], r["mwu_p_A_vs_Csm"], r["frac_Csm_gt_medA"] = auc_and_frac(agg["A"][2], agg["C_sm"][2])
        rows.append(r)
        # ---- verb-pair granularity (soft_l2, hard_l2, nn_l2, seq soft only, to keep size manageable)
        if vname in ("soft_l2", "hard_l2", "nn_l2", "pc1_removed_l2", "zscore_l2"):
            uv = sorted(np.unique(verbs))
            nseq_v = {v: len(np.unique(seq[verbs == v])) for v in uv}
            for i1, v1 in enumerate(uv):
                for v2 in uv[i1 + 1:]:
                    if nseq_v[v1] < MIN_SEQ_PER_VERB or nseq_v[v2] < MIN_SEQ_PER_VERB:
                        continue
                    isv1 = verbs == v1; isv2 = verbs == v2
                    pair12 = (isv1[:, None] & isv2[None, :]) | (isv2[:, None] & isv1[None, :])
                    both = (isv1 | isv2)
                    mC = mm["C"] & pair12; mCdm = mm["C_dm"] & pair12; mCsm = mm["C_sm"] & pair12
                    mB = mm["B"] & both[:, None] & both[None, :]
                    mA = mm["A"] & both[:, None] & both[None, :]
                    a2 = {k: seq_agg(D, m, seq_code, n_seq) for k, m in
                          dict(A=mA, B=mB, C=mC, C_dm=mCdm, C_sm=mCsm).items()}
                    if a2["B"][2].size < 5 or a2["C_dm"][2].size < 5:
                        continue
                    sidx = np.unique(seq_code[both])
                    bm2 = dict(zip(a2, boot_means([(S[np.ix_(sidx, sidx)], N[np.ix_(sidx, sidx)]) for S, N, _ in a2.values()], inset[:, sidx])))
                    q = dict(backend=be, level=level, variant=vname, category=cat, role=role, hand=hand,
                             verb_i=v1, verb_j=v2, n_seq_i=nseq_v[v1], n_seq_j=nseq_v[v2])
                    for k, (S, N, d) in a2.items():
                        q[f"n_{k}"] = int(d.size); q[f"mean_{k}"] = float(d.mean()) if d.size else np.nan
                    with np.errstate(invalid="ignore", divide="ignore"):
                        q["ratio_Cdm_over_B"] = sdiv(q["mean_C_dm"], q["mean_B"])
                        q["ratio_lo"], q["ratio_hi"], q["n_boot_valid"] = ci(bm2["C_dm"] / bm2["B"])
                        q["ratio_C_over_B"] = sdiv(q["mean_C"], q["mean_B"])
                        q["ratio_Csm_over_A"] = sdiv(q["mean_C_sm"], q["mean_A"])
                        q["ratio_Csm_lo"], q["ratio_Csm_hi"], _ = ci(bm2["C_sm"] / bm2["A"])
                    q["auc_Cdm_gt_B"], q["mwu_p"], q["frac_Cdm_gt_medB"] = auc_and_frac(a2["B"][2], a2["C_dm"][2])
                    q["auc_Csm_gt_A"], _, _ = auc_and_frac(a2["A"][2], a2["C_sm"][2])
                    verbpair_rows.append(q)


# ----------------------------------------------------------------------------- pooled rows
def pooled(rows_df):
    """Pool pairs over categories per (backend, level, variant, role, hand) and over hands (role),
    plus macro (mean of per-group CII). Pooled CI: bootstrap ratio of pooled sums can't be rebuilt
    from means alone, so it is recomputed in analyse_pooled using stored S/N (see main)."""
    out = []
    for (be, lv, va, role, hand), g in rows_df.groupby(["backend", "level", "variant", "role", "hand"]):
        r = dict(backend=be, level=lv, variant=va, role=role, hand=hand, scope="ALL_pooled", n_groups=len(g))
        for k in ("A", "B", "C", "C_sm", "C_dm"):
            n = g[f"n_{k}"].sum(); r[f"n_{k}"] = int(n)
            r[f"mean_{k}"] = float((g[f"mean_{k}"] * g[f"n_{k}"]).sum() / n) if n else np.nan
        r["cii"] = r["mean_B"] / r["mean_C"]; r["cii_dm"] = r["mean_B"] / r["mean_C_dm"]
        r["csm_over_a"] = r["mean_C_sm"] / r["mean_A"] if r["n_A"] and r["n_C_sm"] else np.nan
        r["gp"] = r["mean_B"] / r["mean_A"]
        ok = g[np.isfinite(g.cii)]
        r["cii_macro"] = float(ok.cii.mean()); r["cii_dm_macro"] = float(ok.cii_dm.mean())
        r["auc_C_gt_B_macro"] = float(ok.auc_C_gt_B.mean())
        r["frac_C_gt_medB_macro"] = float(ok.frac_C_gt_medB.mean())
        r["csm_over_a_macro"] = float(g.csm_over_a[np.isfinite(g.csm_over_a)].mean())
        out.append(r)
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- verb decoding
def decode(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    out = []
    for (be, cat, role, level), c in data.items():
        if level != "win":
            continue
        meta = c["meta"]
        for hand in ("L", "R"):
            sel = np.where(meta.hand.values == hand)[0]
            mt = meta.iloc[sel]
            y = mt.verb.values; mesh = mt.mesh_id.values; seq = mt.sequence_id.values
            uv, cnt = np.unique(y, return_counts=True)
            keep_v = uv[cnt >= 10]
            if len(keep_v) < 2:
                continue
            k = np.isin(y, keep_v); sel = sel[k]; y = y[k]; mesh = mesh[k]; seq = seq[k]
            if len(np.unique(mesh)) < 3 or len(np.unique(seq)) < 10:
                continue
            for xname in ("X_soft", "X_hard"):
                X = c[xname][sel]
                for split, groups in (("seq_heldout", seq), ("mesh_heldout", mesh)):
                    ng = min(5, len(np.unique(groups)))
                    gkf = GroupKFold(n_splits=ng)
                    pred = np.empty(len(y), object); ok = np.zeros(len(y), bool)
                    for tr, te in gkf.split(X, y, groups):
                        if len(np.unique(y[tr])) < 2:
                            continue
                        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
                        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te]); ok[te] = True
                    if ok.sum() < 10:
                        continue
                    bacc = balanced_accuracy_score(y[ok], pred[ok])
                    # permutation-free chance: 1/n_classes for balanced accuracy
                    out.append(dict(backend=be, category=cat, role=role, hand=hand, features=xname, split=split,
                                    n=int(ok.sum()), n_verbs=len(keep_v), n_meshes=len(np.unique(mesh)),
                                    balanced_acc=float(bacc), chance=1 / len(keep_v),
                                    bacc_minus_chance=float(bacc - 1 / len(keep_v))))
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------- instance-space same-mesh test
def instance_same_mesh():
    s = pd.read_csv(ROOT / "samples_index.csv")
    rows = []; vp_rows = []
    cache = {}
    def vec(file, hand, role, window):
        if file not in cache:
            cache[file] = np.load(ROOT / "dense_window_cache" / file)
        d = cache[file]
        return np.asarray(d[f"soft_{hand}_{role}"][window], np.float64), np.asarray(d[f"hard_{hand}_{role}"][window], np.float64)
    for (cat, role, hand, mesh), g in s.groupby(["category", "role", "hand", "mesh_id"]):
        if g.verb.nunique() < 2 or g.sequence_id.nunique() < 4:
            continue
        Vs, Vh = zip(*[vec(f, hand, role, w) for f, w in zip(g.file, g.window)])
        Vs = np.stack(Vs); Vh = np.stack(Vh)
        verbs = g.verb.values; seq = g.sequence_id.values
        useq, sc = np.unique(seq, return_inverse=True)
        upper = np.triu(np.ones((len(g), len(g)), bool), 1) & (sc[:, None] != sc[None, :])
        sv = verbs[:, None] == verbs[None, :]
        for fname, V in (("inst_soft_l2", Vs), ("inst_hard_l2", Vh)):
            D = pdist_l2(V)
            a = D[upper & sv]; cc = D[upper & ~sv]
            if a.size < 5 or cc.size < 5:
                continue
            auc, p, fr = auc_and_frac(a, cc)
            rows.append(dict(category=cat, role=role, hand=hand, mesh_id=mesh, feature=fname, n_seq=len(useq),
                             n_verbs=g.verb.nunique(), n_A=a.size, n_Csm=cc.size, mean_A=a.mean(), mean_Csm=cc.mean(),
                             ratio_Csm_over_A=sdiv(cc.mean(), a.mean()), auc_Csm_gt_A=auc, frac_Csm_gt_medA=fr))
            if fname == "inst_soft_l2":
                uv = sorted(np.unique(verbs))
                for i, v1 in enumerate(uv):
                    for v2 in uv[i + 1:]:
                        m1 = verbs == v1; m2 = verbs == v2
                        mC = upper & ((m1[:, None] & m2[None, :]) | (m2[:, None] & m1[None, :]))
                        mA = upper & sv & ((m1 | m2)[:, None] & (m1 | m2)[None, :])
                        a2, c2 = D[mA], D[mC]
                        if a2.size < 5 or c2.size < 5:
                            continue
                        auc, p, fr = auc_and_frac(a2, c2)
                        vp_rows.append(dict(category=cat, role=role, hand=hand, mesh_id=mesh, verb_i=v1, verb_j=v2,
                                            n_A=a2.size, n_Csm=c2.size, mean_A=a2.mean(), mean_Csm=c2.mean(),
                                            ratio=sdiv(c2.mean(), a2.mean()), auc=auc))
    return pd.DataFrame(rows), pd.DataFrame(vp_rows)


def main():
    t0 = time.time()
    log("loading caches")
    data = load_all()
    log(f"loaded {len(data)} files in {time.time()-t0:.0f}s")
    rows, vp = [], []
    keys = sorted(data)
    for n, key in enumerate(keys):
        be, cat, role, level = key
        c = data[key]
        for hand in ("L", "R"):
            analyse_group(c, hand, be, cat, role, level, rows, vp, rng_seed=zlib.crc32(f'{cat}|{role}|{hand}'.encode()))
        if n % 10 == 0:
            log(f"{n}/{len(keys)} {key} rows={len(rows)} vp={len(vp)}")
    df = pd.DataFrame(rows); df.to_csv(OUT / "group_metrics.csv", index=False)
    vpd = pd.DataFrame(vp); vpd.to_csv(OUT / "verb_pair_metrics.csv", index=False)
    pooled(df).to_csv(OUT / "pooled_metrics.csv", index=False)
    log("group/verb-pair done; decoding")
    decode(data).to_csv(OUT / "verb_decoding.csv", index=False)
    log("decoding done; instance-space")
    r1, r2 = instance_same_mesh()
    r1.to_csv(OUT / "instance_same_mesh.csv", index=False); r2.to_csv(OUT / "instance_same_mesh_verbpairs.csv", index=False)
    log(f"all done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
