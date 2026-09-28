"""Verification of Step-9 claims (i)-(iii): recompute NN retrieval from the cache under several
conditions.  Writes only under <root>/verify/.

Conditions (all per (backend, category, role, hand), candidates always from a DIFFERENT sequence):
  base_cos / base_l2          X_soft, all other-sequence candidates (replicates step 9; L2 added)
  covonly_cos                 coverage mask as the vector (per-mesh constant) -> top1_mesh
  common_cos                  X_soft restricted to canonical points covered by ALL meshes of the group
  xmesh_cos / xmesh_l2        X_soft, candidates from a DIFFERENT mesh only (function-across-geometry)
  xmesh_centred_cos           same, group mean removed
  xmesh_hard_cos, xmesh_nn_cos  same on X_hard / X_soft_nn
  xmesh_seq_cos               same on the sequence-level rows (__seq.npz)
  inmesh_cos                  candidates from the SAME mesh only (function within geometry)
Also: chance_<attr> = mean over queries of the fraction of eligible candidates sharing the attribute;
null_verb_shuffle = top1_verb when verb labels are permuted across sequences WITHIN each mesh
(keeps the mesh->verb confound, breaks the contact->verb link).
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path("/result/uhnam/dexcore/canonical_contact")
sys.path.insert(0, str(ROOT / "scripts"))
from cc_common import list_cache_files, load_cache, seq_bootstrap_counts, percentile_ci  # noqa
OUT = ROOT / "verify"
BACKENDS = ["normalized", "aligned", "dino", "random_perm"]
ATTRS = ["verb", "mesh_id", "triplet"]
N_BOOT, SEED, K, MIN_TOUCH, N_SHUF = 1000, 0, 5, 0.2, 100
QUADS = ["q_same_mesh_same_verb", "q_same_mesh_diff_verb", "q_diff_mesh_same_verb", "q_diff_mesh_diff_verb"]


def load_full(path, min_touch):
    d = np.load(path, allow_pickle=True)
    c = load_cache(path, min_touch=min_touch)
    keep = c["meta"]["row"].values
    c["X_soft_nn"] = np.asarray(d["X_soft_nn"], np.float32)[keep]
    return c


def sim_matrix(X, metric):
    X = X.astype(np.float64)
    if metric == "cos":
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        Xn = X / np.maximum(nrm, 1e-12)
        S = Xn @ Xn.T
        S[:, nrm[:, 0] == 0] = -np.inf
        S[nrm[:, 0] == 0, :] = -np.inf  # a zero query has no meaningful neighbour
        return S
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    return -np.sqrt(np.maximum(D2, 0))


def flags(S, meta, elig, verb_override=None):
    """per-sample 0/1 flags. S: similarity (higher = closer); elig: (n,n) bool candidate mask."""
    n = len(meta)
    S = S.copy()
    S[~elig] = -np.inf
    n_cand = elig.sum(1)
    valid = (n_cand > 0) & np.isfinite(S.max(1))
    kk = np.minimum(K, n_cand)
    order = np.argsort(-S, axis=1, kind="stable")[:, :K]
    top1 = order[:, 0]
    colmask = np.arange(order.shape[1])[None, :] < kk[:, None]
    out = {"sequence_id": meta.sequence_id.values, "n_candidates": n_cand}
    for a in ATTRS:
        v = meta[a].values.astype(str) if not (a == "verb" and verb_override is not None) else verb_override
        same = v[:, None] == v[None, :]
        out[f"chance_{a}"] = np.where(valid, (same & elig).sum(1) / np.maximum(n_cand, 1), np.nan)
        out[f"top1_{a}"] = np.where(valid, same[np.arange(n), top1], np.nan)
        s5 = same[np.arange(n)[:, None], order] & colmask
        out[f"top5_any_{a}"] = np.where(valid, s5.any(1), np.nan)
        out[f"top5_frac_{a}"] = np.where(valid, s5.sum(1) / np.maximum(kk, 1), np.nan)
    sm, sv = out["top1_mesh_id"] == 1, out["top1_verb"] == 1
    for q, m in zip(QUADS, [sm & sv, sm & ~sv, ~sm & sv, ~sm & ~sv]):
        out[q] = np.where(valid, m, np.nan)
    df = pd.DataFrame(out); df["valid"] = valid
    return df


def shuffle_null(S, meta, elig, rng):
    """top1_verb under verb labels permuted across SEQUENCES within each mesh (mesh structure kept)."""
    seq = meta.sequence_id.values; mesh = meta.mesh_id.values; verb = meta.verb.values.astype(str)
    seq_u, inv = np.unique(seq, return_inverse=True)
    seq_mesh = np.zeros(len(seq_u), int); seq_verb = np.empty(len(seq_u), object)
    seq_mesh[inv] = mesh; seq_verb[inv] = verb
    seq_verb = seq_verb.astype(str)
    n = len(meta); S = S.copy(); S[~elig] = -np.inf
    n_cand = elig.sum(1); valid = (n_cand > 0) & np.isfinite(S.max(1))
    top1 = np.argmax(S, axis=1)
    vals = []
    for _ in range(N_SHUF):
        sv = seq_verb.copy()
        for m in np.unique(seq_mesh):
            idx = np.where(seq_mesh == m)[0]
            sv[idx] = sv[rng.permutation(idx)]
        v = sv[inv]
        vals.append(np.nanmean(np.where(valid, v == v[top1], np.nan)) if valid.any() else np.nan)
    return float(np.nanmean(vals))


METRICS = [f"{p}_{a}" for a in ATTRS for p in ("top1", "top5_any", "top5_frac", "chance")] + QUADS


def summarise(f, with_ci=True):
    f = f[f.valid]
    row = dict(n_queries=len(f), n_sequences=f.sequence_id.nunique())
    if len(f) == 0:
        return row
    seq_ids, inv = np.unique(f.sequence_id.values, return_inverse=True)
    counts = seq_bootstrap_counts(len(seq_ids), N_BOOT, SEED) if with_ci else None
    for c in METRICS:
        v = f[c].values.astype(float)
        row[c] = float(np.nanmean(v))
        if with_ci:
            w = counts[:, inv].astype(float); ws = w.sum(1)
            est = (w @ np.nan_to_num(v)) / np.maximum((w * np.isfinite(v)).sum(1), 1)
            est[ws == 0] = np.nan
            row[f"{c}_lo"], row[f"{c}_hi"], _ = percentile_ci(est)
    return row


def paired_diff(fa, fb):
    """bootstrap CI of mean(fb - fa) for top1_verb, top5_frac_verb (same samples, same sequences)."""
    assert (fa.sequence_id.values == fb.sequence_id.values).all()
    v = fa.valid.values & fb.valid.values
    out = dict(n_queries=int(v.sum()))
    if v.sum() == 0:
        return out
    seq_ids, inv = np.unique(fa.sequence_id.values[v], return_inverse=True)
    counts = seq_bootstrap_counts(len(seq_ids), N_BOOT, SEED)
    w = counts[:, inv].astype(float); ws = w.sum(1)
    for c in ["top1_verb", "top5_frac_verb", "q_diff_mesh_same_verb"]:
        d = (fb[c].values - fa[c].values)[v].astype(float)
        out[f"d_{c}"] = float(d.mean())
        est = (w @ d) / np.maximum(ws, 1); est[ws == 0] = np.nan
        out[f"d_{c}_lo"], out[f"d_{c}_hi"], _ = percentile_ci(est)
        out[f"d_{c}_p_le0"] = float(np.mean(est <= 0))
    return out


def main():
    t0 = time.time()
    rows, per_flags, cov_checks = [], {}, []
    rng = np.random.default_rng(SEED)
    for backend in BACKENDS:
        for cat, role, path in list_cache_files(ROOT, backend):
            c = load_full(path, MIN_TOUCH)
            cs = load_full(path.with_name(path.stem + "__seq.npz"), MIN_TOUCH)
            for hand, m in c["meta"].groupby("hand", sort=True):
                if m.sequence_id.nunique() < 2:
                    continue
                idx = m.index.values; m = m.reset_index(drop=True)
                Xs, Xh, Xn, cov = c["X_soft"][idx], c["X_hard"][idx], c["X_soft_nn"][idx], c["coverage"][idx]
                seq = m.sequence_id.values; mesh = m.mesh_id.values
                diff_seq = seq[:, None] != seq[None, :]
                diff_mesh = mesh[:, None] != mesh[None, :]
                # coverage constant within mesh?
                for mid in np.unique(mesh):
                    cm = cov[mesh == mid]
                    cov_checks.append(dict(backend=backend, category=cat, role=role, hand=hand, mesh_id=mid,
                                           n=len(cm), frac_points_constant=float((cm == cm[0]).all(0).mean())))
                common = cov.all(0)  # canonical points covered in every sample (hence every mesh) of the group
                conds = {}
                Scos, Sl2 = sim_matrix(Xs, "cos"), sim_matrix(Xs, "l2")
                conds["base_cos"] = flags(Scos, m, diff_seq)
                conds["base_l2"] = flags(Sl2, m, diff_seq)
                conds["covonly_cos"] = flags(sim_matrix(cov.astype(np.float32), "cos"), m, diff_seq)
                conds["common_cos"] = flags(sim_matrix(Xs[:, common], "cos"), m, diff_seq) if common.sum() >= 8 else None
                conds["xmesh_cos"] = flags(Scos, m, diff_seq & diff_mesh)
                conds["xmesh_l2"] = flags(Sl2, m, diff_seq & diff_mesh)
                Xc = Xs - Xs.mean(0, keepdims=True)
                conds["xmesh_centred_cos"] = flags(sim_matrix(Xc, "cos"), m, diff_seq & diff_mesh)
                # per-mesh mean removed (removes the mesh signature; the strongest geometry-blind variant)
                Xm = Xs.copy()
                for mid in np.unique(mesh):
                    Xm[mesh == mid] -= Xs[mesh == mid].mean(0, keepdims=True)
                conds["xmesh_meshcentred_cos"] = flags(sim_matrix(Xm, "cos"), m, diff_seq & diff_mesh)
                conds["xmesh_hard_cos"] = flags(sim_matrix(Xh, "cos"), m, diff_seq & diff_mesh)
                conds["xmesh_nn_cos"] = flags(sim_matrix(Xn, "cos"), m, diff_seq & diff_mesh)
                conds["xmesh_common_cos"] = flags(sim_matrix(Xs[:, common], "cos"), m, diff_seq & diff_mesh) if common.sum() >= 8 else None
                conds["inmesh_cos"] = flags(Scos, m, diff_seq & ~diff_mesh)
                # sequence-level rows
                ms = cs["meta"]; ms = ms[ms.hand == hand]
                if ms.sequence_id.nunique() >= 2:
                    si = ms.index.values; ms = ms.reset_index(drop=True)
                    sseq, smesh = ms.sequence_id.values, ms.mesh_id.values
                    dsq = sseq[:, None] != sseq[None, :]; dsm = smesh[:, None] != smesh[None, :]
                    Ss = sim_matrix(cs["X_soft"][si], "cos")
                    conds["seq_base_cos"] = flags(Ss, ms, dsq)
                    conds["xmesh_seq_cos"] = flags(Ss, ms, dsq & dsm)
                null_base = shuffle_null(Scos, m, diff_seq, rng)
                null_x = shuffle_null(Scos, m, diff_seq & diff_mesh, rng)
                for cond, f in conds.items():
                    if f is None:
                        continue
                    per_flags[(backend, cat, role, hand, cond)] = f
                    r = dict(backend=backend, category=cat, role=role, hand=hand, cond=cond, scope="category",
                             n_meshes=int(len(np.unique(mesh))), n_common_points=int(common.sum())) | summarise(f)
                    if cond == "base_cos": r["null_verb_shuffle"] = null_base
                    if cond == "xmesh_cos": r["null_verb_shuffle"] = null_x
                    rows.append(r)
            print(f"{backend}/{cat}/{role} done {time.time()-t0:.0f}s", flush=True)
    # pooled per (backend, role, hand, cond) and per (backend, cond); macro
    keys = sorted({(k[0], k[2], k[3], k[4]) for k in per_flags})
    for backend, role, hand, cond in keys:
        parts = [v for k, v in per_flags.items() if (k[0], k[2], k[3], k[4]) == (backend, role, hand, cond)]
        pooled = pd.concat(parts, ignore_index=True)
        rows.append(dict(backend=backend, category="ALL_pooled", role=role, hand=hand, cond=cond, scope="ALL_pooled") | summarise(pooled))
        cr = [r for r in rows if r["scope"] == "category" and (r["backend"], r["role"], r["hand"], r["cond"]) == (backend, role, hand, cond) and r["n_queries"] > 0]
        mr = dict(backend=backend, category="ALL_macro", role=role, hand=hand, cond=cond, scope="ALL_macro",
                  n_queries=sum(r["n_queries"] for r in cr), n_categories=len(cr))
        for cm in METRICS + ["null_verb_shuffle"]:
            v = [r[cm] for r in cr if np.isfinite(r.get(cm, np.nan))]
            mr[cm] = float(np.mean(v)) if v else np.nan
        rows.append(mr)
    for backend, cond in sorted({(k[0], k[4]) for k in per_flags}):
        parts = [v for k, v in per_flags.items() if (k[0], k[4]) == (backend, cond)]
        rows.append(dict(backend=backend, category="ALL_pooled", role="ALL", hand="ALL", cond=cond, scope="ALL_pooled_all") | summarise(pd.concat(parts, ignore_index=True)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "nn_conditions.csv", index=False)
    pd.DataFrame(cov_checks).to_csv(OUT / "coverage_constancy.csv", index=False)
    # paired dino vs aligned, and aligned vs normalized, per category and pooled, on cross-mesh conditions
    prow = []
    for cond in ["xmesh_cos", "xmesh_centred_cos", "xmesh_meshcentred_cos", "xmesh_hard_cos", "xmesh_seq_cos", "base_cos"]:
        for a, b in [("aligned", "dino"), ("normalized", "aligned"), ("normalized", "dino"), ("random_perm", "aligned")]:
            groups = sorted({(k[1], k[2], k[3]) for k in per_flags if k[0] == a and k[4] == cond})
            fa_all, fb_all = [], []
            for cat, role, hand in groups:
                if (b, cat, role, hand, cond) not in per_flags:
                    continue
                fa, fb = per_flags[(a, cat, role, hand, cond)], per_flags[(b, cat, role, hand, cond)]
                prow.append(dict(cond=cond, from_backend=a, to_backend=b, category=cat, role=role, hand=hand) | paired_diff(fa, fb))
                fa_all.append(fa); fb_all.append(fb)
            for role in ["tool", "target", "ALL"]:
                sel = [i for i, g in enumerate(groups) if (role == "ALL" or g[1] == role) and (b, g[0], g[1], g[2], cond) in per_flags]
                if sel:
                    prow.append(dict(cond=cond, from_backend=a, to_backend=b, category="ALL_pooled", role=role, hand="ALL")
                                | paired_diff(pd.concat([fa_all[i] for i in sel], ignore_index=True), pd.concat([fb_all[i] for i in sel], ignore_index=True)))
    pd.DataFrame(prow).to_csv(OUT / "nn_paired_backend_diffs.csv", index=False)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
