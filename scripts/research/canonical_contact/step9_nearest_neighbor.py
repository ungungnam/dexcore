"""Step 9: cosine nearest-neighbour retrieval inside each (backend, category, role, hand).

For every sample (touch_frac >= --min-touch) the candidates are all samples of the group from a
DIFFERENT sequence. Reported per group (and pooled / macro per (backend, role, hand)):
  top1_<attr>, top5_any_<attr>, top5_frac_<attr>   for attr in verb, mesh_id, counterpart_cat, triplet, subject
      top1     : top-1 neighbour shares the attribute
      top5_any : at least one of the top-5 shares it
      top5_frac: fraction of the top-5 sharing it
  chance_<attr>: mean over samples of the fraction of candidates sharing the attribute (random-pick baseline)
  q_same_mesh_same_verb, q_same_mesh_diff_verb, q_diff_mesh_same_verb, q_diff_mesh_diff_verb: 2x2 breakdown of top-1
Sequence-level bootstrap CIs (resample sequences with replacement, weight samples by multiplicity).

Output: <root>/nearest_neighbor_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import (add_common_args, list_cache_files, load_cache, percentile_ci, resolve_backends,  # noqa: E402
                       seq_bootstrap_counts, setup_logging)

log = setup_logging("step9")
ATTRS = ["verb", "mesh_id", "counterpart_cat", "triplet", "subject"]
QUADS = ["q_same_mesh_same_verb", "q_same_mesh_diff_verb", "q_diff_mesh_same_verb", "q_diff_mesh_diff_verb"]
K = 5


def per_sample_flags(X: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame | None:
    """One row per query sample with 0/1 flags (NaN rows for samples without eligible candidates)."""
    n = len(meta)
    Xn = X.astype(np.float64)
    norm = np.linalg.norm(Xn, axis=1, keepdims=True)
    Xn = Xn / np.maximum(norm, 1e-12)
    sim = Xn @ Xn.T
    seq = meta.sequence_id.values
    same_seq = seq[:, None] == seq[None, :]
    sim[same_seq] = -np.inf  # excludes self and same-sequence samples
    sim[(norm[:, 0] == 0)[None, :].repeat(n, 0)] = -np.inf  # zero vectors never retrieved
    n_cand = (~same_seq).sum(1)
    flags = {"sequence_id": seq, "n_candidates": n_cand}
    valid = (n_cand > 0) & np.isfinite(sim.max(1))  # at least one retrievable candidate
    order = np.argsort(-sim, axis=1, kind="stable")[:, :K]
    top1 = order[:, 0]
    for a in ATTRS:
        v = meta[a].values.astype(str)
        same = v[:, None] == v[None, :]
        cand_same = (same & ~same_seq).sum(1)
        flags[f"chance_{a}"] = np.where(valid, cand_same / np.maximum(n_cand, 1), np.nan)
        t1 = same[np.arange(n), top1].astype(float)
        kk = np.minimum(min(K, order.shape[1]), n_cand)
        s5 = same[np.arange(n)[:, None], order]  # (n,K); columns beyond kk are -inf sims -> masked
        # a group with fewer than K samples yields fewer than K columns in `order`
        colmask = np.arange(order.shape[1])[None, :] < kk[:, None]
        s5 = s5 & colmask
        flags[f"top1_{a}"] = np.where(valid, t1, np.nan)
        flags[f"top5_any_{a}"] = np.where(valid, s5.any(1).astype(float), np.nan)
        flags[f"top5_frac_{a}"] = np.where(valid, s5.sum(1) / np.maximum(kk, 1), np.nan)
    sm = flags["top1_mesh_id"] == 1
    sv = flags["top1_verb"] == 1
    flags["q_same_mesh_same_verb"] = np.where(valid, (sm & sv).astype(float), np.nan)
    flags["q_same_mesh_diff_verb"] = np.where(valid, (sm & ~sv).astype(float), np.nan)
    flags["q_diff_mesh_same_verb"] = np.where(valid, (~sm & sv).astype(float), np.nan)
    flags["q_diff_mesh_diff_verb"] = np.where(valid, (~sm & ~sv).astype(float), np.nan)
    df = pd.DataFrame(flags)
    df["valid"] = valid
    return df


METRIC_COLS = ([f"{p}_{a}" for a in ATTRS for p in ("top1", "top5_any", "top5_frac", "chance")] + QUADS)


def summarise(flags: pd.DataFrame, n_boot: int, seed: int, with_ci: bool = True) -> dict:
    f = flags[flags.valid]
    row = dict(n_queries=len(f), n_sequences=f.sequence_id.nunique())
    if len(f) == 0:
        return row
    seq_ids, inv = np.unique(f.sequence_id.values, return_inverse=True)
    vals = {c: f[c].values.astype(float) for c in METRIC_COLS}
    for c, v in vals.items():
        row[c] = float(np.nanmean(v))
    if with_ci and n_boot > 0:
        counts = seq_bootstrap_counts(len(seq_ids), n_boot, seed)  # (n_boot, S)
        w = counts[:, inv].astype(np.float64)  # (n_boot, n_queries)
        wsum = w.sum(1)
        for c, v in vals.items():
            est = (w @ v) / np.maximum(wsum, 1)
            est[wsum == 0] = np.nan
            row[f"{c}_ci_lo"], row[f"{c}_ci_hi"], _ = percentile_ci(est)
    return row


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--min-sequences", type=int, default=2, dest="min_sequences")
    args = ap.parse_args()
    root = args.root
    rows = []
    per_group_flags = {}
    for backend in resolve_backends(root, args.backends):
        for cat, role, path in list_cache_files(root, backend):
            c = load_cache(path, min_touch=args.min_touch)
            for hand, m in c["meta"].groupby("hand", sort=True):
                if m.sequence_id.nunique() < args.min_sequences:
                    continue
                X = c["X_soft"][m.index.values]
                m = m.reset_index(drop=True)
                fl = per_sample_flags(X, m)
                fl["category"] = cat
                per_group_flags[(backend, role, hand, cat)] = fl
                r = dict(backend=backend, scope="category", category=cat, role=role, hand=hand) | summarise(fl, args.n_boot, args.seed)
                rows.append(r)
                log.info("%s/%s/%s/%s: n=%d top1_verb=%.3f (chance %.3f) top1_mesh=%.3f (chance %.3f)", backend, cat, role, hand,
                         r["n_queries"], r.get("top1_verb", np.nan), r.get("chance_verb", np.nan),
                         r.get("top1_mesh_id", np.nan), r.get("chance_mesh_id", np.nan))
    # pooled + macro per (backend, role, hand)
    keys = sorted({k[:3] for k in per_group_flags})
    for backend, role, hand in keys:
        parts = [v for k, v in per_group_flags.items() if k[:3] == (backend, role, hand)]
        pooled = pd.concat(parts, ignore_index=True)
        rows.append(dict(backend=backend, scope="ALL_pooled", category="ALL_pooled", role=role, hand=hand)
                    | summarise(pooled, args.n_boot, args.seed))
        cat_rows = [r for r in rows if r["backend"] == backend and r["role"] == role and r["hand"] == hand
                    and r["scope"] == "category" and r.get("n_queries", 0) > 0]
        mr = dict(backend=backend, scope="ALL_macro", category="ALL_macro", role=role, hand=hand,
                  n_queries=sum(r["n_queries"] for r in cat_rows), n_sequences=sum(r["n_sequences"] for r in cat_rows),
                  n_categories=len(cat_rows))
        for c in METRIC_COLS:
            v = [r[c] for r in cat_rows if np.isfinite(r.get(c, np.nan))]
            mr[c] = float(np.mean(v)) if v else np.nan
        rows.append(mr)
    out = pd.DataFrame(rows)
    front = ["backend", "scope", "category", "role", "hand", "n_queries", "n_sequences"]
    out = out[front + [c for c in out.columns if c not in front]]
    out.to_csv(root / "nearest_neighbor_metrics.csv", index=False)
    log.info("wrote nearest_neighbor_metrics.csv (%d rows)", len(out))
    show = out[out.scope != "category"][["backend", "scope", "role", "hand", "n_queries", "top1_verb", "chance_verb",
                                          "top1_mesh_id", "chance_mesh_id"] + QUADS]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
