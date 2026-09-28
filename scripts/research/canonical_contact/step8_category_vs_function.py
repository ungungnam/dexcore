"""Step 8: within- vs between-action variance of canonical contact maps.

For every (backend, category, role, hand), samples with touch_frac >= --min-touch:
  * sequence mean = mean of X_soft over the sequence's windows (sequence-balanced)
  * verb mean     = mean of its sequence means
  * within_action_variance  = mean over verbs (with >= 2 sequences) of the mean squared L2 distance
                              of sequence means to their verb mean
  * between_action_variance = mean over verbs of ||verb mean - grand mean||^2, grand mean = mean of
                              verb means (each verb weighs 1/n_verbs);
    between_action_variance_seqw = same but every verb weighted by its number of sequences
  * ratio = between / within, n_verbs, n_sequences, n_samples
  * sequence-level bootstrap CIs (resample sequences with replacement) for within, between, ratio
Also the pairwise verb-mean distance matrix (L2 and cosine).

Outputs:
  <root>/action_variance.csv               one row per (backend, category, role, hand) + macro rows per (backend, role, hand)
  <root>/action_verb_distances.csv         long table of verb-pair distances between verb means
  <root>/figures_data/mean_maps_<backend>_<category>_<role>_<hand>.npz
        verbs (V,), mean_maps (V,K), n_sequences_per_verb (V,), grand_mean (K,),
        verb_dist_l2 (V,V), verb_dist_cosine (V,V), canonical_points (K,3), seq_ids (S,), seq_verbs (S,),
        seq_means (S,K)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import (add_common_args, list_cache_files, load_cache, percentile_ci, resolve_backends,  # noqa: E402
                       safe_ratio, setup_logging)

log = setup_logging("step8")


def sequence_means(X: np.ndarray, meta: pd.DataFrame):
    seq_ids, inv = np.unique(meta.sequence_id.values, return_inverse=True)
    S = len(seq_ids)
    sums = np.zeros((S, X.shape[1]), np.float64)
    np.add.at(sums, inv, X.astype(np.float64))
    cnt = np.bincount(inv, minlength=S)
    means = sums / cnt[:, None]
    verbs = np.empty(S, object)
    verbs[inv] = meta.verb.values
    return seq_ids, means, verbs.astype(str), cnt


def variance_stats(seq_means: np.ndarray, seq_verbs: np.ndarray, weights: np.ndarray | None = None):
    """weights: multiplicity of each sequence (bootstrap); None = all ones. Returns dict of scalars + maps."""
    w = np.ones(len(seq_verbs)) if weights is None else weights.astype(np.float64)
    keep = w > 0
    verbs = np.unique(seq_verbs[keep])
    vmeans, nseq, within_v = [], [], []
    for v in verbs:
        m = keep & (seq_verbs == v)
        ww = w[m]
        mu = (seq_means[m] * ww[:, None]).sum(0) / ww.sum()
        vmeans.append(mu)
        nseq.append(ww.sum())
        if ww.sum() >= 2:
            within_v.append((((seq_means[m] - mu) ** 2).sum(1) * ww).sum() / ww.sum())
    vmeans = np.asarray(vmeans)
    nseq = np.asarray(nseq)
    within = float(np.mean(within_v)) if within_v else np.nan
    if len(verbs) >= 2:
        grand = vmeans.mean(0)
        between = float(((vmeans - grand) ** 2).sum(1).mean())
        grand_w = (vmeans * nseq[:, None]).sum(0) / nseq.sum()
        between_w = float((((vmeans - grand_w) ** 2).sum(1) * nseq).sum() / nseq.sum())
    else:
        grand = vmeans.mean(0) if len(vmeans) else np.full(seq_means.shape[1], np.nan)
        between, between_w = np.nan, np.nan
    return dict(verbs=verbs, vmeans=vmeans, nseq=nseq, within=within, between=between, between_w=between_w,
                ratio=float(safe_ratio(between, within)), ratio_w=float(safe_ratio(between_w, within)),
                grand=grand, n_verbs_within=len(within_v))


def verb_distance_matrices(vmeans: np.ndarray):
    diff = vmeans[:, None, :] - vmeans[None, :, :]
    l2 = np.sqrt((diff ** 2).sum(-1))
    n = np.linalg.norm(vmeans, axis=1)
    den = n[:, None] * n[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = 1.0 - np.clip((vmeans @ vmeans.T) / den, -1, 1)
    cos[den == 0] = np.nan
    return l2, cos


def process_group(backend, cat, role, hand, X, meta, pts, n_boot, seed, fig_dir):
    seq_ids, smeans, sverbs, cnt = sequence_means(X, meta)
    st = variance_stats(smeans, sverbs)
    S = len(seq_ids)
    rng = np.random.default_rng(seed)
    bw, bb, bbw, br, brw = [], [], [], [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, S, S), minlength=S)
        b = variance_stats(smeans, sverbs, w)
        bw.append(b["within"]); bb.append(b["between"]); bbw.append(b["between_w"])
        br.append(b["ratio"]); brw.append(b["ratio_w"])
    row = dict(backend=backend, category=cat, role=role, hand=hand, n_samples=len(meta), n_sequences=S,
               n_verbs=len(st["verbs"]), n_verbs_with_ge2_seq=st["n_verbs_within"],
               within_action_variance=st["within"], between_action_variance=st["between"],
               between_action_variance_seqw=st["between_w"], ratio=st["ratio"], ratio_seqw=st["ratio_w"])
    for name, arr in (("within", bw), ("between", bb), ("between_seqw", bbw), ("ratio", br), ("ratio_seqw", brw)):
        lo, hi, nv = percentile_ci(arr)
        row[f"{name}_ci_lo"], row[f"{name}_ci_hi"], row[f"{name}_n_boot_valid"] = lo, hi, nv
    l2, cos = verb_distance_matrices(st["vmeans"])
    verbs = st["verbs"]
    dist_rows = [dict(backend=backend, category=cat, role=role, hand=hand, verb_i=verbs[i], verb_j=verbs[j],
                      n_seq_i=int(st["nseq"][i]), n_seq_j=int(st["nseq"][j]), l2=float(l2[i, j]), cosine=float(cos[i, j]))
                 for i in range(len(verbs)) for j in range(i + 1, len(verbs))]
    fig_dir.mkdir(parents=True, exist_ok=True)
    safe = lambda s: str(s).replace(" ", "_").replace("/", "_")
    np.savez_compressed(fig_dir / f"mean_maps_{safe(backend)}_{safe(cat)}_{safe(role)}_{safe(hand)}.npz",
                        verbs=verbs.astype(str), mean_maps=st["vmeans"].astype(np.float32),
                        n_sequences_per_verb=st["nseq"], grand_mean=st["grand"].astype(np.float32),
                        verb_dist_l2=l2, verb_dist_cosine=cos,
                        canonical_points=pts if pts is not None else np.zeros((0, 3)),
                        seq_ids=seq_ids.astype(str), seq_verbs=sverbs.astype(str), seq_means=smeans.astype(np.float32),
                        seq_n_windows=cnt)
    return row, dist_rows


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--min-sequences", type=int, default=3, dest="min_sequences")
    args = ap.parse_args()
    root = args.root
    fig_dir = root / "figures_data"
    rows, drows = [], []
    for backend in resolve_backends(root, args.backends):
        for cat, role, path in list_cache_files(root, backend):
            c = load_cache(path, min_touch=args.min_touch)
            for hand, m in c["meta"].groupby("hand", sort=True):
                if m.sequence_id.nunique() < args.min_sequences or m.verb.nunique() < 1:
                    log.info("skip %s/%s/%s/%s: %d sequences", backend, cat, role, hand, m.sequence_id.nunique())
                    continue
                X = c["X_soft"][m.index.values]
                row, dr = process_group(backend, cat, role, hand, X, m.reset_index(drop=True), c["canonical_points"],
                                        args.n_boot, args.seed, fig_dir)
                rows.append(row); drows.extend(dr)
                log.info("%s/%s/%s/%s: within=%.4g between=%.4g ratio=%.3g (verbs=%d seqs=%d)", backend, cat, role,
                         hand, row["within_action_variance"], row["between_action_variance"], row["ratio"],
                         row["n_verbs"], row["n_sequences"])
    av = pd.DataFrame(rows)
    macro = []
    if len(av):
        for (backend, role, hand), g in av.groupby(["backend", "role", "hand"], sort=True):
            ok = g[np.isfinite(g.ratio)]
            macro.append(dict(backend=backend, category="ALL_macro", role=role, hand=hand, n_samples=int(ok.n_samples.sum()),
                              n_sequences=int(ok.n_sequences.sum()), n_verbs=int(ok.n_verbs.sum()),
                              n_verbs_with_ge2_seq=int(ok.n_verbs_with_ge2_seq.sum()),
                              within_action_variance=ok.within_action_variance.mean(),
                              between_action_variance=ok.between_action_variance.mean(),
                              between_action_variance_seqw=ok.between_action_variance_seqw.mean(),
                              ratio=ok.ratio.mean(), ratio_seqw=ok.ratio_seqw.mean(), n_categories=len(ok)))
    av = pd.concat([av, pd.DataFrame(macro)], ignore_index=True)
    av.to_csv(root / "action_variance.csv", index=False)
    pd.DataFrame(drows).to_csv(root / "action_verb_distances.csv", index=False)
    log.info("wrote action_variance.csv (%d rows) and action_verb_distances.csv (%d rows)", len(av), len(drows))
    if len(av):
        print(av[av.category == "ALL_macro"][["backend", "role", "hand", "within_action_variance",
                                              "between_action_variance", "ratio", "n_categories"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))


if __name__ == "__main__":
    main()
