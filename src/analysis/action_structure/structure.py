"""Part 4: does a representation organise chunks by ACTION, and does it survive changing objects?

THE EPISODE RULE RUNS THROUGH EVERYTHING HERE. Two chunks cut from one episode share a tool, a
target, a mesh pair, a person and a moment; they are not two observations of an action. So every
neighbour, every pair and every cohesion term in this module EXCLUDES same-episode partners, and
every aggregate is reported twice: once per chunk, and once episode-balanced with each chunk
weighted 1/n_chunks so a 7-chunk episode does not outvote seven 1-chunk episodes.

The geometry-controlled test in `geometry_controlled` is the one that matters. Everything above it
can be explained by object identity; only that test holds identity fixed-different and asks whether
the action still shows.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Pairs sampled per group for the P(closer) estimate. The exact valid-pair counts are reported
#: separately; at this size the standard error of the estimate is ~5e-4, far below anything the
#: conclusion turns on.
AUC_SAMPLE = 500_000


def pairwise_distances(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """(N,N) distances, float32.

    `metric` is part of a representation's definition, not a detail: on these features cosine beats
    euclidean by more than any feature choice does, because it discards the vector's magnitude --
    which is largely "how much motion is in this window", an object and phase property rather than
    an action one. Euclidean uses the Gram trick and repairs the diagonal, since that trick can
    leave tiny negatives that turn into NaN under the square root.
    """
    if metric != "euclidean":
        from sklearn.metrics import pairwise_distances as skl
        return skl(np.asarray(X, dtype=np.float64), metric=metric).astype(np.float32)
    X = np.asarray(X, dtype=np.float32)
    sq = (X * X).sum(axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(d2, out=d2)


def _codes(labels: Sequence) -> np.ndarray:
    import pandas as pd
    return pd.Categorical(labels).codes.astype(np.int32)


# --------------------------------------------------------------------------------- clustering
def kmeans_scores(X: np.ndarray, meta, k: Optional[int] = None,
                  weights: Optional[np.ndarray] = None, seed: int = 0) -> Dict[str, object]:
    """k-means with k = the number of actions, scored against three labellings.

    Scored against the tool and target categories as well as the action, because a clustering that
    matches the tool better than the action has told us what it really found.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    actions = meta["action"].to_numpy()
    k = k or len(np.unique(actions))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(np.asarray(X, dtype=np.float64))
    cl = km.labels_
    out: Dict[str, object] = {"k": k, "clusters": cl}
    for name in ("action", "tool_cat", "target_cat"):
        lab = meta[name].to_numpy()
        out[f"{name}_ari"] = float(adjusted_rand_score(lab, cl))
        out[f"{name}_nmi"] = float(normalized_mutual_info_score(lab, cl))
        out[f"{name}_purity"] = _purity(cl, lab)
        if weights is not None:
            out[f"{name}_purity_epbal"] = _purity(cl, lab, weights)
    return out


def _purity(clusters: np.ndarray, labels: np.ndarray,
            weights: Optional[np.ndarray] = None) -> float:
    w = np.ones(len(clusters)) if weights is None else np.asarray(weights, dtype=float)
    total = 0.0
    for c in np.unique(clusters):
        m = clusters == c
        lab, wm = labels[m], w[m]
        best = max(wm[lab == u].sum() for u in np.unique(lab))
        total += best
    return float(total / w.sum())


def confusion(clusters: np.ndarray, labels: Sequence):
    import pandas as pd
    return pd.crosstab(pd.Series(labels, name="action"), pd.Series(clusters, name="cluster"))


# ------------------------------------------------------------------------ pairwise separation
def separation(D: np.ndarray, meta, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Within- vs between-action distances, silhouette and 1-NN action retrieval.

    Same-episode pairs are masked out of all four. Without that mask the within-action mean is
    partly a within-EPISODE mean, and 1-NN retrieval mostly recovers the neighbouring chunk of the
    same episode -- which says nothing about actions.
    """
    act = _codes(meta["action"])
    ep = _codes(meta["episode_id"])
    n = len(act)
    same_act = act[:, None] == act[None, :]
    diff_ep = ep[:, None] != ep[None, :]

    within = D[same_act & diff_ep]
    between = D[(~same_act) & diff_ep]
    out = {"within_action_mean": float(within.mean()), "between_action_mean": float(between.mean()),
           "within_over_between": float(within.mean() / between.mean()),
           "n_within_pairs": int(within.size // 2), "n_between_pairs": int(between.size // 2)}

    # leave-episode-out silhouette: cohesion against other episodes of the same action only
    Dm = np.where(diff_ep, D, np.nan)
    a = np.full(n, np.nan)
    b = np.full(n, np.inf)
    for c in np.unique(act):
        col = np.where(act[None, :] == c, Dm, np.nan)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(col, axis=1)
        a = np.where(act == c, m, a)
        b = np.where(act == c, b, np.fmin(b, m))
    s = (b - a) / np.maximum(a, b)
    out["silhouette_action"] = float(np.nanmean(s))
    if weights is not None:
        w = np.asarray(weights, float)
        ok = np.isfinite(s)
        out["silhouette_action_epbal"] = float((s[ok] * w[ok]).sum() / w[ok].sum())

    # 1-NN retrieval, nearest chunk from a DIFFERENT episode
    Dnn = np.where(diff_ep, D, np.inf)
    nn = np.argmin(Dnn, axis=1)
    hit = (act[nn] == act).astype(float)
    out["nn_action_acc"] = float(hit.mean())
    out["nn_action_acc_chance"] = float(sum((np.mean(act == c)) ** 2 for c in np.unique(act)))
    if weights is not None:
        w = np.asarray(weights, float)
        out["nn_action_acc_epbal"] = float((hit * w).sum() / w.sum())
    for name in ("tool_cat", "target_cat"):
        lab = _codes(meta[name])
        out[f"nn_{name}_acc"] = float((lab[nn] == lab).mean())
    return out


# ----------------------------------------------------------------- geometry-controlled pairing
def geometry_controlled(D: np.ndarray, meta, level: str = "mesh",
                        seed: int = 0) -> Dict[str, float]:
    """The central test: with BOTH objects different, are same-action chunks still closer?

    Pairs are restricted to different tool AND different target identity (mesh id, or category for
    `level="category"`), which also excludes same-episode pairs by construction. The score is
    P(d(same action) < d(different action)) over those pairs -- an AUC, where 0.5 is "the
    representation knows nothing about the action once the objects differ".
    """
    rng = np.random.default_rng(seed)
    act = _codes(meta["action"])
    tcol, ocol = ("tool_mesh", "target_mesh") if level == "mesh" else ("tool_cat", "target_cat")
    t, o = _codes(meta[tcol]), _codes(meta[ocol])

    iu = np.triu_indices(len(act), k=1)
    valid = (t[iu[0]] != t[iu[1]]) & (o[iu[0]] != o[iu[1]])
    same = act[iu[0]] == act[iu[1]]
    d = D[iu]

    ds, dd = d[valid & same], d[valid & ~same]
    out = {f"{level}_n_same_pairs": int(ds.size), f"{level}_n_diff_pairs": int(dd.size),
           f"{level}_same_mean": float(ds.mean()) if ds.size else np.nan,
           f"{level}_diff_mean": float(dd.mean()) if dd.size else np.nan}
    if ds.size == 0 or dd.size == 0:
        out[f"{level}_p_closer"] = np.nan
        return out
    a = ds if ds.size <= AUC_SAMPLE else rng.choice(ds, AUC_SAMPLE, replace=False)
    b = dd if dd.size <= AUC_SAMPLE else rng.choice(dd, AUC_SAMPLE, replace=False)
    out[f"{level}_p_closer"] = _auc_less(a, b)
    out[f"{level}_auc_sample"] = int(min(a.size, b.size))
    return out


def _auc_less(a: np.ndarray, b: np.ndarray) -> float:
    """P(A < B) with ties counted as half, by ranking -- the Mann-Whitney U statistic."""
    from scipy.stats import rankdata

    both = np.concatenate([a, b])
    r = rankdata(both)
    ra = r[:len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2.0
    return float(1.0 - u / (len(a) * len(b)))       # 1-U/(nm): probability a ranks BELOW b


# ------------------------------------------------------------------------ nearest-neighbour look
def neighbour_examples(D: np.ndarray, meta, n_query: int = 6, n_show: int = 3,
                       seed: int = 0) -> List[Dict[str, object]]:
    """For a few chunks, the nearest neighbours under three different constraints."""
    rng = np.random.default_rng(seed)
    act = meta["action"].to_numpy()
    ep = meta["episode_id"].to_numpy()
    tm, om = meta["tool_mesh"].to_numpy(), meta["target_mesh"].to_numpy()
    tc, oc = meta["tool_cat"].to_numpy(), meta["target_cat"].to_numpy()
    ci = meta["chunk_index"].to_numpy()
    ids = meta["chunk_id"].to_numpy()

    counts = meta["action"].value_counts()
    pool = [i for i in range(len(meta)) if counts[act[i]] >= 20]
    queries = rng.choice(pool, size=min(n_query, len(pool)), replace=False)

    out = []
    for q in queries:
        other_ep = ep != ep[q]
        rows = {}
        masks = {
            "same action, both objects different":
                other_ep & (act == act[q]) & (tm != tm[q]) & (om != om[q]),
            "shared object categories, different action":
                other_ep & (act != act[q]) & (tc == tc[q]) & (oc == oc[q]),
            "unconstrained nearest":
                other_ep,
        }
        for label, m in masks.items():
            if not m.any():
                rows[label] = []
                continue
            idx = np.where(m)[0]
            order = idx[np.argsort(D[q, idx])][:n_show]
            rows[label] = [{"chunk_id": ids[j], "action": act[j], "tool": tc[j], "target": oc[j],
                            "chunk_index": int(ci[j]), "distance": float(D[q, j])} for j in order]
        out.append({"query": {"chunk_id": ids[q], "action": act[q], "tool": tc[q],
                              "target": oc[q], "chunk_index": int(ci[q])}, "neighbours": rows})
    return out
