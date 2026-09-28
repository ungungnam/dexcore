"""Does an object-trajectory representation DETERMINE the hand trajectory?

This replaces the verb as the criterion. The verb was a proxy for "does this representation
preserve function"; the hand motion is the thing itself, so it is measured directly.

THE MEASUREMENT is retrieval, not regression, because a distance is all a representation gives us
and fitting a model would measure the model instead:

    for each chunk, take its k nearest chunks in OBJECT-representation space, average THEIR hand
    trajectories, and compare that against the chunk's own hand trajectory.

`relative_error` is that error over the error of predicting the dataset's mean hand trajectory.
1.0 means the representation told us nothing; 0.0 would mean it told us everything. It is the same
quantity whatever the representation's dimension, so the numbers compare directly.

TWO CONSTRAINTS, and the second is the real test:

    any        the nearest chunk from a different EPISODE. Same-episode neighbours are excluded
               everywhere -- they share a grasp, a take and a tool instance, so retrieving them
               measures nothing.
    geometry   the nearest chunk whose tool AND target CATEGORY both differ. This is transfer:
               a spoon-and-bowl chunk may only be predicted from, say, a spatula-and-pan one.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_K = 5


def _codes(labels) -> np.ndarray:
    import pandas as pd
    return pd.Categorical(labels).codes.astype(np.int32)


def flatten_target(H: np.ndarray) -> np.ndarray:
    """(N,T,C) hand target -> (N, T*C), channel z-scored so every channel weighs the same.

    Without the z-score the 3 position channels (metres, ~0.1) would be drowned by the 6 rotation
    and 90 finger channels (order 1), and the error would be a finger error wearing a hand's name.
    """
    X = np.asarray(H, dtype=np.float64)
    mu = np.nanmean(X, axis=(0, 1), keepdims=True)
    sd = np.nanstd(X, axis=(0, 1), keepdims=True)
    Z = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    return Z.reshape(len(Z), -1)


def _neighbour_mask(meta, constraint: str) -> np.ndarray:
    ep = _codes(meta["episode_id"])
    ok = ep[:, None] != ep[None, :]
    if constraint == "geometry":
        t, o = _codes(meta["tool_cat"]), _codes(meta["target_cat"])
        ok &= (t[:, None] != t[None, :]) & (o[:, None] != o[None, :])
    elif constraint != "any":
        raise ValueError(f"constraint must be 'any' or 'geometry', got {constraint!r}")
    return ok


def retrieval_error(D: np.ndarray, Y: np.ndarray, meta, k: int = DEFAULT_K,
                    constraint: str = "any") -> Dict[str, float]:
    """Predict each chunk's hand trajectory from its k nearest neighbours in `D`.

    Chunks whose hand annotation is missing are dropped from the scoring but stay available as
    neighbours -- a NaN target cannot be scored, but its object trajectory is still real.
    """
    Y = np.asarray(Y, dtype=np.float64)
    scorable = np.isfinite(Y).all(axis=1)
    ok = _neighbour_mask(meta, constraint) & scorable[None, :]

    n = len(Y)
    Dm = np.where(ok, D, np.inf).astype(np.float32, copy=False)
    order = np.argpartition(Dm, kth=min(k, n - 1) - 1, axis=1)[:, :k]
    # argpartition does not sort, and an all-inf row means "no admissible neighbour"
    has = np.isfinite(np.take_along_axis(Dm, order, axis=1)).all(axis=1) & scorable
    if not has.any():
        return {"relative_error": np.nan, "n_scored": 0, "k": k, "constraint": constraint}

    # Accumulate the neighbour mean one neighbour at a time. `Y[order]` would materialise an
    # (N, k, D) gather -- 5 GB at k=20 for the wrist+fingers target, on a shared machine.
    pred = np.zeros_like(Y)
    for c in range(order.shape[1]):
        pred += Y[order[:, c]]
    pred /= order.shape[1]
    err = np.linalg.norm(pred[has] - Y[has], axis=1)
    base = np.linalg.norm(Y[scorable].mean(axis=0)[None, :] - Y[has], axis=1)
    return {"relative_error": float(err.mean() / base.mean()),
            "median_relative_error": float(np.median(err / np.maximum(base, 1e-12))),
            "n_scored": int(has.sum()), "k": k, "constraint": constraint}


def random_baseline(Y: np.ndarray, meta, k: int = DEFAULT_K, constraint: str = "any",
                    seed: int = 0, repeats: int = 3) -> float:
    """The same estimator with RANDOM admissible neighbours.

    This is the number that matters: it holds the estimator, the k and the constraint fixed and
    removes only the representation, so the gap between it and `retrieval_error` is exactly what
    the representation contributed.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.float64)
    scorable = np.isfinite(Y).all(axis=1)
    ok = _neighbour_mask(meta, constraint) & scorable[None, :]
    out = []
    for _ in range(repeats):
        D = rng.random(ok.shape).astype(np.float32)
        out.append(retrieval_error(D, Y, meta, k, constraint)["relative_error"])
    return float(np.mean(out))


def distance_agreement(D: np.ndarray, Y: np.ndarray, meta, constraint: str = "any",
                       sample: int = 400_000, seed: int = 0) -> float:
    """Spearman correlation between object-representation distance and hand-trajectory distance.

    Retrieval asks about the nearest few; this asks whether the whole ordering agrees. A
    representation can win one and lose the other -- good locally, wrong globally, or vice versa.
    """
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=np.float64)
    scorable = np.isfinite(Y).all(axis=1)
    ok = np.triu(_neighbour_mask(meta, constraint) & scorable[None, :] & scorable[:, None], 1)
    i, j = np.nonzero(ok)
    if len(i) == 0:
        return np.nan
    if len(i) > sample:
        pick = rng.choice(len(i), sample, replace=False)
        i, j = i[pick], j[pick]
    hand_d = np.linalg.norm(Y[i] - Y[j], axis=1)
    return float(spearmanr(D[i, j], hand_d).statistic)


def evaluate(D: np.ndarray, targets: Dict[str, np.ndarray], meta, name: str,
             ks: Sequence[int] = (1, 5, 20), seed: int = 0) -> list:
    """Every (target, constraint, k) score for one representation. One row each."""
    rows = []
    for tname, H in targets.items():
        Y = flatten_target(H)
        for constraint in ("any", "geometry"):
            base = random_baseline(Y, meta, k=ks[1], constraint=constraint, seed=seed)
            agree = distance_agreement(D, Y, meta, constraint, seed=seed)
            for k in ks:
                r = retrieval_error(D, Y, meta, k, constraint)
                rows.append({"representation": name, "target": tname,
                             "constraint": constraint, "k": k,
                             "relative_error": r["relative_error"],
                             "median_relative_error": r["median_relative_error"],
                             "random_baseline_k5": base, "spearman_dist_agreement": agree,
                             "n_scored": r["n_scored"]})
    return rows
