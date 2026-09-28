"""Part 1: is the action label just a restatement of which objects are in the scene?

TACO pairs a verb with a tool and a target as a named triplet, so the audit has to come BEFORE any
structure finding: if knowing the tool category tells you the action, then a representation that
encodes object identity will look action-structured without containing any action information.

ASSOCIATION IS REPORTED THREE WAYS because each answers a different question:

    NMI          symmetric, 0..1: how much the two labellings share, normalised for their entropies
    U(action|X)  the uncertainty coefficient H(action) - H(action|X) over H(action), i.e. the share
                 of the action's entropy that X removes. DIRECTED, which is the question here --
                 "does the tool tell you the action", not "are they related"
    Cramer's V   the chi-square effect size, for a reader who wants a contingency-table number

All three are also computed episode-balanced (each chunk weighted 1/n_chunks), since an episode
that produced 7 chunks would otherwise count seven times toward the association.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

EPS = 1e-12


def _counts(a: Sequence, b: Sequence, w: Optional[np.ndarray] = None):
    """Weighted contingency table plus the label orders, as a dense array."""
    import pandas as pd

    w = np.ones(len(a)) if w is None else np.asarray(w, dtype=float)
    df = pd.DataFrame({"a": list(a), "b": list(b), "w": w})
    tab = df.pivot_table(index="a", columns="b", values="w", aggfunc="sum", fill_value=0.0)
    return tab.to_numpy(dtype=float), list(tab.index), list(tab.columns)


def _entropy(p: np.ndarray) -> float:
    p = p[p > EPS]
    return float(-(p * np.log(p)).sum())


def association(a: Sequence, b: Sequence, w: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Every association measure for one pair of labellings. `a` is the DEPENDENT one for U."""
    n, _, _ = _counts(a, b, w)
    tot = n.sum()
    if tot <= 0:
        return {"nmi": np.nan, "u_a_given_b": np.nan, "cramers_v": np.nan,
                "n_a": 0, "n_b": 0, "n": 0.0}
    P = n / tot
    pa, pb = P.sum(axis=1), P.sum(axis=0)
    Ha, Hb = _entropy(pa), _entropy(pb)
    Hab = _entropy(P.ravel())
    mi = max(0.0, Ha + Hb - Hab)
    nmi = mi / np.sqrt(Ha * Hb) if Ha > EPS and Hb > EPS else np.nan
    expected = np.outer(pa, pb) * tot
    chi2 = float(((n - expected) ** 2 / np.maximum(expected, EPS)).sum())
    k = min(len(pa), len(pb))
    v = np.sqrt(chi2 / (tot * (k - 1))) if k > 1 else np.nan
    return {"nmi": float(nmi), "u_a_given_b": float(mi / Ha) if Ha > EPS else np.nan,
            "cramers_v": float(v), "n_a": len(pa), "n_b": len(pb), "n": float(tot)}


def audit(meta, weights: Optional[np.ndarray] = None, action: str = "action"):
    """Association between the action and each object-identity labelling. One row per labelling."""
    import pandas as pd

    pair_cat = meta["tool_cat"] + " / " + meta["target_cat"]
    pair_mesh = meta["tool_mesh"] + " / " + meta["target_mesh"]
    against = {"tool_cat": meta["tool_cat"], "target_cat": meta["target_cat"],
               "tool+target_cat": pair_cat, "tool+target_mesh": pair_mesh,
               "tool_mesh": meta["tool_mesh"], "target_mesh": meta["target_mesh"]}
    rows: List[Dict[str, object]] = []
    for name, col in against.items():
        r = {"against": name, **association(meta[action], col)}
        if weights is not None:
            rb = association(meta[action], col, weights)
            r.update({"nmi_epbal": rb["nmi"], "u_a_given_b_epbal": rb["u_a_given_b"]})
        rows.append(r)
    return pd.DataFrame(rows)


def per_action_coverage(meta, action: str = "action"):
    """For each action: how many distinct objects it spans, and how many pairs it SHARES.

    The shared-pair count is the one that decides whether Part 4D is possible at all -- a
    geometry-controlled comparison needs actions that occur with more than one object pair.
    """
    import pandas as pd

    meta = meta.copy()
    meta["_pair"] = meta["tool_cat"] + " / " + meta["target_cat"]
    meta["_mesh_pair"] = meta["tool_mesh"] + " / " + meta["target_mesh"]
    pairs_by_action = {a: set(g["_pair"]) for a, g in meta.groupby(action)}
    rows = []
    for a, g in meta.groupby(action):
        others = set().union(*(p for b, p in pairs_by_action.items() if b != a)) \
            if len(pairs_by_action) > 1 else set()
        rows.append({
            action: a, "n_chunks": len(g), "n_episodes": g["episode_id"].nunique(),
            "n_tool_cat": g["tool_cat"].nunique(), "n_target_cat": g["target_cat"].nunique(),
            "n_tool_mesh": g["tool_mesh"].nunique(), "n_target_mesh": g["target_mesh"].nunique(),
            "n_cat_pairs": g["_pair"].nunique(), "n_mesh_pairs": g["_mesh_pair"].nunique(),
            "cat_pairs_shared_with_others": len(pairs_by_action[a] & others),
        })
    return pd.DataFrame(rows).sort_values("n_chunks", ascending=False).reset_index(drop=True)


def contingency(meta, rows: str, cols: str, weights: Optional[np.ndarray] = None):
    """A plain contingency table, for the report."""
    import pandas as pd

    n, ri, ci = _counts(meta[rows], meta[cols], weights)
    return pd.DataFrame(n, index=pd.Index(ri, name=rows), columns=pd.Index(ci, name=cols))
