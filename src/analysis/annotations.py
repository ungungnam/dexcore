"""How TACO's annotation fields relate to EACH OTHER -- the dataset's design, before any motion.

The fields are not independent, and the ways they are not independent decide what any label-based
result can mean. A verb almost picks its tool; a triplet was often shot on one day; a mesh instance
belongs to exactly one category. None of that is visible from a trajectory, and all of it bounds
what "this representation separates actions" could possibly be measuring.

THREE UNITS OF ANALYSIS, because they answer different questions and can disagree:

    episode   one recording. Association here mixes the DESIGN with how many takes each triplet
              happened to get -- which is what a model trained on episodes actually sees.
    triplet   one (verb, tool, target) cell, counted once. This is the design alone, with the
              take counts removed.
    chunk     supplied by the caller when a chunked study needs matching numbers.

DIRECTED ASSOCIATION IS THE POINT. "Are verb and tool related" is the wrong question; "does the
tool tell you the verb" and "does the verb tell you the tool" have different answers and only the
first one threatens a verb-labelled result. `association` reports both directions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# The association maths is shared with the trajectory study's confound audit -- one implementation,
# so the numbers in the two reports are comparable rather than merely similar.
from src.analysis.action_structure.confound import EPS, _counts, association, contingency

#: The annotation fields this module relates to one another.
FIELDS = ("verb", "tool_cat", "target_cat", "tool_mesh", "target_mesh", "triplet", "date")


def episode_table(meta):
    """One row per episode, with the recording date and take parsed out of the sequence id.

    TACO's sequence directories are named `<YYYYMMDD>_<take>`, so the recording session is in the
    path. It is an annotation like any other, and a strong one: a triplet shot entirely on one day
    is confounded with that day's lighting, subject and setup.
    """
    df = meta.drop_duplicates("episode_id").reset_index(drop=True).copy()
    seq = df["episode_id"].str.split("/").str[-1]
    df["date"] = seq.str.split("_").str[0]
    df["take_no"] = seq.str.split("_").str[1].astype(int)
    if "action" in df.columns and "verb" not in df.columns:
        df["verb"] = df["action"]
    return df


def triplet_table(df):
    """One row per (verb, tool, target) cell -- the design, with take counts removed."""
    g = df.groupby("triplet", as_index=False).agg(
        verb=("verb", "first"), tool_cat=("tool_cat", "first"), target_cat=("target_cat", "first"),
        n_episodes=("episode_id", "size"), n_dates=("date", "nunique"),
        n_tool_mesh=("tool_mesh", "nunique"), n_target_mesh=("target_mesh", "nunique"))
    # a triplet has many mesh instances, so mesh columns carry the count, not an identity
    g["tool_mesh"] = g["triplet"]
    g["target_mesh"] = g["triplet"]
    g["date"] = g["triplet"]
    return g


def association_matrix(df, fields: Sequence[str] = FIELDS):
    """Every ordered pair of fields: NMI, and the share of `a`'s entropy that `b` removes.

    Read the diagonal as 1 and ignore it. Read a row as "how well does each other field predict
    this one" -- that is the direction that matters when the row is a label you intend to use.
    """
    import pandas as pd

    fields = [f for f in fields if f in df.columns and df[f].nunique() > 1]
    rows: List[Dict[str, object]] = []
    for a in fields:
        for b in fields:
            if a == b:
                continue
            r = association(df[a], df[b])
            rows.append({"target": a, "predictor": b, "nmi": r["nmi"],
                         "u_target_given_predictor": r["u_a_given_b"],
                         "cramers_v": r["cramers_v"], "n_target": r["n_a"],
                         "n_predictor": r["n_b"]})
    return pd.DataFrame(rows)


def occupancy(df):
    """How much of the possible label space the dataset actually uses.

    A design that used every combination would make the fields independent by construction. TACO
    uses a small, deliberate corner of it, and the size of that corner is the confound.
    """
    import pandas as pd

    rows = []
    pairs = [("verb", "tool_cat"), ("verb", "target_cat"), ("tool_cat", "target_cat")]
    for a, b in pairs:
        na, nb = df[a].nunique(), df[b].nunique()
        obs = df.groupby([a, b]).ngroups
        rows.append({"space": f"{a} x {b}", "possible": na * nb, "observed": obs,
                     "occupancy": obs / (na * nb)})
    na, nb, nc = df["verb"].nunique(), df["tool_cat"].nunique(), df["target_cat"].nunique()
    obs = df.groupby(["verb", "tool_cat", "target_cat"]).ngroups
    rows.append({"space": "verb x tool x target", "possible": na * nb * nc, "observed": obs,
                 "occupancy": obs / (na * nb * nc)})
    return pd.DataFrame(rows)


def field_profile(df, key: str, against: str):
    """Row-normalised contingency: for each level of `key`, its distribution over `against`.

    This is what "which verbs are interchangeable" is asked of -- two verbs with the same tool
    profile are, as far as the annotation goes, the same experiment done twice.
    """
    tab = contingency(df, key, against)
    return tab.div(tab.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def profile_similarity(df, key: str, against: str):
    """Cosine similarity between the `against`-profiles of every pair of `key` levels."""
    import pandas as pd

    P = field_profile(df, key, against)
    X = P.to_numpy()
    n = np.linalg.norm(X, axis=1, keepdims=True)
    S = (X / np.maximum(n, EPS)) @ (X / np.maximum(n, EPS)).T
    return pd.DataFrame(S, index=P.index, columns=P.index)


def exclusivity(df, key: str, against: str):
    """For each level of `key`: how many `against` levels it uses, and how many it SHARES.

    A level that shares nothing cannot be compared against anything with the shared part held
    fixed -- which is exactly when a controlled comparison becomes impossible.
    """
    import pandas as pd

    sets = {k: set(g[against]) for k, g in df.groupby(key)}
    rows = []
    for k, s in sets.items():
        others = set().union(*(v for j, v in sets.items() if j != k)) if len(sets) > 1 else set()
        rows.append({key: k, "n_levels": len(s), "n_shared": len(s & others),
                     "n_exclusive": len(s - others),
                     "exclusive_fraction": len(s - others) / max(len(s), 1),
                     "n_episodes": int((df[key] == k).sum())})
    return pd.DataFrame(rows).sort_values("n_episodes", ascending=False).reset_index(drop=True)


def mesh_structure(df):
    """Mesh instances against their categories, and how widely each instance is reused."""
    import pandas as pd

    rows = []
    for role in ("tool", "target"):
        cat, mesh = f"{role}_cat", f"{role}_mesh"
        per_cat = df.groupby(cat)[mesh].nunique()
        per_mesh_cat = df.groupby(mesh)[cat].nunique()
        per_mesh_trip = df.groupby(mesh)["triplet"].nunique()
        rows.append({"role": role, "n_categories": df[cat].nunique(),
                     "n_meshes": df[mesh].nunique(),
                     "meshes_per_category_median": float(per_cat.median()),
                     "meshes_per_category_max": int(per_cat.max()),
                     "meshes_in_more_than_one_category": int((per_mesh_cat > 1).sum()),
                     "triplets_per_mesh_median": float(per_mesh_trip.median()),
                     "triplets_per_mesh_max": int(per_mesh_trip.max())})
    return pd.DataFrame(rows)
