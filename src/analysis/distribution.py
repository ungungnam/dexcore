"""Group the per-sequence features by action and say which of them the action actually explains.

TWO OUTPUTS, and the second is the point:

    `group_summary`     the distribution of every feature within every verb -- n, mean, sd, and
                        the quartiles, because these features are skewed and a mean alone lies.
    `discriminability`  how strongly each feature separates the verbs at all, ranked.

WHY A RANKING AND NOT A CLASSIFIER. The question is which physical quantities the action label is
informative about, not how well a model can guess the verb. Kruskal-Wallis answers exactly that,
makes no normality assumption (speeds and path lengths are heavy-tailed), and survives the very
uneven group sizes here -- TACO has 3 `hit` triplets against 25 `pour in some`.

READ THE EFFECT SIZE, NOT THE p-VALUE. With 2317 sequences almost everything is "significant";
epsilon-squared (the share of rank variance the grouping explains) is what says whether it matters.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from src.analysis.features import LABEL_COLUMNS, feature_columns

log = logging.getLogger(__name__)

#: Below this many non-null samples a group is not summarised -- its quartiles would be noise.
MIN_GROUP_N = 3

#: Effect-size floor used as a denominator in `compare_groupings`. Epsilon-squared is in [0,1], so
#: this caps a confound ratio at 1000 -- "the confound explains essentially none of it".
MIN_EFFECT_SIZE = 1e-3


def feature_table(rows: Iterable[Dict[str, object]]):
    """Feature dicts -> DataFrame, label columns first. pandas is imported here, not at module
    import, so `features.py` stays usable in a plain-numpy context."""
    import pandas as pd

    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    first = [c for c in LABEL_COLUMNS if c in df.columns]
    return df[first + [c for c in df.columns if c not in first]]


def group_summary(df, group: str = "verb", features: Optional[Sequence[str]] = None):
    """Long-format distribution table: one row per (group, feature).

    Long rather than wide because there are ~50 features and ~15 verbs, and a long table is what
    both a plot and a `query()` want.
    """
    import pandas as pd

    feats = list(features or feature_columns(df.columns))
    out: List[Dict[str, object]] = []
    for key, sub in df.groupby(group, sort=True):
        for f in feats:
            x = pd.to_numeric(sub[f], errors="coerce").dropna()
            if len(x) < MIN_GROUP_N:
                continue
            out.append({group: key, "feature": f, "n": int(len(x)),
                        "mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                        "q25": float(x.quantile(0.25)), "median": float(x.median()),
                        "q75": float(x.quantile(0.75)),
                        "min": float(x.min()), "max": float(x.max())})
    return pd.DataFrame(out)


def discriminability(df, group: str = "verb", features: Optional[Sequence[str]] = None,
                     min_group_n: int = MIN_GROUP_N):
    """Kruskal-Wallis per feature, ranked by effect size. One row per feature.

    `epsilon_sq = (H - k + 1) / (n - k)` is the rank-variance share explained by the grouping:
    0 means the feature is the same in every verb, 1 means the verbs are perfectly ordered by it.
    Groups smaller than `min_group_n` are dropped from a feature's test (and counted in `n_groups`
    so a feature tested on 4 verbs is not compared against one tested on 15 without warning).
    """
    import pandas as pd
    from scipy import stats

    feats = list(features or feature_columns(df.columns))
    rows: List[Dict[str, object]] = []
    for f in feats:
        samples, labels = [], []
        for key, sub in df.groupby(group, sort=True):
            x = pd.to_numeric(sub[f], errors="coerce").dropna().to_numpy()
            if len(x) >= min_group_n:
                samples.append(x)
                labels.append(key)
        k, n = len(samples), int(sum(len(s) for s in samples))
        if k < 2 or n <= k:
            continue
        try:
            H, p = stats.kruskal(*samples)
        except ValueError as e:                    # every value identical -> no ranks to compare
            log.debug("kruskal skipped for %s: %s", f, e)
            continue
        eps = (float(H) - k + 1) / (n - k)
        top = max(zip(labels, samples), key=lambda t: np.median(t[1]))
        bot = min(zip(labels, samples), key=lambda t: np.median(t[1]))
        rows.append({"feature": f, "H": float(H), "p": float(p),
                     "epsilon_sq": max(0.0, eps), "n": n, "n_groups": k,
                     f"highest_{group}": top[0], "highest_median": float(np.median(top[1])),
                     f"lowest_{group}": bot[0], "lowest_median": float(np.median(bot[1]))})
    out = pd.DataFrame(rows)
    return out.sort_values("epsilon_sq", ascending=False).reset_index(drop=True) if len(out) else out


def group_counts(df, group: str = "verb"):
    """How many sequences each action has -- the first thing to look at before any comparison."""
    import pandas as pd

    c = df.groupby(group, sort=True).size().rename("n_sequences").reset_index()
    return c.sort_values("n_sequences", ascending=False).reset_index(drop=True)


def compare_groupings(df, groups: Sequence[str] = ("verb", "tool_name"),
                      features: Optional[Sequence[str]] = None):
    """Effect size of each feature under SEVERAL groupings, side by side. The confound check.

    TACO's design confounds the two things one would most like to separate: a `brush` verb is
    almost always performed with a brush, so a feature that really measures the TOOL scores highly
    "by verb" as well. Running the same test against `tool_name` and comparing is the cheapest
    honest diagnostic there is.

    `ratio` is the first grouping's effect size over the largest of the others: above 1 the feature
    tracks the action more than the tool, below 1 it is mostly the tool talking. It is a warning
    flag, not a correction -- properly separating the two needs the verbs that share a tool, which
    `--verbs` can select.
    """
    import pandas as pd

    out = None
    for g in groups:
        if g not in df.columns or df[g].nunique() < 2:
            continue
        d = discriminability(df, g, features)
        if d.empty:                               # no group large enough to test: no columns either
            continue
        d = d[["feature", "epsilon_sq", "n_groups"]].rename(
            columns={"epsilon_sq": f"epsilon_sq_{g}", "n_groups": f"n_{g}s"})
        out = d if out is None else out.merge(d, on="feature", how="outer")

    # A grouping with one level, or with no testable group, produces no column -- and without the
    # FIRST grouping's column there is nothing to compare against, so say so instead of indexing a
    # column that is not there. Filtering to a single verb reaches this on an ordinary command line.
    first = f"epsilon_sq_{groups[0]}"
    if out is None or len(out) == 0 or first not in out.columns:
        return pd.DataFrame()
    rest = [f"epsilon_sq_{g}" for g in groups[1:] if f"epsilon_sq_{g}" in out.columns]
    if rest:
        # Floor the denominator rather than dropping it: a feature the confound explains NOTHING of
        # is the best possible case, and it must come out as a large ratio, not a missing one.
        out["ratio"] = out[first] / out[rest].max(axis=1).clip(lower=MIN_EFFECT_SIZE)
    return out.sort_values(first, ascending=False).reset_index(drop=True)
