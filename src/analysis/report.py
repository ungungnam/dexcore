"""Write the action-conditioned analysis to disk: tables first, plots second.

TABLES ARE THE RESULT, plots are a view of them. Everything a figure shows is also written as CSV,
so a number in a plot can always be traced back and re-plotted without re-running the sweep.

    features.csv          one row per sequence -- the raw material, keep this
    group_counts.csv      sequences per action, the sample sizes every other table depends on
    group_summary.csv     per (action, feature): n, mean, sd, quartiles, min, max
    discriminability.csv  per feature: Kruskal-Wallis H, p, epsilon-squared, ranked -- omitted
                          when the grouping has one level, which leaves nothing to rank
    confounds.csv         the same effect size under a second grouping (verb vs tool), because in
                          TACO the verb and the tool are confounded by construction
    summary.json          what was run, when, over what, and the ranking's head
    plots/                counts, per-feature box plots for the top features, median heatmap
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.analysis import distribution as D
from src.analysis.features import feature_columns

log = logging.getLogger(__name__)

DEFAULT_TOP_K = 12


def _plt():
    """matplotlib with a headless backend, chosen before pyplot is imported."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------------- plotting
def plot_group_counts(df, group: str, path: Path) -> None:
    plt = _plt()
    counts = D.group_counts(df, group)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(counts))))
    ax.barh(counts[group][::-1], counts["n_sequences"][::-1], color="#4C72B0")
    ax.set_xlabel("sequences")
    ax.set_title(f"sequences per {group}  (total {int(counts['n_sequences'].sum())})")
    for y, v in enumerate(counts["n_sequences"][::-1]):
        ax.text(v, y, f" {v}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_feature_by_group(df, feature: str, group: str, path: Path,
                          stat: Optional[Dict[str, float]] = None) -> None:
    """Box plot of one feature across the groups, ordered by median so the trend is readable."""
    import pandas as pd
    plt = _plt()

    data, labels = [], []
    for key, sub in df.groupby(group, sort=True):
        x = pd.to_numeric(sub[feature], errors="coerce").dropna().to_numpy()
        if len(x) >= D.MIN_GROUP_N:
            data.append(x); labels.append(f"{key} (n={len(x)})")
    if not data:
        return
    order = np.argsort([np.median(d) for d in data])
    data = [data[i] for i in order]; labels = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.34 * len(data))))
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=labels, showfliers=False,
                    patch_artist=True, medianprops={"color": "black"})
    for patch in bp["boxes"]:
        patch.set_facecolor("#8FA8D0")
    rng = np.random.default_rng(0)               # one stream: re-seeding per box would draw the
    for i, d in enumerate(data, 1):              # same jitter in every one and look like structure
        ax.plot(d, rng.normal(i, 0.05, len(d)), ".", ms=2.5, color="#22314E", alpha=0.45)
    title = feature
    if stat:
        title += f"   (eps^2={stat['epsilon_sq']:.3f}, p={stat['p']:.2g})"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(feature)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_median_heatmap(df, group: str, features, path: Path) -> None:
    """z-scored median of each feature per group: the whole comparison on one page.

    z-scored ACROSS groups, per feature -- the features have incompatible units, and the question
    is which action is high or low on each, not how many metres it is.
    """
    plt = _plt()

    med = df.groupby(group, sort=True)[list(features)].median(numeric_only=True)
    z = (med - med.mean()) / med.std(ddof=0).replace(0, np.nan)
    z = z.dropna(axis=1, how="all")
    if z.empty:
        return
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * len(z.columns), 1.2 + 0.36 * len(z.index)))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(z.columns)), z.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(z.index)), z.index, fontsize=8)
    ax.set_title(f"median feature value per {group} (z-scored per feature)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.7, label="z")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------------------------------- report
def _jsonable(v):
    """Keep integers integral in summary.json -- `n_groups: 14.0` is a small lie about a count."""
    if isinstance(v, (bool, str)):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return v


def write_report(df, out_dir, group: str = "verb", top_k: int = DEFAULT_TOP_K,
                 meta: Optional[Dict[str, object]] = None, plots: bool = True,
                 confound_group: Optional[str] = "tool_name") -> Dict[str, object]:
    """Write every table, then the plots.

    Returns the tables it computed, keyed by name, plus "dir" -- so a caller can print a ranking
    without re-running Kruskal-Wallis over every feature, and without reading back a file that may
    legitimately be empty (a single-level grouping has nothing to rank).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = feature_columns(df.columns)

    df.to_csv(out / "features.csv", index=False)
    counts = D.group_counts(df, group); counts.to_csv(out / "group_counts.csv", index=False)
    summary = D.group_summary(df, group, feats); summary.to_csv(out / "group_summary.csv", index=False)
    disc = D.discriminability(df, group, feats)
    if len(disc):
        disc.to_csv(out / "discriminability.csv", index=False)
    else:
        log.warning("nothing to rank: no %s has %d or more sequences with a measurable feature",
                    group, D.MIN_GROUP_N)
    conf = None
    if confound_group and confound_group != group and confound_group in df.columns \
            and df[confound_group].nunique() > 1 and df[group].nunique() > 1:
        conf = D.compare_groupings(df, (group, confound_group), feats)
        if len(conf):
            conf.to_csv(out / "confounds.csv", index=False)
    log.info("wrote tables to %s", out)

    top = list(disc["feature"][:top_k]) if len(disc) else feats[:top_k]
    meta_out = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "n_sequences": int(len(df)),
        "n_features": len(feats),
        "group": group,
        f"n_{group}s": int(df[group].nunique()),
        "top_features": [
            {k: _jsonable(v)
             for k, v in row.items() if k in ("feature", "epsilon_sq", "p", "n_groups",
                                              f"highest_{group}", f"lowest_{group}")}
            for row in (disc.head(top_k).to_dict("records") if len(disc) else [])],
        **({"confound_group": confound_group,
            "features_tracking_the_group_more_than_the_confound":
                int((conf["ratio"] > 1).sum()) if conf is not None and "ratio" in conf else None}
           if conf is not None else {}),
        **(meta or {}),
    }
    (out / "summary.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))

    if plots:
        pdir = out / "plots"; pdir.mkdir(exist_ok=True)
        plot_group_counts(df, group, pdir / "group_counts.png")
        stats = {r["feature"]: r for r in disc.to_dict("records")} if len(disc) else {}
        for i, f in enumerate(top):
            plot_feature_by_group(df, f, group, pdir / f"{i:02d}_{f}.png", stats.get(f))
        plot_median_heatmap(df, group, top, pdir / "median_heatmap.png")
        # count what was written: a feature that is all-NaN or too sparse produces no file
        log.info("wrote %d plots to %s", len(list(pdir.glob("*.png"))), pdir)
    return {"dir": out, "features": df, "group_counts": counts, "group_summary": summary,
            "discriminability": disc, "confounds": conf}
