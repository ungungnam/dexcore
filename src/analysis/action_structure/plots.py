"""Figures for Parts 3 and 4A. Every figure is a view of a table that is also written to disk.

THE EMBEDDING FIGURES ARE THE POINT OF THIS FILE: the same 2-D embedding, drawn five times with
five different colourings. If the action colouring looks structured and the tool colouring looks
identically structured, the structure belongs to the tool. Drawing them side by side is what makes
that impossible to miss, and it is the reason the colourings share one embedding rather than each
getting its own.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Colourings every embedding is drawn with. `chunk_index` is continuous; the rest are categorical.
COLOURINGS = ("action", "tool_cat", "target_cat", "cat_pair", "chunk_index")
MAX_LEGEND = 12


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _scatter(ax, xy, values, title: str, continuous: bool = False):
    plt = _plt()
    if continuous:
        v = np.asarray(values, dtype=float)
        s = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=3, cmap="viridis", alpha=0.7)
        plt.colorbar(s, ax=ax, shrink=0.75)
    else:
        vals = np.asarray(values, dtype=object)
        order = [u for u, _ in sorted(((u, (vals == u).sum()) for u in np.unique(vals)),
                                      key=lambda t: -t[1])]
        cmap = plt.get_cmap("tab20")
        for i, u in enumerate(order):
            m = vals == u
            if i < MAX_LEGEND:
                ax.scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.75, color=cmap(i % 20),
                           label=f"{u} ({m.sum()})")
            else:
                ax.scatter(xy[m, 0], xy[m, 1], s=2, alpha=0.3, color="0.75")
        if len(order) <= MAX_LEGEND + 1:
            ax.legend(fontsize=5, markerscale=2, loc="best", framealpha=0.85)
        else:
            ax.legend(fontsize=5, markerscale=2, loc="best", framealpha=0.85,
                      title=f"top {MAX_LEGEND} of {len(order)}", title_fontsize=5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def embedding_panel(xy: np.ndarray, meta, path: Path, name: str) -> None:
    """One embedding, five colourings, one row."""
    plt = _plt()
    cat_pair = (meta["tool_cat"] + "/" + meta["target_cat"]).to_numpy()
    cols = {"action": meta["action"].to_numpy(), "tool_cat": meta["tool_cat"].to_numpy(),
            "target_cat": meta["target_cat"].to_numpy(), "cat_pair": cat_pair,
            "chunk_index": meta["chunk_index"].to_numpy()}
    fig, axes = plt.subplots(1, len(COLOURINGS), figsize=(4.1 * len(COLOURINGS), 4.3))
    for ax, key in zip(axes, COLOURINGS):
        _scatter(ax, xy, cols[key], f"{key}", continuous=(key == "chunk_index"))
    fig.suptitle(name, fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def confusion_heatmap(tab, path: Path, title: str) -> None:
    plt = _plt()
    m = tab.to_numpy(dtype=float)
    frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)   # row-normalised: per action
    fig, ax = plt.subplots(figsize=(1.2 + 0.4 * m.shape[1], 1.5 + 0.34 * m.shape[0]))
    im = ax.imshow(frac, cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(m.shape[1]), tab.columns, fontsize=7)
    ax.set_yticks(range(m.shape[0]), [f"{a} ({int(n)})" for a, n in zip(tab.index, m.sum(axis=1))],
                  fontsize=7)
    ax.set_xlabel("k-means cluster", fontsize=8); ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="share of the action's chunks")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def stat_by_group(df, stat: str, group: str, path: Path, min_n: int = 10) -> None:
    """Box plot of one descriptive statistic across the levels of one grouping."""
    import pandas as pd
    plt = _plt()

    data, labels = [], []
    for key, sub in df.groupby(group, sort=True):
        x = pd.to_numeric(sub[stat], errors="coerce").dropna().to_numpy()
        if len(x) >= min_n:
            data.append(x); labels.append(f"{key} ({len(x)})")
    if not data:
        return
    order = np.argsort([np.median(d) for d in data])
    data, labels = [data[i] for i in order], [labels[i] for i in order]
    fig, ax = plt.subplots(figsize=(7.5, max(2.6, 0.32 * len(data))))
    bp = ax.boxplot(data, orientation="horizontal", tick_labels=labels, showfliers=False,
                    patch_artist=True, medianprops={"color": "black"})
    for p in bp["boxes"]:
        p.set_facecolor("#8FA8D0")
    ax.set_xlabel(stat, fontsize=9); ax.set_title(f"{stat} by {group}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)


def stat_overview(df, stats: Sequence[str], path: Path) -> None:
    """Global distribution of every descriptive statistic, on a log-friendly axis."""
    import pandas as pd
    plt = _plt()

    stats = [s for s in stats if pd.to_numeric(df[s], errors="coerce").notna().any()]
    ncol = 4
    nrow = int(np.ceil(len(stats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.1 * nrow))
    for ax, s in zip(np.ravel(axes), stats):
        x = pd.to_numeric(df[s], errors="coerce").dropna()
        ax.hist(x, bins=50, color="#4C72B0")
        ax.set_title(s, fontsize=7); ax.tick_params(labelsize=6)
    for ax in np.ravel(axes)[len(stats):]:
        ax.axis("off")
    fig.suptitle("chunk-level descriptive statistics (all chunks)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def chunk_index_effect(df, stats: Sequence[str], path: Path) -> None:
    """Does a statistic depend on WHERE in the episode the chunk came from?

    A phase effect here is the alternative explanation for any clustering: chunk 0 of everything
    is a reach, chunk 2 of everything is a retreat, and a representation could be organising by
    that rather than by the action.
    """
    import pandas as pd
    plt = _plt()

    fig, axes = plt.subplots(1, min(4, len(stats)), figsize=(3.6 * min(4, len(stats)), 3.0))
    for ax, s in zip(np.ravel([axes]), stats[:4]):
        g = df.assign(k=df["chunk_index"].clip(upper=4)).groupby("k")[s]
        med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(med.index, q1, q3, alpha=0.25, color="#4C72B0")
        ax.plot(med.index, med, "o-", color="#22314E")
        ax.set_xlabel("chunk index in episode (4 = 4+)", fontsize=8)
        ax.set_title(s, fontsize=8); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=125); plt.close(fig)
