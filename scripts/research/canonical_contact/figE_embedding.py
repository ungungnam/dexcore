"""Figure E: 2D embeddings of window-level X_soft (touch_frac >= 0.2) for the six study
(category, role, hand) triples x backends {normalized, aligned, dino}. PCA to 50 dims, then UMAP
(n_neighbors 15, min_dist 0.1, seed 0); plain PCA-2 as an inset. Each panel pair: coloured by VERB
and by MESH ID, with silhouette scores (sklearn, on the 50-d PCA) in the titles. The SAME <= 1500
sub-sample (seed 0) is used for every backend of a triple, so panels are comparable.
Outputs figures/figE_<cat>_<role>_<hand>_<backend>.png, figE_all.png, figE_silhouettes.csv.
Run: python scripts/figE_embedding.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import BACKENDS_CANON, FIG, SIX, load_cache

MAX_N, MIN_TOUCH, SEED = 1500, 0.2, 0
# fixed categorical hue order (tab20 without the pale pairs first); assigned by sorted label, never cycled
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
           "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"]


def colours(labels):
    u = sorted(set(labels))
    return u, {l: PALETTE[i % len(PALETTE)] for i, l in enumerate(u)}


def embed(X):
    n_comp = min(50, X.shape[0] - 1, X.shape[1])
    Z50 = PCA(n_components=n_comp, random_state=SEED).fit_transform(X)
    Z2 = Z50[:, :2]
    U = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=SEED).fit_transform(Z50)
    return Z50, Z2, U


def sil(Z, labels):
    labels = np.asarray(labels)
    if len(set(labels)) < 2 or len(set(labels)) >= len(labels):
        return np.nan
    return float(silhouette_score(Z, labels))


def scatter(ax, U, Z2, labels, title, legend_max=12):
    u, cmap = colours(labels)
    labels = np.asarray(labels)
    for l in u:
        m = labels == l
        ax.scatter(U[m, 0], U[m, 1], s=7, c=cmap[l], alpha=0.75, lw=0, label=f"{l} ({m.sum()})")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#bbb")
    ins = ax.inset_axes([0.72, 0.02, 0.27, 0.27])
    for l in u:
        m = labels == l
        ins.scatter(Z2[m, 0], Z2[m, 1], s=2, c=cmap[l], alpha=0.7, lw=0)
    ins.set_xticks([]); ins.set_yticks([]); ins.set_title("PCA-2", fontsize=6, pad=1)
    ins.patch.set_alpha(0.85)
    if len(u) <= legend_max:
        ax.legend(fontsize=6, markerscale=1.5, loc="upper left", frameon=True, framealpha=0.8,
                  handletextpad=0.2, borderpad=0.3, labelspacing=0.2)
    else:
        ax.text(0.01, 0.99, f"{len(u)} classes", transform=ax.transAxes, fontsize=7, va="top")


def main():
    rng = np.random.default_rng(SEED)
    rows, all_panels = [], []
    for cat, role, hand in SIX:
        caches = {be: load_cache(be, cat, role) for be in BACKENDS_CANON}
        ref = caches["aligned"]
        key = lambda z: list(zip(z["sequence_id"], z["window"], z["hand"]))
        for be in BACKENDS_CANON:
            assert key(caches[be]) == key(ref), (cat, role, be, "cache rows differ between backends")
        keep = np.where((ref["touch_frac"] >= MIN_TOUCH) & (ref["hand"] == hand))[0]
        if len(keep) > MAX_N:
            keep = np.sort(rng.choice(keep, MAX_N, replace=False))
        verbs = ref["verb"][keep].astype(str)
        meshes = ref["mesh_id"][keep].astype(str)
        panels = {}
        for be in BACKENDS_CANON:
            X = caches[be]["X_soft"][keep].astype(np.float64)
            Z50, Z2, U = embed(X)
            sv, sm = sil(Z50, verbs), sil(Z50, meshes)
            rows.append(dict(category=cat, role=role, hand=hand, backend=be, n=len(keep),
                             n_verbs=len(set(verbs)), n_meshes=len(set(meshes)),
                             silhouette_verb=sv, silhouette_mesh=sm))
            panels[be] = (U, Z2, sv, sm)
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            scatter(axes[0], U, Z2, verbs, f"{cat}/{role} ({hand}) {be}: colour = VERB   silhouette(verb) = {sv:.3f}")
            scatter(axes[1], U, Z2, meshes, f"{cat}/{role} ({hand}) {be}: colour = MESH ID   silhouette(mesh) = {sm:.3f}")
            fig.suptitle(f"Figure E: UMAP of PCA-50 of X_soft, window level, touch_frac>={MIN_TOUCH}, n={len(keep)} "
                         f"(inset: PCA-2; silhouettes on PCA-50)", fontsize=9)
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(FIG / f"figE_{cat}_{role}_{hand}_{be}.png", dpi=130)
            plt.close(fig)
            print(cat, role, hand, be, f"n={len(keep)} sil_verb={sv:.3f} sil_mesh={sm:.3f}")
        all_panels.append((cat, role, hand, verbs, meshes, panels))
    # combined figure: one row per triple, columns = backend x {verb, mesh}
    nb = len(BACKENDS_CANON)
    fig, axes = plt.subplots(len(SIX), 2 * nb, figsize=(3.4 * 2 * nb, 3.3 * len(SIX)))
    for r, (cat, role, hand, verbs, meshes, panels) in enumerate(all_panels):
        for b, be in enumerate(BACKENDS_CANON):
            U, Z2, sv, sm = panels[be]
            scatter(axes[r, 2 * b], U, Z2, verbs, f"{cat}/{role}({hand}) {be} | verb  s={sv:.2f}", legend_max=8)
            scatter(axes[r, 2 * b + 1], U, Z2, meshes, f"{cat}/{role}({hand}) {be} | mesh  s={sm:.2f}", legend_max=0)
    fig.suptitle("Figure E: UMAP(PCA-50) of window-level X_soft; s = silhouette on PCA-50 by verb / by mesh id "
                 "(same sub-sample per row)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "figE_all.png", dpi=110)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.to_csv(FIG / "figE_silhouettes.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
