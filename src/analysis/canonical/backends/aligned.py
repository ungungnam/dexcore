"""The "method" backend -- STILL GEOMETRIC, still a baseline. Per category:

    template   the medoid: after aligning every mesh onto every other, the mesh with the smallest
               mean symmetric Chamfer from the others. The canonical frame is that mesh, centred,
               scaled to max radius 1 and rotated onto its own PCA axes (axis 0 = longest extent),
               which is why the template's transform is nothing but its centre/scale/PCA and its
               ICP correction is the identity.
    per mesh   centre at the centroid, scale to max radius 1, rotate onto PCA axes, then resolve
               the axis-sign ambiguity by trying the four PROPER rotations (pairs of axes flipped;
               a single flip would mirror the object) and keeping the one with the lowest
               symmetric Chamfer to the template, then refine with similarity ICP (rigid +
               isotropic scale). The composed map is stored as one (R, t, s, c) so
               to_canonical / from_canonical are exact inverses.

Why this is not correspondence: ICP overlaps SURFACES. A spatula whose blade is 20% longer than
the template's lands its blade tip on empty space or on the template's blade middle; nothing tells
canonical point k that it should mean "tip". The per-mesh Chamfer after ICP, exposed through
`quality`, is the honest measure of how far spatial overlap gets -- read it before trusting any
index-matched comparison across meshes.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.canonical.backends._similarity import CategoryFit, SimilarityBackend
from src.analysis.canonical.geometry import (PROPER_FLIPS, Similarity, centre_scale, chamfer,
                                             compose, icp_similarity, pca_axes)
from src.analysis.canonical.interface import N_CANONICAL

log = logging.getLogger(__name__)


def align_to_template(P: np.ndarray, T: np.ndarray, tree_t: cKDTree,
                      icp_iters: int = 30) -> Tuple[Similarity, float, float]:
    """P (mesh, in its own PCA frame) onto T (template, canonical frame).
    -> (Similarity about the origin, Chamfer after the best proper flip, Chamfer after ICP)."""
    best_flip, best_ch = None, np.inf
    for F in PROPER_FLIPS:
        ch = chamfer(P @ F, T, tree_b=tree_t)
        if ch < best_ch:
            best_flip, best_ch = F, ch
    Q = P @ best_flip
    icp = icp_similarity(Q, T, tree_t, iters=icp_iters)
    ch_icp = chamfer(icp.apply(Q), T, tree_b=tree_t)
    if ch_icp > best_ch:             # ICP is a local method; never accept a step backwards
        icp, ch_icp = Similarity.identity(), best_ch
    # q = icp(F p): rotation F then the ICP similarity, as one map about the origin
    return Similarity(icp.R @ best_flip, icp.t, icp.s, np.zeros(3)), best_ch, ch_icp


class AlignedBackend(SimilarityBackend):
    """Medoid template + PCA + proper-flip search + similarity ICP. Geometric; not semantic."""

    name = "aligned"

    def __init__(self, n_sample: int = 2000, k: int = N_CANONICAL, icp_iters: int = 30):
        super().__init__(n_sample=n_sample, k=k)
        self.icp_iters = int(icp_iters)

    def _fit_category(self, category: str, verts: Dict[str, np.ndarray]) -> CategoryFit:
        ids = sorted(verts)
        M = len(ids)
        # stage 1: centre + scale, then PCA rotation, per mesh (independent of any template)
        plain = {i: centre_scale(verts[i]) for i in ids}
        base = {}
        for i in ids:
            axes = pca_axes(plain[i].apply(verts[i]))
            base[i] = Similarity(axes, np.zeros(3), plain[i].s, plain[i].c)
        samples = self._samples(verts)
        s_plain = {i: plain[i].apply(samples[i]) for i in ids}
        s_pca = {i: base[i].apply(samples[i]) for i in ids}
        # stage 2: align every mesh onto every candidate template; [i, j] = mesh i onto template j
        pair_plain = self._pairwise_chamfer(s_plain, ids)
        pair_pca = np.zeros((M, M)); pair_icp = np.zeros((M, M))
        refine: Dict[Tuple[str, str], Similarity] = {}
        for b, j in enumerate(ids):
            tree_j = cKDTree(s_pca[j])
            for a, i in enumerate(ids):
                if i == j:
                    continue
                sim, ch_pca, ch_icp = align_to_template(s_pca[i], s_pca[j], tree_j, self.icp_iters)
                refine[(i, j)] = sim
                pair_pca[a, b], pair_icp[a, b] = ch_pca, ch_icp
        # stage 3: the medoid under the FULL alignment is the template
        mean_other = pair_icp.sum(0) / max(M - 1, 1)
        tj = int(np.argmin(mean_other))
        template = ids[tj]
        sims, quality = {}, {}
        for a, i in enumerate(ids):
            if i == template:
                sims[i] = base[i]
                quality[i] = {"chamfer_centred": 0.0, "chamfer_pca": 0.0, "chamfer_icp": 0.0}
            else:
                sims[i] = compose(refine[(i, template)], base[i])
                quality[i] = {"chamfer_centred": float(pair_plain[a, tj]),
                              "chamfer_pca": float(pair_pca[a, tj]),
                              "chamfer_icp": float(pair_icp[a, tj])}
        log.info("%-12s %2d meshes  template %s  chamfer centred %.4f -> pca %.4f -> icp %.4f",
                 category, M, template,
                 *[np.nan if M == 1 else float(np.mean([quality[i][k] for i in ids if i != template]))
                   for k in ("chamfer_centred", "chamfer_pca", "chamfer_icp")])
        return CategoryFit(category, ids, sims, template,
                           self._template_points(verts[template], sims[template]), verts, quality,
                           {"chamfer_centred": pair_plain, "chamfer_pca": pair_pca,
                            "chamfer_icp": pair_icp})
