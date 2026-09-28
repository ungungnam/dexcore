"""BASELINE B: centre at the centroid, isotropic scale so the farthest vertex sits at radius 1,
NO rotation. Each mesh stays in the orientation its scan frame happens to give it.

This is the weakest map that still yields a fixed-length vector, and it is here to be beaten: two
cups whose scans were oriented differently will overlap only where their unit spheres happen to,
and the canonical index says nothing about the cup beyond "somewhere in the unit ball". It is
exactly invertible. The category template (for the K canonical points) is the medoid under this
same map -- the mesh with the smallest mean symmetric Chamfer to the others after centring and
scaling -- so the baseline is judged on its own terms.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.analysis.canonical.backends._similarity import CategoryFit, SimilarityBackend
from src.analysis.canonical.geometry import centre_scale


class NormalizedBackend(SimilarityBackend):
    """Geometric baseline: centroid + unit-radius scale, no rotation. Not a correspondence."""

    name = "normalized"

    def _fit_category(self, category: str, verts: Dict[str, np.ndarray]) -> CategoryFit:
        ids = sorted(verts)
        sims = {i: centre_scale(verts[i]) for i in ids}
        samples = self._samples(verts)
        mapped = {i: sims[i].apply(samples[i]) for i in ids}
        pair = self._pairwise_chamfer(mapped, ids)
        # medoid: smallest mean Chamfer to the OTHER meshes (a 1-mesh category is its own template)
        mean_other = pair.sum(0) / max(len(ids) - 1, 1)
        template = ids[int(np.argmin(mean_other))]
        tj = ids.index(template)
        quality = {i: {"chamfer_centred": float(pair[n, tj]), "chamfer_pca": np.nan,
                       "chamfer_icp": np.nan} for n, i in enumerate(ids)}
        return CategoryFit(category, ids, sims, template,
                           self._template_points(verts[template], sims[template]),
                           verts, quality, {"chamfer_centred": pair})
