"""SANITY backend: `aligned` with the correspondence destroyed and nothing else changed.

The K entries of every canonical vector are shuffled by a random permutation drawn with seed 0.
The permutation is drawn PER MESH (one generator per category, seeded 0, one draw per mesh in
sorted-id order) and not once per category, for a reason worth stating: a single permutation
applied to every mesh of a category is only a relabelling of the K indices -- index k would still
mean the same canonical point on every mesh, and every index-matched comparison across meshes
would come out identical to `aligned`. Only when the shuffle differs between meshes does "index k
on mesh A" stop meaning "index k on mesh B", which is the null hypothesis this backend exists to
represent: the same alignment, the same coverage, the same value distribution, but chance-level
agreement across instances.

The point map (`to_canonical` / `from_canonical`) is untouched -- it is the vector index that is
scrambled, so `canonical_points` still returns the template points and, for a given mesh, entry k
of the vector belongs to canonical point `permutation(mesh)[k]`.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.analysis.canonical.backends._similarity import CategoryFit
from src.analysis.canonical.backends.aligned import AlignedBackend
from src.analysis.canonical.interface import DEFAULT_RADIUS, N_CANONICAL

SEED = 0


class RandomPermBackend(AlignedBackend):
    """`aligned` whose canonical vectors are index-shuffled per mesh. Chance level."""

    name = "random_perm"

    def __init__(self, n_sample: int = 2000, k: int = N_CANONICAL, icp_iters: int = 30,
                 seed: int = SEED):
        super().__init__(n_sample=n_sample, k=k, icp_iters=icp_iters)
        self.seed = int(seed)
        self._perms: Dict[str, Dict[str, np.ndarray]] = {}

    @classmethod
    def from_aligned(cls, aligned: AlignedBackend, seed: int = SEED) -> "RandomPermBackend":
        """Wrap an already fitted `aligned` backend (no refitting)."""
        self = cls(n_sample=aligned.n_sample, k=aligned.k, icp_iters=aligned.icp_iters, seed=seed)
        for cat, fit in aligned._cats.items():
            self._cats[cat] = fit
            self._perms[cat] = self._draw(fit)
        return self

    def _draw(self, fit: CategoryFit) -> Dict[str, np.ndarray]:
        rng = np.random.default_rng(self.seed)
        return {i: rng.permutation(len(fit.canonical_points)) for i in fit.mesh_ids}

    def _fit_category(self, category: str, verts: Dict[str, np.ndarray]) -> CategoryFit:
        fit = super()._fit_category(category, verts)
        self._perms[category] = self._draw(fit)
        return fit

    def permutation(self, category: str, mesh_id: str) -> np.ndarray:
        """(K,) canonical-point index that entry k of this mesh's vector actually holds."""
        return self._perms[category][mesh_id]

    def splat(self, category: str, mesh_id: str, per_vertex_values: np.ndarray,
              kernel: str = "nearest", radius: float = DEFAULT_RADIUS,
              sigma: Optional[float] = None, return_coverage: bool = False):
        out = super().splat(category, mesh_id, per_vertex_values, kernel, radius, sigma, True)
        vec, covered = out
        p = self.permutation(category, mesh_id)
        return (vec[p], covered[p]) if return_coverage else vec[p]

    # ---- persistence: base arrays plus the permutations, so the npz stands alone -------------
    def _extra_arrays(self, fit: CategoryFit):
        return {"perms": np.stack([self._perms[fit.category][i] for i in fit.mesh_ids]),
                "seed": np.array(self.seed)}

    def _restore(self, fit: CategoryFit, z) -> None:
        self.seed = int(z["seed"])
        self._perms[fit.category] = {i: z["perms"][n] for n, i in enumerate(fit.mesh_ids)}
