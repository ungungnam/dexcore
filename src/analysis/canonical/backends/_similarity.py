"""Shared machinery for backends whose per-mesh map is a similarity transform.

Both geometric backends (and the permuted sanity wrapper) store the same thing per category: one
`Similarity` per mesh, one template mesh, K canonical points on that template, the object-frame
vertices (kept so `splat` needs only the per-vertex values), and the alignment-quality numbers.
Subclasses differ only in how `_fit_category` produces that state, so everything else -- the
map, the inverse, the splat, the npz round trip -- lives here once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.canonical.geometry import Similarity, chamfer, fps_indices
from src.analysis.canonical.interface import (DEFAULT_RADIUS, N_CANONICAL, CanonicalBackend,
                                              MeshDict)

#: Quality columns stored per mesh. NaN where a stage does not exist for the backend.
QUALITY_KEYS = ("chamfer_centred", "chamfer_pca", "chamfer_icp")


@dataclass
class CategoryFit:
    category: str
    mesh_ids: List[str]
    sims: Dict[str, Similarity]
    template: str
    canonical_points: np.ndarray                      # (K, 3)
    verts: Dict[str, np.ndarray]                      # object frame, metres
    quality: Dict[str, Dict[str, float]]              # mesh_id -> {QUALITY_KEYS}
    pairwise: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (M, M), [i, j]: i onto j
    _trees: Dict[str, Tuple[np.ndarray, cKDTree]] = field(default_factory=dict, repr=False)

    def mapped(self, mesh_id: str) -> Tuple[np.ndarray, cKDTree]:
        """Instance vertices in the canonical frame and a KD-tree over them (cached)."""
        if mesh_id not in self._trees:
            V = self.sims[mesh_id].apply(self.verts[mesh_id])
            self._trees[mesh_id] = (V, cKDTree(V))
        return self._trees[mesh_id]

    # ---- npz round trip ---------------------------------------------------------------------
    def to_arrays(self) -> Dict[str, np.ndarray]:
        ids = self.mesh_ids
        offsets = np.cumsum([0] + [len(self.verts[i]) for i in ids])
        out = {
            "category": np.array(self.category), "mesh_ids": np.array(ids), "template": np.array(self.template),
            "R": np.stack([self.sims[i].R for i in ids]), "t": np.stack([self.sims[i].t for i in ids]),
            "s": np.array([self.sims[i].s for i in ids]), "c": np.stack([self.sims[i].c for i in ids]),
            "canonical_points": self.canonical_points,
            "verts_all": np.concatenate([self.verts[i] for i in ids]), "verts_offsets": offsets,
        }
        for k in QUALITY_KEYS:
            out[k] = np.array([self.quality[i].get(k, np.nan) for i in ids])
        for k, v in self.pairwise.items():
            out["pairwise_" + k] = v
        return out

    @classmethod
    def from_arrays(cls, z) -> "CategoryFit":
        ids = [str(i) for i in z["mesh_ids"]]
        off = z["verts_offsets"]
        sims = {i: Similarity(z["R"][n], z["t"][n], float(z["s"][n]), z["c"][n]) for n, i in enumerate(ids)}
        verts = {i: z["verts_all"][off[n]:off[n + 1]] for n, i in enumerate(ids)}
        quality = {i: {k: float(z[k][n]) for k in QUALITY_KEYS if k in z} for n, i in enumerate(ids)}
        pairwise = {k[len("pairwise_"):]: z[k] for k in z.files if k.startswith("pairwise_")}
        return cls(str(z["category"]), ids, sims, str(z["template"]), z["canonical_points"],
                   verts, quality, pairwise)


def slug(category: str) -> str:
    return category.replace(" ", "_")


def splat_values(tree: cKDTree, vals: np.ndarray, P: np.ndarray, kernel: str = "nearest",
                 radius: float = DEFAULT_RADIUS, sigma: Optional[float] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """The splat kernel every backend shares: per-vertex `vals` on the mapped vertices held by
    `tree` (canonical frame) -> values on the K canonical points `P`, plus the coverage mask.
    See `CanonicalBackend.splat` for the kernel semantics."""
    out = np.zeros(len(P))
    if kernel == "nearest":
        d, idx = tree.query(P)
        covered = d <= radius
        out[covered] = vals[idx[covered]]
    elif kernel == "gauss":
        sig = float(sigma) if sigma is not None else radius / 2.0
        covered = np.zeros(len(P), dtype=bool)
        for k, nb in enumerate(tree.query_ball_point(P, radius)):
            if not nb:
                continue
            nb = np.asarray(nb)
            w = np.exp(-0.5 * ((tree.data[nb] - P[k]) ** 2).sum(1) / sig ** 2)
            out[k] = float((w * vals[nb]).sum() / w.sum())
            covered[k] = True
    else:
        raise ValueError(f"kernel must be 'nearest' or 'gauss', got {kernel!r}")
    return out, covered


class SimilarityBackend(CanonicalBackend):
    """Backend whose per-mesh map is one `Similarity`. Subclasses implement `_fit_category`."""

    def __init__(self, n_sample: int = 2000, k: int = N_CANONICAL):
        self.n_sample = int(n_sample)
        self.k = int(k)
        self._cats: Dict[str, CategoryFit] = {}

    # ---- subclass hook ----------------------------------------------------------------------
    def _fit_category(self, category: str, verts: Dict[str, np.ndarray]) -> CategoryFit:
        raise NotImplementedError

    # ---- CanonicalBackend -------------------------------------------------------------------
    def fit(self, category: str, meshes: MeshDict) -> None:
        verts = {str(i): np.asarray(v, dtype=np.float64) for i, (v, _f) in sorted(meshes.items())}
        self._cats[category] = self._fit_category(category, verts)

    def categories(self) -> List[str]:
        return sorted(self._cats)

    def mesh_ids(self, category: str) -> List[str]:
        return list(self._cat(category).mesh_ids)

    def template(self, category: str) -> str:
        return self._cat(category).template

    def transform(self, category: str, mesh_id: str) -> Similarity:
        return self._cat(category).sims[mesh_id]

    def quality(self, category: str) -> Dict[str, Dict[str, float]]:
        """mesh_id -> Chamfer-to-template after each stage (canonical units, template excluded
        from means by callers since its own value is 0 by construction)."""
        return {i: dict(q) for i, q in self._cat(category).quality.items()}

    def to_canonical(self, category: str, mesh_id: str, points: np.ndarray) -> np.ndarray:
        return self._cat(category).sims[mesh_id].apply(points)

    def from_canonical(self, category: str, mesh_id: str, coords: np.ndarray) -> np.ndarray:
        return self._cat(category).sims[mesh_id].invert(coords)

    def canonical_points(self, category: str) -> np.ndarray:
        return self._cat(category).canonical_points

    def splat(self, category: str, mesh_id: str, per_vertex_values: np.ndarray,
              kernel: str = "nearest", radius: float = DEFAULT_RADIUS,
              sigma: Optional[float] = None, return_coverage: bool = False):
        fit = self._cat(category)
        vals = np.asarray(per_vertex_values, dtype=np.float64)
        if vals.shape != (len(fit.verts[mesh_id]),):
            raise ValueError(f"expected ({len(fit.verts[mesh_id])},) values for mesh {mesh_id}, "
                             f"got {vals.shape}")
        _V, tree = fit.mapped(mesh_id)
        out, covered = splat_values(tree, vals, fit.canonical_points, kernel, radius, sigma)
        return (out, covered) if return_coverage else out

    def coverage(self, category: str, mesh_id: str, radius: float = DEFAULT_RADIUS) -> float:
        """Fraction of canonical points within `radius` of some mapped vertex of the mesh."""
        _V, tree = self._cat(category).mapped(mesh_id)
        return float((tree.query(self._cat(category).canonical_points)[0] <= radius).mean())

    def roundtrip_error(self, category: str, mesh_id: str) -> float:
        V = self._cat(category).verts[mesh_id]
        return float(np.abs(self.from_canonical(category, mesh_id,
                                                self.to_canonical(category, mesh_id, V)) - V).max())

    # ---- persistence ------------------------------------------------------------------------
    def _extra_arrays(self, fit: CategoryFit) -> Dict[str, np.ndarray]:
        return {}

    def _restore(self, fit: CategoryFit, z) -> None:
        pass

    def save(self, directory: Union[str, Path]) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        summary = {}
        for cat, fit in self._cats.items():
            arrays = fit.to_arrays()
            arrays.update(self._extra_arrays(fit))
            np.savez_compressed(directory / f"{slug(cat)}.npz", **arrays)
            summary[cat] = {"template": fit.template, "n_meshes": len(fit.mesh_ids),
                            "meshes": {i: {k: (None if np.isnan(v) else float(v))
                                           for k, v in fit.quality[i].items()}
                                       for i in fit.mesh_ids}}
        (directory / "templates.json").write_text(json.dumps(
            {"backend": self.name, "n_sample": self.n_sample, "k": self.k, "categories": summary},
            indent=1))
        return directory

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "SimilarityBackend":
        directory = Path(directory)
        meta = {}
        if (directory / "templates.json").exists():
            meta = json.loads((directory / "templates.json").read_text())
        self = cls(n_sample=meta.get("n_sample", 2000), k=meta.get("k", N_CANONICAL))
        for path in sorted(directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                fit = CategoryFit.from_arrays(z)
                self._restore(fit, z)
            self._cats[fit.category] = fit
        return self

    # ---- helpers for subclasses -------------------------------------------------------------
    def _cat(self, category: str) -> CategoryFit:
        try:
            return self._cats[category]
        except KeyError:
            raise KeyError(f"{self.name}: category {category!r} not fitted; "
                           f"have {self.categories()}") from None

    def _samples(self, verts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """~n_sample farthest-point-sampled vertices per mesh, object frame."""
        return {i: V[fps_indices(V, self.n_sample)] for i, V in verts.items()}

    def _template_points(self, fit_verts: np.ndarray, sim: Similarity) -> np.ndarray:
        """K canonical points: FPS on the template's full vertex set in the canonical frame."""
        V = sim.apply(fit_verts)
        return V[fps_indices(V, self.k)]

    @staticmethod
    def _pairwise_chamfer(mapped_samples: Dict[str, np.ndarray], ids: List[str]) -> np.ndarray:
        trees = {i: cKDTree(mapped_samples[i]) for i in ids}
        M = np.zeros((len(ids), len(ids)))
        for a, i in enumerate(ids):
            for b, j in enumerate(ids):
                if a < b:
                    M[a, b] = M[b, a] = chamfer(mapped_samples[i], mapped_samples[j], trees[i], trees[j])
        return M
