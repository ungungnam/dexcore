"""The contract every canonical-mapping backend fulfils, and the two module-level entry points
the study spec names (`object_to_canonical`, `canonical_to_object`).

A backend owns, per category, (1) a map from each mesh's own object frame into a shared canonical
frame and back, (2) a FIXED set of K canonical surface points every mesh of the category is read
at, and (3) `splat`, which turns a per-vertex value on one mesh instance into a K-vector over
those points. (3) is the whole reason the package exists: contact on mesh A (N_A vertices) and on
mesh B (N_B vertices, different topology) become two vectors of the same length whose index k
means "the same place on the category template" -- to whatever extent the backend's map makes
that true. For the geometric baselines shipped here it is true only up to spatial overlap after
alignment; nothing in them knows what a handle is.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

#: Canonical points per category. Matches the 512-point BPS / FPS budgets used elsewhere so a
#: canonical contact vector is the same width as the per-object contact label it stands in for.
N_CANONICAL = 512
#: Default splat radius in canonical units (template scaled to max radius 1). 512 points spread
#: over a unit sphere sit ~0.16 apart, so a canonical point normally sees a few instance
#: vertices and is left uncovered only where the instance has no surface near the template's.
DEFAULT_RADIUS = 0.15

MeshDict = Dict[str, Tuple[np.ndarray, np.ndarray]]      # mesh_id -> (verts (N,3), faces (F,3))


class CanonicalBackend(ABC):
    """Per-category object-frame <-> canonical-frame mapping with a fixed canonical point set."""

    name: str = "abstract"

    # ---- fitting -------------------------------------------------------------------------
    @abstractmethod
    def fit(self, category: str, meshes: MeshDict) -> None:
        """Fit the category from its meshes (object-frame vertices in metres). Replaces any
        earlier fit of the same category."""

    @abstractmethod
    def categories(self) -> List[str]:
        """Categories this backend has been fitted (or loaded) for."""

    @abstractmethod
    def mesh_ids(self, category: str) -> List[str]:
        """Mesh ids the category was fitted with, in a stable order."""

    # ---- the map ---------------------------------------------------------------------------
    @abstractmethod
    def to_canonical(self, category: str, mesh_id: str, points: np.ndarray) -> np.ndarray:
        """(N,3) object-frame points -> (N,3) canonical-frame points."""

    @abstractmethod
    def from_canonical(self, category: str, mesh_id: str, coords: np.ndarray) -> np.ndarray:
        """(N,3) canonical-frame points -> (N,3) object frame. Exact inverse of `to_canonical`
        for every backend here (they are all similarity transforms)."""

    @abstractmethod
    def canonical_points(self, category: str) -> np.ndarray:
        """(K,3) canonical surface points shared by every mesh of the category."""

    # ---- the fixed-dimensional vector ------------------------------------------------------
    @abstractmethod
    def splat(self, category: str, mesh_id: str, per_vertex_values: np.ndarray,
              kernel: str = "nearest", radius: float = DEFAULT_RADIUS,
              sigma: Union[float, None] = None,
              return_coverage: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """(N,) values on the instance's vertices -> (K,) values on the canonical points.

        'nearest': each canonical point takes the value of the nearest mapped instance vertex.
        'gauss':   Gaussian-weighted average (std `sigma`, default radius/2) over mapped vertices
                   within `radius`.
        A canonical point with no instance vertex within `radius` gets 0 and is flagged False in
        the coverage mask, returned alongside the vector when `return_coverage` is set.
        """

    # ---- persistence -------------------------------------------------------------------------
    @abstractmethod
    def save(self, directory: Union[str, Path]) -> Path:
        """One `<category>.npz` per fitted category under `directory`."""

    @classmethod
    @abstractmethod
    def load(cls, directory: Union[str, Path]) -> "CanonicalBackend":
        """Inverse of `save`."""


def object_to_canonical(category: str, mesh_id: str, points: np.ndarray,
                        backend: CanonicalBackend) -> np.ndarray:
    """(N,3) object-frame points of `mesh_id` -> (N,3) canonical coordinates of `category`."""
    return backend.to_canonical(category, mesh_id, points)


def canonical_to_object(category: str, mesh_id: str, coords: np.ndarray,
                        backend: CanonicalBackend) -> np.ndarray:
    """(N,3) canonical coordinates -> (N,3) object frame of `mesh_id`, where the backend's map
    is invertible (all backends in this package)."""
    return backend.from_canonical(category, mesh_id, coords)
