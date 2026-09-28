"""Basis point set encoding of object SHAPE, as a representation the hand can be predicted from.

WHY SHAPE ENTERS HERE. The trajectory study deliberately excluded geometry, and its result was that
the wrist is largely predictable from the tool's orientation while the FINGERS are not predictable
at all -- under a geometry-controlled comparison every trajectory representation scored worse than
predicting the dataset mean. Shape is the obvious missing cause: what the fingers do is set by what
they are wrapped around. BPS (Prokudin et al. 2019) is the cheapest honest way to put it in.

THE ENCODING. A fixed basis of points is drawn once inside a ball and reused for every object; the
code is the distance from each basis point to the nearest point of the object's surface, clipped at
the ball radius. Fixed basis, fixed length, no training, no correspondence -- two objects are
comparable because they were measured against the same ruler.

THE BALL IS METRIC AND THE OBJECT IS NOT RESCALED. A 12 cm spoon and a 30 cm kettle must not encode
alike: how a hand wraps an object depends on how big it actually is. The basis sits in the object's
OWN frame -- the frame its recorded pose is applied to, and the same frame the wrist target is
expressed in -- so the code describes shape as the hand meets it, with pose already factored out.

One code per MESH, not per chunk: the release has 130 tool and 81 target meshes against 4562 chunks,
so the encoding is computed once per mesh and looked up.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

#: Basis size. 256 is enough to separate these objects and keeps the code comparable in width to
#: the trajectory representations rather than swamping them.
N_BASIS = 256
#: Ball radius in metres. Covers the largest TACO tool with room to spare; distances clip here, so
#: a basis point far from every surface reads as "nothing nearby" instead of an unbounded number.
BALL_RADIUS_M = 0.20
#: Surface samples per mesh for the nearest-point query.
N_SURFACE = 20000
BASIS_SEED = 0


def basis(n: int = N_BASIS, radius: float = BALL_RADIUS_M, seed: int = BASIS_SEED) -> np.ndarray:
    """(n,3) points uniform in a ball. Drawn once and shared by every object, which is the whole
    point of the method -- a code only means something against a fixed ruler."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = radius * rng.random(n) ** (1.0 / 3.0)        # uniform by volume, not by radius
    return v * r[:, None]


def encode_mesh(mesh_path: str, scale: float, B: Optional[np.ndarray] = None,
                n_surface: int = N_SURFACE, radius: float = BALL_RADIUS_M,
                seed: int = BASIS_SEED) -> np.ndarray:
    """(n_basis,) distances from the basis to the mesh surface, in the mesh's own frame, metres."""
    import trimesh
    from scipy.spatial import cKDTree

    B = basis(seed=seed) if B is None else B
    mesh = trimesh.load(mesh_path, process=False, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    surf = np.asarray(pts, dtype=np.float64) * scale
    d, _ = cKDTree(surf).query(B, k=1, workers=-1)
    return np.minimum(d, radius)


def encode_dataset(tracks: Iterable, n_basis: int = N_BASIS, radius: float = BALL_RADIUS_M,
                   seed: int = BASIS_SEED) -> Dict[str, np.ndarray]:
    """mesh id -> code, for every distinct mesh among `tracks` (objects with a resolved mesh)."""
    B = basis(n_basis, radius, seed)
    seen: Dict[str, np.ndarray] = {}
    for track in tracks:
        if track is None or not track.mesh_path or track.name in seen:
            continue
        try:
            seen[track.name] = encode_mesh(track.mesh_path, track.mesh_scale, B, radius=radius,
                                           seed=seed)
        except Exception as e:                      # noqa: BLE001 -- reported, then skipped
            log.warning("mesh %s: %s: %s", track.name, type(e).__name__, e)
    return seen


def build_codebook(root=None, refs: Optional[Sequence] = None, **kw) -> Dict[str, np.ndarray]:
    """Encode every mesh the TACO index refers to, tool and target alike."""
    from src.analysis.loaders import taco

    root = Path(root or taco.default_root())
    refs = list(refs if refs is not None else taco.index(root))
    tracks, seen = [], set()
    for ref in refs:
        traj = taco.load(ref, root=root, with_hands=False)
        for t in (traj.tool, traj.target):
            if t is not None and t.name not in seen:
                seen.add(t.name)
                tracks.append(t)
    log.info("encoding %d distinct meshes", len(tracks))
    return encode_dataset(tracks, **kw)


def chunk_codes(meta, codebook: Dict[str, np.ndarray], which: str = "tool") -> np.ndarray:
    """(N, n_basis) per chunk, by looking its mesh id up. Missing meshes take the codebook mean."""
    col = f"{which}_mesh"
    dim = len(next(iter(codebook.values())))
    fallback = np.mean(list(codebook.values()), axis=0)
    return np.stack([codebook.get(str(m), fallback) for m in meta[col]])
