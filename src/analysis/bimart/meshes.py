"""TACO object meshes in the form BimArt's preprocessing expects.

BimArt keeps two mesh dictionaries per object: one at the ORIGINAL metric scale, used for contact
distances, and one normalised so the farthest vertex sits at radius 0.85, used for the BPS features.
The normalisation factor is not thrown away -- it rides along in the global state vector, so the
network still knows a 12 cm spoon from a 40 cm kettle even though both fill the same unit sphere.
`no_recenter` in the asset's name is literal: the centroid is left where the mesh frame puts it,
because that frame is the one the recorded poses are applied to.

TWO DIFFERENCES FROM ARCTIC, both forced by the data:

    keyed by MESH, not by category   ARCTIC has 11 object templates, one per category. TACO has 206
                                     mesh instances across 26 categories, and two spoons are not
                                     interchangeable, so the dictionary is keyed by mesh id.
    decimated on the way in          ARCTIC templates ship at 2-4k vertices. TACO ships 10k-100k
                                     faces, which would make the per-frame nearest-point queries
                                     dominate preprocessing for no gain in a 512-point BPS.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

#: Radius the farthest vertex is scaled to. BimArt's own value; kept so the pretrained
#: normalisation statistics and the BPS basis stay meaningful.
UNIT_SPHERE_RADIUS = 0.85
#: Target face count after decimation, chosen to land near ARCTIC's 2-4k vertices.
TARGET_FACES = 8000


def normalise(verts: np.ndarray, radius: float = UNIT_SPHERE_RADIUS):
    """-> (scaled verts, scale). Scales about the MESH ORIGIN, not the centroid."""
    r = float(np.linalg.norm(verts, axis=1).max())
    scale = radius / max(r, 1e-9)
    return verts * scale, scale


def _decimate(mesh, target_faces: int = TARGET_FACES):
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception as e:                          # noqa: BLE001 -- reported, then kept as is
        log.warning("decimation failed (%s: %s); keeping %d faces", type(e).__name__, e,
                    len(mesh.faces))
        return mesh


def build(root=None, refs=None, target_faces: int = TARGET_FACES,
          radius: float = UNIT_SPHERE_RADIUS) -> Dict[str, dict]:
    """mesh id -> {verts, verts_original, faces, scale, centroid, category, role}.

    `verts` is normalised for BPS; `verts_original` keeps metres for contact distances.
    """
    import trimesh

    from src.analysis.loaders import taco

    root = Path(root or taco.default_root())
    refs = list(refs if refs is not None else taco.index(root))
    seen: Dict[str, dict] = {}
    for ref in refs:
        traj = taco.load(ref, root=root, with_hands=False)
        for track, role, cat in ((traj.tool, "tool", ref.action.tool),
                                 (traj.target, "target", ref.action.target)):
            if track is None or not track.mesh_path or track.name in seen:
                continue
            mesh = _decimate(trimesh.load(track.mesh_path, process=False, force="mesh"),
                             target_faces)
            V = np.asarray(mesh.vertices, dtype=np.float64) * track.mesh_scale   # cm -> m
            Vn, scale = normalise(V, radius)
            seen[track.name] = {"verts": Vn, "verts_original": V,
                                "faces": np.asarray(mesh.faces),
                                "scale": float(scale), "centroid": np.zeros(3),
                                "category": cat, "role": role}
    log.info("%d meshes; scale %.2f - %.2f", len(seen),
             min(v["scale"] for v in seen.values()), max(v["scale"] for v in seen.values()))
    return seen


def save(mesh_dict: Dict[str, dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mesh_dict, allow_pickle=True)
    return path


def load(path) -> Dict[str, dict]:
    return np.load(path, allow_pickle=True).item()
