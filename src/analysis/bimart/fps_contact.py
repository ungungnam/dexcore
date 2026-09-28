"""Farthest-point sample indices per mesh, for gathering the contact label.

WHY THIS EXISTS. The contact label was originally read at the BPS indices -- the object vertices
each basis point happens to be nearest to. Both objects are scaled into the basis by the TARGET's
normalisation factor, so whenever the tool is large relative to the target its far parts, the handle
included, are nearest to no basis point and their contact is never recorded. Measured against the
dense per-vertex truth on 36 sequences the tool's contact fraction came out 0.204 too low on
average and, in the worst cases, 0.02 where the truth was 0.71: the label said the tool was
untouched while the hand was holding it. The error is present for EVERY target category (cup 0.06
against 0.42, teapot 0.11 against 0.35, plate 0.24 against 0.40), worst where the target is small.

Sampling each mesh's own surface instead makes coverage a property of the mesh and not of the other
object's size. Against the same dense truth the new label is 0.002 off on the tool half, and the
target half -- which was already fine, the target being the object the basis is fitted to -- stays
within 0.004. The BPS input is untouched, so this changes the training target and the motion
model's conditioning and nothing else.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

N_PER_OBJECT = 512   # matches the BPS count, so the 2048-wide contact vector keeps its shape


def fps_indices(verts: np.ndarray, k: int = N_PER_OBJECT) -> np.ndarray:
    """k vertex indices spread over the surface. Deterministic: starts at the vertex furthest
    from the centroid, so the same mesh always yields the same set."""
    verts = np.asarray(verts, dtype=np.float64)
    n = len(verts)
    if n <= k:
        return np.arange(n, dtype=np.int64)
    sel = np.empty(k, dtype=np.int64)
    sel[0] = int(np.argmax(np.linalg.norm(verts - verts.mean(0), axis=1)))
    d = np.linalg.norm(verts - verts[sel[0]], axis=1)
    for i in range(1, k):
        sel[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(verts - verts[sel[i]], axis=1))
    return sel


def build_table(mesh_dict: Dict[str, dict], k: int = N_PER_OBJECT) -> Dict[str, np.ndarray]:
    out = {}
    for i, (name, m) in enumerate(mesh_dict.items()):
        out[name] = fps_indices(m["verts_original"], k)
        if (i + 1) % 40 == 0 or i + 1 == len(mesh_dict):
            log.info("  fps %d/%d meshes", i + 1, len(mesh_dict))
    return out


def load_table(path) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in np.load(path, allow_pickle=True).item().items()}


def mesh_key(mesh_dict, mesh_id) -> str:
    """Mesh ids appear as ints in the index and as strings (sometimes zero-padded) as dict keys."""
    cands = [mesh_id, str(mesh_id)]
    try:                                   # ARCTIC's keys are names, not numbers
        cands.append(f"{int(mesh_id):03d}")
    except (TypeError, ValueError):
        pass
    for cand in cands:
        if cand in mesh_dict:
            return cand
    raise KeyError(f"mesh {mesh_id!r} not in mesh dict")


def gather_indices(table, mesh_dict, tool_mesh, target_mesh) -> np.ndarray:
    """(1024,) indices into the concatenated [tool verts, target verts] axis of contact_left/right,
    the same axis and the same ordering the BPS indices used."""
    tk, gk = mesh_key(mesh_dict, tool_mesh), mesh_key(mesh_dict, target_mesh)
    n_tool = len(mesh_dict[tk]["verts_original"])
    return np.concatenate([table[tk], table[gk] + n_tool]).astype(np.int64)


def gather(z, inds: np.ndarray) -> np.ndarray:
    """(T, 2048) contact at `inds`, left hand then right -- the layout store.ARRAYS expects."""
    return np.concatenate([z["contact_left"][:, inds], z["contact_right"][:, inds]],
                          axis=-1).astype(np.float32)
