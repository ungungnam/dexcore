"""Contact between the hands and the two TACO objects, in the demo format dexcore already uses.

TACO SHIPS NO CONTACT ANNOTATION. The official contents list poses, meshes, video, camera
parameters and 2D segmentations, and nothing else; the README's tactile-annotation section points
at a third-party sensor-labelling tool that takes tactile hardware streams TACO does not have. So
contact is DERIVED here, from geometry that is already in the dataset.

That turns out to be well posed rather than marginal. Measured on sampled sequences, the minimum
distance from the hand holding the tool to the tool surface has a median of 0.4-1.6 mm, while the
other hand sits at 80-161 mm. Three orders of magnitude apart, so the threshold is not a knob the
answer depends on.

THE DEFINITION IS `src/data/contacts.py`'s, NOT A NEW ONE: a contact is an OBJECT surface point
within `threshold` of the HAND SKIN, the points are farthest-point-sampled to at most 50, and their
hand-side partners collapse onto the 16 MANO links. Matching it is what lets a TACO demo be read by
the same code that reads an ARCTIC one.

ONE DELIBERATE REINTERPRETATION. ARCTIC has one articulated object and uses `part_id` for its two
parts (1 = lid, 2 = base). TACO has two rigid objects, so `part_id` carries WHICH OBJECT instead:

    1 = tool     2 = target

The channel keeps its shape and its role -- "which piece of geometry was touched" -- but its
meaning is dataset-specific, so anything reading it must know which dataset it came from. Recording
both objects rather than the tool alone matters: in `(measure, ruler, bowl)` the left hand holds the
ruler while the right works the bowl, and a tool-only record would show that hand as idle.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

TOOL_PART_ID = 1
TARGET_PART_ID = 2
#: Surface samples per object, matching `src/data/contacts.N_OBJ_SAMPLE`.
N_OBJ_SAMPLE = 3580
#: Contact slots per frame per hand, matching the demo format.
MAX_CONTACT_PER_STEP = 50
#: Metres. Chosen an order of magnitude above the measured grasp distance (0.4-1.6 mm) and two
#: below the free hand (80+ mm), so it sits in the empty gap between the two populations.
DEFAULT_THRESHOLD_M = 0.005

_MANO_ROOT = os.environ.get(
    "DEXCORE_MANO_ROOT",
    "/home/uhnam/workspace/TACO-Instructions/dataset_utils/manopth/mano/models")
_MANOPTH = os.environ.get(
    "DEXCORE_MANOPTH", "/home/uhnam/workspace/TACO-Instructions/dataset_utils")


def _install_numpy_aliases() -> None:
    """chumpy, which the MANO pickles need, imports names numpy removed in 1.24.

    Restoring the aliases is the whole fix. It is done here rather than asked of the environment so
    that a worker process gets it too, and it is idempotent.
    """
    for name, value in (("bool", bool), ("int", int), ("float", float), ("complex", complex),
                        ("object", object), ("str", str), ("unicode", str),
                        ("nan", float("nan")), ("inf", float("inf"))):
        if not hasattr(np, name):
            setattr(np, name, value)


def mano_layer(side: str, device: str = "cpu"):
    """A MANO layer in TACO's own configuration: no PCA, 45 finger dimensions, wrist at origin."""
    import sys

    _install_numpy_aliases()
    if _MANOPTH not in sys.path:
        sys.path.insert(0, _MANOPTH)
    from manopth.manopth.manolayer import ManoLayer

    return ManoLayer(mano_root=_MANO_ROOT, use_pca=False, ncomps=45, side=side,
                     center_idx=0).to(device)


def hand_geometry(track, layer, device: str = "cpu"):
    """A `HandTrack` -> (skin (T,778,3), joints (T,21,3)) in world metres.

    The layer is run with a ZERO global rotation and the recorded rotation applied afterwards,
    because `center_idx=0` already puts the wrist at the origin: rotating there and translating by
    `root_pos` reproduces exactly what TACO's own loader produces.
    """
    import torch

    from src import geometry as G

    n = len(track)
    theta = np.concatenate([np.zeros((n, 3)), track.finger_pose], axis=1)
    with torch.no_grad():
        verts, joints, _ = layer(
            torch.as_tensor(theta, dtype=torch.float32, device=device),
            torch.as_tensor(np.repeat(track.shape[None], n, 0), dtype=torch.float32, device=device))
    verts = verts.cpu().numpy() / 1000.0
    joints = joints.cpu().numpy() / 1000.0
    R = G.wxyz_to_R(track.root_quat)
    skin = np.einsum("tij,tvj->tvi", R, verts) + track.root_pos[:, None, :]
    kp = np.einsum("tij,tvj->tvi", R, joints) + track.root_pos[:, None, :]
    return skin, kp


def object_samples(track, n: int = N_OBJ_SAMPLE, seed: int = 0) -> Optional[np.ndarray]:
    """(n,3) surface samples of an object track's mesh, in the object's own frame, metres."""
    import trimesh

    if not track.mesh_path:
        return None
    mesh = trimesh.load(track.mesh_path, process=False, force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, dtype=np.float64) * track.mesh_scale


def episode_contacts(traj, threshold: float = DEFAULT_THRESHOLD_M, layers: Optional[Dict] = None,
                     n_obj_sample: int = N_OBJ_SAMPLE, seed: int = 0) -> Dict[str, np.ndarray]:
    """Every contact field for one episode, both hands, both objects.

    Returns `contacts_{side}` (T,50,4), `valid_{side}` (T,50), `links_{side}` (T,16,4).
    """
    from scipy.spatial import cKDTree

    from src import geometry as G
    from src.data.contacts import closest_link, farthest_sample

    layers = layers or {s: mano_layer(s) for s in ("left", "right")}
    parts = [(traj.tool, TOOL_PART_ID), (traj.target, TARGET_PART_ID)]
    local, pid = [], []
    for track, p in parts:
        if track is None:
            continue
        pts = object_samples(track, n_obj_sample, seed)
        if pts is None:
            continue
        local.append((track, pts))
        pid.append(np.full(len(pts), p, dtype=np.int64))
    if not local:
        raise ValueError(f"{traj.sequence_id}: no object mesh resolved")
    pid = np.concatenate(pid)

    T = traj.num_frames
    world = np.empty((T, len(pid), 3))
    off = 0
    for track, pts in local:
        R = G.wxyz_to_R(track.quat)
        world[:, off:off + len(pts)] = np.einsum("tij,vj->tvi", R, pts) + track.pos[:, None, :]
        off += len(pts)

    out: Dict[str, np.ndarray] = {}
    for side in ("left", "right"):
        h = traj.hands.get(side)
        if h is None:
            continue
        skin, joints = hand_geometry(h, layers[side])
        oc = np.zeros((T, MAX_CONTACT_PER_STEP, 4), dtype=np.float32)
        ov = np.zeros((T, MAX_CONTACT_PER_STEP), dtype=bool)
        links = np.zeros((T, 16, 4), dtype=np.float32)
        for t in range(T):
            d, idx = cKDTree(skin[t]).query(world[t], k=1, workers=1)
            hit = d < threshold
            if not np.any(hit):
                continue
            c_obj = np.concatenate([world[t][hit], pid[hit][:, None]], axis=-1)
            c_hand = np.concatenate([skin[t][idx[hit]], pid[hit][:, None]], axis=-1)
            k = min(MAX_CONTACT_PER_STEP, len(c_obj))
            oc[t, :k] = farthest_sample(c_obj, k)
            ov[t, :k] = True
            links[t] = closest_link(farthest_sample(c_hand, k), joints[t])
        out[f"contacts_{side}"] = oc
        out[f"valid_{side}"] = ov
        out[f"links_{side}"] = links
    return out


def _worker(task):
    dataset_root, ref, threshold, out_dir = task
    from src.analysis.loaders import taco

    try:
        traj = taco.load(ref, root=dataset_root)
        fields = episode_contacts(traj, threshold)
        path = Path(out_dir) / f"{ref.sequence_id.replace('/', '__')}.npz"
        np.savez_compressed(path, **fields)
        touched = {s: float(fields[f"valid_{s}"].any(axis=1).mean())
                   for s in ("left", "right") if f"valid_{s}" in fields}
        return {"episode_id": ref.sequence_id, "frames": traj.num_frames, **
                {f"contact_frac_{s}": v for s, v in touched.items()}}
    except Exception as e:                          # noqa: BLE001 -- reported, then skipped
        log.warning("%s: %s: %s", ref.sequence_id, type(e).__name__, e)
        return None


def run(out_dir, root=None, refs: Optional[Sequence] = None, threshold: float = DEFAULT_THRESHOLD_M,
        workers: int = 8, limit: Optional[int] = None):
    """Derive contacts for every episode, one compressed file each, in parallel."""
    import multiprocessing as mp

    from src.analysis.loaders import taco

    root = Path(root or taco.default_root())
    refs = list(refs if refs is not None else taco.index(root))
    if limit:
        refs = refs[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(root, r, threshold, out_dir) for r in refs]

    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(processes=max(1, min(workers, avail))) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            if r is not None:
                rows.append(r)
            if i % 100 == 0 or i == len(tasks):
                log.info("  %d/%d", i, len(tasks))
    return rows
