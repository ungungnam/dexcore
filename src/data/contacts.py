"""Re-deriving the contact fields the demo format carries, against a target geometry.

THE DEFINITION IS THE FORMAT'S, NOT ONE OF OUR OWN. A contact is an OBJECT surface point within
`threshold` of the HAND SKIN (the MANO visual meshes, not the collision proxy). Those points are
farthest-point-sampled to at most 50 -- `contacts.{side}` -- and their hand-side partners are then
collapsed onto the 16 MANO links by nearest-average-joint distance, with the part id decided by
majority vote -- `contact_links_{side}`.

Matching this matters beyond tidiness: `map_contacts.py` reads `contacts.{side}` and fails outright
without it, and the contact reward is computed from `contact_links_{side}`. A plausible but
different definition here would train against a different notion of touching than every other demo
in the dataset.

WHY THIS IS A MODULE AND NOT A METHOD. Every synthesis method has to produce these fields, and they
have to be produced THE SAME WAY or the metrics computed from them are not comparable across
methods. It used to live inside `LinearKeypointReconstructor`, which made it reachable only by the
staged pipeline; an end-to-end method would have had to reimplement it, and "reimplemented the
contact definition slightly differently" is exactly the kind of difference that shows up as a
result.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src import geometry as G
from src.data.hand import get_hand
from src.data.mano import MANO_HAND_LINKS
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory
from src.paths import BASE_PART_ID, LID_PART_ID

MAX_CONTACT_PER_STEP = 50    # process_arctic default
N_OBJ_SAMPLE = 3580          # matches ARCTIC's object vertex density
SAMPLES_PER_LINK = 32


def recompute_contacts(side: str, configs: Dict[str, np.ndarray], joints: np.ndarray,
                       traj: ObjectTrajectory, obj: ArticulatedObject, threshold: float,
                       samples_per_link: int = SAMPLES_PER_LINK,
                       n_obj_sample: int = N_OBJ_SAMPLE, seed: int = 0) -> Tuple[np.ndarray, ...]:
    """-> (contact_pos (T,16,3), contact_part (T,16), contact_mask (T,16),
           obj_contacts (T,50,4), obj_valid (T,50))

    `configs` poses the hand; `joints` are that hand's 21 world keypoints (used to assign each
    contact to a link). `traj` and `obj` are the TARGET object's trajectory and geometry.
    """
    from scipy.spatial import cKDTree
    T = len(traj)
    hand = get_hand(side, device="cpu", samples_per_link=samples_per_link)
    q = hand.fk.dofs_from_cfg(configs, frames=T)
    skin = hand.skin_points_world(q)                                   # (T,V,3) hand surface

    # object surface samples, part-local, once -- the object is rigid per part
    obj_local, obj_pid = [], []
    for pid in (LID_PART_ID, BASE_PART_ID):
        pts = obj.sample_part_surface(pid, num_samples=n_obj_sample, seed=seed)
        obj_local.append(pts)
        obj_pid.append(np.full(len(pts), pid, dtype=np.int64))
    obj_pid = np.concatenate(obj_pid)

    pos = np.zeros((T, 16, 3))
    part = np.zeros((T, 16), dtype=np.int8)
    mask = np.zeros((T, 16), dtype=bool)
    oc = np.zeros((T, MAX_CONTACT_PER_STEP, 4))
    ov = np.zeros((T, MAX_CONTACT_PER_STEP), dtype=bool)

    Tw = {pid: obj.part_to_world_batch(pid, traj.obj_arti, traj.obj_pos, traj.obj_quat)
          for pid in (LID_PART_ID, BASE_PART_ID)}
    for t in range(T):
        ov_world = np.concatenate([G.transform_points(obj_local[k], Tw[pid][t])
                                   for k, pid in enumerate((LID_PART_ID, BASE_PART_ID))])
        d, idx = cKDTree(skin[t]).query(ov_world, k=1, workers=-1)
        hit = d < threshold
        if not np.any(hit):
            continue
        c_obj = np.concatenate([ov_world[hit], obj_pid[hit][:, None]], axis=-1)
        c_hand = np.concatenate([skin[t][idx[hit]], obj_pid[hit][:, None]], axis=-1)
        n = min(MAX_CONTACT_PER_STEP, len(c_obj))
        c_obj = farthest_sample(c_obj, n)
        c_hand = farthest_sample(c_hand, n)
        oc[t, :n] = c_obj
        ov[t, :n] = True
        links = closest_link(c_hand, joints[t])
        pos[t] = links[:, :3]
        part[t] = links[:, 3].astype(np.int8)
        mask[t] = np.linalg.norm(links[:, :3], axis=-1) > 0
    return pos, part, mask, oc, ov


def farthest_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Farthest-point sample `n` rows of (N,4) contacts, on xyz only. Mirrors process_arctic."""
    if len(pts) <= n:
        return pts[:n]
    out = [pts[0]]
    while len(out) < n:
        d = np.linalg.norm(pts[:, :3] - np.stack(out)[:, :3][:, None], axis=-1)
        out.append(pts[np.argmax(np.min(d, axis=0))])
    return np.stack(out)


def closest_link(contacts: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """(N,4) hand-side contacts + (21,3) keypoints -> (16,4) per-link [pos(3), part id].

    A contact belongs to the link whose spanned joints it is closest to ON AVERAGE; the link's
    position is the inverse-distance-weighted mean of its contacts, and its part id is the majority
    vote. Zero rows where a link has no contact -- the format's "no contact" encoding.
    """
    avg = np.stack([np.mean([np.linalg.norm(contacts[:, :3] - joints[j], axis=-1)
                             for j in idxs], axis=0) for _, idxs in MANO_HAND_LINKS])
    nearest = np.argmin(avg, axis=0)
    ndist = np.min(avg, axis=0)
    out = np.zeros((len(MANO_HAND_LINKS), 4))
    for i in range(len(MANO_HAND_LINKS)):
        m = nearest == i
        if not np.any(m):
            continue
        voted = np.argmax(np.bincount(contacts[m, 3].astype(int)))
        w = 1.0 / np.maximum(ndist[m], 1e-9)
        out[i] = np.concatenate([np.average(contacts[m][:, :3], axis=0, weights=w), [voted]])
    return out
