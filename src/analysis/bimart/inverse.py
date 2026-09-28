"""Back from the canonical frame to the world, and from keypoints to a MANO hand.

The motion model predicts 100 surface points per hand plus their direction vectors, in the TARGET's
frame. Neither is a hand: a viewer, a penetration metric and a retargeting pipeline all need a
posed MANO mesh. This module closes that gap, reusing BimArt's own postprocessing and fitting so
the two ports stay comparable.

WHY THE POSE IS READ BACK FROM TACO. `global_states` carries the anchor's translation RELATIVE to
the window's first frame, which is what the network should see -- an absolute table position is a
nuisance variable. But a relative translation cannot be inverted on its own, so the absolute pose
is fetched from the dataset rather than duplicated into every `.npz`. The object pose files are
17 KB each; re-reading them costs nothing and keeps one source of truth.

THE ANCHOR IS THE TARGET, matching `features.py`. Handing BimArt's postprocessing the tool's pose
instead would silently place every hand in the wrong frame -- the reconstruction would still look
like a hand, just in the wrong place, which is the kind of error that survives a visual check.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

HAND_KEYPOINTS = 100


def anchor_pose(sequence_id: str, root=None) -> Tuple[np.ndarray, np.ndarray]:
    """(rot (T,3) axis-angle, trans (T,3) metres) of the canonical anchor -- TACO's target."""
    from scipy.spatial.transform import Rotation

    from src import geometry as G
    from src.analysis.loaders import taco

    root = Path(root or taco.default_root())
    triplet, seq = sequence_id.split("/")
    ref = next(r for r in taco.index(root) if r.triplet == triplet and r.sequence == seq)
    traj = taco.load(ref, root=root, with_hands=False)
    tgt = traj.target
    return Rotation.from_matrix(G.wxyz_to_R(tgt.quat)).as_rotvec(), tgt.pos


def to_world(points_canonical: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """(T,K,3) canonical -> world. The exact inverse of `features.to_canonical`."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_rotvec(np.asarray(rot)).as_matrix()
    return np.einsum("tij,tkj->tki", R, points_canonical) + np.asarray(trans)[:, None, :]


def split_action(action: np.ndarray, hand_keypoints: int = HAND_KEYPOINTS) -> Dict[str, np.ndarray]:
    """(T,1200) -> the four blocks the model actually predicts, each (T,100,3).

    Layout, fixed by `features.sequence_features`: keypoints left then right, then the direction
    vectors in the same order.
    """
    k = hand_keypoints
    a = np.asarray(action)
    kp = a[..., : k * 6].reshape(*a.shape[:-1], k * 2, 3)
    dv = a[..., k * 6: k * 12].reshape(*a.shape[:-1], k * 2, 3)
    return {"kp_left": kp[..., :k, :], "kp_right": kp[..., k:, :],
            "dirvec_left": dv[..., :k, :], "dirvec_right": dv[..., k:, :]}


def action_to_world(action: np.ndarray, sequence_id: str, start: int = 0, root=None
                    ) -> Dict[str, np.ndarray]:
    """One predicted window -> hand keypoints in world metres, both hands.

    `start` is the window's first frame in the source sequence, so the anchor pose lines up.
    """
    parts = split_action(action)
    T = len(action)
    rot, trans = anchor_pose(sequence_id, root)
    sl = slice(start, start + T)
    r, t = rot[sl], trans[sl]
    if len(r) != T:
        raise ValueError(f"{sequence_id}: window [{start}, {start+T}) runs past {len(rot)} frames")
    return {"left": to_world(parts["kp_left"], r, t),
            "right": to_world(parts["kp_right"], r, t),
            "dirvec_left": np.einsum("tij,tkj->tki",
                                     __import__("scipy.spatial.transform", fromlist=["Rotation"])
                                     .Rotation.from_rotvec(r).as_matrix(), parts["dirvec_left"]),
            "dirvec_right": np.einsum("tij,tkj->tki",
                                      __import__("scipy.spatial.transform", fromlist=["Rotation"])
                                      .Rotation.from_rotvec(r).as_matrix(), parts["dirvec_right"])}


def fit_mano(kp_left_world: np.ndarray, kp_right_world: np.ndarray, steps: int = 4000,
             hand_keypoints: int = HAND_KEYPOINTS, device: str = "cuda:0"):
    """Fit MANO to predicted world keypoints, via BimArt's own optimiser.

    Returns whatever `mano_utils.fit_mano` returns: parameters plus left and right vertices.
    `steps` is BimArt's 4000 by default -- the expensive part of inference, not of training.
    """
    import sys

    import torch

    bim = "/home/uhnam/workspace/dexcore/third_party/BimArt"
    if bim not in sys.path:
        sys.path.insert(0, bim)
    from utils import mano_utils

    layer = mano_utils.create_mano_layer()
    l = torch.as_tensor(kp_left_world, dtype=torch.float32, device=device)
    r = torch.as_tensor(kp_right_world, dtype=torch.float32, device=device)
    if l.ndim == 3:                                   # (T,K,3) -> one batch item
        l, r = l[None], r[None]
    idx = np.load(f"{bim}/assets/part_fps_hand_index_100.npy").astype(np.int32)
    return mano_utils.fit_mano(l, r, layer, ph=l.shape[1], idxl=idx, idxr=idx,
                               hand_keypoints=hand_keypoints, steps=steps)
