"""The four trajectory representations, and the rotation encodings they share.

ROTATION ENCODING IS FIXED HERE AND NOWHERE ELSE.

    absolute rotation     6D = the first two COLUMNS of R, flattened (Zhou et al. 2019). Continuous
                          on SO(3), unlike a quaternion (double cover) or Euler angles (gimbal,
                          discontinuous, order-dependent).
    incremental rotation  SO(3) log map -> rotation vector, axis*angle in radians.

Euler angles are never produced.

THE FOUR REPRESENTATIONS, and what each one removes:

    raw world          nothing removed. Absolute position in the capture volume is a feature.
    world-origin       the chunk's global translation offset only (minus the target's position at
                       the chunk's first frame). World ORIENTATION, hence gravity, is kept.
    virtual base       the target is the frame. Y(t) = inv(X_O(t)) @ X_T(t) is the full relative
                       SE(3) motion -- not reduced to an articulation angle -- carried alongside
                       the target's own world motion G_O(t), so the target's trajectory is still
                       available but the tool is described relative to it.
    differential       the same relative trajectory as local increments in the previous frame's
                       relative frame, plus the initial relative pose so the trajectory is
                       reconstructible. This is the one representation with no absolute pose in it
                       past frame 0.

Translations stay in metres throughout; no per-chunk magnitude normalisation happens here, because
path length and displacement are exactly what Part 3 measures. Z-scoring for distance work is a
separate, explicit step (`zscore`).
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.analysis.action_structure.chunks import CHUNK_LEN, ChunkSet


# ---------------------------------------------------------------------------- rotation codecs
def rot6d(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,6): the first two columns, stacked. Inverse is Gram-Schmidt."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def so3_log(R: np.ndarray) -> np.ndarray:
    """(...,3,3) -> (...,3) rotation vector. scipy handles the small-angle and pi branches."""
    from scipy.spatial.transform import Rotation

    flat = np.asarray(R, dtype=np.float64).reshape(-1, 3, 3)
    return Rotation.from_matrix(flat).as_rotvec().reshape(R.shape[:-2] + (3,))


# --------------------------------------------------------------------------- relative motion
def relative_pose(cs: ChunkSet):
    """Y(t) = inv(X_O(t)) @ X_T(t), the tool in the target's moving frame.

    Returns (p_Y, R_Y) with shapes (N,64,3) and (N,64,3,3). Written as R_O^T (p_T - p_O) rather
    than by building and inverting 4x4 matrices -- same result, no 64x4x4 temporaries per chunk.
    """
    R_Y = np.einsum("ntji,ntjk->ntik", cs.targ_R, cs.tool_R)          # R_O^T R_T
    p_Y = np.einsum("ntji,ntj->nti", cs.targ_R, cs.tool_p - cs.targ_p)  # R_O^T (p_T - p_O)
    return p_Y, R_Y


def increments(p_Y: np.ndarray, R_Y: np.ndarray):
    """Local increments of the relative trajectory, per the differential protocol.

        dR(t) = R_Y(t-1)^T R_Y(t)          t = 1..63
        dp(t) = R_Y(t-1)^T (p_Y(t) - p_Y(t-1))

    Returns (log dR, dp), both (N,63,3).
    """
    dR = np.einsum("ntji,ntjk->ntik", R_Y[:, :-1], R_Y[:, 1:])
    dp = np.einsum("ntji,ntj->nti", R_Y[:, :-1], p_Y[:, 1:] - p_Y[:, :-1])
    return so3_log(dR), dp


# ------------------------------------------------------------------------------ the four reps
def raw_world(cs: ChunkSet) -> np.ndarray:
    """(N,64,18) = [p_T, rot6d(R_T), p_O, rot6d(R_O)], all in world coordinates."""
    return np.concatenate([cs.tool_p, rot6d(cs.tool_R), cs.targ_p, rot6d(cs.targ_R)], axis=-1)


def world_origin_aligned(cs: ChunkSet) -> np.ndarray:
    """(N,64,18). As `raw_world` but both positions are offset by p_O(0) of the same chunk.

    The control for "is this just where on the table the episode happened?". Only the chunk's
    global translation goes; the world's orientation, and therefore gravity, stays.
    """
    origin = cs.targ_p[:, :1, :]                                       # (N,1,3)
    return np.concatenate([cs.tool_p - origin, rot6d(cs.tool_R),
                           cs.targ_p - origin, rot6d(cs.targ_R)], axis=-1)


def virtual_base(cs: ChunkSet) -> np.ndarray:
    """(N,64,18) = [p_Y, rot6d(R_Y), p_O - p_O(0), rot6d(R_O)].

    The trajectory-only analogue of a base / moving-part factorisation: the target is the base,
    the tool is the moving part, and the full relative SE(3) motion is kept.
    """
    p_Y, R_Y = relative_pose(cs)
    G_O = np.concatenate([cs.targ_p - cs.targ_p[:, :1, :], rot6d(cs.targ_R)], axis=-1)
    return np.concatenate([p_Y, rot6d(R_Y), G_O], axis=-1)


def differential(cs: ChunkSet) -> np.ndarray:
    """(N, 9 + 63*6) = [p_Y(0), rot6d(R_Y(0)), then 63 increments of [log dR, dp]].

    Frame 0's relative pose is retained deliberately: without it the increments do not reconstruct
    the trajectory. Returned flat (not (N,T,C)) because its frames are not homogeneous -- frame 0
    is an absolute pose and the rest are increments.
    """
    p_Y, R_Y = relative_pose(cs)
    log_dR, dp = increments(p_Y, R_Y)
    head = np.concatenate([p_Y[:, 0], rot6d(R_Y[:, 0])], axis=-1)      # (N,9)
    tail = np.concatenate([log_dR, dp], axis=-1).reshape(len(p_Y), -1)  # (N,63*6)
    return np.concatenate([head, tail], axis=-1)


#: name -> builder. The analysis iterates this, so adding a representation changes nothing else.
REPRESENTATIONS = {
    "raw_world": raw_world,
    "world_origin": world_origin_aligned,
    "virtual_base": virtual_base,
    "differential": differential,
}


# ------------------------------------------------------------------------------- normalisation
def zscore(X: np.ndarray, channels: int = 0) -> np.ndarray:
    """Channel-wise z-score across chunks, for distance and clustering only.

    "Channel-wise" is per CHANNEL, with the mean and sd pooled over chunks AND frames -- not per
    (frame, channel) cell. Pooling over time is what keeps frame 10 and frame 50 of the same
    channel on one scale; standardising each cell separately would instead give every frame its own
    unit of measurement and destroy the temporal shape the representation exists to carry.

    `channels` selects the layout: 0 for an already-flat (N,D) array (each column is its own
    channel), or C for an (N,T,C) tensor. Output is always flat (N, T*C).
    """
    X = np.asarray(X, dtype=np.float64)
    if channels:
        X = X.reshape(len(X), -1, channels)
        mu = X.mean(axis=(0, 1), keepdims=True)
        sd = X.std(axis=(0, 1), keepdims=True)
    else:
        X = X.reshape(len(X), -1)
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
    out = (X - mu) / np.where(sd > 1e-12, sd, 1.0)
    return out.reshape(len(X), -1)


def build_all(cs: ChunkSet) -> Dict[str, np.ndarray]:
    """Every representation, as (N,T,C) tensors except `differential`, which is flat by nature."""
    return {name: fn(cs) for name, fn in REPRESENTATIONS.items()}


#: channel counts, for `zscore`. `differential` is flat and heterogeneous, so it z-scores per
#: column; the increment block is homogeneous in units anyway (rad and metres per frame).
CHANNELS = {"raw_world": 18, "world_origin": 18, "virtual_base": 18, "differential": 0}
