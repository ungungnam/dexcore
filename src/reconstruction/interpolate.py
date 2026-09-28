"""Object-relative keypoint interpolation -- the in-betweening the reconstructor rests on.

    world hand keypoint
        |  target object inverse transform
    object-relative hand keypoint
        |  linear interpolation between keyframes
    dense object-relative keypoint trajectory
        |  dense target object transform
    world-space dense hand keypoint trajectory

    ^O p_j(k) = T_O(k)^-1 p_j(k)
    ^O p_j(t) = (1 - a) ^O p_j(k_i) + a ^O p_j(k_{i+1}),   a = (t - k_i) / (k_{i+1} - k_i)
      p_j(t) = T_Ot(t) ^O p_j(t)

WHY OBJECT-RELATIVE AND NOT WORLD. A hand holding a moving object is nearly STATIC in the object's
frame and travels a long arc in the world's. Interpolating in the world frame across a 10% budget
cuts the corner of that arc and pulls the hand off the object; interpolating in the object frame
preserves the grasp and lets the object carry it. This is the single most important choice in the
reconstruction stage, and it is why tau_O has to stay dense.

THE FRAME IS THE OBJECT ROOT, AND THAT IS AN APPROXIMATION. An ARCTIC object is ARTICULATED: the
root frame is the base part, so a keypoint riding the LID is not static in this frame even when the
grasp never changes. The articulation is interpolated alongside, which is exact at the keyframes
and correct to first order between them, but a grasp on the lid across a large articulation change
will show it. `frame="root"` is the honest name for what this does; a part-aware frame is the
obvious next variant and the interface takes it as an argument for that reason.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from src import geometry as G
from src.data.trajectory import ObjectTrajectory

FRAMES = ("root", "world")


def bracket(keyframes: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each `target` frame, the surrounding keyframe rows and the blend weight.

    -> (lo_row, hi_row, alpha), each (T,). Refuses to EXTRAPOLATE: a target outside
    [keyframes[0], keyframes[-1]] raises, because silently clamping would produce a frozen hand at
    the ends that looks like a real (and quite reasonable) result.
    """
    kf = np.asarray(keyframes, dtype=np.float64)
    t = np.asarray(targets, dtype=np.float64)
    if kf.ndim != 1 or kf.size < 2:
        raise ValueError(f"need >= 2 keyframes, got {kf.size}")
    if np.any(np.diff(kf) <= 0):
        raise ValueError("keyframes must be strictly increasing")
    lo_v, hi_v = kf[0], kf[-1]
    out = (t < lo_v - 1e-9) | (t > hi_v + 1e-9)
    if np.any(out):
        bad = t[out]
        raise ValueError(
            f"{out.sum()} target frames lie outside the keyframe span [{lo_v:g}, {hi_v:g}] "
            f"(e.g. {bad[:5]}); reconstruction interpolates and refuses to extrapolate. "
            f"Keep the clip endpoints in the selection.")

    hi = np.clip(np.searchsorted(kf, t, side="left"), 1, kf.size - 1)
    lo = hi - 1
    span = kf[hi] - kf[lo]
    alpha = np.where(span > 0, (t - kf[lo]) / np.where(span > 0, span, 1.0), 0.0)
    # a target that lands exactly on a keyframe must reproduce it bit-exactly, not to 1e-16
    exact = np.isclose(t[:, None], kf[None, :], rtol=0, atol=0)
    hit = exact.any(axis=1)
    if np.any(hit):
        j = np.argmax(exact[hit], axis=1)
        lo = lo.copy(); hi = hi.copy(); alpha = alpha.copy()
        lo[hit] = j
        hi[hit] = j
        alpha[hit] = 0.0
    return lo.astype(int), hi.astype(int), alpha


def lerp(values: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Linear blend along axis 0: values is (K, ...), result is (T, ...)."""
    a = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
    return (1.0 - a) * values[lo] + a * values[hi]


def interpolate_keypoints(keypoints: np.ndarray, key_transforms: np.ndarray,
                          dense_transforms: np.ndarray, keyframes: np.ndarray,
                          targets: np.ndarray, frame: str = "root") -> np.ndarray:
    """The pipeline in the module docstring. -> (T, N, 3) world keypoints.

    keypoints        (K,N,3) world keypoints at the selected keyframes
    key_transforms   (K,4,4) object pose AT those keyframes (the TARGET object's, post-transfer)
    dense_transforms (T,4,4) object pose at every output frame
    keyframes        (K,)    frame index of each keyframe, in the same numbering as `targets`
    targets          (T,)    the output frames
    frame            "root"  interpolate in the object root frame (see the module docstring)
                     "world" interpolate in the world frame -- the ablation, not the default
    """
    if frame not in FRAMES:
        raise ValueError(f"frame must be one of {FRAMES}, got {frame!r}")
    kp = np.asarray(keypoints, dtype=np.float64)
    lo, hi, a = bracket(keyframes, targets)

    if frame == "world":
        return lerp(kp, lo, hi, a)

    rel = G.transform_points(kp, G.invert(np.asarray(key_transforms, dtype=np.float64)))
    rel_dense = lerp(rel, lo, hi, a)
    return G.transform_points(rel_dense, np.asarray(dense_transforms, dtype=np.float64))


def interpolate_scalar(values: np.ndarray, keyframes: np.ndarray,
                       targets: np.ndarray) -> np.ndarray:
    """Linear interpolation of a (K,) or (K,D) signal onto `targets`."""
    lo, hi, a = bracket(keyframes, targets)
    return lerp(np.asarray(values, dtype=np.float64), lo, hi, a)


def interpolate_configs(configs: Dict[str, np.ndarray], keyframes: np.ndarray,
                        targets: np.ndarray) -> Dict[str, np.ndarray]:
    """Linearly interpolate a `configs_{side}` dict. Used as the IK WARM START, not as an answer.

    These are URDF joint angles including three Euler wrist angles, and linear interpolation of
    Euler angles is not a rotation interpolation. That is acceptable HERE precisely because the
    result is only an initial guess that IK then corrects against the interpolated keypoints --
    which are interpolated properly, in the object frame. It would not be acceptable as output.
    """
    lo, hi, a = bracket(keyframes, targets)
    return {k: lerp(np.asarray(v, dtype=np.float64), lo, hi, a) for k, v in configs.items()}
