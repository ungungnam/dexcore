"""Reconstruction error against a reference demonstration, and trajectory smoothness.

RECONSTRUCTION ERROR ONLY MEANS SOMETHING FOR THE SAME OBJECT. When the target geometry differs
there is no ground-truth hand trajectory to compare against -- that is the whole reason the
cross-object experiment leans on downstream utility (ADD-AUC) instead. These functions therefore
REFUSE a reference whose object differs, rather than returning a number that reads like a score.

The criteria are separate small functions so the reconstruction-aware selector can swap one for
another (fingertip-only, contact-weighted, qpos RMS) without touching the selector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.mano import FINGERTIP_JOINTS
from src.paths import SIDES


# --------------------------------------------------------------------------- per-frame criteria
def keypoint_error(demo: Demonstration, ref: Demonstration, side: str,
                   joints=None, object_relative: bool = False) -> np.ndarray:
    """(T,) mean Euclidean keypoint error, metres.

    `object_relative=True` measures the error in the OBJECT's frame.

    IT CHANGES NOTHING WHEN BOTH DEMOS SHARE AN OBJECT TRAJECTORY, which is exactly the same-object
    case this metric is defined for: the two clouds are mapped by the SAME rigid transform, and a
    rigid transform preserves distance, so ||R(a-b)|| == ||a-b|| identically. It is kept because it
    does differ once the two demos' object trajectories differ -- comparing two reconstructions of
    the same clip against each other, say -- but for a same-object reconstruction-vs-source
    comparison it will report the world-frame number to the last bit. Do not read agreement between
    the two as corroboration; it is an algebraic identity.
    """
    a, b = demo.joints[side], ref.joints[side]
    if joints is not None:
        a, b = a[:, joints], b[:, joints]
    if object_relative:
        a = demo.world_to_object(a)
        b = ref.world_to_object(b)
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


def fingertip_error(demo: Demonstration, ref: Demonstration, side: str, **kw) -> np.ndarray:
    """(T,) mean fingertip error -- the five points that actually do the manipulating."""
    return keypoint_error(demo, ref, side, joints=FINGERTIP_JOINTS, **kw)


def qpos_error(demo: Demonstration, ref: Demonstration, side: str) -> np.ndarray:
    """(T,) RMS over the 51 MANO DoF, radians (and metres on the 3 prismatic wrist DoF)."""
    keys = sorted(set(demo.configs[side]) & set(ref.configs[side]))
    if not keys:
        raise ValueError(f"no shared configs_{side} joints between the two demos")
    a = np.stack([demo.configs[side][k] for k in keys], axis=1)
    b = np.stack([ref.configs[side][k] for k in keys], axis=1)
    return np.sqrt(((a - b) ** 2).mean(axis=1))


CRITERIA: Dict[str, Callable] = {
    "keypoint": keypoint_error,
    "keypoint_object_relative": lambda d, r, s: keypoint_error(d, r, s, object_relative=True),
    "fingertip": fingertip_error,
    "qpos": qpos_error,
}


# ------------------------------------------------------------------------------- the metric
@dataclass
class ReconstructionErrorResult:
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict)     # "criterion.side" -> (T,)

    def aggregate(self) -> dict:
        out = {}
        for key, v in self.per_frame.items():
            unit = 1.0 if key.startswith("qpos") else 1000.0     # rad vs mm
            out[f"{key}.mean"] = float(v.mean() * unit)
            out[f"{key}.rms"] = float(np.sqrt((v ** 2).mean()) * unit)
            out[f"{key}.max"] = float(v.max() * unit)
        return out

    def summary(self) -> str:
        lines = ["reconstruction error vs reference:"]
        for key, v in sorted(self.per_frame.items()):
            unit, u = (1.0, "rad") if key.startswith("qpos") else (1000.0, "mm")
            lines.append(f"  {key:34s} mean {v.mean()*unit:8.3f} {u}  max {v.max()*unit:8.3f} {u}")
        return "\n".join(lines)


def reconstruction_error(demo: Demonstration, ref: Demonstration,
                         criteria=("keypoint", "fingertip", "qpos"),
                         sides=SIDES) -> ReconstructionErrorResult:
    """Compare a reconstructed demo against a dense reference. Same object, same length, or refuse."""
    if demo.obj_name != ref.obj_name:
        raise ValueError(
            f"reconstruction error needs the SAME object: got {demo.obj_name!r} vs {ref.obj_name!r}. "
            f"Across a geometry change there is no ground-truth hand trajectory to compare to -- "
            f"use downstream utility (ADD-AUC) instead.")
    if demo.num_frames != ref.num_frames:
        raise ValueError(f"length mismatch: {demo.num_frames} vs {ref.num_frames}")

    res = ReconstructionErrorResult()
    for name in criteria:
        if name not in CRITERIA:
            raise ValueError(f"unknown criterion {name!r}. Known: {', '.join(sorted(CRITERIA))}")
        for side in sides:
            if not (demo.configs.get(side) and ref.configs.get(side)) and name == "qpos":
                continue
            res.per_frame[f"{name}.{side}"] = CRITERIA[name](demo, ref, side)
    return res


# ------------------------------------------------------------------------------- smoothness
@dataclass
class SmoothnessResult:
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict)

    def aggregate(self) -> dict:
        return {f"{k}.mean": float(v.mean()) for k, v in self.per_frame.items()} | \
               {f"{k}.max": float(v.max()) for k, v in self.per_frame.items()}

    def summary(self) -> str:
        lines = ["smoothness (keypoint acceleration):"]
        for k, v in sorted(self.per_frame.items()):
            lines.append(f"  {k:22s} mean {v.mean():8.4f} m/s^2   max {v.max():9.4f} m/s^2")
        return "\n".join(lines)


def trajectory_smoothness(demo: Demonstration, sides=SIDES) -> SmoothnessResult:
    """Mean keypoint acceleration magnitude per frame -- the diagnostic for in-betweening artefacts.

    A linear interpolant is piecewise-constant in velocity, so its acceleration is an impulse train
    at the keyframes. A spike here at exactly the selected frames is the expected signature of the
    method, not a bug -- but it is what a smoother reconstructor would have to improve on, so it is
    measured rather than assumed.
    """
    dt = 1.0 / demo.fps
    res = SmoothnessResult()
    for side in sides:
        p = demo.joints.get(side)
        if p is None or len(p) < 3:
            continue
        acc = (p[2:] - 2 * p[1:-1] + p[:-2]) / (dt ** 2)
        res.per_frame[f"accel.{side}"] = np.linalg.norm(acc, axis=-1).mean(axis=-1)
    return res
