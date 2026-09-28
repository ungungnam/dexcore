"""Reconstruction: sparse transferred states + dense tau_O -> a dense hand trajectory.

    K_t  --I-->  D_hat_t

THE CONTRACT. `reconstruct()` takes the transferred SPARSE demo and the DENSE target object
trajectory, and returns a dense `Demonstration` whose length is the trajectory's -- not the
selection's. That asymmetry is the whole point of the stage: selection removed frames, tau_O never
lost any, and reconstruction is what puts the hand back on every frame the object occupies.

WHAT COMES BACK IS A COMPLETE DEMONSTRATION, not a keypoint cloud: 51-DoF `configs_{side}`, world
keypoints that are the FK of those configs (so the two cannot disagree), and recomputed contact
links. Anything less would not be loadable by DexMachina, which is the point of the format.

DIAGNOSTICS ARE PART OF THE RETURN. Per-frame IK residual and convergence come back alongside the
demo, because "the reconstruction ran" and "the reconstruction is usable" are different claims and
only the second one matters.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.trajectory import ObjectTrajectory


@dataclass
class ReconstructionDiagnostics:
    """Per-frame evidence about the reconstructed trajectory."""
    method: str = ""
    per_frame_rms: Dict[str, np.ndarray] = field(default_factory=dict)   # side -> (T,) metres
    converged: Dict[str, np.ndarray] = field(default_factory=dict)       # side -> (T,) bool
    ik_history: Dict[str, list] = field(default_factory=dict)
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def num_unconverged(self) -> int:
        return int(sum((~c).sum() for c in self.converged.values()))

    @property
    def all_converged(self) -> bool:
        return self.num_unconverged == 0

    def summary(self) -> str:
        lines = [f"{self.method}:"]
        for side, rms in sorted(self.per_frame_rms.items()):
            conv = self.converged.get(side)
            n_bad = int((~conv).sum()) if conv is not None else 0
            worst = int(np.argmax(rms))
            lines.append(
                f"  {side:5s} keypoint RMS {np.sqrt(np.mean(rms ** 2)) * 1000:7.3f} mm  "
                f"worst {rms.max() * 1000:7.3f} mm @ row {worst}"
                + (f"  -- {n_bad} frames NOT converged" if n_bad else ""))
        return "\n".join(lines)


@dataclass
class ReconstructionResult:
    demo: Demonstration
    diagnostics: ReconstructionDiagnostics

    def require_converged(self) -> "ReconstructionResult":
        if not self.diagnostics.all_converged:
            raise RuntimeError("reconstruction did not converge on some frames:\n"
                               + self.diagnostics.summary())
        return self


class Reconstructor(abc.ABC):
    """Base class. Subclasses implement `reconstruct`; the registry and naming live here."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.options = dict(kwargs)

    @abc.abstractmethod
    def reconstruct(self, transferred_sparse_demo: Demonstration,
                    target_object_trajectory: ObjectTrajectory,
                    **kwargs) -> ReconstructionResult:
        """Dense hand motion between the transferred sparse states."""

    def describe(self) -> str:
        if not self.options:
            return self.name
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in sorted(self.options.items())) + ")"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"

    @staticmethod
    def _check_inputs(sparse: Demonstration, traj: ObjectTrajectory) -> np.ndarray:
        """Validate the sparse/dense pairing and return the keyframe rows in TRAJECTORY numbering.

        The two are related through absolute source frame indices, which both sides carry. Matching
        on those rather than on position is what keeps a selection that dropped the first frame, or
        a trajectory that starts at a non-zero offset, from silently shifting the whole hand.
        """
        if sparse.num_frames < 2:
            raise ValueError(f"need >= 2 sparse frames to interpolate, got {sparse.num_frames}")
        traj_frames = traj.frames()
        pos = {int(f): i for i, f in enumerate(traj_frames)}
        rows = []
        for f in sparse.frames():
            if int(f) not in pos:
                raise ValueError(
                    f"selected frame {int(f)} is not present in the target object trajectory "
                    f"(which covers {traj_frames.min()}..{traj_frames.max()}). The trajectory must "
                    f"be dense over the window the selection came from.")
            rows.append(pos[int(f)])
        rows = np.asarray(rows, dtype=int)
        if np.any(np.diff(rows) <= 0):
            raise ValueError("sparse frames are not strictly increasing after matching to tau_O")
        return rows


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Reconstructor:
    if name not in _REGISTRY:
        raise ValueError(f"unknown reconstructor {name!r}. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def available() -> list:
    return sorted(_REGISTRY)
