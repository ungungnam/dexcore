"""Transfer: given a sparse hand-object interaction state, ground it onto a target geometry.

    K_s  --Phi-->  K_t

THE OPERATOR IS PER-FRAME. Phi acts on one selected keyframe at a time, independently. That is not
an implementation detail, it is the research claim being tested: if a sparse set of per-frame
geometric groundings is enough, then dense per-frame transfer was doing more work than it needed
to. A transfer that quietly smoothed across frames would answer a different question.

FAILURE IS REPORTED, NOT REPAIRED. A frame whose grounding is infeasible -- no palm placement
found, IK did not converge, the result penetrates -- is recorded in `TransferDiagnostics` with a
reason and marked not-ok. It is NEVER silently replaced with the source pose or with a neighbour,
because a reconstructed trajectory built from invisibly-substituted keyframes looks exactly like a
working one. Callers decide what to do; `TransferResult.require_all_ok()` is the strict option.

THE OBJECT TRAJECTORY IS AN INPUT, NOT AN OUTPUT. tau_O for the target is produced upstream by the
canonical-frame object transfer and handed in dense. Transfer places the HAND relative to it; it
must not modify it.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory


@dataclass
class FrameDiagnostic:
    """What happened to ONE selected keyframe under the transfer."""
    row: int                          # index into the sparse demo
    frame: int                        # absolute source frame
    side: str
    ok: bool = True
    reason: str = ""                  # why it failed, empty when ok
    metrics: Dict[str, float] = field(default_factory=dict)   # palm err, ik residual, pen depth...

    def describe(self) -> str:
        m = "  ".join(f"{k}={v:.4g}" for k, v in sorted(self.metrics.items()))
        return (f"frame {self.frame:5d} [{self.side:5s}] "
                f"{'ok ' if self.ok else 'FAIL'}  {m}{('  <- ' + self.reason) if self.reason else ''}")


@dataclass
class TransferDiagnostics:
    """Per-frame outcomes plus whatever the operator wants to expose for inspection.

    `extras` is deliberately untyped: palm correspondence points, source/target fingertip
    locations, the finger adaptation deltas. Each operator exposes what it actually computed
    rather than conforming to a schema that would fit none of them.
    """
    method: str = ""
    frames: List[FrameDiagnostic] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def num_failed(self) -> int:
        return sum(1 for f in self.frames if not f.ok)

    @property
    def all_ok(self) -> bool:
        return self.num_failed == 0

    def failures(self) -> List[FrameDiagnostic]:
        return [f for f in self.frames if not f.ok]

    def summary(self) -> str:
        n = len(self.frames)
        head = f"{self.method}: {n - self.num_failed}/{n} keyframe-sides transferred"
        if self.all_ok:
            return head
        lines = [head + f"  ({self.num_failed} FAILED)"]
        lines += ["    " + f.describe() for f in self.failures()[:20]]
        if self.num_failed > 20:
            lines.append(f"    ... and {self.num_failed - 20} more")
        return "\n".join(lines)


@dataclass
class TransferResult:
    """The transferred sparse states, the dense target object trajectory, and the diagnostics."""
    demo: Demonstration
    trajectory: ObjectTrajectory
    diagnostics: TransferDiagnostics

    def require_all_ok(self) -> "TransferResult":
        """Raise unless every keyframe transferred. The strict path for an automated sweep."""
        if not self.diagnostics.all_ok:
            raise RuntimeError("transfer failed on some keyframes:\n"
                               + self.diagnostics.summary())
        return self


class Transfer(abc.ABC):
    """Base class. Subclasses implement `transfer`; the registry and naming live here."""

    name: str = "base"

    def __init__(self, **kwargs):
        self.options = dict(kwargs)

    @abc.abstractmethod
    def transfer(self, sparse_demo: Demonstration, source_object: ArticulatedObject,
                 target_object: ArticulatedObject,
                 target_object_trajectory: ObjectTrajectory, **kwargs) -> TransferResult:
        """Ground each selected hand-object state onto `target_object`."""

    def describe(self) -> str:
        if not self.options:
            return self.name
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in sorted(self.options.items())) + ")"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"

    # ------------------------------------------------------------------ shared helpers
    @staticmethod
    def _check_trajectory(sparse_demo: Demonstration, traj: ObjectTrajectory,
                          target_object: ArticulatedObject) -> None:
        """The three ways a caller can hand in a mismatched trajectory, caught before any geometry."""
        if traj.obj_name != target_object.name:
            raise ValueError(f"target object trajectory is for {traj.obj_name!r} but the target "
                             f"object is {target_object.name!r}")
        n = sparse_demo.provenance.source_num_frames or sparse_demo.num_frames
        if len(traj) != n:
            raise ValueError(f"target object trajectory has {len(traj)} frames but the selection "
                             f"came from a {n}-frame clip; tau_O must stay dense")
        if sparse_demo.frames().max() - sparse_demo.frame_start >= len(traj):
            raise IndexError("a selected frame lies outside the target object trajectory")


_REGISTRY = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Transfer:
    if name not in _REGISTRY:
        raise ValueError(f"unknown transfer {name!r}. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def available() -> list:
    return sorted(_REGISTRY)
