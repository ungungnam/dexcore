"""IdentityTransfer -- no geometric change. The control for the temporal-redundancy experiment.

    Demo -> Selection -> IdentityTransfer -> Reconstruction -> Evaluation

With the target object equal to the source, Phi is the identity and every difference between the
reconstructed demo and the source is attributable to SELECTION and RECONSTRUCTION alone. That is
what makes "how much of the dense hand trajectory is actually necessary?" a measurable question:
there is no geometric grounding error mixed into the answer.

It refuses a genuinely different target rather than silently pretending the geometry matches --
transferring a box grasp onto a ketchup bottle by doing nothing is not a baseline, it is a bug.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory
from src.paths import SIDES
from src.transfer.base import (FrameDiagnostic, Transfer, TransferDiagnostics, TransferResult,
                               register)


@register
class IdentityTransfer(Transfer):
    """Pass the sparse states through untouched.

    `allow_different_object=True` opts into using it as a deliberate NO-OP BASELINE against a
    different target -- the "what if we did not transfer at all" row of a cross-object table. It is
    off by default because that is a very different claim from "identity transfer", and one that
    should have to be asked for by name.
    """

    name = "identity"

    def __init__(self, allow_different_object: bool = False, **kwargs):
        super().__init__(allow_different_object=allow_different_object, **kwargs)
        self.allow_different_object = bool(allow_different_object)

    def describe(self) -> str:
        """'identity', or 'identity(no-op baseline)' when pointed at a different object."""
        return "identity(no-op baseline)" if self.allow_different_object else "identity"

    def transfer(self, sparse_demo: Demonstration, source_object: ArticulatedObject,
                 target_object: ArticulatedObject,
                 target_object_trajectory: ObjectTrajectory, **kwargs) -> TransferResult:
        if source_object.name != target_object.name and not self.allow_different_object:
            raise ValueError(
                f"IdentityTransfer got source {source_object.name!r} != target "
                f"{target_object.name!r}. Doing nothing is not a transfer. Use a real transfer "
                f"operator, or pass allow_different_object=True to record this as an explicit "
                f"no-op baseline.")
        self._check_trajectory(sparse_demo, target_object_trajectory, target_object)

        out = replace(sparse_demo,
                      obj_name=target_object.name,
                      provenance=replace(sparse_demo.provenance,
                                         target_object=target_object.name,
                                         transfer=self.describe()))
        diag = TransferDiagnostics(method=self.describe())
        for row, frame in enumerate(sparse_demo.frames()):
            for side in SIDES:
                diag.frames.append(FrameDiagnostic(
                    row=row, frame=int(frame), side=side, ok=True,
                    metrics={"displacement": 0.0}))
        diag.extras["note"] = "identity: hand states and tau_O are bit-identical to the source"
        return TransferResult(demo=out, trajectory=target_object_trajectory, diagnostics=diag)
