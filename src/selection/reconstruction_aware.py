"""ReconstructionAwareSelector -- greedily add the frame the current reconstruction gets most wrong.

    1. start from the two endpoints
    2. reconstruct the trajectory from the frames selected so far
    3. score every unselected frame by its reconstruction error
    4. add the worst one
    5. repeat to the budget

THE SCORING RECONSTRUCTION IS THE CHEAP PROXY, NOT THE REAL ONE. It is object-relative keypoint
interpolation only -- no IK. Two reasons: the greedy loop runs it once per added frame (a 10% budget
on an 889-frame clip is ~89 solves), and IK's residual is dominated by whether the pose is
reachable, which is not what selection is choosing between. The proxy measures exactly the quantity
selection controls: how badly linear in-betweening misses this frame.

The consequence is that this selector optimises the interpolation stage, not the full pipeline. That
is a real limitation and it is the reason the criterion is pluggable -- swapping in an IK-aware or
contact-aware criterion is a one-line change and needs no change to the search.

OBJECT-RELATIVE BY DEFAULT, matching the reconstructor: a hand that is stationary on a moving object
must score as easy, and a world-frame criterion would score it as hard.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from src.data.demo import Demonstration
from src.paths import SIDES
from src.reconstruction import interpolate as I
from src.selection.base import Selector, SelectionConfig, register


def keypoint_criterion(demo: Demonstration, selected: np.ndarray,
                       object_relative: bool = True, joints=None) -> np.ndarray:
    """(T,) per-frame error of interpolating the hand keypoints from `selected` alone.

    The error of a selected frame is 0 by construction, which is what stops the greedy loop from
    re-picking it without needing a separate mask.
    """
    T = demo.num_frames
    targets = np.arange(T, dtype=np.float64)
    key_at = np.asarray(selected, dtype=np.float64)
    dense_T = demo.object_transform()
    key_T = dense_T[np.asarray(selected, dtype=int)]

    err = np.zeros(T)
    for side in SIDES:
        kp = demo.joints[side]
        if joints is not None:
            kp = kp[:, joints]
        approx = I.interpolate_keypoints(
            kp[np.asarray(selected, dtype=int)], key_T, dense_T, key_at, targets,
            frame="root" if object_relative else "world")
        err += np.linalg.norm(approx - kp, axis=-1).mean(axis=-1)
    return err / len(SIDES)


CRITERIA: dict = {
    "keypoint": keypoint_criterion,
    "keypoint_world": lambda d, s: keypoint_criterion(d, s, object_relative=False),
    "fingertip": lambda d, s: keypoint_criterion(d, s, joints=[16, 17, 18, 19, 20]),
}


@register
class ReconstructionAwareSelector(Selector):
    """Greedy worst-first selection under a pluggable reconstruction-error criterion."""

    name = "reconstruction_aware"

    def __init__(self, config: Optional[SelectionConfig] = None, criterion: str = "keypoint",
                 **kwargs):
        super().__init__(config, **kwargs)
        if criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {criterion!r}. Known: {', '.join(sorted(CRITERIA))}")
        self.criterion_name = criterion
        self.criterion: Callable = CRITERIA[criterion]

    def describe(self) -> str:
        return f"{super().describe()[:-1]}, criterion={self.criterion_name})" \
            if "(" in super().describe() else f"{self.name}(criterion={self.criterion_name})"

    def frame_indices(self, demo: Demonstration) -> np.ndarray:
        total = demo.num_frames
        budget = self.config.resolve_budget(total)
        selected = [0, total - 1]
        while len(selected) < budget:
            err = self.criterion(demo, np.array(sorted(selected)))
            err[np.asarray(selected, dtype=int)] = -np.inf     # never re-pick
            nxt = int(np.argmax(err))
            if not np.isfinite(err[nxt]) or err[nxt] <= 0:
                break                                          # nothing left to improve
            selected.append(nxt)
        return np.array(sorted(selected))
