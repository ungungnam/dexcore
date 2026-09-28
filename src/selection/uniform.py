"""UniformSelector -- N evenly spaced frames. The baseline every other selector must beat.

Uniform sampling ignores the demonstration's content entirely, which is exactly why it is the
control: without it, "reconstruction-aware selection retains the motion at 10%" is a claim with
nothing to be relative to.

ENDPOINTS ARE ALWAYS KEPT (`keep_endpoints`, default True). The reconstruction stage interpolates
BETWEEN selected frames, so a clip whose last keyframe is frame 800 of 889 has 89 frames that can
only be extrapolated -- which the linear reconstructor refuses to do. Keeping the endpoints is what
makes the budget an interpolation problem end to end.
"""
from __future__ import annotations

import numpy as np

from src.data.demo import Demonstration
from src.selection.base import Selector, register


@register
class UniformSelector(Selector):
    """`n` frames spaced evenly over the clip, endpoints included.

    Rounding `linspace` to integers can collide on very small budgets over short clips (two
    requested frames landing on the same row). That is de-duplicated, so the realised count can be
    below the requested budget -- `provenance.selection_ratio` records what was actually kept, not
    what was asked for.
    """

    name = "uniform"

    def frame_indices(self, demo: Demonstration) -> np.ndarray:
        total = demo.num_frames
        n = self.config.resolve_budget(total)
        return np.unique(np.round(np.linspace(0, total - 1, n)).astype(int))
