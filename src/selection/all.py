"""AllSelector -- keep every frame. The dense reference the other budgets are measured against."""
from __future__ import annotations

import numpy as np

from src.data.demo import Demonstration
from src.selection.base import Selector, register


@register
class AllSelector(Selector):
    """Identity selection.

    It exists so the dense case travels the SAME pipeline as every sparse case: `All` ->
    `IdentityTransfer` -> `Reconstruction` must reproduce the source demo, which is the only
    end-to-end check that the transfer and reconstruction stages are not quietly distorting
    something. A dense baseline that skipped the pipeline could not catch that.
    """

    name = "all"

    def frame_indices(self, demo: Demonstration) -> np.ndarray:
        return np.arange(demo.num_frames)

    def describe(self) -> str:
        return "all"
