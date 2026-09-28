"""Selection: WHICH frames of the human demonstration need to be retained.

    D_s  --S-->  K_s

A selector answers one question and nothing else: given a dense demonstration, which rows survive.
It never changes geometry and never changes the object trajectory -- both of those belong to later
stages, and keeping selection ignorant of them is what lets the same selector be used for the
same-object redundancy experiment and for cross-object transfer without modification.

THE CONTRACT. `select()` returns a `Demonstration` of the same type with fewer frames, carrying
`provenance.selected_frames` / `source_frames` / `source_num_frames`. Subclasses implement
`frame_indices()` -- the algorithm -- and inherit provenance stamping, budget resolution and the
endpoint guarantee from here, so a new selector cannot forget to record what it did.

BUDGET. Either `ratio` (fraction of the dense clip) or `num_frames` (an absolute count). A ratio is
what the redundancy experiment sweeps (100 / 25 / 10 / 5 / 1 %); a count is what a fair comparison
against another method at equal budget needs. Giving both is an error rather than a precedence rule.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from src.data.demo import Demonstration


@dataclass
class SelectionConfig:
    """Everything a selector may vary. A selector's NAME plus this config fully describes a run."""
    ratio: Optional[float] = None          # fraction of the dense clip to keep, in (0, 1]
    num_frames: Optional[int] = None       # absolute budget; mutually exclusive with `ratio`
    keep_endpoints: bool = True            # always retain the first and last frame
    seed: Optional[int] = None             # only used by selectors that are actually stochastic

    def validate(self) -> "SelectionConfig":
        if self.ratio is not None and self.num_frames is not None:
            raise ValueError("give ratio OR num_frames, not both -- otherwise a run's budget is "
                             "ambiguous and two configs with the same name can disagree")
        if self.ratio is not None and not (0.0 < self.ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {self.ratio}")
        if self.num_frames is not None and self.num_frames < 2:
            raise ValueError(f"num_frames must be >= 2 (both endpoints), got {self.num_frames}")
        return self

    def resolve_budget(self, total: int) -> int:
        """-> how many frames to keep, given a dense clip of `total` frames.

        Rounds a ratio to the NEAREST frame and clamps to [2, total]. The clamp at 2 is why a 1%
        budget on a short clip still produces a usable pair rather than a degenerate single frame.
        """
        if self.num_frames is not None:
            n = int(self.num_frames)
        elif self.ratio is not None:
            n = int(round(self.ratio * total))
        else:
            n = total
        return int(np.clip(n, 2, total))


class Selector(abc.ABC):
    """Base class. Subclasses implement `frame_indices`; everything else is handled here."""

    name: str = "base"

    def __init__(self, config: Optional[SelectionConfig] = None, **kwargs):
        cfg = config or SelectionConfig(**kwargs)
        self.config = cfg.validate()

    # ------------------------------------------------------------------ the algorithm
    @abc.abstractmethod
    def frame_indices(self, demo: Demonstration) -> np.ndarray:
        """-> sorted, unique row indices into `demo` to retain. Must include >= 2 frames."""

    # ------------------------------------------------------------------ the contract
    def select(self, demo: Demonstration, **kwargs) -> Demonstration:
        """Dense demo -> sparse demo, with the selection recorded in the provenance."""
        idx = np.asarray(self.frame_indices(demo), dtype=int)
        idx = self._finalize(idx, demo.num_frames)
        ratio = len(idx) / demo.num_frames
        return demo.subset(idx, selector=self.describe(), ratio=ratio, **kwargs)

    def _finalize(self, idx: np.ndarray, total: int) -> np.ndarray:
        """Sort, de-duplicate, bounds-check, and honour `keep_endpoints`."""
        idx = np.unique(np.asarray(idx, dtype=int))
        if idx.size and (idx.min() < 0 or idx.max() >= total):
            raise IndexError(f"{self.name}: frame index out of range [0,{total}): "
                             f"got [{idx.min()}, {idx.max()}]")
        if self.config.keep_endpoints:
            idx = np.unique(np.concatenate([idx, [0, total - 1]]))
        if idx.size < 2:
            raise ValueError(f"{self.name}: selected {idx.size} frames; need at least 2")
        return idx

    def describe(self) -> str:
        """The name a result carries, e.g. 'uniform(ratio=0.1)'. Must round-trip to a config."""
        c = self.config
        if c.num_frames is not None:
            return f"{self.name}(n={c.num_frames})"
        if c.ratio is not None:
            return f"{self.name}(ratio={c.ratio:g})"
        return f"{self.name}(all)"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


# ------------------------------------------------------------------------------- the registry
_REGISTRY = {}


def register(cls):
    """Register a selector so a config file can name it. Adding a variant touches nothing else."""
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Selector:
    """Selector by name, e.g. build('uniform', ratio=0.1)."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown selector {name!r}. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def available() -> list:
    return sorted(_REGISTRY)
