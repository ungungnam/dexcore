"""The dense object-state trajectory -- tau_O^{1:T} in the factorization.

    D  ->  ( tau_O^{1:T} ,  K_HO )

This is the first half, and it is given its own type because it is the half that stays DENSE
through the whole pipeline. Selection thins the hand states; the object trajectory is never
thinned, and after a cross-object transfer it is not even the source's any more -- it is the
target's, produced by the canonical-frame object transfer. Reconstruction consumes it as the frame
to place the interpolated hand into.

Keeping it separate from `Demonstration` is what makes "reconstruct this hand motion against THAT
object trajectory" expressible at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src import geometry as G
from src.paths import DEFAULT_FPS


@dataclass
class ObjectTrajectory:
    """A dense object state sequence: root pose + articulation, one row per frame."""

    obj_name: str
    obj_pos: np.ndarray            # (T,3) metres, world
    obj_quat: np.ndarray           # (T,4) wxyz, sign-unrolled
    obj_arti: np.ndarray           # (T,) radians
    fps: float = DEFAULT_FPS
    frame_index: Optional[np.ndarray] = None   # absolute source frame of each row
    source: str = ""                           # how this trajectory was produced

    def __post_init__(self):
        self.obj_pos = np.asarray(self.obj_pos, dtype=np.float64)
        self.obj_quat = G.unroll_quat(np.asarray(self.obj_quat, dtype=np.float64))
        self.obj_arti = np.asarray(self.obj_arti, dtype=np.float64).reshape(-1)
        n = {len(self.obj_pos), len(self.obj_quat), len(self.obj_arti)}
        if len(n) != 1:
            raise ValueError(f"ragged object trajectory: pos {len(self.obj_pos)}, "
                             f"quat {len(self.obj_quat)}, arti {len(self.obj_arti)}")

    def __len__(self) -> int:
        return len(self.obj_arti)

    @property
    def num_frames(self) -> int:
        return len(self)

    def frames(self) -> np.ndarray:
        if self.frame_index is not None:
            return np.asarray(self.frame_index, dtype=int)
        return np.arange(len(self))

    @classmethod
    def from_demo(cls, demo, source: str = "source demo") -> "ObjectTrajectory":
        """The object trajectory a demonstration already carries. Identity transfer uses this."""
        return cls(obj_name=demo.obj_name, obj_pos=demo.obj_pos.copy(),
                   obj_quat=demo.obj_quat.copy(), obj_arti=demo.obj_arti.copy(),
                   fps=demo.fps, frame_index=demo.frames(), source=source)

    def transform(self, t=None) -> np.ndarray:
        """Object ROOT pose T_O(t) as (4,4), or (T,4,4) for `t=None`."""
        if t is None:
            return G.affine(G.wxyz_to_R(self.obj_quat), self.obj_pos)
        return G.affine(G.wxyz_to_R(self.obj_quat[t]), self.obj_pos[t])

    def subset(self, rows) -> "ObjectTrajectory":
        idx = np.asarray(rows, dtype=int)
        return ObjectTrajectory(
            obj_name=self.obj_name, obj_pos=self.obj_pos[idx], obj_quat=self.obj_quat[idx],
            obj_arti=self.obj_arti[idx], fps=self.fps, frame_index=self.frames()[idx],
            source=self.source)

    def state(self) -> np.ndarray:
        """(T,8) [pos(3), quat wxyz(4), arti(1)] -- DexMachina's ADD evaluation layout."""
        return np.concatenate([self.obj_pos, self.obj_quat, self.obj_arti[:, None]], axis=1)

    def summary(self) -> str:
        span = np.round((self.obj_pos.max(0) - self.obj_pos.min(0)) * 100, 1)
        return (f"tau_O[{self.obj_name}]  {len(self)} frames @ {self.fps:g} fps  "
                f"({self.source or 'unknown source'})\n"
                f"  arti    {np.degrees(self.obj_arti.min()):7.2f} .. "
                f"{np.degrees(self.obj_arti.max()):7.2f} deg\n"
                f"  pos     span {span} cm")
