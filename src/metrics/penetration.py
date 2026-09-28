"""Hand-object penetration: how far the hand's surface is INSIDE the object's.

    depth(t, link) = max(0, -sdf_part(p))    over the link's surface samples, worst part

Signed distance is POSITIVE OUTSIDE throughout dexcore (src/data/object.py negates trimesh's
convention), so penetration is a clamped negative distance and a non-penetrating frame scores 0.

WHY THIS IS THE EXPENSIVE METRIC. Exact signed distance runs at ~470 queries/s and a clip asks for
~1e6. The KD-tree prefilter is what makes it tractable: a point whose nearest SAMPLED surface point
is further than `cutoff` cannot be penetrating by more than `cutoff`, so only the candidates are
evaluated exactly. `cutoff` therefore bounds the deepest penetration this metric can SEE -- it is
reported in the result so a saturated run is visible rather than silently clipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from src import geometry as G
from src.data.demo import Demonstration
from src.data.hand import get_hand
from src.data.object import ArticulatedObject, get_object
from src.paths import BASE_PART_ID, LID_PART_ID, SIDES


@dataclass
class PenetrationResult:
    """Per-frame depths plus the aggregate. Per-frame is kept so failure frames can be rendered."""
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict)        # side -> (T,) max depth, m
    per_frame_link: Dict[str, np.ndarray] = field(default_factory=dict)   # side -> (T,16) depth, m
    cutoff: float = 0.05
    saturated: Dict[str, int] = field(default_factory=dict)  # frames at/over the cutoff bound

    def aggregate(self) -> dict:
        out = {}
        for side, d in self.per_frame.items():
            out[f"{side}.max_depth_mm"] = float(d.max() * 1000)
            out[f"{side}.mean_depth_mm"] = float(d.mean() * 1000)
            out[f"{side}.frames_penetrating"] = int((d > 0).sum())
            out[f"{side}.frac_penetrating"] = float((d > 0).mean())
            out[f"{side}.saturated_frames"] = int(self.saturated.get(side, 0))
        if self.per_frame:
            out["max_depth_mm"] = max(v for k, v in out.items() if k.endswith("max_depth_mm"))
        out["cutoff_mm"] = float(self.cutoff * 1000)
        return out

    def summary(self) -> str:
        lines = ["penetration (hand into object):"]
        for side, d in sorted(self.per_frame.items()):
            sat = self.saturated.get(side, 0)
            lines.append(f"  {side:5s} max {d.max()*1000:7.2f} mm  mean {d.mean()*1000:6.2f} mm  "
                         f"penetrating {int((d>0).sum())}/{len(d)} frames"
                         + (f"  [{sat} frames AT the {self.cutoff*1000:.0f} mm cutoff]" if sat else ""))
        return "\n".join(lines)


def hand_object_penetration(demo: Demonstration, obj: Optional[ArticulatedObject] = None,
                            samples_per_link: int = 32, cutoff: float = 0.05,
                            sides=SIDES) -> PenetrationResult:
    """Penetration of each hand into the (articulated) object, per frame and per link."""
    obj = obj or get_object(demo.obj_name)
    T = demo.num_frames
    res = PenetrationResult(cutoff=float(cutoff))

    for side in sides:
        if not demo.configs.get(side):
            continue
        hand = get_hand(side, device="cpu", samples_per_link=samples_per_link)
        q = hand.fk.dofs_from_cfg(demo.configs[side], frames=T)
        pts = hand.link_points_world(q)                                  # (T,16,S,3)

        depth = np.zeros((T, 16))
        for pid in (LID_PART_ID, BASE_PART_ID):
            Tw = obj.part_to_world_batch(pid, demo.obj_arti, demo.obj_pos, demo.obj_quat)
            inv = G.invert(Tw)
            flat = pts.reshape(T, -1, 3)
            local = np.einsum("tij,tnj->tni", inv[:, :3, :3], flat) + inv[:, None, :3, 3]
            sd = obj.signed_distance(pid, local.reshape(-1, 3), cutoff=cutoff)
            sd = sd.reshape(T, 16, -1)
            depth = np.maximum(depth, np.clip(-sd, 0.0, None).max(axis=-1))

        res.per_frame_link[side] = depth
        res.per_frame[side] = depth.max(axis=1)
        res.saturated[side] = int((res.per_frame[side] >= cutoff * 0.999).sum())
    return res
