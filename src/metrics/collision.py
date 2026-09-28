"""Self-collision within a hand, and inter-hand collision between the two.

WHAT IS MEASURED: the minimum distance between the SURFACE SAMPLES of two links,

    gap(t, a, b) = min over samples  || p_a - p_b ||

reported as a distance, not a penetration depth. Two links that interpenetrate have intersecting
surfaces and therefore surface samples arbitrarily close together, so a gap at or near zero is the
signature of a collision; the sample spacing bounds how finely that can be resolved.

THIS IS A PROXIMITY DIAGNOSTIC, NOT A PHYSICS ENGINE. It flags frames worth rendering. It cannot
report how deep a collision is, and it will miss a collision whose intersection curve falls between
samples.

A SPHERE-PER-LINK APPROXIMATION WAS TRIED FIRST AND DISCARDED. The palm link is large and flat, so
its bounding sphere swallows the thumb: it reported 33 mm of "collision" on every frame of the
SOURCE human demonstration, which is physically valid by construction. Any metric that fails on
ground truth is measuring its own approximation error, so the sphere version was removed rather
than thresholded around.

CALIBRATE AGAINST THE SOURCE. The generic MANO URDF is not the ARCTIC MANO fit, so even a real
human demo shows small non-zero proximities. Compare a reconstruction's numbers to the SOURCE
demo's, not to zero -- `baseline_gaps()` computes exactly that.

ADJACENT LINKS ARE EXCLUDED: a finger segment always touches its own neighbour; that is a joint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.demo import Demonstration
from src.data.hand import get_hand
from src.data.mano import LINK_NAMES, MANO_HAND_LINKS
from src.paths import SIDES

# a gap at or below this is reported as a collision. Chosen from the source-demo baseline, not
# from zero -- see the module docstring.
DEFAULT_GAP_THRESHOLD = 1e-3       # metres


def _adjacent_pairs() -> set:
    """Link pairs sharing a MANO joint, i.e. articulated neighbours. Excluded from collision."""
    adj = set()
    for i, (_, ja) in enumerate(MANO_HAND_LINKS):
        for j, (_, jb) in enumerate(MANO_HAND_LINKS):
            if i < j and set(ja) & set(jb):
                adj.add((i, j))
    return adj


ADJACENT = _adjacent_pairs()


@dataclass
class CollisionResult:
    """Per-frame minimum gaps, plus which link pair was responsible."""
    per_frame_gap: Dict[str, np.ndarray] = field(default_factory=dict)   # key -> (T,) metres
    worst_pairs: Dict[str, list] = field(default_factory=dict)
    threshold: float = DEFAULT_GAP_THRESHOLD

    def aggregate(self) -> dict:
        out = {"gap_threshold_mm": float(self.threshold * 1000)}
        for key, g in self.per_frame_gap.items():
            out[f"{key}.min_gap_mm"] = float(g.min() * 1000)
            out[f"{key}.mean_gap_mm"] = float(g.mean() * 1000)
            out[f"{key}.frames_colliding"] = int((g <= self.threshold).sum())
            out[f"{key}.frac_colliding"] = float((g <= self.threshold).mean())
        return out

    def summary(self) -> str:
        lines = [f"collision (min inter-link surface gap, threshold "
                 f"{self.threshold*1000:.1f} mm):"]
        for key, g in sorted(self.per_frame_gap.items()):
            worst = self.worst_pairs.get(key, [])
            lines.append(f"  {key:16s} min {g.min()*1000:7.2f} mm  mean {g.mean()*1000:7.2f} mm  "
                         f"below threshold {int((g<=self.threshold).sum())}/{len(g)} frames"
                         + (f"  tightest {worst[0][0]}-{worst[0][1]}" if worst else ""))
        return "\n".join(lines)


def _link_points(demo: Demonstration, side: str, samples_per_link: int) -> np.ndarray:
    hand = get_hand(side, device="cpu", samples_per_link=samples_per_link)
    q = hand.fk.dofs_from_cfg(demo.configs[side], frames=demo.num_frames)
    return hand.link_points_world(q)                                    # (T,16,S,3)


def _min_gaps(pa: np.ndarray, pb: np.ndarray, skip: Optional[set]) -> Tuple[np.ndarray, list]:
    """Min surface-sample distance over all non-skipped link pairs. -> ((T,), worst pairs)."""
    T, L = pa.shape[0], pa.shape[1]
    best = np.full(T, np.inf)
    pair_min = np.full((L, L), np.inf)
    same = pa is pb
    for i in range(L):
        for j in range(L):
            if same and i >= j:
                continue
            if skip is not None and ((i, j) in skip or (j, i) in skip):
                continue
            # (T,S,1,3) - (T,1,S,3) -> (T,S,S); one pair at a time keeps this a few MB
            d = np.linalg.norm(pa[:, i, :, None, :] - pb[:, j, None, :, :], axis=-1).min(axis=(1, 2))
            pair_min[i, j] = d.min()
            best = np.minimum(best, d)
    flat = pair_min.ravel()
    order = np.argsort(flat)[:3]
    worst = [(LINK_NAMES[k // L], LINK_NAMES[k % L], round(float(flat[k] * 1000), 2))
             for k in order if np.isfinite(flat[k])]
    return best, worst


def hand_collision(demo: Demonstration, samples_per_link: int = 32,
                   self_collision: bool = True, inter_hand: bool = True,
                   threshold: float = DEFAULT_GAP_THRESHOLD) -> CollisionResult:
    """Minimum inter-link surface gap, per hand (self) and between hands."""
    res = CollisionResult(threshold=float(threshold))
    pts = {s: _link_points(demo, s, samples_per_link) for s in SIDES if demo.configs.get(s)}

    if self_collision:
        for side, p in pts.items():
            g, worst = _min_gaps(p, p, skip=ADJACENT)
            res.per_frame_gap[f"self.{side}"] = g
            res.worst_pairs[f"self.{side}"] = worst

    if inter_hand and len(pts) == 2:
        g, worst = _min_gaps(pts["left"], pts["right"], skip=None)
        res.per_frame_gap["inter_hand"] = g
        res.worst_pairs["inter_hand"] = worst
    return res


def baseline_gaps(source: Demonstration, **kw) -> dict:
    """The source demonstration's own collision numbers -- the floor a reconstruction is judged against.

    A human demo is physically realisable by construction, so whatever this reports is the metric's
    own approximation error (generic URDF vs the ARCTIC MANO fit, plus sample spacing). Numbers from
    a reconstruction only mean something relative to these.
    """
    return hand_collision(source, **kw).aggregate()
