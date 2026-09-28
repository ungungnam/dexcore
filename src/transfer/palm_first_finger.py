"""PalmFirstFingerTransfer -- preserve the palm-object relationship, then adapt the fingers.

    source sparse hand-object state
            |  palm placement          rigid, global   (vendored: dexcore/functional_wrist.py)
            |  finger adaptation       local, per finger (vendored: dexcore/close_realize.py)
    target sparse hand-object state

THE PRINCIPLE. The palm carries the FUNCTION of a grasp -- which face is approached, from what
direction, at what stand-off -- and it is the part that must survive a change of geometry intact.
The fingers carry the FIT, and fit is what should be allowed to change when the surface changes.
Placing the palm first and letting the fingers close onto the new surface encodes that ordering;
solving for everything at once does not.

HOW THE PALM FRAME IS DEFINED, and this is the part worth knowing: NOT by a designated part. Source
surface points near the palm are sampled PART-LOCALLY, transferred onto the target, and each frame's
wrist is carried by the rigid Kabsch transform mapping one point set onto the other. Because the
points ride part-locally on both sides, a grasp on the lid follows the lid through articulation, a
grasp on the base follows the base, and a mixed grasp blends -- with no rule anywhere saying which
part a hand "belongs to".

Then each finger closes from an open palm onto the target until it reaches a soft contact band,
against the CONVEX DECOMPOSITION the simulator itself collides with. A pose that stops there is one
the simulator will also find collision-free.

PER SELECTED FRAME. The operator answers each keyframe on its own; nothing is smoothed across them.
That is the claim being tested, not an implementation detail.
"""
from __future__ import annotations

from src.data.demo import Demonstration
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory
from src.transfer.base import Transfer, TransferResult, register


@register
class PalmFirstFingerTransfer(Transfer):
    """Palm placement first, then per-finger closing. One selected frame at a time."""

    name = "palm_first_finger"

    def __init__(self, contact_band_m: float = 0.002, **kwargs):
        super().__init__(contact_band_m=contact_band_m, **kwargs)
        self.contact_band_m = float(contact_band_m)

    def describe(self) -> str:
        return f"{self.name}(band={self.contact_band_m * 1000:g}mm)"

    def transfer(self, sparse_demo: Demonstration, source_object: ArticulatedObject,
                 target_object: ArticulatedObject,
                 target_object_trajectory: ObjectTrajectory, **kwargs) -> TransferResult:
        raise NotImplementedError(
            "PalmFirstFingerTransfer is not implemented yet -- this is dexcore's OWN method, so "
            "it is written here rather than adapted from a third party.\n"
            "The pieces already in place for it: `src/data/object.py` carries the canonical frame "
            "and articulation phase, `src/geometry.py` has `kabsch` for the palm fit, and "
            "`src/data/mano.py` has `wrist_pose_from_q` / `apply_wrist_transform` for turning a "
            "rigid palm placement back into the six wrist DOFs."
        )
