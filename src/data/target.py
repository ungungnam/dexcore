"""Building the DENSE target object trajectory, tau_O, for a chosen target object.

THIS IS THE ONE PLACE THAT DECIDES "IS THE TARGET THE SAME OBJECT?", and the answer is by NAME
equality, nothing looser.

A SCALED OBJECT IS A DIFFERENT OBJECT. `box_s110` is not "the box, bigger" -- it is a separately
baked asset with its own geometry, and a demonstration transferred onto it has to be grounded onto
that geometry like any other cross-object transfer. There is deliberately NO "box family" shortcut
that would notice the shared `box` prefix and copy the source trajectory through: that shortcut
produces a trajectory which looks perfectly reasonable, trains without complaint, and answers a
question nobody asked. If the names differ, the canonical transfer runs.

THE CANONICAL TRANSFER, in one line:

    the TARGET's canonical frame is placed where the SOURCE's canonical frame is,
    and the lid angle is carried across by normalised PHASE, not by raw angle.

Both halves matter. Placing by canonical frame means the target is oriented the way the source was
-- hinge along the hinge, up along up -- rather than sharing a root pose that means something
different on each mesh. Carrying phase rather than angle means "half open" stays half open when the
two joints have different ranges; copying the angle would drive a lid past its own limit.

The computation is ELEMENTWISE in the frame index, so evaluating it on the whole clip and on a
keyframe subset give the same answer -- which is what lets the dense trajectory reconstruction uses
and the sparse one transfer aims at come from one function instead of two that could disagree.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from src import geometry as G
from src.data.demo import Demonstration
from src.data.object import ArticulatedObject, get_object
from src.data.trajectory import ObjectTrajectory


def is_same_object(source: str, target: str) -> bool:
    """Name equality, and nothing looser. See the module docstring on why there is no prefix rule."""
    return str(source) == str(target)


def target_object_states(source: ArticulatedObject, target: ArticulatedObject,
                         src_pos, src_quat, src_arti):
    """-> (pos (F,3), quat (F,4) wxyz, arti (F,)): where the TARGET is, given where the source is.

    For each frame the source's canonical frame is located in the world,

        T_canon_world = T_root_world(k) @ C_src

    and the target's root is placed so that ITS canonical frame lands there,

        T_root_world_target(k) = T_canon_world @ C_tgt^-1
    """
    src_pos = np.asarray(src_pos, dtype=np.float64)
    src_quat = np.asarray(src_quat, dtype=np.float64)
    src_arti = np.asarray(src_arti, dtype=np.float64).reshape(-1)
    F = len(src_arti)

    C_src = source.canonical_pose()
    C_tgt_inv = G.invert(target.canonical_pose())

    pos = np.zeros((F, 3))
    quat = np.zeros((F, 4))
    arti = np.zeros(F)
    for k in range(F):
        root_world = G.affine(G.wxyz_to_R(src_quat[k]), src_pos[k])
        base_world = (root_world @ C_src) @ C_tgt_inv
        pos[k] = base_world[:3, 3]
        quat[k] = G.R_to_wxyz(base_world[:3, :3])
        arti[k] = target.alpha_to_q(source.q_to_alpha(float(src_arti[k])))
    return pos, quat, arti


def build_target_trajectory(demo: Demonstration, target_object: ArticulatedObject,
                            source_object: Optional[ArticulatedObject] = None) -> ObjectTrajectory:
    """The dense tau_O the target object should follow.

    Same object   -> the demonstration's own trajectory, EXACTLY. Identity must be bit-exact, which
                     is what makes the temporal-redundancy experiment measure selection and
                     reconstruction alone.
    Different     -> the canonical transfer above.
    """
    source_object = source_object or get_object(demo.obj_name)
    if is_same_object(source_object.name, target_object.name):
        return ObjectTrajectory.from_demo(demo, source="identity (same object)")

    pos, quat, arti = target_object_states(
        source_object, target_object, demo.obj_pos, demo.obj_quat, demo.obj_arti)

    lo, hi = sorted((target_object.q_closed, target_object.q_open))
    n_out = int(((arti < lo - 1e-6) | (arti > hi + 1e-6)).sum())
    if n_out:
        # `alpha_to_q` clamps phase, so this can only fire if the target's own limits are
        # inconsistent -- an asset problem, not something to quietly clip
        raise ValueError(
            f"canonical transfer produced {n_out}/{len(arti)} frames outside "
            f"{target_object.name}'s joint range [{lo:.3f}, {hi:.3f}] rad "
            f"(got {arti.min():.3f}..{arti.max():.3f}). Check the target's URDF joint limits.")

    return ObjectTrajectory(
        obj_name=target_object.name, obj_pos=pos, obj_quat=quat, obj_arti=arti,
        fps=demo.fps, frame_index=demo.frames(),
        source=f"canonical transfer {source_object.name} -> {target_object.name}")
