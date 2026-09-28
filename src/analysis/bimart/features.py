"""TACO sequences in BimArt's feature format: object BPS, contact distances, hand kp and dirvec.

THE ONE DECISION THAT SHAPES THIS PORT: WHICH FRAME IS CANONICAL.

ARCTIC has a single articulated object, and BimArt canonicalises into that object's frame with the
ARTICULATION LEFT IN. One part (the lid) therefore moves inside the canonical frame and gets its
BPS recomputed every frame; the other (the base) is fixed there and gets one BPS repeated. The
object channel carries the articulation state over time, and the hands are described relative to
the thing they manipulate.

TACO has two RIGID objects and no articulation. Canonicalising each into its own frame would leave
both BPS constant in time and the object channel would carry no motion at all -- the port would
silently throw away exactly what BimArt's object channel exists to encode. The roles are assigned
by WHICH OBJECT MOVES, measured rather than assumed: over all 4562 chunks the tool travels 3.95x
further and turns 3.91x more than the target, and it is the larger of the two in 95% of them.

    canonical frame  =  the TARGET's frame
    target  <->  ARCTIC's base    fixed in the canonical frame, one BPS repeated
    tool    <->  ARCTIC's lid     moves in the canonical frame, BPS recomputed per frame

The tool-target relative motion lands in the object channel, which is the closest analogue of what
articulation was doing. The choice was also checked against the thing that actually depends on it --
how predictable the hand is from the object trajectory. The target frame wins on every measurement:
wrist 0.494 against the tool frame's 0.524 and the world frame's 0.631, and under a
geometry-controlled comparison 0.753 against 0.800 and 0.977. In the tool frame the hand barely
moves -- it is gripping the tool -- so the motion that makes the action drains out of the channel
the model has to predict and into the one it is conditioned on.

GLOBAL STATE IS 8, NOT 7. ARCTIC packs [rot(3), relative trans(3), scale(1)]. Two objects means two
scales, so the vector is [target rot(3), target relative trans(3), target scale(1), tool scale(1)].
`global_state_dim` is a config key, so this costs a config line rather than a code change.

TWO SCALES COEXIST, as they do upstream. BPS runs on unit-sphere-normalised meshes; contact and
`dirvec` run on the ORIGINAL metric meshes, matching the metric canonical hand vertices. BimArt does
the same -- `obj_feature_preprocess` loads the normalised dictionary, `hand_feature_preprocess` loads
the raw one -- and mixing them up would put the direction vectors in the wrong units.

ONE SCALE FOR THE WHOLE SCENE, FITTED TO BOTH OBJECTS. ARCTIC's asset
(`mesh_dict_unit_sphere_085_no_recenter.npy`) carries a single scale per OBJECT, chosen so the union
of its two parts reaches radius 0.85; each part alone is inside that -- ketchup's lid only reaches
0.211 -- so neither part can ever fall outside the basis. TACO's two objects are the analogue of
those two parts, so the scale has to be fitted to the pair, not to one of them.

An earlier version of this file scaled both objects by the TARGET's factor. That broke the analogy
and, with it, the contact label: a bowl is small, so its factor is large, the tool was blown up past
the basis, and its handle was nearest to no basis point. Measured against the dense per-vertex truth
the tool's contact fraction came out 0.204 too low on average, and in the worst cases 0.02 where the
truth was 0.71 -- the label said the tool was untouched while the hand held it -- for every target
category, worst where the target is small. Gathering the label somewhere else instead (per-mesh
farthest-point samples) restores coverage but costs more than it buys: the BPS indices are anchored
to a PLACE, not to the object -- the tool's slot changes vertex on 35% of frame steps -- so the label
is spatially registered with the BPS input, and breaking that registration made the hand error worse
on every split (test_1 39 -> 52 mm, test_3 70 -> 117 mm) even though the label itself became almost
exact. See VERDICT_label_fix.json. Scaling the scene as a whole fixes the cause rather than the
symptom: it is a uniform scaling, so the gap between the objects is undistorted, the tool is inside
the basis, and contact stays co-indexed with the input.

CONTACT IS the distance from each OBJECT vertex to its nearest HAND vertex -- the opposite direction
from `action_structure.contact`, which asks which object points are near the hand. Same geometry,
different indexing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

#: BimArt's own basis, 512 points on/in the unit sphere.
DEFAULT_BPS = Path("/home/uhnam/workspace/dexcore/third_party/BimArt/assets/"
                   "bps_normalized_part_based.npy")
#: Hand vertex subset BimArt predicts, 100 per hand, farthest-point sampled per MANO part. The
#: indices are a property of MANO's topology, not of ARCTIC, so they port unchanged.
DEFAULT_HAND_INDEX = Path("/home/uhnam/workspace/dexcore/third_party/BimArt/assets/"
                          "part_fps_hand_index_100.npy")
GLOBAL_STATE_DIM = 8
#: ARCTIC normalises each object so the union of its parts reaches this radius.
BASIS_RADIUS = 0.85


def load_basis(path=None) -> np.ndarray:
    return np.load(path or DEFAULT_BPS)


def load_hand_index(path=None) -> np.ndarray:
    return np.load(path or DEFAULT_HAND_INDEX).reshape(-1)


# --------------------------------------------------------------------------------- canonical
def to_canonical(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(T,N,3) world points -> the frame given by per-frame rotation `R` and translation `t`."""
    return np.einsum("tji,tnj->tni", R, points - t[:, None, :])


# --------------------------------------------------------------------------------- BPS + contact
def bps_from_points(points: np.ndarray, basis: np.ndarray):
    """-> (deltas (T,B,3), indices (T,B)). BimArt's BPS is the VECTOR to the nearest point, not a
    scalar distance, which is why `bps_feature_dim` is 3."""
    from scipy.spatial import cKDTree

    deltas = np.empty((len(points), len(basis), 3))
    inds = np.empty((len(points), len(basis)), dtype=np.int64)
    for t in range(len(points)):
        d, i = cKDTree(points[t]).query(basis, k=1, workers=-1)
        deltas[t] = points[t][i] - basis
        inds[t] = i
    return deltas, inds


def bps_static(points: np.ndarray, basis: np.ndarray, frames: int):
    """One BPS, repeated. For the object that does not move in the canonical frame."""
    from scipy.spatial import cKDTree

    d, i = cKDTree(points).query(basis, k=1, workers=-1)
    delta = points[i] - basis
    return np.repeat(delta[None], frames, axis=0), np.repeat(i[None], frames, axis=0)


def contact_distance(obj_verts_cano: np.ndarray, hand_verts_cano: np.ndarray) -> np.ndarray:
    """(T,N) distance from every object vertex to the nearest hand vertex, both canonical."""
    from scipy.spatial import cKDTree

    out = np.empty((len(obj_verts_cano), obj_verts_cano.shape[1]))
    for t in range(len(obj_verts_cano)):
        out[t] = cKDTree(hand_verts_cano[t]).query(obj_verts_cano[t], k=1, workers=-1)[0]
    return out


# ------------------------------------------------------------------------------- hand features
def dirvec(hand_pts: np.ndarray, obj_pts: np.ndarray) -> np.ndarray:
    """(T,K,3) vector from each hand keypoint to its nearest object point, canonical frame.

    BimArt predicts this alongside the keypoints so that contact and penetration are explicit in
    the output rather than implied by two independent position streams.
    """
    from scipy.spatial import cKDTree

    out = np.empty_like(hand_pts)
    for t in range(len(hand_pts)):
        _, i = cKDTree(obj_pts[t]).query(hand_pts[t], k=1, workers=-1)
        out[t] = obj_pts[t][i] - hand_pts[t]
    return out


def global_states(traj, anchor_scale: float, moving_scale: float) -> np.ndarray:
    """(T,8) [anchor rot (axis-angle, 3), anchor translation relative to frame 0 (3), two scales].

    `anchor_scale` is the SCENE's normalisation factor -- the analogue of ARCTIC's single per-object
    scale -- and `moving_scale` the tool mesh's own, which keeps the tool's true size recoverable.

    The anchor is the target -- the object whose frame is canonical -- so this is the pose of the
    coordinate system itself, exactly as ARCTIC reports the pose of its single object.
    """
    from scipy.spatial.transform import Rotation

    from src import geometry as G

    anchor = traj.target
    rot = Rotation.from_matrix(G.wxyz_to_R(anchor.quat)).as_rotvec()
    rel = anchor.pos - anchor.pos[0]
    n = len(rot)
    return np.concatenate([rot, rel,
                           np.full((n, 1), anchor_scale), np.full((n, 1), moving_scale)], axis=1)


# ------------------------------------------------------------------------------ the whole thing
def sequence_features(traj, mesh_dict: Dict[str, dict], basis: np.ndarray,
                      hand_index: np.ndarray, layers: Dict) -> Dict[str, object]:
    """Every feature BimArt's two datasets read, for one TACO sequence.

    Returns one dict; the caller splits it into the object file and the hand file if it wants to
    mirror BimArt's on-disk layout exactly.
    """
    from src import geometry as G
    from src.analysis.action_structure import contact as C

    tool, targ = traj.tool, traj.target
    mt, mo = mesh_dict[tool.name], mesh_dict[targ.name]
    T = traj.num_frames
    R_tool, p_tool = G.wxyz_to_R(tool.quat), tool.pos
    R_targ, p_targ = G.wxyz_to_R(targ.quat), targ.pos
    # ---- objects in the canonical (target) frame
    # metric first; the normalised copy is the same points times the SCENE's factor
    tool_world = np.einsum("tij,nj->tni", R_tool, mt["verts_original"]) + p_tool[:, None, :]
    tool_cano = to_canonical(tool_world, R_targ, p_targ)                     # (T,N,3) metres, moves
    targ_cano = np.repeat(mo["verts_original"][None], T, axis=0)             # fixed by definition

    # one scale for the pair, as ARCTIC fits one scale to an object's two parts: the union reaches
    # BASIS_RADIUS and neither object can fall outside the basis. No recentring, matching
    # `mesh_dict_unit_sphere_085_no_recenter`.
    r_scene = max(float(np.linalg.norm(tool_cano.reshape(-1, 3), axis=1).max()),
                  float(np.linalg.norm(mo["verts_original"], axis=1).max()))
    s_scene = BASIS_RADIUS / r_scene

    bps_tool, ind_tool = bps_from_points(tool_cano * s_scene, basis)         # moving part first,
    bps_targ, ind_targ = bps_static(mo["verts_original"] * s_scene, basis, T)  # as ARCTIC orders it
    obj_bps = np.concatenate([bps_tool, bps_targ], axis=1)                   # (T, 1024, 3)
    obj_inds = np.concatenate([ind_tool, ind_targ + len(tool_cano[0])], axis=1)

    # ---- hands
    hand_v, hand_cano, sampled_cano, contact = {}, {}, {}, {}
    for side in ("left", "right"):
        skin, _ = C.hand_geometry(traj.hands[side], layers[side])            # (T,778,3) world
        hand_v[side] = skin
        hand_cano[side] = to_canonical(skin, R_targ, p_targ)                 # metres, canonical
        sampled_cano[side] = hand_cano[side][:, hand_index]                  # (T,100,3)

    obj_all_cano = np.concatenate([tool_cano, targ_cano], axis=1)            # metric, canonical
    for side in ("left", "right"):
        contact[side] = {"dist": contact_distance(obj_all_cano, hand_cano[side])}

    kp = np.concatenate([sampled_cano["left"], sampled_cano["right"]], axis=1)
    dv = dirvec(kp, obj_all_cano)

    return {
        "sequence_id": traj.sequence_id, "triplet": traj.action.raw,
        "tool_mesh": tool.name, "target_mesh": targ.name,
        "verb": traj.action.verb, "tool_cat": traj.action.tool, "target_cat": traj.action.target,
        "n_frames": T,
        "obj_cano_bps": obj_bps.astype(np.float32),
        "obj_cano_bps_inds": obj_inds.astype(np.int32),
        "obj_cano_verts_dense": obj_all_cano.astype(np.float32),
        "global_states": global_states(traj, s_scene, mt["scale"]).astype(np.float32),
        "contact_dict": {s: {"dist": contact[s]["dist"].astype(np.float32)} for s in contact},
        "kp": kp.reshape(T, -1).astype(np.float32),
        "dirvec": dv.reshape(T, -1).astype(np.float32),
        "left_hand_sampled_verts_cano": sampled_cano["left"].astype(np.float32),
        "right_hand_sampled_verts_cano": sampled_cano["right"].astype(np.float32),
    }
