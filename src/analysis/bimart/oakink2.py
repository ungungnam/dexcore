"""OakInk2 in BimArt's feature format.

The decisions below were made by the user on 2026-09-30 after a review of upstream BimArt, the TACO
port and the OakInk2 release; the evidence is in
/result/uhnam/dexcore/oakink2/30_bimart_port/review/ (DECISIONS.md, plan.md).

WHAT BECOMES BIMART'S "ONE OBJECT, TWO PARTS". Upstream BimArt reads one articulated ARCTIC object
with exactly two parts: part0 moves in the object's frame (the lid), part1 defines it (the base).
The TACO port maps its two objects onto those slots. OakInk2 has no fixed object count: a recording
is a long task cut into primitive segments, and each segment handles one, two or more objects. Only
segments that handle EXACTLY TWO objects fit upstream's structure, so the unit here is one such
segment (1,128 of 2,840). Jointed objects (a laptop's lid and base, a bottle's cap and body) are
separately tracked rigid parts in OakInk2, which is exactly upstream's two-part picture; distinct
objects (a beaker and a test tube) are the TACO picture. Both go through the same code.

WHICH PART IS WHICH. Where the names say so -- inside one object's part group, cap/lid/gate/lever/
drawer/handle move and body/base/cavity/frame/tube hold -- the names decide, as ARCTIC's parts.json
does. Elsewhere the object whose vertex centroid travels LESS over all of the pair's segments is
part1. That fallback has no upstream analogue (ARCTIC always has a semantic base); it follows the
TACO port, whose less-moving target is the frame. The role is fixed per PAIR, not per segment.

ONE SCALE PER PAIR. Upstream fits one constant per object so the union of its two parts reaches the
basis radius 0.85. The TACO port kept the union rule and fitted it per take. Here the union rule is
kept and held constant per object pair (the maximum over every segment of that pair), which is the
closest analogue of upstream's per-object constant and guarantees both parts are inside the basis in
every frame of every segment.

UNITS AND FRAMES. Meshes and object poses are metres, `v_world = R @ v + t`. The canonical frame is
part1's, per frame, with no recentring: part groups share a pivot frame, as ARCTIC's parts do. BPS
runs on the metric canonical points times the pair scale; contact and dirvec stay metric, matching
upstream, which uses the normalised mesh for BPS and the original-scale one for contact.

MANO. OakInk2's official layer is `rot_mode="quat", center_idx=0, use_pca=False,
flat_hand_mean=True`, poses as wxyz quaternions, `tsl` the wrist in world coordinates. The repo's
manopth layer (`use_pca=False, center_idx=0`, `flat_hand_mean=True` by default) reproduces it after a
wxyz -> axis-angle conversion; the dynamic-contact study verified 0.2-0.3 mm hand-object minima
during grips with exactly this setting.
"""
from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path("/data/uhnam/oakink2")
ANNO = ROOT / "anno_preview"
PROG = ROOT / "extracted/program/program_info"
EXT = ROOT / "extracted/program_extension/extension"
DESC = ROOT / "extracted/object_raw/obj_desc.json"
PART_TREE = ROOT / "extracted/object_affordance/object_part_tree.json"
MESHES = Path("/result/uhnam/dexcore/oakink2/00_assets/object_raw/align_ds")
PORT = Path("/result/uhnam/dexcore/oakink2/30_bimart_port")
SPLIT_DIR = PORT / "assets/split"

#: Mocap is 120 Hz; every OakInk2 study in this repo samples every 4th frame, i.e. 30 Hz.
STEP = 4
#: Frames trimmed at each end of a unit. Upstream uses 100 to skip ARCTIC's T-pose lead-in; OakInk2
#: has none, so the TACO port's 8 is kept.
BASE_FRAME = 8
PRED_HORIZON = 64
#: A window is dropped when a hand comes this close to a tracked object outside the pair (B5).
NONPAIR_TOUCH_M = 0.005

PART0_WORDS = ("cap", "lid", "gate", "lever", "drawer", "handle")
PART1_WORDS = ("body", "base", "cavity", "frame", "tube")


# ------------------------------------------------------------------------------------ the split
def load_split() -> Dict[str, str]:
    """Recording name (the pkl stem, `++` form) -> "train" / "val" / "test".

    TaMF's official split (oakink/OakInk2-TaMF `asset/split`), stored under SPLIT_DIR. The files use
    `scene/seq` where the release uses `scene++seq`.
    """
    out = {}
    for split in ("train", "val", "test"):
        for line in (SPLIT_DIR / f"{split}.txt").read_text().split():
            out[line.strip().replace("/", "++")] = split
    return out


# -------------------------------------------------------------------------------- the segments
def _norm_key(k):
    """A program key as a hashable tuple: ((lh_start, lh_end) | None, (rh_start, rh_end) | None)."""
    raw = ast.literal_eval(k) if isinstance(k, str) else k
    return tuple(None if r is None else (int(r[0]), int(r[1])) for r in raw)


def extended_range(key, ext_entry: Optional[dict]):
    """(lo, hi) mocap frames over both active hands, widened to each hand's annotated approach
    start and retreat end when `ext_entry` has them. With `ext_entry=None` it is the interaction
    range alone."""
    los, his = [], []
    for h, rng in enumerate(key):
        if rng is None:
            continue
        s, e = rng
        if ext_entry:
            ap = (ext_entry.get("approach") or [None, None])[h]
            rt = (ext_entry.get("retreat") or [None, None])[h]
            if ap:
                s = int(ap[0])
            if rt:
                e = int(rt[1])
        los.append(s)
        his.append(e)
    return min(los), max(his)


def n_frames(lo: int, hi: int) -> int:
    return len(range(lo, hi, STEP))


def n_train_windows(frames: int) -> int:
    """Stride-1 windows after trimming BASE_FRAME at both ends, as upstream's MotionDataset does."""
    return max(0, frames - PRED_HORIZON - 2 * BASE_FRAME)


def n_test_windows(frames: int) -> int:
    """Non-overlapping windows under upstream's test rule: start at BASE_FRAME, step PRED_HORIZON,
    keep while start + PRED_HORIZON < end, with end = frames - PRED_HORIZON - BASE_FRAME."""
    end = frames - PRED_HORIZON - BASE_FRAME
    s, n = BASE_FRAME, 0
    while s + PRED_HORIZON < end:
        n += 1
        s += PRED_HORIZON
    return n


def semantic_role(name: str) -> Optional[int]:
    """0 if the name marks a moving part, 1 if it marks a base, None if it names neither.

    The LAST role word decides, because an English compound's head noun comes last: a "drawer
    frame" is a frame (the base the drawer slides in), a "microwave oven gate lever" is a lever.
    """
    n = name.lower()
    hits = [(m.start(), 0) for w in PART0_WORDS for m in re.finditer(rf"\b{w}\b", n)]
    hits += [(m.start(), 1) for w in PART1_WORDS for m in re.finditer(rf"\b{w}\b", n)]
    return max(hits)[1] if hits else None


def segments(two_objects_only: bool = True) -> List[dict]:
    """Every primitive segment, one row each. With `two_objects_only`, only those handling exactly
    two objects -- the unit this port trains on."""
    split = load_split()
    desc = json.load(open(DESC))
    tree = json.load(open(PART_TREE))
    parent = {c: p for p, cs in tree.items() for c in cs}
    rows = []
    for pf in sorted(PROG.glob("*.json")):
        rec = pf.stem
        prog = json.load(open(pf))
        ext_file = EXT / pf.name
        ext = {_norm_key(k): v for k, v in json.load(open(ext_file)).items()} if ext_file.exists() else {}
        for k, v in prog.items():
            key = _norm_key(k)
            objs = sorted(set(v.get("obj_list") or []))
            if two_objects_only and len(objs) != 2:
                continue
            lo, hi = extended_range(key, ext.get(key))
            ilo, ihi = extended_range(key, None)
            n = n_frames(lo, hi)
            row = dict(recording=rec, key=k, lo=lo, hi=hi, lo_int=ilo, hi_int=ihi,
                       n_frames=n, n_train_windows=n_train_windows(n), n_test_windows=n_test_windows(n),
                       hands="bh" if key[0] and key[1] else ("lh" if key[0] else "rh"),
                       primitive=v.get("primitive"), primitive_lh=v.get("primitive_lh"),
                       primitive_rh=v.get("primitive_rh"), interaction_mode=v.get("interaction_mode"),
                       has_extension=key in ext, split=split.get(rec))
            if len(objs) == 2:
                a, b = objs
                row.update(obj_a=a, obj_b=b,
                           name_a=desc.get(a, {}).get("obj_name", a), name_b=desc.get(b, {}).get("obj_name", b),
                           part_group=bool(parent.get(a)) and parent.get(a) == parent.get(b))
            else:
                row.update(objects=";".join(objs))
            rows.append(row)
    return rows


def name_roles(name_a: str, name_b: str, part_group: bool) -> Optional[str]:
    """Which member of a pair is part0 by name: "a", "b", or None when the names do not decide.

    Names decide only inside one object's part group; a distinct pair has no semantic base. A group
    of two has one part of each kind, so when only one name carries a role word the other part takes
    the complement: "alcohol burner cap" moves, so "alcohol burner" is its base. Two parts with the
    same role word (a gate and its lever) or identical names (the two halves of tongs) stay None.
    """
    if not part_group:
        return None
    ra, rb = semantic_role(name_a), semantic_role(name_b)
    if ra is not None and rb is None:
        rb = 1 - ra
    elif rb is not None and ra is None:
        ra = 1 - rb
    if ra == 0 and rb == 1:
        return "a"
    if ra == 1 and rb == 0:
        return "b"
    return None


# ------------------------------------------------------------------------------------- meshes
def load_mesh(obj_id: str):
    """(verts (N,3) metres, faces (F,3)). `process=False`, as upstream's `load_mesh_dict` does, so
    the vertex order is the file's own."""
    import trimesh

    d = MESHES / obj_id
    files = sorted(d.glob("*.ply")) + sorted(d.glob("*.obj"))
    if not files:
        raise FileNotFoundError(f"no mesh for {obj_id} under {d}")
    m = trimesh.load(files[0], process=False, force="mesh")
    return np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces, dtype=np.int64)


# -------------------------------------------------------------------------- recording cache
def load_recording(rec: str, frames: Iterable[int]) -> dict:
    """Hands and every object's pose at the given mocap frames, from one annotation pickle.

    -> frames (F,), {lh,rh}_pose (F,16,4) wxyz, {lh,rh}_tsl (F,3), {lh,rh}_betas (10,),
       obj_ids (n,), obj_T (n,F,4,4). Frames outside the recording are dropped and logged.
    """
    import pickle

    import torch  # noqa: F401 -- the pickles hold torch tensors

    with open(ANNO / f"{rec}.pkl", "rb") as fh:
        d = pickle.load(fh)
    have = set(int(f) for f in d["mocap_frame_id_list"])
    want = sorted(set(int(f) for f in frames))
    fids = [f for f in want if f in have]
    if len(fids) != len(want):
        log.warning("%s: %d requested frames are outside the recording", rec, len(want) - len(fids))
    mano = d["raw_mano"]

    def stack(key, shape):
        return np.stack([np.asarray(mano[f][key].detach().cpu().numpy()).reshape(shape) for f in fids])

    out = {"frames": np.asarray(fids, dtype=np.int64)}
    for side in ("lh", "rh"):
        out[f"{side}_pose"] = stack(f"{side}__pose_coeffs", (16, 4)).astype(np.float32)
        out[f"{side}_tsl"] = stack(f"{side}__tsl", (3,)).astype(np.float32)
        out[f"{side}_betas"] = np.asarray(mano[fids[0]][f"{side}__betas"].detach().cpu().numpy()).reshape(10)
    obj_ids = list(d["obj_list"])
    out["obj_ids"] = np.asarray(obj_ids)
    out["obj_T"] = np.stack([np.stack([np.asarray(d["obj_transf"][o][f]) for f in fids])
                             for o in obj_ids]).astype(np.float64)
    return out


def frame_index(cache: dict, lo: int, hi: int) -> np.ndarray:
    """Positions in `cache["frames"]` of range(lo, hi, STEP)."""
    pos = {int(f): i for i, f in enumerate(cache["frames"])}
    want = list(range(lo, hi, STEP))
    missing = [f for f in want if f not in pos]
    if missing:
        raise KeyError(f"{len(missing)} frames of [{lo},{hi}) not cached (first {missing[0]})")
    return np.asarray([pos[f] for f in want], dtype=np.int64)


def obj_pose(cache: dict, obj_id: str, idx: np.ndarray):
    """(R (T,3,3), t (T,3)) of one object at the cached positions `idx`."""
    k = int(np.flatnonzero(cache["obj_ids"] == obj_id)[0])
    T = cache["obj_T"][k][idx]
    return T[:, :3, :3], T[:, :3, 3]


# --------------------------------------------------------------------------------- the hands
def mano_layers(device: str = "cpu"):
    from src.analysis.action_structure import contact as C

    return {"lh": C.mano_layer("left", device), "rh": C.mano_layer("right", device)}


def mano_verts(layer, pose_wxyz: np.ndarray, tsl: np.ndarray, betas: np.ndarray,
               device: str = "cpu") -> np.ndarray:
    """(T,16,4) wxyz quaternions, (T,3) wrist position, (10,) betas -> (T,778,3) world metres.

    Same conversion as scripts/research/canonical_contact/dyn/build_oakink2_takes.py: the layer's
    wrist is at the origin (`center_idx=0`), so translating by `tsl` places the hand.
    """
    import torch
    from scipy.spatial.transform import Rotation

    T = len(pose_wxyz)
    aa = Rotation.from_quat(pose_wxyz[..., [1, 2, 3, 0]].reshape(-1, 4)).as_rotvec().reshape(T, 16, 3)
    theta = torch.tensor(aa.reshape(T, 48), dtype=torch.float32, device=device)
    beta = torch.tensor(np.repeat(betas[None], T, 0), dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, T, 2048):
            v, _ = layer(theta[i:i + 2048], beta[i:i + 2048])[:2]
            out.append(v.cpu().numpy())
    return np.concatenate(out) / 1000.0 + tsl[:, None, :]


# --------------------------------------------------------------------------- per-pair geometry
def part_radius(cache: dict, mesh_dict: dict, part0: str, part1: str, lo: int, hi: int):
    """Largest distance from part1's origin, in part1's frame, reached by each part over a segment:
    -> (r0 (T,), r1 scalar). The union radius is max(r0.max(), r1)."""
    from src.analysis.bimart.features import to_canonical

    idx = frame_index(cache, lo, hi)
    R0, t0 = obj_pose(cache, part0, idx)
    R1, t1 = obj_pose(cache, part1, idx)
    w0 = np.einsum("tij,nj->tni", R0, mesh_dict[part0]["verts"]) + t0[:, None, :]
    c0 = to_canonical(w0, R1, t1)
    return np.linalg.norm(c0, axis=2).max(1), float(np.linalg.norm(mesh_dict[part1]["verts"], axis=1).max())


def centroid_travel(cache: dict, mesh_dict: dict, obj_id: str, lo: int, hi: int) -> float:
    """Path length of an object's posed vertex centroid over a segment, metres. The mesh origin can
    sit far from the geometry (part groups share a pivot), so the centroid, not the origin, moves."""
    idx = frame_index(cache, lo, hi)
    R, t = obj_pose(cache, obj_id, idx)
    c = np.einsum("tij,j->ti", R, mesh_dict[obj_id]["verts"].mean(0)) + t
    return float(np.linalg.norm(np.diff(c, axis=0), axis=1).sum())


def canonical_parts(cache: dict, mesh_dict: dict, part0: str, part1: str, lo: int, hi: int):
    """(c0 (T,N0,3), c1 (N1,3)) metric points of both parts in part1's frame."""
    from src.analysis.bimart.features import to_canonical

    idx = frame_index(cache, lo, hi)
    R0, t0 = obj_pose(cache, part0, idx)
    R1, t1 = obj_pose(cache, part1, idx)
    c0 = to_canonical(np.einsum("tij,nj->tni", R0, mesh_dict[part0]["verts"]) + t0[:, None, :], R1, t1)
    return c0, mesh_dict[part1]["verts"]


def bps_indices(cache: dict, mesh_dict: dict, part0: str, part1: str, lo: int, hi: int,
                scale: float, basis: np.ndarray) -> np.ndarray:
    """(T,1024) BPS indices under a given scale, without the hands. For comparing scale choices:
    the dense contact does not depend on the scale, only which vertices the slots land on does."""
    from src.analysis.bimart import features as F

    c0, v1 = canonical_parts(cache, mesh_dict, part0, part1, lo, hi)
    _, ind0 = F.bps_from_points(c0 * scale, basis)
    _, ind1 = F.bps_static(v1 * scale, basis, len(c0))
    return np.concatenate([ind0, ind1 + len(mesh_dict[part0]["verts"])], axis=1)


# ---------------------------------------------------------------------------- one segment
def segment_features(cache: dict, mesh_dict: dict, part0: str, part1: str, lo: int, hi: int,
                     scale: float, basis: np.ndarray, hand_index: np.ndarray, layers: dict,
                     keep_dense: bool = False, nonpair: bool = True) -> Dict[str, np.ndarray]:
    """Every feature BimArt's two datasets read, for one segment.

    Layout mirrors the TACO port and upstream: slots [0:512] are part0 (BPS recomputed every
    frame), [512:1024] part1 (one BPS, repeated); part1's indices are offset by part0's vertex
    count, so they index the concatenated [part0, part1] vertex axis the contact is computed on.
    `global_abs` is [part1 rotvec (3), part1 ABSOLUTE translation (3), pair scale (1)]; the
    dataset subtracts its own reference frame's translation, as upstream does.
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    from src.analysis.bimart import features as F

    idx = frame_index(cache, lo, hi)
    T = len(idx)
    R0, t0 = obj_pose(cache, part0, idx)
    R1, t1 = obj_pose(cache, part1, idx)
    v0, v1 = mesh_dict[part0]["verts"], mesh_dict[part1]["verts"]
    c0 = F.to_canonical(np.einsum("tij,nj->tni", R0, v0) + t0[:, None, :], R1, t1)   # (T,N0,3) m
    c1 = np.repeat(v1[None], T, axis=0)                                                  # fixed

    bps0, ind0 = F.bps_from_points(c0 * scale, basis)          # moving part first, as ARCTIC
    bps1, ind1 = F.bps_static(v1 * scale, basis, T)
    obj_bps = np.concatenate([bps0, bps1], axis=1)
    inds = np.concatenate([ind0, ind1 + len(v0)], axis=1)

    world, cano = {}, {}
    for side in ("lh", "rh"):
        world[side] = mano_verts(layers[side], cache[f"{side}_pose"][idx], cache[f"{side}_tsl"][idx],
                                 cache[f"{side}_betas"])
        cano[side] = F.to_canonical(world[side], R1, t1)
    kp = np.concatenate([cano["lh"][:, hand_index], cano["rh"][:, hand_index]], axis=1)   # (T,200,3)

    obj_all = np.concatenate([c0, c1], axis=1)
    dense = {s: F.contact_distance(obj_all, cano[s]) for s in ("lh", "rh")}
    rt = np.arange(T)[:, None]
    label = np.concatenate([dense["lh"][rt, inds], dense["rh"][rt, inds]], axis=1)       # (T,2048)
    dv = F.dirvec(kp, obj_all)

    rot = Rotation.from_matrix(R1).as_rotvec()
    gabs = np.concatenate([rot, t1, np.full((T, 1), scale)], axis=1)

    out = {"obj_cano_bps": obj_bps.astype(np.float32), "obj_cano_bps_inds": inds.astype(np.int32),
           "contact": label.astype(np.float32), "global_abs": gabs.astype(np.float64),
           "kp": kp.reshape(T, -1).astype(np.float32), "dirvec": dv.reshape(T, -1).astype(np.float32),
           "frames": cache["frames"][idx], "n_part0": np.int64(len(v0))}

    if nonpair:
        # each hand's closest approach to every tracked object outside the pair, per frame (B5).
        # Kept per object, so the filter can be restated (e.g. part-tree siblings) without
        # recomputing. The closest pair of points is symmetric, so one tree per hand per frame
        # is queried with every other object's vertices at once.
        others = [o for o in cache["obj_ids"] if o not in (part0, part1)]
        missing = [o for o in others if o not in mesh_dict]
        if missing:
            raise KeyError(f"tracked non-pair objects without a mesh: {missing}")
        near = np.full((T, 2, len(others)), np.inf)
        if others:
            poses = [obj_pose(cache, o, idx) for o in others]
            verts = [mesh_dict[o]["verts"] for o in others]
            starts = np.cumsum([0] + [len(v) for v in verts[:-1]])
            for f in range(T):
                wo = np.concatenate([v @ R[f].T + t[f] for v, (R, t) in zip(verts, poses)])
                for j, side in enumerate(("lh", "rh")):
                    near[f, j] = np.minimum.reduceat(cKDTree(world[side][f]).query(wo, k=1)[0], starts)
        out["nonpair_dist"] = near.astype(np.float32)                      # (T,2,n_other) m
        out["nonpair_obj_ids"] = np.array(others, dtype="U32")
        out["nonpair_min_dist"] = near.min(axis=2, initial=np.inf).astype(np.float32)   # (T,2)
        out["n_nonpair_objects"] = np.int64(len(others))

    if keep_dense:
        out["contact_dense_lh"] = dense["lh"].astype(np.float32)
        out["contact_dense_rh"] = dense["rh"].astype(np.float32)
        out["hand_world_lh"] = world["lh"].astype(np.float32)
        out["hand_world_rh"] = world["rh"].astype(np.float32)
    return out
