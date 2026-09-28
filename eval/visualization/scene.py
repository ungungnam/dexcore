"""Turn a `Demonstration` into a scene the browser can render without doing any kinematics.

THE SPLIT IS DELIBERATE: every convention stays on the Python side, where it is tested.

The browser receives, per frame, the WORLD TRANSFORM of each object part and each of the 32 hand
links -- already composed. It never sees a hinge axis, an articulation angle, a URDF, or a wxyz
quaternion. It multiplies nothing. A convention bug therefore cannot be introduced by the viewer,
because the viewer has no convention to get wrong.

Two conversions happen here, once, for that reason:
  * quaternions are emitted as **xyzw**, which is what three.js takes. Everything else in dexcore is
    wxyz (see src/geometry.py); this is the single crossing point.
  * the lid's hinge transform is baked into its per-frame world pose via `ArticulatedObject`.

Bulk arrays go out as base64 float32 rather than JSON numbers: an 889-frame clip is ~1.2 MB packed
against ~4 MB as text, and the browser gets typed arrays it can hand straight to three.js.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src import geometry as G
from src.data.demo import Demonstration
from src.data.hand import get_hand
from src.data.mano import LINK_NAMES, MANO_HAND_LINKS
from src.data.object import ArticulatedObject, get_object
from src.paths import BASE_PART_ID, LID_PART_ID, PART_NAME, SIDES

SCENE_FORMAT_VERSION = 1


def _f32(a) -> Dict[str, object]:
    """Pack an array as base64 float32 + its shape."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape),
            "dtype": "float32"}


def _u8(a) -> Dict[str, object]:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.uint8))
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape),
            "dtype": "uint8"}


def _u32(a) -> Dict[str, object]:
    a = np.ascontiguousarray(np.asarray(a, dtype=np.uint32))
    return {"b64": base64.b64encode(a.tobytes()).decode("ascii"), "shape": list(a.shape),
            "dtype": "uint32"}


def _wxyz_to_xyzw(q) -> np.ndarray:
    """The one place dexcore's wxyz meets three.js's xyzw. See the module docstring."""
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)


def _pose_arrays(T: np.ndarray):
    """(N,4,4) world transforms -> (position (N,3), quaternion xyzw (N,4))."""
    return T[..., :3, 3], _wxyz_to_xyzw(G.R_to_wxyz(T[..., :3, :3]))


def skeleton_edges() -> List[List[int]]:
    """Bone pairs between the 21 MANO keypoints, derived from MANO_HAND_LINKS's joint spans."""
    edges = set()
    for name, idxs in MANO_HAND_LINKS:
        if name == "palm":
            for j in idxs[1:]:
                edges.add((idxs[0], j))          # wrist -> each finger base
        else:
            edges.add((idxs[0], idxs[1]))
    return [list(e) for e in sorted(edges)]


def source_reference(demo: Demonstration) -> Optional[dict]:
    """Where this demonstration came from, so a viewer can put the two side by side.

    A generated demo records `provenance.source_path` and, through `dexcore_demo`, the ABSOLUTE
    source frames it covers. That pair is enough to reopen the original at exactly the matching
    window without the user having to know which file or which frames -- which is the whole point:
    "load the target" should be sufficient to see the comparison.

    Returns None when this IS a source demo, or when the recorded path no longer resolves (a demo
    generated on another machine, or a source that has since moved). The viewer then shows one
    pane rather than inventing a comparison.
    """
    sp = demo.provenance.source_path
    if not sp:
        return None
    p = Path(sp)
    same_file = demo.path and Path(demo.path).resolve() == p.resolve() if p.exists() else False
    frames = demo.frames()
    return {
        "path": str(p),
        "available": p.exists() and not same_file,
        "sequence": demo.provenance.source_sequence,
        "object": demo.provenance.source_object,
        "start": int(frames.min()),
        "end": int(frames.max()) + 1,
    }

def selected_rows(demo: Demonstration) -> List[int]:
    """Which ROWS of this demo were selected keyframes, valid under any re-slicing.

    `provenance.selected_frames` are local indices into the window the selection was made in. Load
    the same file with a different `frame_start` and those indices point somewhere else entirely --
    the timeline ticks would silently land on the wrong frames, which is worse than having none.
    `source_frames` are ABSOLUTE, so they survive re-slicing; this maps them back to rows and drops
    any that fall outside the current window.
    """
    # `selector` empty means no selection ever happened. That matters because slicing a demo for
    # DISPLAY goes through `subset()`, which stamps every retained row as "selected" -- without
    # this guard, viewing the raw source over frames 30-230 marks all 200 of them as keyframes.
    if not demo.provenance.selector:
        return []
    src = demo.provenance.source_frames
    if not src:
        return []
    pos = {int(f): i for i, f in enumerate(demo.frames())}
    return sorted(pos[int(f)] for f in src if int(f) in pos)


def _mesh_payload(mesh) -> dict:
    return {"verts": _f32(np.asarray(mesh.vertices, dtype=np.float32)),
            "faces": _u32(np.asarray(mesh.faces, dtype=np.uint32).reshape(-1))}


def build_scene(demo: Demonstration, obj: Optional[ArticulatedObject] = None,
                samples_per_link: int = 8) -> dict:
    """A complete, self-contained scene for one demonstration.

    `samples_per_link` only affects the cached hand model; the viewer uses meshes, not samples.
    """
    obj = obj or get_object(demo.obj_name)
    T = demo.num_frames

    # ---- object: static part meshes + per-frame WORLD pose of each part (hinge already applied)
    parts = []
    for pid in (LID_PART_ID, BASE_PART_ID):
        Tw = obj.part_to_world_batch(pid, demo.obj_arti, demo.obj_pos, demo.obj_quat)
        pos, quat = _pose_arrays(Tw)
        parts.append({"id": int(pid), "name": PART_NAME[pid],
                      "mesh": _mesh_payload(obj.mesh(pid)),
                      "pos": _f32(pos), "quat": _f32(quat)})

    # ---- hands: static link meshes + per-frame WORLD pose of each link, plus keypoints & contacts
    hands = {}
    for side in SIDES:
        if not demo.configs.get(side):
            continue
        hand = get_hand(side, device="cpu", samples_per_link=samples_per_link)
        q = hand.fk.dofs_from_cfg(demo.configs[side], frames=T)
        frames = hand.link_frames(q)                                   # (T,16,4,4)
        pos, quat = _pose_arrays(frames)
        hands[side] = {
            "links": list(LINK_NAMES),
            "meshes": [_mesh_payload(hand.link_mesh(n)) for n in LINK_NAMES],
            "pos": _f32(pos), "quat": _f32(quat),
            "keypoints": _f32(demo.joints[side]),
            "contact": _u8(demo.contact_mask[side]),
            "contact_part": _u8(demo.contact_part[side]),
            "contact_pos": _f32(demo.contact_pos[side]),
        }

    # ---- a sensible initial camera: centred on the object, framed to the whole clip's extent
    allpts = [demo.obj_pos]
    for side in hands:
        allpts.append(demo.joints[side].reshape(-1, 3))
    pts = np.vstack(allpts)
    centre = (pts.min(0) + pts.max(0)) / 2
    radius = float(np.linalg.norm(pts.max(0) - pts.min(0)) / 2) + 0.15

    return {
        "format_version": SCENE_FORMAT_VERSION,
        "name": demo.name,
        "num_frames": T,
        "fps": float(demo.fps),
        "frame_index": [int(f) for f in demo.frames()],
        "object": {"name": obj.name, "parts": parts},
        "hands": hands,
        "skeleton": skeleton_edges(),
        "selected_frames": selected_rows(demo),
        "is_sparse": bool(demo.is_sparse),
        "provenance": demo.provenance.to_dict(),
        "summary": demo.summary(),
        "camera": {"centre": [float(c) for c in centre], "radius": radius},
        "source": source_reference(demo),
    }
