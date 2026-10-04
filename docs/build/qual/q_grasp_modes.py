#!/usr/bin/env python
"""Qualitative figure "grasp_modes" (page block F1-B): what an initial grasp mode is.

Upper two rows: the recorded hand on the ARCTIC scissors at the first contact frame of ten test
sequences, with the recorded first contact map on the surface. Lower two rows: ten first contact maps
sampled from the model that is given only the object's shape and which hand, for the first of these
sequences. One camera and one colour scale for all twenty panels; the sampled panels show the horizontal
band of that same view in which the closed scissors lie (the rest of the view is empty without a hand).

Selection (computed here, nothing is picked by eye): the rule of figure 7 of the hierarchical report
(scripts/research/hier_contact_gen/figures.py::fig7): the (group, mesh) with the most test sequences,
its first ten test sequences in file order, and the ten saved samples of the first of them in stored
order. On ARCTIC this is the group "scissors, left hand".

Two things the picture cannot say by itself are measured here and written into the caption and meta.json:
the 512-value map is defined on the closed scissors, so contact on one blade is also coloured on the other
(the distance of each half of the scissors from the drawn hand is computed per panel), and only the hand
whose contact the map records is drawn (the distance of the other hand to the surface is computed per panel).

Data only: the samples are read from the study's saved predictions (eval/fixed0/preds.npz, S0_G); no
model is loaded. Run as

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_grasp_modes.py
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import qlib

log = logging.getLogger("q_grasp_modes")

FIG_ID = "grasp_modes"
BLOCK = "F1-B"
DATASET = "arctic"
N_SHOW = 10                 # sequences shown, and samples per sequence saved by the study
N_COLS = 5                  # panels per row: ten panels wrap into two rows of five
PANEL_WIDTH = 440           # pixels
TILT_DEG = 25.0             # camera: angle between the view direction and the hinge axis
CAMERA_MARGIN = 0.05
BAND_PAD = 0.08             # sampled panels: free space above and below the object, as a share of the panel height
SPLIT_SHARE = 0.25          # a map with less than this share on the blade side is counted as "on the handle side"
VMAX_STEP = 0.1             # the colour scale ends at the next multiple of this above the largest drawn value
FAR_MM = 30.0               # a half of the scissors farther than this from the drawn hand has recorded contact below exp(-1.5) = 0.22
STRONG_SHARE = 0.5          # "strongly coloured": drawn at this share of the panel's largest value, or more
HOLD_MM = 2.0               # the hand that is not drawn counts as holding the object when its mesh is this close to the surface
NEAR_MM = 5.0               # hand vertices / surface vertices within this distance are counted as "at the surface"
HANDLE_SHARE = 0.95         # "holding the handles": at least this share of the surface at that hand lies behind the hinge
OTHER_HAND = {"left": "right", "right": "left"}
HINGE_AXIS = np.array([0.0, 0.0, 1.0])   # ARCTIC object frame: the top part turns about z through the origin
LONG_AXIS = np.array([1.0, 0.0, 0.0])    # the closed scissors lie along x
MID_AXIS = np.array([0.0, 1.0, 0.0])

ROOT = qlib.HIER[DATASET]
PREDS = ROOT / "eval/fixed0/preds.npz"
PER_EXAMPLE = ROOT / "eval/fixed0/per_example.csv"
META_CSV = ROOT / "sequences_meta.csv"
MANIFEST = ROOT / "manifests/fixed0.json"
SEQUENCES = ROOT / "sequences.npz"
MESH_DICT = qlib.ARCTIC_PROBE / "assets/arctic_mesh_dict.npy"
SAMPLER_CKPT = qlib.RESULT_ROOT / "hier_contact_gen_ckpt" / DATASET / "fixed0/sampler_G.pt"   # named in meta, not loaded


# ------------------------------------------------------------------------------ selection
@dataclass(frozen=True)
class Selection:
    group: str
    mesh_id: str
    rows: np.ndarray            # rows of the prediction file that are drawn (first N_SHOW of the group, file order)
    examples: np.ndarray        # the same sequences as rows of sequences_meta.csv
    group_rows: np.ndarray      # every test row of the group
    ranking: list[dict[str, Any]]


def select(pred_examples: np.ndarray, meta: pd.DataFrame) -> Selection:
    """Figure 7 of the hierarchical report: the (group, mesh) with the most test sequences, first ten in file order."""
    test = meta.iloc[pred_examples]
    counts = test.groupby(["group", "mesh_id"]).size()
    group, mesh_id = counts.idxmax()
    group_rows = np.flatnonzero((test.group == group).to_numpy() & (test.mesh_id == mesh_id).to_numpy())
    ranked = counts.sort_values(ascending=False, kind="stable")
    ranking = [{"rank": i + 1, "group": g, "mesh_id": m, "test_sequences": int(c)} for i, ((g, m), c) in enumerate(ranked.items())]
    rows = group_rows[:N_SHOW]
    return Selection(group, mesh_id, rows, pred_examples[rows], group_rows, ranking)


# ------------------------------------------------------------------------------ geometry helpers (not in qlib)
def closed_mesh(example: qlib.Example) -> tuple[np.ndarray, int]:
    """Vertices of the object's closed rest pose (the pose in which the 512 canonical points are defined), in
    qlib's vertex order: the turning (top) part first; and the number of vertices of that top part.
    Checked against Example.mesh."""
    parts = np.load(MESH_DICT, allow_pickle=True).item()
    top = np.asarray(parts[f"{example.meta.category}_top"]["verts_original"], float)
    bottom = np.asarray(parts[f"{example.meta.category}_bottom"]["verts_original"], float)
    turned = top @ Rotation.from_rotvec(-example.articulation(0) * HINGE_AXIS).as_matrix().T
    if not np.allclose(np.concatenate([turned, bottom]), example.mesh(0)[0], atol=1e-9):
        raise RuntimeError("the closed mesh turned by the recorded angle does not reproduce Example.mesh(0)")
    return np.concatenate([top, bottom]), len(top)


@lru_cache(maxsize=None)
def take_files(category: str, sequence_id: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """One ARCTIC take, read-only: the object state per frame ([opening, rotvec, translation]) and the world
    vertices of BOTH hands. qlib.Example.hand gives the contacting hand only; the other hand is needed here to
    say where it is, although it is not drawn."""
    stem, subject = sequence_id.split("/")
    base = qlib.ARCTIC_PROC / category / subject
    state = np.load(base / f"{stem}_processed_obj_features.npy", allow_pickle=True).item()["obj_world_state"].astype(np.float64)
    hands = np.load(base / f"{stem}_processed_hand_features.npy", allow_pickle=True).item()
    return state, {side: hands[f"{side}_hand_verts"] for side in OTHER_HAND}


def hand_in_object_frame(example: qlib.Example, side: str, t: int) -> np.ndarray:
    """(778, 3) vertices of the ``side`` ("left" / "right") hand at sequence frame t in the object's rigid frame,
    by qlib's own transform R^T (x - p)."""
    state, hands = take_files(example.meta.category, example.meta.sequence_id)
    frame = example.take_frame(t)
    rot = Rotation.from_rotvec(state[frame, 1:4]).as_matrix()
    return (np.asarray(hands[side][frame], np.float64) - state[frame, 4:7]) @ rot


def panel_geometry(example: qlib.Example, mesh: np.ndarray, hand: np.ndarray, drawn: np.ndarray, closed: np.ndarray,
                   n_top: int, blade_sign: float, dense_file: pd.Series) -> dict[str, Any]:
    """What one recorded panel does not show, measured at the drawn frame.

    (a) How far each half of the scissors (the turning top part, the bottom part) is from the drawn hand, and
        how strongly the half that is farther away is coloured: the map lives on the closed scissors, so a map
        point between the closed blades colours both.
    (b) Where the hand that is not drawn is.

    Distances are computed from the drawn geometry (nearest hand vertex per mesh vertex) and compared with the
    recorded distances of the dense contact files; ``mesh`` / ``hand`` / ``drawn`` are the arrays of the panel.
    """
    side = qlib.SIDE[example.meta.hand]
    other = OTHER_HAND[side]
    frame = example.take_frame(0)
    if not np.allclose(hand_in_object_frame(example, side, 0), hand, atol=1e-9):
        raise RuntimeError("the two-hand loader does not reproduce the hand that qlib draws")
    other_hand = hand_in_object_frame(example, other, 0)
    with np.load(qlib.ARCTIC_PROBE / "sequences" / dense_file[example.meta.sequence_id]) as z:
        recorded = {s: z[f"contact_{s}"][frame].astype(np.float64) for s in (side, other)}    # (V,) metres
    d_hand = cKDTree(hand).query(mesh)[0]
    d_other = cKDTree(other_hand).query(mesh)[0]
    top = np.arange(len(mesh)) < n_top
    part_mm = {"top": float(d_hand[top].min() * 1000), "bottom": float(d_hand[~top].min() * 1000)}
    far_part = max(part_mm, key=part_mm.get)
    far = top if far_part == "top" else ~top
    on_blade = np.sign(closed @ LONG_AXIS) == blade_sign          # per vertex, in the closed pose
    dense = np.exp(-recorded[side] / qlib.SOFT_SCALE)             # recorded per-vertex contact of the drawn hand
    strong = drawn >= STRONG_SHARE * np.nanmax(drawn)
    nearest_other = int(d_other.argmin())
    touched = d_other < NEAR_MM / 1000                            # surface vertices at the hand that is not drawn
    return {
        "frame": frame, "other_hand": other_hand, "d_hand": d_hand, "d_other": d_other, "part_mm": part_mm,
        "far_part": far_part, "far_mm": part_mm[far_part], "far_blade_mm": float(d_hand[far & on_blade].min() * 1000),
        "drawn_max": float(np.nanmax(drawn)), "drawn_max_far": float(np.nanmax(drawn[far])),
        "recorded_max_far": float(dense[far].max()),
        "strong_far_share": float((strong & (d_hand > FAR_MM / 1000)).sum() / strong.sum()),
        "correlation": float(np.corrcoef(drawn, dense)[0, 1]),
        "other_mm": float(d_other.min() * 1000),
        "other_recorded_mm": float(recorded[other].min() * 1000),
        "other_vertices_near": int((cKDTree(mesh).query(other_hand)[0] < NEAR_MM / 1000).sum()),
        "other_nearest_part": "top" if nearest_other < n_top else "bottom",
        # position of the nearest surface point along the closed scissors, from the hinge axis toward the handles
        "other_nearest_behind_hinge_mm": float(-blade_sign * (closed[nearest_other] @ LONG_AXIS) * 1000),
        "other_touched_on_handle_side": float((touched & ~on_blade).sum() / touched.sum()) if touched.any() else float("nan"),
        "check_hand_mm": float(np.abs(d_hand - recorded[side]).max() * 1000),
        "check_other_mm": float(np.abs(d_other - recorded[other]).max() * 1000),
    }


def camera_rule(closed: np.ndarray, hands: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """View direction (scene -> camera) and up vector, from the geometry alone.

    The scissors are thin along the hinge axis. The camera looks along that axis, from the side on which fewer
    recorded hand vertices lie, tilted by TILT_DEG away from the mean hand position, so that the thin object is
    drawn in front of the hands and its colours are not hidden. The closed scissors lie horizontally.
    """
    centre = closed.mean(0)
    all_hands = np.concatenate(hands)
    above = int(((all_hands - centre) @ HINGE_AXIS > 0).sum())
    below = len(all_hands) - above
    side = 1.0 if above <= below else -1.0
    toward_hands = 1.0 if (all_hands.mean(0) - centre) @ MID_AXIS >= 0 else -1.0
    up = toward_hands * MID_AXIS
    tilt = np.radians(TILT_DEG)
    direction = np.cos(tilt) * side * HINGE_AXIS - np.sin(tilt) * up
    _, axes = np.linalg.eigh(np.cov((closed - centre).T))          # columns: thinnest, middle, longest principal axis
    off = [float(np.degrees(np.arccos(min(1.0, abs(axes[:, j] @ a))))) for j, a in enumerate((HINGE_AXIS, MID_AXIS, LONG_AXIS))]
    evidence = {"hand_vertices_on_plus_z_side": above, "hand_vertices_on_minus_z_side": below,
                "camera_side_of_the_object": "+z" if side > 0 else "-z", "up": up.tolist(), "tilt_deg": TILT_DEG,
                "direction_scene_to_camera": direction.tolist(),
                "principal_axes_off_frame_axes_deg": {"thinnest_vs_hinge_z": round(off[0], 1), "middle_vs_y": round(off[1], 1),
                                                      "longest_vs_x": round(off[2], 1)}}
    return direction, up, evidence


def blade_side(closed: np.ndarray, canonical: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """(512,) bool: canonical points on the blade side of the hinge axis in the closed pose.

    The hinge axis is z through the origin of the object frame; the side of it on which the closed scissors are
    narrower is the blades (the handle rings are wide)."""
    centre = closed.mean(0)
    radius = np.linalg.norm(closed - centre, axis=1).max()
    points_m = canonical.astype(np.float64) * radius + centre       # canonical units -> metres, as in qlib.Example
    along = closed @ LONG_AXIS
    width = {s: float(np.ptp((closed @ MID_AXIS)[np.sign(along) == s])) for s in (-1.0, 1.0)}
    blade_sign = min(width, key=width.get)
    mask = np.sign(points_m @ LONG_AXIS) == blade_sign
    info = {"hinge_axis": "z axis through the origin of the object frame", "blade_side": "x > 0" if blade_sign > 0 else "x < 0",
            "blade_sign": blade_sign, "width_across_m": {"x<0": round(width[-1.0], 4), "x>0": round(width[1.0], 4)},
            "canonical_points_on_blade_side": int(mask.sum()), "canonical_points": int(len(mask)),
            "share": "sum of the map over the blade-side canonical points / sum over all 512 (values below 0 counted as 0)"}
    return mask, info


def blade_share(maps: np.ndarray, blade: np.ndarray) -> np.ndarray:
    """Share of a map's total (negative values counted as 0) that lies on the blade side of the hinge."""
    m = np.clip(np.asarray(maps, np.float64), 0.0, None)
    return m[..., blade].sum(-1) / m.sum(-1)


# ------------------------------------------------------------------------------ layout helpers (not in qlib)
def layout_font(panel_width: int) -> ImageFont.FreeTypeFont:
    """The font qlib.grid will use for a figure of N_COLS panels of this width with a row-label column."""
    probe = qlib.grid([[np.full((8, panel_width, 3), 255, np.uint8)] * N_COLS], row_labels=["x"])
    px = int(probe.info["qlib_layout"]["font_px"])
    path = next((p for p in qlib.FONT_PATHS if Path(p).exists()), None)
    if path is None:
        import matplotlib                                           # ships the same DejaVu Sans
        path = str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")
    return ImageFont.truetype(path, px)


def with_title(panel: np.ndarray, text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Put a one-line title in a white strip above a panel (grid has one label per column only, and the ten
    panels of a group wrap into two rows here, so each panel carries its own number)."""
    line = int(np.ceil(font.size * 1.32))
    if font.getlength(text) > panel.shape[1]:
        raise ValueError(f"panel title {text!r} is wider than the panel")
    strip = Image.new("RGB", (panel.shape[1], line + line // 4), "white")
    ImageDraw.Draw(strip).text((panel.shape[1] / 2, line // 8), text, font=font, fill=qlib.INK, anchor="ma")
    return np.concatenate([np.asarray(strip), panel])


def crop_band(images: list[np.ndarray], pad: int) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Cut the same horizontal band out of every image: the rows in which any of them draws something, plus
    ``pad`` rows on both sides. The camera is untouched; only empty rows are dropped."""
    drawn = np.flatnonzero(np.stack([(im < 250).any(axis=(1, 2)) for im in images]).any(0))
    y0, y1 = max(0, int(drawn[0]) - pad), min(images[0].shape[0], int(drawn[-1]) + 1 + pad)
    return [im[y0:y1] for im in images], (y0, y1)


def stack(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """Two grids of the same width, one above the other; the layout record of the first is kept."""
    if top.width != bottom.width or top.info["qlib_layout"]["font_px"] != bottom.info["qlib_layout"]["font_px"]:
        raise ValueError("the two grids differ in width or font size")
    canvas = Image.new("RGB", (top.width, top.height + bottom.height), "white")
    canvas.paste(top, (0, 0))
    canvas.paste(bottom, (0, top.height))
    canvas.info["qlib_layout"] = dict(top.info["qlib_layout"], height=canvas.height,
                                      panel_lower_grid=bottom.info["qlib_layout"]["panel"])
    return canvas


def span(v: list[int]) -> str:
    """[1, 2, 3, 4] -> '1 to 4'; [9, 10] -> '9 and 10'; anything else -> a comma list."""
    if len(v) > 2 and v == list(range(v[0], v[-1] + 1)):
        return f"{v[0]} to {v[-1]}"
    return " and ".join(map(str, v)) if len(v) == 2 else ", ".join(map(str, v))


def r(x: float, digits: int = 3) -> float:
    return round(float(x), digits)


def pct(x: float) -> str:
    return f"{100 * float(x):.0f}"


def pct_up(x: float) -> str:
    """Per cent rounded UP to one decimal, for "at most" statements (0.09256 -> '9.3', never '9')."""
    return f"{np.ceil(1000 * float(x) - 1e-9) / 10:.1f}"


def listed(values: list[str]) -> str:
    """['54', '32'] -> '54 and 32'; three or more -> 'a, b and c'."""
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


# ------------------------------------------------------------------------------ the figure
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    qlib.require_headless()

    # ---- selection
    with np.load(PREDS) as z:
        pred_examples = z["example"]
        gt_first = z["gt"][:, 0].astype(np.float32)                # (N, 512) recorded first maps of all test sequences
        samples_all = z["S0_G"].astype(np.float32)                 # (N, 10, 512) saved samples
    meta = pd.read_csv(META_CSV, dtype={"mesh_id": str, "subject": str, "sequence_id": str})
    manifest = json.loads(MANIFEST.read_text())
    if not np.array_equal(pred_examples, np.asarray(manifest["test"])):
        raise RuntimeError("the prediction file is not in the order of the test manifest")
    sel = select(pred_examples, meta)
    rows_meta = meta.iloc[sel.examples]
    log.info("selected group %s (%d test sequences), examples %s", sel.group, len(sel.group_rows), sel.examples.tolist())
    if samples_all.shape[1] != N_SHOW:
        raise RuntimeError(f"expected {N_SHOW} saved samples per sequence, found {samples_all.shape[1]}")
    samples = samples_all[sel.rows[0]]                             # (10, 512), stored order, for the first sequence

    train_rows = np.asarray(manifest["train"])
    train_rows = train_rows[(meta.group.to_numpy() == sel.group)[train_rows]]
    with np.load(SEQUENCES) as z:
        if list(z["state_names"])[0] != "arti":
            raise RuntimeError("the first object state is not the articulation")
        canonical_all = dict(zip(map(str, z["canonical_cats"]), z["canonical_points"]))
        static_inputs = np.stack([z[k][pred_examples[sel.group_rows]] for k in ("g_index", "hand", "role")], 1)
        train_first = z["C"][train_rows, 0].astype(np.float32)     # first maps of the group's training sequences
        opening_state = z["O"][sel.examples, qlib.PAD, 0].astype(np.float64)   # the study's own opening angle at frame 0
    same_sampler_input = bool((static_inputs == static_inputs[0]).all())

    # ---- data
    examples = [qlib.load_example(DATASET, int(n)) for n in sel.examples]
    first = examples[0]
    category, hand_word = first.meta.category, qlib.SIDE[first.meta.hand]
    if any(e.meta.category != category or e.meta.hand != first.meta.hand or not e.covered.all() for e in examples):
        raise RuntimeError("the ten sequences do not share one object and hand, or the map does not reach every vertex")
    for e, row in zip(examples, sel.rows):
        if not np.array_equal(e.C[0], gt_first[row]):
            raise RuntimeError(f"recorded first map of example {e.meta.example} differs between sequences.npz and preds.npz")
    closed, n_top = closed_mesh(first)
    faces = first.mesh(0)[1]
    blade, blade_info = blade_side(closed, canonical_all[category])

    recorded_512 = np.stack([e.C[0] for e in examples])            # (10, 512)
    recorded_vertex = np.stack([e.to_vertices(e.C[0]) for e in examples])
    sample_vertex = first.to_vertices(samples)                     # (10, V): one object, so one operator
    meshes = [e.mesh(0)[0] for e in examples]
    hands = [e.hand(0)[0] for e in examples]
    hand_faces = first.hand(0)[1]

    # ---- what the recorded panels do not show: the half of the scissors far from the hand, and the other hand
    other_word = OTHER_HAND[hand_word]
    dense_file = pd.read_csv(qlib.ARCTIC_PROBE / "sequence_index.csv").set_index("sequence_id")["file"]
    geo = [panel_geometry(e, meshes[i], hands[i], recorded_vertex[i], closed, n_top, blade_info["blade_sign"], dense_file)
           for i, e in enumerate(examples)]
    far_mm = np.array([g["far_mm"] for g in geo])
    other_mm = np.array([g["other_mm"] for g in geo])
    # a half of the scissors that is far from the hand and is nevertheless strongly coloured
    mixed = (far_mm > FAR_MM) & np.array([g["drawn_max_far"] >= STRONG_SHARE * g["drawn_max"] for g in geo])
    mixed_seq = (np.flatnonzero(mixed) + 1).tolist()
    held = other_mm < HOLD_MM                                      # the hand that is not drawn is on the object
    held_seq = (np.flatnonzero(held) + 1).tolist()
    for i in np.flatnonzero(mixed):
        if abs(geo[i]["far_blade_mm"] - geo[i]["far_mm"]) > 1e-6:
            raise RuntimeError("the caption calls the far half 'the second coloured blade': its nearest point must lie on its blade")
    if not held.any() or held.all() or held_seq != list(range(held_seq[0], N_SHOW + 1)):
        raise RuntimeError("the caption's wording assumes the other hand is on the object from one sequence on, and not in all ten")
    held_at_handles = all(geo[i]["other_touched_on_handle_side"] >= HANDLE_SHARE and geo[i]["other_nearest_behind_hinge_mm"] > 0
                          for i in np.flatnonzero(held))
    for i, g in enumerate(geo):
        log.info("sequence %d: top / bottom part %.1f / %.1f mm from the %s hand; far part (%s) drawn up to %.3f of %.3f, recorded up to "
                 "%.3f; correlation drawn vs recorded %.2f; %s hand %.2f mm from the surface, %d of its vertices within %g mm, "
                 "%.0f mm behind the hinge", i + 1, g["part_mm"]["top"], g["part_mm"]["bottom"], hand_word, g["far_part"],
                 g["drawn_max_far"], g["drawn_max"], g["recorded_max_far"], g["correlation"], other_word, g["other_mm"],
                 g["other_vertices_near"], NEAR_MM, g["other_nearest_behind_hinge_mm"])

    # ---- values computed from the drawn arrays, and their cross-checks against the study's tables
    opening_deg = np.degrees([e.articulation(0) for e in examples])
    recorded_min_mm = np.array([-qlib.SOFT_SCALE * np.log(e.dense(0).max()) * 1000 for e in examples])   # recorded distances
    drawn_min_mm = np.array([cKDTree(h).query(v)[0].min() * 1000 for h, v in zip(hands, meshes)])       # drawn geometry
    share_recorded = blade_share(recorded_512, blade)
    share_samples = blade_share(samples, blade)
    share_test = blade_share(gt_first[sel.group_rows], blade)
    share_train = blade_share(train_first, blade)
    share_group_samples = blade_share(samples_all[sel.group_rows], blade)     # (test sequences of the group, 10)
    share_undrawn = share_test[N_SHOW:]                            # the group's test sequences that are not drawn
    handle_recorded, handle_samples = share_recorded < SPLIT_SHARE, share_samples < SPLIT_SHARE
    if not (handle_recorded.any() and (~handle_recorded).any() and handle_samples.any() and (~handle_samples).any()):
        raise RuntimeError("the caption's wording assumes maps on both sides of the hinge among the recordings and the samples")
    if not np.allclose(share_test[:N_SHOW], share_recorded) or len(share_undrawn) == 0 or (share_undrawn < SPLIT_SHARE).any():
        raise RuntimeError("the caption's wording assumes that every test sequence that is not drawn is at the hinge and the blades")
    shown = np.concatenate([share_recorded, share_samples])
    split_gap = [float(shown[shown < SPLIT_SHARE].max()), float(shown[shown >= SPLIT_SHARE].min())]
    dist = np.linalg.norm(samples[:, None] - recorded_512[None], axis=2)      # (sample, recorded sequence), map units
    nearest = dist.argmin(1)
    per_example = pd.read_csv(PER_EXAMPLE)
    table = per_example[(per_example.example == first.meta.example) & (per_example.model == "samplerG_static")].set_index("K")
    table_first, table_mean = float(table.loc[1, "s0_err"]), float(table.loc[-1, "s0_err"])
    checks = {
        "sample 1 to the recorded first map of sequence 1: computed vs per_example.csv (samplerG_static, K = 1, s0_err)":
            [r(dist[0, 0], 4), r(table_first, 4)],
        "mean over the ten samples of that distance: computed vs per_example.csv (samplerG_static, K = -1, s0_err)":
            [r(dist[:, 0].mean(), 4), r(table_mean, 4)],
        "sum of the recorded first map vs sequences_meta.csv mass0 (max abs difference over the ten)":
            r(np.abs(recorded_512.astype(np.float64).sum(1) - rows_meta.mass0.to_numpy()).max(), 4),
        "opening angle: ARCTIC object state vs the study's stored state O[:, frame 0, 'arti'] (max abs difference, rad)":
            r(np.abs(np.radians(opening_deg) - opening_state).max(), 5),
        "nearest hand-to-surface distance: drawn geometry vs recorded distances (max abs difference, mm)":
            r(np.abs(recorded_min_mm - drawn_min_mm).max(), 4),
        f"distance of every mesh vertex to the drawn {hand_word} hand: drawn geometry vs recorded contact_{hand_word} "
        "(max abs difference over the ten panels, mm)": r(max(g["check_hand_mm"] for g in geo), 6),
        f"distance of every mesh vertex to the {other_word} hand (not drawn): this script's transform vs recorded contact_{other_word} "
        "(max abs difference over the ten panels, mm)": r(max(g["check_other_mm"] for g in geo), 6),
    }
    disagreements = []
    if abs(dist[0, 0] - table_first) > 2e-3 or abs(dist[:, 0].mean() - table_mean) > 2e-3:
        disagreements.append("distance of the samples to the recorded first map differs from per_example.csv")
    if np.abs(recorded_512.astype(np.float64).sum(1) - rows_meta.mass0.to_numpy()).max() > 1e-2:
        disagreements.append("sum of the recorded first map differs from sequences_meta.csv mass0")
    if np.abs(np.radians(opening_deg) - opening_state).max() > 2e-3:
        disagreements.append("opening angle differs between the ARCTIC object state and the study's stored state")
    if np.abs(recorded_min_mm - drawn_min_mm).max() > 0.05 or max(g["check_hand_mm"] for g in geo) > 0.05:
        disagreements.append("the drawn hand is not where the recorded hand-to-surface distances say it is")
    if max(g["check_other_mm"] for g in geo) > 0.05:
        disagreements.append("the hand that is not drawn is not where the recorded hand-to-surface distances say it is")
    for name, value in checks.items():
        log.info("check: %s = %s", name, value)

    # ---- colour scale and camera: one of each for all twenty panels
    largest = float(max(np.nanmax(recorded_vertex), np.nanmax(sample_vertex)))
    vmax = float(np.ceil(largest / VMAX_STEP - 1e-9) * VMAX_STEP)
    direction, up, camera_evidence = camera_rule(closed, hands)
    scene = meshes + hands + [closed]
    size = (PANEL_WIDTH, PANEL_WIDTH)
    camera = qlib.fit_camera(scene, direction, up, margin=CAMERA_MARGIN, aspect=1.0)

    # ---- render
    take_letter = {key: "ABCDEFGHIJ"[i] for i, key in enumerate(dict.fromkeys(rows_meta.take_key))}   # order of appearance
    font = layout_font(PANEL_WIDTH)
    recorded_images = [qlib.render([qlib.mesh_item(meshes[i], faces, rgb=qlib.contact_rgb(recorded_vertex[i], vmax)),
                                    qlib.hand_item(hands[i], hand_faces)], camera, size) for i in range(N_SHOW)]
    sample_full = [qlib.render([qlib.mesh_item(closed, faces, rgb=qlib.contact_rgb(sample_vertex[k], vmax))], camera, size)
                   for k in range(N_SHOW)]
    sample_images, band = crop_band(sample_full, int(round(BAND_PAD * size[1])))
    log.info("rendered %d panels of %s px; sampled panels show rows %s of that view", 2 * N_SHOW, size, band)
    recorded_titled = [with_title(im, f"{i + 1} · take {take_letter[row.take_key]} ({row.verb})", font)
                       for i, (im, row) in enumerate(zip(recorded_images, rows_meta.itertuples()))]
    sample_titled = [with_title(im, f"sample {k + 1}", font) for k, im in enumerate(sample_images)]

    one_cm = float(np.exp(-0.01 / qlib.SOFT_SCALE))
    bar = qlib.colorbar("contact", 0.0, vmax, f"contact value of the map (1 = hand touching, {one_cm:.2f} = hand 1 cm away)")
    upper = qlib.grid([recorded_titled[:N_COLS], recorded_titled[N_COLS:]],
                      row_labels=[f"Recorded: {hand_word} hand and contact at the first contact frame, sequences 1 to {N_COLS}",
                                  f"Recorded, sequences {N_COLS + 1} to {N_SHOW}"])
    lower = qlib.grid([sample_titled[:N_COLS], sample_titled[N_COLS:]],
                      row_labels=[f"Sampled first contact, samples 1 to {N_COLS} (no hand)", f"Sampled, samples {N_COLS + 1} to {N_SHOW}"],
                      colorbars=[bar])
    figure = stack(upper, lower)
    if figure.info["qlib_layout"]["font_px"] != font.size:
        raise RuntimeError("panel titles and grid labels ended up with different font sizes")

    # ---- what was drawn, for the verifier
    panels: dict[str, Any] = {"mesh_faces": faces, "hand_faces": hand_faces, "closed_mesh_verts": closed,
                              "canonical_points_canonical_units": canonical_all[category], "blade_side_mask": blade,
                              "sample_to_recorded_distance": dist, "sample_band_rows": np.asarray(band),
                              "top_part_vertex_count": np.asarray(n_top)}
    for i in range(N_SHOW):
        # the last three arrays are NOT drawn: they are the evidence for the caption's two reading notes
        panels[f"recorded_{i + 1:02d}"] = {"mesh_verts": meshes[i], "hand_verts": hands[i], "contact_512": recorded_512[i],
                                           "contact_vertex": recorded_vertex[i], "image": recorded_images[i],
                                           "not_drawn_distance_to_hand_vertex": geo[i]["d_hand"],
                                           f"not_drawn_{other_word}_hand_verts": geo[i]["other_hand"],
                                           f"not_drawn_distance_to_{other_word}_hand_vertex": geo[i]["d_other"]}
        panels[f"sample_{i + 1:02d}"] = {"contact_512": samples[i], "contact_vertex": sample_vertex[i], "image": sample_images[i]}

    # ---- every number a caption could quote, with its source
    row0 = int(sel.rows[0])
    side = blade_info["blade_side"]
    arctic_state = str(qlib.ARCTIC_PROC / category) + "/<subject>/<take>_processed_obj_features.npy"
    dense_files = str(qlib.ARCTIC_PROBE / "sequences")
    split_files = f"{META_CSV}; {MANIFEST}"
    numbers: list[dict[str, Any]] = []

    def num(label: str, value: Any, source: Any, locator: str) -> None:
        numbers.append({"label": label, "value": value, "dataset": "ARCTIC", "source": str(source), "locator": locator})

    takes = list(take_letter)
    take_sequences = {key: (np.flatnonzero(rows_meta.take_key.to_numpy() == key) + 1).tolist() for key in takes}
    handle_seq, other_seq = (np.flatnonzero(handle_recorded) + 1).tolist(), (np.flatnonzero(~handle_recorded) + 1).tolist()
    handle_smp = (np.flatnonzero(handle_samples) + 1).tolist()
    num(f"test sequences of {category}, {hand_word} hand", len(sel.group_rows), split_files, f"manifest key 'test', group == {sel.group}")
    num("test sequences of the group with the second most", sel.ranking[1]["test_sequences"], split_files, f"group {sel.ranking[1]['group']}")
    num("groups (object, hand) with test sequences", len(sel.ranking), split_files, "distinct (group, mesh_id) among the test rows")
    num("sequences drawn", N_SHOW, PREDS, f"rows {sel.rows.tolist()}: the first {N_SHOW} of the group in file order")
    num("samples drawn", N_SHOW, PREDS, f"S0_G[{row0}, 0..{N_SHOW - 1}]: every saved sample, stored order")
    num("takes among the ten sequences", len(takes), META_CSV, f"column take_key: {takes}")
    num("other takes among the ten sequences", len(takes) - 1, META_CSV, f"column take_key: {takes[1:]}")
    for key in takes:
        num(f"sequences from take {take_letter[key]} ({key})", len(take_sequences[key]), META_CSV,
            f"sequences {take_sequences[key]}; contact episodes {rows_meta.episode_index[rows_meta.take_key == key].add(1).tolist()} "
            f"of {int(rows_meta.n_episodes[rows_meta.take_key == key].iloc[0])} of that take and hand")
    num("colour scale maximum", r(vmax, 2), "computed in q_grasp_modes.py", f"largest drawn value ({largest:.3f}) rounded up to a multiple of {VMAX_STEP}")
    num("largest value drawn, recorded maps", r(np.nanmax(recorded_vertex)), "panels.npz", "max of recorded_*/contact_vertex")
    num("largest value drawn, sampled maps", r(np.nanmax(sample_vertex)), "panels.npz", "max of sample_*/contact_vertex")
    num("contact value at a hand distance of 1 cm", r(one_cm), "definition of the contact value, exp(-distance / 2 cm)", "exp(-0.5)")
    num("nearest hand-to-surface distance at the drawn frame, smallest of the ten (mm)", r(recorded_min_mm.min(), 1), dense_files,
        f"contact_{hand_word}[t0].min(), sequence {int(recorded_min_mm.argmin()) + 1}")
    num("nearest hand-to-surface distance at the drawn frame, largest of the ten (mm)", r(recorded_min_mm.max(), 1), dense_files,
        f"contact_{hand_word}[t0].min(), sequence {int(recorded_min_mm.argmax()) + 1}")
    num("opening angle at the drawn frame, smallest of the ten (degrees)", r(opening_deg.min(), 1), arctic_state,
        f"obj_world_state[t0, 0], sequence {int(opening_deg.argmin()) + 1}")
    num("opening angle at the drawn frame, largest of the ten (degrees)", r(opening_deg.max(), 1), arctic_state,
        f"obj_world_state[t0, 0], sequence {int(opening_deg.argmax()) + 1}")
    num("drawn sequences with the map on the handle side (blade-side share below 0.25)", len(handle_seq), PREDS, f"sequences {handle_seq}")
    num("highest blade-side share among those sequences", r(share_recorded[handle_recorded].max(), 4), PREDS, f"gt[rows, 0], sequences {handle_seq}")
    num("lowest blade-side share among the other drawn sequences", r(share_recorded[~handle_recorded].min(), 4), PREDS, f"gt[rows, 0], sequences {other_seq}")
    num("highest blade-side share among the other drawn sequences", r(share_recorded[~handle_recorded].max(), 4), PREDS, f"gt[rows, 0], sequences {other_seq}")
    num("samples with the map on the handle side (blade-side share below 0.25)", len(handle_smp), PREDS, f"S0_G[{row0}], samples {handle_smp}")
    num("highest blade-side share among those samples", r(share_samples[handle_samples].max(), 4), PREDS, f"S0_G[{row0}], samples {handle_smp}")
    num("other samples", int((~handle_samples).sum()), PREDS, f"S0_G[{row0}], samples {(np.flatnonzero(~handle_samples) + 1).tolist()}")
    num("lowest blade-side share among the other samples", r(share_samples[~handle_samples].min(), 4), PREDS, f"S0_G[{row0}]")
    num("highest blade-side share among the other samples", r(share_samples[~handle_samples].max(), 4), PREDS, f"S0_G[{row0}]")
    num("samples whose nearest recorded first map (of the ten drawn) is a handle-side sequence", int(handle_recorded[nearest].sum()), PREDS,
        f"argmin over the drawn rows of ||S0_G[{row0}, k] - gt[row, 0]||")
    num("test sequences of the group with the map on the handle side", int((share_test < SPLIT_SHARE).sum()), PREDS,
        f"gt[rows of the group, 0], {len(share_test)} sequences")
    num("training sequences of the group", len(share_train), f"{SEQUENCES}; {MANIFEST}", f"manifest key 'train', group == {sel.group}")
    num("training sequences of the group with the map on the handle side", int((share_train < SPLIT_SHARE).sum()), f"{SEQUENCES}; {MANIFEST}",
        "C[training rows of the group, 0]")
    undrawn_rows = sel.group_rows[N_SHOW:].tolist()
    num("test sequences of the group that are not drawn", len(share_undrawn), PREDS,
        f"rows {undrawn_rows}: the group's test sequences after the first {N_SHOW} in file order")
    num("test sequences that are not drawn with the map on the handle side", int((share_undrawn < SPLIT_SHARE).sum()), PREDS,
        f"gt[rows {undrawn_rows[0]} to {undrawn_rows[-1]}, 0]")
    num("lowest blade-side share among the test sequences that are not drawn", r(share_undrawn.min(), 4), PREDS,
        f"gt[rows {undrawn_rows[0]} to {undrawn_rows[-1]}, 0]")
    num("highest blade-side share among the test sequences that are not drawn", r(share_undrawn.max(), 4), PREDS,
        f"gt[rows {undrawn_rows[0]} to {undrawn_rows[-1]}, 0]")
    num("saved samples of the group (ten per test sequence)", int(share_group_samples.size), PREDS, "S0_G[rows of the group]")
    num("saved samples of the group with the map on the handle side", int((share_group_samples < SPLIT_SHARE).sum()), PREDS,
        f"S0_G[rows of the group]; per test sequence {(share_group_samples < SPLIT_SHARE).sum(1).tolist()}")

    # the half of the scissors that is far from the drawn hand (reading note 1 of the caption)
    part_rows = {"top": f"[:{n_top}]", "bottom": f"[{n_top}:]"}
    far_rule = (f"the part (top or bottom half) farther from the drawn hand is more than {FAR_MM:g} mm from it and is drawn at "
                f"{STRONG_SHARE:g} of the panel's largest value or more")
    num("drawn sequences with a strongly coloured half of the scissors that is far from the drawn hand", len(mixed_seq),
        "computed in q_grasp_modes.py; arrays in panels.npz", f"sequences {mixed_seq}; rule: {far_rule}")
    num("recorded contact value at a hand distance of 30 mm", r(np.exp(-FAR_MM / 1000 / qlib.SOFT_SCALE)),
        "definition of the contact value, exp(-distance / 2 cm)", "exp(-1.5)")
    for i, (e, g) in enumerate(zip(examples, geo)):
        where = f"{e.meta.sequence_id}, contact_{hand_word}[{g['frame']}]{part_rows[g['far_part']]}"
        tag = ", the second coloured blade" if mixed[i] else ""
        num(f"sequence {i + 1}: nearest distance from the drawn hand to the half of the scissors farther from it "
            f"({g['far_part']} part{tag}) (mm)", r(g["far_mm"], 1), dense_files, f"{where}.min()")
        if mixed[i]:
            key = f"recorded_{i + 1:02d}"
            num(f"sequence {i + 1}: largest drawn value on that half", r(g["drawn_max_far"]), "panels.npz",
                f"{key}/contact_vertex{part_rows[g['far_part']]}.max()")
            num(f"sequence {i + 1}: largest drawn value in the panel", r(g["drawn_max"]), "panels.npz", f"{key}/contact_vertex.max()")
            num(f"sequence {i + 1}: largest recorded per-vertex contact on that half", r(g["recorded_max_far"]), dense_files,
                f"exp(-d / 2 cm) with d = {where}.min()")
            num(f"sequence {i + 1}: share of the vertices drawn at {STRONG_SHARE:g} of the panel's largest value or more that are more than "
                f"{FAR_MM:g} mm from the drawn hand", r(g["strong_far_share"]), "panels.npz",
                f"{key}/contact_vertex and {key}/not_drawn_distance_to_hand_vertex")
            num(f"sequence {i + 1}: correlation of the drawn values with the recorded per-vertex contact", r(g["correlation"]),
                f"panels.npz; {dense_files}", f"{key}/contact_vertex against exp(-contact_{hand_word}[{g['frame']}] / 2 cm)")
    correlation = np.array([g["correlation"] for g in geo])
    num("lowest correlation of the drawn values with the recorded per-vertex contact among the other drawn sequences",
        r(correlation[~mixed].min()), f"panels.npz; {dense_files}", f"sequences {(np.flatnonzero(~mixed) + 1).tolist()}")
    num("highest correlation of the drawn values with the recorded per-vertex contact among the other drawn sequences",
        r(correlation[~mixed].max()), f"panels.npz; {dense_files}", f"sequences {(np.flatnonzero(~mixed) + 1).tolist()}")

    # the hand that is not drawn (reading note 2 of the caption)
    hand_files = str(qlib.ARCTIC_PROC / category) + f"/<subject>/<take>_processed_hand_features.npy ({other_word}_hand_verts)"
    near = np.array([g["other_vertices_near"] for g in geo])
    behind = np.array([g["other_nearest_behind_hinge_mm"] for g in geo])
    num(f"drawn sequences in which the {other_word} hand (not drawn) is on the object (nearest distance below {HOLD_MM:g} mm)",
        len(held_seq), dense_files, f"contact_{other_word}[t0].min(), sequences {held_seq}")
    num(f"largest nearest {other_word}-hand-to-surface distance among those sequences (mm)", r(other_mm[held].max(), 2), dense_files,
        f"contact_{other_word}[t0].min(), sequence {int(np.flatnonzero(held)[other_mm[held].argmax()]) + 1}")
    num(f"smallest nearest {other_word}-hand-to-surface distance among the other drawn sequences (mm)", r(other_mm[~held].min(), 2),
        dense_files, f"contact_{other_word}[t0].min(), sequence {int(np.flatnonzero(~held)[other_mm[~held].argmin()]) + 1}")
    num(f"fewest {other_word}-hand vertices within {NEAR_MM:g} mm of the surface among those sequences (of 778)", int(near[held].min()),
        hand_files, f"sequences {held_seq}, take frame t0, against the posed mesh of the panel")
    num(f"most {other_word}-hand vertices within {NEAR_MM:g} mm of the surface among those sequences (of 778)", int(near[held].max()),
        hand_files, f"sequences {held_seq}, take frame t0, against the posed mesh of the panel")
    num(f"nearest surface point to the {other_word} hand among those sequences: smallest distance behind the hinge, toward the handles (mm)",
        r(behind[held].min(), 1), hand_files, f"sequences {held_seq}; position of that vertex along the closed {category}")
    num(f"nearest surface point to the {other_word} hand among those sequences: largest distance behind the hinge, toward the handles (mm)",
        r(behind[held].max(), 1), hand_files, f"sequences {held_seq}; position of that vertex along the closed {category}")
    handle_side_share = np.array([g["other_touched_on_handle_side"] for g in geo])
    num(f"smallest share of the surface vertices within {NEAR_MM:g} mm of the {other_word} hand that lie behind the hinge, among those "
        "sequences", r(handle_side_share[held].min()), hand_files,
        f"sequences {held_seq}: {[r(v) for v in handle_side_share[held]]}")
    for i, (e, g) in enumerate(zip(examples, geo)):
        num(f"sequence {i + 1}: nearest distance from the {other_word} hand (not drawn) to the surface at the drawn frame (mm)",
            r(g["other_recorded_mm"], 2), dense_files, f"{e.meta.sequence_id}, contact_{other_word}[{g['frame']}].min()")
        num(f"sequence {i + 1}: {other_word}-hand vertices within {NEAR_MM:g} mm of the surface at the drawn frame (of 778)",
            g["other_vertices_near"], hand_files, f"{e.meta.sequence_id}, take frame {g['frame']}, against the posed mesh of the panel")
    num("distance of sample 1 to the recorded first map of sequence 1 (map units)", r(table_first), PER_EXAMPLE,
        f"example == {first.meta.example}, model == samplerG_static, K == 1, column s0_err")
    num("mean distance of the ten samples to the recorded first map of sequence 1 (map units)", r(table_mean), PER_EXAMPLE,
        f"example == {first.meta.example}, model == samplerG_static, K == -1, column s0_err")
    for i, e in enumerate(examples):
        num(f"sequence {i + 1}: share of the recorded first map on the blade side of the hinge", r(share_recorded[i], 4), PREDS,
            f"gt[{int(sel.rows[i])}, 0] (= sequences.npz C[{e.meta.example}, 0]); canonical points with {side}")
        num(f"sequence {i + 1}: opening angle at the drawn frame (degrees)", r(opening_deg[i], 1), arctic_state,
            f"{e.meta.sequence_id}, obj_world_state[{e.meta.t0}, 0]")
        num(f"sequence {i + 1}: nearest hand-to-surface distance at the drawn frame (mm)", r(recorded_min_mm[i], 1), dense_files,
            f"{e.meta.sequence_id}, contact_{hand_word}[{e.meta.t0}].min()")
    for k in range(N_SHOW):
        num(f"sample {k + 1}: share of the map on the blade side of the hinge", r(share_samples[k], 4), PREDS,
            f"S0_G[{row0}, {k}]; canonical points with {side}")
        num(f"sample {k + 1}: distance to the recorded first map of sequence 1 (map units)", r(dist[k, 0]), PREDS,
            f"||S0_G[{row0}, {k}] - gt[{row0}, 0]||")
        num(f"sample {k + 1}: nearest of the ten recorded first maps (sequence number)", int(nearest[k]) + 1, PREDS,
            f"argmin over the drawn rows of ||S0_G[{row0}, {k}] - gt[row, 0]||")

    # ---- caption, built from the values above
    saved_on = time.strftime("%Y-%m-%d", time.localtime(PREDS.stat().st_mtime))
    letters = [take_letter[k] for k in takes]
    other_takes = letters[1:]
    verbs = list(dict.fromkeys(rows_meta.verb))                    # the dataset's names of the takes, in order of appearance
    if handle_seq != take_sequences[takes[0]] or len(handle_smp) != 1:
        raise RuntimeError("the caption's wording assumes that the handle-side sequences are those of the first take, and one handle-side sample")
    # reading note 1: the map lives on the closed object, so both blades are coloured (sequences named by the rule far_rule)
    mixing_note = f"The contact map is defined on the closed {category}, so contact on one blade is also coloured on the other"
    if mixed_seq:
        mixing_note += (f": in sequence{'s' if len(mixed_seq) > 1 else ''} {span(mixed_seq)} the second coloured blade is "
                        f"{listed([f'{far_mm[i - 1]:.0f} mm' for i in mixed_seq])} from the hand")
    # reading note 2: the hand that is not drawn
    held_where = "the handles" if held_at_handles else f"the {category}"
    # six sentences; "what" = what is drawn, "note" = what a reader must know to read it correctly
    sentences = [
        ("what",
         f"ARCTIC {category}, {hand_word} hand: the first {N_SHOW} of the {len(sel.group_rows)} test sequences of this object and hand, "
         f"selected as the group with the most test sequences; each title gives the recording (take {', '.join(letters[:-1])} or "
         f"{letters[-1]}) and, in brackets, the dataset's name of the recording ({' or '.join(verbs)})."),
        ("what, with the note on the hand that is not drawn",
         f"Upper rows: the recorded {hand_word} hand at the first frame of firm contact, still {recorded_min_mm.min():.0f} to "
         f"{recorded_min_mm.max():.0f} mm from the surface, with the recorded contact map on the {category}; only the {hand_word} hand is "
         f"drawn, and in sequences {span(held_seq)} the {other_word} hand, not drawn, is already holding {held_where}."),
        ("what",
         f"Sequences {span(handle_seq)} are {len(handle_seq)} contact episodes of one recording (take {letters[0]}), with the hand at a "
         f"handle ring (at most {pct_up(share_recorded[handle_recorded].max())} % of the map lies on the blade side of the hinge); sequences "
         f"{span(other_seq)} come from {len(takes) - 1} other recordings (takes {' and '.join(other_takes)}), with the hand at the hinge and "
         f"the blades ({pct(share_recorded[~handle_recorded].min())} to {pct(share_recorded[~handle_recorded].max())} %)."),
        ("note on colour that is far from the hand", f"{mixing_note}."),
        ("what",
         f"Lower rows: {N_SHOW} first contact maps sampled for sequence 1 by the model that is given only the object's shape and which hand, "
         f"drawn on the closed {category} and without a hand because that model outputs a map and is not given the opening angle; "
         f"sample {handle_smp[0]} lies on a handle ring ({pct(share_samples[handle_samples].max())} % on the blade side), the other "
         f"{int((~handle_samples).sum())} around the hinge and on the blades ({pct(share_samples[~handle_samples].min())} to "
         f"{pct(share_samples[~handle_samples].max())} %)."),
        ("note on the sequences that are not drawn, and the shared camera and colour scale",
         f"The {len(share_undrawn)} test sequences not drawn are all at the hinge and the blades, so the figure does not compare how often "
         f"each grasp occurs; all panels share one camera and one colour scale (0 to {vmax:g})."),
    ]
    caption = " ".join(text for _, text in sentences)
    alt = (f"Twenty renders of a pair of {category}: ten with a recorded {hand_word} hand at a handle ring or at the blades and the contact "
           f"coloured on the surface, and ten sampled contact maps on the closed {category} without a hand.")

    info: dict[str, Any] = {
        "id": FIG_ID,
        "block": BLOCK,
        "title": "What an initial grasp mode is: recorded and sampled first contact on one object",
        "panels": {
            "upper two rows (ten panels, numbered 1 to 10 in file order)":
                f"the recorded {hand_word} hand on the {category} at the first frame of firm contact of one test sequence each; the surface "
                "is coloured by the recorded first contact map (512 values) of that sequence; the object is opened as recorded at that "
                "frame; the title gives the sequence number, the take it comes from (a letter) and, in brackets, the dataset's name of "
                f"that take ({' or '.join(verbs)}; column verb of sequences_meta.csv); the {other_word} hand is not drawn",
            "lower two rows (ten panels, sample 1 to sample 10 in stored order)":
                "the ten saved samples of the first contact map for sequence 1, from the sampler that is given the object's shape and "
                f"which hand; drawn on the closed {category}, no hand; each panel is the horizontal band of the same camera view in which "
                "the closed object lies",
            "layout": f"each group of ten panels wraps into two rows of {N_COLS}; one camera and one colour scale for all twenty panels",
        },
        "examples": [{"panel": f"recorded {i + 1}", "dataset": "ARCTIC", "example": e.meta.example, "sequence_id": e.meta.sequence_id,
                      "object": category, "hand": hand_word, "split": e.meta.split,
                      "frames_drawn": {"sequence_frame": 0, "take_frame": e.take_frame(0)},
                      "take": take_letter[row.take_key], "take_label_in_dataset": row.verb, "subject": row.subject,
                      "contact_episode_in_take": f"{int(row.episode_index) + 1} of {int(row.n_episodes)}",
                      "opening_angle_deg": r(opening_deg[i], 1), "nearest_hand_to_surface_mm": r(recorded_min_mm[i], 1),
                      "blade_side_share": r(share_recorded[i], 4), "prediction_file_row": int(sel.rows[i])}
                     for i, (e, row) in enumerate(zip(examples, rows_meta.itertuples()))],
        "selection_rule": (
            "The rule of figure 7 of the hierarchical report (scripts/research/hier_contact_gen/figures.py::fig7): the (group, mesh) with the "
            f"most test sequences, its first {N_SHOW} test sequences in the order of the prediction file, and the {N_SHOW} saved samples of the "
            "first of them. All ten samples are shown in stored order; none is picked."),
        "selection_evidence": {
            "ranked_test_sequences_per_group": sel.ranking,
            "chosen": {"group": sel.group, "mesh_id": sel.mesh_id, "rank": 1, "test_sequences": len(sel.group_rows),
                       "runner_up": sel.ranking[1], "candidates": len(sel.ranking)},
            "prediction_file_rows_of_the_group": sel.group_rows.tolist(),
            "rows_drawn": sel.rows.tolist(), "examples_drawn": sel.examples.tolist(),
            "samples_drawn": {"prediction_file_row": row0, "example": int(sel.examples[0]), "sample_indices": list(range(N_SHOW))},
            "prediction_file_order_equals_test_manifest": True,
            "sampler_input_identical_for_all_test_sequences_of_the_group": same_sampler_input,
            "sampler_input_g_index_hand_role": static_inputs[0].tolist(),
        },
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (qlib.Example.to_vertices); recorded and sampled maps both go through it",
            "colour_scales": [{"name": "contact", "range": [0.0, r(vmax, 2)], "colours": "grey (0) through orange to dark red-brown (maximum)",
                               "rule": f"one scale for all twenty panels; maximum = largest drawn value ({largest:.3f}) rounded up to a "
                                       f"multiple of {VMAX_STEP}",
                               "negative_values": f"the sampled maps go down to {float(samples.min()):.3f}; values below 0 are drawn as 0"}],
            "camera": {"rule": "the camera looks along the hinge axis (the object is thin along it), from the side on which fewer recorded "
                               f"hand vertices lie, tilted by {TILT_DEG:g} degrees away from the mean hand position, so that the thin "
                               "object is drawn in front of the hands; the closed object lies horizontally",
                       "evidence": camera_evidence,
                       "fitted_on": "the ten posed meshes, the ten hands and the closed mesh; one camera for all twenty panels",
                       "camera": camera, "panel_px": list(size), "margin": CAMERA_MARGIN,
                       "sampled_panels": f"rows {band[0]} to {band[1]} of the same {size[1]} px view (the rows outside are empty)"},
            "hand_style": f"recorded MANO {hand_word} hand (the hand whose contact the map records), opaque skin colour, smooth shading; "
                          f"the {other_word} hand is not drawn in any panel; no hand in the sampled panels",
            "object_pose": {"recorded": "top part turned by the recorded opening angle at the drawn frame",
                            "sampled": "closed rest pose, the pose in which the 512 canonical points are defined (the sampler has no "
                                       "opening-angle input)"},
            "panel_titles": "drawn by this script in the grid's font (qlib.grid has one label per column; the ten panels wrap into two rows)",
            "blade_side_of_the_hinge": dict(blade_info, split_at=SPLIT_SHARE,
                                            shares_nearest_to_the_split_among_the_twenty_maps=[r(v, 4) for v in split_gap]),
        },
        "numbers": numbers,
        "per_panel_values": {
            "recorded_blade_side_share": [r(v, 4) for v in share_recorded], "sample_blade_side_share": [r(v, 4) for v in share_samples],
            "sample_to_recorded_distance_rows_samples_cols_sequences": np.round(dist.astype(np.float64), 3),
            "sample_nearest_recorded_sequence": (nearest + 1).tolist(),
            "group_test_blade_side_share_file_order": [r(v, 4) for v in share_test],
            "group_train_blade_side_share_histogram": {"bin_edges": [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
                                                       "counts": np.histogram(share_train, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0001])[0]},
            "group_samples_on_the_handle_side_per_test_sequence_file_order": (share_group_samples < SPLIT_SHARE).sum(1).tolist(),
            "recorded_panels_at_the_drawn_frame": [
                {"sequence": i + 1, "take_frame": g["frame"],
                 f"nearest_distance_{hand_word}_hand_to_top_part_mm": r(g["part_mm"]["top"], 2),
                 f"nearest_distance_{hand_word}_hand_to_bottom_part_mm": r(g["part_mm"]["bottom"], 2),
                 "part_farther_from_the_drawn_hand": g["far_part"],
                 "nearest_distance_to_the_blade_of_that_part_mm": r(g["far_blade_mm"], 2),
                 "largest_drawn_value": r(g["drawn_max"]), "largest_drawn_value_on_that_part": r(g["drawn_max_far"]),
                 "largest_recorded_per_vertex_contact_on_that_part": r(g["recorded_max_far"]),
                 f"share_of_strongly_drawn_vertices_more_than_{FAR_MM:g}_mm_from_the_hand": r(g["strong_far_share"]),
                 "correlation_drawn_vs_recorded_per_vertex_contact": r(g["correlation"]),
                 "named_in_the_caption_as_second_coloured_blade": bool(mixed[i]),
                 f"nearest_distance_{other_word}_hand_to_surface_mm": r(g["other_mm"], 2),
                 f"{other_word}_hand_vertices_within_{NEAR_MM:g}_mm_of_the_surface": g["other_vertices_near"],
                 f"part_nearest_to_the_{other_word}_hand": g["other_nearest_part"],
                 "that_point_behind_the_hinge_toward_the_handles_mm": r(g["other_nearest_behind_hinge_mm"], 1),
                 f"share_of_surface_vertices_within_{NEAR_MM:g}_mm_of_the_{other_word}_hand_on_the_handle_side":
                     None if np.isnan(g["other_touched_on_handle_side"]) else r(g["other_touched_on_handle_side"]),
                 f"{other_word}_hand_counted_as_holding": bool(held[i])}
                for i, g in enumerate(geo)],
        },
        "reading_note_rules": {
            "second_coloured_blade": f"a sequence is named when {far_rule}; the distance quoted is the smallest distance of a vertex of "
                                     "that part to the drawn hand, and that vertex lies on the part's blade",
            "hand_not_drawn": f"the {other_word} hand is counted as holding the object when its nearest vertex is within {HOLD_MM:g} mm of "
                              f"the surface; among the ten panels the distances on the two sides of this threshold are "
                              f"{other_mm[held].max():.2f} mm and {other_mm[~held].min():.2f} mm, so the threshold does not decide the "
                              f"count; 'holding the handles' is written when, in each of those sequences, the surface point nearest to "
                              f"that hand and at least {HANDLE_SHARE:g} of the surface vertices within {NEAR_MM:g} mm of it lie behind the "
                              f"hinge (smallest share {handle_side_share[held].min():.3f})",
            "at_most": "a percentage after 'at most' is rounded up to one decimal; other percentages and distances are rounded to the nearest unit",
        },
        "cross_checks": checks,
        "disagreements": disagreements,
        "inference": {
            "newly_computed": "none (data and saved predictions only; no checkpoint was loaded)",
            "device": "CPU (rendering only)",
            "seed": "none used here",
            "read_from_saved_predictions": f"{PREDS}: S0_G[{row0}] (ten samples), gt[rows, 0] (recorded first maps)",
            "how_the_saved_samples_were_made": (
                f"by the study's scripts/research/hier_contact_gen/rollout_eval.py (file written {saved_on}) with {SAMPLER_CKPT}, the "
                "diffusion sampler of the first map given the object descriptor and the hand flag; per that script's defaults: 100 DDIM "
                "steps, one generator per sample index k, seeded 100000 + k, for the whole test batch"),
        },
        "caveats": [
            f"The two visibly different grasps coincide with different recordings: sequences {span(take_sequences[takes[0]])} are contact "
            f"episodes of one take ({takes[0]}), the others come from {len(takes) - 1} other takes by other subjects "
            f"({', '.join(takes[1:])}). The figure does not show one subject choosing between two grasps.",
            f"The first ten sequences in file order are not a random draw of the group: they contain all "
            f"{int((share_test < SPLIT_SHARE).sum())} handle-side sequences of the {len(share_test)} test sequences; the other "
            f"{len(share_undrawn)} are all at the hinge and the blades (blade-side share {share_undrawn.min():.2f} to "
            f"{share_undrawn.max():.2f}). {len(handle_seq)} of {N_SHOW} recorded panels against {len(handle_smp)} of {N_SHOW} sampled panels "
            f"is therefore not a comparison of how often each grasp occurs. Handle-side counts: {int((share_test < SPLIT_SHARE).sum())} of "
            f"{len(share_test)} test sequences, {int((share_train < SPLIT_SHARE).sum())} of {len(share_train)} training sequences, "
            f"{int((share_group_samples < SPLIT_SHARE).sum())} of the {share_group_samples.size} saved samples of the group.",
            "The sampler is given the object's shape and which hand only, so its input is identical for every test sequence of the group: "
            "the samples are draws from one distribution and belong to sequence 1 only in that they were stored in its row.",
            "The sampled maps are drawn on the closed object, the recorded maps on the object as opened at that frame; a map does not "
            "contain the opening angle.",
            f"The 512-value map is defined on the closed {category}: a map point between the closed blades covers both, so contact on one "
            "blade is also coloured on the other, in the recorded map itself and in what is drawn from it. "
            + (" ".join(
                f"Sequence {i + 1}: the {geo[i]['far_part']} part is {geo[i]['far_mm']:.1f} mm from the hand at its nearest point (recorded "
                f"per-vertex contact at most {geo[i]['recorded_max_far']:.2f}) and is drawn up to {geo[i]['drawn_max_far']:.2f} "
                f"(largest value of the panel {geo[i]['drawn_max']:.2f}); {100 * geo[i]['strong_far_share']:.0f} % of the vertices drawn at half "
                f"the panel's largest value or more are over {FAR_MM:g} mm from the hand; correlation of the drawn values with the recorded "
                f"per-vertex contact {geo[i]['correlation']:.2f}." for i in np.flatnonzero(mixed))
               + f" In the other {int((~mixed).sum())} recorded panels that correlation is {correlation[~mixed].min():.2f} to "
                 f"{correlation[~mixed].max():.2f}." if mixed_seq else "No drawn sequence has a strongly coloured half far from the hand."),
            f"Only the {hand_word} hand is drawn, the hand whose contact the map records. The {other_word} hand is on the object in sequences "
            f"{span(held_seq)} (nearest distance {other_mm[held].min():.2f} to {other_mm[held].max():.2f} mm, {int(near[held].min())} to "
            f"{int(near[held].max())} of its 778 vertices within {NEAR_MM:g} mm of the surface, {behind[held].min():.0f} to "
            f"{behind[held].max():.0f} mm behind the hinge), so the object is held there and is not floating; in the other drawn sequences "
            f"it is " + listed([f"{other_mm[i]:.1f} mm" for i in np.flatnonzero(~held)]) + " from the surface (sequences "
            f"{span((np.flatnonzero(~held) + 1).tolist())}). The {other_word} hand's own contact is not part of the drawn map.",
            "'First contact' is the study's first frame of firm contact (an object vertex within 1 cm of the hand and enough total contact, "
            f"for three frames), so the hand is still {recorded_min_mm.min():.1f} to {recorded_min_mm.max():.1f} mm from the surface and "
            f"the values stay below {vmax:g}.",
            f"In recorded panels {span(other_seq)} the camera is on the far side of the object from the hand, so the hand is seen behind it.",
            "The blade-side share is a coarse summary (canonical points beyond the hinge axis); it does not separate the two handle rings.",
            "No error or quality value of the sampler is shown; whether the samples cover the recorded grasps in the right proportions "
            "cannot be read from ten samples.",
        ],
        "suggested_caption": caption,
        "suggested_caption_sentences": [{"role": role, "text": text} for role, text in sentences],
        "suggested_alt": alt,
    }
    paths = qlib.save(FIG_ID, figure, panels, info)
    log.info("layout %s", figure.info["qlib_layout"])
    log.info("caption: %s", caption)
    for name, path in paths.items():
        log.info("%s: %s", name, path)
    if disagreements:
        log.warning("DISAGREEMENTS: %s", disagreements)


if __name__ == "__main__":
    main()
