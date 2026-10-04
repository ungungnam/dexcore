"""Figure ``contact_map_primer`` (page block: introduction of Finding 1, what a contact map is).

One recorded TACO sequence, data only (no model is loaded or run):

* top row, one frame, read left to right: the recorded hand on the knife | the knife alone with that
  frame's contact map drawn on its surface | the same map as it is stored, one value on each of the 512
  canonical points;
* bottom row: frames 0, 8, 16, 32, 48 and 63 of the sequence, with the hand, the knife and the map.

Selection (computed below, never by eye).
Sequence: among the test sequences of the hierarchical contact study, the one with the largest Euclidean
distance between its last and its first contact map (the rule of that study's "high-drift" example). The
script stops if this is not the sequence the page already shows (example 3011), instead of switching.
Frame of the top row: the frame of that sequence with the largest contact amount (sum of the 512 values).

Run from anywhere (CPU, software renderer; see qlib.py):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_contact_map_primer.py
"""
from __future__ import annotations

import json
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

import qlib

log = logging.getLogger("q_contact_map_primer")

FIG_ID = "contact_map_primer"
BLOCK = "F1-intro"                     # the lead of Finding 1; the page has no data-evidence code for it yet
DATASET = "taco"
PAGE_EXAMPLE = 3011                    # the sequence of the page's existing image f1b_taco_knife_rollout.png
STRIP_FRAMES = (0, 8, 16, 32, 48, 63)  # the frames of that image
VMAX = 1.0                             # the soft contact exp(-d / 2 cm) lies in [0, 1]
POINT_RADIUS = 0.0035                  # metres, the dots of the 512-point view
CAMERA_TILT = 0.5                      # argument of qlib.hand_side_view
CAMERA_MARGIN = 0.08
# Panel widths for which qlib.grid gives the three-panel block and the six-panel block the same total
# width (2472 px), hence the same font; stack() checks it.
TOP_PANEL_WIDTH, STRIP_PANEL_WIDTH = 794, 400

HIER = qlib.HIER[DATASET]
SEQUENCES, SEQUENCES_META = HIER / "sequences.npz", HIER / "sequences_meta.csv"
PREDS, PER_EXAMPLE = HIER / "eval/fixed0/preds.npz", HIER / "eval/fixed0/per_example.csv"
SEQUENCE_CONFIG = HIER / "config/sequence_construction.json"   # the study's rule for the first frame of a sequence
MESH_DICT = qlib.TACO_GEN3 / "assets/taco_mesh_dict.npy"       # the meshes qlib draws


# ------------------------------------------------------------------------------ selection
def select(contact: np.ndarray, meta: pd.DataFrame) -> dict[str, Any]:
    """The rule: test sequence with the largest ||C_63 - C_0||, then its frame with the largest sum of C.

    contact: (N, 64, 512) recorded canonical maps; meta: sequences_meta.csv (row n = example n)."""
    test = np.flatnonzero((meta.split_set == "test").to_numpy())
    maps = contact[test].astype(np.float32)                       # (n_test, 64, 512)
    drift = np.linalg.norm(maps[:, -1] - maps[:, 0], axis=1)      # (n_test,)
    order = np.argsort(-drift, kind="stable")
    example = int(test[order[0]])
    amount = contact[example].astype(np.float64).sum(1)           # (64,) contact amount per frame
    return {"test": test, "drift": drift, "order": order, "example": example, "amount": amount,
            "frame": int(np.argmax(amount))}


def cross_checks(sel: dict[str, Any], contact: np.ndarray, meta: pd.DataFrame) -> dict[str, Any]:
    """Compare what the script computes with the study's own saved tables (all opened read-only)."""
    n = sel["example"]
    maps = contact[n].astype(np.float32)
    with np.load(PREDS) as z:
        pred_examples, pred_gt = z["example"], z["gt"].astype(np.float32)
    study_drift = np.linalg.norm(pred_gt[:, -1] - pred_gt[:, 0], axis=1)       # hier_contact_gen/figures.py, fig6
    table = pd.read_csv(PER_EXAMPLE)
    row = table[(table.model == "static_gt") & (table.K == 0) & (table.example == n)]
    assert len(row) == 1, "expected one static_gt row for the example in per_example.csv"
    row, m = row.iloc[0], meta.iloc[n]
    hold_first = float(np.linalg.norm(maps - maps[0], axis=1).mean())           # error of holding the first map
    step = float(np.linalg.norm(maps[1:] - maps[:-1], axis=1).mean())           # mean frame-to-frame change
    return {
        "test_set_equals_saved_prediction_file": bool(np.array_equal(pred_examples, sel["test"])),
        "max_abs_difference_between_sequences_C_and_saved_gt": float(np.abs(pred_gt - contact[sel["test"]].astype(np.float32)).max()),
        "study_high_drift_example": int(pred_examples[int(np.argmax(study_drift))]),
        "contact_amount_frame0": {"script": float(sel["amount"][0]), "sequences_meta.csv mass0": float(m.mass0)},
        "contact_amount_mean": {"script": float(sel["amount"].mean()), "sequences_meta.csv mass_mean": float(m.mass_mean)},
        "error_of_holding_first_map": {"script": hold_first, "per_example.csv static_gt E_C": float(row.E_C)},
        "mean_frame_to_frame_change": {"script": step, "per_example.csv static_gt dmag_true": float(row.dmag_true)},
    }


# ------------------------------------------------------------------------------ geometry helpers
def canonical_points(category: str) -> np.ndarray:
    """(512, 3) canonical points of ``category`` in canonical units (object centre at 0, object radius 1)."""
    with np.load(SEQUENCES) as z:
        return z["canonical_points"][list(z["canonical_cats"]).index(category)].astype(np.float64)


def to_object_frame(canonical: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Canonical points in the object's frame, metres.

    The study maps an instance to the canonical frame by centring on the vertex mean and dividing by the
    largest vertex distance from it (no rotation); this is the inverse. The points were sampled on the
    category's template mesh, so they lie near, not on, the surface of another instance."""
    centre = verts.mean(0)
    return canonical * np.linalg.norm(verts - centre, axis=1).max() + centre


def template_mesh(category: str, canonical: np.ndarray) -> dict[str, Any]:
    """The mesh of ``category`` the canonical points were sampled on: the one whose normalised vertices contain them."""
    meshes = np.load(MESH_DICT, allow_pickle=True).item()
    best: dict[str, Any] = {}
    for mesh_id, mesh in sorted(meshes.items()):
        if mesh.get("category") != category:
            continue
        v = np.asarray(mesh["verts_original"], float)
        centre = v.mean(0)
        gap = float(cKDTree((v - centre) / np.linalg.norm(v - centre, axis=1).max()).query(canonical)[0].max())
        if not best or gap < best["largest_gap_canonical_units"]:
            best = {"mesh_id": mesh_id, "largest_gap_canonical_units": gap, "extent_m": np.round(np.sort(np.ptp(v, 0))[::-1], 3)}
    return best


def nearest_hand_distance(object_verts: np.ndarray, hand_verts: np.ndarray) -> np.ndarray:
    """(V,) distance from every object vertex to the nearest hand vertex, metres."""
    return cKDTree(hand_verts).query(object_verts)[0]


def row_camera(points: Sequence[np.ndarray], direction: np.ndarray, up: np.ndarray, width: int) -> tuple[qlib.Camera, tuple[int, int]]:
    """Camera and (width, height) in pixels for one row: the panel gets the proportions of ``points`` seen from
    ``direction``, and the HORIZONTAL field of view is qlib.VIEW_ANGLE in every row, so a wide panel is not
    drawn with a wider lens (and more perspective) than a narrow one."""
    back = direction / np.linalg.norm(direction)
    upward = up - (up @ back) * back
    upward /= np.linalg.norm(upward)
    q = np.concatenate([np.asarray(p, float).reshape(-1, 3) for p in points])
    extent_w, extent_h = np.ptp(q @ np.cross(upward, back)), np.ptp(q @ upward)
    size = (width, int(round(width * extent_h / extent_w)))
    aspect = size[0] / size[1]
    view_angle = float(np.degrees(2 * np.arctan(np.tan(np.radians(qlib.VIEW_ANGLE) / 2) / max(aspect, 1.0))))
    return qlib.fit_camera(points, direction, up, margin=CAMERA_MARGIN, aspect=aspect, view_angle=view_angle), size


def border_pixels(image: np.ndarray) -> int:
    """Number of non-white pixels on the outermost ring of a panel; 0 = nothing is cut by the panel's edge."""
    ring = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    return int((ring < 250).any(1).sum())


def tinted_pixels(image: np.ndarray) -> int:
    """Number of pixels with the pale-violet tint of qlib.NO_DATA (blue clearly above green); 0 = none visible."""
    rgb = image.astype(int)
    return int(((rgb[..., 2] - rgb[..., 1] > 8) & (rgb[..., 0] > 150)).sum())


def stack(blocks: Sequence[Image.Image]) -> Image.Image:
    """Put qlib.grid images of one width and one font under each other (qlib.grid needs equal panels)."""
    layouts = [b.info["qlib_layout"] for b in blocks]
    if len({b.width for b in blocks}) != 1 or len({lay["font_px"] for lay in layouts}) != 1:
        raise ValueError(f"blocks differ in width or font: {[(b.width, lay['font_px']) for b, lay in zip(blocks, layouts)]}")
    canvas = Image.new("RGB", (blocks[0].width, sum(b.height for b in blocks)), "white")
    y = 0
    for block in blocks:
        canvas.paste(block, (0, y))
        y += block.height
    canvas.info["qlib_layout"] = {"width": canvas.width, "height": canvas.height, "panel": [lay["panel"] for lay in layouts],
                                  "font_px": layouts[0]["font_px"], "cap_px": layouts[0]["cap_px"],
                                  "cap_px_at_1100": layouts[0]["cap_px_at_1100"], "blocks": layouts}
    return canvas


# ------------------------------------------------------------------------------ figure
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    qlib.require_headless()

    # ---- selection, from the study's sequence file
    with np.load(SEQUENCES) as z:
        contact = z["C"]                                           # (N, 64, 512) float16
    meta = pd.read_csv(SEQUENCES_META, dtype={"mesh_id": str})
    sel = select(contact, meta)
    if sel["example"] != PAGE_EXAMPLE:
        raise RuntimeError(f"the rule selects example {sel['example']}, the page shows {PAGE_EXAMPLE}; not switching silently")
    checks = cross_checks(sel, contact, meta)
    t_top, drift, order, amount = sel["frame"], sel["drift"], sel["order"], sel["amount"]
    log.info("selected example %d (largest of %d first-to-last differences: %.3f), top-row frame %d (amount %.1f)",
             sel["example"], len(drift), drift[order[0]], t_top, amount[t_top])

    # ---- data of the example (object frame, metres)
    ex = qlib.load_example(DATASET, sel["example"])
    assert np.array_equal(ex.C, contact[sel["example"]].astype(np.float32))
    verts, faces = ex.mesh(0)                                      # rigid object: the same at every frame
    frames = sorted(set(STRIP_FRAMES) | {t_top})
    hands = {t: ex.hand(t)[0] for t in frames}                     # (778, 3) each
    hand_faces = ex.hand(0)[1]
    on_surface = {t: ex.to_vertices(ex.C[t]) for t in frames}      # (V,) display rule; NaN where not reached
    canonical = canonical_points(ex.meta.category)                 # (512, 3), canonical units
    points = to_object_frame(canonical, verts)                     # (512, 3), metres
    template = template_mesh(ex.meta.category, canonical)
    distance = {t: nearest_hand_distance(verts, hands[t]) for t in frames}
    drawn_max = max(float(np.nanmax(on_surface[t])) for t in frames)
    assert max(drawn_max, float(ex.C[frames].max())) <= VMAX, "a drawn value exceeds the colour scale"

    # ---- cameras: one direction for the figure, one fitted camera per row.
    # hand_side_view looks at the knife's broad side from the side of the hand's centre, which here shows the
    # back of the hand in front of the handle; the opposite broad side shows the handle with the fingers
    # closing around it. A display choice made by looking at both sides (the example was not changed).
    side, up = qlib.hand_side_view(verts, hands[t_top], tilt=CAMERA_TILT)
    direction = -side
    top_cam, top_size = row_camera([verts, hands[t_top], points], direction, up, TOP_PANEL_WIDTH)
    strip_cam, strip_size = row_camera([verts] + [hands[t] for t in STRIP_FRAMES], direction, up, STRIP_PANEL_WIDTH)

    # ---- top row
    surface_rgb = {t: qlib.contact_rgb(on_surface[t], VMAX) for t in frames}
    point_rgb = qlib.contact_rgb(ex.C[t_top].astype(np.float64), VMAX)
    top = [
        qlib.render([qlib.mesh_item(verts, faces), qlib.hand_item(hands[t_top], hand_faces)], top_cam, top_size),
        qlib.render([qlib.mesh_item(verts, faces, rgb=surface_rgb[t_top])], top_cam, top_size),
        qlib.render([qlib.mesh_item(verts, faces, **qlib.GHOST)]
                    + [qlib.points_item(points[k], point_rgb[k], POINT_RADIUS) for k in range(len(points))], top_cam, top_size),
    ]
    # ---- bottom row
    strip = [qlib.render([qlib.mesh_item(verts, faces, rgb=surface_rgb[t]), qlib.hand_item(hands[t], hand_faces)],
                         strip_cam, strip_size) for t in STRIP_FRAMES]

    # ---- layout
    top_block = qlib.grid(
        [top],
        col_labels=[f"Frame {t_top}: the recorded hand on the knife",
                    "The contact map of this frame, drawn on the knife",
                    "The same map as it is stored: one value on each of 512 points"],
        footnote="The same sequence over time, with the hand:")
    strip_block = qlib.grid(
        [strip], col_labels=[f"Frame {t}" for t in STRIP_FRAMES],
        colorbars=[qlib.colorbar("contact", 0.0, VMAX, "Contact value: near 1 where the hand touches, towards 0 away from the hand")])
    image = stack([top_block, strip_block])

    # ---- numbers a caption may quote, each with its source
    seq_file, meta_file = str(SEQUENCES), str(SEQUENCES_META)
    arrays = f"panels.npz of {FIG_ID}"
    second = int(order[1])
    amount_order = np.argsort(-amount, kind="stable")
    covered = ex.covered
    dense_top = ex.dense(t_top)
    high = covered & (on_surface[t_top] >= 0.5 * np.nanmax(on_surface[t_top]))
    low = covered & (on_surface[t_top] <= 0.05 * np.nanmax(on_surface[t_top]))
    off_surface = cKDTree(verts).query(points)[0]                  # canonical point -> nearest mesh vertex
    onset_rule = json.loads(SEQUENCE_CONFIG.read_text(encoding="utf-8"))
    violet = sum(tinted_pixels(p) for p in [top[1]] + strip)
    extent = np.round(np.sort(np.ptp(verts, 0))[::-1], 3)          # this knife: length, blade height, thickness

    def number(label: str, value: float | int, source: str, locator: str) -> dict[str, Any]:
        return {"label": label, "value": value, "dataset": "TACO", "source": source, "locator": locator}

    numbers = [
        number("values per contact map (canonical points)", int(ex.C.shape[1]), seq_file, "key C, last dimension"),
        number("frames per sequence", int(ex.C.shape[0]), seq_file, "key C, second dimension"),
        number("frame of the top row", t_top, arrays, "selection/amount_per_frame, argmax"),
        number("frames of the bottom row", list(STRIP_FRAMES), "docs/build/qual/q_contact_map_primer.py",
               "STRIP_FRAMES (the frames of the page's image f1b_taco_knife_rollout.png)"),
        number("test sequences ranked", int(len(drift)), meta_file, "rows with split_set == test"),
        number("rank of the selected sequence by first-to-last map difference", 1, arrays, "selection/drift, position of the largest value"),
        number("first-to-last map difference of the selected sequence (rank 1)", round(float(drift[order[0]]), 3), arrays,
               f"selection/drift at selection/test_examples == {sel['example']}; computed from {seq_file} key C"),
        number("first-to-last map difference of the runner-up", round(float(drift[second]), 3), arrays,
               f"selection/drift at selection/test_examples == {int(sel['test'][second])}"),
        number("median first-to-last map difference of the test sequences", round(float(np.median(drift)), 3), arrays, "selection/drift, median"),
        number("contact amount at frame 0 (sum of the 512 values)", round(float(amount[0]), 1), arrays,
               f"selection/amount_per_frame[0]; equals mass0 of row {sel['example']} in {meta_file}"),
        number(f"contact amount at frame {t_top} (sum of the 512 values)", round(float(amount[t_top]), 1), arrays,
               f"selection/amount_per_frame[{t_top}]"),
        number("largest of the 512 values at frame 0", round(float(ex.C[0].max()), 2), arrays, "strip_frame_00/map_512, max"),
        number(f"largest of the 512 values at frame {t_top}", round(float(ex.C[t_top].max()), 2), arrays, "top_points/values, max"),
        number(f"largest value drawn on the surface at frame {t_top}", round(float(np.nanmax(on_surface[t_top])), 2), arrays,
               "top_surface/vertex_values, nanmax"),
        number("soft-contact length scale, cm (value = exp(-distance / 2 cm) per mesh vertex, then averaged around each point)",
               round(qlib.SOFT_SCALE * 100, 1), "docs/build/qual/qlib.py", "SOFT_SCALE; hier_contact_gen report, data section"),
        *[number(f"smallest distance between a hand vertex and a knife vertex at frame {t}, mm", round(float(distance[t].min()) * 1000, 1),
                 arrays, f"strip_frame_{t:02d}/hand_distance, min") for t in STRIP_FRAMES],
        number("mesh vertices of this knife", int(len(verts)), arrays, "top_surface/object_verts, rows"),
        number("mesh vertices the 512 points do not reach (drawn pale violet)", int((~covered).sum()), arrays, "top_surface/vertex_values, NaN count"),
        number("median distance of the 512 points from this knife's nearest mesh vertex, mm", round(float(np.median(off_surface)) * 1000, 1),
               arrays, "top_points/xyz against top_surface/object_verts"),
        number("largest distance of the 512 points from this knife's nearest mesh vertex, mm", round(float(off_surface.max()) * 1000, 1),
               arrays, "top_points/xyz against top_surface/object_verts"),
        number(f"correlation between the surface drawing and the per-vertex recorded contact at frame {t_top}",
               round(float(np.corrcoef(dense_top[covered], on_surface[t_top][covered])[0, 1]), 3), arrays,
               "top_surface/vertex_values against check/dense_per_vertex (reached vertices)"),
        number("length of the knife, m", float(extent[0]), arrays, "top_surface/object_verts, largest axis extent"),
        number("blade height of this knife, m", float(extent[1]), arrays, "top_surface/object_verts, second axis extent"),
        number("blade height of the template knife the points were sampled on, m", float(template["extent_m"][1]), str(MESH_DICT),
               f"mesh {template['mesh_id']}, verts_original, second axis extent"),
        number("colour scale, lowest and highest value", [0.0, VMAX], "docs/build/qual/q_contact_map_primer.py", "VMAX"),
        number("the study's distance for a hard contact, cm (a sequence starts where a vertex is this close and the amount is at least zero_mass)",
               round(float(onset_rule["hard_mm"]) * 100, 1), str(SEQUENCE_CONFIG), "key hard_mm (metres); keys firm_rule, zero_mass, firm_persist_frames"),
    ]
    d_first = round(float(distance[0].min()) * 1000, 1)            # the frame-0 entry of the list above

    caption = (
        f"One recorded TACO test sequence, a right hand taking hold of a knife, selected as the test sequence whose contact map differs most "
        f"between its first and its last frame (rank 1 of {len(drift)}), so it is not a typical one. "
        f"Top row, frame {t_top} (the frame of this sequence with the largest contact amount): the recorded hand on the knife; the knife "
        f"alone with that frame's contact map drawn on its surface; and the same map as it is stored, one value on each of {ex.C.shape[1]} points. "
        f"The points come from a template knife of the category, scaled to this knife, so their outline differs from it; the pale shape "
        f"behind them is this knife. "
        f"Bottom row: frames {', '.join(str(t) for t in STRIP_FRAMES[:-1])} and {STRIP_FRAMES[-1]} of the {ex.C.shape[0]}, with the hand. "
        f"The value falls off with the distance to the hand (scale {qlib.SOFT_SCALE * 100:g} cm) instead of switching on at touch, so at "
        f"frame 0, when the hand is still beside the handle ({d_first:g} mm from the knife at its nearest point), the handle is already "
        f"faintly coloured, and the colour deepens as the hand closes on it. "
        f"All panels share one colour scale, 0 to {VMAX:g}.")
    alt = ("Renders of one recorded sequence of a right hand grasping a knife: the hand on the knife, the knife with the contact coloured "
           "on its handle, the same contact as coloured dots, and six frames over time in which the colour on the handle deepens as the hand closes.")

    panels: dict[str, Any] = {
        "selection": {"test_examples": sel["test"], "drift": drift, "amount_per_frame": amount},
        "top_hand_on_object": {"object_verts": verts, "object_faces": faces, "hand_verts": hands[t_top], "hand_faces": hand_faces},
        "top_surface": {"object_verts": verts, "object_faces": faces, "map_512": ex.C[t_top], "vertex_values": on_surface[t_top]},
        "top_points": {"xyz": points, "values": ex.C[t_top], "ghost_object_verts": verts},
        "check": {"dense_per_vertex": dense_top, "hand_distance": distance[t_top], "covered": covered},
    }
    for t in STRIP_FRAMES:
        panels[f"strip_frame_{t:02d}"] = {"hand_verts": hands[t], "map_512": ex.C[t], "vertex_values": on_surface[t],
                                          "hand_distance": distance[t]}

    info = {
        "id": FIG_ID,
        "block": BLOCK,
        "title": "What a contact map is: one recorded TACO sequence, on the object and as 512 points",
        "panels": {
            "top row, left": f"frame {t_top}: the recorded right hand (skin colour) on the knife (grey, no contact drawn)",
            "top row, middle": f"frame {t_top}: the knife alone, its surface coloured by the recorded contact map of this frame",
            "top row, right": f"frame {t_top}: the 512 stored values as dots at the 512 canonical points, in front of a pale ghost of the knife",
            "bottom row": "frames " + ", ".join(str(t) for t in STRIP_FRAMES) + ": the recorded hand on the knife, the knife coloured by the recorded "
                          "contact map of that frame",
            "colour bar": f"contact value 0 to {VMAX:g}, shared by every panel",
        },
        "examples": [{"dataset": "TACO", "example": sel["example"], "sequence_id": ex.meta.sequence_id,
                      "object": f"{ex.meta.category} (a cleaver), mesh {ex.meta.mesh_id}, role {ex.meta.role}", "hand": ex.meta.hand,
                      "split": ex.meta.split, "first_take_frame": ex.meta.t0, "frames_drawn": {"top row": [t_top], "bottom row": list(STRIP_FRAMES)}}],
        "selection_rule": (
            "Sequence: among the TACO test sequences of the hierarchical contact study (split_set == test in sequences_meta.csv), the one with the "
            "largest Euclidean distance between its last and its first recorded 512-value contact map; this is the sequence the page already "
            "shows (example 3011). Top-row frame: the frame of that sequence with the largest contact amount (sum of the 512 values)."),
        "selection_evidence": {
            "metric": "||C_63 - C_0||_2 over the 512 canonical values, float32, per test sequence",
            "candidates": int(len(drift)),
            "chosen": {"example": sel["example"], "rank": 1, "value": float(drift[order[0]])},
            "top_10": [{"rank": r + 1, "example": int(sel["test"][i]), "value": float(drift[i]), "group": meta.group.iloc[int(sel["test"][i])]}
                       for r, i in enumerate(order[:10])],
            "median": float(np.median(drift)), "minimum": float(drift.min()),
            "sorted_values": np.round(drift[order], 4),
            "frame_metric": "sum of the 512 canonical values of a frame",
            "frame_chosen": {"frame": t_top, "value": float(amount[t_top])},
            "frame_top_5": [{"rank": r + 1, "frame": int(t), "value": float(amount[t])} for r, t in enumerate(amount_order[:5])],
            "amount_per_frame": np.round(amount, 2),
            "cross_checks_against_the_study": checks,
        },
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (the study's operator transposed and normalised: a smoothing, not an inverse)",
            "colour_scale": {"function": "qlib.contact_rgb", "range": [0.0, VMAX],
                             "stops": {"0": qlib.ZERO, str(VMAX / 2): qlib.ORANGE, str(VMAX): qlib.DARK},
                             "why": "the soft contact exp(-d / 2 cm) lies in [0, 1], so the scale does not depend on the example",
                             "largest_value_drawn": {"on the surface": drawn_max, "on the points": float(ex.C[t_top].max())},
                             "not_reached": f"{int((~covered).sum())} of {len(verts)} vertices are not reached by the 512 points and are drawn in {qlib.NO_DATA}; "
                                            f"pixels of the coloured panels with that tint: {violet} (no legend entry is drawn for it)"},
            "camera": {"direction_rule": f"the opposite of qlib.hand_side_view(object, hand at frame {t_top}, tilt={CAMERA_TILT}): the knife's broad side, "
                                         "longest axis horizontal, from the side on which the fingers (not the back of the hand) face the viewer; "
                                         "chosen by looking at both broad sides",
                       "direction": direction, "up": up, "margin": CAMERA_MARGIN,
                       "lens": f"perspective; the horizontal field of view is {qlib.VIEW_ANGLE:g} degrees in both rows (the vertical view angle follows from "
                               "the panel's proportions), so the knife has the same perspective in both rows",
                       "top row": {"camera": top_cam, "panel_px": list(top_size), "fitted_on": f"knife, hand at frame {t_top}, the 512 points"},
                       "bottom row": {"camera": strip_cam, "panel_px": list(strip_size), "fitted_on": "knife and the hand at all six frames"}},
            "hand": f"MANO mesh of the recorded right hand, opaque, {qlib.HAND_SKIN}; in the top-left panel it is drawn on the uncoloured knife",
            "points": f"spheres of radius {POINT_RADIUS * 1000:g} mm at canonical point * object radius + object centre (the inverse of the study's "
                      "normalisation), in front of the knife drawn as a ghost (qlib.GHOST)",
            "template_of_the_points": template,
            "layout": "two qlib.grid blocks of equal width stacked by this script (qlib.grid needs equal panels)",
        },
        "numbers": numbers,
        "self_check": {
            f"median distance to the hand of vertices drawn at >= half the largest value, frame {t_top}, mm": round(float(np.median(distance[t_top][high])) * 1000, 1),
            f"median distance to the hand of vertices drawn at <= 5 % of the largest value, frame {t_top}, mm": round(float(np.median(distance[t_top][low])) * 1000, 1),
            "max |W s - C| over the drawn frames (float16 storage)": float(max(np.abs(ex.W @ ex.dense(t).astype(np.float32) - ex.C[t]).max() for t in frames)),
            "max |cached distance - distance of the drawn hand| over the drawn frames, mm":
                float(max(np.abs(-qlib.SOFT_SCALE * np.log(ex.dense(t)) - distance[t]).max() for t in frames) * 1000),
            "pixels of the coloured panels tinted pale violet (vertices the map does not reach)": violet,
            "non-white pixels on the edge of any panel (0 = nothing clipped)": sum(border_pixels(p) for p in top + strip),
        },
        "inference": "none: recorded data only; no checkpoint was loaded and no model was run (the MANO hand mesh is computed on CPU from the recorded hand parameters)",
        "caveats": [
            "One sequence. It is the test sequence with the largest change between its first and last map, so it is the least typical one in that respect "
            f"(its difference is {drift[order[0]] / np.median(drift):.1f} times the test median).",
            f"The top-row frame ({t_top}) is also the last frame of the bottom row, so the bottom row's last panel repeats it with the hand in place.",
            f"In frames 0, 8 and 16 the hand is beside the handle and not closed on it (nearest hand vertex {d_first:g} mm from the knife at frame 0); the faint "
            f"colour there is proximity, not touch. The study starts a sequence at the first frame of 'firm contact', which it defines as "
            f"'{onset_rule['firm_rule']}' (a vertex within {onset_rule['hard_mm'] * 100:g} cm of the hand, amount at least {onset_rule['zero_mass']:g}) for "
            f"{onset_rule['firm_persist_frames']} frames in a row; frame 0 meets it without a grasp.",
            "The picture is coarse twice over: the value decays over 2 cm, and the 512-value map is spread back onto the mesh by the study's own operator, "
            "which smooths. It shows that the whole handle is in contact, not which finger touches where, and the colour reaches a little onto the blade "
            "next to the handle.",
            f"The 512 points come from the category's template knife (mesh {template['mesh_id']}, blade {template['extent_m'][1]:g} m high against "
            f"{extent[1]:g} m here) scaled to this knife, not from this knife: their outline is another knife's, and they lie up to "
            f"{off_surface.max() * 1000:.1f} mm from this knife's nearest mesh vertex (median {np.median(off_surface) * 1000:.1f} mm).",
            "Frames 0, 8 and 16 look almost alike: the hand hardly moves before it closes on the handle.",
            "The opaque hand hides the part of the handle under the fingers in the bottom row; the top-middle panel shows that frame's map without the hand.",
            "No model output is shown; TACO only.",
        ],
        "suggested_caption": caption,
        "suggested_alt": alt,
    }
    paths = qlib.save(FIG_ID, image, panels, info)
    for name, path in paths.items():
        log.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
