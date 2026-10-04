"""Qualitative figure "structure_on_a_grasp" for page block F2: what the 48 numbers are on a real grasp.

One recorded frame of one TACO sequence and of one ARCTIC sequence (one row each, never mixed), each at three
levels of description, all three panels of a row from one camera:

    1  the recorded hand on the object mesh, the six hand parts in the six part colours
    2  the object surface coloured by the hand part that is nearest to it, where the hand is within 1 cm
       (full colour: the vertices that count as touching) or within 2 cm (pale: they only add to the amount)
    3  the object as a ghost with nothing but the 48 numbers drawn: per touching part one translucent disc at its
       contact centroid (area proportional to its contact amount) and one arrow along its contact normal

Under each row a strip of six bars shows which parts touch in each of the 64 frames of the sequence.

What a reader cannot see is measured here and written to meta.json (key ``visibility``) instead of being left to
the eye: which touching vertices the object hides from the camera in panel 2, how far each hand part is from the
object surface, how much the discs overlap in panel 3, how much of every arrow shows there (the script stops if an
arrow does not show, because the column label promises a disc and an arrow for each touching part) and whether an
arrowhead lies behind a disc. The caption's clauses about these things are written from the measured values.

Data only, no model. The 48 numbers are read from the feature cache of the structure / variance study
(``reports/structure_variance_boundary/<dataset>/cache/features.npz``: n_hard, m, p, nrm, length, centroid). Before
anything is drawn, the script recomputes them from the geometry it draws (the mesh, the hand, the recorded
distances) with the study's rules and stops if they do not agree: this is the check that the rescaled centroids
lie on the contact region of the same frame.

Selection (computed here, evidence in meta.json). The frame is the sequence's frame of largest contact amount
(sum of the six amounts; first such frame). TACO: the sequence already shown on the page (example 3011).
ARCTIC: the test sequence whose number of touching parts at that frame is the median over the test sequences
(smallest example index among the ties).

Run (CPU, software renderer; see README.md):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_structure_on_a_grasp.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")      # shared machine: keep the few linear-algebra calls small

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import qlib

log = logging.getLogger("q_structure_on_a_grasp")

FIG_ID = "structure_on_a_grasp"
BLOCK = "F2"
TITLE = "What the 48 numbers are on a real grasp"
DATASETS: tuple[str, ...] = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
PAGE_EXAMPLE = 3011                     # TACO sequence already shown on the page (docs/evidence/images.json)

# ------------------------------------------------------------------------------ inputs (all read-only)
STUDY = qlib.RESULT_ROOT / "reports/structure_variance_boundary"
DEFINITIONS = STUDY / "representation_definitions.json"
PAGE_IMAGES = qlib.REPO / "docs/evidence/images.json"
PAGE_IMAGE_KEY = "assets/img/qual_contact_map_primer.webp"   # the page figure that introduces this sequence
ARCTIC_TEMPLATES = qlib.REPO / "third_party/BimArt/data/arctic_raw/meta/object_vtemplates"   # millimetres
RULES_SOURCE = ("scripts/research/structure_variance_boundary/build_features.py (part_features), sv_common.py "
                "(causal_majority, N_MIN_VERTS), scripts/research/wrench_counterfactual/wc_common.py (contact_geometry)")


def feature_path(dataset: str) -> Path:
    return STUDY / dataset / "cache/features.npz"


def manifest_path(dataset: str) -> Path:
    return qlib.HIER[dataset] / "manifests/fixed0.json"


def second_table_path(dataset: str) -> Path:
    """A later study's per-sequence table that stores the recorded participation of every test sequence (a_true)."""
    return qlib.RESULT_ROOT / "reports/structure_aware_temporal_generation" / dataset / "metrics/D0_seed0_A.npz"


# ------------------------------------------------------------------------------ the study's rules (re-implemented, checked below)
HARD = 0.01                 # metres: a vertex within 1 cm of the hand counts as touching
SOFT = 0.02                 # metres: vertices within 2 cm add to the contact amount
MIN_VERTS = 3               # a part touches in a frame when at least 3 of its vertices are within HARD ...
MAJORITY = 3                # ... in at least 2 of the frames t-2, t-1, t
CANCEL = 0.2                # mean normal shorter than this: the normal of the vertex nearest the centroid is stored

# ------------------------------------------------------------------------------ display choices
PANEL = (598, 380)          # pixels (width, height) of one rendered panel; the figure stays below qlib.ASSET_MAX_WIDTH
NEAR = 0.06                 # metres: the camera frames the hand and the object surface within NEAR of the hand
CAMERA_MARGIN = 0.06
N_DIRECTIONS = 4000         # candidate view directions (Fibonacci lattice on the sphere)
DISC_AREA_SHARE = 0.25      # disc area = DISC_AREA_SHARE * amount * object surface area
DISC_OPACITY = 0.5          # discs are translucent: a disc or an arrow behind another disc stays visible
ARROW_LENGTH = 0.035        # metres
VISIBLE_STEP = 32           # of 255: a pixel shows an arrow when leaving the arrow out changes a channel by more than this
ARROW_MIN_SHARE = 0.5       # the script stops if less than this share of an arrow's pixels shows in panel 3
HEAD_SHARE = 0.3            # qlib draws the last 30 % of an arrow as its head (tip_length in qlib._polydata)
HEAD_DIM = 0.5              # the caption names an arrow if less than this share of its head shows with no disc in front
RAY_TOLERANCE = 5e-4        # metres: a vertex is visible when nothing is hit this much before it on the ray from the camera
FAR_SIDE = 0.5              # the caption names the part with the smallest visible share of touching vertices if below this
HOVER = 0.005               # metres: the caption names a touching part that stays further than this from the object surface
PALE = 0.45                 # share of the part colour on vertices between HARD and SOFT
HAND_TINT = 0.55            # share of the part colour on the hand (the rest is the page's hand colour)
STRIP_OFF = "#eceae4"       # a bar where the part does not touch
BAR_SHARE = 0.66            # height of a bar as a share of the distance between two bars
EDGE_ON_DEG = 20.0          # the caption says "seen edge-on" when the view is this close to the broad side of an object
FLAT = 0.3                  # ... whose thinnest extent is below this share of its middle extent (standard deviations of the vertices)
STRIP_TICKS = (0, 16, 32, 48, 63)
STRIP_TITLE = "Which hand parts touch the object in each of the 64 frames (\N{BLACK DOWN-POINTING TRIANGLE} marks the frame drawn above)"
SLOT = (255, 0, 255)        # placeholder colour of the strip slot until the strip is pasted
COLUMN_LABELS = ("1. Hand on the object",
                 "2. Contact on the surface, coloured by hand part",
                 "3. Only the 48 numbers: a disc and an arrow for each touching part")
LEGEND = "Hand part (same colours on the hand, on the surface, for the discs and arrows, and in the bars)"


# ------------------------------------------------------------------------------ the feature cache
@dataclass(frozen=True)
class Features:
    """The cached per-frame descriptor of every sequence of one dataset (stored frame i = sequence frame i - PAD)."""
    path: Path
    n_hard: np.ndarray        # (N, 80, 6) vertices of the part within HARD
    amount: np.ndarray        # (N, 80, 6) area-weighted soft contact of the part, a share of the object's surface
    centroid: np.ndarray      # (N, 80, 6, 3) (centroid of the part's touching vertices - mesh centroid) / length
    normal: np.ndarray        # (N, 80, 6, 3) unit normal of the part's contact, pointing to the hand
    length: np.ndarray        # (N,) metres
    origin: np.ndarray        # (N, 3) mesh centroid, metres
    touching: np.ndarray      # (N, 64, 6) bool, the study's participation for sequence frames 0 .. 63
    total: np.ndarray         # (N, 64) sum of the six amounts

    def centre(self, n: int, frame: int) -> np.ndarray:
        """(6, 3) contact centroids in the object frame, metres."""
        return self.centroid[n, frame + qlib.PAD].astype(np.float64) * float(self.length[n]) + self.origin[n]


def causal_majority(raw: np.ndarray) -> np.ndarray:
    """(N, F, 6) bool -> the same, true where the indicator holds in at least 2 of the frames i-2, i-1, i
    (the first frame is repeated before the start), as in the study's sv_common.causal_majority."""
    a = raw.astype(np.int8)
    prev1 = np.concatenate([a[:, :1], a[:, :-1]], 1)
    prev2 = np.concatenate([a[:, :1], a[:, :1], a[:, :-2]], 1)
    return (a + prev1 + prev2) >= MAJORITY - 1


def load_features(dataset: str) -> Features:
    path = feature_path(dataset)
    with np.load(path) as z:
        n_hard, amount = z["n_hard"], z["m"]
        frames = slice(qlib.PAD, qlib.PAD + qlib.T)
        return Features(path, n_hard, amount, z["p"], z["nrm"], z["length"], z["centroid"],
                        causal_majority(n_hard >= MIN_VERTS)[:, frames], amount[:, frames].astype(np.float64).sum(-1))


def test_examples(dataset: str) -> np.ndarray:
    return np.sort(np.asarray(json.loads(manifest_path(dataset).read_text())["test"], dtype=np.int64))


# ------------------------------------------------------------------------------ selection
@dataclass(frozen=True)
class Selection:
    dataset: str
    example: int
    frame: int
    rule: str
    evidence: dict[str, Any]


def peak_frames(feat: Features) -> np.ndarray:
    """(N,) per sequence the first frame of largest contact amount."""
    return feat.total.argmax(1)


def frame_evidence(feat: Features, n: int) -> dict[str, Any]:
    total = feat.total[n]
    order = np.argsort(-total, kind="stable")
    frame = int(order[0])
    return {"contact_amount_by_frame": [round(float(x), 5) for x in total], "frame": frame,
            "contact_amount_at_frame": float(total[frame]),
            "five_largest_frames": [{"frame": int(i), "contact_amount": float(total[i])} for i in order[:5]],
            "touching_parts_at_frame": int(feat.touching[n, frame].sum())}


def count_evidence(feat: Features, test: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Per test sequence the number of touching parts at its own frame of largest contact amount."""
    counts = feat.touching[test, peak_frames(feat)[test]].sum(-1)
    ranked = np.sort(counts)
    return counts, {"n_test_sequences": int(len(test)),
                    "histogram_of_touching_parts": {str(k): int((counts == k).sum()) for k in range(len(qlib.PART_NAMES) + 1)},
                    "lower_middle_value": int(ranked[(len(ranked) - 1) // 2]), "upper_middle_value": int(ranked[len(ranked) // 2]),
                    "median": float(np.median(counts))}


def select_taco(feat: Features) -> Selection:
    test = test_examples("taco")
    entry = json.loads(PAGE_IMAGES.read_text()).get(PAGE_IMAGE_KEY, {})
    shown = [e for e in entry.get("examples", []) if e.get("dataset") == "TACO" and e.get("example") == PAGE_EXAMPLE]
    page = f"{entry.get('shows', '')} (TACO example {PAGE_EXAMPLE}, {shown[0].get('sequence_id', '')})" if shown else ""
    if not shown:
        raise RuntimeError(f"{PAGE_IMAGES} no longer documents example {PAGE_EXAMPLE} as the page's sequence (entry "
                           f"{PAGE_IMAGE_KEY!r}); point PAGE_IMAGE_KEY at a page figure that shows it, or change the rule")
    counts, evidence = count_evidence(feat, test)
    own = frame_evidence(feat, PAGE_EXAMPLE)
    evidence.update(own, page_reference={"file": str(PAGE_IMAGES), "key": PAGE_IMAGE_KEY, "text": page},
                    in_test_split=bool(PAGE_EXAMPLE in set(test.tolist())),
                    test_sequences_with_the_same_number_of_touching_parts=int((counts == own["touching_parts_at_frame"]).sum()))
    rule = (f"TACO: the sequence already shown on the page (example {PAGE_EXAMPLE}), at its frame of largest contact "
            "amount (sum of the six per-part amounts of the feature cache; first such frame).")
    return Selection("taco", PAGE_EXAMPLE, own["frame"], rule, evidence)


def select_arctic(feat: Features) -> Selection:
    test = test_examples("arctic")
    counts, evidence = count_evidence(feat, test)
    median = evidence["lower_middle_value"]
    ties = test[counts == median]
    n = int(ties.min())
    ranked = np.sort(counts)
    evidence.update(frame_evidence(feat, n), touching_parts_sorted=[int(c) for c in ranked],
                    ranks_holding_the_median_value=[int(np.searchsorted(ranked, median, "left")) + 1,
                                                    int(np.searchsorted(ranked, median, "right"))],
                    n_sequences_with_the_median_value=int(len(ties)), examples_with_the_median_value=[int(i) for i in ties],
                    chosen="smallest example index among them")
    rule = ("ARCTIC: every test sequence is taken at its frame of largest contact amount (sum of the six per-part "
            "amounts; first such frame) and the number of touching parts there is counted; the sequence with the median "
            "count is drawn (lower middle value if the two middle values differ; smallest example index among the ties).")
    return Selection("arctic", n, int(peak_frames(feat)[n]), rule, evidence)


# ------------------------------------------------------------------------------ one frame: geometry, cached numbers, checks
@dataclass(frozen=True)
class Grasp:
    selection: Selection
    example: qlib.Example
    verts: np.ndarray          # (V, 3) object, metres, the object's own frame
    faces: np.ndarray          # (F, 3)
    hand_verts: np.ndarray     # (778, 3)
    hand_faces: np.ndarray     # (1538, 3)
    distance: np.ndarray       # (V,) recorded distance from each object vertex to the hand, metres
    part: np.ndarray           # (V,) part of the hand vertex nearest to each object vertex
    area: float                # surface area of the drawn mesh, m^2
    touching: np.ndarray       # (6,) bool   } the 48 numbers of this frame, from the feature cache:
    amount: np.ndarray         # (6,)        } 6 on/off + 6 amounts + 6 centroids (18) + 6 normals (18);
    centre: np.ndarray         # (6, 3) m    } centroid and normal count only for touching parts
    normal: np.ndarray         # (6, 3)      }
    n_hard: np.ndarray         # (6,) cached number of vertices within HARD
    radius: np.ndarray         # (6,) disc radius, metres (0 where the part does not touch)
    strip: np.ndarray          # (64, 6) bool, the cached participation of the whole sequence
    surface_gap: np.ndarray    # (6,) smallest distance from the part's hand vertices to the object surface, metres
    edge_near: float           # median edge length of the object's triangles that have a vertex within SOFT of the hand, metres
    checks: dict[str, Any]


def recompute(verts: np.ndarray, faces: np.ndarray, hand: np.ndarray, distance: np.ndarray, part: np.ndarray,
              nearest: np.ndarray) -> dict[str, np.ndarray]:
    """The study's per-part descriptor of one frame, from the geometry that is drawn (build_features.part_features)."""
    mesh = trimesh.Trimesh(verts, faces, process=False)
    trimesh.repair.fix_normals(mesh)
    normals = np.asarray(mesh.vertex_normals, float)
    vertex_area = np.zeros(len(verts))
    for j in range(3):
        np.add.at(vertex_area, faces[:, j], mesh.area_faces / 3.0)
    to_hand = np.where((((hand[nearest] - verts) * normals).sum(1) > 0)[:, None], normals, -normals)
    k_parts = len(qlib.PART_NAMES)
    out = {"n_hard": np.zeros(k_parts, np.int64), "amount": np.zeros(k_parts), "centre": np.full((k_parts, 3), np.nan),
           "normal": np.full((k_parts, 3), np.nan), "mean_normal_length": np.full(k_parts, np.nan),
           "to_nearest_touching_vertex": np.full(k_parts, np.nan), "area": np.asarray(vertex_area.sum())}
    for k in range(k_parts):
        soft = (part == k) & (distance < SOFT)
        hard = soft & (distance < HARD)
        out["amount"][k] = (np.exp(-distance[soft] / qlib.SOFT_SCALE) * vertex_area[soft]).sum() / vertex_area.sum()
        out["n_hard"][k] = hard.sum()
        if hard.any():
            centre = verts[hard].mean(0)
            mean = to_hand[hard].mean(0)
            out["mean_normal_length"][k] = np.linalg.norm(mean)
            away = np.linalg.norm(verts[hard] - centre, axis=1)
            if out["mean_normal_length"][k] < CANCEL:        # normals cancel (the part touches both sides of a thin shape)
                mean = to_hand[hard][away.argmin()]
            out["centre"][k], out["normal"][k] = centre, mean / np.linalg.norm(mean)
            out["to_nearest_touching_vertex"][k] = away.min()
    return out


def study_area(example: qlib.Example, drawn_area: float) -> float:
    """Surface area the study divided by. TACO: the drawn mesh. ARCTIC: the full template mesh; the drawn mesh
    lacks its few triangles that join the two parts."""
    if example.meta.dataset != "arctic":
        return drawn_area
    template = trimesh.load(ARCTIC_TEMPLATES / example.meta.category / "mesh.obj", process=False)
    return float(template.area) / 1e6


def load_grasp(sel: Selection, feat: Features) -> Grasp:
    n, t = sel.example, sel.frame
    ex = qlib.load_example(sel.dataset, n)
    verts, faces = ex.mesh(t)
    hand, hand_faces = ex.hand(t)
    distance = -qlib.SOFT_SCALE * np.log(ex.dense(t))
    by_tree, nearest = cKDTree(hand).query(verts)
    part = qlib.mano_parts()[nearest]
    i = t + qlib.PAD
    touching, amount = feat.touching[n, t], feat.amount[n, i].astype(np.float64)
    centre, normal, n_hard = feat.centre(n, t), feat.normal[n, i].astype(np.float64), feat.n_hard[n, i].astype(np.int64)
    mine = recompute(verts, faces, hand, distance, part, nearest)
    area = float(mine["area"])
    has = n_hard >= 1

    # ---- the cache must describe the frame that is drawn
    close = distance < 0.05
    hand_gap = float(np.abs(by_tree - distance)[close].max())
    if hand_gap > 1e-4:
        raise RuntimeError(f"{sel.dataset} {n}: the drawn hand is not the hand of the recorded distances ({hand_gap:.2e} m)")
    if not np.array_equal(mine["n_hard"], n_hard):
        raise RuntimeError(f"{sel.dataset} {n}: touching vertices per part {mine['n_hard']} differ from the cache {n_hard}")
    if touching.any() and not has[touching].all():
        raise RuntimeError(f"{sel.dataset} {n}: a part counted as touching has no vertex within 1 cm in frame {t}")
    centre_gap = np.linalg.norm(mine["centre"][has] - centre[has], axis=1)
    cosine = (mine["normal"][has] * normal[has]).sum(1)
    ratio = area / study_area(ex, area)
    amount_gap = np.abs(mine["amount"][amount > 0] * ratio / amount[amount > 0] - 1)
    raw_gap = np.abs(mine["amount"][amount > 0] / amount[amount > 0] - 1)
    if centre_gap.max() > 1e-4 or cosine.min() < 0.999 or amount_gap.max() > 2e-3:
        raise RuntimeError(f"{sel.dataset} {n}: cache and recomputation disagree: centroid {centre_gap.max():.2e} m, "
                           f"normal cosine {cosine.min():.4f}, amount {amount_gap.max():.2e}")
    if mine["to_nearest_touching_vertex"][touching].max() > HARD:
        raise RuntimeError(f"{sel.dataset} {n}: a centroid is more than 1 cm from the part's nearest touching vertex")
    with np.load(second_table_path(sel.dataset)) as z:
        rows = np.flatnonzero(z["example"] == n)
        second = bool(len(rows) == 1 and np.array_equal(z["a_true"][rows[0]] > 0.5, feat.touching[n]))
    if not second:
        raise RuntimeError(f"{sel.dataset} {n}: participation differs from {second_table_path(sel.dataset)}")

    radius = np.where(touching, np.sqrt(DISC_AREA_SHARE * amount * area / np.pi), 0.0)
    # ---- what "touching" means on this mesh: the gap between each hand part and the surface, the size of the triangles
    to_surface = trimesh.proximity.closest_point(trimesh.Trimesh(verts, faces, process=False), hand)[1]
    surface_gap = np.array([to_surface[qlib.mano_parts() == k].min() for k in range(len(qlib.PART_NAMES))])
    edges = np.linalg.norm(verts[faces] - verts[faces[:, [1, 2, 0]]], axis=2)                 # (F, 3)
    edge_near = float(np.median(edges[(distance[faces] < SOFT).any(1)]))
    checks = {
        "hand_vs_recorded_distance_max_m": hand_gap,
        "touching_vertices_per_part": {"cache": n_hard.tolist(), "recomputed": mine["n_hard"].tolist()},
        "centroid_cache_vs_recomputed_max_mm": float(centre_gap.max() * 1e3),
        "centroid_to_nearest_touching_vertex_of_its_part_mm": _by_part(mine["to_nearest_touching_vertex"] * 1e3, touching),
        "normal_cosine_cache_vs_recomputed_min": float(cosine.min()),
        "mean_normal_length_before_normalising": _by_part(mine["mean_normal_length"], has),
        "parts_whose_normal_is_the_nearest_vertex_normal": [qlib.PART_NAMES[k] for k in np.flatnonzero(has)
                                                             if mine["mean_normal_length"][k] < CANCEL],
        "amount": {"cache": amount.tolist(), "recomputed_on_the_drawn_mesh": mine["amount"].tolist()},
        "amount_recomputed_vs_cache_max_relative_difference": float(raw_gap.max()),
        "amount_after_area_correction_max_relative_difference": float(amount_gap.max()),
        "drawn_mesh_area_m2": area, "study_mesh_area_m2": area / ratio, "area_ratio_drawn_to_study": ratio,
        "participation_equals_second_table": second, "second_table": str(second_table_path(sel.dataset)),
        "hand_part_to_object_surface_mm": _by_part(surface_gap * 1e3, np.ones(len(surface_gap), bool)),
        "nearest_object_vertex_of_each_touching_part_mm": {qlib.PART_NAMES[k]: float(distance[part == k].min() * 1e3)
                                                           for k in np.flatnonzero(touching)},
        "median_edge_of_triangles_within_2cm_of_the_hand_mm": edge_near * 1e3,
        "median_edge_of_all_triangles_mm": float(np.median(edges) * 1e3),
    }
    return Grasp(sel, ex, verts, faces, hand, hand_faces, distance, part, area, touching, amount, centre, normal, n_hard,
                 radius, feat.touching[n], surface_gap, edge_near, checks)


def _by_part(values: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {qlib.PART_NAMES[k]: float(values[k]) for k in np.flatnonzero(mask)}


# ------------------------------------------------------------------------------ camera
def fibonacci_sphere(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    polar, turn = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(turn) * np.sin(polar), np.sin(turn) * np.sin(polar), np.cos(polar)], 1)


def wrist_axis(hand: np.ndarray, hand_faces: np.ndarray) -> np.ndarray:
    """Unit vector from the hand's centre to the centre of the open rim of the hand mesh (the wrist)."""
    edges = np.sort(np.concatenate([hand_faces[:, [0, 1]], hand_faces[:, [1, 2]], hand_faces[:, [2, 0]]]), 1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    rim = np.unique(unique[count == 1])
    axis = hand[rim].mean(0) - hand.mean(0)
    return axis / np.linalg.norm(axis)


def glyph_view(g: Grasp) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """(direction, up) for qlib.fit_camera, computed from the frame's own numbers.

    direction: of N_DIRECTIONS directions spread evenly over the sphere, leaving out those that look at the hand
    from behind the wrist (into the open end of the hand mesh), the one that maximises
    sum_k max(c_k, 0) sqrt(1 - c_k^2), c_k = cosine between the direction and the contact normal of touching part k.
    A disc seen at 45 degrees shows both its face and its arrow. up: the direction in which the framed points
    (hand, nearby object surface, discs, arrows) extend most, as seen from that direction, runs horizontally, with
    the wrist on the left."""
    directions, cos, score, wrist = view_candidates(g)
    behind = directions @ wrist
    best = int(np.where(behind <= 0, score, -np.inf).argmax())
    back = directions[best]
    up = view_up(np.concatenate(framed_points(g)), back, wrist)
    spread, axes = np.linalg.eigh(np.cov((g.verts - g.verts.mean(0)).T))   # columns: thinnest, middle, longest axis of the object
    info = {"direction": back.tolist(), "up": up.tolist(), "score": float(score[best]),
            "best_score_of_any_direction": float(score.max()),
            # how far a camera could go towards seeing every disc from the front: over the allowed directions, the
            # largest value of the smallest cosine (negative: from every direction some disc is seen from behind)
            "largest_smallest_cosine_of_any_allowed_direction": float(np.where(behind <= 0, cos.min(1), -np.inf).max()),
            "angle_to_broad_side_deg": float(np.degrees(np.arcsin(min(1.0, abs(back @ axes[:, 0]))))),
            "thinnest_to_middle_extent": float(np.sqrt(spread[0] / spread[1])),
            "cosine_to_contact_normals": {qlib.PART_NAMES[k]: float(c) for k, c in zip(np.flatnonzero(g.touching), cos[best])},
            "cosine_to_wrist_axis": float(behind[best])}
    return back, up, info


def view_candidates(g: Grasp) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The candidate view directions (N, 3), their cosines to the contact normals of the touching parts (N, parts),
    the score of the camera rule (N,) and the wrist axis (3,)."""
    directions = fibonacci_sphere(N_DIRECTIONS)
    cos = directions @ g.normal[g.touching].T
    score = (np.clip(cos, 0, 1) * np.sqrt(np.clip(1 - cos ** 2, 0, None))).sum(1)
    return directions, cos, score, wrist_axis(g.hand_verts, g.hand_faces)


def view_up(framed: np.ndarray, back: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """The camera's up vector for view direction ``back``: the framed points' longest extent, as the camera sees
    them, runs horizontally, with the wrist on the left."""
    flat = framed - framed.mean(0)
    flat -= np.outer(flat @ back, back)                               # the framed points as the camera sees them
    right = np.linalg.svd(flat, full_matrices=False)[2][0]
    right -= (right @ back) * back
    right /= np.linalg.norm(right)
    if right @ wrist > 0:
        right = -right
    return np.cross(back, right)


def clear_head_alternative(g: Grasp) -> dict[str, Any]:
    """Not used for the figure; recorded to show what a stricter camera rule would cost. The camera rule's choice
    if the directions from which an arrowhead lies behind a disc were left out as well (geometry test of
    heads_behind_discs, with the camera fitted for every allowed direction)."""
    directions, cos, score, wrist = view_candidates(g)
    allowed = np.flatnonzero(directions @ wrist <= 0)
    framed = framed_points(g)
    stacked = np.concatenate(framed)
    clear = np.zeros(len(directions), bool)
    for i in allowed:
        camera = qlib.fit_camera(framed, directions[i], view_up(stacked, directions[i], wrist), margin=CAMERA_MARGIN,
                                 aspect=PANEL[0] / PANEL[1])
        clear[i] = not any(heads_behind_discs(g, camera).values())
    out: dict[str, Any] = {"allowed_directions": int(len(allowed)), "of_them_with_no_arrowhead_behind_a_disc": int(clear.sum())}
    if clear.any():
        best = int(np.where(clear, score, -np.inf).argmax())
        out.update(direction=directions[best].tolist(), score=float(score[best]),
                   cosine_to_contact_normals={qlib.PART_NAMES[k]: float(c) for k, c in zip(np.flatnonzero(g.touching), cos[best])})
    return out


def framed_points(g: Grasp) -> list[np.ndarray]:
    """What the camera must contain: the hand, the object surface within NEAR of it, every disc and arrow."""
    points = [g.hand_verts, g.verts[g.distance < NEAR]]
    for k in np.flatnonzero(g.touching):
        points.append(np.vstack([disc_rim(g, k, 4), g.centre[k] + ARROW_LENGTH * g.normal[k]]))
    return points


def disc_rim(g: Grasp, k: int, n: int) -> np.ndarray:
    """(n, 3) points evenly spaced on the rim of part k's disc."""
    u = np.cross(g.normal[k], (1.0, 0.0, 0.0) if abs(g.normal[k][0]) < 0.9 else (0.0, 1.0, 0.0))
    u /= np.linalg.norm(u)
    w = np.cross(g.normal[k], u)
    turn = 2 * np.pi * np.arange(n) / n
    return g.centre[k] + g.radius[k] * (np.outer(np.cos(turn), u) + np.outer(np.sin(turn), w))


def project(points: np.ndarray, camera: qlib.Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(x, y, depth) of ``points`` for the perspective camera: x and y run from -1 to 1 across the picture (x to the
    right, y upwards), depth is the distance in front of the camera along its axis."""
    position, back = np.asarray(camera.position), np.asarray(camera.position) - np.asarray(camera.focal_point)
    back /= np.linalg.norm(back)
    up = np.asarray(camera.up)
    q = np.asarray(points, float).reshape(-1, 3) - position
    depth = -(q @ back)
    tan_y = math.tan(math.radians(camera.view_angle) / 2)
    return (q @ np.cross(up, back)) / (depth * tan_y * camera.aspect), (q @ up) / (depth * tan_y), depth


def to_pixels(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """project()'s x, y -> (column, row) of a PANEL-sized image, not rounded."""
    return (x + 1) / 2 * PANEL[0], (1 - y) / 2 * PANEL[1]


def share_in_frame(points: np.ndarray, camera: qlib.Camera) -> float:
    """Share of ``points`` that the perspective camera sees."""
    x, y, depth = project(points, camera)
    return float(((depth > 0) & (np.abs(x) <= 1) & (np.abs(y) <= 1)).mean())


# ------------------------------------------------------------------------------ panels
def hand_colours() -> np.ndarray:
    """(778, 3) the page's hand colour tinted with the colour of each vertex's part."""
    return (1 - HAND_TINT) * _rgb(qlib.HAND_SKIN) + HAND_TINT * qlib.PART_COLORS[qlib.mano_parts()]


def surface_colours(g: Grasp) -> np.ndarray:
    """(V, 3) part colour where the hand is within HARD, a pale part colour up to SOFT, neutral grey elsewhere."""
    weight = np.where(g.distance < HARD, 1.0, np.where(g.distance < SOFT, PALE, 0.0))[:, None]
    return (1 - weight) * _rgb(qlib.ZERO) + weight * qlib.PART_COLORS[g.part]


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0


def disc_item(g: Grasp, k: int) -> dict[str, Any]:
    """Part k's disc, translucent. qlib.discs_item has no opacity parameter; qlib.render reads the ``opacity`` key of
    any item (and then draws the item's front faces only), so the key is added here."""
    return {**qlib.discs_item(g.centre[k], g.normal[k], g.radius[k], qlib.PART_HEX[k]), "opacity": DISC_OPACITY}


def arrow_item(g: Grasp, k: int) -> dict[str, Any]:
    return qlib.arrows_item(g.centre[k], g.normal[k], qlib.PART_HEX[k], scale=ARROW_LENGTH)


def number_items(g: Grasp, without_arrow: int | None = None) -> list[dict[str, Any]]:
    """Panel 3: the ghost object, a disc and an arrow per touching part (optionally leaving one arrow out)."""
    items = [qlib.mesh_item(g.verts, g.faces, **qlib.GHOST)]
    for k in np.flatnonzero(g.touching):
        items.append(disc_item(g, k))
        if k != without_arrow:
            items.append(arrow_item(g, k))
    return items


def render_row(g: Grasp, camera: qlib.Camera) -> list[np.ndarray]:
    hand_rgb, surface_rgb = hand_colours(), surface_colours(g)
    scenes = ([qlib.mesh_item(g.verts, g.faces), qlib.hand_item(g.hand_verts, g.hand_faces, rgb=hand_rgb)],
              [qlib.mesh_item(g.verts, g.faces, rgb=surface_rgb)],
              number_items(g))
    return [clean_border(qlib.render(items, camera, PANEL)) for items in scenes]


def clean_border(image: np.ndarray) -> np.ndarray:
    """Replace the outermost pixel line of a render by its neighbour. The renderer's anti-aliasing wraps around the
    image border, so an object that leaves the picture on one side leaves a faint line on the opposite side."""
    image = image.copy()
    image[:, 0], image[:, -1] = image[:, 1], image[:, -2]
    image[0], image[-1] = image[1], image[-2]
    return image


# ------------------------------------------------------------------------------ what the camera hides (measured, for caption and caveats)
def visible_touching(g: Grasp, camera: qlib.Camera) -> np.ndarray:
    """(V,) bool: the vertex is within HARD of the hand, inside the picture and not hidden by the object itself (on
    the ray from the camera nothing is hit more than RAY_TOLERANCE before the vertex)."""
    index = np.flatnonzero(g.distance < HARD)
    position = np.asarray(camera.position)
    ray = g.verts[index] - position
    length = np.linalg.norm(ray, axis=1)
    mesh = trimesh.Trimesh(g.verts, g.faces, process=False)
    hits, ray_id, _ = mesh.ray.intersects_location(np.repeat(position[None], len(index), 0), ray / length[:, None],
                                                   multiple_hits=True)
    first = np.full(len(index), np.inf)
    np.minimum.at(first, ray_id, np.linalg.norm(hits - position, axis=1))
    x, y, depth = project(g.verts[index], camera)
    out = np.zeros(len(g.verts), bool)
    out[index] = (first > length - RAY_TOLERANCE) & (depth > 0) & (np.abs(x) <= 1) & (np.abs(y) <= 1)
    return out


def disc_overlap(g: Grasp, camera: qlib.Camera) -> dict[str, dict[str, Any]]:
    """Per touching part, from the projected outline of its disc: its size in pixels, the share that other discs
    overlap, the share that lies behind a disc whose centre is nearer to the camera, and which discs these are."""
    parts = [int(k) for k in np.flatnonzero(g.touching)]
    masks, depth = {}, {}
    for k in parts:
        x, y, _ = project(disc_rim(g, k, 96), camera)
        outline = Image.new("1", PANEL, 0)
        ImageDraw.Draw(outline).polygon([(float(a), float(b)) for a, b in zip(*to_pixels(x, y))], fill=1)
        masks[k], depth[k] = np.asarray(outline, bool), float(project(g.centre[k], camera)[2][0])
    out = {}
    for k in parts:
        others = [j for j in parts if j != k and (masks[j] & masks[k]).any()]
        nearer = [j for j in others if depth[j] < depth[k]]
        size = int(masks[k].sum())
        out[qlib.PART_NAMES[k]] = {
            "disc_pixels": size,
            "share_overlapped_by_other_discs": float((masks[k] & np.any([masks[j] for j in others], 0)).sum() / size) if others else 0.0,
            "share_behind_a_nearer_disc": float((masks[k] & np.any([masks[j] for j in nearer], 0)).sum() / size) if nearer else 0.0,
            "overlaps": [qlib.PART_NAMES[j] for j in others], "behind": [qlib.PART_NAMES[j] for j in nearer]}
    return out


def heads_behind_discs(g: Grasp, camera: qlib.Camera) -> dict[str, dict[str, float]]:
    """Per touching part, from the geometry alone: the share of its arrowhead (9 points on the arrow's axis over the
    last HEAD_SHARE of its length) that lies behind each disc as the camera sees it; discs with share 0 are left out.
    The arrow of a disc seen from behind can lie behind its own disc."""
    parts = [int(k) for k in np.flatnonzero(g.touching)]
    position = np.asarray(camera.position)
    out: dict[str, dict[str, float]] = {}
    for k in parts:
        points = g.centre[k] + np.outer(np.linspace(1 - HEAD_SHARE, 1, 9) * ARROW_LENGTH, g.normal[k])
        ray = position - points
        reach = np.linalg.norm(ray, axis=1)
        ray /= reach[:, None]
        out[qlib.PART_NAMES[k]] = {}
        for j in parts:
            with np.errstate(divide="ignore", invalid="ignore"):
                step = ((g.centre[j] - points) @ g.normal[j]) / (ray @ g.normal[j])     # along the ray to the disc's plane
            crossing = points + step[:, None] * ray
            behind = (step > 1e-9) & (step < reach) & (np.linalg.norm(crossing - g.centre[j], axis=1) <= g.radius[j])
            if behind.any():
                out[qlib.PART_NAMES[k]][qlib.PART_NAMES[j]] = float(behind.mean())
    return out


def arrow_visibility(g: Grasp, camera: qlib.Camera, panel: np.ndarray) -> dict[str, dict[str, Any]]:
    """Per touching part, measured on renders from the row's camera.

    arrow_pixels_alone: the pixels the arrow covers when it is rendered alone. arrow_pixels_showing: those of them
    where ``panel`` (the third panel as drawn) differs from the same panel rendered without that arrow by more than
    VISIBLE_STEP in some channel; an arrow behind a translucent disc or the ghost object still counts if it changes
    the colour that much. head_pixels: the arrow's pixels over the last HEAD_SHARE of its projected length.
    head_pixels_clear: those of them that show and that look the same (to VISIBLE_STEP) when all discs are left out,
    that is, the head seen with no disc of another colour in front of it.
    The projected middle of the shaft must fall on the arrow's own pixels (this checks project())."""
    panel = panel.astype(np.int16)
    no_discs = clean_border(qlib.render([item for item in number_items(g) if item["kind"] != "discs"], camera, PANEL)).astype(np.int16)
    behind_discs = heads_behind_discs(g, camera)
    rows, cols = np.mgrid[:PANEL[1], :PANEL[0]]
    out = {}
    for k in np.flatnonzero(g.touching):
        name = qlib.PART_NAMES[k]
        alone = clean_border(qlib.render([arrow_item(g, k)], camera, PANEL)).astype(np.int16)
        without = clean_border(qlib.render(number_items(g, without_arrow=k), camera, PANEL)).astype(np.int16)
        covered = (255 - alone).max(-1) > VISIBLE_STEP
        shows = covered & (np.abs(panel - without).max(-1) > VISIBLE_STEP)
        clear = shows & (np.abs(panel - no_discs).max(-1) <= VISIBLE_STEP)
        base, mid, tip = (np.array([float(v[0]) for v in to_pixels(*project(g.centre[k] + s * ARROW_LENGTH * g.normal[k], camera)[:2])])
                          for s in (0.0, 0.35, 1.0))
        col, row = int(round(mid[0])), int(round(mid[1]))
        if not covered[max(row - 2, 0):row + 3, max(col - 2, 0):col + 3].any():
            raise RuntimeError(f"{g.selection.dataset}: the projected shaft of the {name} arrow at pixel ({col}, {row}) is not "
                               "on the rendered arrow; project() and the renderer disagree")
        along = ((cols - base[0]) * (tip[0] - base[0]) + (rows - base[1]) * (tip[1] - base[1])) / max(float(((tip - base) ** 2).sum()), 1e-9)
        head = covered & (along >= 1 - HEAD_SHARE)
        share = float(shows.sum() / covered.sum())
        if share < ARROW_MIN_SHARE:
            raise RuntimeError(f"{g.selection.dataset}: only {share:.0%} of the {name} arrow shows in panel 3; the column label "
                               "promises an arrow for each touching part")
        cosine = sight_cosine(g, camera, k)
        out[name] = {"arrow_pixels_alone": int(covered.sum()), "arrow_pixels_showing": int(shows.sum()), "share_showing": share,
                     "head_pixels": int(head.sum()), "head_pixels_clear": int((head & clear).sum()),
                     "head_share_clear": float((head & clear).sum() / max(int(head.sum()), 1)),
                     "head_behind_discs": behind_discs[name],
                     "cosine_normal_to_line_of_sight": cosine,
                     "share_of_length_in_projection": math.sqrt(max(0.0, 1 - cosine ** 2)),
                     "length_on_screen_px": float(np.linalg.norm(tip - base)),
                     "points": "towards the viewer" if cosine >= 0 else "into the picture"}
    return out


def sight_cosine(g: Grasp, camera: qlib.Camera, k: int) -> float:
    """Cosine between part k's contact normal and the line of sight from its disc centre to the camera (the camera
    rule uses the cosine to the view axis instead; with a perspective camera the two differ a little)."""
    to_camera = np.asarray(camera.position) - g.centre[k]
    return float(g.normal[k] @ to_camera / np.linalg.norm(to_camera))


def measure_visibility(g: Grasp, camera: qlib.Camera, panel_3: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Everything the caption says about what can and cannot be seen, and the (V,) mask of visible touching vertices."""
    seen = visible_touching(g, camera)
    touching_vertices = {}
    for k, name in enumerate(qlib.PART_NAMES):
        mine = (g.part == k) & (g.distance < HARD)
        if mine.any():
            touching_vertices[name] = {"within_1cm": int(mine.sum()), "visible": int((mine & seen).sum()),
                                       "share_visible": float((mine & seen).sum() / mine.sum())}
    return {"panel_2_touching_vertices": touching_vertices, "panel_3_discs": disc_overlap(g, camera),
            "panel_3_arrows": arrow_visibility(g, camera, panel_3),
            "hand_part_to_object_surface_mm": {qlib.PART_NAMES[k]: float(g.surface_gap[k] * 1e3) for k in np.flatnonzero(g.touching)}}, seen


# ------------------------------------------------------------------------------ the strip under a row
def load_font(px: int) -> ImageFont.FreeTypeFont:
    for path in qlib.FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    import matplotlib                                  # ships the same DejaVu Sans
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), px)


@dataclass(frozen=True)
class StripLayout:
    font: ImageFont.FreeTypeFont
    cap: int        # height of a capital letter
    line: int       # height of a text line
    pitch: int      # distance between the tops of two bars

    @classmethod
    def for_font(cls, px: int) -> "StripLayout":
        font = load_font(px)
        box = font.getbbox("H")
        cap = box[3] - box[1]
        return cls(font, cap, math.ceil(px * 1.32), math.ceil(px * 1.15))

    @property
    def top(self) -> int:           # y of the first bar: margin, title line, marker band
        return self.cap // 2 + self.line + self.cap + self.cap // 3

    @property
    def height(self) -> int:        # bars, tick labels, margin
        return self.top + len(qlib.PART_NAMES) * self.pitch + self.line + self.cap // 2


def draw_strip(width: int, touching: np.ndarray, frame: int, lay: StripLayout) -> Image.Image:
    """Six bars (one per hand part) over the 64 frames: part colour where the part touches, pale grey where it does
    not; the drawn frame is outlined and marked by a triangle."""
    font, cap, pitch = lay.font, lay.cap, lay.pitch
    img = Image.new("RGB", (width, lay.height), "white")
    draw = ImageDraw.Draw(img)
    n_frames = touching.shape[0]
    x0 = math.ceil(max(font.getlength(s) for s in (*qlib.PART_NAMES, "frame"))) + cap
    x1 = width - math.ceil(font.getlength(str(STRIP_TICKS[-1])) / 2) - 2
    edges = np.round(np.linspace(x0, x1, n_frames + 1)).astype(int)
    bar = round(pitch * BAR_SHARE)
    y_title = cap // 2
    if x0 + font.getlength(STRIP_TITLE) > width:
        raise ValueError("the strip title is wider than the strip; shorten it")
    draw.text((x0, y_title), STRIP_TITLE, font=font, fill=qlib.INK)
    half = round(cap * 0.45)                                # half width of the marker triangle
    for k, name in enumerate(qlib.PART_NAMES):
        y = lay.top + k * pitch
        draw.text((x0 - cap // 2, y + bar / 2), name, font=font, fill=qlib.INK, anchor="rm")
        draw.rectangle((x0, y, x1 - 1, y + bar - 1), fill=STRIP_OFF)
        for t in np.flatnonzero(touching[:, k]):
            draw.rectangle((edges[t], y, edges[t + 1] - 1, y + bar - 1), fill=qlib.PART_HEX[k])
    y_end = lay.top + (len(qlib.PART_NAMES) - 1) * pitch + bar
    stroke = max(2, cap // 7)
    draw.rectangle((edges[frame] - stroke, lay.top - stroke, edges[frame + 1] - 1 + stroke, y_end - 1 + stroke),
                   outline=qlib.INK, width=stroke)
    xc = (edges[frame] + edges[frame + 1]) / 2
    draw.polygon([(xc - half, lay.top - stroke - 2 * half - 2), (xc + half, lay.top - stroke - 2 * half - 2),
                  (xc, lay.top - stroke - 2)], fill=qlib.INK)
    y_tick = y_end + stroke + 2
    draw.text((x0 - cap // 2, y_tick), "frame", font=font, fill=qlib.INK, anchor="ra")
    for t in STRIP_TICKS:
        draw.text(((edges[t] + edges[t + 1]) / 2, y_tick), str(t), font=font, fill=qlib.INK, anchor="ma")
    return img


# ------------------------------------------------------------------------------ layout
def row_label(g: Grasp) -> str:
    m = g.example.meta
    return f"{LABEL[m.dataset]}\n{qlib.SIDE[m.hand]} hand on a {m.category}\nframe {g.selection.frame}"


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    """Lines of ``text`` no wider than ``width`` (explicit line breaks are kept); a word that cannot fit raises."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            if font.getlength(word) > width:
                raise ValueError(f"label word {word!r} is wider than its {width:.0f} px slot; shorten the label")
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def compose(grasps: list[Grasp], rendered: list[list[np.ndarray]]) -> tuple[Image.Image, dict[str, Any]]:
    """qlib.grid of the rendered panels, each extended downwards by a slot into which the row's strip is pasted
    (qlib.grid has equal panels only; the strip spans the three panels of its row). qlib.grid would centre a row
    label on the renders plus the strip, so the label column is left empty and the labels are drawn here, in the
    same font, centred on the renders."""
    width, height = PANEL
    probe = qlib.grid([[np.full((height, width, 3), 255, np.uint8)] * len(COLUMN_LABELS)], row_labels=["probe"])
    lay = StripLayout.for_font(probe.info["qlib_layout"]["font_px"])
    slot = np.broadcast_to(np.array(SLOT, np.uint8), (lay.height, width, 3))
    rows = [[np.concatenate([panel, slot], 0) for panel in row] for row in rendered]
    image = qlib.grid(rows, row_labels=[""] * len(grasps), col_labels=list(COLUMN_LABELS),
                      colorbars=[qlib.colorbar("parts", label=LEGEND)])
    if image.info["qlib_layout"]["font_px"] != lay.font.size:
        raise RuntimeError("the label size changed between the probe layout and the figure")
    mask = (np.asarray(image) == np.array(SLOT, np.uint8)).all(-1)
    lines = np.flatnonzero(mask.any(1))
    starts = lines[np.r_[True, np.diff(lines) > 1]]
    if len(starts) != len(grasps) or len(lines) != len(grasps) * lay.height:
        raise RuntimeError("could not find the strip slots in the layout")
    draw = ImageDraw.Draw(image)
    boxes, label_boxes = [], []
    for g, y in zip(grasps, starts):
        columns = np.flatnonzero(mask[y])
        x, x_end = int(columns[0]), int(columns[-1]) + 1
        image.paste(draw_strip(x_end - x, g.strip, g.selection.frame, lay), (x, int(y)))
        boxes.append([x, int(y), x_end, int(y) + lay.height])
        margin = image.width - x_end                        # the layout's outer margin; the label keeps half of it clear of the panels
        text = wrap(row_label(g), lay.font, x - margin - margin // 2)
        top = int(y) - height + (height - len(text) * lay.line) // 2
        if top < int(y) - height:
            raise ValueError("the row label is taller than the renders; shorten it")
        for i, s in enumerate(text):
            draw.text((margin, top + i * lay.line), s, font=lay.font, fill=qlib.INK)
        label_boxes.append([margin, top, x - margin // 2, top + len(text) * lay.line])
    if (np.asarray(image) == np.array(SLOT, np.uint8)).all(-1).any():
        raise RuntimeError("a strip did not cover its slot")
    return image, {"strip_boxes_px": boxes, "strip_font_px": lay.font.size, "strip_height_px": lay.height,
                   "row_label_boxes_px": label_boxes,
                   "row_labels": "drawn by this script, centred on the three renders of the row (not on renders plus strip)"}


# ------------------------------------------------------------------------------ meta
def number(label: str, value: Any, dataset: str, source: Path | str, locator: str) -> dict[str, Any]:
    return {"label": label, "value": value, "dataset": dataset, "source": str(source), "locator": locator}


def sig(value: float, digits: int = 2) -> float:
    """``value`` to ``digits`` significant digits (rounding to fixed decimals would store 6e-08 as 0.0)."""
    return float(f"{float(value):.{digits}g}")


def english_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def name_parts(parts: list[str]) -> str:
    """["palm", "index", "little"] -> "the palm and the index and little fingers"."""
    fingers = [p for p in parts if p not in ("palm", "thumb")]
    phrases = [f"the {p}" for p in parts if p in ("palm", "thumb")]
    if fingers:
        phrases.append(f"the {english_list(fingers)} finger" + ("s" if len(fingers) > 1 else ""))
    return english_list(phrases)


def build_caption(grasps: list[Grasp], vis: list[dict[str, Any]], edge_on: list[str]) -> str:
    """The suggested caption. Every clause about what can or cannot be seen is written from the measured values
    (``vis``, one record per row from measure_visibility) and is left out where the measurement does not support it."""
    by_ds = {g.selection.dataset: g for g in grasps}
    taco, arctic = by_ds["taco"], by_ds["arctic"]
    ev = arctic.selection.evidence
    n_taco, n_arctic = int(taco.touching.sum()), int(arctic.touching.sum())
    far_side, coarse, away, dim, hover, overlap = [], [], [], [], [], False
    for g, v in zip(grasps, vis):
        thing = g.example.meta.category
        touching = [qlib.PART_NAMES[k] for k in np.flatnonzero(g.touching)]
        seen = {p: v["panel_2_touching_vertices"][p] for p in touching}
        least = min(seen, key=lambda p: seen[p]["share_visible"])
        if seen[least]["share_visible"] < FAR_SIDE:
            far_side.append(f"contact on the far side of the {thing} is hidden in this panel, most of all {name_parts([least])}'s "
                            f"({seen[least]['visible']} of its {seen[least]['within_1cm']} touching vertices are visible)")
        if g.edge_near > HARD:
            coarse.append(f"the {thing} mesh is coarse (triangle edges of about {g.edge_near * 1e3:.0f} mm near the hand), so its "
                          "coloured patches are blurred")
        behind = [p for p in touching if v["panel_3_arrows"][p]["points"] == "into the picture"]
        if behind:
            away.append(f"on the {thing} {'those' if len(behind) > 1 else 'that'} of {name_parts(behind)}")
        for p in touching:                                  # an arrowhead that a disc of another part dims
            arrow = v["panel_3_arrows"][p]
            if arrow["head_share_clear"] < HEAD_DIM:
                front = [d for d, share in arrow["head_behind_discs"].items() if d != p and share >= 0.5]
                dim.append(f"on the {thing} the head of {name_parts([p])}'s arrow shows only dimly"
                           + (f" behind {name_parts(front)}'s disc" if len(front) == 1 else
                              f" behind the discs of {name_parts(front)}" if front else " behind its own disc"))
        overlap = overlap or any(v["panel_3_discs"][p]["overlaps"] for p in touching)
        farthest = max(touching, key=lambda p: v["hand_part_to_object_surface_mm"][p])
        if v["hand_part_to_object_surface_mm"][farthest] > HOVER * 1e3:
            hover.append(f"on the {thing} {name_parts([farthest])} counts although it stays "
                         f"{v['hand_part_to_object_surface_mm'][farthest]:.0f} mm from the surface")
    reasons = (["neighbouring ones overlap"] if overlap else []) + (
        [f"some face away from the viewer ({'; '.join(away)}), their arrows then pointing into the picture"] if away else [])
    translucent = ("The discs are translucent because " + " and ".join(reasons) if reasons
                   else "The discs are translucent so that nothing behind a disc is hidden") + "".join("; " + clause for clause in dim)
    return (
        f"One recorded frame from one TACO sequence (a {qlib.SIDE[taco.example.meta.hand]} hand on a {taco.example.meta.category}; "
        "the sequence already shown on this page, drawn at its frame of largest contact amount) and from one ARCTIC sequence "
        f"(a {qlib.SIDE[arctic.example.meta.hand]} hand on a {arctic.example.meta.category}; each of the {ev['n_test_sequences']} test "
        f"sequences was taken at its own frame of largest contact amount, {ev['n_sequences_with_the_median_value']} of them have the "
        f"median number of touching parts, {n_arctic}, and the first of these in index order is drawn). "
        "Left: the recorded hand on the object, its six parts in six colours (the palm is dark grey). "
        "Middle: the object surface coloured by the nearest hand part, strongly where the hand is within 1 cm and faintly up to 2 cm"
        + "".join("; " + clause for clause in far_side + coarse) + ". "
        "Right: the same frame reduced to its 48 numbers: for each of the six parts whether it touches (at least "
        f"{MIN_VERTS} vertices of the object mesh within 1 cm of it" + (", so " + " and ".join(hover) if hover else "") + "), how "
        "much (disc area; disc areas compare amounts within one row only), where (disc centre, the centroid of its contact) and "
        f"facing which way (arrow, its contact normal); {n_taco} parts touch on the {taco.example.meta.category}, {n_arctic} on the "
        f"{arctic.example.meta.category}. "
        + translucent + ". "
        "The bars show which parts touch in each of the 64 frames of the sequence, the outlined column being the frame drawn; the "
        "discs and arrows are a drawing of recorded numbers, not a model output, and the camera frames the hand, so part of each "
        "object lies outside the picture" + "".join(f", and the {name} is seen edge-on" for name in edge_on) + "."
    )


def numbers_for(g: Grasp, feat: Features, cam_info: dict[str, Any], vis: dict[str, Any]) -> list[dict[str, Any]]:
    sel, name = g.selection, LABEL[g.selection.dataset]
    n, t = sel.example, sel.frame
    cache, at = feat.path, f"[{n}, {t + qlib.PAD}] (stored frame = frame + {qlib.PAD})"
    drawn = "computed in this script from the arrays in panels.npz"
    ev, checks, key = sel.evidence, g.checks, sel.dataset
    out = [
        number("example index", n, name, qlib.HIER[sel.dataset] / "sequences_meta.csv", f"row {n}"),
        number("frame drawn (0-based, of 64)", t, name, cache, f"argmax over frames 0..63 of sum_k m[{n}, frame + {qlib.PAD}, k]"),
        number("frame of the recorded take", g.example.take_frame(t), name, qlib.HIER[sel.dataset] / "sequences_meta.csv",
               f"t0 of row {n} + frame"),
        number("contact amount at the drawn frame (sum of the six parts; share of the object's surface, distance-weighted)",
               round(float(g.amount.sum()), 5), name, cache, f"m{at}.sum()"),
        number("hand parts that touch in the drawn frame", int(g.touching.sum()), name, cache,
               f"n_hard[{n}, {t + qlib.PAD - 2}:{t + qlib.PAD + 1}] >= {MIN_VERTS} in at least 2 of the 3 frames, per part"),
        number("test sequences", ev["n_test_sequences"], name, manifest_path(sel.dataset), "len(test)"),
        number("test sequences with that many touching parts at their own frame of largest contact amount",
               int(ev["histogram_of_touching_parts"][str(int(g.touching.sum()))]), name, cache,
               "meta.selection_evidence: histogram_of_touching_parts"),
        number("median number of touching parts over the test sequences, each at its frame of largest contact amount",
               ev["median"], name, cache, "meta.selection_evidence: median"),
        number("object surface area of the drawn mesh, m^2", round(g.area, 5), name, drawn,
               f"sum of triangle areas of {key}_1_hand_on_object/object_verts, object_faces"),
        number("object vertices", int(len(g.verts)), name, drawn, f"{key}_1_hand_on_object/object_verts"),
        number("object vertices within 1 cm of the hand (touching)", int((g.distance < HARD).sum()), name, drawn,
               f"{key}_2_contact_by_part/distance < {HARD}"),
        number("object vertices within 2 cm of the hand", int((g.distance < SOFT).sum()), name, drawn,
               f"{key}_2_contact_by_part/distance < {SOFT}"),
        number("share of the object's vertices inside the picture", round(cam_info["object_vertices_in_frame"], 3), name, drawn,
               "projection of object_verts with meta.display_choices.camera.per_row"),
        number("angle between the view direction and the object's broad side, degrees", round(cam_info["angle_to_broad_side_deg"], 1),
               name, drawn, "meta.display_choices.camera.per_row: asin |direction . thinnest principal axis of object_verts|"),
        number("largest distance of a drawn disc centre from the nearest touching vertex of its part, mm",
               round(max(checks["centroid_to_nearest_touching_vertex_of_its_part_mm"].values()), 2), name, drawn,
               "meta.checks.centroid_to_nearest_touching_vertex_of_its_part_mm"),
        number("largest difference between cached and recomputed contact centroid, mm",
               round(checks["centroid_cache_vs_recomputed_max_mm"], 5), name, drawn, "meta.checks.centroid_cache_vs_recomputed_max_mm"),
        number("largest relative difference between cached and recomputed amount",
               sig(checks["amount_recomputed_vs_cache_max_relative_difference"]), name, drawn,
               "meta.checks.amount_recomputed_vs_cache_max_relative_difference"),
        number("disc area per unit of contact amount, cm^2", round(DISC_AREA_SHARE * g.area * 1e4, 1), name, drawn,
               f"{DISC_AREA_SHARE} * surface area of the drawn mesh; pi * {key}_3_numbers/radius^2 / {key}_3_numbers/amount"),
        number("median edge length of the object's triangles with a vertex within 2 cm of the hand, mm",
               round(g.edge_near * 1e3, 1), name, drawn,
               f"{key}_1_hand_on_object/object_verts, object_faces, {key}_2_contact_by_part/distance < {SOFT}"),
    ]
    for k, part in enumerate(qlib.PART_NAMES):
        out.append(number(f"{part}: frames (of 64) in which it touches", int(g.strip[:, k].sum()), name, cache,
                          f"causal 3-frame majority of n_hard[{n}, :, {k}] >= {MIN_VERTS}, frames 0..63"))
        out.append(number(f"{part}: contact amount at the drawn frame", round(float(g.amount[k]), 5), name, cache,
                          f"m[{n}, {t + qlib.PAD}, {k}]"))
        out.append(number(f"{part}: object vertices within 1 cm at the drawn frame", int(g.n_hard[k]), name, cache,
                          f"n_hard[{n}, {t + qlib.PAD}, {k}]"))
        if part in vis["panel_2_touching_vertices"]:
            out.append(number(f"{part}: of these, vertices the camera sees in panel 2 (inside the picture, not hidden by the object)",
                              vis["panel_2_touching_vertices"][part]["visible"], name, drawn,
                              f"ray from the camera to each vertex; {key}_2_contact_by_part/visible & (part == {k})"))
        if g.touching[k]:
            out.append(number(f"{part}: disc radius, mm", round(float(g.radius[k]) * 1e3, 1), name, drawn,
                              f"sqrt({DISC_AREA_SHARE} * amount * surface area / pi), {key}_3_numbers/radius[{k}]"))
            out.append(number(f"{part}: smallest distance from the hand part to the object surface, mm",
                              round(float(g.surface_gap[k]) * 1e3, 1), name, drawn,
                              f"closest point on the object mesh of every hand vertex of the part; {key}_1_hand_on_object/"
                              f"surface_gap[{k}]"))
            out.append(number(f"{part}: share of its arrow's pixels that show in panel 3",
                              round(vis["panel_3_arrows"][part]["share_showing"], 3), name, drawn,
                              f"meta.visibility.{name}.panel_3_arrows.{part} ({key}_3_numbers/arrow_pixels_showing[{k}] / "
                              f"arrow_pixels_alone[{k}])"))
            out.append(number(f"{part}: share of its arrowhead's pixels that show in panel 3 with no disc in front",
                              round(vis["panel_3_arrows"][part]["head_share_clear"], 3), name, drawn,
                              f"meta.visibility.{name}.panel_3_arrows.{part} ({key}_3_numbers/head_pixels_clear[{k}] / "
                              f"head_pixels[{k}])"))
            out.append(number(f"{part}: share of its arrow's length that the picture shows (sine of the angle to the line of sight)",
                              round(vis["panel_3_arrows"][part]["share_of_length_in_projection"], 3), name, drawn,
                              f"meta.visibility.{name}.panel_3_arrows.{part}; {key}_3_numbers/centre[{k}], normal[{k}], "
                              f"{key}_camera/position"))
            out.append(number(f"{part}: share of its disc that other discs overlap in panel 3",
                              round(vis["panel_3_discs"][part]["share_overlapped_by_other_discs"], 3), name, drawn,
                              f"meta.visibility.{name}.panel_3_discs.{part} (projected disc outlines)"))
    return out


def build_meta(grasps: list[Grasp], feats: dict[str, Features], cams: list[dict[str, Any]], vis: list[dict[str, Any]],
               layout: dict[str, Any]) -> dict[str, Any]:
    definitions = json.loads(DEFINITIONS.read_text())
    if definitions["dims"]["R2"] != 48 or [definitions["blocks"][b] for b in definitions["ladder"]["R2"]] != [6, 6, 36]:
        raise RuntimeError(f"{DEFINITIONS} no longer describes 48 numbers as 6 + 6 + 36")
    by_ds = {g.selection.dataset: g for g in grasps}
    taco, arctic = by_ds["taco"], by_ds["arctic"]
    n_taco, n_arctic = int(taco.touching.sum()), int(arctic.touching.sum())
    numbers = [
        number("numbers per frame in the structural description", definitions["dims"]["R2"], "both", DEFINITIONS, "dims.R2"),
        number("of these: on/off per part, amount per part, centroid (3) and normal (3) per part",
               [definitions["blocks"]["part"], definitions["blocks"]["amount"], definitions["blocks"]["geom"]], "both", DEFINITIONS,
               "blocks.part, blocks.amount, blocks.geom"),
        number("hand parts (palm and five fingers)", len(qlib.PART_NAMES), "both", DEFINITIONS, "blocks.part"),
        number("frames per sequence", qlib.T, "both", feature_path("taco"), "n_hard.shape[1] - 2 * 8 padding frames"),
        number("distance within which a vertex counts as touching, cm", round(HARD * 100, 2), "both", DEFINITIONS,
               "participation, geometry (rule re-implemented here; reproduces the cached counts exactly)"),
        number("distance up to which a vertex adds to the contact amount, cm", round(SOFT * 100, 2), "both", DEFINITIONS, "amount"),
        number("smallest number of touching vertices for a part to count as touching", MIN_VERTS, "both", DEFINITIONS, "participation"),
        number("share of (amount x object surface area) drawn as disc area", DISC_AREA_SHARE, "both", "this script", "DISC_AREA_SHARE"),
        number("disc area for the same contact amount: ARCTIC row relative to TACO row", round(arctic.area / taco.area, 1), "both",
               "computed in this script from the arrays in panels.npz", "ratio of the surface areas of the two drawn meshes"),
        number("disc opacity in panel 3", DISC_OPACITY, "both", "this script", "DISC_OPACITY"),
        number("arrow length, cm", round(ARROW_LENGTH * 100, 2), "both", "this script", "ARROW_LENGTH"),
    ]
    for g, cam, v in zip(grasps, cams, vis):
        numbers += numbers_for(g, feats[g.selection.dataset], cam, v)

    def example(g: Grasp) -> dict[str, Any]:
        m = g.example.meta
        return {"dataset": LABEL[m.dataset], "example": m.example, "sequence_id": m.sequence_id, "object": m.category,
                "mesh_id": m.mesh_id, "hand": qlib.SIDE[m.hand], "role": m.role, "split": m.split, "frames_drawn": [g.selection.frame],
                "take_frames_drawn": [g.example.take_frame(g.selection.frame)], "strip_frames": [0, qlib.T - 1],
                "touching_parts_at_the_drawn_frame": [qlib.PART_NAMES[k] for k in np.flatnonzero(g.touching)]}

    fallback = {LABEL[g.selection.dataset]: g.checks["parts_whose_normal_is_the_nearest_vertex_normal"] for g in grasps}
    no_disc = {LABEL[g.selection.dataset]: {qlib.PART_NAMES[k]: round(float(g.amount[k]), 5) for k in range(len(qlib.PART_NAMES))
                                             if not g.touching[k] and g.amount[k] > 0} for g in grasps}
    rows = [LABEL[g.selection.dataset] for g in grasps]
    back_facing = {row: [p for p, a in v["panel_3_arrows"].items() if a["points"] == "into the picture"] for row, v in zip(rows, vis)}
    arrows_showing = {row: {p: f"{a['share_showing']:.0%}" for p, a in v["panel_3_arrows"].items()} for row, v in zip(rows, vis)}
    arrows_length = {row: {p: f"{a['share_of_length_in_projection']:.0%}" for p, a in v["panel_3_arrows"].items()
                           if a["points"] == "into the picture"} for row, v in zip(rows, vis)}
    heads_clear = {row: {p: f"{a['head_share_clear']:.0%}" for p, a in v["panel_3_arrows"].items()} for row, v in zip(rows, vis)}
    heads_behind = {row: {p: {d: f"{share:.0%}" for d, share in a["head_behind_discs"].items()}
                          for p, a in v["panel_3_arrows"].items() if a["head_behind_discs"]} for row, v in zip(rows, vis)}
    stricter = []           # per row with a dim arrowhead: what leaving out the directions with a hidden arrowhead would cost
    for g, cam, v in zip(grasps, cams, vis):
        other = cam["if_no_arrowhead_might_lie_behind_a_disc"]
        if "direction" in other and any(a["head_share_clear"] < HEAD_DIM for a in v["panel_3_arrows"].values()):
            largest = qlib.PART_NAMES[int(np.argmax(g.radius))]
            would, now = other["cosine_to_contact_normals"][largest], cam["cosine_to_contact_normals"][largest]
            stricter.append(
                f" {LABEL[g.selection.dataset]}: {other['of_them_with_no_arrowhead_behind_a_disc']} of the "
                f"{other['allowed_directions']} allowed view directions have no arrowhead behind a disc by that geometry test; "
                f"the best of them by the camera rule's score ({other['score']:.2f} against {cam['score']:.2f}) would see the "
                f"largest disc ({largest}) at cosine {would:.2f} instead of {now:.2f}"
                + (", that is, closer to edge-on" if abs(would) < abs(now) else "")
                + ". The camera rule was kept and the caption names the arrowhead.")
    discs_overlapped = {row: {p: f"{d['share_overlapped_by_other_discs']:.0%}" for p, d in v["panel_3_discs"].items()}
                        for row, v in zip(rows, vis)}
    discs_behind = {row: {p: f"{d['share_behind_a_nearer_disc']:.0%} behind {english_list(d['behind'])}"
                          for p, d in v["panel_3_discs"].items() if d["behind"]} for row, v in zip(rows, vis)}
    vertices_seen = {row: {p: f"{s['visible']} of {s['within_1cm']}" for p, s in v["panel_2_touching_vertices"].items()}
                     for row, v in zip(rows, vis)}
    surface_gaps = {row: {p: round(d, 1) for p, d in v["hand_part_to_object_surface_mm"].items()} for row, v in zip(rows, vis)}
    per_amount = {LABEL[g.selection.dataset]: DISC_AREA_SHARE * g.area * 1e4 for g in grasps}      # cm^2 of disc per unit amount
    centre_gap = ", ".join(f"{LABEL[g.selection.dataset]} {max(g.checks['centroid_to_nearest_touching_vertex_of_its_part_mm'].values()):.1f} mm"
                           for g in grasps)
    in_frame = ", ".join(f"{LABEL[g.selection.dataset]} {cam['object_vertices_in_frame']:.0%}" for g, cam in zip(grasps, cams))
    edge_on = [g.example.meta.category for g, cam in zip(grasps, cams)
               if cam["angle_to_broad_side_deg"] < EDGE_ON_DEG and cam["thinnest_to_middle_extent"] < FLAT]
    caption = build_caption(grasps, vis, edge_on)
    return {
        "id": FIG_ID, "block": BLOCK, "title": TITLE,
        "panels": {
            "rows": {LABEL[g.selection.dataset]: f"one recorded frame of one sequence ({row_label(g).splitlines()[1]}, frame "
                                                  f"{g.selection.frame} of 0..63); the three panels share one camera" for g in grasps},
            "column_1": "The recorded hand mesh on the object mesh. The hand is tinted by hand part (palm, thumb, index, middle, "
                        "ring, little).",
            "column_2": "The object alone. Each vertex within 1 cm of the hand has the colour of the hand part nearest to it; vertices "
                        "between 1 and 2 cm have a pale version of that colour; all others are neutral grey.",
            "column_3": "The object as a translucent ghost and nothing but the frame's 48 numbers: for every touching part a flat "
                        "translucent disc centred at the centroid of its touching vertices, facing along its contact normal, with area "
                        "proportional to its contact amount (within the row), and an opaque arrow of fixed length along that normal. "
                        "Parts that do not touch have no disc.",
            "strip_under_each_row": "Six bars, one per hand part, over the 64 frames of the sequence: coloured where the part touches, "
                                    "pale grey where it does not. The outlined column with the triangle is the frame drawn above.",
        },
        "examples": [example(g) for g in grasps],
        "selection_rule": " ".join(g.selection.rule for g in grasps),
        "selection_evidence": {LABEL[g.selection.dataset]: g.selection.evidence for g in grasps},
        "display_choices": {
            "canonical_to_surface_rule": "not used: no 512-point map is drawn. Panel 2 is coloured per mesh vertex from the recorded "
                                         "hand-to-vertex distances (qlib Example.dense, inverted to metres).",
            "contact_by_part": "part of the hand vertex nearest to the object vertex (the study's rule); full part colour below "
                               f"{HARD * 100:g} cm, {PALE:g} of the part colour mixed into the neutral grey between {HARD * 100:g} and "
                               f"{SOFT * 100:g} cm, neutral grey above",
            "colour_scales": {"hand parts": dict(zip(qlib.PART_NAMES, qlib.PART_HEX)), "no contact": qlib.ZERO,
                              "numeric_range": "categorical; the two tint levels are distance bands (0 to 1 cm, 1 to 2 cm), there is "
                                               "no continuous scale",
                              "strip": {"touching": "part colour", "not touching": STRIP_OFF}},
            "discs": "centre = cached centroid * length + mesh centroid; normal = cached normal; disc area = a quarter of the part's "
                     f"distance-weighted contact area ({DISC_AREA_SHARE:g} x amount x that object's surface area; radius = sqrt of "
                     f"that over pi), {qlib.DISC_THICKNESS * 1e3:g} mm thick. The amount is a share of the object's surface, so for "
                     f"the same amount the disc is {arctic.area / taco.area:.1f} times larger on the {arctic.example.meta.category} "
                     f"than on the {taco.example.meta.category} (disc area per unit amount: "
                     + ", ".join(f"{row} {value:.1f} cm^2" for row, value in per_amount.items())
                     + f"); disc areas compare amounts within one row only. Opacity {DISC_OPACITY:g} (front faces only), so that a "
                       "disc or an arrow behind a disc stays visible; arrows are opaque",
            "arrows": "from the disc centre along the cached unit normal (pointing from the surface to the hand), "
                      f"{ARROW_LENGTH * 100:g} cm long",
            "object_in_panel_3": f"ghost ({qlib.GHOST['color']}, opacity {qlib.GHOST['opacity']}), because a centroid of touching "
                                 "vertices can lie inside the object",
            "hand_style": f"panel 1 only: recorded MANO mesh, each vertex {HAND_TINT:g} part colour + {1 - HAND_TINT:g} hand colour "
                          f"{qlib.HAND_SKIN}; no hand in panels 2 and 3",
            "camera": {"rule": " ".join(glyph_view.__doc__.split()).replace("N_DIRECTIONS", str(N_DIRECTIONS)),
                       "framing": f"perspective, {qlib.VIEW_ANGLE:g} degrees; fitted to the hand, the object vertices within "
                                  f"{NEAR * 100:g} cm of the hand and all discs and arrows (margin {CAMERA_MARGIN:g}); the rest of the "
                                  "object may leave the picture",
                       "per_row": {LABEL[g.selection.dataset]: cam for g, cam in zip(grasps, cams)}},
            "panel_px": list(PANEL), "layout": layout,
            "render_border": " ".join(clean_border.__doc__.split()),
            "strip": "drawn by this script (PIL) in the label font and size of qlib.grid and pasted into a slot reserved under each row",
        },
        "numbers": numbers,
        "checks": {LABEL[g.selection.dataset]: g.checks for g in grasps},
        "visibility": {
            "how": {
                "panel_2_touching_vertices": "per part, the object vertices within 1 cm of the hand and how many of them the camera "
                                             "sees: inside the picture and, on the ray from the camera, nothing of the object hit more "
                                             f"than {RAY_TOLERANCE * 1e3:g} mm before the vertex (trimesh ray casting)",
                "panel_3_discs": "from the projected outline of each disc (96-gon in the disc's middle plane, the camera of the row; "
                                 f"the {qlib.DISC_THICKNESS * 1e3:g} mm thickness of the drawn disc is left out): pixels inside it, the "
                                 "share that the outlines of other discs overlap, the share behind a disc whose centre is nearer to "
                                 "the camera",
                "panel_3_arrows": "arrow_pixels_alone: pixels the arrow covers when rendered alone; arrow_pixels_showing: those of "
                                  f"them where the third panel differs by more than {VISIBLE_STEP}/255 in some channel from the same "
                                  f"panel rendered without the arrow. The script stops below a share of {ARROW_MIN_SHARE:g}. "
                                  f"head_pixels: the arrow's pixels over the last {HEAD_SHARE:g} of its projected length; "
                                  "head_pixels_clear: those of them that show and whose colour stays within "
                                  f"{VISIBLE_STEP}/255 when all discs are left out (no disc of another colour in front); the caption "
                                  f"names an arrow whose head_share_clear is below {HEAD_DIM:g}. head_behind_discs: from the geometry "
                                  "alone, the share of 9 points along the head's axis that lie behind each disc as the camera sees "
                                  "it. "
                                  "cosine_normal_to_line_of_sight: between the contact normal and the line from the disc centre to "
                                  "the camera (negative: the arrow points into the picture); share_of_length_in_projection: the "
                                  "sine of that angle, the share of the arrow's length that the picture shows",
                "hand_part_to_object_surface_mm": "smallest distance from the hand vertices of the part to the object mesh surface "
                                                  "(closest point on a triangle, not on a vertex)",
            },
            **{row: v for row, v in zip(rows, vis)},
        },
        "inference": "none (data only): no checkpoint was loaded and no model was run; every value is read from the feature cache "
                     "or recomputed from recorded geometry on CPU",
        "caveats": [
            "The third panel is a drawing of 48 numbers, not a surface and not a model output. A contact map decoded from these numbers "
            "is not shown because the study's decoder was never saved.",
            f"Disc area = a quarter of the part's distance-weighted contact area ({DISC_AREA_SHARE:g} x amount x that object's "
            "surface area); the quarter is a display choice. The stored amount is a share of the object's surface, so disc areas "
            f"compare amounts within one row only: for the same amount the disc is {arctic.area / taco.area:.1f} times larger on the "
            f"{arctic.example.meta.category} than on the {taco.example.meta.category}. Arrow length is fixed and carries no value.",
            "A disc centre is the mean position of the part's touching vertices. It need not lie on the surface (a finger wrapped around a "
            "handle gives a centre inside the handle), which is why the object is translucent in the third panel. Largest distance to the "
            f"nearest touching vertex of the part: {centre_gap}.",
            "The arrow is the mean of the contact normals of the part, except where these nearly cancel (mean shorter than "
            f"{CANCEL:g}): there the study stores the normal of the touching vertex nearest the centroid. Affected here: "
            f"{json.dumps(fallback)}.",
            f"Discs whose normal points away from the camera are seen from behind through the ghost object: {json.dumps(back_facing)}. "
            "Where the parts wrap around the object no camera avoids this; over the allowed view directions the largest attainable "
            "value of the smallest cosine between view direction and contact normal is "
            + ", ".join(f"{row} {cam['largest_smallest_cosine_of_any_allowed_direction']:.2f}" for row, cam in zip(rows, cams))
            + " (negative: some disc is always seen from behind). "
            "Their arrows point into the picture, are seen through the translucent disc and are foreshortened; share of the arrow's "
            f"length that the picture shows: {json.dumps(arrows_length)}. Share of each arrow's pixels that show in the third panel "
            f"(the colour changes by more than {VISIBLE_STEP}/255 when the arrow is left out): {json.dumps(arrows_showing)}.",
            "An arrowhead behind a disc of another part shows only as a dull mixed colour, and the arrow's bright base at the disc "
            "centre can then be mistaken for its head. Share of each arrowhead's pixels that show with no disc in front (the colour "
            f"stays within {VISIBLE_STEP}/255 when all discs are left out): {json.dumps(heads_clear)}; arrowheads that lie behind a "
            f"disc by the geometry, with the share of the head: {json.dumps(heads_behind)}." + "".join(stricter),
            f"Neighbouring discs overlap, which is why they are drawn at opacity {DISC_OPACITY:g}; the overlapped part of a disc shows "
            f"a mixed colour. Share of each disc's outline that other discs overlap: {json.dumps(discs_overlapped)}; of these, behind "
            f"a disc whose centre is nearer to the camera: {json.dumps(discs_behind)}.",
            "Contact on faces turned away from the camera is not visible in the second panel. Touching vertices (within 1 cm) that "
            f"the camera sees, per part: {json.dumps(vertices_seen)}.",
            f"A part counts as touching when at least {MIN_VERTS} vertices of the object mesh are within 1 cm of the hand and nearest "
            "to that part (3-frame majority); this does not require the part to rest on the surface. Smallest distance from each "
            f"touching hand part to the object surface, mm: {json.dumps(surface_gaps)}.",
            f"The dark grey of the palm ({qlib.PART_HEX[0]}, the palette shared by all qualitative figures) is close to a shadow on "
            "the light grey object, so palm contact in the second panel can be mistaken for shading.",
            f"A part that does not count as touching has no disc although its amount is one of the 48 numbers: {json.dumps(no_disc)}.",
            "A part counts as touching by a 3-frame majority (frames t-2, t-1, t), so the bars can lag the surface colouring by one frame.",
            f"The camera frames the hand and the object within {NEAR * 100:g} cm of it. Share of the object's vertices inside the "
            f"picture: {in_frame}."
            + "".join(f" The {name} is seen edge-on (the view direction is within {EDGE_ON_DEG:g} degrees of its broad side), so a flat "
                      "part of it appears as a thin wedge." for name in edge_on),
            "TACO: the sequence was not chosen for being typical; it is the page's example (docs/evidence/f1b_hier_contact_gen.json "
            "describes it as the test sequence with the largest change between first and last frame; not recomputed here) and the "
            f"frame is the extreme of that sequence. {n_taco} parts touch there, as in "
            f"{taco.selection.evidence['test_sequences_with_the_same_number_of_touching_parts']} of "
            f"{taco.selection.evidence['n_test_sequences']} test sequences at their own frame of largest contact amount.",
            f"ARCTIC: typical only in the number of touching parts ({n_arctic}, the median); it is the smallest index among "
            f"{arctic.selection.evidence['n_sequences_with_the_median_value']} test sequences with that count. The mesh is coarse for "
            f"the size of the object ({len(arctic.verts)} vertices; median triangle edge near the hand {arctic.edge_near * 1e3:.1f} mm, "
            f"against {taco.edge_near * 1e3:.1f} mm on the {taco.example.meta.category}), and colours are interpolated across each "
            "triangle, so the coloured patch is blurred and looks larger than the 1 cm and 2 cm bands: "
            f"{int((arctic.distance < HARD).sum())} vertices are within 1 cm of the hand, {int((arctic.distance < SOFT).sum())} "
            "within 2 cm.",
            "ARCTIC: the amounts recomputed on the drawn mesh are "
            f"{arctic.checks['amount_recomputed_vs_cache_max_relative_difference']:.1%} above the cached ones at most, because the "
            "drawn mesh lacks the template's triangles that join the two parts (area ratio "
            f"{arctic.checks['area_ratio_drawn_to_study']:.4f}); after that correction they agree to "
            f"{arctic.checks['amount_after_area_correction_max_relative_difference']:.0e}.",
            "The hand is tinted by part in the first panel only to show which colour is which part; it is the recorded hand mesh.",
        ],
        "suggested_caption": caption,
        "suggested_alt": "Two rows of 3-D renders, " + " and ".join(
            f"a {qlib.SIDE[g.example.meta.hand]} hand on a {g.example.meta.category} ({LABEL[g.selection.dataset]})" for g in grasps)
            + ", each shown as the hand on the object, as contact on the surface coloured by hand part, and as one disc and arrow per "
              "touching hand part, with bars underneath showing which parts touch over 64 frames.",
        "study_rules_source": RULES_SOURCE,
        "script": str(Path(__file__).resolve()),
        "command": qlib.RUN_COMMAND.replace("<script>", "docs/build/qual/" + Path(__file__).name),
    }


# ------------------------------------------------------------------------------ main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    qlib.require_headless()
    feats = {ds: load_features(ds) for ds in DATASETS}
    selections = [select_taco(feats["taco"]), select_arctic(feats["arctic"])]
    grasps, rendered, cams, vis, panels = [], [], [], [], {}
    for sel in selections:
        log.info("%s: example %d, frame %d (%d touching parts)", LABEL[sel.dataset], sel.example, sel.frame,
                 sel.evidence["touching_parts_at_frame"])
        g = load_grasp(sel, feats[sel.dataset])
        direction, up, cam_info = glyph_view(g)
        camera = qlib.fit_camera(framed_points(g), direction, up, margin=CAMERA_MARGIN, aspect=PANEL[0] / PANEL[1])
        cam_info.update(position=list(camera.position), focal_point=list(camera.focal_point), view_angle=camera.view_angle,
                        object_vertices_in_frame=share_in_frame(g.verts, camera),
                        hand_vertices_in_frame=share_in_frame(g.hand_verts, camera),
                        if_no_arrowhead_might_lie_behind_a_disc=clear_head_alternative(g))
        images = render_row(g, camera)
        seen_by, seen = measure_visibility(g, camera, images[2])
        for part, arrow in seen_by["panel_3_arrows"].items():
            log.info("%s: %s arrow shows %.0f%%, its head with no disc in front %.0f%% (%s)", LABEL[sel.dataset], part,
                     100 * arrow["share_showing"], 100 * arrow["head_share_clear"], arrow["points"])
        grasps.append(g), rendered.append(images), cams.append(cam_info), vis.append(seen_by)
        key = sel.dataset

        def per_part(record: dict[str, dict[str, Any]], field: str) -> np.ndarray:
            return np.array([record.get(name, {}).get(field, 0) for name in qlib.PART_NAMES])

        panels[f"{key}_1_hand_on_object"] = {"image": images[0], "object_verts": g.verts, "object_faces": g.faces,
                                             "hand_verts": g.hand_verts, "hand_faces": g.hand_faces,
                                             "hand_part": qlib.mano_parts(), "hand_rgb": hand_colours(),
                                             "surface_gap": g.surface_gap}
        panels[f"{key}_2_contact_by_part"] = {"image": images[1], "distance": g.distance, "part": g.part,
                                              "rgb": surface_colours(g), "visible": seen}
        panels[f"{key}_3_numbers"] = {"image": images[2], "touching": g.touching, "amount": g.amount, "centre": g.centre,
                                      "normal": g.normal, "radius": g.radius, "n_hard": g.n_hard,
                                      "disc_opacity": np.asarray(DISC_OPACITY),
                                      "arrow_pixels_alone": per_part(seen_by["panel_3_arrows"], "arrow_pixels_alone"),
                                      "arrow_pixels_showing": per_part(seen_by["panel_3_arrows"], "arrow_pixels_showing"),
                                      "head_pixels": per_part(seen_by["panel_3_arrows"], "head_pixels"),
                                      "head_pixels_clear": per_part(seen_by["panel_3_arrows"], "head_pixels_clear"),
                                      "disc_pixels": per_part(seen_by["panel_3_discs"], "disc_pixels")}
        panels[f"{key}_strip"] = {"touching": g.strip, "frame": np.asarray(sel.frame),
                                  "contact_amount": feats[key].total[sel.example]}
        panels[f"{key}_camera"] = {"position": np.asarray(camera.position), "focal_point": np.asarray(camera.focal_point),
                                   "up": np.asarray(camera.up), "view_angle": np.asarray(camera.view_angle),
                                   "aspect": np.asarray(camera.aspect)}
    image, layout = compose(grasps, rendered)
    paths = qlib.save(FIG_ID, image, panels, build_meta(grasps, feats, cams, vis, layout))
    for name, path in paths.items():
        log.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
