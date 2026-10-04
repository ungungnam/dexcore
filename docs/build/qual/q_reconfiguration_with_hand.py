"""Figure ``reconfiguration_with_hand`` (page block F1-D): the hand on the object before and after a
"persistent spatial" contact event, and the modelled wrench capability before and after.

Data only, no model is run. Everything comes from the wrench counterfactual study
(``<result root>/reports/wrench_counterfactual``), all of it opened read-only:

* ``events.csv``: one row per test event (class, frames, retention, cosine, relative change, touching parts);
* ``<dataset>/geometry/<event>.npz``: the object vertices within 2 cm of the hand at the frame before and
  the frame after the event, their distance, the hand part nearest to each, and the recorded MANO hand;
* ``<dataset>/capacity/primary.npz``: the two capability profiles over the study's 76 directions.

The object mesh and an independent copy of the hand and of the hand-to-vertex distances come from
``qlib.load_example``; the script checks the two sources against each other and stores the result.

Four events are drawn, two per dataset, each chosen by a rule computed here (``select_events``):

* typical: the persistent spatial test event whose retention is the median of its class;
* a part joins: among the persistent spatial test events in which a hand part starts touching, the one
  with the median relative capability change.

Per event (one row): the frame before and the frame after it from one camera (``choose_camera``: the
direction that sees the most contact; in the rows in which a part joins, the most contact of that part),
the object grey with the vertices within 1 cm of the hand in the colour of the nearest hand part, the
recorded hand see-through and faintly tinted by part; then the two capability profiles as a line plot.

What a viewer can and cannot see of every contact patch is counted in pixels of flat renders from the row's
camera (``visible_pixels``); the caption and the caveats are written from these counts.

Run (CPU, software renderer):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_reconfiguration_with_hand.py
"""
from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib  # noqa: E402

FIG_ID = "reconfiguration_with_hand"
BLOCK = "F1-D"
STUDY = qlib.RESULT_ROOT / "reports/wrench_counterfactual"
EVENTS_CSV = STUDY / "events.csv"
REPORT_EXAMPLES_CSV = STUDY / "figures/figF_examples.csv"      # the example events of the study's own report
DATASETS = ("taco", "arctic")
DATASET_LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
KINDS = ("typical", "part_joins")

CONTACT_THR = 0.010            # metres: the study's contact rule (object vertex within 1 cm of the hand)
PANEL = (592, 444)             # pixels (width, height) of every panel
HAND_OPACITY = 0.5
HAND_TINT = 0.25               # share of the part colour in the hand's colour (the rest is skin)
N_VIEWS = 400                  # candidate camera directions (Fibonacci sphere)
HAND_BALL = 0.007              # metres: a hand vertex covers what lies within this distance of the line of sight
CAMERA_MARGIN = 0.10
HIDDEN_SHARE = 0.5             # a patch is "mostly hidden" when under this share of its pixels can be seen (hidden_parts)
END_SHARE = 0.05               # a patch is "at the end" of the object when it reaches within this share of its length
PART_COLOUR_WORD = {"palm": "dark grey", "thumb": "blue", "index": "orange", "middle": "green", "ring": "yellow", "little": "purple"}
PART_WORD = {"palm": "palm", "thumb": "thumb", "index": "index finger", "middle": "middle finger", "ring": "ring finger",
             "little": "little finger"}                                 # the parts in running text
FRAME_PX = 2                   # outline of the rendered panels: they are close-ups, the object may run out of them
BEFORE_COLOR, AFTER_COLOR = "#9a9890", qlib.INK   # the two lines of the capability plots
TOL = 1e-6                     # metres: tolerance of the geometry cross-checks


# ------------------------------------------------------------------------------ selection
def event_key(event_id: str) -> tuple[int, int]:
    """'3069_1' -> (3069, 1): the order meant by "smaller event id" (example number, then event number)."""
    example, index = event_id.split("_")
    return int(example), int(index)


def parts_of(text: Any) -> frozenset[str]:
    """'index+thumb' -> {'index', 'thumb'} (empty for a missing value)."""
    return frozenset(str(text).split("+")) if isinstance(text, str) and text else frozenset()


def median_event(table: pd.DataFrame, column: str) -> tuple[str, dict[str, Any]]:
    """The event whose ``column`` value is the median of ``table``: sort by (value, event id); an odd count has
    one middle event; with an even count no event sits at the median, the two middle events are equally far
    from it and the one with the smaller event id is taken. Event ids are compared as numbers (event_key:
    example number, then event number), not as text. Returns the event id and the ranking that proves the pick."""
    order = table.assign(_key=table.event_id.map(event_key)).sort_values([column, "_key"]).reset_index(drop=True)
    n = len(order)
    middle = sorted({(n - 1) // 2, n // 2})
    chosen = min(middle, key=lambda i: event_key(order.event_id[i]))
    evidence = {
        "metric": column, "n_candidates": n, "median": float(order[column].median()),
        "chosen_event": order.event_id[chosen], "chosen_value": float(order[column][chosen]), "chosen_rank": chosen + 1,
        "middle_ranks": [i + 1 for i in middle],
        "tie": (None if len(middle) == 1 else
                f"ranks {middle[0] + 1} and {middle[1] + 1} ({order.event_id[middle[0]]} at {order[column][middle[0]]:.4f}, "
                f"{order.event_id[middle[1]]} at {order[column][middle[1]]:.4f}) are equally far from the median of "
                f"{order[column].median():.4f}; the smaller event id is taken, ids compared as numbers (example number, then event "
                f"number), so {order.event_id[chosen]} comes first"),
        "sorted": [[e, float(v)] for e, v in zip(order.event_id, order[column])],
    }
    return order.event_id[chosen], evidence


def select_events(events: pd.DataFrame) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """Both rules, per dataset. Returns {(dataset, kind): event id} and the evidence for meta.json."""
    chosen: dict[tuple[str, str], str] = {}
    evidence: dict[str, Any] = {}
    report = pd.read_csv(REPORT_EXAMPLES_CSV)
    for ds in DATASETS:
        split = pd.read_csv(qlib.HIER[ds] / "sequences_meta.csv").split_set
        cls = events[(events.dataset == ds) & (events.class_report == "persistent_spatial")]
        cls = cls[split.iloc[cls.example.to_numpy()].to_numpy() == "test"]
        assert cls.R_pre_to_post.notna().all() and cls.event_id.is_unique
        gained = cls[[bool(parts_of(b) - parts_of(a)) for a, b in zip(cls.fingers_pre, cls.fingers_post)]]
        assert (gained.n_appeared >= 1).all() and len(gained) == int((cls.n_appeared >= 1).sum())   # the study's own count agrees
        chosen[ds, "typical"], typical = median_event(cls, "R_pre_to_post")
        chosen[ds, "part_joins"], joins = median_event(gained, "rel_change_Q")
        with np.load(STUDY / ds / "capacity" / "primary.npz", allow_pickle=True) as cap:
            n_directions = len(cap["direction"])
        evidence[ds] = {
            "class": "events.csv rows of this dataset with class_report == 'persistent_spatial' whose sequence is in the test split",
            "n_class": len(cls), "n_part_added": len(gained),
            "class_median_cosine": float(cls.cosine.median()), "class_median_rel_change_Q": float(cls.rel_change_Q.median()),
            "class_share_same_parts": float(np.mean([parts_of(a) == parts_of(b) for a, b in zip(cls.fingers_pre, cls.fingers_post)])),
            "n_directions": n_directions, "study_report_examples": sorted(report.event_id[report.dataset == ds]),
            "typical": typical, "part_joins": joins,
        }
    return chosen, evidence


# ------------------------------------------------------------------------------ the study's summaries of two profiles
def profile_numbers(q_pre: np.ndarray, q_post: np.ndarray) -> dict[str, float]:
    """Retention, cosine and relative change of two capability profiles, by the study's formulas
    (scripts/research/wrench_counterfactual/wrench_lp.py: retention, profile_metrics)."""
    both = np.maximum(q_pre, q_post) > 1e-6
    mean_pre, mean_post = float(q_pre.mean()), float(q_post.mean())
    return {
        "retention": float(np.mean(np.minimum(q_pre[both] / (q_post[both] + 1e-6), 1.0))),
        "cosine": float((q_pre * q_post).sum() / (np.linalg.norm(q_pre) * np.linalg.norm(q_post) + 1e-9)),
        "rel_change": (mean_post - mean_pre) / (mean_pre + 1e-9),
        "mean_before": mean_pre, "mean_after": mean_post,
    }


# ------------------------------------------------------------------------------ one event
@dataclass(frozen=True)
class Frame:
    t: int                     # sequence frame
    take_frame: int
    verts: np.ndarray          # (V, 3) object vertices, object frame, metres
    faces: np.ndarray          # (F, 3)
    hand: np.ndarray           # (778, 3) recorded MANO hand, object frame
    part: np.ndarray           # (V,) hand part touching the vertex (index into qlib.PART_NAMES), -1 = no contact
    normal: np.ndarray         # (V, 3) unit surface normal pointing to the hand at the vertices of the event file, 0 elsewhere
    checks: dict[str, Any]

    # The three per-contact-vertex arrays below are all cut from per-vertex arrays with the same mask, so they
    # are in the same (ascending mesh index) order whatever the order of the event file.
    @property
    def contact(self) -> np.ndarray:
        """(n contact, 3) positions of the contact vertices."""
        return self.verts[self.part >= 0]

    @property
    def toward(self) -> np.ndarray:
        """(n contact, 3) surface normals at the contact vertices, pointing to the hand."""
        return self.normal[self.part >= 0]

    @property
    def labels(self) -> np.ndarray:
        """(n contact,) hand part of the contact vertices."""
        return self.part[self.part >= 0]

    @property
    def parts(self) -> list[str]:
        return [name for k, name in enumerate(qlib.PART_NAMES) if (self.part == k).any()]


@dataclass(frozen=True)
class Event:
    dataset: str
    kind: str
    event_id: str
    row: pd.Series             # the event's row of events.csv
    example: qlib.Example
    before: Frame
    after: Frame
    q_before: np.ndarray       # (76,) capability profile
    q_after: np.ndarray
    numbers: dict[str, float]  # recomputed from the two profiles
    hand_faces: np.ndarray

    @property
    def key(self) -> str:
        return f"{self.dataset}_{self.kind}"

    @property
    def added(self) -> list[int]:
        """Hand parts (indices into qlib.PART_NAMES) that touch after the event and did not before it."""
        return [k for k in range(len(qlib.PART_NAMES)) if (self.after.part == k).any() and not (self.before.part == k).any()]


def load_frame(ex: qlib.Example, geom: Any, tag: str, t: int, row: pd.Series) -> Frame:
    """The frame before (tag 'pre') or after ('post') an event: mesh and contact on qlib's mesh, hand from the
    event file; every quantity that exists twice is compared."""
    verts, faces = ex.mesh(t)
    gap, vertex = cKDTree(verts).query(geom[f"{tag}_pos"].astype(np.float64))   # the event file's vertices on qlib's mesh
    assert gap.max() < TOL and len(np.unique(vertex)) == len(vertex), "event vertices do not match the mesh"
    dist = geom[f"{tag}_dist"].astype(np.float64)
    touching = dist < CONTACT_THR
    part = np.full(len(verts), -1, np.int64)
    part[vertex[touching]] = geom[f"{tag}_label"][touching]
    hand = geom[f"hand_{tag}"].astype(np.float64)
    toward = geom[f"{tag}_nrm"].astype(np.float64) * np.where(geom[f"{tag}_outward"], 1.0, -1.0)[:, None]
    normal = np.zeros((len(verts), 3))
    normal[vertex] = toward                                            # event-file order -> mesh order

    # The event file lists its vertices in its own order (not ascending on the ARCTIC notebook, whose mesh is
    # stored top part first): everything per contact vertex must be in mesh order before it is paired.
    order = np.argsort(vertex[touching])
    assert np.allclose(verts[part >= 0], geom[f"{tag}_pos"][touching][order], atol=TOL)
    assert np.array_equal(normal[part >= 0], toward[touching][order])
    assert np.array_equal(part[part >= 0], geom[f"{tag}_label"][touching][order])

    # cross-checks against qlib's independent copies and against the study's event table
    hand_qlib, _ = ex.hand(t)
    dist_qlib = -qlib.SOFT_SCALE * np.log(ex.dense(t))                 # distance of every vertex to the hand
    nearest = qlib.mano_parts()[cKDTree(hand).query(verts[vertex[touching]])[1]]
    names = "+".join(sorted(qlib.PART_NAMES[k] for k in np.unique(part[part >= 0])))
    checks = {
        "max_vertex_mismatch_m": float(gap.max()),
        "max_hand_mismatch_m": float(np.abs(hand_qlib - hand).max()),
        "max_distance_mismatch_m": float(np.abs(dist_qlib[vertex] - dist).max()),
        "same_contact_set_as_qlib_distances": bool(np.array_equal(np.flatnonzero(dist_qlib < CONTACT_THR), np.flatnonzero(part >= 0))),
        "n_contact_vertices": int(touching.sum()), "n_contact_vertices_events_csv": int(row[f"n_contact_{tag}"]),
        "parts": names, "parts_events_csv": "+".join(sorted(parts_of(row[f"fingers_{tag}"]))),
        "share_labels_equal_to_nearest_hand_vertex": float(np.mean(nearest == geom[f"{tag}_label"][touching])),
        "event_file_lists_contact_vertices_in_mesh_order": bool(np.all(np.diff(vertex[touching]) > 0)),
    }
    assert checks["max_hand_mismatch_m"] < TOL and checks["max_distance_mismatch_m"] < TOL
    assert checks["same_contact_set_as_qlib_distances"] and checks["n_contact_vertices"] == checks["n_contact_vertices_events_csv"]
    assert checks["parts"] == checks["parts_events_csv"]
    return Frame(t, ex.take_frame(t), verts, faces, hand, part, normal, checks)


def load_event(ds: str, kind: str, event_id: str, events: pd.DataFrame) -> Event:
    row = events[(events.dataset == ds) & (events.event_id == event_id)].iloc[0]
    ex = qlib.load_example(ds, int(row.example))
    assert ex.meta.sequence_id == row.sequence_id and ex.meta.split == "test"
    with np.load(STUDY / ds / "geometry" / f"{event_id}.npz") as geom:
        frames = json.loads(str(geom["frames"]))
        assert (frames["pre"], frames["post"]) == (int(row.pre_frame), int(row.post_frame))
        before = load_frame(ex, geom, "pre", frames["pre"], row)
        after = load_frame(ex, geom, "post", frames["post"], row)
    with np.load(STUDY / ds / "capacity" / "primary.npz", allow_pickle=True) as cap:
        i = list(cap["event_id"]).index(event_id)
        q_before, q_after = cap["q_pre"][i].astype(np.float64), cap["q_post"][i].astype(np.float64)
    return Event(ds, kind, event_id, row, ex, before, after, q_before, q_after, profile_numbers(q_before, q_after),
                 qlib.mano_faces(qlib.SIDE[ex.meta.hand]))


# ------------------------------------------------------------------------------ camera
def fibonacci_sphere(n: int) -> np.ndarray:
    """(n, 3) unit vectors spread evenly over the sphere."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z ** 2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def contact_in_view(direction: np.ndarray, frame: Frame) -> np.ndarray:
    """(n contact,) how much of each contact vertex of a frame a camera in ``direction`` sees: the cosine between
    the surface normal (pointing to the hand) and the direction where the vertex faces the camera, 0 where it
    does not. A vertex whose line of sight passes a hand vertex within HAND_BALL is seen through the see-through
    hand and counts 1 - HAND_OPACITY. In the order of frame.contact and frame.labels."""
    facing = np.clip(frame.toward @ direction, 0.0, None)
    rel = frame.hand[None] - frame.contact[:, None]                     # (n, 778, 3)
    along = rel @ direction
    across2 = (rel ** 2).sum(-1) - along ** 2
    covered = ((along > HAND_BALL) & (across2 < HAND_BALL ** 2)).any(1)
    return facing * np.where(covered, 1.0 - HAND_OPACITY, 1.0)


def choose_camera(event: Event) -> tuple[qlib.Camera, dict[str, Any]]:
    """One camera for both frames of an event, computed from the geometry alone, out of N_VIEWS directions.

    Typical event: the direction from which the most contact is seen (contact_in_view summed over all contact
    vertices of both frames). Event in which a part joins: the direction from which the most contact of the
    part that joins is seen (the same sum over that part's vertices only), the all-contact sum breaking ties:
    the joining part is what the row is about, and the all-contact view can leave it on the far side.

    The object's longest axis is horizontal and the hand is in the upper half; fitted to the two hands and the
    contact vertices."""
    frames = (event.before, event.after)
    views = fibonacci_sphere(N_VIEWS)
    added = event.added
    all_scores, added_scores = np.zeros(N_VIEWS), np.zeros(N_VIEWS)
    for i, d in enumerate(views):
        for f in frames:
            seen = contact_in_view(d, f)
            all_scores[i] += seen.sum()
            added_scores[i] += seen[np.isin(f.labels, added)].sum()
    if event.kind == "part_joins":
        assert added, "an event in which a part joins has an added part"
        rule = "the part that joins first"
        best = int(np.lexsort((all_scores, added_scores))[-1])          # last key is the primary one
    else:
        rule = "all contact"
        best = int(all_scores.argmax())
    direction = views[best]
    n_added = sum(int(np.isin(f.labels, added).sum()) for f in frames)
    centre = event.before.verts.mean(0)
    _, axes = np.linalg.eigh(np.cov((event.before.verts - centre).T))   # columns: thinnest, middle, longest axis
    up = np.cross(direction, axes[:, 2])
    if np.linalg.norm(up) < 0.3:                                        # looking along the longest axis: use the middle one
        up = np.cross(direction, axes[:, 1])
    hands = np.concatenate([f.hand for f in frames])
    contact = np.concatenate([f.contact for f in frames])
    if up @ (hands.mean(0) - contact.mean(0)) < 0:
        up = -up
    camera = qlib.fit_camera([hands, contact], direction, up, margin=CAMERA_MARGIN, aspect=PANEL[0] / PANEL[1])
    record = {"rule": rule, "candidate": best, "direction": direction.tolist(), "up": list(camera.up), "position": list(camera.position),
              "focal_point": list(camera.focal_point), "view_angle": camera.view_angle, "n_candidates": N_VIEWS,
              "score": float(all_scores[best]), "score_per_contact_vertex": float(all_scores[best] / sum(len(f.contact) for f in frames)),
              "best_all_contact_candidate": int(all_scores.argmax()), "best_all_contact_score": float(all_scores.max()),
              "share_of_best_all_contact_score": float(all_scores[best] / all_scores.max()),
              "added_parts": [qlib.PART_NAMES[k] for k in added],
              "added_part_score": float(added_scores[best]) if added else None,
              "added_part_score_per_vertex": float(added_scores[best] / n_added) if added else None,
              "best_added_part_score": float(added_scores.max()) if added else None,
              "added_part_score_at_best_all_contact_candidate": float(added_scores[all_scores.argmax()]) if added else None}
    return camera, record


def facing_camera(frame: Frame, direction: np.ndarray) -> dict[str, list[int]]:
    """Per touching part: [contact vertices whose surface normal (pointing to the hand) faces the camera, all
    contact vertices]. The sign of a normal, not what is visible: see visible_pixels for that."""
    faces_camera = frame.toward @ direction > 0
    return {name: [int(faces_camera[frame.labels == k].sum()), int((frame.labels == k).sum())]
            for k, name in enumerate(qlib.PART_NAMES) if (frame.labels == k).any()}


# ------------------------------------------------------------------------------ what can be seen
def flat_render(meshes: list[tuple[np.ndarray, np.ndarray, np.ndarray]], camera: qlib.Camera) -> np.ndarray:
    """(height, width, 3) uint8 picture of ``meshes`` (vertices, faces, per-vertex rgb in [0, 1]) of PANEL size:
    no lighting, no anti-aliasing, black background, both sides of every triangle. Only used to count pixels;
    qlib.render always lights and anti-aliases, so this goes to pyvista directly, with the same checks."""
    qlib.require_headless()
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=list(PANEL))
    try:
        if pl.render_window.GetClassName() != "vtkOSOpenGLRenderWindow":
            raise RuntimeError("VTK did not select the OSMesa window: " + pl.render_window.GetClassName())
        pl.set_background("black")
        for verts, faces, rgb in meshes:
            poly = pv.PolyData(np.asarray(verts, float), np.hstack([np.full((len(faces), 1), 3), faces]))
            poly.point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
            pl.add_mesh(poly, scalars="rgb", rgb=True, lighting=False, show_scalar_bar=False)
        pl.camera.position, pl.camera.focal_point, pl.camera.up = camera.position, camera.focal_point, camera.up
        pl.camera.view_angle = camera.view_angle
        pl.reset_camera_clipping_range()
        image = pl.screenshot(None, return_img=True)
    finally:
        pl.close()
    return np.ascontiguousarray(image[..., :3])


def visible_pixels(event: Event, frame: Frame, camera: qlib.Camera) -> tuple[dict[str, dict[str, int]], dict[str, np.ndarray]]:
    """Per touching part, how many pixels of its contact patch a viewer of the panel can see, counted in flat
    renders from the panel's camera. A pixel belongs to a part's patch where the part's indicator (1 on its
    contact vertices, 0 elsewhere, interpolated over the triangles like the colours of the figure) is above 0.5.

    * patch_alone: the pixels the patch covers when nothing else is drawn (the triangles that have a contact
      vertex of the part, seen from either side): what there would be to see if nothing hid it;
    * on_object: the pixels of the patch on the whole object with the hand removed (the rest of the object and
      the patch's own far side hide the others);
    * clear_of_hand: of those, the pixels the hand is not in front of (the rest is seen through the hand).

    Also returns the two part maps ((height, width) int8, -1 = no patch) for panels.npz."""
    n_parts = len(qlib.PART_NAMES)
    onehot = np.zeros((len(frame.verts), n_parts))
    onehot[np.flatnonzero(frame.part >= 0), frame.labels] = 1.0
    black_hand = (frame.hand, event.hand_faces, np.zeros((len(frame.hand), 3)))

    def part_map(with_hand: bool) -> np.ndarray:
        """(height, width) int8: the part whose patch the pixel shows, -1 where none. Three parts per render, one
        per colour channel; the largest indicator wins (the rasteriser overshoots on a few sliver triangles)."""
        halves = [flat_render([(frame.verts, frame.faces, onehot[:, c:c + 3])] + ([black_hand] if with_hand else []), camera)
                  for c in range(0, n_parts, 3)]
        value = np.concatenate(halves, -1)
        return np.where(value.max(-1) > 127, value.argmax(-1), -1).astype(np.int8)

    on_object, clear = part_map(False), part_map(True)
    assert ((clear == -1) | (clear == on_object)).all()                # the hand only takes pixels away
    counts: dict[str, dict[str, int]] = {}
    alone_bits = np.zeros(on_object.shape, np.uint8)                   # bit k: the patch of part k drawn alone covers the pixel
    for k, name in enumerate(qlib.PART_NAMES):
        if not (frame.part == k).any():
            continue
        patch = frame.faces[(frame.part[frame.faces] == k).any(1)]
        alone = flat_render([(frame.verts, patch, np.repeat(onehot[:, k:k + 1], 3, 1))], camera)[..., 0] > 127
        alone_bits |= (alone.astype(np.uint8) << k).astype(np.uint8)
        counts[name] = {"vertices": int((frame.part == k).sum()), "patch_alone": int(alone.sum()),
                        "on_object": int((on_object == k).sum()), "clear_of_hand": int((clear == k).sum())}
    return counts, {"patch_on_object": on_object, "patch_clear_of_hand": clear, "patch_alone_bits": alone_bits}


def hidden_parts(pixels: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    """Which parts' contact a viewer of the row cannot see well, from the pixel counts of both frames summed
    (HIDDEN_SHARE is the threshold of both tests):

    * far_side: under HIDDEN_SHARE of the patch's pixels are visible on the object (of these, not_visible: none);
    * under_hand: not far_side, and under HIDDEN_SHARE of the visible pixels are clear of the hand."""
    total: dict[str, dict[str, int]] = {}
    for frame in pixels.values():
        for part, n in frame.items():
            sums = total.setdefault(part, dict.fromkeys(("patch_alone", "on_object", "clear_of_hand"), 0))
            for key in sums:
                sums[key] += n[key]
    present = [p for p in qlib.PART_NAMES if p in total]
    far = [p for p in present if total[p]["on_object"] < HIDDEN_SHARE * total[p]["patch_alone"]]
    return {"far_side": far, "not_visible": [p for p in far if total[p]["on_object"] == 0],
            "under_hand": [p for p in present if p not in far and total[p]["clear_of_hand"] < HIDDEN_SHARE * total[p]["on_object"]],
            "pixels_both_frames": {p: total[p] for p in present}}


def patch_place(event: Event, frame: Frame, part: int, camera: qlib.Camera) -> str:
    """Where the patch of hand part ``part`` is in the picture, in words, from the geometry alone:
    'at the left end of the <object>' when the patch lies left of the patches of all other parts in the picture
    and reaches the end of the object that is on that side (within END_SHARE of the object's length along its
    longest axis); 'left of the other patches' when only the first holds; '' otherwise. Same for right."""
    back = np.asarray(camera.position) - np.asarray(camera.focal_point)
    right = np.cross(camera.up, back / np.linalg.norm(back))
    x = {k: float(frame.verts[frame.part == k].mean(0) @ right) for k in np.unique(frame.labels)}
    others = [v for k, v in x.items() if k != part]
    if not others or min(others) <= x[part] <= max(others):
        return ""
    side = "left" if x[part] < min(others) else "right"
    centre = frame.verts.mean(0)
    _, axes = np.linalg.eigh(np.cov((frame.verts - centre).T))
    along = (frame.verts - centre) @ axes[:, 2]                         # position along the object's longest axis
    patch = along[frame.part == part]
    low_end = patch.min() - along.min() < along.max() - patch.max()     # which end of the object the patch is nearer to
    gap = patch.min() - along.min() if low_end else along.max() - patch.max()
    end_vertex = frame.verts[along.argmin() if low_end else along.argmax()]
    end_side = "left" if (end_vertex - centre) @ right < 0 else "right"
    if gap < END_SHARE * (along.max() - along.min()) and end_side == side:
        return f"at the {side} end of the {event.example.meta.category}"
    return f"{side} of the other patches"


# ------------------------------------------------------------------------------ drawing
def hex_rgb(color: str) -> np.ndarray:
    """'#rrggbb' -> (3,) float in [0, 1]."""
    return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0


def object_colors(frame: Frame) -> np.ndarray:
    """(V, 3): the colour of the touching hand part on contact vertices, neutral grey elsewhere."""
    rgb = np.tile(hex_rgb(qlib.ZERO), (len(frame.verts), 1))
    rgb[frame.part >= 0] = qlib.PART_COLORS[frame.part[frame.part >= 0]]
    return rgb


def hand_colors() -> np.ndarray:
    """(778, 3): skin mixed with the colour of the vertex's hand part."""
    return HAND_TINT * qlib.PART_COLORS[qlib.mano_parts()] + (1 - HAND_TINT) * hex_rgb(qlib.HAND_SKIN)


def render_frame(event: Event, frame: Frame, camera: qlib.Camera) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    obj_rgb, hand_rgb = object_colors(frame), hand_colors()
    image = qlib.render([qlib.mesh_item(frame.verts, frame.faces, rgb=obj_rgb),
                         qlib.hand_item(frame.hand, event.hand_faces, rgb=hand_rgb, opacity=HAND_OPACITY)], camera, PANEL)
    outline = (hex_rgb(qlib.RULE) * 255).round().astype(np.uint8)
    for edge in (np.s_[:FRAME_PX], np.s_[-FRAME_PX:]):
        image[edge], image[:, edge] = outline, outline
    drawn = {"object_verts": frame.verts, "object_faces": frame.faces, "object_rgb": obj_rgb, "contact_part": frame.part,
             "hand_verts": frame.hand, "hand_faces": event.hand_faces, "hand_rgb": hand_rgb,
             "camera": np.array([*camera.position, *camera.focal_point, *camera.up, camera.view_angle])}
    return image, drawn


def profile_panel(event: Event, y_max: float, font_px: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The two capability profiles as an image of PANEL size; directions sorted by the value before the event."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.argsort(-event.q_before, kind="stable")
    dpi = 100
    pt = font_px * 72.0 / dpi
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": pt, "text.color": qlib.INK, "axes.labelcolor": qlib.INK,
                         "axes.edgecolor": qlib.INK, "xtick.color": qlib.INK, "ytick.color": qlib.INK, "axes.linewidth": 1.5}):
        fig, ax = plt.subplots(figsize=(PANEL[0] / dpi, PANEL[1] / dpi), dpi=dpi)
        x = np.arange(1, len(order) + 1)
        ax.plot(x, event.q_before[order], color=BEFORE_COLOR, lw=5, label="before", solid_capstyle="round")
        ax.plot(x, event.q_after[order], color=AFTER_COLOR, lw=2.5, label="after")
        ax.set_xlim(0.5, len(order) + 0.5)
        ax.set_ylim(0, y_max)
        ax.set_xticks([1, len(order)])
        ax.set_yticks(np.arange(0, y_max + 1e-9, 2))
        ax.set_xlabel(f"{len(order)} directions, sorted", labelpad=-pt * 0.6)
        ax.set_ylabel("capability")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=6, width=1.5)
        ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.0, labelspacing=0.2)
        fig.subplots_adjust(left=0.17, right=0.95, top=0.96, bottom=0.20)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, facecolor="white")
        plt.close(fig)
    image = np.asarray(Image.open(buffer).convert("RGB"))
    assert image.shape == (PANEL[1], PANEL[0], 3)
    return image, {"q_before": event.q_before, "q_after": event.q_after, "direction_order": order}


# ------------------------------------------------------------------------------ text
def object_phrase(event: Event) -> str:
    m = event.example.meta
    return f"{m.category}, {qlib.SIDE[m.hand]} hand"


def row_label(event: Event) -> str:
    kind = "typical event (median retention)" if event.kind == "typical" else "event in which a part joins (median change)"
    return f"{DATASET_LABEL[event.dataset]}\n{kind}:\n{object_phrase(event)}"


def joined(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def pct(x: float) -> str:
    return f"{100 * x:+.0f}%".replace("-", "\N{MINUS SIGN}")


def describe(event: Event, camera: qlib.Camera, direction: np.ndarray, pixels: dict[str, dict[str, dict[str, int]]]) -> dict[str, Any]:
    """Everything printed and stored about one event; ``camera`` and ``direction`` are those of its row,
    ``pixels`` the counts of visible_pixels for its two frames."""
    before, after = set(event.before.parts), set(event.after.parts)
    move = np.linalg.norm(event.after.hand - event.before.hand, axis=1)

    def count(frame: Frame) -> dict[str, int]:
        return {name: int((frame.part == k).sum()) for k, name in enumerate(qlib.PART_NAMES) if (frame.part == k).any()}

    m = event.example.meta
    return {
        "dataset": event.dataset, "kind": event.kind, "event_id": event.event_id, "example": m.example, "sequence_id": m.sequence_id,
        "object": m.category, "mesh_id": m.mesh_id, "hand": qlib.SIDE[m.hand], "split": m.split,
        "frames_drawn": {"before": event.before.t, "after": event.after.t},
        "take_frames_drawn": {"before": event.before.take_frame, "after": event.after.take_frame},
        "parts_before": event.before.parts, "parts_after": event.after.parts,
        "parts_added": [p for p in qlib.PART_NAMES if p in after - before],
        "parts_removed": [p for p in qlib.PART_NAMES if p in before - after],
        "contact_vertices_before": count(event.before), "contact_vertices_after": count(event.after),
        "contact_pixels": pixels, "contact_visibility": hidden_parts(pixels),
        "added_patch_place": {qlib.PART_NAMES[k]: patch_place(event, event.after, k, camera) for k in event.added},
        "contact_vertices_with_normal_facing_camera": {"before": facing_camera(event.before, direction),
                                                       "after": facing_camera(event.after, direction)},
        "hand_displacement_mm": {"mean": float(move.mean() * 1000), "max": float(move.max() * 1000)},
        "retention": event.numbers["retention"], "cosine": event.numbers["cosine"], "rel_change": event.numbers["rel_change"],
        "mean_capability_before": event.numbers["mean_before"], "mean_capability_after": event.numbers["mean_after"],
        "share_of_directions_with_lower_capability_after": float(np.mean(event.q_after < event.q_before)),
        "events_csv": {"R_pre_to_post": float(event.row.R_pre_to_post), "cosine": float(event.row.cosine),
                       "rel_change_Q": float(event.row.rel_change_Q), "Q_pre": float(event.row.Q_pre), "Q_post": float(event.row.Q_post),
                       "fingers_pre": event.row.fingers_pre, "fingers_post": event.row.fingers_post, "dC": float(event.row.dC),
                       "contact_threshold_m": float(event.row.thr), "friction": float(event.row.mu), "budget": event.row.budget,
                       "part_changes": {p: event.row[f"fc_{p}"] for p in qlib.PART_NAMES if event.row[f"fc_{p}"] != "absent"}},
        "cross_checks": {"before": event.before.checks, "after": event.after.checks},
    }


def numbers_for(events: list[Event], facts: dict[str, dict[str, Any]], evidence: dict[str, Any],
                cameras: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Every number a caption could quote, with the file and the row or key it comes from."""
    csv = str(EVENTS_CSV)
    panels_file = str(qlib.OUT_ROOT / FIG_ID / "panels.npz")
    out: list[dict[str, Any]] = []

    def add(label: str, value: Any, dataset: str, source: str, locator: str) -> None:
        out.append({"label": label, "value": value, "dataset": dataset, "source": source, "locator": locator})

    for ds in DATASETS:
        e = evidence[ds]
        cls = f"rows with dataset == '{ds}' and class_report == 'persistent_spatial'"
        add("persistent spatial events in the test split", e["n_class"], ds, csv, cls + " (count)")
        add("median retention of the class", e["typical"]["median"], ds, csv, cls + ", column R_pre_to_post (median)")
        add("median profile cosine of the class", e["class_median_cosine"], ds, csv, cls + ", column cosine (median)")
        add("median relative capability change of the class", e["class_median_rel_change_Q"], ds, csv, cls + ", column rel_change_Q (median)")
        add("share of the class with the same touching parts before and after", e["class_share_same_parts"], ds, csv,
            cls + ", fingers_pre == fingers_post (mean)")
        add("events of the class in which a hand part is added", e["n_part_added"], ds, csv,
            cls + ", fingers_post has a part that fingers_pre lacks (count)")
        add("median relative capability change among the events with an added part", e["part_joins"]["median"], ds, csv,
            cls + " with an added part, column rel_change_Q (median)")
        for kind in KINDS:
            pool = cls + (" with an added part" if kind == "part_joins" else "")
            add(f"rank of the chosen event ({kind})", f"{e[kind]['chosen_rank']} of {e[kind]['n_candidates']}", ds, csv,
                f"computed in this script: {pool}, sorted by {e[kind]['metric']}, then event id")
        add("directions of a capability profile", e["n_directions"], ds, str(STUDY / ds / "capacity/primary.npz"), "length of key 'direction'")
    for ev in events:
        f, who = facts[ev.key], f"{ev.event_id} ({ev.kind})"
        row = f"row with dataset == '{ev.dataset}' and event_id == '{ev.event_id}'"
        profiles = str(STUDY / ev.dataset / "capacity/primary.npz")
        at = f"q_pre, q_post in the row where event_id == '{ev.event_id}'"
        geometry = str(STUDY / ev.dataset / "geometry" / f"{ev.event_id}.npz")
        add(f"retention, event {who}", f["events_csv"]["R_pre_to_post"], ev.dataset, csv, row + ", column R_pre_to_post")
        add(f"profile cosine, event {who}", f["events_csv"]["cosine"], ev.dataset, csv, row + ", column cosine")
        add(f"relative capability change, event {who}", f["events_csv"]["rel_change_Q"], ev.dataset, csv, row + ", column rel_change_Q")
        add(f"retention recomputed from the drawn profiles, event {who}", f["retention"], ev.dataset, profiles, at)
        add(f"profile cosine recomputed from the drawn profiles, event {who}", f["cosine"], ev.dataset, profiles, at)
        add(f"relative change recomputed from the drawn profiles, event {who}", f["rel_change"], ev.dataset, profiles, at)
        add(f"mean capability before, event {who}", f["mean_capability_before"], ev.dataset, profiles,
            f"mean of q_pre in the row where event_id == '{ev.event_id}'")
        add(f"mean capability after, event {who}", f["mean_capability_after"], ev.dataset, profiles,
            f"mean of q_post in the row where event_id == '{ev.event_id}'")
        add(f"frame before / after, event {who}", [ev.before.t, ev.after.t], ev.dataset, csv, row + ", columns pre_frame, post_frame")
        add(f"contact threshold (m), event {who}", f["events_csv"]["contact_threshold_m"], ev.dataset, csv, row + ", column thr")
        add(f"friction coefficient of the capability model, event {who}", f["events_csv"]["friction"], ev.dataset, csv, row + ", column mu")
        add(f"units of normal force per touching hand part in the capability model, event {who}", 1, ev.dataset, csv,
            row + f", column budget == '{f['events_csv']['budget']}' (one unit per finger and for the palm)")
        add(f"mean displacement of the hand vertices between the two frames (mm), event {who}", f["hand_displacement_mm"]["mean"],
            ev.dataset, geometry, "mean over the 778 vertices of |hand_post - hand_pre|")
        add(f"touching hand parts before, event {who}", len(f["parts_before"]), ev.dataset, geometry, "distinct pre_label where pre_dist < 0.01")
        add(f"touching hand parts after, event {who}", len(f["parts_after"]), ev.dataset, geometry, "distinct post_label where post_dist < 0.01")
        add(f"contact vertices before, event {who}", sum(f["contact_vertices_before"].values()), ev.dataset, geometry, "pre_dist < 0.01 (count)")
        add(f"contact vertices after, event {who}", sum(f["contact_vertices_after"].values()), ev.dataset, geometry, "post_dist < 0.01 (count)")
        for part in f["parts_added"]:
            add(f"contact vertices of the added part ({part}) after, event {who}", f["contact_vertices_after"][part], ev.dataset, geometry,
                f"post_dist < 0.01 and post_label == {qlib.PART_NAMES.index(part)} (count)")
        add(f"share of the directions in which the capability is lower after than before, event {who}",
            f["share_of_directions_with_lower_capability_after"], ev.dataset, profiles, f"mean of q_post < q_pre in the row where event_id == '{ev.event_id}'")
        vis = f["contact_visibility"]
        for part in qlib.PART_NAMES:                    # the pixel counts behind every statement about what can be seen
            if part in vis["far_side"] or part in vis["under_hand"] or part in f["parts_added"]:
                n, k = vis["pixels_both_frames"][part], qlib.PART_NAMES.index(part)
                add(f"pixels of the {part}'s contact patch, both frames summed [drawn alone, visible on the object, of those clear of "
                    f"the hand], event {who}", [n["patch_alone"], n["on_object"], n["clear_of_hand"]], ev.dataset, panels_file,
                    f"keys '{ev.key}_before/...' and '{ev.key}_after/...': pixels with bit {k} set in patch_alone_bits, pixels == {k} in "
                    f"patch_on_object, pixels == {k} in patch_clear_of_hand")
        camera = cameras[ev.key]
        if camera["rule"] != "all contact":
            add(f"all-contact score of the row's camera as a share of the best all-contact score, event {who}",
                camera["share_of_best_all_contact_score"], ev.dataset, geometry,
                "computed in this script (choose_camera) from pre/post_pos, _nrm, _outward, _dist, _label and hand_pre/post; "
                "stored in display_choices.camera.per_event")
    return out


def fingers(parts: list[str]) -> str:
    """['index', 'middle'] -> 'index finger and middle finger'."""
    return joined([PART_WORD[p] for p in parts])


def added_note(f: dict[str, Any], part: str) -> str:
    """'the dark grey patch at the left end of the knife': how to find the patch of a part that joins in the picture."""
    text = f"the {PART_COLOUR_WORD[part]} patch"
    if f["added_patch_place"][part]:
        text += " " + f["added_patch_place"][part]
    if part in f["contact_visibility"]["under_hand"]:
        text += ", seen through the hand"
    return text


def event_clause(f: dict[str, Any]) -> str:
    """One event in words, from its computed values."""
    if f["parts_added"]:
        what = " and ".join(f"the {PART_WORD[p]} starts touching with {f['contact_vertices_after'][p]} mesh vertices, {added_note(f, p)}"
                            for p in f["parts_added"])
        if f["parts_removed"]:
            what += f" and the {fingers(f['parts_removed'])} stops touching"
        if f["kind"] == "typical":
            what += ", so the median event itself gains a part"
    elif f["parts_removed"]:
        what = f"the {fingers(f['parts_removed'])} stops touching"
    else:
        what = f"the same {len(f['parts_before'])} parts touch before and after"
    return (f"on the {f['object']} the hand moves {f['hand_displacement_mm']['mean']:.0f} mm on average and {what} "
            f"(retention {f['events_csv']['R_pre_to_post']:.2f}, capability {pct(f['events_csv']['rel_change_Q'])})")


def unseen_clause(f: dict[str, Any]) -> str:
    """What a viewer of the row of one event cannot see, in words ('' if every patch is at least half visible)."""
    vis = f["contact_visibility"]
    gone = vis["not_visible"]
    partly = [p for p in vis["far_side"] if p not in gone]
    bits = []
    if gone:
        bits.append(f"the contact of the {fingers(gone)} is on the far side and cannot be seen")
    if partly:
        bits.append(f"less than half of the contact of the {fingers(partly)} can be seen")
    return f"on the {f['object']} " + ", and ".join(bits) if bits else ""


def class_size(e: dict[str, Any], kind: str, label: str) -> str:
    """'26 on ARCTIC, an even count, so ...': the number of events a rule chose from, with the tie if there is one."""
    pick = e[kind]
    text = f"{pick['n_candidates']} on {label}"
    if pick["tie"]:
        text += (f", an even count, so the two middle events are equally far from the median of {pick['median']:.2f} and the one "
                 "with the smaller id is drawn")
    return text


def caption(facts: dict[str, dict[str, Any]], evidence: dict[str, Any]) -> str:
    """The caption, written from the computed values so that it cannot drift from them."""
    t, a = evidence["taco"], evidence["arctic"]
    unseen = "; ".join(c for c in (unseen_clause(f) for f in facts.values()) if c)
    return (
        "Four recorded events in which the contact moves over the object while the hand keeps hold of it, two from TACO and two from "
        "ARCTIC, each drawn at the frame before and the frame after the event: the object, the recorded hand (see-through) and the "
        "object surface within 1 cm of the hand, in the colour of the nearest hand part. "
        "In each dataset the first event is selected as the one at the median retention among the events of this kind in the test "
        f"split ({class_size(t, 'typical', 'TACO')}; {class_size(a, 'typical', 'ARCTIC')}), the second as the one at the median "
        f"capability change among those in which a hand part starts touching ({class_size(t, 'part_joins', 'TACO')}; "
        f"{class_size(a, 'part_joins', 'ARCTIC')}). "
        f"TACO: {event_clause(facts['taco_typical'])}; {event_clause(facts['taco_part_joins'])}. "
        f"ARCTIC: {event_clause(facts['arctic_typical'])}; {event_clause(facts['arctic_part_joins'])}. "
        "Each row is drawn from one camera, the direction that sees the most contact or, where a part joins, the most contact of that "
        "part" + (f", so not every patch can be seen: {unseen}. " if unseen else ". ") +
        f"The plots show the modelled capability along {t['n_directions']} directions before (grey) and after (black), sorted by the "
        "value before, and retention is the share of the capability after the event that the hand already had before it; the "
        "capability comes from a friction model that gives every touching hand part one unit of force, so a part that starts "
        "touching can only add to it by construction, and it is not a measurement."
    )


def caveats_for(events: list[Event], facts: dict[str, dict[str, Any]], evidence: dict[str, Any],
                cameras: dict[str, dict[str, Any]]) -> list[str]:
    """What the figure does not show, and how typical the examples are; written from the computed values."""
    out = ["Four single events, one per row. They show what such an event looks like, not how the class is spread; the class "
           "statistics are in the charts of the same page block."]
    for ev in events:
        f, e = facts[ev.key], evidence[ev.dataset]
        name = f"{DATASET_LABEL[ev.dataset]} event {ev.event_id} ({f['object']})"
        if ev.kind == "typical":
            pick = e["typical"]
            if pick["tie"]:
                text = (f"{name} stands for the median retention of its class, but the class has an even number of events "
                        f"({e['n_class']}), so no event sits at the median: {pick['tie']}. Its retention is "
                        f"{pick['chosen_value']:.3f} (rank {pick['chosen_rank']}); it is typical in retention only.")
            else:
                text = (f"{name} is the event at the median retention of its class ({pick['chosen_value']:.3f}; rank "
                        f"{pick['chosen_rank']} of {e['n_class']}), so it is typical in retention only.")
            if f["parts_added"] or f["parts_removed"]:
                text += (f" It is itself an event in which the set of touching parts changes ({joined(f['parts_added'] + f['parts_removed'])}), "
                         f"with a capability change of {pct(f['events_csv']['rel_change_Q'])}, so the two {DATASET_LABEL[ev.dataset]} rows "
                         "differ in degree, not in kind.")
            text += (f" In {100 * e['class_share_same_parts']:.0f}% of the {e['n_class']} events of the class the set of touching parts "
                     "is the same before and after.")
            out.append(text)
        else:
            n_before, n_after = sum(f["contact_vertices_before"].values()), sum(f["contact_vertices_after"].values())
            added = ", ".join(f"{p} {f['contact_vertices_after'][p]}" for p in f["parts_added"])
            out.append(f"{name} is the event at the median capability change ({e['part_joins']['chosen_value']:+.3f}; rank "
                       f"{e['part_joins']['chosen_rank']} of {e['n_part_added']}) among the events of its class in which a part is added. "
                       f"Contact vertices of the added part after the event: {added}; all contact vertices: {n_before} before, {n_after} after. "
                       "The capability model counts touching parts and where their patches are, not the contact area.")
            camera = cameras[ev.key]
            if camera["candidate"] != camera["best_all_contact_candidate"]:
                out.append(f"{name}: the camera is the direction that sees the most contact of the part that joins "
                           f"({joined(f['parts_added'])}); it reaches {100 * camera['share_of_best_all_contact_score']:.0f}% of the score "
                           "of the direction that sees the most contact overall, so the other patches are shown less well than they could be.")
        vis = f["contact_visibility"]
        total = vis["pixels_both_frames"]
        if vis["far_side"]:
            partly = [p for p in vis["far_side"] if p not in vis["not_visible"]]
            what = " and ".join(([f"all of the contact of the {joined(vis['not_visible'])}"] if vis["not_visible"] else [])
                                + ([f"more than half of the contact of the {joined(partly)}"] if partly else []))
            detail = "; ".join(f"{p}: {total[p]['on_object']} of {total[p]['patch_alone']}" for p in vis["far_side"])
            out.append(f"{name}: seen from the row's camera, the object hides {what} (pixels of the patch visible on the object, of "
                       f"the pixels the patch covers when drawn alone, both frames summed: {detail}). One view cannot show both sides "
                       "of the object.")
        if vis["under_hand"]:
            detail = "; ".join(f"{p}: {total[p]['clear_of_hand']} of {total[p]['on_object']}" for p in vis["under_hand"])
            out.append(f"{name}: the contact of the {joined(vis['under_hand'])} is seen mostly through the see-through hand, at reduced "
                       f"strength (visible patch pixels that the hand is not in front of, both frames summed: {detail}).")
        if f["event_id"] in e["study_report_examples"]:
            out.append(f"{name} is also one of the example events of the study's own report, chosen there by a different rule; here it is "
                       "the event the stated rule returns.")
    gaps = ", ".join(f"{ev.event_id}: {ev.after.t - ev.before.t}" for ev in events)
    lower = ", ".join(f"{100 * facts[ev.key]['share_of_directions_with_lower_capability_after']:.0f}%" for ev in events)
    out += [
        f"Only the frame before and the frame after each event are drawn (frames apart: {gaps}); the motion in between is not shown.",
        "Contact is proximity (an object vertex within 1 cm of the recorded hand mesh), not measured force. The capability is a model "
        "output (linearised friction cones, friction coefficient 0.5, one unit of normal force per touching hand part, single hand): "
        "every touching part adds a term that is never negative, so a part that starts touching can only add to it. The total can "
        f"still fall where the other parts' contact moves: it is lower after the event in {lower} of the "
        f"{evidence[events[0].dataset]['n_directions']} directions of the four events (rows top to bottom).",
        "What can be seen is counted in pixels of flat renders from the row's camera (visible_pixels in the script; maps in "
        "panels.npz). A patch that wraps around the object covers the same pixels with its near and its far side, so the counts say "
        "what the picture shows, not which share of the contact area faces the camera.",
        *[f"{DATASET_LABEL[ev.dataset]} event {ev.event_id} ({facts[ev.key]['object']}): the palm's patch ("
          f"{facts[ev.key]['contact_vertices_after']['palm']} contact vertices, {facts[ev.key]['contact_pixels']['after']['palm']['on_object']} "
          "visible pixels in the frame after) is dark grey on a grey object and can be taken for shading; the caption says where it is."
          for ev in events if "palm" in facts[ev.key]["parts_added"]],
        "The renders are close-ups fitted to the hand and the contact region; a larger object runs out of the panel (the grey outline "
        "is the panel edge). Colours are interpolated between mesh vertices, so the outline of a patch follows the mesh triangles. "
        f"The see-through hand is skin colour mixed with {100 * HAND_TINT:.0f}% of the colour of each part, so the strong colours are on "
        "the object only.",
    ]
    return out


# ------------------------------------------------------------------------------ main
def main() -> None:
    qlib.require_headless()
    events_table = pd.read_csv(EVENTS_CSV)
    chosen, evidence = select_events(events_table)
    events = [load_event(ds, kind, chosen[ds, kind], events_table) for ds in DATASETS for kind in KINDS]
    cameras = {ev.key: choose_camera(ev) for ev in events}
    pixels: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    pixel_maps: dict[str, dict[str, np.ndarray]] = {}
    for ev in events:                                   # what a viewer of each panel can see, from flat renders
        pixels[ev.key] = {}
        for name, frame in (("before", ev.before), ("after", ev.after)):
            pixels[ev.key][name], pixel_maps[f"{ev.key}_{name}"] = visible_pixels(ev, frame, cameras[ev.key][0])
    facts = {ev.key: describe(ev, cameras[ev.key][0], np.asarray(cameras[ev.key][1]["direction"]), pixels[ev.key]) for ev in events}

    # the study's per-event table against the values recomputed from the profiles that are drawn
    disagreements: list[str] = []
    for ev in events:
        f, e = facts[ev.key], evidence[ev.dataset][ev.kind]
        for mine, theirs in (("retention", "R_pre_to_post"), ("cosine", "cosine"), ("rel_change", "rel_change_Q"),
                             ("mean_capability_before", "Q_pre"), ("mean_capability_after", "Q_post")):
            if abs(f[mine] - f["events_csv"][theirs]) > 1e-6:
                disagreements.append(f"{ev.event_id}: {mine} recomputed {f[mine]:.6f}, events.csv {f['events_csv'][theirs]:.6f}")
        print(f"{DATASET_LABEL[ev.dataset]} {ev.kind}: event {ev.event_id}, {f['sequence_id']}, {object_phrase(ev)}, "
              f"frames {ev.before.t} -> {ev.after.t}")
        print(f"   retention {f['retention']:.4f}  cosine {f['cosine']:.4f}  relative change {f['rel_change']:+.4f}  "
              f"(events.csv: {f['events_csv']['R_pre_to_post']:.4f}, {f['events_csv']['cosine']:.4f}, {f['events_csv']['rel_change_Q']:+.4f})")
        print(f"   parts before: {'+'.join(f['parts_before'])}   after: {'+'.join(f['parts_after'])}   "
              f"added: {'+'.join(f['parts_added']) or 'none'}   removed: {'+'.join(f['parts_removed']) or 'none'}")
        print(f"   contact vertices before {f['contact_vertices_before']}  after {f['contact_vertices_after']}")
        rec = cameras[ev.key][1]
        print(f"   camera: rule '{rec['rule']}', candidate {rec['candidate']} of {rec['n_candidates']}, all-contact score {rec['score']:.4f} "
              f"({100 * rec['share_of_best_all_contact_score']:.0f}% of the best, candidate {rec['best_all_contact_candidate']})"
              + (f", score of the added part {rec['added_part_score']:.4f} (best {rec['best_added_part_score']:.4f}, at the best "
                 f"all-contact candidate {rec['added_part_score_at_best_all_contact_candidate']:.4f})" if rec["added_parts"] else ""))
        for name in ("before", "after"):
            print(f"   pixels of each patch, {name} [alone, on the object, clear of the hand]: "
                  + ", ".join(f"{p} {[n['patch_alone'], n['on_object'], n['clear_of_hand']]}" for p, n in f["contact_pixels"][name].items()))
        vis = f["contact_visibility"]
        print(f"   mostly on the far side: {vis['far_side'] or 'none'} (not visible at all: {vis['not_visible'] or 'none'})   "
              f"mostly under the hand: {vis['under_hand'] or 'none'}   place of the added patch: {f['added_patch_place'] or 'n/a'}")
        seen = f["contact_vertices_with_normal_facing_camera"]
        print(f"   contact vertices whose normal faces the camera [facing, all]: before {seen['before']}  after {seen['after']}")
        print(f"   directions in which the capability is lower after: {100 * f['share_of_directions_with_lower_capability_after']:.0f}%")
        print(f"   hand displacement between the frames: mean {f['hand_displacement_mm']['mean']:.1f} mm, "
              f"max {f['hand_displacement_mm']['max']:.1f} mm")
        for tag, c in f["cross_checks"].items():
            worst = max(c["max_vertex_mismatch_m"], c["max_hand_mismatch_m"], c["max_distance_mismatch_m"])
            print(f"   check, {tag}: geometry within {worst:.1e} m of qlib's copy, "
                  f"{100 * c['share_labels_equal_to_nearest_hand_vertex']:.1f}% of the part labels equal the nearest hand vertex's part")
        print(f"   rule: rank {e['chosen_rank']} of {e['n_candidates']} by {e['metric']} (median {e['median']:.4f})"
              + (f"; even count, {e['tie']}" if e["tie"] else ""))
    print("disagreements with events.csv:", disagreements or "none")

    row_labels = [row_label(ev) for ev in events]
    col_labels = ["before the event", "after the event", "modelled capability, before and after"]
    legend = [qlib.colorbar("parts", label="hand part. Strong colour: object surface within 1 cm of that part. "
                                           "Pale colour: the see-through hand itself.")]
    blank = np.full((PANEL[1], PANEL[0], 3), 255, np.uint8)
    font_px = qlib.grid([[blank] * 3] * len(events), row_labels, col_labels, legend).info["qlib_layout"]["font_px"]
    y_max = float(2 * np.ceil(max(max(ev.q_before.max(), ev.q_after.max()) for ev in events) / 2))

    rows, panels = [], {}
    for ev in events:
        camera = cameras[ev.key][0]
        image_before, panels[f"{ev.key}_before"] = render_frame(ev, ev.before, camera)
        image_after, panels[f"{ev.key}_after"] = render_frame(ev, ev.after, camera)
        for name in ("before", "after"):                # not drawn: the maps behind the pixel counts, for the verifier
            panels[f"{ev.key}_{name}"].update(pixel_maps[f"{ev.key}_{name}"])
        image_plot, panels[f"{ev.key}_profiles"] = profile_panel(ev, y_max, font_px)
        rows.append([image_before, image_after, image_plot])
    image = qlib.grid(rows, row_labels, col_labels, legend)

    text = caption(facts, evidence)
    records = {key: record for key, (_, record) in cameras.items()}
    objects = ", ".join(f"a {facts[ev.key]['object']}" for ev in events)
    meta = {
        "id": FIG_ID, "block": BLOCK,
        "title": "The hand on the object before and after an event that moves the contact, with the modelled capability",
        "panels": {
            "rows": {ev.key: f"{DATASET_LABEL[ev.dataset]}, event {ev.event_id}, {object_phrase(ev)}: "
                             + ("the event at the median retention of its class" if ev.kind == "typical" else
                                "the event at the median capability change among the events in which a hand part is added")
                     for ev in events},
            "columns": {
                "before the event": "object mesh, recorded hand and contact at the frame before the event (pre_frame of the study)",
                "after the event": "the same at the first frame after the event (post_frame), same camera and colours",
                "modelled capability, before and after": "the study's capability profile over its 76 directions before (grey) and after "
                                                         "(black), directions sorted by the value before, the same axes in all four plots"},
        },
        "examples": [{k: facts[ev.key][k] for k in ("dataset", "kind", "event_id", "example", "sequence_id", "object", "mesh_id", "hand", "split",
                                                    "frames_drawn", "take_frames_drawn")} for ev in events],
        "events": facts,
        "selection_rule": (
            "Typical event: among the persistent spatial events of the test split (the study's class: firm contact before, after and "
            "throughout the event), the one whose retention is the median of the class. With an even count no event sits at the median: "
            "the two middle events are equally far from it and the one with the smaller event id is taken. Event ids are compared as "
            "numbers, the example number first and then the number of the event within the example (910_4 before 1286_3), not as text. "
            "Event in which a part joins: among the events of the same class in which the set of touching hand parts gains a part, the "
            "one whose relative capability change is the median, with the same rule for an even count. Both are computed per dataset by "
            "this script from events.csv."),
        "selection_evidence": evidence,
        "display_choices": {
            "canonical_to_surface_rule": "not used, no canonical map is drawn. Contact is the study's own per-vertex rule: an object vertex "
                                         "within 1 cm of the recorded hand mesh, coloured by the hand part of the nearest hand vertex "
                                         "(labels read from the event file)",
            "colour_scales": {"hand parts": dict(zip(qlib.PART_NAMES, qlib.PART_HEX)), "no contact": qlib.ZERO,
                              "range": "categorical, no numeric range",
                              "note": "colours are interpolated between mesh vertices, so the outline of a patch follows the mesh triangles"},
            "hand_style": f"recorded MANO mesh at opacity {HAND_OPACITY}; each vertex is {100 * HAND_TINT:.0f}% the colour of its part and "
                          f"{100 * (1 - HAND_TINT):.0f}% skin {qlib.HAND_SKIN}",
            "camera": {"rule": f"One camera per event, the same for both frames, out of {N_VIEWS} directions spread evenly over the "
                               "sphere. The score of a direction over a set of contact vertices is the sum, over those of them that face "
                               "the camera, of the cosine between the direction and the surface normal (pointing to the hand); a vertex "
                               f"whose line of sight passes a hand vertex within {1000 * HAND_BALL:g} mm is seen through the hand and counts "
                               f"{1 - HAND_OPACITY:g}. Rows of typical events (rule 'all contact'): the direction with the highest score "
                               "over all contact vertices of both frames. Rows of events in which a part joins (rule 'the part that joins "
                               "first'): the direction with the highest score over the contact vertices of the joining part alone, the "
                               "all-contact score breaking ties, because the all-contact direction can leave the joining part on the far "
                               "side of the object. Up: the object's longest axis horizontal, the hand in the upper half. Fitted to the "
                               f"two hands and the contact vertices (margin {CAMERA_MARGIN}), so a larger object runs out of the panel.",
                       "per_event": records},
            "visibility": {"rule": "What a viewer can see of every part's contact patch is counted in pixels of flat renders (no lighting, "
                                   "no anti-aliasing) from the row's camera at the panel's size. A pixel belongs to a part's patch where "
                                   "the part's indicator (1 on its contact vertices, 0 elsewhere, interpolated like the colours) is above "
                                   "0.5. patch_alone: the patch drawn with nothing else; on_object: the patch on the whole object with "
                                   "the hand removed; clear_of_hand: of those, the pixels the hand is not in front of. With both frames "
                                   f"summed, a part is 'far_side' when on_object is under {HIDDEN_SHARE:g} of patch_alone ('not_visible' "
                                   f"when it is 0) and 'under_hand' when it is not far_side and clear_of_hand is under {HIDDEN_SHARE:g} "
                                   "of on_object.",
                           "counts": "events.<row>.contact_pixels and events.<row>.contact_visibility; the maps are in panels.npz "
                                     "(<row>_<before|after>/patch_on_object, patch_clear_of_hand, patch_alone_bits)",
                           "place_of_an_added_patch": "in the caption, 'at the left (right) end of the object' is computed: the patch lies "
                                                      "left (right) of the patches of all other parts in the picture and reaches within "
                                                      f"{100 * END_SHARE:g}% of the object's length of the end of the object on that side"},
            "plots": {"x": "the 76 directions of the study, sorted by the capability before the event, largest first",
                      "y": f"capability h(u) from 0 to {y_max:g} in the study's normalised unit (force, and torque divided by the object's "
                           "length); the same range in all four plots",
                      "before": BEFORE_COLOR, "after": AFTER_COLOR},
            "panel_px": list(PANEL), "panel_outline": f"{FRAME_PX} px in {qlib.RULE} around the rendered panels",
        },
        "numbers": numbers_for(events, facts, evidence, records),
        "inference": "none: recorded data and the saved analysis outputs of the study only; no checkpoint loaded, no device, no seed",
        "disagreements_with_study_tables": disagreements,
        "caveats": caveats_for(events, facts, evidence, records),
        "suggested_caption": text,
        "suggested_alt": (f"Four rows of pictures, one event each ({objects}): the object with a see-through hand on it before and after the "
                          "event, coloured patches on the object where each hand part touches, and a line plot of the modelled capability "
                          "before and after."),
    }
    paths = qlib.save(FIG_ID, image, panels, meta)
    print("layout:", image.info["qlib_layout"])
    print("caption:", text)
    print("caveats:", *meta["caveats"], sep="\n   - ")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
