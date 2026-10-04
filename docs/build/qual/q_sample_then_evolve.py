"""Qualitative figure "sample_then_evolve" for page block F1-B: one contact sequence on the real object.

One TACO test sequence (a right hand taking hold of a knife), drawn on the knife mesh in the knife's own
frame at six of its 64 frames. Rows, top to bottom:

    recorded      the recorded 512-point contact map, with the recorded hand as a ghost
    evolved       the true first map, evolved by the deterministic model            (preds.npz: gtinit_vf)
    sampled       a sampled first map, the best of ten picked with the ground truth,
                  evolved by the same model                                          (preds.npz: samplerG_vf__K10)
    held          the true first map held for all frames, the reference              (preds.npz: static_gt)

Nothing is computed by a model here: every map is read from the predictions the study saved
(``taco/50_hier_contact_gen/eval/fixed0/preds.npz``). The sequence is chosen by the rule of the dot strip this
figure replaces (the study's "high drift" example): the test sequence whose first and last recorded maps differ
most. The script recomputes that ranking and stops if it no longer gives the sequence documented on the page.

Each model row carries this sequence's own error from the study's per-sequence table (``per_example.csv``); the
same value is recomputed from the drawn arrays and both are stored in ``meta.json`` (key ``cross_checks``).

Run (CPU, software renderer; see README.md):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_sample_then_evolve.py
"""
from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import qlib

log = logging.getLogger("q_sample_then_evolve")

FIG_ID = "sample_then_evolve"
BLOCK = "F1-B"
TITLE = "Sample a first contact map, then evolve it: one sequence on the real object"
SCRIPT = "docs/build/qual/q_sample_then_evolve.py"
DATASET, DATASET_NAME = "taco", "TACO"
FRAMES: tuple[int, ...] = (0, 8, 16, 32, 48, 63)
PAGE_EXAMPLE = 3011            # the sequence of the dot strip this figure replaces (docs/evidence_manifest.json, F1-B images)
N_SAMPLES = 10                 # first maps sampled per sequence by the study; the best one is drawn
SPLIT, FOLD = "fixed", 0

STUDY = qlib.HIER[DATASET]
EVAL = STUDY / "eval" / f"{SPLIT}{FOLD}"
PREDS, TABLE = EVAL / "preds.npz", EVAL / "per_example.csv"
AGGREGATE, MANIFEST = STUDY / "results/aggregate.csv", STUDY / "manifests" / f"{SPLIT}{FOLD}.json"

PANEL = (480, 320)             # pixels (width, height) of one rendered panel
CAMERA_MARGIN = 0.04
CAMERA_TILT = 0.5              # how far the view is raised from the broad side's normal towards the object's middle axis
VMAX_STEP = 0.1                # the colour scale ends at the largest drawn value, rounded up to this step
NODATA_LEGEND_FRACTION = 0.005 # the "not reached" swatch is shown only if at least this share of the vertices is not reached
TOLERANCE = 1e-3               # allowed difference between a stored value and its recomputation (predictions are saved as float16)


@dataclass(frozen=True)
class Row:
    key: str                   # panel key in panels.npz
    array: str                 # array of preds.npz, (test sequences, 64, 512)
    model: str | None          # `model` column of per_example.csv; None for the recording
    k: int | None              # `K` column of per_example.csv
    label: str                 # row label in the figure
    what: str                  # the same in one sentence, for meta.json


ROWS: tuple[Row, ...] = (
    Row("recorded", "gt", None, None, "Recorded contact, with the recorded hand",
        "the recorded contact map of each frame, with the recorded hand drawn as a translucent grey ghost"),
    Row("evolved", "gtinit_vf", "gtinit_vf", 0, "True first map, evolved by the model",
        "the true first map, then the deterministic model's evolution over the following frames"),
    Row("sampled", f"samplerG_vf__K{N_SAMPLES}", "samplerG_vf", N_SAMPLES, "Sampled first map (best of ten), evolved by the same model",
        "a first map sampled from the object condition alone (the one of ten samples whose evolved sequence is closest "
        "to the recording, so picked with the ground truth), then the same deterministic evolution"),
    Row("held", "static_gt", "static_gt", 0, "True first map, held still",
        "the true first map repeated unchanged in every frame: the reference without any evolution"),
)
MODEL_ROWS = tuple(r for r in ROWS if r.model is not None)


def rel(path: Path) -> str:
    """Path relative to the result root, as the page's evidence files cite it."""
    return str(path.relative_to(qlib.RESULT_ROOT))


def number(label: str, value: Any, source: str, locator: str) -> dict[str, Any]:
    """One entry of meta.json's ``numbers``: a value a caption may quote, with where it comes from."""
    return {"label": label, "value": value, "dataset": DATASET_NAME, "source": source, "locator": locator}


def check(quantity: str, stored: float, stored_in: str, recomputed: float, tolerance: float = TOLERANCE) -> dict[str, Any]:
    """One entry of meta.json's ``cross_checks``: a stored value against its recomputation in this script."""
    difference = abs(float(stored) - float(recomputed))
    log.info("cross-check %-58s stored %.6f, recomputed %.6f -> %s", quantity, stored, recomputed,
             "agrees" if difference < tolerance else "DISAGREES")
    return {"quantity": quantity, "stored": float(stored), "stored_in": stored_in, "recomputed": float(recomputed),
            "difference": difference, "tolerance": tolerance, "agrees": bool(difference < tolerance)}


# ------------------------------------------------------------------------------ selection
def select_example(example: np.ndarray, recorded: np.ndarray) -> tuple[int, np.ndarray, dict[str, Any]]:
    """The rule of the figure this one replaces: the test sequence with the largest L2 distance between its first
    and its last recorded map. Returns (row in the prediction file, the metric of every test sequence, evidence)."""
    test = json.loads(MANIFEST.read_text(encoding="utf-8"))["test"]
    assert sorted(test) == sorted(example.tolist()), "the prediction file does not hold exactly the test sequences"
    change = np.linalg.norm(recorded[:, -1] - recorded[:, 0], axis=1)          # (test sequences,)
    order = np.argsort(-change, kind="stable")
    row = int(order[0])
    names = pd.read_csv(STUDY / "sequences_meta.csv", usecols=["sequence_id"], dtype=str).sequence_id   # row = example index
    evidence = {
        "metric": "L2 distance between the last and the first recorded 512-point map of the sequence, ||C_63 - C_0||",
        "computed_from": f"{rel(PREDS)}, arrays gt and example",
        "candidates": int(len(example)),
        "candidate_set": f"all test sequences of the study's fixed split ({rel(MANIFEST)}, key test; checked equal to the rows of the prediction file)",
        "chosen": {"example": int(example[row]), "rank": 1, "value": float(change[row])},
        "top_10": [{"rank": r + 1, "example": int(example[i]), "sequence_id": str(names.iloc[int(example[i])]), "value": float(change[i])}
                   for r, i in enumerate(order[:10])],
        "test_set": {"min": float(change.min()), "median": float(np.median(change)), "mean": float(change.mean()), "max": float(change.max())},
        "same_rule_as": "scripts/research/hier_contact_gen/figures.py, fig6 'high_drift' (np.argmax of this metric), whose output "
                        "taco/50_hier_contact_gen/figures/fig6_qual_high_drift_fixed.png is the dot strip now on the page",
        "page_example": PAGE_EXAMPLE,
    }
    return row, change, evidence


# ------------------------------------------------------------------------------ this sequence's numbers
def table_row(table: pd.DataFrame, model: str, k: int, example: int) -> pd.Series:
    """The single row of per_example.csv for (model, K, example)."""
    hit = table[(table.model == model) & (table.K == k) & (table.example == example)]
    if len(hit) != 1:
        raise ValueError(f"expected one row for model={model}, K={k}, example={example}, found {len(hit)}")
    return hit.iloc[0]


def rank_of(values: pd.Series, example: int, descending: bool = False) -> int:
    """1-based rank of ``example`` among the test sequences (ascending unless ``descending``)."""
    v = values.loc[example]
    return int(((values > v) if descending else (values < v)).sum()) + 1


def sequence_numbers(maps: dict[str, np.ndarray], first_samples: np.ndarray, example: np.ndarray, row: int,
                     change: np.ndarray) -> tuple[dict[str, float], list[dict], list[dict], dict[str, Any]]:
    """Errors of this sequence from the study's tables, each recomputed from the arrays that are drawn.

    Returns (error per model row as printed in the figure, entries for ``numbers``, entries for ``cross_checks``,
    a few values the caveats quote)."""
    n, n_test = int(example[row]), len(example)
    table = pd.read_csv(TABLE)
    table = table[(table.split == SPLIT) & (table.fold == FOLD)]
    aggregate = pd.read_csv(AGGREGATE)
    aggregate = aggregate[aggregate.split == SPLIT]
    errors: dict[str, float] = {}
    test_means: dict[str, float] = {}
    numbers: list[dict] = []
    checks: list[dict] = []
    for r in MODEL_ROWS:
        line = table_row(table, r.model, r.k, n)
        per_frame = np.linalg.norm(maps[r.key] - maps["recorded"], axis=1)     # (64,) L2 over the 512 points
        errors[r.key] = float(line.E_C)
        column = table[(table.model == r.model) & (table.K == r.k)].set_index("example").E_C.reindex(example)
        mean_row = aggregate[(aggregate.model == r.model) & (aggregate.K == r.k)]
        assert len(mean_row) == 1 and int(mean_row.n_examples.iloc[0]) == n_test
        test_means[r.key] = float(mean_row.E_C.iloc[0])
        where = f"model={r.model}, K={r.k}"
        numbers += [
            number(f"error of the row '{r.label}' for this sequence: mean over the 64 frames of the L2 distance between its "
                   "512-point map and the recorded one (E_C)", float(line.E_C), rel(TABLE), f"row {where}, example={n}, split={SPLIT}; column E_C"),
            number(f"the same error recomputed from the drawn arrays, row '{r.label}'", float(per_frame.mean()), rel(PREDS),
                   f"mean over the 64 frames of ||{r.array}[{row}] - gt[{row}]||_2 (row {row} = example {n})"),
            number(f"L2 distance to the recorded map at the drawn frames {list(FRAMES)}, row '{r.label}'",
                   [float(per_frame[t]) for t in FRAMES], rel(PREDS), f"||{r.array}[{row}, t] - gt[{row}, t]||_2 for t in {list(FRAMES)}"),
            number(f"rank of this sequence's error among the {n_test} test sequences (1 = lowest), row '{r.label}'", rank_of(column, n),
                   rel(TABLE), f"rows {where}, split={SPLIT}; column E_C; position of example {n}"),
            number(f"mean of the same error over the {n_test} test sequences (context, not drawn), row '{r.label}'", test_means[r.key],
                   rel(AGGREGATE), f"row split={SPLIT}, {where}; column E_C"),
        ]
        checks.append(check(f"E_C of this sequence, {where}", line.E_C, rel(TABLE), per_frame.mean()))

    # the sampled row: which of the ten samples it is, and that its first frame is that sample
    sampled = table_row(table, "samplerG_vf", N_SAMPLES, n)
    k_star = int(sampled.k_star)
    assert np.array_equal(maps["sampled"][0], first_samples[k_star]), "frame 0 of the sampled row is not the saved sample k_star"
    first_errors = np.linalg.norm(first_samples - maps["recorded"][0], axis=1)  # (10,) first-map error of every sample
    checks.append(check("first-map error of the drawn sample (s0_err)", sampled.s0_err, rel(TABLE), first_errors[k_star]))
    single = table_row(table, "samplerG_vf", 1, n)
    by_model = {m: table[(table.model == m) & (table.K == 0)].set_index("example").E_C.reindex(example) for m in ("static_gt", "gtinit_vf")}
    gain = by_model["static_gt"] - by_model["gtinit_vf"]                        # what evolving the true first map gains over holding it
    here = f"example={n}"
    numbers += [
        number("index (0 to 9) of the drawn sample among the ten sampled first maps", k_star, rel(TABLE),
               f"row model=samplerG_vf, K={N_SAMPLES}, {here}; column k_star"),
        number("L2 distance between the drawn sampled first map and the true first map", float(sampled.s0_err), rel(TABLE),
               f"row model=samplerG_vf, K={N_SAMPLES}, {here}; column s0_err"),
        number("L2 distance to the true first map of each of the ten sampled first maps (sample 0 to 9)", [float(v) for v in first_errors],
               rel(PREDS), f"||S0_G[{row}, k] - gt[{row}, 0]||_2, k = 0..9"),
        number("error of this sequence when sample 0 is evolved instead of the best of ten (not drawn)", float(single.E_C), rel(TABLE),
               f"row model=samplerG_vf, K=1, {here}; column E_C"),
        number("number of first maps sampled, of which the best is drawn", N_SAMPLES, rel(PREDS), "array S0_G, second dimension"),
        number("selection metric of this sequence: L2 distance between its last and first recorded map", float(change[row]), rel(PREDS),
               f"||gt[{row}, 63] - gt[{row}, 0]||_2"),
        number("rank of this sequence by the selection metric (1 = largest)", 1, rel(PREDS), "argsort of ||gt[:, 63] - gt[:, 0]||_2 over all rows"),
        number("number of test sequences the selection ranks", n_test, rel(PREDS), "length of array example"),
        number("median of the selection metric over the test sequences", float(np.median(change)), rel(PREDS), "median of ||gt[:, 63] - gt[:, 0]||_2"),
        number("error of holding minus error of evolving the true first map, this sequence", float(gain.loc[n]), rel(TABLE),
               f"E_C(static_gt, K=0) - E_C(gtinit_vf, K=0), {here}"),
        number(f"rank of that difference among the {n_test} test sequences (1 = largest gain from evolving)", rank_of(gain, n, descending=True),
               rel(TABLE), f"E_C(static_gt, K=0) - E_C(gtinit_vf, K=0) per example, split={SPLIT}"),
        number("frames per sequence", int(maps["recorded"].shape[0]), rel(PREDS), "array gt, second dimension"),
        number("canonical points per contact map", int(maps["recorded"].shape[1]), rel(PREDS), "array gt, third dimension"),
        number("frames drawn", list(FRAMES), SCRIPT, "constant FRAMES"),
        number("length scale of the contact value exp(-distance / scale), in cm", qlib.SOFT_SCALE * 100, "docs/build/qual/qlib.py",
               "constant SOFT_SCALE (the studies' definition of soft contact)"),
    ]
    context = {"k_star": k_star, "single_sample_error": float(single.E_C), "gain": float(gain.loc[n]),
               "gain_rank": rank_of(gain, n, descending=True), "test_means": test_means}
    return errors, numbers, checks, context


# ------------------------------------------------------------------------------ camera
def far_side_view(object_verts: np.ndarray, hand_verts: np.ndarray, tilt: float = CAMERA_TILT) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """A (direction, up) pair for qlib.fit_camera, and the numbers it rests on: look onto the object's broad side
    (along its thinnest principal axis) from the side AWAY from the hand, raised by ``tilt`` towards the middle axis
    on the hand's side, with the longest axis horizontal.

    qlib.hand_side_view looks from the hand's side; there the ghost of a hand that is closed around a handle lies
    in front of the contact and dims its colours. From the far side the object is in front of the hand."""
    centre = object_verts.mean(0)
    _, axes = np.linalg.eigh(np.cov((object_verts - centre).T))                 # columns: thinnest, middle, longest axis
    to_hand = hand_verts.mean(0) - centre
    thin, mid = (a if a @ to_hand >= 0 else -a for a in (axes[:, 0], axes[:, 1]))
    direction = -thin + tilt * mid
    direction /= np.linalg.norm(direction)
    up = mid - (mid @ direction) * direction
    offsets = {"hand_centre_along_thin_axis_mm": float(to_hand @ thin * 1000), "hand_centre_along_middle_axis_mm": float(to_hand @ mid * 1000)}
    return direction, up / np.linalg.norm(up), offsets


# ------------------------------------------------------------------------------ figure
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    qlib.require_headless()

    # ---- saved predictions and the selection rule
    with np.load(PREDS) as z:
        example = z["example"]
        row, change, evidence = select_example(example, z["gt"].astype(np.float32))
        maps = {r.key: z[r.array][row].astype(np.float32) for r in ROWS}       # each (64, 512)
        first_samples = z["S0_G"][row].astype(np.float32)                      # (10, 512) the ten sampled first maps
    n, n_test, n_frames = int(example[row]), len(example), maps["recorded"].shape[0]
    if n != PAGE_EXAMPLE:
        raise RuntimeError(f"the selection rule gives example {n}, not the documented page example {PAGE_EXAMPLE}")
    ex = qlib.load_example(DATASET, n)
    assert ex.meta.split == "test" and ex.test_row(example) == row
    assert np.array_equal(maps["recorded"], ex.C), "the saved ground truth is not the study's recorded sequence"
    log.info("selected example %d (%s, %s hand): first-to-last difference %.3f, rank 1 of %d", n, ex.meta.sequence_id,
             qlib.SIDE[ex.meta.hand], change[row], n_test)
    errors, numbers, checks, context = sequence_numbers(maps, first_samples, example, row, change)

    # ---- geometry (the knife is fixed in its own frame, the recorded hand moves) and the values that are drawn
    verts, faces = ex.mesh(0)
    hands = {t: ex.hand(t)[0] for t in FRAMES}
    hand_faces = ex.hand(FRAMES[0])[1]
    gap_mm = [float(cKDTree(hands[t]).query(verts)[0].min() * 1000) for t in FRAMES]         # hand mesh to knife mesh
    cached_mm = [float(-qlib.SOFT_SCALE * np.log(ex.dense(t).max()) * 1000) for t in FRAMES]  # the study's cached distances
    for t, a, b in zip(FRAMES, cached_mm, gap_mm):
        checks.append(check(f"smallest hand-to-knife distance at frame {t}, mm", a, "dense distances of the take (qlib.Example.dense)", b))
    values = {r.key: ex.to_vertices(maps[r.key][list(FRAMES)]) for r in ROWS}                 # each (6, V), NaN where not reached
    drawn_min = min(float(np.nanmin(v)) for v in values.values())
    drawn_max = max(float(np.nanmax(v)) for v in values.values())
    vmax = math.ceil(drawn_max / VMAX_STEP - 1e-9) * VMAX_STEP
    unreached = int((~ex.covered).sum())
    numbers += [
        number(f"smallest distance between the recorded hand mesh and the knife mesh at frames {list(FRAMES)}, in mm", gap_mm,
               "recorded MANO hand and object pose of the take (qlib.Example.hand, .mesh)",
               f"example {n}, minimum over knife vertices of the distance to the nearest hand vertex"),
        number("lower end of the colour scale (no contact)", 0.0, SCRIPT, "first argument of qlib.colorbar / qlib.contact_rgb"),
        number("upper end of the colour scale (largest drawn surface value, rounded up to 0.1)", vmax, SCRIPT,
               f"largest drawn value {drawn_max:.4f} over the {len(ROWS) * len(FRAMES)} panels"),
        number("vertices of the knife mesh", int(len(verts)), "taco/30_bimart_gen3_scene_scale/assets/taco_mesh_dict.npy", f"mesh {ex.meta.mesh_id}, verts_original"),
        number("knife vertices that the 512-point map does not reach (pale violet if visible)", unreached, SCRIPT, "qlib.Example.covered"),
    ]

    # ---- render: one camera and one colour scale for all panels
    direction, up, hand_offsets = far_side_view(verts, hands[FRAMES[-1]])
    camera = qlib.fit_camera([verts, *hands.values()], direction, up, margin=CAMERA_MARGIN, aspect=PANEL[0] / PANEL[1])
    images: list[list[np.ndarray]] = []
    panels: dict[str, Any] = {
        "shared": {"mesh_verts": verts, "mesh_faces": faces, "hand_faces": hand_faces, "reached": ex.covered, "frames": np.array(FRAMES)},
        "selection": {"example": example, "first_to_last_difference": change},
    }
    for r in ROWS:
        images.append([])
        for j, t in enumerate(FRAMES):
            items = [qlib.mesh_item(verts, faces, rgb=qlib.contact_rgb(values[r.key][j], vmax))]
            drawn = {"vertex_values": values[r.key][j], "map_512": maps[r.key][t]}
            if r.model is None:                                                 # only the recording has a hand
                items.append(qlib.hand_item(hands[t], hand_faces, **qlib.GHOST))
                drawn["hand_verts"] = hands[t]
            images[-1].append(qlib.render(items, camera, PANEL))
            panels[f"{r.key}_t{t:02d}"] = drawn
        log.info("rendered row %s", r.key)
    row_labels = [r.label if r.model is None else f"{r.label}\nerror {errors[r.key]:.2f}" for r in ROWS]
    legend = qlib.colorbar("contact", 0.0, vmax, "contact, exp(\N{MINUS SIGN}distance to the hand / 2 cm)",
                           nodata="not reached by the map" if unreached / len(verts) >= NODATA_LEGEND_FRACTION else None)
    image = qlib.grid(images, row_labels, [f"frame {t}" for t in FRAMES], colorbars=[legend])

    # ---- caption and record
    e = {k: f"{v:.2f}" for k, v in errors.items()}
    caption = (
        f"One {DATASET_NAME} test sequence, a right hand taking hold of a knife, drawn on the knife's surface at frames "
        f"{', '.join(str(t) for t in FRAMES[:-1])} and {FRAMES[-1]} of its {n_frames}; it was selected as the one of the {n_test} test "
        "sequences whose first and last recorded contact maps differ most, so a case where the evolution has the most to do and not a "
        "typical one. "
        "Top row: the recorded contact, with the recorded hand as a pale ghost; the contact starts at the corner of the blade next to the "
        "handle and ends on the whole handle. "
        "Below it: the true first map evolved by the deterministic model, then a sampled first map (the best of ten samples, picked with "
        "the ground truth) evolved by the same model, then the true first map simply held. "
        f"The error in each label is this sequence's distance to the recorded maps, averaged over all {n_frames} frames: {e['evolved']} "
        f"evolved from the true first map, {e['sampled']} evolved from the sampled one, {e['held']} when held. "
        "Colour is the contact value, highest where the hand touches and falling off with the distance to the hand, on one scale from "
        f"0 to {vmax:g} in every panel."
    )
    alt = ("A knife drawn six times per row over the course of one recorded sequence, coloured by contact: in the recording and in the two "
           "evolved rows the handle turns from pale to dark red as the hand closes on it, while the held row stays unchanged.")
    meta = {
        "id": FIG_ID,
        "block": BLOCK,
        "title": TITLE,
        "replaces": "docs/assets/img/f1b_taco_knife_rollout.png (the dot strip of the same sequence on the 512 canonical points)",
        "script": SCRIPT,
        "command": qlib.RUN_COMMAND.replace("<script>", SCRIPT),
        "panels": {
            "rows": [{"key": r.key, "label": label.replace("\n", ", "), "shows": r.what, "source_array": f"{rel(PREDS)}: {r.array}"}
                     for r, label in zip(ROWS, row_labels)],
            "columns": [f"frame {t} of the {n_frames}-frame sequence (take frame {ex.take_frame(t)})" for t in FRAMES],
            "note": "Every panel shows the same knife from the same camera with the same colour scale; the knife is fixed because "
                    "everything is drawn in the knife's own frame. Column 'frame 0' of the third row is the sampled first map itself.",
        },
        "examples": [{"dataset": DATASET_NAME, "example": n, "sequence_id": ex.meta.sequence_id,
                      "object": f"{ex.meta.category} (mesh {ex.meta.mesh_id}, the {ex.meta.role} of the take)", "hand": qlib.SIDE[ex.meta.hand],
                      "frames": list(FRAMES), "take_frames": [ex.take_frame(t) for t in FRAMES], "split": ex.meta.split}],
        "selection_rule": "The same documented sequence as the dot strip this figure replaces: among all test sequences of the TACO fixed split, "
                          "the one with the largest L2 distance between its first and its last recorded contact map, a case where the evolution "
                          "has the most to do. The script recomputes the ranking and stops unless it gives that sequence.",
        "selection_evidence": evidence,
        "display_choices": {
            "canonical_to_surface": f"{qlib.DISPLAY_RULE}. Applied to every row, the recording included (the recorded row is the study's 512-point "
                                    f"map, not the per-vertex recording), so rows are comparable. {unreached} of {len(verts)} knife vertices are "
                                    f"not reached by the map and would be drawn in pale violet ({qlib.NO_DATA}); the legend swatch is left out "
                                    f"below a share of {NODATA_LEGEND_FRACTION:g}.",
            "colour_scale": {"function": "qlib.contact_rgb", "range": [0.0, vmax],
                             "stops": {"0": qlib.ZERO, f"{vmax / 2:g}": qlib.ORANGE, f"{vmax:g}": qlib.DARK},
                             "rule": f"upper end = largest drawn surface value ({drawn_max:.4f}) rounded up to {VMAX_STEP}; one scale for all "
                                     f"{len(ROWS) * len(FRAMES)} panels",
                             "drawn_value_range": [drawn_min, drawn_max],
                             "clipping": f"model outputs below 0 (lowest drawn value {drawn_min:.4f}) are drawn as 0; nothing exceeds the upper end"},
            "camera": {"rule": "far_side_view (this script): onto the knife's broad side from the side away from the recorded hand at the last "
                               f"drawn frame, raised by {CAMERA_TILT} towards the knife's middle axis on the hand's side, longest axis horizontal; "
                               "fitted once on the knife and the recorded hand at all six frames and reused for every panel",
                       "why": "from the hand's own side (qlib.hand_side_view) the ghost of the closed hand lies over the handle and dims the "
                              "recorded contact exactly where the rows are compared; from the far side the handle is in front of the hand",
                       "sides_decided_by": hand_offsets, "margin": CAMERA_MARGIN, "panel_px": list(PANEL), **dataclasses.asdict(camera)},
            "hand": f"recorded MANO hand, top row only, as a ghost ({qlib.GHOST['color']} at {qlib.GHOST['opacity']:.0%} opacity, front faces only), "
                    "mostly behind the knife in this view; the other rows have no hand because the models output contact maps, not hands",
            "frame": "the knife's own rigid frame, metres: the knife stands still and the hand moves",
        },
        "numbers": numbers,
        "cross_checks": {"note": "values stored by the studies against their recomputation here from the arrays that are drawn "
                                 "(saved predictions are float16) and from the recorded hand mesh",
                         "checks": checks, "all_agree": bool(all(c["agrees"] for c in checks))},
        "inference": {
            "newly_computed": "nothing; no model was run and no checkpoint was loaded",
            "checkpoints_loaded": [],
            "device": "CPU (rendering only)",
            "seed": "none in this script",
            "read_from_saved_predictions": {
                "file": rel(PREDS), "arrays": ["example", "gt", "S0_G"] + [r.array for r in MODEL_ROWS], "row": row,
                "produced_by": "the study's evaluation, scripts/research/hier_contact_gen/rollout_eval.py --split fixed --fold 0 --vf-tag _unroll8 "
                               "(log: reports/hier_logs/taco/eval.log), with hier_contact_gen_ckpt/taco/fixed0/sampler_G.pt and vf_unroll8.pt; "
                               f"sample k of the first-map sampler was drawn with generator seed 100000 + k (here k = {context['k_star']})",
            },
        },
        "caveats": [
            f"One sequence, and by construction not a typical one: it has the largest first-to-last difference of the {n_test} test sequences "
            f"({change[row]:.2f} against a test median of {np.median(change):.2f}). Its gain from evolving over holding ({context['gain']:.2f}) "
            f"is rank {context['gain_rank']} of {n_test}, and its evolved error ({e['evolved']}) is below the test mean "
            f"({context['test_means']['evolved']:.2f}), so the evolution looks better here than it does on average.",
            "'Best of ten' is chosen with the ground truth (the sample whose evolved sequence has the lowest error), so the third row is an "
            f"oracle pick, not what one draw gives: evolving sample 0 of the same ten gives {context['single_sample_error']:.2f} for this "
            "sequence. The other nine samples are not drawn.",
            "The contact value is a soft proximity, exp(-distance / 2 cm), so it is above zero before the hand touches: in the first two drawn "
            f"frames the recorded hand is about {gap_mm[0]:.0f} mm from the knife and the map already shows contact at the blade corner.",
            f"The 512-point maps are smoothed onto the mesh vertices for display, and model outputs below zero (down to {drawn_min:.3f}) are "
            f"drawn as zero. The errors in the row labels are computed on the 512 points over all {n_frames} frames, not on the six drawn surfaces.",
            "The models output contact maps only. The hand is the recorded one and appears only in the top row. It is drawn as a pale ghost, "
            "mostly behind the knife; where its fingers reach around the handle towards the viewer the colours beneath look slightly lighter "
            "than the same value in the rows without a hand.",
            "The knife is drawn in its own frame, so the object motion that conditions the evolution is not visible.",
            "TACO only; the figure says nothing about ARCTIC or OakInk2.",
        ],
        "suggested_caption": caption,
        "suggested_alt": alt,
    }
    paths = qlib.save(FIG_ID, image, panels, meta)
    for name, path in paths.items():
        log.info("%s: %s", name, path)
    log.info("layout: %s", image.info["qlib_layout"])


if __name__ == "__main__":
    main()
