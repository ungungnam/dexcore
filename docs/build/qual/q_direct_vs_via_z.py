"""Qualitative figure "direct_vs_via_z" for page blocks F3-C and F3-diagA: direct generation, generation
via z, and the z of the true frames through the same decoder, on the real object.

One test sequence per dataset (TACO, ARCTIC; never mixed), drawn on the object mesh in the object's own frame
at six of its 64 frames. Rows of each dataset block, top to bottom:

    recorded + hand   the recorded 512-point contact map, with the recorded hand as a ghost
    recorded          the same map without the hand: the ghost mutes the colours under it, so this row
                      is the reference the rows below are compared with
    direct            the direct generator: first map, object and object motion -> the 63 later maps
                      (saved prediction, preds/B0_seed0_A.npz)
    via z, predicted  the generator that predicts the latent z of every later frame and decodes it
                      (saved prediction, preds/B1_seed0_A.npz)
    via z, true       the same decoder fed the z of the TRUE frames, an analysis-only input that no
                      generator has (decoded here on CPU; the study saved only the error of this path)

The sequence is chosen by a rule the script computes: the test sequence at the median dense error of the
generator via z. The dense error of a row is the L2 distance between its 512-value map and the recorded one,
averaged over frames 1 to 63 (the study's E_C); it is recomputed here from the arrays that are drawn, printed
in the row labels next to the test-set mean of the study's tables, and cross-checked against the study's
per-sequence table.

Frame 0 is the given first map: the two generator rows show the recorded map there, the last row the decoder's
output for the true z of frame 0 (not part of the error). One camera per dataset serves every row and frame; its
direction is the one, on the hand's side, from which the most recorded contact faces the camera (contact_view).
One colour scale serves the whole figure.

Inference. Only the last row is computed: the decoder of the via-z generator is loaded read-only from its
selected checkpoint with the study's own model class, and fed the cached latents of the true frames. The
decoder's inputs (object descriptor, per-point geometry) are rebuilt from the caches as the study's data
classes build them; decoding the SAVED predicted latents this way must reproduce the saved prediction, which
the script checks before it draws anything. Nothing is trained and no file of the study is written.

Run (CPU, software renderer; see README.md):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_direct_vs_via_z.py
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree

import qlib

log = logging.getLogger("q_direct_vs_via_z")

FIG_ID = "direct_vs_via_z"
BLOCK = "F3-C, F3-diagA"
TITLE = "Direct generation, generation via z, and the true z through the same decoder: one sequence per dataset"
DATASETS: tuple[str, ...] = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
FRAMES: tuple[int, ...] = (0, 8, 16, 32, 48, 63)
T = qlib.T                     # frames per sequence; frame 0 is given, frames 1..63 are generated

STUDY = qlib.RESULT_ROOT / "reports/contact_latent_temporal_stage2"
STUDY_CODE = qlib.REPO / "scripts/research/contact_latent_temporal_stage2"
CKPT_ROOT = Path(os.environ.get("DEXCORE_CKPT_ROOT", "/ckpt/uhnam/dexcore")) / "contact_latent_temporal_stage2"
GEOMETRY_CACHE = "reports/structure_aware_temporal_generation/{dataset}/cache/canonical_geometry.npz"
DIRECT, VIA_Z = "B0_seed0", "B1_seed0"           # the study's run names (file names only; never shown in the figure)
MAIN_TABLE, TRUE_Z_TABLE = STUDY / "main_metrics.csv", STUDY / "bottleneck_metrics.csv"

PANEL = (400, 300)             # pixels (width, height) of one rendered panel
CAMERA_MARGIN = 0.04
N_DIRECTIONS = 2000            # candidate view directions (Fibonacci sphere)
VMAX_STEP = 0.1                # the colour scale ends at the largest drawn value, rounded up to this step
NODATA_LEGEND_FRACTION = 0.005 # the "not reached" swatch is shown if at least this share of a drawn mesh is not reached
TABLE_TOLERANCE = 1e-3         # allowed difference between a table value and its recomputation (predictions are saved as float16)
DECODE_TOLERANCE = 0.02        # allowed |re-decoded saved z - saved prediction| per map value (CPU float32 against the study's
                               # GPU bfloat16 pass; the scouting pass measured 0.006)
N_POINT_CHANNELS = 10          # per-point decoder input: position (3), normal (3), nearest, coverage, cell area, valid


@dataclass(frozen=True)
class Row:
    key: str                   # panel key in panels.npz
    maps: str                  # which map sequence the row draws (key of Block.maps)
    label: str                 # row label in the figure (plain words)
    what: str                  # the same in one sentence, for meta.json
    hand: bool = False         # draw the recorded hand as a ghost
    table: Path | None = None  # table that holds the test-set mean of this row's dense error
    locator: Mapping[str, str] | None = None     # row of that table (the value column is `column`)
    column: str | None = None


ROWS: tuple[Row, ...] = (
    Row("recorded_with_hand", "recorded", "Recorded contact, with the recorded hand",
        "the recorded contact map of each frame, with the recorded hand drawn as a translucent grey ghost (the ghost mutes the "
        "colours under it)", hand=True),
    Row("recorded", "recorded", "Recorded contact alone",
        "the same recorded contact map without the hand: the reference for the rows below"),
    Row("direct", "direct", "Direct generator",
        "the direct generator: from the first recorded map, the object and its motion it outputs the 63 later maps directly "
        "(saved prediction)", table=MAIN_TABLE, locator={"model": "B0", "metric": "E_C"}, column="value"),
    Row("via_z_predicted", "via_z_predicted", "Via z, with the predicted z",
        "the generator via z: from the same inputs it predicts the latent z of each later frame, and its decoder turns each z "
        "into a map (saved prediction)", table=MAIN_TABLE, locator={"model": "B1", "metric": "E_C"}, column="value"),
    Row("via_z_true", "via_z_true", "Via z, with the z of the true frames (analysis only)",
        "the decoder of the generator via z, fed the z that the study's encoder computed from the TRUE map of each frame: an "
        "analysis-only input that no generator has (decoded on CPU by this script)", table=TRUE_Z_TABLE, locator={"model": "B1"},
        column="E_C_oracle_z"),
)
SCORED: tuple[Row, ...] = tuple(row for row in ROWS if row.table is not None)      # the rows that carry a dense error


def rel(path: Path) -> str:
    """Path relative to the result root, as the page's evidence files cite it."""
    return str(path.relative_to(qlib.RESULT_ROOT))


def dense_error(maps: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    """The study's dense error E_C: maps, recorded (..., 64, 512) -> (...,), the L2 distance over the 512 values,
    averaged over frames 1..63 (frame 0 is given, never generated)."""
    return np.linalg.norm(maps[..., 1:, :] - recorded[..., 1:, :], axis=-1).mean(-1)


def frame_error(maps: np.ndarray, recorded: np.ndarray) -> np.ndarray:
    """(64, 512), (64, 512) -> (64,) L2 distance of each frame's map from the recorded one."""
    return np.linalg.norm(maps - recorded, axis=-1)


# ------------------------------------------------------------------------------ selection
@dataclass(frozen=True)
class Selection:
    example: int               # row of sequences_meta.csv
    row: int                   # row of the prediction files
    examples: np.ndarray       # (N,) example index of every row of the prediction files (the test sequences)
    recorded: np.ndarray       # (64, 512) recorded maps of the sequence
    direct: np.ndarray         # (64, 512) saved prediction of the direct generator
    via_z: np.ndarray          # (64, 512) saved prediction of the generator via z
    z_predicted: np.ndarray    # (63, 64) its predicted latents of frames 1..63, standardised
    test_mean: dict[str, float]          # row key -> mean dense error over the test sequences, recomputed from the saved predictions
    evidence: dict[str, Any]

    @property
    def n_test(self) -> int:
        return len(self.examples)


def prediction_file(dataset: str, run: str) -> Path:
    return STUDY / dataset / "preds" / f"{run}_A.npz"


def select_example(dataset: str) -> Selection:
    """The rule: the test sequence at the median dense error of the generator via z. Sequences are sorted by that
    error (ties: smaller example index); an even count has two middle sequences, equally close to the median, and
    the one with the smaller example index is taken."""
    with np.load(prediction_file(dataset, DIRECT)) as z:
        example, direct = z["example"], z["pred"][:, 0].astype(np.float32)             # (N,), (N, 64, 512)
    with np.load(prediction_file(dataset, VIA_Z)) as z:
        assert np.array_equal(z["example"], example), "the two prediction files list different sequences"
        via_z, z_predicted = z["pred"][:, 0].astype(np.float32), z["z_hat"]
    tables = qlib._tables(dataset)
    assert (tables["meta"].split_set.values[example] == "test").all(), "a prediction is not of a test sequence"
    assert len(example) == int((tables["meta"].split_set.values == "test").sum()), "the prediction files do not cover the test set"
    recorded = tables["C"][example].astype(np.float32)                                 # (N, 64, 512)
    error = {"direct": dense_error(direct, recorded).astype(np.float64), "via_z_predicted": dense_error(via_z, recorded).astype(np.float64)}
    metric = error["via_z_predicted"]
    n = len(metric)
    order = np.lexsort((example, metric))                                              # ascending error, ties: smaller example index
    middle = order[[n // 2]] if n % 2 else order[n // 2 - 1:n // 2 + 1]
    row = int(middle[np.argmin(example[middle])])
    rank = int(np.flatnonzero(order == row)[0]) + 1
    meta = tables["meta"]
    names, chosen, train = meta.sequence_id, meta.iloc[int(example[row])], meta[meta.split_set == "train"]
    lower = error["via_z_predicted"] < error["direct"]
    evidence = {
        "metric": "dense error of the generator via z: L2 distance between its 512-value map and the recorded map, averaged over "
                  "frames 1 to 63",
        "computed_from": f"{rel(prediction_file(dataset, VIA_Z))} (arrays pred, example) against the recorded maps "
                         f"{rel(qlib.HIER[dataset] / 'sequences.npz')} (array C)",
        "candidates": n,
        "candidate_set": "all test sequences of the study's fixed split (the sequences of the prediction file)",
        "median": float(np.median(metric)),
        "tie_rule": "the count is even, so the two middle sequences are equally close to the median; the smaller example index "
                    "is taken" if n % 2 == 0 else "the count is odd: one middle sequence",
        "middle": [{"rank": int(np.flatnonzero(order == i)[0]) + 1, "example": int(example[i]), "sequence_id": str(names.iloc[int(example[i])]),
                    "value": float(metric[i])} for i in middle],
        "chosen": {"example": int(example[row]), "rank": rank, "value": float(metric[row]),
                   "distance_to_median": float(abs(metric[row] - np.median(metric)))},
        "test_set": {"min": float(metric.min()), "median": float(np.median(metric)), "mean": float(metric.mean()), "max": float(metric.max())},
        "sorted": {"rank_1_is": "smallest error", "example": example[order].tolist(), "value": [round(float(v), 6) for v in metric[order]]},
        "typicality": {
            "direct_error_of_chosen": float(error["direct"][row]),
            "via_z_minus_direct_of_chosen": float(metric[row] - error["direct"][row]),
            "via_z_minus_direct_test_mean": float((metric - error["direct"]).mean()),
            "sequences_where_via_z_is_lower": int(lower.sum()),
            "share_where_via_z_is_lower": float(lower.mean()),
            "training_sequences": int(len(train)),
            "training_sequences_with_this_mesh": int((train.mesh_id == chosen.mesh_id).sum()),
            "training_sequences_of_this_category": int((train.category == chosen.category).sum()),
            "subject": str(chosen.subject),
            "subject_in_training": bool((train.subject == chosen.subject).any()),
            "test_subjects_absent_from_training": sorted(set(meta.subject.values[example]) - set(train.subject)),
        },
    }
    return Selection(int(example[row]), row, example, recorded[row], direct[row], via_z[row], z_predicted[row].astype(np.float32),
                     {k: float(v.mean()) for k, v in error.items()}, evidence)


# ------------------------------------------------------------------------------ the decoder of the generator via z, on CPU
def checkpoint_path(dataset: str) -> Path:
    return CKPT_ROOT / dataset / f"{VIA_Z}.pt"


def md5_head(path: Path, n: int = 12) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()[:n]


def load_generator(dataset: str) -> tuple[Any, dict[str, Any]]:
    """The generator via z in its selected (best validation) state, on CPU, built with the study's model class.
    Returns (network in eval mode, checkpoint dict). The checkpoint is opened read-only."""
    import torch
    torch.set_num_threads(1)
    if str(STUDY_CODE) not in sys.path:
        sys.path.insert(0, str(STUDY_CODE))
    import s2_models                                    # the study's model definitions (imports nothing dataset-bound)
    ck = torch.load(checkpoint_path(dataset), map_location="cpu", weights_only=False)
    assert ck["dataset"] == dataset and ck["model"] == "B1", (ck["dataset"], ck["model"])
    net = s2_models.Stage2Model(ck["model"], ck["d_traj"], ck["d_static"], d_geo=N_POINT_CHANNELS)
    net.load_state_dict(ck["state"], strict=True)       # the latent mean / std buffers are part of the state
    net.eval()
    return net, ck


def decoder_inputs(dataset: str, example: int, hier_stats: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """The decoder's inputs for one sequence, rebuilt from the caches exactly as the study's data classes build them:
    the standardised static descriptor (1032,) (hier_contact_gen/data.py, FoldData.S) and the per-point tokens of every
    frame (64, 512, 10) (contact_latent_temporal_stage2/s2_data.py, S2Data.geo_tokens)."""
    with np.load(qlib.HIER[dataset] / "sequences.npz") as z:
        g = int(z["g_index"][example])
        nearest, coverage = z["G_nearest"][g] / 0.3, z["G_coverage"][g]                 # (512,), (512,)
        static = np.concatenate([nearest, coverage, z["G_radius"][g:g + 1], z["G_extent"][g],
                                 np.eye(2, dtype=np.float32)[z["hand"][example]], np.eye(2, dtype=np.float32)[z["role"][example]]])
        articulation = z["O"][example, qlib.PAD:qlib.PAD + T, 0].astype(np.float32)      # (64,) radians; used on ARCTIC only
    static = ((static - hier_stats["muS"]) / hier_stats["sdS"]).astype(np.float32)
    with np.load(qlib.RESULT_ROOT / GEOMETRY_CACHE.format(dataset=dataset)) as c:
        k = int(c["geo_index"][example])
        assert k >= 0, "the sequence has no canonical geometry"
        x_top, x_bot, n_top, n_bot = (c[name][k].astype(np.float32) for name in ("X_top", "X_bot", "N_top", "N_bot"))   # (512, 3)
        centroid, length = c["centroid"][k].astype(np.float32), np.float32(c["length"][k])
        area, valid = (c["alpha"][k] * 512).astype(np.float32), c["valid"][k].astype(np.float32)
    if dataset == "arctic":                             # the top part turns about z by -articulation(t)
        cos, sin = np.cos(-articulation)[:, None, None], np.sin(-articulation)[:, None, None]

        def turn(v: np.ndarray) -> np.ndarray:
            return np.concatenate([cos * v[None, :, :1] - sin * v[None, :, 1:2], sin * v[None, :, :1] + cos * v[None, :, 1:2],
                                   np.broadcast_to(v[None, :, 2:], (T, 512, 1))], -1)
        position, normal = turn(x_top) + x_bot, turn(n_top) + n_bot
    else:
        position, normal = np.broadcast_to(x_top + x_bot, (T, 512, 3)), np.broadcast_to(n_top + n_bot, (T, 512, 3))
    position = (position - centroid) / length
    normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-9)
    per_point = np.stack([nearest, coverage, area, valid], -1).astype(np.float32)        # (512, 4), the same in every frame
    tokens = np.concatenate([position, normal, np.broadcast_to(per_point, (T, 512, 4))], -1).astype(np.float32)
    return static, tokens


def decode(net: Any, z_standardised: np.ndarray, tokens: np.ndarray, static: np.ndarray, hier_stats: Mapping[str, Any]) -> np.ndarray:
    """z (M, 64) in the standardised coordinates the generator predicts, tokens (M, 512, 10), static (1032,) ->
    raw contact maps (M, 512), through the generator's decoder."""
    import torch
    with torch.no_grad():
        maps, _ = net.decode_maps(torch.from_numpy(np.ascontiguousarray(z_standardised, dtype=np.float32)), None,
                                  torch.from_numpy(np.ascontiguousarray(tokens)), torch.from_numpy(static)[None].expand(len(tokens), -1))
    return maps.numpy() * float(hier_stats["s"]) + np.asarray(hier_stats["mu"], np.float32)


def decode_true_z(dataset: str, sel: Selection) -> tuple[np.ndarray, dict[str, Any]]:
    """The maps of the last row, (64, 512): the cached z of the true frames 0..63 through the generator's decoder.
    Also re-decodes the SAVED predicted z and compares it with the saved prediction; raises if they differ by more
    than DECODE_TOLERANCE, because the row would then not show 'the same decoder'."""
    net, ck = load_generator(dataset)
    stats = ck["stats"]
    static, tokens = decoder_inputs(dataset, sel.example, stats["sat"]["hier"])
    teacher_file = STUDY / dataset / "cache" / "teacher_z.npz"
    with np.load(teacher_file) as z:
        assert bool(z["has"][sel.example]), "no cached latent for this sequence"
        z_true = (z["z"][sel.example] - stats["z_mu"]) / stats["z_sd"]                  # (64, 64) standardised
        encoder = {"checkpoint": str(z["ckpt"]), "md5": str(z["md5"]), "step": int(z["best_step"])}
    true_maps = decode(net, z_true, tokens, static, stats["sat"]["hier"])               # (64, 512)
    again = decode(net, sel.z_predicted, tokens[1:], static, stats["sat"]["hier"])      # (63, 512)
    difference = float(np.abs(again - sel.via_z[1:]).max())
    if difference > DECODE_TOLERANCE:
        raise RuntimeError(f"{dataset}: the saved predicted z, decoded here, differs from the saved prediction by {difference:.4f}")
    path = checkpoint_path(dataset)
    info = {
        "checkpoint": str(path), "checkpoint_md5_head": md5_head(path), "state": "selected (best validation) state, key 'state'",
        "selected_at_step": int(ck["best_step"]), "training_seed": int(ck["seed"]), "parameters": {k: int(v) for k, v in ck["n_params"].items()},
        "model_class": "scripts/research/contact_latent_temporal_stage2/s2_models.py: Stage2Model, decode_maps (strict state load)",
        "latents_of_the_true_frames": {"file": rel(teacher_file), "array": f"z[{sel.example}] (64 frames x 64), standardised with the "
                                       "checkpoint's training mean and std", "encoder": encoder},
        "check_same_decoder": {
            "what": "the saved predicted z of this sequence (z_hat of the prediction file), decoded here, against the saved prediction",
            "max_abs_difference_per_value": difference, "tolerance": DECODE_TOLERANCE,
            "dense_error_re_decoded": float(dense_error(np.concatenate([sel.recorded[:1], again]), sel.recorded)),
            "dense_error_saved": float(dense_error(sel.via_z, sel.recorded)),
            "note": "the remaining difference is CPU float32 here against the study's GPU bfloat16 pass",
        },
        "z_rms_error_of_the_prediction": float(np.sqrt(((sel.z_predicted - z_true[1:]) ** 2).mean())),
    }
    del net
    return true_maps.astype(np.float32), info


# ------------------------------------------------------------------------------ view
def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outward unit normal (V, 3) and surface area (V,) per vertex, from the triangles (area-weighted)."""
    tri = (verts - verts.mean(0))[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])          # twice the triangle area, along its normal
    if (tri[:, 0] * cross).sum() < 0:                                        # signed volume: the winding is inward
        cross = -cross
    normal, area = np.zeros_like(verts), np.zeros(len(verts))
    for corner in range(3):
        np.add.at(normal, faces[:, corner], cross)
        np.add.at(area, faces[:, corner], np.linalg.norm(cross, axis=1) / 6)
    return normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12), area


def sphere_directions(n: int) -> np.ndarray:
    """(n, 3) unit vectors spread evenly over the sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    polar, azimuth = np.arccos(1 - 2 * i / n), np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], 1)


def contact_view(meshes: Sequence[tuple[np.ndarray, np.ndarray]], hands: Sequence[np.ndarray],
                 recorded: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """A (direction, up) pair for qlib.fit_camera and the numbers it rests on, computed from the drawn frames alone.

    direction: of N_DIRECTIONS evenly spread directions on the hand's side of the object, the one from which the most
    recorded contact faces the camera (sum over the drawn frames and the vertices of area x recorded value x cosine
    between the outward normal and the direction). up: the object's longest principal axis is horizontal, with the
    hand on the left. The model outputs play no part in it."""
    candidates = sphere_directions(N_DIRECTIONS)
    score, weight = np.zeros(len(candidates)), 0.0
    for (verts, faces), values in zip(meshes, recorded):
        normal, area = vertex_normals(verts, faces)
        mass = area * np.nan_to_num(values)
        score += np.clip(candidates @ normal.T, 0.0, None) @ mass
        weight += float(mass.sum())
    centre = np.concatenate([m[0] for m in meshes]).mean(0)
    to_hand = np.concatenate(hands).mean(0) - centre
    allowed = np.flatnonzero(candidates @ to_hand > 0)
    best = int(allowed[np.argmax(score[allowed])])
    direction = candidates[best]
    first = meshes[0][0]
    long_axis = np.linalg.eigh(np.cov((first - first.mean(0)).T))[1][:, 2]
    right = long_axis if long_axis @ to_hand < 0 else -long_axis              # pointing away from the hand: the hand is on the left
    up = np.cross(direction, right)
    up = up / np.linalg.norm(up)
    info = {"direction": direction.tolist(), "up": up.tolist(), "candidates": N_DIRECTIONS, "candidates_on_the_hand_side": int(len(allowed)),
            "share_of_recorded_contact_facing_the_camera": float(score[best] / weight),
            "best_share_over_all_directions": float(score.max() / weight), "object_longest_axis": long_axis.tolist()}
    return direction, up, info


# ------------------------------------------------------------------------------ one dataset block
@dataclass
class Block:
    dataset: str
    example: qlib.Example
    selection: Selection
    maps: dict[str, np.ndarray]                # row key -> (64, 512)
    values: dict[str, np.ndarray]              # row key -> (frames drawn, V) per-vertex values that are drawn
    meshes: list[tuple[np.ndarray, np.ndarray]]
    hands: list[np.ndarray]
    hand_faces: np.ndarray
    errors: dict[str, float]                   # row key -> dense error of this sequence (frames 1..63)
    decoder: dict[str, Any]


def build_block(dataset: str) -> Block:
    sel = select_example(dataset)
    log.info("%s: example %d, rank %d of %d by the via-z dense error %.4f (median %.4f)", dataset, sel.example,
             sel.evidence["chosen"]["rank"], sel.n_test, sel.evidence["chosen"]["value"], sel.evidence["median"])
    ex = qlib.load_example(dataset, sel.example)
    assert ex.meta.split == "test" and np.array_equal(ex.C, sel.recorded)
    assert sel.row == ex.test_row(sel.examples)
    assert np.array_equal(sel.direct[0], sel.recorded[0]) and np.array_equal(sel.via_z[0], sel.recorded[0]), "frame 0 is not the given map"
    meshes = [ex.mesh(t) for t in FRAMES]
    hands = [ex.hand(t)[0] for t in FRAMES]              # loads the take (TACO: MANO on CPU) before the study's modules are imported
    true_maps, decoder = decode_true_z(dataset, sel)
    maps = {"recorded": sel.recorded, "direct": sel.direct, "via_z_predicted": sel.via_z, "via_z_true": true_maps}
    values = {key: ex.to_vertices(m[list(FRAMES)]) for key, m in maps.items()}          # (6, V) each, by the page's display rule
    errors = {key: float(dense_error(m, sel.recorded)) for key, m in maps.items() if key != "recorded"}
    log.info("%s: dense error of this sequence %s", dataset, {k: round(v, 4) for k, v in errors.items()})
    return Block(dataset, ex, sel, maps, values, meshes, hands, ex.hand(FRAMES[0])[1], errors, decoder)


def table_value(table: Path, dataset: str, locator: Mapping[str, str], column: str) -> float:
    frame = pd.read_csv(table)
    rows = frame[(frame.dataset == dataset) & np.logical_and.reduce([frame[k] == v for k, v in locator.items()])]
    assert len(rows) == 1, (table, dataset, locator)
    return float(rows[column].iloc[0])


def render_block(block: Block, vmax: float, test_means: Mapping[str, float], legend: Sequence[qlib.Colorbar]) -> tuple[Image.Image, dict[str, Any]]:
    """The grid of one dataset (rows x frames): one camera for every panel, one colour scale for the whole figure."""
    recorded_values = list(block.values["recorded"])
    direction, up, view = contact_view(block.meshes, block.hands, recorded_values)
    camera = qlib.fit_camera([m[0] for m in block.meshes] + block.hands, direction, up, margin=CAMERA_MARGIN, aspect=PANEL[0] / PANEL[1])
    images = []
    for row in ROWS:
        images.append([])
        for i, (verts, faces) in enumerate(block.meshes):
            items = [qlib.mesh_item(verts, faces, rgb=qlib.contact_rgb(block.values[row.maps][i], vmax))]
            if row.hand:
                items.append(qlib.hand_item(block.hands[i], block.hand_faces, **qlib.GHOST))
            images[-1].append(qlib.render(items, camera, PANEL))
    labels = [row.label if row.table is None else f"{row.label}\nerror {block.errors[row.key]:.2f}\ntest mean {test_means[row.key]:.2f}"
              for row in ROWS]
    image = qlib.grid(images, labels, [f"frame {t}" for t in FRAMES], colorbars=legend)
    return image, {"camera": camera, "view_rule_numbers": view, "row_labels": [s.replace("\n", ", ") for s in labels]}


def heading_font(px: int) -> ImageFont.FreeTypeFont:
    """DejaVu Sans Bold at the size of the grid's labels (the regular face if the bold one is missing)."""
    regular = next((Path(p) for p in qlib.FONT_PATHS if Path(p).exists()), None)
    if regular is None:
        import matplotlib
        regular = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
    bold = regular.with_name("DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(bold if bold.exists() else regular), px)


def stack_blocks(blocks: Sequence[tuple[str, Image.Image]]) -> Image.Image:
    """Stack the dataset grids (equal widths) under one heading each; keeps the grid's layout record."""
    layout = dict(blocks[0][1].info["qlib_layout"])
    assert all(image.width == layout["width"] for _, image in blocks)
    font = heading_font(layout["font_px"])
    line = math.ceil(layout["font_px"] * 1.32)
    margin = 2 * max(6, layout["panel"][0] // 50)        # the left margin qlib.grid uses
    canvas = Image.new("RGB", (layout["width"], sum(image.height + line + margin for _, image in blocks)), "white")
    draw, y = ImageDraw.Draw(canvas), 0
    for heading, image in blocks:
        draw.text((margin, y + margin), heading, font=font, fill=qlib.INK)
        canvas.paste(image, (0, y + margin + line))
        y += image.height + line + margin
    layout["height"] = canvas.height
    canvas.info["qlib_layout"] = layout
    return canvas


# ------------------------------------------------------------------------------ meta
def number(label: str, value: Any, dataset: str, source: str, locator: str, shown_as: str | None = None) -> dict[str, Any]:
    """One entry of meta.numbers; ``shown_as`` is the rounded text used in the figure or the caption."""
    entry = {"label": label, "value": value, "dataset": dataset, "source": source, "locator": locator}
    if shown_as is not None:
        entry["shown_as"] = shown_as
    return entry


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    qlib.require_headless()
    blocks = [build_block(dataset) for dataset in DATASETS]

    # one colour scale for the whole figure: the largest drawn value, rounded up
    largest = {LABEL[b.dataset]: max(float(np.nanmax(v)) for v in b.values.values()) for b in blocks}
    drawn_max = max(largest.values())
    drawn_min = min(float(np.nanmin(v)) for b in blocks for v in b.values.values())
    vmax = round(math.ceil(drawn_max / VMAX_STEP - 1e-9) * VMAX_STEP, 6)
    unreached = {b.dataset: float(1 - b.example.covered.mean()) for b in blocks}
    legend = [qlib.colorbar("contact", 0.0, vmax, "contact, exp(\N{MINUS SIGN}distance to the hand / 2 cm)",
                            nodata="not reached by the map" if max(unreached.values()) >= NODATA_LEGEND_FRACTION else None)]

    numbers: list[dict[str, Any]] = []
    cross_checks: list[dict[str, Any]] = []
    panels: dict[str, dict[str, np.ndarray]] = {}
    examples, cameras, row_labels, grids, inference_runs = [], {}, {}, [], {}
    for b in blocks:
        ds, name, sel, m = b.dataset, LABEL[b.dataset], b.selection, b.example.meta
        test_means = {row.key: table_value(row.table, ds, row.locator, row.column) for row in SCORED}
        hand = qlib.SIDE[m.hand]
        image, drawn = render_block(b, vmax, test_means, legend if b is blocks[-1] else ())
        grids.append((f"{name}: one test sequence ({m.category}, {hand} hand)", image))
        cameras[ds], row_labels[ds], inference_runs[ds] = {**drawn["view_rule_numbers"], "camera": drawn["camera"]}, drawn["row_labels"], b.decoder

        # ---- what was drawn, for the verifier
        for row in ROWS:
            panels[f"{ds}_{row.key}"] = {"c512": b.maps[row.maps], **{f"frame{t:02d}": b.values[row.maps][i] for i, t in enumerate(FRAMES)}}
        panels[f"{ds}_geometry"] = {"frames": np.asarray(FRAMES), "mesh_verts": np.stack([v for v, _ in b.meshes]), "mesh_faces": b.meshes[0][1],
                                    "hand_verts": np.stack(b.hands), "hand_faces": b.hand_faces, "covered": b.example.covered}

        # ---- numbers: this sequence (from the drawn arrays), the test set (from the study's tables), and their cross-checks
        per_example = {"direct": STUDY / ds / "metrics" / f"{DIRECT}_A.npz", "via_z_predicted": STUDY / ds / "metrics" / f"{VIA_Z}_A.npz"}
        for row in SCORED:
            numbers.append(number(f"dense error of this sequence, {row.label.lower()}", b.errors[row.key], name,
                                  "computed in the script from the drawn arrays: mean over frames 1..63 of the L2 distance over the 512 values",
                                  f"panels.npz: {ds}_{row.key}/c512 against {ds}_recorded/c512", f"{b.errors[row.key]:.2f}"))
            locator = ", ".join(f"{k}={v}" for k, v in {"dataset": ds, **row.locator}.items()) + f", column {row.column}"
            numbers.append(number(f"test-set mean dense error, {row.label.lower()}", test_means[row.key], name, rel(row.table), "row " + locator,
                                  f"{test_means[row.key]:.2f}"))
            if row.key in per_example:
                with np.load(per_example[row.key]) as z:
                    assert np.array_equal(z["example"], sel.examples)
                    listed = float(z["E_C"][sel.row, 0])
                cross_checks.append({"dataset": name, "quantity": f"dense error of this sequence, {row.label.lower()}", "computed": b.errors[row.key],
                                     "table": listed, "table_file": rel(per_example[row.key]), "table_key": f"E_C[{sel.row}, 0] (example {sel.example})",
                                     "difference": abs(b.errors[row.key] - listed), "agrees": abs(b.errors[row.key] - listed) <= TABLE_TOLERANCE})
                cross_checks.append({"dataset": name, "quantity": f"test-set mean dense error, {row.label.lower()}", "computed": sel.test_mean[row.key],
                                     "computed_from": f"{rel(prediction_file(ds, DIRECT if row.key == 'direct' else VIA_Z))} against the recorded maps",
                                     "table": test_means[row.key], "table_file": rel(row.table), "table_key": "row " + locator,
                                     "difference": abs(sel.test_mean[row.key] - test_means[row.key]),
                                     "agrees": abs(sel.test_mean[row.key] - test_means[row.key]) <= TABLE_TOLERANCE})
            else:
                cross_checks.append({"dataset": name, "quantity": f"dense error of this sequence, {row.label.lower()}", "computed": b.errors[row.key],
                                     "table": None, "agrees": None,
                                     "note": "the study saved no per-sequence value of this path (only the test-set mean and curve), so this "
                                             "number has no table to be checked against; the decoder itself is checked by re-decoding the "
                                             "saved predicted z (inference.runs.*.check_same_decoder)"})
        evidence = sel.evidence
        numbers += [
            number("number of test sequences", sel.n_test, name, rel(prediction_file(ds, VIA_Z)), "length of the array example", str(sel.n_test)),
            number("rank of the chosen sequence by the via-z dense error (1 = smallest)", evidence["chosen"]["rank"], name,
                   "computed in the script from the saved predictions", "meta.selection_evidence"),
            number("median over the test sequences of the via-z dense error", evidence["median"], name,
                   "computed in the script from the saved predictions", "meta.selection_evidence", f"{evidence['median']:.2f}"),
            number("test sequences on which the via-z dense error is below the direct generator's",
                   evidence["typicality"]["sequences_where_via_z_is_lower"], name,
                   "computed in the script from the saved predictions of both generators", "meta.selection_evidence.typicality",
                   f"{evidence['typicality']['sequences_where_via_z_is_lower']} of {sel.n_test}"),
            number("share of the mesh vertices the 512-point map does not reach", unreached[ds], name,
                   "computed in the script from the study's operator W (qlib.Example.covered)", f"panels.npz: {ds}_geometry/covered",
                   f"{unreached[ds]:.0%}"),
            number("largest difference between the saved prediction and the saved predicted z decoded by this script",
                   b.decoder["check_same_decoder"]["max_abs_difference_per_value"], name, "computed in the script",
                   "meta.inference.runs.*.check_same_decoder"),
        ]
        for row in SCORED:
            distance = frame_error(b.maps[row.key], sel.recorded)
            for t in FRAMES:
                numbers.append(number(f"L2 distance from the recorded map at frame {t}, {row.label.lower()}", float(distance[t]), name,
                                      "computed in the script from the drawn arrays", f"panels.npz: {ds}_{row.key}/c512[{t}] against {ds}_recorded/c512[{t}]"))
        tree = [cKDTree(v) for v, _ in b.meshes]
        examples.append({
            "dataset": name, "example": sel.example, "sequence_id": m.sequence_id, "object": f"{m.category} (mesh {m.mesh_id}, role {m.role})",
            "hand": hand, "frames": list(FRAMES), "take_frames": [b.example.take_frame(t) for t in FRAMES], "split": m.split, "group": m.group,
            "row_in_prediction_files": sel.row, "subject": evidence["typicality"]["subject"],
            "training_sequences_with_this_mesh": evidence["typicality"]["training_sequences_with_this_mesh"],
            "training_sequences_of_this_category": evidence["typicality"]["training_sequences_of_this_category"],
            "hand_to_object_distance_mm": [round(float(tr.query(h)[0].min()) * 1000, 2) for tr, h in zip(tree, b.hands)],
            "articulation_rad": [round(b.example.articulation(t), 4) for t in FRAMES],
        })
    numbers.append(number("upper end of the colour scale", vmax, "TACO and ARCTIC", "computed in the script: the largest value drawn in any panel "
                          f"({drawn_max:.4f}), rounded up to a multiple of {VMAX_STEP:g}", "meta.display_choices.colour_scale", f"{vmax:g}"))
    sequences = "sequences.npz of the two studies (" + ", ".join(rel(qlib.HIER[ds] / "sequences.npz") for ds in DATASETS) + ")"
    numbers += [
        number("frames per sequence", T, "TACO and ARCTIC", sequences, "array C, axis 1", str(T)),
        number("values per contact map (canonical points)", blocks[0].maps["recorded"].shape[1], "TACO and ARCTIC", sequences, "array C, axis 2",
               str(blocks[0].maps["recorded"].shape[1])),
        number("frames drawn per sequence", len(FRAMES), "TACO and ARCTIC", "constant of the script",
               "FRAMES = " + ", ".join(map(str, FRAMES)), "six"),
        number("first and last frame of the error average", [1, T - 1], "TACO and ARCTIC", "the study's definition of the dense error (frame 0 is "
               "given, never generated)", "scripts/research/contact_latent_temporal_stage2/s2_generate.py, s2_diag_bottleneck.py", f"1 to {T - 1}"),
    ]
    assert all(b.maps["recorded"].shape == (T, 512) for b in blocks)
    for check in cross_checks:
        if check["agrees"] is False:
            log.warning("DISAGREEMENT with the study's table: %s", check)

    image = stack_blocks(grids)

    # ---- caption, from the numbers above
    by = {b.dataset: b for b in blocks}

    def errors_text(ds: str) -> str:
        e = by[ds].errors
        return f"{e['direct']:.2f} (direct), {e['via_z_predicted']:.2f} (predicted z) and {e['via_z_true']:.2f} (true z)"

    def means_text(ds: str) -> str:
        direct, predicted, true = (f"{table_value(r.table, ds, r.locator, r.column):.2f}" for r in SCORED)
        return f"{direct}, {predicted} and {true}"

    objects = {ds: f"a {qlib.SIDE[by[ds].example.meta.hand]} hand on a {by[ds].example.meta.category}" for ds in DATASETS}
    lower = {ds: by[ds].errors["via_z_predicted"] < by[ds].errors["direct"] for ds in DATASETS}
    caption = (
        f"One test sequence per dataset on the real object, at six of its 64 frames: {objects['taco']} (TACO) and {objects['arctic']} (ARCTIC), "
        "each selected as the test sequence at the median error of the generator that goes through z. "
        "The first two rows show the recorded contact, with the recorded hand as a ghost and without it; below them are the direct generator, the "
        "generator via z with its own predicted z, and the same decoder fed the z of the true frames, an analysis-only input that no generator has. "
        "Frame 0 is given to both generators; in the last row it is decoded as well. "
        "The error of a row is the distance between its maps and the recorded maps, averaged over frames 1 to 63: on the TACO sequence "
        f"{errors_text('taco')}, against test-set means of {means_text('taco')}; on the ARCTIC sequence {errors_text('arctic')}, against "
        f"{means_text('arctic')}. "
        + ("On both of these sequences the error with the predicted z is below the direct generator's, which the test-set means do not show. "
           if all(lower.values()) else "")
        + (f"Pale violet marks the part of the {by['taco'].example.meta.category} that the 512-point map does not reach."
           if unreached["taco"] >= NODATA_LEGEND_FRACTION else "")).strip()

    meta = {
        "id": FIG_ID,
        "block": BLOCK,
        "title": TITLE,
        "panels": {
            "layout": f"two blocks, TACO above ARCTIC, each a grid of {len(ROWS)} rows x {len(FRAMES)} columns under a heading with the dataset, the "
                      "object and the hand",
            "columns": [f"frame {t} of the 64-frame sequence (30 frames per second)" for t in FRAMES],
            "rows": [{"key": r.key, "label": {LABEL[ds]: row_labels[ds][i] for ds in DATASETS}, "shows": r.what} for i, r in enumerate(ROWS)],
            "frame_0": "frame 0 is the given first map: the two generator rows show the recorded map there (the saved predictions equal it "
                       "exactly); the last row shows the decoder's output for the true z of frame 0, which is not part of the error",
            "two_recorded_rows": "the first row adds the recorded hand as a ghost, which mutes the colours of the contact under it; the second "
                                 "row draws the same recorded maps without the hand and is the one to compare the model rows with",
            "row_label_numbers": "'error' is this sequence's dense error (frames 1 to 63), 'test mean' the mean over the dataset's test sequences",
            "arrays": "panels.npz: <dataset>_<row>/c512 is the (64, 512) map sequence, <dataset>_<row>/frameNN the per-vertex values drawn, "
                      "<dataset>_geometry/* the meshes, hands and covered mask of the drawn frames",
        },
        "examples": examples,
        "selection_rule": "Per dataset, the test sequence at the median dense error of the generator via z: the test sequences are sorted by that "
                          "error (mean over frames 1 to 63 of the L2 distance between the saved prediction and the recorded map; ties: smaller "
                          "example index). The test sets have an even number of sequences, so the two middle ones are equally close to the "
                          "median and the one with the smaller example index is taken.",
        "selection_evidence": {LABEL[b.dataset]: b.selection.evidence for b in blocks},
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (qlib.Example.to_vertices; the same rule for the recorded maps and every model output; "
                                    "a smoothing, not an inverse). The errors are computed on the 512 values, not on the surface.",
            "colour_scale": {"kind": "contact (grey, orange, dark red-brown)", "min": 0.0, "max": vmax,
                             "shared_by": "every panel of both datasets, so that one colour bar serves the figure (a display constant; no "
                                          "statistic is pooled over the datasets)",
                             "rule": f"the largest value drawn in any panel, rounded up to a multiple of {VMAX_STEP:g}",
                             "largest_drawn_value": drawn_max, "largest_drawn_value_by_dataset": largest, "smallest_drawn_value": drawn_min,
                             "below_zero": "model outputs are not clamped; values below 0 are drawn in the colour of 0",
                             "not_reached": f"vertices no canonical point reaches are drawn in {qlib.NO_DATA} "
                                            f"({', '.join(f'{LABEL[d]} {u:.1%}' for d, u in unreached.items())} of the vertices)"},
            "camera": {"rule": f"one camera per dataset for all rows and frames. Direction: of {N_DIRECTIONS} evenly spread directions on the hand's side "
                               "of the object, the one from which the most recorded contact faces the camera (area x recorded value x cosine of "
                               "the outward normal, summed over the drawn frames). The object's longest principal axis is horizontal, the hand "
                               "on the left. Fitted (qlib.fit_camera) on the object and the hand of all drawn frames; perspective projection. "
                               "The model outputs play no part in the view.",
                       "margin": CAMERA_MARGIN, "panel_px": list(PANEL), **{LABEL[d]: c for d, c in cameras.items()}},
            "hand": f"recorded MANO hand, top row only, as a ghost ({qlib.GHOST['color']} at {qlib.GHOST['opacity']:.0%} opacity, front faces only); "
                    "the generators output no hand, so the other rows show the object alone. The ghost mutes the colours under it, hence the "
                    "second recorded row without the hand",
            "frames": "six fixed frames of the 64 (0, 8, 16, 32, 48, 63); ARCTIC: the mesh is articulated at each frame",
        },
        "numbers": numbers,
        "cross_checks": cross_checks,
        "inference": {
            "device": "cpu (1 thread, float32)",
            "seed": "none needed: the decoder is deterministic (the checkpoints are of training seed 0)",
            "newly_computed": "last row only: for each of the two sequences, the 64 maps decoded from the cached z of the true frames 0..63 by "
                              "the decoder of the generator via z; and, as a check, the 63 maps decoded from the saved predicted z",
            "read_from_saved_predictions": {LABEL[ds]: {"direct": rel(prediction_file(ds, DIRECT)), "via_z_predicted": rel(prediction_file(ds, VIA_Z))}
                                            for ds in DATASETS},
            "decoder_inputs": "rebuilt by this script from sequences.npz and the canonical geometry cache as the study's data classes build them "
                              "(hier_contact_gen/data.py FoldData.S; contact_latent_temporal_stage2/s2_data.py S2Data.geo_tokens), with the "
                              "normalisation statistics stored in the checkpoint",
            "runs": {LABEL[ds]: run for ds, run in inference_runs.items()},
            "not_run": "no training; none of the study's scripts; the generators themselves were not re-run (their saved predictions are drawn)",
        },
        "caveats": [
            "One sequence per dataset and one training seed; a picture of two sequences, not a result. The dataset-level statements rest on the "
            "study's tables (test-set means in the row labels).",
            "The sequence is at the median of the VIA-Z error only. On both chosen sequences the via-z error is below the direct generator's: "
            + "; ".join(f"{LABEL[b.dataset]} {b.errors['via_z_predicted']:.2f} against {b.errors['direct']:.2f}, while the test-set means are "
                        f"{table_value(MAIN_TABLE, b.dataset, {'model': 'B1', 'metric': 'E_C'}, 'value'):.2f} against "
                        f"{table_value(MAIN_TABLE, b.dataset, {'model': 'B0', 'metric': 'E_C'}, 'value'):.2f} (via z lower on "
                        f"{b.selection.evidence['typicality']['sequences_where_via_z_is_lower']} of {b.selection.n_test} test sequences)" for b in blocks)
            + ". The figure therefore does not illustrate the test-set comparison of the two generators (tie on TACO, via z worse on ARCTIC).",
            "The error distribution is skewed: the median via-z error ("
            + ", ".join(f"{LABEL[b.dataset]} {b.selection.evidence['median']:.2f}" for b in blocks) + ") is below the mean, so a median sequence "
            "has a smaller error than the test-set mean printed next to it.",
            "The last row uses the z of the true future frames: an analysis-only diagnostic of what the decoder can do with a perfect latent, "
            "not a generator. The decoder was trained on predicted latents. On the TACO sequence this row's error "
            f"({by['taco'].errors['via_z_true']:.2f}) is well above its test-set mean "
            f"({table_value(TRUE_Z_TABLE, 'taco', {'model': 'B1'}, 'E_C_oracle_z'):.2f}); the study saved no per-sequence values of this path "
            "to say where the sequence lies in that distribution.",
            "The differences between the rows are small against the colour range (a dense error of 1 is a root-mean-square difference of "
            f"{1 / math.sqrt(512):.3f} per map value, on a colour scale from 0 to {vmax:g}); the rows differ mostly in the shape and strength "
            "of one contact patch.",
            "Every map is 512 canonical values spread onto the mesh by a smoothing rule, so fine detail on the surface is not meaningful. "
            f"On the TACO {by['taco'].example.meta.category} {unreached['taco']:.0%} of the vertices are not reached by the map (pale violet). "
            "A thin object shows contact on both of its faces; only the face towards the camera is seen.",
            "Model outputs are not clamped: values below 0 are drawn as 0.",
            "The last row was decoded on CPU in float32; the study's test-set mean of this path was computed on GPU with bfloat16 autocast "
            "(the same difference changes the re-decoded predicted maps by at most "
            f"{max(b.decoder['check_same_decoder']['max_abs_difference_per_value'] for b in blocks):.3f} per value).",
            "Seen and unseen: " + "; ".join(
                f"{LABEL[b.dataset]}: this object ({b.example.meta.category}, mesh {b.example.meta.mesh_id}) occurs in "
                f"{b.selection.evidence['typicality']['training_sequences_with_this_mesh']} of the "
                f"{b.selection.evidence['typicality']['training_sequences']} training sequences, the subject "
                f"{'also appears' if b.selection.evidence['typicality']['subject_in_training'] else 'does not appear'} in training takes"
                for b in blocks) + ".",
        ],
        "suggested_caption": caption,
        "suggested_alt": "Two blocks of 3-D renders, a plate with a left hand and a box with a right hand, each with five rows of orange contact "
                         "patches on the object over six frames: recorded with the hand, recorded alone, direct generator, generator via the "
                         "predicted z, and the decoder fed the z of the true frames.",
        "script": "docs/build/qual/q_direct_vs_via_z.py",
        "command": qlib.RUN_COMMAND.replace("<script>", "docs/build/qual/q_direct_vs_via_z.py"),
    }
    paths = qlib.save(FIG_ID, image, panels, meta)
    log.info("saved %s", {k: str(v) for k, v in paths.items()})
    log.info("caption: %s", caption)


if __name__ == "__main__":
    main()
