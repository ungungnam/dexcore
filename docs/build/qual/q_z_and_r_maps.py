"""Qualitative figure ``z_and_r_maps`` (page block F3-B): the z-only map, the correction r adds, and
a latent swap, on the real object meshes.

The Stage-1 study (``reports/contact_factorization_stage1``) is a single-frame autoencoder of the
512-value contact map: a latent z is decoded to a "z-only" map, and a second code r adds a
correction to it. The figure has two parts, each with one TACO row and one ARCTIC row.

* Reconstruction, one test frame per dataset. Columns: the recorded contact with the recorded hand,
  the z-only map, what r adds (z + r map minus z-only map, on its own finer scale), the z + r map.
  The maps are the ones the study saved (``<ds>/latents/A3_seed0_test.npz``: ``C_bar``, ``C_hat``).
* Swap, one pair of test frames A and B per dataset (same object, two takes). Columns: frame A and
  frame B as recorded, the z of A decoded with the r of B, the z of B decoded with the r of A. Under
  each swapped map a second row shows what the exchanged r changes: the swapped map minus the map
  decoded from the same z with the frame's own r, on the finer difference scale. The study saved
  swapped maps for its first six pairs only, so the two swaps are computed here with the study's own
  model and data classes on the CPU (float32, weights ``ema_state`` of the saved checkpoint, opened
  read-only).

The recorded hand is a line drawing laid over the render of the map alone (its outline and the depth
jumps between its fingers, where the hand is in front of the object). A filled or translucent hand
repaints the map under it; with lines, every pixel that is not on a line is the pixel of the hand-free
render, so the recorded panel can be compared with the decoded panels beside it.

Selection, computed below from the study's saved tables (nothing is chosen by eye):

* frame: the lower median of the z-only reconstruction error ``e_zonly`` over the unique test frames
  (the study's test frames: a frame shared by two overlapping sequences of one take counts once, at
  its first occurrence in the saved file; rank ``(N - 1) // 2`` of the stable ascending order);
* pair: the lower median, over the controlled pairs of ``swap_pairs_A3.csv``, of the pair's detail
  score = mean of ``detail_cos_rdonor`` over its two swaps (ties: first in file).

The study's loader binds one dataset per process, so the model runs in a child process per dataset,
one after the other. Geometry is in the contacted object's own frame, in metres (see ``qlib``).

Run (CPU, software renderer; see README.md):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_z_and_r_maps.py
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib  # noqa: E402

log = logging.getLogger("z_and_r_maps")

FIG_ID = "z_and_r_maps"
BLOCK = "F3-B"
TITLE = "The z-only map, the correction r adds, and a latent swap"
STUDY = qlib.RESULT_ROOT / "reports/contact_factorization_stage1"
STUDY_SCRIPTS = qlib.REPO / "scripts/research/contact_factorization_stage1"
MODEL = "A3"                     # the study's name of the z / r model trained with the relational loss
RUN = f"{MODEL}_seed0"           # its one training seed
DATASETS = ("taco", "arctic")
NAME = {"taco": "TACO", "arctic": "ARCTIC"}
OBJECT_WORDS = {"espressomachine": "espresso machine", "capsulemachine": "capsule machine", "waffleiron": "waffle iron"}
HAND_WORDS = {"L": "left hand", "R": "right hand"}
SCRIPT = "computed in q_z_and_r_maps.py"

PANEL_WIDTH = 560                # pixels; the height follows the row's scene (ASPECT_RANGE)
ASPECT_RANGE = (1.25, 2.0)       # panel width / height
CONTACT_STEP, DIFFERENCE_STEP = 0.1, 0.01   # a colour limit is the largest drawn value rounded up to this step
RECON_COLUMNS = ("recorded, hand outlined", "decoded from z alone", "what r adds", "decoded from z and r")
SWAP_COLUMNS = ("frame A, recorded", "frame B, recorded", "z of A with r of B", "z of B with r of A")
SWAPS = {"AB": "z of A with r of B", "BA": "z of B with r of A"}
CHANGE_KEYS = {"AB": "z_A_r_B", "BA": "z_B_r_A"}             # panel keys of the two swaps
CHANGE_LABEL = "{dataset}, same pair\nwhat the exchanged r changes\n(swapped map minus the frame's own z and r map)"
# The recorded hand as a line drawing (pixels of one panel; a panel is PANEL_WIDTH wide): the outline of the part of
# the hand that is in front of the object, and the places inside it where the depth jumps (one finger over another).
HAND_LINE = {"colour": "#3c3b37", "outline_px": 2, "depth_jump_m": 0.006, "soften_px": 0.6}
# columns of the study's swap table, in plain words; the label-free ones are recomputed here from the drawn maps
SCORE_WORDS = {
    "struct_to_zdonor": "structure distance of the swapped map to the frame that gave z",
    "struct_to_rdonor": "structure distance of the swapped map to the frame that gave r",
    "detail_cos_rdonor": "detail score: cosine between the correction after the swap and the detail the r donor's z-only map misses",
    "detail_cos_zdonor": "cosine between the correction after the swap and the detail the z donor's z-only map misses",
    "dense_change": "L2 distance between the swapped map and the z donor's own z + r map",
    "dense_AB": "L2 distance between the two recorded maps",
    "dense_to_zdonor": "L2 distance between the swapped map and the z donor's recorded map",
    "dense_to_rdonor": "L2 distance between the swapped map and the r donor's recorded map"}
RECOMPUTED = tuple(c for c in SCORE_WORDS if not c.startswith("struct"))   # structure needs the hand-part labels


# ------------------------------------------------------------------------------ selection
def lower_median(values: np.ndarray) -> tuple[int, int, np.ndarray]:
    """(index, rank, order) of the lower median of ``values``: rank (N - 1) // 2 of the stable ascending order,
    so equal values keep their order in the file."""
    order = np.argsort(values, kind="stable")
    rank = (len(values) - 1) // 2
    return int(order[rank]), rank, order


def ranking(values: np.ndarray, rank: int, order: np.ndarray, describe: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
    """What proves a pick: the number of candidates, the rank and value of the pick, its neighbours in rank."""
    near = [{"rank": k, "value": float(values[order[k]]), **describe(int(order[k]))}
            for k in range(max(0, rank - 2), min(len(values), rank + 3))]
    return {"n_candidates": int(len(values)), "rank_0_based": rank, "value": float(values[order[rank]]),
            "numpy_median": float(np.median(values)), "mean": float(values.mean()),
            "quantiles": {f"{q:g}": float(np.quantile(values, q)) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)},
            "neighbours_in_rank": near}


def frame_table(dataset: str, examples: np.ndarray) -> pd.DataFrame:
    """One row per entry of the saved test file (sequence after sequence, 64 frames each): the take and object
    (``key``) and the take frame. Two overlapping sequences of one take share frames; the study counts such a frame
    once, at its first occurrence (cf_data.FrameData._frame_table, the same key and the same order)."""
    meta = pd.read_csv(qlib.HIER[dataset] / "sequences_meta.csv").iloc[examples]
    key = np.repeat((meta.take_key.astype(str) + "|" + meta.group.astype(str)).to_numpy(), qlib.T)
    frame = np.repeat(meta.t0.to_numpy(), qlib.T) + np.tile(np.arange(qlib.T), len(meta))
    return pd.DataFrame({"key": key, "frame": frame})


@dataclass(frozen=True)
class FramePick:
    dataset: str
    example: int                      # row of sequences_meta.csv
    row: int                          # row of the saved test file
    t: int                            # sequence frame
    e_zonly: float                    # saved errors of this frame (L2 over the 512 raw values)
    e_full: float
    z_only: np.ndarray                # (512,) saved maps of this frame
    z_and_r: np.ndarray
    examples: np.ndarray              # (sequences,) the example of every row of the saved file
    unique: np.ndarray                # (sequences, 64) first occurrences = the unique test frames
    errors: dict[str, np.ndarray]     # the saved (sequences, 64) error tables
    evidence: dict[str, Any]
    path: Path


def select_frame(dataset: str) -> FramePick:
    """The test frame at the lower median of the saved z-only reconstruction error over the unique test frames."""
    path = STUDY / dataset / "latents" / f"{RUN}_test.npz"
    with np.load(path) as z:
        examples, e_zonly, e_full = z["n"], z["e_zonly"], z["e_full"]
        table = frame_table(dataset, examples)
        unique = ~table.duplicated().to_numpy()
        flat = np.flatnonzero(unique)                        # the unique frames, in the order of the saved file
        values = e_zonly.ravel()[flat]
        k, rank, order = lower_median(values)
        row, t = divmod(int(flat[k]), e_zonly.shape[1])
        z_only, z_and_r = z["C_bar"][row, t].astype(np.float32), z["C_hat"][row, t].astype(np.float32)
    unique = unique.reshape(e_zonly.shape)
    # a repeated frame has one error, whichever sequence it is read from (so the first occurrence loses nothing)
    copies = table.assign(e=e_zonly.ravel()).groupby(["key", "frame"]).e
    spread = copies.max() - copies.min()
    every, every_rank, _ = lower_median(e_zonly.ravel())     # the other reading of the rule: every saved entry
    evidence = ranking(values, rank, order, lambda i: {"example": int(examples[flat[i] // qlib.T]), "t": int(flat[i] % qlib.T)})
    evidence.update(metric=f"e_zonly of {path} (L2 norm over the 512 values of the raw map)",
                    candidates=f"the unique test frames: {e_zonly.shape[0]} test sequences x {e_zonly.shape[1]} frames as saved, "
                               "a frame shared by overlapping sequences of one take counted once (first occurrence)",
                    n_saved_entries=int(e_zonly.size), n_repeated_entries=int(e_zonly.size - len(flat)),
                    largest_error_difference_between_the_copies_of_a_repeated_frame=float(spread.max()),
                    share_of_unique_frames_with_smaller_z_and_r_error=float((e_full[unique] < e_full[row, t]).mean()),
                    other_reading_median_over_every_saved_entry={
                        "example": int(examples[every // qlib.T]), "t": int(every % qlib.T), "value": float(e_zonly.ravel()[every]),
                        "rank_0_based": every_rank, "n_entries": int(e_zonly.size), "numpy_median": float(np.median(e_zonly)),
                        "share_of_saved_entries_below_the_pick": float((e_zonly < e_zonly[row, t]).mean())})
    return FramePick(dataset, int(examples[row]), int(row), int(t), float(e_zonly[row, t]), float(e_full[row, t]),
                     z_only, z_and_r, examples, unique, {"e_zonly": e_zonly, "e_full": e_full}, evidence, path)


@dataclass(frozen=True)
class PairPick:
    dataset: str
    pair: int                         # row of swap_pairs_selected.csv
    i_a: int                          # indices into the study's table of unique test frames
    i_b: int
    swaps: dict[str, pd.Series]       # "AB" (z of A, r of B) and "BA": the pair's two rows of the swap table
    scores: np.ndarray                # (pairs,) the detail score of every pair
    evidence: dict[str, Any]
    path: Path


def select_pair(dataset: str) -> PairPick:
    """The controlled pair at the lower median of the pair's detail score (mean over its two swaps)."""
    path = STUDY / dataset / f"swap_pairs_{MODEL}.csv"
    table = pd.read_csv(path, dtype={"mesh": str})
    chosen = pd.read_csv(STUDY / dataset / "swap_pairs_selected.csv", dtype={"mesh": str})
    assert table.groupby("pair").swap.apply(lambda s: sorted(s) == ["AB", "BA"]).all(), "a pair does not have two swaps"
    scores = table.groupby("pair", sort=True).detail_cos_rdonor.mean()
    assert np.array_equal(scores.index.to_numpy(), np.arange(len(chosen))), "the pair ids are not the rows of the pair table"
    pair, rank, order = lower_median(scores.to_numpy())
    swaps = {s: table[(table.pair == pair) & (table.swap == s)].iloc[0] for s in SWAPS}
    single = table.detail_cos_rdonor.to_numpy()
    other, _, _ = lower_median(single)                       # the other reading of the rule: the median single swap
    evidence = ranking(scores.to_numpy(), rank, order, lambda i: {"pair": i})
    evidence.update(metric=f"mean of detail_cos_rdonor over the two swaps (AB, BA) of a pair, {path}",
                    candidates=f"{len(scores)} controlled pairs of {STUDY / dataset / 'swap_pairs_selected.csv'}",
                    detail_cos_rdonor_of_the_two_swaps={s: float(swaps[s].detail_cos_rdonor) for s in SWAPS},
                    share_of_all_swaps_below_each={s: float((single < swaps[s].detail_cos_rdonor).mean()) for s in SWAPS},
                    other_reading_median_over_single_swaps={"pair": int(table.pair.iloc[other]), "swap": str(table.swap.iloc[other]),
                                                            "value": float(single[other]), "n_swaps": int(len(single))})
    return PairPick(dataset, pair, int(chosen.iA.iloc[pair]), int(chosen.iB.iloc[pair]), swaps, scores.to_numpy(), evidence, path)


# ------------------------------------------------------------------------------ the model, on the CPU
def stage1_forward(dataset: str, i_a: int, i_b: int, frame: tuple[int, int], threads: int) -> dict[str, Any]:
    """Encode test frames A and B (indices into the study's unique test-frame table), decode each with its own r
    and with the other's r, and re-decode the reconstruction frame ``(example, t)`` as a check of the saved maps.
    Runs in a process of its own: the study's loader binds one dataset per process. Maps are raw contact values
    clamped to [0, 1], as the study saves them. Nothing is written."""
    qlib.require_headless()
    sys.path.insert(0, str(STUDY_SCRIPTS))
    import torch
    torch.set_num_threads(threads)
    import cf_common
    from cf_data import FrameData
    from cf_models import build as build_model, n_params

    path = cf_common.ckpt_path(dataset, RUN)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    data = FrameData(dataset, torch.device("cpu"), stats=ck["stats"], need_masks=False)
    net = build_model(ck["model"], data.d_geo(), data.d_static(), res_scale=ck.get("res_scale", 1.0))
    net.load_state_dict(ck["ema_state"])
    net.eval()
    qlib.require_headless()                                  # torch is loaded now: it must see no CUDA device
    n, t = data.split_frames("test")
    pair = torch.tensor([i_a, i_b])
    with torch.no_grad():
        b = data.batch(n[pair], t[pair])                     # row 0 = frame A, row 1 = frame B
        out = net(b["Cn"], b["geo"], b["S"])
        swapped = out["C_bar"] + net.decode_r(out["z"], out["r"].flip(0), b["geo"], b["S"], out["C_bar"])   # own z, the other's r
        f = data.batch(torch.tensor([frame[0]]), torch.tensor([frame[1]]))
        check = net(f["Cn"], f["geo"], f["S"])

    def raw(x: "torch.Tensor") -> np.ndarray:
        return data.to_raw(x).clamp(0, 1).numpy()

    return {"checkpoint": str(path), "model": str(ck["model"]), "best_step": int(ck["best_step"]), "n_params": int(n_params(net)),
            "torch": torch.__version__, "n_unique_test_frames": int(data.frames["test"]["n_unique"]),
            "unique_example": n.numpy(), "unique_t": t.numpy(),          # the study's table of unique test frames
            "example": n[pair].numpy(), "t": t[pair].numpy(), "take": [str(data.frames["test"]["take"][i]) for i in (i_a, i_b)],
            "recorded": b["C"].numpy(), "z_only": raw(out["C_bar"]), "z_and_r": raw(out["C_hat"]), "swapped": raw(swapped),
            "frame_z_only": raw(check["C_bar"])[0], "frame_z_and_r": raw(check["C_hat"])[0]}


def run_forward(pick: PairPick, frame: FramePick, threads: int) -> dict[str, Any]:
    """stage1_forward in a fresh child process (one at a time)."""
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(stage1_forward, pick.dataset, pick.i_a, pick.i_b, (frame.example, frame.t), threads).result()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def swap_scores(fwd: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The label-free columns of the study's swap table (eval_swap.py), recomputed from the maps drawn here.
    Swap AB: z donor = A (row 0), r donor = B (row 1); swap BA: the reverse."""
    c, bar, own, swapped = fwd["recorded"], fwd["z_only"], fwd["z_and_r"], fwd["swapped"]
    scores = {}
    for name, zd, rd in (("AB", 0, 1), ("BA", 1, 0)):
        delta = swapped[zd] - bar[zd]                        # the correction after the swap
        scores[name] = {"detail_cos_rdonor": cosine(delta, c[rd] - bar[rd]), "detail_cos_zdonor": cosine(delta, c[zd] - bar[zd]),
                        "dense_change": float(np.linalg.norm(swapped[zd] - own[zd])), "dense_AB": float(np.linalg.norm(c[zd] - c[rd])),
                        "dense_to_zdonor": float(np.linalg.norm(swapped[zd] - c[zd])),
                        "dense_to_rdonor": float(np.linalg.norm(swapped[zd] - c[rd]))}
    return scores


# ------------------------------------------------------------------------------ what one dataset contributes
@dataclass
class Drawn:
    """Everything one dataset contributes: its reconstruction frame and its swap pair."""
    frame: FramePick
    example: qlib.Example             # the sequence of the reconstruction frame
    pair: PairPick
    forward: dict[str, Any]           # stage1_forward of the pair
    a: qlib.Example                   # the sequences of frames A and B
    b: qlib.Example
    ta: int                           # their sequence frames
    tb: int
    recomputed: dict[str, dict[str, float]]

    @property
    def recorded(self) -> np.ndarray:
        """(512,) recorded map of the reconstruction frame."""
        return self.example.C[self.frame.t]

    @property
    def changes(self) -> dict[str, np.ndarray]:
        """Per swap, (512,) what the exchanged r changes: the swapped map minus the map decoded from the same z with
        the frame's own r (the difference whose L2 norm is the study's ``dense_change``)."""
        return {s: self.forward["swapped"][k] - self.forward["z_and_r"][k] for k, s in enumerate(SWAPS)}


def load_drawn(frame: FramePick, pair: PairPick, fwd: dict[str, Any]) -> Drawn:
    ds = frame.dataset
    example = qlib.load_example(ds, frame.example)
    a, b = (qlib.load_example(ds, int(n)) for n in fwd["example"])
    ta, tb = (int(t) for t in fwd["t"])
    assert example.meta.split.startswith("test") and a.meta.split.startswith("test") and b.meta.split.startswith("test")
    assert a.meta.mesh_id == b.meta.mesh_id == str(pair.swaps["AB"].mesh) and a.meta.sequence_id != b.meta.sequence_id
    assert fwd["take"] == [str(pair.swaps[s].take_z) for s in SWAPS], "the pair's frames are not the takes of the swap table"
    assert np.array_equal(fwd["recorded"], np.stack([a.C[ta], b.C[tb]])), "the model's input is not the recorded map"
    # the frames ranked here are the study's unique test frames, the same occurrence of each
    ranked = (np.repeat(frame.examples, qlib.T) * qlib.T + np.tile(np.arange(qlib.T), len(frame.examples)))[frame.unique.ravel()]
    assert np.array_equal(ranked, fwd["unique_example"] * qlib.T + fwd["unique_t"]), "not the study's table of unique test frames"
    return Drawn(frame, example, pair, fwd, a, b, ta, tb, swap_scores(fwd))


def object_words(meta: qlib.Meta) -> str:
    return OBJECT_WORDS.get(meta.category, meta.category)


def two(value: float) -> str:
    """Two decimals for labels and the caption (no minus sign on a rounded zero)."""
    text = f"{value:.2f}"
    return "0.00" if text == "-0.00" else text


def example_record(part: str, e: qlib.Example, t: int, **extra: Any) -> dict[str, Any]:
    m = e.meta
    return {"part": part, "dataset": m.dataset, "example": m.example, "sequence_id": m.sequence_id,
            "object": f"{m.category} (mesh {m.mesh_id}, role {m.role})", "hand": m.hand, "frames_drawn": [t],
            "take_frames": [e.take_frame(t)], "split": m.split, **extra}


def examples_of(d: Drawn) -> list[dict[str, Any]]:
    return [example_record("reconstruction", d.example, d.frame.t, row_of_saved_file=d.frame.row),
            example_record("swap, frame A", d.a, d.ta, pair=d.pair.pair, opening_angle_rad=d.a.articulation(d.ta)),
            example_record("swap, frame B", d.b, d.tb, pair=d.pair.pair, opening_angle_rad=d.b.articulation(d.tb))]


def number(label: str, value: float, dataset: str, source: Path | str, locator: str) -> dict[str, Any]:
    return {"label": label, "value": float(value), "dataset": dataset, "source": str(source), "locator": locator}


def study_row(path: Path) -> pd.Series:
    """The row of the drawn model in one of the study's summary tables."""
    table = pd.read_csv(path)
    return table[table.model == MODEL].iloc[0]


def numbers_of(d: Drawn) -> list[dict[str, Any]]:
    """Every number a caption could quote for one dataset, with the file and the row or key it comes from."""
    fr, pk, ds = d.frame, d.pair, d.frame.dataset
    at = f"[{fr.row}, {fr.t}] (row of example {fr.example}, sequence frame {fr.t})"
    seq, recon = qlib.HIER[ds] / "sequences.npz", STUDY / ds / "reconstruction.csv"
    study = study_row(recon)
    out = [
        number("error of the z-only map in the drawn frame (L2 over the 512 map values), as saved", fr.e_zonly, ds, fr.path, "e_zonly" + at),
        number("error of the z + r map in the drawn frame (L2 over the 512 map values), as saved", fr.e_full, ds, fr.path, "e_full" + at),
        number("error of the z-only map in the drawn frame, recomputed from the drawn arrays", np.linalg.norm(fr.z_only - d.recorded), ds, SCRIPT,
               f"norm(C_bar{at} of {fr.path} - C[{fr.example}, {fr.t}] of {seq}); panels.npz recon_{ds}/z_only_c512, recorded_c512"),
        number("error of the z + r map in the drawn frame, recomputed from the drawn arrays", np.linalg.norm(fr.z_and_r - d.recorded), ds, SCRIPT,
               f"norm(C_hat{at} of {fr.path} - C[{fr.example}, {fr.t}] of {seq}); panels.npz recon_{ds}/z_and_r_c512, recorded_c512"),
        number("L2 norm of the recorded map of the drawn frame", np.linalg.norm(d.recorded), ds, SCRIPT,
               f"norm(C[{fr.example}, {fr.t}]) of {seq}; panels.npz recon_{ds}/recorded_c512"),
        number("number of values of a contact map", d.recorded.size, ds, seq, "C.shape[-1]"),
        number("largest absolute value of what r adds, on the 512 map values of the drawn frame", np.abs(fr.z_and_r - fr.z_only).max(), ds, SCRIPT,
               f"max |C_hat - C_bar|{at} of {fr.path}; panels.npz recon_{ds}/r_adds_c512"),
        number("median z-only error over the test frames (numpy median over the unique frames)", fr.evidence["numpy_median"], ds, fr.path,
               "median(e_zonly[first occurrences])"),
        number("number of test frames (unique frames: overlapping sequences counted once)", fr.evidence["n_candidates"], ds, SCRIPT,
               f"first occurrences of (take_key, group, t0 + t) over the test sequences of {qlib.HIER[ds] / 'sequences_meta.csv'}"),
        number("number of entries of the saved test file (sequences x 64)", fr.evidence["n_saved_entries"], ds, fr.path, "e_zonly.size"),
        number("share of test frames with a smaller z + r error than the drawn frame",
               fr.evidence["share_of_unique_frames_with_smaller_z_and_r_error"], ds, fr.path, f"mean(e_full[first occurrences] < e_full{at})"),
        number("mean z-only error over the unique test frames (study table)", study.E_zonly, ds, recon, f"row model == {MODEL}, column E_zonly"),
        number("mean z + r error over the unique test frames (study table)", study.E_full, ds, recon, f"row model == {MODEL}, column E_full"),
        number("mean z-only error over the unique test frames, recomputed", fr.errors["e_zonly"][fr.unique].mean(), ds, fr.path,
               "mean(e_zonly[first occurrences])"),
        number("mean z + r error over the unique test frames, recomputed", fr.errors["e_full"][fr.unique].mean(), ds, fr.path,
               "mean(e_full[first occurrences])"),
        number("detail score of the drawn pair (mean over its two swaps)", pk.evidence["value"], ds, pk.path,
               f"mean of column detail_cos_rdonor over rows pair == {pk.pair}"),
        number("median detail score over the controlled pairs (numpy median)", pk.evidence["numpy_median"], ds, pk.path,
               "median over pairs of the pair mean of detail_cos_rdonor"),
        number("number of controlled pairs", pk.evidence["n_candidates"], ds, pk.path, "number of distinct values of column pair"),
    ]
    for s, said in SWAPS.items():
        for col, words in SCORE_WORDS.items():
            out.append(number(f"{words} ({said}), study table", pk.swaps[s][col], ds, pk.path, f"row pair == {pk.pair}, swap == {s}, column {col}"))
        for col in RECOMPUTED:
            out.append(number(f"{SCORE_WORDS[col]} ({said}), recomputed on the CPU from the drawn maps", d.recomputed[s][col], ds, SCRIPT,
                              f"definition of eval_swap.py on panels.npz swap_{ds}/frame_A_c512, frame_B_c512, z_A_r_B_c512, z_B_r_A_c512, "
                              "z_only_c512_A/B, z_and_r_c512_A/B"))
        out.append(number(f"largest absolute change by the exchanged r, on the 512 map values ({said})", np.abs(d.changes[s]).max(), ds, SCRIPT,
                          f"max |swapped map - own z + r map|; panels.npz swap_{ds}_change/change_{CHANGE_KEYS[s]}_c512"))
    out.append(number("largest absolute change by the exchanged r over the two swaps of the pair, on the 512 map values",
                      max(np.abs(c).max() for c in d.changes.values()), ds, SCRIPT,
                      f"max over panels.npz swap_{ds}_change/change_z_A_r_B_c512 and change_z_B_r_A_c512"))
    metrics = study_row(STUDY / ds / "swap_metrics.csv")
    for col, words in (("frac_struct_follows_z", "share of all swaps whose structure is closer to the z donor"),
                       ("frac_detail_follows_r", "share of all swaps whose detail is closer to the r donor")):
        out.append(number(words + " (study table)", metrics[col], ds, STUDY / ds / "swap_metrics.csv", f"row model == {MODEL}, column {col}"))
    return out


def checks_of(d: Drawn) -> dict[str, Any]:
    """The drawn arrays against what the study saved (differences, not hidden): saved errors against errors of
    the saved maps, the study's test means against this script's unique-frame means, the float32 CPU run against
    the saved bfloat16 GPU maps, and the swap table against the scores of the maps drawn here."""
    fr, fwd = d.frame, d.forward
    with np.load(fr.path) as z:
        rows = [(e.test_row(z["n"]), t) for e, t in ((d.a, d.ta), (d.b, d.tb))]
        saved_bar = np.stack([z["C_bar"][r, t] for r, t in rows]).astype(np.float32)
        saved_hat = np.stack([z["C_hat"][r, t] for r, t in rows]).astype(np.float32)
    study = study_row(STUDY / fr.dataset / "reconstruction.csv")
    unique = {"this_script": fr.evidence["n_candidates"], "study_loader": fwd["n_unique_test_frames"],
              "same_frames_in_the_same_order": True}          # asserted in load_drawn
    assert unique["this_script"] == unique["study_loader"], unique
    return {
        "saved_error_minus_recomputed": {"z_only": fr.e_zonly - float(np.linalg.norm(fr.z_only - d.recorded)),
                                         "z_and_r": fr.e_full - float(np.linalg.norm(fr.z_and_r - d.recorded))},
        "study_mean_minus_recomputed_over_unique_frames": {"z_only": float(study.E_zonly - fr.errors["e_zonly"][fr.unique].mean()),
                                                           "z_and_r": float(study.E_full - fr.errors["e_full"][fr.unique].mean())},
        "unique_test_frames": unique,
        "cpu_float32_minus_saved_maps_max_abs": {"reconstruction_frame_z_only": float(np.abs(fwd["frame_z_only"] - fr.z_only).max()),
                                                 "reconstruction_frame_z_and_r": float(np.abs(fwd["frame_z_and_r"] - fr.z_and_r).max()),
                                                 "swap_frames_z_only": float(np.abs(fwd["z_only"] - saved_bar).max()),
                                                 "swap_frames_z_and_r": float(np.abs(fwd["z_and_r"] - saved_hat).max())},
        "swap_table_minus_recomputed": {s: {col: float(d.pair.swaps[s][col] - d.recomputed[s][col]) for col in RECOMPUTED} for s in SWAPS},
    }


# ------------------------------------------------------------------------------ panels and layout
@dataclass
class Panel:
    key: str
    example: qlib.Example
    t: int
    c512: np.ndarray                  # (512,) the canonical map (or difference of two maps) behind the panel
    kind: str                         # "contact" or "difference"
    with_hand: bool = False
    compare: str | None = None        # recorded panels: key of the decoded panel of the row it is read against

    @property
    def values(self) -> np.ndarray:
        """(V,) what is drawn: the map on the mesh vertices by the page's one display rule."""
        return self.example.to_vertices(self.c512)


@dataclass
class Row:
    key: str
    label: str
    columns: Sequence[str]
    panels: list[Panel | None]        # None = an empty cell
    view: str | None = None           # key of the row whose camera and panel size this row reuses


def reconstruction_row(d: Drawn) -> Row:
    fr, ex = d.frame, d.example
    label = (f"{NAME[fr.dataset]}\n{object_words(ex.meta)}, {HAND_WORDS[ex.meta.hand]}\n"
             f"error {two(fr.e_zonly)} with z alone, {two(fr.e_full)} with z and r")
    return Row(f"recon_{fr.dataset}", label, RECON_COLUMNS, [
        Panel("recorded", ex, fr.t, d.recorded, "contact", with_hand=True, compare="z_only"),
        Panel("z_only", ex, fr.t, fr.z_only, "contact"),
        Panel("r_adds", ex, fr.t, fr.z_and_r - fr.z_only, "difference"),
        Panel("z_and_r", ex, fr.t, fr.z_and_r, "contact")])


def swap_row(d: Drawn) -> Row:
    """A swapped map is decoded on the geometry of the frame that gave z, so it is drawn on that frame's object."""
    a, b, fwd = d.a.meta, d.b.meta, d.forward
    hands = HAND_WORDS[a.hand] if a.hand == b.hand else f"A {HAND_WORDS[a.hand]}, B {HAND_WORDS[b.hand]}"
    return Row(f"swap_{a.dataset}", f"{NAME[a.dataset]}\n{object_words(a)}, {hands}\nA and B from two takes", SWAP_COLUMNS, [
        Panel("frame_A", d.a, d.ta, fwd["recorded"][0], "contact", with_hand=True, compare="z_A_r_B"),
        Panel("frame_B", d.b, d.tb, fwd["recorded"][1], "contact", with_hand=True, compare="z_B_r_A"),
        Panel("z_A_r_B", d.a, d.ta, fwd["swapped"][0], "contact"),
        Panel("z_B_r_A", d.b, d.tb, fwd["swapped"][1], "contact")])


def change_row(d: Drawn) -> Row:
    """Under the two swapped maps of a pair: what the exchanged r changes, with the camera of the swap row. The two
    cells under the recorded frames stay empty (a recorded frame has no such difference)."""
    ds = d.a.meta.dataset
    return Row(f"swap_{ds}_change", CHANGE_LABEL.format(dataset=NAME[ds]), SWAP_COLUMNS, [
        None, None,
        Panel(f"change_{CHANGE_KEYS['AB']}", d.a, d.ta, d.changes["AB"], "difference"),
        Panel(f"change_{CHANGE_KEYS['BA']}", d.b, d.tb, d.changes["BA"], "difference")], view=f"swap_{ds}")


def row_view(row: Row) -> tuple[qlib.Camera, tuple[int, int]]:
    """One camera and one panel size for a row: look onto the object's broad side from the side of the first
    panel's hand (qlib.hand_side_view), fitted to every mesh and hand the row shows. The panel is as wide as
    PANEL_WIDTH and as flat as the scene, within ASPECT_RANGE."""
    panels = [p for p in row.panels if p is not None]
    meshes = [p.example.mesh(p.t)[0] for p in panels]
    hands = [p.example.hand(p.t)[0] for p in panels if p.with_hand]
    direction, up = qlib.hand_side_view(meshes[0], hands[0])
    points = np.concatenate(meshes + hands)
    side = np.cross(up, direction)
    aspect = float(np.clip(np.ptp(points @ side) / np.ptp(points @ up), *ASPECT_RANGE))
    size = (PANEL_WIDTH, 2 * int(round(PANEL_WIDTH / aspect / 2)))
    return qlib.fit_camera(meshes + hands, direction, up, aspect=size[0] / size[1]), size


def depth_image(verts: np.ndarray, faces: np.ndarray, camera: qlib.Camera, size: tuple[int, int]) -> np.ndarray:
    """(height, width) distance of the mesh from the camera along the viewing axis, in metres; NaN where the mesh is
    not seen. A helper of this script (qlib.render returns colours only): the camera and window of qlib.render on
    the same software renderer, without anti-aliasing."""
    qlib.require_headless()
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])])
    try:
        if pl.render_window.GetClassName() != "vtkOSOpenGLRenderWindow":
            raise RuntimeError("VTK did not select the OSMesa window: " + pl.render_window.GetClassName())
        faces = np.asarray(faces, np.int64)
        pl.add_mesh(pv.PolyData(np.asarray(verts, float), np.hstack([np.full((len(faces), 1), 3), faces])), color="white")
        pl.camera.position, pl.camera.focal_point, pl.camera.up = camera.position, camera.focal_point, camera.up
        pl.camera.view_angle = camera.view_angle
        pl.reset_camera_clipping_range()
        pl.screenshot(None, return_img=True)                 # the first render, as in qlib.render
        depth = -np.asarray(pl.get_image_depth(fill_value=np.nan, reset_camera_clipping_range=False), float)
    finally:
        pl.close()
    return depth


def hand_lines(obj: tuple[np.ndarray, np.ndarray], hand: tuple[np.ndarray, np.ndarray], camera: qlib.Camera,
               size: tuple[int, int]) -> dict[str, np.ndarray]:
    """The recorded hand as a line drawing for one panel: "hand_line" (height, width) opacity of the lines in
    [0, 1], "hand_mask" where the hand is in front of the object (or of the background), "object_mask" where the
    object is. Lines: the outline of the hand mask, and the pixels inside it where the hand's depth jumps by more
    than HAND_LINE["depth_jump_m"] between its two neighbours (a finger in front of another part of the hand)."""
    d_hand, d_obj = depth_image(*hand, camera, size), depth_image(*obj, camera, size)
    seen = np.isfinite(d_hand) & ~(d_obj <= d_hand)          # NaN (no object) compares False: the hand is seen
    depth = np.where(seen, d_hand, np.nan)
    jump = np.zeros_like(seen)
    jump[1:-1] |= np.abs(depth[2:] - depth[:-2]) > HAND_LINE["depth_jump_m"]
    jump[:, 1:-1] |= np.abs(depth[:, 2:] - depth[:, :-2]) > HAND_LINE["depth_jump_m"]
    line = (seen & ~ndimage.binary_erosion(seen, iterations=HAND_LINE["outline_px"])) | (jump & seen)
    alpha = np.clip(ndimage.gaussian_filter(line.astype(float), HAND_LINE["soften_px"]), 0, 1)
    return {"hand_line": alpha.astype(np.float32), "hand_mask": seen, "object_mask": np.isfinite(d_obj)}


def render_row(row: Row, camera: qlib.Camera, size: tuple[int, int], vmax: float, vabs: float) -> list[dict[str, np.ndarray]]:
    """Per cell the arrays that were drawn. "image" is the panel. A recorded panel also has "image_without_hand"
    (the render of the map alone, which the hand lines are laid over) and the arrays of hand_lines."""
    ink = np.array([int(HAND_LINE["colour"][i:i + 2], 16) for i in (1, 3, 5)], float)
    cells = []
    for p in row.panels:
        if p is None:
            cells.append({"image": np.full((size[1], size[0], 3), 255, np.uint8)})
            continue
        verts, faces = p.example.mesh(p.t)
        rgb = qlib.contact_rgb(p.values, vmax) if p.kind == "contact" else qlib.diverging_rgb(p.values, vabs)
        cell = {"image": qlib.render([qlib.mesh_item(verts, faces, rgb=rgb)], camera, size)}
        if p.with_hand:                                      # lines, so every pixel off a line is the map as rendered
            cell.update(hand_lines((verts, faces), p.example.hand(p.t), camera, size))
            plain, alpha = cell["image"], cell["hand_line"][..., None].astype(float)
            cell.update(image=(plain * (1 - alpha) + ink * alpha).round().astype(np.uint8), image_without_hand=plain)
        cells.append(cell)
    return cells


def hand_checks(row: Row, cells: Sequence[dict[str, np.ndarray]]) -> dict[str, dict[str, Any]]:
    """What the hand lines do to each recorded panel of a row, in 8-bit colour levels over the object's pixels:
    how many pixels they touch, that every other pixel is the hand-free render, and the panel's distance to the
    decoded panel it is read against (same object, same pose, same camera) with and without the lines."""
    by_key = {p.key: (p, c) for p, c in zip(row.panels, cells) if p is not None}
    out = {}
    for p, c in by_key.values():
        if not p.with_hand:
            continue
        other, other_cell = by_key[p.compare]
        assert other.example is p.example and other.t == p.t, "the compared panel shows another object pose"
        drawn, plain, decoded = (x.astype(float) for x in (c["image"], c["image_without_hand"], other_cell["image"]))
        obj, on_line = c["object_mask"], c["hand_line"] > 0
        coloured = (c["image_without_hand"] != 255).any(-1)  # the object in the colour render (anti-aliased rim included)
        out[p.key] = {
            "object_pixels": int(obj.sum()),
            "share_of_object_pixels_with_the_hand_in_front": float(c["hand_mask"][obj].mean()),
            "share_of_object_pixels_changed_by_the_lines": float((drawn != plain).any(-1)[obj].mean()),
            "mean_abs_change_by_the_lines_on_the_object": float(np.abs(drawn - plain)[obj].mean()),
            "max_abs_change_off_the_lines": float(np.abs(drawn - plain)[~on_line].max()),
            "compared_with": p.compare,
            "mean_abs_difference_to_the_compared_panel_as_drawn": float(np.abs(drawn - decoded)[obj].mean()),
            "mean_abs_difference_to_the_compared_panel_without_hand": float(np.abs(plain - decoded)[obj].mean()),
            "depth_and_colour_render_object_overlap_iou": float((obj & coloured).sum() / (obj | coloured).sum()),
        }
    return out


def round_up(value: float, step: float) -> float:
    return float(np.ceil(value / step - 1e-9) * step)


def stack(parts: Sequence[Sequence[Image.Image]]) -> Image.Image:
    """Put the one-row grids of each part under each other, with a thin rule between two parts. qlib.grid lays out
    panels of one size; the rows here differ in height, so each row is a grid of its own (same width, same font)."""
    layouts = [g.info["qlib_layout"] for part in parts for g in part]
    width, font_px = layouts[0]["width"], layouts[0]["font_px"]
    assert all(lay["width"] == width and lay["font_px"] == font_px for lay in layouts), "the row grids differ in width or font"
    canvas = Image.new("RGB", (width, sum(lay["height"] for lay in layouts)), "white")
    y, rules = 0, []
    for part in parts:
        rules.append(y)
        for g in part:
            canvas.paste(g, (0, y))
            y += g.height
    draw, margin = ImageDraw.Draw(canvas), layouts[0]["panel"][0] // 25
    for y in rules[1:]:
        draw.line((margin, y, width - margin, y), fill=qlib.RULE, width=max(2, width // 1100))
    canvas.info["qlib_layout"] = {**layouts[0], "height": canvas.height, "panel": [lay["panel"] for lay in layouts]}
    return canvas


def arrays_of(row: Row, cells: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """What a row drew, for panels.npz: per panel the surface values, the 512-value map, the geometry and the
    rendered arrays (image; for a recorded panel also image_without_hand, hand_line, hand_mask, object_mask)."""
    out: dict[str, np.ndarray] = {}
    for p, cell in zip(row.panels, cells):
        if p is None:
            continue
        verts, faces = p.example.mesh(p.t)
        out.update({f"{p.key}_surface": p.values, f"{p.key}_c512": p.c512,
                    f"{p.key}_verts": verts.astype(np.float32), f"{p.key}_faces": faces.astype(np.int32)})
        out.update({f"{p.key}_{name}": array for name, array in cell.items()})
        if p.with_hand:
            hand_verts, hand_faces = p.example.hand(p.t)
            out.update({f"{p.key}_hand_verts": hand_verts.astype(np.float32), f"{p.key}_hand_faces": hand_faces.astype(np.int32)})
    return out


# ------------------------------------------------------------------------------ words
def with_article(noun: str) -> str:
    """The noun with its indefinite article."""
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def caption_of(drawn: dict[str, Drawn], vmax: float, vabs: float) -> str:
    t, c = drawn["taco"], drawn["arctic"]

    def both(d: Drawn, col: str) -> str:
        return f"{two(d.pair.swaps['AB'][col])} and {two(d.pair.swaps['BA'][col])}"

    return (
        f"Top two rows: one test frame per dataset, selected as the frame whose error with z alone is the median over all test frames "
        f"(TACO: {with_article(object_words(t.example.meta))}; ARCTIC: {with_article(object_words(c.example.meta))}), shown from left as the recorded contact "
        f"with the recorded hand drawn as an outline, the map decoded from z alone, what r adds to it, and the map decoded from z and r. "
        f"In these two frames adding r lowers the error from {two(t.frame.e_zonly)} to {two(t.frame.e_full)} (TACO) and from "
        f"{two(c.frame.e_zonly)} to {two(c.frame.e_full)} (ARCTIC), as a distance over the {t.recorded.size} map values (the recorded maps "
        f"have norms {two(np.linalg.norm(t.recorded))} and {two(np.linalg.norm(c.recorded))}); the difference panels use a scale of "
        f"±{vabs:g}, about {vmax / vabs:.0f} times finer than the contact scale. "
        f"Lower rows: one swap pair per dataset, selected as the pair whose detail score, averaged over its two swaps, is the median over "
        f"the controlled pairs ({two(t.pair.evidence['value'])} in the TACO pair, {with_article(object_words(t.a.meta))}; "
        f"{two(c.pair.evidence['value'])} in the ARCTIC pair, {with_article(object_words(c.a.meta))}): frames A and B as recorded, the z of A "
        f"decoded with the r of B and the z of B decoded with the r of A, and under each swapped map what the exchanged r changes "
        f"(the swapped map minus the map decoded from the same z with that frame's own r). "
        f"Each swapped map keeps the layout of the frame that gave z: the study's structure distance to that frame is "
        f"{both(t, 'struct_to_zdonor')} (TACO) and {both(c, 'struct_to_zdonor')} (ARCTIC), against "
        f"{both(t, 'struct_to_rdonor')} (TACO) and {both(c, 'struct_to_rdonor')} (ARCTIC) to the frame that gave r. "
        f"This is expected from the model's form: a swapped map is the z donor's z-only map plus a small correction decoded from r, "
        f"so the swap tests what r carries, not whether z holds the layout. "
        f"What the exchanged r changes is small, a distance of {both(t, 'dense_change')} (TACO) and {both(c, 'dense_change')} (ARCTIC) "
        f"over the {t.recorded.size} map values where the two recorded frames are {two(t.pair.swaps['AB'].dense_AB)} and "
        f"{two(c.pair.swaps['AB'].dense_AB)} apart; its detail score, the cosine between the correction after the swap and the detail "
        f"that the r donor's own z-only map misses, is {both(t, 'detail_cos_rdonor')} for the two swaps of the TACO pair and "
        f"{both(c, 'detail_cos_rdonor')} for those of the ARCTIC pair.")


def meta_of(drawn: dict[str, Drawn], rows: Sequence[Row], cameras: dict[str, Any], numbers: list[dict[str, Any]],
            hand_check: dict[str, Any], vmax: float, vabs: float, threads: int) -> dict[str, Any]:
    t, c = drawn["taco"], drawn["arctic"]
    percentiles = " and ".join(f"{100 * d.frame.evidence['share_of_unique_frames_with_smaller_z_and_r_error']:.0f} ({NAME[ds]})"
                               for ds, d in drawn.items())
    details = "; ".join(f"{NAME[ds]} " + " and ".join(two(d.pair.swaps[s].detail_cos_rdonor) for s in SWAPS) for ds, d in drawn.items())
    n_panels = sum(p is not None for row in rows for p in row.panels)
    touched = [v["share_of_object_pixels_changed_by_the_lines"] for v in hand_check.values()]
    meshes = {f"{NAME[ds]} {object_words(e.meta)}": e for ds, d in drawn.items() for e in (d.example, d.a)}   # A and B share a mesh
    unreached = "; ".join(f"{name} {100 * (1 - e.covered.mean()):.0f} %" for name, e in meshes.items() if not e.covered.all())
    evidence = {}
    for ds, d in drawn.items():
        evidence[ds] = {
            "reconstruction_frame": {"picked": {"example": d.frame.example, "t": d.frame.t, "row_of_saved_file": d.frame.row}, **d.frame.evidence},
            "swap_pair": {"picked": {"pair": d.pair.pair, "iA": d.pair.i_a, "iB": d.pair.i_b,
                                     "frame_A": {"example": d.a.meta.example, "t": d.ta}, "frame_B": {"example": d.b.meta.example, "t": d.tb}},
                          **d.pair.evidence},
            "sorted_metrics": f"panels.npz selection_{ds}/e_zonly_unique_sorted, selection_{ds}/pair_detail_score_sorted"}
    return {
        "id": FIG_ID, "block": BLOCK, "title": TITLE,
        "panels": {
            "layout": "Six rows of four cells. Rows 1-2: reconstruction of one test frame (TACO, ARCTIC). Rows 3-6: one latent swap between "
                      "two test frames of the same object per dataset (rows 3-4 TACO, rows 5-6 ARCTIC): a row with the two recorded frames "
                      "and the two swapped maps, and under it a row with what the exchanged r changes in each swapped map (the two cells "
                      "under the recorded frames are empty). Every panel of a row has the same camera; a change row has the camera of the "
                      "swap row above it.",
            "reconstruction_columns": {
                RECON_COLUMNS[0]: "the recorded 512-value contact map of the frame on the object, with the recorded MANO hand as a line drawing",
                RECON_COLUMNS[1]: "the map the study saved for the frame, decoded from the latent z alone (C_bar)",
                RECON_COLUMNS[2]: "the saved z + r map minus the saved z-only map (C_hat - C_bar), on the difference scale",
                RECON_COLUMNS[3]: "the map the study saved for the frame, decoded from z and r (C_hat)"},
            "swap_columns": {
                SWAP_COLUMNS[0]: "recorded contact map and recorded hand (line drawing) of frame A, on the object as posed in frame A",
                SWAP_COLUMNS[1]: "recorded contact map and recorded hand (line drawing) of frame B, on the object as posed in frame B",
                SWAP_COLUMNS[2]: "computed here: z-only map of A plus the correction decoded from the z of A and the r of B; on the object as posed in frame A",
                SWAP_COLUMNS[3]: "computed here: z-only map of B plus the correction decoded from the z of B and the r of A; on the object as posed in frame B"},
            "change_rows": {
                "label": CHANGE_LABEL.format(dataset="<dataset>").replace("\n", " | "),
                "reference": "each swap frame's own z + r map (computed here, not drawn; panels.npz swap_<ds>/z_and_r_c512_A/B)",
                SWAP_COLUMNS[2]: "computed here: the swapped map (z of A, r of B) minus the map decoded from the z of A and the r of A, on the "
                                 "difference scale; on the object as posed in frame A",
                SWAP_COLUMNS[3]: "computed here: the swapped map (z of B, r of A) minus the map decoded from the z of B and the r of B, on the "
                                 "difference scale; on the object as posed in frame B",
                "empty_cells": "the two cells under the recorded frames (no such difference exists for a recorded frame)"},
            "row_labels": [row.label.replace("\n", " | ") for row in rows],
            "arrays": "panels.npz: <row>/<panel>_surface (drawn per-vertex values), _c512 (the 512-value map behind it), _image (the panel as "
                      "drawn), _verts, _faces; for a recorded panel also _image_without_hand (the render the hand lines are laid over), "
                      "_hand_line (opacity of the lines), _hand_mask (hand in front of the object or background), _object_mask, _hand_verts, "
                      "_hand_faces; swap_<ds>/z_only_c512_A/B and z_and_r_c512_A/B (the model's own maps of the swap frames, not drawn); "
                      "swap_<ds>_change/change_z_A_r_B_* and change_z_B_r_A_* (the change panels); selection_<ds>/e_zonly_unique_sorted and "
                      "pair_detail_score_sorted (the ranked selection metrics).",
        },
        "examples": [e for d in drawn.values() for e in examples_of(d)],
        "selection_rule": (
            "Reconstruction rows: per dataset, the test frame whose saved z-only reconstruction error (e_zonly) is the lower median over the "
            "unique test frames (every frame of every test sequence, a frame shared by two overlapping sequences of one take counted once; "
            "rank (N - 1) // 2 in ascending order, ties in file order). "
            f"Swap rows: per dataset, the controlled pair whose detail score (mean of detail_cos_rdonor over the pair's two swaps in "
            f"swap_pairs_{MODEL}.csv) is the lower median over all controlled pairs (ties: first in file)."),
        "selection_evidence": evidence,
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (qlib.Example.to_vertices; the same rule for recorded maps, decoded maps and the differences)",
            "contact_scale": {"range": [0.0, vmax], "colours": "grey at 0, orange at half, dark red-brown at the limit",
                              "rule": f"largest drawn contact value on the surface over all rows, rounded up to {CONTACT_STEP:g}"},
            "difference_scale": {"range": [-vabs, vabs], "colours": "blue = less contact, grey = no change, orange = more contact",
                                 "rule": "largest drawn absolute difference on the surface over all difference panels (reconstruction and "
                                         f"swap), rounded up to {DIFFERENCE_STEP:g}",
                                 "panels": "the third column of rows 1-2 (what r adds) and the two panels of rows 4 and 6 (what the "
                                           "exchanged r changes)"},
            "camera": {"rule": "per row: qlib.hand_side_view(mesh and hand of the first panel) (onto the object's broad side, from the side of that "
                               "hand), then qlib.fit_camera on every mesh and hand the row shows; the same camera for the four panels; a change "
                               "row reuses the camera of its swap row",
                       "panel_size_rule": f"width {PANEL_WIDTH} px; width / height = projected scene width / height, kept within {list(ASPECT_RANGE)}",
                       "per_row": cameras},
            "hand": {"style": "the recorded MANO hand of the frame as a line drawing laid over the render of the map alone, in the recorded "
                              "panels only; decoded maps and differences are drawn without a hand",
                     "lines": f"colour {HAND_LINE['colour']}; the {HAND_LINE['outline_px']} px outline of the part of the hand that is in "
                              "front of the object from the row's camera (parts behind the object are not drawn), and the pixels inside it "
                              f"where the hand's depth jumps by more than {1000 * HAND_LINE['depth_jump_m']:g} mm between the two neighbouring "
                              f"pixels (a finger in front of another part of the hand); softened with a Gaussian of {HAND_LINE['soften_px']:g} px",
                     "why": "a filled or translucent hand repaints the contact map under it; with lines, every pixel that is not on a line is "
                            "the pixel of the hand-free render (checks.hand_lines), so the recorded panel is comparable with the decoded "
                            "panels beside it",
                     "how": "depth_image of this script: two extra renders per recorded panel (hand alone, object alone) with the camera and "
                            "window of qlib.render on the same software renderer; qlib.render returns colours only",
                     "parameters": HAND_LINE},
            "object_pose": "the contacted object's own frame; an ARCTIC object is drawn with the opening angle of the frame whose geometry the map was decoded on",
            "layout": "one qlib.grid per row, stacked by this script (rows differ in height), a thin rule between the two parts; an empty cell "
                      "is a white panel",
        },
        "numbers": numbers,
        "inference": {
            "checkpoints": {ds: d.forward["checkpoint"] for ds, d in drawn.items()},
            "weights": "ema_state (the weights the study evaluated; best validation step "
                       + ", ".join(f"{NAME[ds]} {d.forward['best_step']}" for ds, d in drawn.items()) + ")",
            "model": f"Stage-1 factorised autoencoder, z and r of 64 dimensions each, {t.forward['n_params']} parameters, one training seed (0)",
            "device": f"cpu, float32, torch {t.forward['torch']}, {threads} threads; one child process per dataset, run one after the other",
            "seed": "none needed: the encoders and decoders are deterministic in evaluation mode",
            "newly_computed": "rows 3-6: the two swapped maps of each pair and each swap frame's own z + r map (the reference of the change "
                              "panels); for checks only, the z-only maps of the swap frames and both maps of the reconstruction frames",
            "read_from_saved": f"rows 1-2: C_bar and C_hat of <ds>/latents/{RUN}_test.npz (float16, clamped to [0, 1], written by the study on a GPU "
                               f"in bfloat16); every recorded map: sequences.npz of the hierarchical study; swap scores: swap_pairs_{MODEL}.csv",
            "study_code_used": [str(STUDY_SCRIPTS / f) for f in ("cf_common.py", "cf_data.py", "cf_models.py")],
        },
        "checks": {**{ds: checks_of(d) for ds, d in drawn.items()}, "hand_lines": hand_check},
        "caveats": [
            "One frame and one pair per dataset, from a model trained with one seed. Each is at the median of its selection metric, so it is typical "
            f"in that metric only; the z + r errors of the two reconstruction frames are at percentile {percentiles} of the test frames.",
            "The frame is ranked over the unique test frames, as the study's tables are. Ranking over every entry of the saved file (a frame "
            "shared by two overlapping sequences counted twice) picks another frame; it is listed in selection_evidence "
            "(other_reading_median_over_every_saved_entry).",
            f"A pair has two swaps with different detail scores; the pair is at the median of their mean, the single swaps are not ({details}).",
            "A swapped map is the z donor's z-only map plus a small correction by construction, so its resemblance to the z donor is largely built "
            "into the architecture, and the structure distance is computed with the z donor's own hand-part labels. The informative half of "
            "the swap is what the exchanged r changes (rows 4 and 6) and its detail score.",
            "The change panels show the swapped map minus the map decoded with the frame's own r. They do not show whether the change points "
            "towards the r donor's detail; that is the detail score, a number, and it is near zero in the ARCTIC pair.",
            "The controlled pairs are the study's: two test frames on the same object mesh from different takes, with a small distance between "
            "their structure and wrench descriptors relative to other pairs on that mesh and a dense distance above the median.",
            f"The difference panels use a scale about {vmax / vabs:.0f} times finer than the contact panels; equal colour strength does not mean equal size.",
            f"The hand is a line drawing, not a surface: its lines replace the map on {100 * min(touched):.0f} to {100 * max(touched):.0f} % of the "
            "object's pixels in the recorded panels, parts of the hand behind the object are not drawn, and the drawing does not show how far "
            "the hand is from the surface.",
            "Every map is drawn through the smoothing display rule, so the surface values are smoother and lower than the 512 map values; the quoted "
            "errors, scores and largest changes are computed on the 512 values, not on the drawn surface.",
            "ARCTIC: the display operator is built on the closed object, so contact near the hinge is drawn faintly on both parts.",
            *([f"Mesh vertices that none of the 512 map points reaches have no value and are drawn pale violet (share of the vertices: {unreached}). "
               "The map cannot represent contact there, recorded or decoded."] if unreached else []),
            "The decoded and swapped maps come from contact maps; no hand is generated.",
            "z was trained with a relational loss built from the structure and wrench descriptors; the structure distances quoted for the swap use "
            "those same descriptors and the true hand-part labels of the reference frame, and are read from the study's table (not recomputed here).",
            "The swapped maps are a float32 CPU run; the study's tables come from a bfloat16 GPU run. The differences are listed under checks.",
        ],
        "suggested_caption": caption_of(drawn, vmax, vabs),
        "suggested_alt": (
            f"{n_panels} renders of four objects with contact painted on their surface: {with_article(object_words(t.example.meta))} and "
            f"{with_article(object_words(c.example.meta))} with the recorded contact and the outline of the hand, the map decoded from z alone, the small "
            f"correction r adds and the map decoded from z and r; then {with_article(object_words(t.a.meta))} and {with_article(object_words(c.a.meta))} with two "
            "recorded frames, the two maps obtained by exchanging r between them, each looking like the frame that gave z, and the small "
            "change the exchanged r makes in each."),
    }


# ------------------------------------------------------------------------------ the figure
def build(threads: int) -> dict[str, Path]:
    qlib.require_headless()
    drawn: dict[str, Drawn] = {}
    for ds in DATASETS:                                      # one child process after the other
        frame, pair = select_frame(ds), select_pair(ds)
        log.info("%s: frame = example %d, t %d (rank %d of %d unique test frames, error with z alone %.4f); pair %d (rank %d of %d, "
                 "detail score %.4f)", NAME[ds], frame.example, frame.t, frame.evidence["rank_0_based"], frame.evidence["n_candidates"],
                 frame.e_zonly, pair.pair, pair.evidence["rank_0_based"], pair.evidence["n_candidates"], pair.evidence["value"])
        drawn[ds] = load_drawn(frame, pair, run_forward(pair, frame, threads))
        log.info("%s: model run on the CPU (%s)", NAME[ds], drawn[ds].forward["checkpoint"])

    parts = [[reconstruction_row(d) for d in drawn.values()], [row for d in drawn.values() for row in (swap_row(d), change_row(d))]]
    rows = [row for part in parts for row in part]
    panels = [p for row in rows for p in row.panels if p is not None]
    # one contact scale and one difference scale for the whole figure
    vmax = round_up(max(np.nanmax(p.values) for p in panels if p.kind == "contact"), CONTACT_STEP)
    vabs = round_up(max(np.nanmax(np.abs(p.values)) for p in panels if p.kind == "difference"), DIFFERENCE_STEP)
    not_reached = "not reached by the map" if any(not p.example.covered.all() for p in panels) else None
    bars = [qlib.colorbar("contact", 0.0, vmax, "contact on the object surface (0 = none)", nodata=not_reached),
            qlib.colorbar("diverging", -vabs, vabs, "difference panels: less or more contact")]

    grids, saved, cameras, views, hand_check = [], {}, {}, {}, {}
    for part in parts:                                       # one camera and one grid per row (the rows differ in height)
        grids.append([])
        for k, row in enumerate(part):
            camera, size = views[row.key] = views[row.view] if row.view else row_view(row)
            cells = render_row(row, camera, size, vmax, vabs)
            grids[-1].append(qlib.grid([[cell["image"] for cell in cells]], row_labels=[row.label],
                                       col_labels=list(row.columns) if k == 0 else None, colorbars=bars if row is rows[-1] else ()))
            cameras[row.key] = {"camera": camera, "panel_size": list(size), **({"same_as": row.view} if row.view else {})}
            saved[row.key] = arrays_of(row, cells)
            hand_check.update({f"{row.key}/{key}": value for key, value in hand_checks(row, cells).items()})
            log.info("%s rendered at %dx%d", row.key, *size)
    image = stack(grids)

    numbers = [n for d in drawn.values() for n in numbers_of(d)]
    numbers += [number("upper limit of the contact colour scale", vmax, "both", SCRIPT,
                       f"largest drawn contact value on the surface, rounded up to {CONTACT_STEP:g}"),
                number("limit of the difference colour scale (plus and minus)", vabs, "both", SCRIPT,
                       f"largest drawn absolute difference on the surface, rounded up to {DIFFERENCE_STEP:g}"),
                number("contact scale limit divided by the difference scale limit", vmax / vabs, "both", SCRIPT, "ratio of the two limits")]
    for key, check in hand_check.items():                    # key = <row>/<panel>, row = <part>_<dataset>
        ds, where = key.split("/")[0].split("_")[1], f"checks.hand_lines[{key}]: panels.npz {key}_image against {key}_image_without_hand over {key}_object_mask"
        numbers += [number(f"share of the object's pixels changed by the hand lines ({key})",
                           check["share_of_object_pixels_changed_by_the_lines"], ds, SCRIPT, where),
                    number(f"mean absolute colour change by the hand lines over the object's pixels, in 8-bit levels ({key})",
                           check["mean_abs_change_by_the_lines_on_the_object"], ds, SCRIPT, where)]
    for ds, d in drawn.items():                              # the model's own maps of the swap frames, and the ranked metrics
        for k, name in enumerate("AB"):
            saved[f"swap_{ds}"].update({f"z_only_c512_{name}": d.forward["z_only"][k], f"z_and_r_c512_{name}": d.forward["z_and_r"][k]})
        saved[f"selection_{ds}"] = {"e_zonly_unique_sorted": np.sort(d.frame.errors["e_zonly"][d.frame.unique], kind="stable"),
                                    "pair_detail_score_sorted": np.sort(d.pair.scores, kind="stable")}
    meta = meta_of(drawn, rows, cameras, numbers, hand_check, vmax, vabs, threads)
    paths = qlib.save(FIG_ID, image, saved, meta)

    for ds, d in drawn.items():
        log.info("%s reconstruction frame: error with z alone %.4f, with z and r %.4f (saved); norm of the recorded map %.3f",
                 NAME[ds], d.frame.e_zonly, d.frame.e_full, float(np.linalg.norm(d.recorded)))
        for s in SWAPS:
            row, own = d.pair.swaps[s], d.recomputed[s]
            log.info("%s pair %d, %s: structure distance to the z donor %.4f, to the r donor %.4f | detail cosine to the r donor %.4f "
                     "(here %.4f), to the z donor %.4f (here %.4f) | change by the swap %.4f (here %.4f, largest map value %.4f), "
                     "distance between the frames %.4f",
                     NAME[ds], d.pair.pair, SWAPS[s], row.struct_to_zdonor, row.struct_to_rdonor, row.detail_cos_rdonor, own["detail_cos_rdonor"],
                     row.detail_cos_zdonor, own["detail_cos_zdonor"], row.dense_change, own["dense_change"], float(np.abs(d.changes[s]).max()),
                     row.dense_AB)
        log.info("%s checks: %s", NAME[ds], meta["checks"][ds])
    for key, value in hand_check.items():
        log.info("hand lines, %s: %s", key, value)
    log.info("colour limits: contact 0 to %g, difference -%g to +%g", vmax, vabs, vabs)
    log.info("layout: %s", image.info["qlib_layout"])
    log.info("caption: %s", meta["suggested_caption"])
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--threads", type=int, default=2, help="CPU threads of the model run (the machine is shared)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for name, path in build(args.threads).items():
        log.info("%s: %s", name, path)


if __name__ == "__main__":
    main()
