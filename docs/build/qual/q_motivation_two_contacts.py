"""Qualitative figure ``motivation_two_contacts`` (page block MOTIVATION).

One TACO test window of this project's BimArt port (generation 3,
``taco/30_bimart_gen3_scene_scale``): the same motion model, run with the same noise, is given two
contact maps, the recorded one and the one the contact stage predicts. The figure shows, on the
real tool and target meshes at four frames,

* the two contact maps painted on the objects (rows 1 and 2), and
* the hand surface points the motion stage generates from each of them, 100 per hand (rows 3 and 4),

every panel over the recorded MANO hands drawn as translucent grey meshes.

The window is chosen by a rule computed here from the study's saved per-window table
(``contact_probe/test_2.csv``): the window of split test_2 whose increase of the mean hand-point
error (predicted contact minus recorded contact) is the median over all windows of the split.

The study never saved the motion stage's outputs, so the script samples both stages again on the
CPU with the study's own sampler (``scripts/eval_bimart_taco.py``: ``load_model``, ``sample``). That
is a NEW sample of the same models, not the sample behind the table; its two errors are recorded
next to the table's values. One difference from the study's scripts is deliberate: their ``seed``
fixes only the initial noise, and the noise of the later denoising steps comes from torch's global
generator, which they never seed. Here that generator is seeded before every run, so the figure is
reproducible and its two motion runs share all of their noise.

Only the sample with later-noise seed 0 is drawn. The same window is sampled five more times (both
stages, later-noise seeds 1 to 5, same initial noise) as an undrawn control: it says how far the
sampler's noise alone moves the hand, and where the drawn sample lies among six samples. The errors
of all six go to meta.json and panels.npz.

The recorded hands are drawn translucent with all of their faces. ``qlib.render`` draws only the
front faces of a translucent mesh; the MANO mesh is open at the wrist and the camera looks from the
wrist side, so each hand is passed a second time with reversed triangle winding (see
``recorded_hand_items``).

Checkpoints and data are opened read-only. Geometry is in the target object's rigid frame (the
frame the models work in), in metres.

Run (CPU, software renderer; see README.md):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_motivation_two_contacts.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import qlib

log = logging.getLogger("q_motivation_two_contacts")

FIG_ID = "motivation_two_contacts"
BLOCK = "MOTIVATION"
TITLE = "One test window, one motion model, the same noise, two contact maps"
DATASET = "taco"

# ------------------------------------------------------------------------------ the study
GEN3 = qlib.TACO_GEN3
SPLIT = "test_2"                                  # tool meshes held out; not the split that selected the checkpoints
WINDOW_TABLE = GEN3 / "contact_probe" / f"{SPLIT}.csv"                 # one row per (window, hand, frame)
SPLIT_TABLE = GEN3 / "evaluation" / "test_split_results.csv"           # the split means the page quotes
SAVED_MAPS = GEN3 / "contact_probe" / f"{SPLIT}_contact_maps.npz"      # the study's saved contact maps
SAVED_MAPS_INDEX = GEN3 / "contact_probe" / f"{SPLIT}_contact_maps_index.csv"
SEQUENCE_INDEX = GEN3 / "sequence_index.csv"
CONTACT_RUN = GEN3 / "experiments/contact_model_taco_scene_20260927_174803"
MOTION_RUN = GEN3 / "experiments/motion_model_taco_scene_20260927_174902"
CONTACT_CONFIG = qlib.REPO / "configs/bimart_taco/train_contact_config_scene.yaml"
MOTION_CONFIG = qlib.REPO / "configs/bimart_taco/train_motion_config_scene.yaml"
HAND_INDEX = qlib.REPO / "third_party/BimArt/assets/part_fps_hand_index_100.npy"   # the 100 MANO vertices per hand
SEED = 0                                          # initial noise of every run, and the later noise of the drawn runs
CONTROL_SEED = 1                                  # later noise of the control run
SPREAD_SEEDS = (0, 1, 2, 3, 4, 5)                 # later-noise seeds of the six samples of the window; only SEED is drawn
FAR_MM = 200.0                                    # a hand "starts far away" if its frame-0 error exceeds this (20 cm)
HORIZON = 64                                      # frames per window
HAND_KP = 100                                     # generated surface points per hand
SLOTS = 512                                       # contact-map entries per object and hand
FPS = 30.0                                        # TACO frame rate
SIDES = ("left", "right")                         # order of the hands in the models' arrays
HAND_SLICE = {"left": slice(0, HAND_KP), "right": slice(HAND_KP, 2 * HAND_KP)}   # a hand's points among the 200

# ------------------------------------------------------------------------------ display choices
FRAMES = (0, 21, 42, 63)                          # evenly spaced over the window
LATE = slice(FRAMES[1], HORIZON)                  # the window from the second drawn frame on
DISPLAY_SIGMA = 0.01                              # metres: Gaussian that spreads the map entries over the mesh
DISPLAY_RADIUS = 0.03                             # metres: a vertex farther than this from every entry has no value
CONTACT_VMAX = 1.0
POINT_RADIUS = 0.005                              # metres: radius of a drawn hand point
COLOR_FROM_RECORDED = qlib.BLUE                   # hand points generated from the recorded contact
COLOR_FROM_PREDICTED = qlib.PART_HEX[3]           # green: hand points generated from the predicted contact
RECORDED_HAND = {"color": "#93928c", "opacity": 0.4}    # per face layer (front and back faces are both drawn); darker
#                                                         than qlib.GHOST, which vanishes at this scene size
AZIMUTH_DEG = 45.0                                # camera turn about the vertical, from the hands' side
ELEVATION_DEG = 25.0                              # camera height above the horizontal plane
CAMERA_MARGIN = 0.04
PANEL = (480, 480)


# ------------------------------------------------------------------------------ selection
@dataclass(frozen=True)
class Selection:
    window: int
    rank: int                    # 1-based position in the ascending order of the error increase
    table: pd.DataFrame          # one row per window, ascending by gap_mm

    @property
    def row(self) -> pd.Series:
        return self.table.iloc[self.rank - 1]


def select_window(path: Path) -> Selection:
    """The window whose error increase (predicted minus recorded contact) is the median of the split.

    The table has one row per (window, hand, frame): ``motion_mm`` is the mean distance of that
    hand's 100 generated points from the recorded ones with the recorded contact as input,
    ``total_mm`` the same with the predicted contact. Their means over a window's 128 rows are the
    two per-window errors in the sense of ``eval_bimart_taco.py::score``.
    """
    rows = pd.read_csv(path, usecols=["window", "sequence_id", "start", "verb", "tool_cat", "target_cat", "tool_mesh",
                                      "target_mesh", "motion_mm", "total_mm", "contact_map_err_mm"])
    assert (rows.groupby("window").size() == 2 * HORIZON).all(), "a window does not have 2 hands x 64 frames"
    per = rows.groupby("window").agg(
        sequence_id=("sequence_id", "first"), start=("start", "first"), verb=("verb", "first"),
        tool=("tool_cat", "first"), target=("target_cat", "first"), tool_mesh=("tool_mesh", "first"),
        target_mesh=("target_mesh", "first"), recorded_mm=("motion_mm", "mean"), predicted_mm=("total_mm", "mean"),
        contact_map_err_mm=("contact_map_err_mm", "mean")).reset_index()
    per["gap_mm"] = per["predicted_mm"] - per["recorded_mm"]
    ordered = per.sort_values(["gap_mm", "window"]).reset_index(drop=True)
    position = (len(ordered) - 1) // 2                      # the middle window (the lower one if the count is even)
    if len(ordered) % 2:
        assert np.isclose(ordered["gap_mm"].iloc[position], per["gap_mm"].median())
    return Selection(int(ordered["window"].iloc[position]), position + 1, ordered)


def frame_profile(path: Path, window: int) -> dict[str, Any]:
    """Per window frame (64 values each): the split's mean error increase, and the chosen window's two errors in the
    study's table (mean over both hands); plus that window's four frame-0 rows by hand (side -> (recorded, predicted))."""
    rows = pd.read_csv(path, usecols=["window", "hand", "frame", "motion_mm", "total_mm"])
    split = rows.groupby("frame")[["motion_mm", "total_mm"]].mean().sort_index()
    mine = rows[rows["window"] == window]
    own = mine.groupby("frame")[["motion_mm", "total_mm"]].mean().sort_index()
    assert len(split) == len(own) == HORIZON
    first = mine[mine["frame"] == 0].set_index("hand")
    assert sorted(first.index) == ["L", "R"]
    return {"split_gap_mm": (split["total_mm"] - split["motion_mm"]).to_numpy(),
            "window_recorded_mm": own["motion_mm"].to_numpy(), "window_predicted_mm": own["total_mm"].to_numpy(),
            "window_frame0_by_hand_mm": {side: (float(first.loc[side[0].upper(), "motion_mm"]),
                                                float(first.loc[side[0].upper(), "total_mm"])) for side in SIDES}}


# ------------------------------------------------------------------------------ inference (CPU, read-only)
@dataclass(frozen=True)
class Sample:
    sequence_id: str
    start: int                         # take frame of window frame 0
    sequence_file: str
    tool_mesh: str
    target_mesh: str
    kp_recorded: np.ndarray            # (64, 200, 3) recorded hand points, left hand first, metres
    kp_from_recorded: np.ndarray       # (64, 200, 3) generated with the recorded contact as input
    kp_from_predicted: np.ndarray      # (64, 200, 3) generated with the predicted contact as input
    kp_control: np.ndarray             # (64, 200, 3) recorded contact again, other noise after the first step
    contact_recorded: np.ndarray       # (64, 2048) metres: left hand [tool 512, target 512], then right hand
    contact_predicted: np.ndarray      # (64, 2048) metres, the contact stage's sample
    spread_from_recorded: np.ndarray   # (6, 64, 200, 3) one sample per later-noise seed of SPREAD_SEEDS, recorded contact
    spread_from_predicted: np.ndarray  # (6, 64, 200, 3) the same seeds, each with its own predicted contact
    checkpoints: dict[str, Any]


def run_models(window: int) -> Sample:
    """Sample the contact stage and the motion stage for both contact maps, once per later-noise seed of SPREAD_SEEDS.
    The sample with later-noise seed SEED is the drawn one; the others are the undrawn control."""
    for path in (str(qlib.REPO), str(qlib.REPO / "third_party/BimArt")):
        if path not in sys.path:
            sys.path.insert(0, path)
    import torch
    import yaml
    from diffusers import DDPMScheduler

    from scripts.eval_bimart_taco import CFG_GUIDE_STRENGTH, load_model, sample   # the study's sampler
    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from utils import data_util                                                   # third_party/BimArt

    qlib.require_headless()                                   # torch is imported now: it must see no CUDA device
    torch.set_num_threads(1)
    device = "cpu"
    ccfg = yaml.safe_load(CONTACT_CONFIG.read_text())
    mcfg = yaml.safe_load(MOTION_CONFIG.read_text())
    assert Path(ccfg["base_dir"]) == GEN3 and Path(mcfg["data"]["base_dir"]) == GEN3
    steps = ccfg["num_timesteps"]
    with contextlib.redirect_stdout(io.StringIO()):           # the upstream helpers print every array shape
        contact_model, contact_tag, contact_epoch, _ = load_model("contact", CONTACT_RUN, ccfg, device)
        motion_model, motion_tag, motion_epoch, _ = load_model("motion", MOTION_RUN, mcfg, device)
        cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
        mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], device)
    scheduler = DDPMScheduler(num_train_timesteps=steps, beta_schedule="squaredcos_cap_v2", clip_sample=False,
                              prediction_type="sample")
    contact_data = TacoContactDataset(root=GEN3, split=SPLIT, pred_horizon=ccfg["pred_horizon"], base_frame=ccfg["base_frame"])
    motion_data = TacoMotionDataset(root=GEN3, split=SPLIT, pred_horizon=mcfg["data"]["pred_horizon"],
                                    base_frame=mcfg["data"]["base_frame"])
    row = motion_data.index.iloc[motion_data.windows[window]["file_idx"]]

    def batch_of_one(item: dict) -> dict:
        as_tensor = lambda a: torch.as_tensor(np.array(a))[None]            # noqa: E731
        return {"action": as_tensor(item["action"]), "obs": {k: as_tensor(v) for k, v in item["obs"].items()}}

    cn = data_util.preprocess_contact_batch(batch_of_one(contact_data[window]), device, cstat, True)
    mn = data_util.preprocess_batch(batch_of_one(motion_data[window]), mstat, device)
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]
    a_mean, a_std = mstat["action"]["mean"], mstat["action"]["std"]

    def draw(model: "torch.nn.Module", shape: "torch.Size", cond: dict, kind: str, later_seed: int = SEED) -> "torch.Tensor":
        """One run of the study's sampler. Its ``seed`` fixes only the initial noise; the noise the scheduler adds at
        the later steps comes from torch's global generator, which the study's scripts never seed. It is seeded
        here before every run, so the runs are reproducible and two runs with one ``later_seed`` share ALL noise."""
        torch.manual_seed(later_seed)
        return sample(model, shape, scheduler, device, cond, kind, steps, SEED)

    def hand_points(action_norm: "torch.Tensor") -> np.ndarray:
        """The first 600 of the 1200 output numbers per frame: 200 points x 3 (the rest are direction vectors)."""
        return (action_norm * a_std + a_mean)[0, :, :HAND_KP * 6].reshape(HORIZON, 2 * HAND_KP, 3).numpy()

    # stage 1: the contact map from the objects alone
    contact_predicted = draw(contact_model, cn["action"].shape, cn["obs"], "contact") * c_std + c_mean      # metres
    contact_recorded = cn["action"] * c_std + c_mean
    # stage 2, twice with identical noise: the recorded map, then the stage-1 map (in the motion model's units)
    given_recorded = dict(mn["obs"])
    given_predicted = dict(mn["obs"])
    given_predicted["contact_points"] = (contact_predicted - mstat["contact_points"]["mean"]) / mstat["contact_points"]["std"]
    x_recorded = draw(motion_model, mn["action"].shape, given_recorded, "motion")
    x_predicted = draw(motion_model, mn["action"].shape, given_predicted, "motion")
    # undrawn control: the whole pipeline again with other later noise (same initial noise) -> how far the sampler's
    # noise alone moves the hand, and where the drawn sample lies among the samples of this window
    spread_recorded, spread_predicted = {SEED: x_recorded}, {SEED: x_predicted}
    for later_seed in SPREAD_SEEDS:
        if later_seed == SEED:
            continue
        contact_again = draw(contact_model, cn["action"].shape, cn["obs"], "contact", later_seed) * c_std + c_mean
        given_again = dict(mn["obs"])
        given_again["contact_points"] = (contact_again - mstat["contact_points"]["mean"]) / mstat["contact_points"]["std"]
        spread_recorded[later_seed] = draw(motion_model, mn["action"].shape, given_recorded, "motion", later_seed)
        spread_predicted[later_seed] = draw(motion_model, mn["action"].shape, given_again, "motion", later_seed)
    x_control = spread_recorded[CONTROL_SEED]      # the recorded map again, same initial noise, other later noise

    return Sample(
        spread_from_recorded=np.stack([hand_points(spread_recorded[s]) for s in SPREAD_SEEDS]),
        spread_from_predicted=np.stack([hand_points(spread_predicted[s]) for s in SPREAD_SEEDS]),
        sequence_id=row["sequence_id"], start=int(motion_data.windows[window]["start"]), sequence_file=row["file"],
        tool_mesh=f"{int(row['tool_mesh']):03d}", target_mesh=f"{int(row['target_mesh']):03d}",
        kp_recorded=hand_points(mn["action"]), kp_from_recorded=hand_points(x_recorded),
        kp_from_predicted=hand_points(x_predicted), kp_control=hand_points(x_control),
        contact_recorded=contact_recorded[0].numpy(), contact_predicted=contact_predicted[0].numpy(),
        checkpoints={"contact_stage": {"file": CONTACT_RUN / "model_best.pth", "weights": contact_tag, "epoch": contact_epoch},
                     "motion_stage": {"file": MOTION_RUN / "model_best.pth", "weights": motion_tag, "epoch": motion_epoch},
                     "configs": [CONTACT_CONFIG, MOTION_CONFIG],
                     "sampler": "scripts/eval_bimart_taco.py::sample (50-step DDPM, x0-prediction)",
                     "classifier_free_guidance": CFG_GUIDE_STRENGTH, "contact_guidance": "off",
                     "initial_noise_seed": SEED, "later_noise_seed": SEED, "control_later_noise_seed": CONTROL_SEED,
                     "undrawn_later_noise_seeds": [s for s in SPREAD_SEEDS if s != SEED], "device": device, "threads": 1})


def point_distance_mm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(..., K, 3) x 2 -> (...,) mean distance between corresponding points, millimetres."""
    return np.linalg.norm(a - b, axis=-1).mean(-1) * 1000.0


def hand_error_mm(points: np.ndarray, recorded: np.ndarray) -> float:
    """Mean distance of all generated points of a window from the recorded ones (eval_bimart_taco.py::score)."""
    return float(np.linalg.norm(points - recorded, axis=-1).mean() * 1000.0)


# ------------------------------------------------------------------------------ recorded geometry
@dataclass(frozen=True)
class Scene:
    tool: np.ndarray                   # (64, N_tool, 3) tool vertices in the target frame
    target: np.ndarray                 # (N_target, 3) target vertices (fixed: its frame is the canonical one)
    tool_faces: np.ndarray
    target_faces: np.ndarray
    hands: dict[str, np.ndarray]       # side -> (64, 778, 3) recorded MANO vertices in the target frame
    up: np.ndarray                     # (3,) the world's vertical axis in the target frame at window frame 0
    slot_vertex: np.ndarray            # (64, 1024) vertex of each contact-map entry (tool vertices first)
    dense: dict[str, np.ndarray]       # side -> (64, N_tool + N_target) recorded distance to that hand, metres


def load_scene(sample: Sample) -> Scene:
    """The window's recorded objects and hands, built like the study's features (``src/analysis/bimart/features.py``)."""
    from src import geometry
    from src.analysis.action_structure import contact
    from src.analysis.bimart import features
    from src.analysis.loaders import taco

    triplet, sequence = sample.sequence_id.rsplit("/", 1)
    traj = taco.load(taco.SequenceRef(triplet, sequence, taco.parse_triplet(triplet)), with_hands=True)
    assert (traj.tool.name, traj.target.name) == (sample.tool_mesh, sample.target_mesh)
    meshes = np.load(GEN3 / "assets/taco_mesh_dict.npy", allow_pickle=True).item()
    tool_mesh, target_mesh = meshes[traj.tool.name], meshes[traj.target.name]
    window = slice(sample.start, sample.start + HORIZON)
    rot_target, pos_target = geometry.wxyz_to_R(traj.target.quat)[window], traj.target.pos[window]
    rot_tool, pos_tool = geometry.wxyz_to_R(traj.tool.quat)[window], traj.tool.pos[window]
    tool_world = np.einsum("tij,nj->tni", rot_tool, tool_mesh["verts_original"]) + pos_tool[:, None]
    hands = {}
    for side in SIDES:
        skin, _ = contact.hand_geometry(traj.hands[side], contact.mano_layer(side))       # MANO on CPU, world frame
        hands[side] = features.to_canonical(skin[window], rot_target, pos_target)
    with np.load(GEN3 / "sequences" / sample.sequence_file) as z:
        slot_vertex = z["obj_cano_bps_inds"][window].astype(np.int64)
        dense = {side: z[f"contact_{side}"][window] for side in SIDES}
    return Scene(tool=features.to_canonical(tool_world, rot_target, pos_target),
                 target=np.asarray(target_mesh["verts_original"], float),
                 tool_faces=np.asarray(tool_mesh["faces"], np.int64), target_faces=np.asarray(target_mesh["faces"], np.int64),
                 hands=hands, up=rot_target[0].T @ np.array([0.0, 0.0, 1.0]), slot_vertex=slot_vertex, dense=dense)


def check_scene(scene: Scene, sample: Sample) -> dict[str, float]:
    """The meshes drawn are the geometry the models were trained on: three checks against stored arrays (metres)."""
    index = np.load(HAND_INDEX).reshape(-1)
    points = np.concatenate([scene.hands[side][:, index] for side in SIDES], axis=1)
    checks = {"max |MANO vertices at the 100 point indices - stored recorded hand points|":
              float(np.abs(points - sample.kp_recorded).max())}
    worst = 0.0
    for t in FRAMES:                   # stored per-vertex distance to the hand = distance between the drawn meshes
        verts = np.concatenate([scene.tool[t], scene.target])
        for side in SIDES:
            dist = cKDTree(scene.hands[side][t]).query(verts)[0]
            worst = max(worst, float(np.abs(dist - scene.dense[side][t]).max()))
    checks["max |object-to-hand distance of the drawn meshes - stored dense distance|, drawn frames"] = worst
    rows = np.arange(HORIZON)[:, None]
    gathered = np.concatenate([scene.dense[side][rows, scene.slot_vertex] for side in SIDES], axis=1)
    checks["max |stored dense distance at the map's vertices - recorded contact map|"] = float(
        np.abs(gathered - sample.contact_recorded).max())
    assert all(v < 1e-4 for v in checks.values()), checks
    return checks


# ------------------------------------------------------------------------------ display rule for the contact map
def soft_contact(distance: np.ndarray) -> np.ndarray:
    """Distance to a hand in metres -> contact exp(-distance / 2 cm); a negative predicted distance counts as 0."""
    return np.exp(-np.clip(distance, 0.0, None) / qlib.SOFT_SCALE)


def nearer_hand(contact_map: np.ndarray) -> np.ndarray:
    """(…, 2048) map in metres -> (…, 1024) contact of the nearer hand per entry (tool 512, target 512)."""
    soft = soft_contact(contact_map)
    return np.maximum(soft[..., :2 * SLOTS], soft[..., 2 * SLOTS:])


def to_surface(verts: np.ndarray, slot_vertex: np.ndarray, slot_value: np.ndarray) -> np.ndarray:
    """One object's 512 map entries -> a value per mesh vertex.

    Entry k sits on vertex ``slot_vertex[k]``. A vertex u takes the Gaussian-weighted mean of the entries within
    DISPLAY_RADIUS: sum_k w_k v_k / sum_k w_k with w_k = exp(-|x_u - x_k|^2 / (2 DISPLAY_SIGMA^2)); NaN if there is none.
    """
    anchors = verts[slot_vertex]
    out = np.full(len(verts), np.nan)
    for u, near in enumerate(cKDTree(anchors).query_ball_point(verts, DISPLAY_RADIUS)):
        if near:
            near = np.asarray(near)
            w = np.exp(-0.5 * ((anchors[near] - verts[u]) ** 2).sum(1) / DISPLAY_SIGMA ** 2)
            out[u] = (w * slot_value[near]).sum() / w.sum()
    return out


def surface_contact(scene: Scene, contact_map: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex contact on (tool, target) at window frame t for a (64, 2048) map."""
    soft = nearer_hand(contact_map[t])
    n_tool = scene.tool.shape[1]
    return (to_surface(scene.tool[t], scene.slot_vertex[t, :SLOTS], soft[:SLOTS]),
            to_surface(scene.target, scene.slot_vertex[t, SLOTS:] - n_tool, soft[SLOTS:]))


def display_fidelity(scene: Scene, sample: Sample) -> dict[str, float]:
    """How close the displayed recorded map is to the exact recorded per-vertex contact, over the drawn frames."""
    shown, exact = [], []
    for t in FRAMES:
        shown.append(np.concatenate(surface_contact(scene, sample.contact_recorded, t)))
        exact.append(np.maximum(*(soft_contact(scene.dense[side][t]) for side in SIDES)))
    shown, exact = np.concatenate(shown), np.concatenate(exact)
    ok = ~np.isnan(shown)
    return {"correlation": float(np.corrcoef(shown[ok], exact[ok])[0, 1]),
            "mean_abs_difference": float(np.abs(shown[ok] - exact[ok]).mean()),
            "max_shown": float(shown[ok].max()), "max_exact": float(exact.max()),
            "share_of_vertices_without_value": float(1.0 - ok.mean())}


# ------------------------------------------------------------------------------ camera
def view_direction(scene: Scene, elevation_deg: float, azimuth_deg: float) -> np.ndarray:
    """Scene -> camera: horizontally from the objects towards the recorded hands (the side the person works from),
    turned by ``azimuth_deg`` about the vertical and raised by ``elevation_deg``."""
    up = scene.up / np.linalg.norm(scene.up)
    objects = np.concatenate([scene.tool[list(FRAMES)].reshape(-1, 3), scene.target])
    hands = np.concatenate([scene.hands[side][list(FRAMES)].reshape(-1, 3) for side in SIDES])
    towards = hands.mean(0) - objects.mean(0)
    towards -= (towards @ up) * up
    towards = Rotation.from_rotvec(np.radians(azimuth_deg) * up).apply(towards / np.linalg.norm(towards))
    return np.cos(np.radians(elevation_deg)) * towards + np.sin(np.radians(elevation_deg)) * up


# ------------------------------------------------------------------------------ numbers and cross-checks
@dataclass(frozen=True)
class Errors:
    """Hand-point errors of this sample, millimetres (the metric of eval_bimart_taco.py::score)."""
    recorded: float                    # whole window, recorded contact as input
    predicted: float                   # whole window, predicted contact as input
    frame_recorded: np.ndarray         # (64,) per frame, both hands
    frame_predicted: np.ndarray        # (64,)
    per_hand: dict[str, dict[str, float]]
    between: float                     # mean distance between the two generated hands
    noise_only: float                  # mean distance between the drawn recorded-contact run and the control run
    frame0_by_hand: dict[str, dict[str, float]]   # side -> contact input -> error at frame 0


@dataclass(frozen=True)
class Spread:
    """Hand-point errors of the six samples of the window (later-noise seeds SPREAD_SEEDS, in that order), millimetres."""
    window_recorded: np.ndarray        # (6,) whole window, recorded contact as input
    window_predicted: np.ndarray       # (6,) whole window, predicted contact as input
    frame_recorded: np.ndarray         # (6, 64) per frame, both hands
    frame_predicted: np.ndarray        # (6, 64)
    frame0_predicted_by_hand: dict[str, np.ndarray]    # side -> (6,) frame 0, predicted contact as input

    @property
    def drawn(self) -> int:
        return SPREAD_SEEDS.index(SEED)

    def rank_from_top(self, values: np.ndarray) -> int:
        """1 if the drawn sample has the largest of ``values`` (6,), 2 if the second largest, ..."""
        return int((values > values[self.drawn]).sum()) + 1


def measure_spread(sample: Sample) -> Spread:
    rec = sample.kp_recorded[None]
    a, b = sample.spread_from_recorded, sample.spread_from_predicted
    assert np.array_equal(a[SPREAD_SEEDS.index(SEED)], sample.kp_from_recorded)
    assert np.array_equal(b[SPREAD_SEEDS.index(SEED)], sample.kp_from_predicted)
    frame_a, frame_b = point_distance_mm(a, rec), point_distance_mm(b, rec)
    return Spread(window_recorded=frame_a.mean(1), window_predicted=frame_b.mean(1), frame_recorded=frame_a,
                  frame_predicted=frame_b,
                  frame0_predicted_by_hand={side: point_distance_mm(b[:, 0, sl], rec[:, 0, sl]) for side, sl in HAND_SLICE.items()})


def low_mid_high(values: np.ndarray) -> tuple[float, float, float]:
    return float(np.min(values)), float(np.median(values)), float(np.max(values))


def measure(sample: Sample) -> Errors:
    rec, a, b = sample.kp_recorded, sample.kp_from_recorded, sample.kp_from_predicted
    per_hand = {side: {"recorded contact": hand_error_mm(a[:, sl], rec[:, sl]),
                       "predicted contact": hand_error_mm(b[:, sl], rec[:, sl])} for side, sl in HAND_SLICE.items()}
    frame0 = {side: {"recorded contact": float(point_distance_mm(a[0, sl], rec[0, sl])),
                     "predicted contact": float(point_distance_mm(b[0, sl], rec[0, sl]))} for side, sl in HAND_SLICE.items()}
    return Errors(recorded=hand_error_mm(a, rec), predicted=hand_error_mm(b, rec),
                  frame_recorded=point_distance_mm(a, rec), frame_predicted=point_distance_mm(b, rec), per_hand=per_hand,
                  between=float(point_distance_mm(b, a).mean()), noise_only=float(point_distance_mm(sample.kp_control, a).mean()),
                  frame0_by_hand=frame0)


def saved_map_check(window: int, sample: Sample) -> dict[str, float]:
    """This run's contact maps against the maps the study saved for the same window (its CUDA sample), millimetres."""
    index = pd.read_csv(SAVED_MAPS_INDEX)
    position = int(np.flatnonzero(index["window"].to_numpy() == window)[0])
    assert index["sequence_id"].iloc[position] == sample.sequence_id and int(index["start"].iloc[position]) == sample.start
    with np.load(SAVED_MAPS) as z:
        saved_recorded = z["gt"][position].astype(np.float32)
        saved_predicted = z["pred"][position].astype(np.float32)
    mae = lambda a, b: float(np.abs(a - b).mean() * 1000.0)                 # noqa: E731
    return {"max_abs_recorded_map_difference_mm": float(np.abs(saved_recorded - sample.contact_recorded).max() * 1000.0),
            "predicted_map_this_run_vs_saved_mean_abs_mm": mae(saved_predicted, sample.contact_predicted),
            "predicted_map_this_run_vs_recorded_mean_abs_mm": mae(sample.contact_predicted, sample.contact_recorded),
            "predicted_map_saved_vs_recorded_mean_abs_mm": mae(saved_predicted, saved_recorded)}


def training_overlap(sample: Sample) -> dict[str, int]:
    """How many training sequences share the window's tool mesh, target mesh and (verb, tool, target) triplet."""
    index = pd.read_csv(SEQUENCE_INDEX)
    train = index[index["split"] == "train"]
    split = index[index["split"] == SPLIT]
    here = index[index["sequence_id"] == sample.sequence_id].iloc[0]
    assert here["split"] == SPLIT
    return {"training_sequences": int(len(train)),
            "training_sequences_with_this_tool_mesh": int((train["tool_mesh"] == here["tool_mesh"]).sum()),
            "training_sequences_with_this_tool_mesh_as_target": int((train["target_mesh"] == here["tool_mesh"]).sum()),
            "training_sequences_with_this_target_mesh": int((train["target_mesh"] == here["target_mesh"]).sum()),
            "training_sequences_with_this_triplet": int((train["triplet"] == here["triplet"]).sum()),
            "split_sequences": int(len(split)),
            "split_sequences_whose_tool_mesh_is_the_tool_of_a_training_sequence": int(split["tool_mesh"].isin(train["tool_mesh"]).sum())}


def number(label: str, value: float, source: Path | str, locator: str) -> dict[str, Any]:
    return {"label": label, "value": value, "dataset": DATASET, "source": str(source), "locator": locator}


def collect_numbers(selection: Selection, sample: Sample, scene: Scene, errors: Errors, spread: Spread,
                    profile: dict[str, Any], split_row: pd.Series, saved: dict[str, float],
                    overlap: dict[str, int]) -> list[dict[str, Any]]:
    """Every number a caption may quote, each with the file (or saved array) and the row / key it comes from."""
    chosen, table = selection.row, selection.table
    here = f"computed in this script; arrays saved in panels.npz of {FIG_ID} under sample/"
    in_spread = f"computed in this script; arrays saved in panels.npz of {FIG_ID} under spread/"
    in_table = f"rows with window == {selection.window} (2 hands x {HORIZON} frames), mean of "
    frame_gap = errors.frame_predicted - errors.frame_recorded
    table_gap = profile["window_predicted_mm"] - profile["window_recorded_mm"]
    six = f"six samples of this window (later-noise seeds {SPREAD_SEEDS[0]}-{SPREAD_SEEDS[-1]}, initial seed {SEED})"
    numbers = [
        number("this sample: hand-point error with the recorded contact, mm", errors.recorded, here,
               "mean over 64 frames x 200 points of |hand_points_from_recorded_contact - hand_points_recorded|"),
        number("this sample: hand-point error with the predicted contact, mm", errors.predicted, here,
               "mean over 64 frames x 200 points of |hand_points_from_predicted_contact - hand_points_recorded|"),
        number("this sample: error increase, mm", errors.predicted - errors.recorded, here, "difference of the two errors above"),
        number("this sample: mean distance between the two generated hands, mm", errors.between, here,
               "mean of |hand_points_from_predicted_contact - hand_points_from_recorded_contact|"),
        number("control: recorded contact, same initial noise, other later noise; mean distance to the drawn run, mm",
               errors.noise_only, here, "mean of |hand_points_control - hand_points_from_recorded_contact|"),
        number(f"this sample: error increase over frames {LATE.start}-{LATE.stop - 1}, mm", float(frame_gap[LATE].mean()), here,
               f"mean over frames {LATE.start}-{LATE.stop - 1} of error_per_frame_predicted_contact_mm minus "
               "error_per_frame_recorded_contact_mm"),
        number("this sample: mean absolute difference between predicted and recorded contact map, mm",
               saved["predicted_map_this_run_vs_recorded_mean_abs_mm"], here, "mean of |contact_predicted_m - contact_recorded_m| x 1000"),
        number("seed of the new sample (initial noise and later noise of every drawn run)", SEED, "this script", "SEED"),
        number("study's table, this window: hand-point error with the recorded contact, mm", float(chosen["recorded_mm"]),
               WINDOW_TABLE, in_table + "motion_mm"),
        number("study's table, this window: hand-point error with the predicted contact, mm", float(chosen["predicted_mm"]),
               WINDOW_TABLE, in_table + "total_mm"),
        number("study's table, this window: error increase (the selection metric), mm", float(chosen["gap_mm"]),
               WINDOW_TABLE, in_table + "total_mm minus mean of motion_mm"),
        number("study's table, this window: mean absolute difference between predicted and recorded contact map, mm",
               float(chosen["contact_map_err_mm"]), WINDOW_TABLE, in_table + "contact_map_err_mm"),
        number("study's table, this window, frame 0: error with the recorded contact, mm", float(profile["window_recorded_mm"][0]),
               WINDOW_TABLE, f"rows with window == {selection.window} and frame == 0, mean of motion_mm over the two hands"),
        number("study's table, this window, frame 0: error with the predicted contact, mm", float(profile["window_predicted_mm"][0]),
               WINDOW_TABLE, f"rows with window == {selection.window} and frame == 0, mean of total_mm over the two hands"),
        number(f"number of windows in {SPLIT}", int(len(table)), WINDOW_TABLE, "distinct values of column window"),
        number(f"rank of the window by error increase (ascending, 1-based) among the windows of {SPLIT}", selection.rank,
               WINDOW_TABLE, "per-window error increase, sorted"),
        number(f"median error increase over the windows of {SPLIT}, mm", float(table["gap_mm"].median()), WINDOW_TABLE,
               "median over windows of (mean total_mm - mean motion_mm)"),
        number(f"mean error increase over the windows of {SPLIT}, mm", float(table["gap_mm"].mean()), WINDOW_TABLE,
               "mean over windows of (mean total_mm - mean motion_mm)"),
        number(f"windows of {SPLIT} whose error increases", int((table["gap_mm"] > 0).sum()), WINDOW_TABLE,
               "count of windows with mean total_mm > mean motion_mm"),
        number(f"{SPLIT} mean over all its windows: error with the recorded contact, mm", float(table["recorded_mm"].mean()),
               WINDOW_TABLE, "mean over windows of mean motion_mm"),
        number(f"{SPLIT} mean over all its windows: error with the predicted contact, mm", float(table["predicted_mm"].mean()),
               WINDOW_TABLE, "mean over windows of mean total_mm"),
        number(f"{SPLIT} mean error increase at frame 0, mm", float(profile["split_gap_mm"][0]), WINDOW_TABLE,
               "rows with frame == 0: mean total_mm - mean motion_mm"),
        number(f"{SPLIT} mean error increase over frames {LATE.start}-{LATE.stop - 1}, mm", float(profile["split_gap_mm"][LATE].mean()),
               WINDOW_TABLE, f"mean over frames {LATE.start}-{LATE.stop - 1} of (mean total_mm - mean motion_mm)"),
        number(f"{SPLIT} mean the page quotes: error with the recorded contact, mm", float(split_row["motion_kp_err_mm_gt_contact"]),
               SPLIT_TABLE, f"row split == {SPLIT}, column motion_kp_err_mm_gt_contact"),
        number(f"{SPLIT} mean the page quotes: error with the predicted contact, mm", float(split_row["motion_kp_err_mm_pred_contact"]),
               SPLIT_TABLE, f"row split == {SPLIT}, column motion_kp_err_mm_pred_contact"),
        number(f"windows behind the {SPLIT} means the page quotes", int(split_row["sampled"]), SPLIT_TABLE,
               f"row split == {SPLIT}, column sampled"),
        number("frames per window", HORIZON, MOTION_CONFIG, "data.pred_horizon"),
        number("window length, seconds", HORIZON / FPS, "src/analysis/loaders/taco.py", f"DEFAULT_FPS = {FPS:g}; {HORIZON} frames / {FPS:g}"),
        number("generated surface points per hand", HAND_KP, MOTION_CONFIG, "data.hand_keypoints"),
        number("training sequences with this tool mesh", overlap["training_sequences_with_this_tool_mesh"], SEQUENCE_INDEX,
               "rows with split == train and the window's tool_mesh"),
        number("training sequences with this tool mesh as their target", overlap["training_sequences_with_this_tool_mesh_as_target"],
               SEQUENCE_INDEX, "rows with split == train whose target_mesh is the window's tool_mesh"),
        number(f"sequences in {SPLIT}", overlap["split_sequences"], SEQUENCE_INDEX, f"rows with split == {SPLIT}"),
        number(f"sequences in {SPLIT} whose tool mesh is the tool of a training sequence",
               overlap["split_sequences_whose_tool_mesh_is_the_tool_of_a_training_sequence"], SEQUENCE_INDEX,
               f"rows with split == {SPLIT} whose tool_mesh occurs in column tool_mesh of a row with split == train"),
        number("study's table, this window: frame with the largest error increase", int(np.argmax(table_gap)), WINDOW_TABLE,
               f"rows with window == {selection.window}: frame maximising mean total_mm - mean motion_mm over the two hands"),
        number("study's table, this window: largest per-frame error increase, mm", float(table_gap.max()), WINDOW_TABLE,
               f"rows with window == {selection.window}: maximum over frames of mean total_mm - mean motion_mm"),
        number("number of samples of this window (the drawn one and the undrawn control)", len(SPREAD_SEEDS), "this script",
               "SPREAD_SEEDS"),
        number(f"{six}: rank of the drawn sample by frame-0 error with the predicted contact (1 = largest)",
               spread.rank_from_top(spread.frame_predicted[:, 0]), in_spread, "error_per_frame_predicted_contact_mm[:, 0]"),
        number(f"{six}: rank of the drawn sample by whole-window error with the predicted contact (1 = largest)",
               spread.rank_from_top(spread.window_predicted), in_spread, "mean over frames of error_per_frame_predicted_contact_mm"),
        number(f"{six}: samples whose left hand starts more than {FAR_MM:g} mm from the recorded one with the predicted contact",
               int((spread.frame0_predicted_by_hand["left"] > FAR_MM).sum()), in_spread, "frame0_left_hand_predicted_contact_mm"),
        number(f"{six}: samples whose right hand starts more than {FAR_MM:g} mm from the recorded one with the predicted contact",
               int((spread.frame0_predicted_by_hand["right"] > FAR_MM).sum()), in_spread, "frame0_right_hand_predicted_contact_mm"),
    ]
    for what, values, key in (
            ("frame-0 error with the predicted contact", spread.frame_predicted[:, 0], "error_per_frame_predicted_contact_mm[:, 0]"),
            ("frame-0 error with the recorded contact", spread.frame_recorded[:, 0], "error_per_frame_recorded_contact_mm[:, 0]"),
            ("whole-window error with the predicted contact", spread.window_predicted,
             "mean over frames of error_per_frame_predicted_contact_mm"),
            ("whole-window error with the recorded contact", spread.window_recorded,
             "mean over frames of error_per_frame_recorded_contact_mm"),
            ("frame-0 error of the left hand with the predicted contact", spread.frame0_predicted_by_hand["left"],
             "frame0_left_hand_predicted_contact_mm"),
            ("frame-0 error of the right hand with the predicted contact", spread.frame0_predicted_by_hand["right"],
             "frame0_right_hand_predicted_contact_mm")):
        for stat, value in zip(("minimum", "median", "maximum"), low_mid_high(values)):
            numbers.append(number(f"{six}: {stat} of the {what}, mm", value, in_spread, f"{stat} over the six samples of {key}"))
    for side in SIDES:
        for arm, value in errors.per_hand[side].items():
            numbers.append(number(f"this sample: {side} hand error with the {arm}, mm", value, here,
                                  f"points {HAND_SLICE[side].start}-{HAND_SLICE[side].stop - 1} of the hand-point arrays"))
        for arm, value in errors.frame0_by_hand[side].items():
            numbers.append(number(f"this sample, frame 0: {side} hand error with the {arm}, mm", value, here,
                                  f"frame 0, points {HAND_SLICE[side].start}-{HAND_SLICE[side].stop - 1} of the hand-point arrays"))
        for arm, value, column in zip(("recorded contact", "predicted contact"), profile["window_frame0_by_hand_mm"][side],
                                      ("motion_mm", "total_mm")):
            numbers.append(number(f"study's table, this window, frame 0: {side} hand error with the {arm}, mm", value, WINDOW_TABLE,
                                  f"row with window == {selection.window}, hand == {side[0].upper()}, frame == 0, column {column}"))
    for t in FRAMES:
        numbers.append(number(f"this sample, frame {t}: error with the recorded contact, mm", float(errors.frame_recorded[t]),
                              here, f"error_per_frame_recorded_contact_mm[{t}]"))
        numbers.append(number(f"this sample, frame {t}: error with the predicted contact, mm", float(errors.frame_predicted[t]),
                              here, f"error_per_frame_predicted_contact_mm[{t}]"))
    for side in SIDES:
        numbers.append(number(f"frame 0: distance of the recorded {side} hand from the nearest object surface, mm",
                              float(scene.dense[side][0].min() * 1000.0), GEN3 / "sequences" / sample.sequence_file,
                              f"min over the vertices of both objects of contact_{side}[{sample.start}]"))
    return numbers


# ------------------------------------------------------------------------------ figure
def render_inside(items: list[dict[str, Any]], camera: qlib.Camera) -> np.ndarray:
    """qlib.render, plus the guarantee that nothing is cut by the panel's edge (its outermost pixels are background)."""
    panel = qlib.render(items, camera, PANEL)
    edge = np.concatenate([e.reshape(-1, 3) for e in (panel[:3], panel[-3:], panel[:, :3], panel[:, -3:])])
    assert (edge >= 250).all(), "something is drawn on the edge of a panel: widen CAMERA_MARGIN"
    return panel


def recorded_hand_items(scene: Scene, t: int) -> list[dict[str, Any]]:
    """Both recorded hands at window frame t as translucent meshes with ALL of their faces.

    qlib.render draws only the front faces of a translucent mesh (it culls the back faces). The MANO mesh is open at
    the wrist and this camera looks from the wrist side, so with the front faces alone a hand reads as an open shell
    (about a third of its triangles at frame 0). Each hand is therefore passed twice, the second time with reversed
    triangle winding: same vertices, colour and opacity, and the renderer then draws the other faces as well. The
    lighter ellipse at the wrist is the opening of the MANO mesh, where only the far wall is seen.
    """
    items = []
    for side in SIDES:
        verts, faces = scene.hands[side][t], qlib.mano_faces(side)
        items.append(qlib.hand_item(verts, faces, **RECORDED_HAND))
        items.append(qlib.hand_item(verts, faces[:, ::-1], **RECORDED_HAND))
    return items


def render_rows(sample: Sample, scene: Scene, camera: qlib.Camera) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """The 4 x 4 panels (rows: recorded contact, predicted contact, hand from each; columns: FRAMES) and what they show."""
    contact_rows = (("recorded_contact", sample.contact_recorded), ("predicted_contact", sample.contact_predicted))
    hand_rows = (("hand_from_recorded_contact", sample.kp_from_recorded, COLOR_FROM_RECORDED),
                 ("hand_from_predicted_contact", sample.kp_from_predicted, COLOR_FROM_PREDICTED))
    rows: list[list[np.ndarray]] = [[] for _ in range(len(contact_rows) + len(hand_rows))]
    panels: dict[str, Any] = {}
    for t in FRAMES:
        recorded_hands = recorded_hand_items(scene, t)
        for r, (name, contact_map) in enumerate(contact_rows):
            tool_value, target_value = surface_contact(scene, contact_map, t)
            items = [qlib.mesh_item(scene.tool[t], scene.tool_faces, rgb=qlib.contact_rgb(tool_value, CONTACT_VMAX)),
                     qlib.mesh_item(scene.target, scene.target_faces, rgb=qlib.contact_rgb(target_value, CONTACT_VMAX))]
            rows[r].append(render_inside(items + recorded_hands, camera))
            panels[f"{name}_t{t:02d}"] = {"tool_value": tool_value, "target_value": target_value, "map_m": contact_map[t]}
        plain = [qlib.mesh_item(scene.tool[t], scene.tool_faces), qlib.mesh_item(scene.target, scene.target_faces)]
        for r, (name, points, colour) in enumerate(hand_rows, len(contact_rows)):
            rows[r].append(render_inside(plain + recorded_hands + [qlib.points_item(points[t], colour, POINT_RADIUS)], camera))
            panels[f"{name}_t{t:02d}"] = {"points": points[t]}
        panels[f"recorded_t{t:02d}"] = {"tool_verts": scene.tool[t], "hand_left": scene.hands["left"][t],
                                        "hand_right": scene.hands["right"][t], "hand_points": sample.kp_recorded[t],
                                        "map_vertex": scene.slot_vertex[t]}
    panels["geometry"] = {"target_verts": scene.target, "tool_faces": scene.tool_faces, "target_faces": scene.target_faces,
                          "hand_faces_left": qlib.mano_faces("left"), "hand_faces_right": qlib.mano_faces("right"),
                          "up": scene.up}
    return rows, panels


def caption_text(selection: Selection, errors: Errors, spread: Spread, profile: dict[str, Any], split_row: pd.Series) -> str:
    chosen = selection.row
    first = spread.frame_predicted[:, 0]                    # frame-0 error with the predicted contact, six samples
    low, _, high = low_mid_high(first)
    rank = spread.rank_from_top(first)
    place = "the " + ("", "second ", "third ", "fourth ", "fifth ", "sixth ")[rank - 1] + "largest"
    far = int((spread.frame0_predicted_by_hand["left"] > FAR_MM).sum())
    as_drawn = ", as drawn," if errors.frame0_by_hand["left"]["predicted contact"] > FAR_MM else ""
    word = ("none", "one", "two", "three", "four", "five", "six")            # counts up to len(SPREAD_SEEDS) in words
    return (
        f"One {HORIZON}-frame test window of TACO (about {HORIZON / FPS:.0f} seconds; a teapot is picked up and tipped over a "
        f"kettle; this teapot mesh is in no training sequence), selected as the window whose error increase in the study's "
        f"saved per-window table is the median of the {len(selection.table)} windows of its test split (the split whose "
        f"tool meshes are held out). "
        f"Top two rows: the recorded contact and the contact predicted by the contact stage, drawn on the object meshes; "
        f"the grey hands are the recorded hands. "
        f"Bottom two rows: the {HAND_KP} surface points per hand that the same motion model generates, with the same noise, "
        f"from the recorded contact (blue; an oracle input, because this map is measured from the recorded hands "
        f"themselves) and from the predicted contact (green). "
        f"In this one new sample (seed {SEED}) the points are on average {errors.recorded:.1f} mm from the recorded hands with "
        f"the recorded contact and {errors.predicted:.1f} mm with the predicted contact; the saved table of the study gives "
        f"{chosen['recorded_mm']:.1f} and {chosen['predicted_mm']:.1f} mm for this window, and the split means quoted above are "
        f"{split_row['motion_kp_err_mm_gt_contact']:.1f} and {split_row['motion_kp_err_mm_pred_contact']:.1f} mm. "
        f"In this sample the two results differ most in the first frame ({errors.frame_recorded[0]:.0f} against "
        f"{errors.frame_predicted[0]:.0f} mm; the study's saved sample of the same window gives "
        f"{profile['window_recorded_mm'][0]:.0f} against {profile['window_predicted_mm'][0]:.0f} mm there), before either "
        f"recorded hand touches an object: the maps the motion model reads are distances to the hand, and the colour shows "
        f"only their last few centimetres. "
        f"Of {word[len(first)]} samples of this window that share the starting noise and differ in the noise added during "
        f"sampling, the drawn one has {place} first-frame error with the predicted contact (range {low:.0f} to {high:.0f} mm), "
        f"and the left hand starts more than {FAR_MM / 10:g} cm from the recorded one{as_drawn} in {word[far]} of the "
        f"{word[len(first)]}.")


def check_caption(scene: Scene, errors: Errors, overlap: dict[str, int]) -> None:
    """The statements of the caption that are not plain numbers."""
    assert int(np.argmax(errors.frame_predicted - errors.frame_recorded)) == 0, "the first frame does not differ most"
    assert min(float(scene.dense[side][0].min()) for side in SIDES) > 0.02, "a recorded hand is within 2 cm of an object at frame 0"
    assert overlap["training_sequences_with_this_tool_mesh"] == 0, "the tool mesh occurs in training"
    assert overlap["training_sequences_with_this_tool_mesh_as_target"] == 0, "the tool mesh occurs in training as a target"
    assert overlap["split_sequences_whose_tool_mesh_is_the_tool_of_a_training_sequence"] == 0, \
        f"{SPLIT} is not a split whose tool meshes are held out"
    up = scene.up / np.linalg.norm(scene.up)
    first, last = scene.tool[0], scene.tool[-1]
    assert (last.mean(0) - first.mean(0)) @ up > 0.1, "the tool is not lifted by more than 10 cm"
    turn, _ = Rotation.align_vectors(last - last.mean(0), first - first.mean(0))
    assert np.degrees(np.arccos(np.clip(turn.apply(up) @ up, -1.0, 1.0))) > 30.0, "the tool is not tipped by more than 30 degrees"


def describe(args: argparse.Namespace, selection: Selection, sample: Sample, scene: Scene, errors: Errors, spread: Spread,
             direction: np.ndarray, camera: qlib.Camera) -> dict[str, Any]:
    """The record written to meta.json next to the figure."""
    chosen, table = selection.row, selection.table
    profile = frame_profile(WINDOW_TABLE, selection.window)
    split_row = pd.read_csv(SPLIT_TABLE).set_index("split").loc[SPLIT]
    saved = saved_map_check(selection.window, sample)
    overlap = training_overlap(sample)
    check_caption(scene, errors, overlap)
    late_gap = float((errors.frame_predicted - errors.frame_recorded)[LATE].mean())
    radius_cm = f"{DISPLAY_RADIUS * 100:g} cm"
    display_rule = (
        "Not the 512-point canonical map of the other figures. This study's map has, per hand, 512 entries on the tool "
        "and 512 on the target; entry k is the distance from mesh vertex obj_cano_bps_inds[k] to that hand. Each entry "
        "becomes exp(-max(distance, 0) / 2 cm), the larger of the two hands is kept, and a mesh vertex takes the "
        f"Gaussian-weighted mean (sigma {DISPLAY_SIGMA * 100:g} cm) of the entries of its own object within {radius_cm}. "
        f"A vertex with no entry within {radius_cm} has no value and is drawn in pale violet. Recorded and predicted "
        "maps go through the same rule.")
    first, first_left = spread.frame_predicted[:, 0], spread.frame0_predicted_by_hand["left"]
    table_gap = profile["window_predicted_mm"] - profile["window_recorded_mm"]
    table_left, table_right = (profile["window_frame0_by_hand_mm"][side] for side in SIDES)
    listed = lambda values: ", ".join(f"{v:.0f}" for v in values)                  # noqa: E731
    return {
        "id": FIG_ID, "block": BLOCK, "title": TITLE,
        "display_rule": display_rule,              # qlib.save's own top-level field describes a rule this figure does not use
        "panels": {
            "columns": f"window frames {list(FRAMES)} of {HORIZON} (take frames {[sample.start + t for t in FRAMES]}, "
                       f"{FPS:g} frames per second)",
            "row 1": "the recorded contact map of the window, painted on the tool (teapot) and target (kettle) meshes",
            "row 2": "the contact map the contact stage predicts from the object motion alone, painted the same way",
            "row 3": "blue spheres: the 100 surface points per hand the motion stage generates when its contact input is "
                     "the recorded map",
            "row 4": "green spheres: the same motion stage, same noise, when its contact input is the predicted map",
            "every panel": "translucent grey meshes: the recorded MANO hands (both hands) at that frame; objects at their "
                           "recorded poses",
        },
        "examples": [{
            "dataset": DATASET, "study": GEN3, "split": SPLIT, "n": selection.window,
            "n_is": f"window index among the study's {SPLIT} windows (TacoMotionDataset order = column window of {WINDOW_TABLE.name})",
            "sequence_id": sample.sequence_id, "take_frames": [sample.start, sample.start + HORIZON - 1],
            "object": {"tool": f"{chosen['tool']} (mesh {sample.tool_mesh})",
                       "target": f"{chosen['target']} (mesh {sample.target_mesh})", "verb": chosen["verb"]},
            "hand": "both (left and right)", "frames_drawn": list(FRAMES), **overlap}],
        "selection_rule": (
            f"Among all windows of split {SPLIT}, the one whose increase of the mean hand-point error (predicted contact minus "
            f"recorded contact; per window the mean of total_mm minus the mean of motion_mm over its 128 rows of "
            f"{WINDOW_TABLE.name}) is the median: the windows are sorted by that increase and the middle one is taken."),
        "selection_evidence": {
            "table": WINDOW_TABLE, "metric": "mean(total_mm) - mean(motion_mm) per window, mm", "candidates": int(len(table)),
            "rank_ascending_1_based": selection.rank, "window": selection.window, "value_mm": float(chosen["gap_mm"]),
            "median_mm": float(table["gap_mm"].median()), "mean_mm": float(table["gap_mm"].mean()),
            "min_mm": float(table["gap_mm"].min()), "max_mm": float(table["gap_mm"].max()),
            "quartiles_mm": [float(v) for v in table["gap_mm"].quantile([0.25, 0.5, 0.75])],
            "neighbours": table.iloc[selection.rank - 4:selection.rank + 3][
                ["window", "sequence_id", "recorded_mm", "predicted_mm", "gap_mm"]].to_dict("records"),
            "sorted_window": table["window"].tolist(), "sorted_gap_mm": [round(float(v), 4) for v in table["gap_mm"]],
        },
        "display_choices": {
            "frame": "the target object's rigid frame (the frame the models work in), metres; the tool moves in it",
            "canonical_to_surface_rule": display_rule,
            "display_rule_fidelity_recorded_map_vs_exact_per_vertex_contact": display_fidelity(scene, sample),
            "colour_scales": [{"what": "contact on the objects (rows 1 and 2)", "scale": "qlib.contact_rgb",
                               "range": [0.0, CONTACT_VMAX], "no_value": qlib.NO_DATA}],
            "objects_rows_3_4": f"plain {qlib.ZERO} (no map painted)",
            "camera": {
                "rule": "one camera for all 16 panels, fitted to everything drawn at the four frames; horizontal direction from "
                        "the objects' centroid towards the recorded hands' centroid, turned by the azimuth about the vertical "
                        "and raised by the elevation; vertical = the world's z axis in the target frame at frame 0. Azimuth and "
                        "elevation were set by looking at renders (handle of the kettle in profile, teapot and both hands "
                        "unoccluded).",
                "azimuth_deg": args.azimuth, "elevation_deg": args.elevation, "margin": CAMERA_MARGIN,
                "direction": direction, "fitted": camera},
            "hand_style": {
                "recorded_hands": {
                    **RECORDED_HAND, "faces_drawn": "front and back (no back-face culling), the same in all four rows",
                    "note": "MANO meshes, translucent; the opacity is per face layer. qlib.render culls the back faces of a "
                            "translucent mesh, and the MANO mesh is open at the wrist, which this camera faces, so with front "
                            "faces alone the hands read as open shells. Each hand is passed to the renderer twice, the second "
                            "time with reversed triangle winding, so that all of its faces are drawn. The lighter ellipse at "
                            "the wrist is the opening of the mesh. Darker than qlib.GHOST, which is nearly invisible at this "
                            "scene size."},
                "generated_hand_points": {
                    "radius_m": POINT_RADIUS, "from_recorded_contact": COLOR_FROM_RECORDED,
                    "from_predicted_contact": COLOR_FROM_PREDICTED,
                    "note": "the motion stage outputs 100 surface points per hand, not MANO parameters; no mesh is fitted"}},
            "panel_px": list(PANEL),
        },
        "numbers": collect_numbers(selection, sample, scene, errors, spread, profile, split_row, saved, overlap),
        "inference": {
            "checkpoints": sample.checkpoints,
            "newly_computed": [
                "the predicted contact map of the window (contact stage, 50 denoising steps)",
                "the hand points generated from the recorded contact and from the predicted contact (motion stage, 50 steps "
                "each, classifier-free guidance 0.5, no contact guidance), with identical noise",
                f"an undrawn control: the same window sampled {len(SPREAD_SEEDS) - 1} more times (contact stage, then the motion "
                f"stage from the recorded and from the predicted contact; same initial noise, later-noise seeds "
                f"{[s for s in SPREAD_SEEDS if s != SEED]}). Its errors are in sampling_spread and its hand points in panels.npz "
                "under spread/; nothing of it is drawn, and it played no part in choosing the window or the drawn seed",
            ],
            "read_from_disk": [
                f"recorded contact map, recorded hand points and object features: {GEN3 / 'train_store'}",
                f"object meshes: {GEN3 / 'assets/taco_mesh_dict.npy'}",
                f"map vertices and dense hand distances: {GEN3 / 'sequences' / sample.sequence_file}",
                "recorded MANO hands and object poses: the TACO release through src/analysis/loaders/taco.py",
                f"per-window errors (selection and table values): {WINDOW_TABLE}; split means: {SPLIT_TABLE}",
            ],
            "noise": ("The study's sampler seeds only the initial noise; the noise of the later denoising steps comes from "
                      "torch's global generator, which the study's scripts do not seed. Here it is seeded before every run: the "
                      "two drawn runs share all noise and the figure is reproducible. The study's table was sampled on CUDA in "
                      "batches of 32."),
            "cross_checks": {"geometry_m": check_scene(scene, sample), "contact_maps_vs_saved": saved,
                             "errors_vs_table_mm": {"this_sample": [errors.recorded, errors.predicted],
                                                    "table": [float(chosen["recorded_mm"]), float(chosen["predicted_mm"])]}},
        },
        "per_frame_error_mm": {"frames": list(range(HORIZON)), "this_sample_recorded_contact": errors.frame_recorded,
                               "this_sample_predicted_contact": errors.frame_predicted,
                               "table_this_window_recorded_contact": profile["window_recorded_mm"],
                               "table_this_window_predicted_contact": profile["window_predicted_mm"],
                               "table_split_mean_increase": profile["split_gap_mm"]},
        "sampling_spread": {
            "what": f"The window sampled {len(SPREAD_SEEDS)} times on the CPU: initial noise seed {SEED} every time, the noise of "
                    "the later denoising steps (both stages) from the listed seed. The sample with later-noise seed "
                    f"{SEED} is the drawn one; the others are an undrawn control. Errors in mm against the recorded hand points.",
            "later_noise_seeds": list(SPREAD_SEEDS), "drawn_later_noise_seed": SEED,
            "window_error_recorded_contact": spread.window_recorded, "window_error_predicted_contact": spread.window_predicted,
            "frame0_error_recorded_contact": spread.frame_recorded[:, 0], "frame0_error_predicted_contact": first,
            "frame0_left_hand_error_predicted_contact": first_left,
            "frame0_right_hand_error_predicted_contact": spread.frame0_predicted_by_hand["right"],
            "min_median_max": {"frame0_error_predicted_contact": low_mid_high(first),
                               "window_error_predicted_contact": low_mid_high(spread.window_predicted),
                               "frame0_error_recorded_contact": low_mid_high(spread.frame_recorded[:, 0]),
                               "window_error_recorded_contact": low_mid_high(spread.window_recorded)},
            "rank_of_the_drawn_sample_from_the_top": {"frame0_error_predicted_contact": spread.rank_from_top(first),
                                                      "window_error_predicted_contact": spread.rank_from_top(spread.window_predicted)},
            "far_threshold_mm": FAR_MM, "samples_with_left_hand_far_at_frame0": int((first_left > FAR_MM).sum()),
        },
        "caveats": [
            "One window and one sample (seed 0). The drawn hands are a new CPU sample of the same two checkpoints, not the "
            "sample behind the saved table or the page's split means; this sample's errors differ from the table's for the "
            "same window.",
            f"The drawn sample is not a typical sample of this window in its first frame, which is where the figure shows its "
            f"effect. Over the {len(SPREAD_SEEDS)} samples (later-noise seeds {list(SPREAD_SEEDS)}) the frame-0 error with the "
            f"predicted contact is {listed(first)} mm and the whole-window error {listed(spread.window_predicted)} mm; the drawn "
            f"sample (seed {SEED}) has rank {spread.rank_from_top(first)} and {spread.rank_from_top(spread.window_predicted)} from "
            f"the top. The cloud of green points beside the kettle's handle is the left hand "
            f"({errors.frame0_by_hand['left']['predicted contact']:.0f} mm from the recorded left hand at frame 0); the left hand "
            f"starts more than {FAR_MM:g} mm away in {int((first_left > FAR_MM).sum())} of the {len(SPREAD_SEEDS)} samples "
            f"({listed(first_left)} mm). In the study's saved sample of this window the left hand is not displaced at frame 0 "
            f"({table_left[0]:.0f} mm with the recorded contact, {table_left[1]:.0f} mm with the predicted one), the right hand is "
            f"({table_right[0]:.0f} against {table_right[1]:.0f} mm), and the largest per-frame increase is at frame "
            f"{int(np.argmax(table_gap))} ({table_gap.max():.0f} mm), not at frame 0. The seed was fixed in advance ({SEED}); "
            "neither the window nor the seed was chosen by looking at these samples.",
            f"The window is the median of the per-window error increase ({chosen['gap_mm']:.1f} mm in the table); the mean "
            f"increase over the split is larger ({table['gap_mm'].mean():.1f} mm) because the distribution has a long upper "
            "tail, so the page's split means describe a larger gap than this window shows.",
            f"Most of what the picture shows is in the first column. From frame {LATE.start} on, this sample's error increase "
            f"averages {late_gap:.1f} mm, which is a few pixels at this panel size: the two generated hands look alike there.",
            f"The first-frame difference of this sample ({errors.frame_recorded[0]:.0f} against {errors.frame_predicted[0]:.0f} mm) "
            f"is larger than in the study's saved sample of the same window ({profile['window_recorded_mm'][0]:.0f} against "
            f"{profile['window_predicted_mm'][0]:.0f} mm): the contact stage's sample differs. A larger increase at the start "
            f"of a window is the pattern of the whole split ({profile['split_gap_mm'][0]:.1f} mm at frame 0 against "
            f"{profile['split_gap_mm'][LATE].mean():.1f} mm over frames {LATE.start}-{LATE.stop - 1}).",
            "The contact maps are distances from object vertices to each hand (up to tens of centimetres). The colour is "
            "exp(-distance / 2 cm), so it shows only the last few centimetres; the motion stage also reads the larger "
            "distances, which is why the hands differ at frame 0 where neither map shows much contact.",
            "The recorded-contact condition is an oracle: its map comes from the recorded hand, so it tells the motion stage "
            "where the hand is even before it touches.",
            "Generated hands are 100 surface points per hand (the motion stage's output); no hand mesh was fitted to them.",
            f"The painted map is a smoothed display of 512 entries per object; vertices without an entry within {radius_cm} are "
            "pale violet. Both hands are merged (the nearer one).",
            "The grey hands in the two contact rows are the recorded hands in both rows; the predicted contact has no hand of "
            "its own.",
            f"The split is defined by held-out tool meshes (none of its {overlap['split_sequences']} sequences has a tool mesh "
            f"that is the tool of a training sequence). This window's triplet {sample.sequence_id.rsplit('/', 1)[0]} occurs in "
            f"{overlap['training_sequences_with_this_triplet']} training sequences, and its target mesh ({chosen['target']}) is "
            f"the target of {overlap['training_sequences_with_this_target_mesh']} training sequences.",
        ],
        "suggested_caption": caption_text(selection, errors, spread, profile, split_row),
        "suggested_alt": ("A kettle and a teapot at four moments of a recorded pouring motion, with the recorded hands in grey: "
                          "two rows show recorded and predicted contact as orange areas on the objects, two rows show generated "
                          "hand points that sit on the recorded hands when the recorded contact is used and away from them in "
                          "the first frame when the predicted contact is used."),
    }


def build(args: argparse.Namespace) -> dict[str, Path]:
    qlib.require_headless()
    selection = select_window(WINDOW_TABLE)
    chosen = selection.row
    log.info("selected %s window %d (rank %d of %d): %s, table %.1f -> %.1f mm", SPLIT, selection.window, selection.rank,
             len(selection.table), chosen["sequence_id"], chosen["recorded_mm"], chosen["predicted_mm"])
    sample = run_models(selection.window)
    assert (sample.sequence_id, sample.start) == (chosen["sequence_id"], int(chosen["start"])), "window index mismatch"
    errors = measure(sample)
    log.info("this sample (seed %d, CPU): %.1f mm with the recorded contact, %.1f mm with the predicted contact; "
             "the two generated hands are %.1f mm apart; sampler noise alone moves the hand by %.1f mm",
             SEED, errors.recorded, errors.predicted, errors.between, errors.noise_only)
    spread = measure_spread(sample)
    for i, later_seed in enumerate(SPREAD_SEEDS):
        log.info("later-noise seed %d%s: window %.1f -> %.1f mm, frame 0 %.1f -> %.1f mm (predicted contact: left %.0f, right %.0f)",
                 later_seed, " (drawn)" if later_seed == SEED else "", spread.window_recorded[i], spread.window_predicted[i],
                 spread.frame_recorded[i, 0], spread.frame_predicted[i, 0], spread.frame0_predicted_by_hand["left"][i],
                 spread.frame0_predicted_by_hand["right"][i])
    scene = load_scene(sample)

    # one camera for every panel, fitted to everything any panel shows
    frames = list(FRAMES)
    drawn = [scene.tool[frames], scene.target, *(scene.hands[side][frames] for side in SIDES),
             sample.kp_from_recorded[frames], sample.kp_from_predicted[frames]]
    direction = view_direction(scene, args.elevation, args.azimuth)
    camera = qlib.fit_camera(drawn, direction, up=scene.up, margin=CAMERA_MARGIN, aspect=PANEL[0] / PANEL[1])
    rows, panels = render_rows(sample, scene, camera)
    image = qlib.grid(
        rows,
        row_labels=["Recorded contact", "Contact predicted by the contact stage",
                    "Hand generated from the recorded contact (blue points)",
                    "Hand generated from the predicted contact (green points)"],
        col_labels=[f"frame {t}" for t in FRAMES],
        colorbars=[qlib.colorbar("contact", 0.0, CONTACT_VMAX, "contact on the objects: exp(−distance to the nearer hand / 2 cm)",
                                 nodata=f"no map entry within {DISPLAY_RADIUS * 100:g} cm")],
        footnote="Grey translucent hands: the recorded hands, the same in all four rows. The recorded contact is measured "
                 "from these hands.")

    table = selection.table
    panels["sample"] = {
        "hand_points_recorded": sample.kp_recorded, "hand_points_from_recorded_contact": sample.kp_from_recorded,
        "hand_points_from_predicted_contact": sample.kp_from_predicted, "hand_points_control": sample.kp_control,
        "contact_recorded_m": sample.contact_recorded, "contact_predicted_m": sample.contact_predicted,
        "error_per_frame_recorded_contact_mm": errors.frame_recorded, "error_per_frame_predicted_contact_mm": errors.frame_predicted}
    panels["selection"] = {"window": table["window"].to_numpy(), "gap_mm": table["gap_mm"].to_numpy(),
                           "recorded_contact_mm": table["recorded_mm"].to_numpy(),
                           "predicted_contact_mm": table["predicted_mm"].to_numpy()}
    panels["spread"] = {                                     # not drawn: the six samples of the window (first axis)
        "later_noise_seed": np.array(SPREAD_SEEDS), "hand_points_from_recorded_contact": sample.spread_from_recorded,
        "hand_points_from_predicted_contact": sample.spread_from_predicted,
        "error_per_frame_recorded_contact_mm": spread.frame_recorded, "error_per_frame_predicted_contact_mm": spread.frame_predicted,
        "frame0_left_hand_predicted_contact_mm": spread.frame0_predicted_by_hand["left"],
        "frame0_right_hand_predicted_contact_mm": spread.frame0_predicted_by_hand["right"]}
    paths = qlib.save(FIG_ID, image, panels, describe(args, selection, sample, scene, errors, spread, direction, camera))
    state_display_rule(paths["meta"])
    return paths


def state_display_rule(meta_path: Path) -> None:
    """Make the record-level ``display_rule`` of meta.json the rule this figure uses.

    qlib.save writes there the rule of the figures that draw a 512-point canonical map through the study's operator W.
    This figure draws another study's map by its own rule (``display_rule`` of ``describe``); two different rules in
    one record would contradict each other, so the top-level field is replaced and the replacement is noted.
    """
    record = json.loads(meta_path.read_text(encoding="utf-8"))
    own = record["meta"]["display_rule"]
    if record.get("display_rule") != own:
        record["display_rule"] = own
        record["display_rule_note"] = (f"set by {Path(__file__).name}: qlib.save writes the rule of the 512-point canonical "
                                       "maps here (qlib.DISPLAY_RULE), which this figure does not use")
        meta_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--azimuth", type=float, default=AZIMUTH_DEG, help="camera turn about the vertical, degrees")
    parser.add_argument("--elevation", type=float, default=ELEVATION_DEG, help="camera elevation, degrees")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    for name, path in build(args).items():
        log.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
