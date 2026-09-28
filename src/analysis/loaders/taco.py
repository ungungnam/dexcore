"""TACO -> `HOITrajectory`. The (verb, tool, target) dataset, 2317 sequences.

LAYOUT, as the release ships it (v1):

    <root>/Hand_Poses/<triplet>/<seq>/{left,right}_hand.pkl        {frame: {hand_pose(48), hand_trans(3)}}
    <root>/Hand_Poses/<triplet>/<seq>/{left,right}_hand_shape.pkl  {"hand_shape": (10,)}
    <root>/Object_Poses/<triplet>/<seq>/tool_<id>.npy              (T,4,4) SE(3), world, metres
    <root>/Object_Poses/<triplet>/<seq>/target_<id>.npy            (T,4,4)
    <root>/object_models_released/<id>_cm.obj                      mesh, CENTIMETRES

`<triplet>` is the literal directory name "(verb, tool, target)" -- the action label is in the
path, which is why an index can be built without opening a single file.

THREE CONVENTIONS TAKEN FROM TACO'S OWN LOADER (dataset_utils/hand_pose_loader.py), not guessed:
  * `hand_pose` is 48 = 3 global axis-angle + 45 finger axis-angle, `use_pca=False`.
  * the MANO layer is built with `center_idx=0`, so joint 0 sits at the origin before translation:
    the WRIST IN WORLD IS EXACTLY `hand_trans`, and no MANO forward pass is needed to get it.
  * MANO output is millimetres and is divided by 1000 -- so `hand_trans` is already metres, the
    same unit as the object matrices. Nothing here rescales poses.

The pickles hold torch tensors, so unpickling needs torch importable even though this module does
no tensor maths; `_to_numpy` is the only place that touches them.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Sequence

import numpy as np

from src import geometry as G
from src.analysis.schema import ActionLabel, HandTrack, HOITrajectory, ObjectTrack

log = logging.getLogger(__name__)

DATASET = "taco"
DEFAULT_FPS = 30.0                 # the release's RGB/depth videos are decoded at fps=30
MESH_SCALE = 0.01                  # "<id>_cm.obj" -> metres
_TRIPLET_RE = re.compile(r"^\((?P<verb>[^,]+),(?P<tool>[^,]+),(?P<target>[^,]+)\)$")


def default_root() -> Path:
    """$DEXCORE_TACO_ROOT, or the server's copy. Never hard-coded anywhere else."""
    return Path(os.environ.get("DEXCORE_TACO_ROOT", "/backups/uhnam/TACO"))


# ------------------------------------------------------------------------------------- index
class SequenceRef(NamedTuple):
    """Where one sequence lives and what it is -- built from paths alone, so indexing is cheap."""
    triplet: str
    sequence: str
    action: ActionLabel

    @property
    def sequence_id(self) -> str:
        return f"{self.triplet}/{self.sequence}"


def parse_triplet(triplet: str) -> ActionLabel:
    """'(pour in some, bowl, plate)' -> ActionLabel. Verbs contain spaces but never commas."""
    m = _TRIPLET_RE.match(triplet.strip())
    if not m:
        raise ValueError(f"unparseable TACO triplet directory name: {triplet!r}")
    return ActionLabel(verb=m.group("verb").strip(), tool=m.group("tool").strip(),
                       target=m.group("target").strip(), raw=triplet)


def index(root=None, verbs: Optional[Sequence[str]] = None,
          tools: Optional[Sequence[str]] = None,
          targets: Optional[Sequence[str]] = None) -> List[SequenceRef]:
    """Every sequence under `root`, optionally filtered by action parts. Opens no data file."""
    root = Path(root or default_root())
    hand_root = root / "Hand_Poses"
    if not hand_root.is_dir():
        raise FileNotFoundError(f"TACO Hand_Poses not found under {root}. "
                                f"Set $DEXCORE_TACO_ROOT or pass --root.")
    out: List[SequenceRef] = []
    for triplet in sorted(os.listdir(hand_root)):
        if not (hand_root / triplet).is_dir():
            continue
        try:
            action = parse_triplet(triplet)
        except ValueError:
            log.warning("skipping unparseable triplet dir %r", triplet)
            continue
        if verbs and action.verb not in verbs:
            continue
        if tools and action.tool not in tools:
            continue
        if targets and action.target not in targets:
            continue
        for seq in sorted(os.listdir(hand_root / triplet)):
            if (hand_root / triplet / seq).is_dir():
                out.append(SequenceRef(triplet, seq, action))
    return out


# -------------------------------------------------------------------------------- unpickling
def _to_numpy(x) -> np.ndarray:
    """torch tensor or array -> float64 numpy. The pickles store tensors; nothing else does."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:          # the pickles reference torch by module name
        raise RuntimeError(
            f"cannot unpickle {path}: {e}. TACO hand poses are pickled torch tensors -- run in an "
            f"environment with torch (conda activate dexmachina).") from e


# ------------------------------------------------------------------------------------- pieces
def _object_tracks(root: Path, ref: SequenceRef) -> List[ObjectTrack]:
    d = root / "Object_Poses" / ref.triplet / ref.sequence
    tracks: List[ObjectTrack] = []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".npy"):
            continue
        role, _, obj_id = fn[:-4].partition("_")
        if role not in ("tool", "target"):
            log.warning("%s: unexpected object pose file %r", ref.sequence_id, fn)
            continue
        T = np.load(d / fn).astype(np.float64)          # (T,4,4), world, metres
        if T.ndim != 3 or T.shape[1:] != (4, 4):
            raise ValueError(f"{d/fn}: expected (T,4,4), got {T.shape}")
        mesh = root / "object_models_released" / f"{obj_id}_cm.obj"
        tracks.append(ObjectTrack(
            name=obj_id, role=role, pos=T[:, :3, 3], quat=G.R_to_wxyz(T[:, :3, :3]),
            arti=None,                                   # TACO objects are rigid
            mesh_path=str(mesh) if mesh.exists() else None, mesh_scale=MESH_SCALE))
    return tracks


def _hand_track(root: Path, ref: SequenceRef, side: str) -> Optional[HandTrack]:
    d = root / "Hand_Poses" / ref.triplet / ref.sequence
    pose_path = d / f"{side}_hand.pkl"
    if not pose_path.exists():
        return None
    frames = _load_pickle(pose_path)
    keys = sorted(frames)                                 # '00001', '00002', ... zero-padded
    if not keys:
        return None
    # `sorted` on the string keys is only the frame order if the keys are zero-padded AND there is
    # no gap. A missing middle frame would otherwise shorten the hand track, `load` would trim a
    # frame off the END of the object track to match, and every frame after the gap would be
    # silently misaligned against the object -- with a plausible-looking note recording the trim.
    nums = [int(k) for k in keys]
    if nums != list(range(nums[0], nums[0] + len(nums))):
        raise ValueError(f"{pose_path}: frame keys are not a contiguous run "
                         f"({nums[0]}..{nums[-1]} over {len(nums)} frames)")
    theta = np.stack([_to_numpy(frames[k]["hand_pose"]).reshape(-1) for k in keys])   # (T,48)
    trans = np.stack([_to_numpy(frames[k]["hand_trans"]).reshape(3) for k in keys])   # (T,3)

    shape_path = d / f"{side}_hand_shape.pkl"
    beta = _to_numpy(_load_pickle(shape_path)["hand_shape"]) if shape_path.exists() else None

    from scipy.spatial.transform import Rotation
    quat = G.R_to_wxyz(Rotation.from_rotvec(theta[:, :3]).as_matrix())
    return HandTrack(side=side, root_pos=trans, root_quat=quat,
                     finger_pose=theta[:, 3:], shape=beta)


# ------------------------------------------------------------------------------------ loading
def load(ref: SequenceRef, root=None, fps: float = DEFAULT_FPS,
         with_hands: bool = True) -> HOITrajectory:
    """One TACO sequence, normalised onto the common schema.

    Hand and object annotations are counted independently upstream, so this trims every track to
    the shortest and records the trim in `notes` rather than papering over it -- every temporal
    metric divides by the frame count. No sequence in release v1 actually needs it: all 2317 come
    out with the hand, tool and target arrays already the same length.
    """
    root = Path(root or default_root())
    objects = _object_tracks(root, ref)
    # Skipping the hands is worth a parameter: they are two 14 MB pickles per sequence against a
    # 17 KB pose array, so an object-only sweep over the release is an order of magnitude cheaper.
    hands = {s: h for s in ("left", "right")
             if with_hands and (h := _hand_track(root, ref, s)) is not None}
    if not objects and not hands:
        raise FileNotFoundError(f"{ref.sequence_id}: neither object nor hand annotations found")

    lens = [len(o) for o in objects] + [len(h) for h in hands.values()]
    n = min(lens)
    notes: List[str] = []
    if max(lens) != n:
        notes.append(f"tracks trimmed {max(lens)} -> {n} frames")
        objects = [ObjectTrack(name=o.name, role=o.role, pos=o.pos[:n], quat=o.quat[:n],
                               arti=None if o.arti is None else o.arti[:n],
                               mesh_path=o.mesh_path, mesh_scale=o.mesh_scale) for o in objects]
        hands = {s: HandTrack(side=h.side, root_pos=h.root_pos[:n], root_quat=h.root_quat[:n],
                              finger_pose=h.finger_pose[:n], shape=h.shape)
                 for s, h in hands.items()}

    return HOITrajectory(
        dataset=DATASET, sequence_id=ref.sequence_id, action=ref.action, objects=objects,
        hands=hands, fps=fps, source_path=str(root / "Hand_Poses" / ref.triplet / ref.sequence),
        notes=notes)


def iter_sequences(root=None, refs: Optional[Iterable[SequenceRef]] = None,
                   fps: float = DEFAULT_FPS, skip_errors: bool = True,
                   **index_kwargs) -> Iterator[HOITrajectory]:
    """Stream sequences. A sweep over 2317 of them must never die on one bad directory."""
    root = Path(root or default_root())
    for ref in (refs if refs is not None else index(root, **index_kwargs)):
        try:
            yield load(ref, root=root, fps=fps)
        except Exception as e:                     # noqa: BLE001 -- reported, then skipped
            if not skip_errors:
                raise
            log.warning("%s: %s: %s", ref.sequence_id, type(e).__name__, e)
