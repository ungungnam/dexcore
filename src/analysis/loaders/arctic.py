"""ARCTIC raw_seqs -> `HOITrajectory`. The articulated-object dataset, 301 sequences.

LAYOUT:

    <root>/raw_seqs/<subject>/<obj>_<action>_<clip>.object.npy       (T,7)
    <root>/raw_seqs/<subject>/<obj>_<action>_<clip>.mano.npy         {"left"|"right": {...}}
    <root>/raw_seqs/<subject>/<obj>_<action>_<clip>.egocam.dist.npy  camera, not read here
    <root>/raw_seqs/<subject>/<obj>_<action>_<clip>.smplx.npy        full body, not read here
    <root>/meta/object_vtemplates/<obj>/mesh.obj                     mesh, MILLIMETRES

UNITS, verified against the files rather than assumed:

    object.npy[:, 0]     articulation, RADIANS          -> kept as is
    object.npy[:, 1:4]   root rotation, axis-angle
    object.npy[:, 4:7]   root translation, MILLIMETRES  -> /1000
    mano[side]["trans"]  MANO transl, METRES            -> kept as is

ARCTIC's action vocabulary is two words, `grab` and `use`, and the object is the only other label
-- which is why an action-conditioned study starts with TACO and uses this loader for contrast.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Iterable, Iterator, List, NamedTuple, Optional, Sequence

import numpy as np

from src import geometry as G
from src.analysis.schema import ActionLabel, HandTrack, HOITrajectory, ObjectTrack

log = logging.getLogger(__name__)

DATASET = "arctic"
DEFAULT_FPS = 30.0                 # raw_seqs are the image-synced rate; override with fps=
MESH_SCALE = 0.001                 # meta/object_vtemplates meshes are millimetres
POS_SCALE = 0.001                  # object.npy translation is millimetres
# "_retake" / "_retake2" suffixes are re-recordings of a clip, 9 of them in the release. They
# are ordinary sequences and are kept -- dropping them would silently shrink the dataset by 3%.
_SEQ_RE = re.compile(r"^(?P<obj>[a-z]+)_(?P<action>grab|use)_(?P<clip>\d+)(?P<retake>_retake\d*)?$")


def default_root() -> Path:
    """$DEXCORE_ARCTIC_ROOT, or the server's copy."""
    return Path(os.environ.get("DEXCORE_ARCTIC_ROOT", "/data/uhnam/ARCTIC/data"))


class SequenceRef(NamedTuple):
    subject: str
    name: str                      # "box_grab_01"
    action: ActionLabel

    @property
    def sequence_id(self) -> str:
        return f"{self.subject}/{self.name}"


def parse_name(name: str) -> ActionLabel:
    """'box_grab_01' -> ActionLabel(verb='grab', tool='box'). The clip index is not a label."""
    m = _SEQ_RE.match(name)
    if not m:
        raise ValueError(f"unparseable ARCTIC sequence name: {name!r}")
    return ActionLabel(verb=m.group("action"), tool=m.group("obj"), target="", raw=name)


def index(root=None, verbs: Optional[Sequence[str]] = None,
          objects: Optional[Sequence[str]] = None,
          subjects: Optional[Sequence[str]] = None) -> List[SequenceRef]:
    """Every sequence under `root`, from filenames alone."""
    root = Path(root or default_root())
    seq_root = root / "raw_seqs"
    if not seq_root.is_dir():
        raise FileNotFoundError(f"ARCTIC raw_seqs not found under {root}. "
                                f"Set $DEXCORE_ARCTIC_ROOT or pass --root.")
    out: List[SequenceRef] = []
    for subject in sorted(os.listdir(seq_root)):
        if subjects and subject not in subjects:
            continue
        if not (seq_root / subject).is_dir():
            continue
        for fn in sorted(os.listdir(seq_root / subject)):
            if not fn.endswith(".object.npy"):
                continue
            name = fn[: -len(".object.npy")]
            try:
                action = parse_name(name)
            except ValueError:
                log.warning("skipping unparseable sequence %r", name)
                continue
            if verbs and action.verb not in verbs:
                continue
            if objects and action.tool not in objects:
                continue
            out.append(SequenceRef(subject, name, action))
    return out


def load(ref: SequenceRef, root=None, fps: float = DEFAULT_FPS) -> HOITrajectory:
    """One ARCTIC sequence, normalised onto the common schema."""
    from scipy.spatial.transform import Rotation

    root = Path(root or default_root())
    base = root / "raw_seqs" / ref.subject / ref.name
    obj = np.load(f"{base}.object.npy").astype(np.float64)        # (T,7)
    if obj.ndim != 2 or obj.shape[1] != 7:
        raise ValueError(f"{base}.object.npy: expected (T,7), got {obj.shape}")

    mesh = root / "meta" / "object_vtemplates" / ref.action.tool / "mesh.obj"
    track = ObjectTrack(
        name=ref.action.tool, role="object",
        pos=obj[:, 4:7] * POS_SCALE,
        quat=G.R_to_wxyz(Rotation.from_rotvec(obj[:, 1:4]).as_matrix()),
        arti=obj[:, 0],                                           # radians, single revolute joint
        mesh_path=str(mesh) if mesh.exists() else None, mesh_scale=MESH_SCALE)

    hands = {}
    mano_path = Path(f"{base}.mano.npy")
    if mano_path.exists():
        mano = np.load(mano_path, allow_pickle=True).item()
        for side in ("left", "right"):
            d = mano.get(side)
            if d is None:
                continue
            rot = np.asarray(d["rot"], dtype=np.float64)          # (T,3) axis-angle
            hands[side] = HandTrack(
                side=side, root_pos=np.asarray(d["trans"], dtype=np.float64),
                root_quat=G.R_to_wxyz(Rotation.from_rotvec(rot).as_matrix()),
                finger_pose=np.asarray(d["pose"], dtype=np.float64),
                shape=np.asarray(d["shape"], dtype=np.float64))

    lens = [len(track)] + [len(h) for h in hands.values()]
    n, notes = min(lens), []
    if max(lens) != n:
        notes.append(f"tracks trimmed {max(lens)} -> {n} frames")
        track = ObjectTrack(name=track.name, role=track.role, pos=track.pos[:n],
                            quat=track.quat[:n], arti=track.arti[:n],
                            mesh_path=track.mesh_path, mesh_scale=track.mesh_scale)
        hands = {s: HandTrack(side=h.side, root_pos=h.root_pos[:n], root_quat=h.root_quat[:n],
                              finger_pose=h.finger_pose[:n], shape=h.shape)
                 for s, h in hands.items()}

    return HOITrajectory(dataset=DATASET, sequence_id=ref.sequence_id, action=ref.action,
                         objects=[track], hands=hands, fps=fps, source_path=str(base),
                         notes=notes)


def iter_sequences(root=None, refs: Optional[Iterable[SequenceRef]] = None,
                   fps: float = DEFAULT_FPS, skip_errors: bool = True,
                   **index_kwargs) -> Iterator[HOITrajectory]:
    root = Path(root or default_root())
    for ref in (refs if refs is not None else index(root, **index_kwargs)):
        try:
            yield load(ref, root=root, fps=fps)
        except Exception as e:                     # noqa: BLE001 -- reported, then skipped
            if not skip_errors:
                raise
            log.warning("%s: %s: %s", ref.sequence_id, type(e).__name__, e)
