"""One dataset-neutral trajectory type, so every analysis metric is written once.

WHY A SEPARATE TYPE FROM `src/data/demo.Demonstration`. A `Demonstration` is DexMachina's format:
exactly one articulated object, two MANO hands, contact arrays. That is the right type for the
synthesis pipeline and the wrong one for dataset analysis, where a sequence may carry

    ARCTIC   one object with a 1-DoF revolute joint, two hands, action = grab | use
    TACO     two RIGID objects (a tool and a target), two hands, action = a (verb, tool, target)

`HOITrajectory` is the intersection that both fit into without either being distorted: a list of
object tracks that each MAY have an articulation channel, plus per-side hand tracks, plus the
action label the analysis groups by. Loaders normalise units and conventions ON THE WAY IN, so a
metric never asks which dataset it is looking at:

    positions   metres, world frame of the source capture
    rotations   wxyz quaternions, sign-unrolled (src/geometry's convention)
    articulation radians, or None for a rigid object
    time        frame index + an explicit `fps`

WHAT IS DELIBERATELY NOT HERE. No image, no camera, no mesh loading. An object track carries the
PATH to its mesh so a metric that needs geometry can load it, and the 99% of metrics that only
need poses stay cheap over thousands of sequences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src import geometry as G

SIDES = ("left", "right")


# ------------------------------------------------------------------------------ action label
@dataclass(frozen=True)
class ActionLabel:
    """What the sequence IS, as the dataset labels it.

    `verb` is the grouping key for every action-conditioned distribution. TACO labels a full
    (verb, tool, target) triplet; ARCTIC labels only grab/use and the object, so `tool` carries the
    object name and `target` is empty. `raw` keeps the dataset's own string for traceability.
    """
    verb: str
    tool: str = ""
    target: str = ""
    raw: str = ""

    @property
    def triplet(self) -> str:
        return f"({self.verb}, {self.tool}, {self.target})"

    def __str__(self) -> str:
        return self.raw or self.triplet


# ------------------------------------------------------------------------------ object track
@dataclass
class ObjectTrack:
    """A dense pose sequence for one object, plus its role in the interaction.

    `role` is what the action grammar calls this object -- "tool" is the one the hand drives,
    "target" the one acted upon, "object" when the dataset does not distinguish (ARCTIC). Metrics
    that talk about tool-relative motion select by role, never by index.
    """
    name: str                              # dataset-local id, e.g. "035" or "box"
    role: str                              # "tool" | "target" | "object"
    pos: np.ndarray                        # (T,3) metres, world
    quat: np.ndarray                       # (T,4) wxyz, sign-unrolled
    arti: Optional[np.ndarray] = None      # (T,) radians, or None when rigid
    mesh_path: Optional[str] = None
    mesh_scale: float = 1.0                # multiply mesh vertices by this to get metres

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.quat = G.unroll_quat(np.asarray(self.quat, dtype=np.float64).reshape(-1, 4))
        if self.arti is not None:
            self.arti = np.asarray(self.arti, dtype=np.float64).reshape(-1)
        n = {len(self.pos), len(self.quat)} | ({len(self.arti)} if self.arti is not None else set())
        if len(n) != 1:
            raise ValueError(f"ragged object track {self.name!r}: {sorted(n)}")

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def is_articulated(self) -> bool:
        return self.arti is not None

    def R(self) -> np.ndarray:
        """(T,3,3) rotation matrices."""
        return G.wxyz_to_R(self.quat)

    def transform(self) -> np.ndarray:
        """(T,4,4) world pose."""
        return G.affine(self.R(), self.pos)


# -------------------------------------------------------------------------------- hand track
@dataclass
class HandTrack:
    """A MANO hand over time, in the parameterisation both datasets actually store.

    `root_pos` / `root_quat` are the MANO ROOT -- read straight from the annotation, with no MANO
    forward pass, which is what keeps a 2317-sequence sweep cheap. The root is NOT the same thing
    in both datasets and the difference is a real offset, not a naming quibble:

        TACO     the layer is built with `center_idx=0`, so joint 0 sits at the origin before
                 translation and the root IS the wrist joint, exactly.
        ARCTIC   `trans` is the smplx-convention `transl`, applied after skinning; the wrist joint
                 is that plus the template's own joint-0 offset (order 10 cm, beta-dependent).

    So root-position differences are comparable WITHIN a dataset and only approximately comparable
    across the two. An exact cross-dataset wrist needs a MANO forward pass, which is what `joints`
    is for -- it stays None unless a caller fills it.
    """
    side: str
    root_pos: np.ndarray                  # (T,3) metres, world
    root_quat: np.ndarray                 # (T,4) wxyz, sign-unrolled
    finger_pose: np.ndarray                # (T,45) axis-angle, MANO order
    shape: Optional[np.ndarray] = None     # (10,) MANO beta
    joints: Optional[np.ndarray] = None    # (T,21,3) metres, world -- only if computed

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        self.root_pos = np.asarray(self.root_pos, dtype=np.float64).reshape(-1, 3)
        self.root_quat = G.unroll_quat(np.asarray(self.root_quat, dtype=np.float64).reshape(-1, 4))
        if self.shape is not None:
            self.shape = np.asarray(self.shape, dtype=np.float64).reshape(-1)
        # Check the frame count BEFORE reshaping. A (2T, 45) finger_pose against a (T,3) root
        # reshapes to (T, 90) without complaint and welds pairs of frames together, which nothing
        # downstream can detect -- `len(track)` reads T either way.
        fp = np.asarray(self.finger_pose, dtype=np.float64)
        lens = {len(self.root_pos), len(self.root_quat), len(fp)}
        if len(lens) != 1:
            raise ValueError(f"ragged hand track {self.side!r}: root_pos {len(self.root_pos)}, "
                             f"root_quat {len(self.root_quat)}, finger_pose {len(fp)}")
        self.finger_pose = fp.reshape(len(self.root_pos), -1)

    def __len__(self) -> int:
        return len(self.root_pos)


# --------------------------------------------------------------------------------- sequence
@dataclass
class HOITrajectory:
    """One hand-object interaction sequence, normalised.

    Every track shares one timeline: `num_frames` rows at `fps`. A loader that cannot make that
    true (a hand annotated over fewer frames than the object, say) must say so by trimming and
    recording the trim in `notes` -- silently ragged tracks would corrupt every temporal metric.
    """
    dataset: str
    sequence_id: str
    action: ActionLabel
    objects: List[ObjectTrack] = field(default_factory=list)
    hands: Dict[str, HandTrack] = field(default_factory=dict)
    fps: float = 30.0
    source_path: str = ""
    notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        lens = {len(o) for o in self.objects} | {len(h) for h in self.hands.values()}
        if len(lens) > 1:
            raise ValueError(f"{self.sequence_id}: tracks disagree on length: {sorted(lens)}")

    def __len__(self) -> int:
        return self.num_frames

    @property
    def num_frames(self) -> int:
        if self.objects:
            return len(self.objects[0])
        if self.hands:
            return len(next(iter(self.hands.values())))
        return 0

    @property
    def duration_s(self) -> float:
        return self.num_frames / float(self.fps)

    def by_role(self, role: str) -> Optional[ObjectTrack]:
        """The first object with this role, or None. Roles are unique in both datasets so far."""
        for o in self.objects:
            if o.role == role:
                return o
        return None

    @property
    def tool(self) -> Optional[ObjectTrack]:
        """The driven object: TACO's tool, or ARCTIC's single object."""
        return self.by_role("tool") or self.by_role("object")

    @property
    def target(self) -> Optional[ObjectTrack]:
        return self.by_role("target")

    def times(self) -> np.ndarray:
        return np.arange(self.num_frames) / float(self.fps)

    def summary(self) -> str:
        objs = ", ".join(f"{o.role}:{o.name}{'(arti)' if o.is_articulated else ''}"
                         for o in self.objects)
        hands = ", ".join(sorted(self.hands))
        return (f"{self.dataset}/{self.sequence_id}  {self.action}\n"
                f"  {self.num_frames} frames @ {self.fps:g} fps ({self.duration_s:.1f} s)\n"
                f"  objects: {objs or '(none)'}\n"
                f"  hands:   {hands or '(none)'}"
                + ("\n  notes:   " + "; ".join(self.notes) if self.notes else ""))
