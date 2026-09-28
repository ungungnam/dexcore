"""One demonstration-synthesis method: dense tau_O + a target object -> a dense target demo.

WHY THIS LEVEL EXISTS. dexcore factorizes synthesis into three stages,

    D_s  --S-->  K_s  --Phi-->  K_t  --I-->  D_hat_t

but that factorization is itself a research CLAIM, not a law. An end-to-end generative model
(BimArt) answers the same question -- "produce a demonstration for this object trajectory" -- with
no selection, no transfer and no reconstruction anywhere in it. Comparing the two requires a level
at which they are the same kind of object, and the three stage registries are below that level:
a `Selector` takes a budget, a `Transfer` takes a sparse demo, a `Reconstructor` takes sparse +
tau_O. None of those signatures fit a method that consumes only tau_O.

So `Synthesizer` is that level, and it is deliberately the WHOLE of the arrow:

    SynthesisInput  ->  Demonstration (dense, on the target object)  +  diagnostics

`StagedSynthesizer` runs the three stages; `BimArtSynthesizer` runs a diffusion model. The
evaluation in `eval/` reads a saved .npy and cannot tell which produced it, which is exactly the
property that makes the comparison mean something.

WHAT EACH METHOD IS ALLOWED TO READ, AND WHY IT IS ENFORCED
-----------------------------------------------------------
Every synthesizer is given tau_O and the target object. Everything else about the source is
OPTIONAL and must be DECLARED in `consumes`; reading an undeclared input raises.

This is enforced rather than documented because the difference between the two families is
precisely which inputs they consume. "BimArt is conditioned on the object trajectory alone" is the
claim the comparison rests on, and a claim that lives only in a docstring is one a later edit can
break silently -- the run would still finish, the numbers would still look reasonable, and the
table would be wrong. Here it raises instead, and `consumes` is written into the provenance, so a
saved demonstration states what its method was allowed to see.

    source_object   the source geometry (NOT its motion -- that is already in tau_O)
    source_clip     which clip/window this is: names, frame numbers, fps. Identity, not content;
                    every method needs it to write a well-formed demonstration
    source_hand     the human hand states: joints, configs, contacts. THE gated one. A method
                    that declares this is transferring a demonstration; one that does not is
                    generating from the object trajectory

THE BUDGET IS PART OF THE RESULT. `hand_states_used` is how many source hand states the method
consumed at inference: the selection budget for the staged pipeline, 0 for an end-to-end model,
T for a dense per-frame transfer. That single number is the x-axis Experiment 1 sweeps, and
putting it in the common result is what lets a method with no budget be plotted against one that
has one -- as a horizontal line rather than as a missing row.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.object import ArticulatedObject
from src.data.trajectory import ObjectTrajectory

# The optional inputs a synthesizer may declare. tau_O and the target object are the mandatory
# contract and are never declared -- a method that needs neither is not a synthesizer.
OPTIONAL_INPUTS = ("source_object", "source_clip", "source_hand")


# --------------------------------------------------------------------------------- source clip
@dataclass(frozen=True)
class SourceClip:
    """WHICH clip this is: names, window, fps. Identity and timing, never hand content.

    Separated from the demonstration so that a method which must not see the human hand motion can
    still write a well-formed output -- `use_clip` and the absolute frame window are what the
    `dexcore_demo` block records, and getting them wrong makes a demo forget where it came from
    (see src/data/demo.py). Withholding them would force an end-to-end method to invent them.
    """
    sequence: str                  # "box_use_01"
    obj_name: str                  # the SOURCE object
    use_clip: str
    subject: str
    fps: float
    frame_start: int               # absolute index of local row 0
    frame_end: int                 # absolute, half-open
    total_frames: int              # length of the file the window was cut from
    path: str = ""

    @classmethod
    def of(cls, demo: Demonstration) -> "SourceClip":
        return cls(sequence=demo.name, obj_name=demo.obj_name, use_clip=demo.use_clip,
                   subject=demo.subject, fps=float(demo.fps),
                   frame_start=int(demo.frame_start), frame_end=int(demo.frame_end),
                   total_frames=int(demo.total_frames), path=str(demo.path))


# --------------------------------------------------------------------------------- the input
@dataclass(frozen=True)
class SynthesisInput:
    """What a synthesizer is given. Undeclared inputs raise on access; see the module docstring."""

    trajectory: ObjectTrajectory        # tau_O, DENSE, already on the target object
    target_object: ArticulatedObject
    # The SOURCE object's own trajectory. Not gated, and deliberately so: tau_O is a deterministic
    # function of it (src/data/target.py), so withholding it would protect nothing -- and a method
    # that drives an external reference generally has to hand over the trajectory in the source's
    # own frame, because the reference applies the canonical transfer itself.
    source_trajectory: Optional[ObjectTrajectory] = None
    _source_object: ArticulatedObject = None
    _source_clip: SourceClip = None
    _source_demo: Optional[Demonstration] = None
    allowed: frozenset = frozenset()
    seed: int = 0

    def __len__(self) -> int:
        """Frames the synthesizer must produce. tau_O is the length contract."""
        return len(self.trajectory)

    def _get(self, name: str, value):
        if name in self.allowed:
            return value
        declared = ", ".join(sorted(self.allowed)) or "nothing beyond tau_O and the target object"
        if name == "source_hand":
            why = ("Reading the human hand motion would make this a TRANSFER of a demonstration "
                   "rather than a synthesis from the object trajectory, which is the distinction "
                   "the comparison rests on. Declare it deliberately if that is what it is.")
        else:
            why = f"Add {name!r} to `consumes` if the method genuinely needs it."
        raise AttributeError(
            f"this synthesizer did not declare {name!r} in `consumes`, so reading it here is "
            f"refused. It declared: {declared}.\n" + why)

    @property
    def source_object(self) -> ArticulatedObject:
        return self._get("source_object", self._source_object)

    @property
    def source_clip(self) -> SourceClip:
        return self._get("source_clip", self._source_clip)

    @property
    def source_demo(self) -> Demonstration:
        """The dense source demonstration, INCLUDING its human hand states."""
        d = self._get("source_hand", self._source_demo)
        if d is None:
            raise ValueError("source_hand was declared but no source demonstration was supplied")
        return d

    @property
    def is_same_object(self) -> bool:
        """Object identity by NAME, the same rule src/data/target.py uses. Not gated: a method
        may always know whether the geometry changed."""
        return self._source_clip.obj_name == self.target_object.name


# --------------------------------------------------------------------------------- diagnostics
@dataclass
class SynthesisDiagnostics:
    """Per-frame evidence about one synthesis, whatever produced it.

    `stages` holds the sub-diagnostics a composite method already produces (transfer,
    reconstruction) rather than flattening them: each of those has its own summary and its own
    notion of failure, and collapsing them into one schema would lose exactly the detail they
    exist to report.
    """
    method: str = ""
    num_failed: int = 0                                    # frames the method could not answer
    failures: List[str] = field(default_factory=list)      # human-readable, one per failure
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict)   # name -> (T,)
    stages: Dict[str, object] = field(default_factory=dict)          # sub-diagnostic objects
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def all_ok(self) -> bool:
        return self.num_failed == 0

    def summary(self) -> str:
        lines = [f"{self.method}: {'ok' if self.all_ok else f'{self.num_failed} FAILED'}"]
        for name, d in self.stages.items():
            s = d.summary() if hasattr(d, "summary") else str(d)
            lines += [f"  [{name}] " + s.replace("\n", "\n  ")]
        lines += [f"  ! {f}" for f in self.failures[:20]]
        if len(self.failures) > 20:
            lines.append(f"  ... and {len(self.failures) - 20} more")
        lines += [f"  - {n}" for n in self.notes]
        return "\n".join(lines)


# --------------------------------------------------------------------------------- the result
@dataclass
class SynthesisResult:
    """The generated demonstration, plus what it cost and how it went."""

    demo: Demonstration                 # DENSE, on the target object -- the thing the run is for
    trajectory: ObjectTrajectory        # tau_O, returned unchanged; synthesis must not touch it
    diagnostics: SynthesisDiagnostics
    method: str = ""
    consumes: frozenset = frozenset()   # what the method was ALLOWED to read
    # source hand states consumed AT INFERENCE. Staged: the selection budget. End-to-end: 0.
    # Dense per-frame transfer: T. See the module docstring.
    hand_states_used: int = 0
    extras: Dict[str, object] = field(default_factory=dict)   # method-specific intermediates

    @property
    def budget(self) -> float:
        """`hand_states_used` as a fraction of the dense clip -- Experiment 1's x-axis."""
        return self.hand_states_used / max(len(self.trajectory), 1)

    def require_ok(self) -> "SynthesisResult":
        if not self.diagnostics.all_ok:
            raise RuntimeError("synthesis failed on some frames:\n" + self.diagnostics.summary())
        return self

    def summary(self) -> str:
        return "\n".join([
            f"{self.method}  budget {self.hand_states_used}/{len(self.trajectory)} "
            f"source hand states ({self.budget:.1%})  consumes={sorted(self.consumes) or '[]'}",
            self.demo.summary(),
            self.diagnostics.summary(),
        ])


# --------------------------------------------------------------------------------- the method
class Synthesizer(abc.ABC):
    """A whole demonstration-synthesis method. Subclasses implement `synthesize`."""

    name: str = "base"
    # Optional inputs this method reads, beyond tau_O and the target object. Enforced, and written
    # into the output's provenance -- see the module docstring on why this is not just a comment.
    consumes: frozenset = frozenset()

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        unknown = set(cls.consumes) - set(OPTIONAL_INPUTS)
        if unknown:
            raise TypeError(f"{cls.__name__}.consumes has unknown input(s) {sorted(unknown)}. "
                            f"Known: {list(OPTIONAL_INPUTS)}")

    def __init__(self, **kwargs):
        self.options = dict(kwargs)

    @abc.abstractmethod
    def synthesize(self, inp: SynthesisInput) -> SynthesisResult:
        """tau_O + a target object -> a dense demonstration of the same length as tau_O."""

    # ------------------------------------------------------------------ the contract
    def run(self, inp: SynthesisInput) -> SynthesisResult:
        """`synthesize` plus the checks every method must satisfy. Drivers call THIS."""
        got = self.synthesize(inp)
        self._check(got, inp)
        return got

    def _check(self, res: SynthesisResult, inp: SynthesisInput) -> None:
        """The three ways an output can be wrong in a way nothing downstream would catch."""
        n = len(inp)
        if len(res.demo) != n:
            raise RuntimeError(
                f"{self.describe()}: produced {len(res.demo)} frames for a {n}-frame tau_O. "
                f"Synthesis must answer EVERY frame the object occupies; a shorter output is a "
                f"different clip, not a worse one.")
        if res.demo.obj_name != inp.target_object.name:
            raise RuntimeError(
                f"{self.describe()}: output is named for object {res.demo.obj_name!r} but the "
                f"target is {inp.target_object.name!r}. DexMachina resolves geometry from that "
                f"field, so this would train against the wrong mesh with no error.")
        for f in ("obj_pos", "obj_arti"):
            if not np.array_equal(getattr(res.demo, f), getattr(inp.trajectory, f)):
                raise RuntimeError(
                    f"{self.describe()}: the output's {f} differs from tau_O. The object "
                    f"trajectory is an INPUT; a method that moves it has answered a different "
                    f"question and its metrics are not comparable.")

    def describe(self) -> str:
        if not self.options:
            return self.name
        return f"{self.name}(" + ", ".join(f"{k}={v}" for k, v in sorted(self.options.items())) + ")"

    def __repr__(self) -> str:
        return f"<{self.describe()}>"


# ------------------------------------------------------------------------------- the registry
_REGISTRY: Dict[str, type] = {}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def build(name: str, **kwargs) -> Synthesizer:
    if name not in _REGISTRY:
        raise ValueError(f"unknown synthesizer {name!r}. Known: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[name](**kwargs)


def available() -> list:
    return sorted(_REGISTRY)
