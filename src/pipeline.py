"""The driver: everything that is the same whatever method produces the demonstration.

    load D_s  ->  resolve objects  ->  build tau_O  ->  <SYNTHESIZER>  ->  save

Only the boxed step differs between dexcore's three-stage pipeline and an end-to-end generative
model, which is why the synthesizer is the unit of replacement (src/synthesis/base.py) and this
module holds NO research logic at all.

TAU_O IS BUILT HERE, BEFORE THE METHOD RUNS, AND THAT IS DELIBERATE. The target object's dense
trajectory is a property of the OBJECT -- for the same object it is the source's own, and for a
different one it is the canonical-frame transfer (src/data/target.py). Every method receives the
same tau_O, so nothing about the object trajectory can differ between two rows of a comparison.
It also means an end-to-end method gets tau_O without ever touching the source demonstration's
hand states, which is exactly the input separation `SynthesisInput` enforces.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import PipelineConfig
from src.data.demo import Demonstration
from src.data.object import ArticulatedObject, get_object
from src.data.target import build_target_trajectory
from src.data.trajectory import ObjectTrajectory
from src.synthesis.base import SourceClip, SynthesisInput, SynthesisResult, Synthesizer
from src.synthesis.base import build as build_synthesizer
from src.synthesis import staged as _staged   # noqa: F401  populates the synthesizer registry


@dataclass
class PipelineOutput:
    """One run: the config, the inputs it resolved, and what the method produced."""

    config: PipelineConfig
    source_demo: Demonstration
    source_object: ArticulatedObject
    target_object: ArticulatedObject
    trajectory: ObjectTrajectory
    result: SynthesisResult

    # ------------------------------------------------------------------ common view
    @property
    def demo(self) -> Demonstration:
        """The generated dense target demonstration -- the thing the run is for."""
        return self.result.demo

    @property
    def hand_states_used(self) -> int:
        return self.result.hand_states_used

    @property
    def budget(self) -> float:
        return self.result.budget

    # ------------------------------------------------------------------ staged-only view
    def _staged(self, attr: str):
        """Intermediates only a staged run has. Raises with WHY, rather than AttributeError."""
        if not hasattr(self.result, attr):
            raise AttributeError(
                f"{attr!r} is an intermediate of the three-stage pipeline; this run used "
                f"synthesizer={self.config.synthesizer!r}, which has no such stage. Use "
                f"`output.demo` for the result, or `output.result.diagnostics` for what happened.")
        return getattr(self.result, attr)

    @property
    def sparse_demo(self) -> Demonstration:
        return self._staged("sparse_demo")

    @property
    def transfer(self):
        return self._staged("transfer")

    @property
    def reconstruction(self):
        return self._staged("reconstruction")

    def summary(self) -> str:
        return self.result.summary()

    # ------------------------------------------------------------------ saving
    def save(self, out_dir=None) -> Path:
        """Write the generated demo, the method's intermediates, and the config into one directory."""
        out = Path(out_dir or self.config.output_path())
        out.mkdir(parents=True, exist_ok=True)
        self.config.save(out / "config.yaml")
        self.demo.save(out / f"{self.demo.name}.npy")

        files = {"target_demo": f"{self.demo.name}.npy"}
        diag = {
            "synthesizer": self.result.method,
            "consumes": sorted(self.result.consumes),
            "hand_states_used": self.result.hand_states_used,
            "budget": round(self.result.budget, 6),
            "num_failed": self.result.diagnostics.num_failed,
            "failures": self.result.diagnostics.failures[:200],
            "notes": self.result.diagnostics.notes,
            "per_frame": {k: (np.asarray(v) * 1000).round(4).tolist()
                          for k, v in self.result.diagnostics.per_frame.items()},
        }
        if hasattr(self.result, "sparse_demo"):
            # K_s -- what SELECTION chose, still on the source object
            self.sparse_demo.save(out / "selected_keyframes.npy")
            # K_t -- the same keyframes after TRANSFER, on the target object. Saved because it is
            # the only place Phi's output can be inspected on its own: everything downstream of it
            # has been through interpolation and IK, so a fault here is otherwise
            # indistinguishable from a fault there.
            self.transfer.demo.save(out / "transferred_keyframes.npy")
            files.update(selected_keyframes="selected_keyframes.npy",
                         transferred_keyframes="transferred_keyframes.npy")
            diag["selected_frames"] = [int(f) for f in self.sparse_demo.frames()]
        diag["files"] = files
        (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
        return out


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def build_input(config: PipelineConfig, synthesizer: Synthesizer):
    """Resolve everything the method may read, and withhold everything it did not declare.

    -> (SynthesisInput, source_demo, source_object, target_object, trajectory)
    """
    source_demo = Demonstration.load(
        path=config.demo_path, obj_name=config.obj_name, use_clip=config.use_clip,
        subject=config.subject, frame_start=config.frame_start, frame_end=config.frame_end)
    source_object = get_object(config.obj_name)
    target_object = get_object(config.resolved_target)
    trajectory = build_target_trajectory(source_demo, target_object, source_object)

    source_demo.provenance.config = config.to_dict()
    source_demo.provenance.seed = config.seed
    source_demo.provenance.source_object = source_object.name
    source_demo.provenance.target_object = target_object.name

    inp = SynthesisInput(
        trajectory=trajectory, target_object=target_object,
        source_trajectory=ObjectTrajectory.from_demo(source_demo, source="source demo"),
        _source_object=source_object, _source_clip=SourceClip.of(source_demo),
        _source_demo=source_demo, allowed=frozenset(synthesizer.consumes), seed=config.seed)
    return inp, source_demo, source_object, target_object, trajectory


def run(config: PipelineConfig, verbose: bool = True) -> PipelineOutput:
    """Synthesize one demonstration from one config, whatever method the config names."""
    config.validate()
    seed_everything(config.seed)

    synthesizer = build_synthesizer(config.synthesizer, **config.resolve_synthesizer_options())
    inp, source_demo, source_object, target_object, trajectory = build_input(config, synthesizer)
    if verbose:
        print(f"[synthesize] {synthesizer.describe()}")
        print(f"             consumes {sorted(synthesizer.consumes) or '[]'} "
              f"| tau_O {len(trajectory)} frames [{trajectory.source}]")

    result = synthesizer.run(inp)
    if verbose:
        print(f"[result]     {result.hand_states_used}/{len(trajectory)} source hand states used "
              f"({result.budget:.1%})")
        print(result.diagnostics.summary())

    return PipelineOutput(config=config, source_demo=source_demo, source_object=source_object,
                          target_object=target_object, trajectory=trajectory, result=result)
