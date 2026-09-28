"""Demo/CV-level metrics on a SAVED demonstration -- evaluation level A.

Everything here takes a `.npy` off disk and knows nothing about how it was made, so a dexcore
output, a DexMachina reference and a baseline's demo are all scored by the same code. A metric that
can only be computed inside the pipeline that produced the data cannot be used to compare that
pipeline against anything else.

This is the counterpart of `eval/downstream/` (level B, external frameworks). The two live in
sibling packages because they answer different questions and fail in different ways: a CV metric is
a property of the demonstration, a downstream metric is a property of a policy trained on it.

Per-frame values are saved alongside the aggregates so failure frames can be found and rendered
later, rather than averaged into a single number that hides them.

READ THE NUMBERS AGAINST THE SOURCE DEMO'S OWN, NOT AGAINST ZERO. The generic MANO URDF is not the
ARCTIC MANO fit and the object collision meshes are decimated, so a real human demonstration scores
~5 mm of penetration and a sub-millimetre self-collision gap. Only the excess over that baseline is
attributable to a reconstruction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.data.demo import Demonstration
from src.data.object import get_object
from src.metrics.collision import hand_collision
from src.metrics.penetration import hand_object_penetration
from src.metrics.reconstruction import reconstruction_error, trajectory_smoothness

METRICS = ("penetration", "collision", "smoothness", "reconstruction")


@dataclass
class EvalReport:
    demo_name: str = ""
    demo_path: str = ""
    aggregate: Dict[str, float] = field(default_factory=dict)
    per_frame: Dict[str, np.ndarray] = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(
            {"demo": self.demo_name, "path": self.demo_path,
             "aggregate": self.aggregate, "provenance": self.provenance,
             "notes": self.notes}, indent=2))
        np.savez_compressed(out / "per_frame.npz",
                            **{k: np.asarray(v) for k, v in self.per_frame.items()})
        return out

    def summary(self) -> str:
        lines = [f"eval: {self.demo_name}"]
        lines += [f"  {k:44s} {v:12.4f}" for k, v in sorted(self.aggregate.items())
                  if isinstance(v, (int, float))]
        lines += [f"  ! {n}" for n in self.notes]
        return "\n".join(lines)


def evaluate(demo: Demonstration, metrics=("penetration", "collision", "smoothness"),
             reference: Optional[Demonstration] = None,
             samples_per_link: int = 32) -> EvalReport:
    """Score a demonstration. `reference` enables the same-object reconstruction-error metric."""
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metric(s) {sorted(unknown)}. Known: {', '.join(METRICS)}")

    rep = EvalReport(demo_name=demo.name, demo_path=demo.path,
                     provenance=demo.provenance.to_dict())
    obj = get_object(demo.obj_name)

    if "penetration" in metrics:
        r = hand_object_penetration(demo, obj, samples_per_link=samples_per_link)
        rep.aggregate.update({f"penetration.{k}": v for k, v in r.aggregate().items()})
        for s, v in r.per_frame.items():
            rep.per_frame[f"penetration.{s}"] = v
        if any(v > 0 for k, v in r.aggregate().items() if k.endswith("saturated_frames")):
            rep.notes.append("penetration saturated at the cutoff on some frames; "
                             "raise `cutoff` to measure them")

    if "collision" in metrics:
        r = hand_collision(demo, samples_per_link=samples_per_link)
        rep.aggregate.update({f"collision.{k}": v for k, v in r.aggregate().items()})
        for s, v in r.per_frame_gap.items():
            rep.per_frame[f"collision.{s}"] = v

    if "smoothness" in metrics:
        r = trajectory_smoothness(demo)
        rep.aggregate.update({f"smoothness.{k}": v for k, v in r.aggregate().items()})
        for s, v in r.per_frame.items():
            rep.per_frame[f"smoothness.{s}"] = v

    if "reconstruction" in metrics:
        if reference is None:
            rep.notes.append("reconstruction error skipped: no reference given "
                             "(only defined for the same object)")
        else:
            r = reconstruction_error(demo, reference)
            rep.aggregate.update({f"reconstruction.{k}": v for k, v in r.aggregate().items()})
            for s, v in r.per_frame.items():
                rep.per_frame[f"reconstruction.{s}"] = v
    return rep
