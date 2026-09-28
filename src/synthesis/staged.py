"""StagedSynthesizer -- dexcore's own three-stage pipeline, as one `Synthesizer`.

    D_s  --S-->  K_s  --Phi-->  K_t  --I-->  D_hat_t

This holds NO research logic. It wires the three registries together and reports what they did in
the common `SynthesisResult` shape, so the staged pipeline sits in the same table as a method that
has no stages at all. A new selector, transfer or reconstructor is still added by registering it;
nothing here changes.

IT DECLARES `source_hand`, AND THAT IS THE POINT. Selection reads the human hand states to choose
frames, and transfer grounds those states onto the target -- this method IS a transfer of a
demonstration. An end-to-end model declares no such thing, and `hand_states_used` then reports 0
against this method's selection budget. Those two numbers in one column are the comparison.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.data.demo import Demonstration
from src.reconstruction.base import ReconstructionResult
from src.reconstruction.base import build as build_reconstructor
from src.selection.base import build as build_selector
from src.synthesis.base import (Synthesizer, SynthesisDiagnostics, SynthesisInput,
                                SynthesisResult, register)
from src.transfer.base import TransferResult
from src.transfer.base import build as build_transfer

# importing the implementations is what populates the three stage registries. It happens HERE,
# in the only synthesizer that has stages, rather than in the driver -- a method with no stages
# must not drag them in.
from src.reconstruction import linear_keypoint as _r_linear      # noqa: F401
from src.selection import all as _s_all                          # noqa: F401
from src.selection import reconstruction_aware as _s_ra          # noqa: F401
from src.selection import uniform as _s_uniform                  # noqa: F401
from src.transfer import cordex as _t_cordex                     # noqa: F401
from src.transfer import identity as _t_identity                 # noqa: F401
from src.transfer import palm_first_finger as _t_pff             # noqa: F401
from src.transfer import shapegen as _t_shapegen                 # noqa: F401


class StagedResult(SynthesisResult):
    """A `SynthesisResult` that also exposes the intermediates only a staged method has.

    K_s and K_t are kept because they are the only place selection and transfer can be inspected
    on their own: everything downstream of them has been through interpolation and IK, so a fault
    in one is otherwise indistinguishable from a fault in the other.
    """

    def __init__(self, sparse_demo: Demonstration, transfer: TransferResult,
                 reconstruction: ReconstructionResult, **kw):
        super().__init__(**kw)
        self.sparse_demo = sparse_demo
        self.transfer = transfer
        self.reconstruction = reconstruction


@register
class StagedSynthesizer(Synthesizer):
    """Selection -> transfer -> reconstruction, named by registry entry."""

    name = "staged"
    consumes = frozenset({"source_object", "source_clip", "source_hand"})

    def __init__(self, selector: str = "uniform", transfer: str = "identity",
                 reconstructor: str = "linear_keypoint",
                 selector_options: Optional[Dict[str, Any]] = None,
                 transfer_options: Optional[Dict[str, Any]] = None,
                 reconstructor_options: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self.selector = build_selector(selector, **(selector_options or {}))
        self.transfer = build_transfer(transfer, **(transfer_options or {}))
        self.reconstructor = build_reconstructor(reconstructor, **(reconstructor_options or {}))

    def describe(self) -> str:
        return (f"{self.name}({self.selector.describe()} -> {self.transfer.describe()} -> "
                f"{self.reconstructor.describe()})")

    def synthesize(self, inp: SynthesisInput) -> StagedResult:
        sparse = self.selector.select(inp.source_demo)
        tr = self.transfer.transfer(sparse, inp.source_object, inp.target_object, inp.trajectory)
        rec = self.reconstructor.reconstruct(tr.demo, tr.trajectory,
                                             target_object=inp.target_object)

        diag = SynthesisDiagnostics(method=self.describe())
        diag.stages["transfer"] = tr.diagnostics
        diag.stages["reconstruction"] = rec.diagnostics
        for side, rms in rec.diagnostics.per_frame_rms.items():
            diag.per_frame[f"ik_residual.{side}"] = rms
        # Both stages' notions of failure are counted, because both mean "this frame was not
        # answered": a keyframe Phi could not ground, and a frame whose IK did not reach the
        # convergence threshold. They stay distinguishable in `failures`.
        diag.failures += [f"transfer: {f.describe()}" for f in tr.diagnostics.failures()]
        for side, conv in rec.diagnostics.converged.items():
            n = int((~conv).sum())
            if n:
                diag.failures.append(f"reconstruction: {side} IK did not converge on {n} frames")
        diag.num_failed = tr.diagnostics.num_failed + rec.diagnostics.num_unconverged

        return StagedResult(
            sparse_demo=sparse, transfer=tr, reconstruction=rec,
            demo=rec.demo, trajectory=tr.trajectory, diagnostics=diag,
            method=self.describe(), consumes=self.consumes,
            hand_states_used=len(sparse),
            extras={"selector": self.selector.describe(),
                    "transfer": self.transfer.describe(),
                    "reconstructor": self.reconstructor.describe()},
        )
