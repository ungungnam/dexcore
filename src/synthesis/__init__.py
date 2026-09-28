"""Synthesis: the whole of D_s -> D_hat_t, as ONE replaceable object.

`Synthesizer` is the level at which dexcore's own three-stage pipeline and an end-to-end
generative model are the same kind of thing, and therefore comparable.
"""
from src.synthesis.base import (OPTIONAL_INPUTS, SourceClip, SynthesisDiagnostics,
                                SynthesisInput, SynthesisResult, Synthesizer, available, build,
                                register)
from src.synthesis import staged as _staged   # noqa: F401  populates the registry
from src.synthesis import bimart as _bimart   # noqa: F401

__all__ = ["Synthesizer", "SynthesisInput", "SynthesisResult", "SynthesisDiagnostics",
           "SourceClip", "OPTIONAL_INPUTS", "build", "available", "register"]
