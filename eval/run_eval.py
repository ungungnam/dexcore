"""Evaluation entry point: the two levels, in one place.

    A. demo / CV-level      `eval.cv`          metrics on a saved demonstration
    B. downstream utility   `eval.downstream`  DexMachina (ADD-AUC), CHORD

Evaluation is deliberately separated from synthesis: everything below operates on a `.npy` off
disk, so how a demonstration was produced cannot influence how it is scored.

This module is the library-level dispatcher; `scripts/evaluate.py` is its CLI.
"""
from __future__ import annotations

from eval.cv import METRICS, EvalReport, evaluate

__all__ = ["METRICS", "EvalReport", "evaluate", "downstream_adapters"]


def downstream_adapters() -> dict:
    """The level-B adapters, and whether each can actually run here.

    Reported rather than assumed: CHORD is a stub until its repository is present, and saying so
    at lookup time is better than a NotImplementedError from three call sites deep.
    """
    from eval.downstream import chord, dexmachina
    return {
        "dexmachina": {"module": dexmachina, "available": dexmachina.DEXMACHINA_ROOT.exists(),
                       "metric": "ADD-AUC (object trajectory tracking)"},
        "chord": {"module": chord, "available": chord.available(),
                  "metric": "not wired -- CHORD repository not present"},
    }
