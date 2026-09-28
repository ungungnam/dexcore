"""CHORD adapter -- interface only; the CHORD repository is not present on this machine.

Deliberately a stub with a real signature rather than a guess at CHORD's format. Writing an
exporter against a remembered schema would produce files that look right and are not, and the
failure would surface as a bad number in a downstream table rather than as an error here.

To implement: point `CHORD_ROOT` at the checkout, then mirror `eval/downstream/dexmachina.py` --
`export()` writes CHORD's demonstration format, and a reader parses its evaluation output.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.data.demo import Demonstration

CHORD_ROOT: Optional[Path] = None


def available() -> bool:
    return CHORD_ROOT is not None and Path(CHORD_ROOT).exists()


def export(demo: Demonstration, out_dir, **kwargs):
    raise NotImplementedError(
        "CHORD export is not implemented: the CHORD repository is not available on this machine, "
        "so its demonstration format has not been inspected. Set eval.downstream.chord.CHORD_ROOT "
        "to a checkout and implement export() against the real format -- guessing it would produce "
        "plausible files that fail silently downstream."
    )


def read_results(path, **kwargs):
    raise NotImplementedError("CHORD result parsing is not implemented; see export().")
