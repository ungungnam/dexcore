"""Backend registry. `get(name)` returns the class, `load(name, root)` a backend restored from
`root/<name>/`, the layout `fit_all` writes."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Type, Union

from src.analysis.canonical.backends.aligned import AlignedBackend
from src.analysis.canonical.backends.dino import DinoBackend
from src.analysis.canonical.backends.normalized import NormalizedBackend
from src.analysis.canonical.backends.random_perm import RandomPermBackend
from src.analysis.canonical.interface import CanonicalBackend

BACKENDS: Dict[str, Type[CanonicalBackend]] = {
    NormalizedBackend.name: NormalizedBackend,
    AlignedBackend.name: AlignedBackend,
    RandomPermBackend.name: RandomPermBackend,
    DinoBackend.name: DinoBackend,
}


def get(name: str) -> Type[CanonicalBackend]:
    try:
        return BACKENDS[name]
    except KeyError:
        raise KeyError(f"unknown backend {name!r}; known: {sorted(BACKENDS)}") from None


def load(name: str, root: Union[str, Path]) -> CanonicalBackend:
    return get(name).load(Path(root) / name)


__all__ = ["BACKENDS", "AlignedBackend", "DinoBackend", "NormalizedBackend", "RandomPermBackend",
           "get", "load"]
