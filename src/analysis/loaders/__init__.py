"""Dataset loaders, one module per dataset, all producing `HOITrajectory`.

A loader is chosen BY NAME so a CLI flag (`--dataset taco`) is the only place a dataset is named.
Adding a dataset means adding a module with `index`/`load`/`iter_sequences`/`default_root` and one
line in `LOADERS` -- no metric and no report code changes.
"""
from __future__ import annotations

from types import ModuleType
from typing import Dict

from src.analysis.loaders import arctic, taco

LOADERS: Dict[str, ModuleType] = {"taco": taco, "arctic": arctic}


def get(dataset: str) -> ModuleType:
    try:
        return LOADERS[dataset]
    except KeyError:
        raise KeyError(f"unknown dataset {dataset!r}; have {sorted(LOADERS)}") from None


__all__ = ["LOADERS", "get", "taco", "arctic"]
