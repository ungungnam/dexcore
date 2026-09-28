#!/usr/bin/env python
"""Run SELECTION alone and save the sparse demonstration. For inspecting a budget in isolation.

  python scripts/select.py --sequence box_use_01 --selector reconstruction_aware --ratio 0.05
"""

import pathlib as _pl
import sys as _sys

# Python puts this script's directory first on sys.path, where `scripts/select.py` SHADOWS the
# standard library's `select` module (imported by subprocess, imported by numpy). Strip it first,
# before any import that could reach it.
_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse

from src.cli import add_config_args, add_data_args, add_stage_args, build_config

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p); add_data_args(p); add_stage_args(p)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = build_config(args)

    from src.data.demo import Demonstration
    from src.selection.base import build as build_selector
    import src.pipeline  # noqa: F401  (populates the registries)

    demo = Demonstration.load(path=cfg.demo_path, obj_name=cfg.obj_name, use_clip=cfg.use_clip,
                              subject=cfg.subject, frame_start=cfg.frame_start,
                              frame_end=cfg.frame_end)
    sel = build_selector(cfg.selector, **cfg.selector_options)
    sparse = sel.select(demo)
    print(sparse.summary())
    print(f"\nselected frames: {list(sparse.frames())}")

    out = args.out or (cfg.output_path() / "selected_keyframes.npy")
    sparse.save(out)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
