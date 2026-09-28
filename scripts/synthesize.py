#!/usr/bin/env python
"""Run the full pipeline: select -> transfer -> reconstruct, and save the generated demonstration.

  python scripts/synthesize.py --sequence box_use_01 --selector uniform --ratio 0.1 \
      --transfer identity --reconstructor linear_keypoint

  python scripts/synthesize.py -c configs/experiment1_redundancy.yaml --ratio 0.05
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
    p.add_argument("--no-save", action="store_true", help="run but write nothing")
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args()

    cfg = build_config(args)
    if not args.quiet:
        print(cfg.describe()); print()

    from src import pipeline
    out = pipeline.run(cfg, verbose=not args.quiet)

    print(); print(out.demo.summary())
    if not args.no_save:
        d = out.save()
        print(f"\nwrote {d}")
        for f in sorted(d.iterdir()):
            print(f"  {f.name}")

if __name__ == "__main__":
    main()
