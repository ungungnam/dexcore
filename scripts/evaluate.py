#!/usr/bin/env python
"""Evaluate a SAVED demonstration. Independent of how it was generated.

  python scripts/evaluate.py --input outputs/<run>/box_use_01.npy --metrics penetration collision
  python scripts/evaluate.py --input outputs/<run>/box_use_01.npy \
      --reference-sequence box_use_01 --metrics reconstruction
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

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="a saved demonstration .npy")
    p.add_argument("--metrics", "-m", nargs="+",
                   default=["penetration", "collision", "smoothness"])
    p.add_argument("--reference-sequence", default=None,
                   help="demo stem to compare against, e.g. box_use_01 (same object only)")
    p.add_argument("--reference-path", default=None)
    p.add_argument("--samples-per-link", type=int, default=32)
    p.add_argument("--out", default=None, help="directory for metrics.json + per_frame.npz "
                                               "(default: alongside the input)")
    p.add_argument("--asset-root", default=None)
    args = p.parse_args()

    if args.asset_root:
        from src import paths
        paths.set_asset_path(*[r.strip() for r in args.asset_root.split(":") if r.strip()])

    from eval.run_eval import evaluate
    from src.data.demo import Demonstration
    from src.paths import parse_demo_stem

    demo = Demonstration.load(path=args.input)
    ref = None
    if args.reference_path or args.reference_sequence:
        if args.reference_sequence:
            obj, clip = parse_demo_stem(args.reference_sequence)
            ref = Demonstration.load(obj_name=obj, use_clip=clip,
                                     frame_start=demo.frame_start, frame_end=demo.frame_end)
        else:
            ref = Demonstration.load(path=args.reference_path)

    rep = evaluate(demo, metrics=tuple(args.metrics), reference=ref,
                   samples_per_link=args.samples_per_link)
    print(rep.summary())
    out = rep.save(args.out or _pl.Path(args.input).parent)
    print(f"\nwrote {out}/metrics.json and per_frame.npz")

if __name__ == "__main__":
    main()
