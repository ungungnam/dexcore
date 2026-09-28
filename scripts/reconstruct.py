#!/usr/bin/env python
"""Run RECONSTRUCTION alone, from a saved sparse demonstration.

  python scripts/reconstruct.py --sparse outputs/<run>/selected_keyframes.npy \
      --source-sequence box_use_01
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
    p.add_argument("--sparse", required=True, help="a saved sparse demonstration .npy")
    p.add_argument("--source-sequence", required=True,
                   help="the dense source the selection came from, e.g. box_use_01")
    p.add_argument("--target-object", default=None)
    p.add_argument("--reconstructor", default="linear_keypoint")
    p.add_argument("--ik-iters", type=int, default=300)
    p.add_argument("--out", default=None)
    p.add_argument("--asset-root", default=None)
    args = p.parse_args()

    if args.asset_root:
        from src import paths
        paths.set_asset_path(*[r.strip() for r in args.asset_root.split(":") if r.strip()])

    from src.data.demo import Demonstration
    from src.data.object import get_object
    from src.data.target import build_target_trajectory
    from src.paths import parse_demo_stem
    from src.reconstruction.base import build as build_reconstructor
    import src.pipeline  # noqa: F401

    sparse = Demonstration.load(path=args.sparse)
    obj, clip = parse_demo_stem(args.source_sequence)
    # the SOURCE WINDOW the selection was made in, which is NOT `sparse.frame_end`: that is
    # frame_start plus the number of retained ROWS, so a 6-frame selection out of 60 would reload
    # the source as frames 0..6 and every keyframe past the sixth would fall outside tau_O.
    # `source_num_frames` is what `subset()` stamps for exactly this.
    n_dense = sparse.provenance.source_num_frames or (
        int(sparse.frames().max()) + 1 - sparse.frame_start)
    source = Demonstration.load(obj_name=obj, use_clip=clip,
                                frame_start=sparse.frame_start,
                                frame_end=sparse.frame_start + int(n_dense))
    target = get_object(args.target_object or sparse.obj_name)
    tau = build_target_trajectory(source, target, get_object(obj))

    rec = build_reconstructor(args.reconstructor, ik_iters=args.ik_iters)
    res = rec.reconstruct(sparse, tau, target_object=target)
    print(res.diagnostics.summary())
    print(); print(res.demo.summary())

    out = args.out or f"outputs/reconstruct_{res.demo.name}.npy"
    res.demo.save(out)
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()
