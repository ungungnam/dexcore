"""Shared CLI plumbing: the same config flags for every script, in one place.

This lives in `src/` rather than in `scripts/` deliberately. Python puts a script's OWN directory
first on `sys.path`, so a helper imported as a bare top-level module from `scripts/` puts that whole
directory on the import path -- and `scripts/select.py` then SHADOWS the standard library's
`select`, which `subprocess` imports, which `numpy` imports. The failure is spectacular and its
traceback points at numpy. See `_strip_script_dir` below for the other half of the fix.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import PipelineConfig


def add_config_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("config (files merge left to right; --set overrides everything)")
    g.add_argument("--config", "-c", action="append", default=[],
                   help="YAML config file; repeatable, later files win")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="dotted override, e.g. --set selector_options.ratio=0.05")
    g.add_argument("--asset-root", default=None, help="override the asset SEARCH PATH (colon-separated, highest priority first)")
    return p


def add_data_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("data")
    g.add_argument("--sequence", default=None,
                   help="demo stem, e.g. box_use_01 (sets obj_name and use_clip)")
    g.add_argument("--demo-path", default=None, help="explicit .npy, wins over --sequence")
    g.add_argument("--subject", default=None)
    g.add_argument("--frame-start", type=int, default=None)
    g.add_argument("--frame-end", type=int, default=None)
    g.add_argument("--target-object", default=None, help="default: the source object")
    return p


def add_stage_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("method")
    g.add_argument("--synthesizer", default=None,
                   help="synthesis method: 'staged' (dexcore's S->Phi->I) or an end-to-end one. "
                        "The stage flags below apply only to 'staged'.")
    g.add_argument("--selector", default=None)
    g.add_argument("--ratio", type=float, default=None, help="selection budget as a fraction")
    g.add_argument("--num-frames", type=int, default=None, help="selection budget as a count")
    g.add_argument("--transfer", default=None)
    g.add_argument("--reconstructor", default=None)
    g.add_argument("--ik-iters", type=int, default=None)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--output-dir", default=None)
    g.add_argument("--run-name", default=None)
    return p


def build_config(args) -> PipelineConfig:
    """CLI flags -> dotted overrides -> a validated PipelineConfig."""
    over = {}
    for item in getattr(args, "set", []):
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        over[k] = v

    def put(key, val):
        if val is not None:
            over.setdefault(key, val)

    if getattr(args, "sequence", None):
        from src.paths import parse_demo_stem
        obj, clip = parse_demo_stem(args.sequence)
        if clip is None:
            raise SystemExit(f"--sequence {args.sequence!r} has no _use_ token; "
                             f"expected e.g. box_use_01")
        put("obj_name", obj)
        put("use_clip", clip)
    put("demo_path", getattr(args, "demo_path", None))
    put("subject", getattr(args, "subject", None))
    put("frame_start", getattr(args, "frame_start", None))
    put("frame_end", getattr(args, "frame_end", None))
    put("target_object", getattr(args, "target_object", None))
    put("synthesizer", getattr(args, "synthesizer", None))
    put("selector", getattr(args, "selector", None))
    put("transfer", getattr(args, "transfer", None))
    put("reconstructor", getattr(args, "reconstructor", None))
    put("seed", getattr(args, "seed", None))
    put("output_dir", getattr(args, "output_dir", None))
    put("run_name", getattr(args, "run_name", None))
    put("asset_root", getattr(args, "asset_root", None))
    if getattr(args, "ratio", None) is not None:
        over.setdefault("selector_options.ratio", args.ratio)
    if getattr(args, "num_frames", None) is not None:
        over.setdefault("selector_options.num_frames", args.num_frames)
        over.setdefault("selector_options.ratio", None)
    if getattr(args, "ik_iters", None) is not None:
        over.setdefault("reconstructor_options.ik_iters", args.ik_iters)
    return PipelineConfig.load(*getattr(args, "config", []), overrides=over)


def _strip_script_dir(script_file) -> None:
    """Remove the script's own directory from sys.path and add the repo root.

    Must run BEFORE importing numpy (or anything that reaches the standard library's `select`).
    Every script in `scripts/` calls this as its first statement; see the module docstring.
    """
    here = str(Path(script_file).resolve().parent)
    repo = str(Path(script_file).resolve().parent.parent)
    sys.path[:] = [p for p in sys.path if p not in ("", here)]
    if repo not in sys.path:
        sys.path.insert(0, repo)
