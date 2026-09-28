#!/usr/bin/env python
"""Action-conditioned trajectory analysis over a source dataset.

  # every TACO sequence, grouped by verb
  python scripts/analyze_actions.py --dataset taco --workers 16

  # one question at a time: are the brushing verbs different from each other?
  python scripts/analyze_actions.py --dataset taco --verbs brush dust smear --group verb

  # ARCTIC, whose only verbs are grab and use -- group by object instead
  python scripts/analyze_actions.py --dataset arctic --group tool_name

Results go to $DEXCORE_RESULT_ROOT (default /result/uhnam/dexcore) under a timestamped run
directory, so a run never overwrites an earlier one.
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path


#: The label columns a distribution may be conditioned on. Checked BEFORE the sweep runs -- a typo
#: should not cost 2317 sequences of work.
GROUPS = ("verb", "tool_name", "target_name", "action")


def result_root() -> Path:
    """Where analysis artefacts go. $DEXCORE_RESULT_ROOT overrides the server convention."""
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="taco", choices=("taco", "arctic"))
    p.add_argument("--root", default=None, help="dataset root (default: the loader's own)")
    p.add_argument("--out", default=None, help="output directory (default: a timestamped run)")
    p.add_argument("--run-name", default=None)
    p.add_argument("--group", default="verb", choices=GROUPS,
                   help="label column to condition the distributions on")
    p.add_argument("--verbs", nargs="*", default=None, help="keep only these verbs")
    p.add_argument("--tools", nargs="*", default=None)
    p.add_argument("--targets", nargs="*", default=None, help="TACO only")
    p.add_argument("--subjects", nargs="*", default=None, help="ARCTIC only")
    p.add_argument("--limit", type=int, default=None, help="first N sequences (a smoke run)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--fps", type=float, default=None, help="override the dataset's frame rate")
    p.add_argument("--top-k", type=int, default=12, help="features to plot, by effect size")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("analyze_actions")

    from src.analysis import loaders, report, sweep

    mod = loaders.get(args.dataset)
    root = Path(args.root) if args.root else mod.default_root()
    if args.fps is not None and args.fps <= 0:
        p.error("--fps must be positive")
    index_kwargs = {"verbs": args.verbs}
    if args.dataset == "taco":
        if args.subjects:
            p.error("--subjects is an ARCTIC filter; TACO sequences have no subject label")
        index_kwargs.update(tools=args.tools, targets=args.targets)
    else:
        if args.targets:
            p.error("--targets is a TACO filter; ARCTIC sequences have no target object")
        index_kwargs.update(objects=args.tools, subjects=args.subjects)
    refs = mod.index(root, **{k: v for k, v in index_kwargs.items() if v})
    if args.limit:
        refs = refs[: args.limit]
    if not refs:
        p.error("no sequences matched the filters")

    t0 = time.time()
    rows = sweep.run(args.dataset, root=root, refs=refs, workers=args.workers, fps=args.fps)
    if not rows:
        p.error("every sequence failed to load; see the warnings above")
    elapsed = time.time() - t0
    log.info("extracted %d/%d rows in %.1fs (%.2fs/seq)", len(rows), len(refs), elapsed,
             elapsed / max(1, len(refs)))

    from src.analysis.distribution import feature_table
    df = feature_table(rows)
    if df[args.group].nunique() < 2:
        log.warning("every sequence has the same %s (%r): there is nothing to compare",
                    args.group, df[args.group].iloc[0])

    run = args.run_name or f"{args.dataset}_by_{args.group}_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out) if args.out else result_root() / "analysis" / args.dataset / run
    result = report.write_report(
        df, out, group=args.group, top_k=args.top_k, plots=not args.no_plots,
        meta={"dataset": args.dataset, "root": str(root), "n_requested": len(refs),
              "n_failed": len(refs) - len(rows), "elapsed_s": round(elapsed, 1),
              "fps_override": args.fps, "filters": {k: v for k, v in
                                                    (("verbs", args.verbs), ("tools", args.tools),
                                                     ("targets", args.targets)) if v}})

    disc = result["discriminability"]      # already computed by the report; do not run it twice
    print(f"\n{len(df)} sequences, {df[args.group].nunique()} {args.group}s -> {out}\n")
    if len(disc):
        print(f"most {args.group}-dependent features:")
        cols = ["feature", "epsilon_sq", "p", f"highest_{args.group}", f"lowest_{args.group}"]
        print(disc[cols].head(args.top_k).to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
