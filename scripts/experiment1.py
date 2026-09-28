#!/usr/bin/env python
"""Experiment 1 -- temporal redundancy on a fixed object.

    Source Demo -> Selector -> IdentityTransfer -> Reconstruction -> CV evaluation

Sweeps selector x budget with the object held fixed, so every difference is attributable to
selection and reconstruction alone. Writes one row per run to a CSV plus the usual per-run outputs.

  python scripts/experiment1.py --sequence box_use_01 --budgets 0.25 0.1 0.05 0.01
"""

import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import csv
import time


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sequence", default="box_use_01")
    p.add_argument("--frame-start", type=int, default=0)
    p.add_argument("--frame-end", type=int, default=None)
    p.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.10, 0.05, 0.01])
    p.add_argument("--selectors", nargs="+", default=["uniform", "reconstruction_aware"])
    p.add_argument("--ik-iters", type=int, default=300)
    p.add_argument("--out", default="outputs/exp1_redundancy")
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    from eval.run_eval import evaluate
    from src import pipeline, viz
    from src.config import PipelineConfig
    from src.paths import parse_demo_stem

    obj, clip = parse_demo_stem(args.sequence)
    runs = [("all", 1.0)] + [(s, b) for s in args.selectors for b in args.budgets]
    rows = []

    for selector, budget in runs:
        opts = {} if selector == "all" else {"ratio": budget}
        cfg = PipelineConfig(
            obj_name=obj, use_clip=clip, frame_start=args.frame_start, frame_end=args.frame_end,
            selector=selector, selector_options=opts, transfer="identity",
            reconstructor="linear_keypoint", reconstructor_options={"ik_iters": args.ik_iters},
            output_dir=args.out)
        t0 = time.time()
        out = pipeline.run(cfg, verbose=False)
        rep = evaluate(out.demo, metrics=("penetration", "collision", "smoothness",
                                          "reconstruction"), reference=out.source_demo)
        out.save()
        rep.save(cfg.output_path())
        if args.figures:
            viz.report(out)

        a = rep.aggregate
        row = {
            "selector": selector,
            "budget": round(len(out.sparse_demo) / len(out.source_demo), 5),
            "frames_kept": len(out.sparse_demo),
            "frames_total": len(out.source_demo),
            "kp_err_mean_mm": round((a["reconstruction.keypoint.left.mean"]
                                     + a["reconstruction.keypoint.right.mean"]) / 2, 3),
            "kp_err_max_mm": round(max(a["reconstruction.keypoint.left.max"],
                                       a["reconstruction.keypoint.right.max"]), 3),
            "fingertip_mean_mm": round((a["reconstruction.fingertip.left.mean"]
                                        + a["reconstruction.fingertip.right.mean"]) / 2, 3),
            "qpos_rms_rad": round((a["reconstruction.qpos.left.rms"]
                                   + a["reconstruction.qpos.right.rms"]) / 2, 4),
            "penetration_max_mm": round(a["penetration.max_depth_mm"], 3),
            "self_collision_frac": round(max(a["collision.self.left.frac_colliding"],
                                             a["collision.self.right.frac_colliding"]), 4),
            "accel_mean": round((a["smoothness.accel.left.mean"]
                                 + a["smoothness.accel.right.mean"]) / 2, 3),
            "seconds": round(time.time() - t0, 1),
            "run": cfg.resolve_run_name(),
        }
        rows.append(row)
        print(f"{selector:22s} {row['budget']:7.3f}  "
              f"kp {row['kp_err_mean_mm']:7.3f} mm (max {row['kp_err_max_mm']:8.3f})  "
              f"pen {row['penetration_max_mm']:6.2f} mm  {row['seconds']:5.1f}s", flush=True)

    out_csv = _pl.Path(args.out) / "results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
