"""Run the DINO semantic backend for every TACO mesh and cache it next to the geometric ones.

    PYTHONPATH=/home/uhnam/workspace/dexcore:/home/uhnam/workspace/dexmachina \\
    CUDA_VISIBLE_DEVICES=1 python -m src.analysis.canonical.fit_dino --workers 6

Reads the cached `aligned` backend from `<root>/aligned` (the template per category is TAKEN from
it, never refitted, so `dino` and `aligned` share canonical points) and writes

    <root>/dino/<category>.npz     the per-category map (aligned fit + FPS points -> template)
    <root>/dino/templates.json     config, template id and per-mesh quality
    <root>/dino/quality.csv        one row per (category, mesh): n_valid, fallback_frac,
                                   conf mean/median, displacement vs aligned, distance of the
                                   mapped points to the template surface, wall time
    <root>/dino/timing.log         per-mesh timing lines from the workers

Work is split by CATEGORY over `--workers` processes (each renders its templates once and holds
its own DINOv2 on the GPU; the rasteriser is pure Python so the CPU workers are the parallelism).
Categories are dispatched largest first so the 15-mesh spatula does not start last.
"""
from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

from src.analysis.canonical.fit_all import DEFAULT_MESH_DICT, DEFAULT_ROOT, by_category

log = logging.getLogger("canonical.fit_dino")

CSV_COLUMNS = ["category", "mesh_id", "is_template", "n_points", "n_valid", "fallback_frac",
               "conf_mean", "conf_median", "top1_conf_mean", "disp_mean", "disp_median",
               "surf_dist_mean", "n_views_mean", "flip", "flip_margin", "coverage", "time_s"]

_W: dict = {}          # per-worker state


def _init_worker(mesh_dict_path: str, root: str, device: str, n_sample: int, min_top1_conf: float,
                 engine_kw: dict, threads: int, log_path: str):
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    import torch
    torch.set_num_threads(threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(processName)s %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.FileHandler(log_path),
                                                       logging.StreamHandler()])
    for noisy in ("dexmachina", "dinov2", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from src.analysis.bimart import meshes as M
    from src.analysis.canonical.backends.aligned import AlignedBackend
    from src.analysis.canonical.backends.dino import DinoBackend
    _W["cats"] = by_category(M.load(mesh_dict_path))
    aligned = AlignedBackend.load(Path(root) / "aligned")
    _W["backend"] = DinoBackend(aligned, n_sample=n_sample, min_top1_conf=min_top1_conf,
                                device=device, **engine_kw)
    _W["out"] = Path(root) / "dino"


def _run_category(category: str) -> str:
    t0 = time.time()
    bk = _W["backend"]
    bk.fit(category, _W["cats"][category])
    path = bk.save_category(category, _W["out"])
    q = bk.quality(category)
    n = len(q)
    logging.getLogger("canonical.fit_dino").info(
        "%-12s DONE %2d meshes in %.0fs  mean fallback %.3f -> %s", category, n, time.time() - t0,
        sum(v["fallback_frac"] for v in q.values()) / n, path)
    return category


def write_quality_csv(backend, root: Path) -> Path:
    rows = []
    for cat in backend.categories():
        tpl = backend.template(cat)
        q = backend.quality(cat)
        for mid in backend.mesh_ids(cat):
            r = {"category": cat, "mesh_id": mid, "is_template": int(mid == tpl),
                 "coverage": backend.coverage(cat, mid)}
            r.update(q[mid])
            rows.append(r)
    path = root / "quality.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def fit_dino(mesh_dict_path: str, root: Path, categories: Optional[List[str]], workers: int,
             device: str, n_sample: int, min_top1_conf: float, engine_kw: dict, threads: int):
    from src.analysis.bimart import meshes as M
    cats = by_category(M.load(mesh_dict_path))
    todo = sorted(categories or cats, key=lambda c: -len(cats[c]))
    out = root / "dino"
    out.mkdir(parents=True, exist_ok=True)
    log_path = str(out / "timing.log")
    t0 = time.time()
    init_args = (mesh_dict_path, str(root), device, n_sample, min_top1_conf, engine_kw, threads,
                 log_path)
    if workers <= 1:
        _init_worker(*init_args)
        for c in todo:
            _run_category(c)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            for c in pool.imap_unordered(_run_category, todo):
                log.info("collected %s  (%.0fs elapsed)", c, time.time() - t0)
    # assemble: load everything back, write templates.json + quality.csv
    from src.analysis.canonical.backends.dino import DinoBackend
    backend = DinoBackend.load(out)
    backend.write_templates_json(out)
    csv_path = write_quality_csv(backend, out)
    log.info("all done in %.1f min -> %s", (time.time() - t0) / 60, csv_path)
    return backend, csv_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-dict", default=DEFAULT_MESH_DICT)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--threads", type=int, default=4, help="BLAS/torch threads per worker")
    p.add_argument("--device", default="cuda", help="DINO + feature matching device")
    p.add_argument("--n-sample", type=int, default=2000, help="FPS points mapped per mesh")
    p.add_argument("--min-top1-conf", type=float, default=0.5,
                   help="below this top-1 cluster confidence the point falls back to aligned")
    p.add_argument("--n-views", type=int, default=24)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--upsample", type=int, default=2, help="render upsampling before DINO")
    p.add_argument("--mutual-px-tol", type=float, default=16.0)
    p.add_argument("--min-match-confidence", type=float, default=0.25)
    p.add_argument("--dbscan-eps", type=float, default=0.06, help="canonical units")
    p.add_argument("--no-flip-search", action="store_true",
                   help="render the instance only in the aligned frame (no sign search)")
    p.add_argument("--flip-min-margin", type=float, default=0.05,
                   help="a flip replaces the aligned frame only with this fraction of the points "
                        "more valid matches")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    engine_kw = dict(n_views=args.n_views, image_size=args.image_size, upsample=args.upsample,
                     mutual_px_tol=args.mutual_px_tol,
                     min_match_confidence=args.min_match_confidence, dbscan_eps=args.dbscan_eps,
                     flip_search=not args.no_flip_search, flip_min_margin=args.flip_min_margin)
    backend, csv_path = fit_dino(args.mesh_dict, Path(args.root), args.categories, args.workers,
                                 args.device, args.n_sample, args.min_top1_conf, engine_kw,
                                 args.threads)
    print(f"\n{'category':12s} {'n':>3s} {'tpl':>4s} {'fallback':>9s} {'conf':>6s} {'disp':>6s} "
          f"{'surf':>6s} {'cover':>6s} {'flipped':>7s} {'s/mesh':>7s}")
    for cat in backend.categories():
        q = backend.quality(cat)
        others = [m for m in q if m != backend.template(cat)] or list(q)
        import numpy as np
        mean = lambda k: float(np.nanmean([q[m][k] for m in others]))       # noqa: E731
        cov = float(np.mean([backend.coverage(cat, m) for m in q]))
        flipped = sum(q[m]["flip"] != "+++" for m in others)
        print(f"{cat:12s} {len(q):3d} {backend.template(cat):>4s} {mean('fallback_frac'):9.3f} "
              f"{mean('conf_mean'):6.3f} {mean('disp_mean'):6.3f} {mean('surf_dist_mean'):6.4f} "
              f"{cov:6.3f} {flipped:7d} {mean('time_s'):7.1f}")
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
