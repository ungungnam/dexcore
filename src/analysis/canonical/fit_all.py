"""Fit and cache every canonical backend for every TACO category, and write the alignment
quality table.

    python -m src.analysis.canonical.fit_all            # from the repo root

Output layout (root defaults to /result/uhnam/dexcore/taco/40_representation_study/canonical_backend):

    <root>/<backend>/<category>.npz      the fitted state, `backends.load(name, root)` restores it
    <root>/<backend>/templates.json      template id and per-mesh Chamfer per stage
    <root>/alignment_quality.csv         one row per category:
        n_meshes, template (aligned's medoid), mean symmetric Chamfer to that template over the
        OTHER meshes of the category (canonical units, template at max radius 1) after
        (a) centring + scaling only, (b) PCA axes + proper-flip sign, (c) similarity ICP;
        the worst-case round-trip error of to_canonical -> from_canonical in metres; and the
        mean fraction of canonical points a mesh covers at the default splat radius.

`random_perm` is built from the fitted `aligned` backend rather than refitted: it is meant to
share the alignment exactly, differing only in the vector index shuffle.
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.analysis.canonical.backends.aligned import AlignedBackend
from src.analysis.canonical.backends.normalized import NormalizedBackend
from src.analysis.canonical.backends.random_perm import RandomPermBackend
from src.analysis.canonical.interface import DEFAULT_RADIUS

log = logging.getLogger("canonical.fit_all")

DEFAULT_MESH_DICT = "/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/assets/taco_mesh_dict.npy"
DEFAULT_ROOT = "/result/uhnam/dexcore/taco/40_representation_study/canonical_backend"

CSV_COLUMNS = ["category", "n_meshes", "template", "template_normalized",
               "chamfer_centred_scaled", "chamfer_pca_sign", "chamfer_icp",
               "roundtrip_err_m", "coverage_nearest"]


def by_category(mesh_dict: Dict[str, dict]) -> Dict[str, Dict[str, tuple]]:
    cats: Dict[str, Dict[str, tuple]] = {}
    for mesh_id, m in mesh_dict.items():
        cats.setdefault(m["category"], {})[mesh_id] = (m["verts_original"], m["faces"])
    return cats


def quality_row(category: str, aligned: AlignedBackend, normalized: NormalizedBackend,
                radius: float = DEFAULT_RADIUS) -> dict:
    ids = aligned.mesh_ids(category)
    tpl = aligned.template(category)
    q = aligned.quality(category)
    others = [i for i in ids if i != tpl]

    def mean_stage(key: str) -> float:
        return float(np.mean([q[i][key] for i in others])) if others else float("nan")

    rt = max(max(aligned.roundtrip_error(category, i), normalized.roundtrip_error(category, i))
             for i in ids)
    cov = float(np.mean([aligned.coverage(category, i, radius) for i in ids]))
    return {"category": category, "n_meshes": len(ids), "template": tpl,
            "template_normalized": normalized.template(category),
            "chamfer_centred_scaled": mean_stage("chamfer_centred"),
            "chamfer_pca_sign": mean_stage("chamfer_pca"), "chamfer_icp": mean_stage("chamfer_icp"),
            "roundtrip_err_m": rt, "coverage_nearest": cov}


def fit_all(mesh_dict: Dict[str, dict], root: Path, n_sample: int = 2000, k: int = 512,
            icp_iters: int = 30, categories: List[str] = None, radius: float = DEFAULT_RADIUS):
    cats = by_category(mesh_dict)
    if categories:
        cats = {c: cats[c] for c in categories}
    aligned = AlignedBackend(n_sample=n_sample, k=k, icp_iters=icp_iters)
    normalized = NormalizedBackend(n_sample=n_sample, k=k)
    rows = []
    for cat in sorted(cats):
        t0 = time.time()
        aligned.fit(cat, cats[cat])
        normalized.fit(cat, cats[cat])
        rows.append(quality_row(cat, aligned, normalized, radius))
        log.info("%-12s done in %.1fs  (roundtrip %.1e m, coverage %.3f)", cat, time.time() - t0,
                 rows[-1]["roundtrip_err_m"], rows[-1]["coverage_nearest"])
    random_perm = RandomPermBackend.from_aligned(aligned)
    for backend in (normalized, aligned, random_perm):
        out = backend.save(root / backend.name)
        log.info("%-12s -> %s", backend.name, out)
    csv_path = root / "alignment_quality.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info("-> %s", csv_path)
    return rows, csv_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh-dict", default=DEFAULT_MESH_DICT)
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--n-sample", type=int, default=2000, help="vertices per mesh for Chamfer/ICP")
    p.add_argument("--k", type=int, default=512, help="canonical points per category")
    p.add_argument("--icp-iters", type=int, default=30)
    p.add_argument("--radius", type=float, default=DEFAULT_RADIUS, help="coverage radius")
    p.add_argument("--categories", nargs="*", default=None, help="subset (default: all)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    from src.analysis.bimart import meshes as M

    md = M.load(args.mesh_dict)
    log.info("%d meshes from %s", len(md), args.mesh_dict)
    rows, csv_path = fit_all(md, Path(args.root), args.n_sample, args.k, args.icp_iters,
                             args.categories, args.radius)
    print(f"\n{'category':12s} {'n':>3s} {'tpl':>4s} {'centred':>8s} {'pca+sign':>9s} "
          f"{'icp':>8s} {'roundtrip':>10s} {'cover':>6s}")
    for r in rows:
        print(f"{r['category']:12s} {r['n_meshes']:3d} {r['template']:>4s} "
              f"{r['chamfer_centred_scaled']:8.4f} {r['chamfer_pca_sign']:9.4f} "
              f"{r['chamfer_icp']:8.4f} {r['roundtrip_err_m']:10.1e} {r['coverage_nearest']:6.3f}")
    print(f"\n-> {csv_path}")


if __name__ == "__main__":
    main()
