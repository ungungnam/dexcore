"""Step 3: canonical contact cache for the canonical contact study.

For every SAMPLE (sequence, window, hand, role) in samples_index.csv and for every whole-sequence
average (sequence x hand x role with sequence-level touch >= TOUCH_MIN), the instance's
per-vertex soft / hard contact maps (dense_window_cache) are splatted onto the category's K=512
canonical points of each requested backend.

Output (root = /result/uhnam/dexcore/canonical_contact/canonical_contact_cache):
    <root>/<backend>/<category_slug>__<role>.npz         window samples
    <root>/<backend>/<category_slug>__<role>__seq.npz    whole-sequence averages
        X_soft, X_hard (M, K) float32   kernel 'gauss', radius RADIUS (canonical units)
        X_soft_nn      (M, K) float32   kernel 'nearest', same radius (sensitivity check)
        coverage       (M, K) bool      canonical point had an instance vertex within RADIUS
        canonical_points (K, 3)
        meta arrays (M,): sequence_id, window, start, hand, mesh_id, verb, triplet,
                          counterpart_cat, split, subject, touch_frac
        (seq files: window = -1, start = -1, touch_frac = mean of the per-window touch fractions)
    <root>/cache_index.csv   one row per file: backend, category, role, level, file, M, K

Run from the repo root with PYTHONPATH set (scripts/ must not be sys.path[0]):
    PYTHONPATH=/home/uhnam/workspace/dexcore python \
        /result/uhnam/dexcore/canonical_contact/scripts/build_canonical_cache.py \
        --backends aligned normalized random_perm --workers 24
    ... --backends dino          # later, once a 'dino' backend is registered in the package
    ... --verify                 # sanity checks (a)-(c) + coverage / count report on the cache
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.analysis.canonical import backends as backend_registry
from src.analysis.canonical.backends._similarity import slug
from src.analysis.canonical.interface import DEFAULT_RADIUS

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
DENSE = ROOT / "dense_window_cache"
BACKEND_ROOT = ROOT / "canonical_backend"
OUT_ROOT = ROOT / "canonical_contact_cache"
TOUCH_MIN = 0.2
RADIUS = DEFAULT_RADIUS          # 0.15 canonical units, the coverage radius of alignment_quality.csv
GEOMETRIC = ["aligned", "normalized", "random_perm"]

META_STR = ["sequence_id", "hand", "mesh_id", "verb", "triplet", "counterpart_cat", "split"]
META_NUM = {"window": np.int32, "start": np.int32, "subject": np.int32, "touch_frac": np.float32}

log = logging.getLogger("build_canonical_cache")


# ----------------------------------------------------------------------------- sample tables
def window_samples() -> pd.DataFrame:
    """samples_index.csv joined with start / triplet from window_index.csv."""
    s = pd.read_csv(ROOT / "samples_index.csv")
    w = pd.read_csv(ROOT / "window_index.csv")[["sequence_id", "window", "start", "triplet"]]
    out = s.merge(w, on=["sequence_id", "window"], how="left", validate="many_to_one")
    if out["start"].isna().any():
        raise RuntimeError("samples_index rows without a window_index match")
    out["mesh_id"] = out["mesh_id"].map(lambda i: f"{int(i):03d}")
    out["level"] = "window"
    return out


def sequence_samples() -> pd.DataFrame:
    """One row per (sequence, hand, role) with sequence-level touch >= TOUCH_MIN, where the
    sequence-level touch is the mean of the per-window touch fractions (equal 64-frame windows)."""
    w = pd.read_csv(ROOT / "window_index.csv")
    rows = []
    for (seq, file), g in w.groupby(["sequence_id", "file"], sort=True):
        first = g.iloc[0]
        for role in ("tool", "target"):
            cp = "target_cat" if role == "tool" else "tool_cat"
            for hand in ("L", "R"):
                touch = float(g[f"touch_{hand}_{role}"].mean())
                if touch < TOUCH_MIN:
                    continue
                rows.append({"category": first[f"{role}_cat"], "role": role, "hand": hand,
                             "sequence_id": seq, "file": file, "window": -1, "start": -1,
                             "mesh_id": f"{int(first[f'{role}_mesh']):03d}", "verb": first["verb"],
                             "triplet": first["triplet"], "counterpart_cat": first[cp],
                             "subject": int(first["subject"]), "split": first["split"],
                             "touch_frac": touch, "level": "seq"})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- workers
_BACKENDS: Dict[str, object] = {}


def _get_backend(name: str):
    if name not in _BACKENDS:
        _BACKENDS[name] = backend_registry.load(name, BACKEND_ROOT)
    return _BACKENDS[name]


def _splat_group(task: dict) -> dict:
    """One (backend, category, role, level) file. Loads each dense npz once, splats every
    sample of the group with the backend's own `splat` (KD-tree gather)."""
    be = _get_backend(task["backend"])
    cat, role, level = task["category"], task["role"], task["level"]
    rows: pd.DataFrame = task["rows"]
    K = len(be.canonical_points(cat))
    M = len(rows)
    X_soft = np.zeros((M, K), np.float32)
    X_hard = np.zeros((M, K), np.float32)
    X_soft_nn = np.zeros((M, K), np.float32)
    cov = np.zeros((M, K), bool)
    inst_hard_mean = np.zeros(M, np.float32)     # sum(hard on instance vertices) / n_verts
    done = np.zeros(M, bool)
    # rows of one file are NOT contiguous (hand L rows precede hand R rows), so every splat is
    # written at the row's own index `n` to stay aligned with the metadata arrays below
    for file, g in rows.groupby("file", sort=False):
        with np.load(DENSE / file) as z:
            for n, r in g.iterrows():
                hand, mesh = r["hand"], r["mesh_id"]
                if level == "seq":
                    soft = z[f"seq_soft_{hand}_{role}"].astype(np.float64)
                    hard = z[f"seq_hard_{hand}_{role}"].astype(np.float64)
                else:
                    wi = int(r["window"])
                    soft = z[f"soft_{hand}_{role}"][wi].astype(np.float64)
                    hard = z[f"hard_{hand}_{role}"][wi].astype(np.float64)
                xs, c = be.splat(cat, mesh, soft, kernel="gauss", radius=RADIUS,
                                 return_coverage=True)
                xh = be.splat(cat, mesh, hard, kernel="gauss", radius=RADIUS)
                xn = be.splat(cat, mesh, soft, kernel="nearest", radius=RADIUS)
                X_soft[n], X_hard[n], X_soft_nn[n], cov[n] = xs, xh, xn, c
                inst_hard_mean[n] = hard.mean()
                done[n] = True
    assert done.all()
    out = {"X_soft": X_soft, "X_hard": X_hard, "X_soft_nn": X_soft_nn, "coverage": cov,
           "inst_hard_mean": inst_hard_mean, "canonical_points": be.canonical_points(cat),
           "radius": np.array(RADIUS), "kernel": np.array("gauss"), "backend": np.array(task["backend"]),
           "category": np.array(cat), "role": np.array(role), "level": np.array(level)}
    for k in META_STR:
        out[k] = np.array([str(v) for v in rows[k]], dtype=str)   # unicode, not object
    for k, dt in META_NUM.items():
        out[k] = rows[k].to_numpy().astype(dt)
    path = OUT_ROOT / task["backend"] / task["fname"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return {"backend": task["backend"], "category": cat, "role": role, "level": level,
            "file": str(path.relative_to(OUT_ROOT)), "M": M, "K": K,
            "coverage_mean": float(cov.mean())}


def build(backends: List[str], workers: int, categories: Optional[List[str]] = None):
    win = window_samples()
    seq = sequence_samples()
    log.info("%d window samples, %d sequence samples", len(win), len(seq))
    tasks = []
    for name in backends:
        for df in (win, seq):
            for (cat, role), g in df.groupby(["category", "role"], sort=True):
                if categories and cat not in categories:
                    continue
                level = g["level"].iloc[0]
                fname = f"{slug(cat)}__{role}" + ("__seq" if level == "seq" else "") + ".npz"
                tasks.append({"backend": name, "category": cat, "role": role, "level": level,
                              "rows": g.reset_index(drop=True), "fname": fname})
    tasks.sort(key=lambda t: -len(t["rows"]))          # rows keep samples_index order (RangeIndex)          # biggest first for load balance
    log.info("%d tasks -> %s", len(tasks), OUT_ROOT)
    t0 = time.time()
    with Pool(workers) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_splat_group, tasks)):
            results.append(r)
            log.info("[%3d/%d] %-12s %-12s %-6s %-6s M=%5d cov=%.3f (%.0fs)", i + 1, len(tasks),
                     r["backend"], r["category"], r["role"], r["level"], r["M"],
                     r["coverage_mean"], time.time() - t0)
    return results


def write_index(results: List[dict]):
    """cache_index.csv: merge new rows into the existing index (so 'dino' can be added later)."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / "cache_index.csv"
    cols = ["backend", "category", "role", "level", "file", "M", "K", "coverage_mean"]
    new = pd.DataFrame(results)[cols]
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["file"].isin(new["file"])]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["backend", "level", "category", "role"]).reset_index(drop=True)
    new.to_csv(path, index=False)
    log.info("-> %s (%d files)", path, len(new))


# ----------------------------------------------------------------------------- verification
def verify(seed: int = 0):
    from scipy.stats import spearmanr

    idx = pd.read_csv(OUT_ROOT / "cache_index.csv")
    rng = np.random.default_rng(seed)
    lines = []

    # (a) aligned: mean(X_hard) tracks instance-level contact area
    al = idx[(idx.backend == "aligned") & (idx.level == "window")]
    pool_x, pool_i = [], []
    for _, r in al.iterrows():
        z = np.load(OUT_ROOT / r["file"])
        pool_x.append(z["X_hard"].mean(1)); pool_i.append(z["inst_hard_mean"])
    pool_x, pool_i = np.concatenate(pool_x), np.concatenate(pool_i)
    pick = rng.choice(len(pool_x), 100, replace=False)
    rho = spearmanr(pool_i[pick], pool_x[pick]).correlation
    rho_all = spearmanr(pool_i, pool_x).correlation
    lines.append(f"(a) aligned: Spearman(sum(hard)/n_verts, mean(X_hard)) = {rho:.3f} "
                 f"over 100 random samples ({rho_all:.3f} over all {len(pool_x)})")

    # (b) random_perm == per-mesh permutation of aligned
    rp = backend_registry.load("random_perm", BACKEND_ROOT)
    n_checked, n_bad_exact, n_bad_multiset = 0, 0, 0
    for _, r in idx[idx.backend == "aligned"].iterrows():
        za = np.load(OUT_ROOT / r["file"])
        zr = np.load(OUT_ROOT / r["file"].replace("aligned/", "random_perm/"))
        for key in ("X_soft", "X_hard", "X_soft_nn", "coverage"):
            A, B = za[key], zr[key]
            perms = np.stack([rp.permutation(r["category"], m) for m in za["mesh_id"]])
            exact = np.array_equal(np.take_along_axis(A, perms, 1), B)
            same_ms = np.array_equal(np.sort(A, 1), np.sort(B, 1))
            n_bad_exact += (not exact); n_bad_multiset += (not same_ms)
        n_checked += 1
    lines.append(f"(b) random_perm: {n_checked} aligned files checked; files failing "
                 f"exact-permutation check = {n_bad_exact}, failing same-multiset check = {n_bad_multiset}")

    # (c) normalized differs from aligned
    diffs, n_zero = [], 0
    for _, r in idx[(idx.backend == "aligned")].iterrows():
        za = np.load(OUT_ROOT / r["file"]); zn = np.load(OUT_ROOT / r["file"].replace("aligned/", "normalized/"))
        d = np.abs(za["X_soft"] - zn["X_soft"]).mean(1)
        diffs.append(d); n_zero += int((d == 0).sum())
    diffs = np.concatenate(diffs)
    lines.append(f"(c) mean |X_soft_aligned - X_soft_normalized| = {diffs.mean():.5f} "
                 f"(min per-sample {diffs.min():.2e}; samples with zero diff: {n_zero}/{len(diffs)})")

    # coverage per category / counts
    cov = (idx[idx.level == "window"].assign(w=lambda d: d.M * d.coverage_mean)
           .groupby(["backend", "category"])[["w", "M"]].sum().pipe(lambda d: d.w / d.M).unstack(0))
    lines.append("\nper-category coverage mean (window samples):\n" + cov.round(3).to_string())
    counts = idx.pivot_table(index=["category", "role"], columns=["backend", "level"], values="M",
                             aggfunc="sum", fill_value=0)
    lines.append("\nsamples per (backend, category, role):\n" + counts.to_string())
    lines.append(f"\ntotal files: {len(idx)}; total window samples per backend: "
                 f"{idx[idx.level == 'window'].groupby('backend').M.sum().to_dict()}; "
                 f"seq samples per backend: {idx[idx.level == 'seq'].groupby('backend').M.sum().to_dict()}")
    report = "\n".join(lines)
    (OUT_ROOT / "verification.txt").write_text(report)
    print(report)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backends", nargs="+", default=GEOMETRIC,
                   help="backend names registered in src.analysis.canonical.backends")
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--verify", action="store_true", help="only run the sanity checks on the cache")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    if args.verify:
        verify()
        return
    results = build(args.backends, args.workers, args.categories)
    write_index(results)


if __name__ == "__main__":
    main()
