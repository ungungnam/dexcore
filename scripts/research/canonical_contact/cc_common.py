"""Shared helpers for the canonical-contact analysis scripts (Steps 5-11).

Cache layout follows CACHE_FORMAT.md:
  <root>/canonical_contact_cache/<backend>/<category>__<role>.npz
    X_soft (M,K) float32, X_hard (M,K) float32, coverage (M,K) bool, canonical_points (K,3),
    meta parallel arrays of length M: sequence_id, window, start, hand, mesh_id, verb, triplet,
    counterpart_cat, split, subject, touch_frac
A SAMPLE is (sequence, window, hand, role). touch_frac >= MIN_TOUCH (0.2) unless stated.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("/result/uhnam/dexcore/canonical_contact")
DEFAULT_BACKENDS = ["aligned", "normalized", "random_perm"]
OPTIONAL_BACKENDS = ["dino"]
META_COLS = ["sequence_id", "window", "start", "hand", "mesh_id", "verb", "triplet",
             "counterpart_cat", "split", "subject", "touch_frac"]
PAIR_KEY = ["category", "role", "hand", "pair_type", "seq_i", "window_i", "seq_j", "window_j"]
PAIR_ATTRS = ["mesh_i", "mesh_j", "verb_i", "verb_j", "same_mesh", "same_counterpart_cat",
              "same_triplet_ex_mesh", "same_subject", "same_split", "both_train",
              "counterpart_cat_i", "counterpart_cat_j"]
DISTANCES = ["l1", "l2", "cosine"]
N_BOOT = 1000
SEED = 0
MIN_TOUCH = 0.2

log = logging.getLogger("cc")


def setup_logging(name: str = "cc") -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def add_common_args(ap: argparse.ArgumentParser, with_backends: bool = True) -> argparse.ArgumentParser:
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="output root (holds the cache and csvs)")
    if with_backends:
        ap.add_argument("--backends", nargs="+", default=None,
                        help="cache backends; default: aligned normalized random_perm (+ dino if present)")
    ap.add_argument("--min-touch", type=float, default=MIN_TOUCH, dest="min_touch")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, dest="n_boot")
    ap.add_argument("--seed", type=int, default=SEED)
    return ap


def cache_dir(root: Path, backend: str) -> Path:
    return Path(root) / "canonical_contact_cache" / backend


def resolve_backends(root: Path, backends: list[str] | None) -> list[str]:
    """Default backends plus any optional one (dino) whose cache directory exists."""
    if backends:
        return list(backends)
    out = list(DEFAULT_BACKENDS)
    for b in OPTIONAL_BACKENDS:
        if cache_dir(root, b).is_dir() and any(cache_dir(root, b).glob("*__*.npz")):
            out.append(b)
    missing = [b for b in out if not cache_dir(root, b).is_dir()]
    if missing:
        log.warning("cache directories missing for backends %s", missing)
    return out


def list_cache_files(root: Path, backend: str) -> list[tuple[str, str, Path]]:
    """(category, role, path) for every per-window cache file of a backend (skips *__seq.npz)."""
    out = []
    for p in sorted(cache_dir(root, backend).glob("*__*.npz")):
        stem = p.stem
        if stem.endswith("__seq"):
            continue
        cat, role = stem.rsplit("__", 1)
        out.append((cat, role, p))
    return out


def _as_str(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).astype(str)


def load_cache(path: Path, min_touch: float | None = None) -> dict:
    """Load one cache file. Returns dict(X_soft, X_hard, coverage, canonical_points, meta: DataFrame).

    meta columns: META_COLS + 'row' (row index in the file). If min_touch is given, rows with
    touch_frac < min_touch are dropped from every array.
    """
    d = np.load(path, allow_pickle=True)
    keys = set(d.files)
    meta = {}
    if "meta" in keys:  # tolerate a single pickled dict / structured array named 'meta'
        m = d["meta"]
        if m.dtype.names:
            for c in m.dtype.names:
                meta[c] = m[c]
        else:
            m = m.item() if m.shape == () else m
            for c in META_COLS:
                if c in m:
                    meta[c] = np.asarray(m[c])
    for c in META_COLS:
        if c in keys:
            meta[c] = d[c]
    missing = [c for c in META_COLS if c not in meta]
    if missing:
        raise KeyError(f"{path}: missing meta arrays {missing}; have {sorted(keys)}")
    M = len(meta["sequence_id"])
    df = pd.DataFrame({
        "sequence_id": _as_str(meta["sequence_id"]),
        "window": np.asarray(meta["window"]).astype(int),
        "start": np.asarray(meta["start"]).astype(int),
        "hand": _as_str(meta["hand"]),
        "mesh_id": np.asarray(meta["mesh_id"]).astype(int),
        "verb": _as_str(meta["verb"]),
        "triplet": _as_str(meta["triplet"]),
        "counterpart_cat": _as_str(meta["counterpart_cat"]),
        "split": _as_str(meta["split"]),
        "subject": _as_str(meta["subject"]),
        "touch_frac": np.asarray(meta["touch_frac"]).astype(float),
    })
    df["row"] = np.arange(M)
    X_soft = np.asarray(d["X_soft"], dtype=np.float32)
    X_hard = np.asarray(d["X_hard"], dtype=np.float32) if "X_hard" in keys else None
    cov = np.asarray(d["coverage"]) if "coverage" in keys else None
    pts = np.asarray(d["canonical_points"]) if "canonical_points" in keys else None
    assert X_soft.shape[0] == M, (path, X_soft.shape, M)
    keep = np.ones(M, bool)
    if min_touch is not None:
        keep = df["touch_frac"].values >= min_touch
        df = df[keep].reset_index(drop=True)
        X_soft = X_soft[keep]
        X_hard = X_hard[keep] if X_hard is not None else None
        cov = cov[keep] if cov is not None else None
    return dict(X_soft=X_soft, X_hard=X_hard, coverage=cov, canonical_points=pts, meta=df, path=path)


def sample_key_index(meta: pd.DataFrame) -> dict[tuple[str, int, str], int]:
    """(sequence_id, window, hand) -> position in the loaded arrays."""
    keys = zip(meta["sequence_id"].values, meta["window"].values.astype(int), meta["hand"].values)
    idx = {}
    for i, k in enumerate(keys):
        if k in idx:
            raise ValueError(f"duplicate sample key {k}")
        idx[k] = i
    return idx


# --------------------------------------------------------------------------- distances
def pair_distances(X: np.ndarray, ia: np.ndarray, ib: np.ndarray, which=("l1", "l2", "cosine")) -> dict:
    """Row-wise distances between X[ia] and X[ib]; cosine = 1 - cos(sim); NaN where a row is all-zero."""
    A = X[ia].astype(np.float64)
    B = X[ib].astype(np.float64)
    out = {}
    if "l1" in which:
        out["l1"] = np.abs(A - B).sum(1)
    if "l2" in which:
        out["l2"] = np.sqrt(((A - B) ** 2).sum(1))
    if "cosine" in which:
        na = np.linalg.norm(A, axis=1)
        nb = np.linalg.norm(B, axis=1)
        den = na * nb
        cos = np.full(len(ia), np.nan)
        ok = den > 0
        cos[ok] = (A[ok] * B[ok]).sum(1) / den[ok]
        out["cosine"] = 1.0 - np.clip(cos, -1.0, 1.0)
    return out


# --------------------------------------------------------------------------- bootstrap
def seq_bootstrap_draws(n_seq: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_seq) bool: which sequences are present in each resample (with replacement).

    Deterministic in (n_seq, n_boot, seed) so every representation / distance of the same group
    shares identical draws.
    """
    rng = np.random.default_rng(seed)
    inset = np.zeros((n_boot, n_seq), bool)
    draws = rng.integers(0, n_seq, (n_boot, n_seq))
    rows = np.repeat(np.arange(n_boot), n_seq)
    inset[rows, draws.ravel()] = True
    return inset


def seq_bootstrap_counts(n_seq: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_seq) int multiplicity of each sequence in each resample (same rng as draws)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_seq, (n_boot, n_seq))
    counts = np.zeros((n_boot, n_seq), np.int32)
    for b in range(n_boot):
        counts[b] = np.bincount(draws[b], minlength=n_seq)
    return counts


def boot_pair_means(inset: np.ndarray, ia: np.ndarray, ib: np.ndarray, d: np.ndarray,
                    chunk: int = 200) -> np.ndarray:
    """Mean of d over pairs whose BOTH sequences are in the resample; (n_boot,) with NaN if none.

    Pairs with NaN distance are ignored. Inclusion-based (matches step4's baseline-A bootstrap).
    """
    n_boot = inset.shape[0]
    out = np.full(n_boot, np.nan)
    ok = np.isfinite(d)
    ia, ib, d = ia[ok], ib[ok], d[ok]
    if d.size == 0:
        return out
    for s in range(0, n_boot, chunk):
        sub = inset[s:s + chunk]
        keep = sub[:, ia] & sub[:, ib]
        cnt = keep.sum(1)
        tot = keep @ d
        with np.errstate(invalid="ignore", divide="ignore"):
            out[s:s + chunk] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    return out


def percentile_ci(x: np.ndarray, min_valid: int = 10) -> tuple[float, float, int]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < min_valid:
        return (np.nan, np.nan, int(x.size))
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), int(x.size))


def safe_ratio(num, den):
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = np.where(np.isfinite(num) & np.isfinite(den) & (den > 0), r, np.nan)
    return r


def encode_sequences(*seq_arrays: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Map sequence ids from several arrays to a shared integer index. Returns (uniq, [codes...])."""
    allv = np.concatenate([np.asarray(a).astype(str) for a in seq_arrays]) if seq_arrays else np.array([])
    uniq = np.unique(allv)
    pos = {s: i for i, s in enumerate(uniq)}
    codes = [np.array([pos[s] for s in np.asarray(a).astype(str)], dtype=int) for a in seq_arrays]
    return uniq, codes


def read_csv_comments(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", **kw)


def read_pairs(root: Path) -> pd.DataFrame:
    p = read_csv_comments(Path(root) / "contact_pairs.csv")
    for c in ("window_i", "window_j", "mesh_i", "mesh_j"):
        p[c] = p[c].astype(int)
    return p
