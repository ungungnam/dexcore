"""Shared definitions for the dynamic-contact study (time_decomp/dynamic_contact/).

A contact map is the (512,) canonical vector X of build_frame_cache.py (soft = exp(-d/0.02) per
vertex, splat with the fixed per-mesh gauss operator onto the category's canonical points).
This module fixes, once, how X is split into AMOUNT and PATTERN and how "no contact" is decided:

  mass      m(X)   = sum_k X_k
  pattern   P(X)   = X / (m + EPS)                       only meaningful when m > ZERO_MASS
  centroid  mu(X)  = sum_k P_k p_k                       p_k = canonical points (canonical units)
  zero      m < ZERO_MASS  and  n_hard == 0              n_hard = vertices within 1 cm of the hand

ZERO_MASS is chosen from the mass histogram of the full cache (see exp1 log); the value is
recorded in cache/zero_threshold.json so every script reads the same number.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/uhnam/workspace/dexcore")

STUDY = Path("/result/uhnam/dexcore/taco/40_representation_study")
TD = STUDY / "time_decomp"
SEQ = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
CP_DIR = STUDY / "canonical_contact_cache"
BACKEND_DIR = STUDY / "canonical_backend"
TACO_ROOT = Path("/backups/uhnam/TACO")

# ------------------------------------------------------------------ dataset switch
# DC_DATASET selects the output root, the groups, the pair axes and the second hold-out split.
# "taco" is the original study (defaults unchanged); "arctic" and "oakink2" reuse the same scripts
# on caches built by build_arctic_cache.py / build_oakink2_cache.py.
DATASET = os.environ.get("DC_DATASET", "taco")


def _axis(name, kind, lo=None, hi=None, mesh=None, verb=None, subject=None):
    """A pair axis. within: same take, phase gap in [lo, hi). cross: other take, with each of
    mesh/verb/subject either equal (True), different (False) or free (None), phase gap < TOL."""
    return dict(name=name, kind=kind, lo=lo, hi=hi, mesh=mesh, verb=verb, subject=subject)


FLOOR = _axis("floor", "within", 0.0, 0.05)
TIME = _axis("time", "within", 0.5, 1.01)
if DATASET == "taco":
    OUT = TD / "dynamic_contact"
    GROUPS = [("plate", "target", "L"), ("pan", "target", "L"), ("spoon", "tool", "R"),
              ("bowl", "target", "L"), ("spatula", "tool", "R"), ("roller", "tool", "R"),
              ("knife", "tool", "R"), ("bowl", "tool", "R")]
    AXES = [FLOOR, TIME, _axis("take", "cross", mesh=True, verb=True), _axis("mesh", "cross", mesh=False, verb=True),
            _axis("action", "cross", mesh=True, verb=False)]
    RATIOS = [("time", "take"), ("time", "mesh"), ("mesh", "take"), ("action", "take"), ("time", "floor")]
    AX2 = "mesh"            # the second identity axis (what "mesh" was in TACO)
    SPLIT2 = "mesh"         # the second hold-out split of Experiment 3 / 4
    ROLE_SPLIT = ("target", "tool")
elif DATASET == "arctic":
    OUT = Path("/result/uhnam/dexcore/arctic/30_dynamic_contact")
    AXES = [FLOOR, TIME, _axis("take", "cross", verb=True), _axis("subject", "cross", verb=True, subject=False),
            _axis("take_same_subject", "cross", verb=True, subject=True), _axis("action", "cross", verb=False)]
    RATIOS = [("time", "take"), ("time", "subject"), ("subject", "take_same_subject"), ("action", "take"), ("time", "floor")]
    AX2 = "subject"
    SPLIT2 = "subject"
    ROLE_SPLIT = ("L", "R")
elif DATASET in ("oakink2", "oakink2ext"):
    OUT = Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact" if DATASET == "oakink2" else
               "/result/uhnam/dexcore/oakink2/11_dynamic_contact_extended")
    AXES = [FLOOR, TIME, _axis("take", "cross", mesh=True, verb=True), _axis("mesh", "cross", mesh=False, verb=True),
            _axis("action", "cross", mesh=True, verb=False), _axis("subject", "cross", mesh=True, verb=True, subject=False)]
    RATIOS = [("time", "take"), ("time", "mesh"), ("mesh", "take"), ("action", "take"), ("subject", "take"), ("time", "floor")]
    AX2 = "mesh"
    SPLIT2 = "mesh"
    ROLE_SPLIT = ("L", "R")
else:
    raise ValueError(DATASET)
CACHE = OUT / "cache"
if DATASET != "taco":
    _gj = CACHE / "groups.json"
    if _gj.exists():
        GROUPS = [tuple(g) for g in json.loads(_gj.read_text())]
    else:
        GROUPS = sorted(tuple(p.stem.split("__")) for p in (CACHE / "frames_full").glob("*__*__*.npz")) if (CACHE / "frames_full").exists() else []
AXIS_NAMES = [a["name"] for a in AXES]
CROSS_AXES = [a["name"] for a in AXES if a["kind"] == "cross"]

FPS = 30
STRIDE_PAIRS = 2          # the four-axis pairing lives on every 2nd frame (15 Hz)
TOUCH_MIN = 0.2           # window-level touch fraction used by the earlier study
EPS = 1e-6
HARD_MM = 0.01            # a vertex within 1 cm counts as hard contact
META_KEYS = ("sequence_id", "mesh_id", "verb", "split", "subject", "window", "frame",
             "frame_in_window", "phase", "touch_window", "n_hard", "min_d")


def gname(cat, role, hand):
    return f"{cat}__{role}__{hand}"


def zero_threshold() -> float:
    p = CACHE / "zero_threshold.json"
    if p.exists():
        return float(json.loads(p.read_text())["zero_mass"])
    raise FileNotFoundError("run exp1_threshold.py first (writes cache/zero_threshold.json)")


def load_group(cat, role, hand, backend="normalized", stride=1, touching_windows_only=False):
    """X (n,512) float32, meta DataFrame, canonical points (512,3)."""
    sub = "frames_full" if backend == "normalized" else f"frames_full_{backend}"
    z = np.load(CACHE / sub / f"{gname(cat, role, hand)}.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in META_KEYS})
    M["sequence_id"] = M.sequence_id.astype(str)
    M["mesh_id"] = M.mesh_id.astype(str)
    M["verb"] = M.verb.astype(str)
    M["subject"] = M.subject.astype(str)
    keep = np.ones(len(M), bool)
    if stride > 1:
        keep &= (M.frame_in_window.values % stride) == 0
    if touching_windows_only:
        keep &= M.touch_window.values >= TOUCH_MIN
    if not keep.all():
        X, M = X[keep], M[keep].reset_index(drop=True)
    P = z["canonical_points"].astype(np.float64)
    return X, M, P


def mass(X):
    return X.sum(-1)


def pattern(X):
    return X / (X.sum(-1, keepdims=True) + EPS)


def centroid(X, P):
    """(n,512),(512,3) -> (n,3) mass-weighted centroid in canonical units."""
    return pattern(X) @ P


def is_zero(m, n_hard, thr):
    return (m < thr) & (n_hard == 0)


# --------------------------------------------------------------------------- statistics
def seq_bootstrap(values, seq_a, seq_b=None, n_boot=1000, stat=np.mean, seed=0):
    """Cluster bootstrap over SEQUENCES. Each pair carries the sequence ids of its two members
    (seq_b None = one sequence per value, e.g. per-sequence curves). A bootstrap replicate
    resamples sequences with replacement and weights each value by the product of the draw
    counts of its members (the standard multiplier bootstrap for pair statistics). Returns
    (point, lo, hi) with a 95% percentile interval."""
    values = np.asarray(values, float)
    seqs = np.unique(np.concatenate([seq_a, seq_b]) if seq_b is not None else seq_a)
    idx = {s: i for i, s in enumerate(seqs)}
    ia = np.array([idx[s] for s in seq_a])
    ib = np.array([idx[s] for s in seq_b]) if seq_b is not None else None
    rng = np.random.default_rng(seed)
    out = []
    S = len(seqs)
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        w = cnt[ia] * (cnt[ib] if ib is not None else 1.0)
        if w.sum() == 0:
            continue
        out.append(wstat(values, w, stat))
    out = np.asarray(out)
    return float(stat(values)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def wstat(v, w, stat):
    if stat is np.mean:
        return float((v * w).sum() / w.sum())
    if stat is np.median:
        o = np.argsort(v)
        cw = np.cumsum(w[o])
        return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])
    raise ValueError(stat)


def paired_ratio_bootstrap(num, den, seq_num, seq_den, n_boot=1000, stat=np.mean, seed=0):
    """Ratio of two pair-statistics (e.g. time / mesh) with a paired sequence bootstrap: the same
    sequence draw is applied to both numerator and denominator samples."""
    seqs = np.unique(np.concatenate([seq_num.ravel(), seq_den.ravel()]))
    idx = {s: i for i, s in enumerate(seqs)}
    S = len(seqs)
    f = lambda arr: np.vectorize(idx.get)(arr)
    inum, iden = f(seq_num), f(seq_den)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        cnt = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        wn = np.prod(cnt[inum], axis=1) if inum.ndim == 2 else cnt[inum]
        wd = np.prod(cnt[iden], axis=1) if iden.ndim == 2 else cnt[iden]
        if wn.sum() == 0 or wd.sum() == 0:
            continue
        out.append(wstat(num, wn, stat) / max(wstat(den, wd, stat), 1e-12))
    out = np.asarray(out)
    return float(stat(num) / stat(den)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
