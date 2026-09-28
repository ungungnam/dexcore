#!/usr/bin/env python
"""Per-FRAME canonical contact vectors, for decomposing mesh identity against time.

The study's cache averages each 64-frame window into one vector, so time was never measured. The
splat is a fixed linear map per mesh (gauss weights over that mesh's canonical-mapped vertices),
so the same operator applies frame by frame: build it once per mesh as a (K, N) matrix and hit the
whole window with one matmul. Verified against the cached window vectors in the same run.

  PYTHONPATH=/home/uhnam/workspace/dexcore python .../build_frame_cache.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.analysis.canonical import backends as BR
from src.analysis.canonical.interface import DEFAULT_RADIUS

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
OUT = ROOT / "time_decomp"
SEQ = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
BACKEND = "normalized"
RADIUS = DEFAULT_RADIUS
SIGMA = RADIUS / 2.0
STRIDE = 2                      # every 2nd frame of the 64-frame window
TOUCH_MIN = 0.2
GROUPS = [("plate", "target", "L"), ("pan", "target", "L"), ("spoon", "tool", "R"),
          ("bowl", "target", "L"), ("spatula", "tool", "R"), ("roller", "tool", "R"),
          ("knife", "tool", "R"), ("bowl", "tool", "R")]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("frames")


def weight_matrix(be, cat: str, mesh: str) -> np.ndarray:
    """(K, N) row-normalised gauss weights = exactly what `splat(kernel='gauss')` computes."""
    fit = be._cat(cat)
    V, tree = fit.mapped(mesh)
    P = fit.canonical_points
    W = np.zeros((len(P), len(V)), np.float32)
    for k, nb in enumerate(tree.query_ball_point(P, RADIUS)):
        if not nb:
            continue
        nb = np.asarray(nb)
        w = np.exp(-0.5 * ((tree.data[nb] - P[k]) ** 2).sum(1) / SIGMA ** 2)
        W[k, nb] = w / w.sum()
    return W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    be = BR.load(BACKEND, str(ROOT / "canonical_backend"))
    wi = pd.read_csv(ROOT / "window_index.csv")
    files = pd.read_csv(SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    checks = []
    for cat, role, hand in GROUPS:
        cache = np.load(ROOT / f"canonical_contact_cache/{BACKEND}/{cat}__{role}.npz", allow_pickle=True)
        cm = pd.DataFrame({k: cache[k] for k in ("sequence_id", "window", "hand")})
        cm["row"] = np.arange(len(cm))
        key = cm[cm.hand == hand].set_index(["sequence_id", "window"]).row
        rows = wi[wi[f"{role}_cat"] == cat].copy()
        rows = rows[rows[f"touch_{hand}_{role}"] >= TOUCH_MIN]
        mesh_col = f"{role}_mesh"
        X, meta = [], []
        known = set(be.mesh_ids(cat))
        for mesh, g in rows.groupby(mesh_col):
            mesh = f"{int(mesh):03d}"          # the backend keys meshes zero-padded
            if mesh not in known:              # a mesh the category was never fitted with
                log.info("  %s/%s: mesh %s not in the fit, skipped (%d windows)", cat, role, mesh, len(g))
                continue
            W = weight_matrix(be, cat, mesh)
            for seq, gg in g.groupby("sequence_id"):
                with np.load(SEQ / "sequences" / files[seq]) as z:
                    d = z[f"contact_{'left' if hand == 'L' else 'right'}"]
                n_tool = int(gg.n_tool.iloc[0])
                d = d[:, :n_tool] if role == "tool" else d[:, n_tool:]
                starts = sorted(gg.start.astype(int))
                lo, hi = starts[0], starts[-1] + 63
                for _, r in gg.iterrows():
                    s = int(r.start)
                    fr = np.arange(s, min(s + 64, len(d)), STRIDE)
                    soft = np.exp(-d[fr] / 0.02).astype(np.float32)      # (f, N)
                    x = soft @ W.T                                        # (f, K)
                    X.append(x.astype(np.float16))
                    meta.append(pd.DataFrame({
                        "sequence_id": seq, "mesh_id": mesh, "verb": r.verb,
                        "split": r.split, "subject": r.subject, "window": int(r.window),
                        "frame": fr, "frame_in_window": fr - s,
                        "phase": (fr - lo) / max(hi - lo, 1)}))
                    if len(checks) < 40 and (seq, int(r.window)) in key.index:
                        ref = cache["X_soft"][int(key.loc[(seq, int(r.window))])]
                        checks.append(float(np.abs(x.mean(0) - ref).max()))
        X = np.concatenate(X); M = pd.concat(meta, ignore_index=True)
        np.savez_compressed(OUT / f"{cat}__{role}__{hand}.npz", X=X,
                            **{c: M[c].values for c in M.columns})
        log.info("%s/%s/%s: %d frames, %d windows, %d sequences, %d meshes",
                 cat, role, hand, len(X), M.groupby(["sequence_id", "window"]).ngroups,
                 M.sequence_id.nunique(), M.mesh_id.nunique())
    log.info("window-mean vs cached X_soft: max abs diff over %d checks = %.2e",
             len(checks), max(checks))


if __name__ == "__main__":
    main()
