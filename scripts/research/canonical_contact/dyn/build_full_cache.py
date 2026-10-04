#!/usr/bin/env python
"""Per-frame canonical contact vectors at STRIDE 1 over EVERY window of every sequence (no touch
filter), plus per-mesh geometry descriptors. Same operator as build_frame_cache.py (verified
against it below), so the stride-2 / touch>=0.2 subset of this cache IS the earlier cache.

  python build_full_cache.py --backend normalized
  python build_full_cache.py --backend aligned        (verification rerun)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import dc_common as C
from src.analysis.canonical import backends as BR
from src.analysis.canonical.interface import DEFAULT_RADIUS

RADIUS, SIGMA = DEFAULT_RADIUS, DEFAULT_RADIUS / 2.0
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("full_cache")


def weight_matrix(be, cat, mesh):
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


def geometry_descriptor(be, cat, mesh, mesh_dict):
    """Mesh geometry in the canonical index space: distance from each canonical point to the
    nearest mapped vertex (clipped at 2*RADIUS), coverage, plus the metric scale."""
    fit = be._cat(cat)
    V, tree = fit.mapped(mesh)
    d, _ = tree.query(fit.canonical_points)
    verts = np.asarray(mesh_dict[mesh]["verts_original"], float)
    return dict(nearest=np.minimum(d, 2 * RADIUS).astype(np.float32),
                coverage=(d <= RADIUS), radius_m=float(np.linalg.norm(verts - verts.mean(0), axis=1).max()),
                extent_m=(verts.max(0) - verts.min(0)).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="normalized")
    a = ap.parse_args()
    sub = "frames_full" if a.backend == "normalized" else f"frames_full_{a.backend}"
    out = C.CACHE / sub
    out.mkdir(parents=True, exist_ok=True)
    be = BR.load(a.backend, str(C.BACKEND_DIR))
    mesh_dict = np.load(C.SEQ / "assets/taco_mesh_dict.npy", allow_pickle=True).item()
    wi = pd.read_csv(C.STUDY / "window_index.csv")
    files = pd.read_csv(C.SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    old_checks = []
    geo_written = set()
    for cat, role, hand in C.GROUPS:
        rows = wi[wi[f"{role}_cat"] == cat].copy()
        touch_col = f"touch_{hand}_{role}"
        mesh_col = f"{role}_mesh"
        known = set(be.mesh_ids(cat))
        old = None
        if a.backend == "normalized":
            zo = np.load(C.TD / f"{C.gname(cat, role, hand)}.npz", allow_pickle=True)
            old = (zo["X"].astype(np.float32),
                   pd.DataFrame({k: zo[k] for k in ("sequence_id", "frame")}))
            old[1]["sequence_id"] = old[1].sequence_id.astype(str)
            old_key = {(s, f): i for i, (s, f) in enumerate(zip(old[1].sequence_id, old[1].frame))}
        X, meta, geo = [], [], {}
        for mesh, g in rows.groupby(mesh_col):
            mesh = f"{int(mesh):03d}"
            if mesh not in known:
                log.info("  %s/%s: mesh %s not in the fit, skipped (%d windows)", cat, role, mesh, len(g))
                continue
            W = weight_matrix(be, cat, mesh)
            geo[mesh] = geometry_descriptor(be, cat, mesh, mesh_dict)
            for seq, gg in g.groupby("sequence_id"):
                with np.load(C.SEQ / "sequences" / files[seq]) as z:
                    d = z[f"contact_{'left' if hand == 'L' else 'right'}"]
                n_tool = int(gg.n_tool.iloc[0])
                d = d[:, :n_tool] if role == "tool" else d[:, n_tool:]
                touching = gg[gg[touch_col] >= C.TOUCH_MIN]
                if len(touching):
                    lo, hi = int(touching.start.min()), int(touching.start.max()) + 63
                else:
                    lo, hi = np.nan, np.nan
                for _, r in gg.iterrows():
                    s = int(r.start)
                    fr = np.arange(s, min(s + 64, len(d)))
                    dd = d[fr]
                    soft = np.exp(-dd / 0.02).astype(np.float32)
                    x = soft @ W.T
                    X.append(x.astype(np.float16))
                    meta.append(pd.DataFrame({
                        "sequence_id": seq, "mesh_id": mesh, "verb": r.verb, "split": r.split,
                        "subject": int(r.subject), "window": int(r.window), "frame": fr,
                        "frame_in_window": fr - s,
                        "phase": (fr - lo) / max(hi - lo, 1) if lo == lo else np.nan,
                        "touch_window": float(r[touch_col]),
                        "n_hard": (dd < C.HARD_MM).sum(1).astype(np.int32),
                        "min_d": dd.min(1).astype(np.float32)}))
                    if old is not None and len(old_checks) < 200:
                        for j, f in enumerate(fr):
                            if (seq, int(f)) in old_key:
                                old_checks.append(float(np.abs(x[j] - old[0][old_key[(seq, int(f))]]).max()))
                                break
        X = np.concatenate(X)
        M = pd.concat(meta, ignore_index=True)
        np.savez_compressed(out / f"{C.gname(cat, role, hand)}.npz", X=X,
                            canonical_points=be.canonical_points(cat).astype(np.float32),
                            **{c: M[c].values for c in M.columns})
        if cat not in geo_written:
            ids = sorted(geo)
            np.savez_compressed(C.CACHE / f"geometry_{a.backend}__{cat}.npz", mesh_ids=np.array(ids),
                                nearest=np.stack([geo[i]["nearest"] for i in ids]),
                                coverage=np.stack([geo[i]["coverage"] for i in ids]),
                                radius_m=np.array([geo[i]["radius_m"] for i in ids]),
                                extent_m=np.stack([geo[i]["extent_m"] for i in ids]))
            geo_written.add(cat)
        log.info("%s/%s/%s: %d frames, %d windows, %d sequences, %d meshes, %.0f%% touching windows",
                 cat, role, hand, len(X), M.groupby(["sequence_id", "window"]).ngroups,
                 M.sequence_id.nunique(), M.mesh_id.nunique(),
                 100 * (M.drop_duplicates(["sequence_id", "window"]).touch_window >= C.TOUCH_MIN).mean())
    if old_checks:
        log.info("agreement with the stride-2 cache (max abs diff, %d frames, fp16 storage): %.2e",
                 len(old_checks), max(old_checks))


if __name__ == "__main__":
    main()
