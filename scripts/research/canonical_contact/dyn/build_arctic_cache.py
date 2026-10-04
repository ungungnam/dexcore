#!/usr/bin/env python
"""ARCTIC -> the dynamic-contact frame cache (same format as build_full_cache.py for TACO).

Inputs (all existing):
  /result/uhnam/dexcore/arctic/20_bimart_contact_probe/sequences/<seq>.npz  contact_left/right (T, N)
      per-vertex hand-surface distance [m]; vertex order = top part then bottom part
  .../assets/arctic_mesh_dict.npy   <obj>_top / <obj>_bottom: verts_original (rest pose, metres)
  /data/uhnam/ARCTIC/data/raw_seqs/<subj>/<name>.object.npy   (T,7) arti [rad], rotvec, trans [mm]
One object per category, so the canonical support is the object's own rest-pose mesh: centre at
the centroid, scale to unit max radius, 512 farthest-point-sampled canonical points, W = the same
row-normalised gauss operator as TACO (radius 0.15, sigma 0.075). Windows: 64 frames from frame 0,
non-overlapping. Groups: (object, "obj", hand). Object state (17 dims): articulation, root
rotation 6D, translation relative to frame 0, linear velocity, angular velocity, articulation rate.
Output: /result/uhnam/dexcore/arctic/30_dynamic_contact/cache/{frames_full/<obj>__obj__<L|R>.npz,
geometry_normalized__<obj>.npz, object_states.npz, window_index.csv}
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

PROBE = Path("/result/uhnam/dexcore/arctic/20_bimart_contact_probe")
RAW = Path("/data/uhnam/ARCTIC/data/raw_seqs")
OUT = Path("/result/uhnam/dexcore/arctic/30_dynamic_contact/cache")
RADIUS, SIGMA, K, WIN, TOUCH_MIN, HARD = 0.15, 0.075, 512, 64, 0.2, 0.01
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("arctic_cache")


def fps(P, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(k - 1):
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)


def weight_matrix(V, P):
    tree = cKDTree(V)
    W = np.zeros((len(P), len(V)), np.float32)
    for k, nb in enumerate(tree.query_ball_point(P, RADIUS)):
        if not nb:
            continue
        nb = np.asarray(nb)
        w = np.exp(-0.5 * ((V[nb] - P[k]) ** 2).sum(1) / SIGMA ** 2)
        W[k, nb] = w / w.sum()
    return W, tree


def rot6d(R):
    return R[:, :, :2].reshape(len(R), 6)


def angvel(R):
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    w = Rotation.from_matrix(rel).as_rotvec()
    out = np.zeros((len(R), 3)); out[1:-1] = 0.5 * (w[:-1] + w[1:]); out[0], out[-1] = w[0], w[-1]
    return out


def vel(p):
    out = np.zeros_like(p); out[1:-1] = 0.5 * (p[2:] - p[:-2]); out[0], out[-1] = p[1] - p[0], p[-1] - p[-2]
    return out


def main():
    (OUT / "frames_full").mkdir(parents=True, exist_ok=True)
    md = np.load(PROBE / "assets/arctic_mesh_dict.npy", allow_pickle=True).item()
    si = pd.read_csv(PROBE / "sequence_index.csv")
    si["subject"] = si.sequence_id.str.split("/").str[1]
    si["name"] = si.sequence_id.str.split("/").str[0]
    # ---- object states
    states, off, ids = [], [0], []
    for r in si.itertuples():
        o = np.load(RAW / r.subject / f"{r.name}.object.npy").astype(np.float64)[: r.n_frames]
        arti, R, p = o[:, :1], Rotation.from_rotvec(o[:, 1:4]).as_matrix(), o[:, 4:7] / 1000.0
        f = np.concatenate([arti, rot6d(R), p - p[0], vel(p), angvel(R), vel(arti)], 1).astype(np.float32)
        states.append(f); ids.append(r.sequence_id); off.append(off[-1] + len(f))
    names = ["arti"] + [f"rot6d_{i}" for i in range(6)] + ["rel_x", "rel_y", "rel_z", "vel_x", "vel_y", "vel_z",
                                                          "angvel_x", "angvel_y", "angvel_z", "arti_rate"]
    np.savez_compressed(OUT / "object_states.npz", sequence_id=np.array(ids), offset=np.array(off),
                        states=np.concatenate(states), names=np.array(names))
    log.info("object states: %d sequences, %d frames", len(ids), off[-1])
    # ---- per object: canonical support and frames
    win_rows = []
    for obj in sorted(si.tool_cat.unique()):
        V = np.concatenate([md[f"{obj}_top"]["verts_original"], md[f"{obj}_bottom"]["verts_original"]]).astype(np.float64)
        c = V.mean(0); rad = np.linalg.norm(V - c, axis=1).max()
        Vn = (V - c) / rad
        P = Vn[fps(Vn, K)]
        W, tree = weight_matrix(Vn, P)
        d_near, _ = tree.query(P)
        np.savez_compressed(OUT / f"geometry_normalized__{obj}.npz", mesh_ids=np.array([obj]),
                            nearest=np.minimum(d_near, 2 * RADIUS)[None].astype(np.float32),
                            coverage=(d_near <= RADIUS)[None], radius_m=np.array([rad]),
                            extent_m=(V.max(0) - V.min(0))[None].astype(np.float32))
        for hand, key in (("L", "contact_left"), ("R", "contact_right")):
            X, meta = [], []
            for r in si[si.tool_cat == obj].itertuples():
                with np.load(PROBE / "sequences" / r.file) as z:
                    d = z[key]
                T = len(d)
                starts = list(range(0, T - WIN + 1, WIN))
                touch = [float((d[s:s + WIN].min(1) < HARD).mean()) for s in starts]
                touching = [s for s, t in zip(starts, touch) if t >= TOUCH_MIN]
                lo, hi = (touching[0], touching[-1] + WIN - 1) if touching else (np.nan, np.nan)
                for w, (s, t) in enumerate(zip(starts, touch)):
                    win_rows.append(dict(sequence_id=r.sequence_id, object=obj, hand=hand, window=w, start=s,
                                         verb=r.verb, subject=r.subject, split=r.split, touch=t))
                    fr = np.arange(s, s + WIN)
                    dd = d[fr]
                    x = (np.exp(-dd / 0.02).astype(np.float32) @ W.T)
                    X.append(x.astype(np.float16))
                    meta.append(pd.DataFrame({
                        "sequence_id": r.sequence_id, "mesh_id": obj, "verb": r.verb, "split": r.split,
                        "subject": r.subject, "window": w, "frame": fr, "frame_in_window": fr - s,
                        "phase": (fr - lo) / max(hi - lo, 1) if lo == lo else np.nan,
                        "touch_window": t, "n_hard": (dd < HARD).sum(1).astype(np.int32),
                        "min_d": dd.min(1).astype(np.float32)}))
            X = np.concatenate(X); M = pd.concat(meta, ignore_index=True)
            np.savez_compressed(OUT / "frames_full" / f"{obj}__obj__{hand}.npz", X=X,
                                canonical_points=P.astype(np.float32), **{c: M[c].values for c in M.columns})
            tw = M.drop_duplicates(["sequence_id", "window"])
            log.info("%s/%s: %d frames, %d windows (%.0f%% touching), %d takes, %d subjects", obj, hand, len(X),
                     len(tw), 100 * (tw.touch_window >= TOUCH_MIN).mean(), M.sequence_id.nunique(), M.subject.nunique())
    pd.DataFrame(win_rows).to_csv(OUT / "window_index.csv", index=False)


if __name__ == "__main__":
    main()
