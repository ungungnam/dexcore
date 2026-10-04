#!/usr/bin/env python
"""OakInk2 -> the dynamic-contact frame cache (stage 2; same format as build_full_cache.py).

Groups = (category, "obj", hand) with at least MIN_TAKES takes. Category canonical support: the
instance with the most takes is the template; each instance is centred and scaled to unit max
radius (the 'normalized' backend of the TACO study), the template's 512 farthest-point-sampled
vertices are the canonical points and every instance gets its own row-normalised gauss operator
W (radius 0.15, sigma 0.075) onto them. Windows: 64 frames (30 Hz) from the take's first frame.
Meta: mesh_id = object instance id, verb = program primitive, subject = actor, take = (sequence,
hand, object, segment). Object state (15 dims, world frame): rotation 6D, translation relative to
the take's first frame, linear and angular velocity.
Output: /result/uhnam/dexcore/oakink2/10_dynamic_contact/cache/{frames_full/<cat>__obj__<L|R>.npz,
geometry_normalized__<cat>.npz, object_states.npz, groups.json, window_index.csv}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import os
OUT = Path(os.environ.get("OAKINK2_OUT", "/result/uhnam/dexcore/oakink2/10_dynamic_contact"))
CACHE = OUT / "cache"
MESH_CACHE = Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact/meshes")
RADIUS, SIGMA, K, WIN, TOUCH_MIN, HARD, MIN_TAKES = 0.15, 0.075, 512, 64, 0.2, 0.01, 20
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("oakink2_cache")


def fps(P, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(min(k, len(P)) - 1):
        i = int(np.argmax(d)); idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)


def normalise(V):
    c = V.mean(0); r = np.linalg.norm(V - c, axis=1).max()
    return (V - c) / r, r


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


def angvel(R):
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:]); w = Rotation.from_matrix(rel).as_rotvec()
    out = np.zeros((len(R), 3)); out[1:-1] = 0.5 * (w[:-1] + w[1:]); out[0], out[-1] = w[0], w[-1]; return out


def vel(p):
    out = np.zeros_like(p); out[1:-1] = 0.5 * (p[2:] - p[:-2]); out[0], out[-1] = p[1] - p[0], p[-1] - p[-2]; return out


def main():
    (CACHE / "frames_full").mkdir(parents=True, exist_ok=True)
    ix = pd.read_csv(OUT / "takes_index.csv")
    ix = ix[ix.n_frames >= WIN].copy()
    ix["take_id"] = ix.file.str.replace(".npz", "", regex=False)
    ix["hand_LR"] = ix.hand.map({"lh": "L", "rh": "R"})
    counts = ix.groupby(["category", "hand_LR"]).take_id.nunique()
    groups = [(c, "obj", h) for (c, h), n in counts.items() if n >= MIN_TAKES]
    log.info("groups with >= %d takes: %s", MIN_TAKES, groups)
    json.dump(groups, open(CACHE / "groups.json", "w"))
    # ---- object states (per take, 30 Hz frames of the take)
    states, off, ids = [], [0], []
    for r in ix.itertuples():
        with np.load(OUT / "takes" / r.file) as z:
            Tm = z["obj_transf"].astype(np.float64)
        R, p = Tm[:, :3, :3], Tm[:, :3, 3]
        f = np.concatenate([R[:, :, :2].reshape(len(R), 6), p - p[0], vel(p), angvel(R)], 1).astype(np.float32)
        states.append(f); ids.append(r.take_id); off.append(off[-1] + len(f))
    names = [f"rot6d_{i}" for i in range(6)] + ["rel_x", "rel_y", "rel_z", "vel_x", "vel_y", "vel_z", "angvel_x", "angvel_y", "angvel_z"]
    np.savez_compressed(CACHE / "object_states.npz", sequence_id=np.array(ids), offset=np.array(off),
                        states=np.concatenate(states), names=np.array(names))
    # ---- per group
    win_rows = []
    done_geo = set()
    for cat, _, hand in groups:
        sub = ix[(ix.category == cat) & (ix.hand_LR == hand)]
        insts = sub.groupby("object").take_id.nunique().sort_values(ascending=False)
        template = insts.index[0]
        meshes = {o: np.load(MESH_CACHE / f"{o}.npz")["verts"].astype(np.float64) for o in ix[ix.category == cat].object.unique()}
        Vt, _ = normalise(meshes[template]); P = Vt[fps(Vt, K)]
        Ws, geo = {}, {}
        for o, V in meshes.items():
            Vn, rad = normalise(V); W, tree = weight_matrix(Vn, P); Ws[o] = W
            dn, _ = tree.query(P)
            geo[o] = dict(nearest=np.minimum(dn, 2 * RADIUS).astype(np.float32), coverage=dn <= RADIUS, radius_m=rad, extent_m=(V.max(0) - V.min(0)).astype(np.float32))
        if cat not in done_geo:
            ids_ = sorted(geo)
            np.savez_compressed(CACHE / f"geometry_normalized__{cat}.npz", mesh_ids=np.array(ids_), nearest=np.stack([geo[i]["nearest"] for i in ids_]),
                                coverage=np.stack([geo[i]["coverage"] for i in ids_]), radius_m=np.array([geo[i]["radius_m"] for i in ids_]),
                                extent_m=np.stack([geo[i]["extent_m"] for i in ids_]))
            done_geo.add(cat)
        X, meta = [], []
        for r in sub.itertuples():
            with np.load(OUT / "takes" / r.file) as z:
                d = z["contact"].astype(np.float32)
            W = Ws[r.object]
            T = len(d); starts = list(range(0, T - WIN + 1, WIN))
            touch = [float((d[s:s + WIN].min(1) < HARD).mean()) for s in starts]
            touching = [s for s, t in zip(starts, touch) if t >= TOUCH_MIN]
            lo, hi = (touching[0], touching[-1] + WIN - 1) if touching else (np.nan, np.nan)
            for w, (s, t) in enumerate(zip(starts, touch)):
                win_rows.append(dict(take_id=r.take_id, category=cat, hand=hand, object=r.object, window=w, start=s, verb=r.primitive,
                                     subject=r.actor, touch=t))
                fr = np.arange(s, s + WIN); dd = d[fr]
                x = np.exp(-dd / 0.02).astype(np.float32) @ W.T
                X.append(x.astype(np.float16))
                meta.append(pd.DataFrame({"sequence_id": r.take_id, "mesh_id": r.object, "verb": r.primitive, "split": "all",
                                          "subject": r.actor, "window": w, "frame": fr, "frame_in_window": fr - s,
                                          "phase": (fr - lo) / max(hi - lo, 1) if lo == lo else np.nan, "touch_window": t,
                                          "n_hard": (dd < HARD).sum(1).astype(np.int32), "min_d": dd.min(1).astype(np.float32)}))
        X = np.concatenate(X); M = pd.concat(meta, ignore_index=True)
        np.savez_compressed(CACHE / "frames_full" / f"{cat}__obj__{hand}.npz", X=X, canonical_points=P.astype(np.float32),
                            **{c: M[c].values for c in M.columns})
        tw = M.drop_duplicates(["sequence_id", "window"])
        log.info("%s/%s: %d frames, %d windows (%.0f%% touching), %d takes, %d instances, %d subjects, %d verbs", cat, hand, len(X), len(tw),
                 100 * (tw.touch_window >= TOUCH_MIN).mean(), M.sequence_id.nunique(), M.mesh_id.nunique(), M.subject.nunique(), M.verb.nunique())
    pd.DataFrame(win_rows).to_csv(CACHE / "window_index.csv", index=False)


if __name__ == "__main__":
    main()
