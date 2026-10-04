#!/usr/bin/env python
"""Step 2: semantic contact patches and directional wrench capacity for every selected event.
    python wrench_analysis.py --dataset taco [--workers 32] [--split test]

Settings (varied one at a time from the primary: contact threshold 1 cm, patch merge radius 1 cm,
finger-level unit budgets, mu = 0.5): mu in {0.3, 0.5, 0.8, 1.0}; merge radius 0 / 2 cm; contact
threshold 0.5 / 1.5 cm; budgets per patch / whole hand.
Per event and setting:
  patches at the pre frame (s) and the post frame (e + 1); capability profiles q_pre, q_post over the
  shared 76 directions (64 random + 12 axes); scalar summaries (Q = mean, min, median, coverage,
  cosine, relative L1 distance, retention both ways);
  Metric B: capacities along the event's future-motion directions (K = 8, 16 frames after the event
  start): the translational-acceleration direction (pure force), the rotational-acceleration axis
  (pure torque) and the gravity-support direction, all expressed in the object frame at the frame
  where the grasp is evaluated -- for the POST grasp at its own frame, for the PRE grasp transported
  (attached to its vertex ids; ARCTIC's top part articulated) to the post frame, and for both grasps
  transported to s + K;
  finger-level participation and per-finger patch changes (appeared / disappeared / slid / stable).
Writes <ds>/events_metrics.csv (event x setting), <ds>/directional_capacity.csv (primary setting,
event x direction), <ds>/capacity/<setting>.npz (all profiles), <ds>/finger_transitions.csv.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import wc_common as W
import wrench_lp as L

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("wrench")
U_SHARED, U_NAMES = L.direction_set()
SLIDE_M = 0.010                                                       # a finger 'slid' when its patch centroid moved > 1 cm
_MESH = {}


def settings():
    base = dict(thr=W.CONTACT_THR["primary"], merge=W.MERGE_R["primary"], budget=W.BUDGET_PRIMARY, mu=W.MU_PRIMARY)
    out = [dict(name="primary", **base)]
    for mu in W.MU_SET:
        if mu != W.MU_PRIMARY:
            out.append(dict(name=f"mu{mu}", **{**base, "mu": mu}))
    for k in ("none", "wide"):
        out.append(dict(name=f"merge_{k}", **{**base, "merge": W.MERGE_R[k]}))
    for k in ("tight", "loose"):
        out.append(dict(name=f"thr_{k}", **{**base, "thr": W.CONTACT_THR[k]}))
    for k in ("per_patch", "hand_total"):
        out.append(dict(name=f"budget_{k}", **{**base, "budget": k}))
    return out


def load_geometry(ds, event_id):
    z = np.load(W.ds_out(ds) / "geometry" / f"{event_id}.npz", allow_pickle=True)
    g = {k: z[k] for k in z.files}
    g["frames"] = json.loads(str(g["frames"])); g["mesh"] = json.loads(str(g["mesh"]))
    return g


def geom_dict(g, k, frame=None):
    """contact_geometry dict of the pre / post vertex set at its own frame or transported to `frame`."""
    d = dict(ids=g[f"{k}_ids"], dist=g[f"{k}_dist"], pos=g[f"{k}_pos"], nrm=g[f"{k}_nrm"], label=g[f"{k}_label"], outward=g[f"{k}_outward"])
    if frame is not None and frame != k:
        d = dict(d, pos=g[f"tr_{k}_{frame}_pos"], nrm=g[f"tr_{k}_{frame}_nrm"])
    return d


def mesh_of(ds, ev):
    get = (lambda k: ev[k]) if isinstance(ev, dict) else (lambda k: getattr(ev, k))
    key = str(get("mesh_id")) if ds == "taco" else str(get("category"))
    if key not in _MESH:
        _MESH[key] = W.taco_mesh(key) if ds == "taco" else W.arctic_mesh(key)
    return _MESH[key]


def motion_directions(g, frame_key, K):
    """Pure-force, pure-torque and support directions (6-D unit) in the object frame at frame_key."""
    R = g[f"motion_R_{frame_key}"] if frame_key in ("pre", "post") else g[f"motion_R_{frame_key}"]
    dv, dom = g.get(f"motion_dv_K{K}"), g.get(f"motion_dom_K{K}")
    if dv is None:
        return {}
    out = {}
    f = W.unit(R.T @ dv); t = W.unit(R.T @ dom); up = W.unit(R.T @ g["motion_up_world"])
    if np.linalg.norm(f) > 0:
        out[f"force_K{K}"] = np.r_[f, 0, 0, 0]
    if np.linalg.norm(t) > 0:
        out[f"torque_K{K}"] = np.r_[0, 0, 0, t]
    out["support"] = np.r_[up, 0, 0, 0]
    return out


def finger_changes(p_pre, p_post):
    """Per finger: stable / slid / appeared / disappeared / absent, from patch centroids."""
    out = {}
    for part in range(len(W.PARTS)):
        a = [p for p in p_pre if p["part"] == part]; b = [p for p in p_post if p["part"] == part]
        if not a and not b:
            out[W.PARTS[part]] = "absent"
        elif a and not b:
            out[W.PARTS[part]] = "disappeared"
        elif b and not a:
            out[W.PARTS[part]] = "appeared"
        else:
            ca = np.mean([p["centroid"] for p in a], 0); cb = np.mean([p["centroid"] for p in b], 0)
            out[W.PARTS[part]] = "slid" if np.linalg.norm(ca - cb) > SLIDE_M else "stable"
    return out


def analyze_event(args):
    ds, ev, sets = args
    try:
        g = load_geometry(ds, ev["event_id"])
        mesh = mesh_of(ds, ev)
        o, l = mesh.centroid, mesh.length
        rows, profiles = [], {}
        for st in sets:
            p_pre = W.build_patches(geom_dict(g, "pre"), mesh.adj, st["thr"], st["merge"])
            p_post = W.build_patches(geom_dict(g, "post"), mesh.adj, st["thr"], st["merge"])
            q_pre, s_pre = L.capacity(p_pre, o, l, st["mu"], U_SHARED, st["budget"]); q_post, s_post = L.capacity(p_post, o, l, st["mu"], U_SHARED, st["budget"])
            row = dict(dataset=ds, event_id=ev["event_id"], setting=st["name"], thr=st["thr"], merge=st["merge"], budget=st["budget"], mu=st["mu"],
                       n_contact_pre=int((g["pre_dist"] < st["thr"]).sum()), n_contact_post=int((g["post_dist"] < st["thr"]).sum()),
                       n_patches_pre=len(p_pre), n_patches_post=len(p_post),
                       n_fingers_pre=len({p["part"] for p in p_pre}), n_fingers_post=len({p["part"] for p in p_post}),
                       fingers_pre="+".join(sorted({W.PARTS[p["part"]] for p in p_pre})), fingers_post="+".join(sorted({W.PARTS[p["part"]] for p in p_post})),
                       length=l)
            row.update(L.profile_metrics(q_pre, q_post))                     # support-function capacity (primary)
            row.update(L.profile_metrics(s_pre, s_post, prefix="strict_"))    # exact-direction capacity (strict)
            fc = finger_changes(p_pre, p_post)
            row.update({f"fc_{k}": v for k, v in fc.items()})
            row["n_slid"] = sum(v == "slid" for v in fc.values()); row["n_appeared"] = sum(v == "appeared" for v in fc.values())
            row["n_disappeared"] = sum(v == "disappeared" for v in fc.values()); row["n_stable"] = sum(v == "stable" for v in fc.values())
            # Metric B: future-motion directions
            for K in W.K_FUTURE:
                if f"motion_dv_K{K}" not in g:
                    continue
                dirs_post = motion_directions(g, "post", K)
                names = [k for k in dirs_post]
                U = np.stack([dirs_post[k] for k in names])
                h_post, a_post = L.capacity(p_post, o, l, st["mu"], U, st["budget"])
                p_preT = W.build_patches(geom_dict(g, "pre", "post"), mesh.adj, st["thr"], st["merge"])
                h_preT, a_preT = L.capacity(p_preT, o, l, st["mu"], U, st["budget"])
                for i, k in enumerate(names):
                    row[f"B_{k}_pre"] = float(h_preT[i]); row[f"B_{k}_post"] = float(h_post[i])
                    row[f"Bs_{k}_pre"] = float(a_preT[i]); row[f"Bs_{k}_post"] = float(a_post[i])
                # both grasps transported to s + K
                fk = f"s{K}"
                if f"motion_R_{fk}" in g and f"tr_pre_{fk}_pos" in g:
                    dirs_k = motion_directions(g, fk, K)
                    Uk = np.stack([dirs_k[k] for k in names])
                    h_pre_k, a_pre_k = L.capacity(W.build_patches(geom_dict(g, "pre", fk), mesh.adj, st["thr"], st["merge"]), o, l, st["mu"], Uk, st["budget"])
                    h_post_k, a_post_k = L.capacity(W.build_patches(geom_dict(g, "post", fk), mesh.adj, st["thr"], st["merge"]), o, l, st["mu"], Uk, st["budget"])
                    for i, k in enumerate(names):
                        row[f"B_{k}_pre_at{fk}"] = float(h_pre_k[i]); row[f"B_{k}_post_at{fk}"] = float(h_post_k[i])
                        row[f"Bs_{k}_pre_at{fk}"] = float(a_pre_k[i]); row[f"Bs_{k}_post_at{fk}"] = float(a_post_k[i])
                row[f"motion_dv_norm_K{K}"] = float(np.linalg.norm(g[f"motion_dv_K{K}"])); row[f"motion_dom_norm_K{K}"] = float(np.linalg.norm(g[f"motion_dom_K{K}"]))
                row[f"motion_disp_norm_K{K}"] = float(np.linalg.norm(g[f"motion_disp_K{K}"]))
                if f"motion_darti_K{K}" in g:
                    row[f"motion_darti_K{K}"] = float(g[f"motion_darti_K{K}"])
            rows.append(row); profiles[st["name"]] = (q_pre, q_post, s_pre, s_post)
        return dict(event_id=ev["event_id"], rows=rows, profiles=profiles)
    except Exception as ex:  # noqa: BLE001
        return dict(event_id=ev["event_id"], error=f"{type(ex).__name__}: {ex}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=W.DATASETS)
    ap.add_argument("--workers", type=int, default=32); ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--max-transient", type=int, default=150, help="transient events analysed (diagnostic only)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(); ds = a.dataset; t_start = time.time()
    S = W.load_selected(ds) if a.split == "test" else pd.read_csv(W.ds_out(ds) / "events_selected_train.csv", dtype={"event_id": str, "mesh_id": str})
    if a.split == "test":
        rng = np.random.default_rng(0)
        tr = S[S.cls == "transient"]
        drop = tr.sample(max(0, len(tr) - a.max_transient), random_state=0).index if len(tr) > a.max_transient else []
        S = S.drop(index=drop).reset_index(drop=True)
    have = {p.stem for p in (W.ds_out(ds) / "geometry").glob("*.npz")}
    S = S[S.event_id.isin(have)].reset_index(drop=True)
    if a.limit:
        S = S.head(a.limit)
    sets = settings() if a.split == "test" else [settings()[0]]
    log.info("%s (%s): %d events x %d settings, %d workers", ds, a.split, len(S), len(sets), a.workers)
    jobs = [(ds, r, sets) for r in S.to_dict("records")]
    rows, profiles, errors = [], {st["name"]: {} for st in sets}, []
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(analyze_event, jobs, chunksize=2)):
            if "error" in res:
                errors.append(res); continue
            rows.extend(res["rows"])
            for name, prof in res["profiles"].items():
                profiles[name][res["event_id"]] = prof
            if (i + 1) % 200 == 0:
                log.info("  %d / %d events (%.0fs)", i + 1, len(S), time.time() - t_start)
    M = pd.DataFrame(rows).merge(S, on=["dataset", "event_id"], how="left")
    suffix = "" if a.split == "test" else "_train"
    M.to_csv(W.ds_out(ds) / f"events_metrics{suffix}.csv", index=False)
    (W.ds_out(ds) / "capacity").mkdir(exist_ok=True)
    for name, d in profiles.items():
        ids = sorted(d)
        stack = lambda j: np.stack([d[i][j] for i in ids]) if ids else np.zeros((0, len(U_NAMES)))
        np.savez_compressed(W.ds_out(ds) / "capacity" / f"{name}{suffix}.npz", event_id=np.array(ids), q_pre=stack(0), q_post=stack(1), strict_pre=stack(2), strict_post=stack(3),
                            direction=np.array(U_NAMES), U=U_SHARED)
    if a.split == "test":
        d = profiles["primary"]; ids = sorted(d)
        long = pd.DataFrame(dict(event_id=np.repeat(ids, len(U_NAMES)), direction=np.tile(U_NAMES, len(ids)),
                                 h_pre=np.concatenate([d[i][0] for i in ids]), h_post=np.concatenate([d[i][1] for i in ids]),
                                 alpha_strict_pre=np.concatenate([d[i][2] for i in ids]), alpha_strict_post=np.concatenate([d[i][3] for i in ids])))
        long = long.merge(S[["event_id", "cls", "take_key", "primary", "secondary"]], on="event_id")
        long.to_csv(W.ds_out(ds) / "directional_capacity.csv", index=False, float_format="%.5g")
        P = M[M.setting == "primary"]
        fc = P[["event_id", "cls", "take_key", "dC", "Q_pre", "Q_post", "rel_change_Q", "R_pre_to_post"] + [f"fc_{p}" for p in W.PARTS] + ["n_slid", "n_appeared", "n_disappeared", "n_stable", "fingers_pre", "fingers_post"]]
        fc.to_csv(W.ds_out(ds) / "finger_transitions.csv", index=False)
    W.write_json(W.ds_out(ds) / "logs" / f"wrench_analysis{suffix}_errors.json", errors)
    log.info("done %s (%s): %d rows, %d errors, %.0fs", ds, a.split, len(M), len(errors), time.time() - t_start)
    if a.split == "test" and len(M):
        P = M[M.setting == "primary"]
        pd.set_option("display.width", 250)
        print(P.groupby("cls")[["dC", "Q_pre", "Q_post", "rel_change_Q", "R_pre_to_post", "cosine", "n_patches_pre", "n_patches_post"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
