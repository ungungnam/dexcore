#!/usr/bin/env python
"""Step 1: contact geometry of every selected test event.   python build_geometry.py --dataset taco
For each event: contact vertices (hand distance < 2 cm, thresholds applied later) at the pre and post
frames with positions, outward mesh normals, finger labels (nearest MANO vertex -> skinning part) and
the side the hand is on; the same vertex sets transported to the post frame and to s + 8 / s + 16
(rigid objects: identity; ARCTIC: articulated top part); the hand vertices in the object frame (for
figures); the object's world motion around the event (velocity / angular velocity per frame, the
window changes over K = 8 / 16, the object orientation at the pre / post / window frames, the
gravity direction). Writes <ds>/geometry/<event_id>.npz, <ds>/events_selected.csv and
<ds>/sanity/geometry_sanity.json (recomputed-vs-cached contact minima, label coverage, normal sides).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import wc_common as W

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("geometry")
_LAB = None
_LAYERS = {}


def labels():
    global _LAB
    if _LAB is None:
        _LAB = W.mano_labels()
    return _LAB


# ------------------------------------------------------------------------------ per-take loaders
def taco_take(sequence_id):
    """Raw TACO take: world hand skins (both sides), world object poses (tool, target), gen3 contact
    distances (T, n_tool + n_target), n_tool."""
    import sys
    sys.path.insert(0, str(W.REPO))
    from src.analysis.action_structure import contact as C
    from src.analysis.loaders import taco
    from src import geometry as G
    root = taco.default_root()
    ref = [r for r in taco.index(root) if r.sequence_id == sequence_id][0]
    traj = taco.load(ref, root=root, with_hands=True)
    for s in ("left", "right"):
        if s not in _LAYERS:
            _LAYERS[s] = C.mano_layer(s)
    skins = {s: C.hand_geometry(traj.hands[s], _LAYERS[s])[0] for s in ("left", "right") if s in traj.hands}
    files = pd.read_csv(W.TACO_GEN3 / "sequence_index.csv").set_index("sequence_id")["file"]
    wi = pd.read_csv(W.TACO_WINDOW_INDEX).drop_duplicates("sequence_id").set_index("sequence_id")
    with np.load(W.TACO_GEN3 / "sequences" / files[sequence_id]) as z:
        dist = {"left": z["contact_left"], "right": z["contact_right"]}
    poses = {"tool": (G.wxyz_to_R(traj.tool.quat), traj.tool.pos), "target": (G.wxyz_to_R(traj.target.quat), traj.target.pos)}
    return dict(skins=skins, dist=dist, n_tool=int(wi.loc[sequence_id].n_tool), poses=poses, n_frames=min(len(traj.tool.pos), len(dist["left"])))


def arctic_take(sequence_id, category):
    stem, subj = sequence_id.split("/")
    h = np.load(W.ARCTIC_PROC / category / subj / f"{stem}_processed_hand_features.npy", allow_pickle=True).item()
    o = np.load(W.ARCTIC_PROC / category / subj / f"{stem}_processed_obj_features.npy", allow_pickle=True).item()
    return dict(skins={"left": h["left_hand_verts"].astype(np.float64), "right": h["right_hand_verts"].astype(np.float64)},
                dist={s: o["contact_dict"][s]["dist"] for s in ("left", "right")}, index={s: o["contact_dict"][s]["index"] for s in ("left", "right")},
                state=o["obj_world_state"].astype(np.float64), n_frames=len(o["obj_world_state"]))


# ------------------------------------------------------------------------------ per-event geometry
def event_frames(ev):
    fr = {"pre": ev.pre_frame, "post": ev.post_frame}
    for K in W.K_FUTURE:
        if ev[f"valid_K{K}"]:
            fr[f"s{K}"] = ev.start + K
    return fr


def process_take(args):
    ds, sequence_id, events = args
    lab = labels()
    side_of = {"L": "left", "R": "right"}
    out = []
    try:
        if ds == "taco":
            tk = taco_take(sequence_id)
        else:
            tk = arctic_take(sequence_id, events[0]["category"])
    except Exception as ex:  # noqa: BLE001
        return [dict(event_id=e["event_id"], error=f"{type(ex).__name__}: {ex}") for e in events]
    meshes = {}
    for ev in events:
        ev = pd.Series(ev)
        side = side_of[ev.hand]; t0 = int(ev.t0)
        fr = event_frames(ev)
        rec = dict(event_id=ev.event_id, frames={k: int(v) for k, v in fr.items()})
        try:
            if ds == "taco":
                key = ev.mesh_id
                if key not in meshes:
                    meshes[key] = W.taco_mesh(key)
                mesh = meshes[key]
                n_tool = tk["n_tool"]
                cols = slice(0, n_tool) if ev.role == "tool" else slice(n_tool, None)
                R_all, p_all = tk["poses"][ev.role]
                n_fr = tk["n_frames"]
                def geom(f):
                    tf = t0 + f
                    d = tk["dist"][side][tf][cols]
                    hand_obj = (tk["skins"][side][tf] - p_all[tf]) @ R_all[tf]                   # object frame
                    g = W.contact_geometry(d, None, mesh.verts, mesh.normals, hand_obj, lab)
                    return g, hand_obj, mesh.verts, mesh.normals
                R_w, p_w = R_all[:n_fr], p_all[:n_fr]
                arti = None
            else:
                key = ev.category
                if key not in meshes:
                    meshes[key] = W.arctic_mesh(key)
                mesh = meshes[key]
                st = tk["state"]; n_fr = tk["n_frames"]
                def geom(f):
                    tf = t0 + f
                    d = tk["dist"][side][tf]; idx = tk["index"][side][tf]
                    hand_obj = W.arctic_rigid(tk["skins"][side][tf], st[tf])
                    v = mesh.verts_at(st[tf, 0]); n = mesh.normals_at(st[tf, 0])
                    g = W.contact_geometry(d, idx, v, n, hand_obj, lab)
                    return g, hand_obj, v, n
                R_w = Rotation.from_rotvec(st[:n_fr, 1:4]).as_matrix(); p_w = st[:n_fr, 4:7]
                arti = st[:n_fr, 0]
            # contact geometry at pre and post
            for k in ("pre", "post"):
                g, hand_obj, v, n = geom(fr[k])
                rec[k] = g; rec[f"hand_{k}"] = hand_obj.astype(np.float32)
                # sanity: recomputed minimum vs cached minimum (TACO: independent KD-tree; ARCTIC: stored)
                tf = t0 + fr[k]
                d_all = tk["dist"][side][tf][cols] if ds == "taco" else tk["dist"][side][tf]
                rec[f"min_d_cached_{k}"] = float(d_all.min())
                rec[f"min_d_recomputed_{k}"] = float(cKDTree(hand_obj).query(v)[0].min())
            # transport: pre ids and post ids at the other frames (positions / normals in the rigid frame)
            for k in ("pre", "post"):
                ids = rec[k]["ids"]
                tr = {}
                for name, f in fr.items():
                    if ds == "taco":
                        tr[name] = (mesh.verts[ids].astype(np.float32), mesh.normals[ids].astype(np.float32))
                    else:
                        a = arti[t0 + f]
                        tr[name] = (mesh.verts_at(a)[ids].astype(np.float32), mesh.normals_at(a)[ids].astype(np.float32))
                rec[f"transport_{k}"] = tr
            # object motion (world frame) around the event
            v_lin, om = W.world_rates(R_w, p_w)
            s_tf = t0 + ev.start
            motion = dict(R_pre=R_w[t0 + fr["pre"]], R_post=R_w[t0 + fr["post"]], up_world=W.UP_WORLD)
            for K in W.K_FUTURE:
                if f"s{K}" in fr:
                    a, b = s_tf, min(s_tf + K, n_fr - 1)
                    motion[f"dv_K{K}"] = v_lin[b] - v_lin[a]; motion[f"dom_K{K}"] = om[b] - om[a]
                    motion[f"vmean_K{K}"] = v_lin[a:b + 1].mean(0); motion[f"ommean_K{K}"] = om[a:b + 1].mean(0)
                    motion[f"disp_K{K}"] = p_w[b] - p_w[a]
                    motion[f"R_s{K}"] = R_w[b]
                    if arti is not None:
                        motion[f"darti_K{K}"] = float(arti[b] - arti[a])
            motion["speed_pre"] = float(np.linalg.norm(v_lin[t0 + fr["pre"]])); motion["speed_post"] = float(np.linalg.norm(v_lin[t0 + fr["post"]]))
            rec["motion"] = motion
            rec["mesh"] = dict(name=mesh.name, length=mesh.length, extent=mesh.extent, centroid=mesh.centroid, frac_outward=mesh.frac_outward, watertight=mesh.watertight, volume=mesh.volume, n_verts=len(mesh.verts))
            rec["arti_pre"] = None if arti is None else float(arti[t0 + fr["pre"]]); rec["arti_post"] = None if arti is None else float(arti[t0 + fr["post"]])
        except Exception as ex:  # noqa: BLE001
            rec["error"] = f"{type(ex).__name__}: {ex}"
        out.append(rec)
    return out


def save_event(ds, rec):
    p = W.ds_out(ds) / "geometry" / f"{rec['event_id']}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    flat = {"frames": json.dumps(rec["frames"]), "event_id": rec["event_id"]}
    for k in ("pre", "post"):
        for kk, vv in rec[k].items():
            flat[f"{k}_{kk}"] = vv
        flat[f"hand_{k}"] = rec[f"hand_{k}"]
        for name, (pos, nrm) in rec[f"transport_{k}"].items():
            flat[f"tr_{k}_{name}_pos"] = pos; flat[f"tr_{k}_{name}_nrm"] = nrm
        flat[f"min_d_cached_{k}"] = rec[f"min_d_cached_{k}"]; flat[f"min_d_recomputed_{k}"] = rec[f"min_d_recomputed_{k}"]
    for k, v in rec["motion"].items():
        flat[f"motion_{k}"] = v
    flat["mesh"] = json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rec["mesh"].items()})
    flat["arti_pre"] = np.nan if rec["arti_pre"] is None else rec["arti_pre"]; flat["arti_post"] = np.nan if rec["arti_post"] is None else rec["arti_post"]
    np.savez(p, **flat)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=W.DATASETS)
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int, default=None, help="takes (smoke test)")
    ap.add_argument("--split", default="test", choices=["test", "train"], help="train: persistent spatial events only (thresholds)")
    a = ap.parse_args(); ds = a.dataset; t_start = time.time()
    S = W.select_events(ds) if a.split == "test" else W.select_events(ds, "train", ["persistent_spatial"])
    log.info("%s: %d test events (%s)", ds, len(S), S.cls.value_counts().to_dict())
    takes = [(ds, sid, g.to_dict("records")) for sid, g in S.groupby("sequence_id")]
    if a.limit:
        takes = takes[:a.limit]
    log.info("%s: %d takes -> %d workers", ds, len(takes), a.workers)
    san = dict(dataset=ds, n_events=0, n_errors=0, errors=[], min_d_abs_diff=[], n_contact_pre=[], n_contact_post=[], frac_hand_outward=[], label_hist=np.zeros(len(W.PARTS), int).tolist())
    lab_hist = np.zeros(len(W.PARTS), int)
    with Pool(a.workers) as pool:
        for recs in pool.imap_unordered(process_take, takes):
            for rec in recs:
                if "error" in rec:
                    san["n_errors"] += 1; san["errors"].append(dict(event_id=rec["event_id"], error=rec["error"])); continue
                save_event(ds, rec); san["n_events"] += 1
                for k in ("pre", "post"):
                    san["min_d_abs_diff"].append(abs(rec[f"min_d_cached_{k}"] - rec[f"min_d_recomputed_{k}"]))
                    g = rec[k]; sel = g["dist"] < W.CONTACT_THR["primary"]
                    san[f"n_contact_{k}"].append(int(sel.sum()))
                    if sel.any():
                        san["frac_hand_outward"].append(float(g["outward"][sel].mean())); lab_hist += np.bincount(g["label"][sel], minlength=len(W.PARTS))
            if san["n_events"] % 100 < len(recs):
                log.info("  %d events done (%.0fs)", san["n_events"], time.time() - t_start)
    san["label_hist"] = dict(zip(W.PARTS, lab_hist.tolist()))
    san["min_d_abs_diff_max"] = float(np.max(san["min_d_abs_diff"])) if san["min_d_abs_diff"] else None
    san["min_d_abs_diff_median"] = float(np.median(san["min_d_abs_diff"])) if san["min_d_abs_diff"] else None
    san["frac_hand_outward_mean"] = float(np.mean(san["frac_hand_outward"])) if san["frac_hand_outward"] else None
    san["n_contact_pre_median"] = float(np.median(san["n_contact_pre"])) if san["n_contact_pre"] else None
    san["n_contact_post_median"] = float(np.median(san["n_contact_post"])) if san["n_contact_post"] else None
    for k in ("min_d_abs_diff", "n_contact_pre", "n_contact_post", "frac_hand_outward"):
        del san[k]
    W.write_json(W.ds_out(ds) / "sanity" / ("geometry_sanity.json" if a.split == "test" else f"geometry_sanity_{a.split}.json"), san)
    log.info("done %s: %d events, %d errors, min-d agreement max %.2e, hand-outward %.3f, labels %s, %.0fs", ds, san["n_events"], san["n_errors"],
             san["min_d_abs_diff_max"] or -1, san["frac_hand_outward_mean"] or -1, san["label_hist"], time.time() - t_start)


if __name__ == "__main__":
    main()
