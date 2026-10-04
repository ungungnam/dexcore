#!/usr/bin/env python
"""Step 1: per-frame feature cache for every train / val / test sequence of the fixed split.
    python build_features.py --dataset taco --workers 48

For every sequence and every stored frame t in [-8, 72) (take frames clamped at the boundaries, as the hand cache
does) the contacted hand's 778-vertex MANO skin is expressed in the object frame with the loaders of the wrench
report (TACO: raw skins + gen3 distances + tool / target poses; ARCTIC: BimArt skins + contact dict + rigid frame with
the articulated top part), the contact vertices (<= 2 cm) are labelled by the nearest MANO vertex's semantic part and
the per-part descriptors of the representation ladder are computed:
  n_hard_k  vertices within 1 cm                   m_k  area-weighted soft mass  sum exp(-d/2cm) A_v / A_total
  p_k       hard-contact centroid (object frame, relative to the mesh centroid, / l)
  nrm_k     mean toward-hand normal (unit)         s_k  RMS radius of the hard-contact vertices / l
  c_k       number of contact patches (wrench report's patches: 1 cm / merge 1 cm)
  hand      42-D coarse hand: wrist / l, palm frame (e1 toward the fingers, e2 toward the thumb side, e3 dorsal),
            5 fingertips / l, 5 MCP -> tip unit directions
and, for t in [0, 64), the 76-D support-function wrench profile q_t (and the strict LP profile) of the patches with
the wrench report's primary setting. The 512-D canonical contact vector is re-evaluated on the padded frames with the
same operator (TACO) / taken from the dynamic-contact frame cache (ARCTIC) and checked against sequences.npz.
Writes <ds>/cache/features.npz and <ds>/cache/features_build.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sv_common as S
import wc_common as W
import wrench_lp as L
import build_geometry as BG  # noqa: E402  (wrench_counterfactual: taco_take / arctic_take)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("features")
U_SHARED, U_NAMES = L.direction_set()
ARCTIC_FRAMES = Path("/result/uhnam/dexcore/arctic/30_dynamic_contact/cache/frames_full")
_MESH, _JREG, _OPER, _GROUP = {}, {}, {}, {}
_LAB = None
_BE = None


def labels():
    global _LAB
    if _LAB is None:
        _LAB = W.mano_labels()
    return _LAB


def j_regressor(side):
    if side not in _JREG:
        for name, val in (("bool", bool), ("int", int), ("float", float), ("complex", complex), ("object", object), ("str", str), ("unicode", str)):
            if not hasattr(np, name):
                setattr(np, name, val)
        m = pickle.load(open(W.MANO_ROOT / f"MANO_{side.upper()}.pkl", "rb"), encoding="latin1")
        J = m["J_regressor"]; J = J.toarray() if hasattr(J, "toarray") else np.asarray(J)
        _JREG[side] = np.asarray(J, float)
    return _JREG[side]


def mesh_with_areas(ds, key):
    if key not in _MESH:
        import trimesh
        mesh = W.taco_mesh(key) if ds == "taco" else W.arctic_mesh(key)
        fa = trimesh.Trimesh(mesh.verts, mesh.faces, process=False).area_faces
        va = np.zeros(len(mesh.verts))
        for j in range(3):
            np.add.at(va, mesh.faces[:, j], fa / 3.0)
        mesh.vertex_area = va; mesh.area_total = float(va.sum())
        _MESH[key] = mesh
    return _MESH[key]


def unit_rows(x, eps=1e-9):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.where(n > eps, x / np.maximum(n, 1e-12), 0.0)


def hand_coarse(skin, Jreg, side, centroid, l):
    """42-D coarse hand state in the object frame (see the module docstring)."""
    J = Jreg @ skin
    wrist = J[0]; mcp = J[S.MCP]; tips = skin[S.TIPS]
    e1 = W.unit(J[[1, 4, 10, 7]].mean(0) - wrist)                          # toward the fingers (index, middle, ring, little MCPs)
    e2 = J[1] - J[7]; e2 = W.unit(e2 - (e2 @ e1) * e1)                     # toward the thumb side (index MCP - little MCP)
    e3 = np.cross(e1, e2) if side == "right" else np.cross(e2, e1)          # dorsal for both hands
    return np.concatenate([(wrist - centroid) / l, e1, e2, e3, ((tips - centroid) / l).ravel(), unit_rows(tips - mcp).ravel()]).astype(np.float32)


def part_features(g, mesh, thr_hard, merge):
    nh = np.zeros(S.N_PARTS, np.int32); m = np.zeros(S.N_PARTS); p = np.zeros((S.N_PARTS, 3)); n = np.zeros((S.N_PARTS, 3))
    s = np.zeros(S.N_PARTS); c = np.zeros(S.N_PARTS, np.int32); patches = []
    if len(g["ids"]):
        lab, d, ids = g["label"], g["dist"], g["ids"]
        w = np.exp(-d / 0.02) * mesh.vertex_area[ids] / mesh.area_total
        nrm_all = np.where(g["outward"][:, None], g["nrm"], -g["nrm"])       # toward the hand
        for k in range(S.N_PARTS):
            mk = lab == k
            if not mk.any():
                continue
            m[k] = w[mk].sum()
            hk = mk & (d < thr_hard)
            nh[k] = int(hk.sum())
            if nh[k] >= 1:
                pos = g["pos"][hk]; cen = pos.mean(0)
                p[k] = (cen - mesh.centroid) / mesh.length
                nm = nrm_all[hk].mean(0)
                if np.linalg.norm(nm) < 0.2:                                   # cancelling normals (thin plate / edge): nearest vertex to the centroid
                    nm = nrm_all[hk][np.argmin(np.linalg.norm(pos - cen, axis=1))]
                n[k] = W.unit(nm)
                s[k] = np.sqrt(((pos - cen) ** 2).sum(1).mean()) / mesh.length
        patches = W.build_patches(g, mesh.adj, thr_hard, merge)
        for pt in patches:
            c[pt["part"]] += 1
    return dict(n_hard=nh, m=m, p=p, nrm=n, s=s, c=c), patches


def wrench_profile(patches, mesh, strict):
    if not patches:
        return np.zeros(S.N_DIR, np.float32), np.zeros(S.N_DIR, np.float32)
    h, a = L.capacity(patches, mesh.centroid, mesh.length, S.MU, U_SHARED, S.BUDGET, strict=strict)
    return h.astype(np.float32), (a if strict else np.full(S.N_DIR, np.nan)).astype(np.float32)


# ------------------------------------------------------------------------------ padded canonical contact
def taco_operator(cat, mesh_id):
    global _BE
    if _BE is None:
        os.environ["DC_DATASET"] = "taco"
        dyn = str(S.REPO / "scripts/research/canonical_contact/dyn")
        if dyn not in sys.path:
            sys.path.insert(0, dyn)
        import dc_common as DC
        from src.analysis.canonical import backends as BR
        _BE = BR.load("normalized", str(DC.BACKEND_DIR))
    key = (cat, mesh_id)
    if key not in _OPER:
        from build_full_cache import weight_matrix
        _OPER[key] = weight_matrix(_BE, cat, mesh_id)
    return _OPER[key]


def arctic_group_frames(cat, hand):
    key = (cat, hand)
    if key not in _GROUP:
        z = np.load(ARCTIC_FRAMES / f"{cat}__obj__{hand}.npz", allow_pickle=True)
        sid = z["sequence_id"].astype(str); fr = z["frame"].astype(int)
        _GROUP[key] = (dict(zip(zip(sid, fr), range(len(sid)))), z["X"])
    return _GROUP[key]


# ------------------------------------------------------------------------------ per take / per sequence
def process_sequence(ds, tk, r, lab, strict):
    n, t0, hand, role = int(r["n"]), int(r["t0"]), r["hand"], r["role"]
    side = {"L": "left", "R": "right"}[hand]
    n_fr = tk["n_frames"]
    tt = t0 + np.arange(-S.PAD, S.T + S.PAD); fr = np.clip(tt, 0, n_fr - 1); clamped = tt != fr
    if ds == "taco":
        mesh = mesh_with_areas(ds, r["mesh_id"]); n_tool = tk["n_tool"]
        cols = slice(0, n_tool) if role == "tool" else slice(n_tool, None)
        R_all, p_all = tk["poses"][role]

        def frame(tf):
            d = tk["dist"][side][tf][cols]
            hand_obj = (tk["skins"][side][tf] - p_all[tf]) @ R_all[tf]
            return d, None, hand_obj, mesh.verts, mesh.normals
    else:
        mesh = mesh_with_areas(ds, r["category"]); st = tk["state"]

        def frame(tf):
            d = tk["dist"][side][tf]; idx = tk["index"][side][tf]
            hand_obj = W.arctic_rigid(tk["skins"][side][tf], st[tf])
            return d, idx, hand_obj, mesh.verts_at(st[tf, 0]), mesh.normals_at(st[tf, 0])
    Jreg = j_regressor(side)
    o = dict(n_hard=np.zeros((S.NF, 6), np.int16), m=np.zeros((S.NF, 6), np.float32), p=np.zeros((S.NF, 6, 3), np.float32),
             nrm=np.zeros((S.NF, 6, 3), np.float32), s=np.zeros((S.NF, 6), np.float32), c=np.zeros((S.NF, 6), np.int8),
             hand=np.zeros((S.NF, S.HAND_COARSE_DIM), np.float32), q=np.zeros((S.T, S.N_DIR), np.float32), q_strict=np.zeros((S.T, S.N_DIR), np.float32),
             min_d=np.zeros(S.NF, np.float32), min_d_recomputed=np.full(S.NF, np.nan, np.float32), n_patches=np.zeros(S.NF, np.int16),
             n_contact=np.zeros(S.NF, np.int16), frac_outward=np.full(S.NF, np.nan, np.float32))
    palm_dot = []
    for i, tf in enumerate(fr):
        d, idx, hand_obj, v, nrm = frame(int(tf))
        g = W.contact_geometry(d, idx, v, nrm, hand_obj, lab)               # vertices <= 2 cm
        feats, patches = part_features(g, mesh, S.CONTACT_THR, S.MERGE_R)
        for k, val in feats.items():
            o[k][i] = val
        o["hand"][i] = hand_coarse(hand_obj, Jreg, side, mesh.centroid, mesh.length)
        o["min_d"][i] = float(d.min()); o["n_patches"][i] = len(patches); o["n_contact"][i] = int((g["dist"] < S.CONTACT_THR).sum())
        if len(g["ids"]):
            o["frac_outward"][i] = float(g["outward"].mean())
        if i % 8 == 0:
            o["min_d_recomputed"][i] = float(cKDTree(hand_obj).query(v)[0].min())
        if feats["n_hard"][0] >= S.N_MIN_VERTS:                               # toward-hand palm contact normal vs the dorsal direction e3 (expected > 0)
            palm_dot.append(float(feats["nrm"][0] @ o["hand"][i][9:12]))
        t = i - S.PAD
        if 0 <= t < S.T:
            o["q"][t], o["q_strict"][t] = wrench_profile(patches, mesh, strict)
    # canonical contact on the padded frames
    if ds == "taco":
        Wop = taco_operator(r["category"], r["mesh_id"])
        Dm = np.stack([tk["dist"][side][int(tf)][cols] for tf in fr])
        C_pad = (np.exp(-Dm / 0.02).astype(np.float32) @ Wop.T).astype(np.float32); found = np.ones(S.NF, bool)
    else:
        lut, X = arctic_group_frames(r["category"], hand)
        C_pad = np.full((S.NF, 512), np.nan, np.float32); found = np.zeros(S.NF, bool)
        for i, tf in enumerate(fr):
            j = lut.get((r["sequence_id"], int(tf)))
            if j is not None:
                C_pad[i] = X[j].astype(np.float32); found[i] = True
    o.update(n=n, C_pad=C_pad, C_found=found, clamped=clamped, length=float(mesh.length), centroid=mesh.centroid.astype(np.float32),
             palm_dot=float(np.mean(palm_dot)) if palm_dot else np.nan, n_palm_frames=len(palm_dot))
    return o


def process_take(args):
    ds, sid, rows, strict = args
    try:
        import torch
        torch.set_num_threads(1)
        tk = BG.taco_take(sid) if ds == "taco" else BG.arctic_take(sid, rows[0]["category"])
    except Exception as ex:  # noqa: BLE001
        return [dict(n=int(r["n"]), error=f"{type(ex).__name__}: {ex}") for r in rows]
    lab = labels(); out = []
    for r in rows:
        try:
            out.append(process_sequence(ds, tk, r, lab, strict))
        except Exception as ex:  # noqa: BLE001
            out.append(dict(n=int(r["n"]), error=f"{type(ex).__name__}: {ex}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None, help="takes (for tests)"); ap.add_argument("--no-strict", action="store_true")
    a = ap.parse_args()
    ds = a.dataset; t_start = time.time()
    D = S.HP.load_raw(ds); meta = D["meta"]; man = D["man"]; C_seq = D["C"]
    use = np.sort(np.unique(np.concatenate([man["train"], man["val"], man["test"]])))
    tasks = []
    for sid, idx in meta.iloc[use].groupby("sequence_id").indices.items():
        rows = [dict(n=int(use[i]), sequence_id=sid, t0=int(meta.iloc[use[i]].t0), hand=meta.iloc[use[i]].hand, role=meta.iloc[use[i]].role,
                     mesh_id=str(meta.iloc[use[i]].mesh_id), category=meta.iloc[use[i]].category) for i in idx]
        tasks.append((ds, sid, rows, not a.no_strict))
    if a.limit:
        tasks = tasks[:a.limit]
    N = len(meta)
    F = dict(n_hard=np.zeros((N, S.NF, 6), np.int16), m=np.zeros((N, S.NF, 6), np.float32), p=np.zeros((N, S.NF, 6, 3), np.float32),
             nrm=np.zeros((N, S.NF, 6, 3), np.float32), s=np.zeros((N, S.NF, 6), np.float32), c=np.zeros((N, S.NF, 6), np.int8),
             hand=np.zeros((N, S.NF, S.HAND_COARSE_DIM), np.float32), q=np.zeros((N, S.T, S.N_DIR), np.float32), q_strict=np.zeros((N, S.T, S.N_DIR), np.float32),
             min_d=np.zeros((N, S.NF), np.float32), min_d_recomputed=np.full((N, S.NF), np.nan, np.float32), n_patches=np.zeros((N, S.NF), np.int16),
             n_contact=np.zeros((N, S.NF), np.int16), frac_outward=np.full((N, S.NF), np.nan, np.float32), C_pad=np.zeros((N, S.NF, 512), np.float16),
             C_found=np.zeros((N, S.NF), bool), clamped=np.zeros((N, S.NF), bool), length=np.zeros(N, np.float32), centroid=np.zeros((N, 3), np.float32),
             palm_dot=np.full(N, np.nan, np.float32), n_palm_frames=np.zeros(N, np.int32), included=np.zeros(N, bool), C_check=np.full(N, np.nan, np.float32))
    errors = []; done = 0
    log.info("%s: %d sequences in %d takes, %d workers, strict LP %s", ds, len(use), len(tasks), a.workers, not a.no_strict)
    with mp.Pool(a.workers, maxtasksperchild=16) as pool:
        for res in pool.imap_unordered(process_take, tasks, chunksize=1):
            for o in res:
                n = o["n"]
                if "error" in o:
                    errors.append(dict(example=n, error=o["error"])); continue
                for k in F:
                    if k in o:
                        F[k][n] = o[k]
                # padded C: missing frames (ARCTIC) -> the nearest stored sequence frame; check against sequences.npz
                Cp = o["C_pad"].copy()
                if not o["C_found"].all():
                    seq = C_seq[n].astype(np.float32)
                    for i in np.where(~o["C_found"])[0]:
                        Cp[i] = seq[int(np.clip(i - S.PAD, 0, S.T - 1))]
                F["C_check"][n] = float(np.abs(Cp[S.PAD:S.PAD + S.T] - C_seq[n].astype(np.float32)).max())
                F["C_pad"][n] = Cp.astype(np.float16); F["included"][n] = True
            done += 1
            if done % 100 == 0:
                log.info("  %d / %d takes (%.0f s)", done, len(tasks), time.time() - t_start)
    F["example"] = np.arange(N); F["U"] = U_SHARED; F["direction_names"] = np.array(U_NAMES)
    out = S.feature_path(ds); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **F)
    inc = F["included"]
    summary = dict(dataset=ds, n_sequences=int(inc.sum()), n_takes=len(tasks), n_errors=len(errors), errors=errors[:50], seconds=time.time() - t_start,
                   strict=not a.no_strict, C_check_max=float(np.nanmax(F["C_check"][inc])) if inc.any() else None,
                   C_padded_found_frac=float(F["C_found"][inc].mean()) if inc.any() else None, clamped_frac=float(F["clamped"][inc].mean()) if inc.any() else None,
                   min_d_check_max=float(np.nanmax(np.abs(F["min_d_recomputed"][inc] - F["min_d"][inc]))) if inc.any() else None,
                   palm_dot_mean=float(np.nanmean(F["palm_dot"][inc])) if inc.any() else None,
                   frac_frames_in_contact=float((F["n_contact"][inc][:, S.PAD:S.PAD + S.T] > 0).mean()) if inc.any() else None,
                   mean_active_parts=float((F["n_hard"][inc][:, S.PAD:S.PAD + S.T] >= S.N_MIN_VERTS).sum(-1).mean()) if inc.any() else None)
    S.write_json(out.parent / "features_build.json", summary)
    log.info("done: %s", json.dumps({k: v for k, v in summary.items() if k != "errors"}))
    if errors:
        log.warning("%d errors, first: %s", len(errors), errors[0])


if __name__ == "__main__":
    main()
