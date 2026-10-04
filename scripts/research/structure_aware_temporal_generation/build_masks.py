#!/usr/bin/env python
"""Step 1: the training-time privileged signal and the canonical geometry.    python build_masks.py --dataset taco --workers 48

For every train / val / test sequence and frame t in [0, 64) (take frame t0 + t, clamped at the take boundaries like every
previous cache) the contacted hand's 778-vertex MANO skin is placed in the object frame with the wrench report's loaders, EVERY
mesh vertex u gets the semantic part of its nearest MANO vertex (palm / thumb / index / middle / ring / little, the wrench
report's skinning-weight labels) and the side the hand is on (outward flag), and both are transported to the 512 canonical
points with the canonical operator W (rows normalised to 1):
    M_t(v, k) = sum_u W[v, u] 1[part_t(u) = k]                 (stored as uint8 / 255)
    sgn_t(v)  = sign(sum_u W[v, u] (2 outward_t(u) - 1))        (+1: the hand is on the outward side -> keep the outward normal)
    nh_t(v)   = normalise(sum_u W[v, u] (+-n_u))                 the toward-hand normal field (every vertex normal oriented toward
                                                                 the hand BEFORE averaging, as the exact R2 does; int8 / 127)
Per mesh the canonical geometry: W-averaged vertex positions and outward normals split into the articulated top part and the
fixed bottom part (ARCTIC; TACO objects are rigid: everything in 'bot'), the cell area of every canonical point
(alpha_v = sum_u W[v, u] / colsum_u * A_u / A_total), the mesh centroid and characteristic length l of the R2 definition.
W is verified against the cached canonical contact (exp(-d / 2 cm) @ W^T on sampled frames).
Writes <ds>/cache/masks.npz, <ds>/cache/canonical_geometry.npz, <ds>/cache/masks_build.json.
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import sat_common as S
import wc_common as W
import build_geometry as BG

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("masks")
ARCTIC_DICT = "/result/uhnam/dexcore/arctic/20_bimart_contact_probe/assets/arctic_mesh_dict.npy"
ARCTIC_PROBE = "/result/uhnam/dexcore/arctic/20_bimart_contact_probe/sequences"
TACO_BACKEND = "/result/uhnam/dexcore/taco/40_representation_study/canonical_backend"
RADIUS, SIGMA = 0.15, 0.075
_CTX = {}


# ------------------------------------------------------------------------------ meshes in the canonical operator's vertex order
def gauss_operator(V_norm, P):
    """Row-normalised gauss operator from normalised vertices to canonical points (build_full_cache / build_arctic_cache)."""
    tree = cKDTree(V_norm)
    Wop = np.zeros((len(P), len(V_norm)), np.float32)
    for k, nb in enumerate(tree.query_ball_point(P, RADIUS)):
        if not nb:
            continue
        nb = np.asarray(nb)
        w = np.exp(-0.5 * ((V_norm[nb] - P[k]) ** 2).sum(1) / SIGMA ** 2)
        Wop[k, nb] = w / w.sum()
    return Wop


def vertex_areas(verts, faces):
    fa = trimesh.Trimesh(verts, faces, process=False).area_faces
    va = np.zeros(len(verts))
    for j in range(3):
        np.add.at(va, faces[:, j], fa / 3.0)
    return va


def taco_object(mesh_id, category, P_by_cat):
    """TACO mesh in the backend's vertex order (= verts_original, asserted): W, verts, outward normals, areas, centroid, l."""
    if "be" not in _CTX:
        from src.analysis.canonical import backends as BR
        _CTX["be"] = BR.load("normalized", TACO_BACKEND)
    be = _CTX["be"]
    fit = be._cat(category)
    mesh = W.taco_mesh(mesh_id)
    assert np.abs(fit.verts[mesh_id] - mesh.verts).max() < 1e-6, "backend vertex order differs from the mesh dict"
    V, tree = fit.mapped(mesh_id)
    Wop = gauss_operator(V, fit.canonical_points)
    assert np.abs(fit.canonical_points - P_by_cat[category]).max() < 1e-6
    return dict(W=Wop, verts=mesh.verts, normals=mesh.normals, faces=mesh.faces, top=np.zeros(len(mesh.verts), bool),
                centroid=mesh.centroid, length=mesh.length, area=vertex_areas(mesh.verts, mesh.faces))


def arctic_object(category, P_by_cat):
    """ARCTIC object in the probe / cache vertex order (top part, then bottom part) with the template's outward normals
    mapped by exact vertex matching; W rebuilt from the normalised rest mesh and the stored canonical points."""
    if "arctic_dict" not in _CTX:
        _CTX["arctic_dict"] = np.load(ARCTIC_DICT, allow_pickle=True).item()
    md = _CTX["arctic_dict"]
    top, bot = md[f"{category}_top"]["verts_original"], md[f"{category}_bottom"]["verts_original"]
    verts = np.concatenate([top, bot]).astype(float)
    tm = W.arctic_mesh(category)                                        # template order, outward normals, parts
    d, perm = cKDTree(tm.verts).query(verts)
    assert d.max() < 1e-6 and len(np.unique(perm)) == len(perm), "template <-> mesh-dict vertex matching failed"
    assert (tm.parts[perm[:len(top)]] == 0).all() and (tm.parts[perm[len(top):]] == 1).all(), "top / bottom labels differ"
    inv = np.empty(len(perm), int); inv[perm] = np.arange(len(perm))
    faces = inv[tm.faces]
    c = verts.mean(0); r = np.linalg.norm(verts - c, axis=1).max()
    Wop = gauss_operator((verts - c) / r, P_by_cat[category])
    assert abs(r - tm.length) < 1e-9 and np.abs(c - tm.centroid).max() < 1e-9
    return dict(W=Wop, verts=verts, normals=tm.normals[perm], faces=faces, top=np.r_[np.ones(len(top), bool), np.zeros(len(bot), bool)],
                centroid=tm.centroid, length=tm.length, area=vertex_areas(verts, faces))


def arctic_verts_at(obj, arti):
    """Vertices AND outward normals of the articulated object (top part rotated by rotvec (0, 0, -arti))."""
    v = obj["verts"].copy(); nr = obj["normals"].copy()
    R = Rotation.from_rotvec((0, 0, -float(arti))).as_matrix()
    v[obj["top"]] = v[obj["top"]] @ R.T; nr[obj["top"]] = nr[obj["top"]] @ R.T
    return v, nr


def canonical_geometry(obj):
    """Per-canonical-point geometry of one object (raw metres, uncentred; the frame geometry centres and scales later)."""
    Wop, top = obj["W"], obj["top"]
    colsum = Wop.sum(0); share = np.where(colsum > 0, Wop / np.maximum(colsum, 1e-12), 0.0)
    alpha = share @ (obj["area"] / obj["area"].sum())
    return dict(X_top=Wop[:, top] @ obj["verts"][top], X_bot=Wop[:, ~top] @ obj["verts"][~top],
                N_top=Wop[:, top] @ obj["normals"][top], N_bot=Wop[:, ~top] @ obj["normals"][~top],
                alpha=alpha, valid=Wop.sum(1) > 0, centroid=obj["centroid"], length=obj["length"], n_verts=len(obj["verts"]),
                frac_top=float(top.mean()))


# ------------------------------------------------------------------------------ per take
def process_take(args):
    ds, sid, rows, P_by_cat = args
    try:
        import torch
        torch.set_num_threads(1)
        lab = W.mano_labels()
        tk = BG.taco_take(sid) if ds == "taco" else BG.arctic_take(sid, rows[0]["category"])
    except Exception as ex:  # noqa: BLE001
        return [dict(n=int(r["n"]), error=f"{type(ex).__name__}: {ex}") for r in rows]
    out = []
    for r in rows:
        try:
            out.append(process_sequence(ds, tk, r, lab, P_by_cat))
        except Exception as ex:  # noqa: BLE001
            out.append(dict(n=int(r["n"]), error=f"{type(ex).__name__}: {ex}"))
    return out


def process_sequence(ds, tk, r, lab, P_by_cat):
    n, t0, side = int(r["n"]), int(r["t0"]), {"L": "left", "R": "right"}[r["hand"]]
    n_fr = tk["n_frames"]
    tt = t0 + np.arange(S.T); fr = np.clip(tt, 0, n_fr - 1)
    key = r["mesh_id"] if ds == "taco" else r["category"]
    if key not in _CTX:
        _CTX[key] = taco_object(r["mesh_id"], r["category"], P_by_cat) if ds == "taco" else arctic_object(r["category"], P_by_cat)
    obj = _CTX[key]; Wop = obj["W"]
    M = np.zeros((S.T, 512, 6), np.uint8); sgn = np.zeros((S.T, 512), np.int8); nh = np.zeros((S.T, 512, 3), np.int8)
    frac_out = np.zeros(S.T, np.float32); min_d = np.zeros(S.T, np.float32)
    checks = []
    if ds == "taco":
        n_tool = tk["n_tool"]; cols = slice(0, n_tool) if r["role"] == "tool" else slice(n_tool, None)
        R_all, p_all = tk["poses"][r["role"]]
        d_src = tk["dist"][side]
    else:
        st = tk["state"]
        probe = np.load(f"{ARCTIC_PROBE}/{r['sequence_id'].replace('/', '__')}.npz")
        d_src = probe[f"contact_{side}"]; cols = slice(None)
    for i, tf in enumerate(fr):
        tf = int(tf)
        if ds == "taco":
            v, nr = obj["verts"], obj["normals"]; hand_obj = (tk["skins"][side][tf] - p_all[tf]) @ R_all[tf]
        else:
            v, nr = arctic_verts_at(obj, st[tf, 0]); hand_obj = W.arctic_rigid(tk["skins"][side][tf], st[tf])
        d, nn = cKDTree(hand_obj).query(v)
        part = lab[nn]
        onehot = np.zeros((len(v), 6), np.float32); onehot[np.arange(len(v)), part] = 1.0
        m = Wop @ onehot
        M[i] = np.clip(np.rint(m * 255), 0, 255).astype(np.uint8)
        outward = ((hand_obj[nn] - v) * nr).sum(1) > 0
        s = Wop @ (2.0 * outward - 1.0)
        sgn[i] = np.where(s < 0, -1, 1).astype(np.int8)
        nv = Wop @ (np.where(outward[:, None], nr, -nr))                       # toward-hand normals averaged per canonical point
        nv = nv / np.maximum(np.linalg.norm(nv, axis=1, keepdims=True), 1e-9)
        nh[i] = np.clip(np.rint(nv * 127), -127, 127).astype(np.int8)
        frac_out[i] = float(outward[d < 0.02].mean()) if (d < 0.02).any() else np.nan
        min_d[i] = float(d.min())
        if i % 21 == 0 and tf < len(d_src):                                  # W check against the cached contact (3 frames)
            dd = np.asarray(d_src[tf][cols], np.float32)
            checks.append((i, (np.exp(-dd / 0.02) @ Wop.T).astype(np.float32), float(dd.min())))
    return dict(n=n, M=M, sgn=sgn, nh=nh, frac_outward=frac_out, min_d=min_d, checks=checks, key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ds = a.dataset; t_start = time.time()
    D = S.load_raw(ds); meta = D["meta"]; man = D["man"]; C_seq = D["C"]
    z = np.load(S.SV.HP.ROOTS[ds] / "sequences.npz", allow_pickle=True)
    P_by_cat = {str(c): p.astype(np.float64) for c, p in zip(z["canonical_cats"], z["canonical_points"])}
    use = np.sort(np.unique(np.concatenate([man["train"], man["val"], man["test"]])))
    tasks = []
    for sid, idx in meta.iloc[use].groupby("sequence_id").indices.items():
        rows = [dict(n=int(use[i]), sequence_id=sid, t0=int(meta.iloc[use[i]].t0), hand=meta.iloc[use[i]].hand, role=meta.iloc[use[i]].role,
                     mesh_id=str(meta.iloc[use[i]].mesh_id), category=meta.iloc[use[i]].category) for i in idx]
        tasks.append((ds, sid, rows, P_by_cat))
    if a.limit:
        tasks = tasks[:a.limit]
    N_used = sum(len(t[2]) for t in tasks)
    row_of = np.full(len(meta), -1, int); example = np.array([r["n"] for t in tasks for r in t[2]]); row_of[example] = np.arange(N_used)
    M = np.zeros((N_used, S.T, 512, 6), np.uint8); sgn = np.zeros((N_used, S.T, 512), np.int8); nh = np.zeros((N_used, S.T, 512, 3), np.int8)
    frac_out = np.full((N_used, S.T), np.nan, np.float32); min_d = np.zeros((N_used, S.T), np.float32)
    done_ex = np.zeros(N_used, bool); errors = []; c_err = []; d_err = []; keys = {}
    log.info("%s: %d sequences in %d takes, %d workers", ds, N_used, len(tasks), a.workers)
    done = 0
    with mp.Pool(a.workers, maxtasksperchild=64) as pool:
        for res in pool.imap_unordered(process_take, tasks, chunksize=1):
            for o in res:
                if "error" in o:
                    errors.append(dict(example=o["n"], error=o["error"])); continue
                j = row_of[o["n"]]
                M[j] = o["M"]; sgn[j] = o["sgn"]; nh[j] = o["nh"]; frac_out[j] = o["frac_outward"]; min_d[j] = o["min_d"]; done_ex[j] = True
                keys[o["n"]] = o["key"]
                for i, Xr, dmin in o["checks"]:
                    c_err.append(float(np.abs(Xr - C_seq[o["n"], i].astype(np.float32)).max()))
                    d_err.append(abs(dmin - o["min_d"][i]))
            done += 1
            if done % 100 == 0:
                log.info("  %d / %d takes (%.0f s)", done, len(tasks), time.time() - t_start)
    # canonical geometry of every object in use (main process)
    objs = {}
    for n, key in keys.items():
        if key in objs:
            continue
        r = meta.iloc[n]
        objs[key] = canonical_geometry(taco_object(str(r.mesh_id), r.category, P_by_cat) if ds == "taco" else arctic_object(r.category, P_by_cat))
    gk = sorted(objs); g_of = {k: i for i, k in enumerate(gk)}
    geo_index = np.array([g_of[keys[n]] if n in keys else -1 for n in range(len(meta))])
    out = S.mask_path(ds); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, M=M, sgn=sgn, nh=nh, example=example, row_of=row_of, frac_outward=frac_out, min_d=min_d, done=done_ex)
    np.savez(S.geometry_path(ds), keys=np.array(gk), geo_index=geo_index,
             **{k: np.stack([objs[g][k] for g in gk]) for k in ("X_top", "X_bot", "N_top", "N_bot", "alpha", "valid", "centroid", "length", "n_verts", "frac_top")})
    inc = done_ex
    summary = dict(dataset=ds, n_sequences=int(inc.sum()), n_takes=len(tasks), n_errors=len(errors), errors=errors[:50], n_objects=len(gk),
                   seconds=time.time() - t_start, W_check_max_abs=float(np.max(c_err)) if c_err else None, W_check_n=len(c_err),
                   min_d_check_max=float(np.max(d_err)) if d_err else None,
                   frac_outward_mean=float(np.nanmean(frac_out[inc])), sgn_negative_frac=float((sgn[inc] < 0).mean()),
                   M_rowsum_mean=float((M[inc][:, ::8].astype(np.float32) / 255).sum(-1).mean()),
                   mean_label_entropy=float(_entropy(M[inc][:, ::8])), valid_points_frac=float(np.mean([objs[g]["valid"].mean() for g in gk])),
                   alpha_sum_mean=float(np.mean([objs[g]["alpha"].sum() for g in gk])), part_share=(M[inc][:, ::8].astype(np.float32) / 255).mean((0, 1, 2)).tolist())
    S.write_json(out.parent / "masks_build.json", summary)
    log.info("done: %s", {k: v for k, v in summary.items() if k != "errors"})
    if errors:
        log.warning("%d errors, first: %s", len(errors), errors[0])


def _entropy(Mu8):
    p = Mu8.astype(np.float32) / 255.0
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-6)
    return -(p * np.log(np.maximum(p, 1e-9))).sum(-1).mean()


if __name__ == "__main__":
    main()
