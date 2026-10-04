"""Temporal contact-change events: shared loading, per-frame quantities and statistics.

Everything is read from the hierarchical contact-generation experiment roots (no retraining, no new
splits). Arrays used (Step 1 inventory):
  <root>/sequences.npz        C (N,64,512) f16 raw canonical contact; O (N,80,D) object states for
                              t in [-8,72) (names in state_names); canonical_points per category
  <root>/sequences_meta.csv   sequence_id, take_key, group, category, role, hand, t0, split_set, ...
  <root>/manifests/fixed0.json  train / val / test indices (the fixed split)
  <root>/eval/fixed0/preds.npz  gt, static_gt, gtinit_vf, samplerG_vf__K1/K10, samplerGT_vf__K10 (B,64,512)
  <root>/eval/fixed0/bimart_preds.npz (TACO, ARCTIC)  gt_restricted, bimart_K1, bimart_K10 (B,64,512)
  firm-contact state: n_hard (vertices within 1 cm) recomputed per frame from the raw distance
                      arrays (TACO) or read from the dynamic-contact frame cache (ARCTIC, OakInk2);
                      firm = n_hard > 0 and mass >= zero_mass, held for 3 consecutive frames
                      (the sequence-construction rule)
  hand-surface points in the object canonical frame: TACO gen3 sequences 'kp' (T,2,100,3) in the
                      target frame (tool-role sequences are re-expressed in the tool frame with
                      the object states); ARCTIC *_processed_hand_features.npy
                      '{left,right}_hand_sampled_verts_cano' (T,100,3); OakInk2: not cached ->
                      unavailable
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

ROOTS = {"taco": Path("/result/uhnam/dexcore/taco/50_hier_contact_gen"),
         "arctic": Path("/result/uhnam/dexcore/arctic/40_hier_contact_gen"),
         "oakink2": Path("/result/uhnam/dexcore/oakink2/20_hier_contact_gen")}
CACHES = {"taco": Path("/result/uhnam/dexcore/taco/40_representation_study/time_decomp/dynamic_contact/cache"),
          "arctic": Path("/result/uhnam/dexcore/arctic/30_dynamic_contact/cache"),
          "oakink2": Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact/cache")}
OUT = Path("/result/uhnam/dexcore/reports/temporal_contact_events")
T = 64
CTX_PAD = 8
HARD_MM = 0.01
FIRM_PERSIST = 3
EPS = 1e-9
LABEL = {"taco": "TACO", "arctic": "ARCTIC", "oakink2": "OakInk2"}
MODELS = {"B0_static": "static_gt", "B1_gtinit_vf": "gtinit_vf", "B2_samplerG_vf_K10": "samplerG_vf__K10",
          "B2_samplerG_vf_K1": "samplerG_vf__K1", "B3_samplerGT_vf_K10": "samplerGT_vf__K10"}
BIMART_MODELS = {"B4_bimart_K10": "bimart_K10", "B4_bimart_K1": "bimart_K1"}


def zero_threshold(ds):
    return float(json.loads((CACHES[ds] / "zero_threshold.json").read_text())["zero_mass"])


def load(ds):
    r = ROOTS[ds]
    z = np.load(r / "sequences.npz", allow_pickle=True)
    meta = pd.read_csv(r / "sequences_meta.csv", dtype={"mesh_id": str, "subject": str, "sequence_id": str})
    man = json.load(open(r / "manifests" / "fixed0.json"))
    preds = np.load(r / "eval" / "fixed0" / "preds.npz")
    bim = np.load(r / "eval" / "fixed0" / "bimart_preds.npz") if (r / "eval" / "fixed0" / "bimart_preds.npz").exists() else None
    return dict(C=z["C"], O=z["O"], names=[str(n) for n in z["state_names"]], cats=[str(c) for c in z["canonical_cats"]],
                P=z["canonical_points"], meta=meta, man={k: np.asarray(v, int) for k, v in man.items() if isinstance(v, list)},
                preds=preds, bimart=bim, thr=zero_threshold(ds))


# ------------------------------------------------------------------ firm-contact state
def firm_mask(mass, n_hard, thr):
    raw = (n_hard > 0) & (mass >= thr)
    n = len(raw)
    if n < FIRM_PERSIST:
        return np.zeros(n, bool)
    runs = np.ones(n - FIRM_PERSIST + 1, bool)
    for j in range(FIRM_PERSIST):
        runs &= raw[j:n - FIRM_PERSIST + 1 + j]
    out = np.zeros(n, bool)
    for j in range(FIRM_PERSIST):
        out[j:n - FIRM_PERSIST + 1 + j] |= runs
    return out


def hard_counts_and_hand(ds, meta_rows, D):
    """For the given meta rows: n_hard (n,64) int and hand-surface speed relative to the contacted
    object (n,63) in m/frame (NaN where unavailable)."""
    n = len(meta_rows)
    nh = np.zeros((n, T), np.int32); hand_speed = np.full((n, T - 1), np.nan, np.float32)
    if ds == "taco":
        seq_root = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
        files = pd.read_csv(seq_root / "sequence_index.csv").set_index("sequence_id")["file"]
        wi = pd.read_csv("/result/uhnam/dexcore/taco/40_representation_study/window_index.csv").drop_duplicates("sequence_id").set_index("sequence_id")
        O = D["O"]; names = D["names"]
        ip = [names.index(k) for k in ("tool_pos_x", "tool_pos_y", "tool_pos_z")]
        ir = [names.index(f"tool_rot6d_{i}") for i in range(6)]
        for i, r in enumerate(meta_rows.itertuples()):
            with np.load(seq_root / "sequences" / files[r.sequence_id]) as z:
                d = z[f"contact_{'left' if r.hand == 'L' else 'right'}"][r.t0:r.t0 + T]
                kp = z["kp"][r.t0:r.t0 + T].reshape(T, 2, 100, 3)[:, 0 if r.hand == "L" else 1]      # target frame
            n_tool = int(wi.loc[r.sequence_id].n_tool)
            dd = d[:, :n_tool] if r.role == "tool" else d[:, n_tool:]
            nh[i] = (dd < HARD_MM).sum(1)
            if r.role == "tool":                                            # re-express in the tool frame
                o = O[r.Index, CTX_PAD:CTX_PAD + T]
                p = o[:, ip]; a = o[:, ir[:3]]; b = o[:, ir[3:]]
                a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b - (a * b).sum(1, keepdims=True) * a
                b = b / np.linalg.norm(b, axis=1, keepdims=True); c = np.cross(a, b)
                R = np.stack([a, b, c], 2)                                   # (T,3,3) columns = axes
                kp = np.einsum("tji,tkj->tki", R, kp - p[:, None])          # R^T (kp - p)
            hand_speed[i] = np.linalg.norm(np.diff(kp, axis=0), axis=2).mean(1)
    else:
        cache = CACHES[ds] / "frames_full"
        by_group = {}
        for g, gm in meta_rows.groupby("group"):
            zc = np.load(cache / f"{g}.npz", allow_pickle=True)
            key = pd.Series(zc["n_hard"], index=pd.MultiIndex.from_arrays([zc["sequence_id"].astype(str), zc["frame"]]))
            key = key[~key.index.duplicated()]
            by_group[g] = key
        hand_root = Path("/home/uhnam/workspace/dexcore/third_party/BimArt/data/arctic_processed_data") if ds == "arctic" else None
        hcache = {}
        for i, r in enumerate(meta_rows.itertuples()):
            k = by_group[r.group]
            nh[i] = k.loc[[(r.sequence_id, f) for f in range(r.t0, r.t0 + T)]].values
            if ds == "arctic":
                stem, subj = r.sequence_id.split("/")
                f = hand_root / r.category / subj / f"{stem}_processed_hand_features.npy"
                if str(f) not in hcache:
                    h = np.load(f, allow_pickle=True).item()
                    hcache[str(f)] = {"L": h["left_hand_sampled_verts_cano"], "R": h["right_hand_sampled_verts_cano"]}
                kp = hcache[str(f)][r.hand][r.t0:r.t0 + T]
                hand_speed[i] = np.linalg.norm(np.diff(kp, axis=0), axis=2).mean(1)
    return nh, hand_speed


# ------------------------------------------------------------------ per-frame quantities
def decompose(C, thr):
    """C (T,512) float32 -> dict of per-transition arrays (T-1,): d, m0, m1, dm, A, S, r, valid."""
    Cn = C[1:]; Cp = C[:-1]
    dC = Cn - Cp
    d = np.linalg.norm(dC, axis=1)
    m0, m1 = Cp.sum(1), Cn.sum(1)
    valid = (m0 >= thr) & (m1 >= thr)
    A = np.full(len(d), np.nan, np.float32); S = np.full(len(d), np.nan, np.float32); resid = np.full(len(d), np.nan, np.float32)
    if valid.any():
        P0 = Cp[valid] / m0[valid, None]; P1 = Cn[valid] / m1[valid, None]
        Pbar = 0.5 * (P0 + P1); mbar = 0.5 * (m0[valid] + m1[valid])
        dm = (m1 - m0)[valid]; dP = P1 - P0
        a_vec = dm[:, None] * Pbar; s_vec = mbar[:, None] * dP
        A[valid] = np.linalg.norm(a_vec, axis=1); S[valid] = np.linalg.norm(s_vec, axis=1)
        resid[valid] = np.linalg.norm(a_vec + s_vec - dC[valid], axis=1)
    r = A / (A + S + EPS)
    return dict(d=d, m0=m0, m1=m1, dm=m1 - m0, A=A, S=S, r=r, valid=valid, resid=resid)


def persistence(C, h):
    """(T,) array: ||C_{t+h} - C_{t-h}|| / sum_{j=t-h}^{t+h-1} ||C_{j+1}-C_j||, NaN at the edges."""
    d = np.linalg.norm(np.diff(C, axis=0), axis=1)
    out = np.full(len(C), np.nan, np.float32)
    for t in range(h, len(C) - h):
        out[t] = np.linalg.norm(C[t + h] - C[t - h]) / (d[t - h:t + h].sum() + EPS)
    return out


def smoothed_d(C, k):
    Cs = uniform_filter1d(C, size=k, axis=0, mode="nearest")
    return np.linalg.norm(np.diff(Cs, axis=0), axis=1)


def kinematics(ds, D, idx):
    """Per-frame signals (n,64) for the sequences idx: dict name -> array (NaN if unavailable)."""
    O = D["O"][idx, CTX_PAD:CTX_PAD + T]; names = D["names"]
    col = lambda ks: O[:, :, [names.index(k) for k in ks]]
    out = {}
    if ds == "taco":
        out["obj_lin_speed"] = np.linalg.norm(col(["targ_vel_x", "targ_vel_y", "targ_vel_z"]), axis=2)          # target, world
        out["obj_ang_speed"] = np.linalg.norm(col(["targ_angvel_x", "targ_angvel_y", "targ_angvel_z"]), axis=2)
        out["tool_rel_lin_speed"] = np.linalg.norm(col(["tool_vel_x", "tool_vel_y", "tool_vel_z"]), axis=2)    # tool in the target frame
        out["tool_rel_ang_speed"] = np.linalg.norm(col(["tool_angvel_x", "tool_angvel_y", "tool_angvel_z"]), axis=2)
        out["arti_rate"] = np.full(O.shape[:2], np.nan, np.float32)
    else:
        out["obj_lin_speed"] = np.linalg.norm(col(["vel_x", "vel_y", "vel_z"]), axis=2)
        out["obj_ang_speed"] = np.linalg.norm(col(["angvel_x", "angvel_y", "angvel_z"]), axis=2)
        out["tool_rel_lin_speed"] = np.full(O.shape[:2], np.nan, np.float32); out["tool_rel_ang_speed"] = np.full(O.shape[:2], np.nan, np.float32)
        out["arti_rate"] = np.abs(col(["arti_rate"])[:, :, 0]) if ds == "arctic" else np.full(O.shape[:2], np.nan, np.float32)
    return out


KIN_AVAILABLE = {"taco": ["obj_lin_speed", "obj_ang_speed", "tool_rel_lin_speed", "tool_rel_ang_speed", "hand_rel_speed"],
                 "arctic": ["obj_lin_speed", "obj_ang_speed", "arti_rate", "hand_rel_speed"],
                 "oakink2": ["obj_lin_speed", "obj_ang_speed"]}
KIN_LABEL = {"obj_lin_speed": "object linear speed [m/frame]", "obj_ang_speed": "object angular speed [rad/frame]",
             "tool_rel_lin_speed": "tool speed relative to target [m/frame]", "tool_rel_ang_speed": "tool angular speed relative to target [rad/frame]",
             "arti_rate": "|articulation rate| [rad/frame]", "hand_rel_speed": "hand-surface speed relative to object [m/frame]"}


# ------------------------------------------------------------------ statistics
def cluster_bootstrap(values, clusters, n_boot=1000, seed=0, stat="mean"):
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    ok = ~np.isnan(values); values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    u, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    if stat == "mean":
        sums = np.bincount(inv, weights=values, minlength=len(u)); cnt = np.bincount(inv, minlength=len(u)).astype(float)
        reps = []
        for _ in range(n_boot):
            w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
            reps.append((w * sums).sum() / max((w * cnt).sum(), 1e-12))
        return float(values.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
    reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))
        sel = np.repeat(np.arange(len(values)), w[inv])
        reps.append(np.median(values[sel]) if len(sel) else np.nan)
    return float(np.median(values)), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5))


def cluster_bootstrap_ratio(num, den, clusters, n_boot=1000, seed=0):
    """Ratio of cluster-weighted sums (e.g. energy shares) with a cluster bootstrap CI."""
    num = np.asarray(num, float); den = np.asarray(den, float); clusters = np.asarray(clusters)
    u, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(u)); sd = np.bincount(inv, weights=den, minlength=len(u))
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        reps.append((w * sn).sum() / max((w * sd).sum(), 1e-12))
    return float(num.sum() / max(den.sum(), 1e-12)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def spearman_cluster(x, y, clusters, n_boot=500, seed=0):
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float); clusters = np.asarray(clusters)
    ok = ~np.isnan(x) & ~np.isnan(y); x, y, clusters = x[ok], y[ok], clusters[ok]
    if len(x) < 10:
        return np.nan, np.nan, np.nan, int(len(x))
    rho = spearmanr(x, y).correlation
    u, inv = np.unique(clusters, return_inverse=True); rng = np.random.default_rng(seed); reps = []
    groups = [np.where(inv == g)[0] for g in range(len(u))]
    for _ in range(n_boot):
        pick = rng.integers(0, len(u), len(u)); sel = np.concatenate([groups[g] for g in pick])
        reps.append(spearmanr(x[sel], y[sel]).correlation)
    return float(rho), float(np.nanpercentile(reps, 2.5)), float(np.nanpercentile(reps, 97.5)), int(len(x))


def cliffs_delta(a, b):
    """P(a > b) - P(a < b) for two samples (subsampled to <= 4000 each for speed)."""
    a = np.asarray(a, float); b = np.asarray(b, float); a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    rng = np.random.default_rng(0)
    if len(a) > 4000: a = rng.choice(a, 4000, replace=False)
    if len(b) > 4000: b = rng.choice(b, 4000, replace=False)
    gt = (a[:, None] > b[None]).mean(); lt = (a[:, None] < b[None]).mean()
    return float(gt - lt)
