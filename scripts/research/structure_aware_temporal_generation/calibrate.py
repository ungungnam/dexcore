#!/usr/bin/env python
"""Step 2: calibrate the grid R2 extraction ONCE on the training GT maps and measure its agreement with the exact R2.
    CUDA_VISIBLE_DEVICES=4 python calibrate.py --dataset taco

Fitted on the TRAIN sequences (GT canonical maps, GT part labels of the mask cache):
  c_hard, n_min   the hard-contact threshold on the canonical value and the point count of the participation rule
                  (causal 3-frame majority of [count >= n_min]) that maximise the macro F1 against the exact participation
  beta            the scale of the cell-area weighted amount against the exact amount (least squares through the origin)
Frozen for every model (train.py reads <ds>/calibration.json); the soft temperatures tau_c, tau_n are fixed constants.
Reported (train / test): the agreement of the hard-rule grid extraction and of the soft training operator with the exact R2
(sanity check 10), and the wrench floor of the grid extraction (q from one patch per active part of the grid R2 of the GT map
vs the exact 76-D profile; also from the exact R2 itself, which isolates the one-patch-per-part simplification).
Writes <ds>/calibration.json, <ds>/calibration_grid.csv.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import sat_common as S
from sat_data import SeqData
from r2_ops import causal_majority, hard_r2, soft_r2, wrench_support, angle_deg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("calibrate")
C_GRID = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
N_GRID = [1, 2, 3, 4, 5, 6, 8, 10]


def batches(data, idx, bs=32):
    idx_t = torch.from_numpy(idx).to(data.device)
    for i in range(0, len(idx_t), bs):
        n = idx_t[i:i + bs]
        t = torch.arange(S.T, device=data.device)[None].expand(len(n), -1)
        M, nh, lab = data.masks(n, t); x, alpha, valid = data.geometry(n, t)
        yield n, t, data.C[n], M, nh, lab, x, alpha


def prev_flags(data, n, lab0, c_hard, n_min):
    """[count >= n_min] of the two frames before t = 0 (C_pad frames -2, -1 with the labels of frame 0)."""
    w = data.C_pad_prev[n].clamp(min=0)
    oneh = F.one_hot(lab0, 6).float()
    cnt = torch.einsum("btv,bvk->btk", (w >= c_hard).float(), oneh)
    return cnt >= n_min


@torch.no_grad()
def grid_search(data, idx):
    tp = np.zeros((len(C_GRID), len(N_GRID), 6)); fp = np.zeros_like(tp); fn = np.zeros_like(tp)
    for n, t, C, M, nh, lab, x, alpha in batches(data, idx):
        a_true = data.a[n]
        w = C.clamp(min=0); oneh = F.one_hot(lab, 6).float()
        w_prev = data.C_pad_prev[n].clamp(min=0); oneh0 = oneh[:, 0]
        for i, ch in enumerate(C_GRID):
            cnt = torch.einsum("btv,btvk->btk", (w >= ch).float(), oneh)
            cnt_prev = torch.einsum("btv,bvk->btk", (w_prev >= ch).float(), oneh0)
            for j, nm in enumerate(N_GRID):
                a = causal_majority(cnt >= nm, cnt_prev >= nm)
                tp[i, j] += ((a > 0.5) & (a_true > 0.5)).float().sum((0, 1)).cpu().numpy()
                fp[i, j] += ((a > 0.5) & (a_true < 0.5)).float().sum((0, 1)).cpu().numpy()
                fn[i, j] += ((a < 0.5) & (a_true > 0.5)).float().sum((0, 1)).cpu().numpy()
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    macro = f1.mean(-1)
    rows = [dict(c_hard=ch, n_min=nm, macro_f1=macro[i, j], **{f"f1_{p}": f1[i, j, k] for k, p in enumerate(S.PARTS)}) for i, ch in enumerate(C_GRID) for j, nm in enumerate(N_GRID)]
    i, j = np.unravel_index(np.argmax(macro), macro.shape)
    return pd.DataFrame(rows), float(C_GRID[i]), int(N_GRID[j]), float(macro[i, j])


@torch.no_grad()
def amount_scale(data, idx, cal):
    sxy = sxx = 0.0; xs, ys = [], []
    for n, t, C, M, nh, lab, x, alpha in batches(data, idx):
        oneh = F.one_hot(lab, 6).float()
        mg = torch.einsum("btv,btvk,bv->btk", C.clamp(min=0), oneh, alpha); mt = data.m[n]
        sxy += float((mg * mt).sum()); sxx += float((mg * mg).sum())
        xs.append(mg.flatten().cpu().numpy()); ys.append(mt.flatten().cpu().numpy())
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    return sxy / max(sxx, 1e-12), float(np.corrcoef(xs, ys)[0, 1])


@torch.no_grad()
def agreement(data, idx, cal, split):
    """Hard grid / soft operator R2 of the GT map vs the exact R2; wrench floors."""
    calt = dict(cal)
    acc = {}; scores_soft, scores_hard, truth = [], [], []
    q_rel = {"grid": [], "exactR2": []}; q_cos = {"grid": [], "exactR2": []}; q_cnt = 0
    for n, t, C, M, nh, lab, x, alpha in batches(data, idx):
        a_t, m_t, p_t, n_t, q_t = data.a[n], data.m[n], data.p[n], data.nrm[n], data.q[n]
        pf = prev_flags(data, n, lab[:, 0], cal["c_hard"], cal["n_min"])
        H = hard_r2(C, lab, nh, x, alpha, calt, pf)
        Sf = soft_r2(C, M, nh, x, alpha, calt)
        act = a_t > 0.5; both_h = act & (H["a"] > 0.5)
        def add(k, v, w=None):
            v = v.float()
            if w is None:
                acc.setdefault(k, []).append(v.flatten().cpu().numpy())
            else:
                acc.setdefault(k, []).append(v[w].flatten().cpu().numpy())
        add("hard_part_hamming", (H["a"] - a_t).abs())
        add("hard_amount_rel_l1", (H["m"] - m_t).abs().sum(-1) / data.tstats["m_bar"])
        add("hard_centroid_dist_both", (H["p"] - p_t).norm(dim=-1), both_h)
        add("hard_centroid_dist_gtactive", (H["p"] - p_t).norm(dim=-1), act)
        add("hard_normal_angle_both", angle_deg(H["n"], n_t), both_h)
        add("hard_normal_angle_gtactive", angle_deg(H["n"], n_t), act)
        add("soft_amount_abs_err_vs_exact", (Sf["m"] - m_t).abs())
        add("soft_centroid_dist_gtactive", (Sf["p"] - p_t).norm(dim=-1), act)
        add("soft_normal_angle_gtactive", angle_deg(Sf["n"], n_t), act)
        add("soft_vs_hard_centroid", (Sf["p"] - H["p"]).norm(dim=-1), both_h)
        add("soft_vs_hard_normal_angle", angle_deg(Sf["n"], H["n"]), both_h)
        add("hard_n_points_active", H["n_hard_pts"], H["a"] > 0.5)
        scores_soft.append(torch.sigmoid(Sf["a_logit"]).cpu().numpy().reshape(-1, 6)); scores_hard.append(H["a"].cpu().numpy().reshape(-1, 6)); truth.append(a_t.cpu().numpy().reshape(-1, 6))
        acc.setdefault("m_grid", []).append(H["m"].flatten().cpu().numpy()); acc.setdefault("m_soft", []).append(Sf["m"].flatten().cpu().numpy()); acc.setdefault("m_true", []).append(m_t.flatten().cpu().numpy())
        # wrench floors
        qs = q_t.sum(-1); ok = qs > 1e-6
        for key, (pp, nn, aa) in {"grid": (H["p"], H["n"], H["a"]), "exactR2": (p_t, n_t, a_t)}.items():
            qh = wrench_support(pp, nn, aa, data.U)
            q_rel[key].append(((qh - q_t).abs().sum(-1) / qs.clamp(min=1e-9))[ok].cpu().numpy())
            q_cos[key].append(((qh * q_t).sum(-1) / (qh.norm(dim=-1) * q_t.norm(dim=-1) + 1e-9))[ok].cpu().numpy())
        q_cnt += int(ok.sum())
    out = {k: float(np.mean(np.concatenate(v))) for k, v in acc.items() if k not in ("m_grid", "m_soft", "m_true")}
    out.update({k + "_median": float(np.median(np.concatenate(v))) for k, v in acc.items() if "dist" in k or "angle" in k})
    ss, sh, tr = np.concatenate(scores_soft), np.concatenate(scores_hard), np.concatenate(truth)
    f1 = lambda pred: np.mean([2 * ((pred[:, k] > 0.5) & (tr[:, k] > 0.5)).sum() / max(2 * ((pred[:, k] > 0.5) & (tr[:, k] > 0.5)).sum() + ((pred[:, k] > 0.5) & (tr[:, k] < 0.5)).sum() + ((pred[:, k] < 0.5) & (tr[:, k] > 0.5)).sum(), 1) for k in range(6)])
    out["hard_part_macro_f1"] = float(f1(sh)); out["soft_part_macro_f1"] = float(f1(ss))
    out["soft_part_macro_auprc"] = float(np.mean([average_precision_score(tr[:, k] > 0.5, ss[:, k]) for k in range(6) if (tr[:, k] > 0.5).any()]))
    mg, ms, mt = (np.concatenate(acc[k]) for k in ("m_grid", "m_soft", "m_true"))
    out["hard_amount_pearson"] = float(np.corrcoef(mg, mt)[0, 1]); out["soft_amount_pearson"] = float(np.corrcoef(ms, mt)[0, 1])
    out["soft_amount_scale_vs_exact"] = float((ms * mt).sum() / max((ms * ms).sum(), 1e-12))
    for key in ("grid", "exactR2"):
        out[f"wrench_floor_{key}_rel_l1"] = float(np.mean(np.concatenate(q_rel[key]))); out[f"wrench_floor_{key}_cos"] = float(np.mean(np.concatenate(q_cos[key])))
    out["n_frames_with_wrench"] = q_cnt; out["split"] = split; out["n_sequences"] = int(len(idx))
    return out


def wrench_self_check(data):
    """The batched support capacity equals wrench_lp.capacity(strict=False) with one patch per finger."""
    import wrench_lp as L
    rng = np.random.default_rng(0); U = data.U.cpu().numpy(); errs = []
    for _ in range(20):
        act = rng.random(6) < 0.6
        p = rng.normal(size=(6, 3)) * 0.5; nr = rng.normal(size=(6, 3)); nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        patches = [dict(part=k, centroid=p[k], normal=nr[k]) for k in range(6) if act[k]]
        h, _ = L.capacity(patches, np.zeros(3), 1.0, S.MU, U, "finger_equal", strict=False)
        q = wrench_support(torch.tensor(p, dtype=torch.float32, device=data.device)[None], torch.tensor(nr, dtype=torch.float32, device=data.device)[None],
                           torch.tensor(act, dtype=torch.float32, device=data.device)[None], data.U)[0].cpu().numpy()
        errs.append(float(np.abs(q - h).max()))
    return float(max(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS)
    a = ap.parse_args()
    ds = a.dataset; t0 = time.time()
    dev = torch.device("cuda")
    data = SeqData(ds, dev)
    tr, va, te = data.idx["train"], data.idx["val"], data.idx["test"]
    grid, c_hard, n_min, f1 = grid_search(data, tr)
    log.info("%s: participation rule c_hard %.2f n_min %d (train macro F1 %.3f) [%.0f s]", ds, c_hard, n_min, f1, time.time() - t0)
    cal = dict(c_hard=c_hard, n_min=n_min, tau_c=S.TAU_C, tau_n=S.TAU_N)
    beta, r = amount_scale(data, tr, cal)
    cal["beta"] = beta
    log.info("%s: amount scale beta %.4f (train Pearson %.3f)", ds, beta, r)
    wcheck = wrench_self_check(data)
    agr = {sp: agreement(data, idx, cal, sp) for sp, idx in (("train", tr), ("test", te))}
    for sp in agr:
        log.info("%s %s agreement: %s", ds, sp, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agr[sp].items()})
    S.ds_out(ds).mkdir(parents=True, exist_ok=True)
    grid.to_csv(S.ds_out(ds) / "calibration_grid.csv", index=False)
    S.write_json(S.calibration_path(ds), dict(dataset=ds, **cal, train_macro_f1=f1, amount_train_pearson=r, m_bar_train=data.tstats["m_bar"], sigma_r=data.sigma_r,
                                              wrench_batched_vs_wrench_lp_max_abs=wcheck, agreement=agr, fitted_on="train sequences (GT maps, GT part labels)",
                                              note="c_hard / n_min / beta fitted once on the training GT maps; tau_c / tau_n fixed; shared by every model"))
    log.info("done %s (%.0f s)", ds, time.time() - t0)


if __name__ == "__main__":
    main()
