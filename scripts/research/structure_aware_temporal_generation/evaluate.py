#!/usr/bin/env python
"""Metrics of one prediction file (or of the GT / persistence references).
    CUDA_VISIBLE_DEVICES=4 python evaluate.py --dataset taco --name D1_seed0 --protocol A
    CUDA_VISIBLE_DEVICES=4 python evaluate.py --dataset taco --name GT --protocol A        (the grid-extraction floor)
    CUDA_VISIBLE_DEVICES=4 python evaluate.py --dataset taco --name PERSIST --protocol A   (C_t = s_0; B: the sampled s_0 held)

Per example and sample (frames 1..63 under protocol A, 0..63 under B):
  dense      E_C = mean_t ||C_hat_t - C_t|| (raw units, the previous studies' metric), E_C of the last 16 frames, s0 error
  R2         exact-rule grid extraction (r2_ops.hard_r2, the GT part label of every canonical point) vs the exact R2 of the
             feature cache: participation Hamming error + per-part TP / FP / FN (macro F1) + the soft score (macro AUPRC);
             amount normalised L1 = sum_k |m_hat_k - m_k| / train-mean total amount and RMSE; centroid distance (units of l)
             and normal angle (degrees) over the parts active in the GT frame (and over the parts active in both)
  wrench     one patch per active part of the grid R2 -> support-function profile over the 76 directions vs the exact q_t:
             relative L1 (sum |q_hat - q| / sum q), cosine, relative Q error; also vs the GT map's own grid wrench
  temporal   frame-to-frame dense change (jitter), total variation, frame-to-frame z-scored R2 change; error-vs-time curves
  diversity  (K >= 2) mean pairwise distances over the K samples: dense, participation, amount, centroid, normal, wrench
             (symmetric relative L1), z-scored R2 vector; per-frame dense / R2 diversity curves
  events     the four event classes of the test labels: metrics at the post frame and over the event window, transition
             timing of the participation set, within-window sample variance
Writes <ds>/metrics/<name>_<protocol>.npz and <ds>/metrics/<name>_<protocol>_events.csv.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

import sat_common as S
from sat_data import SeqData
from r2_ops import hard_r2, soft_r2, wrench_support, angle_deg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")
EVENT_PRE, EVENT_POST = 4, 8


def zscore_r2(data, a, m, p, n):
    """[a | m / sd_m | p / sd_p | n] (..., 48): the scale-free R2 vector of the temporal / diversity metrics."""
    mu_m, sd_m = data.tstats["m"]; mu_g, sd_g = data.tstats["geom"]
    return torch.cat([a, m / sd_m, p.flatten(-2) / sd_g[:18], n.flatten(-2)], -1)


def pairwise_mean(fn, K):
    """mean over k < l of fn(k, l) (tensors)."""
    acc, cnt = 0.0, 0
    for k in range(K):
        for l in range(k + 1, K):
            acc = acc + fn(k, l); cnt += 1
    return acc / max(cnt, 1)


@torch.no_grad()
def evaluate_chunk(data, n, P, protocol):
    """n (b,) example indices, P (b, K, 64, 512) raw predictions (frame 0 = s_0). Returns dict of per-(b, K) metrics, curves
    (b, K, 64) and per-frame R2 arrays."""
    b, K = P.shape[:2]; dev = data.device
    t_all = torch.arange(S.T, device=dev)[None].expand(b, -1)
    C = data.C[n]                                                                   # (b, 64, 512)
    M, nh, lab = data.masks(n, t_all); x, alpha, valid = data.geometry(n, t_all)
    a_t, m_t, p_t, n_t, q_t = data.a[n], data.m[n], data.p[n], data.nrm[n], data.q[n]
    cal = data.cal
    rep = lambda z: z.repeat_interleave(K, 0)                                        # (b, ...) -> (b K, ...)
    Pf = P.reshape(b * K, S.T, 512)
    H = hard_r2(Pf, rep(lab), rep(nh), rep(x), rep(alpha), cal)                        # first-frame padding of the majority
    Sf = soft_r2(Pf, rep(M), rep(nh), rep(x), rep(alpha), cal)
    a_h, m_h, p_h, n_h = (H[k].reshape(b, K, S.T, *H[k].shape[2:]) for k in ("a", "m", "p", "n"))
    a_soft = torch.sigmoid(Sf["a_logit"]).reshape(b, K, S.T, 6)
    q_h = wrench_support(p_h, n_h, a_h, data.U)                                       # (b, K, 64, 76)
    # the GT map's own grid wrench (floor reference)
    Hg = hard_r2(C, lab, nh, x, alpha, cal); q_g = wrench_support(Hg["p"], Hg["n"], Hg["a"], data.U)
    ev = slice(1, S.T) if protocol == "A" else slice(0, S.T)
    A_t, M_t, P_t, N_t, Q_t = (z[:, None] for z in (a_t, m_t, p_t, n_t, q_t))
    out, cur, r2 = {}, {}, {}
    err_t = (P - C[:, None]).norm(dim=-1)                                              # (b, K, 64)
    cur["dense"] = err_t
    out["E_C"] = err_t[:, :, ev].mean(-1); out["E_C_last16"] = err_t[:, :, -16:].mean(-1); out["s0_err"] = err_t[:, :, 0]
    # participation
    act = A_t > 0.5; both = act & (a_h > 0.5)
    ham_t = (a_h - A_t).abs().mean(-1); cur["part"] = ham_t; out["part_hamming"] = ham_t[:, :, ev].mean(-1)
    pred_pos = a_h[:, :, ev] > 0.5; true_pos = act[:, :, ev]
    out["tp"] = (pred_pos & true_pos).float().sum(2); out["fp"] = (pred_pos & ~true_pos).float().sum(2); out["fn"] = (~pred_pos & true_pos).float().sum(2)
    # amount
    am_t = (m_h - M_t).abs().sum(-1) / data.tstats["m_bar"]; cur["amount"] = am_t; out["amount_l1"] = am_t[:, :, ev].mean(-1)
    out["amount_rmse"] = ((m_h - M_t)[:, :, ev] ** 2).mean((-1, -2)).sqrt()
    # centroid / normal
    cd = (p_h - P_t).norm(dim=-1); ang = angle_deg(n_h, N_t)
    nact = act.float().sum(-1); nboth = both.float().sum(-1)
    cur["centroid"] = torch.where(nact > 0, (cd * act).sum(-1) / nact.clamp(min=1), torch.nan); cur["normal"] = torch.where(nact > 0, (ang * act).sum(-1) / nact.clamp(min=1), torch.nan)
    out["centroid"] = (cd * act)[:, :, ev].sum((-1, -2)) / act[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    out["normal"] = (ang * act)[:, :, ev].sum((-1, -2)) / act[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    out["centroid_both"] = (cd * both)[:, :, ev].sum((-1, -2)) / both[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    out["normal_both"] = (ang * both)[:, :, ev].sum((-1, -2)) / both[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    out["both_frac"] = both[:, :, ev].float().sum((-1, -2)) / act[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    # wrench
    qs = Q_t.sum(-1); okq = qs > 1e-6
    qrel_t = torch.where(okq, (q_h - Q_t).abs().sum(-1) / qs.clamp(min=1e-9), torch.nan); cur["wrench"] = qrel_t
    out["q_rel_l1"] = torch.nanmean(qrel_t[:, :, ev], -1)
    qcos = (q_h * Q_t).sum(-1) / (q_h.norm(dim=-1) * Q_t.norm(dim=-1) + 1e-9); out["q_cos"] = torch.nanmean(torch.where(okq, qcos, torch.nan)[:, :, ev], -1)
    out["q_Qerr"] = torch.nanmean(torch.where(okq, (q_h.mean(-1) - Q_t.mean(-1)).abs() / Q_t.mean(-1).clamp(min=1e-9), torch.nan)[:, :, ev], -1)
    qg = q_g[:, None]; qgs = qg.sum(-1); okg = qgs > 1e-6
    out["q_rel_l1_vs_grid"] = torch.nanmean(torch.where(okg, (q_h - qg).abs().sum(-1) / qgs.clamp(min=1e-9), torch.nan)[:, :, ev], -1)
    # the same structural metrics against the GT map's OWN grid extraction (self-consistent, zero floor)
    Ag, Mg, Pg, Ng = (Hg[k][:, None] for k in ("a", "m", "p", "n")); actg = Ag > 0.5; bothg = actg & (a_h > 0.5)
    out["part_hamming_vs_grid"] = (a_h - Ag).abs().mean(-1)[:, :, ev].mean(-1)
    out["amount_l1_vs_grid"] = ((m_h - Mg).abs().sum(-1) / data.tstats["m_bar"])[:, :, ev].mean(-1)
    out["centroid_vs_grid"] = ((p_h - Pg).norm(dim=-1) * bothg)[:, :, ev].sum((-1, -2)) / bothg[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    out["normal_vs_grid"] = (angle_deg(n_h, Ng) * bothg)[:, :, ev].sum((-1, -2)) / bothg[:, :, ev].float().sum((-1, -2)).clamp(min=1)
    # temporal
    dj = (P[:, :, 1:] - P[:, :, :-1]).norm(dim=-1)                                      # (b, K, 63)
    cur["jitter"] = dj
    lo = 1 if protocol == "A" else 0
    out["jitter_dense"] = dj[:, :, lo:].mean(-1); out["tv_dense"] = dj[:, :, lo:].sum(-1)
    Z = zscore_r2(data, a_h, m_h, p_h, n_h); dZ = (Z[:, :, 1:] - Z[:, :, :-1]).norm(dim=-1); cur["jitter_r2"] = dZ
    out["jitter_r2"] = dZ[:, :, lo:].mean(-1)
    # diversity
    if K >= 2:
        out["D_C"] = pairwise_mean(lambda k, l: (P[:, k] - P[:, l]).norm(dim=-1)[:, ev].mean(-1), K)
        cur["D_C_t"] = pairwise_mean(lambda k, l: (P[:, k] - P[:, l]).norm(dim=-1), K)
        out["D_part"] = pairwise_mean(lambda k, l: (a_h[:, k] - a_h[:, l]).abs().mean(-1)[:, ev].mean(-1), K)
        out["D_amount"] = pairwise_mean(lambda k, l: ((m_h[:, k] - m_h[:, l]).abs().sum(-1) / data.tstats["m_bar"])[:, ev].mean(-1), K)
        def cent_pair(k, l):
            bb = (a_h[:, k] > 0.5) & (a_h[:, l] > 0.5); d = (p_h[:, k] - p_h[:, l]).norm(dim=-1)
            return (d * bb)[:, ev].sum((-1, -2)) / bb[:, ev].float().sum((-1, -2)).clamp(min=1)
        def norm_pair(k, l):
            bb = (a_h[:, k] > 0.5) & (a_h[:, l] > 0.5); d = angle_deg(n_h[:, k], n_h[:, l])
            return (d * bb)[:, ev].sum((-1, -2)) / bb[:, ev].float().sum((-1, -2)).clamp(min=1)
        out["D_centroid"] = pairwise_mean(cent_pair, K); out["D_normal"] = pairwise_mean(norm_pair, K)
        def q_pair(k, l):
            den = (q_h[:, k] + q_h[:, l]).sum(-1); v = torch.where(den > 1e-6, (q_h[:, k] - q_h[:, l]).abs().sum(-1) / den.clamp(min=1e-9), torch.nan)
            return torch.nanmean(v[:, ev], -1)
        out["D_q"] = pairwise_mean(q_pair, K)
        out["D_Z"] = pairwise_mean(lambda k, l: (Z[:, k] - Z[:, l]).norm(dim=-1)[:, ev].mean(-1), K)
        cur["D_Z_t"] = pairwise_mean(lambda k, l: (Z[:, k] - Z[:, l]).norm(dim=-1), K)
    r2.update(a=a_h, m=m_h, p=p_h, n=n_h, a_soft=a_soft, q=q_h)
    return out, cur, r2


def event_rows(data, E, example_pos, R2, cur, a_true, protocol, name):
    """Per (event, sample) rows for the test events of the four classes."""
    rows = []
    a_h = R2["a"]                                                                     # (B, K, 64, 6) numpy
    K = a_h.shape[1]
    for ev in E.itertuples():
        if ev.example not in example_pos or ev.cls not in S.EVENT_CLASSES:
            continue
        i = example_pos[ev.example]; pre, post = int(ev.pre_frame), int(ev.post_frame)
        if post > S.T - 1:
            continue
        ap, aq = a_true[i, pre], a_true[i, post]; changes = bool((ap != aq).any())
        lo, hi = max(pre - EVENT_PRE, 0), min(post + EVENT_POST, S.T - 1)
        win = slice(pre, post + 1)
        for k in range(K):
            r = dict(dataset=data.ds, name=name, protocol=protocol, event_id=ev.event_id, example=int(ev.example), take_key=ev.take_key, cls=ev.cls,
                     pre_frame=pre, post_frame=post, k=k, participation_changes=changes, n_parts_changed=int((ap != aq).sum()))
            for key in ("dense", "part", "amount", "centroid", "normal", "wrench"):
                c = cur[key][i, k]
                r[f"{key}_post"] = float(c[post]); r[f"{key}_window"] = float(np.nanmean(c[win])); r[f"{key}_pre"] = float(c[pre]) if (protocol == "B" or pre > 0) else np.nan
            r["jitter_window"] = float(np.mean(cur["jitter"][i, k, pre:post]))
            if changes:
                hit = np.where((a_h[i, k, lo:hi + 1] > 0.5) == (aq[None] > 0.5), True, False).all(-1)
                t_hat = lo + int(np.argmax(hit)) if hit.any() else -1
                r["transition_realised"] = bool(hit.any()); r["transition_timing_error"] = (t_hat - post) if hit.any() else np.nan
                r["post_set_correct_at_post"] = bool(((a_h[i, k, post] > 0.5) == (aq > 0.5)).all())
            if "D_C_t" in cur and k == 0:
                r["D_C_window"] = float(np.mean(cur["D_C_t"][i, win])); r["D_Z_window"] = float(np.mean(cur["D_Z_t"][i, win]))
                r["D_C_post"] = float(cur["D_C_t"][i, post]); r["D_Z_post"] = float(cur["D_Z_t"][i, post])
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--name", required=True)
    ap.add_argument("--protocol", required=True, choices=S.PROTOCOLS); ap.add_argument("--part", default="test", choices=["test", "val"])
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    suffix = a.protocol + ("" if a.part == "test" else "_val")
    data = SeqData(ds, dev)
    te = data.idx[a.part]
    if a.name == "GT":
        P_all = data.C[te][:, None].cpu().numpy(); example = te
    elif a.name == "PERSIST":
        if a.protocol == "A":
            s0 = data.C[te, 0][:, None]                                                     # (B, 1, 512)
        else:
            z = np.load(S.s0_path(ds)); assert np.array_equal(z[f"example_{a.part}"], te); s0 = torch.from_numpy(z[f"s0_{a.part}"]).to(dev)
        P_all = s0[:, :, None].expand(-1, -1, S.T, -1).cpu().numpy(); example = te
    else:
        z = np.load(S.preds_path(ds, a.name, suffix)); P_all = z["pred"].astype(np.float32); example = z["example"]
        assert np.array_equal(example, te)
    B, K = P_all.shape[:2]
    outs, curs, r2s = {}, {}, {}
    for i in range(0, B, a.chunk):
        n = torch.from_numpy(example[i:i + a.chunk]).to(dev)
        P = torch.from_numpy(P_all[i:i + a.chunk]).to(dev)
        o, c, r = evaluate_chunk(data, n, P, a.protocol)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
        for k, v in c.items():
            curs.setdefault(k, []).append(v.cpu().numpy())
        for k, v in r.items():
            r2s.setdefault(k, []).append(v.cpu().numpy())
    outs = {k: np.concatenate(v) for k, v in outs.items()}; curs = {k: np.concatenate(v) for k, v in curs.items()}; r2s = {k: np.concatenate(v) for k, v in r2s.items()}
    a_true = data.a[torch.from_numpy(example).to(dev)].cpu().numpy()
    E = S.load_events(ds) if a.part == "test" else pd.DataFrame()
    rows = event_rows(data, E, {int(e): i for i, e in enumerate(example)}, r2s, curs, a_true, a.protocol, a.name) if len(E) else []
    out = S.metrics_path(ds, a.name, suffix); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, example=example, take=data.take[example], K=K, protocol=a.protocol, name=a.name,
                        **outs, **{f"curve_{k}": v.astype(np.float32) for k, v in curs.items()},
                        r2_a=r2s["a"].astype(np.int8), r2_m=r2s["m"].astype(np.float16), r2_p=r2s["p"].astype(np.float16), r2_n=r2s["n"].astype(np.float16),
                        r2_a_soft=r2s["a_soft"].astype(np.float16), q=r2s["q"].astype(np.float16), a_true=a_true.astype(np.int8))
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + "_events.csv"), index=False)
    summ = {k: (float(np.nanmean(v[:, 0])), float(np.nanmean(v)), float(np.nanmean(np.nanmin(v, 1)))) for k, v in outs.items() if v.ndim == 2 and k not in ("tp", "fp", "fn")}
    log.info("%s %s %s (%d x %d): K1 / mean / best  %s  (%d event rows, %.0f s)", ds, a.name, suffix, B, K,
             {k: tuple(round(x, 3) for x in v) for k, v in summ.items() if k in ("E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "jitter_dense", "D_C", "D_Z", "D_q")}, len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
