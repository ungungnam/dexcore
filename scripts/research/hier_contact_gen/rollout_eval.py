#!/usr/bin/env python
"""Free-rollout evaluation of one fold:  DC_DATASET=<ds> python rollout_eval.py --split take --fold 0

Variants (all on the test examples of the fold, T = 64 frames, raw contact units):
  B0  static_gt          C_t = C_0^GT
  B1  gtinit_vf          C_0^GT, then Euler rollout of the vector field
  B2  samplerG_vf        S_0 ~ p(S_0 | G),            K = 10 samples, each rolled out
  B3  samplerGT_vf       S_0 ~ p(S_0 | G, tau_local,0), K = 10 samples, each rolled out
  +   samplerG_static / samplerGT_static   the sampled S_0 held fixed (sampler quality alone)
  (dense / vf_noise variants are evaluated only when their checkpoints exist; not part of the run)
Best-of-K: k* = argmin_k E_C over the full rollout, and EVERY metric of the row is read from that
k*. Rows: K = 1 (sample 0), 5 (samples 0-4), 10, and the average over the 10 samples.
Per-example metrics (raw units):  E_C = mean_t ||C^_t - C_t||,  E_dC = mean_t ||dC^_t - dC_t||,
  dmag_pred / dmag_true = mean_t ||dC_t||,  pattern_L2 (frames with GT mass >= zero),  mass_abs,
  s0_err = ||C^_0 - C_0||,  plus the per-t curves of ||C^_t - C_t||, ||dC^_t - dC_t||, ||dC^_t||.
Writes OUT/eval/<split><k>/{per_example.csv, curves.npz, preds.npz}.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

import hc_common as H
import models as MD
from data import FoldData
from hc_common import C
from train import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("eval")
KS = (1, 5, 10)
N_SAMPLES = 10


def load(model, fd, dev, tag=""):
    ck = torch.load(H.CKPT / f"{fd.split}{fd.fold}" / f"{model}{tag}.pt", map_location=dev, weights_only=False)
    net = build(model, fd, ck.get("width"), ck.get("dropout", 0.0)).to(dev); net.load_state_dict(ck["state"]); net.eval()
    return net, ck


@torch.no_grad()
def rollout(vf, fd, n, c0):
    """Euler free rollout: c0 (B,512) standardised -> (B,T,512)."""
    out = [c0]; c = c0
    for t in range(fd.T - 1):
        tt = torch.full((len(n),), t, device=fd.device)
        c = c + vf(c, fd.S[n], fd.context(n, tt))
        out.append(c)
    return torch.stack(out, 1)


@torch.no_grad()
def dense_predict(net, fd, n):
    out = []
    for t in range(fd.T):
        tt = torch.full((len(n),), t, device=fd.device)
        out.append(net(fd.S[n], fd.context(n, tt)))
    return torch.stack(out, 1)


def metrics(pred, gt, thr, P):
    """pred, gt: (B,T,512) raw. Returns dict of (B,) arrays and (B,T)/(B,T-1) curves."""
    err_t = np.linalg.norm(pred - gt, axis=2)                          # (B,T)
    dp, dg = np.diff(pred, axis=1), np.diff(gt, axis=1)
    derr_t = np.linalg.norm(dp - dg, axis=2); dmag_p = np.linalg.norm(dp, axis=2); dmag_g = np.linalg.norm(dg, axis=2)
    pc = np.clip(pred, 0, None)
    mg, mp = gt.sum(2), pc.sum(2)
    nz = mg >= thr
    pat = np.linalg.norm(C.pattern(pc) - C.pattern(gt), axis=2)
    cen = np.linalg.norm(np.einsum("btk,kd->btd", C.pattern(pc), P) - np.einsum("btk,kd->btd", C.pattern(gt), P), axis=2)
    with np.errstate(invalid="ignore"):
        pat_m = np.where(nz.sum(1) > 0, (pat * nz).sum(1) / np.maximum(nz.sum(1), 1), np.nan)
        cen_m = np.where(nz.sum(1) > 0, (cen * nz).sum(1) / np.maximum(nz.sum(1), 1), np.nan)
    return dict(E_C=err_t.mean(1), E_dC=derr_t.mean(1), dmag_pred=dmag_p.mean(1), dmag_true=dmag_g.mean(1),
                pattern_L2=pat_m, centroid=cen_m, mass_abs=np.abs(mp - mg).mean(1), s0_err=err_t[:, 0], E_C_last16=err_t[:, -16:].mean(1),
                max_abs=np.abs(pred).max((1, 2)), neg_frac=(pred < -0.05 * gt.max()).mean((1, 2))), dict(err=err_t, derr=derr_t, dmag=dmag_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES); ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--part", default="test", choices=["test", "test_extra"], help="which manifest part to evaluate")
    ap.add_argument("--vf-tag", default="", help="checkpoint tag of the vector field to use as 'vf' (e.g. _unroll8)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    ck_vf = torch.load(H.CKPT / f"{a.split}{a.fold}" / f"vf{a.vf_tag}.pt", map_location="cpu", weights_only=False)
    fd = FoldData(a.split, a.fold, dev, stats=ck_vf["stats"])
    thr = C.zero_threshold()
    nets = {m: load(m, fd, dev, a.vf_tag if m == "vf" else "")[0] for m in ("sampler_G", "sampler_GT", "vf", "vf_noise", "dense")
            if (H.CKPT / f"{a.split}{a.fold}" / f"{m}{a.vf_tag if m == 'vf' else ''}.pt").exists()}
    log.info("models found: %s (vf tag '%s': %s)", sorted(nets), a.vf_tag, {k: v for k, v in ck_vf.items() if k in ("dropout", "wd", "input_noise", "unroll", "selection", "best_val", "best_step")})
    te = fd.idx[a.part]; n_all = torch.from_numpy(te).to(dev)
    meta = fd.meta.iloc[te].reset_index(drop=True)
    P_by_cat = fd.canonical
    gt_raw = fd.X_raw[te]                                                     # (B,T,512)
    B = len(te)
    preds, rows, curves = {}, [], {}
    out_dir = H.OUT / "eval" / (f"{a.split}{a.fold}" + ("" if a.part == "test" else "_extra")); out_dir.mkdir(parents=True, exist_ok=True)

    def run_batches(fn, bs=256):
        """fn(example indices, positional slice) -> tensor; concatenated on the CPU."""
        outs = []
        for i in range(0, B, bs):
            outs.append(fn(n_all[i:i + bs], slice(i, i + bs)).cpu())
        return torch.cat(outs)

    # deterministic variants
    C0 = fd.C[n_all, 0]
    preds["static_gt"] = np.repeat(gt_raw[:, :1], fd.T, 1)
    preds["gtinit_vf"] = fd.to_raw(run_batches(lambda n, p: rollout(nets["vf"], fd, n, fd.C[n, 0])))
    if "vf_noise" in nets:
        preds["gtinit_vfnoise"] = fd.to_raw(run_batches(lambda n, p: rollout(nets["vf_noise"], fd, n, fd.C[n, 0])))
    if "dense" in nets:
        preds["dense"] = fd.to_raw(run_batches(lambda n, p: dense_predict(nets["dense"], fd, n)))
    # samplers: K samples each, seeded per (fold, sampler, sample index)
    S0 = {}
    for si, sm in enumerate(("sampler_G", "sampler_GT")):
        samples = []
        for k in range(a.n_samples):
            g = torch.Generator(device=dev); g.manual_seed(100000 * (si + 1) + 1000 * a.fold + k)
            def f(n, p, k=k, g=g):
                traj = fd.context(n, torch.zeros(len(n), dtype=torch.long, device=dev)) if sm == "sampler_GT" else None
                return nets[sm].sample(fd.S[n], traj, n_steps=a.ddim_steps, generator=g)
            samples.append(run_batches(f))
        S0[sm] = torch.stack(samples, 1).to(dev)                               # (B,K,512) standardised
    log.info("%s %s%d: %d test examples, deterministic variants + %d samples x 2 samplers drawn (%.0fs)", H.DATASET, a.split, a.fold, B, a.n_samples, time.time() - t0)
    # per-category canonical points for the centroid metric
    P = np.stack([P_by_cat[c] for c in meta.category])                        # (B,512,3)

    def cat_metrics(pred):
        m, cv = {}, {}
        for c in np.unique(meta.category):
            sel = np.where(meta.category.values == c)[0]
            mm, cc = metrics(pred[sel], gt_raw[sel], thr, P_by_cat[c])
            for k, v in mm.items():
                m.setdefault(k, np.full(B, np.nan))[sel] = v
            for k, v in cv.items() if False else cc.items():
                cv.setdefault(k, np.full((B, v.shape[1]), np.nan))[sel] = v
        return m, cv

    def add(name, K, pred, m, cv):
        df = pd.DataFrame(m); df["model"] = name; df["K"] = K; df["example"] = te; df["group"] = meta.group.values
        df["take_key"] = meta.take_key.values; df["first_onset"] = meta.first_onset.values; df["leaves_contact"] = meta.leaves_contact.values
        df["category"] = meta.category.values; df["split"] = a.split; df["fold"] = a.fold
        rows.append(df); curves[(name, K)] = cv
        if pred is not None:
            preds[f"{name}__K{K}"] = pred

    for name in ("static_gt", "gtinit_vf", "gtinit_vfnoise", "dense"):
        if name in preds:
            m, cv = cat_metrics(preds[name]); add(name, 0, None, m, cv)
    # sampled variants: static hold and rollouts, with best-of-K selection on E_C
    for sm, tag in (("sampler_G", "samplerG"), ("sampler_GT", "samplerGT")):
        for vf_name, suffix in (("vf", "vf"), ("vf_noise", "vfnoise"), (None, "static")):
            if vf_name is not None and vf_name not in nets:
                continue
            per_k = []                                                       # list over samples of (pred_raw, m, cv)
            for k in range(a.n_samples):
                if vf_name is None:
                    pr = fd.to_raw(S0[sm][:, k]); pr = np.repeat(pr[:, None], fd.T, 1)
                else:
                    pr = fd.to_raw(run_batches(lambda n, p, k=k: rollout(nets[vf_name], fd, n, S0[sm][p, k])))
                m, cv = cat_metrics(pr); per_k.append((pr, m, cv))
            E = np.stack([p[1]["E_C"] for p in per_k], 1)                     # (B,K)
            name = f"{tag}_{suffix}"
            for K in KS:
                kstar = np.argmin(E[:, :K], 1)
                m = {key: np.array([per_k[kstar[i]][1][key][i] for i in range(B)]) for key in per_k[0][1]}
                cv = {key: np.stack([per_k[kstar[i]][2][key][i] for i in range(B)]) for key in per_k[0][2]}
                pred = np.stack([per_k[kstar[i]][0][i] for i in range(B)]) if K == max(KS) or K == 1 else None
                m["k_star"] = kstar
                add(name, K, pred, m, cv)
            # average over samples ("typical" sample quality)
            m = {key: np.mean([p[1][key] for p in per_k], 0) for key in per_k[0][1]}
            cv = {key: np.mean([p[2][key] for p in per_k], 0) for key in per_k[0][2]}
            add(name, -1, None, m, cv)
            log.info("  %s: E_C K1 %.3f K5 %.3f K10 %.3f mean %.3f | s0_err K1 %.3f K10 %.3f", name, E[:, 0].mean(), E[:, :5].min(1).mean(), E.min(1).mean(), E.mean(),
                     per_k[0][1]["s0_err"].mean(), np.array([per_k[np.argmin(E[i])][1]["s0_err"][i] for i in range(B)]).mean())
    R = pd.concat(rows, ignore_index=True)
    R.to_csv(out_dir / "per_example.csv", index=False)
    np.savez_compressed(out_dir / "curves.npz", **{f"{n}__K{K}__{key}": v for (n, K), cv in curves.items() for key, v in cv.items()}, example=te)
    np.savez_compressed(out_dir / "preds.npz", example=te, gt=gt_raw.astype(np.float16),
                        S0_G=fd.to_raw(S0["sampler_G"]).astype(np.float16), S0_GT=fd.to_raw(S0["sampler_GT"]).astype(np.float16),
                        **{k: v.astype(np.float16) for k, v in preds.items()})
    summ = R[R.K.isin([0, 1, 10])].groupby(["model", "K"])[["E_C", "E_dC", "dmag_pred", "dmag_true", "pattern_L2", "mass_abs", "s0_err"]].mean().round(3)
    log.info("%s %s%d summary (%d examples, %.0fs)\n%s", H.DATASET, a.split, a.fold, B, time.time() - t0, summ.to_string())


if __name__ == "__main__":
    main()
