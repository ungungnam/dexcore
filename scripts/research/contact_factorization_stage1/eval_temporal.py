#!/usr/bin/env python
"""Evaluation question 5 — temporal persistence of the learned codes (no temporal training: GT test trajectories encoded
frame by frame).    CUDA_VISIBLE_DEVICES=4 python eval_temporal.py --dataset taco
Codes: z / r (A3, A2), z (A1), h (A0); references: the teacher descriptor [u_R2 | u_q], the dense map C~, the dense residuals
C~ - C_bar~ (A3) and C~ - C_hat~ (A3). Per code (centred by its train mean):
  cos(x_t, x_0)                 mean over test sequences, t = 0..63
  displacement ratio            mean ||x_{t+h} - x_t|| / mean ||x_i - x_j|| over random frame pairs of different test sequences
  autocorrelation               sum_d Cov(x_t,d, x_{t+h},d) / sum_d Var(x_d) over the test frames
  initial dependence            ridge regression on train pairs (t, t+h): R^2 on test of x_{t+h} from x_t, from z3_{t+h}
                                (the current structure, A3's z) and from both; gain = R^2(both) - R^2(z3_{t+h})
Take-cluster bootstrap over sequences. Writes <ds>/temporal_persistence.csv and <ds>/temporal_curves.npz (figure 7).
"""
from __future__ import annotations

import argparse
import os
import logging

import numpy as np
import pandas as pd
import torch

import cf_common as S
from encode import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("temporal")
H_ALL = (1, 2, 4, 8, 16, 32, 63)


def ridge_r2(Xtr, Ytr, Xte, Yte, lam=1e-2):
    """Ridge with an intercept, lambda relative to the mean feature variance; R^2 on test around the train mean of Y."""
    mx, my = Xtr.mean(0), Ytr.mean(0); Xc, Yc = Xtr - mx, Ytr - my
    A = Xc.T @ Xc / len(Xc); lam_ = lam * float(A.diagonal().mean()); W = torch.linalg.solve(A + lam_ * torch.eye(len(A), device=A.device), Xc.T @ Yc / len(Xc))
    pred = (Xte - mx) @ W + my
    return float(1 - ((Yte - pred) ** 2).sum() / ((Yte - my) ** 2).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); ds = a.dataset; dev = torch.device(os.environ.get("CF_DEVICE", "cuda"))
    data = load_data(ds, dev, S.run_name("A3", a.seed))
    n_te, n_tr = data.idx["test"], data.idx["train"]; take = data.take[n_te]
    f = lambda x: torch.from_numpy(np.asarray(x, np.float32)).to(dev)
    codes = {}                                                                         # name -> (train (Ntr, 64, d), test (Nte, 64, d))
    codes["teacher"] = (torch.cat([data.u_r2[f(n_tr).long()], data.u_q[f(n_tr).long()]], -1), torch.cat([data.u_r2[f(n_te).long()], data.u_q[f(n_te).long()]], -1))
    codes["dense_C"] = (data.Cn[f(n_tr).long()], data.Cn[f(n_te).long()])
    z3 = None
    for model in S.ALL_MODELS:
        name = S.run_name(model, a.seed)
        if not S.latents_path(ds, name, "test").exists():
            continue
        tr, te = np.load(S.latents_path(ds, name, "train")), np.load(S.latents_path(ds, name, "test"))
        assert (tr["n"] == n_tr).all() and (te["n"] == n_te).all()
        codes["A0:h" if model == "A0" else f"{model}:z"] = (f(tr["z"]), f(te["z"]))
        if "r" in te:
            codes[f"{model}:r"] = (f(tr["r"]), f(te["r"]))
        if model == "A3":
            z3 = (f(tr["z"]), f(te["z"]))
        if "r" in te:
            codes[f"{model}:resid_z"] = (None, data.Cn[f(n_te).long()] - data.to_std(f(te["C_bar"].astype(np.float32))))
            codes[f"{model}:resid_full"] = (None, data.Cn[f(n_te).long()] - data.to_std(f(te["C_hat"].astype(np.float32))))
    rows, curves = [], {}
    g = torch.Generator(device=dev); g.manual_seed(0); Nte = len(n_te)
    pi = torch.randint(0, Nte, (20000,), device=dev, generator=g); pj = torch.randint(0, Nte, (20000,), device=dev, generator=g); ok = pi != pj
    ti = torch.randint(0, S.T, (20000,), device=dev, generator=g); tj = torch.randint(0, S.T, (20000,), device=dev, generator=g)
    for name, (Xtr, Xte) in codes.items():
        mu = (Xtr if Xtr is not None else Xte).reshape(-1, Xte.shape[-1]).mean(0); X = Xte - mu
        cosv = torch.nn.functional.cosine_similarity(X, X[:, :1], dim=-1)                 # (N, 64)
        curves[f"{name}__cos"] = cosv.mean(0).cpu().numpy()
        rand = (X[pi[ok], ti[ok]] - X[pj[ok], tj[ok]]).norm(dim=-1).mean()
        var = (X.reshape(-1, X.shape[-1]) - X.reshape(-1, X.shape[-1]).mean(0)).pow(2).sum(-1).mean()
        acf = []
        for h in range(0, S.T):
            if h == 0:
                acf.append(1.0); continue
            A_, B_ = X[:, :-h].reshape(-1, X.shape[-1]), X[:, h:].reshape(-1, X.shape[-1]); m_ = X.reshape(-1, X.shape[-1]).mean(0)
            acf.append(float(((A_ - m_) * (B_ - m_)).sum(-1).mean() / var))
        curves[f"{name}__acf"] = np.array(acf)
        for h in H_ALL:
            disp = (X[:, h:] - X[:, :-h]).norm(dim=-1).mean(1).cpu().numpy()             # per sequence
            m, lo, hi = S.cluster_bootstrap(disp / float(rand), take, n_boot=S.N_BOOT)
            rows.append(dict(dataset=ds, code=name, h=h, metric="displacement_ratio", value=m, lo=lo, hi=hi))
            c_h = cosv[:, h].cpu().numpy(); m, lo, hi = S.cluster_bootstrap(c_h, take, n_boot=S.N_BOOT)
            rows.append(dict(dataset=ds, code=name, h=h, metric="cos_to_frame0", value=m, lo=lo, hi=hi))
            rows.append(dict(dataset=ds, code=name, h=h, metric="autocorrelation", value=acf[h], lo=np.nan, hi=np.nan))
        # initial dependence (needs train codes)
        if Xtr is not None and z3 is not None and name not in ("teacher",):
            Xtr_c = Xtr - mu; stride = 2
            for h in (1, 4, 8, 16, 32):
                tt = torch.arange(0, S.T - h, stride, device=dev)
                A_tr, B_tr = Xtr_c[:, tt].reshape(-1, X.shape[-1]), Xtr_c[:, tt + h].reshape(-1, X.shape[-1]); Z_tr = (z3[0] - z3[0].reshape(-1, z3[0].shape[-1]).mean(0))[:, tt + h].reshape(-1, z3[0].shape[-1])
                A_te, B_te = X[:, tt].reshape(-1, X.shape[-1]), X[:, tt + h].reshape(-1, X.shape[-1]); Z_te = (z3[1] - z3[0].reshape(-1, z3[0].shape[-1]).mean(0))[:, tt + h].reshape(-1, z3[1].shape[-1])
                r_self = ridge_r2(A_tr, B_tr, A_te, B_te); r_z = ridge_r2(Z_tr, B_tr, Z_te, B_te); r_both = ridge_r2(torch.cat([A_tr, Z_tr], -1), B_tr, torch.cat([A_te, Z_te], -1), B_te)
                r_persist = float(1 - ((B_te - A_te) ** 2).sum() / ((B_te - B_tr.mean(0)) ** 2).sum())
                for metric, v in (("r2_from_past_self", r_self), ("r2_from_current_z3", r_z), ("r2_from_both", r_both), ("r2_gain_past_beyond_z3", r_both - r_z), ("r2_persistence_copy", r_persist)):
                    rows.append(dict(dataset=ds, code=name, h=h, metric=metric, value=v, lo=np.nan, hi=np.nan))
        log.info("%s %s: acf h=1/8/32 %.3f/%.3f/%.3f disp ratio h=8 %.3f", ds, name, acf[1], acf[8], acf[32], float(np.mean(((X[:, 8:] - X[:, :-8]).norm(dim=-1) / rand).cpu().numpy())))
    pd.DataFrame(rows).to_csv(S.ds_out(ds) / "temporal_persistence.csv", index=False)
    np.savez_compressed(S.ds_out(ds) / "temporal_curves.npz", **curves)


if __name__ == "__main__":
    main()
