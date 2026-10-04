#!/usr/bin/env python
"""Experiment A — functional sufficiency: R_i(t) -> q_t (the 76-D support-function wrench profile), framewise.
    CUDA_VISIBLE_DEVICES=<gpu> python probe_functional.py --dataset taco

Probe: a 2-layer MLP (d -> 256 -> 256 -> 76, SiLU) on the standardised single-frame representation, trained on the
TRAIN frames in contact (Q_t > 0) with the z-scored profile as target (MSE), early-stopped on the validation frames,
evaluated on the TEST frames in contact; 3 seeds.
  mlpG   (primary)      input = representation + the task's static object descriptor G (the same 1032-D standardised
                        S of every temporal model: canonical-space surface descriptor, radius, extent, hand / role flags);
                        dropout / weight decay chosen on validation from a 2 x 2 grid (first seed; the other seeds reuse the choice), the same grid for every representation
  mlp    (sensitivity)  representation only, fixed dropout 0.1 / wd 1e-4
  ridgeG, ridge         closed-form ridge (lambda chosen on validation) with / without G
Secondary target: the strict LP profile (mlpG, seed 0). G is included because the wrench profile is a property of the
hand-object pair: the dense canonical map has no object geometry or finger identity of its own.
Per-frame metrics (raw units): relative L1  sum|q^ - q| / sum q; cosine; nRMSE (RMSE / train sd of q); Q error
|Q^ - Q| / Q; overlap mean_u min(q^ / q, 1). R^2 = 1 - SSE / SST over all outputs (test).
Writes <ds>/functional/{probe_<rep>_<probe>_seed<s>.npz (per-frame metrics on the test frames), functional_probe.csv}.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sv_common as S
from sv_data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probe")
CFG = dict(width=256, dropout=0.1, lr=1e-3, wd=1e-4, batch=2048, max_epochs=60, patience=6)
GRID = [dict(dropout=0.1, wd=1e-4), dict(dropout=0.1, wd=1e-2), dict(dropout=0.3, wd=1e-4), dict(dropout=0.3, wd=1e-2)]


def inp(D, rep, n, t, with_g):
    x = D.frame(rep, n, t)
    return torch.cat([x, D.fd.S[n]], 1) if with_g else x


class MLP(nn.Module):
    def __init__(self, d_in, d_out, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, d_out))

    def forward(self, x):
        return self.net(x)


def contact_frames(D, part, key):
    n, t = D.frames(part)
    q = D.tgt[key][n, t]
    keep = q.sum(1) > 1e-6
    return n[keep], t[keep]


def per_frame_metrics(qhat, q, sd_train):
    qhat = np.maximum(qhat, 0.0)
    rel_l1 = np.abs(qhat - q).sum(1) / np.maximum(q.sum(1), 1e-9)
    cos = (qhat * q).sum(1) / (np.linalg.norm(qhat, axis=1) * np.linalg.norm(q, axis=1) + 1e-9)
    nrmse = np.sqrt(((qhat - q) ** 2).mean(1)) / sd_train
    Q, Qh = q.mean(1), qhat.mean(1)
    q_err = np.abs(Qh - Q) / np.maximum(Q, 1e-9)
    overlap = np.minimum(qhat / np.maximum(q, 1e-9), 1.0)
    overlap = np.where(q > 1e-6, overlap, np.nan)
    return dict(rel_l1=rel_l1, cosine=cos, nrmse=nrmse, Q_err=q_err, overlap=np.nanmean(overlap, 1), abs_err=np.abs(qhat - q).mean(1))


def fit_mlp(D, rep, key, seed, dev, with_g=False, dropout=None, wd=None):
    torch.manual_seed(seed); np.random.seed(seed)
    n_tr, t_tr = contact_frames(D, "train", key); n_va, t_va = contact_frames(D, "val", key)
    d_in = S.REP_DIM[rep] + (D.fd.S.shape[1] if with_g else 0)
    net = MLP(d_in, S.N_DIR, CFG["width"], CFG["dropout"] if dropout is None else dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"] if wd is None else wd)
    mu, sd = D.tstats[key]
    g = torch.Generator(device=dev); g.manual_seed(seed)
    best, best_state, bad, ep = np.inf, None, 0, 0
    steps_per_epoch = max(1, len(n_tr) // CFG["batch"])
    for ep in range(CFG["max_epochs"]):
        net.train()
        for _ in range(steps_per_epoch):
            sel = torch.randint(0, len(n_tr), (CFG["batch"],), device=dev, generator=g)
            n, t = n_tr[sel], t_tr[sel]
            pred = net(inp(D, rep, n, t, with_g)); target = (D.tgt[key][n, t] - mu) / sd
            loss = ((pred - target) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = 0.0
            for i in range(0, len(n_va), 8192):
                n, t = n_va[i:i + 8192], t_va[i:i + 8192]
                v += float((((net(inp(D, rep, n, t, with_g)) - (D.tgt[key][n, t] - mu) / sd) ** 2).mean(1)).sum())
            v /= max(len(n_va), 1)
        if v < best - 1e-5:
            best, bad = v, 0; best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best, ep + 1


def fit_mlp_grid(D, rep, key, seed, dev, with_g=True):
    """The 2 x 2 dropout / weight-decay grid, chosen on the validation loss."""
    best = None
    for g in GRID:
        net, val, ep = fit_mlp(D, rep, key, seed, dev, with_g, g["dropout"], g["wd"])
        if best is None or val < best[1]:
            best = (net, val, ep, g)
    return best


@torch.no_grad()
def predict_mlp(net, D, rep, key, n, t, with_g=False):
    mu, sd = D.tstats[key]
    out = []
    for i in range(0, len(n), 8192):
        out.append((net(inp(D, rep, n[i:i + 8192], t[i:i + 8192], with_g)) * sd + mu).cpu().numpy())
    return np.concatenate(out)


def fit_ridge(D, rep, key, dev, with_g=False):
    """Closed-form ridge on the standardised representation; lambda from a grid on the validation frames."""
    n_tr, t_tr = contact_frames(D, "train", key); n_va, t_va = contact_frames(D, "val", key)
    X = torch.cat([inp(D, rep, n_tr[i:i + 16384], t_tr[i:i + 16384], with_g) for i in range(0, len(n_tr), 16384)]).double()
    mu, sd = D.tstats[key]
    Y = ((D.tgt[key][n_tr, t_tr] - mu) / sd).double()
    Xv = torch.cat([inp(D, rep, n_va[i:i + 16384], t_va[i:i + 16384], with_g) for i in range(0, len(n_va), 16384)]).double()
    Yv = ((D.tgt[key][n_va, t_va] - mu) / sd).double()
    xm = X.mean(0); ym = Y.mean(0); Xc = X - xm; Yc = Y - ym
    G = Xc.T @ Xc; B = Xc.T @ Yc
    best = (np.inf, None, None)
    for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        Wt = torch.linalg.solve(G + lam * len(X) * torch.eye(G.shape[0], device=dev, dtype=G.dtype), B)
        v = float((((Xv - xm) @ Wt + ym - Yv) ** 2).mean())
        if v < best[0]:
            best = (v, lam, Wt)
    return dict(W=best[2], xm=xm, ym=ym, lam=best[1], val=best[0])


@torch.no_grad()
def predict_ridge(model, D, rep, key, n, t, with_g=False):
    mu, sd = D.tstats[key]
    out = []
    for i in range(0, len(n), 16384):
        X = inp(D, rep, n[i:i + 16384], t[i:i + 16384], with_g).double()
        out.append((((X - model["xm"]) @ model["W"] + model["ym"]).float() * sd + mu).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--reps", nargs="*", default=S.REPS)
    ap.add_argument("--seeds", nargs="*", type=int, default=list(S.SEEDS)); ap.add_argument("--no-strict", action="store_true"); ap.add_argument("--no-ridge", action="store_true")
    ap.add_argument("--variants", nargs="*", default=["mlpG", "mlp"], help="mlpG: + G, validation-chosen regularisation (primary); mlp: representation only, fixed")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = a.dataset; D = Data(ds, dev)
    out = S.ds_out(ds) / "functional"; out.mkdir(parents=True, exist_ok=True)
    rows = []
    n_te, t_te = contact_frames(D, "test", "q")
    take_te = D.take[n_te.cpu().numpy()]
    q_te = D.tgt["q"][n_te, t_te].cpu().numpy(); sd_q = float(D.tg["q"][D.idx["train"]].std())
    qs_te = D.tgt["q_strict"][n_te, t_te].cpu().numpy(); sd_qs = float(D.tg["q_strict"][D.idx["train"]].std())
    log.info("%s: %d test frames in contact (%d takes), %d train frames", ds, len(n_te), len(np.unique(take_te)), len(contact_frames(D, "train", "q")[0]))
    # reference: the train-mean profile
    qbar = D.tg["q"][D.idx["train"]].reshape(-1, S.N_DIR); qbar = qbar[qbar.sum(1) > 1e-6].mean(0)
    m0 = per_frame_metrics(np.repeat(qbar[None], len(q_te), 0), q_te, sd_q)
    np.savez_compressed(out / "probe_MEAN_const_seed0.npz", example=n_te.cpu().numpy(), t=t_te.cpu().numpy(), take=take_te, **m0)
    rows.append(dict(dataset=ds, rep="MEAN", probe="const", target="q", seed=0, dim=0, **{k: float(np.nanmean(v)) for k, v in m0.items()},
                     r2=float(1 - ((q_te - qbar) ** 2).sum() / ((q_te - q_te.mean(0)) ** 2).sum()), n_frames=len(q_te), epochs=0, val=np.nan))
    for rep in a.reps:
        for variant in a.variants:
            with_g = variant == "mlpG"
            for seed in a.seeds:
                t0 = time.time()
                if with_g and seed == a.seeds[0]:
                    net, best, ep, g = fit_mlp_grid(D, rep, "q", seed, dev, True); chosen = g       # the grid is run once (first seed); the other seeds reuse the chosen configuration
                elif with_g:
                    g = chosen; net, best, ep = fit_mlp(D, rep, "q", seed, dev, True, g["dropout"], g["wd"])
                else:
                    net, best, ep = fit_mlp(D, rep, "q", seed, dev, False); g = dict(dropout=CFG["dropout"], wd=CFG["wd"])
                qh = predict_mlp(net, D, rep, "q", n_te, t_te, with_g)
                m = per_frame_metrics(qh, q_te, sd_q)
                np.savez_compressed(out / f"probe_{rep}_{variant}_seed{seed}.npz", example=n_te.cpu().numpy(), t=t_te.cpu().numpy(), take=take_te, **m)
                r2 = float(1 - ((qh - q_te) ** 2).sum() / ((q_te - q_te.mean(0)) ** 2).sum())
                rows.append(dict(dataset=ds, rep=rep, probe=variant, target="q", seed=seed, dim=S.REP_DIM[rep], **{k: float(np.nanmean(v)) for k, v in m.items()},
                                 r2=r2, n_frames=len(q_te), epochs=ep, val=best, dropout=g["dropout"], wd=g["wd"]))
                log.info("  %s %s seed %d: rel L1 %.3f cos %.3f R2 %.3f (%d epochs, dropout %g wd %g, %.0f s)", rep, variant, seed, m["rel_l1"].mean(), m["cosine"].mean(), r2, ep, g["dropout"], g["wd"], time.time() - t0)
                if seed == a.seeds[0] and not a.no_strict and with_g:
                    g_s = chosen; net_s, best_s, ep_s = fit_mlp(D, rep, "q_strict", seed, dev, True, g_s["dropout"], g_s["wd"])
                    qh = predict_mlp(net_s, D, rep, "q_strict", n_te, t_te, True)
                    keep = qs_te.sum(1) > 1e-6
                    ms = per_frame_metrics(qh[keep], qs_te[keep], sd_qs)
                    np.savez_compressed(out / f"probe_{rep}_{variant}_strict_seed{seed}.npz", example=n_te.cpu().numpy()[keep], t=t_te.cpu().numpy()[keep], take=take_te[keep], **ms)
                    rows.append(dict(dataset=ds, rep=rep, probe=variant, target="q_strict", seed=seed, dim=S.REP_DIM[rep], **{k: float(np.nanmean(v)) for k, v in ms.items()},
                                     r2=float(1 - ((qh[keep] - qs_te[keep]) ** 2).sum() / ((qs_te[keep] - qs_te[keep].mean(0)) ** 2).sum()), n_frames=int(keep.sum()), epochs=ep_s, val=best_s, dropout=g_s["dropout"], wd=g_s["wd"]))
            if not a.no_ridge:
                model = fit_ridge(D, rep, "q", dev, with_g)
                qh = predict_ridge(model, D, rep, "q", n_te, t_te, with_g)
                m = per_frame_metrics(qh, q_te, sd_q)
                rname = "ridgeG" if with_g else "ridge"
                np.savez_compressed(out / f"probe_{rep}_{rname}_seed0.npz", example=n_te.cpu().numpy(), t=t_te.cpu().numpy(), take=take_te, **m)
                rows.append(dict(dataset=ds, rep=rep, probe=rname, target="q", seed=0, dim=S.REP_DIM[rep], **{k: float(np.nanmean(v)) for k, v in m.items()},
                                 r2=float(1 - ((qh - q_te) ** 2).sum() / ((q_te - q_te.mean(0)) ** 2).sum()), n_frames=len(q_te), epochs=0, val=model["val"], lam=model["lam"]))
                log.info("  %s %s (lambda %g): rel L1 %.3f cos %.3f", rep, rname, model["lam"], m["rel_l1"].mean(), m["cosine"].mean())
        pd.DataFrame(rows).to_csv(out / "functional_probe.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "functional_probe.csv", index=False)
    log.info("done %s", ds)


if __name__ == "__main__":
    main()
