#!/usr/bin/env python
"""Experiment D — residual predictability after the boundary.
    CUDA_VISIBLE_DEVICES=<gpu> python residual.py --dataset taco --zstar R2 --seed 0

Decoder   D(Z*_t) -> [C_t, H_t] (standardised with the scalar scales; an MLP d -> 512 -> 512 -> 812), fitted framewise
          on the TRAIN frames, early-stopped on validation. Residual r_t = [C_t, H_t] - D(Z*_t) on every stored frame.
Predictors of the future residual r_{t+h}, h in (1, 4, 8), the temporal recipe of Experiment B (same trunk, 812 x 3 outputs):
  history   FULL window (t-7 .. t) + G + tau_local            causal
  condmean  Z*_t (one frame) + G + tau_local                  the residual's conditional mean given the current structured state
  oracle    FULL window + G + tau_local + GT Z*_{t+h}         NON-CAUSAL diagnostic: is the dense realisation determined once the
                                                              future structured state is known?
  history_delta / oracle_delta   the same inputs plus the residual window r_{t-7..t} (a function of the past full state and the
                                 fixed decoder), predicting the CHANGE of the residual on top of r_t — i.e. starting from the
                                 last-residual baseline, so that "predictable beyond its own persistence" is tested directly.
Baselines zero (r^ = 0) and last (r^ = r_t).
Metrics   per frame and block (C: 512 dims, H: 300 dims): mean squared residual error, the residual's own energy (zero
          baseline) and the last-residual error; aggregated as error-sum ratios (1 - R^2 against the zero baseline).
Writes <ds>/residual/{decoder_<zstar>_seed<s>.json, metrics_<zstar>_<model>_seed<s>.npz}.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sv_common as S
from sv_data import Data
from train_temporal import CFG, RepPredictor, n_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("residual")
DEC = dict(width=512, dropout=0.1, lr=1e-3, wd=1e-4, batch=2048, max_epochs=60, patience=6)
D_C, D_H = 512, 300


class Decoder(nn.Module):
    def __init__(self, d_in, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, D_C + D_H))

    def forward(self, x):
        return self.net(x)


def dense_std(D, n, i):
    """standardised [C, H] at stored frame index i (0 .. 79)."""
    return torch.cat([D.std["C"][n, i], D.std["H"][n, i]], -1)


def fit_decoder(D, zstar, seed, dev):
    torch.manual_seed(seed)
    n_tr, t_tr = D.frames("train"); n_va, t_va = D.frames("val")
    net = Decoder(S.REP_DIM[zstar], DEC["width"], DEC["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=DEC["lr"], weight_decay=DEC["wd"])
    g = torch.Generator(device=dev); g.manual_seed(seed)
    best, best_state, bad = np.inf, None, 0
    for ep in range(DEC["max_epochs"]):
        net.train()
        for _ in range(max(1, len(n_tr) // DEC["batch"])):
            sel = torch.randint(0, len(n_tr), (DEC["batch"],), device=dev, generator=g); n, t = n_tr[sel], t_tr[sel]
            loss = ((net(D.frame(zstar, n, t)) - dense_std(D, n, t + S.PAD)) ** 2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = np.mean([float(((net(D.frame(zstar, n_va[i:i + 8192], t_va[i:i + 8192])) - dense_std(D, n_va[i:i + 8192], t_va[i:i + 8192] + S.PAD)) ** 2).mean()) for i in range(0, len(n_va), 8192)])
        if v < best - 1e-5:
            best, bad = v, 0; best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= DEC["patience"]:
                break
    net.load_state_dict(best_state); net.eval()
    return net, best, ep + 1


@torch.no_grad()
def residual_field(net, D, zstar):
    """r on every stored frame of every sequence: (N, NF, 812) on the device (float32)."""
    N = D.std["C"].shape[0]; R = torch.zeros((N, S.NF, D_C + D_H), device=D.device)
    ii = torch.arange(S.NF, device=D.device)
    for n0 in range(0, N, 64):
        n = torch.arange(n0, min(n0 + 64, N), device=D.device)
        nn_ = n.repeat_interleave(S.NF); tt = ii.repeat(len(n))
        z = torch.cat([D.std[b][nn_, tt] for b in S.LADDER[zstar]], -1)
        R[n0:n0 + len(n)] = (dense_std(D, nn_, tt) - net(z)).view(len(n), S.NF, -1)
    return R


def decoder_r2(R, D, part):
    """1 - residual energy / total energy of the standardised dense state on the frames of a split part, per block."""
    idx = torch.from_numpy(D.idx[part]).to(D.device); sl = slice(S.PAD, S.PAD + S.T)
    out = {}
    for name, cols in (("C", slice(0, D_C)), ("H", slice(D_C, D_C + D_H))):
        r = R[idx][:, sl, cols]; x = torch.cat([D.std["C"][idx][:, sl], D.std["H"][idx][:, sl]], -1)[..., cols]
        out[name] = float(1 - r.pow(2).sum() / (x - x.mean((0, 1))).pow(2).sum())
    return out


class ResidualPredictor(nn.Module):
    def __init__(self, d_in, d_static, d_traj, n_h):
        super().__init__()
        self.inner = RepPredictor(d_in, d_static, d_traj, D_C + D_H, n_h)

    def forward(self, x, s, traj):
        return self.inner(x, s, traj)


def base_model(model):
    return model[:-6] if model.endswith("_delta") else model


def residual_window(R, n, t):
    """r_{t-7 .. t} flattened (B, HIST * 812); stored index t + PAD - HIST + 1 .. t + PAD."""
    idx = t[:, None] + S.PAD + torch.arange(-S.HIST + 1, 1, device=t.device)[None]
    return R[n[:, None], idx].reshape(len(n), -1)


def inputs(model, D, R, zstar, n, t, h_list):
    base = base_model(model)
    x = [D.window("FULL", n, t)] if base in ("history", "oracle") else [D.frame(zstar, n, t)]
    if base == "oracle":
        for h in h_list:                                                   # GT future structured state, one per horizon
            th = torch.clamp(t + h, max=S.T - 1)
            x.append(D.frame(zstar, n, th))
    if model.endswith("_delta"):
        x.append(residual_window(R, n, t))
    return torch.cat(x, -1)


def predict(net, model, D, R, zstar, n, t):
    """(B, n_h, 812): the residual prediction; the delta variants add the last residual r_t."""
    out = net(inputs(model, D, R, zstar, n, t, S.HORIZONS), *D.cond(n, t))
    if model.endswith("_delta"):
        out = out + R[n, t + S.PAD][:, None, :]
    return out


def loss_fn(pred, R, n, t):
    total = 0.0
    for j, h in enumerate(S.HORIZONS):
        th = torch.clamp(t + h, max=S.T - 1); v = (t + h <= S.T - 1).float()
        l = ((pred[:, j] - R[n, th + S.PAD]) ** 2).mean(1)
        total = total + (l * v).sum() / v.sum().clamp(min=1)
    return total


@torch.no_grad()
def evaluate(net, D, R, zstar, model, n_all, t_all, bs=4096):
    net.eval(); tot = 0.0
    for i in range(0, len(n_all), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]
        tot += float(loss_fn(predict(net, model, D, R, zstar, n, t), R, n, t)) * len(n)
    return tot / len(n_all)


@torch.no_grad()
def test_metrics(net, D, R, zstar, model, out_path, bs=4096):
    n_te, t_te = D.pairs("test"); N = len(n_te); nh = len(S.HORIZONS)
    M = {k: np.full((N, nh), np.nan, np.float32) for k in ("mse_C", "mse_H", "zero_C", "zero_H", "last_C", "last_H", "valid")}
    for i in range(0, N, bs):
        n, t = n_te[i:i + bs], t_te[i:i + bs]; sl = slice(i, i + len(n))
        pred = predict(net, model, D, R, zstar, n, t) if net is not None else None
        for j, h in enumerate(S.HORIZONS):
            th = torch.clamp(t + h, max=S.T - 1); r = R[n, th + S.PAD]; r0 = R[n, t + S.PAD]
            M["valid"][sl, j] = (t + h <= S.T - 1).float().cpu().numpy()
            p = pred[:, j] if pred is not None else torch.zeros_like(r)
            M["mse_C"][sl, j] = ((p[:, :D_C] - r[:, :D_C]) ** 2).mean(1).cpu().numpy(); M["mse_H"][sl, j] = ((p[:, D_C:] - r[:, D_C:]) ** 2).mean(1).cpu().numpy()
            M["zero_C"][sl, j] = (r[:, :D_C] ** 2).mean(1).cpu().numpy(); M["zero_H"][sl, j] = (r[:, D_C:] ** 2).mean(1).cpu().numpy()
            M["last_C"][sl, j] = ((r0[:, :D_C] - r[:, :D_C]) ** 2).mean(1).cpu().numpy(); M["last_H"][sl, j] = ((r0[:, D_C:] - r[:, D_C:]) ** 2).mean(1).cpu().numpy()
    np.savez_compressed(out_path, example=n_te.cpu().numpy(), t=t_te.cpu().numpy(), take=D.take[n_te.cpu().numpy()], horizons=np.array(S.HORIZONS), **M)


def train_predictor(D, R, zstar, model, seed, dev, max_steps=None):
    torch.manual_seed(seed)
    base = base_model(model)
    d_in = S.REP_DIM["FULL"] * S.HIST if base == "history" else (S.REP_DIM["FULL"] * S.HIST + len(S.HORIZONS) * S.REP_DIM[zstar] if base == "oracle" else S.REP_DIM[zstar])
    if model.endswith("_delta"):
        d_in += S.HIST * (D_C + D_H)
    net = ResidualPredictor(d_in, D.d_static(), D.d_traj(), len(S.HORIZONS)).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"]); lr_scale = [1.0]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]) * lr_scale[0])
    n_tr, t_tr = D.pairs("train"); n_va, t_va = D.pairs("val")
    g = torch.Generator(device=dev); g.manual_seed(seed)
    best, best_state, bad, best_step, t0 = np.inf, None, 0, 0, time.time()
    for step in range(1, (max_steps or CFG["max_steps"]) + 1):
        net.train()
        sel = torch.randint(0, len(n_tr), (CFG["batch"],), device=dev, generator=g); n, t = n_tr[sel], t_tr[sel]
        loss = loss_fn(predict(net, model, D, R, zstar, n, t), R, n, t)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"]); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        if step % CFG["eval_every"] == 0:
            v = evaluate(ema, D, R, zstar, model, n_va, t_va)
            if v < best - 1e-5:
                best, bad, best_step = v, 0, step; best_state = {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
                if bad % CFG["plateau_patience"] == 0:
                    lr_scale[0] *= CFG["lr_factor"]
            if bad >= CFG["patience"]:
                break
    ema.load_state_dict(best_state)
    log.info("  %s: best val %.4f at step %d of %d (%.0f s, %d params)", model, best, best_step, step, time.time() - t0, n_params(net))
    return ema, dict(best_val=best, best_step=best_step, steps=step, n_params=n_params(net))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--zstar", required=True, choices=S.REPS)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--models", nargs="*", default=["history", "condmean", "oracle"], choices=["history", "condmean", "oracle", "history_delta", "oracle_delta"]); ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--tag", default="", help="suffix for smoke runs")
    a = ap.parse_args(); ds = a.dataset; tag = f"_{a.tag}" if a.tag else ""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    D = Data(ds, dev)
    out = S.ds_out(ds) / "residual"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    dec, val, ep = fit_decoder(D, a.zstar, a.seed, dev)
    R = residual_field(dec, D, a.zstar)
    info = dict(dataset=ds, zstar=a.zstar, seed=a.seed, decoder_val=val, decoder_epochs=ep, decoder_r2_test=decoder_r2(R, D, "test"), decoder_r2_train=decoder_r2(R, D, "train"),
                residual_energy_test={k: float(v) for k, v in zip(("C", "H"), (R[torch.from_numpy(D.idx["test"]).to(dev)][:, S.PAD:S.PAD + S.T, :D_C].pow(2).mean(), R[torch.from_numpy(D.idx["test"]).to(dev)][:, S.PAD:S.PAD + S.T, D_C:].pow(2).mean()))})
    log.info("%s Z* = %s seed %d: decoder val %.4f (%d epochs), test R2 %s", ds, a.zstar, a.seed, val, ep, info["decoder_r2_test"])
    test_metrics(None, D, R, a.zstar, "zero", out / f"metrics_{a.zstar}_zero_seed{a.seed}{tag}.npz")
    for model in a.models:
        net, st = train_predictor(D, R, a.zstar, model, a.seed, dev, a.max_steps)
        info[model] = st
        test_metrics(net, D, R, a.zstar, model, out / f"metrics_{a.zstar}_{model}_seed{a.seed}{tag}.npz")
    info["seconds"] = time.time() - t0
    path = out / f"decoder_{a.zstar}_seed{a.seed}{tag}.json"
    if path.exists():                                                      # keep the entries of the models run earlier
        prev = json.load(open(path)); prev.update(info); info = prev
    S.write_json(path, info)
    log.info("done %s Z* = %s seed %d (%.0f s)", ds, a.zstar, a.seed, time.time() - t0)


if __name__ == "__main__":
    main()
