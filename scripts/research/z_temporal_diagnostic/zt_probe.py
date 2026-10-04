#!/usr/bin/env python
"""Experiment A: train one local latent-dynamics probe  P_h(z_t, G, tau_t, tau_{t+h}) -> z_{t+h}  (one seed).
    CUDA_VISIBLE_DEVICES=5 python zt_probe.py --dataset taco --h 4 [--variant probe_notau]
Inputs: the CURRENT GT latent z_t (standardised), the static descriptor G (1032-D) and the object-trajectory windows of the two frames
(tau_local,t and tau_local,t+h: object states at -8 .. +8 around each frame, the conditioning of the previous studies).  Nothing about
the future contact, the future latent, the hand, R2 or the wrench enters.  Model: residual MLP around persistence,
z_hat_{t+h} = z_t + f(z_t, G, tau) (width 512, 4 pre-LN residual blocks with 4x GELU expansion, dropout 0.1, zero-initialised output:
the probe starts as the persistence predictor).  Loss: MSE in standardised z.  AdamW 1e-4, batch 1024, <= 30k steps, validation every
100 steps on all validation pairs, early stopping after >= 5k steps with patience 10; the best-validation state is saved and used.
Writes CKPT/<ds>/<variant>_h<h>_seed0.pt, <ds>/train_logs/<variant>_h<h>.csv, <ds>/preds/<variant>_h<h>.npz (test + val predictions).
"""
from __future__ import annotations

import argparse
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import zt_common as Z
from zt_data import Cache

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zt_probe")


class Block(nn.Module):
    def __init__(self, w, e, p):
        super().__init__()
        self.n = nn.LayerNorm(w); self.f1 = nn.Linear(w, e * w); self.f2 = nn.Linear(e * w, w); self.d = nn.Dropout(p)

    def forward(self, x):
        return x + self.d(self.f2(self.d(F.gelu(self.f1(self.n(x))))))


class Probe(nn.Module):
    def __init__(self, dz, d_static, d_tau, use_tau, width, depth, expansion, dropout):
        super().__init__()
        self.use_tau = use_tau
        self.inp = nn.Linear(dz + d_static + (2 * d_tau if use_tau else 0), width)
        self.blocks = nn.ModuleList([Block(width, expansion, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, dz)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z_t, S, tau_t, tau_h):
        x = torch.cat([z_t, S, tau_t, tau_h] if self.use_tau else [z_t, S], -1)
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return z_t + self.out(self.norm(h))


def build(cache, variant):
    return Probe(cache.dz, cache.d_static, cache.d_tau, Z.VARIANTS[variant]["tau"], **Z.PROBE)


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = min(1.0, (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"]))
    return cfg["lr"] * (cfg["final_lr_frac"] + (1 - cfg["final_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def predict(net, cache, r, t, h, bs=8192):
    net.eval(); out = []
    rr, tt = torch.from_numpy(r).to(cache.device), torch.from_numpy(t).to(cache.device)
    for i in range(0, len(r), bs):
        z_t, S, ta, tb, _ = cache.inputs(rr[i:i + bs], tt[i:i + bs], h)
        out.append(net(z_t, S, ta, tb))
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--h", type=int, required=True, choices=Z.HORIZONS); ap.add_argument("--variant", default="probe", choices=list(Z.VARIANTS))
    a = ap.parse_args(); ds, h, variant = a.dataset, a.h, a.variant; cfg = dict(Z.TRAIN)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(Z.SEED); np.random.seed(Z.SEED); t0 = time.time()
    cache = Cache(ds, dev)
    net = build(cache, variant).to(dev); n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr_r, tr_t = (torch.from_numpy(x).to(dev) for x in cache.pairs("train", h)); va_r, va_t = cache.pairs("val", h)
    va_target = cache.t_zs[torch.from_numpy(va_r).to(dev), torch.from_numpy(va_t + h).to(dev)]
    g = torch.Generator(device=dev); g.manual_seed(Z.SEED)
    val0 = float(F.mse_loss(predict(net, cache, va_r, va_t, h), va_target))                 # = persistence (zero-initialised output)
    rows, best, best_state, bad, best_step, stopped_by, run = [], np.inf, None, 0, 0, "max_steps", None
    for step in range(1, cfg["max_steps"] + 1):
        net.train()
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step, cfg)
        i = torch.randint(0, len(tr_r), (cfg["batch"],), device=dev, generator=g)
        z_t, S, ta, tb, tgt = cache.inputs(tr_r[i], tr_t[i], h)
        loss = F.mse_loss(net(z_t, S, ta, tb), tgt)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if step % cfg["eval_every"] == 0:
            v = float(F.mse_loss(predict(net, cache, va_r, va_t, h), va_target))
            rows.append(dict(step=step, train=run, val=v, lr=lr_at(step, cfg), time=time.time() - t0))
            if v < best - 1e-7:
                best, bad, best_step = v, 0, step; best_state = {k: x.detach().clone() for k, x in net.state_dict().items()}
            else:
                bad += 1
            if step >= cfg["min_steps"] and bad >= cfg["patience"]:
                stopped_by = "patience"; break
    net.load_state_dict(best_state)
    out = {}
    for split in ("test", "val"):
        r, t = cache.pairs(split, h); out[f"{split}_r"], out[f"{split}_t"] = r, t
        out[f"{split}_z_hat"] = predict(net, cache, r, t, h).cpu().numpy().astype(np.float32)
    p = Z.preds_path(ds, variant, h); p.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, h=h, variant=variant, **out)
    ck = Z.ckpt_path(ds, variant, h); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, dataset=ds, h=h, variant=variant, seed=Z.SEED, cfg=cfg, probe=Z.PROBE, n_params=n_params, best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by,
                    val_persistence=val0, d_in=net.inp.in_features, teacher_md5=cache.md5, seconds=time.time() - t0), ck)
    Z.train_log_path(ds, variant, h).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(Z.train_log_path(ds, variant, h), index=False)
    log.info("done %s %s h=%d: %d params, val MSE %.4f at step %d of %d (%s); persistence val MSE %.4f; %.0f s -> %s", ds, variant, h, n_params, best, best_step, step, stopped_by, val0, time.time() - t0, ck)


if __name__ == "__main__":
    main()
