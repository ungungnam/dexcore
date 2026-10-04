#!/usr/bin/env python
"""Train one model of one fold:  DC_DATASET=<ds> python train.py --split take --fold 0 --model <m>
  sampler_G    p(S_0 | G)                      DDPM, static condition only
  sampler_GT   p(S_0 | G, tau_local,0)         same net, + the 9 object states around t = 0
  vf           v_phi(C_t, G, tau_local,t)      teacher-forced one-step L2 on GT transitions
  vf_noise     same, with Gaussian input noise on C_t (sigma = NOISE_FRAC x rms one-step delta)
  dense        f(G, tau_local,t) -> C_t        the existing dense predictor (family C), pooled
Optimiser AdamW (lr 3e-4, wd 1e-4, warm-up then constant, halved on validation plateaus), EMA 0.999,
early stopping on the validation loss (checked every EVAL_EVERY steps; stop after PATIENCE checks
without improvement or when the LR has decayed 32x). Writes
  CKPT/<split><k>/<model>.pt        weights (EMA), normalisation stats, config, val curve
  OUT/train_logs/<split><k>_<model>.csv
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import pandas as pd
import torch

import hc_common as H
import models as MD
from data import FoldData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")
CFG = dict(lr=3e-4, wd=1e-4, warmup=500, max_steps=dict(sampler_G=200000, sampler_GT=200000, vf=200000, vf_noise=200000, dense=200000),
           batch=dict(sampler_G=256, sampler_GT=256, vf=1024, vf_noise=1024, dense=1024), ema=0.999, eval_every=500,
           plateau_patience=4, lr_factor=0.5, min_lr_factor=1 / 32, patience=12,
           width=1024, blocks=dict(sampler_G=4, sampler_GT=4, vf=3, vf_noise=3, dense=3), noise_frac=0.5, val_timesteps=[50, 150, 250, 350, 450, 550, 650, 750, 850, 950])
# Convergence rule: constant LR after warm-up; when the validation loss has not improved for
# plateau_patience checks the LR is halved; training stops when it has not improved for `patience`
# checks (12 x 500 = 6000 steps) or the LR has fallen below lr * min_lr_factor. max_steps is only a
# safety cap; a run that hits it is flagged converged = False.


def build(model, fd, width=None, dropout=0.0):
    d_static, d_traj = fd.S.shape[1], len(H.LOCAL_OFFSETS) * fd.D
    width = width or CFG["width"]
    if model == "sampler_G":
        return MD.Diffusion(d_static, 0, width, CFG["blocks"][model], dropout)
    if model == "sampler_GT":
        return MD.Diffusion(d_static, d_traj, width, CFG["blocks"][model], dropout)
    if model in ("vf", "vf_noise"):
        return MD.VectorField(d_static, d_traj, width, CFG["blocks"][model], dropout)
    if model == "dense":
        return MD.Dense(d_static, d_traj, width, CFG["blocks"][model], dropout)
    raise ValueError(model)


class Batcher:
    """Uniform minibatches over the flat index set of a part; sampler: examples, vf/dense: (n, t)."""
    def __init__(self, model, fd, part, seed):
        self.model, self.fd = model, fd
        self.g = torch.Generator(device=fd.device); self.g.manual_seed(seed)
        if model.startswith("sampler"):
            self.n = torch.from_numpy(fd.idx[part]).to(fd.device); self.t = torch.zeros_like(self.n)
        elif model.startswith("vf"):
            self.n, self.t = fd.frame_pairs(part)
        else:
            self.n, self.t = fd.frames(part)
        self.size = len(self.n)

    def inputs(self, sel):
        n, t = self.n[sel], self.t[sel]
        fd = self.fd
        s = fd.S[n]; traj = fd.context(n, t)
        if self.model == "sampler_G":
            return dict(x0=fd.C[n, 0], s=s)
        if self.model == "sampler_GT":
            return dict(x0=fd.C[n, 0], s=s, traj=traj)
        if self.model.startswith("vf"):
            return dict(c=fd.C[n, t], target=fd.C[n, t + 1] - fd.C[n, t], s=s, traj=traj)
        return dict(target=fd.C[n, t], s=s, traj=traj)

    def random(self, bs):
        return self.inputs(torch.randint(0, self.size, (bs,), device=self.fd.device, generator=self.g))

    def all(self, bs=8192):
        for i in range(0, self.size, bs):
            yield self.inputs(torch.arange(i, min(i + bs, self.size), device=self.fd.device))


def unrolled_loss(net, fd, n, t, k, noise_sd=0.0):
    """Free rollout of k steps from the GT C_t (n, t: index tensors), L2 against GT C_{t+1..t+k}.
    t is clamped so that t + k <= T - 1."""
    t = torch.clamp(t, max=fd.T - 1 - k)
    c = fd.C[n, t]
    if noise_sd > 0:
        c = c + noise_sd * torch.randn_like(c)
    s = fd.S[n]; loss = 0.0
    for j in range(k):
        c = c + net(c, s, fd.context(n, t + j))
        loss = loss + ((c - fd.C[n, t + j + 1]) ** 2).sum(1).mean()
    return loss / k


@torch.no_grad()
def val_rollout(net, fd, part="val", bs=512):
    """Mean over val sequences of the free-rollout E_C (standardised units) from the GT C_0."""
    net.eval(); tot = 0.0
    n_all = torch.from_numpy(fd.idx[part]).to(fd.device)
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; c = fd.C[n, 0]; err = torch.zeros(len(n), device=fd.device)
        for t in range(fd.T - 1):
            c = c + net(c, fd.S[n], fd.context(n, torch.full((len(n),), t, device=fd.device)))
            err += (c - fd.C[n, t + 1]).norm(dim=1)
        tot += float(err.sum()) / fd.T
    net.train()
    return tot / len(n_all)


def step_loss(model, net, b, noise_sd=0.0, val_k=None, gen=None):
    if model.startswith("sampler"):
        if val_k is None:
            return net.loss(b["x0"], b["s"], b.get("traj"))
        k = torch.full((len(b["x0"]),), val_k, device=b["x0"].device)
        noise = torch.randn(b["x0"].shape, device=b["x0"].device, generator=gen)
        return net.loss(b["x0"], b["s"], b.get("traj"), k=k, noise=noise)
    if model.startswith("vf"):
        c = b["c"]
        if noise_sd > 0:
            c = c + noise_sd * torch.randn_like(c)
        pred = net(c, b["s"], b["traj"])
        return ((pred - b["target"]) ** 2).sum(1).mean()
    pred = net(b["s"], b["traj"])
    return ((pred - b["target"]) ** 2).sum(1).mean()


@torch.no_grad()
def evaluate(model, net, batcher, noise_sd):
    net.eval(); tot, n = 0.0, 0
    for b in batcher.all():
        m = len(next(iter(b.values())))
        if model.startswith("sampler"):
            gen = torch.Generator(device=batcher.fd.device); gen.manual_seed(1234)
            l = torch.stack([step_loss(model, net, b, val_k=k, gen=gen) for k in CFG["val_timesteps"]]).mean()
        else:
            l = step_loss(model, net, b, 0.0)
        tot += float(l) * m; n += m
    net.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model", required=True, choices=list(CFG["batch"])); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--tag", default="", help="suffix of the checkpoint / log name (variants)")
    ap.add_argument("--width", type=int, default=None); ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--input-noise", type=float, default=0.0, help="VF: Gaussian noise on C_t, as a fraction of the rms one-step delta")
    ap.add_argument("--unroll", type=int, default=0, help="VF: add a k-step free-rollout L2 term (0 = one-step teacher forcing only)")
    ap.add_argument("--select", default="auto", choices=["auto", "loss", "rollout"], help="VF checkpoint selection: one-step val loss or val free-rollout E_C (auto = rollout for vf)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    fd = FoldData(a.split, a.fold, dev)
    net = build(a.model, fd, a.width, a.dropout).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr or CFG["lr"], weight_decay=CFG["wd"] if a.wd is None else a.wd)
    max_steps, bs = a.max_steps or CFG["max_steps"][a.model], CFG["batch"][a.model]
    lr_scale = [1.0]                                       # plateau decay multiplier
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]) * lr_scale[0])
    tr, va = Batcher(a.model, fd, "train", a.seed), Batcher(a.model, fd, "val", a.seed + 1)
    noise_sd = 0.0
    if a.model == "vf_noise" or (a.model == "vf" and a.input_noise > 0):
        d = (fd.C[fd.idx["train"], 1:] - fd.C[fd.idx["train"], :-1])
        noise_sd = (a.input_noise if a.model == "vf" else CFG["noise_frac"]) * float(d.pow(2).mean().sqrt())
    is_vf = a.model.startswith("vf")
    select_rollout = is_vf and a.select in ("auto", "rollout")
    name = a.model + a.tag
    log.info("%s %s%d %s: %d params, %d train items, %d val items, noise_sd %.4f", H.DATASET, a.split, a.fold, a.model,
             MD.n_params(net), tr.size, va.size, noise_sd)
    rows, best, best_state, bad, run, best_step, stopped_by = [], np.inf, None, 0, 0.0, 0, "max_steps"
    for step in range(1, max_steps + 1):
        b = tr.random(bs)
        loss = step_loss(a.model, net, b, noise_sd)
        if is_vf and a.unroll > 0:
            sel = torch.randint(0, tr.size, (bs // 4,), device=dev, generator=tr.g)
            loss = loss + unrolled_loss(net, fd, tr.n[sel], tr.t[sel], a.unroll, noise_sd)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % CFG["eval_every"] == 0:
            v_loss = evaluate(a.model, ema, va, noise_sd)
            v_roll = val_rollout(ema, fd) if is_vf else float("nan")
            v = v_roll if select_rollout else v_loss
            rows.append(dict(step=step, train=run, val=v, val_loss=v_loss, val_rollout=v_roll, lr=sched.get_last_lr()[0], time=time.time() - t0))
            if v < best - 1e-5:
                best, bad = v, 0; best_state = {k: t.detach().clone() for k, t in ema.state_dict().items()}; best_step = step
            else:
                bad += 1
                if bad % CFG["plateau_patience"] == 0:
                    lr_scale[0] *= CFG["lr_factor"]
            if step % (CFG["eval_every"] * 4) == 0:
                log.info("  step %d train %.4f val %.4f (loss %.4f, rollout %.4f) best %.4f (step %d) lr x%.4f (%.0fs)", step, run, v, v_loss, v_roll, best, best_step, lr_scale[0], time.time() - t0)
            if bad >= CFG["patience"]:
                stopped_by = "patience"; break
            if lr_scale[0] < CFG["min_lr_factor"]:
                stopped_by = "lr_floor"; break
    if best_state is None:
        raise RuntimeError(f"{H.DATASET} {a.model}: no finite validation loss was ever recorded (last rows: {rows[-3:]})")
    ck = H.CKPT / f"{a.split}{a.fold}"; ck.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=a.model, state=best_state, stats=fd.stats, cfg=CFG, d_static=fd.S.shape[1], d_traj=len(H.LOCAL_OFFSETS) * fd.D,
                    noise_sd=noise_sd, best_val=best, best_step=best_step, steps=step, seed=a.seed, split=a.split, fold=a.fold,
                    stopped_by=stopped_by, converged=stopped_by != "max_steps", width=a.width or CFG["width"], dropout=a.dropout,
                    wd=CFG["wd"] if a.wd is None else a.wd, lr=a.lr or CFG["lr"], input_noise=a.input_noise, unroll=a.unroll,
                    selection="val free-rollout E_C" if select_rollout else "val loss", tag=a.tag), ck / f"{name}.pt")
    (H.OUT / "train_logs").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(H.OUT / "train_logs" / f"{a.split}{a.fold}_{name}.csv", index=False)
    log.info("done %s %s%d %s: best val %.4f at step %d of %d, stopped by %s, %.0fs", H.DATASET, a.split, a.fold, name, best, best_step, step, stopped_by, time.time() - t0)


if __name__ == "__main__":
    main()
