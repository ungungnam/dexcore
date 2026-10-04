#!/usr/bin/env python
"""Step 2: one multi-horizon Delta-C regressor per (dataset, condition, seed).
    CUDA_VISIBLE_DEVICES=<gpu> python train_delta.py --dataset taco --cond F2k8 --seed 0

Target  Delta_h C_t = C_{t+h} - C_t (standardised units) for h in (1, 4, 8), from ONE network with
        three 512-D heads (the simpler implementation; every condition shares it). Loss =
        sum_h mean_valid ||Delta^_h - Delta_h||^2 / E_train ||Delta_h||^2 (equal relative weight per
        horizon; a pair (n, t) is valid for h when t + h <= 63).
Model   the existing VF trunk (3 FiLM residual blocks, width 1024, dropout 0.1, zero-initialised
        output = the zero-change predictor at step 0); condition = static + trajectory encoders as
        in the VF + a hand encoder (d_hand -> 256 -> 256) added into the same 256-D FiLM vector.
        Only the hand input differs between the conditions (see hp_common.CONDITIONS).
Optim   AdamW 3e-4, wd 0.05, batch 1024, 500 warm-up steps, EMA 0.999; validation loss every 500
        steps; LR halved after 4 checks without improvement; stop after 10 checks without
        improvement (5 000 steps) or 40 000 steps. The EMA weights of the best check are kept.
Writes  OUT/<ds>/ckpt/delta_<cond>_seed<s>.pt, OUT/<ds>/train_logs/delta_<cond>_seed<s>.csv,
        OUT/<ds>/preds/delta_<cond>_seed<s>.npz  (test predictions Delta^_h C_t in RAW contact
        units, (N_test, 63, 3, 512) f16, plus example ids)
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

import hp_common as P
import hp_data as HD
from hp_models import DeltaPredictor, n_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_delta")
# Recipe (identical for every condition, chosen on F0 trials before any hand model was compared; see
# logs/trial_*.log): the existing VF trunk (width 1024, 3 blocks, dropout 0.1, wd 0.05) with the
# final VF's Gaussian input noise (0.5 x rms one-step delta). On ARCTIC every capacity (0.25-1024
# width) reaches its best validation loss within ~750 steps and then overfits, so the checkpoint is
# selected at a 100-step resolution with a short EMA (0.99) and stops after 20 checks without
# improvement; TACO converges in a few thousand steps under the same rule.
CFG = dict(lr=3e-4, wd=0.05, warmup=300, batch=1024, ema=0.99, eval_every=100, plateau_patience=6, lr_factor=0.5, patience=20,
           max_steps=30000, dropout=0.1, width=1024, blocks=3, grad_clip=1.0, input_noise=0.5)


def loss_fn(pred, target, mask, w):
    """pred/target (B, n_h, 512), mask (B, n_h), w (n_h,) -> scalar."""
    se = ((pred - target) ** 2).sum(2) * mask                         # (B, n_h)
    return ((se.sum(0) / mask.sum(0).clamp(min=1)) * w).sum()


@torch.no_grad()
def evaluate(net, fd, ht, cond, n_all, t_all, w, perm, bs=4096):
    net.eval(); tot = 0.0
    for i in range(0, len(n_all), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]
        target, mask = HD.delta_targets(fd, n, t)
        pred = net(fd.C[n, t], fd.S[n], fd.context(n, t), HD.hand_input(ht, cond, n, t, perm))
        se = ((pred - target) ** 2).sum(2) * mask
        tot = tot + (se.sum(0) * w) if i else se.sum(0) * w
    net.train()
    # normalise per horizon by the number of valid pairs
    cnt = torch.stack([(t_all + h <= fd.T - 1).float().sum() for h in P.HORIZONS])
    return float((tot / cnt).sum())


@torch.no_grad()
def predict(net, fd, ht, cond, n_all, t_all, perm, bs=4096):
    net.eval(); out = []
    for i in range(0, len(n_all), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]
        out.append(net(fd.C[n, t], fd.S[n], fd.context(n, t), HD.hand_input(ht, cond, n, t, perm)).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=P.DATASETS); ap.add_argument("--cond", required=True, choices=list(P.CONDITIONS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--width", type=int, default=None); ap.add_argument("--blocks", type=int, default=None); ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--wd", type=float, default=None); ap.add_argument("--input-noise", type=float, default=None, help="Gaussian noise on C~_t as a fraction of the rms one-step delta")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(); ds, cond = a.dataset, a.cond
    for k in ("width", "blocks", "dropout", "wd", "input_noise"):
        if getattr(a, k) is not None:
            CFG[k] = getattr(a, k)
    P.assert_causal()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    fd = P.fold(ds, dev)
    ht = HD.HandTensors(ds, fd, dev)
    perm = HD.shuffle_map(fd, a.seed)
    d_static, d_traj, d_hand = fd.S.shape[1], fd.context(torch.zeros(1, dtype=torch.long, device=dev), torch.zeros(1, dtype=torch.long, device=dev)).shape[1], P.hand_dim(cond)
    net = DeltaPredictor(d_static, d_traj, d_hand, width=CFG["width"], blocks=CFG["blocks"], dropout=CFG["dropout"]).to(dev)
    noise_sd = 0.0
    if CFG["input_noise"] > 0:
        d1 = fd.C[fd.idx["train"], 1:] - fd.C[fd.idx["train"], :-1]
        noise_sd = CFG["input_noise"] * float(d1.pow(2).mean().sqrt())
    n_hand_params = n_params(net.cond.hand) if net.cond.hand is not None else 0
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    lr_scale = [1.0]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]) * lr_scale[0])
    n_tr, t_tr = HD.pairs(fd, "train"); n_va, t_va = HD.pairs(fd, "val"); n_te, t_te = HD.pairs(fd, "test")
    # per-horizon loss weights: 1 / E_train ||Delta_h||^2 over the valid training pairs
    with torch.no_grad():
        tg, mk = HD.delta_targets(fd, n_tr, t_tr)
        w = 1.0 / (((tg ** 2).sum(2) * mk).sum(0) / mk.sum(0))
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    v_zero = evaluate(ema, fd, ht, cond, n_va, t_va, w, perm)                            # zero-initialised output = the zero-change predictor
    log.info("%s %s seed %d: %d params (hand encoder %d, d_hand %d), %d train / %d val / %d test pairs, horizon weights %s, cfg %s, noise_sd %.4f, zero-change val loss %.4f",
             ds, cond, a.seed, n_params(net), n_hand_params, d_hand, len(n_tr), len(n_va), len(n_te), [round(float(x), 3) for x in w], CFG, noise_sd, v_zero)
    rows, best, best_state, bad, run, best_step, stopped_by = [], np.inf, None, 0, 0.0, 0, "max_steps"
    max_steps = a.max_steps or CFG["max_steps"]
    for step in range(1, max_steps + 1):
        sel = torch.randint(0, len(n_tr), (CFG["batch"],), device=dev, generator=g)
        n, t = n_tr[sel], t_tr[sel]
        target, mask = HD.delta_targets(fd, n, t)
        c_in = fd.C[n, t]
        if noise_sd > 0:
            c_in = c_in + noise_sd * torch.randn_like(c_in)
        pred = net(c_in, fd.S[n], fd.context(n, t), HD.hand_input(ht, cond, n, t, perm))
        loss = loss_fn(pred, target, mask, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"])
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % CFG["eval_every"] == 0:
            v = evaluate(ema, fd, ht, cond, n_va, t_va, w, perm)
            rows.append(dict(step=step, train=run, val=v, lr=sched.get_last_lr()[0], time=time.time() - t0))
            if v < best - 1e-5:
                best, bad, best_step = v, 0, step; best_state = {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
                if bad % CFG["plateau_patience"] == 0:
                    lr_scale[0] *= CFG["lr_factor"]
            if step % (CFG["eval_every"] * 10) == 0:
                log.info("  step %d train %.4f val %.4f best %.4f (step %d) lr x%.3f (%.0fs)", step, run, v, best, best_step, lr_scale[0], time.time() - t0)
            if bad >= CFG["patience"]:
                stopped_by = "patience"; break
    assert best_state is not None
    ema.load_state_dict(best_state)
    out = P.ds_out(ds)
    for sub in ("ckpt", "preds", "train_logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name = f"delta_{cond}_seed{a.seed}{a.tag}"
    torch.save(dict(state=best_state, cond=cond, seed=a.seed, cfg=CFG, stats=fd.stats, hand_stats=ht.stats, d_static=d_static, d_traj=d_traj, d_hand=d_hand,
                    best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by, n_params=n_params(net), n_hand_params=n_hand_params,
                    horizon_weights=[float(x) for x in w], zero_change_val=v_zero, noise_sd=noise_sd), out / "ckpt" / f"{name}.pt")
    pd.DataFrame(rows).to_csv(out / "train_logs" / f"{name}.csv", index=False)
    pred = predict(ema, fd, ht, cond, n_te, t_te, perm) * fd.stats["s"]                      # raw units
    n_test = len(fd.idx["test"])
    np.savez_compressed(out / "preds" / f"{name}.npz", pred=pred.reshape(n_test, fd.T - 1, len(P.HORIZONS), 512).astype(np.float16),
                        example=fd.idx["test"], horizons=np.array(P.HORIZONS))
    log.info("done %s %s seed %d: best val %.4f at step %d of %d (%s), %.0fs", ds, cond, a.seed, best, best_step, step, stopped_by, time.time() - t0)


if __name__ == "__main__":
    main()
