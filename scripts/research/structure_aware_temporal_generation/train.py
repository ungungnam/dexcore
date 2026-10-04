#!/usr/bin/env python
"""Train one model of the matrix.
    CUDA_VISIBLE_DEVICES=4 python train.py --dataset taco --model D1 --seed 0 [--lambda-struct 1.0] [--lambda-aux 0.3] [--sweep]

Recipe (identical for the six models): AdamW lr 3e-4, wd 0.01, 1000 warm-up steps then constant, halved after 4 validation
checks without improvement; batch 64 sequences; grad clip 1; EMA 0.999 (the EMA weights are validated and saved); validation
every 500 steps; early stopping after 12 checks without improvement or when the LR has decayed 32x, with a cap of 30 000 steps (the
diffusion v-loss keeps improving slowly; a run that hits the cap is flagged converged = False). Deterministic validation loss = the training objective on the whole validation set;
diffusion validation loss = the objective at 10 fixed steps k with a fixed noise generator (the initial sampler's rule).
The lambdas come from <ds>/lambda_choice.json unless given; --sweep puts them into the run name (the selection runs).
Writes CKPT/<ds>/<name>.pt and <ds>/train_logs/<name>.csv.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import pandas as pd
import torch

import sat_common as S
import sat_models as MD
from sat_data import SeqData, assert_causal_conditioning
from losses import StructuralLoss, dense_loss, persistence_reference, aux_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")
CFG = dict(lr=3e-4, wd=0.01, warmup=1000, batch=64, ema=0.999, eval_every=500, plateau_patience=4, lr_factor=0.5, min_lr_factor=1 / 32,
           patience=12, max_steps=30000, grad_clip=1.0, width=512, depth=6, heads=8, dropout=0.1,
           val_timesteps=[50, 150, 250, 350, 450, 550, 650, 750, 850, 950])


def step_losses(model, net, b, data, cal, sl, lam_s, lam_a, k=None, noise=None):
    """Training objective of one batch and its components."""
    c = S.MODELS[model]; parts = {}
    if c["family"] == "det":
        r_hat, aux = net(b["s0_in"], b["tau"], b["S"])
        base = dense_loss(r_hat, b["r"]); parts["dense"] = float(base)
        C_hat = data.to_raw(r_hat, b["s0_raw"])
    else:
        base, x0_hat, aux, _ = net.forward_loss(b["r"], b["s0_in"], b["tau"], b["S"], k=k, noise=noise); parts["diffusion"] = float(base)
        C_hat = data.to_raw(x0_hat, b["s0_raw"])
    loss = base
    if c["struct"]:
        ls, comp = sl(C_hat, b, cal); loss = loss + lam_s * ls
        parts["struct"] = float(ls); parts.update(comp)
    if c["aux"]:
        la, comp = aux_loss(aux, b, data); loss = loss + lam_a * la
        parts["aux"] = float(la); parts.update(comp)
    return loss, parts


@torch.no_grad()
def evaluate(model, net, data, idx, cal, sl, lam_s, lam_a, bs=128):
    net.eval(); tot = 0.0; n_tot = 0; comp = {}
    gen = torch.Generator(device=data.device); gen.manual_seed(1234)
    is_diff = S.MODELS[model]["family"] == "diff"
    for i in range(0, len(idx), bs):
        n = idx[i:i + bs]; b = data.batch(n)
        if is_diff:
            ls = []
            for kv in CFG["val_timesteps"]:
                k = torch.full((len(n),), kv, device=data.device)
                noise = torch.randn(b["r"].shape, device=data.device, generator=gen)
                l, p = step_losses(model, net, b, data, cal, sl, lam_s, lam_a, k=k, noise=noise); ls.append(l)
                for kk, v in p.items():
                    comp[kk] = comp.get(kk, 0.0) + v * len(n) / len(CFG["val_timesteps"])
            l = torch.stack(ls).mean()
        else:
            l, p = step_losses(model, net, b, data, cal, sl, lam_s, lam_a)
            for kk, v in p.items():
                comp[kk] = comp.get(kk, 0.0) + v * len(n)
        tot += float(l) * len(n); n_tot += len(n)
    net.train()
    return tot / n_tot, {k: v / n_tot for k, v in comp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--model", required=True, choices=list(S.MODELS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lambda-struct", type=float, default=None); ap.add_argument("--lambda-aux", type=float, default=None)
    ap.add_argument("--sweep", action="store_true", help="put the lambdas into the run name (selection runs)")
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ds, model = a.dataset, a.model; c = S.MODELS[model]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    data = SeqData(ds, dev); assert_causal_conditioning(data)
    cal = data.cal; assert cal is not None, "run calibrate.py first"
    choice = S.chosen_lambdas(ds) or {}
    lam_s = a.lambda_struct if a.lambda_struct is not None else choice.get("lambda_struct")
    lam_a = a.lambda_aux if a.lambda_aux is not None else choice.get("lambda_aux")
    if c["struct"]:
        assert lam_s is not None, "lambda_struct: give --lambda-struct or run select_lambda.py"
    if c["aux"]:
        assert lam_a is not None, "lambda_aux: give --lambda-aux or run select_lambda.py"
    name = S.run_name(model, a.seed, lam_s if (a.sweep and c["struct"]) else None, lam_a if (a.sweep and c["aux"]) else None) + a.tag
    net = MD.build(model, data.d_traj(), data.d_static(), CFG).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ref = persistence_reference(data, cal) if c["struct"] else None
    sl = StructuralLoss(ref) if c["struct"] else None
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    lr_scale = [1.0]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]) * lr_scale[0])
    tr = torch.from_numpy(data.idx["train"]).to(dev); va = torch.from_numpy(data.idx["val"]).to(dev)
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    v0, comp0 = evaluate(model, ema, data, va, cal, sl, lam_s, lam_a)
    log.info("%s %s seed %d (%s): %d params, lambda_struct %s, lambda_aux %s, %d train / %d val sequences, sigma_r %.4f, struct refs %s, val at init %.4f %s",
             ds, model, a.seed, name, MD.n_params(net), lam_s, lam_a, len(tr), len(va), data.sigma_r, ref, v0, comp0)
    rows, best, best_state, bad, run, best_step, stopped_by = [], np.inf, None, 0, 0.0, 0, "max_steps"
    max_steps = a.max_steps or CFG["max_steps"]
    for step in range(1, max_steps + 1):
        net.train()
        sel = torch.randint(0, len(tr), (CFG["batch"],), device=dev, generator=g)
        b = data.batch(tr[sel])
        loss, parts = step_losses(model, net, b, data, cal, sl, lam_s, lam_a)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"])
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % CFG["eval_every"] == 0:
            v, comp = evaluate(model, ema, data, va, cal, sl, lam_s, lam_a)
            rows.append(dict(step=step, train=run, val=v, lr=sched.get_last_lr()[0], time=time.time() - t0, **{f"val_{k}": x for k, x in comp.items()}, **{f"train_{k}": x for k, x in parts.items()}))
            if v < best - 1e-5:
                best, bad, best_step = v, 0, step; best_state = {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
                if bad % CFG["plateau_patience"] == 0:
                    lr_scale[0] *= CFG["lr_factor"]
            if step % (CFG["eval_every"] * 4) == 0:
                log.info("  step %d train %.4f val %.4f best %.4f (step %d) lr x%.4f %s (%.0f s)", step, run, v, best, best_step, lr_scale[0],
                         {k: round(x, 4) for k, x in comp.items()}, time.time() - t0)
            if bad >= CFG["patience"]:
                stopped_by = "patience"; break
            if lr_scale[0] < CFG["min_lr_factor"]:
                stopped_by = "lr_floor"; break
    if best_state is None:
        raise RuntimeError("no validation check recorded (max_steps < eval_every?)")
    ck = S.ckpt_path(ds, name); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, model=model, seed=a.seed, lambda_struct=lam_s if c["struct"] else None, lambda_aux=lam_a if c["aux"] else None,
                    cfg=CFG, stats=data.stats, calibration=cal, struct_ref=ref, best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by,
                    converged=stopped_by != "max_steps", n_params=MD.n_params(net), val_init=v0, d_traj=data.d_traj(), d_static=data.d_static(), name=name), ck)
    (S.ds_out(ds) / "train_logs").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(S.ds_out(ds) / "train_logs" / f"{name}.csv", index=False)
    log.info("done %s %s: best val %.4f at step %d of %d (%s), %.0f s -> %s", ds, name, best, best_step, step, stopped_by, time.time() - t0, ck)


if __name__ == "__main__":
    main()
