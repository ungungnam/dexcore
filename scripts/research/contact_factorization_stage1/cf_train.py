#!/usr/bin/env python
"""Train one model of one dataset.    CUDA_VISIBLE_DEVICES=4 python cf_train.py --dataset taco --model A3 --seed 0

Single-latent models (A0, A1): one phase of TRAIN.steps_single steps,  L = L_rec(C_bar) [+ lambda_rel L_rel].
Factorised models (A2, A3):  phase A (z branch alone, steps_phase_a)  L_A = L_coarse + lambda_rel L_rel
                             phase B (+ r branch, steps_phase_b)      L_B = L_full + lambda_coarse L_coarse + lambda_rel L_rel
                                                                            + lambda_delta mean |Delta_C|
                             (z branch at 0.1 x lr for the first z_slow_steps of phase B, then joint fine-tuning).
Reconstruction losses are MSE in the standardised contact units (1.0 = the per-point train mean). L_rel = SmoothL1 between the
batch-mean-normalised pairwise distances of z and of the R2 + wrench teacher (Section 7 of the plan). AdamW, warmup + cosine,
EMA 0.999 (evaluated), bf16 autocast, batch 128 of balanced frames. Logs train / val reconstruction, latent spreads, |Delta_C|,
per-branch gradient norms; keeps the EMA weights of the best validation full reconstruction and the last weights.
Writes <ckpt>/<ds>/<name>.pt and <out>/<ds>/train_logs/<name>.csv.
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import cf_common as S
from cf_data import FrameData
from cf_models import build, n_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train")


def rel_loss(z, u_r2, u_q):
    """SmoothL1(d_z / mean, d_teacher / mean) over the pairs of the batch."""
    B = len(z); iu = torch.triu_indices(B, B, 1, device=z.device)
    dz = torch.cdist(z, z)[iu[0], iu[1]]
    dt = (S.TEACHER["w_r2"] * torch.cdist(u_r2, u_r2) + S.TEACHER["w_q"] * torch.cdist(u_q, u_q))[iu[0], iu[1]]
    return F.smooth_l1_loss(dz / (dz.mean() + 1e-8), dt / (dt.mean() + 1e-8))


def step_losses(net, b, rel, phase, cfg):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("autocast", True)):
        out = net(b["Cn"], b["geo"], b["S"], use_r=(phase == "B"))
    Cn = b["Cn"]; C_bar = out["C_bar"].float(); z = out["z"].float()
    t = dict(coarse=F.mse_loss(C_bar, Cn))
    total = (cfg["lambda_coarse"] if phase == "B" else 1.0) * t["coarse"]
    if rel:
        t["rel"] = rel_loss(z, b["u_r2"], b["u_q"]); total = total + cfg["lambda_rel"] * t["rel"]
    if phase == "B":
        C_hat = out["C_hat"].float(); delta = out["delta"].float()
        t["full"] = F.mse_loss(C_hat, Cn); t["delta"] = delta.abs().mean()
        total = total + t["full"] + cfg["lambda_delta"] * t["delta"]
    return total, t, out


def lr_at(step, n_steps, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(n_steps - cfg["warmup"], 1)
    return cfg["lr"] * (cfg["final_lr_frac"] + (1 - cfg["final_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(net, data, split, rel, use_r, n_max=None, bs=512):
    n, t = data.split_frames(split)
    if n_max is not None and len(n) > n_max:
        g = torch.Generator(device=data.device); g.manual_seed(7); sel = torch.randperm(len(n), device=data.device, generator=g)[:n_max]; n, t = n[sel], t[sel]
    acc = dict(E_zonly=0.0, E_full=0.0, mse_zonly=0.0, mse_full=0.0, rel=0.0, delta_abs=0.0); zs, rs = [], []; cnt = 0; nrel = 0
    for i in range(0, len(n), bs):
        b = data.batch(n[i:i + bs], t[i:i + bs])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = net(b["Cn"], b["geo"], b["S"], use_r=use_r)
        C_bar = out["C_bar"].float(); C_hat = out["C_hat"].float(); m = len(b["n"])
        acc["E_zonly"] += float((data.to_raw(C_bar) - b["C"]).norm(dim=-1).sum()); acc["E_full"] += float((data.to_raw(C_hat) - b["C"]).norm(dim=-1).sum())
        acc["mse_zonly"] += float(((C_bar - b["Cn"]) ** 2).mean(-1).sum()); acc["mse_full"] += float(((C_hat - b["Cn"]) ** 2).mean(-1).sum())
        if out["delta"] is not None:
            acc["delta_abs"] += float((out["delta"].float() * data.s).abs().mean(-1).sum())
        zs.append(out["z"].float())
        if out["r"] is not None:
            rs.append(out["r"].float())
        if rel:
            for j in range(0, m, 128):
                if j + 128 <= m:
                    acc["rel"] += float(rel_loss(out["z"].float()[j:j + 128], b["u_r2"][j:j + 128], b["u_q"][j:j + 128])); nrel += 1
        cnt += m
    res = {k: v / cnt for k, v in acc.items() if k != "rel"}
    res["rel"] = acc["rel"] / max(nrel, 1) if rel else float("nan")
    z = torch.cat(zs); res["z_std"] = float(z.std(0).mean()); res["z_eff_rank"] = eff_rank(z)
    if rs:
        r = torch.cat(rs); res["r_std"] = float(r.std(0).mean()); res["r_eff_rank"] = eff_rank(r)
    else:
        res["r_std"] = float("nan"); res["r_eff_rank"] = float("nan")
    return res


def eff_rank(x):
    """Participation ratio of the covariance eigenvalues (sum λ)^2 / sum λ^2."""
    xc = x - x.mean(0); ev = torch.linalg.eigvalsh(xc.T @ xc / len(xc))
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp(min=1e-12))


def grad_norm(params):
    g = [p.grad.norm() ** 2 for p in params if p.grad is not None]
    return float(torch.stack(g).sum().sqrt()) if g else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=S.SEED); ap.add_argument("--steps-scale", type=float, default=1.0, help="smoke tests only")
    ap.add_argument("--init-z-from", default=None, help="checkpoint name whose E_z / D_z (last raw weights) initialise the z branch")
    ap.add_argument("--phase-b-only", action="store_true", help="skip phase A (the z branch comes from --init-z-from)")
    ap.add_argument("--tag", default="", help="suffix of the run name (pilots)"); ap.add_argument("--batch", type=int, default=None); ap.add_argument("--lambda-delta", type=float, default=None)
    ap.add_argument("--steps-b", type=int, default=None); ap.add_argument("--freeze-z", action="store_true", help="pilots: the z branch is not trained in phase B")
    ap.add_argument("--no-autocast", action="store_true"); ap.add_argument("--log-every", type=int, default=None); ap.add_argument("--res-scale", type=float, default=None)
    a = ap.parse_args(); dev = torch.device("cuda")
    cfg = dict(S.TRAIN); mc = S.ALL_MODELS[a.model]; name = S.run_name(a.model, a.seed) + a.tag
    if a.res_scale is None:
        a.res_scale = cfg["res_scale"]
    if a.batch:
        cfg["batch"] = a.batch
    if a.lambda_delta is not None:
        cfg["lambda_delta"] = a.lambda_delta
    if a.log_every:
        cfg["log_every"] = a.log_every
    cfg["res_scale"] = a.res_scale; cfg["freeze_z"] = a.freeze_z; cfg["autocast"] = not a.no_autocast
    if a.steps_b:
        cfg["steps_phase_b"] = a.steps_b
    for k in ("steps_single", "steps_phase_a", "steps_phase_b", "z_slow_steps", "z_freeze_steps", "warmup", "eval_every"):
        cfg[k] = max(int(cfg[k] * a.steps_scale), 1)
    if a.log_every:
        cfg["log_every"] = a.log_every
    cfg["res_scale"] = a.res_scale; cfg["freeze_z"] = a.freeze_z; cfg["autocast"] = not a.no_autocast
    if a.steps_b:
        cfg["steps_phase_b"] = a.steps_b; cfg["eval_every"] = min(cfg["eval_every"], max(a.steps_b // 4, 1)); cfg["z_slow_steps"] = min(cfg["z_slow_steps"], a.steps_b // 4); cfg["z_freeze_steps"] = min(cfg["z_freeze_steps"], a.steps_b // 4)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    data = FrameData(a.dataset, dev, seed=a.seed)
    net = build(a.model, data.d_geo(), data.d_static(), res_scale=a.res_scale).to(dev)
    init_info = None
    if a.init_z_from:
        ck0 = torch.load(S.ckpt_path(a.dataset, a.init_z_from), map_location="cpu", weights_only=False)
        zs = {k: v for k, v in ck0["last_state"].items() if k.startswith(("E_z.", "D_z."))}
        missing, unexpected = net.load_state_dict(zs, strict=False); assert not unexpected and all(k.startswith(("E_r.", "D_r.")) for k in missing), (unexpected, missing[:5])
        init_info = dict(name=a.init_z_from, steps=ck0["steps"], best_val=ck0["best_val"], model=ck0["model"]); log.info("z branch initialised from %s (%d steps)", a.init_z_from, ck0["steps"])
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    log.info("%s %s: %.1fM params; train frames %d unique of %d (%d take/object groups); val %d; test %d", a.dataset, name, n_params(net) / 1e6,
             data.frames["train"]["n_unique"], data.frames["train"]["n_all"], data.frames["train"]["n_groups"], data.frames["val"]["n_unique"], data.frames["test"]["n_unique"])
    n_single = max(int(cfg["steps_override"].get(a.model, cfg["steps_single"]) * a.steps_scale), 1)
    phases = [("single", n_single)] if mc["dr"] == 0 else ([("B", cfg["steps_phase_b"])] if a.phase_b_only else [("A", cfg["steps_phase_a"]), ("B", cfg["steps_phase_b"])])
    rows, best, gstep, t0 = [], dict(E_full=float("inf")), 0, time.time()
    S.ckpt_path(a.dataset, name).parent.mkdir(parents=True, exist_ok=True); S.train_log_path(a.dataset, name).parent.mkdir(parents=True, exist_ok=True)
    for phase, n_steps in phases:
        groups = [] if (a.freeze_z and phase == "B") else [dict(params=net.z_params(), name="z")]
        if phase == "B":
            groups.append(dict(params=net.r_params(), name="r"))
        def z_mult(step):
            """phase B: z branch frozen for z_freeze_steps, then 0.1 x lr for z_slow_steps, then joint."""
            if phase != "B":
                return 1.0
            if a.freeze_z:
                return 0.0
            return 0.0 if step < cfg["z_freeze_steps"] else (0.1 if step < cfg["z_freeze_steps"] + cfg["z_slow_steps"] else 1.0)
        for p_ in net.z_params():
            p_.requires_grad_(z_mult(0) > 0)
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], weight_decay=cfg["wd"], betas=(0.9, 0.99))
        net.train()
        for step in range(n_steps):
            lr = lr_at(step, n_steps, cfg)
            zm = z_mult(step)
            if zm > 0 and not next(iter(net.z_params())).requires_grad:
                for p_ in net.z_params():
                    p_.requires_grad_(True)
            for g in opt.param_groups:
                g["lr"] = lr * (zm if g["name"] == "z" else 1.0)
            n, t = data.sample_train(cfg["batch"]); b = data.batch(n, t)
            total, terms, out = step_losses(net, b, mc["rel"], phase, cfg)
            opt.zero_grad(set_to_none=True); total.backward()
            gz, gr = grad_norm(net.z_params()), grad_norm(net.r_params())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(cfg["ema"]).add_(pn.detach(), alpha=1 - cfg["ema"])
            gstep += 1
            if step % cfg["log_every"] == 0 or step == n_steps - 1:
                row = dict(kind="train", phase=phase, step=gstep, phase_step=step, lr=lr, total=float(total), grad_z=gz, grad_r=gr,
                           z_std=float(out["z"].float().std(0).mean()), r_std=float(out["r"].float().std(0).mean()) if out["r"] is not None else np.nan,
                           delta_abs=float((out["delta"].float() * data.s).abs().mean()) if out["delta"] is not None else np.nan,
                           **{k: float(v) for k, v in terms.items()}, elapsed=time.time() - t0)
                rows.append(row)
                if step % (cfg["log_every"] * 10) == 0:
                    log.info("%s step %d/%d %s", phase, step, n_steps, {k: round(v, 4) for k, v in row.items() if isinstance(v, float) and not np.isnan(v)})
            if (step + 1) % cfg["eval_every"] == 0 or step == n_steps - 1:
                use_r = phase == "B"
                v = evaluate(ema, data, "val", mc["rel"], use_r); tr = evaluate(ema, data, "train", mc["rel"], use_r, n_max=8192)
                row = dict(kind="eval", phase=phase, step=gstep, phase_step=step, **{f"val_{k}": x for k, x in v.items()}, **{f"train_{k}": x for k, x in tr.items()}, elapsed=time.time() - t0)
                rows.append(row); pd.DataFrame(rows).to_csv(S.train_log_path(a.dataset, name), index=False)
                log.info("EVAL %s step %d val E_zonly %.4f E_full %.4f mse_full %.4f rel %.4f z_std %.3f r_std %.3f | train E_full %.4f", phase, gstep, v["E_zonly"], v["E_full"], v["mse_full"], v["rel"], v["z_std"], v["r_std"], tr["E_full"])
                if phase == phases[-1][0] and v["E_full"] < best["E_full"]:
                    best = dict(E_full=v["E_full"], step=gstep, val=v, train=tr, ema_state=copy.deepcopy(ema.state_dict()))
                net.train()
    ck = dict(name=name, model=a.model, dataset=a.dataset, seed=a.seed, cfg=cfg, arch=S.ARCH, teacher=S.TEACHER, stats=data.stats, res_scale=a.res_scale,
              n_params=n_params(net), d_geo=data.d_geo(), d_static=data.d_static(), best_step=best["step"], best_val=best["val"], best_train=best["train"],
              last_val=rows[-1], steps=gstep, phases=phases, init_z_from=init_info, ema_state=best["ema_state"], last_ema_state=ema.state_dict(), last_state=net.state_dict())
    torch.save(ck, S.ckpt_path(a.dataset, name))
    pd.DataFrame(rows).to_csv(S.train_log_path(a.dataset, name), index=False)
    log.info("done %s: best val E_full %.4f at step %d (%.0f s)", name, best["E_full"], best["step"], time.time() - t0)


if __name__ == "__main__":
    main()
