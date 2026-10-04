#!/usr/bin/env python
"""Train one Stage-2 model (one seed).
    CUDA_VISIBLE_DEVICES=5 python s2_train.py --dataset taco --model B1 [--seed 0] [--lambda-z 1.0] [--tag _lz0.3] [--wandb] [--resume]

Recipe (identical for B0 / B1 / B2): AdamW lr 1e-4, wd 0.01, 5000 warm-up steps then cosine decay to 0.05 x over 80 000 steps; batch 128
sequences (GT s_0); grad clip 1.0; EMA 0.999 (the EMA weights are validated and saved); bf16 autocast; validation every 1000 steps on the
whole validation set with the training objective (all 63 frames decoded); early stopping only after 20 000 steps and only after 15
validation checks without improvement; best-validation and final checkpoints; a resumable "last" checkpoint every validation.
Losses (all dense terms in the residual units rho = (C - s_0) / sigma_r; L_z in standardised teacher coordinates, mean over frames x dims):
  B0  L_C(rho_hat, rho)                                                         all 63 frames
  B1  L_C(C_bar) + lambda_z L_z                                                 dense term on n_dec_frames random future frames per sequence, L_z on all frames
  B2  L_full(C_bar + Delta) + lambda_c L_zonly(C_bar) + lambda_z L_z + lambda_res mean|Delta|   (Delta in standardised contact units)
Per-term gradient norms are logged at a few early steps (the Section-12 inspection); every run writes <ds>/train_logs/<name>.csv (validation
rows), <ds>/train_logs/<name>_train.csv (training rows) and CKPT/<ds>/<name>.pt (best EMA), <name>_final.pt (last EMA + raw weights).
"""
from __future__ import annotations

import argparse
import copy
import logging
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import s2_common as S
import s2_models as MD
from s2_data import S2Data, assert_causal_conditioning

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2_train")


def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = min(1.0, (step - cfg["warmup"]) / max(1, cfg["max_steps"] - cfg["warmup"]))
    return cfg["lr"] * (cfg["final_lr_frac"] + (1 - cfg["final_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * p)))


def sample_frames(B, K, device, gen):
    """(B, K) distinct future frames in 1..63 per sequence."""
    return torch.rand(B, S.TF, device=device, generator=gen).argsort(1)[:, :K] + 1


def step_losses(net, b, data, lam, K=None, gen=None, all_frames=False):
    """Objective of one batch and its (tensor) components; also the raw-unit dense error of the frames used (diagnostic)."""
    out = net.latents(b["s0_in"], b["tau"], b["S"])
    parts = {}
    if "rho" in out:
        rho_hat = out["rho"].float()
        L_C = F.mse_loss(rho_hat, b["rho"]); parts["L_C"] = L_C
        parts["E_C"] = (rho_hat - b["rho"]).norm(dim=-1).mean() * data.sigma_r
        return L_C, parts
    z_std = out["z_std"].float()
    L_z = F.mse_loss(z_std, b["z_star"]); parts["L_z"] = L_z
    B = z_std.shape[0]
    t_sel = torch.arange(1, S.T, device=z_std.device)[None].expand(B, -1) if all_frames else sample_frames(B, K, z_std.device, gen)
    Kf = t_sel.shape[1]; idx = t_sel - 1
    g = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1])).reshape(B * Kf, x.shape[-1])
    z_sel = g(z_std); r_sel = g(out["r"].float()) if "r" in out else None
    n_rep = b["n"].repeat_interleave(Kf); Sg = b["S"].repeat_interleave(Kf, 0); s0 = b["s0_raw"].repeat_interleave(Kf, 0)
    geo = data.geo_tokens(n_rep, t_sel.reshape(-1))
    C_bar, delta = net.decode_maps(z_sel, r_sel, geo, Sg)
    rho_gt = g(b["rho"])
    to_rho = lambda Cn: (data.std_to_raw(Cn.float()) - s0) / data.sigma_r
    rho_bar = to_rho(C_bar)
    if delta is None:
        L_C = F.mse_loss(rho_bar, rho_gt); parts["L_C"] = L_C
        parts["E_C"] = (rho_bar - rho_gt).norm(dim=-1).mean() * data.sigma_r
        return L_C + lam["z"] * L_z, parts
    rho_hat = to_rho(C_bar + delta)
    L_full = F.mse_loss(rho_hat, rho_gt); L_zonly = F.mse_loss(rho_bar, rho_gt); L_res = delta.float().abs().mean()
    parts.update(L_full=L_full, L_zonly=L_zonly, L_res=L_res)
    parts["E_C"] = (rho_hat - rho_gt).norm(dim=-1).mean() * data.sigma_r; parts["E_C_zonly"] = (rho_bar - rho_gt).norm(dim=-1).mean() * data.sigma_r
    parts["delta_rms_raw"] = (delta.float() * data.s).pow(2).mean().sqrt()
    return L_full + lam["c"] * L_zonly + lam["z"] * L_z + lam["res"] * L_res, parts


@torch.no_grad()
def evaluate(net, data, idx, lam, bs=16):
    """Training objective on the whole split (all 63 frames decoded), EMA / eval mode."""
    net.eval(); tot = 0.0; cnt = 0; comp = {}
    for i in range(0, len(idx), bs):
        n = idx[i:i + bs]; b = data.train_batch(n)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            l, p = step_losses(net, b, data, lam, all_frames=True)
        tot += float(l) * len(n); cnt += len(n)
        for k, v in p.items():
            comp[k] = comp.get(k, 0.0) + float(v) * len(n)
    net.train()
    return tot / cnt, {k: v / cnt for k, v in comp.items()}


def term_grad_norms(net, parts, lam):
    """Gradient norm of every WEIGHTED loss term w.r.t. all parameters (the Section-12 inspection)."""
    params = [p for p in net.parameters() if p.requires_grad]
    weights = {"L_C": 1.0, "L_full": 1.0, "L_zonly": lam["c"], "L_z": lam["z"], "L_res": lam["res"]}
    out = {}
    for k, w in weights.items():
        if k in parts:
            gs = torch.autograd.grad(w * parts[k], params, retain_graph=True, allow_unused=True)
            out[f"gnorm_{k}"] = float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))
    return out


def ema_update(ema, net, decay):
    with torch.no_grad():
        for pe, pn in zip(ema.parameters(), net.parameters()):
            pe.lerp_(pn, 1 - decay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--model", required=True, choices=list(S.MODELS))
    ap.add_argument("--seed", type=int, default=S.SEED); ap.add_argument("--tag", default="")
    ap.add_argument("--lambda-z", type=float, default=S.TRAIN["lambda_z"]); ap.add_argument("--lambda-c", type=float, default=S.TRAIN["lambda_c"]); ap.add_argument("--lambda-res", type=float, default=S.TRAIN["lambda_res"])
    ap.add_argument("--max-steps", type=int, default=S.TRAIN["max_steps"]); ap.add_argument("--batch", type=int, default=S.TRAIN["batch"]); ap.add_argument("--n-dec-frames", type=int, default=S.TRAIN["n_dec_frames"])
    ap.add_argument("--min-steps", type=int, default=S.TRAIN["min_steps"]); ap.add_argument("--patience", type=int, default=S.TRAIN["patience"]); ap.add_argument("--eval-every", type=int, default=S.TRAIN["eval_every"])
    ap.add_argument("--wandb", action="store_true"); ap.add_argument("--resume", action="store_true", help="continue from CKPT/<ds>/<name>_last.pt")
    a = ap.parse_args()
    ds, model = a.dataset, a.model
    cfg = dict(S.TRAIN, max_steps=a.max_steps, batch=a.batch, n_dec_frames=a.n_dec_frames, min_steps=a.min_steps, patience=a.patience, eval_every=a.eval_every,
               lambda_z=a.lambda_z, lambda_c=a.lambda_c, lambda_res=a.lambda_res)
    lam = dict(z=a.lambda_z, c=a.lambda_c, res=a.lambda_res)
    name = S.run_name(model, a.seed, a.tag)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    data = S2Data(ds, dev, need_masks=False, teacher=True); assert_causal_conditioning(data)
    tck_path = S.teacher_ckpt_path(ds); tck = torch.load(tck_path, map_location="cpu", weights_only=False)
    assert S.md5(tck_path) == data.teacher_info["md5"], "teacher cache was not produced from the recorded Stage-1 checkpoint"
    net = MD.build(model, data, stage1_state=tck["ema_state"] if S.MODELS[model]["z"] else None).to(dev)
    pc = net.param_counts()
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = torch.from_numpy(data.idx["train"]).to(dev); va = torch.from_numpy(data.idx["val"]).to(dev)
    g_batch = torch.Generator(device=dev); g_batch.manual_seed(a.seed); g_frames = torch.Generator(device=dev); g_frames.manual_seed(1000 + a.seed)
    rows, train_rows, inspect_rows = [], [], []
    best, best_state, bad, best_step, start, stopped_by, run = np.inf, None, 0, 0, 1, "max_steps", None
    last_path = S.ckpt_path(ds, name, "last")
    if a.resume and last_path.exists():
        st = torch.load(last_path, map_location=dev, weights_only=False)
        net.load_state_dict(st["state"]); ema.load_state_dict(st["ema_state"]); opt.load_state_dict(st["opt"])
        g_batch.set_state(st["g_batch"].cpu()); g_frames.set_state(st["g_frames"].cpu())        # Generator.set_state needs a CPU ByteTensor (map_location moved them)
        rows, train_rows, inspect_rows = st["rows"], st["train_rows"], st["inspect_rows"]
        best, best_state, bad, best_step, start, run = st["best"], st["best_state"], st["bad"], st["best_step"], st["step"] + 1, st["run"]
        t0 -= st["elapsed"]
        log.info("resumed %s %s at step %d (best %.4f at %d)", ds, name, start - 1, best, best_step)
    wb = None
    if a.wandb:
        try:
            import wandb
            wb = wandb.init(project="dexcore-contact-stage2", name=f"{ds}_{name}", config=dict(dataset=ds, model=model, seed=a.seed, **cfg, backbone=S.BACKBONE, n_params=pc,
                                                                                               teacher=data.teacher_info, ckpt=str(S.ckpt_path(ds, name)), out=str(S.ds_out(ds))),
                            dir=str(S.OUT / "wandb"), resume="allow", id=f"{ds}_{name}", reinit=True)
        except Exception as e:                                                             # noqa: BLE001
            log.warning("wandb disabled: %s", e); wb = None
    v0, comp0 = evaluate(ema, data, va, lam) if start == 1 else (np.nan, {})
    log.info("%s %s (%s): params %s; %d train / %d val sequences; sigma_r %.4f; lambdas %s; teacher %s (md5 %s); val at init %.4f %s",
             ds, model, name, pc, len(tr), len(va), data.sigma_r, lam, data.teacher_info.get("ckpt"), data.teacher_info.get("md5"), v0, {k: round(v, 4) for k, v in comp0.items()})
    net.train()
    for step in range(start, cfg["max_steps"] + 1):
        lr = lr_at(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr
        sel = torch.randint(0, len(tr), (cfg["batch"],), device=dev, generator=g_batch)
        b = data.train_batch(tr[sel])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, parts = step_losses(net, b, data, lam, K=cfg["n_dec_frames"], gen=g_frames)
        if step in cfg["grad_inspect_steps"]:
            gn = term_grad_norms(net, parts, lam); inspect_rows.append(dict(step=step, **gn, **{k: float(v) for k, v in parts.items()}))
            log.info("  grad inspection step %d: %s", step, {k: round(v, 4) for k, v in gn.items()})
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]))
        opt.step(); ema_update(ema, net, cfg["ema"])
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if step % cfg["log_every"] == 0:
            r = dict(step=step, loss=float(loss), loss_ema=run, lr=lr, grad_norm=gnorm, time=time.time() - t0, **{k: float(v) for k, v in parts.items()})
            train_rows.append(r)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in r.items() if k != "step"}, step=step)
        if step % cfg["eval_every"] == 0:
            v, comp = evaluate(ema, data, va, lam)
            rows.append(dict(step=step, train=run, val=v, lr=lr, time=time.time() - t0, **{f"val_{k}": x for k, x in comp.items()}, **{f"train_{k}": float(x) for k, x in parts.items()}))
            if wb is not None:
                wb.log({"val/loss": v, **{f"val/{k}": x for k, x in comp.items()}, "val/best": min(best, v)}, step=step)
            if v < best - 1e-6:
                best, bad, best_step = v, 0, step; best_state = {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
            if step % (cfg["eval_every"] * 5) == 0 or step == cfg["eval_every"]:
                log.info("  step %d train %.4f val %.4f best %.4f (step %d) lr %.2e gnorm %.2f %s (%.0f s)", step, run, v, best, best_step, lr, gnorm,
                         {k: round(x, 4) for k, x in comp.items()}, time.time() - t0)
            torch.save(dict(state=net.state_dict(), ema_state=ema.state_dict(), opt=opt.state_dict(), g_batch=g_batch.get_state(), g_frames=g_frames.get_state(), rows=rows,
                            train_rows=train_rows, inspect_rows=inspect_rows, best=best, best_state=best_state, bad=bad, best_step=best_step, step=step, run=run, elapsed=time.time() - t0),
                       last_path if last_path.parent.exists() else (last_path.parent.mkdir(parents=True, exist_ok=True) or last_path))
            if step >= cfg["min_steps"] and bad >= cfg["patience"]:
                stopped_by = "patience"; break
    if best_state is None:
        raise RuntimeError("no validation check recorded")
    meta = dict(model=model, dataset=ds, seed=a.seed, name=name, cfg=cfg, backbone=S.BACKBONE, lambdas=lam, stats=data.stats, n_params=pc, d_traj=data.d_traj(), d_static=data.d_static(),
                teacher=dict(data.teacher_info, ckpt_md5_at_training=S.md5(tck_path)), stage1_decoder_init=S.MODELS[model]["z"], best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by,
                converged=stopped_by != "max_steps", val_init=v0, val_init_components=comp0, seconds=time.time() - t0, inspect=inspect_rows)
    ck = S.ckpt_path(ds, name); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=best_state, **meta), ck)
    torch.save(dict(ema_state=ema.state_dict(), raw_state=net.state_dict(), **meta), S.ckpt_path(ds, name, "final"))
    S.train_log_path(ds, name).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(S.train_log_path(ds, name), index=False); pd.DataFrame(train_rows).to_csv(S.train_log_path(ds, name + "_train"), index=False)
    pd.DataFrame(inspect_rows).to_csv(S.train_log_path(ds, name + "_gradinspect"), index=False)
    if wb is not None:
        wb.summary.update(dict(best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by, ckpt=str(ck))); wb.finish()
    log.info("done %s: best val %.4f at step %d of %d (%s), %.0f s -> %s", name, best, best_step, step, stopped_by, time.time() - t0, ck)


if __name__ == "__main__":
    main()
