#!/usr/bin/env python
"""Train one model of the 2 x 2 design (one seed).
    CUDA_VISIBLE_DEVICES=4 python zf_train.py --dataset taco --model M11 [--wandb] [--resume] [--mem-cap-mib 20000]

Objective (nothing else):
  M01            L_C + lambda_z L_z                         whole-sequence z_hat_1:63, C_t = s_0 + D_delta(s_0, z_hat_t, G_t)
  M10 / M11      L_C + lambda_z L_z + lambda_z0 L_z0        z_hat_0 = I(s_0, G), 63-step rollout on the model's OWN latents, full back-propagation through the
                                                            rollout (no teacher forcing); absolute (M10) or s_0-preserving (M11) decoding of the rolled-out latents
L_C: MSE in the residual units rho = (C - s_0) / sigma_r of the direct model, on n_dec_frames random future frames per sequence (all 63 at validation).
L_z: MSE of z_hat_1:63 against the teacher latents (standardised).  L_z0: MSE of z_hat_0 against z*_0; with --init-aug A > 0 the initial-state
module is also asked to reproduce the teacher latent of A random later frames of the same sequence from their GT maps (zf_common.INIT_AUG).
Recipe, validation, selection (EMA state at the minimum validation L_C; the state at the minimum full objective is kept too) and early stopping
are those of the previous follow-up (zj_train.py), whose M2 is this study's M00.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import zf_common as Z
import zf_models as MD
from s2_data import S2Data, assert_causal_conditioning
from s2_train import ema_update, lr_at, sample_frames
from zj_train import save_atomic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zf_train")
S2 = Z.S2


def step_losses(net, data, n, cfg, K=None, gen=None, all_frames=False, oracle=False):
    """Objective of one batch of sequences n and its components."""
    B = len(n); dev = n.device; parts = {}
    s0 = data.C[n, 0]; Sg = data.fd.S[n]; z_star = data.teacher_std(n)                                   # (B, 63, dz) teacher latents of frames 1..63
    z_hat, z0 = MD.predict_latents(net, data, n)
    z_hat = z_hat.float()
    L_z = F.mse_loss(z_hat, z_star); parts["L_z"] = L_z
    loss = cfg["lambda_z"] * L_z
    if z0 is not None:
        z0 = z0.float(); z0_star = data.z_standardise(data.z_star[n, 0])
        se = (z0 - z0_star).pow(2); parts["L_z0_frame0"] = se.mean(); A = 0 if all_frames else cfg["init_aug"]
        if A > 0:                                                                                         # the init module on A random later GT frames (training only)
            ta = torch.randint(1, Z.T, (B, A), device=dev, generator=gen); na = n[:, None].expand(-1, A).reshape(-1); ta = ta.reshape(-1)
            za = net.init_state(data.s0_input(data.C[na, ta]), data.geo_tokens(na, ta), data.fd.S[na]).float()
            se = torch.cat([se, (za - data.z_standardise(data.z_star[na, ta])).pow(2)])
        L_z0 = se.mean(); parts["L_z0"] = L_z0; loss = loss + cfg["lambda_z0"] * L_z0
    t_sel = torch.arange(1, Z.T, device=dev)[None].expand(B, -1) if all_frames else sample_frames(B, K, dev, gen)
    Kf = t_sel.shape[1]; idx = t_sel - 1
    g = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1])).reshape(B * Kf, x.shape[-1])
    n_rep = n.repeat_interleave(Kf); Sr = Sg.repeat_interleave(Kf, 0); s0r = s0.repeat_interleave(Kf, 0)
    geo = data.geo_tokens(n_rep, t_sel.reshape(-1))
    rho_gt = g(data.residual(n, s0))
    C_hat = MD.decode_raw(net, data, g(z_hat), geo, Sr, s0r, cfg["dec_chunk"])                            # the decoder reads the PREDICTED latents
    rho = (C_hat - s0r) / data.sigma_r
    L_C = F.mse_loss(rho, rho_gt); parts["L_C"] = L_C; parts["E_C"] = (rho - rho_gt).norm(dim=-1).mean() * data.sigma_r
    loss = loss + L_C
    if oracle:                                                                                            # reported only: the same decoder on the teacher latents
        with torch.no_grad():
            rho_or = (MD.decode_raw(net, data, g(z_star), geo, Sr, s0r, cfg["dec_chunk"]) - s0r) / data.sigma_r
        parts["E_C_gt"] = (rho_or - rho_gt).norm(dim=-1).mean() * data.sigma_r
    return loss, parts


@torch.no_grad()
def evaluate(net, data, idx, cfg, bs=16):
    net.eval(); tot = 0.0; cnt = 0; comp = {}
    for i in range(0, len(idx), bs):
        n = idx[i:i + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            l, p = step_losses(net, data, n, cfg, all_frames=True, oracle=True)
        tot += float(l) * len(n); cnt += len(n)
        for k, v in p.items():
            comp[k] = comp.get(k, 0.0) + float(v) * len(n)
    net.train()
    return tot / cnt, {k: v / cnt for k, v in comp.items()}


def group_grad_norms(net):
    """Gradient norm per module group after the backward pass (inspection at a few steps)."""
    groups = {"init": "init.", "transition": "trans.", "backbone": "backbone.", "z_head": "z_head.", "decoder": "D_z."}; out = {}
    for name, pre in groups.items():
        sq = sum(float(p.grad.float().pow(2).sum()) for k, p in net.named_parameters() if k.startswith(pre) and p.grad is not None)
        if any(k.startswith(pre) for k, _ in net.named_parameters()):
            out[f"gnorm_{name}"] = sq ** 0.5
    return out


def main():
    T_ = Z.TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", required=True, choices=list(Z.TRAINED_HERE))
    ap.add_argument("--seed", type=int, default=Z.SEED); ap.add_argument("--tag", default="")
    ap.add_argument("--init-aug", type=int, default=-1, help="later frames per sequence for the initial-state module (-1: the decision recorded in OUT/init_aug_decision.json)")
    ap.add_argument("--max-steps", type=int, default=T_["max_steps"]); ap.add_argument("--batch", type=int, default=T_["batch"]); ap.add_argument("--n-dec-frames", type=int, default=T_["n_dec_frames"])
    ap.add_argument("--dec-chunk", type=int, default=T_["dec_chunk"]); ap.add_argument("--min-steps", type=int, default=T_["min_steps"]); ap.add_argument("--patience", type=int, default=T_["patience"])
    ap.add_argument("--eval-every", type=int, default=T_["eval_every"]); ap.add_argument("--ckpt-steps", action="store_true", help="activation checkpointing of the rollout steps (memory)")
    ap.add_argument("--no-dec-ckpt", action="store_true", help="no activation checkpointing in the decoder (faster, more memory; the function and its gradients are the same)")
    ap.add_argument("--wandb", action="store_true"); ap.add_argument("--resume", action="store_true"); ap.add_argument("--mem-cap-mib", type=int, default=0); ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--wandb-suffix", default="", help="appended to the wandb run id / name (a run restarted from step 1 after a code fix must not append to the old wandb run)")
    a = ap.parse_args()
    ds, model = a.dataset, a.model; mc = Z.MODELS[model]
    init_aug = 0
    if mc["stateful"]:
        init_aug = a.init_aug if a.init_aug >= 0 else int(Z.read_json(Z.OUT / "init_aug_decision.json")["init_aug_frames"])
    cfg = dict(T_, max_steps=a.max_steps, batch=a.batch, n_dec_frames=a.n_dec_frames, dec_chunk=a.dec_chunk, min_steps=a.min_steps, patience=a.patience, eval_every=a.eval_every,
               init_aug=init_aug, extra_blocks=Z.DECODER["extra_blocks"], stateful=mc["stateful"], residual=mc["residual"], state=dict(Z.STATE, ckpt_steps=a.ckpt_steps) if mc["stateful"] else None)
    name = Z.run_name(model, a.seed, a.tag)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mem_cap_mib and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_mib * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    data = S2Data(ds, dev, need_masks=False, teacher=True); assert_causal_conditioning(data)
    tck_path = S2.teacher_ckpt_path(ds)
    assert Z.md5(tck_path) == data.teacher_info["md5"], "teacher cache was not produced from the recorded Stage-1 checkpoint"
    stage1 = torch.load(tck_path, map_location="cpu", weights_only=False)["ema_state"]
    net = MD.build(model, data, stage1_state=stage1).to(dev); del stage1
    if mc["stateful"]:
        net.ckpt_steps = a.ckpt_steps
    net.ckpt_dec = not a.no_dec_ckpt
    assert not any(k.startswith(("E_z", "E_r")) for k in net.state_dict()), "the Stage-1 encoder must not be part of the model"
    assert all(p.requires_grad for p in net.parameters()), "nothing is frozen"
    pc = net.param_counts()
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    tr = torch.from_numpy(data.idx["train"]).to(dev); va = torch.from_numpy(data.idx["val"]).to(dev)
    g_batch = torch.Generator(device=dev); g_batch.manual_seed(a.seed); g_frames = torch.Generator(device=dev); g_frames.manual_seed(1000 + a.seed)
    rows, train_rows, inspect_rows = [], [], []
    best = dict(sel=np.inf, total=np.inf); best_step = dict(sel=0, total=0); bad = dict(sel=0, total=0); best_state = dict(sel=None, total=None)
    start, stopped_by, run, v0, comp0 = 1, "max_steps", None, np.nan, {}
    last_path = Z.ckpt_path(ds, name, "last")
    if a.resume and last_path.exists() and not a.smoke:
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["state"]); ema.load_state_dict(st["ema_state"]); opt.load_state_dict(st["opt"])
        g_batch.set_state(st["g_batch"].cpu()); g_frames.set_state(st["g_frames"].cpu())
        rows, train_rows, inspect_rows = st["rows"], st["train_rows"], st["inspect_rows"]
        best, best_step, bad, best_state, start, run = st["best"], st["best_step"], st["bad"], st["best_state"], st["step"] + 1, st["run"]
        t0 -= st["elapsed"]; v0, comp0 = st.get("val_init", np.nan), st.get("val_init_components", {})
        log.info("resumed %s %s at step %d (best val L_C %.4f at %d; best objective %.4f at %d)", ds, name, start - 1, best["sel"], best_step["sel"], best["total"], best_step["total"])
        del st
    wb = None
    if a.wandb and not a.smoke:
        try:
            import wandb
            wb = wandb.init(project="dexcore-z-stateful-s0-factorial", name=f"{ds}_{name}{a.wandb_suffix}", config=dict(dataset=ds, model=model, seed=a.seed, **cfg, backbone=Z.BACKBONE, n_params=pc, teacher=data.teacher_info,
                                                                                                       ckpt=str(Z.ckpt_path(ds, name)), out=str(Z.ds_out(ds))),
                            dir=str(Z.OUT / "wandb"), resume="allow", id=f"{ds}_{name}{a.wandb_suffix}", reinit=True)
        except Exception as e:                                                     # noqa: BLE001
            log.warning("wandb disabled: %s", e); wb = None
    if start == 1 and not a.smoke:
        v0, comp0 = evaluate(ema, data, va, cfg)
    log.info("%s %s (%s): params %s; %d train / %d val sequences; sigma_r %.4f; cfg %s; val at init %.4f %s", ds, model, name, pc, len(tr), len(va), data.sigma_r,
             {k: cfg[k] for k in ("lambda_z", "lambda_z0", "init_aug", "batch", "n_dec_frames", "dec_chunk", "max_steps", "min_steps", "patience", "stateful", "residual")}, v0, {k: round(v, 4) for k, v in comp0.items()})
    net.train()
    max_steps = a.smoke or cfg["max_steps"]; step = start - 1
    if a.resume and start > 1 and start - 1 >= cfg["min_steps"] and bad["sel"] >= cfg["patience"] and bad["total"] >= cfg["patience"]:
        stopped_by, max_steps = "patience", start - 1
    for step in range(start, max_steps + 1):
        lr = lr_at(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr
        n = tr[torch.randint(0, len(tr), (cfg["batch"],), device=dev, generator=g_batch)]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, parts = step_losses(net, data, n, cfg, K=cfg["n_dec_frames"], gen=g_frames)
        opt.zero_grad(set_to_none=True); loss.backward()
        if step in cfg["grad_inspect_steps"]:
            gn = group_grad_norms(net); inspect_rows.append(dict(step=step, **gn, **{k: float(v) for k, v in parts.items()}))
            log.info("  grad inspection step %d: %s", step, {k: round(v, 4) for k, v in gn.items()})
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]))
        opt.step(); ema_update(ema, net, cfg["ema"])
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if a.smoke:
            if step == 1:
                t_smoke = time.time()
            if step in (1, 5) or step == max_steps:
                torch.cuda.synchronize(); log.info("  smoke step %d: loss %.4f %s, %.2f s/step, peak GPU memory %.0f MiB (reserved %.0f)", step, float(loss), {k: round(float(v), 4) for k, v in parts.items()},
                                                   (time.time() - t_smoke) / (step - 1) if step > 1 else float("nan"), torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20)
            if step == max_steps:
                ts = time.time(); v, comp = evaluate(ema, data, va, cfg); torch.cuda.synchronize()
                log.info("  smoke validation: %.1f s, objective %.4f %s; peak %.0f MiB (reserved %.0f)", time.time() - ts, v, {k: round(x, 4) for k, x in comp.items()},
                         torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20)
            continue
        if step % cfg["log_every"] == 0:
            r = dict(step=step, loss=float(loss), loss_ema=run, lr=lr, grad_norm=gnorm, time=time.time() - t0, **{k: float(v) for k, v in parts.items()})
            train_rows.append(r)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in r.items() if k != "step"}, step=step)
        if step % cfg["eval_every"] == 0:
            v, comp = evaluate(ema, data, va, cfg)
            crit = dict(sel=comp["L_C"], total=v)
            rows.append(dict(step=step, train=run, val=v, lr=lr, time=time.time() - t0, **{f"val_{k}": x for k, x in comp.items()}, **{f"train_{k}": float(x) for k, x in parts.items()}))
            for c in ("sel", "total"):
                if crit[c] < best[c] - 1e-6:
                    best[c], bad[c], best_step[c] = crit[c], 0, step
                    best_state[c] = {k: x.detach().cpu().clone() for k, x in ema.state_dict().items()}
                else:
                    bad[c] += 1
            if wb is not None:
                wb.log({"val/loss": v, **{f"val/{k}": x for k, x in comp.items()}, "val/best_L_C": best["sel"], "val/best_objective": best["total"]}, step=step)
            if step % (cfg["eval_every"] * 5) == 0 or step == cfg["eval_every"]:
                log.info("  step %d train %.4f val %.4f | val L_C %.4f best %.4f (step %d) | objective best %.4f (step %d) | lr %.2e gnorm %.2f %s (%.0f s)", step, run, v, comp["L_C"], best["sel"], best_step["sel"],
                         best["total"], best_step["total"], lr, gnorm, {k: round(x, 4) for k, x in comp.items()}, time.time() - t0)
            save_atomic(dict(state=net.state_dict(), ema_state=ema.state_dict(), opt=opt.state_dict(), g_batch=g_batch.get_state(), g_frames=g_frames.get_state(), rows=rows, train_rows=train_rows,
                             inspect_rows=inspect_rows, best=best, best_step=best_step, bad=bad, best_state=best_state, step=step, run=run, elapsed=time.time() - t0, val_init=v0, val_init_components=comp0), last_path)
            if step >= cfg["min_steps"] and bad["sel"] >= cfg["patience"] and bad["total"] >= cfg["patience"]:
                stopped_by = "patience"; break
    if a.smoke:
        return
    if best_state["sel"] is None:
        raise RuntimeError("no validation check recorded")
    meta = dict(model=model, dataset=ds, seed=a.seed, name=name, cfg=cfg, backbone=Z.BACKBONE, stats=data.stats, n_params=pc, d_traj=data.d_traj(), d_static=data.d_static(),
                teacher=dict(data.teacher_info, ckpt_md5_at_training=Z.md5(tck_path)), selection="validation L_C of the predicted-z path",
                best_val=best["sel"], best_step=best_step["sel"], best_val_total=best["total"], best_step_total=best_step["total"], steps=step, stopped_by=stopped_by, converged=stopped_by != "max_steps",
                val_init=v0, val_init_components=comp0, seconds=time.time() - t0, inspect=inspect_rows)
    save_atomic(dict(state=best_state["sel"], state_total=best_state["total"], **meta), Z.ckpt_path(ds, name))
    save_atomic(dict(ema_state=ema.state_dict(), raw_state=net.state_dict(), **meta), Z.ckpt_path(ds, name, "final"))
    Z.train_log_path(ds, name).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Z.train_log_path(ds, name), index=False); pd.DataFrame(train_rows).to_csv(Z.train_log_path(ds, name + "_train"), index=False)
    pd.DataFrame(inspect_rows).to_csv(Z.train_log_path(ds, name + "_gradinspect"), index=False)
    if wb is not None:
        wb.summary.update(dict(best_val_L_C=best["sel"], best_step=best_step["sel"], best_objective=best["total"], best_step_objective=best_step["total"], steps=step, stopped_by=stopped_by, ckpt=str(Z.ckpt_path(ds, name)))); wb.finish()
    log.info("done %s: best val L_C %.4f at step %d (objective %.4f at step %d) of %d (%s), %.0f s -> %s", name, best["sel"], best_step["sel"], best["total"], best_step["total"], step, stopped_by, time.time() - t0, Z.ckpt_path(ds, name))


if __name__ == "__main__":
    main()
