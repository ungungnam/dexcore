#!/usr/bin/env python
"""Train one model of the follow-up (one seed).
    CUDA_VISIBLE_DEVICES=5 python zj_train.py --dataset taco --model M2 [--seed 0] [--wandb] [--resume] [--mem-cap-mib 14000]

M2   L = beta_pred L_C(D_phi(z_hat)) + lambda_z L_z                         the decoder's only training input is the PREDICTED z_hat
M3   L = beta_pred L_C(D_phi(z_hat)) + beta_gt L_C(D_phi(z*)) + lambda_z L_z   same decoder, also fed the teacher z* (0.25 vs 1.0; the teacher term on
                                                                             n_gt_frames = 1 of the 4 random frames per sequence: an unbiased estimate at +25 % cost)
M0r  L = L_C(rho_hat)                                                        the direct model under this protocol (budget-parity check)
L_C: MSE in the residual units rho = (C - s_0) / sigma_r (the B0 loss); L_z: MSE in standardised teacher coordinates over all 63 frames.
No other loss term.  The dense terms use n_dec_frames random future frames per sequence per step; validation decodes all 63 frames.

Recipe: AdamW lr 1e-4, wd 0.01, 5000 warm-up steps, cosine decay to 0.05 x over 100 000 steps, batch 128 sequences (GT s_0), grad clip 1.0,
EMA 0.999 (the EMA weights are validated and saved), bf16 autocast, validation every 1000 steps on the whole validation set.
Selection: the EMA state at the minimum validation L_C of the predicted-z path (the final-task loss; for M0r this is its whole objective).
The state at the minimum of the full objective (the Stage-2 B1 criterion) is kept too ("state_total").
Early stopping: not before 25 000 steps, and only when neither criterion improved for 15 consecutive checks.
Writes CKPT/<ds>/<name>.pt (selected state + state_total + metadata), <name>_final.pt, a resumable <name>_last.pt at every validation,
<ds>/train_logs/<name>.csv (validation rows), <name>_train.csv (training rows), <name>_gradinspect.csv.
"""
from __future__ import annotations

import argparse
import copy
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import zj_common as Z
import zj_models as MD
from s2_data import S2Data, assert_causal_conditioning
from s2_train import ema_update, lr_at, sample_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_train")
S2 = Z.S2


def step_losses(net, b, data, cfg, K=None, gen=None, all_frames=False, dual=False, oracle=False):
    """Objective of one batch and its components.  dual: add beta_gt * L_C of the teacher-z path (M3).  oracle: also report that path
    without using it in the objective (validation of M2: the decoder's floor on the clean teacher latent)."""
    out = net.latents(b["s0_in"], b["tau"], b["S"])
    parts = {}
    if "rho" in out:                                                               # direct model
        rho_hat = out["rho"].float()
        L_C = F.mse_loss(rho_hat, b["rho"]); parts["L_C"] = L_C
        parts["E_C"] = (rho_hat - b["rho"]).norm(dim=-1).mean() * data.sigma_r
        return L_C, parts
    z_hat = out["z_std"].float()                                                   # (B, 63, dz) predicted, standardised teacher coordinates
    L_z = F.mse_loss(z_hat, b["z_star"]); parts["L_z"] = L_z
    B = z_hat.shape[0]
    t_sel = torch.arange(1, Z.T, device=z_hat.device)[None].expand(B, -1) if all_frames else sample_frames(B, K, z_hat.device, gen)
    Kf = t_sel.shape[1]; idx = t_sel - 1
    g = lambda x: torch.gather(x, 1, idx[..., None].expand(-1, -1, x.shape[-1])).reshape(B * Kf, x.shape[-1])
    n_rep = b["n"].repeat_interleave(Kf); Sg = b["S"].repeat_interleave(Kf, 0); s0 = b["s0_raw"].repeat_interleave(Kf, 0)
    geo = data.geo_tokens(n_rep, t_sel.reshape(-1))
    rho_gt = g(b["rho"])
    to_rho = lambda Cn, s0_=s0: (data.std_to_raw(Cn.float()) - s0_) / data.sigma_r
    ch = cfg["dec_chunk"]
    rho_pred = to_rho(MD.decode(net, g(z_hat), geo, Sg, ch))                       # the decoder reads the PREDICTED z_hat (gradient flows into F_theta too)
    L_C = F.mse_loss(rho_pred, rho_gt); parts["L_C"] = L_C
    parts["E_C"] = (rho_pred - rho_gt).norm(dim=-1).mean() * data.sigma_r
    loss = cfg["beta_pred"] * L_C + cfg["lambda_z"] * L_z
    if dual or oracle:
        # the same decoder on the teacher z* : all selected frames at validation, the first n_gt_frames of the K random frames in training
        rows = slice(None) if all_frames else ((torch.arange(B * Kf, device=z_hat.device) % Kf) < cfg["n_gt_frames"])
        with torch.set_grad_enabled(dual and torch.is_grad_enabled()):
            rho_or = to_rho(MD.decode(net, g(b["z_star"])[rows], geo[rows], Sg[rows], ch), s0[rows])
            L_gt = F.mse_loss(rho_or, rho_gt[rows])
        parts["L_C_gt"] = L_gt; parts["E_C_gt"] = (rho_or - rho_gt[rows]).norm(dim=-1).mean() * data.sigma_r
        if dual:
            loss = loss + cfg["beta_gt"] * L_gt
    return loss, parts


@torch.no_grad()
def evaluate(net, data, idx, cfg, dual, bs=16):
    """Objective and components on a whole split (all 63 frames decoded; teacher-z path reported for the z models), eval mode."""
    net.eval(); tot = 0.0; cnt = 0; comp = {}
    for i in range(0, len(idx), bs):
        n = idx[i:i + bs]; b = data.train_batch(n)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            l, p = step_losses(net, b, data, cfg, all_frames=True, dual=dual, oracle=True)
        tot += float(l) * len(n); cnt += len(n)
        for k, v in p.items():
            comp[k] = comp.get(k, 0.0) + float(v) * len(n)
    net.train()
    return tot / cnt, {k: v / cnt for k, v in comp.items()}


def term_grad_norms(net, parts, cfg, dual):
    """Gradient norm of every WEIGHTED loss term w.r.t. the backbone (+ z head) and w.r.t. the decoder."""
    weights = {"L_C": cfg["beta_pred"], "L_z": cfg["lambda_z"]}
    if dual:
        weights["L_C_gt"] = cfg["beta_gt"]
    groups = {"dec": [p for k, p in net.named_parameters() if k.startswith("D_z.")], "trunk": [p for k, p in net.named_parameters() if not k.startswith("D_z.")]}
    out = {}
    for k, w in weights.items():
        if k not in parts:
            continue
        for gname, params in groups.items():
            if not params:
                continue
            if not parts[k].requires_grad:
                continue
            gs = torch.autograd.grad(w * parts[k], params, retain_graph=True, allow_unused=True)
            out[f"gnorm_{k}_{gname}"] = sum(float((g.float() ** 2).sum()) for g in gs if g is not None) ** 0.5   # 0.0 when the term does not reach the group (L_z -> decoder)
    return out


def decoder_change(net, init_dec):
    """How far the decoder moved from its initialisation (Stage-1 weights for the loaded blocks, identity for the extra ones)."""
    if init_dec is None:
        return {}
    num = den = 0.0; gate = 0.0
    for k, p in net.D_z.state_dict().items():
        if k in init_dec:
            num += float((p.float() - init_dec[k].to(p.device)).pow(2).sum()); den += float(init_dec[k].pow(2).sum())
    n1 = getattr(net, "n_stage1_blocks", len(net.D_z.blocks))
    for i in range(n1, len(net.D_z.blocks)):                                       # extra blocks: norm of the zero-initialised FiLM layer (0 = still the identity)
        gate += float(net.D_z.blocks[i].ada[1].weight.float().pow(2).sum() + net.D_z.blocks[i].ada[1].bias.float().pow(2).sum())
    return dict(dec_rel_change=(num / max(den, 1e-12)) ** 0.5, dec_extra_ada_norm=gate ** 0.5)


def save_atomic(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp); os.replace(tmp, path)


def main():
    T_ = Z.TRAIN
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", required=True, choices=list(Z.TRAINED_HERE))
    ap.add_argument("--seed", type=int, default=Z.SEED); ap.add_argument("--tag", default="")
    ap.add_argument("--extra-blocks", type=int, default=Z.DECODER["extra_blocks"])
    ap.add_argument("--max-steps", type=int, default=T_["max_steps"]); ap.add_argument("--batch", type=int, default=T_["batch"]); ap.add_argument("--n-dec-frames", type=int, default=T_["n_dec_frames"])
    ap.add_argument("--dec-chunk", type=int, default=T_["dec_chunk"]); ap.add_argument("--n-gt-frames", type=int, default=T_["n_gt_frames"])
    ap.add_argument("--min-steps", type=int, default=T_["min_steps"]); ap.add_argument("--patience", type=int, default=T_["patience"]); ap.add_argument("--eval-every", type=int, default=T_["eval_every"])
    ap.add_argument("--wandb", action="store_true"); ap.add_argument("--resume", action="store_true", help="continue from CKPT/<ds>/<name>_last.pt")
    ap.add_argument("--mem-cap-mib", type=int, default=0, help="hard cap of this process's GPU memory (shared GPUs: fail here instead of squeezing other jobs)")
    ap.add_argument("--smoke", type=int, default=0, help="run this many steps with a validation at the end, write nothing (timing / memory check)")
    a = ap.parse_args()
    ds, model = a.dataset, a.model; mc = Z.MODELS[model]; dual = mc["dual"]; joint = mc["joint"]
    cfg = dict(T_, max_steps=a.max_steps, batch=a.batch, n_dec_frames=a.n_dec_frames, min_steps=a.min_steps, patience=a.patience, eval_every=a.eval_every, extra_blocks=a.extra_blocks if joint else 0,
               dec_chunk=a.dec_chunk, n_gt_frames=a.n_gt_frames)
    name = Z.run_name(model, a.seed, a.tag)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.mem_cap_mib and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_cap_mib * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time()
    data = S2Data(ds, dev, need_masks=False, teacher=True); assert_causal_conditioning(data)
    tck_path = S2.teacher_ckpt_path(ds)
    assert Z.md5(tck_path) == data.teacher_info["md5"], "teacher cache was not produced from the recorded Stage-1 checkpoint"
    stage1 = torch.load(tck_path, map_location="cpu", weights_only=False)["ema_state"] if joint else None
    if joint:
        net = MD.JointModel(data.d_traj(), data.d_static(), data.d_geo(), data.z_mu, data.z_sd, stage1_state=stage1, extra_blocks=a.extra_blocks).to(dev)
        assert not any(k.startswith(("E_z", "E_r")) for k in net.state_dict()), "the Stage-1 encoder must not be part of the model"
        init_dec = {k: v.detach().float().cpu().clone() for k, v in net.D_z.state_dict().items()}
    else:
        net = MD.build(model, data).to(dev); init_dec = None
    del stage1
    pc = net.param_counts()
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    assert all(p.requires_grad for p in net.parameters()), "nothing is frozen"
    tr = torch.from_numpy(data.idx["train"]).to(dev); va = torch.from_numpy(data.idx["val"]).to(dev)
    g_batch = torch.Generator(device=dev); g_batch.manual_seed(a.seed); g_frames = torch.Generator(device=dev); g_frames.manual_seed(1000 + a.seed)
    rows, train_rows, inspect_rows = [], [], []
    best = dict(sel=np.inf, total=np.inf); best_step = dict(sel=0, total=0); bad = dict(sel=0, total=0); best_state = dict(sel=None, total=None)
    start, stopped_by, run, v0, comp0 = 1, "max_steps", None, np.nan, {}
    last_path = Z.ckpt_path(ds, name, "last")
    if a.resume and last_path.exists() and not a.smoke:
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["state"]); ema.load_state_dict(st["ema_state"]); opt.load_state_dict(st["opt"])
        g_batch.set_state(st["g_batch"].cpu()); g_frames.set_state(st["g_frames"].cpu())   # Generator.set_state needs a CPU ByteTensor
        rows, train_rows, inspect_rows = st["rows"], st["train_rows"], st["inspect_rows"]
        best, best_step, bad, best_state, start, run = st["best"], st["best_step"], st["bad"], st["best_state"], st["step"] + 1, st["run"]
        t0 -= st["elapsed"]; v0, comp0 = st.get("val_init", np.nan), st.get("val_init_components", {})
        log.info("resumed %s %s at step %d (best val L_C %.4f at %d; best objective %.4f at %d)", ds, name, start - 1, best["sel"], best_step["sel"], best["total"], best_step["total"])
        del st
    wb = None
    if a.wandb and not a.smoke:
        try:
            import wandb
            wb = wandb.init(project="dexcore-z-joint-decoder", name=f"{ds}_{name}", config=dict(dataset=ds, model=model, seed=a.seed, **cfg, backbone=Z.BACKBONE, n_params=pc, teacher=data.teacher_info,
                                                                                               ckpt=str(Z.ckpt_path(ds, name)), out=str(Z.ds_out(ds))),
                            dir=str(Z.OUT / "wandb"), resume="allow", id=f"{ds}_{name}", reinit=True)
        except Exception as e:                                                     # noqa: BLE001
            log.warning("wandb disabled: %s", e); wb = None
    if start == 1 and not a.smoke:
        v0, comp0 = evaluate(ema, data, va, cfg, dual)
    log.info("%s %s (%s): params %s; %d train / %d val sequences; sigma_r %.4f; cfg %s; teacher md5 %s; val at init %.4f %s", ds, model, name, pc, len(tr), len(va), data.sigma_r,
             {k: cfg[k] for k in ("lambda_z", "beta_pred", "beta_gt", "batch", "n_dec_frames", "n_gt_frames", "dec_chunk", "max_steps", "min_steps", "patience", "extra_blocks")}, data.teacher_info.get("md5"), v0, {k: round(v, 4) for k, v in comp0.items()})
    net.train()
    max_steps = a.smoke or cfg["max_steps"]
    step = start - 1                                                               # defined even when a resumed run is already at its last step
    if a.resume and start > 1 and start - 1 >= cfg["min_steps"] and bad["sel"] >= cfg["patience"] and bad["total"] >= cfg["patience"]:
        stopped_by, max_steps = "patience", start - 1                              # the run had already met the stopping rule before it was interrupted
    for step in range(start, max_steps + 1):
        lr = lr_at(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr
        sel = torch.randint(0, len(tr), (cfg["batch"],), device=dev, generator=g_batch)
        b = data.train_batch(tr[sel])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            loss, parts = step_losses(net, b, data, cfg, K=cfg["n_dec_frames"], gen=g_frames, dual=dual)
        if joint and step in cfg["grad_inspect_steps"]:
            gn = term_grad_norms(net, parts, cfg, dual); inspect_rows.append(dict(step=step, **gn, **{k: float(v) for k, v in parts.items()}))
            log.info("  grad inspection step %d: %s", step, {k: round(v, 4) for k, v in gn.items()})
        opt.zero_grad(set_to_none=True); loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"]))
        opt.step(); ema_update(ema, net, cfg["ema"])
        run = float(loss) if run is None else 0.98 * run + 0.02 * float(loss)
        if a.smoke:
            if step in (1, 5) or step == max_steps:
                torch.cuda.synchronize(); log.info("  smoke step %d: loss %.4f, %.2f s/step, peak GPU memory %.0f MiB (reserved %.0f)", step, float(loss), (time.time() - t_smoke) / (step - 1) if step > 1 else float("nan"),
                                                   torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20)
            if step == 1:
                t_smoke = time.time()
            if step == max_steps:
                torch.cuda.synchronize(); ts = time.time(); v, comp = evaluate(ema, data, va, cfg, dual); torch.cuda.synchronize()
                log.info("  smoke validation: %.1f s, objective %.4f %s; peak %.0f MiB (reserved %.0f); decoder change %s", time.time() - ts, v, {k: round(x, 4) for k, x in comp.items()},
                         torch.cuda.max_memory_allocated() / 2 ** 20, torch.cuda.max_memory_reserved() / 2 ** 20, decoder_change(net, init_dec))
            continue
        if step % cfg["log_every"] == 0:
            r = dict(step=step, loss=float(loss), loss_ema=run, lr=lr, grad_norm=gnorm, time=time.time() - t0, **{k: float(v) for k, v in parts.items()})
            train_rows.append(r)
            if wb is not None:
                wb.log({f"train/{k}": v for k, v in r.items() if k != "step"}, step=step)
        if step % cfg["eval_every"] == 0:
            v, comp = evaluate(ema, data, va, cfg, dual)
            crit = dict(sel=comp["L_C"], total=v)
            dc = decoder_change(ema, init_dec)
            rows.append(dict(step=step, train=run, val=v, lr=lr, time=time.time() - t0, **{f"val_{k}": x for k, x in comp.items()}, **{f"train_{k}": float(x) for k, x in parts.items()}, **dc))
            for c in ("sel", "total"):
                if crit[c] < best[c] - 1e-6:
                    best[c], bad[c], best_step[c] = crit[c], 0, step
                    best_state[c] = {k: x.detach().cpu().clone() for k, x in ema.state_dict().items()}   # kept on the CPU (shared GPUs)
                else:
                    bad[c] += 1
            if wb is not None:
                wb.log({"val/loss": v, **{f"val/{k}": x for k, x in comp.items()}, "val/best_L_C": best["sel"], "val/best_objective": best["total"], **{f"decoder/{k}": x for k, x in dc.items()}}, step=step)
            if step % (cfg["eval_every"] * 5) == 0 or step == cfg["eval_every"]:
                log.info("  step %d train %.4f val %.4f | val L_C %.4f best %.4f (step %d) | objective best %.4f (step %d) | lr %.2e gnorm %.2f %s (%.0f s)", step, run, v, comp["L_C"], best["sel"], best_step["sel"],
                         best["total"], best_step["total"], lr, gnorm, {k: round(x, 4) for k, x in {**comp, **dc}.items()}, time.time() - t0)
            save_atomic(dict(state=net.state_dict(), ema_state=ema.state_dict(), opt=opt.state_dict(), g_batch=g_batch.get_state(), g_frames=g_frames.get_state(), rows=rows, train_rows=train_rows,
                             inspect_rows=inspect_rows, best=best, best_step=best_step, bad=bad, best_state=best_state, step=step, run=run, elapsed=time.time() - t0,
                             val_init=v0, val_init_components=comp0), last_path)
            if step >= cfg["min_steps"] and bad["sel"] >= cfg["patience"] and bad["total"] >= cfg["patience"]:
                stopped_by = "patience"; break
    if a.smoke:
        return
    if best_state["sel"] is None:
        raise RuntimeError("no validation check recorded")
    meta = dict(model=model, dataset=ds, seed=a.seed, name=name, cfg=cfg, backbone=Z.BACKBONE, stats=data.stats, n_params=pc, d_traj=data.d_traj(), d_static=data.d_static(),
                teacher=dict(data.teacher_info, ckpt_md5_at_training=Z.md5(tck_path)), stage1_decoder_init=joint, selection="validation L_C of the predicted-z path" if joint else "validation L_C",
                best_val=best["sel"], best_step=best_step["sel"], best_val_total=best["total"], best_step_total=best_step["total"], steps=step, stopped_by=stopped_by, converged=stopped_by != "max_steps",
                val_init=v0, val_init_components=comp0, seconds=time.time() - t0, inspect=inspect_rows, decoder_change_final=decoder_change(ema, init_dec))
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
