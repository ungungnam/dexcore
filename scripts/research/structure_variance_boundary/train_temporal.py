#!/usr/bin/env python
"""Experiment B — temporal predictive sufficiency: one direct multi-horizon predictor per (dataset, representation,
seed, head).
    CUDA_VISIBLE_DEVICES=<gpu> python train_temporal.py --dataset taco --rep R2 --seed 0 --head structured

Input   the causal window R_i(t-7 .. t) (standardised, flattened) projected by ONE linear layer to the trunk's 512-D
        input — the same width for every representation — plus the FiLM condition of the previous studies: the static
        descriptor G (+ hand / role flags) and the local object trajectory tau_local,t (O_{t-8 .. t+8 step 2}).
Model   the existing VF trunk (3 FiLM residual blocks, width 1024, dropout 0.1, zero-initialised output = the
        train-mean predictor at step 0); one head per horizon h in (1, 4, 8) — direct prediction, no rollout.
Heads   structured: [a logits (6) | m (6) | p, n (36) | q (76)] = 124 per horizon (T1–T4)
        dense:      [C (512) | H (300)] = 812 per horizon (T5, diagnostic)
Loss    BCE for a; MSE on z-scored m, q, C, H (C, H with the scalar scales); MSE on the z-scored p, n restricted to
        the parts active at t + h (GT activity mask); every block weight 1, every horizon weight 1, valid pairs only.
Optim   the previous diagnostic's recipe: AdamW 3e-4, wd 0.05, batch 1024, 300 warm-up steps, EMA 0.99, validation every
        100 steps, LR halved after 6 checks without improvement, stop after 20 (max 30 000 steps); Gaussian input noise
        of 0.5 x the rms one-step change of every input dimension (train statistics).
Writes  <ds>/temporal/{ckpt, train_logs, metrics}/<rep>_<head>_seed<s>.*: per-frame test metrics in raw units.
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

import sv_common as S
from sv_data import Data, part_expand, assert_causal

if str(S.REPO / "scripts/research/hier_contact_gen") not in sys.path:
    sys.path.insert(0, str(S.REPO / "scripts/research/hier_contact_gen"))
if str(S.REPO / "scripts/research/hand_contact_predictive_info") not in sys.path:
    sys.path.insert(0, str(S.REPO / "scripts/research/hand_contact_predictive_info"))
from models import Trunk  # noqa: E402
from hp_models import Cond, n_params  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("temporal")
CFG = dict(lr=3e-4, wd=0.05, warmup=300, batch=1024, ema=0.99, eval_every=100, plateau_patience=6, lr_factor=0.5, patience=20,
           max_steps=30000, dropout=0.1, width=1024, blocks=3, grad_clip=1.0, input_noise=0.5, d_embed=512)
HEADS = {"structured": dict(a=6, m=6, geom=36, q=76), "dense": dict(C=512, H=300)}


class RepPredictor(nn.Module):
    def __init__(self, d_window, d_static, d_traj, d_out, n_h):
        super().__init__()
        self.proj = nn.Linear(d_window, CFG["d_embed"])
        self.cond = Cond(d_static, d_traj, 0)
        self.net = Trunk(CFG["d_embed"], d_out * n_h, 256, CFG["width"], CFG["blocks"], CFG["dropout"])
        self.n_h, self.d_out = n_h, d_out

    def forward(self, x, s, traj):
        return self.net(self.proj(x), self.cond(s, traj)).view(len(x), self.n_h, self.d_out)


def split_out(pred, head):
    out, j = {}, 0
    for k, d in HEADS[head].items():
        out[k] = pred[..., j:j + d]; j += d
    return out


def loss_fn(pred, D, n, t, head):
    """Sum over horizons and blocks of the block-mean losses over the valid pairs."""
    outs = split_out(pred, head); total = 0.0; parts = {}
    for j, h in enumerate(S.HORIZONS):
        tg, valid = D.targets(n, t, h, list(HEADS[head]))
        v = valid.float(); nv = v.sum().clamp(min=1)
        for k in HEADS[head]:
            p = outs[k][:, j]
            if k == "a":
                l = Fnn.binary_cross_entropy_with_logits(p, tg["a"], reduction="none").mean(1)
            elif k == "geom":
                mask = part_expand(tg["a"]); z = D.standardise("geom", tg["geom"])
                l = (((p - z) ** 2) * mask).sum(1) / mask.sum(1).clamp(min=1)
                l = l * (mask.sum(1) > 0).float()
            else:
                l = ((p - D.standardise(k, tg[k])) ** 2).mean(1)
            lb = (l * v).sum() / nv
            total = total + lb; parts[f"{k}_h{h}"] = float(lb)
    return total, parts


@torch.no_grad()
def evaluate(net, D, n_all, t_all, head, bs=4096):
    net.eval(); tot = 0.0
    for i in range(0, len(n_all), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]
        pred = net(D.window(rep_g, n, t), *D.cond(n, t))
        tot += float(loss_fn(pred, D, n, t, head)[0]) * len(n)
    return tot / len(n_all)


def noise_vector(D, rep):
    """0.5 x rms one-step change per input dimension (train sequences), tiled over the window."""
    tr = torch.from_numpy(D.idx["train"]).to(D.device)
    parts = []
    for b in S.LADDER[rep]:
        x = D.std[b][tr][:, S.PAD:S.PAD + S.T]
        parts.append((x[:, 1:] - x[:, :-1]).pow(2).mean((0, 1)).sqrt())
    per_dim = torch.cat(parts)
    return CFG["input_noise"] * per_dim.repeat(S.HIST)


@torch.no_grad()
def test_metrics(net, D, head, out_path, bs=4096):
    n_te, t_te = D.pairs("test"); N = len(n_te)
    M = {}; nh = len(S.HORIZONS)
    ex = n_te.cpu().numpy(); tt = t_te.cpu().numpy(); take = D.take[ex]
    if head == "structured":
        M.update(a_prob=np.zeros((N, nh, 6), np.float16), a_true=np.zeros((N, nh, 6), np.int8), a_now=np.zeros((N, nh, 6), np.int8))
        for k in ("T2_rel_l1", "T2_rel_l1_persist", "T3_cent", "T3_ang", "T3_cent_persist", "T3_ang_persist", "T4_rel_l1", "T4_cos", "T4_Qerr",
                  "T4_rel_l1_persist", "mse_m", "mse_geom", "mse_q", "mse_m_persist", "mse_geom_persist", "mse_q_persist", "valid", "geom_valid"):
            M[k] = np.full((N, nh), np.nan, np.float32)
    else:
        for k in ("E_C", "E0_C", "cos_C", "E_H", "E0_H", "mse_C", "mse_H", "mse_C_persist", "mse_H_persist", "valid"):
            M[k] = np.full((N, nh), np.nan, np.float32)
    for i in range(0, N, bs):
        n, t = n_te[i:i + bs], t_te[i:i + bs]
        pred = net(D.window(rep_g, n, t), *D.cond(n, t)); outs = split_out(pred, head)
        sl = slice(i, i + len(n))
        for j, h in enumerate(S.HORIZONS):
            keys = list(HEADS[head]) + (["a"] if head == "structured" else [])
            tg, valid = D.targets(n, t, h, keys)
            now, _ = D.targets(n, t, 0, keys)
            M["valid"][sl, j] = valid.float().cpu().numpy()
            if head == "structured":
                pa = torch.sigmoid(outs["a"][:, j]); M["a_prob"][sl, j] = pa.cpu().numpy(); M["a_true"][sl, j] = tg["a"].cpu().numpy(); M["a_now"][sl, j] = now["a"].cpu().numpy()
                m_hat = D.destandardise("m", outs["m"][:, j]).clamp(min=0); m_t = tg["m"]
                M["T2_rel_l1"][sl, j] = ((m_hat - m_t).abs().sum(1) / m_t.sum(1).clamp(min=1e-9)).cpu().numpy()
                M["T2_rel_l1_persist"][sl, j] = ((now["m"] - m_t).abs().sum(1) / m_t.sum(1).clamp(min=1e-9)).cpu().numpy()
                M["mse_m"][sl, j] = ((outs["m"][:, j] - D.standardise("m", m_t)) ** 2).mean(1).cpu().numpy()
                M["mse_m_persist"][sl, j] = ((D.standardise("m", now["m"]) - D.standardise("m", m_t)) ** 2).mean(1).cpu().numpy()
                g_hat = D.destandardise("geom", outs["geom"][:, j]); g_t = tg["geom"]; act = tg["a"] > 0.5
                p_hat = g_hat[:, :18].view(-1, 6, 3); p_t = g_t[:, :18].view(-1, 6, 3); n_hat = g_hat[:, 18:].view(-1, 6, 3); n_t = g_t[:, 18:].view(-1, 6, 3)
                p_now = now["geom"][:, :18].view(-1, 6, 3); n_now = now["geom"][:, 18:].view(-1, 6, 3)
                cent = (p_hat - p_t).norm(dim=2); cent_p = (p_now - p_t).norm(dim=2)
                cosn = (n_hat * n_t).sum(2) / (n_hat.norm(dim=2) * n_t.norm(dim=2) + 1e-9); ang = torch.rad2deg(torch.acos(cosn.clamp(-1, 1)))
                cosp = (n_now * n_t).sum(2) / (n_now.norm(dim=2) * n_t.norm(dim=2) + 1e-9); ang_p = torch.rad2deg(torch.acos(cosp.clamp(-1, 1)))
                na = act.float().sum(1); ok = na > 0
                M["geom_valid"][sl, j] = ok.float().cpu().numpy()
                M["T3_cent"][sl, j] = torch.where(ok, (cent * act).sum(1) / na.clamp(min=1), torch.nan).cpu().numpy()
                M["T3_ang"][sl, j] = torch.where(ok, (ang * act).sum(1) / na.clamp(min=1), torch.nan).cpu().numpy()
                act_now = now["a"] > 0.5; both = act & act_now; nb = both.float().sum(1)
                M["T3_cent_persist"][sl, j] = torch.where(nb > 0, (cent_p * both).sum(1) / nb.clamp(min=1), torch.nan).cpu().numpy()
                M["T3_ang_persist"][sl, j] = torch.where(nb > 0, (ang_p * both).sum(1) / nb.clamp(min=1), torch.nan).cpu().numpy()
                mask = part_expand(tg["a"]); z = D.standardise("geom", g_t)
                M["mse_geom"][sl, j] = torch.where(ok, (((outs["geom"][:, j] - z) ** 2) * mask).sum(1) / mask.sum(1).clamp(min=1), torch.nan).cpu().numpy()
                M["mse_geom_persist"][sl, j] = torch.where(ok, (((D.standardise("geom", now["geom"]) - z) ** 2) * mask).sum(1) / mask.sum(1).clamp(min=1), torch.nan).cpu().numpy()
                q_hat = D.destandardise("q", outs["q"][:, j]).clamp(min=0); q_t = tg["q"]; qs = q_t.sum(1); okq = qs > 1e-6
                M["T4_rel_l1"][sl, j] = torch.where(okq, (q_hat - q_t).abs().sum(1) / qs.clamp(min=1e-9), torch.nan).cpu().numpy()
                M["T4_rel_l1_persist"][sl, j] = torch.where(okq, (now["q"] - q_t).abs().sum(1) / qs.clamp(min=1e-9), torch.nan).cpu().numpy()
                M["T4_cos"][sl, j] = torch.where(okq, (q_hat * q_t).sum(1) / (q_hat.norm(dim=1) * q_t.norm(dim=1) + 1e-9), torch.nan).cpu().numpy()
                M["T4_Qerr"][sl, j] = torch.where(okq, (q_hat.mean(1) - q_t.mean(1)).abs() / q_t.mean(1).clamp(min=1e-9), torch.nan).cpu().numpy()
                M["mse_q"][sl, j] = ((outs["q"][:, j] - D.standardise("q", q_t)) ** 2).mean(1).cpu().numpy()
                M["mse_q_persist"][sl, j] = ((D.standardise("q", now["q"]) - D.standardise("q", q_t)) ** 2).mean(1).cpu().numpy()
            else:
                C_hat = D.destandardise("C", outs["C"][:, j]).clamp(min=0); C_t = tg["C"]; C_now = now["C"]
                dC = C_t - C_now; dCh = C_hat - C_now
                M["E_C"][sl, j] = (C_hat - C_t).norm(dim=1).cpu().numpy(); M["E0_C"][sl, j] = dC.norm(dim=1).cpu().numpy()
                M["cos_C"][sl, j] = ((dCh * dC).sum(1) / (dCh.norm(dim=1) * dC.norm(dim=1) + 1e-9)).cpu().numpy()
                M["mse_C"][sl, j] = ((outs["C"][:, j] - D.standardise("C", C_t)) ** 2).mean(1).cpu().numpy()
                M["mse_C_persist"][sl, j] = ((D.standardise("C", C_now) - D.standardise("C", C_t)) ** 2).mean(1).cpu().numpy()
                H_hat = D.destandardise("H", outs["H"][:, j]); H_t = tg["H"]; H_now = now["H"]
                M["E_H"][sl, j] = (H_hat - H_t).view(-1, 100, 3).norm(dim=2).pow(2).mean(1).sqrt().cpu().numpy()
                M["E0_H"][sl, j] = (H_now - H_t).view(-1, 100, 3).norm(dim=2).pow(2).mean(1).sqrt().cpu().numpy()
                M["mse_H"][sl, j] = ((outs["H"][:, j] - D.standardise("H", H_t)) ** 2).mean(1).cpu().numpy()
                M["mse_H_persist"][sl, j] = ((D.standardise("H", H_now) - D.standardise("H", H_t)) ** 2).mean(1).cpu().numpy()
    np.savez_compressed(out_path, example=ex, t=tt, take=take, horizons=np.array(S.HORIZONS), **M)


def main():
    global rep_g
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--rep", required=True, choices=S.REPS)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--head", default="structured", choices=list(HEADS))
    ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    assert_causal()
    rep_g = a.rep
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    t0 = time.time(); ds = a.dataset
    D = Data(ds, dev, need_hand100=True)
    d_out = sum(HEADS[a.head].values())
    net = RepPredictor(S.REP_DIM[a.rep] * S.HIST, D.d_static(), D.d_traj(), d_out, len(S.HORIZONS)).to(dev)
    noise = noise_vector(D, a.rep)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    lr_scale = [1.0]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]) * lr_scale[0])
    n_tr, t_tr = D.pairs("train"); n_va, t_va = D.pairs("val")
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    v_zero = evaluate(ema, D, n_va, t_va, a.head)
    log.info("%s %s %s seed %d: %d params, window %d-D, %d train / %d val pairs, zero-output val loss %.4f", ds, a.rep, a.head, a.seed, n_params(net),
             S.REP_DIM[a.rep] * S.HIST, len(n_tr), len(n_va), v_zero)
    rows, best, best_state, bad, run, best_step, stopped_by = [], np.inf, None, 0, 0.0, 0, "max_steps"
    max_steps = a.max_steps or CFG["max_steps"]
    for step in range(1, max_steps + 1):
        net.train()
        sel = torch.randint(0, len(n_tr), (CFG["batch"],), device=dev, generator=g)
        n, t = n_tr[sel], t_tr[sel]
        x = D.window(a.rep, n, t)
        x = x + noise[None] * torch.randn_like(x)
        pred = net(x, *D.cond(n, t))
        loss, _ = loss_fn(pred, D, n, t, a.head)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), CFG["grad_clip"])
        opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % CFG["eval_every"] == 0:
            v = evaluate(ema, D, n_va, t_va, a.head)
            rows.append(dict(step=step, train=run, val=v, lr=sched.get_last_lr()[0], time=time.time() - t0))
            if v < best - 1e-5:
                best, bad, best_step = v, 0, step; best_state = {k: x_.detach().clone() for k, x_ in ema.state_dict().items()}
            else:
                bad += 1
                if bad % CFG["plateau_patience"] == 0:
                    lr_scale[0] *= CFG["lr_factor"]
            if step % (CFG["eval_every"] * 10) == 0:
                log.info("  step %d train %.4f val %.4f best %.4f (step %d) (%.0f s)", step, run, v, best, best_step, time.time() - t0)
            if bad >= CFG["patience"]:
                stopped_by = "patience"; break
    ema.load_state_dict(best_state)
    out = S.ds_out(ds) / "temporal"
    for sub in ("ckpt", "train_logs", "metrics"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name = f"{a.rep}_{a.head}_seed{a.seed}{a.tag}"
    torch.save(dict(state=best_state, rep=a.rep, head=a.head, seed=a.seed, cfg=CFG, best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by,
                    n_params=n_params(net), zero_val=v_zero, d_window=S.REP_DIM[a.rep] * S.HIST), out / "ckpt" / f"{name}.pt")
    pd.DataFrame(rows).to_csv(out / "train_logs" / f"{name}.csv", index=False)
    test_metrics(ema, D, a.head, out / "metrics" / f"{name}.npz")
    log.info("done %s %s %s seed %d: best val %.4f at step %d of %d (%s), %.0f s", ds, a.rep, a.head, a.seed, best, best_step, step, stopped_by, time.time() - t0)


if __name__ == "__main__":
    main()
