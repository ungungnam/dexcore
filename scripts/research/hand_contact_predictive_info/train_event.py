#!/usr/bin/env python
"""Step 3: the event-prediction classifier per (dataset, condition, seed).
    CUDA_VISIBLE_DEVICES=<gpu> python train_event.py --dataset taco --cond F2k8 --seed 0

Question  does the information available at frame t tell that a PERSISTENT contact-mode
          transition (persistent spatial / mixed, release, onset+release, onset = regrasp) is about
          to START within the next h frames, h in (4, 8)?
Label     y_t^(h) = 1 iff such an event's first spike transition s satisfies t < s <= t + h; the
          inputs of frame t (C_t, tau_local,t, hand frames <= t) therefore end strictly before the
          event's first transition (anti-leakage rule; hp_common.event_labels). Frames with
          t > 62 - h have no label.
Model     the same MLP for every condition: [C~_t, G, tau_local,t, hand window] -> 512 -> 256 -> 2
          logits (one per horizon), SiLU, dropout 0.1. BCE with pos_weight = n_neg / n_pos of the
          training labels per horizon; AdamW 3e-4, wd 0.05, batch 1024; validation mean AUPRC
          every 200 steps, stop after 10 checks without improvement or 20 000 steps.
Writes    OUT/<ds>/ckpt/event_<cond>_seed<s>.pt, OUT/<ds>/train_logs/event_<cond>_seed<s>.csv,
          OUT/<ds>/preds/event_<cond>_seed<s>.npz (test probabilities (N_test, 63, 2), labels, masks)
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F_
from sklearn.metrics import average_precision_score, roc_auc_score

import hp_common as P
import hp_data as HD
from hp_models import EventClassifier, n_params

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("train_event")
CFG = dict(lr=3e-4, wd=0.05, batch=1024, eval_every=100, patience=15, max_steps=20000, dropout=0.1, width=512, warmup=200, ema=0.99)


def labels(ds, fd, dev):
    Fr, E = P.load_frames(ds), P.load_events(ds)
    ys, vs = [], []
    for h in P.EVENT_HORIZONS:
        y, v = P.event_labels(Fr, E, h); ys.append(y); vs.append(v)
    Y = torch.from_numpy(np.stack(ys, -1).astype(np.float32)).to(dev)         # (N, 63, n_h)
    V = torch.from_numpy(np.stack(vs, -1).astype(np.float32)).to(dev)
    return Y, V


@torch.no_grad()
def predict(net, fd, ht, cond, n_all, t_all, perm, bs=8192):
    net.eval(); out = []
    for i in range(0, len(n_all), bs):
        n, t = n_all[i:i + bs], t_all[i:i + bs]
        out.append(torch.sigmoid(net(HD.classifier_input(fd, ht, cond, n, t, perm))).cpu())
    net.train()
    return torch.cat(out).numpy()


def scores(prob, y, v):
    """prob, y, v (n, n_h) -> per-horizon AUPRC / AUROC over the valid frames."""
    ap, au = [], []
    for j in range(prob.shape[1]):
        m = v[:, j] > 0
        ap.append(average_precision_score(y[m, j], prob[m, j])); au.append(roc_auc_score(y[m, j], prob[m, j]))
    return np.array(ap), np.array(au)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=P.DATASETS); ap.add_argument("--cond", required=True, choices=list(P.CONDITIONS))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(); ds, cond = a.dataset, a.cond
    P.assert_causal()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed); t0 = time.time()
    fd = P.fold(ds, dev); ht = HD.HandTensors(ds, fd, dev); perm = HD.shuffle_map(fd, a.seed)
    Y, V = labels(ds, fd, dev)
    n_tr, t_tr = HD.pairs(fd, "train"); n_va, t_va = HD.pairs(fd, "val"); n_te, t_te = HD.pairs(fd, "test")
    d_in = HD.classifier_input(fd, ht, cond, n_tr[:2], t_tr[:2], perm).shape[1]
    net = EventClassifier(d_in, width=CFG["width"], dropout=CFG["dropout"]).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    y_tr, v_tr = Y[n_tr, t_tr], V[n_tr, t_tr]
    pos = (y_tr * v_tr).sum(0); neg = ((1 - y_tr) * v_tr).sum(0)
    pos_weight = neg / pos.clamp(min=1)
    opt = torch.optim.AdamW(net.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0, (st + 1) / CFG["warmup"]))
    g = torch.Generator(device=dev); g.manual_seed(a.seed)
    y_va, v_va = Y[n_va, t_va].cpu().numpy(), V[n_va, t_va].cpu().numpy()
    log.info("%s %s seed %d: %d params, d_in %d, train pos %s of %s valid (pos_weight %s)", ds, cond, a.seed, n_params(net), d_in,
             pos.tolist(), v_tr.sum(0).tolist(), [round(float(x), 2) for x in pos_weight])
    rows, best, best_state, bad, best_step, stopped_by, run = [], -np.inf, None, 0, 0, "max_steps", 0.0
    for step in range(1, (a.max_steps or CFG["max_steps"]) + 1):
        sel = torch.randint(0, len(n_tr), (CFG["batch"],), device=dev, generator=g)
        n, t = n_tr[sel], t_tr[sel]
        logit = net(HD.classifier_input(fd, ht, cond, n, t, perm))
        y, v = Y[n, t], V[n, t]
        loss = (F_.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight, reduction="none") * v).sum() / v.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - CFG["ema"])
        run = 0.98 * run + 0.02 * float(loss) if step > 1 else float(loss)
        if step % CFG["eval_every"] == 0:
            ap_, au_ = scores(predict(ema, fd, ht, cond, n_va, t_va, perm), y_va, v_va)
            v_score = float(ap_.mean())
            rows.append(dict(step=step, train=run, val_auprc_mean=v_score, **{f"val_auprc_h{h}": float(x) for h, x in zip(P.EVENT_HORIZONS, ap_)},
                             **{f"val_auroc_h{h}": float(x) for h, x in zip(P.EVENT_HORIZONS, au_)}, time=time.time() - t0))
            if v_score > best + 1e-5:
                best, bad, best_step = v_score, 0, step; best_state = {k: x.detach().clone() for k, x in ema.state_dict().items()}
            else:
                bad += 1
            if step % (CFG["eval_every"] * 10) == 0:
                log.info("  step %d train %.4f val AUPRC %s best %.4f (step %d) (%.0fs)", step, run, [round(float(x), 3) for x in ap_], best, best_step, time.time() - t0)
            if bad >= CFG["patience"]:
                stopped_by = "patience"; break
    ema.load_state_dict(best_state)
    out = P.ds_out(ds)
    for sub in ("ckpt", "preds", "train_logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    name = f"event_{cond}_seed{a.seed}"
    prob = predict(ema, fd, ht, cond, n_te, t_te, perm)
    y_te, v_te = Y[n_te, t_te].cpu().numpy(), V[n_te, t_te].cpu().numpy()
    ap_, au_ = scores(prob, y_te, v_te)
    n_test = len(fd.idx["test"])
    np.savez_compressed(out / "preds" / f"{name}.npz", prob=prob.reshape(n_test, fd.T - 1, -1).astype(np.float32), y=y_te.reshape(n_test, fd.T - 1, -1).astype(bool),
                        valid=v_te.reshape(n_test, fd.T - 1, -1).astype(bool), example=fd.idx["test"], horizons=np.array(P.EVENT_HORIZONS))
    torch.save(dict(state=best_state, cond=cond, seed=a.seed, cfg=CFG, d_in=d_in, best_val=best, best_step=best_step, steps=step, stopped_by=stopped_by,
                    n_params=n_params(net), pos_weight=pos_weight.tolist(), test_auprc=ap_.tolist(), test_auroc=au_.tolist()), out / "ckpt" / f"{name}.pt")
    pd.DataFrame(rows).to_csv(out / "train_logs" / f"{name}.csv", index=False)
    log.info("done %s %s seed %d: best val AUPRC %.4f at step %d of %d (%s); test AUPRC %s AUROC %s; %.0fs", ds, cond, a.seed, best, best_step, step, stopped_by,
             [round(float(x), 3) for x in ap_], [round(float(x), 3) for x in au_], time.time() - t0)


if __name__ == "__main__":
    main()
