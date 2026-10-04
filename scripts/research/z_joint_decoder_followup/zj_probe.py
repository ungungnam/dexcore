#!/usr/bin/env python
"""Held-out adaptation probe (post-hoc diagnostic, NOT one of the compared generators; run only if the trigger of zj_common.DECISION fires).
    CUDA_VISIBLE_DEVICES=5 python zj_probe.py --dataset taco [--model M2]

Why.  Joint training exposes the decoder to the predicted z_hat of the TRAINING sequences.  If the temporal network fits those almost
exactly (train RMSE of z_hat - z* far below the held-out value), the decoder never meets the kind of latent error it gets at test time,
and "trained on predicted z" does not yet mean "adapted to the held-out predicted-z distribution".  This probe gives the decoder exactly
that distribution and asks how much dense error it can recover: an optimistic bound on what decoder adaptation alone can buy.

What.  The temporal network F_theta of the selected model is frozen.  Its z_hat on the VALIDATION sequences (held-out errors) is the
decoder's training input; a copy of the model's decoder is fine-tuned on (z_hat_val_t, G_t) -> C_val_t with the same dense loss L_C.
  1. two-fold cross-fitting over the validation takes: fit on one fold, score the dense error on the other, every EVAL steps; the number
     of fine-tuning steps s* is the minimum of the out-of-fold curve (s* = 0 means adaptation does not help even on validation);
  2. the decoder is fine-tuned on all validation sequences for s* steps and scored on the TEST sequences with the frozen network's test z_hat.
A second, overfitting-free variant (--shrink): the decoder is left untouched and only FIVE numbers are fitted on validation, one
shrinkage factor a_b per horizon bin, C_hat_t = D(a_b * z_hat_t) (standardised coordinates, so a < 1 pulls the latent towards the
training mean = the simplest way a decoder can hedge against an uncertain latent); a_b is the grid value in SHRINK_GRID with the lowest
validation dense error in its bin; the chosen factors are then applied to the test latents.
No test quantity is used for any choice.  Caveat stated in the report: this decoder has seen the validation contact maps, which no
compared model has; the probe can therefore only bound the benefit from above.
Writes <ds>/probe/probe_<model>.npz (per-sequence test errors: adapted, unadapted, teacher-z through the adapted decoder; the
out-of-fold curves) and probe_<model>.json.
"""
from __future__ import annotations

import argparse
import copy
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F

import zj_common as Z
import zj_infer as I
import zj_models as MD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_probe")
CFG = dict(lr=3e-5, wd=0.01, batch_maps=256, max_steps=1500, eval_every=100, grad_clip=1.0, seed=0, dec_chunk=128)
SHRINK_GRID = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def shrink_probe(ds, model):
    """Five-parameter hedge: one shrinkage factor of z_hat per horizon bin, fitted on validation, applied to test."""
    t0 = time.time(); dev = torch.device("cuda")
    data = I.load_data(ds, dev, model); net, ck = MD.load(ds, model, data, dev)
    n_va = torch.from_numpy(data.idx["val"]).to(dev); n_te = torch.from_numpy(data.idx["test"]).to(dev)
    z_va, z_te = I.predict_z(net, data, n_va), I.predict_z(net, data, n_te)
    err = lambda n, z: I.dense_err(I.decode_seq(net, data, n, z), data, n)                    # (B, 63)
    val = {a: err(n_va, a * z_va) for a in SHRINK_GRID}
    a_bin, a_frame = {}, torch.ones(Z.TF, device=dev)
    for lo, hi in Z.HORIZON_BINS:
        best = min(SHRINK_GRID, key=lambda a: val[a][:, lo - 1:hi].mean()); a_bin[f"{lo}_{hi}"] = best; a_frame[lo - 1:hi] = best
    e_un = err(n_te, z_te); e_sh = err(n_te, z_te * a_frame[None, :, None])
    val_sh = np.concatenate([val[a_bin[f"{lo}_{hi}"]][:, lo - 1:hi] for lo, hi in Z.HORIZON_BINS], 1)
    d = Z.ds_out(ds) / "probe"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"shrink_{model}.npz", example=data.idx["test"], take=data.take[data.idx["test"]], E_C_unadapted=e_un.mean(1), E_C_shrunk=e_sh.mean(1), err_unadapted=e_un.astype(np.float32),
                        err_shrunk=e_sh.astype(np.float32), grid=np.array(SHRINK_GRID), val_curve=np.array([val[a].mean() for a in SHRINK_GRID]))
    Z.write_json(d / f"shrink_{model}.json", dict(dataset=ds, model=model, grid=SHRINK_GRID, factors=a_bin, val_E_C_by_factor={str(a): float(val[a].mean()) for a in SHRINK_GRID},
                                                 val_E_C_unadapted=float(val[1.0].mean()), val_E_C_shrunk=float(val_sh.mean()), test_E_C_unadapted=float(e_un.mean()), test_E_C_shrunk=float(e_sh.mean()),
                                                 test_gain_rel=float(1 - e_sh.mean() / e_un.mean()), seconds=time.time() - t0))
    log.info("done %s %s shrink probe: factors %s; validation E_C %.4f -> %.4f; test E_C %.4f -> %.4f (%.1f %%) (%.0f s)", ds, model, a_bin, val[1.0].mean(), val_sh.mean(), e_un.mean(), e_sh.mean(),
             100 * (1 - e_sh.mean() / e_un.mean()), time.time() - t0)


def fine_tune(net, data, n_fit, z_fit, steps, eval_sets, gen):
    """Fine-tune the decoder of `net` on (z_fit[i, t], frame t + 1 of sequence n_fit[i]); every eval_every steps score eval_sets
    {name: (n, z)} -> {name: [(step, per-sequence E_C array)]}.  The temporal network is not touched (z_fit is precomputed)."""
    dev = data.device; params = list(net.D_z.parameters()); opt = torch.optim.AdamW(params, lr=CFG["lr"], weight_decay=CFG["wd"])
    hist = {k: [] for k in eval_sets}
    def score(step):
        net.eval()
        for k, (n, z) in eval_sets.items():
            hist[k].append((step, I.dense_err(I.decode_seq(net, data, n, z), data, n).mean(1)))
        net.train()                                                            # training mode: decoder dropout as in joint training, activation checkpointing of the decoder blocks
    score(0)
    for step in range(1, steps + 1):
        i = torch.randint(0, len(n_fit), (CFG["batch_maps"],), device=dev, generator=gen); t = torch.randint(1, Z.T, (CFG["batch_maps"],), device=dev, generator=gen)
        n = n_fit[i]; s0 = data.C[n, 0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Cn = MD.decode(net, z_fit[i, t - 1], data.geo_tokens(n, t), data.fd.S[n], CFG["dec_chunk"])
        rho = (data.std_to_raw(Cn.float()) - s0) / data.sigma_r; rho_gt = (data.C[n, t] - s0) / data.sigma_r
        loss = F.mse_loss(rho, rho_gt)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, CFG["grad_clip"]); opt.step()
        if step % CFG["eval_every"] == 0:
            score(step)
    net.eval()
    return hist


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", default="M2", choices=["M1", "M2", "M3"]); ap.add_argument("--shrink", action="store_true")
    a = ap.parse_args(); ds, model = a.dataset, a.model; t0 = time.time(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    if a.shrink:
        with torch.no_grad():
            shrink_probe(ds, model)
        return
    data = I.load_data(ds, dev, model); net0, ck = MD.load(ds, model, data, dev)
    n_va = torch.from_numpy(data.idx["val"]).to(dev); n_te = torch.from_numpy(data.idx["test"]).to(dev)
    z_va, z_te = I.predict_z(net0, data, n_va), I.predict_z(net0, data, n_te)          # frozen temporal network, held-out latents
    zs_te = data.teacher_std(n_te)
    take_va = data.take[data.idx["val"]]; rng = np.random.default_rng(CFG["seed"])
    takes = rng.permutation(np.unique(take_va)); fold_of = {t: i % 2 for i, t in enumerate(takes)}; fold = np.array([fold_of[t] for t in take_va])
    # 1 cross-fitting on the validation takes
    oof = {}
    for k in (0, 1):
        fit = torch.from_numpy(np.where(fold != k)[0]).to(dev); ev = torch.from_numpy(np.where(fold == k)[0]).to(dev)
        net = copy.deepcopy(net0); gen = torch.Generator(device=dev); gen.manual_seed(CFG["seed"] + k)
        h = fine_tune(net, data, n_va[fit], z_va[fit], CFG["max_steps"], {"oof": (n_va[ev], z_va[ev])}, gen)["oof"]
        for step, e in h:
            oof.setdefault(step, np.full(len(n_va), np.nan))[ev.cpu().numpy()] = e
        del net
    steps = sorted(oof); curve = np.array([oof[s].mean() for s in steps]); s_star = int(steps[int(np.argmin(curve))])
    log.info("%s %s: out-of-fold validation E_C by fine-tuning steps: %s -> s* = %d", ds, model, {s: round(float(c), 4) for s, c in zip(steps, curve)}, s_star)
    # 2 final fit on all validation sequences, scored on test
    net = copy.deepcopy(net0); gen = torch.Generator(device=dev); gen.manual_seed(CFG["seed"] + 10)
    h = fine_tune(net, data, n_va, z_va, max(s_star, 0), {"test": (n_te, z_te), "test_oracle": (n_te, zs_te)}, gen)
    e_un = h["test"][0][1]; e_ad = h["test"][-1][1]; e_or_un = h["test_oracle"][0][1]; e_or_ad = h["test_oracle"][-1][1]
    d = Z.ds_out(ds) / "probe"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / f"probe_{model}.npz", example=data.idx["test"], take=data.take[data.idx["test"]], E_C_unadapted=e_un, E_C_adapted=e_ad, E_C_oracle_unadapted=e_or_un, E_C_oracle_adapted=e_or_ad,
                        oof_steps=np.array(steps), oof_curve=curve, oof_per_sequence=np.stack([oof[s] for s in steps]), val_fold=fold, test_curve_steps=np.array([s for s, _ in h["test"]]),
                        test_curve=np.array([e.mean() for _, e in h["test"]]))
    Z.write_json(d / f"probe_{model}.json", dict(dataset=ds, model=model, cfg=CFG, selected_step_of_model=int(ck["best_step"]), n_val=int(len(n_va)), n_val_takes=int(len(takes)), s_star=s_star,
                                                oof_val_E_C_unadapted=float(curve[0]), oof_val_E_C_best=float(curve.min()), oof_val_gain_rel=float(1 - curve.min() / curve[0]),
                                                test_E_C_unadapted=float(e_un.mean()), test_E_C_adapted=float(e_ad.mean()), test_gain_rel=float(1 - e_ad.mean() / e_un.mean()),
                                                test_E_C_teacher_z_unadapted=float(e_or_un.mean()), test_E_C_teacher_z_adapted=float(e_or_ad.mean()), seconds=time.time() - t0))
    log.info("done %s %s probe: test E_C %.4f -> %.4f (%.1f %%) after %d steps on validation; teacher-z floor %.4f -> %.4f (%.0f s)", ds, model, e_un.mean(), e_ad.mean(), 100 * (1 - e_ad.mean() / e_un.mean()), s_star,
             e_or_un.mean(), e_or_ad.mean(), time.time() - t0)


if __name__ == "__main__":
    main()
