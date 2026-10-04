#!/usr/bin/env python
"""z diagnostics and the decoder-mismatch diagnostic (analysis only: the teacher z* is used here to probe the trained decoders; it is
never an input of the generated trajectories).      CUDA_VISIBLE_DEVICES=5 python zj_diag.py --dataset taco
For every z-mediated model m (M1 = Stage-2 B1, M2, M3 when trained) on the TEST sequences, frames 1..63:
  z error            z_hat_m - z*  (standardised teacher coordinates); also on TRAIN and VALIDATION (EMA weights, eval mode): the error
                     distribution the decoder actually met in training versus the held-out one
  decoder inputs     A  oracle        D_m(z*)                     decoder floor on the clean teacher latent
                     B  predicted     D_m(z_hat_m)                the generated trajectory
                     C  perturbed     D_m(z* + eps)               ONE level, matched to the model's own test error:
                          gauss  eps_t ~ N(mu_bin, Sigma_bin): mean and covariance of z_hat_m - z* over the test frames of the horizon bin of t
                          perm   eps = the error trajectory of ANOTHER test sequence of a different take (exact error distribution, independent of the content)
                     S  swapped       D_m(z_hat_m'), D_m(z* + eps_m')   the same predicted z, and the same perturbed teacher latents, through the other models' decoders (decoder robustness, same input)
  the same A / B on a fixed random subset of TRAIN sequences (what the decoder saw in training).
Writes <ds>/diag/diag.npz (per-sequence, per-frame dense errors and squared z errors) and <ds>/diag/summary.json.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import zj_common as Z
import zj_infer as I
import zj_models as MD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_diag")
N_TRAIN_DECODE = 160


def gaussian_matched(err, gen):
    """err (B, 63, dz) -> eps (B, 63, dz) with eps_t ~ N(mu_bin, Sigma_bin) of the horizon bin of t (independent across sequences and frames)."""
    eps = torch.zeros_like(err); info = {}
    for lo, hi in Z.HORIZON_BINS:
        e = err[:, lo - 1:hi].reshape(-1, err.shape[-1]).double()
        mu = e.mean(0); cov = torch.cov(e.T) + 1e-6 * torch.eye(e.shape[1], device=e.device, dtype=e.dtype)
        Lc = torch.linalg.cholesky(cov)
        x = torch.randn(err.shape[0], hi - lo + 1, err.shape[-1], device=err.device, generator=gen, dtype=torch.float64)
        eps[:, lo - 1:hi] = (mu + x @ Lc.T).float()
        info[f"{lo}_{hi}"] = dict(rms_error=float(e.pow(2).mean().sqrt()), rms_eps=float(eps[:, lo - 1:hi].pow(2).mean().sqrt()), bias_share=float(mu.pow(2).sum() / e.pow(2).sum(1).mean()))
    return eps, info


def other_take_permutation(take, rng):
    """perm[i] = a test sequence of a different take than i (random, fixed seed)."""
    n = len(take); perm = rng.permutation(n)
    for _ in range(200):
        bad = np.where(take[perm] == take)[0]
        if len(bad) == 0:
            break
        for i in bad:
            j = rng.integers(n)
            if take[perm[j]] != take[i] and take[perm[i]] != take[j]:
                perm[i], perm[j] = perm[j], perm[i]
    assert (take[perm] != take).all(), "could not build an other-take permutation"
    return perm


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS)
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    models = [m for m in Z.Z_MODELS if Z.model_ckpt(ds, m).exists()]
    data = I.load_data(ds, dev, "M1")                                            # the checkpoints share the normalisation statistics (sanity check 6 / 7)
    nets = {m: MD.load(ds, m, data, dev)[0] for m in models}
    idx = {k: torch.from_numpy(data.idx[k]).to(dev) for k in ("train", "val", "test")}
    n_te = idx["test"]; take = data.take[data.idx["test"]]
    zs = data.teacher_std(n_te)                                                  # (B, 63, dz) teacher, standardised with the TRAIN statistics
    out, summ = {}, dict(dataset=ds, models=models, n_test=int(len(n_te)), noise=Z.NOISE, bins=Z.HORIZON_BINS)
    zh, err = {}, {}
    for m in models:
        zh[m] = I.predict_z(nets[m], data, n_te); err[m] = zh[m] - zs
        out[f"{m}|zerr2|test"] = err[m].pow(2).mean(-1).cpu().numpy()                                       # (B, 63) mean over the 64 dims
        out[f"{m}|zerr2_dim|test"] = err[m].pow(2).mean((0, 1)).cpu().numpy()                               # (dz,)
        for part in ("train", "val"):
            e = I.predict_z(nets[m], data, idx[part]) - data.teacher_std(idx[part])
            out[f"{m}|zerr2|{part}"] = e.pow(2).mean(-1).cpu().numpy()
        # stored prediction file vs the recomputed latent (the generated trajectories used exactly this z_hat)
        pz = np.load(Z.model_preds(ds, m))["z_hat"]; summ[f"{m}_zhat_vs_preds_file_max_abs"] = float(np.abs(pz - zh[m].cpu().numpy()).max())
    out["teacher|z2|test"] = zs.pow(2).mean(-1).cpu().numpy(); out["teacher|var_dim|test"] = zs.reshape(-1, zs.shape[-1]).var(0).cpu().numpy()
    out["hold_z0|zerr2|test"] = (data.z_standardise(data.z_star[n_te, :1]) - zs).pow(2).mean(-1).cpu().numpy()
    rng = np.random.default_rng(Z.NOISE["seed"]); perm = other_take_permutation(take, rng); perm_t = torch.from_numpy(perm).to(dev)
    tr_sub = idx["train"][torch.from_numpy(np.sort(rng.choice(len(idx["train"]), N_TRAIN_DECODE, replace=False))).to(dev)]
    summ.update(train_subset=tr_sub.cpu().numpy().tolist(), perm=perm.tolist())
    eps = {}
    for m in models:                                                             # one perturbation per model, matched to that model's own test error
        gen = torch.Generator(device=dev); gen.manual_seed(Z.NOISE["seed"])
        eps[m], summ[f"{m}_gauss_level"] = gaussian_matched(err[m], gen)
    for m in models:
        net = nets[m]
        dec = lambda z, n=n_te: I.dense_err(I.decode_seq(net, data, n, z), data, n)
        out[f"{m}|oracle"] = dec(zs)
        out[f"{m}|pred"] = dec(zh[m])
        out[f"{m}|noisy_gauss"] = dec(zs + eps[m])
        out[f"{m}|noisy_perm"] = dec(zs + err[m][perm_t])
        for m2 in models:
            if m2 != m:                                                          # the SAME latents the other decoder was given: its predicted z and its two perturbed teacher latents
                out[f"{m}|zhat_{m2}"] = dec(zh[m2])
                out[f"{m}|noisy_gauss_{m2}"] = dec(zs + eps[m2]); out[f"{m}|noisy_perm_{m2}"] = dec(zs + err[m2][perm_t])
        # the same decoder on TRAIN sequences: predicted vs teacher latent (the input distribution of joint training)
        out[f"{m}|pred|train"] = dec(I.predict_z(net, data, tr_sub), tr_sub); out[f"{m}|oracle|train"] = dec(data.teacher_std(tr_sub), tr_sub)
        # consistency with the stored trajectory
        stored = np.load(Z.model_preds(ds, m))["pred"][:, 0, 1:].astype(np.float32); gt = data.C[n_te, 1:].cpu().numpy()
        summ[f"{m}_pred_path_vs_stored_E_C"] = [float(out[f"{m}|pred"].mean()), float(np.linalg.norm(stored - gt, axis=-1).mean())]
        log.info("%s %s: z rmse test %.3f (train %.3f, val %.3f); E_C oracle %.3f | pred %.3f | gauss %.3f | perm %.3f | train pred %.3f oracle %.3f; %s", ds, m,
                 np.sqrt(out[f"{m}|zerr2|test"].mean()), np.sqrt(out[f"{m}|zerr2|train"].mean()), np.sqrt(out[f"{m}|zerr2|val"].mean()), out[f"{m}|oracle"].mean(), out[f"{m}|pred"].mean(),
                 out[f"{m}|noisy_gauss"].mean(), out[f"{m}|noisy_perm"].mean(), out[f"{m}|pred|train"].mean(), out[f"{m}|oracle|train"].mean(),
                 {k: round(float(v.mean()), 3) for k, v in out.items() if k.startswith((f"{m}|zhat_", f"{m}|noisy_gauss_", f"{m}|noisy_perm_"))})
    d = Z.ds_out(ds) / "diag"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "diag.npz", take=take, example=data.idx["test"], **{k: np.asarray(v, np.float32) for k, v in out.items()})
    summ["seconds"] = time.time() - t0; Z.write_json(d / "summary.json", summ)
    log.info("done %s diag (%.0f s)", ds, time.time() - t0)


if __name__ == "__main__":
    main()
