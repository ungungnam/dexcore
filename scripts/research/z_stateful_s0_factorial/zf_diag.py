#!/usr/bin/env python
"""Latent generalisation, error accumulation, s_0 preservation and the stateful failure check (analysis only).
    CUDA_VISIBLE_DEVICES=4 python zf_diag.py --dataset taco
For every z model (M00, M01, M10, M11) that has a checkpoint:
  z error        (z_hat_t - z*_t)^2 per sequence and frame on TRAIN, VALIDATION and TEST (the model's own generation path: (s_0, G, tau) only);
                 for the stateful models also the error of z_hat_0
  dense error    of the generated trajectory per sequence and frame (test), of the teacher latents through the same decoder (test), and both on
                 160 training sequences
  drift          ||C_hat_t - s_0|| per frame (test), next to the data's ||C_t - s_0||
  use of z       the dense error when the decoder is given, instead of the model's own z_hat, (a) the z_hat predicted for ANOTHER test sequence of a
                 different take and (b) the mean training latent (zero in standardised coordinates); if these were as good as the model's own z_hat,
                 the decoder would not be using the latent (relevant for the s_0-preserving decoder, which also sees s_0)
Stateful models only (the teacher latent is used here as an INPUT, which never happens in generation):
  h-step         z_hat_t+h obtained by rolling the transition h = 1, 4, 8 steps from the TRUE z*_t  (one-step = teacher-forced prediction)
  true start     the 63-step rollout started from the TRUE z*_0 instead of I(s_0, G), and its decoded dense error
References: holding z*_t for h steps (persistence), holding z*_0, the direct model D0's dense error per frame.
Writes <ds>/diag/diag.npz and <ds>/diag/summary.json.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import zf_common as Z
import zf_infer as I
import zf_models as MD

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zf_diag")
N_TRAIN_DECODE = 160
H_STEPS = (1, 4, 8)


@torch.no_grad()
def roll_from_true(net, data, n_all, h, bs=32):
    """Roll the transition h steps from the TRUE state of every start frame t = 0 .. 63 - h -> squared error of the latent at t + h, (B, 64 - h)."""
    out = []; k = net.trans.k
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; B = len(n); S = Z.T - h
        zs = data.z_standardise(data.z_star[n])                                                # (B, 64, dz) teacher, frames 0..63
        tau = MD.tau_all(data, n, k)                                                           # (B, 64 + k, d_traj)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sc, sh = net.trans.static(data.fd.S[n])
            sc = sc[:, None].expand(-1, S, -1, -1).reshape(B * S, *sc.shape[1:]); sh = sh[:, None].expand(-1, S, -1, -1).reshape(B * S, *sh.shape[1:])
            z = zs[:, :S].reshape(B * S, -1)
            for j in range(h):
                ctx = torch.stack([tau[:, t + j:t + j + k + 1] for t in range(S)], 1).reshape(B * S, k + 1, -1)
                z = z + net.trans.step(z, sc, sh, ctx).float()
        out.append((z.view(B, S, -1) - zs[:, h:]).pow(2).mean(-1))
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS)
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(min(1.0, 10000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    models = [m for m in Z.Z_MODELS if Z.model_ckpt(ds, m).exists()]
    data = I.load_data(ds, dev, "M00")
    idx = {k: torch.from_numpy(data.idx[k]).to(dev) for k in ("train", "val", "test")}
    n_te = idx["test"]; take = data.take[data.idx["test"]]
    rng = np.random.default_rng(0); tr_sub = idx["train"][torch.from_numpy(np.sort(rng.choice(len(idx["train"]), N_TRAIN_DECODE, replace=False))).to(dev)]
    out, summ = {}, dict(dataset=ds, models=models, n_test=int(len(n_te)), train_subset=tr_sub.cpu().numpy().tolist(), h_steps=H_STEPS)
    zs_all = {k: data.z_standardise(data.z_star[v]) for k, v in idx.items()}                   # (N, 64, dz)
    zs_te = zs_all["test"]
    out["teacher|z2|test"] = zs_te[:, 1:].pow(2).mean(-1).cpu().numpy(); out["teacher|var_dim|test"] = zs_te[:, 1:].reshape(-1, zs_te.shape[-1]).var(0).cpu().numpy()
    out["hold_z0|zerr2|test"] = (zs_te[:, :1] - zs_te[:, 1:]).pow(2).mean(-1).cpu().numpy()
    for h in H_STEPS:
        out[f"persist|zerr2_h{h}"] = (zs_te[:, h:] - zs_te[:, :-h]).pow(2).mean(-1).cpu().numpy()
    s0_te = data.C[n_te, :1]
    other = (np.arange(len(n_te)) + len(n_te) // 2) % len(n_te)                                # a fixed partner sequence for the "another sequence's latents" check
    for j in np.where(take[other] == take)[0]:                                                 # make sure the partner comes from a different take
        other[j] = next(int(q) for q in np.roll(np.arange(len(n_te)), -int(other[j])) if take[q] != take[j])
    summ["other_sequence_same_take"] = int((take[other] == take).sum()); out["other_index"] = other; other_t = torch.from_numpy(other).to(dev)
    out["gt|drift"] = (data.C[n_te, 1:] - s0_te).norm(dim=-1).cpu().numpy()
    d0 = np.load(Z.model_preds(ds, Z.DIRECT))["pred"][:, 0, 1:].astype(np.float32)
    out["D0|pred"] = np.linalg.norm(d0 - data.C[n_te, 1:].cpu().numpy(), axis=-1); out["D0|drift"] = np.linalg.norm(d0 - s0_te.cpu().numpy(), axis=-1)
    for m in models:
        net = MD.load(ds, m, data, dev)[0]; st = MD.is_stateful(net)
        for part, n in idx.items():
            z, z0 = I.predict_z(net, data, n)
            out[f"{m}|zerr2|{part}"] = (z - zs_all[part][:, 1:]).pow(2).mean(-1).cpu().numpy()
            if z0 is not None:
                out[f"{m}|z0err2|{part}"] = (z0 - zs_all[part][:, 0]).pow(2).mean(-1).cpu().numpy()
            if part == "test":
                z_te = z; out[f"{m}|zerr2_dim|test"] = (z - zs_te[:, 1:]).pow(2).mean((0, 1)).cpu().numpy()
        C = I.decode_seq(net, data, n_te, z_te)
        out[f"{m}|pred"] = I.dense_err(C, data, n_te); out[f"{m}|drift"] = (C - s0_te).norm(dim=-1).cpu().numpy()
        out[f"{m}|oracle"] = I.dense_err(I.decode_seq(net, data, n_te, zs_te[:, 1:]), data, n_te)
        out[f"{m}|pred_z_other"] = I.dense_err(I.decode_seq(net, data, n_te, z_te[other_t]), data, n_te)       # another sequence's predicted latents, this sequence's s_0 / G
        out[f"{m}|pred_z_mean"] = I.dense_err(I.decode_seq(net, data, n_te, torch.zeros_like(z_te)), data, n_te)  # the mean training latent at every frame
        out[f"{m}|pred|train"] = I.dense_err(I.decode_seq(net, data, tr_sub, I.predict_z(net, data, tr_sub)[0]), data, tr_sub)
        out[f"{m}|oracle|train"] = I.dense_err(I.decode_seq(net, data, tr_sub, data.teacher_std(tr_sub)), data, tr_sub)
        stored = np.load(Z.model_preds(ds, m)); gt = data.C[n_te, 1:].cpu().numpy()
        summ[f"{m}_pred_path_vs_stored_E_C"] = [float(out[f"{m}|pred"].mean()), float(np.linalg.norm(stored["pred"][:, 0, 1:].astype(np.float32) - gt, axis=-1).mean())]
        summ[f"{m}_zhat_vs_preds_file_max_abs"] = float(np.abs(stored["z_hat"] - z_te.cpu().numpy()).max())
        msg = ""
        if st:                                                                                 # failure check: the transition from TRUE states, the rollout from the TRUE initial latent
            for h in H_STEPS:
                out[f"{m}|zerr2_h{h}"] = roll_from_true(net, data, n_te, h)
            z_true0, _ = I.predict_z(net, data, n_te, z0=zs_te[:, 0])
            out[f"{m}|zerr2_true_z0"] = (z_true0 - zs_te[:, 1:]).pow(2).mean(-1).cpu().numpy()
            out[f"{m}|pred_true_z0"] = I.dense_err(I.decode_seq(net, data, n_te, z_true0), data, n_te)
            msg = (f"; z0 rmse test {np.sqrt(out[f'{m}|z0err2|test'].mean()):.3f} (train {np.sqrt(out[f'{m}|z0err2|train'].mean()):.3f}); from true states: "
                   + ", ".join(f"h={h} {np.sqrt(out[f'{m}|zerr2_h{h}'].mean()):.3f} (hold {np.sqrt(out[f'persist|zerr2_h{h}'].mean()):.3f})" for h in H_STEPS)
                   + f"; rollout from true z0: z rmse {np.sqrt(out[f'{m}|zerr2_true_z0'].mean()):.3f}, E_C {out[f'{m}|pred_true_z0'].mean():.3f}")
        log.info("%s %s: z rmse train %.3f / val %.3f / test %.3f; E_C pred %.3f (frames 1-16 %.3f) | teacher z %.3f | another sequence's z %.3f | mean z %.3f | train pred %.3f%s", ds, m, np.sqrt(out[f"{m}|zerr2|train"].mean()),
                 np.sqrt(out[f"{m}|zerr2|val"].mean()), np.sqrt(out[f"{m}|zerr2|test"].mean()), out[f"{m}|pred"].mean(), out[f"{m}|pred"][:, :16].mean(), out[f"{m}|oracle"].mean(), out[f"{m}|pred_z_other"].mean(),
                 out[f"{m}|pred_z_mean"].mean(), out[f"{m}|pred|train"].mean(), msg)
        del net
    d = Z.ds_out(ds) / "diag"; d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "diag.npz", take=take, example=data.idx["test"], **{k: np.asarray(v, np.float32) for k, v in out.items()})
    summ["seconds"] = time.time() - t0; Z.write_json(d / "summary.json", summ)
    log.info("done %s diag (%.0f s)", ds, time.time() - t0)


if __name__ == "__main__":
    main()
