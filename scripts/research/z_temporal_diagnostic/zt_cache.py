#!/usr/bin/env python
"""Frame cache of the diagnostic: every GT frame of the train / val / test sequences encoded with the FROZEN Stage-1 A3 encoder.
    CUDA_VISIBLE_DEVICES=5 python zt_cache.py --dataset taco
Writes <ds>/cache/frames.npz:
  seq (M,) sequence indices of the inherited sequence store, split (M,) in {train, val, test}, take (M,), mesh (M,)
  C (M, 64, 512) raw canonical contact (f16), z (M, 64, 64) raw Stage-1 latent z_t = E_z(C_t, G_t), S (M, 1032) static descriptor G,
  tau (M, 64, 9 D) the per-frame local trajectory window tau_local,t (object states at t-8 .. t+8, stride 2; f16),
  u_r2 (M, 64, 48), u_q (M, 64, 76) the Stage-1 teacher descriptors (exact R2 / wrench, z-scored and scaled with TRAIN statistics; used
  only in the temporal-geometry analysis, never as a probe input), a (M, 64, 6) participation, m (M, 64, 6) amount,
  z_mu, z_sd (64,) TRAIN mean / std of z, the teacher checkpoint path / md5, and the agreement with the Stage-2 teacher cache.
Nothing is trained; the encoder weights are the EMA weights of the recorded Stage-1 checkpoint.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import zt_common as Z
from cf_data import FrameData
from cf_models import build as cf_build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zt_cache")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=Z.DATASETS); a = ap.parse_args(); ds = a.dataset; t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = Z.CF.ckpt_path(ds, Z.TEACHER_NAME); ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    data = FrameData(ds, dev, stats=ck["stats"], need_masks=False)
    net = cf_build(ck["model"], data.d_geo(), data.d_static(), res_scale=ck.get("res_scale", 1.0)).to(dev)
    net.load_state_dict(ck["ema_state"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    seq = np.concatenate([data.idx[k] for k in ("train", "val", "test")]); split = np.concatenate([[k] * len(data.idx[k]) for k in ("train", "val", "test")])
    n_seq = torch.from_numpy(seq).to(dev); M = len(seq)
    n = n_seq.repeat_interleave(Z.T); t = torch.arange(Z.T, device=dev).repeat(M)
    zs = []
    for i in range(0, len(n), 512):
        b = data.batch(n[i:i + 512], t[i:i + 512])
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            zs.append(net.encode_z(b["Cn"], b["geo"], b["S"]).float().cpu().numpy())
    z = np.concatenate(zs).reshape(M, Z.T, -1)
    # agreement with the Stage-2 teacher cache (the same frozen encoder, same frames)
    s2 = np.load(Z.S2.teacher_cache_path(ds), allow_pickle=True)
    assert str(s2["md5"]) == Z.md5(ck_path), "Stage-2 teacher cache was produced from a different Stage-1 checkpoint"
    z_fresh = z
    diff = float(np.abs(s2["z"][seq] - z_fresh).max()); diff_std = float((np.abs(s2["z"][seq] - z_fresh) / s2["z_train_std"]).max())
    z = s2["z"][seq].astype(np.float32)          # the latents Stage 2 was trained against (identical encoder; the fresh pass differs only by bf16 rounding)
    fd = data.fd
    tt = torch.arange(Z.T, device=dev)
    tau = fd.O[n_seq[:, None, None], tt[None, :, None] + fd.offsets[None, None]].reshape(M, Z.T, -1)          # (M, 64, 9 D)
    tr = split == "train"
    zt = z[tr].reshape(-1, z.shape[-1])
    out = Z.cache_path(ds); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seq=seq, split=split, take=data.take[seq], mesh=data.mesh[seq], C=data.C[n_seq].cpu().numpy().astype(np.float16), z=z.astype(np.float32),
                        S=data.S[n_seq].cpu().numpy().astype(np.float32), tau=tau.cpu().numpy().astype(np.float16), u_r2=data.u_r2[n_seq].cpu().numpy().astype(np.float32),
                        u_q=data.u_q[n_seq].cpu().numpy().astype(np.float32), a=data.a[n_seq].cpu().numpy().astype(np.int8), m=data.m[n_seq].cpu().numpy().astype(np.float32),
                        z_mu=zt.mean(0), z_sd=zt.std(0), ckpt=str(ck_path), md5=Z.md5(ck_path), best_step=int(ck["best_step"]), max_abs_diff_vs_stage2_cache=diff, max_abs_diff_vs_stage2_cache_in_std=diff_std,
                        tau_offsets=(fd.offsets - Z.PAD).cpu().numpy(), d_state=int(fd.D))
    Z.write_json(out.with_suffix(".json"), dict(ckpt=str(ck_path), md5=Z.md5(ck_path), best_step=int(ck["best_step"]), n_sequences={k: int((split == k).sum()) for k in ("train", "val", "test")},
                                               n_takes={k: int(len(set(data.take[seq][split == k]))) for k in ("train", "val", "test")}, dz=int(z.shape[-1]), d_static=int(data.S.shape[1]), d_tau=int(tau.shape[-1]),
                                               z_train_std_range=[float(zt.std(0).min()), float(zt.std(0).max())], fresh_encoding_max_abs_diff_vs_stage2_cache=diff, fresh_encoding_max_abs_diff_in_train_std=diff_std,
                                               z_source="Stage-2 teacher cache (frozen A3 encoder); re-encoded here as a check",
                                               manifest_md5=Z.md5(Z.SAT.HP.ROOTS[ds] / "manifests" / "fixed0.json"), seconds=time.time() - t0))
    log.info("done %s: %d sequences (%s), z %s, tau %s, fresh encoding vs the Stage-2 teacher cache: max |diff| %.4f raw = %.4f train std, %.0f s -> %s", ds, M, {k: int((split == k).sum()) for k in ("train", "val", "test")}, z.shape, tuple(tau.shape), diff, diff_std, time.time() - t0, out)


if __name__ == "__main__":
    main()
