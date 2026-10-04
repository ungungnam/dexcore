#!/usr/bin/env python
"""Cache the teacher latent z*_t = E_z^Stage1(C_t^GT, G_t) of every sequence frame of the three splits with the FROZEN Stage-1 A3
encoder (EMA weights of its best validation step).
    CUDA_VISIBLE_DEVICES=5 python s2_cache_teacher.py --dataset taco
Writes <ds>/cache/teacher_z.npz:  z (N, 64, 64) raw Stage-1 z coordinates (NaN for sequences outside train / val / test), has (N,) bool,
r_train_mean (64,) [the Stage-1 train-mean realisation code: used ONLY to initialise the bias of the B2 r head, never as a target],
z_train_mean / z_train_std, the teacher checkpoint path / md5 / best step, and the agreement with the Stage-1 latents cache
(the same encoder applied by the Stage-1 encode.py) as a check that the teacher is the recorded Stage-1 model.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import s2_common as S
from cf_data import FrameData
from cf_models import build as cf_build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("cache_teacher")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=S.DATASETS)
    a = ap.parse_args(); ds = a.dataset; t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = S.teacher_ckpt_path(ds)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    assert ck["model"] == S.TEACHER_MODEL and ck["dataset"] == ds, (ck["model"], ck["dataset"])
    data = FrameData(ds, dev, stats=ck["stats"], need_masks=False)
    net = cf_build(ck["model"], data.d_geo(), data.d_static(), res_scale=ck.get("res_scale", 1.0)).to(dev)
    net.load_state_dict(ck["ema_state"]); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    N = data.N
    Z = np.full((N, S.T, net.dz), np.nan, np.float32); R = np.full((N, S.T, net.dr), np.nan, np.float32); has = np.zeros(N, bool)
    agree = {}
    for split in ("train", "val", "test"):
        n_seq = torch.from_numpy(data.idx[split]).to(dev)
        n = n_seq.repeat_interleave(S.T); t = torch.arange(S.T, device=dev).repeat(len(n_seq))
        zs, rs = [], []
        for i in range(0, len(n), 512):
            b = data.batch(n[i:i + 512], t[i:i + 512])
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                o = net(b["Cn"], b["geo"], b["S"], use_r=True)
            zs.append(o["z"].float().cpu().numpy()); rs.append(o["r"].float().cpu().numpy())
        z = np.concatenate(zs).reshape(len(n_seq), S.T, -1); r = np.concatenate(rs).reshape(len(n_seq), S.T, -1)
        Z[data.idx[split]] = z; R[data.idx[split]] = r; has[data.idx[split]] = True
        # agreement with the Stage-1 latents cache (same encoder, same frames; bf16 autocast in both)
        p1 = S.CF.latents_path(ds, S.TEACHER_NAME, split)
        if p1.exists():
            z1 = np.load(p1)
            assert np.array_equal(z1["n"], data.idx[split])
            d = np.abs(z1["z"] - z); agree[split] = dict(max_abs_diff=float(d.max()), mean_abs_diff=float(d.mean()), z_std=float(z.std()))
            log.info("%s %s: %d sequences encoded; vs Stage-1 cache max |diff| %.4f (z std %.2f)", ds, split, len(n_seq), d.max(), z.std())
    tr = data.idx["train"]
    zt = Z[tr].reshape(-1, Z.shape[-1]); rt = R[tr].reshape(-1, R.shape[-1])
    out = S.teacher_cache_path(ds); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, z=Z, has=has, r_train_mean=rt.mean(0), z_train_mean=zt.mean(0), z_train_std=zt.std(0), ckpt=str(ck_path), md5=S.md5(ck_path),
                        best_step=int(ck["best_step"]), teacher_model=ck["model"], teacher_seed=int(ck["seed"]), n_train_frames=len(zt),
                        agreement=np.array(agree, dtype=object), encoded_with="E_z of the Stage-1 A3 EMA weights, bf16 autocast, GT frames only")
    S.write_json(out.with_suffix(".json"), dict(ckpt=str(ck_path), md5=S.md5(ck_path), best_step=int(ck["best_step"]), n_params_teacher=int(ck["n_params"]),
                                               dz=int(Z.shape[-1]), dr=int(R.shape[-1]), n_sequences=int(has.sum()), n_train_frames=int(len(zt)),
                                               z_train_std_range=[float(zt.std(0).min()), float(zt.std(0).max())], r_train_mean_abs=float(np.abs(rt.mean(0)).mean()),
                                               agreement_with_stage1_cache=agree, seconds=time.time() - t0))
    log.info("done %s: %d sequences, z std per dim [%.2f, %.2f], %.0f s -> %s", ds, has.sum(), zt.std(0).min(), zt.std(0).max(), time.time() - t0, out)


if __name__ == "__main__":
    main()
