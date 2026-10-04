#!/usr/bin/env python
"""Encode every sequence frame of a split with a trained model (EMA weights of the best validation step).
    CUDA_VISIBLE_DEVICES=4 python encode.py --dataset taco --model A3
Writes <ds>/latents/<name>_<split>.npz: n (sequence indices), z (N, 64, dz), r (N, 64, dr) [factorised], per-frame errors
e_zonly / e_full (raw L2), mse_zonly / mse_full (standardised), and for val / test the decoded maps C_bar / C_hat (raw, f16).
"""
from __future__ import annotations

import argparse
import os
import logging

import numpy as np
import torch

import cf_common as S
from cf_data import FrameData
from cf_models import build

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("encode")


def load_model(ds, name, data, dev, which="ema_state"):
    ck = torch.load(S.ckpt_path(ds, name), map_location="cpu", weights_only=False)
    net = build(ck["model"], data.d_geo(), data.d_static(), res_scale=ck.get("res_scale", 1.0)).to(dev)
    net.load_state_dict(ck[which]); net.eval()
    return net, ck


def load_data(ds, dev, name, need_masks=False):
    ck = torch.load(S.ckpt_path(ds, name), map_location="cpu", weights_only=False)
    return FrameData(ds, dev, stats=ck["stats"], need_masks=need_masks)


@torch.no_grad()
def encode_frames(net, data, n, t, bs=512, keep_maps=False):
    out = dict(z=[], r=[], e_zonly=[], e_full=[], mse_zonly=[], mse_full=[], C_bar=[], C_hat=[])
    for i in range(0, len(n), bs):
        b = data.batch(n[i:i + bs], t[i:i + bs])
        with torch.autocast(data.device.type, dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            o = net(b["Cn"], b["geo"], b["S"], use_r=True)
        C_bar, C_hat = o["C_bar"].float(), o["C_hat"].float()
        out["z"].append(o["z"].float().cpu().numpy())
        if o["r"] is not None:
            out["r"].append(o["r"].float().cpu().numpy())
        out["e_zonly"].append((data.to_raw(C_bar) - b["C"]).norm(dim=-1).cpu().numpy()); out["e_full"].append((data.to_raw(C_hat) - b["C"]).norm(dim=-1).cpu().numpy())
        out["mse_zonly"].append(((C_bar - b["Cn"]) ** 2).mean(-1).cpu().numpy()); out["mse_full"].append(((C_hat - b["Cn"]) ** 2).mean(-1).cpu().numpy())
        if keep_maps:
            out["C_bar"].append(data.to_raw(C_bar).clamp(0, 1).half().cpu().numpy()); out["C_hat"].append(data.to_raw(C_hat).clamp(0, 1).half().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--model", required=True); ap.add_argument("--seed", type=int, default=S.SEED)
    a = ap.parse_args(); dev = torch.device(os.environ.get("CF_DEVICE", "cuda")); name = S.run_name(a.model, a.seed)
    data = load_data(a.dataset, dev, name); net, ck = load_model(a.dataset, name, data, dev)
    for split in ("train", "val", "test"):
        n_seq = torch.from_numpy(data.idx[split]).to(dev)
        n = n_seq.repeat_interleave(S.T); t = torch.arange(S.T, device=dev).repeat(len(n_seq))
        enc = encode_frames(net, data, n, t, keep_maps=split != "train")
        enc = {k: v.reshape(len(n_seq), S.T, *v.shape[1:]) for k, v in enc.items()}
        p = S.latents_path(a.dataset, name, split); p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, n=data.idx[split], best_step=ck["best_step"], **enc)
        log.info("%s %s %s: %d sequences, E_zonly %.4f E_full %.4f", a.dataset, name, split, len(n_seq), enc["e_zonly"].mean(), enc["e_full"].mean())


if __name__ == "__main__":
    main()
