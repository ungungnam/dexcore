#!/usr/bin/env python
"""Generate the test (or validation) trajectories of one trained Stage-2 model under the fixed-s_0 protocol (s_0 = the GT frame 0,
G, tau of every example; one deterministic trajectory each).
    CUDA_VISIBLE_DEVICES=5 python s2_generate.py --dataset taco --name B1_seed0 [--part val]
Writes <ds>/preds/<name>_A[_val].npz: pred (B, 1, 64, 512) float16 raw contact (frame 0 = s_0, never generated), example indices,
and for B1 / B2 the predicted latents z_hat (B, 63, dz; standardised teacher coordinates) [+ r_hat (B, 63, dr)], and for B2 the z-only
path C_bar (B, 1, 64, 512) and the correction Delta (B, 63, 512) in raw units; raw-range statistics for the validity check.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import torch

import s2_common as S
import s2_models as MD
from s2_data import S2Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2_generate")


def load_model(ds, name, data, dev, which="best"):
    ck = torch.load(S.ckpt_path(ds, name, which), map_location="cpu", weights_only=False)
    net = MD.build(ck["model"], data).to(dev)
    net.load_state_dict(ck["state"] if which == "best" else ck["ema_state"]); net.eval()
    return net, ck


def load_data(ds, dev, name, need_masks=False):
    ck = torch.load(S.ckpt_path(ds, name), map_location="cpu", weights_only=False)
    return S2Data(ds, dev, stats=ck["stats"], need_masks=need_masks, teacher=True)


@torch.no_grad()
def predict(net, data, n_all, bs=16):
    """-> dict of numpy arrays: C_hat (B, 64, 512) raw, [z_std (B, 63, dz), r (B, 63, dr), C_bar (B, 64, 512), delta (B, 63, 512)]."""
    out = {}
    add = lambda k, v: out.setdefault(k, []).append(v.float().cpu().numpy())
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; B = len(n)
        s0 = data.C[n, 0]; s0_in, tau, Sg = data.s0_input(s0), data.tau(n), data.fd.S[n]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            o = net.latents(s0_in, tau, Sg)
            if "rho" in o:
                C_hat = data.to_raw(o["rho"].float(), s0)
            else:
                z_std = o["z_std"].float(); r = o["r"].float() if "r" in o else None
                t = torch.arange(1, S.T, device=data.device)[None].expand(B, -1)
                n_rep = n.repeat_interleave(S.TF); geo = data.geo_tokens(n_rep, t.reshape(-1))
                C_bar, delta = net.decode_maps(z_std.reshape(B * S.TF, -1), None if r is None else r.reshape(B * S.TF, -1), geo, Sg.repeat_interleave(S.TF, 0))
                C_bar = data.std_to_raw(C_bar.float()).view(B, S.TF, 512)
                add("z_std", z_std)
                if delta is None:
                    C_hat = C_bar
                else:
                    delta = (delta.float() * data.s).view(B, S.TF, 512)
                    C_hat = C_bar + delta
                    add("r", r); add("C_bar", torch.cat([s0[:, None], C_bar], 1)); add("delta", delta)
        add("C_hat", torch.cat([s0[:, None], C_hat], 1))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--name", required=True); ap.add_argument("--part", default="test", choices=["test", "val"])
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    data = load_data(ds, dev, a.name); net, ck = load_model(ds, a.name, data, dev)
    te = data.idx[a.part]; n_all = torch.from_numpy(te).to(dev)
    P = predict(net, data, n_all)
    C_hat = P["C_hat"]; gt = data.C[n_all].cpu().numpy()
    err = np.linalg.norm(C_hat[:, 1:] - gt[:, 1:], axis=-1).mean(-1)
    extra = {}
    if "z_std" in P:
        zs = data.teacher_std(n_all).cpu().numpy(); extra["z_err_rms"] = float(np.sqrt(((P["z_std"] - zs) ** 2).mean()))
        extra["z_hat"] = P["z_std"].astype(np.float32)
        if "r" in P:
            extra["r_hat"] = P["r"].astype(np.float32); extra["C_bar"] = P["C_bar"][:, None].astype(np.float16); extra["delta"] = P["delta"].astype(np.float16)
            extra["E_C_zonly"] = float(np.linalg.norm(P["C_bar"][:, 1:] - gt[:, 1:], axis=-1).mean())
    out = S.preds_path(ds, a.name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, pred=C_hat[:, None].astype(np.float16), example=te, s0=gt[:, 0].astype(np.float32), protocol=S.PROTOCOL, part=a.part, model=ck["model"], seed=ck["seed"],
                        best_step=ck["best_step"], raw_min=float(C_hat.min()), raw_max=float(C_hat.max()), frac_below_0=float((C_hat < 0).mean()), frac_above_1=float((C_hat > 1).mean()),
                        n_nonfinite=int((~np.isfinite(C_hat)).sum()), frame0_equals_s0=bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0), **extra)
    log.info("%s %s %s: %d examples, E_C(1..63) %.3f%s, raw range [%.3f, %.3f], frame0 == s0 %s (%.0f s) -> %s", ds, a.name, a.part, len(te), err.mean(),
             "".join(f", {k} {v:.3f}" for k, v in extra.items() if isinstance(v, float)), C_hat.min(), C_hat.max(), bool(np.abs(C_hat[:, 0] - gt[:, 0]).max() == 0), time.time() - t0, out)


if __name__ == "__main__":
    main()
