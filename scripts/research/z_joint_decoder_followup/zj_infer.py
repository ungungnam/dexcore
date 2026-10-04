"""Inference helpers shared by generation, the diagnostics and the sanity checks (memory-light: the GPUs are shared).
Inference path of a z-mediated model: (s_0, G, tau) -> backbone -> z_hat_1:63 -> decoder(geo_t, [G | z_hat_t]) -> C_hat_t.  The teacher z*
is used here only by the diagnostics that say so (oracle / perturbed inputs); it never enters `generate`.
"""
from __future__ import annotations

import numpy as np
import torch

import zj_common as Z
import zj_models as MD
from s2_data import S2Data


def load_data(ds, dev, model, need_masks=False, seed=Z.SEED, tag=None):
    path = Z.model_ckpt(ds, model, seed) if tag is None else Z.ckpt_path(ds, Z.run_name(model, seed, tag))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return S2Data(ds, dev, stats=ck["stats"], need_masks=need_masks, teacher=True)


@torch.no_grad()
def predict_z(net, data, n_all, bs=64):
    """Predicted latent trajectories z_hat (B, 63, dz), standardised teacher coordinates."""
    out = []
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; s0 = data.C[n, 0]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            out.append(net.latents(data.s0_input(s0), data.tau(n), data.fd.S[n])["z_std"].float())
    return torch.cat(out)


@torch.no_grad()
def decode_seq(net, data, n_all, z_std, bs=4, chunk=252):
    """Decode a latent trajectory z_std (B, 63, dz; any source) of the sequences n_all with the decoder of `net` -> raw contact (B, 63, 512)."""
    out = []; t = torch.arange(1, Z.T, device=data.device)
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; B = len(n)
        geo = data.geo_tokens(n.repeat_interleave(Z.TF), t.repeat(B))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            C = MD.decode(net, z_std[i:i + bs].reshape(B * Z.TF, -1), geo, data.fd.S[n].repeat_interleave(Z.TF, 0), chunk)
        out.append(data.std_to_raw(C.float()).view(B, Z.TF, Z.N_PTS))
    return torch.cat(out)


@torch.no_grad()
def generate(net, data, n_all):
    """One deterministic trajectory per example from (s_0, G, tau) -> dict: C_hat (B, 64, 512) raw with frame 0 = s_0 [, z_hat (B, 63, dz)]."""
    s0 = data.C[n_all, 0]
    if not net.use_z:
        outs = []
        for i in range(0, len(n_all), 64):
            n = n_all[i:i + 64]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
                rho = net.latents(data.s0_input(s0[i:i + 64]), data.tau(n), data.fd.S[n])["rho"].float()
            outs.append(data.to_raw(rho, s0[i:i + 64]))
        return dict(C_hat=torch.cat([s0[:, None], torch.cat(outs)], 1))
    z = predict_z(net, data, n_all)
    return dict(C_hat=torch.cat([s0[:, None], decode_seq(net, data, n_all, z)], 1), z_hat=z)


def dense_err(C_hat_future, data, n_all):
    """(B, 63, 512) raw predictions of frames 1..63 -> per-(sequence, frame) dense error ||C_hat_t - C_t|| (B, 63), numpy float64."""
    return (C_hat_future - data.C[n_all, 1:]).norm(dim=-1).double().cpu().numpy()
