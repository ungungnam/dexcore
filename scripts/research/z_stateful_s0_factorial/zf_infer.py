"""Inference helpers shared by generation, diagnostics and sanity checks (memory-light: the GPUs may be shared).
The generated trajectory of every model depends on (s_0, G, tau) only.  The teacher latents are used here only by the diagnostics that say so."""
from __future__ import annotations

import numpy as np
import torch

import zf_common as Z
import zf_models as MD
from s2_data import S2Data


def load_data(ds, dev, model="M00", need_masks=False):
    ck = torch.load(Z.model_ckpt(ds, model), map_location="cpu", weights_only=False)
    return S2Data(ds, dev, stats=ck["stats"], need_masks=need_masks, teacher=True)


@torch.no_grad()
def predict_z(net, data, n_all, bs=64, z0=None):
    """-> z_hat_1:63 (B, 63, dz) and z_hat_0 (B, dz) or None, standardised teacher coordinates.  z0: start the rollout from given initial latents (analysis)."""
    zs, z0s = [], []
    for i in range(0, len(n_all), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            z, zi = MD.predict_latents(net, data, n_all[i:i + bs], None if z0 is None else z0[i:i + bs])
        zs.append(z.float()); z0s.append(None if zi is None else zi.float())
    return torch.cat(zs), (None if z0s[0] is None else torch.cat(z0s))


@torch.no_grad()
def decode_seq(net, data, n_all, z_std, bs=4, chunk=252):
    """Decode latent trajectories z_std (B, 63, dz; any source) of the sequences n_all -> raw contact (B, 63, 512)."""
    out = []; t = torch.arange(1, Z.T, device=data.device)
    for i in range(0, len(n_all), bs):
        n = n_all[i:i + bs]; B = len(n)
        geo = data.geo_tokens(n.repeat_interleave(Z.TF), t.repeat(B))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.device.type == "cuda"):
            C = MD.decode_raw(net, data, z_std[i:i + bs].reshape(B * Z.TF, -1), geo, data.fd.S[n].repeat_interleave(Z.TF, 0), data.C[n, 0].repeat_interleave(Z.TF, 0), chunk)
        out.append(C.float().view(B, Z.TF, Z.N_PTS))
    return torch.cat(out)


@torch.no_grad()
def generate(net, data, n_all):
    """One deterministic trajectory per example from (s_0, G, tau): C_hat (B, 64, 512) raw with frame 0 = s_0, z_hat (B, 63, dz), z0_hat (B, dz) or None."""
    z, z0 = predict_z(net, data, n_all)
    return dict(C_hat=torch.cat([data.C[n_all, :1], decode_seq(net, data, n_all, z)], 1), z_hat=z, z0_hat=z0)


def dense_err(C_future, data, n_all):
    """(B, 63, 512) raw maps of frames 1..63 -> ||C_hat_t - C_t|| (B, 63), numpy float64."""
    return (C_future - data.C[n_all, 1:]).norm(dim=-1).double().cpu().numpy()
