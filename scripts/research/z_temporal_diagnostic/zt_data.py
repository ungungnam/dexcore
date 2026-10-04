"""Cache access of the diagnostic: standardised latents, static descriptor, trajectory windows and the (sequence, t, t + h) pair tables.
All probe inputs come from <ds>/cache/frames.npz (written by zt_cache.py); z is standardised with the TRAIN mean / std stored there."""
from __future__ import annotations

import numpy as np
import torch

import zt_common as Z


class Cache:
    def __init__(self, ds, device=None, with_contact=False):
        z = np.load(Z.cache_path(ds), allow_pickle=True)
        self.ds, self.device = ds, device
        self.seq, self.split, self.take, self.mesh = z["seq"], z["split"].astype(str), z["take"].astype(str), z["mesh"].astype(str)
        self.z_mu, self.z_sd = z["z_mu"].astype(np.float32), z["z_sd"].astype(np.float32)
        self.z_raw = z["z"].astype(np.float32)
        self.zs = (self.z_raw - self.z_mu) / self.z_sd                                # (M, 64, dz) standardised with TRAIN statistics
        self.S = z["S"].astype(np.float32); self.tau = z["tau"].astype(np.float32)
        self.u_r2, self.u_q, self.a, self.m = z["u_r2"], z["u_q"], z["a"], z["m"]
        self.C = z["C"].astype(np.float32) if with_contact else None
        self.md5, self.ckpt = str(z["md5"]), str(z["ckpt"])
        self.dz, self.d_static, self.d_tau = self.zs.shape[-1], self.S.shape[1], self.tau.shape[-1]
        if device is not None:
            f = lambda a: torch.from_numpy(a).to(device)
            self.t_zs, self.t_S, self.t_tau = f(self.zs), f(self.S), f(self.tau)

    def rows(self, split):
        return np.where(self.split == split)[0]

    def pairs(self, split, h):
        """All (row, t) with t in [0, 63 - h] of the split's sequences, row-major."""
        r = self.rows(split); nt = Z.T - h
        return np.repeat(r, nt), np.tile(np.arange(nt), len(r))

    def inputs(self, r, t, h):
        """Torch probe inputs for pair arrays r, t (tensors on the device): z_t, S, tau_t, tau_{t+h}; target z_{t+h}."""
        return self.t_zs[r, t], self.t_S[r], self.t_tau[r, t], self.t_tau[r, t + h], self.t_zs[r, t + h]
