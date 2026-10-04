"""Tensors of the Stage-2 study: the previous temporal study's SeqData (raw / standardised contact, G = fd.S, tau_local, the
residual rho_t = (C_t - s_0) / sigma_r, exact R2 / wrench targets, masks for the evaluation) plus
  * the Stage-1 per-point geometry tokens geo_v = [x_v / l | outward normal (ARCTIC top part articulated per frame) | nearest / 0.3 |
    coverage | cell area x 512 | valid]  (10 channels) that the Stage-1 decoders D_z / D_r read, and
  * the cached teacher latent z*_t = E_z^Stage1(C_t^GT, G_t) of every sequence frame (cache/teacher_z.npz, written by
    s2_cache_teacher.py) with its TRAIN mean / std, so that the networks predict z in standardised coordinates.
One dataset per process (the hierarchical loader is import-bound).
"""
from __future__ import annotations

import numpy as np
import torch

import s2_common as S
from sat_data import SeqData, assert_causal_conditioning  # noqa: F401


class S2Data(SeqData):
    def __init__(self, ds, device, stats=None, need_masks=False, teacher=True):
        super().__init__(ds, device, need_masks=need_masks, stats=stats["sat"] if stats else None)
        z = np.load(S.SAT.HP.ROOTS[ds] / "sequences.npz")
        gi = z["g_index"]
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.pstat = f(np.stack([z["G_nearest"][gi] / 0.3, z["G_coverage"][gi]], -1))     # (N, 512, 2)
        self.Cn = self.fd.C                                                                   # (N, 64, 512) standardised (hier)
        self.mu = f(self.fd.stats["mu"]); self.s = float(self.fd.stats["s"])
        self.z_star = self.z_mu = self.z_sd = self.r_mu = None
        self.teacher_info = {}
        if teacher:
            tc = np.load(S.teacher_cache_path(ds), allow_pickle=True)
            self.z_star = f(tc["z"])                                                           # (N, 64, dz) raw Stage-1 coordinates (NaN where not encoded)
            self.teacher_info = dict(ckpt=str(tc["ckpt"]), md5=str(tc["md5"]), best_step=int(tc["best_step"]), dz=int(tc["z"].shape[-1]), dr=int(tc["r_train_mean"].shape[-1]))
            if stats is None:
                tr = self.idx["train"]
                zt = tc["z"][tr].reshape(-1, tc["z"].shape[-1])
                self.z_mu, self.z_sd = f(zt.mean(0)), f(zt.std(0).clip(min=1e-6))
                self.r_mu = f(tc["r_train_mean"])
            else:
                self.z_mu, self.z_sd, self.r_mu = f(stats["z_mu"]), f(stats["z_sd"]), f(stats["r_mu"])
        self.stats = dict(sat=self.stats, z_mu=None if self.z_mu is None else self.z_mu.cpu().numpy(), z_sd=None if self.z_sd is None else self.z_sd.cpu().numpy(),
                          r_mu=None if self.r_mu is None else self.r_mu.cpu().numpy(), teacher=self.teacher_info)

    # ------------------------------------------------------------------ Stage-1 geometry tokens for frame pairs (n, t)
    def geo_tokens(self, n, t):
        """n, t (B,) -> (B, 512, 10) per-point channels of the Stage-1 decoders (positions / l, outward normals, nearest, coverage, area, valid)."""
        geo = self.geo; g = geo.geo_index[n]
        Xt, Xb, Nt, Nb = geo.X_top[g], geo.X_bot[g], geo.N_top[g], geo.N_bot[g]
        if geo.articulated:
            th = -geo.arti[n, t + S.PAD]
            c, s = torch.cos(th)[:, None, None], torch.sin(th)[:, None, None]
            rot = lambda v: torch.cat([c * v[..., :1] - s * v[..., 1:2], s * v[..., :1] + c * v[..., 1:2], v[..., 2:]], -1)
            X, Nn = rot(Xt) + Xb, rot(Nt) + Nb
        else:
            X, Nn = Xt + Xb, Nt + Nb
        x = (X - geo.centroid[g][:, None]) / geo.length[g][:, None, None]
        Nn = Nn / (Nn.norm(dim=-1, keepdim=True) + 1e-9)
        alpha, valid = geo.alpha[g], geo.valid[g]
        return torch.cat([x, Nn, self.pstat[n], (alpha * S.N_PTS)[..., None], valid.float()[..., None]], -1)

    # ------------------------------------------------------------------ latent coordinates
    def z_standardise(self, z_raw):
        return (z_raw - self.z_mu) / self.z_sd

    def z_destandardise(self, z_std):
        return z_std * self.z_sd + self.z_mu

    def teacher_std(self, n, t=None):
        """Standardised teacher z*_t of frames 1..63 (default) of the sequences n: (B, 63, dz)."""
        z = self.z_star[n, 1:] if t is None else self.z_star[n[:, None], t]
        return self.z_standardise(z)

    # ------------------------------------------------------------------ contact unit conversions
    def std_to_raw(self, Cn):
        return Cn * self.s + self.mu

    def raw_to_rho(self, C_raw, s0_raw):
        """raw maps (B, T', 512) of frames t and s_0 (B, 512) -> the residual units of the dense loss."""
        return (C_raw - s0_raw[:, None]) / self.sigma_r

    # ------------------------------------------------------------------ training batch (GT s_0; no masks)
    def train_batch(self, n):
        s0 = self.C[n, 0]
        return dict(n=n, s0_raw=s0, s0_in=self.s0_input(s0), S=self.fd.S[n], tau=self.tau(n), rho=self.residual(n, s0), C=self.C[n, 1:],
                    z_star=None if self.z_star is None else self.teacher_std(n))

    def d_geo(self):
        return 10
