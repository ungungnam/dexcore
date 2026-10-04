"""Networks of the hierarchical study. All three models share one static-condition encoder and the
same residual-MLP trunk so the sampler, the vector field and the dense predictor are comparable.

  Sampler   DDPM on the standardised contact vector (512-D): 1000 training steps, cosine
            alpha-bar schedule, v-prediction, MSE loss; DDIM sampling (eta = 0, 100 steps), so
            all stochasticity is the Gaussian initial noise. Conditioning: static G (+ hand/role)
            and, for the GT variant, tau_local,0 (9 object states), through FiLM in every block.
  VF        v_phi(C_t, G, tau_local,t) -> C_{t+1} - C_t (standardised units, Delta t = 1 frame)
  Dense     f(G, tau_local,t) -> C_t                      (the earlier study's family C, pooled)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMBlock(nn.Module):
    def __init__(self, width, cond, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(cond, 2 * width)
        # the Dropout module is inserted only when used, so checkpoints trained without it keep their keys
        layers = [nn.Linear(width, width), nn.SiLU()] + ([nn.Dropout(dropout)] if dropout > 0 else []) + [nn.Linear(width, width)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, -1)
        return h + self.mlp(self.norm(h) * (1 + scale) + shift)


class Trunk(nn.Module):
    def __init__(self, d_in, d_out, cond, width=1024, blocks=4, dropout=0.0):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([FiLMBlock(width, cond, dropout) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, d_out))
        nn.init.zeros_(self.out[1].weight); nn.init.zeros_(self.out[1].bias)

    def forward(self, x, c):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, c)
        return self.out(h)


class CondEncoder(nn.Module):
    """static (1032) [+ trajectory context (9*D)] [+ diffusion time] -> 256-D FiLM condition."""
    def __init__(self, d_static, d_traj=0, use_time=False, width=256):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(d_static, 512), nn.SiLU(), nn.Linear(512, width))
        self.traj = nn.Sequential(nn.Linear(d_traj, 256), nn.SiLU(), nn.Linear(256, width)) if d_traj else None
        self.time = nn.Sequential(nn.Linear(128, width), nn.SiLU(), nn.Linear(width, width)) if use_time else None
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, s, traj=None, t=None):
        c = self.static(s)
        if self.traj is not None:
            c = c + self.traj(traj)
        if self.time is not None:
            c = c + self.time(timestep_embedding(t, 128))
        return self.out(c)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


# ------------------------------------------------------------------------------ diffusion
class Diffusion(nn.Module):
    N_STEPS = 1000

    def __init__(self, d_static, d_traj=0, width=1024, blocks=4, dropout=0.0):
        super().__init__()
        self.cond = CondEncoder(d_static, d_traj, use_time=True)
        self.net = Trunk(512, 512, 256, width, blocks, dropout)
        s = torch.arange(self.N_STEPS + 1, dtype=torch.float64) / self.N_STEPS
        ab = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = (ab / ab[0]).clamp(1e-5, 1.0)
        self.register_buffer("alpha_bar", ab.float())          # alpha_bar[k], k = 0..N (k=0: data)

    def v_pred(self, x_t, k, s, traj=None):
        return self.net(x_t, self.cond(s, traj, k))

    def loss(self, x0, s, traj=None, k=None, noise=None):
        B = len(x0)
        if k is None:
            k = torch.randint(1, self.N_STEPS + 1, (B,), device=x0.device)
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.alpha_bar[k][:, None]
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        v = ab.sqrt() * noise - (1 - ab).sqrt() * x0
        return F.mse_loss(self.v_pred(x_t, k, s, traj), v)

    @torch.no_grad()
    def sample(self, s, traj=None, n_steps=100, generator=None):
        B = len(s)
        x = torch.randn(B, 512, device=s.device, generator=generator)
        ks = torch.linspace(self.N_STEPS, 0, n_steps + 1, device=s.device).round().long()
        for i in range(n_steps):
            k, k_next = ks[i], ks[i + 1]
            ab, ab_next = self.alpha_bar[k], self.alpha_bar[k_next]
            v = self.v_pred(x, torch.full((B,), int(k), device=s.device), s, traj)
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps        # DDIM, eta = 0
        return x


class VectorField(nn.Module):
    def __init__(self, d_static, d_traj, width=1024, blocks=3, dropout=0.0):
        super().__init__()
        self.cond = CondEncoder(d_static, d_traj)
        self.net = Trunk(512, 512, 256, width, blocks, dropout)

    def forward(self, c_t, s, traj):
        return self.net(c_t, self.cond(s, traj))


class Dense(nn.Module):
    def __init__(self, d_static, d_traj, width=1024, blocks=3, dropout=0.0):
        super().__init__()
        self.cond = CondEncoder(d_static, d_traj)
        self.net = Trunk(256, 512, 256, width, blocks, dropout)

    def forward(self, s, traj):
        c = self.cond(s, traj)
        return self.net(c, c)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
