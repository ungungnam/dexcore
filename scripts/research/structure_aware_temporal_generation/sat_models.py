"""One sequence backbone for both families, so capacity and conditioning are matched.

Tokens   one per generated frame t = 1..63: [x_t (512) | s_0~ (512) | tau_local,t (9 D)] -> width, + learned position.
         x_t is the noisy residual for the diffusion family and ZERO for the deterministic family (same input width).
Cond     c = enc(G, hand / role flags) + enc(s_0~) [+ emb(diffusion step k)]  (256-D), applied by adaLN-zero FiLM in every
         block (the previous studies' FiLM conditioning, with the DiT zero-initialised gates), 6 blocks of self-attention
         over the 63 frames + MLP (width 512, 8 heads, dropout 0.1).
Output   per-frame 512-D residual r_t (zero-initialised: the deterministic model starts as the persistence predictor
         C_t = s_0; the diffusion model starts at v = 0) and, when enabled, the auxiliary head h_t -> Z_aux,t (48-D exact R2,
         training only).
Diffusion over the residual sequence r_1:63 as ONE sample: 1000 steps, cosine alpha-bar, v-prediction (the initial sampler's
schedule and parameterisation), DDIM sampling (eta = 0, 100 steps); frame 0 is never part of the generated tensor.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import sat_common as S


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], -1)


class AdaLNBlock(nn.Module):
    def __init__(self, width, cond, heads, dropout):
        super().__init__()
        self.n1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond, 6 * width))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        x = self.n1(h) * (1 + s1) + b1
        h = h + g1 * self.drop(self.attn(x, x, x, need_weights=False)[0])
        x = self.n2(h) * (1 + s2) + b2
        return h + g2 * self.mlp(x)


class SeqBackbone(nn.Module):
    def __init__(self, d_traj, d_static, width=512, depth=6, heads=8, n_frames=S.TF, use_time=False, dropout=0.1, cond_width=256, aux=False):
        super().__init__()
        self.inp = nn.Linear(512 + 512 + d_traj, width)
        self.pos = nn.Parameter(torch.randn(1, n_frames, width) * 0.02)
        self.static = nn.Sequential(nn.Linear(d_static, 512), nn.SiLU(), nn.Linear(512, cond_width))
        self.s0enc = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, cond_width))
        self.time = nn.Sequential(nn.Linear(128, cond_width), nn.SiLU(), nn.Linear(cond_width, cond_width)) if use_time else None
        self.cond_out = nn.Sequential(nn.SiLU(), nn.Linear(cond_width, cond_width))
        self.blocks = nn.ModuleList([AdaLNBlock(width, cond_width, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, 512)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.aux = nn.Linear(width, S.R2_DIM) if aux else None

    def forward(self, x, s0_in, tau, s, k=None):
        """x (B, T', 512) frame input (noisy residual or zeros), s0_in (B, 512), tau (B, T', 9 D), s (B, d_static), k (B,) step."""
        c = self.static(s) + self.s0enc(s0_in)
        if self.time is not None:
            c = c + self.time(timestep_embedding(k, 128))
        c = self.cond_out(c)
        h = self.inp(torch.cat([x, s0_in[:, None].expand(-1, x.shape[1], -1), tau], -1)) + self.pos
        for b in self.blocks:
            h = b(h, c)
        h = self.norm(h)
        out = self.out(h)
        return out, (self.aux(h) if self.aux is not None else None)


class Deterministic(nn.Module):
    """r_hat = F(s_0, G, tau) over the 63 future frames; C_hat_t = s_0 + sigma_r r_hat_t."""
    def __init__(self, d_traj, d_static, aux=False, **kw):
        super().__init__()
        self.net = SeqBackbone(d_traj, d_static, use_time=False, aux=aux, **kw)

    def forward(self, s0_in, tau, s):
        x = torch.zeros(s0_in.shape[0], tau.shape[1], 512, device=s0_in.device)
        return self.net(x, s0_in, tau, s)


class SeqDiffusion(nn.Module):
    N_STEPS = 1000

    def __init__(self, d_traj, d_static, aux=False, **kw):
        super().__init__()
        self.net = SeqBackbone(d_traj, d_static, use_time=True, aux=aux, **kw)
        st = torch.arange(self.N_STEPS + 1, dtype=torch.float64) / self.N_STEPS
        ab = torch.cos((st + 0.008) / 1.008 * math.pi / 2) ** 2
        ab = (ab / ab[0]).clamp(1e-5, 1.0)
        self.register_buffer("alpha_bar", ab.float())

    def v_pred(self, x_k, k, s0_in, tau, s):
        return self.net(x_k, s0_in, tau, s, k)

    def forward_loss(self, r0, s0_in, tau, s, k=None, noise=None):
        """v-prediction MSE on the whole sequence; also returns the x0 estimate (for the structural loss) and the aux output."""
        B = len(r0)
        if k is None:
            k = torch.randint(1, self.N_STEPS + 1, (B,), device=r0.device)
        if noise is None:
            noise = torch.randn_like(r0)
        ab = self.alpha_bar[k][:, None, None]
        x_k = ab.sqrt() * r0 + (1 - ab).sqrt() * noise
        v = ab.sqrt() * noise - (1 - ab).sqrt() * r0
        v_hat, aux = self.v_pred(x_k, k, s0_in, tau, s)
        x0_hat = ab.sqrt() * x_k - (1 - ab).sqrt() * v_hat
        return F.mse_loss(v_hat, v), x0_hat, aux, k

    @torch.no_grad()
    def sample(self, s0_in, tau, s, n_steps=100, generator=None):
        B, Tp = s0_in.shape[0], tau.shape[1]
        x = torch.randn(B, Tp, 512, device=s0_in.device, generator=generator)
        ks = torch.linspace(self.N_STEPS, 0, n_steps + 1, device=s0_in.device).round().long()
        for i in range(n_steps):
            k, k_next = ks[i], ks[i + 1]
            ab, ab_next = self.alpha_bar[k], self.alpha_bar[k_next]
            v, _ = self.v_pred(x, torch.full((B,), int(k), device=s0_in.device), s0_in, tau, s)
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps                 # DDIM, eta = 0
        return x


def build(model, d_traj, d_static, cfg):
    c = S.MODELS[model]
    kw = dict(width=cfg["width"], depth=cfg["depth"], heads=cfg["heads"], dropout=cfg["dropout"])
    if c["family"] == "det":
        return Deterministic(d_traj, d_static, aux=c["aux"], **kw)
    return SeqDiffusion(d_traj, d_static, aux=c["aux"], **kw)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
