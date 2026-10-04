"""Point-token networks of the factorisation study.

Every module is a transformer over the 512 canonical points (token = per-point channels + a learned index embedding, width 256,
4 blocks of self-attention + MLP, 8 heads) conditioned by adaLN-zero FiLM (the previous studies' block) on a vector
c = MLP([G | latents]).  Encoders pool the tokens with 4 learned attention queries and map to the latent; decoders read the
geometry tokens only (no contact) and emit one value per point.

  E_z : [geo (10) | C~]                 , c = f(G)           -> z            structure encoder
  D_z : [geo]                           , c = f(G, z)        -> C_bar        structure decoder
  E_r : [geo | C~ | C~ - C_bar (sg)]    , c = f(G, sg(z))    -> r            realisation encoder (conditioned on z)
  D_r : [geo | sg(C_bar) | c]           , c = f(G, sg(z), r) -> Delta_C      residual decoder (c also concatenated to the tokens,
                                                                            output layer initialised at 0.1 x the default scale)
  C_hat = C_bar + Delta_C.   A0 (one latent h) and A1 (z only) use E_z / D_z alone.
The r branch reads z and C_bar through a stop-gradient: z is shaped only by its own reconstruction, the relational loss and
the full reconstruction THROUGH C_bar; the r branch adapts to z rather than the other way round.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import cf_common as S


class Attention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.h, self.d = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width); self.proj = nn.Linear(width, width); self.p = dropout

    def forward(self, x, kv=None):
        B, L, W = x.shape
        if kv is None:
            q, k, v = self.qkv(x).view(B, L, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        else:
            q = F.linear(x, self.qkv.weight[:W], self.qkv.bias[:W]).view(B, L, self.h, self.d).transpose(1, 2)
            k, v = F.linear(kv, self.qkv.weight[W:], self.qkv.bias[W:]).view(B, kv.shape[1], 2, self.h, self.d).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.p if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, L, W))


class AdaLNBlock(nn.Module):
    """zero_gate=True: adaLN-zero (the block starts as the identity, DiT); False: the gates start at 1 (a standard pre-LN block
    with learned FiLM modulation) — used in the residual decoder, where a zero-initialised block behind a near-zero output
    layer never receives gradient (its output layer is driven to zero in the first phase-B steps, see the launch history)."""
    def __init__(self, width, cond, heads, dropout, zero_gate=True):
        super().__init__()
        self.gate_base = 0.0 if zero_gate else 1.0
        self.n1 = nn.LayerNorm(width, elementwise_affine=False); self.attn = Attention(width, heads, dropout)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(4 * width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond, 6 * width))
        nn.init.zeros_(self.ada[1].weight); nn.init.zeros_(self.ada[1].bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        h = h + (self.gate_base + g1) * self.drop(self.attn(self.n1(h) * (1 + s1) + b1))
        return h + (self.gate_base + g2) * self.mlp(self.n2(h) * (1 + s2) + b2)


def cond_mlp(d_in, cond_width):
    return nn.Sequential(nn.Linear(d_in, 512), nn.SiLU(), nn.Linear(512, cond_width), nn.SiLU(), nn.Linear(cond_width, cond_width))


class PointEncoder(nn.Module):
    def __init__(self, c_in, d_cond, d_out, width, depth, heads, dropout, cond_width, n_pool):
        super().__init__()
        self.inp = nn.Linear(c_in, width); self.pos = nn.Parameter(torch.randn(1, S.N_PTS, width) * 0.02)
        self.cond = cond_mlp(d_cond, cond_width)
        self.blocks = nn.ModuleList([AdaLNBlock(width, cond_width, heads, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.pool_q = nn.Parameter(torch.randn(1, n_pool, width) * 0.02); self.pool = Attention(width, heads, 0.0)
        self.out = nn.Sequential(nn.Linear(n_pool * width, width), nn.SiLU(), nn.Linear(width, d_out))

    def forward(self, tokens, cond_in):
        c = self.cond(cond_in)
        h = self.inp(tokens) + self.pos
        for b in self.blocks:
            h = b(h, c)
        h = self.norm(h)
        pooled = self.pool(self.pool_q.expand(h.shape[0], -1, -1), kv=h)
        return self.out(pooled.flatten(1))


class PointDecoder(nn.Module):
    """inject_cond: the condition vector is also concatenated to every token at the input (the residual decoder), so that the
    latents reach the tokens without passing through the zero-initialised FiLM gates; out_scale < 1 shrinks the output layer's
    initialisation (a near-zero but not exactly zero start: an exactly zero output layer behind zero gates receives no gradient
    through the gates and the r branch never activates — observed in the first launch)."""
    def __init__(self, c_in, d_cond, width, depth, heads, dropout, cond_width, inject_cond=False, out_scale=1.0, zero_gate=True, code_dim=0, basis_dim=64):
        super().__init__()
        self.inject = inject_cond
        # structure-conditioned linear basis (residual decoder): Delta_C_v += <K h_v, W r> / sqrt(basis_dim), a per-point readout of the
        # code r through a basis produced by the tokens — an immediately expressive, saddle-free path from r to every point
        self.key = nn.Linear(width, basis_dim) if code_dim else None
        self.code = nn.Linear(code_dim, basis_dim, bias=False) if code_dim else None
        self.basis_dim = basis_dim
        self.inp = nn.Linear(c_in + (cond_width if inject_cond else 0), width); self.pos = nn.Parameter(torch.randn(1, S.N_PTS, width) * 0.02)
        self.cond = cond_mlp(d_cond, cond_width)
        self.blocks = nn.ModuleList([AdaLNBlock(width, cond_width, heads, dropout, zero_gate=zero_gate) for _ in range(depth)])
        self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, 1)
        if out_scale != 1.0:
            with torch.no_grad():
                self.out.weight.mul_(out_scale); self.out.bias.zero_()

    def forward(self, geo, cond_in, code=None):
        c = self.cond(cond_in)
        if self.inject:
            geo = torch.cat([geo, c[:, None].expand(-1, geo.shape[1], -1)], -1)
        h = self.inp(geo) + self.pos
        for b in self.blocks:
            h = b(h, c)
        h = self.norm(h); y = self.out(h).squeeze(-1)
        if code is not None and self.key is not None:
            y = y + torch.einsum("bvd,bd->bv", self.key(h), self.code(code)) / self.basis_dim ** 0.5
        return y


class Factorized(nn.Module):
    """dz > 0 always; dr = 0 gives the single-latent models (A0 with dz = 128, A1 with dz = 64)."""
    def __init__(self, dz, dr, d_geo, d_static, arch=S.ARCH, res_scale=1.0):
        super().__init__()
        kw = dict(width=arch["width"], depth=arch["depth"], heads=arch["heads"], dropout=arch["dropout"], cond_width=arch["cond_width"])
        self.dz, self.dr, self.res_scale = dz, dr, res_scale              # res_scale: gain of the residual input channel of E_r
        self.E_z = PointEncoder(d_geo + 1, d_static, dz, n_pool=arch["n_pool"], **kw)
        self.D_z = PointDecoder(d_geo, d_static + dz, **kw)
        if dr > 0:
            self.E_r = PointEncoder(d_geo + 2, d_static + dz, dr, n_pool=arch["n_pool"], **kw)
            self.D_r = PointDecoder(d_geo + 1, d_static + dz + dr, inject_cond=True, out_scale=0.1, zero_gate=False, code_dim=dr, **kw)      # tokens [geo | C_bar]

    def z_params(self):
        return list(self.E_z.parameters()) + list(self.D_z.parameters())

    def r_params(self):
        return (list(self.E_r.parameters()) + list(self.D_r.parameters())) if self.dr > 0 else []

    def encode_z(self, Cn, geo, Sg):
        return self.E_z(torch.cat([geo, Cn[..., None]], -1), Sg)

    def decode_z(self, z, geo, Sg):
        return self.D_z(geo, torch.cat([Sg, z], -1))

    def encode_r(self, Cn, C_bar, z, geo, Sg):
        return self.E_r(torch.cat([geo, Cn[..., None], ((Cn - C_bar) * self.res_scale)[..., None]], -1), torch.cat([Sg, z], -1))

    def decode_r(self, z, r, geo, Sg, C_bar):
        return self.D_r(torch.cat([geo, C_bar[..., None]], -1), torch.cat([Sg, z, r], -1), code=r)

    def forward(self, Cn, geo, Sg, use_r=True):
        z = self.encode_z(Cn, geo, Sg); C_bar = self.decode_z(z, geo, Sg)
        out = dict(z=z, C_bar=C_bar, r=None, delta=None, C_hat=C_bar)
        if self.dr > 0 and use_r:
            zd, Cd = z.detach(), C_bar.detach()
            r = self.encode_r(Cn, Cd, zd, geo, Sg); delta = self.decode_r(zd, r, geo, Sg, Cd)
            out.update(r=r, delta=delta, C_hat=C_bar + delta)
        return out


def build(model, d_geo, d_static, res_scale=1.0):
    c = S.ALL_MODELS[model]
    return Factorized(c["dz"], c["dr"], d_geo, d_static, res_scale=res_scale)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
