"""One shared temporal backbone, three output pathways.

Backbone (identical for B0 / B1 / B2).  One token per future frame t = 1..63: [s_0~ (512) | tau_local,t (9 D)] -> width 768 + learned
position; condition c = enc(G) + enc(s_0~) (256-D) applied by adaLN-zero FiLM in each of the 8 blocks (12 heads, MLP x4, dropout 0.1;
the previous studies' block, zero-initialised gates so the network starts as the identity); final LayerNorm -> h_t (768).
The network therefore predicts the whole 63-frame future jointly from (s_0, G, tau) and nothing else: no hand pose, no GT labels,
no future contact, no teacher encoder at inference.

B0  h_t -> MLP 768-2048-2048-512 (last layer zero-initialised: the model starts as the persistence predictor C_t = s_0) -> rho_hat_t;
    C_hat_t = s_0 + sigma_r rho_hat_t.  The head is wider than a linear read-out so that the total parameter count stays within
    ~10 % of B1 / B2, which carry the point-token decoders (Section 13 of the plan).
B1  h_t -> linear -> z_hat_t (standardised teacher coordinates; zero-initialised: starts at the train-mean z) -> de-standardise ->
    C_bar_t = D_z(geo_t, [G | z_hat_t])  (the Stage-1 A3 structure decoder, loaded and fine-tuned end to end); C_hat_t = C_bar_t.
B2  as B1 plus h_t -> linear -> r_hat_t (64; zero weights, bias = the Stage-1 train-mean r), Delta_C_t = D_r([geo_t | sg(C_bar_t)],
    [G | sg(z_hat_t) | r_hat_t], code = r_hat_t) (the Stage-1 residual decoder, loaded and fine-tuned), C_hat_t = C_bar_t + Delta_C_t.
    As in Stage 1 the DECODER of the r pathway reads z_hat and C_bar through a stop-gradient, so D_r cannot pull on z_hat through its
    inputs and z_hat is shaped by L_z, L_zonly and L_full THROUGH C_bar only.  Unlike Stage 1 the two heads share the backbone features
    h_t (the plan asks for one trunk with separate heads), so the L_full gradient through D_r -> r head does reach the shared features;
    whether z stays the primary temporal state (H3) is therefore checked empirically (test L_z of B2 vs B1, Table 5) rather than enforced.
The decoders work in the hierarchical study's standardised contact units; all dense losses are evaluated after converting to the
residual units rho = (C - s_0) / sigma_r, so that B0 / B1 / B2 share one loss convention.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import s2_common as S
import cf_models as CFM


def mlp(d_in, d_hidden, d_out):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))


class TemporalBackbone(nn.Module):
    def __init__(self, d_traj, d_static, width, depth, heads, dropout, cond_width, mlp_ratio=4, n_frames=S.TF):
        super().__init__()
        assert mlp_ratio == 4, "the shared AdaLNBlock uses a 4x MLP"
        self.width = width
        self.inp = nn.Linear(512 + d_traj, width)
        self.pos = nn.Parameter(torch.randn(1, n_frames, width) * 0.02)
        self.static = mlp(d_static, 512, cond_width)
        self.s0enc = mlp(512, 512, cond_width)
        self.cond_out = nn.Sequential(nn.SiLU(), nn.Linear(cond_width, cond_width))
        self.blocks = nn.ModuleList([CFM.AdaLNBlock(width, cond_width, heads, dropout, zero_gate=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)

    def forward(self, s0_in, tau, Sg):
        """s0_in (B, 512) standardised s_0, tau (B, 63, 9 D), Sg (B, d_static) -> h (B, 63, width)."""
        c = self.cond_out(self.static(Sg) + self.s0enc(s0_in))
        h = self.inp(torch.cat([s0_in[:, None].expand(-1, tau.shape[1], -1), tau], -1)) + self.pos
        for b in self.blocks:
            h = b(h, c)
        return self.norm(h)


class Stage2Model(nn.Module):
    def __init__(self, model, d_traj, d_static, d_geo=10, dz=S.DZ, dr=S.DR, z_mu=None, z_sd=None, r_mu=None, stage1_state=None, cfg=S.BACKBONE):
        super().__init__()
        self.model, self.cfg = model, dict(cfg)
        c = S.MODELS[model]
        self.use_z, self.use_r = c["z"], c["r"]
        self.dz, self.dr = dz, dr
        self.ckpt_dec = bool(cfg.get("ckpt_dec", False))
        w = cfg["width"]
        self.backbone = TemporalBackbone(d_traj, d_static, w, cfg["depth"], cfg["heads"], cfg["dropout"], cfg["cond_width"], cfg["mlp_ratio"])
        self.register_buffer("z_mu", torch.zeros(dz) if z_mu is None else z_mu.clone().float())
        self.register_buffer("z_sd", torch.ones(dz) if z_sd is None else z_sd.clone().float())
        if not self.use_z:
            hh = cfg["b0_head_hidden"]
            self.head = nn.Sequential(nn.Linear(w, hh), nn.SiLU(), nn.Linear(hh, hh), nn.SiLU(), nn.Linear(hh, 512))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
            return
        kw = dict(width=CF_ARCH["width"], depth=CF_ARCH["depth"], heads=CF_ARCH["heads"], dropout=CF_ARCH["dropout"], cond_width=CF_ARCH["cond_width"])
        self.z_head = nn.Linear(w, dz)
        nn.init.zeros_(self.z_head.weight); nn.init.zeros_(self.z_head.bias)
        self.D_z = CFM.PointDecoder(d_geo, d_static + dz, **kw)
        if self.use_r:
            self.r_head = nn.Linear(w, dr)
            nn.init.zeros_(self.r_head.weight)
            with torch.no_grad():
                self.r_head.bias.copy_(torch.zeros(dr) if r_mu is None else r_mu.float())
            self.D_r = CFM.PointDecoder(d_geo + 1, d_static + dz + dr, inject_cond=True, out_scale=0.1, zero_gate=False, code_dim=dr, **kw)
        if stage1_state is not None:
            self.load_stage1_decoders(stage1_state)

    def load_stage1_decoders(self, state):
        """Initialise D_z (and D_r) from the Stage-1 A3 EMA weights (strict key match)."""
        sub = lambda pre: {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}
        self.D_z.load_state_dict(sub("D_z."), strict=True)
        if self.use_r:
            self.D_r.load_state_dict(sub("D_r."), strict=True)

    # ------------------------------------------------------------------ latent / dense heads
    def latents(self, s0_in, tau, Sg):
        """-> dict: rho (B, 63, 512) for B0; z_std (B, 63, dz) [and r (B, 63, dr)] for B1 / B2."""
        h = self.backbone(s0_in, tau, Sg)
        if not self.use_z:
            return dict(rho=self.head(h))
        out = dict(z_std=self.z_head(h))
        if self.use_r:
            out["r"] = self.r_head(h)
        return out

    def _decoder(self, dec, geo, cond_in, code=None):
        """cf_models.PointDecoder.forward with optional activation checkpointing of the point-token blocks (training only; the
        decoders process up to 128 x n_dec_frames maps of 512 tokens per step — recomputing the block activations in the backward
        pass trades ~30 % compute for a 3-4 x smaller footprint)."""
        c = dec.cond(cond_in)
        if dec.inject:
            geo = torch.cat([geo, c[:, None].expand(-1, geo.shape[1], -1)], -1)
        h = dec.inp(geo) + dec.pos
        for b in dec.blocks:
            h = checkpoint(b, h, c, use_reentrant=False) if (self.training and self.ckpt_dec and torch.is_grad_enabled()) else b(h, c)
        h = dec.norm(h); y = dec.out(h).squeeze(-1)
        if code is not None and dec.key is not None:
            y = y + torch.einsum("bvd,bd->bv", dec.key(h), dec.code(code)) / dec.basis_dim ** 0.5
        return y

    def decode_maps(self, z_std, r, geo, Sg):
        """Per-map decoding. z_std (M, dz) standardised, r (M, dr) or None, geo (M, 512, 10), Sg (M, d_static) ->
        C_bar (M, 512), delta (M, 512) or None, both in standardised contact units."""
        z_raw = z_std * self.z_sd + self.z_mu
        C_bar = self._decoder(self.D_z, geo, torch.cat([Sg, z_raw], -1))
        delta = None
        if self.use_r:
            assert r is not None
            delta = self._decoder(self.D_r, torch.cat([geo, C_bar.detach()[..., None]], -1), torch.cat([Sg, z_raw.detach(), r], -1), code=r)
        return C_bar, delta

    def param_counts(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        d = dict(backbone=n(self.backbone))
        if not self.use_z:
            d["head"] = n(self.head)
        else:
            d["head"] = n(self.z_head) + (n(self.r_head) if self.use_r else 0)
            d["D_z"] = n(self.D_z)
            if self.use_r:
                d["D_r"] = n(self.D_r)
        d["total"] = n(self)
        return d


CF_ARCH = S.CF.ARCH


def build(model, data, stage1_state=None, cfg=S.BACKBONE):
    return Stage2Model(model, data.d_traj(), data.d_static(), d_geo=data.d_geo(), dz=S.DZ, dr=S.DR, z_mu=data.z_mu, z_sd=data.z_sd, r_mu=data.r_mu,
                       stage1_state=stage1_state, cfg=cfg)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
