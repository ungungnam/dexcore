"""The four z-mediated models of the 2 x 2 design (all standardised teacher coordinates for z).

Temporal part
  whole (A0)     the Stage-2 backbone unchanged: one token per future frame from (s_0, tau_local,t), 768 x 8 blocks, FiLM on (s_0, G);
                 linear head -> z_hat_1:63 in one pass.
  stateful (A1)  z_hat_0 = I(s_0, G)            a point-token contact encoder over the 512 canonical points (tokens = frame-0 geometry | s_0,
                                                FiLM on G; the Stage-1 encoder's architecture, trained from scratch: no Stage-1 weights)
                 z_hat_t+1 = z_hat_t + F(z_hat_t, G, tau_t .. tau_t+k)
                                                a residual MLP (width 1024, 6 blocks, FiLM on G) on the state and the local trajectory window (k = 4);
                                                the update comes from a zero-initialised linear layer, so the model starts as latent persistence.
                                                The same F is applied 63 times on its OWN output (rollout); the teacher latent is never an input.
Decoder (per frame, point-token transformer: the Stage-1 D_z blocks, loaded, + 2 identity-initialised blocks; FiLM on [G | z_hat_t])
  absolute (B0)  tokens = geometry of frame t                       -> C_t (standardised contact units)
  residual (B1)  tokens = geometry of frame t | s_0 at that point   -> Delta_C_t ;  C_t = s_0 + Delta_C_t.  The s_0 channel is a new input
                 column (zero-initialised) and the output layer is zero-initialised, so the model starts as contact persistence C_t = s_0.
M00 is the previous follow-up's network (zj_models.JointModel); the helpers below give all four one interface.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import zf_common as Z
import zj_models as ZJM
import s2_models as S2M
import cf_models as CFM

CF_ARCH = S2M.CF_ARCH


class StateTransition(nn.Module):
    """Delta z = F(z_hat_t, G, tau_t .. tau_t+k): a residual MLP on the state and the flattened local trajectory window, FiLM-conditioned on G
    in every block (depth x [LayerNorm -> FiLM -> Linear 4x -> SiLU -> Linear], width 1024), zero-initialised read-out (latent persistence at start).
    A transformer over [state | G | tau] tokens of the same size was measured at 3.1 x the rollout time and 6 x the memory (63 sequential steps
    are dominated by per-operation overhead), so the plan's other option, a strong residual MLP, is used."""
    def __init__(self, dz, d_traj, d_static, width, depth, dropout, cond_width, k):
        super().__init__()
        self.k, self.depth, self.width = k, depth, width
        self.inp = nn.Linear(dz + (k + 1) * d_traj, width)
        self.cond = S2M.mlp(d_static, 512, cond_width); self.film = nn.Linear(cond_width, depth * 2 * width)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)                   # FiLM starts as the identity
        self.norms = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in range(depth)])
        self.l1 = nn.ModuleList([nn.Linear(width, 4 * width) for _ in range(depth)]); self.l2 = nn.ModuleList([nn.Linear(4 * width, width) for _ in range(depth)])
        self.drop = nn.Dropout(dropout); self.norm = nn.LayerNorm(width); self.out = nn.Linear(width, dz)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def static(self, Sg):
        """G -> FiLM scale and shift of every block (B, depth, width) x 2; computed once per sequence."""
        f = self.film(torch.nn.functional.silu(self.cond(Sg))).view(Sg.shape[0], self.depth, 2, self.width)
        return f[:, :, 0], f[:, :, 1]

    def step(self, z, sc, sh, tau_ctx):
        """z (B, dz), tau_ctx (B, k + 1, d_traj) = tau_local of frames t .. t + k  ->  Delta z (B, dz)."""
        h = self.inp(torch.cat([z, tau_ctx.flatten(1)], -1))
        for i in range(self.depth):
            h = h + self.l2[i](self.drop(torch.nn.functional.silu(self.l1[i](self.norms[i](h) * (1 + sc[:, i]) + sh[:, i]))))
        return self.out(self.norm(h))


class FactorialModel(nn.Module):
    def __init__(self, stateful, residual, d_traj, d_static, d_geo, z_mu, z_sd, dz=Z.S2.DZ, stage1_state=None, state_cfg=Z.STATE, backbone_cfg=Z.BACKBONE, extra_blocks=Z.DECODER["extra_blocks"]):
        super().__init__()
        self.stateful, self.residual, self.dz, self.use_z = stateful, residual, dz, True
        self.cfg_state = dict(state_cfg); self.ckpt_dec = True; self.ckpt_steps = bool(state_cfg.get("ckpt_steps", False))
        self.register_buffer("z_mu", z_mu.clone().float()); self.register_buffer("z_sd", z_sd.clone().float())
        if stateful:
            c = state_cfg
            # The initial-state module has the Stage-1 encoder's architecture.  Built first under the same seed it would also get the Stage-1 encoder's
            # random INITIAL weights (found by a sanity check: cosine 0.89 with the trained encoder).  Its initialisation is therefore drawn from a
            # separate random stream, so that nothing of the teacher, not even its starting point, is shared.
            rs = torch.random.get_rng_state(); torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,))) ^ 0x5F3759DF)
            self.init = CFM.PointEncoder(d_geo + 1, d_static, dz, c["init_width"], c["init_depth"], c["init_heads"], c["dropout"], c["cond_width"], c["init_pool"])
            torch.random.set_rng_state(rs)
            self.trans = StateTransition(dz, d_traj, d_static, c["width"], c["depth"], c["dropout"], c["cond_width"], c["k"])
        else:
            b = backbone_cfg
            self.backbone = S2M.TemporalBackbone(d_traj, d_static, b["width"], b["depth"], b["heads"], b["dropout"], b["cond_width"], b["mlp_ratio"])
            self.z_head = nn.Linear(b["width"], dz); nn.init.zeros_(self.z_head.weight); nn.init.zeros_(self.z_head.bias)
        kw = dict(width=CF_ARCH["width"], depth=CF_ARCH["depth"], heads=CF_ARCH["heads"], dropout=CF_ARCH["dropout"], cond_width=CF_ARCH["cond_width"])
        self.D_z = CFM.PointDecoder(d_geo + (1 if residual else 0), d_static + dz, **kw)
        if stage1_state is not None:
            self.load_stage1_decoder(stage1_state)
        self.n_stage1_blocks = len(self.D_z.blocks)
        for _ in range(extra_blocks):
            self.D_z.blocks.append(CFM.AdaLNBlock(CF_ARCH["width"], CF_ARCH["cond_width"], CF_ARCH["heads"], CF_ARCH["dropout"], zero_gate=True))
        if residual and stage1_state is not None:                                         # contact persistence at initialisation: Delta_C = 0
            nn.init.zeros_(self.D_z.out.weight); nn.init.zeros_(self.D_z.out.bias)

    def load_stage1_decoder(self, state):
        """Stage-1 A3 structure-decoder weights; for the residual decoder the input layer gets one more (zero) column for the s_0 channel."""
        sub = {k[len("D_z."):]: v.clone() for k, v in state.items() if k.startswith("D_z.")}
        if self.residual:
            w = sub["inp.weight"]; sub["inp.weight"] = torch.cat([w, torch.zeros(w.shape[0], 1, dtype=w.dtype)], 1)
        self.D_z.load_state_dict(sub, strict=True)

    # ------------------------------------------------------------------ temporal part
    def init_state(self, C_in, geo, Sg):
        """I(C, G): standardised contact map C_in (M, 512), its geometry tokens (M, 512, 10), G (M, d_static) -> z_hat (M, dz), standardised."""
        return self.init(torch.cat([geo, C_in[..., None]], -1), Sg)

    def rollout(self, z0, tau_all, Sg, steps=Z.TF):
        """z0 (B, dz); tau_all (B, 64 + k, d_traj) = tau_local of frames 0 .. 63 (the last frame repeated k times) -> z_hat_1:steps (B, steps, dz)."""
        sc, sh = self.trans.static(Sg); z, out, k = z0.float(), [], self.trans.k        # the state is carried in float32: under bf16 autocast a 63-step sum of
        for t in range(steps):                                                          # small updates would be rounded at every step (found by the code review)
            ctx = tau_all[:, t:t + k + 1]
            dz = checkpoint(self.trans.step, z, sc, sh, ctx, use_reentrant=False) if (self.training and self.ckpt_steps and torch.is_grad_enabled()) else self.trans.step(z, sc, sh, ctx)
            z = z + dz.float(); out.append(z)
        return torch.stack(out, 1)

    def whole(self, s0_in, tau, Sg):
        return self.z_head(self.backbone(s0_in, tau, Sg))

    # ------------------------------------------------------------------ decoder
    def decode(self, z_std, geo, Sg, s0_in=None, chunk=0):
        """z_std (M, dz), geo (M, 512, 10), G (M, d_static) [, s0_in (M, 512) standardised s_0 for the residual decoder] -> network output (M, 512):
        the contact map (absolute) or its change from s_0 (residual), in standardised contact units."""
        if chunk and z_std.shape[0] > chunk:
            return torch.cat([self.decode(z_std[i:i + chunk], geo[i:i + chunk], Sg[i:i + chunk], None if s0_in is None else s0_in[i:i + chunk]) for i in range(0, z_std.shape[0], chunk)])
        dec = self.D_z
        if self.residual:
            geo = torch.cat([geo, s0_in[..., None]], -1)
        c = dec.cond(torch.cat([Sg, z_std * self.z_sd + self.z_mu], -1))
        h = dec.inp(geo) + dec.pos
        for b in dec.blocks:
            h = checkpoint(b, h, c, use_reentrant=False) if (self.training and self.ckpt_dec and torch.is_grad_enabled()) else b(h, c)
        return dec.out(dec.norm(h)).squeeze(-1)

    def param_counts(self):
        n = lambda m: sum(p.numel() for p in m.parameters())
        d = dict(temporal=(n(self.init) + n(self.trans)) if self.stateful else (n(self.backbone) + n(self.z_head)), D_z=n(self.D_z), total=n(self))
        if self.stateful:
            d.update(init=n(self.init), transition=n(self.trans))
        return d


def build(model, data, stage1_state=None):
    c = Z.MODELS[model]
    if c["reuse"]:
        return ZJM.build("M2", data, stage1_state=stage1_state)
    return FactorialModel(c["stateful"], c["residual"], data.d_traj(), data.d_static(), data.d_geo(), data.z_mu, data.z_sd, stage1_state=stage1_state)


def load(ds, model, data, dev, seed=Z.SEED, key="state", which="best"):
    path = Z.model_ckpt(ds, model, seed) if which == "best" else Z.ckpt_path(ds, Z.run_name(model, seed), which)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = (ZJM.build("M2", data, extra_blocks=ck.get("cfg", {}).get("extra_blocks")) if Z.MODELS[model]["reuse"] else build(model, data)).to(dev)
    net.load_state_dict(ck[key]); net.eval()
    return net, ck


# ---------------------------------------------------------------------- one interface for the four models
def is_stateful(net):
    return bool(getattr(net, "stateful", False))


def is_residual(net):
    return bool(getattr(net, "residual", False))


def tau_all(data, n, k):
    """tau_local of frames 0 .. 63 with the last frame repeated k times: (B, 64 + k, d_traj)."""
    t = torch.arange(0, Z.T + k, device=data.device).clamp(max=Z.T - 1)[None].expand(len(n), -1)
    return data.tau(n, t)


def predict_latents(net, data, n, z0=None):
    """(s_0, G, tau) -> z_hat_1:63 (B, 63, dz) [and z_hat_0 (B, dz) for the stateful models].  z0: start the rollout from a GIVEN initial latent
    instead of I(s_0, G) (analysis only)."""
    s0 = data.C[n, 0]; Sg = data.fd.S[n]
    if is_stateful(net):
        if z0 is None:
            z0 = net.init_state(data.s0_input(s0), data.geo_tokens(n, torch.zeros_like(n)), Sg)
        return net.rollout(z0, tau_all(data, n, net.trans.k), Sg), z0
    if isinstance(net, FactorialModel):
        return net.whole(data.s0_input(s0), data.tau(n), Sg), None
    return net.latents(data.s0_input(s0), data.tau(n), Sg)["z_std"], None


def decode_raw(net, data, z_std, geo, Sg, s0_raw, chunk=0):
    """Latents (M, dz) -> raw contact maps (M, 512) for any of the four models (s0_raw (M, 512): the initial map of each row's sequence)."""
    if isinstance(net, FactorialModel):
        y = net.decode(z_std, geo, Sg, data.s0_input(s0_raw) if net.residual else None, chunk).float()
        return s0_raw + data.s * y if net.residual else data.std_to_raw(y)
    return data.std_to_raw(ZJM.decode(net, z_std, geo, Sg, chunk).float())
