"""Models of the follow-up.  M0 / M1 are the Stage-2 B0 / B1 networks unchanged (s2_models.Stage2Model).  M2 / M3 are the B1 network with
a larger decoder D_phi: the Stage-1 structure decoder D_z (point-token transformer over the 512 canonical points, width 256, 8 heads,
adaLN-zero FiLM on [G | z_hat], reading the per-frame geometry tokens; loaded from the Stage-1 A3 checkpoint) followed by `extra_blocks`
more blocks of the same kind.  adaLN-zero blocks start as the identity, so at initialisation D_phi IS the Stage-1 decoder; all of it is
trained (nothing frozen).  Inference path: (s_0, G, tau) -> backbone -> linear -> z_hat_1:63 -> D_phi(geo_t, [G | z_hat_t]) -> C_hat_t.
The Stage-1 encoder E_z is not part of the model (it only produced the cached teacher targets z*).
"""
from __future__ import annotations

import torch

import zj_common as Z
import s2_models as S2M
import cf_models as CFM


class JointModel(S2M.Stage2Model):
    def __init__(self, d_traj, d_static, d_geo, z_mu, z_sd, stage1_state=None, extra_blocks=Z.DECODER["extra_blocks"], cfg=Z.BACKBONE):
        super().__init__("B1", d_traj, d_static, d_geo=d_geo, z_mu=z_mu, z_sd=z_sd, stage1_state=stage1_state, cfg=cfg)   # loads D_z (strict) when a Stage-1 state is given
        a = S2M.CF_ARCH
        self.n_stage1_blocks = len(self.D_z.blocks); self.extra_blocks = extra_blocks
        for _ in range(extra_blocks):
            self.D_z.blocks.append(CFM.AdaLNBlock(a["width"], a["cond_width"], a["heads"], a["dropout"], zero_gate=True))



def decode(net, z_std, geo, Sg, chunk=0):
    """The contact decoder of a z-mediated network on any latent input.  z_std (M, dz) standardised (predicted or teacher), geo (M, 512, 10),
    Sg (M, d_static) -> C (M, 512) in standardised contact units.  chunk > 0: at most `chunk` maps per decoder call (the attention maps of one
    block over 512 points are the memory peak of a step; chunking changes nothing in the function computed)."""
    if not chunk or z_std.shape[0] <= chunk:
        return net.decode_maps(z_std, None, geo, Sg)[0]
    return torch.cat([net.decode_maps(z_std[i:i + chunk], None, geo[i:i + chunk], Sg[i:i + chunk])[0] for i in range(0, z_std.shape[0], chunk)])


def build(model, data, stage1_state=None, extra_blocks=None):
    """model: a key of zj_common.MODELS.  Reused models are the Stage-2 networks; M2 / M3 the joint model (extra_blocks overrides the default)."""
    c = Z.MODELS[model]
    if c["reuse"] is not None or c["arch"] == "B0":
        return S2M.build(c["arch"], data, stage1_state=stage1_state if c["arch"] != "B0" else None)
    return JointModel(data.d_traj(), data.d_static(), data.d_geo(), data.z_mu, data.z_sd, stage1_state=stage1_state,
                      extra_blocks=Z.DECODER["extra_blocks"] if extra_blocks is None else extra_blocks)


def load(ds, model, data, dev, which="best", seed=Z.SEED, key=None, tag=None):
    """Network + checkpoint dict of a model of this study (reused Stage-2 checkpoints included).  key: state entry to load
    (default "state" = the selected state; "state_total" = the state selected by the full objective, M2 / M3 only)."""
    path = (Z.model_ckpt(ds, model, seed) if which == "best" else Z.ckpt_path(ds, Z.run_name(model, seed), which)) if tag is None else Z.ckpt_path(ds, Z.run_name(model, seed, tag), which)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = build(model, data, extra_blocks=ck.get("cfg", {}).get("extra_blocks")).to(dev)
    net.load_state_dict(ck[key or "state"]); net.eval()
    return net, ck


def n_params(m):
    return sum(p.numel() for p in m.parameters())
