"""Tensors shared by the probes and the temporal predictors: the feature cache as standardised blocks (train
statistics only), the hierarchical study's FoldData (static condition G, trajectory context tau_local, split), the
causal window gather and the future targets.

Standardisation (fitted on the TRAIN sequences, frames t in [0, 64)):
  coarse blocks (part, amount, geom, topo, hand)   per-dimension z-score
  C, H                                             per-dimension mean, ONE scalar scale (the geometry of the vector is kept,
                                                   as in the previous diagnostic's FoldData / HandTensors)
Targets keep their raw units for the metrics; the losses use per-dimension z-scores of m and q, z-scores of the active
entries of geom, the scalar scales of C and H, and BCE for a.
"""
from __future__ import annotations

import numpy as np
import torch

import sv_common as S

GEOM_P = np.arange(18); GEOM_N = 18 + np.arange(18)          # geom block layout: [p (6 x 3) | n (6 x 3)]


def part_expand(mask6):
    """(B, 6) -> (B, 36) mask over the geom block."""
    return torch.cat([mask6.repeat_interleave(3, 1), mask6.repeat_interleave(3, 1)], 1)


class Data:
    def __init__(self, ds, device, need_hand100=True):
        self.ds, self.device = ds, device
        F = S.load_features(ds)
        hand100 = S.HP.load_hand_cache(ds)["hand"] if need_hand100 else None
        self.fd = S.HP.fold(ds, device)                                     # C~, S, O, context(), idx, stats
        self.idx = self.fd.idx; tr = self.idx["train"]
        self.meta = self.fd.meta
        used = np.concatenate([self.idx[k] for k in ("train", "val", "test")])
        assert F["included"][used].all(), "feature cache incomplete"
        self.included = F["included"]
        self.length = torch.from_numpy(F["length"].astype(np.float32)).to(device)
        blocks = S.block_arrays(F, hand100)
        self.raw = blocks                                                    # numpy, (N, NF, d)
        self.norm, self.std = {}, {}
        for b, X in blocks.items():
            mu, sd = S.fit_norm(X, tr)
            if b in ("C", "H"):
                sd = np.full_like(sd, float((X[tr][:, S.PAD:S.PAD + S.T] - mu).std()))
            self.norm[b] = (mu, sd)
            self.std[b] = torch.from_numpy(((X - mu) / sd).astype(np.float32)).to(device)
        # targets on the sequence frames t in [0, 64), raw units
        sl = slice(S.PAD, S.PAD + S.T)
        self.tg = {"a": blocks["part"][:, sl], "m": blocks["amount"][:, sl], "geom": blocks["geom"][:, sl], "q": F["q"],
                   "q_strict": F["q_strict"], "C": blocks["C"][:, sl]}
        if hand100 is not None:
            self.tg["H"] = blocks["H"][:, sl]
        self.tstats = {}
        for k in ("m", "q", "q_strict", "C", "H"):
            if k not in self.tg:
                continue
            x = self.tg[k][tr].reshape(-1, self.tg[k].shape[-1]).astype(np.float64)
            mu = x.mean(0); sd = x.std(0)
            if k in ("C", "H"):
                sd = np.full_like(sd, float((x - mu).std()))
            sd = np.where(sd < 1e-6, 1.0, sd)
            self.tstats[k] = (torch.tensor(mu, dtype=torch.float32, device=device), torch.tensor(sd, dtype=torch.float32, device=device))
        g = self.tg["geom"][tr].reshape(-1, 36).astype(np.float64); act = part_expand(torch.from_numpy(self.tg["a"][tr].reshape(-1, 6))).numpy() > 0
        mu = np.array([g[act[:, j], j].mean() if act[:, j].any() else 0.0 for j in range(36)]); sd = np.array([g[act[:, j], j].std() if act[:, j].any() else 1.0 for j in range(36)])
        sd = np.where(sd < 1e-6, 1.0, sd)
        self.tstats["geom"] = (torch.tensor(mu, dtype=torch.float32, device=device), torch.tensor(sd, dtype=torch.float32, device=device))
        self.tgt = {k: torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)).to(device) for k, v in self.tg.items()}
        self.offsets = torch.arange(-S.HIST + 1, 1, device=device) + S.PAD
        self.take = self.meta.take_key.astype(str).values

    # ---------------------------------------------------------------- inputs
    def window(self, rep, n, t):
        """(B, HIST * d) standardised representation frames t-7 .. t (t in [0, 64), stored index t + PAD)."""
        idx = t[:, None] + self.offsets[None]
        x = torch.cat([self.std[b][n[:, None], idx] for b in S.LADDER[rep]], -1)
        return x.reshape(len(n), -1)

    def frame(self, rep, n, t):
        """(B, d) standardised representation at frame t."""
        return torch.cat([self.std[b][n, t + S.PAD] for b in S.LADDER[rep]], -1)

    def cond(self, n, t):
        return self.fd.S[n], self.fd.context(n, t)

    def d_static(self):
        return self.fd.S.shape[1]

    def d_traj(self):
        return len(self.fd.offsets) * self.fd.D

    # ---------------------------------------------------------------- targets
    def targets(self, n, t, h, keys):
        """dict key -> raw target at t + h (clamped) and the validity mask (t + h <= 63)."""
        th = torch.clamp(t + h, max=S.T - 1); valid = (t + h <= S.T - 1)
        return {k: self.tgt[k][n, th] for k in keys}, valid

    def standardise(self, k, x):
        mu, sd = self.tstats[k]
        return (x - mu) / sd

    def destandardise(self, k, z):
        mu, sd = self.tstats[k]
        return z * sd + mu

    def pairs(self, part):
        return self.fd.frame_pairs(part)                                     # (n, t), t in [0, 62]

    def frames(self, part):
        return self.fd.frames(part)                                          # (n, t), t in [0, 63]


def assert_causal():
    offs = (torch.arange(-S.HIST + 1, 1)).tolist()
    assert max(offs) == 0 and min(offs) == -(S.HIST - 1), offs
