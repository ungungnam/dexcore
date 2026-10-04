"""Tensors shared by the two training scripts: the hierarchical study's FoldData (contact,
object states, static condition, normalisation fitted on the training examples) plus the hand
states, the causal / non-causal window gather, the shuffled-hand control and the (n, t) pair sets.
"""
from __future__ import annotations

import numpy as np
import torch

import hp_common as P


class HandTensors:
    """hand (N, 80, 300) metres -> standardised absolute states and standardised one-frame
    differences (both with ONE scalar scale fitted on the training sequences, so geometry is kept)."""
    def __init__(self, ds, fd, device):
        z = P.load_hand_cache(ds)
        H = z["hand"].reshape(len(z["hand"]), P.T + 2 * P.PAD, P.HAND_DIM).astype(np.float32)
        tr = fd.idx["train"]
        mu = H[tr].reshape(-1, P.HAND_DIM).mean(0); sd = float((H[tr] - mu).std())
        Dh = np.diff(H, axis=1)
        rms = float(np.sqrt((Dh[tr] ** 2).mean()))
        self.abs = torch.from_numpy((H - mu) / sd).to(device)                 # (N, 80, 300)
        self.rel = torch.from_numpy(Dh / rms).to(device)                      # (N, 79, 300): rel[j] = H[j+1] - H[j]
        self.stats = dict(mu=mu.tolist(), sd=sd, rms=rms)
        self.device = device

    def window(self, cond, n, t):
        """(B, d_hand) hand input of condition `cond` for pairs (n, t), or None for F0."""
        c = P.CONDITIONS[cond]
        if c["hand"] is None:
            return None
        offs = torch.tensor(P.window_offsets(cond), device=self.device) + P.PAD
        idx = t[:, None] + offs[None]
        src = self.abs if c["hand"] == "abs" else self.rel
        return src[n[:, None], idx].reshape(len(n), -1)


def shuffle_map(fd, seed):
    """Sequence index -> a random other sequence of the same split part (the hand input of the
    F2shuf control): same dimensionality and statistics, no information about this sequence."""
    rng = np.random.default_rng(1000 + seed)
    perm = np.arange(fd.N)
    for part in ("train", "val", "test", "test_extra"):
        idx = fd.idx[part]
        if len(idx) > 1:
            p = rng.permutation(idx)
            while (p == idx).any():                                  # a derangement
                p = rng.permutation(idx)
            perm[idx] = p
    return torch.from_numpy(perm).to(fd.device)


def hand_input(ht, cond, n, t, perm=None):
    if P.CONDITIONS[cond]["shuffle"]:
        n = perm[n]
    return ht.window(cond, n, t)


def pairs(fd, part):
    """All (n, t), t in [0, T-2], of a part as flat index tensors (t = 62 has only h = 1 valid)."""
    return fd.frame_pairs(part)


def delta_targets(fd, n, t):
    """(B, n_h, 512) standardised Delta_h C_t and (B, n_h) validity masks (t + h <= T - 1)."""
    tg, mk = [], []
    for h in P.HORIZONS:
        th = torch.clamp(t + h, max=fd.T - 1)
        tg.append(fd.C[n, th] - fd.C[n, t]); mk.append((t + h <= fd.T - 1).float())
    return torch.stack(tg, 1), torch.stack(mk, 1)


def classifier_input(fd, ht, cond, n, t, perm=None):
    parts = [fd.C[n, t], fd.S[n], fd.context(n, t)]
    hw = hand_input(ht, cond, n, t, perm)
    if hw is not None:
        parts.append(hw)
    return torch.cat(parts, 1)
