"""Networks of the diagnostic. The Delta-C predictor is the existing vector field's trunk
(scripts/research/hier_contact_gen/models.py: Trunk of FiLM residual blocks) with a multi-horizon
output; the condition encoder is the VF's static + trajectory encoders plus an optional hand
encoder of the same shape (d_hand -> 256 -> 256) summed into the same 256-D FiLM vector. The hand
encoder is the only thing that differs between F0 / F1 / F2 (its parameter count is reported).
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn

import hp_common as P

if str(P.HIER) not in sys.path:
    sys.path.insert(0, str(P.HIER))
from models import Trunk  # noqa: E402  (hier_contact_gen/models.py)


class Cond(nn.Module):
    def __init__(self, d_static, d_traj, d_hand=0, width=256):
        super().__init__()
        self.static = nn.Sequential(nn.Linear(d_static, 512), nn.SiLU(), nn.Linear(512, width))
        self.traj = nn.Sequential(nn.Linear(d_traj, 256), nn.SiLU(), nn.Linear(256, width))
        self.hand = nn.Sequential(nn.Linear(d_hand, 256), nn.SiLU(), nn.Linear(256, width)) if d_hand else None
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(width, width))

    def forward(self, s, traj, hand=None):
        c = self.static(s) + self.traj(traj)
        if self.hand is not None:
            c = c + self.hand(hand)
        return self.out(c)


class DeltaPredictor(nn.Module):
    """(C~_t, G, tau_local,t [, hand window]) -> Delta^_h C~_t for h in HORIZONS (one head each)."""
    def __init__(self, d_static, d_traj, d_hand=0, n_h=len(P.HORIZONS), width=1024, blocks=3, dropout=0.1):
        super().__init__()
        self.cond = Cond(d_static, d_traj, d_hand)
        self.net = Trunk(512, 512 * n_h, 256, width, blocks, dropout)
        self.n_h = n_h

    def forward(self, c_t, s, traj, hand=None):
        return self.net(c_t, self.cond(s, traj, hand)).view(len(c_t), self.n_h, 512)


class EventClassifier(nn.Module):
    """[C~_t, G, tau_local,t, hand window] -> logits of 'a persistent contact-mode transition starts
    within the next h frames' for h in EVENT_HORIZONS. The same small MLP for every condition."""
    def __init__(self, d_in, n_out=len(P.EVENT_HORIZONS), width=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width // 2), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(width // 2, n_out))

    def forward(self, x):
        return self.net(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
