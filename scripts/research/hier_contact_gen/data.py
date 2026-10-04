"""Fold data for the hierarchical study: normalisation fitted on the training examples of the
fold, tensors on the device, and the (static, trajectory, contact) views every model consumes.

  contact     C~ = (X - mu) / s      mu per canonical point (train mean), s ONE scalar (train std
                                     of X - mu over all points) so raw L2 = s * standardised L2
  object      O~ = clip((O - mu_O) / sd_O, +-6)   per dim, sd floored at 1e-3 (as in exp3)
  static      [nearest/0.3, coverage, z(radius), z(extent), onehot(hand), onehot(role)]   1032-D
  context     tau_local,t = O~[t + LOCAL_OFFSETS] flattened (9 states), t in [0, T)
"""
from __future__ import annotations

import json

import numpy as np
import torch

import hc_common as H


class FoldData:
    def __init__(self, split, fold, device, stats=None):
        z, meta = H.load_sequences()
        man = json.load(open(H.OUT / "manifests" / f"{split}{fold}.json"))
        self.split, self.fold, self.device, self.meta = split, fold, device, meta
        self.idx = {k: np.asarray(man.get(k, []), int) for k in ("train", "val", "test", "test_extra")}
        X = z["C"].astype(np.float32)                                  # (N, T, 512)
        O = z["O"].astype(np.float32)                                  # (N, T+16, D)
        G = np.concatenate([z["G_nearest"] / 0.3, z["G_coverage"], z["G_radius"][:, None], z["G_extent"]], 1).astype(np.float32)
        S = np.concatenate([G[z["g_index"]], np.eye(2, dtype=np.float32)[z["hand"]], np.eye(2, dtype=np.float32)[z["role"]]], 1)
        tr = self.idx["train"]
        if stats is None:
            mu = X[tr].reshape(-1, 512).mean(0)
            s = float((X[tr] - mu).std())
            muO = O[tr].reshape(-1, O.shape[-1]).mean(0); sdO = O[tr].reshape(-1, O.shape[-1]).std(0); sdO = np.where(sdO < 1e-3, 1.0, sdO)
            muS = S[tr].mean(0); sdS = S[tr].std(0); sdS = np.where(sdS < 1e-3, 1.0, sdS)
            # only radius / extent are z-scored; nearest, coverage and the flags keep their scale
            keep = np.ones(S.shape[1], bool); keep[1024:1028] = False
            muS[keep] = 0.0; sdS[keep] = 1.0
            stats = dict(mu=mu, s=s, muO=muO, sdO=sdO, muS=muS, sdS=sdS)
        self.stats = stats
        self.N, self.T, self.K = X.shape
        self.D = O.shape[-1]
        f32 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.C = f32((X - stats["mu"]) / stats["s"])
        self.O = f32(np.clip((O - stats["muO"]) / stats["sdO"], -6, 6))
        self.S = f32((S - stats["muS"]) / stats["sdS"])
        self.offsets = torch.tensor(H.LOCAL_OFFSETS, device=device) + H.CTX_PAD
        self.X_raw = X
        self.canonical = {str(c): p for c, p in zip(z["canonical_cats"], z["canonical_points"])}

    # ----------------------------------------------------------------- views
    def context(self, n, t):
        """tau_local for example indices n (tensor) and frame indices t (tensor, same shape) -> (B, 9*D)."""
        idx = t[:, None] + self.offsets[None, :]                        # (B, 9), already padded by CTX_PAD
        return self.O[n[:, None], idx].reshape(len(n), -1)

    def to_raw(self, Ct):
        """standardised (…,512) tensor/array -> raw contact array."""
        Ct = Ct.detach().cpu().numpy() if torch.is_tensor(Ct) else Ct
        return Ct * self.stats["s"] + self.stats["mu"]

    def frame_pairs(self, part):
        """All (n, t) with t in [0, T-2] for the vector field, as flat index tensors."""
        n = torch.from_numpy(self.idx[part]).to(self.device)
        nn_ = n.repeat_interleave(self.T - 1); tt = torch.arange(self.T - 1, device=self.device).repeat(len(n))
        return nn_, tt

    def frames(self, part):
        n = torch.from_numpy(self.idx[part]).to(self.device)
        return n.repeat_interleave(self.T), torch.arange(self.T, device=self.device).repeat(len(n))
