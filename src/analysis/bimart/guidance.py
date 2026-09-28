"""Contact guidance: push each denoising step's hand prediction toward the predicted contact map.

This is the stage between sampling and MANO fitting in BimArt's inference, and skipping it changes
the result substantially -- the raw sampler has nothing tying the hand to the object, so it is free
to place a plausible-looking hand in the wrong place. Guidance closes that loop: at every denoising
step it reads off the contact the CURRENT hand prediction would actually produce, compares it with
the contact map stage one predicted, and steps the prediction down that gradient.

    actual   for each of the 1024 BPS-sampled object vertices, the distance to the nearest
             predicted hand keypoint
    target   stage one's contact map at those same vertices
    loss     MSE between them, per hand
    step     `x <- x - gradient / ||gradient|| * scale`, a normalised step so the magnitude does
             not depend on how large the loss happens to be

TWO DEPARTURES FROM UPSTREAM, both forced and neither behavioural:

    `knn_points` -> `cdist`   pytorch3d is not installed here, and with K=1 over 1024 object points
                              against 100 hand keypoints the two are the same computation. pytorch3d
                              returns SQUARED distances and upstream takes their square root, which
                              `cdist` gives directly.
    object vertices           upstream rebuilds them from ARCTIC's articulation state. TACO's are
                              rigid, so they are rebuilt the way `features.py` builds them -- tool
                              posed into the target frame, target fixed -- rather than stored, which
                              would add a (T, ~8000, 3) array to every archive.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

log = logging.getLogger(__name__)

HAND_KP = 100
#: Contact entries per hand: 512 BPS points x 2 objects, as `features.sequence_features` lays out.
PER_HAND = 1024


def nearest_hand_distance(obj_pts: torch.Tensor, hand_pts: torch.Tensor) -> torch.Tensor:
    """(T, N_obj) distance from each object vertex to its nearest hand keypoint.

    Upstream calls pytorch3d's `knn_points(A, B, K=1)` and square-roots the result; at K=1 that is
    exactly a pairwise distance followed by a min, which is what this does.
    """
    return torch.cdist(obj_pts, hand_pts).min(dim=-1).values


def unnormalize(x: torch.Tensor, stat) -> torch.Tensor:
    return x * stat["std"] + stat["mean"]


_CACHE: Dict[str, object] = {}


def _canonical_objects(root, sequence_id: str) -> np.ndarray:
    """(T, N, 3) both objects' METRIC vertices in the canonical (target) frame, tool first.

    Rebuilt from the dataset rather than stored: it is the same construction `features.py` performs,
    and duplicating a (T, ~8000, 3) array into every archive would cost far more than recomputing
    it for the few windows an evaluation samples.
    """
    from src.analysis.bimart import features as F, meshes as M
    from src.analysis.loaders import taco

    if "meshes" not in _CACHE:
        _CACHE["meshes"] = M.load(Path(root) / "assets" / "taco_mesh_dict.npy")
        _CACHE["refs"] = {r.sequence_id: r for r in taco.index()}
    hit = _CACHE.get(sequence_id)
    if hit is not None:
        return hit

    from src import geometry as G

    traj = taco.load(_CACHE["refs"][sequence_id], with_hands=False)
    md = _CACHE["meshes"]
    mt, mo = md[traj.tool.name], md[traj.target.name]
    R_tool, p_tool = G.wxyz_to_R(traj.tool.quat), traj.tool.pos
    R_targ, p_targ = G.wxyz_to_R(traj.target.quat), traj.target.pos
    tool_world = np.einsum("tij,nj->tni", R_tool, mt["verts_original"]) + p_tool[:, None, :]
    tool_cano = F.to_canonical(tool_world, R_targ, p_targ)
    targ_cano = np.repeat(mo["verts_original"][None], len(tool_cano), axis=0)
    out = np.concatenate([tool_cano, targ_cano], axis=1)          # order matches features.py
    if len(_CACHE) > 40:                                          # bounded; evaluation is sampled
        for k in [k for k in _CACHE if k not in ("meshes", "refs")][:10]:
            _CACHE.pop(k)
    _CACHE[sequence_id] = out
    return out


def sparse_object_vertices(root, sequence_id: str, start: int, horizon: int,
                           device: str = "cuda") -> torch.Tensor:
    """(T, 1024, 3) canonical METRIC object vertices at this window's BPS indices.

    Metric, not the unit-sphere copy: the contact map is a distance in metres and so are the hand
    keypoints guidance compares against.
    """
    verts = _canonical_objects(root, sequence_id)
    z = np.load(Path(root) / "sequences" / f"{sequence_id.replace('/', '__')}.npz")
    sl = slice(start, start + horizon)
    inds = z["obj_cano_bps_inds"][sl]
    rows = np.arange(len(inds))[:, None]
    return torch.as_tensor(verts[sl][rows, inds], dtype=torch.float32, device=device)


class ContactGuidance:
    """One window's guidance state. `step` is called once per denoising step."""

    def __init__(self, sparse_obj_verts: torch.Tensor, contact_pred_norm: torch.Tensor,
                 stat: Dict, w_cm: float = 1.0, guidance_scale: float = 1.0):
        self.obj = sparse_obj_verts                                  # (B,T,1024,3) or (T,1024,3)
        if self.obj.ndim == 3:
            self.obj = self.obj[None]
        self.cm = unnormalize(contact_pred_norm, stat["contact_points"])   # (B,T,2048), metres
        self.stat = stat
        self.w_cm = w_cm
        self.guidance_scale = guidance_scale
        self.history = []

    def _hands(self, action_unnorm: torch.Tensor):
        kp = action_unnorm[..., : HAND_KP * 6]
        kp = kp.reshape(*kp.shape[:-1], HAND_KP * 2, 3)
        return kp[..., :HAND_KP, :], kp[..., HAND_KP:, :]

    def step(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """One guided step on the NORMALISED prediction, as upstream does."""
        with torch.enable_grad():
            x = pred_norm.detach().requires_grad_(True)
            unnorm = unnormalize(x, self.stat["action"])
            left, right = self._hands(unnorm)
            loss = 0.0
            for hand, target in ((left, self.cm[..., :PER_HAND]),
                                 (right, self.cm[..., PER_HAND:])):
                B, T = hand.shape[0], hand.shape[1]
                actual = nearest_hand_distance(self.obj.reshape(B * T, -1, 3),
                                               hand.reshape(B * T, HAND_KP, 3)).reshape(B, T, -1)
                loss = loss + torch.nn.functional.mse_loss(target, actual)
            loss = loss * self.w_cm
            grad = torch.autograd.grad(loss, x)[0]
            # Normalise PER SAMPLE, not over the whole batch. Upstream runs at batch size 1, where
            # the two are identical; batched, a single global norm makes each sample's step shrink
            # as 1/sqrt(B) and the guidance quietly stops doing anything.
            flat = grad.reshape(grad.shape[0], -1)
            scale = self.guidance_scale / (torch.linalg.norm(flat, dim=1) + 1e-7)
            step = grad * scale.reshape(-1, *([1] * (grad.ndim - 1)))
            self.history.append(float(loss.detach()))
            return (x - step).detach()
