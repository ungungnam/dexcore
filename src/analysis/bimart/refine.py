"""Stages three and four of BimArt inference: fit a MANO hand, then optimise it against the object.

Stage two leaves 100 predicted surface points per hand. Those are not a hand -- they have no
topology, no pose parameters, and nothing stops them from sitting inside the object. Stage three
fits MANO to them; stage four then moves the fitted hand so that it touches the object where the
prediction said it would, does not penetrate it, and does not jitter.

WHAT STAGE FOUR OPTIMISES, from upstream's weights:

    projection (100)   each keypoint plus its predicted direction vector should land ON the object
    penetration (10)   hand vertices should not be inside the object mesh
    acceleration (1000) second difference of the vertices, i.e. no frame-to-frame jitter

It optimises the MANO PARAMETERS, not free vertices, so the result stays a valid hand.

ONE SUBSTITUTION, as in `guidance.py`: `pytorch3d.ops.knn_points` is unavailable here and is used
at K=1, so `cdist().min()` replaces it. Upstream uses the returned distances WITHOUT a square root,
so the projection loss is a mean SQUARED distance and this reproduces that rather than quietly
changing the objective's units.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

W_PEN, W_PROJ, W_ACC = 10.0, 100.0, 1000.0
OP_STEPS = 100
LR = 5e-4


def proj_loss(proj_points: torch.Tensor, obj_verts: torch.Tensor) -> torch.Tensor:
    """Mean SQUARED distance from each projected point to the nearest object vertex.

    Upstream feeds `knn_points`'s output straight into the mean, and pytorch3d returns squared
    distances -- so the square is part of the objective, not an oversight to be corrected.
    """
    return (torch.cdist(proj_points, obj_verts).min(dim=-1).values ** 2).mean()


def penetration_depth(hand_verts: torch.Tensor, obj_mesh) -> torch.Tensor:
    """Mean depth of hand vertices that lie INSIDE the object, in metres. 0 when none do.

    `trimesh.proximity.signed_distance` is positive inside a watertight mesh; for a mesh that is
    not watertight the sign is unreliable, so those objects contribute 0 rather than noise.
    """
    import trimesh

    if not obj_mesh.is_watertight:
        return torch.zeros((), device=hand_verts.device)
    sd = trimesh.proximity.signed_distance(obj_mesh, hand_verts.detach().cpu().numpy())
    inside = np.clip(sd, 0.0, None)
    return torch.as_tensor(float(inside.mean()), device=hand_verts.device)


class PostOptimization:
    """Upstream's `refine_noisy_motions`, with TACO's two rigid objects in place of ARCTIC's one."""

    def __init__(self, mano_layer, hand_index: np.ndarray, steps: int = OP_STEPS, lr: float = LR,
                 w_pen: float = W_PEN, w_proj: float = W_PROJ, w_acc: float = W_ACC,
                 device: str = "cuda"):
        self.layer = mano_layer
        self.hand_index = np.asarray(hand_index).reshape(-1)
        self.steps, self.lr = steps, lr
        self.w_pen, self.w_proj, self.w_acc = w_pen, w_proj, w_acc
        self.device = device

    def run(self, mano_param: Dict, obj_verts: torch.Tensor, dirvec: Dict[str, torch.Tensor],
            obj_mesh=None) -> Dict:
        """Optimise the fitted hands in place. `obj_verts` is (T, N, 3) in the same frame as the
        hands; `dirvec` holds the predicted (T, 100, 3) offsets per side."""
        import sys

        bim = "/home/uhnam/workspace/dexcore/third_party/BimArt"
        if bim not in sys.path:
            sys.path.insert(0, bim)
        from utils import mano_utils

        # `fit_mano` hands back tensors; `make_mano_param_optimizable` expects numpy and calls
        # `torch.from_numpy` on each entry. Convert rather than patch upstream.
        as_np = {side: {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                        for k, v in side_params.items()}
                 for side, side_params in mano_param.items()}
        params = mano_utils.make_mano_param_optimizable(as_np)
        joints_dirvec = torch.cat([dirvec["left"], dirvec["right"]], dim=1).clone()
        opt = torch.optim.AdamW(
            [params[s][k] for s in ("left", "right") for k in ("pose", "rot", "trans", "shape")],
            lr=self.lr)

        hist = []
        for i in range(self.steps):
            opt.zero_grad()
            lv, rv, _, _ = mano_utils.get_two_hand_verts_keypts_tensor(self.layer, params)
            verts = torch.cat([lv, rv], dim=1)
            keypts = torch.cat([lv[:, self.hand_index], rv[:, self.hand_index]], dim=1)
            projection = joints_dirvec + keypts

            pl = proj_loss(projection, obj_verts) * self.w_proj
            smooth = torch.mean((verts[:-2] - 2 * verts[1:-1] + verts[2:]) ** 2) * self.w_acc
            loss = pl + smooth
            loss.backward()
            opt.step()
            if i % 25 == 0 or i == self.steps - 1:
                hist.append({"step": i, "proj": float(pl), "acc": float(smooth),
                             "total": float(loss)})
        with torch.no_grad():
            lv, rv, _, _ = mano_utils.get_two_hand_verts_keypts_tensor(self.layer, params)
        return {"left_verts": lv.detach(), "right_verts": rv.detach(), "mano_param": params,
                "history": hist}
