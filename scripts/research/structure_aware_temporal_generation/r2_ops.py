"""R2 operators on the 512-point canonical grid (torch, batched over sequences and frames).

soft_r2    the DIFFERENTIABLE training operator on a dense map C (raw canonical contact values >= 0):
             m_k = sum_v M(v, k) alpha_v C(v)                                       amount (cell-area weighted contact)
             c_k = sum_v M(v, k) sigma((C(v) - c_hard) / tau_c)                     soft count of hard-contact points
             a_k = sigma((c_k - (n_min - 1/2)) / tau_n)                             soft participation (logit returned)
             p_k = sum_v omega_vk x_v / sum_v omega_vk,  omega_vk = M(v, k) sigma((C(v) - c_hard) / tau_c)
             n_k = normalise(sum_v omega_vk nh_v)                                   mean toward-hand normal (nh: the frame's
                                                                                    toward-hand normal field of the mask cache)
hard_r2    the EXACT-RULE grid extraction used for the metrics (no gradient): hard points C(v) >= c_hard with the GT part
           label of the point (argmax_k M), participation = causal 3-frame majority of [count >= n_min], amount = beta x the
           cell-area weighted contact, centroid / normal = the unweighted mean over the hard points of the part (the top-n_min
           points by contact value when the part is active by the majority but has no hard point in this frame); inactive
           parts carry zeros exactly like the exact R2 of the feature cache.
wrench_support   the wrench report's support-function capacity for one patch per active part (8-edge friction cone, mu =
           0.5, one unit of normal force per part) over the shared 76 directions; equals wrench_lp.capacity(..., strict=False)
           for one patch per finger (checked in sanity.py).
c_hard, n_min, beta, tau_c, tau_n are fitted / fixed ONCE per dataset (calibrate.py) and shared by every model.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

import sat_common as S


class CanonGeometry:
    """Per-dataset canonical geometry on the device: positions / outward normals of the 512 canonical points of every
    object, split into the articulated top part (ARCTIC) and the fixed part, the cell areas, the mesh centroid and l."""
    def __init__(self, ds, device, O_raw):
        g = S.load_geometry(ds)
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
        self.X_top, self.X_bot, self.N_top, self.N_bot = f(g["X_top"]), f(g["X_bot"]), f(g["N_top"]), f(g["N_bot"])
        self.alpha, self.valid = f(g["alpha"]), torch.from_numpy(g["valid"]).to(device)
        self.centroid, self.length = f(g["centroid"]), f(g["length"])
        self.geo_index = torch.from_numpy(g["geo_index"]).long().to(device)
        self.articulated = ds == "arctic"
        self.arti = f(O_raw[:, :, 0]) if self.articulated else None          # (N, 80) articulation [rad] for frames -8 .. 71
        self.device = device

    def frame(self, n, t):
        """n (B,) example indices, t (B, T') sequence frames in [0, 64) -> x (B, T', 512, 3) positions centred at the mesh
        centroid / l (the top part articulated per frame on ARCTIC), alpha (B, 512), valid (B, 512)."""
        g = self.geo_index[n]; B, Tp = t.shape
        Xt, Xb = self.X_top[g], self.X_bot[g]
        if self.articulated:
            th = -self.arti[n[:, None], t + S.PAD]                               # rotvec (0, 0, -arti) on the top part
            c, s = torch.cos(th)[..., None, None], torch.sin(th)[..., None, None]
            v = Xt[:, None]
            X = torch.cat([c * v[..., :1] - s * v[..., 1:2], s * v[..., :1] + c * v[..., 1:2], v[..., 2:].expand(B, Tp, 512, 1)], -1) + Xb[:, None]
        else:
            X = (Xt + Xb)[:, None].expand(B, Tp, 512, 3)
        x = (X - self.centroid[g][:, None, None]) / self.length[g][:, None, None, None]
        return x, self.alpha[g], self.valid[g]


def causal_majority(flag, prev=None):
    """flag (B, T, 6) bool/float -> majority of frames {t-2, t-1, t} (prev (B, 2, 6) = the two frames before t = 0, or the
    first frame repeated), float."""
    f = flag.float()
    if prev is None:
        prev = f[:, :1].expand(-1, 2, -1)
    ff = torch.cat([prev.float(), f], 1)
    return ((ff[:, 2:] + ff[:, 1:-1] + ff[:, :-2]) >= 2).float()


def soft_r2(C, M, nh, x, alpha, cal):
    """C (B, T', 512) raw contact, M (B, T', 512, 6) float part weights, nh (B, T', 512, 3) toward-hand unit normals of the
    frame, x (B, T', 512, 3) positions, alpha (B, 512); cal: dict with c_hard, n_min.
    Returns a_logit, m (B, T', 6), p, n (B, T', 6, 3), ws (B, T', 6)."""
    w = C.clamp(min=0)
    m = torch.einsum("btv,btvk,bv->btk", w, M, alpha)
    soft_hard = torch.sigmoid((w - cal["c_hard"]) / S.TAU_C)
    cnt = torch.einsum("btv,btvk->btk", soft_hard, M)
    a_logit = (cnt - (cal["n_min"] - 0.5)) / S.TAU_N
    om = soft_hard[..., None] * M
    ws = om.sum(2)
    p = torch.einsum("btvk,btvd->btkd", om, x) / (ws[..., None] + 1e-6)
    nv = torch.einsum("btvk,btvd->btkd", om, nh)
    n = nv / (nv.norm(dim=-1, keepdim=True) + 1e-6)
    return dict(a_logit=a_logit, m=m, p=p, n=n, ws=ws, cnt=cnt)


@torch.no_grad()
def hard_r2(C, lab, nh, x, alpha, cal, prev_flag=None):
    """Exact-rule grid extraction. C (B, T, 512) raw, lab (B, T, 512) long part label of every point, the rest as in
    soft_r2; prev_flag (B, 2, 6) the [count >= n_min] flags of the two frames before the first one (None: first frame
    repeated). Returns a, m (B, T, 6), p, n (B, T, 6, 3) with inactive parts zeroed, cnt, n_hard_pts (B, T, 6)."""
    w = C.clamp(min=0)
    hard = (w >= cal["c_hard"]).float()
    oneh = F.one_hot(lab, S.N_PARTS).float()
    cnt = torch.einsum("btv,btvk->btk", hard, oneh)
    flag = cnt >= cal["n_min"]
    a = causal_majority(flag, prev_flag)
    m = cal["beta"] * torch.einsum("btv,btvk,bv->btk", w, oneh, alpha)
    sel = hard[..., None] * oneh                                               # (B, T, 512, 6)
    ns = sel.sum(2)
    p = torch.einsum("btvk,btvd->btkd", sel, x) / ns[..., None].clamp(min=1)
    nv = torch.einsum("btvk,btvd->btkd", sel, nh)
    # fallback for active parts without a hard point in this frame: the top-n_min points of the part by contact value
    score = (w[..., None] * oneh).permute(0, 1, 3, 2)                          # (B, T, 6, 512)
    top = score.topk(int(cal["n_min"]), dim=-1).indices                        # (B, T, 6, n_min)
    xg = torch.gather(x[:, :, None].expand(-1, -1, S.N_PARTS, -1, -1), 3, top[..., None].expand(-1, -1, -1, -1, 3))
    ng = torch.gather(nh[:, :, None].expand(-1, -1, S.N_PARTS, -1, -1), 3, top[..., None].expand(-1, -1, -1, -1, 3))
    p_fb, n_fb = xg.mean(3), ng.sum(3)
    use_fb = (ns == 0)[..., None]
    p = torch.where(use_fb, p_fb, p); nv = torch.where(use_fb, n_fb, nv)
    n = nv / (nv.norm(dim=-1, keepdim=True) + 1e-9)
    return dict(a=a, m=m, p=p * a[..., None], n=n * a[..., None], cnt=cnt, n_hard_pts=ns, flag=flag.float())


def wrench_support(p, n, act, U, mu=S.MU, n_edges=S.N_CONE):
    """One friction-cone patch per active part: p (..., 6, 3) centroids / l relative to the mesh centroid, n (..., 6, 3)
    unit normals toward the hand, act (..., 6) in {0, 1}, U (76, 6) unit directions -> h(u) (..., 76)."""
    ex = torch.tensor([1.0, 0, 0], device=p.device); ey = torch.tensor([0, 1.0, 0], device=p.device)
    a = torch.where((n[..., :1].abs() < 0.9), ex, ey)
    t1 = torch.cross(n, a, dim=-1); t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9); t2 = torch.cross(n, t1, dim=-1)
    th = 2 * math.pi * torch.arange(n_edges, device=p.device, dtype=p.dtype) / n_edges
    E = -n[..., None, :] + mu * (torch.cos(th)[:, None] * t1[..., None, :] + torch.sin(th)[:, None] * t2[..., None, :])   # (..., 6, 8, 3)
    tau = torch.cross(p[..., None, :].expand_as(E), E, dim=-1)
    rows = torch.cat([E, tau], -1)                                                                                  # (..., 6, 8, 6)
    Sv = rows @ U.T                                                                                                 # (..., 6, 8, 76)
    return (Sv.max(-2).values.clamp(min=0) * act[..., None]).sum(-2)


def r2_vector(a, m, p, n):
    """[a | m | p | n] -> (..., 48)."""
    return torch.cat([a, m, p.flatten(-2), n.flatten(-2)], -1)


def angle_deg(n1, n2):
    cos = (n1 * n2).sum(-1) / (n1.norm(dim=-1) * n2.norm(dim=-1) + 1e-9)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1)))
