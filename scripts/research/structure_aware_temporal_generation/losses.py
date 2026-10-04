"""Losses.

L_dense     mean squared error of the standardised residual r_hat vs r (the persistence predictor r_hat = 0 scores ~1).
L_struct    (1/4) sum_c L_c / ref_c over c in {part, amount, centroid, normal}, computed with the differentiable grid operator
            (r2_ops.soft_r2) on the generated map in raw contact units (deterministic: the prediction; diffusion: the x0
            estimate). part: BCE of the soft participation logit against the EXACT GT participation; amount: L1 between the
            soft amounts of the generated and the GT map; centroid: Euclidean distance (units of l) between the soft centroids;
            normal: 1 - cos between the soft mean normals; centroid / normal averaged over the parts ACTIVE in the GT frame.
            ref_c = the value of L_c on the TRAIN set for the persistence predictor C_t = s_0 (train statistics; computed once
            per run, identical for every model of a dataset because it does not depend on the model).
L_aux       the previous study's structured loss on the hidden-state head: BCE (a) + MSE (z-scored m) + masked MSE (z-scored
            p, n over the active parts), each a block mean; target = the exact R2 of the frame.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import sat_common as S
from r2_ops import soft_r2

STRUCT_TERMS = ("part", "amount", "centroid", "normal")


def dense_loss(r_hat, r):
    return F.mse_loss(r_hat, r)


def structural_terms(C_hat, b, cal):
    """Unnormalised structural components of a batch (C_hat raw (B, T', 512))."""
    sh = soft_r2(C_hat, b["M"], b["nh"], b["x"], b["alpha"], cal)
    with torch.no_grad():
        sg = soft_r2(b["C"], b["M"], b["nh"], b["x"], b["alpha"], cal)
    act = b["a"]; na = act.sum().clamp(min=1)
    part = F.binary_cross_entropy_with_logits(sh["a_logit"], act)
    amount = (sh["m"] - sg["m"]).abs().mean()
    centroid = ((sh["p"] - sg["p"]).norm(dim=-1) * act).sum() / na
    normal = ((1 - (sh["n"] * sg["n"]).sum(-1)) * act).sum() / na
    return dict(part=part, amount=amount, centroid=centroid, normal=normal)


class StructuralLoss:
    def __init__(self, ref):
        self.ref = {k: float(v) for k, v in ref.items()}

    def __call__(self, C_hat, b, cal):
        terms = structural_terms(C_hat, b, cal)
        norm = {k: terms[k] / self.ref[k] for k in STRUCT_TERMS}
        return sum(norm.values()) / len(STRUCT_TERMS), {k: float(v) for k, v in norm.items()}


@torch.no_grad()
def persistence_reference(data, cal, bs=64, n_max=None):
    """ref_c of the four structural terms for C_t = s_0 over the training sequences."""
    tr = torch.from_numpy(data.idx["train"]).to(data.device)
    if n_max:
        tr = tr[:n_max]
    acc = {k: 0.0 for k in STRUCT_TERMS}; cnt = 0
    for i in range(0, len(tr), bs):
        n = tr[i:i + bs]; b = data.batch(n)
        C_hat = b["s0_raw"][:, None].expand(-1, S.TF, -1)
        t = structural_terms(C_hat, b, cal)
        for k in STRUCT_TERMS:
            acc[k] += float(t[k]) * len(n)
        cnt += len(n)
    return {k: v / cnt for k, v in acc.items()}


def aux_loss(aux, b, data):
    """aux (B, T', 48) = [a logits | z(m) | z(geom)] against the exact R2 of the frames."""
    a_log, m_z, g_z = aux[..., :6], aux[..., 6:12], aux[..., 12:]
    la = F.binary_cross_entropy_with_logits(a_log, b["a"])
    lm = F.mse_loss(m_z, data.standardise("m", b["m"]))
    mask = b["a"].repeat_interleave(3, -1).repeat(1, 1, 2)
    lg = (((g_z - data.standardise("geom", b["geom"])) ** 2) * mask).sum() / mask.sum().clamp(min=1)
    return la + lm + lg, dict(aux_a=float(la), aux_m=float(lm), aux_geom=float(lg))
