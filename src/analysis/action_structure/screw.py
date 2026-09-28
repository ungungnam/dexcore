"""Does the tool-target pair behave like a VIRTUAL JOINT, and does the joint tell you the action?

WHY THIS FILE EXISTS. `stats.py` measures axis stability as the mean resultant length of the
increment rotation vectors -- and that measure is SIGN-SENSITIVE, so a motion that runs back and
forth along one perfectly fixed axis scores near zero, the same as random tumbling. Stirring,
brushing and screwing are all back-and-forth, so the low numbers there say nothing about whether a
joint exists. Everything here is sign-invariant, which is the only way to ask the question.

THE MEASUREMENTS, all per 64-frame chunk, all on the relative trajectory Y(t) = inv(X_O) X_T:

    axis_concentration   largest eigenvalue of sum_t w_t a_t a_t^T, weighted by rotation magnitude.
                         1.0 = every increment turned about ONE line (either direction); 1/3 =
                         isotropic. The outer product is what makes a reversal count as agreement.
    dir_concentration    the same construction on translation directions -- a prismatic joint.
    pitch_m_per_rad      translation ALONG the dominant axis per radian turned. 0 = revolute,
                         large = prismatic/helical.
    axial_fraction       share of translation that lies along the axis rather than across it.
    reversal_fraction    how often the motion changes sign along the dominant axis. This is what
                         `stats.rel_axis_stability` was accidentally measuring.
    net_*                the single screw taking Y(0) to Y(63): one angle, one pitch.

AXIS LOCATION IS DELIBERATELY NOT MEASURED. Per-increment rotations here average ~0.04 rad, and the
distance to a screw axis goes as cot(theta/2) -- at that scale the location is dominated by
annotation noise. Reporting it would be reporting nothing.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.analysis.action_structure import representations as REP
from src.analysis.action_structure.chunks import ChunkSet

EPS = 1e-9


def _concentration(v: np.ndarray, weights: np.ndarray):
    """Sign-invariant directional concentration of (N,T,3) vectors.

    Returns (lambda1, dominant_axis). Builds the weighted scatter of the UNIT directions as outer
    products, so `a` and `-a` reinforce instead of cancelling, and takes its largest eigenvalue:
    1.0 when every direction lies on one line, 1/3 when they are isotropic.
    """
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    u = np.where(n > EPS, v / np.maximum(n, EPS), 0.0)          # (N,T,3) unit directions
    w = np.asarray(weights, dtype=np.float64) * (n[..., 0] > EPS)   # (N,T)
    M = np.einsum("nt,nti,ntj->nij", w, u, u)
    M = M / np.maximum(w.sum(axis=1)[:, None, None], EPS)
    vals, vecs = np.linalg.eigh(M)                       # ascending
    return vals[:, -1], vecs[:, :, -1]


def _net_screw(p0, R0, p1, R1):
    """The single screw displacement from pose 0 to pose 1: angle and pitch (metres per radian)."""
    dR = np.einsum("nji,njk->nik", R0, R1)
    dp = np.einsum("nji,nj->ni", R0, p1 - p0)
    w = REP.so3_log(dR)
    theta = np.linalg.norm(w, axis=-1)
    axis = np.where(theta[:, None] > EPS, w / np.maximum(theta, EPS)[:, None], 0.0)
    along = (axis * dp).sum(axis=-1)
    pitch = np.where(theta > 1e-3, along / np.maximum(theta, EPS), np.nan)
    return theta, pitch, np.linalg.norm(dp, axis=-1)


def chunk_screw_features(cs: ChunkSet) -> Dict[str, np.ndarray]:
    """Every virtual-joint statistic, one value per chunk."""
    p_Y, R_Y = REP.relative_pose(cs)
    w_rot, dp = REP.increments(p_Y, R_Y)                 # (N,63,3) each
    theta = np.linalg.norm(w_rot, axis=-1)
    step = np.linalg.norm(dp, axis=-1)

    lam_rot, axis = _concentration(w_rot, theta)         # weight by how much it turned
    lam_lin, ldir = _concentration(dp, step)             # weight by how far it moved

    # signed rotation about the dominant axis, and how often it reverses
    proj = np.einsum("nti,ni->nt", w_rot, axis)
    sign = np.sign(proj)
    moving = theta > 1e-4
    flips = ((sign[:, 1:] * sign[:, :-1]) < 0) & moving[:, 1:] & moving[:, :-1]
    reversal = flips.sum(axis=1) / np.maximum((moving[:, 1:] & moving[:, :-1]).sum(axis=1), 1)

    # translation resolved along vs across the rotation axis
    along = np.abs(np.einsum("nti,ni->nt", dp, axis))
    across = np.sqrt(np.maximum(step ** 2 - along ** 2, 0.0))
    tot_theta = theta.sum(axis=1)
    pitch = np.where(tot_theta > 1e-3, along.sum(axis=1) / np.maximum(tot_theta, EPS), np.nan)
    axial = along.sum(axis=1) / np.maximum(along.sum(axis=1) + across.sum(axis=1), EPS)

    net_theta, net_pitch, net_disp = _net_screw(p_Y[:, 0], R_Y[:, 0], p_Y[:, -1], R_Y[:, -1])

    return {
        "screw_axis_concentration": lam_rot,
        "screw_dir_concentration": lam_lin,
        "screw_reversal_fraction": reversal,
        "screw_pitch_m_per_rad": pitch,
        "screw_axial_fraction": axial,
        "screw_axis_align_lin": np.abs((axis * ldir).sum(axis=-1)),   # is sliding along the axis?
        "screw_net_angle_rad": net_theta,
        "screw_net_pitch": net_pitch,
        "screw_net_disp_m": net_disp,
        # the dominant axis in the TARGET's frame, sign-invariantly: |components|. Carries some
        # target-mesh identity (the mesh frame's own orientation), so read it with that in mind.
        "screw_axis_absx": np.abs(axis[:, 0]), "screw_axis_absy": np.abs(axis[:, 1]),
        "screw_axis_absz": np.abs(axis[:, 2]),
    }


def features_frame(cs: ChunkSet):
    """Metadata plus every screw statistic, one row per chunk."""
    import pandas as pd
    cols = chunk_screw_features(cs)
    return pd.concat([cs.meta.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


SCREW_COLUMNS = ("screw_axis_concentration", "screw_dir_concentration", "screw_reversal_fraction",
                 "screw_pitch_m_per_rad", "screw_axial_fraction", "screw_axis_align_lin",
                 "screw_net_angle_rad", "screw_net_pitch", "screw_net_disp_m",
                 "screw_axis_absx", "screw_axis_absy", "screw_axis_absz")
