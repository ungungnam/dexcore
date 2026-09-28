"""Part 3: descriptive statistics of each 64-frame chunk, for tool, target and relative motion.

These are the "summary statistics only" representation as well as the descriptive tables -- a
deliberately low-dimensional view, so that whatever structure the 1152-dimensional trajectory
representations show can be compared against what a few dozen scalars already capture.

Magnitudes are NOT normalised per chunk: path length in metres and rotation path in radians are
the quantities of interest, not nuisances.

TWO STABILITY MEASURES, both mean resultant lengths (the circular-statistics R-bar): take the unit
direction of every increment, average the unit vectors, take the norm. 1.0 means every increment
pointed the same way -- a straight slide, or a rotation about one fixed axis. Near 0 means the
directions cancel, as they do for an oscillation or for noise. It is scale-free by construction,
which is what makes it comparable between a 2 cm wrist motion and a 40 cm carry.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from src.analysis.action_structure import representations as REP
from src.analysis.action_structure.chunks import ChunkSet


def _resultant(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """(N,T,3) increments -> (N,) mean resultant length of their unit directions."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    good = n[..., 0] > eps
    u = np.where(n > eps, v / np.maximum(n, eps), 0.0)
    cnt = good.sum(axis=1)
    out = np.linalg.norm(u.sum(axis=1), axis=-1) / np.maximum(cnt, 1)
    return np.where(cnt > 0, out, np.nan)


def _body_increments(p: np.ndarray, R: np.ndarray):
    """World-frame linear increments and body-frame angular increments of a pose sequence."""
    dp = np.diff(p, axis=1)                                            # (N,T-1,3)
    dR = np.einsum("ntji,ntjk->ntik", R[:, :-1], R[:, 1:])
    return dp, REP.so3_log(dR)


def _motion_stats(p: np.ndarray, R: np.ndarray, prefix: str) -> Dict[str, np.ndarray]:
    dp, dw = _body_increments(p, R)
    lin, ang = np.linalg.norm(dp, axis=-1), np.linalg.norm(dw, axis=-1)
    acc_lin = np.linalg.norm(np.diff(dp, axis=1), axis=-1)
    acc_ang = np.linalg.norm(np.diff(dw, axis=1), axis=-1)
    net_R = np.einsum("nji,njk->nik", R[:, 0], R[:, -1])
    return {
        f"{prefix}_trans_path_m": lin.sum(axis=1),
        f"{prefix}_rot_path_rad": ang.sum(axis=1),
        f"{prefix}_trans_disp_m": np.linalg.norm(p[:, -1] - p[:, 0], axis=-1),
        f"{prefix}_rot_disp_rad": np.linalg.norm(REP.so3_log(net_R), axis=-1),
        f"{prefix}_lin_inc_mean": lin.mean(axis=1), f"{prefix}_lin_inc_max": lin.max(axis=1),
        f"{prefix}_ang_inc_mean": ang.mean(axis=1), f"{prefix}_ang_inc_max": ang.max(axis=1),
        f"{prefix}_lin_acc_mean": acc_lin.mean(axis=1), f"{prefix}_lin_acc_max": acc_lin.max(axis=1),
        f"{prefix}_ang_acc_mean": acc_ang.mean(axis=1), f"{prefix}_ang_acc_max": acc_ang.max(axis=1),
        f"{prefix}_dir_stability": _resultant(dp),
        f"{prefix}_axis_stability": _resultant(dw),
    }


def _coupling(cs: ChunkSet) -> Dict[str, np.ndarray]:
    """How much the tool and the target move together, in the world frame.

    Two views, because they answer different questions: the cosine is "do they go the same way",
    the speed correlation is "do they speed up and slow down together". A carried target scores
    high on both; a target held still scores NaN-ish on neither but near zero on both.
    """
    dT, dO = np.diff(cs.tool_p, axis=1), np.diff(cs.targ_p, axis=1)
    nT, nO = np.linalg.norm(dT, axis=-1), np.linalg.norm(dO, axis=-1)
    ok = (nT > 1e-9) & (nO > 1e-9)
    cos = np.where(ok, (dT * dO).sum(axis=-1) / np.maximum(nT * nO, 1e-12), np.nan)
    with np.errstate(invalid="ignore"):
        cos_mean = np.nanmean(np.where(ok, cos, np.nan), axis=1)
    a = nT - nT.mean(axis=1, keepdims=True)
    b = nO - nO.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    speed_corr = np.where(denom > 1e-12, (a * b).sum(axis=1) / np.maximum(denom, 1e-12), np.nan)
    return {"tool_target_dir_cos": cos_mean, "tool_target_speed_corr": speed_corr}


def _effective_rank(X: np.ndarray) -> Dict[str, np.ndarray]:
    """Effective PCA rank of each chunk's own (64,9) relative SE(3) trajectory.

    Two measures because they disagree usefully: the participation ratio
    `(sum lambda)^2 / sum lambda^2` is a smooth "how many directions carry the variance", while
    `n95` counts components needed for 95%. A pure translation along one line gives ~1; a rich
    multi-axis manipulation gives 4-6.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)                            # (N, min(64,9))
    lam = s ** 2
    tot = lam.sum(axis=1, keepdims=True)
    pr = np.where(tot[:, 0] > 0, (lam.sum(axis=1) ** 2) / np.maximum((lam ** 2).sum(axis=1), 1e-30),
                  np.nan)
    frac = np.cumsum(lam, axis=1) / np.maximum(tot, 1e-30)
    n95 = (frac < 0.95).sum(axis=1) + 1
    return {"rel_participation_ratio": pr, "rel_rank95": n95.astype(float)}


def chunk_statistics(cs: ChunkSet):
    """One DataFrame row per chunk: the metadata columns plus every statistic above."""
    import pandas as pd

    p_Y, R_Y = REP.relative_pose(cs)
    cols: Dict[str, np.ndarray] = {}
    cols.update(_motion_stats(cs.tool_p, cs.tool_R, "tool"))
    cols.update(_motion_stats(cs.targ_p, cs.targ_R, "target"))
    cols.update(_motion_stats(p_Y, R_Y, "rel"))
    cols.update(_coupling(cs))
    cols.update(_effective_rank(np.concatenate([p_Y, REP.rot6d(R_Y)], axis=-1)))
    return pd.concat([cs.meta.reset_index(drop=True), pd.DataFrame(cols)], axis=1)


#: The statistic columns, i.e. everything `chunk_statistics` adds to the metadata.
def statistic_columns(df) -> list:
    meta = {"chunk_id", "episode_id", "triplet", "action", "tool_cat", "target_cat",
            "tool_mesh", "target_mesh", "chunk_index", "n_chunks", "episode_frames"}
    return [c for c in df.columns if c not in meta]
