"""The representation that actually separates actions once the objects are different.

WHAT IT IS: twenty scalars per chunk -- the relative tool-target motion and the virtual-joint
parameters -- compared by COSINE distance after a channel z-score. It scores 0.617 on the
geometry-controlled test against 0.497-0.522 for the 1152-dimensional trajectory representations,
on the same chunks, the same pairs and the same protocol.

HOW IT WAS ARRIVED AT, and what did NOT work, because the negative results are the informative part:

    cosine instead of euclidean      +0.033   the single biggest gain. Cosine drops the vector's
                                              magnitude, which is mostly "how much motion is in
                                              this window" -- an object and phase property.
    drop world-frame tool/target     +0.012   keeping only rel_* and screw_*.
    drop net-displacement + the      +0.006   first-to-last displacement barely varies by action
      sign-broken axis stability              (epsilon^2 0.05); `rel_axis_stability` reads near
                                              zero for ANY reversing motion, see `screw.py`.
    rank/quantile gaussianisation    -0.007   helped on the full data and did NOT survive a split
                                              by episode. Dropped. (This is why every choice here
                                              was re-checked on held-out episodes.)
    scale-free shape descriptors     -0.023   oscillation counts, profile skew, time symmetry,
                                              curvature. Seventeen of them, all worse.
    residualising out phase, motion  -0.008   removing the known confounds removed signal with them
      magnitude, or target motion       to     -0.034
    PCA whitening                    -0.083
    label-driven feature selection   -0.02 to  top-k by action/tool effect ratio, fitted on half
      (with an episode split)         -0.05    the episodes: did not generalise to the other half.

The drops were cross-validated the other way too: features chosen as harmful on one half of the
episodes improved the OTHER half, and both halves independently named the same two.

WHAT IT IS NOT. 0.617 is a long way from 0.5, and a long way from "actions cluster". Chunks of the
same action are closer than chunks of different actions about 62% of the time once tool and target
categories both differ; a representation that had really captured the action would be far higher.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

#: Sums and their means differ by a constant factor, so a euclidean or cosine distance counts the
#: same physical quantity twice. The mean is dropped and the sum kept.
REDUNDANT = ("tool_lin_inc_mean", "tool_ang_inc_mean", "target_lin_inc_mean",
             "target_ang_inc_mean", "rel_lin_inc_mean", "rel_ang_inc_mean",
             "screw_net_disp_m", "screw_net_angle_rad")

#: The winning set: relative motion plus virtual-joint parameters, minus the net-displacement
#: quantities and minus `rel_axis_stability`, which `screw_axis_concentration` supersedes.
JOINT_MOTION_COLUMNS = (
    "rel_trans_path_m", "rel_rot_path_rad", "rel_lin_inc_max", "rel_ang_inc_max",
    "rel_lin_acc_mean", "rel_lin_acc_max", "rel_ang_acc_mean", "rel_ang_acc_max",
    "rel_dir_stability", "rel_participation_ratio",
    "screw_axis_concentration", "screw_dir_concentration", "screw_reversal_fraction",
    "screw_pitch_m_per_rad", "screw_axial_fraction", "screw_axis_align_lin", "screw_net_pitch",
    "screw_axis_absx", "screw_axis_absy", "screw_axis_absz",
)

#: The distance this representation is defined with. Changing it changes the result by more than
#: any feature choice does, so it belongs with the feature list, not in a caller's default.
METRIC = "cosine"


def columns(available: Sequence[str], keep_world: bool = False) -> List[str]:
    """The representation's columns, intersected with what a statistics table actually has."""
    if keep_world:
        return [c for c in available if c not in REDUNDANT]
    return [c for c in JOINT_MOTION_COLUMNS if c in available]


def build(stats_df, cols: Optional[Sequence[str]] = None) -> np.ndarray:
    """(N, 20) matrix from a chunk-statistics table. NaNs take their column's median.

    A NaN here means a chunk in which something never moved, so there was no direction to measure;
    the column median is the honest filler and there are only a handful.
    """
    cols = list(cols or columns(list(stats_df.columns)))
    X = stats_df[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
