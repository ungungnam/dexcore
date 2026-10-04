#!/usr/bin/env python
"""Per-frame object state for every TACO sequence, from the raw 4x4 poses (objects only; no hand
pickles are opened). Everything is expressed in the CANONICAL frame of the BimArt port, i.e. the
target's frame, so the features are exactly the information a contact model conditioned on the
object trajectory could see:

  0: 3   tool position in the target frame                 R_T^T (p_tool - p_T)          [m]
  3: 9   tool rotation in the target frame, 6D             first two columns of R_T^T R_tool
  9: 15  target world rotation, 6D                         first two columns of R_T
 15: 18  target world translation relative to frame 0                                    [m]
 18: 21  tool linear velocity in the target frame                                        [m/frame]
 21: 24  tool angular velocity in the target frame (rotvec of R_rel(t)^T R_rel(t+1))     [rad/frame]
 24: 27  target world linear velocity                                                    [m/frame]
 27: 30  target world angular velocity                                                   [rad/frame]
Central differences; first/last frame use one-sided differences.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import dc_common as C
from src.analysis.loaders import taco

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("obj_states")
NAMES = ["tool_pos_x", "tool_pos_y", "tool_pos_z"] + [f"tool_rot6d_{i}" for i in range(6)] + \
        [f"targ_rot6d_{i}" for i in range(6)] + ["targ_rel_x", "targ_rel_y", "targ_rel_z"] + \
        ["tool_vel_x", "tool_vel_y", "tool_vel_z", "tool_angvel_x", "tool_angvel_y", "tool_angvel_z",
         "targ_vel_x", "targ_vel_y", "targ_vel_z", "targ_angvel_x", "targ_angvel_y", "targ_angvel_z"]


def angvel(R):
    """(T,3,3) -> (T,3) rotvec of R_t^T R_{t+1}, central differences (rad/frame)."""
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])           # R_t^T R_{t+1}
    w = Rotation.from_matrix(rel).as_rotvec()
    out = np.zeros((len(R), 3))
    out[1:-1] = 0.5 * (w[:-1] + w[1:])
    out[0], out[-1] = w[0], w[-1]
    return out


def vel(p):
    out = np.zeros_like(p)
    out[1:-1] = 0.5 * (p[2:] - p[:-2])
    out[0], out[-1] = p[1] - p[0], p[-1] - p[-2]
    return out


def main():
    si = pd.read_csv(C.SEQ / "sequence_index.csv")
    refs = {r.sequence_id: r for r in taco.index(C.TACO_ROOT)}
    states, lens = {}, {}
    for sid, n_frames in zip(si.sequence_id, si.n_frames):
        traj = taco.load(refs[sid], root=C.TACO_ROOT, with_hands=False)
        tool, targ = traj.tool, traj.target
        from src import geometry as G
        R_tool, p_tool = G.wxyz_to_R(tool.quat), tool.pos
        R_T, p_T = G.wxyz_to_R(targ.quat), targ.pos
        T = min(len(R_tool), len(R_T), int(n_frames))
        R_tool, p_tool, R_T, p_T = R_tool[:T], p_tool[:T], R_T[:T], p_T[:T]
        p_rel = np.einsum("tji,tj->ti", R_T, p_tool - p_T)
        R_rel = np.einsum("tji,tjk->tik", R_T, R_tool)
        f = np.concatenate([
            p_rel, R_rel[:, :, :2].reshape(T, 6), R_T[:, :, :2].reshape(T, 6), p_T - p_T[0],
            vel(p_rel), angvel(R_rel), vel(p_T), angvel(R_T)], axis=1).astype(np.float32)
        states[sid] = f
        lens[sid] = T
    ids = list(states)
    off = np.cumsum([0] + [lens[i] for i in ids])
    np.savez_compressed(C.CACHE / "object_states.npz", sequence_id=np.array(ids), offset=off,
                        states=np.concatenate([states[i] for i in ids]), names=np.array(NAMES))
    log.info("%d sequences, %d frames, %d dims -> cache/object_states.npz", len(ids), off[-1], len(NAMES))


if __name__ == "__main__":
    main()
