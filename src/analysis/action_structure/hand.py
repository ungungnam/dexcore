"""The PREDICTION TARGET: hand trajectory. Never an input to any representation.

The study's real question is not "does this representation separate verbs" -- the verb was only ever
a proxy -- but "does this representation determine the hand motion". So the hand enters here and
only here, as the thing to be predicted, and no representation in `representations.py`,
`stats.py`, `screw.py` or `compact.py` can see it.

FRAME. The wrist is expressed in the TOOL's frame:

    X_{H|T}(t) = inv(X_T(t)) @ X_H(t)

because that is the quantity a transfer has to get right -- how the hand holds and moves the tool.
In the world frame the same grasp reads as a completely different trajectory depending on where on
the table the episode happened, and against the target frame it mixes the grasp with the approach.
The world and target frames are still available for comparison.

Finger pose is frame-independent by construction (MANO's 45 axis-angle parameters are in the hand's
own root frame), so it is carried through unchanged.

WRIST == `hand_trans`, exactly: TACO builds its MANO layer with `center_idx=0`, so joint 0 sits at
the origin before translation. No MANO forward pass is needed and none is done.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

import numpy as np

from src import geometry as G
from src.analysis.action_structure import representations as REP
from src.analysis.action_structure.chunks import CHUNK_LEN
from src.analysis.loaders import taco

log = logging.getLogger(__name__)

SIDES = ("left", "right")
#: (3 position + 6 rotation) per hand, both hands
WRIST_DIM = 2 * 9
#: MANO's 45 finger parameters per hand
FINGER_DIM = 2 * 45


def _episode_hand(traj, n_chunks: int):
    """(k,64,3) wrist positions, (k,64,3,3) wrist rotations and (k,64,45) finger pose, per side."""
    out = {}
    idx = np.arange(n_chunks * CHUNK_LEN).reshape(n_chunks, CHUNK_LEN)
    for side in SIDES:
        h = traj.hands.get(side)
        if h is None:
            return None
        out[side] = (h.root_pos[idx], G.wxyz_to_R(h.root_quat)[idx], h.finger_pose[idx])
    return out


def build(meta, root=None, refs: Optional[Sequence] = None, workers: int = 1) -> Dict[str, np.ndarray]:
    """Hand arrays aligned ROW FOR ROW with a `ChunkSet`'s metadata.

    Alignment is by `chunk_id`, not by position, because an episode that fails to load must leave a
    hole rather than shift every later row onto the wrong hand.
    """
    from pathlib import Path

    root = Path(root or taco.default_root())
    want = {}
    for ep, g in meta.groupby("episode_id", sort=False):
        want[ep] = int(g["n_chunks"].iloc[0])
    by_id = {r.sequence_id: r for r in (refs if refs is not None else taco.index(root))}

    n = len(meta)
    wp = np.full((n, CHUNK_LEN, 2, 3), np.nan)
    wR = np.full((n, CHUNK_LEN, 2, 3, 3), np.nan)
    fp = np.full((n, CHUNK_LEN, 2, 45), np.nan)
    pos = {cid: i for i, cid in enumerate(meta["chunk_id"])}

    missing = 0
    for ep, k in want.items():
        ref = by_id.get(ep)
        if ref is None:
            missing += k
            continue
        try:
            traj = taco.load(ref, root=root)
            got = _episode_hand(traj, k)
        except Exception as e:                      # noqa: BLE001 -- reported, then left as NaN
            log.warning("%s: %s: %s", ep, type(e).__name__, e)
            got = None
        if got is None:
            missing += k
            continue
        for c in range(k):
            i = pos.get(f"{ep}#{c}")
            if i is None:
                continue
            for s, side in enumerate(SIDES):
                wp[i, :, s], wR[i, :, s], fp[i, :, s] = (got[side][0][c], got[side][1][c],
                                                         got[side][2][c])
    if missing:
        log.warning("%d chunks have no hand annotation and stay NaN", missing)
    return {"wrist_pos": wp, "wrist_R": wR, "finger_pose": fp}


def in_frame(hand: Dict[str, np.ndarray], cs, frame: str = "tool") -> np.ndarray:
    """(N,64,18) wrist of both hands, in `frame`: "tool", "target" or "world"."""
    if frame == "world":
        p = hand["wrist_pos"]
        R = hand["wrist_R"]
    else:
        base_p, base_R = ((cs.tool_p, cs.tool_R) if frame == "tool" else (cs.targ_p, cs.targ_R))
        # R_base^T (p_H - p_base) and R_base^T R_H, per hand
        p = np.einsum("ntji,ntsj->ntsi", base_R, hand["wrist_pos"] - base_p[:, :, None, :])
        R = np.einsum("ntji,ntsjk->ntsik", base_R, hand["wrist_R"])
    six = REP.rot6d(R)                                          # (N,64,2,6)
    return np.concatenate([p, six], axis=-1).reshape(len(p), CHUNK_LEN, -1)


def targets(hand: Dict[str, np.ndarray], cs, frame: str = "tool") -> Dict[str, np.ndarray]:
    """The two prediction targets: wrist alone, and wrist plus fingers.

    Fingers are appended rather than analysed apart because the eventual target is the whole hand;
    reporting the wrist on its own says how much of any success is just "where the hand is".
    """
    wrist = in_frame(hand, cs, frame)                           # (N,64,18)
    fingers = hand["finger_pose"].reshape(len(wrist), CHUNK_LEN, -1)   # (N,64,90)
    return {"wrist": wrist, "wrist+fingers": np.concatenate([wrist, fingers], axis=-1)}
