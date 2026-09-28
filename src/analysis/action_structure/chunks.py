"""The chunking protocol: one 64-frame non-overlapping window is one data point.

    N_chunks = floor(F / 64),  chunk k = frames [64k .. 64k+63],  remainder discarded

No resampling, no interpolation, no padding, no sliding windows. A chunk inherits its episode's
action, tool/target category, tool/target MESH ID, the episode id, its own index k and how many
chunks the episode produced -- the last two because a chunk's position inside its episode is a
candidate explanation for any structure found, and the episode id is what keeps chunks from one
episode from being counted as independent observations.

ONLY RIGID-OBJECT POSES ARE READ. Hands are not loaded at all (`with_hands=False`), so no
hand-derived quantity can leak into a representation by accident.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src import geometry as G
from src.analysis.loaders import taco

log = logging.getLogger(__name__)

CHUNK_LEN = 64


@dataclass
class ChunkSet:
    """Every chunk's poses plus its metadata. Arrays are parallel to `meta`'s rows.

    Rotations are kept as matrices here -- the representations in `representations.py` are the
    only place a rotation is encoded, so the encoding choice (6D, log-map) lives in one file.
    """
    tool_p: np.ndarray        # (N,64,3) metres, world
    tool_R: np.ndarray        # (N,64,3,3)
    targ_p: np.ndarray        # (N,64,3)
    targ_R: np.ndarray        # (N,64,3,3)
    meta: "object"            # pandas DataFrame, N rows

    def __len__(self) -> int:
        return len(self.tool_p)

    @property
    def episode_weights(self) -> np.ndarray:
        """1/N_chunks for each chunk: the episode-balanced weight the protocol asks for."""
        return 1.0 / self.meta["n_chunks"].to_numpy(dtype=float)


def _episode_chunks(traj, ref) -> Optional[dict]:
    """Split one episode into its chunks, or None when it is shorter than one chunk."""
    tool, targ = traj.tool, traj.target
    if tool is None or targ is None:
        return None
    n = min(len(tool), len(targ))
    k = n // CHUNK_LEN
    if k == 0:
        return None
    idx = np.arange(k * CHUNK_LEN).reshape(k, CHUNK_LEN)
    return {
        "tool_p": tool.pos[idx], "tool_R": G.wxyz_to_R(tool.quat)[idx],
        "targ_p": targ.pos[idx], "targ_R": G.wxyz_to_R(targ.quat)[idx],
        "rows": [{"chunk_id": f"{ref.sequence_id}#{i}", "episode_id": ref.sequence_id,
                  "triplet": ref.triplet, "action": ref.action.verb,
                  "tool_cat": ref.action.tool, "target_cat": ref.action.target,
                  "tool_mesh": tool.name, "target_mesh": targ.name,
                  "chunk_index": i, "n_chunks": k, "episode_frames": n} for i in range(k)],
    }


def build(root=None, refs: Optional[Sequence] = None, limit: Optional[int] = None,
          seed: int = 0, **index_kwargs) -> ChunkSet:
    """Chunk the whole TACO release (or a filtered subset of it).

    `limit` takes a RANDOM sample of episodes, not the first N. The index is ordered by triplet,
    so the first N episodes are all the same verb -- a smoke run on them would report perfect
    action structure because there is only one action in the sample.
    """
    import pandas as pd

    root = Path(root or taco.default_root())
    refs = list(refs if refs is not None else taco.index(root, **index_kwargs))
    if limit and limit < len(refs):
        rng = np.random.default_rng(seed)
        refs = [refs[i] for i in sorted(rng.choice(len(refs), limit, replace=False))]

    tp, tR, op, oR, rows = [], [], [], [], []
    skipped = 0
    for ref in refs:
        try:
            traj = taco.load(ref, root=root, with_hands=False)
        except Exception as e:                      # noqa: BLE001 -- reported, then skipped
            log.warning("%s: %s: %s", ref.sequence_id, type(e).__name__, e)
            skipped += 1
            continue
        c = _episode_chunks(traj, ref)
        if c is None:
            skipped += 1
            continue
        tp.append(c["tool_p"]); tR.append(c["tool_R"])
        op.append(c["targ_p"]); oR.append(c["targ_R"])
        rows.extend(c["rows"])

    if not rows:
        raise RuntimeError("no episode produced a full 64-frame chunk")
    log.info("%d episodes -> %d chunks (%d episodes gave none)", len(refs) - skipped, len(rows),
             skipped)
    return ChunkSet(np.concatenate(tp), np.concatenate(tR), np.concatenate(op),
                    np.concatenate(oR), pd.DataFrame(rows))
