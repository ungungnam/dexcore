"""One flat memmap per array, for random-access training.

WHY NOT READ THE `.npz` FILES DIRECTLY. Preprocessing writes one compressed archive per sequence,
which is the right archival form -- it keeps the dense per-vertex contact so a different BPS basis
can be tried later without recomputing anything. It is the wrong TRAINING form, twice over:

    `mmap_mode` does nothing for a compressed npz. numpy silently falls back to a zip handle and
    decompresses on every read, and a forked DataLoader worker inherits that handle, so several
    workers reading at once corrupt each other's zlib stream.

    Shuffled sampling defeats any cache. A batch of 100 random windows touches ~100 different
    sequences, so a per-worker cache misses on nearly every item and pays a full ~10 MB
    decompression for 64 frames.

So the arrays the trainer actually reads are written once, uncompressed, concatenated across
sequences into one file each, with a row offset per sequence. Random access is then a page fault,
workers share the mapping read-only, and nothing decompresses.

THE CONTACT VECTOR IS GATHERED HERE, ONCE. Both datasets index the dense per-vertex distance at
that frame's BPS indices; doing it at build time turns a 2 x 7998 read plus a gather into a flat
2048 read, and drops 22 GiB off the store.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

ARRAYS = {"bps": (3072, np.float32),        # obj_cano_bps, flattened
          "contact": (2048, np.float32),    # gathered at the BPS indices, left then right
          "action": (1200, np.float32),     # kp then dirvec
          "global": (8, np.float32)}


def _gather_contact(z) -> np.ndarray:
    inds = z["obj_cano_bps_inds"]
    rows = np.arange(len(inds))[:, None]
    return np.concatenate([z["contact_left"][rows, inds], z["contact_right"][rows, inds]],
                          axis=-1).astype(np.float32)


def build(root, out_dir=None, index=None) -> Path:
    """Write the store. Returns its directory; `offsets.csv` maps sequence -> row range."""
    import pandas as pd

    root = Path(root)
    out_dir = Path(out_dir or root / "train_store")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = index if index is not None else pd.read_csv(root / "sequence_index.csv")

    total = int(index["n_frames"].sum())
    mm = {name: np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+",
                                          dtype=dt, shape=(total, d))
          for name, (d, dt) in ARRAYS.items()}
    rows, at = [], 0
    for i, r in index.iterrows():
        with np.load(root / "sequences" / r["file"]) as z:
            n = len(z["kp"])
            mm["bps"][at:at + n] = z["obj_cano_bps"].reshape(n, -1)
            mm["contact"][at:at + n] = _gather_contact(z)
            mm["action"][at:at + n] = np.concatenate([z["kp"], z["dirvec"]], axis=-1)
            mm["global"][at:at + n] = z["global_states"]
        rows.append({"sequence_id": r["sequence_id"], "start": at, "n_frames": n})
        at += n
        if (i + 1) % 250 == 0 or i + 1 == len(index):
            log.info("  store %d/%d (%d rows)", i + 1, len(index), at)
    for v in mm.values():
        v.flush()
    off = index.merge(pd.DataFrame(rows), on="sequence_id", suffixes=("", "_store"))
    off.to_csv(out_dir / "offsets.csv", index=False)
    log.info("store at %s: %d rows", out_dir, at)
    return out_dir


def open_store(store_dir) -> Dict[str, np.ndarray]:
    """Read-only memmaps. Safe to open before a fork: each worker gets its own page mapping."""
    store_dir = Path(store_dir)
    return {name: np.load(store_dir / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
