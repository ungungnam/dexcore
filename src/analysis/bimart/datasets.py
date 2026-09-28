"""TACO datasets with BimArt's interface, so its trainers can be pointed at them unchanged.

BimArt's own `ObjectContactData` and `MotionDataset` walk an ARCTIC directory tree, parse a
category out of each filename against a hardcoded list of eleven, and read `obj_world_state` as the
(T,7) [articulation, rotation, translation] vector ARCTIC ships. None of that survives the move to
TACO, but the DICTIONARIES THEY HAND THE TRAINER do -- same keys, same shapes, same units. These
classes reproduce those dictionaries from the preprocessed `.npz` files.

    contact model   {"action": (T,2048), "obs": {"obj_feat": (T,3072),
                                                 "curr_global_states": (T,8)}, "aux": {...}}
    motion model    {"action": (T,1200), "obs": {"contact_points": (T,2048),
                                                 "object": (T,3072),
                                                 "global_states": (T,8)}, "viz": {...}}

READS COME FROM THE FLAT STORE, not from the per-sequence archives. `store.py` explains why at
length; the short version is that a compressed npz cannot be memory-mapped, decompresses on every
read, and corrupts itself when forked DataLoader workers share its handle. The archives stay on
disk as the archival form -- they keep the dense per-vertex contact, so a different BPS basis can
be tried without recomputing anything -- and the store holds exactly what the trainer reads.

WINDOWING follows BimArt: a sliding window of stride 1 for training, non-overlapping windows for
test. `base_frame` drops the head of each sequence -- 100 upstream to skip ARCTIC's T-pose, 8 here
because TACO has no T-pose and a median length of 148 frames, where 100 would discard the release.

THE SPLIT IS TACO'S OWN, not a fresh random one. The release ships four test sets graded by what
they share with training -- `test_1` shares everything, `test_2` holds out the tool MESHES,
`test_3` holds out the (verb, tool, target) triplets, `test_4` holds out both. A random split would
throw that structure away, and `test_2` in particular is the transfer question this port exists to
ask.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from src.analysis.bimart import store

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/result/uhnam/dexcore/taco/10_bimart_gen1_original_label")
TEST_SPLITS = ("test_1", "test_2", "test_3", "test_4")


def _resolve_splits(split: str) -> Optional[List[str]]:
    """"train", one test name, "test" for all four, or "all" for everything."""
    if split == "all":
        return None
    if split == "test":
        return list(TEST_SPLITS)
    return [split]


class _TacoBase(Dataset):
    def __init__(self, root=None, split: str = "train", pred_horizon: int = 64,
                 base_frame: int = 8, end_frame: Optional[int] = None,
                 return_aux_info: bool = False, sequences: Optional[Sequence[str]] = None,
                 store_dir: str = "train_store"):
        import pandas as pd

        self.root = Path(root or DEFAULT_ROOT)
        self.store_dir = store_dir
        self.split = split
        self.pred_horizon = pred_horizon
        self.base_frame = base_frame
        self.end_frame = end_frame
        self.return_aux_info = return_aux_info

        index = pd.read_csv(self.root / "sequence_index.csv")
        wanted = _resolve_splits(split)
        if wanted is not None:
            if "split" not in index.columns:
                raise KeyError("sequence_index.csv has no `split` column; rerun preprocessing")
            index = index[index["split"].isin(wanted)]
        if sequences is not None:
            index = index[index["sequence_id"].isin(set(sequences))]
        self.index = index.reset_index(drop=True)

        self.store = store.open_store(self.root / self.store_dir)
        off = pd.read_csv(self.root / self.store_dir / "offsets.csv")
        starts = dict(zip(off["sequence_id"], off["start"]))
        self.index["store_start"] = self.index["sequence_id"].map(starts)
        if self.index["store_start"].isna().any():
            missing = self.index.loc[self.index["store_start"].isna(), "sequence_id"].tolist()
            raise KeyError(f"{len(missing)} sequences are not in the store, e.g. {missing[:3]}")
        self.index["store_start"] = self.index["store_start"].astype(int)
        self.windows = self._build_windows()
        log.info("%s: %d sequences, %d windows", split, len(self.index), len(self.windows))

    def _build_windows(self) -> List[dict]:
        out = []
        train_like = self.split in ("train", "all")
        for i, n in enumerate(self.index["n_frames"]):
            end = self.end_frame if self.end_frame is not None else n - self.pred_horizon - self.base_frame
            if end <= self.base_frame:
                continue
            if train_like:
                starts = range(self.base_frame, end)                 # stride 1
            else:
                starts = range(self.base_frame, end, self.pred_horizon)   # non-overlapping
            for s in starts:
                out.append({"file_idx": i, "start": int(s)})
        return out

    def __len__(self) -> int:
        return len(self.windows)

    def _load(self, file_idx: int) -> dict:
        """The sequence's arrays, decompressed, from this process's own cache."""
        pid = os.getpid()
        if self._cache_pid != pid:            # a fork inherited the parent's cache; drop it
            self._cache = OrderedDict()
            self._cache_pid = pid
        hit = self._cache.get(file_idx)
        if hit is not None:
            self._cache.move_to_end(file_idx)
            return hit
        with np.load(self._paths[file_idx]) as z:
            data = {k: z[k] for k in z.files}
        self._cache[file_idx] = data
        if len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return data

    def _rows(self, idx: int):
        """(row slice into the store, the sequence's index row)."""
        w = self.windows[idx]
        row = self.index.iloc[w["file_idx"]]
        base = int(row["store_start"]) + w["start"]
        return slice(base, base + self.pred_horizon), row


class TacoContactDataset(_TacoBase):
    """Predicts the contact map from the object alone -- BimArt's stage one."""

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sl, row = self._rows(idx)
        return {"action": np.asarray(self.store["contact"][sl]),
                "obs": {"obj_feat": np.asarray(self.store["bps"][sl]),
                        "curr_global_states": np.asarray(self.store["global"][sl])},
                "aux": {"filename": f"{row['sequence_id']}_{self.windows[idx]['start']}",
                        "category": row["tool_cat"], "triplet": row["triplet"],
                        "verb": row["verb"]}}


class TacoMotionDataset(_TacoBase):
    """Predicts hand keypoints and direction vectors, conditioned on object and contact."""

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sl, row = self._rows(idx)
        out = {"action": np.asarray(self.store["action"][sl]),
               "obs": {"contact_points": np.asarray(self.store["contact"][sl]),
                       "object": np.asarray(self.store["bps"][sl]),
                       "global_states": np.asarray(self.store["global"][sl])}}
        if self.return_aux_info:
            out["viz"] = {"filename": row["sequence_id"], "triplet": row["triplet"],
                          "verb": row["verb"], "tool_mesh": row["tool_mesh"],
                          "target_mesh": row["target_mesh"],
                          "start_index": self.windows[idx]["start"]}
        return out
