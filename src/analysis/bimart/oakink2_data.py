"""OakInk2 store, windows, datasets and normalisation statistics for the BimArt port.

The TACO modules (`store.py`, `datasets.py`, `stats.py`) are left as they are, so the TACO results
stay reproducible. This module is their OakInk2 counterpart, and it restores three things upstream
does that the TACO port does not (review/plan.md, "Restore upstream without asking"):

    7-D GLOBAL STATE, WINDOW-RELATIVE. Upstream's global state is [rotvec (3), translation minus a
    reference translation (3), scale (1)], and the two models use different references:
        motion   t(f) - t(start - 1)    third_party/BimArt/dataset/motion_data.py:105-123
        contact  t(f) - t(start)        third_party/BimArt/dataset/contact_data.py:63-77
    The store keeps the ABSOLUTE part1 translation (`global_abs`, float64) and each dataset
    subtracts its own reference in `__getitem__`. The TACO port subtracts sequence frame 0 for
    both, which leaks the absolute take position into every window.

    TRAIN-ONLY, WINDOW-WEIGHTED STATISTICS. Upstream's released motion stats are the per-dimension
    mean/std over every frame row of every stride-1 train window (reproduced to 0.7% in
    review/report_upstream.md §3.4). A frame covered by k windows counts k times. The translation
    columns depend on the window's reference, so they are summed per window, per model.

    UPSTREAM TEST-WINDOW RULE. `start = base; while start + H < end: start += H`, with
    end = n - H - base, for the test split (motion_data.py:146-166).

AND ONE DATA FILTER (B5, user decision). A window is dropped when, in any of its frames, either
hand is within 5 mm of a tracked object outside the pair -- part-tree siblings included. The
model's input holds only the two parts, so a hand holding a third object makes contact the model
cannot see. Upstream has no such filter because ARCTIC has no third object.

WINDOWS per split: train at stride 1 (upstream); val at stride 16, for a validation pass upstream
does not have; test by the upstream rule above. The per-frame nonpair distance is in the store, so
the filter is applied when windows are built and can be switched off (`nonpair_m=None`).
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

DEFAULT_ROOT = Path("/result/uhnam/dexcore/oakink2/30_bimart_port")
NONPAIR_M = 0.005
VAL_STRIDE = 16
STD_FLOOR = 1e-4            # as stats.py: a constant channel is left unscaled

ARRAYS = {"bps": (3072, np.float32),          # obj_cano_bps, flattened
          "contact": (2048, np.float32),      # gathered at the BPS indices, lh then rh
          "action": (1200, np.float32),       # kp then dirvec
          "global_abs": (7, np.float64),      # part1 rotvec, ABSOLUTE translation, scale
          "nonpair": (2, np.float32)}         # each hand's closest non-pair object, m


# ------------------------------------------------------------------------------------ store
def build_store(root=DEFAULT_ROOT, out_dir=None, index=None) -> Path:
    """One flat, uncompressed memmap per array (see store.py for why). -> the store directory."""
    import pandas as pd

    root = Path(root)
    out_dir = Path(out_dir or root / "train_store")
    out_dir.mkdir(parents=True, exist_ok=True)
    index = index if index is not None else pd.read_csv(root / "sequence_index.csv")
    total = int(index["n_frames"].sum())
    mm = {name: np.lib.format.open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dt, shape=(total, d))
          for name, (d, dt) in ARRAYS.items()}
    starts, at = [], 0
    for i, r in enumerate(index.itertuples()):
        with np.load(root / "sequences" / r.file) as z:
            n = len(z["kp"])
            if n != int(r.n_frames):
                raise ValueError(f"{r.file}: {n} frames on disk, {r.n_frames} in the index")
            mm["bps"][at:at + n] = z["obj_cano_bps"].reshape(n, -1)
            mm["contact"][at:at + n] = z["contact"]
            mm["action"][at:at + n] = np.concatenate([z["kp"], z["dirvec"]], axis=-1)
            mm["global_abs"][at:at + n] = z["global_abs"]
            mm["nonpair"][at:at + n] = z["nonpair_min_dist"]
        starts.append(at)
        at += n
        if (i + 1) % 100 == 0 or i + 1 == len(index):
            log.info("  store %d/%d (%d rows)", i + 1, len(index), at)
    for v in mm.values():
        v.flush()
    off = index.assign(start=starts)
    off.to_csv(out_dir / "offsets.csv", index=False)
    return out_dir


def open_store(store_dir) -> Dict[str, np.ndarray]:
    store_dir = Path(store_dir)
    return {name: np.load(store_dir / f"{name}.npy", mmap_mode="r") for name in ARRAYS}


# ---------------------------------------------------------------------------------- windows
def window_starts(n: int, kind: str, pred_horizon: int = 64, base_frame: int = 8,
                  val_stride: int = VAL_STRIDE) -> np.ndarray:
    """Start frames of one sequence's windows. kind: "train", "val" or "test"."""
    end = n - pred_horizon - base_frame
    if kind == "train":
        return np.arange(base_frame, max(base_frame, end))
    if kind == "val":
        return np.arange(base_frame, max(base_frame, end), val_stride)
    starts, s = [], base_frame                          # upstream test rule
    while s + pred_horizon < end:
        starts.append(s)
        s += pred_horizon
    return np.asarray(starts, dtype=np.int64)


def clean_windows(nonpair: np.ndarray, starts: np.ndarray, pred_horizon: int,
                  nonpair_m: Optional[float]) -> np.ndarray:
    """(len(starts),) bool: no frame of the window has a hand within `nonpair_m` of a non-pair object."""
    if nonpair_m is None or len(starts) == 0:
        return np.ones(len(starts), dtype=bool)
    near = (np.asarray(nonpair).min(axis=1) < nonpair_m).astype(np.int64)
    c = np.concatenate([[0], np.cumsum(near)])
    return (c[starts + pred_horizon] - c[starts]) == 0


def relative_global(g_abs: np.ndarray, ref_trans: np.ndarray) -> np.ndarray:
    """(H,7) [rotvec, translation - reference, scale] from absolute rows."""
    out = np.array(g_abs, dtype=np.float64, copy=True)
    out[:, 3:6] -= ref_trans
    return out.astype(np.float32)


class _OakInk2Base(Dataset):
    def __init__(self, root=None, split: str = "train", pred_horizon: int = 64, base_frame: int = 8,
                 nonpair_m: Optional[float] = NONPAIR_M, window_kind: Optional[str] = None,
                 sequences: Optional[Sequence[str]] = None, store_dir: str = "train_store",
                 return_aux_info: bool = False):
        import pandas as pd

        self.root = Path(root or DEFAULT_ROOT)
        self.split = split
        self.pred_horizon = pred_horizon
        self.base_frame = base_frame
        self.return_aux_info = return_aux_info
        off = pd.read_csv(self.root / store_dir / "offsets.csv")
        if split != "all":
            off = off[off["split"] == split]
        if sequences is not None:
            off = off[off["sequence_id"].isin(set(sequences))]
        self.index = off.reset_index(drop=True)
        self.store = open_store(self.root / store_dir)

        kind = window_kind or ("train" if split in ("train", "all") else split)
        fi, st, n_all = [], [], 0
        for i, r in enumerate(self.index.itertuples()):
            s = window_starts(int(r.n_frames), kind, pred_horizon, base_frame)
            n_all += len(s)
            keep = clean_windows(self.store["nonpair"][r.start:r.start + r.n_frames], s,
                                 pred_horizon, nonpair_m)
            fi.append(np.full(int(keep.sum()), i, dtype=np.int64))
            st.append(s[keep])
        self.win_file = np.concatenate(fi) if fi else np.zeros(0, np.int64)
        self.win_start = np.concatenate(st) if st else np.zeros(0, np.int64)
        self.n_windows_before_filter = n_all
        log.info("%s (%s windows): %d sequences, %d windows (%d dropped by the %s m non-pair filter)",
                 split, kind, len(self.index), len(self.win_start), n_all - len(self.win_start), nonpair_m)

    def __len__(self) -> int:
        return len(self.win_start)

    def _rows(self, idx: int):
        row = self.index.iloc[int(self.win_file[idx])]
        base = int(row["start"]) + int(self.win_start[idx])
        return base, row


class OakInk2ContactDataset(_OakInk2Base):
    """BimArt's stage one: the contact map from the object alone. Reference = window frame 0."""

    def __getitem__(self, idx: int) -> Dict[str, object]:
        base, row = self._rows(idx)
        sl = slice(base, base + self.pred_horizon)
        g = self.store["global_abs"][sl]
        return {"action": np.array(self.store["contact"][sl]),
                "obs": {"obj_feat": np.array(self.store["bps"][sl]),
                        "curr_global_states": relative_global(g, g[0, 3:6])},
                "aux": {"filename": f"{row['sequence_id']}_{int(self.win_start[idx])}",
                        "pair": row["pair"], "part_group": bool(row["part_group"])}}


class OakInk2MotionDataset(_OakInk2Base):
    """Hand keypoints and direction vectors from object and contact. Reference = start - 1."""

    def __getitem__(self, idx: int) -> Dict[str, object]:
        base, row = self._rows(idx)
        sl = slice(base, base + self.pred_horizon)
        out = {"action": np.array(self.store["action"][sl]),
               "obs": {"contact_points": np.array(self.store["contact"][sl]),
                       "object": np.array(self.store["bps"][sl]),
                       "global_states": relative_global(self.store["global_abs"][sl],
                                                        self.store["global_abs"][base - 1, 3:6])}}
        if self.return_aux_info:
            out["viz"] = {"filename": row["sequence_id"], "pair": row["pair"],
                          "part0": row["part0"], "part1": row["part1"],
                          "part_group": bool(row["part_group"]),
                          "start_index": int(self.win_start[idx])}
        return out


# ------------------------------------------------------------------------------- statistics
class _Weighted:
    """Weighted mean and variance, shifted by the first rows seen to keep float64 cancellation small."""

    def __init__(self):
        self.w = 0.0
        self.shift = self.s1 = self.s2 = None

    def update(self, x: np.ndarray, w: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if self.shift is None:
            self.shift = x[w > 0].mean(axis=0) if (w > 0).any() else np.zeros(x.shape[1])
            self.s1 = np.zeros(x.shape[1])
            self.s2 = np.zeros(x.shape[1])
        d = x - self.shift
        self.s1 += w @ d
        self.s2 += w @ (d * d)
        self.w += float(w.sum())

    def add_moments(self, w: float, s1: np.ndarray, s2: np.ndarray) -> None:
        """Pre-summed, unshifted moments (the per-window translation columns)."""
        self.w += w
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2
        self.shift = np.zeros_like(self.s1) if self.shift is None else self.shift

    def finish(self) -> Dict[str, np.ndarray]:
        m = self.s1 / self.w
        std = np.sqrt(np.maximum(self.s2 / self.w - m * m, 0.0))
        return {"mean": m + self.shift, "std": np.where(std > STD_FLOOR, std, 1.0)}


def _translation_moments(t: np.ndarray, starts: np.ndarray, ref: np.ndarray, H: int):
    """Sum and sum of squares, over windows and their frames, of t(f) - t(ref_s). Exact, via prefix sums."""
    c1 = np.concatenate([np.zeros((1, 3)), np.cumsum(t, axis=0)])
    c2 = np.concatenate([np.zeros((1, 3)), np.cumsum(t * t, axis=0)])
    win1 = c1[starts + H] - c1[starts]                  # (W,3) sum of t over the window
    win2 = c2[starts + H] - c2[starts]
    r = t[ref]                                          # (W,3)
    return (win1 - H * r).sum(0), (win2 - 2 * r * win1 + H * r * r).sum(0)


def compute_stats(root=DEFAULT_ROOT, store_dir: str = "train_store", pred_horizon: int = 64,
                  base_frame: int = 8, nonpair_m: Optional[float] = NONPAIR_M) -> Dict[str, Dict]:
    """Train windows only, window-weighted, in the key layout BimArt's `load_stat_dict` expects."""
    ds = _OakInk2Base(root, "train", pred_horizon, base_frame, nonpair_m, store_dir=store_dir)
    st, H = ds.store, pred_horizon
    acc = {k: _Weighted() for k in ("contact", "bps", "action", "rot_scale")}
    tr = {m: _Weighted() for m in ("motion", "contact")}
    for i, r in enumerate(ds.index.itertuples()):
        starts = ds.win_start[ds.win_file == i]
        if len(starts) == 0:
            continue
        n = int(r.n_frames)
        w = np.zeros(n + 1)
        np.add.at(w, starts, 1.0)
        np.add.at(w, starts + H, -1.0)
        w = np.cumsum(w)[:n]                            # windows covering each frame
        sl = slice(int(r.start), int(r.start) + n)
        acc["contact"].update(st["contact"][sl], w)
        acc["bps"].update(st["bps"][sl], w)
        acc["action"].update(st["action"][sl], w)
        g = np.asarray(st["global_abs"][sl])
        acc["rot_scale"].update(g[:, [0, 1, 2, 6]], w)
        for m, ref in (("motion", starts - 1), ("contact", starts)):
            s1, s2 = _translation_moments(g[:, 3:6], starts, ref, H)
            tr[m].add_moments(float(len(starts) * H), s1, s2)
        if (i + 1) % 100 == 0 or i + 1 == len(ds.index):
            log.info("  stats %d/%d sequences", i + 1, len(ds.index))
    s = {k: v.finish() for k, v in acc.items()}

    def gs(model):
        t = tr[model].finish()
        rs = s["rot_scale"]
        return {k: np.concatenate([rs[k][:3], t[k], rs[k][3:]]) for k in ("mean", "std")}

    def cp(d):
        return {k: v.copy() for k, v in d.items()}

    g_c, g_m = gs("contact"), gs("motion")
    # every entry its own dict: `load_stat_dict` converts in place (see stats.py)
    contact = {"action": cp(s["contact"]), "prev_contact": cp(s["contact"]),
               "obj_feat": cp(s["bps"]), "prev_obj_feat": cp(s["bps"]),
               "curr_global_states": {k: v[None, :].copy() for k, v in g_c.items()},
               "prev_global_states": {k: v[None, :].copy() for k, v in g_c.items()}}
    motion = {"action": cp(s["action"]), "contact_points": cp(s["contact"]),
              "object": cp(s["bps"]), "global_states": cp(g_m)}
    meta = {"train_windows": int(len(ds)), "train_windows_before_filter": int(ds.n_windows_before_filter),
            "train_sequences": int(ds.index["sequence_id"].nunique()), "nonpair_m": nonpair_m,
            "floored_channels": {k: int((v["std"] == 1.0).sum()) for k, v in s.items()}}
    return {"contact": contact, "motion": motion, "meta": meta}


def save_stats(stats: Dict[str, Dict], out_dir) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("contact", "motion"):
        p = out_dir / f"oakink2_{name}_norm_stats.pkl"
        with open(p, "wb") as f:
            pickle.dump(stats[name], f)
        paths[name] = p
    return paths
