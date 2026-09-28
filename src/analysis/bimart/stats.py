"""Normalisation statistics, in the layout BimArt's `load_stat_dict` expects.

The shipped `contact_norm_stats.pkl` and `motion_norm_stats.pkl` are ARCTIC's. Reusing them on TACO
would centre the data on the wrong mean -- TACO's objects are smaller, its scales run 3.3-17.8
against ARCTIC's 2.4-8.3, and its canonical frame is the tool's, not an articulated object's. So
they are recomputed here from the preprocessed TACO sequences, with the same keys and shapes.

THE CONTACT ACTION IS GATHERED, NOT DENSE. BimArt stores a distance per object VERTEX and then
indexes it at the BPS indices to get the 2048-dimensional action
(`contact_data.get_frame_contact_feature`). The statistics are computed on the gathered vector, so
they match what the network actually sees.

Standard deviations are floored: a BPS channel that never varies -- a basis point permanently
inside the object, say -- would otherwise divide by ~0 and send that channel to infinity.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)

#: Below this a channel is treated as constant and left unscaled.
STD_FLOOR = 1e-4


class _Running:
    """Streaming mean and variance, so 2317 sequences never have to be held at once."""

    def __init__(self):
        self.n = 0
        self.mean = None
        self.m2 = None

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).reshape(-1, np.shape(x)[-1])
        if self.mean is None:
            self.mean = np.zeros(x.shape[1])
            self.m2 = np.zeros(x.shape[1])
        for chunk in np.array_split(x, max(1, len(x) // 4096)):
            k = len(chunk)
            if k == 0:
                continue
            delta = chunk.mean(axis=0) - self.mean
            new_n = self.n + k
            self.mean += delta * (k / new_n)
            self.m2 += chunk.var(axis=0) * k + (delta ** 2) * (self.n * k / new_n)
            self.n = new_n

    def finish(self) -> Dict[str, np.ndarray]:
        std = np.sqrt(self.m2 / max(self.n, 1))
        return {"mean": self.mean, "std": np.where(std > STD_FLOOR, std, 1.0)}


def gathered_contact(npz) -> np.ndarray:
    """(T, 2048) the contact action: dense per-vertex distance indexed at the BPS points."""
    inds = npz["obj_cano_bps_inds"]
    rows = np.arange(len(inds))[:, None]
    return np.concatenate([npz["contact_left"][rows, inds], npz["contact_right"][rows, inds]],
                          axis=-1)


def compute(sequence_dir, files: Optional[Iterable[str]] = None) -> Dict[str, Dict]:
    """-> {"contact": stat dict, "motion": stat dict}, both ready to pickle."""
    sequence_dir = Path(sequence_dir)
    files = sorted(files or [p.name for p in sequence_dir.glob("*.npz")])
    acc = {k: _Running() for k in ("contact_action", "obj_feat", "global_states", "motion_action")}
    for i, name in enumerate(files, 1):
        z = np.load(sequence_dir / name)
        acc["contact_action"].update(gathered_contact(z))
        acc["obj_feat"].update(z["obj_cano_bps"].reshape(len(z["obj_cano_bps"]), -1))
        acc["global_states"].update(z["global_states"])
        acc["motion_action"].update(np.concatenate([z["kp"], z["dirvec"]], axis=-1))
        if i % 200 == 0 or i == len(files):
            log.info("  stats %d/%d", i, len(files))
    s = {k: v.finish() for k, v in acc.items()}

    # BimArt's own key names, and its (1, D) shape for the contact model's global states.
    # Every entry gets its OWN dict even where two keys hold the same numbers: BimArt's
    # `load_stat_dict` converts each in place, so a shared object would be converted twice and
    # raise on the second pass.
    def gs2d():
        return {k: v[None, :].copy() for k, v in s["global_states"].items()}

    def cp(k):
        return {kk: vv.copy() for kk, vv in s[k].items()}

    contact = {"action": cp("contact_action"), "prev_contact": cp("contact_action"),
               "obj_feat": cp("obj_feat"), "prev_obj_feat": cp("obj_feat"),
               "curr_global_states": gs2d(), "prev_global_states": gs2d()}
    motion = {"action": cp("motion_action"), "contact_points": cp("contact_action"),
              "object": cp("obj_feat"), "global_states": cp("global_states")}
    return {"contact": contact, "motion": motion}


def save(stats: Dict[str, Dict], out_dir) -> Dict[str, Path]:
    import pickle

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, d in stats.items():
        p = out_dir / f"taco_{name}_norm_stats.pkl"
        with open(p, "wb") as f:
            pickle.dump(d, f)
        paths[name] = p
    return paths
