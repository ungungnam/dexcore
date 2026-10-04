"""Shared definitions for the hierarchical contact-generation study (initial contact-mode sampler
+ deterministic contact-evolution vector field), run independently on TACO, ARCTIC and OakInk2.

Everything contact-related is inherited from the dynamic-contact study (scripts/research/
canonical_contact/dyn/dc_common.py, imported as C): the 512-D canonical contact vector X, its mass /
pattern split, the per-dataset zero-mass threshold, the 1 cm hard-contact rule, the object states
and the geometry descriptor. This module adds only what the new study needs:

  firm contact      n_hard(t) > 0  and  mass(t) >= ZERO_MASS, holding for FIRM_PERSIST consecutive
                    frames (the 'both_hard' physical rule of the earlier study plus the dataset
                    zero threshold; the persistence requirement skips one-frame flickers)
  episode onset     first firm frame of the take, plus every later firm frame that follows
                    >= EPISODE_GAP frames without firm contact (a new contact episode)
  sequence          C_0..C_{T-1} = the T = 64 frames (30 Hz, stride 1) starting at an onset; kept
                    only when all T frames exist
  local context     tau_local,t = O_{t-8..t+8 step 2} (9 object states, +-0.27 s), clamped at the
                    take boundaries exactly as the earlier study's input family C
  split             ONE fixed train / val / test split per dataset at the take level (TACO, ARCTIC:
                    the dataset labels; OakInk2: seeded 80/10/10 by take, stratified by group)
  static condition  G = surface descriptor of the mesh in the canonical index space (nearest
                    distance + coverage at the 512 canonical points, 1024-D) + metric radius +
                    extent, plus the hand (L/R) and, for TACO, the role (tool/target) as one-hot
                    flags; one pooled model per dataset (the groups share it)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
sys.path.insert(0, str(REPO / "scripts/research/canonical_contact/dyn"))
sys.path.insert(0, str(REPO))
import dc_common as C  # noqa: E402  (reads DC_DATASET)

DATASET = C.DATASET
T = 64
FPS = 30
LOCAL_OFFSETS = list(range(-8, 9, 2))        # 9 states
CTX_PAD = 8                                  # object states are stored for t in [-8, T+8)
FIRM_PERSIST = 3
EPISODE_GAP = 15                             # 0.5 s without firm contact separates episodes
_ROOTS = {"taco": "/result/uhnam/dexcore/taco/50_hier_contact_gen",
          "arctic": "/result/uhnam/dexcore/arctic/40_hier_contact_gen",
          "oakink2": "/result/uhnam/dexcore/oakink2/20_hier_contact_gen",
          "oakink2ext": "/result/uhnam/dexcore/oakink2/21_hier_contact_gen_extended"}   # ext: built, not run
OUT = Path(_ROOTS[DATASET])
# /ckpt was 100 % full (other users' checkpoints) when the first rollout-trained VF tried to save, so the
# checkpoint store lives under /result; <root>/checkpoints links to it.
CKPT = Path(f"/result/uhnam/dexcore/hier_contact_gen_ckpt/{DATASET}")
SEQ_FILE = OUT / "sequences.npz"
SPLIT, FOLD = "fixed", 0                      # one fixed take-level split per dataset (manifests/fixed0.json)
DEVICE_ENV = "CUDA_VISIBLE_DEVICES"


def firm_mask(mass, n_hard, thr):
    """Per-frame firm contact with the persistence requirement (a frame is firm when it starts a
    run of FIRM_PERSIST firm frames, or belongs to one)."""
    raw = (n_hard > 0) & (mass >= thr)
    if len(raw) < FIRM_PERSIST:
        return np.zeros(len(raw), bool)
    runs = np.ones(len(raw) - FIRM_PERSIST + 1, bool)
    for j in range(FIRM_PERSIST):
        runs &= raw[j:len(raw) - FIRM_PERSIST + 1 + j]
    out = np.zeros(len(raw), bool)
    for j in range(FIRM_PERSIST):
        out[j:len(raw) - FIRM_PERSIST + 1 + j] |= runs
    return out


def episode_onsets(firm):
    """Indices where a contact episode starts: the first firm frame, and every firm frame that
    follows >= EPISODE_GAP consecutive non-firm frames."""
    onsets = []
    last_firm = None
    for i, f in enumerate(firm):
        if f:
            if last_firm is None or i - last_firm > EPISODE_GAP:
                onsets.append(i)
            last_firm = i
    return onsets


class ObjectStates:
    def __init__(self):
        z = np.load(C.CACHE / "object_states.npz", allow_pickle=True)
        self.off = {str(s): (int(a), int(b)) for s, a, b in zip(z["sequence_id"], z["offset"][:-1], z["offset"][1:])}
        self.S = z["states"].astype(np.float32)
        self.names = [str(n) for n in z["names"]]
        self.D = self.S.shape[1]

    def get(self, sid, frames):
        a, b = self.off[sid]
        f = np.clip(np.asarray(frames), 0, b - a - 1)
        return self.S[a + f]


def load_sequences():
    """The built sequence set: dict of arrays (see build_sequences.py) + a meta DataFrame."""
    z = np.load(SEQ_FILE, allow_pickle=True)
    meta = pd.read_csv(OUT / "sequences_meta.csv", dtype={"mesh_id": str, "subject": str, "sequence_id": str})
    return z, meta


def geometry_table():
    """All meshes of the dataset: id -> (category, 1024-D descriptor, radius, extent)."""
    rows = {}
    for p in sorted(C.CACHE.glob("geometry_normalized__*.npz")):
        cat = p.stem.split("__", 1)[1]
        g = np.load(p, allow_pickle=True)
        for i, m in enumerate(g["mesh_ids"].astype(str)):
            rows[(cat, m)] = dict(nearest=g["nearest"][i].astype(np.float32), coverage=g["coverage"][i].astype(np.float32),
                                  radius_m=float(g["radius_m"][i]), extent_m=g["extent_m"][i].astype(np.float32))
    return rows


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=C._json_default))


def cluster_bootstrap(values, clusters, n_boot=1000, seed=0):
    """Mean with a 95 % cluster (take-level) bootstrap interval."""
    values = np.asarray(values, float); clusters = np.asarray(clusters)
    ok = ~np.isnan(values); values, clusters = values[ok], clusters[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    u, inv = np.unique(clusters, return_inverse=True)
    rng = np.random.default_rng(seed)
    sums = np.bincount(inv, weights=values, minlength=len(u)); cnt = np.bincount(inv, minlength=len(u)).astype(float)
    reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        reps.append((w * sums).sum() / max((w * cnt).sum(), 1e-12))
    return float(values.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
