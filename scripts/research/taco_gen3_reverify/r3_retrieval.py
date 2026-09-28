#!/usr/bin/env python
"""R3: G1 (is the model a lookup table?), G3 (does the input shift explain the gap?), G9 (takes).

For each test_1 window the 100 nearest TRAIN windows are found in the same descriptor space the
original baseline used (mean+std over the 64 frames of the BPS + global-state features, z-scored),
and the hand error of each neighbour's trajectory against the query's ground truth is measured.
`oracle_k` is the best of those k -- a retrieval system that is told the answer. The model is a
lookup table only if it cannot beat that.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

G = Path("/result/uhnam/dexcore/bimart_taco_scene/contact_probe/gen/reverify")
ROOT = Path("/result/uhnam/dexcore/bimart_taco_scene")
S = ROOT / "train_store"
H, KP_DIM, K = 64, 600, 100
dev = "cuda"

off = pd.read_csv(S / "offsets.csv")
bps = np.load(S / "bps.npy", mmap_mode="r")
glo = np.load(S / "global.npy", mmap_mode="r")
act = np.load(S / "action.npy", mmap_mode="r")
base = 8


def windows(row, stride):
    end = int(row.n_frames_store) - H - base
    return [] if end <= base else list(range(base, end, stride))


def descriptors(row, starts):
    s0, n = int(row.start), int(row.n_frames_store)
    X = np.concatenate([np.asarray(bps[s0:s0 + n], dtype=np.float32),
                        np.asarray(glo[s0:s0 + n], dtype=np.float32)], axis=1)
    D = np.empty((len(starts), 2 * X.shape[1]), dtype=np.float32)
    for i, st in enumerate(starts):
        w = X[st:st + H]
        D[i, :X.shape[1]] = w.mean(0)
        D[i, X.shape[1]:] = w.std(0)
    return D


def hands(row, starts):
    s0 = int(row.start)
    A = np.asarray(act[s0:s0 + int(row.n_frames_store), :KP_DIM], dtype=np.float32)
    return np.stack([A[st:st + H] for st in starts])          # (n, 64, 600) metres


tr = off[off.split == "train"].reset_index(drop=True)
Xi, Hi, Mi = [], [], []
for i, row in tr.iterrows():
    st = windows(row, 4)
    if not st:
        continue
    Xi.append(descriptors(row, st))
    Hi.append(hands(row, st))
    Mi += [{"sequence_id": row.sequence_id, "start": s} for s in st]
Xi = np.concatenate(Xi); Hi = np.concatenate(Hi); Mi = pd.DataFrame(Mi)
mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
Zi = torch.from_numpy((Xi - mu) / sd).to(dev)
Hi_t = torch.from_numpy(Hi.reshape(len(Hi), -1)).to(dev)
Zi_sq = (Zi ** 2).sum(1)
print(f"index: {len(Zi)} train windows (stride 4), {Zi.shape[1]} dims", flush=True)

rows = []
for split in ("test_1", "test_2", "test_3", "test_4"):
    sub = off[off.split == split].reset_index(drop=True)
    for i, row in sub.iterrows():
        st = windows(row, H)
        if not st:
            continue
        q = torch.from_numpy((descriptors(row, st) - mu) / sd).to(dev)
        qh = torch.from_numpy(hands(row, st).reshape(len(st), -1)).to(dev)
        d2 = (q ** 2).sum(1, keepdim=True) + Zi_sq[None] - 2 * q @ Zi.T
        nnd, nni = torch.topk(d2, K, dim=1, largest=False)
        for b, s in enumerate(st):
            cand = Hi_t[nni[b]]                                        # (K, 64*600)
            err = torch.linalg.norm((cand - qh[b][None]).reshape(K, H, 200, 3),
                                    dim=-1).mean(dim=(1, 2)) * 1000     # (K,) mm
            rows.append({"split": split, "sequence_id": row.sequence_id, "start": s,
                         "nn_dist": float(nnd[b, 0].clamp(min=0).sqrt()),
                         "nn1_mm": float(err[0]), "oracle5_mm": float(err[:5].min()),
                         "oracle20_mm": float(err[:20].min()), "oracle100_mm": float(err.min()),
                         "nn_sequence_id": Mi.sequence_id.iloc[int(nni[b, 0])]})
    print(split, "done", flush=True)
R = pd.DataFrame(rows)
R.to_csv(G / "r3_retrieval_scene.csv", index=False)

# ---- join the model's own error (window level, pooled over hands) -------------------------
w = pd.read_pickle(G / "window_level.pkl")
mw = (w[w.run == "scene"].groupby(["split", "sequence_id", "start"])[["motion_mm", "contact_mm", "total_mm"]]
      .mean().reset_index())
J = R.merge(mw, on=["split", "sequence_id", "start"], how="inner")
J.to_csv(G / "r3_retrieval_joined.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== G1: model vs retrieval (scene run), window level ===")
print(J.groupby("split")[["motion_mm", "nn1_mm", "oracle5_mm", "oracle20_mm", "oracle100_mm", "nn_dist"]]
      .mean().round(1).to_string())
print("  leak (a neighbour from the query's own sequence):",
      int((R.sequence_id == R.nn_sequence_id).sum()))
print("\n=== G3: does the input shift explain the error? spearman(nn_dist, model error) ===")
for split, g in J.groupby("split"):
    r1 = spearmanr(g.nn_dist, g.motion_mm); r2 = spearmanr(g.nn_dist, g.contact_mm)
    sq = g.groupby("sequence_id")[["nn_dist", "motion_mm"]].mean()
    r3 = spearmanr(sq.nn_dist, sq.motion_mm)
    print(f"  {split}: window motion {r1.correlation:+.3f} contact {r2.correlation:+.3f} | "
          f"sequence motion {r3.correlation:+.3f} (n={len(g)}, {len(sq)} seq)")
print("\n=== G3: within-sequence (does it track the shift inside one take?) ===")
for split, g in J.groupby("split"):
    d = g.copy()
    for c in ("nn_dist", "motion_mm"):
        d[c] = d[c] - d.groupby("sequence_id")[c].transform("mean")
    m = d[d.groupby("sequence_id").sequence_id.transform("size") > 1]
    if len(m) > 10:
        print(f"  {split}: {spearmanr(m.nn_dist, m.motion_mm).correlation:+.3f} (n={len(m)})")
print("\n->", G)
