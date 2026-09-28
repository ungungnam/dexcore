#!/usr/bin/env python
"""Nearest-neighbour retrieval baseline and input-shift measure, in the model's own conditioning space.

Two hypotheses, one computation. G1 (memorisation): if the trained model behaves like a lookup table,
copying the hand of the nearest TRAINING window in conditioning space should score about as well as
the model. G3 (input shift): if test windows sit far from the training manifold, the distance to that
nearest window should predict the model's error.

Conditioning = per-frame object BPS (3072) + global states (8), exactly what stage 1 receives.
Window descriptor = [mean over frames, std over frames] of the 3080 channels, z-scored per channel on
the training descriptors. Retrieval copies the neighbour's 64-frame hand keypoints (600 = 100 pts x
2 hands x 3, in each window's own target frame) and its recorded contact map. Train queries use
leave-sequence-out.
"""
import pathlib as _pl
import sys as _sys

# scripts/select.py shadows the stdlib `select` when this directory leads sys.path; drop it first
_HERE = str(_pl.Path(__file__).resolve().parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]

import argparse, json, logging
from pathlib import Path
import numpy as np, pandas as pd

R = Path("/result/uhnam/dexcore/bimart_taco"); S = R / "train_store"
H = 64; KP = 600


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stride", type=int, default=4, help="train window stride for the index")
    p.add_argument("--train-queries", type=int, default=640)
    p.add_argument("--out", default=str(R / "contact_probe" / "knn"))
    p.add_argument("--root", default=str(R), help="dataset root; the _scene variant lives under bimart_taco_scene")
    p.add_argument("--store", default="train_store", help="store subdirectory under --root")
    a = p.parse_args()
    S = Path(a.root) / a.store
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("knn")
    off = pd.read_csv(S / "offsets.csv")
    bps = np.load(S / "bps.npy", mmap_mode="r"); glo = np.load(S / "global.npy", mmap_mode="r")
    act = np.load(S / "action.npy", mmap_mode="r"); con = np.load(S / "contact.npy", mmap_mode="r")
    base = 8

    def windows(row, stride):
        end = int(row.n_frames_store) - H - base
        return [] if end <= base else list(range(base, end, stride))

    def descriptors(row, starts):
        s0, n = int(row.start), int(row.n_frames_store)
        X = np.concatenate([np.asarray(bps[s0:s0 + n], dtype=np.float32),
                            np.asarray(glo[s0:s0 + n], dtype=np.float32)], axis=1)   # (n, 3080)
        D = np.empty((len(starts), 2 * X.shape[1]), dtype=np.float32)
        for i, st in enumerate(starts):
            w = X[st:st + H]; D[i, :X.shape[1]] = w.mean(0); D[i, X.shape[1]:] = w.std(0)
        return D

    # ---- index over train
    tr = off[off.split == "train"].reset_index(drop=True)
    idx_desc, idx_meta = [], []
    for i, row in tr.iterrows():
        st = windows(row, a.stride)
        if not st: continue
        idx_desc.append(descriptors(row, st))
        idx_meta += [{"seq_i": i, "sequence_id": row.sequence_id, "start": s, "abs_start": int(row.start) + s,
                      "triplet": row.triplet, "verb": row.verb, "tool_mesh": row.tool_mesh,
                      "target_mesh": row.target_mesh} for s in st]
    Xi = np.concatenate(idx_desc); Mi = pd.DataFrame(idx_meta)
    mu, sd = Xi.mean(0), Xi.std(0) + 1e-6
    Zi = (Xi - mu) / sd
    log.info("index: %d train windows (stride %d), %d dims", len(Zi), a.stride, Zi.shape[1])

    # ---- queries: every window of each test split (non-overlapping, as evaluated), + train sample
    rng = np.random.default_rng(0); rows = []
    for split in ("test_1", "test_2", "test_3", "test_4", "train"):
        sub = off[off.split == split].reset_index(drop=True)
        qs = []
        for i, row in sub.iterrows():
            st = windows(row, H if split != "train" else 1)
            qs += [(i, s) for s in st]
        if split == "train":
            qs = [qs[k] for k in rng.choice(len(qs), min(a.train_queries, len(qs)), replace=False)]
        # descriptors per sequence
        by_seq = {}
        for i, s in qs: by_seq.setdefault(i, []).append(s)
        for i, starts in by_seq.items():
            row = sub.iloc[i]
            Zq = (descriptors(row, starts) - mu) / sd
            d2 = (Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zi.T + (Zi ** 2).sum(1)[None, :]
            if split == "train":
                d2[:, (Mi.sequence_id == row.sequence_id).values] = np.inf   # leave-sequence-out
            nn = d2.argmin(1); dist = np.sqrt(np.maximum(d2[np.arange(len(nn)), nn], 0))
            # second-nearest from a DIFFERENT sequence, for a stability check
            for k, s in enumerate(starts):
                q0 = int(row.start) + s; m = Mi.iloc[nn[k]]
                kq = np.asarray(act[q0:q0 + H, :KP]).reshape(H, 200, 3)
                kn = np.asarray(act[m.abs_start:m.abs_start + H, :KP]).reshape(H, 200, 3)
                cq = np.asarray(con[q0:q0 + H]); cn = np.asarray(con[m.abs_start:m.abs_start + H])
                dd = np.linalg.norm(kq - kn, axis=-1)
                rows.append({"split": split, "sequence_id": row.sequence_id, "start": s, "verb": row.verb,
                             "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh, "target_cat": row.target_cat,
                             "nn_dist": float(dist[k]), "nn_sequence_id": m.sequence_id, "nn_start": int(m.start),
                             "nn_same_triplet": bool(m.triplet == row.triplet), "nn_same_verb": bool(m.verb == row.verb),
                             "nn_same_tool_mesh": bool(m.tool_mesh == row.tool_mesh),
                             "nn_same_target_mesh": bool(m.target_mesh == row.target_mesh),
                             "retr_hand_mm": float(dd.mean() * 1000), "retr_hand_L_mm": float(dd[:, :100].mean() * 1000),
                             "retr_hand_R_mm": float(dd[:, 100:].mean() * 1000),
                             "retr_cmap_mm": float(np.abs(cq - cn).mean() * 1000)})
        log.info("%s: %d queries", split, sum(len(v) for v in by_seq.values()))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    D = pd.DataFrame(rows); D.to_csv(out / "knn_windows.csv", index=False)
    (out / "meta.json").write_text(json.dumps({"stride": a.stride, "index_windows": int(len(Zi)), "dims": int(Zi.shape[1])}))
    pd.set_option("display.width", 200)
    print("\n=== retrieval baseline (copy nearest train window's hand / contact map) ===")
    print(D.groupby("split")[["nn_dist", "retr_hand_mm", "retr_hand_L_mm", "retr_hand_R_mm", "retr_cmap_mm",
                              "nn_same_triplet", "nn_same_tool_mesh", "nn_same_target_mesh"]].mean().round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
