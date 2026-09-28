#!/usr/bin/env python
"""G3 verification, step 1: recompute the conditioning-space NN for every test_1 (and train-query) window,
decompose the distance into blocks (BPS tool / BPS target / rot / trans / target scale / tool scale, each
split into mean- and std-halves), run block-wise NN searches, and compute a hand-space (600-d kp) NN
distance with the same [mean, std] descriptor. Output: one row per query window."""
import sys, pathlib as _pl
_HERE = str(_pl.Path(__file__).resolve().parent); sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
import json, logging, numpy as np, pandas as pd
from pathlib import Path

R = Path("/result/uhnam/dexcore/bimart_taco"); S = R / "train_store"
OUT = R / "contact_probe" / "gen" / "verify" / "g3"; OUT.mkdir(parents=True, exist_ok=True)
H = 64; KP = 600; base = 8; STRIDE = 4
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("g3")

off = pd.read_csv(S / "offsets.csv")
bps = np.load(S / "bps.npy", mmap_mode="r"); glo = np.load(S / "global.npy", mmap_mode="r"); act = np.load(S / "action.npy", mmap_mode="r")
C = 3080
# channel blocks of the 3080 per-frame conditioning
BLK = {"bps_tool": np.arange(0, 1536), "bps_target": np.arange(1536, 3072), "rot": np.arange(3072, 3075),
       "trans": np.arange(3075, 3078), "tscale": np.array([3078]), "toolscale": np.array([3079])}
# descriptor index sets (mean half = 0..3079, std half = 3080..6159)
DBLK = {}
for k, v in BLK.items():
    DBLK[k + "_mean"] = v; DBLK[k + "_std"] = v + C; DBLK[k] = np.concatenate([v, v + C])
DBLK["all_mean"] = np.arange(0, C); DBLK["all_std"] = np.arange(C, 2 * C)

def windows(row, stride):
    end = int(row.n_frames_store) - H - base
    return [] if end <= base else list(range(base, end, stride))

def descriptors(row, starts):
    s0, n = int(row.start), int(row.n_frames_store)
    X = np.concatenate([np.asarray(bps[s0:s0 + n], dtype=np.float32), np.asarray(glo[s0:s0 + n], dtype=np.float32)], axis=1)
    K = np.asarray(act[s0:s0 + n, :KP], dtype=np.float32)
    D = np.empty((len(starts), 2 * C), np.float32); Dk = np.empty((len(starts), 2 * KP), np.float32)
    raw = np.empty((len(starts), 8), np.float32)  # window-mean of the 8 global states (un-normalised)
    for i, st in enumerate(starts):
        w = X[st:st + H]; D[i, :C] = w.mean(0); D[i, C:] = w.std(0)
        k = K[st:st + H]; Dk[i, :KP] = k.mean(0); Dk[i, KP:] = k.std(0)
        raw[i] = w[:, 3072:].mean(0)
    return D, Dk, raw

# ---- index over train (identical to knn_baseline_taco.py)
tr = off[off.split == "train"].reset_index(drop=True)
idx_desc, idx_k, idx_meta = [], [], []
for i, row in tr.iterrows():
    st = windows(row, STRIDE)
    if not st: continue
    D, Dk, _ = descriptors(row, st); idx_desc.append(D); idx_k.append(Dk)
    idx_meta += [{"sequence_id": row.sequence_id, "start": s, "abs_start": int(row.start) + s, "triplet": row.triplet,
                  "verb": row.verb, "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh} for s in st]
Xi = np.concatenate(idx_desc); Ki = np.concatenate(idx_k); Mi = pd.DataFrame(idx_meta)
mu, sd = Xi.mean(0), Xi.std(0) + 1e-6; Zi = (Xi - mu) / sd; del Xi
muk, sdk = Ki.mean(0), Ki.std(0) + 1e-6; Zk = (Ki - muk) / sdk; del Ki
log.info("index: %d windows, %d cond dims, %d hand dims", len(Zi), Zi.shape[1], Zk.shape[1])
Zi_sq = (Zi ** 2).sum(1); Zk_sq = (Zk ** 2).sum(1)
blk_sq = {k: (Zi[:, v] ** 2).sum(1) for k, v in DBLK.items()}

def nn_search(Zq, Zref, ref_sq, mask_seq=None):
    d2 = (Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zref.T + ref_sq[None, :]
    if mask_seq is not None: d2[:, mask_seq] = np.inf
    nn = d2.argmin(1); return nn, np.sqrt(np.maximum(d2[np.arange(len(nn)), nn], 0))

rng = np.random.default_rng(0); rows = []
for split in ("test_1", "train"):
    sub = off[off.split == split].reset_index(drop=True); qs = []
    for i, row in sub.iterrows():
        qs += [(i, s) for s in windows(row, H if split != "train" else 1)]
    if split == "train":
        qs = [qs[k] for k in rng.choice(len(qs), min(640, len(qs)), replace=False)]
    by_seq = {}
    for i, s in qs: by_seq.setdefault(i, []).append(s)
    for i, starts in by_seq.items():
        row = sub.iloc[i]; D, Dk, raw = descriptors(row, starts)
        Zq = (D - mu) / sd; Zqk = (Dk - muk) / sdk
        mask = (Mi.sequence_id == row.sequence_id).values if split == "train" else None
        nn, dist = nn_search(Zq, Zi, Zi_sq, mask)
        nnk, distk = nn_search(Zqk, Zk, Zk_sq, mask)
        # block-wise NN searches (each block alone)
        bw = {}
        for k in ("bps_tool", "bps_target", "rot", "trans", "tscale", "toolscale", "all_mean", "all_std"):
            v = DBLK[k]; nb, db = nn_search(Zq[:, v], Zi[:, v], blk_sq[k], mask); bw[k] = (nb, db)
        for j, s in enumerate(starts):
            m = Mi.iloc[nn[j]]; q0 = int(row.start) + s
            diff = Zq[j] - Zi[nn[j]]
            r = {"split": split, "sequence_id": row.sequence_id, "start": s, "verb": row.verb, "triplet": row.triplet,
                 "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh, "target_cat": row.target_cat, "tool_cat": row.tool_cat,
                 "nn_dist": float(dist[j]), "nn_sequence_id": m.sequence_id, "nn_start": int(m.start),
                 "nn_same_triplet": bool(m.triplet == row.triplet), "nn_same_verb": bool(m.verb == row.verb),
                 "nn_same_tool_mesh": bool(m.tool_mesh == row.tool_mesh), "nn_same_target_mesh": bool(m.target_mesh == row.target_mesh),
                 "hand_nn_dist": float(distk[j]), "hand_nn_sequence_id": Mi.iloc[nnk[j]].sequence_id,
                 "hand_nn_same_verb": bool(Mi.iloc[nnk[j]].verb == row.verb),
                 "w_rot_x": float(raw[j, 0]), "w_rot_y": float(raw[j, 1]), "w_rot_z": float(raw[j, 2]),
                 "w_trans_x": float(raw[j, 3]), "w_trans_y": float(raw[j, 4]), "w_trans_z": float(raw[j, 5]),
                 "tscale": float(raw[j, 6]), "toolscale": float(raw[j, 7])}
            # per-block (squared) distance to the SAME nearest neighbour, and per-dim RMS version
            for k, v in DBLK.items():
                r[f"d2_{k}"] = float((diff[v] ** 2).sum()); r[f"rms_{k}"] = float(np.sqrt((diff[v] ** 2).mean()))
            for k, (nb, db) in bw.items():
                r[f"bw_{k}_dist"] = float(db[j]); r[f"bw_{k}_same_verb"] = bool(Mi.iloc[nb[j]].verb == row.verb)
            # hand retrieval error of the hand-space NN and of the conditioning NN (recorded kp)
            kq = np.asarray(act[q0:q0 + H, :KP]).reshape(H, 200, 3)
            for tag, idx in (("cond", nn[j]), ("hand", nnk[j])):
                mm = Mi.iloc[idx]; kn = np.asarray(act[mm.abs_start:mm.abs_start + H, :KP]).reshape(H, 200, 3)
                r[f"retr_hand_mm_{tag}nn"] = float(np.linalg.norm(kq - kn, axis=-1).mean() * 1000)
            rows.append(r)
    log.info("%s: %d queries", split, sum(len(v) for v in by_seq.values()))
F = pd.DataFrame(rows); F.to_csv(OUT / "g3_features.csv", index=False)
json.dump({"index_windows": int(len(Zi)), "blocks": {k: int(len(v)) for k, v in DBLK.items()}}, open(OUT / "g3_features_meta.json", "w"), indent=1)
print(F.groupby("split")[["nn_dist", "hand_nn_dist", "retr_hand_mm_condnn", "retr_hand_mm_handnn"]].mean().round(2))
print("-> ", OUT / "g3_features.csv")
