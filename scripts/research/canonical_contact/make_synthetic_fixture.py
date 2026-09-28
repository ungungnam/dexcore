"""Build a SMALL synthetic fixture that follows CACHE_FORMAT.md exactly, for testing Steps 5-11.

Creates under --root:
  window_index.csv, samples_index.csv, contact_pairs.csv, contact_pairs_summary.csv   (via step4's builders)
  baselineA_distances.csv, baselineA_metrics.csv                                      (random Chamfer numbers, real layout)
  canonical_contact_cache/<backend>/<category>__<role>.npz  (+ __seq.npz)             backends aligned/normalized/random_perm
Two categories (alpha, beta) x two roles x two hands, 3 verbs, 3 meshes per category, K canonical points.
The synthetic X_soft = verb signature + mesh signature + noise, so verb / mesh structure is recoverable
for 'aligned'; 'normalized' adds noise; 'random_perm' permutes canonical points per sample (destroys it).

Usage: python scripts/make_synthetic_fixture.py --root <scratch>/fixture [--K 512]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_pairs_baselineA as s4  # noqa: E402

VERBS = ["brush", "cut", "stir"]
CATS = ["alpha", "beta"]
MESHES = {"alpha": [1, 2, 3], "beta": [11, 12, 13]}
N_SEQ_PER_TRIPLET = 4   # sequences per (verb, tool_cat)  -> 3 verbs x 2 orderings x 4 = 24 sequences
N_WIN = 2
HANDS = ["L", "R"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    K = args.K

    # ---------------- sequences and window index (real column layout)
    rows = []
    sid = 0
    for verb in VERBS:
        for tool_cat in CATS:
            target_cat = [c for c in CATS if c != tool_cat][0]
            for k in range(N_SEQ_PER_TRIPLET):
                sid += 1
                tool_mesh = MESHES[tool_cat][k % 3]
                target_mesh = MESHES[target_cat][(k + 1) % 3]
                seq = f"({verb}, {tool_cat}, {target_cat})/2023{sid:04d}"
                file = f"({verb}, {tool_cat}, {target_cat})__2023{sid:04d}.npz"
                subject = 1 + (sid % 4)
                split = "train" if sid % 3 else "test_1"
                for w in range(N_WIN):
                    touch = rng.uniform(0.0, 1.0, 4)
                    touch[rng.random(4) < 0.15] = 0.05  # some samples below the 0.2 threshold
                    rows.append(dict(sequence_id=seq, file=file, window=w, start=8 + 64 * w, verb=verb,
                                     triplet=f"({verb}, {tool_cat}, {target_cat})", tool_cat=tool_cat, target_cat=target_cat,
                                     tool_mesh=tool_mesh, target_mesh=target_mesh, split=split, subject=subject,
                                     n_tool=1000, n_target=1000, touch_L_tool=touch[0], touch_R_tool=touch[1],
                                     touch_L_target=touch[2], touch_R_target=touch[3]))
    w = pd.DataFrame(rows)
    w.to_csv(root / "window_index.csv", index=False)
    samples = s4.build_samples(w)
    samples.to_csv(root / "samples_index.csv", index=False)
    pairs, summ = s4.build_pairs(samples)
    pairs.to_csv(root / "contact_pairs.csv", index=False)
    summ.to_csv(root / "contact_pairs_summary.csv", index=False)
    print(f"samples={len(samples)} pairs={len(pairs)}")
    print(pairs.groupby(["category", "role", "hand", "pair_type"]).size().to_string())

    # ---------------- Baseline A (random numbers, real layout incl. comment header)
    dist = pairs[s4.PAIR_KEY].copy()
    dist["chamfer_raw_m"] = np.abs(rng.normal(0.04, 0.02, len(dist)))
    dist["chamfer_scalednorm"] = dist["chamfer_raw_m"] * 8
    dist["n_pts_i"] = rng.integers(20, 200, len(dist))
    dist["n_pts_j"] = rng.integers(20, 200, len(dist))
    dist["fallback_i"] = 0
    dist["fallback_j"] = 0
    with open(root / "baselineA_distances.csv", "w") as f:
        f.write("# synthetic Baseline A fixture\n")
        dist.to_csv(f, index=False)
    s4.N_BOOT = 100
    met = s4.aggregate(dist, 2)
    met.to_csv(root / "baselineA_metrics.csv", index=False)

    # ---------------- canonical cache
    verb_sig = {v: rng.random(K) for v in VERBS}
    mesh_sig = {}
    for c in CATS:
        for m in MESHES[c]:
            mesh_sig[m] = rng.random(K)
    canon = {c: rng.normal(size=(K, 3)) for c in CATS}
    for backend in ("aligned", "normalized", "random_perm"):
        d = root / "canonical_contact_cache" / backend
        d.mkdir(parents=True, exist_ok=True)
        for cat in CATS:
            for role in ("tool", "target"):
                cat_col, mesh_col = f"{role}_cat", f"{role}_mesh"
                cp_col = "target_cat" if role == "tool" else "tool_cat"
                ws = w[w[cat_col] == cat]
                meta = {k: [] for k in ["sequence_id", "window", "start", "hand", "mesh_id", "verb", "triplet",
                                        "counterpart_cat", "split", "subject", "touch_frac"]}
                Xs, Xh = [], []
                for r in ws.itertuples(index=False):
                    for hand in HANDS:
                        x = 0.6 * verb_sig[r.verb] + 0.3 * mesh_sig[getattr(r, mesh_col)] + 0.1 * rng.random(K)
                        if backend == "normalized":
                            x = x + 0.4 * rng.random(K)
                        if backend == "random_perm":
                            x = x[rng.permutation(K)]
                        x = np.clip(x, 0, 1).astype(np.float32)
                        Xs.append(x)
                        Xh.append((x > 0.5).astype(np.float32))
                        meta["sequence_id"].append(r.sequence_id); meta["window"].append(r.window)
                        meta["start"].append(r.start); meta["hand"].append(hand)
                        meta["mesh_id"].append(getattr(r, mesh_col)); meta["verb"].append(r.verb)
                        meta["triplet"].append(r.triplet); meta["counterpart_cat"].append(getattr(r, cp_col))
                        meta["split"].append(r.split); meta["subject"].append(r.subject)
                        meta["touch_frac"].append(getattr(r, f"touch_{hand}_{role}"))
                Xs = np.stack(Xs); Xh = np.stack(Xh)
                cov = rng.random(Xs.shape) < 0.9
                arrays = dict(X_soft=Xs, X_hard=Xh, coverage=cov, canonical_points=canon[cat].astype(np.float32),
                              sequence_id=np.array(meta["sequence_id"]), window=np.array(meta["window"], dtype=np.int64),
                              start=np.array(meta["start"], dtype=np.int64), hand=np.array(meta["hand"]),
                              mesh_id=np.array(meta["mesh_id"], dtype=np.int64), verb=np.array(meta["verb"]),
                              triplet=np.array(meta["triplet"]), counterpart_cat=np.array(meta["counterpart_cat"]),
                              split=np.array(meta["split"]), subject=np.array(meta["subject"], dtype=np.int64),
                              touch_frac=np.array(meta["touch_frac"], dtype=np.float32))
                np.savez_compressed(d / f"{cat}__{role}.npz", **arrays)
                # whole-sequence averages: M = n_sequences x 2 hands
                mdf = pd.DataFrame({k: v for k, v in meta.items()})
                mdf["row"] = np.arange(len(mdf))
                seq_rows = []
                for (s, h), g in mdf.groupby(["sequence_id", "hand"], sort=True):
                    seq_rows.append((g.row.values, g.iloc[0]))
                sX = np.stack([Xs[idx].mean(0) for idx, _ in seq_rows])
                sH = np.stack([Xh[idx].mean(0) for idx, _ in seq_rows])
                sarr = dict(X_soft=sX, X_hard=sH, coverage=np.stack([cov[idx].any(0) for idx, _ in seq_rows]),
                            canonical_points=canon[cat].astype(np.float32))
                for k in meta:
                    if k == "touch_frac":
                        sarr[k] = np.array([mdf.loc[idx, "touch_frac"].mean() for idx, _ in seq_rows], dtype=np.float32)
                    elif k in ("window", "start"):
                        sarr[k] = np.full(len(seq_rows), -1, dtype=np.int64)
                    else:
                        sarr[k] = np.array([m0[k] for _, m0 in seq_rows])
                np.savez_compressed(d / f"{cat}__{role}__seq.npz", **sarr)
                print(f"{backend}/{cat}__{role}: M={len(Xs)} K={K}")
    (root / "figures_data").mkdir(exist_ok=True)
    print("fixture written to", root)


if __name__ == "__main__":
    main()
