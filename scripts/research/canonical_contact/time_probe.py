#!/usr/bin/env python
"""Is time a separable axis, or is it hidden under identity?

The decomposition says the within-take term is large but a ridge on the raw vector cannot read
phase. Both can be true if identity lives in the take's OFFSET and time in the deviation from it.
This tests exactly that, with the take mean removed the way a test take could remove it (its own
mean, no labels): held-out-sequence folds, two targets (phase, mesh id), two feature sets (raw,
take-centred).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/result/uhnam/dexcore/canonical_contact/time_decomp")
RNG = np.random.default_rng(0)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("probe")


def run(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
    M["take_id"] = M.sequence_id.astype(str)
    keep = M.groupby("mesh_id").sequence_id.transform("nunique").values >= 2
    X, M = X[keep], M[keep].reset_index(drop=True)
    idx = RNG.choice(len(X), min(8000, len(X)), replace=False)
    X, M = X[idx], M.iloc[idx].reset_index(drop=True)
    C = np.empty_like(X)                                  # take-centred (each take minus its own mean)
    for t, i in M.groupby("take_id").indices.items():
        C[i] = X[i] - X[i].mean(0)
    out = {"category": path.stem.split("__")[0], "hand": path.stem.split("__")[2], "n": len(X),
           "n_take": M.take_id.nunique(), "n_mesh": M.mesh_id.nunique()}
    gkf = GroupKFold(n_splits=min(5, M.take_id.nunique()))
    res = {k: [] for k in ("phase_raw", "phase_centred", "mesh_raw", "mesh_centred")}
    for tr, te in gkf.split(X, groups=M.take_id):
        for tag, F in (("raw", X), ("centred", C)):
            sc = StandardScaler().fit(F[tr]); A, B = sc.transform(F[tr]), sc.transform(F[te])
            p = Ridge(alpha=1.0).fit(A, M.phase.iloc[tr]).predict(B)
            y = M.phase.iloc[te].values
            res[f"phase_{tag}"].append(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
            if M.mesh_id.iloc[tr].nunique() > 1:
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(A, M.mesh_id.iloc[tr])
                res[f"mesh_{tag}"].append((clf.predict(B) == M.mesh_id.iloc[te].values).mean())
    for k, v in res.items():
        out[k] = float(np.mean(v)) if v else np.nan
    out["mesh_chance"] = float(M.mesh_id.value_counts(normalize=True).max())
    return out


rows = [run(p) for p in sorted(ROOT.glob("*__*__*.npz")) if p.name != "time_vs_mesh.csv"]
D = pd.DataFrame(rows)
D.to_csv(ROOT / "time_probe.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== phase and mesh, from the raw vector and from the take-centred vector ===")
print(D[["category", "hand", "n", "n_mesh", "phase_raw", "phase_centred",
         "mesh_raw", "mesh_centred", "mesh_chance"]].round(3).to_string(index=False))
print("\n->", ROOT / "time_probe.csv")
