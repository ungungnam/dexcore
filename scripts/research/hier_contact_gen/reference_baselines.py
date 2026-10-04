#!/usr/bin/env python
"""Data-only reference points for the initial-contact sampler (DC_DATASET=<ds>): for every test
sequence, held static over the T frames (so the rows are comparable with the 'held static' sampler
rows and with B0):
  ref_mean_static        the mean first-contact map of the TRAINING sequences of the same group and
                         mesh (the deterministic 'predict the average' baseline; group only if the
                         mesh has no training sequence)
  ref_take_static        the first-contact map of a random training sequence of the same group and
                         mesh, K = 1 / best-of-10 (by the static E_C, the same rule as the sampler)
                         = what best-of-K over the EMPIRICAL distribution achieves
  ref_nearest_static     the training first-contact map closest to the test C_0 (oracle upper bound
                         of any retrieval)
Writes results/reference_baselines.csv (per example) and prints the means.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import hc_common as H

K = 10


def main():
    z, meta = H.load_sequences()
    man = json.load(open(H.OUT / "manifests" / "fixed0.json"))
    tr, te = np.asarray(man["train"]), np.asarray(man["test"])
    X = z["C"].astype(np.float32)
    C0 = X[:, 0]
    key = list(zip(meta.group, meta.mesh_id))
    pool_mesh, pool_group = {}, {}
    for i in tr:
        pool_mesh.setdefault(key[i], []).append(i); pool_group.setdefault(meta.group[i], []).append(i)
    rng = np.random.default_rng(0)
    rows = []
    for i in te:
        cand = pool_mesh.get(key[i]) or pool_group.get(meta.group[i]) or list(tr)
        cand = np.asarray(cand)
        gt = X[i]                                                     # (T,512)
        static_err = lambda s0: float(np.linalg.norm(gt - s0[None], axis=1).mean())
        mean0 = C0[cand].mean(0)
        pick = rng.choice(cand, size=min(K, len(cand)), replace=len(cand) < K)
        errs = np.array([static_err(C0[j]) for j in pick])
        d = np.linalg.norm(C0[cand] - C0[i][None], axis=1); near = cand[np.argmin(d)]
        base = dict(example=int(i), group=meta.group[i], take_key=meta.take_key[i], n_candidates=len(cand), same_mesh=key[i] in pool_mesh)
        rows.append(dict(base, model="ref_mean_static", K=0, E_C=static_err(mean0), s0_err=float(np.linalg.norm(mean0 - C0[i]))))
        rows.append(dict(base, model="ref_take_static", K=1, E_C=float(errs[0]), s0_err=float(np.linalg.norm(C0[pick[0]] - C0[i]))))
        j = pick[np.argmin(errs)]
        rows.append(dict(base, model="ref_take_static", K=K, E_C=float(errs.min()), s0_err=float(np.linalg.norm(C0[j] - C0[i]))))
        rows.append(dict(base, model="ref_take_static", K=-1, E_C=float(errs.mean()), s0_err=float(np.mean([np.linalg.norm(C0[j] - C0[i]) for j in pick]))))
        rows.append(dict(base, model="ref_nearest_static", K=0, E_C=static_err(C0[near]), s0_err=float(d.min())))
    R = pd.DataFrame(rows)
    (H.OUT / "results").mkdir(exist_ok=True)
    R.to_csv(H.OUT / "results" / "reference_baselines.csv", index=False)
    out = []
    for (m, k), g in R.groupby(["model", "K"]):
        e = H.cluster_bootstrap(g.E_C.values, g.take_key.values, n_boot=500); s = H.cluster_bootstrap(g.s0_err.values, g.take_key.values, n_boot=500)
        out.append(dict(model=m, K=k, n=len(g), E_C=e[0], E_C_lo=e[1], E_C_hi=e[2], s0_err=s[0], s0_err_lo=s[1], s0_err_hi=s[2], E_C_macro=float(g.groupby("group").E_C.mean().mean())))
    S = pd.DataFrame(out); S.to_csv(H.OUT / "results" / "reference_baselines_summary.csv", index=False)
    print(H.DATASET, f"({len(te)} test sequences; {int(R[R.model == 'ref_mean_static'].same_mesh.sum())} with a same-mesh training pool)")
    print(S.round(3).to_string())


if __name__ == "__main__":
    main()
