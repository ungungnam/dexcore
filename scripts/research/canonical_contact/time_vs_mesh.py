#!/usr/bin/env python
"""Decompose a TACO contact map into mesh identity, take, and time.

Part B measured one vector per 64-frame window, so time was averaged away before anything was
compared; its A pairs (same verb, same mesh, different take) silently mix take variability with
whatever phase the two windows happened to sit at. Here each frame is its own canonical vector, so
the four sources can be separated on ONE scale (L2 between 512-d canonical vectors of the same
category, `normalized` backend):

  (1) nested variance components   mesh / take within mesh / time within take
  (2) a distance ladder            d(same take, delta t) against d(other take, matched phase)
                                   and d(other mesh, matched phase)
  (3) a shared phase curve         how much of the time term is a category-wide arc rather than
                                   take-specific wobble
  (4) probes                       mesh identity vs normalised phase, from the same vector
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/result/uhnam/dexcore/canonical_contact/time_decomp")
RNG = np.random.default_rng(0)
CAP = 4000                      # pairs sampled per (group, pair type)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tvm")


def load(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    M = pd.DataFrame({k: z[k] for k in z.files if k != "X"})
    M["take_id"] = M.sequence_id.astype(str)
    return X, M


def var_components(X, M):
    """SS_total = SS_mesh + SS_take|mesh + SS_time|take, exactly (nested means)."""
    gm = X.mean(0)
    ss_tot = ((X - gm) ** 2).sum()
    take_mean = {t: X[i].mean(0) for t, i in M.groupby("take_id").indices.items()}
    mesh_mean = {m: X[i].mean(0) for m, i in M.groupby("mesh_id").indices.items()}
    ss_mesh = sum(len(i) * ((mesh_mean[m] - gm) ** 2).sum() for m, i in M.groupby("mesh_id").indices.items())
    ss_take = 0.0
    for t, i in M.groupby("take_id").indices.items():
        m = M.mesh_id.values[i[0]]
        ss_take += len(i) * ((take_mean[t] - mesh_mean[m]) ** 2).sum()
    ss_time = sum(((X[i] - take_mean[t]) ** 2).sum() for t, i in M.groupby("take_id").indices.items())
    return dict(ss_total=ss_tot, mesh=ss_mesh / ss_tot, take=ss_take / ss_tot, time=ss_time / ss_tot,
                n=len(X), n_take=M.take_id.nunique(), n_mesh=M.mesh_id.nunique())


def d(X, a, b):
    return np.linalg.norm(X[a] - X[b], axis=1)


def ladder(X, M):
    """Mean L2 for pair types that hold mesh / take / phase fixed in different combinations."""
    out = {}
    take_idx = M.groupby("take_id").indices
    ph = M.phase.values
    # --- same take, by |delta phase|
    within = {k: [] for k in ("dt_tiny", "dt_small", "dt_mid", "dt_large")}
    for t, i in take_idx.items():
        if len(i) < 4:
            continue
        a = RNG.choice(i, min(len(i), 60), replace=True)
        b = RNG.choice(i, min(len(i), 60), replace=True)
        keep = a != b
        a, b = a[keep], b[keep]
        gap = np.abs(ph[a] - ph[b])
        for key, lo, hi in (("dt_tiny", 0, .05), ("dt_small", .05, .2), ("dt_mid", .2, .5), ("dt_large", .5, 1.01)):
            m = (gap >= lo) & (gap < hi)
            if m.any():
                within[key].append(d(X, a[m], b[m]))
    for k, v in within.items():
        out[k] = float(np.concatenate(v).mean()) if v else np.nan
        out["n_" + k] = int(sum(len(x) for x in v))
    # --- different take / different mesh, phase matched (|dphase| < 0.1) and unmatched
    mesh = M.mesh_id.values, 
    mesh = M.mesh_id.values
    verb = M.verb.values
    take = M.take_id.values
    n = len(M)
    for name, same_mesh in (("other_take", True), ("other_mesh", False)):
        for matched in (True, False):
            acc = []
            tries = 0
            while len(acc) < CAP and tries < 60:
                tries += 1
                a = RNG.integers(0, n, CAP * 4)
                b = RNG.integers(0, n, CAP * 4)
                ok = (take[a] != take[b]) & (verb[a] == verb[b])
                ok &= (mesh[a] == mesh[b]) if same_mesh else (mesh[a] != mesh[b])
                if matched:
                    ok &= np.abs(ph[a] - ph[b]) < 0.1
                if ok.any():
                    acc.append(d(X, a[ok], b[ok]))
            v = np.concatenate(acc)[:CAP] if acc else np.array([np.nan])
            out[f"{name}{'_matched' if matched else '_anyphase'}"] = float(v.mean())
            out[f"n_{name}{'_matched' if matched else '_anyphase'}"] = int(len(v))
    return out


def phase_curve(X, M, bins=10):
    """How much of the within-take variation is a category-wide arc: variance of the shared
    phase profile over the total within-take variance."""
    take_idx = M.groupby("take_id").indices
    Z = np.empty_like(X)
    for t, i in take_idx.items():
        Z[i] = X[i] - X[i].mean(0)                       # take-centred: pure within-take signal
    b = np.clip((M.phase.values * bins).astype(int), 0, bins - 1)
    prof = np.stack([Z[b == k].mean(0) if (b == k).any() else np.zeros(X.shape[1]) for k in range(bins)])
    ss_within = (Z ** 2).sum()
    ss_shared = sum((b == k).sum() * (prof[k] ** 2).sum() for k in range(bins))
    return dict(shared_phase_frac=ss_shared / ss_within, ss_within=ss_within,
                prof_peak=float(np.abs(prof).max()))


def probes(X, M):
    """Mesh identity (accuracy) and phase (R^2) from the same vector, sequence-level folds."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    keep = M.groupby("mesh_id").sequence_id.transform("nunique") >= 2
    Xk, Mk = X[keep.values], M[keep.values].reset_index(drop=True)
    if Mk.mesh_id.nunique() < 2:
        return {}
    idx = RNG.choice(len(Xk), min(6000, len(Xk)), replace=False)
    Xk, Mk = Xk[idx], Mk.iloc[idx].reset_index(drop=True)
    gkf = GroupKFold(n_splits=min(5, Mk.sequence_id.nunique()))
    acc, r2 = [], []
    for tr, te in gkf.split(Xk, groups=Mk.sequence_id):
        sc = StandardScaler().fit(Xk[tr])
        A, B = sc.transform(Xk[tr]), sc.transform(Xk[te])
        if Mk.mesh_id.iloc[tr].nunique() > 1:
            clf = LogisticRegression(max_iter=1500, C=1.0).fit(A, Mk.mesh_id.iloc[tr])
            acc.append((clf.predict(B) == Mk.mesh_id.iloc[te].values).mean())
        rg = Ridge(alpha=1.0).fit(A, Mk.phase.iloc[tr])
        p = rg.predict(B); y = Mk.phase.iloc[te].values
        r2.append(1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9))
    maj = Mk.mesh_id.value_counts(normalize=True).max()
    return dict(mesh_acc=float(np.mean(acc)) if acc else np.nan, mesh_chance=float(maj),
                phase_r2=float(np.mean(r2)), n_probe=len(Xk))


def main():
    rows = []
    for p in sorted(ROOT.glob("*.npz")):
        cat, role, hand = p.stem.split("__")
        X, M = load(p)
        r = dict(category=cat, role=role, hand=hand)
        r |= var_components(X, M)
        r |= ladder(X, M)
        r |= phase_curve(X, M)
        r |= probes(X, M)
        rows.append(r)
        log.info("%s/%s/%s  var mesh %.2f take %.2f time %.2f | mesh acc %.2f phase R2 %.2f",
                 cat, role, hand, r["mesh"], r["take"], r["time"],
                 r.get("mesh_acc", np.nan), r.get("phase_r2", np.nan))
    D = pd.DataFrame(rows)
    D.to_csv(ROOT / "time_vs_mesh.csv", index=False)
    pd.set_option("display.width", 250)
    print("\n=== (1) variance components (share of total) ===")
    print(D[["category", "role", "hand", "n", "n_mesh", "n_take", "mesh", "take", "time"]].round(3).to_string(index=False))
    print("\n=== (2) distance ladder (mean L2) ===")
    cols = ["category", "hand", "dt_tiny", "dt_small", "dt_mid", "dt_large",
            "other_take_matched", "other_take_anyphase", "other_mesh_matched", "other_mesh_anyphase"]
    print(D[cols].round(3).to_string(index=False))
    print("\n=== (3) shared phase arc, and (4) probes ===")
    print(D[["category", "hand", "shared_phase_frac", "mesh_acc", "mesh_chance", "phase_r2"]].round(3).to_string(index=False))
    print("\n->", ROOT / "time_vs_mesh.csv")


if __name__ == "__main__":
    main()
