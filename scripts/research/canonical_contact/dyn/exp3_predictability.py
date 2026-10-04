#!/usr/bin/env python
"""Experiment 3: is the temporal contact variation predictable from the OBJECT trajectory?

Per (category, role, hand) group, one small predictor maps an input family to the 512-D contact
vector of a frame. The predictor (ridge regression, and an MLP 256-256 with the same schedule) is
identical across the four input families; only the input changes:
  A  geometry only                 G = mesh descriptor: PCA (8 comps, fit on the training meshes)
                                       of the nearest-surface distance + coverage at the 512
                                       canonical points, plus metric radius and extent     [12]
  B  geometry + current state      G, O_t                                         [+30]
  C  geometry + local context      G, O_{t-8..t+8 step 2} (9 states, +-0.27 s)    [+270]
  D  geometry + full window        G, O over the 64-frame window (32 states) + position in window
  Dc geometry + coarse full window G, O over the 64-frame window at stride 8 (8 states) + position
                                   (same temporal extent as D with a quarter of the dimensions, to
                                   separate 'more context' from 'more input dimensions')
Object state O_t (build_object_states.py): tool pose in the target frame (pos + 6D rot), target
world pose (6D rot + translation rel. frame 0), tool and target linear/angular velocities.

Splits (3 folds each, training on 2/3):
  take   sequences split 3-way, stratified by mesh   (all meshes seen)
  mesh   meshes split 3-way                           (test meshes unseen)
Baselines: category mean (train), per-mesh mean (train; = category mean when the mesh is unseen),
previous sampled frame of the truth (2 original frames earlier), and the oracle sequence mean.
Metrics per frame, averaged per test sequence, then sequence-bootstrapped per group.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

import dc_common as C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("exp3")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOCAL_OFFSETS = list(range(-8, 9, 2))
FOLDS = 3
CONDS = ["A_geometry", "B_current_state", "C_local_context", "D_full_window", "Dc_coarse_window"]
OUT = C.OUT / "exp3"
PRED_DIR = OUT / "preds"


class ObjectStates:
    def __init__(self):
        z = np.load(C.CACHE / "object_states.npz", allow_pickle=True)
        self.off = {s: (int(a), int(b)) for s, a, b in zip(z["sequence_id"], z["offset"][:-1], z["offset"][1:])}
        self.S = z["states"].astype(np.float32)

    def get(self, sid, frames):
        a, b = self.off[sid]
        f = np.clip(np.asarray(frames), 0, b - a - 1)
        return self.S[a + f]


N_GEO_PCA = 8


def geometry_features(geo, M, train_meshes):
    """Per-frame geometry descriptor: PCA of the 1024-D surface descriptor fitted on the TRAINING
    meshes only (so an unseen mesh is projected, never memorised), plus radius and extent."""
    gidx = {m: i for i, m in enumerate(geo["mesh_ids"])}
    raw = np.concatenate([geo["nearest"], geo["coverage"].astype(np.float32)], 1).astype(np.float32)
    tr = np.array([gidx[m] for m in train_meshes])
    mu = raw[tr].mean(0, keepdims=True)
    U, s, Vt = np.linalg.svd(raw[tr] - mu, full_matrices=False)
    k = min(N_GEO_PCA, len(tr) - 1)
    Z = (raw - mu) @ Vt[:k].T
    Z = Z / (Z[tr].std(0, keepdims=True) + 1e-6)
    desc = np.concatenate([Z, geo["radius_m"][:, None], geo["extent_m"]], 1).astype(np.float32)
    gi = np.array([gidx[m] for m in M.mesh_id.values])
    return desc[gi]


def build_inputs(M, states):
    """Trajectory inputs, cond -> (n, d) float32 (geometry is appended per fold)."""
    n = len(M)
    D = states.S.shape[1]
    O_t = np.zeros((n, D), np.float32)
    O_loc = np.zeros((n, len(LOCAL_OFFSETS) * D), np.float32)
    O_win = np.zeros((n, 32 * D + 1), np.float32)
    O_coarse = np.zeros((n, 8 * D + 1), np.float32)
    fr = M.frame.values; fiw = M.frame_in_window.values
    for sid, idx in M.groupby("sequence_id").indices.items():
        f = fr[idx]
        O_t[idx] = states.get(sid, f)
        O_loc[idx] = np.concatenate([states.get(sid, f + o) for o in LOCAL_OFFSETS], 1)
        s = f - fiw[idx]
        O_win[idx, :-1] = np.concatenate([states.get(sid, s + o) for o in range(0, 64, 2)], 1)
        O_win[idx, -1] = fiw[idx] / 64.0
        O_coarse[idx, :-1] = np.concatenate([states.get(sid, s + o) for o in range(0, 64, 8)], 1)
        O_coarse[idx, -1] = fiw[idx] / 64.0
    return {"A_geometry": np.zeros((n, 0), np.float32), "B_current_state": O_t,
            "C_local_context": O_loc, "D_full_window": O_win, "Dc_coarse_window": O_coarse}


def standardise(tr, *others):
    """Train-set z-scores; features constant on the training set keep scale 1 (so a held-out mesh
    whose descriptor differs there is not blown up), and everything is clipped to +-6."""
    mu, sd = tr.mean(0, keepdims=True), tr.std(0, keepdims=True)
    sd = np.where(sd < 1e-3, 1.0, sd)
    f = lambda X: np.clip((X - mu) / sd, -6, 6).astype(np.float32)
    return (f(tr),) + tuple(f(o) for o in others)


def fit_ridge(Xtr, Ytr, Xva, Yva):
    Xtr_t = torch.from_numpy(Xtr).to(DEV); Ytr_t = torch.from_numpy(Ytr).to(DEV)
    Xva_t = torch.from_numpy(Xva).to(DEV); Yva_t = torch.from_numpy(Yva).to(DEV)
    ones = lambda X: torch.cat([X, torch.ones(len(X), 1, device=DEV)], 1)
    A = ones(Xtr_t); Av = ones(Xva_t)
    AtA = A.T @ A; AtY = A.T @ Ytr_t
    best = None
    for lam in (1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4):
        Wt = torch.linalg.solve(AtA + lam * torch.eye(A.shape[1], device=DEV), AtY)
        err = ((Av @ Wt - Yva_t) ** 2).sum(1).mean().item()
        if best is None or err < best[0]:
            best = (err, lam, Wt)
    return lambda X: (ones(torch.from_numpy(X).to(DEV)) @ best[2]).cpu().numpy(), best[1]


def fit_mlp(Xtr, Ytr, Xva, Yva, seed=0, hidden=256, epochs=60, patience=8, lr=1e-3, bs=1024):
    torch.manual_seed(seed)
    d_in, d_out = Xtr.shape[1], Ytr.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d_in, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(hidden, d_out)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).to(DEV); Ytr_t = torch.from_numpy(Ytr).to(DEV)
    Xva_t = torch.from_numpy(Xva).to(DEV); Yva_t = torch.from_numpy(Yva).to(DEV)
    n = len(Xtr_t); best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = ((net(Xtr_t[j]) - Ytr_t[j]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            va = ((net(Xva_t) - Yva_t) ** 2).sum(1).mean().item()
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state); net.eval()

    def predict(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 8192):
                out.append(net(torch.from_numpy(X[i:i + 8192]).to(DEV)).cpu().numpy())
        return np.concatenate(out)
    return predict, ep + 1


def frame_metrics(pred, Y, P, thr, n_hard):
    """Per-frame metric arrays. pred is clipped at 0 for mass/pattern purposes."""
    pc = np.clip(pred, 0, None)
    m_t, m_p = C.mass(Y), C.mass(pc)
    zero_t = C.is_zero(m_t, n_hard, thr); zero_p = m_p < thr
    Pt, Pp = C.pattern(Y), C.pattern(pc)
    out = dict(raw=np.linalg.norm(pred - Y, axis=1), mass_abs=np.abs(m_p - m_t),
               mass_log=np.abs(np.log1p(m_p) - np.log1p(m_t)),
               pattern_L2=np.where(zero_t, np.nan, np.linalg.norm(Pp - Pt, axis=1)),
               centroid=np.where(zero_t, np.nan, np.linalg.norm(Pp @ P - Pt @ P, axis=1)),
               zero_true=zero_t.astype(float), zero_pred=zero_p.astype(float),
               act_correct=(zero_t == zero_p).astype(float))
    # pattern error at the TRUE mass (isolates the spatial part from the amount part)
    out["raw_at_true_mass"] = np.where(zero_t, np.nan, np.linalg.norm(Pp * m_t[:, None] - Y, axis=1))
    return out


def per_sequence(mets, M_test, extra_cols):
    df = pd.DataFrame(mets); df["sequence_id"] = M_test.sequence_id.values
    agg = df.groupby("sequence_id").mean(numeric_only=True)
    # activation F1 for the zero class is computed at the fold level (rare class), not per sequence
    return agg.reset_index()


def fold_f1(mets):
    zt, zp = mets["zero_true"] > 0.5, mets["zero_pred"] > 0.5
    tp = (zt & zp).sum(); fp = (~zt & zp).sum(); fn = (zt & ~zp).sum()
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return dict(zero_f1=2 * prec * rec / max(prec + rec, 1e-12), zero_prec=prec, zero_rec=rec,
                zero_frac_true=float(zt.mean()), act_acc=float((zt == zp).mean()),
                bal_acc=0.5 * (rec + ((~zt & ~zp).sum() / max((~zt).sum(), 1))))


def main(groups=None, save_preds=True, tag=""):
    thr = C.zero_threshold()
    states = ObjectStates()
    OUT.mkdir(exist_ok=True, parents=True); PRED_DIR.mkdir(exist_ok=True)
    rows_seq, rows_fold = [], []
    for cat, role, hand in (groups or C.GROUPS):
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, stride=C.STRIDE_PAIRS)
        geo = dict(np.load(C.CACHE / f"geometry_normalized__{cat}.npz", allow_pickle=True))
        geo["mesh_ids"] = geo["mesh_ids"].astype(str)
        traj_inputs = build_inputs(M, states)
        seqs = np.array(sorted(M.sequence_id.unique()))
        seq_mesh = M.drop_duplicates("sequence_id").set_index("sequence_id").mesh_id
        meshes = np.array(sorted(M.mesh_id.unique()))
        n_hard = M.n_hard.values
        # fold assignment: take folds stratified by the second identity (mesh or subject); the
        # second split holds out whole values of that identity
        key2 = "mesh_id" if C.SPLIT2 == "mesh" else C.SPLIT2
        seq_key2 = M.drop_duplicates("sequence_id").set_index("sequence_id")[key2].astype(str)
        vals2 = np.array(sorted(seq_key2.unique()))
        take_fold = {}
        for v2 in vals2:
            ss = seqs[seq_key2[seqs].values == v2]
            for i, s in enumerate(ss):
                take_fold[s] = i % FOLDS
        n_fold2 = min(FOLDS, len(vals2))          # the second split needs >= 2 identities
        fold2 = {v: i % n_fold2 for i, v in enumerate(vals2)} if n_fold2 >= 2 else {}
        for split in ("take", C.SPLIT2):
            if split != "take" and n_fold2 < 2:
                log.info("%s: only %d %s identity, second split skipped", g, len(vals2), key2); continue
            for k in range(FOLDS if split == "take" else n_fold2):
                t0 = time.time()
                if split == "take":
                    te_seq = np.array([s for s in seqs if take_fold[s] == k])
                else:
                    te_seq = np.array([s for s in seqs if fold2[seq_key2[s]] == k])
                if len(te_seq) == 0:
                    continue
                tr_seq = np.array([s for s in seqs if s not in set(te_seq)])
                rng = np.random.default_rng(k)
                va_seq = set(rng.choice(tr_seq, max(1, len(tr_seq) // 10), replace=False))
                tr_seq = np.array([s for s in tr_seq if s not in va_seq])
                sel = lambda S: np.isin(M.sequence_id.values, list(S))
                itr, iva, ite = sel(tr_seq), sel(va_seq), sel(te_seq)
                Ytr, Yva, Yte = X[itr], X[iva], X[ite]
                Gf = geometry_features(geo, M, sorted(np.unique(M.mesh_id.values[itr])))
                inputs = {c: np.concatenate([Gf, v], 1) for c, v in traj_inputs.items()}
                Mte = M[ite].reset_index(drop=True)
                preds = {}
                # baselines
                cat_mean = Ytr.mean(0)
                preds["base_cat_mean"] = np.tile(cat_mean, (ite.sum(), 1))
                mm = {m: X[itr & (M.mesh_id.values == m)].mean(0) for m in np.unique(M.mesh_id.values[itr])}
                preds["base_mesh_mean"] = np.stack([mm.get(m, cat_mean) for m in Mte.mesh_id.values])
                prev = np.zeros_like(Yte); prev[:] = np.nan
                fr_te = Mte.frame.values; sid_te = Mte.sequence_id.values
                key = {(s, f): i for i, (s, f) in enumerate(zip(sid_te, fr_te))}
                for i, (s, f) in enumerate(zip(sid_te, fr_te)):
                    j = key.get((s, f - C.STRIDE_PAIRS))
                    prev[i] = Yte[j] if j is not None else Yte[i]
                preds["base_prev_frame"] = prev
                preds["oracle_seq_mean"] = np.concatenate([np.tile(Yte[idx].mean(0), (len(idx), 1))
                                                           for _, idx in sorted(Mte.groupby("sequence_id").indices.items(), key=lambda kv: kv[1][0])])
                # models
                info = {}
                for cond in CONDS:
                    Xtr, Xte, Xva = standardise(inputs[cond][itr], inputs[cond][ite], inputs[cond][iva])
                    pr, lam = fit_ridge(Xtr, Ytr, Xva, Yva)
                    preds[f"ridge_{cond}"] = pr(Xte); info[f"ridge_{cond}"] = dict(lam=lam)
                    pm, ep = fit_mlp(Xtr, Ytr, Xva, Yva, seed=k)
                    preds[f"mlp_{cond}"] = pm(Xte); info[f"mlp_{cond}"] = dict(epochs=ep)
                for name, pr_ in preds.items():
                    mets = frame_metrics(pr_, Yte, P, thr, n_hard[ite])
                    ps = per_sequence(mets, Mte, None)
                    ps["group"] = g; ps["split"] = split; ps["fold"] = k; ps["model"] = name
                    ps["mesh_id"] = seq_mesh[ps.sequence_id].values
                    rows_seq.append(ps)
                    f1 = fold_f1(mets)
                    # temporal error: on consecutive sampled frames of the same sequence
                    dtrue = np.diff(Yte, axis=0); dpred = np.diff(pr_, axis=0)
                    same = (sid_te[1:] == sid_te[:-1]) & (fr_te[1:] - fr_te[:-1] == C.STRIDE_PAIRS)
                    temporal = float(np.linalg.norm(dpred[same] - dtrue[same], axis=1).mean())
                    temporal_true = float(np.linalg.norm(dtrue[same], axis=1).mean())
                    mt, mp = C.mass(Yte), C.mass(np.clip(pr_, 0, None))
                    r2_mass = 1 - ((mp - mt) ** 2).mean() / mt.var()
                    rows_fold.append(dict(group=g, category=cat, role=role, hand=hand, split=split, fold=k, model=name,
                                          n_test_frames=int(ite.sum()), n_test_seq=len(te_seq), n_train_seq=len(tr_seq),
                                          n_test_mesh=int(Mte.mesh_id.nunique()), raw=float(np.nanmean(mets["raw"])),
                                          mass_abs=float(np.nanmean(mets["mass_abs"])), pattern_L2=float(np.nanmean(mets["pattern_L2"])),
                                          centroid=float(np.nanmean(mets["centroid"])), r2_mass=float(r2_mass),
                                          temporal_err=temporal, temporal_true=temporal_true, **f1, **info.get(name, {})))
                if save_preds:
                    np.savez_compressed(PRED_DIR / f"{g}__{split}{k}.npz", sequence_id=sid_te, frame=fr_te,
                                        **{n: preds[n].astype(np.float16) for n in ("mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_D_full_window", "mlp_Dc_coarse_window")})
                get = lambda n: rows_fold[-len(preds) + list(preds).index(n)]["raw"]
                log.info("%s %s fold %d: %d train / %d test seq, %d test frames, %.0fs | mlp raw A %.2f B %.2f C %.2f D %.2f Dc %.2f | ridge A %.2f B %.2f C %.2f D %.2f Dc %.2f | cat-mean %.2f mesh-mean %.2f prev %.2f seq-oracle %.2f",
                         g, split, k, len(tr_seq), len(te_seq), ite.sum(), time.time() - t0,
                         *[get(f"mlp_{c}") for c in CONDS], *[get(f"ridge_{c}") for c in CONDS],
                         get("base_cat_mean"), get("base_mesh_mean"), get("base_prev_frame"), get("oracle_seq_mean"))
        pd.concat(rows_seq).to_csv(OUT / f"per_sequence{tag}.csv", index=False)
        pd.DataFrame(rows_fold).to_csv(OUT / f"per_fold{tag}.csv", index=False)
    if tag:
        return
    aggregate(pd.concat(rows_seq), pd.DataFrame(rows_fold))


def aggregate(PS, PF):
    """Per group/split/model: sequence bootstrap over test sequences (all folds pooled)."""
    agg = []
    for (g, split, model), gp in PS.groupby(["group", "split", "model"]):
        row = dict(group=g, split=split, model=model, n_seq=len(gp))
        for met in ("raw", "mass_abs", "mass_log", "pattern_L2", "centroid", "raw_at_true_mass", "act_correct"):
            v = gp[met].values; ok = ~np.isnan(v)
            pt, lo, hi = C.seq_bootstrap(v[ok], gp.sequence_id.values[ok], None, n_boot=500)
            row[met], row[f"{met}_lo"], row[f"{met}_hi"] = pt, lo, hi
        pf = PF[(PF.group == g) & (PF.split == split) & (PF.model == model)]
        for met in ("zero_f1", "bal_acc", "r2_mass", "temporal_err", "temporal_true"):
            row[met] = float(pf[met].mean())
        agg.append(row)
    A = pd.DataFrame(agg)
    A.to_csv(C.OUT / "prediction_ablation.csv", index=False)
    pd.set_option("display.width", 250)
    print(A.pivot_table(index=["group", "model"], columns="split", values=["raw", "mass_abs", "pattern_L2"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None, help="e.g. knife__tool__R")
    ap.add_argument("--shard", type=int, default=None, help="process groups[shard::n_shards]")
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    gs = [tuple(g.split("__")) for g in a.groups] if a.groups else None
    if a.shard is not None:
        gs = (gs or C.GROUPS)[a.shard::a.n_shards]
        main(gs, tag=f"_shard{a.shard}of{a.n_shards}")
    else:
        main(gs)
