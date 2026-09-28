#!/usr/bin/env python
"""G3 step 3: identity-restricted NN searches for test_1. If the distance to the nearest training window that
uses the SAME tool mesh (or same tool+target mesh, or same triplet) still predicts error, the shift is in how
the objects were handled in this take, not in which objects they are. Also k-NN density (k=5,20)."""
import sys, pathlib as _pl
_HERE = str(_pl.Path(__file__).resolve().parent); sys.path[:] = [p for p in sys.path if p not in ("", _HERE)]
import json, logging, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
R = Path("/result/uhnam/dexcore/bimart_taco"); S = R / "train_store"; OUT = R / "contact_probe" / "gen" / "verify" / "g3"
H = 64; KP = 600; base = 8; STRIDE = 4; C = 3080
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"); log = logging.getLogger("g3r")
off = pd.read_csv(S / "offsets.csv")
bps = np.load(S / "bps.npy", mmap_mode="r"); glo = np.load(S / "global.npy", mmap_mode="r"); act = np.load(S / "action.npy", mmap_mode="r")
TOOL = np.concatenate([np.arange(0, 1536), np.arange(C, C + 1536)])
def windows(row, stride):
    end = int(row.n_frames_store) - H - base; return [] if end <= base else list(range(base, end, stride))
def descriptors(row, starts):
    s0, n = int(row.start), int(row.n_frames_store)
    X = np.concatenate([np.asarray(bps[s0:s0 + n], dtype=np.float32), np.asarray(glo[s0:s0 + n], dtype=np.float32)], axis=1)
    D = np.empty((len(starts), 2 * C), np.float32)
    for i, st in enumerate(starts): w = X[st:st + H]; D[i, :C] = w.mean(0); D[i, C:] = w.std(0)
    return D
tr = off[off.split == "train"].reset_index(drop=True); idx_desc, idx_meta = [], []
for i, row in tr.iterrows():
    st = windows(row, STRIDE)
    if not st: continue
    idx_desc.append(descriptors(row, st))
    idx_meta += [{"sequence_id": row.sequence_id, "start": s, "abs_start": int(row.start) + s, "triplet": row.triplet, "verb": row.verb, "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh} for s in st]
Xi = np.concatenate(idx_desc); Mi = pd.DataFrame(idx_meta); mu, sd = Xi.mean(0), Xi.std(0) + 1e-6; Zi = (Xi - mu) / sd; del Xi
Zi_sq = (Zi ** 2).sum(1); Zt_sq = (Zi[:, TOOL] ** 2).sum(1)
log.info("index %d", len(Zi))
def d2mat(Zq, Zref, ref_sq): return (Zq ** 2).sum(1)[:, None] - 2 * Zq @ Zref.T + ref_sq[None, :]
sub = off[off.split == "test_1"].reset_index(drop=True); rows = []
for i, row in sub.iterrows():
    starts = windows(row, H)
    if not starts: continue
    Zq = (descriptors(row, starts) - mu) / sd
    d2 = d2mat(Zq, Zi, Zi_sq); d2t = d2mat(Zq[:, TOOL], Zi[:, TOOL], Zt_sq)
    masks = {"any": np.ones(len(Mi), bool), "same_tool": (Mi.tool_mesh == row.tool_mesh).values,
             "same_target": (Mi.target_mesh == row.target_mesh).values,
             "same_both": ((Mi.tool_mesh == row.tool_mesh) & (Mi.target_mesh == row.target_mesh)).values,
             "same_triplet": (Mi.triplet == row.triplet).values,
             "same_verb": (Mi.verb == row.verb).values, "other_verb": (Mi.verb != row.verb).values}
    srt = np.sort(d2, axis=1)
    for j, s in enumerate(starts):
        q0 = int(row.start) + s; kq = np.asarray(act[q0:q0 + H, :KP]).reshape(H, 200, 3)
        r = {"sequence_id": row.sequence_id, "start": s, "verb": row.verb, "triplet": row.triplet,
             "knn5_mean": float(np.sqrt(np.maximum(srt[j, :5], 0)).mean()), "knn20_mean": float(np.sqrt(np.maximum(srt[j, :20], 0)).mean())}
        for name, m in masks.items():
            r[f"n_{name}"] = int(m.sum())
            if m.sum() == 0: r[f"d_{name}"] = np.nan; r[f"dtool_{name}"] = np.nan; r[f"retr_{name}"] = np.nan; continue
            dd = np.where(m, d2[j], np.inf); k = dd.argmin(); r[f"d_{name}"] = float(np.sqrt(max(dd[k], 0)))
            ddt = np.where(m, d2t[j], np.inf); r[f"dtool_{name}"] = float(np.sqrt(max(ddt.min(), 0)))
            mm = Mi.iloc[k]; kn = np.asarray(act[mm.abs_start:mm.abs_start + H, :KP]).reshape(H, 200, 3)
            r[f"retr_{name}"] = float(np.linalg.norm(kq - kn, axis=-1).mean() * 1000)
            r[f"nnverb_{name}"] = mm.verb; r[f"nn_same_target_{name}"] = bool(mm.target_mesh == row.target_mesh); r[f"nn_same_tool_{name}"] = bool(mm.tool_mesh == row.tool_mesh)
        rows.append(r)
F = pd.DataFrame(rows); J = pd.read_csv(R / "contact_probe" / "knn" / "knn_joined.csv"); J = J[J.split == "test_1"]
T = F.merge(J[["sequence_id", "start", "motion", "contact", "nn_dist", "nn_same_target_mesh", "nn_same_tool_mesh"]], on=["sequence_id", "start"])
T.to_csv(OUT / "g3_restricted_windows.csv", index=False)
def sp(x, y): m = np.isfinite(x) & np.isfinite(y); r, p = stats.spearmanr(x[m], y[m]); return float(r), float(p), int(m.sum())
out = {}
print("test_1 unrestricted NN: same target mesh %.2f, same tool mesh %.2f" % (T.nn_same_target_mesh.mean(), T.nn_same_tool_mesh.mean()))
Jt = pd.read_csv(R / "contact_probe" / "knn" / "knn_joined.csv"); Jt = Jt[Jt.split == "train"]
print("train  unrestricted NN: same target mesh %.2f, same tool mesh %.2f" % (Jt.nn_same_target_mesh.mean(), Jt.nn_same_tool_mesh.mean()))
out["same_mesh_rate"] = dict(test1_target=float(T.nn_same_target_mesh.mean()), test1_tool=float(T.nn_same_tool_mesh.mean()), train_target=float(Jt.nn_same_target_mesh.mean()), train_tool=float(Jt.nn_same_tool_mesh.mean()))
for name in ("any", "same_tool", "same_target", "same_both", "same_triplet", "same_verb", "other_verb"):
    r, p, n = sp(T[f"d_{name}"].values, T.motion.values); rc, _, _ = sp(T[f"d_{name}"].values, T.contact.values)
    rt, _, nt = sp(T[f"dtool_{name}"].values, T.motion.values); rr, _, _ = sp(T[f"retr_{name}"].values, T.motion.values)
    out[name] = dict(n=n, mean_dist=float(T[f"d_{name}"].mean()), rho_motion=r, p=p, rho_contact=rc, mean_dtool=float(T[f"dtool_{name}"].mean()), rho_tool_motion=rt, mean_retr=float(T[f"retr_{name}"].mean()), rho_retr_motion=rr, mean_n_index=float(T[f"n_{name}"].mean()))
    print(f"NN restricted to {name:12s}: n={n:3d} mean dist {T[f'd_{name}'].mean():5.1f} rho(motion)={r:+.3f} (p={p:.1e}) rho(contact)={rc:+.3f} | tool-block dist {T[f'dtool_{name}'].mean():5.1f} rho={rt:+.3f} | retr hand {T[f'retr_{name}'].mean():6.1f} mm rho={rr:+.3f} | index size {T[f'n_{name}'].mean():.0f}")
for k in ("knn5_mean", "knn20_mean"):
    r, p, n = sp(T[k].values, T.motion.values); rc, _, _ = sp(T[k].values, T.contact.values); out[k] = dict(rho_motion=r, rho_contact=rc); print(f"{k}: rho(motion)={r:+.3f}, rho(contact)={rc:+.3f}")
# extra: is d_same_both - d_any (cost of insisting on identity) related to error?
m = np.isfinite(T.d_same_both)
print("windows with a same-tool+target training window: %d/%d; among them rho(d_same_both, motion)=%+.3f, rho(d_any, motion)=%+.3f, mean d_same_both %.1f vs d_any %.1f" % (m.sum(), len(T), sp(T.d_same_both.values, T.motion.values)[0], sp(T.d_any.values[m], T.motion.values[m])[0], T.d_same_both[m].mean(), T.d_any[m].mean()))
# partial: d_same_triplet controlling verb (verb FE), cluster by sequence
def partial(dcol):
    mm = np.isfinite(T[dcol]); sub = T[mm]; X = pd.get_dummies(sub.verb, drop_first=True).astype(float).values; X = np.column_stack([np.ones(len(sub)), X])
    rx = stats.rankdata(sub[dcol]); ry = stats.rankdata(sub.motion)
    ux = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]; uy = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]; return float(stats.pearsonr(ux, uy)[0]), int(mm.sum())
for dcol in ("d_any", "d_same_tool", "d_same_both", "d_same_triplet", "dtool_same_both"):
    pr, n = partial(dcol); out[dcol + "_partial_verb"] = pr; print(f"partial Spearman({dcol}, motion | verb) = {pr:+.3f} (n={n})")
json.dump(out, open(OUT / "g3_restricted_summary.json", "w"), indent=1)
