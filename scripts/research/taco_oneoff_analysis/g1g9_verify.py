#!/usr/bin/env python
"""Attack the validity of the kNN retrieval baseline (G1) and the take-to-take number (G9).

(a) top-k ORACLE retrieval (best of top-k neighbours by hand error), plus phase-shift oracle.
(b) descriptors that keep temporal order (8-frame concat, PCA-512, global trajectory), and a
    descriptor that also sees the recorded contact map (what the motion stage is conditioned on).
(c) leakage checks recomputed inside this run.
(d) take-to-take spread at matched phase, from the store, per hand, same/different subject,
    plus the take-mean floor and the test_1-vs-train-takes floor.
"""
import pathlib as _pl, sys as _sys
_HERE = str(_pl.Path(__file__).resolve().parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
import json, logging, time, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

R = Path("/result/uhnam/dexcore/bimart_taco"); S = R / "train_store"
OUT = R / "contact_probe" / "gen" / "verify"; OUT.mkdir(parents=True, exist_ok=True)
H = 64; KP = 600; BASE = 8; STRIDE = 4; NTRQ = 640
KS = (1, 5, 20, 100)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("g1g9")

off = pd.read_csv(S / "offsets.csv")
bps = np.load(S / "bps.npy", mmap_mode="r"); glo = np.load(S / "global.npy", mmap_mode="r")
con = np.load(S / "contact.npy", mmap_mode="r")
t0 = time.time()
ACT = np.ascontiguousarray(np.load(S / "action.npy", mmap_mode="r")[:, :KP]).reshape(-1, 200, 3)  # (N,200,3) m
log.info("hand kp in RAM %s (%.0fs)", ACT.shape, time.time() - t0)
seq_n = dict(zip(off.sequence_id, off.n_frames_store)); seq_s0 = dict(zip(off.sequence_id, off.start))


def windows(row, stride):
    end = int(row.n_frames_store) - H - BASE
    return [] if end <= BASE else list(range(BASE, end, stride))


FR8 = np.linspace(0, H - 1, 8).round().astype(int)
FR16 = np.linspace(0, H - 1, 16).round().astype(int)


def descs(row, starts):
    """Return dict of descriptor families for the given window starts of one sequence."""
    s0, n = int(row.start), int(row.n_frames_store)
    X = np.concatenate([np.asarray(bps[s0:s0 + n], dtype=np.float32),
                        np.asarray(glo[s0:s0 + n], dtype=np.float32)], axis=1)   # (n,3080)
    C = np.asarray(con[s0:s0 + n], dtype=np.float32)                            # (n,2048)
    out = {k: [] for k in ("meanstd", "frames8", "globtraj", "meanstd_cmap", "cmap_only")}
    for st in starts:
        w = X[st:st + H]; c = C[st:st + H]
        ms = np.concatenate([w.mean(0), w.std(0)])
        out["meanstd"].append(ms)
        out["frames8"].append(w[FR8].reshape(-1))                                  # 24640
        out["globtraj"].append(np.concatenate([ms, w[FR16, 3072:].reshape(-1)]))   # 6160+128
        cms = np.concatenate([c.mean(0), c.std(0)])
        out["meanstd_cmap"].append(np.concatenate([ms, cms]))                      # 10256
        out["cmap_only"].append(cms)
    return {k: np.stack(v) for k, v in out.items()}


# ------------------------------------------------------------------ index
tr = off[off.split == "train"].reset_index(drop=True)
idx = {k: [] for k in ("meanstd", "frames8", "globtraj", "meanstd_cmap", "cmap_only")}; meta = []
for i, row in tr.iterrows():
    st = windows(row, STRIDE)
    if not st: continue
    d = descs(row, st)
    for k in idx: idx[k].append(d[k])
    meta += [{"seq_i": i, "sequence_id": row.sequence_id, "start": s, "abs_start": int(row.start) + s,
              "triplet": row.triplet, "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh} for s in st]
    if (i + 1) % 200 == 0: log.info("index %d/%d", i + 1, len(tr))
Mi = pd.DataFrame(meta)
idx = {k: np.concatenate(v) for k, v in idx.items()}
stats = {}
for k in idx:
    mu, sd = idx[k].mean(0), idx[k].std(0) + 1e-6
    idx[k] = (idx[k] - mu) / sd; stats[k] = (mu, sd)
    log.info("index %s: %s", k, idx[k].shape)
# PCA-512 of frames8 (randomized SVD on the z-scored index)
from sklearn.decomposition import PCA
pca = PCA(n_components=512, svd_solver="randomized", random_state=0).fit(idx["frames8"])
idx["frames8_pca512"] = pca.transform(idx["frames8"]).astype(np.float32)
log.info("PCA-512 explained variance: %.3f", pca.explained_variance_ratio_.sum())
sq = {k: (v ** 2).sum(1) for k, v in idx.items()}
idx_abs = Mi.abs_start.values; idx_seq = Mi.sequence_id.values; idx_start = Mi.start.values
idx_n = np.array([seq_n[s] for s in idx_seq])

# ------------------------------------------------------------------ queries (same sampling as knn_baseline_taco.py)
rng = np.random.default_rng(0); Q = []
for split in ("test_1", "test_2", "test_3", "test_4", "train"):
    sub = off[off.split == split].reset_index(drop=True)
    qs = []
    for i, row in sub.iterrows():
        qs += [(i, s) for s in windows(row, H if split != "train" else 1)]
    if split == "train":
        qs = [qs[k] for k in rng.choice(len(qs), min(NTRQ, len(qs)), replace=False)]
    by = {}
    for i, s in qs: by.setdefault(i, []).append(s)
    for i, starts in by.items():
        row = sub.iloc[i]; d = descs(row, starts)
        for j, s in enumerate(starts):
            Q.append({"split": split, "sequence_id": row.sequence_id, "start": s, "abs_start": int(row.start) + s,
                      "triplet": row.triplet, "tool_mesh": row.tool_mesh, "target_mesh": row.target_mesh,
                      "desc": {k: d[k][j] for k in d}})
log.info("queries: %d", len(Q))
Qd = {k: np.stack([q["desc"][k] for q in Q]) for k in ("meanstd", "frames8", "globtraj", "meanstd_cmap", "cmap_only")}
for k in Qd: Qd[k] = (Qd[k] - stats[k][0]) / stats[k][1]
Qd["frames8_pca512"] = pca.transform(Qd["frames8"]).astype(np.float32)
qsplit = np.array([q["split"] for q in Q]); qseq = np.array([q["sequence_id"] for q in Q])
qabs = np.array([q["abs_start"] for q in Q]); qtrip = np.array([q["triplet"] for q in Q])
qtool = np.array([q["tool_mesh"] for q in Q]); qtarg = np.array([q["target_mesh"] for q in Q])
train_seqs = set(tr.sequence_id)


def hand_err(qa, na):
    """mean point distance (mm) L, R between query window at qa and neighbour window at na."""
    dd = np.linalg.norm(ACT[qa:qa + H] - ACT[na:na + H], axis=-1) * 1000
    return dd[:, :100].mean(), dd[:, 100:].mean()


def topk_for(name, KMAX=100):
    Zi = idx[name]; Zq = Qd[name]
    out = np.empty((len(Zq), KMAX), dtype=np.int64); dist = np.empty((len(Zq), KMAX), dtype=np.float32)
    for a in range(0, len(Zq), 256):
        z = Zq[a:a + 256]
        d2 = (z ** 2).sum(1)[:, None] - 2 * z @ Zi.T + sq[name][None, :]
        for r in range(len(z)):
            if qsplit[a + r] == "train":
                d2[r, idx_seq == qseq[a + r]] = np.inf
        part = np.argpartition(d2, KMAX, axis=1)[:, :KMAX]
        pd2 = np.take_along_axis(d2, part, 1); o = np.argsort(pd2, 1)
        out[a:a + 256] = np.take_along_axis(part, o, 1); dist[a:a + 256] = np.sqrt(np.maximum(np.take_along_axis(pd2, o, 1), 0))
    return out, dist


# ------------------------------------------------------------------ (a)+(b): nn1 and oracle-k per descriptor
rows = []
DESC_NAMES = ("meanstd", "frames8", "frames8_pca512", "globtraj", "meanstd_cmap", "cmap_only")
NB = {}
for name in DESC_NAMES:
    t0 = time.time(); nb, dist = topk_for(name); NB[name] = nb
    E = np.empty((len(Q), 100, 2), dtype=np.float32)
    for qi in range(len(Q)):
        for j in range(100):
            E[qi, j] = hand_err(qabs[qi], idx_abs[nb[qi, j]])
    Em = E.mean(-1)
    for qi in range(len(Q)):
        r = {"split": qsplit[qi], "sequence_id": qseq[qi], "start": int(qabs[qi] - seq_s0[qseq[qi]]), "desc": name,
             "nn_dist": float(dist[qi, 0]), "nn_sequence_id": idx_seq[nb[qi, 0]],
             "nn1_mm": float(Em[qi, 0]), "nn1_L": float(E[qi, 0, 0]), "nn1_R": float(E[qi, 0, 1]),
             "nn1_same_cond": bool(idx_seq[nb[qi, 0]] != qseq[qi] and Mi.triplet.values[nb[qi, 0]] == qtrip[qi]
                                   and Mi.tool_mesh.values[nb[qi, 0]] == qtool[qi] and Mi.target_mesh.values[nb[qi, 0]] == qtarg[qi]),
             "leak": bool(idx_seq[nb[qi, 0]] == qseq[qi]) or (idx_seq[nb[qi, 0]] not in train_seqs)}
        for k in KS:
            b = int(Em[qi, :k].argmin())
            r[f"oracle{k}_mm"] = float(Em[qi, b]); r[f"oracle{k}_L"] = float(E[qi, b, 0]); r[f"oracle{k}_R"] = float(E[qi, b, 1])
            r[f"oracle{k}_perhand_mm"] = float((E[qi, :k, 0].min() + E[qi, :k, 1].min()) / 2)
        # mean over top-5 / top-20 neighbours' hands ("soft retrieval" = average of neighbours)
        for k in (5, 20):
            avg = np.mean([ACT[idx_abs[nb[qi, j]]:idx_abs[nb[qi, j]] + H] for j in range(k)], axis=0)
            dd = np.linalg.norm(ACT[qabs[qi]:qabs[qi] + H] - avg, axis=-1) * 1000
            r[f"avg{k}_mm"] = float(dd.mean()); r[f"avg{k}_L"] = float(dd[:, :100].mean()); r[f"avg{k}_R"] = float(dd[:, 100:].mean())
        rows.append(r)
    log.info("%s done (%.0fs): nn1 test_1 %.1f  oracle20 %.1f", name, time.time() - t0,
             Em[qsplit == "test_1", 0].mean(), Em[qsplit == "test_1", :20].min(1).mean())

# phase-shift oracle on the meanstd top-20: slide each neighbour window by delta in [-32,32] step 4
nb = NB["meanstd"]; shift_rows = []
deltas = np.arange(-32, 33, 4)
for qi in range(len(Q)):
    best = np.inf; best20 = np.inf; bestL = bestR = np.nan
    for j in range(20):
        n_ab = idx_abs[nb[qi, j]]; st = idx_start[nb[qi, j]]; n_len = idx_n[nb[qi, j]]; s0 = n_ab - st
        for dl in deltas:
            ns = st + dl
            if ns < 0 or ns + H > n_len: continue
            l, rr = hand_err(qabs[qi], s0 + ns); e = (l + rr) / 2
            if e < best: best, bestL, bestR = e, l, rr
            if j == 0 and e < best20: best20 = e
    shift_rows.append({"oracle20_shift_mm": best, "oracle20_shift_L": bestL, "oracle20_shift_R": bestR, "oracle1_shift_mm": best20})
SH = pd.DataFrame(shift_rows)
D = pd.DataFrame(rows)
D0 = D[D.desc == "meanstd"].reset_index(drop=True)
for c in SH: D0[c] = SH[c].values
D = pd.concat([D0, D[D.desc != "meanstd"]], ignore_index=True)

# join model errors (window-level, from knn_joined.csv)
J = pd.read_csv(R / "contact_probe" / "knn" / "knn_joined.csv")[["split", "sequence_id", "start", "motion", "motion_L", "motion_R", "contact", "touch", "retr_hand_mm", "nn_dist"]]
J = J.rename(columns={"retr_hand_mm": "retr_hand_mm_orig", "nn_dist": "nn_dist_orig"})
D = D.merge(J, on=["split", "sequence_id", "start"], how="left")
D.to_csv(OUT / "g1g9_retrieval_per_window.csv", index=False)

# (c) leakage
leak = {"train_lso_violations": int((D[(D.split == "train")].nn_sequence_id == D[(D.split == "train")].sequence_id).sum()),
        "test_nn_not_in_train": int((~D[D.split != "train"].nn_sequence_id.isin(train_seqs)).sum()),
        "any_leak_flag": int(D.leak.sum()), "n_rows": int(len(D))}
# reproduction of the original baseline
rep = D[D.desc == "meanstd"].dropna(subset=["retr_hand_mm_orig"])
leak["repro_maxabs_retr_hand_diff_mm"] = float((rep.nn1_mm - rep.retr_hand_mm_orig).abs().max())
leak["repro_maxabs_nn_dist_diff"] = float((rep.nn_dist - rep.nn_dist_orig).abs().max())
log.info("leak/repro: %s", leak)

# summary table per split x descriptor
cols = ["nn_dist", "nn1_mm", "nn1_L", "nn1_R"] + [f"oracle{k}_mm" for k in KS] + [f"oracle{k}_perhand_mm" for k in KS] + \
       ["oracle20_L", "oracle20_R", "oracle100_L", "oracle100_R", "avg5_mm", "avg20_mm", "oracle1_shift_mm", "oracle20_shift_mm", "oracle20_shift_L", "oracle20_shift_R", "motion", "motion_L", "motion_R"]
Dm = D.dropna(subset=["motion"])
summ = Dm.groupby(["desc", "split"])[cols].mean().round(1)
summ["n"] = Dm.groupby(["desc", "split"]).size()
for k in KS:
    summ[f"frac_model_beats_oracle{k}"] = Dm.groupby(["desc", "split"]).apply(lambda g: (g.motion < g[f"oracle{k}_mm"]).mean()).round(3)
    summ[f"median_ratio_oracle{k}_over_model"] = Dm.groupby(["desc", "split"]).apply(lambda g: (g[f"oracle{k}_mm"] / g.motion).median()).round(2)
summ["frac_model_beats_oracle20_shift"] = Dm.groupby(["desc", "split"]).apply(lambda g: (g.motion < g.oracle20_shift_mm).mean()).round(3)
summ["rho_nn_dist_motion"] = Dm.groupby(["desc", "split"]).apply(lambda g: spearmanr(g.nn_dist, g.motion)[0]).round(3)
summ["rho_nn1_motion"] = Dm.groupby(["desc", "split"]).apply(lambda g: spearmanr(g.nn1_mm, g.motion)[0]).round(3)
summ["rho_oracle20_motion"] = Dm.groupby(["desc", "split"]).apply(lambda g: spearmanr(g.oracle20_mm, g.motion)[0]).round(3)
summ.to_csv(OUT / "g1g9_retrieval_summary.csv")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print("\n=== (a)/(b) retrieval variants vs model motion error (mm), rows with model errors ===")
print(summ.reset_index().to_string())

# ------------------------------------------------------------------ (d) take-to-take spread at matched phase
subj = pd.read_csv(R / "contact_probe" / "gen" / "betas_per_sequence_subjects.csv")[["sequence_id", "subject", "date"]]
subj_of = dict(zip(subj.sequence_id, subj.subject)); date_of = dict(zip(subj.sequence_id, subj.date))
NPH = 32; phases = (np.arange(NPH) + 0.5) / NPH


def kp_at_phase(sid, shift=0.0):
    s0, n = seq_s0[sid], seq_n[sid]
    fr = np.clip(np.round((phases + shift) * (n - 1)).astype(int), 0, n - 1)
    return ACT[s0 + fr]                        # (NPH,200,3)


def pair_dist(A, B):
    dd = np.linalg.norm(A - B, axis=-1) * 1000
    return dd[:, :100].mean(), dd[:, 100:].mean()


def pair_dist_best_shift(sid_a, sid_b, max_shift=0.10, step=0.02):
    A = kp_at_phase(sid_a); best = (np.inf, np.nan, np.nan)
    for sh in np.arange(-max_shift, max_shift + 1e-9, step):
        l, r = pair_dist(A, kp_at_phase(sid_b, sh))
        if (l + r) / 2 < best[0]: best = ((l + r) / 2, l, r)
    return best


def pair_dist_abs_frames(sid_a, sid_b):
    """same absolute frame index (no phase alignment), over the overlap"""
    n = min(seq_n[sid_a], seq_n[sid_b]); fr = np.linspace(0, n - 1, NPH).round().astype(int)
    return pair_dist(ACT[seq_s0[sid_a] + fr], ACT[seq_s0[sid_b] + fr])


cond_cols = ["triplet", "tool_mesh", "target_mesh"]
tk = []
for cond, g in tr.groupby(cond_cols):
    sids = list(g.sequence_id)
    if len(sids) < 2: continue
    for a, b in itertools.combinations(sids, 2):
        l, r = pair_dist(kp_at_phase(a), kp_at_phase(b)); bs = pair_dist_best_shift(a, b); la, ra = pair_dist_abs_frames(a, b)
        tk.append({"triplet": cond[0], "tool_mesh": cond[1], "target_mesh": cond[2], "seq_a": a, "seq_b": b, "n_takes": len(sids),
                   "same_subject": subj_of.get(a) == subj_of.get(b), "same_date": date_of.get(a) == date_of.get(b),
                   "len_a": seq_n[a], "len_b": seq_n[b],
                   "phase_L": l, "phase_R": r, "phase_mm": (l + r) / 2,
                   "shift_mm": bs[0], "shift_L": bs[1], "shift_R": bs[2],
                   "absframe_L": la, "absframe_R": ra, "absframe_mm": (la + ra) / 2})
TK = pd.DataFrame(tk); TK.to_csv(OUT / "g1g9_train_take_pairs.csv", index=False)
print("\n=== (d) train take-to-take spread at matched phase (mm), pairs within (triplet, tool_mesh, target_mesh) ===")
print("pairs:", len(TK), " conditions:", TK.groupby(cond_cols).ngroups, " same_subject frac:", round(TK.same_subject.mean(), 3))
print(TK[["phase_mm", "phase_L", "phase_R", "shift_mm", "shift_L", "shift_R", "absframe_mm", "absframe_L", "absframe_R"]].describe().round(1).T)
print(TK.groupby("same_subject")[["phase_mm", "phase_L", "phase_R", "shift_mm"]].agg(["mean", "median", "count"]).round(1))
print("R/L ratio (phase-matched, mean):", round(TK.phase_R.mean() / TK.phase_L.mean(), 2))
print("per-verb phase-matched:"); print(TK.assign(verb=TK.triplet.str.extract(r"\((.*?),")[0]).groupby("verb")[["phase_mm", "phase_L", "phase_R"]].agg(["mean", "count"]).round(1))

# take-mean floor: conditions with 3 takes: distance from each take to the mean of the other two
tm = []
for cond, g in tr.groupby(cond_cols):
    sids = list(g.sequence_id)
    if len(sids) < 3: continue
    K = {s: kp_at_phase(s) for s in sids}
    for s in sids:
        others = [K[o] for o in sids if o != s]
        l, r = pair_dist(K[s], np.mean(others, 0))
        pl = np.mean([pair_dist(K[s], o)[0] for o in others]); pr = np.mean([pair_dist(K[s], o)[1] for o in others])
        tm.append({"triplet": cond[0], "seq": s, "n_takes": len(sids), "to_mean_L": l, "to_mean_R": r, "to_mean_mm": (l + r) / 2,
                   "pairwise_L": pl, "pairwise_R": pr, "pairwise_mm": (pl + pr) / 2})
TM = pd.DataFrame(tm); TM.to_csv(OUT / "g1g9_train_take_mean_floor_3takes.csv", index=False)
print("\n=== (d) 3-take conditions: distance to mean of the other 2 takes vs mean pairwise ===")
print(TM[["to_mean_mm", "to_mean_L", "to_mean_R", "pairwise_mm", "pairwise_L", "pairwise_R"]].describe().round(1).T)
print("ratio to_mean/pairwise:", round(TM.to_mean_mm.mean() / TM.pairwise_mm.mean(), 3), "(iid-Gaussian prediction sqrt(1.5/2)=0.866; infinite-take limit 0.707)")

# test_1 sequences vs train takes of the same exact condition, at matched phase
t1 = off[off.split == "test_1"]
seqerr = pd.read_csv(R / "contact_probe" / "gen" / "sequence_level_errors_with_subject.csv")
seqerr = seqerr[seqerr.split == "test_1"][["sequence_id", "motion", "contact"]].rename(columns={"motion": "model_motion", "contact": "model_contact"})
# also per-hand model motion at sequence level from knn_joined
Jh = J[J.split == "test_1"].groupby("sequence_id")[["motion_L", "motion_R"]].mean().reset_index()
tv = []
for _, r in t1.iterrows():
    g = tr[(tr.triplet == r.triplet) & (tr.tool_mesh == r.tool_mesh) & (tr.target_mesh == r.target_mesh)]
    if len(g) == 0: continue
    Kq = kp_at_phase(r.sequence_id); Ks = [kp_at_phase(s) for s in g.sequence_id]
    per = [pair_dist(Kq, k) for k in Ks]; best_shift = [pair_dist_best_shift(r.sequence_id, s) for s in g.sequence_id]
    lm, rm = pair_dist(Kq, np.mean(Ks, 0))
    bi = int(np.argmin([(l + rr) / 2 for l, rr in per]))
    tv.append({"sequence_id": r.sequence_id, "triplet": r.triplet, "n_train_takes": len(g),
               "same_subject_any": any(subj_of.get(r.sequence_id) == subj_of.get(s) for s in g.sequence_id),
               "mean_take_L": np.mean([p[0] for p in per]), "mean_take_R": np.mean([p[1] for p in per]),
               "mean_take_mm": np.mean([(p[0] + p[1]) / 2 for p in per]),
               "best_take_mm": (per[bi][0] + per[bi][1]) / 2, "best_take_shift_mm": min(b[0] for b in best_shift),
               "to_train_mean_L": lm, "to_train_mean_R": rm, "to_train_mean_mm": (lm + rm) / 2})
TV = pd.DataFrame(tv).merge(seqerr, on="sequence_id", how="left").merge(Jh, on="sequence_id", how="left")
TV.to_csv(OUT / "g1g9_test1_vs_train_takes.csv", index=False)
print("\n=== (d) test_1 sequences whose exact (triplet, meshes) condition exists in train: phase-matched distance to train takes ===")
print("n test_1 seqs:", len(TV), " with >=2 train takes:", (TV.n_train_takes >= 2).sum(), " same-subject-any:", TV.same_subject_any.sum())
print(TV[["mean_take_mm", "mean_take_L", "mean_take_R", "best_take_mm", "best_take_shift_mm", "to_train_mean_mm", "model_motion", "motion_L", "motion_R"]].describe().round(1).T)
print(TV[TV.n_train_takes >= 2][["mean_take_mm", "best_take_mm", "to_train_mean_mm", "model_motion"]].describe().round(1).T)
print("frac model_motion < mean_take_mm:", round((TV.model_motion < TV.mean_take_mm).mean(), 3),
      "  < best_take_mm:", round((TV.model_motion < TV.best_take_mm).mean(), 3),
      "  < to_train_mean_mm:", round((TV.model_motion < TV.to_train_mean_mm).mean(), 3))
print("Spearman(mean_take_mm, model_motion):", spearmanr(TV.mean_take_mm, TV.model_motion))

# the 'same triplet+meshes neighbour' windows from the original run: are they really other takes?
D0 = D[(D.desc == "meanstd")]
sc = D0[D0.nn1_same_cond]
print("\n=== original-style same-condition neighbours (meanstd nn1) ===")
print(sc.groupby("split")[["nn1_mm", "nn1_L", "nn1_R", "oracle20_mm", "motion"]].agg(["mean", "count"]).round(1))

summary = {"leak": leak, "pca512_evr": float(pca.explained_variance_ratio_.sum()),
           "take_pairs": {"n_pairs": int(len(TK)), "n_conditions": int(TK.groupby(cond_cols).ngroups),
                          "phase_mm_mean": float(TK.phase_mm.mean()), "phase_L_mean": float(TK.phase_L.mean()), "phase_R_mean": float(TK.phase_R.mean()),
                          "phase_mm_median": float(TK.phase_mm.median()), "shift_mm_mean": float(TK.shift_mm.mean()),
                          "absframe_mm_mean": float(TK.absframe_mm.mean()),
                          "same_subject_phase_mm": float(TK[TK.same_subject].phase_mm.mean()), "diff_subject_phase_mm": float(TK[~TK.same_subject].phase_mm.mean()),
                          "n_same_subject": int(TK.same_subject.sum())},
           "take_mean_3takes": {"n": int(len(TM)), "to_mean_mm": float(TM.to_mean_mm.mean()), "pairwise_mm": float(TM.pairwise_mm.mean())},
           "test1_vs_train_takes": {"n": int(len(TV)), "mean_take_mm": float(TV.mean_take_mm.mean()), "best_take_mm": float(TV.best_take_mm.mean()),
                                    "to_train_mean_mm": float(TV.to_train_mean_mm.mean()), "model_motion": float(TV.model_motion.mean()),
                                    "frac_model_below_mean_take": float((TV.model_motion < TV.mean_take_mm).mean())}}
(OUT / "g1g9_summary.json").write_text(json.dumps(summary, indent=1, default=float))
print("\n->", OUT)
