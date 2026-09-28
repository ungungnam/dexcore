"""Step 4: pair tables + Baseline A (raw instance-space Chamfer) for the canonical contact study.

Outputs (under /result/uhnam/dexcore/canonical_contact/):
  contact_pairs.csv, contact_pairs_summary.csv,
  baselineA_distances.csv, baselineA_metrics.csv
"""
from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
CACHE = ROOT / "dense_window_cache"
MESH_DICT = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/assets/taco_mesh_dict.npy")
TOUCH_MIN = 0.2
CAP = 3000
SEED = 0
N_BOOT = 1000
HARD_THR = 0.5
MIN_PTS = 5
TOPK = 20

log = logging.getLogger("step4")

PAIR_KEY = ["category", "role", "hand", "pair_type", "seq_i", "window_i", "seq_j", "window_j"]
PAIR_COLS = PAIR_KEY + [
    "mesh_i", "mesh_j", "verb_i", "verb_j", "same_mesh", "same_counterpart_cat",
    "same_triplet_ex_mesh", "same_subject", "same_split", "both_train",
    "counterpart_cat_i", "counterpart_cat_j",
]


# --------------------------------------------------------------------------- samples
def build_samples(w: pd.DataFrame) -> pd.DataFrame:
    """One row per SAMPLE (sequence, window, hand, role) with touch_frac >= TOUCH_MIN."""
    rows = []
    for role in ("tool", "target"):
        cat_col = f"{role}_cat"
        mesh_col = f"{role}_mesh"
        cp_col = "target_cat" if role == "tool" else "tool_cat"
        for hand in ("L", "R"):
            t = w[f"touch_{hand}_{role}"]
            sub = w[t >= TOUCH_MIN]
            rows.append(pd.DataFrame({
                "category": sub[cat_col].values,
                "role": role,
                "hand": hand,
                "sequence_id": sub["sequence_id"].values,
                "file": sub["file"].values,
                "window": sub["window"].values,
                "mesh_id": sub[mesh_col].values,
                "verb": sub["verb"].values,
                "counterpart_cat": sub[cp_col].values,
                "subject": sub["subject"].values,
                "split": sub["split"].values,
                "touch_frac": sub[f"touch_{hand}_{role}"].values,
            }))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- pairs
def eligible(pt: str, a: dict, b: dict) -> bool:
    if pt == "A":
        return a["verb"] == b["verb"] and a["mesh_id"] == b["mesh_id"]
    if pt == "B":
        return a["verb"] == b["verb"] and a["mesh_id"] != b["mesh_id"]
    return a["verb"] != b["verb"]


def sample_pairs_for_group(g: pd.DataFrame, rng: random.Random) -> tuple[list[dict], dict]:
    """Enumerate / stratified-sample pairs for one (category, role, hand) group.

    Sequence-level attributes (verb, mesh, counterpart, subject, split) are constant within a
    sequence, so eligibility is decided at the sequence level; windows are then drawn.
    """
    seqs = {}
    for sid, gs in g.groupby("sequence_id", sort=True):
        r = gs.iloc[0]
        seqs[sid] = dict(verb=r.verb, mesh_id=int(r.mesh_id), counterpart=r.counterpart_cat,
                         subject=int(r.subject), split=r.split,
                         windows=sorted(int(x) for x in gs.window.tolist()))
    sids = sorted(seqs)
    out, info = [], {}
    for pt in ("A", "B", "C"):
        # adjacency over sequences (undirected, stored both ways for round-robin sampling)
        adj = {s: [] for s in sids}
        n_possible = 0
        for ii, s in enumerate(sids):
            for t in sids[ii + 1:]:
                if eligible(pt, seqs[s], seqs[t]):
                    adj[s].append(t)
                    adj[t].append(s)
                    n_possible += len(seqs[s]["windows"]) * len(seqs[t]["windows"])
        chosen: set[tuple] = set()  # (s, ws, t, wt) with s < t
        if n_possible <= CAP:
            for s in sids:
                for t in adj[s]:
                    if s < t:
                        for ws in seqs[s]["windows"]:
                            for wt in seqs[t]["windows"]:
                                chosen.add((s, ws, t, wt))
        else:
            # Pass 1: round-robin over sequences; each turn a sequence draws one unused partner
            # sequence and one random window pair. This keeps per-sequence coverage balanced.
            used_seq = {s: set() for s in sids}
            active = [s for s in sids if adj[s]]
            while len(chosen) < CAP and active:
                rng.shuffle(active)
                nxt = []
                for s in active:
                    cands = [t for t in adj[s] if t not in used_seq[s]]
                    if not cands:
                        continue
                    t = rng.choice(cands)
                    used_seq[s].add(t)
                    used_seq[t].add(s)
                    ws = rng.choice(seqs[s]["windows"])
                    wt = rng.choice(seqs[t]["windows"])
                    chosen.add((s, ws, t, wt) if s < t else (t, wt, s, ws))
                    if len(chosen) >= CAP:
                        break
                    if len(cands) > 1:
                        nxt.append(s)
                active = nxt
            # Pass 2 (only if all sequence pairs are exhausted but window combos remain):
            # draw additional window pairs uniformly among the remaining combinations.
            if len(chosen) < CAP:
                remaining = []
                for s in sids:
                    for t in adj[s]:
                        if s < t:
                            for ws in seqs[s]["windows"]:
                                for wt in seqs[t]["windows"]:
                                    if (s, ws, t, wt) not in chosen:
                                        remaining.append((s, ws, t, wt))
                rng.shuffle(remaining)
                for c in remaining[: CAP - len(chosen)]:
                    chosen.add(c)
        covered_seq, covered_mesh = set(), set()
        for (s, ws, t, wt) in sorted(chosen):
            a, b = seqs[s], seqs[t]
            covered_seq.update([s, t])
            covered_mesh.update([a["mesh_id"], b["mesh_id"]])
            out.append(dict(
                category=g.category.iloc[0], role=g.role.iloc[0], hand=g.hand.iloc[0], pair_type=pt,
                seq_i=s, window_i=ws, seq_j=t, window_j=wt,
                mesh_i=a["mesh_id"], mesh_j=b["mesh_id"], verb_i=a["verb"], verb_j=b["verb"],
                same_mesh=int(a["mesh_id"] == b["mesh_id"]),
                same_counterpart_cat=int(a["counterpart"] == b["counterpart"]),
                same_triplet_ex_mesh=int(a["verb"] == b["verb"] and a["counterpart"] == b["counterpart"]),
                same_subject=int(a["subject"] == b["subject"]),
                same_split=int(a["split"] == b["split"]),
                both_train=int(a["split"] == "train" and b["split"] == "train"),
                counterpart_cat_i=a["counterpart"], counterpart_cat_j=b["counterpart"],
            ))
        info[pt] = dict(n_possible=n_possible, n_pairs=len(chosen),
                        n_sequences_covered=len(covered_seq), n_meshes_covered=len(covered_mesh))
    return out, info


def build_pairs(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SEED)
    all_rows, summ = [], []
    for (cat, role, hand), g in samples.groupby(["category", "role", "hand"], sort=True):
        n_seq = g.sequence_id.nunique()
        if n_seq < 2:
            log.info("skip %s/%s/%s: %d samples from %d sequence(s)", cat, role, hand, len(g), n_seq)
            for pt in "ABC":
                summ.append(dict(category=cat, role=role, hand=hand, pair_type=pt, n_pairs=0,
                                 n_sequences_covered=0, n_meshes_covered=0, n_strict_counterpart=0,
                                 n_strict_triplet=0, n_same_subject=0, n_possible=0,
                                 n_samples=len(g), n_sequences_available=n_seq,
                                 n_meshes_available=g.mesh_id.nunique(), n_verbs_available=g.verb.nunique()))
            continue
        rows, info = sample_pairs_for_group(g, rng)
        all_rows.extend(rows)
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=PAIR_COLS)
        for pt in "ABC":
            d = df[df.pair_type == pt] if len(df) else df
            summ.append(dict(category=cat, role=role, hand=hand, pair_type=pt,
                             n_pairs=info[pt]["n_pairs"],
                             n_sequences_covered=info[pt]["n_sequences_covered"],
                             n_meshes_covered=info[pt]["n_meshes_covered"],
                             n_strict_counterpart=int(d.same_counterpart_cat.sum()) if len(d) else 0,
                             n_strict_triplet=int(d.same_triplet_ex_mesh.sum()) if len(d) else 0,
                             n_same_subject=int(d.same_subject.sum()) if len(d) else 0,
                             n_possible=info[pt]["n_possible"],
                             n_samples=len(g), n_sequences_available=n_seq,
                             n_meshes_available=g.mesh_id.nunique(), n_verbs_available=g.verb.nunique()))
        log.info("%s/%s/%s: samples=%d seqs=%d  A=%d/%d B=%d/%d C=%d/%d", cat, role, hand, len(g), n_seq,
                 info["A"]["n_pairs"], info["A"]["n_possible"], info["B"]["n_pairs"], info["B"]["n_possible"],
                 info["C"]["n_pairs"], info["C"]["n_possible"])
    pairs = pd.DataFrame(all_rows, columns=PAIR_COLS)
    return pairs, pd.DataFrame(summ)


# --------------------------------------------------------------------------- baseline A
_MESH = None


def _init_worker(mesh_path: str):
    global _MESH
    _MESH = np.load(mesh_path, allow_pickle=True).item()


def _contact_set(stem: str, window: int, hand: str, role: str, mesh_id: int, hard_arr=None, soft_arr=None):
    d = np.load(CACHE / f"{stem}.npz")
    hard = d[f"hard_{hand}_{role}"][window].astype(np.float32)
    idx = np.nonzero(hard > HARD_THR)[0]
    fallback = False
    if idx.size < MIN_PTS:
        soft = d[f"soft_{hand}_{role}"][window].astype(np.float32)
        idx = np.argsort(-soft, kind="stable")[:TOPK]
        fallback = True
    V = _MESH[f"{mesh_id:03d}"]["verts_original"].astype(np.float64)
    assert V.shape[0] == hard.shape[0], (stem, mesh_id, V.shape, hard.shape)
    Vc = V - V.mean(axis=0, keepdims=True)  # centre at vertex centroid; no scaling, no rotation
    radius = float(np.linalg.norm(Vc, axis=1).max())
    return Vc[idx], radius, fallback


def _chamfer(P: np.ndarray, Q: np.ndarray) -> float:
    dpq, _ = cKDTree(Q).query(P, k=1)
    dqp, _ = cKDTree(P).query(Q, k=1)
    return 0.5 * (float(dpq.mean()) + float(dqp.mean()))


def _work_sample(args):
    """Compute contact set for one sample; returns key -> (pts, radius, fallback)."""
    key, stem, window, hand, role, mesh_id = args
    pts, radius, fb = _contact_set(stem, window, hand, role, mesh_id)
    return key, pts, radius, fb


def _work_pairs(args):
    """Compute Chamfer distances for a chunk of pairs, given the contact-set dict."""
    chunk, sets = args
    out = []
    for (ki, kj) in chunk:
        Pi, ri, _ = sets[ki]
        Pj, rj, _ = sets[kj]
        raw = _chamfer(Pi, Pj)
        sn = _chamfer(Pi / ri, Pj / rj)
        out.append((raw, sn, len(Pi), len(Pj)))
    return out


def compute_baselineA(pairs: pd.DataFrame, samples: pd.DataFrame, n_proc: int) -> pd.DataFrame:
    stem_of = dict(zip(samples.sequence_id, samples.file.str.replace(r"\.npz$", "", regex=True)))
    # unique samples involved in pairs
    keys = set()
    for side in ("i", "j"):
        for row in pairs[["category", "role", "hand", f"seq_{side}", f"window_{side}", f"mesh_{side}"]].itertuples(index=False):
            keys.add((row[3], int(row[4]), row[2], row[1], int(row[5])))  # (seq, window, hand, role, mesh)
    tasks = [(k, stem_of[k[0]], k[1], k[2], k[3], k[4]) for k in sorted(keys)]
    log.info("computing %d contact sets with %d processes", len(tasks), n_proc)
    sets = {}
    n_fb = 0
    with Pool(n_proc, initializer=_init_worker, initargs=(str(MESH_DICT),)) as pool:
        for key, pts, radius, fb in pool.imap_unordered(_work_sample, tasks, chunksize=32):
            sets[key] = (pts, radius, fb)
            n_fb += int(fb)
    log.info("contact sets done; %d/%d used top-%d soft fallback", n_fb, len(tasks), TOPK)
    pk = [((r.seq_i, int(r.window_i), r.hand, r.role, int(r.mesh_i)),
           (r.seq_j, int(r.window_j), r.hand, r.role, int(r.mesh_j))) for r in pairs.itertuples(index=False)]
    # ship only the sets each chunk needs
    chunks = [pk[i:i + 500] for i in range(0, len(pk), 500)]
    jobs = []
    for ch in chunks:
        need = {k for p in ch for k in p}
        jobs.append((ch, {k: sets[k] for k in need}))
    log.info("computing Chamfer for %d pairs in %d chunks", len(pk), len(chunks))
    res = []
    with Pool(n_proc) as pool:
        for r in pool.imap(_work_pairs, jobs):
            res.extend(r)
    res = np.asarray(res, dtype=np.float64)
    out = pairs[PAIR_KEY].copy()
    out["chamfer_raw_m"] = res[:, 0]
    out["chamfer_scalednorm"] = res[:, 1]
    out["n_pts_i"] = res[:, 2].astype(int)
    out["n_pts_j"] = res[:, 3].astype(int)
    out["fallback_i"] = [int(sets[a][2]) for a, _ in pk]
    out["fallback_j"] = [int(sets[b][2]) for _, b in pk]
    return out


# --------------------------------------------------------------------------- metrics
def _boot_group(args):
    """Sequence-level bootstrap for one group. dist: per pair-type arrays of (seq_i, seq_j, d)."""
    seqs, per_type, seed = args
    rng = np.random.default_rng(seed)
    seqs = np.asarray(seqs)
    n = len(seqs)
    stats = {pt: [] for pt in "ABC"}
    ratios = {"cii": [], "gp": []}
    seq_index = {s: i for i, s in enumerate(seqs)}
    enc = {pt: (np.array([seq_index[a] for a in v[0]]), np.array([seq_index[b] for b in v[1]]), v[2])
           for pt, v in per_type.items()}
    for _ in range(N_BOOT):
        draw = rng.integers(0, n, n)
        inset = np.zeros(n, bool)
        inset[draw] = True
        m = {}
        for pt in "ABC":
            if pt not in enc:
                m[pt] = np.nan
                continue
            ia, ib, d = enc[pt]
            keep = inset[ia] & inset[ib]
            m[pt] = d[keep].mean() if keep.any() else np.nan
            stats[pt].append(m[pt])
        ratios["cii"].append(m["B"] / m["C"] if np.isfinite(m["B"]) and np.isfinite(m["C"]) and m["C"] > 0 else np.nan)
        ratios["gp"].append(m["B"] / m["A"] if np.isfinite(m["B"]) and np.isfinite(m["A"]) and m["A"] > 0 else np.nan)
    def ci(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size < 10:
            return (np.nan, np.nan, x.size)
        return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), int(x.size))
    return {pt: ci(stats[pt]) for pt in "ABC"} | {"cii": ci(ratios["cii"]), "gp": ci(ratios["gp"])}


def aggregate(dist: pd.DataFrame, n_proc: int) -> pd.DataFrame:
    rows = []
    jobs, meta = [], []
    groups = list(dist.groupby(["category", "role", "hand"], sort=True))
    # pooled rows over all categories, per (role, hand)
    for (role, hand), g in dist.groupby(["role", "hand"], sort=True):
        groups.append((("ALL_pooled", role, hand), g))
    for rep, col in (("baselineA_raw", "chamfer_raw_m"), ("baselineA_scalednorm", "chamfer_scalednorm")):
        for (cat, role, hand), g in groups:
            seqs = sorted(set(g.seq_i) | set(g.seq_j))
            per_type = {}
            means, counts = {}, {}
            for pt in "ABC":
                d = g[g.pair_type == pt]
                counts[pt] = len(d)
                means[pt] = float(d[col].mean()) if len(d) else np.nan
                if len(d):
                    per_type[pt] = (d.seq_i.values, d.seq_j.values, d[col].values.astype(float))
            jobs.append((seqs, per_type, SEED))
            meta.append((rep, cat, role, hand, means, counts, len(seqs)))
    log.info("bootstrapping %d groups", len(jobs))
    with Pool(n_proc) as pool:
        cis = pool.map(_boot_group, jobs)
    for (rep, cat, role, hand, means, counts, nseq), c in zip(meta, cis):
        r = dict(representation=rep, category=cat, role=role, hand=hand, n_sequences=nseq)
        for pt in "ABC":
            r[f"n_pairs_{pt}"] = counts[pt]
            r[f"mean_{pt}"] = means[pt]
            r[f"ci_lo_{pt}"], r[f"ci_hi_{pt}"], _ = c[pt]
        r["cross_instance_invariance"] = means["B"] / means["C"] if counts["B"] and counts["C"] else np.nan
        r["cii_ci_lo"], r["cii_ci_hi"], r["cii_n_boot_valid"] = c["cii"]
        r["geometry_penalty"] = means["B"] / means["A"] if counts["B"] and counts["A"] else np.nan
        r["gp_ci_lo"], r["gp_ci_hi"], r["gp_n_boot_valid"] = c["gp"]
        rows.append(r)
    met = pd.DataFrame(rows)
    # macro rows: mean of per-category ratios (categories with all three pair types), per (role, hand)
    macro = []
    for (rep, role, hand), g in met[~met.category.str.startswith("ALL")].groupby(["representation", "role", "hand"]):
        ok = g[np.isfinite(g.cross_instance_invariance) & np.isfinite(g.geometry_penalty)]
        macro.append(dict(representation=rep, category="ALL_macro", role=role, hand=hand, n_sequences=int(ok.n_sequences.sum()),
                          n_pairs_A=int(ok.n_pairs_A.sum()), n_pairs_B=int(ok.n_pairs_B.sum()), n_pairs_C=int(ok.n_pairs_C.sum()),
                          mean_A=ok.mean_A.mean(), mean_B=ok.mean_B.mean(), mean_C=ok.mean_C.mean(),
                          cross_instance_invariance=ok.cross_instance_invariance.mean(),
                          geometry_penalty=ok.geometry_penalty.mean(), n_categories=len(ok)))
    return pd.concat([met, pd.DataFrame(macro)], ignore_index=True)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_proc", type=int, default=32)
    ap.add_argument("--skip_pairs", action="store_true", help="reuse existing contact_pairs.csv")
    ap.add_argument("--skip_dist", action="store_true", help="reuse existing baselineA_distances.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    w = pd.read_csv(ROOT / "window_index.csv")
    samples = build_samples(w)
    log.info("samples with touch>=%.1f: %d", TOUCH_MIN, len(samples))
    samples.to_csv(ROOT / "samples_index.csv", index=False)

    if args.skip_pairs and (ROOT / "contact_pairs.csv").exists():
        pairs = pd.read_csv(ROOT / "contact_pairs.csv")
    else:
        pairs, summ = build_pairs(samples)
        pairs.to_csv(ROOT / "contact_pairs.csv", index=False)
        summ.to_csv(ROOT / "contact_pairs_summary.csv", index=False)
        log.info("wrote %d pairs", len(pairs))

    if args.skip_dist and (ROOT / "baselineA_distances.csv").exists():
        dist = pd.read_csv(ROOT / "baselineA_distances.csv", comment="#")
    else:
        dist = compute_baselineA(pairs, samples, args.n_proc)
        header = (
            "# Baseline A: raw instance-space Chamfer distance between contact SETS (metres).\n"
            "# Contact set = instance vertices with hard occupancy > 0.5 (fallback: top-20 by soft if < 5), taken in each\n"
            "# mesh's OWN scan frame centred at its vertex centroid -- no scaling, no rotation, no correspondence.\n"
            "# Because different meshes have different scan frames/scales, cross-mesh distances here mix genuine contact\n"
            "# differences with arbitrary frame/scale differences: this is exactly the naive comparison a canonical\n"
            "# representation is supposed to beat. chamfer_scalednorm divides each mesh by its max radius (scale removed,\n"
            "# orientation still arbitrary). One row per pair in contact_pairs.csv (same key columns).\n"
        )
        with open(ROOT / "baselineA_distances.csv", "w") as f:
            f.write(header)
            dist.to_csv(f, index=False)
        log.info("wrote distances")

    met = aggregate(dist, args.n_proc)
    met.to_csv(ROOT / "baselineA_metrics.csv", index=False)
    log.info("wrote metrics (%d rows)", len(met))


if __name__ == "__main__":
    main()
