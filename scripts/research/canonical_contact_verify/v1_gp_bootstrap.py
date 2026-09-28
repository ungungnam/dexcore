"""(a) Recompute pooled + per-category geometry_penalty (B/A) and CII (B/C) for every representation
from contact_pair_distances.csv with an independent sequence-level bootstrap (multiplicity-weighted:
pair weight = m_i * m_j where m = multinomial sequence counts). Also (c): scale controls -- cosine,
hard_l2, ratios normalised by the mean over ALL pairs, and (B-A)/A_sd.
Writes verify/gp_bootstrap_pooled.csv, verify/gp_bootstrap_category.csv."""
import numpy as np, pandas as pd, sys
R = "/result/uhnam/dexcore/canonical_contact/"
N_BOOT, SEED = 1000, 12345
t = pd.read_csv(R + "contact_pair_distances.csv")
print("pairs", len(t), t.pair_type.value_counts().to_dict())
REPS = {"baselineA_raw": "chamfer_raw_m", "baselineA_scalednorm": "chamfer_scalednorm",
        "normalized": "normalized_l2", "aligned": "aligned_l2", "dino": "dino_l2", "random_perm": "random_perm_l2",
        "normalized_cos": "normalized_cosine", "aligned_cos": "aligned_cosine", "dino_cos": "dino_cosine",
        "random_perm_cos": "random_perm_cosine",
        "normalized_hard": "normalized_hard_l2", "aligned_hard": "aligned_hard_l2", "dino_hard": "dino_hard_l2",
        "random_perm_hard": "random_perm_hard_l2",
        "normalized_l1": "normalized_l1", "aligned_l1": "aligned_l1", "dino_l1": "dino_l1"}

def boot_stats(g, n_boot=N_BOOT, seed=SEED, chunk=100):
    seqs = pd.unique(np.concatenate([g.seq_i.values, g.seq_j.values]))
    idx = {s: k for k, s in enumerate(seqs)}
    ia = g.seq_i.map(idx).values; ib = g.seq_j.map(idx).values
    n = len(seqs)
    rng = np.random.default_rng(seed)
    pt = g.pair_type.values
    masks = {"A": pt == "A", "B": pt == "B", "C": pt == "C"}
    out = []
    Dall = {rep: g[col].values.astype(np.float64) for rep, col in REPS.items()}
    # point estimates
    pe = {rep: {s: (np.nanmean(D[m]) if m.any() else np.nan) for s, m in masks.items()} for rep, D in Dall.items()}
    boot = {rep: {s: np.full(n_boot, np.nan) for s in masks} for rep in REPS}
    for c0 in range(0, n_boot, chunk):
        nb = min(chunk, n_boot - c0)
        m = np.zeros((nb, n), np.float32)
        draws = rng.integers(0, n, (nb, n))
        for b in range(nb):
            m[b] = np.bincount(draws[b], minlength=n)
        for s, mk in masks.items():
            if not mk.any():
                continue
            W = m[:, ia[mk]] * m[:, ib[mk]]                    # (nb, n_pairs_s)
            for rep, D in Dall.items():
                d = D[mk]; fin = np.isfinite(d)
                num = W[:, fin] @ d[fin]; den = W[:, fin].sum(1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot[rep][s][c0:c0 + nb] = np.where(den > 0, num / den, np.nan)
    rows = []
    for rep in REPS:
        r = dict(representation=rep, n_sequences=n)
        for s, mk in masks.items():
            r[f"n_{s}"] = int(mk.sum()); r[f"mean_{s}"] = pe[rep][s]
        allmean = np.nanmean(Dall[rep])
        r["mean_all"] = allmean
        with np.errstate(invalid="ignore", divide="ignore"):
            r["GP"] = pe[rep]["B"] / pe[rep]["A"]; r["CII"] = pe[rep]["B"] / pe[rep]["C"]
            gpb = boot[rep]["B"] / boot[rep]["A"]; ciib = boot[rep]["B"] / boot[rep]["C"]
            # scale controls: B and A divided by the overall-mean distance of this representation
            r["A_over_all"] = pe[rep]["A"] / allmean; r["B_over_all"] = pe[rep]["B"] / allmean
            r["C_over_all"] = pe[rep]["C"] / allmean
            # B-A gap in units of the pooled SD of distances (contrast)
            r["BminusA_over_sd"] = (pe[rep]["B"] - pe[rep]["A"]) / np.nanstd(Dall[rep])
        r["GP_lo"], r["GP_hi"] = np.nanpercentile(gpb, [2.5, 97.5]) if np.isfinite(gpb).any() else (np.nan, np.nan)
        r["CII_lo"], r["CII_hi"] = np.nanpercentile(ciib, [2.5, 97.5]) if np.isfinite(ciib).any() else (np.nan, np.nan)
        r["GP_boot"] = gpb
        rows.append(r)
    return rows

def paired_diff(rows, a, b, key="GP_boot"):
    ra = [r for r in rows if r["representation"] == a][0]; rb = [r for r in rows if r["representation"] == b][0]
    d = ra[key] - rb[key]
    return np.nanmean(d), np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5), float(np.nanmean(d > 0))

pooled, cats = [], []
for (role, hand), g in t.groupby(["role", "hand"]):
    rows = boot_stats(g)
    for r in rows:
        r.update(scope="pooled", role=role, hand=hand, category="ALL")
    # paired bootstrap differences (same draws): normalized - aligned, aligned - random_perm, aligned - baselineA
    for a, b in [("normalized", "aligned"), ("normalized", "dino"), ("dino", "aligned"), ("aligned", "random_perm"),
                 ("aligned", "baselineA_raw"), ("normalized", "baselineA_raw"), ("aligned", "baselineA_scalednorm"),
                 ("normalized_cos", "aligned_cos"), ("aligned_cos", "random_perm_cos"), ("normalized_hard", "aligned_hard")]:
        m, lo, hi, p = paired_diff(rows, a, b)
        pooled.append(dict(scope="pooled_diff", role=role, hand=hand, category="ALL", representation=f"{a} - {b}",
                           GP=m, GP_lo=lo, GP_hi=hi, frac_pos=p))
    pooled += rows
    print(f"\n== pooled {role} {hand}  n_seq={rows[0]['n_sequences']}")
    for r in rows:
        print(f"  {r['representation']:22s} A={r['mean_A']:.3f} B={r['mean_B']:.3f} C={r['mean_C']:.3f} "
              f"GP={r['GP']:.3f} [{r['GP_lo']:.3f},{r['GP_hi']:.3f}]  CII={r['CII']:.3f} [{r['CII_lo']:.3f},{r['CII_hi']:.3f}] "
              f"B/all={r['B_over_all']:.3f} A/all={r['A_over_all']:.3f} (B-A)/sd={r['BminusA_over_sd']:.3f}")
    sys.stdout.flush()
for (cat, role, hand), g in t.groupby(["category", "role", "hand"]):
    if (g.pair_type == "B").sum() == 0 or (g.pair_type == "A").sum() == 0:
        continue
    rows = boot_stats(g, n_boot=500)
    for r in rows:
        r.update(scope="category", role=role, hand=hand, category=cat)
    for a, b in [("normalized", "aligned"), ("normalized", "dino"), ("dino", "aligned"), ("aligned", "random_perm")]:
        m, lo, hi, p = paired_diff(rows, a, b)
        cats.append(dict(scope="category_diff", role=role, hand=hand, category=cat, representation=f"{a} - {b}",
                         GP=m, GP_lo=lo, GP_hi=hi, frac_pos=p))
    cats += rows
    print(f"{cat:12s} {role:6s} {hand}  " + "  ".join(f"{r['representation'][:10]}={r['GP']:.2f}[{r['GP_lo']:.2f},{r['GP_hi']:.2f}]"
          for r in rows if r["representation"] in ("normalized", "aligned", "dino", "random_perm", "baselineA_raw")))
    sys.stdout.flush()
df = pd.DataFrame(pooled).drop(columns=["GP_boot"], errors="ignore"); df.to_csv(R + "verify/gp_bootstrap_pooled.csv", index=False)
df = pd.DataFrame(cats).drop(columns=["GP_boot"], errors="ignore"); df.to_csv(R + "verify/gp_bootstrap_category.csv", index=False)
print("done")
