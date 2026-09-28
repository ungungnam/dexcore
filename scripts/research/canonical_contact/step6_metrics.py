"""Step 6: A/B/C pair metrics for every representation, in ONE table.

Input : <root>/contact_pair_distances.csv (Step 5; holds every representation's distance per pair)
Output: <root>/representation_metrics.csv

Rows: per (representation, distance, scope, category, role, hand[, verb]) with
  scope = 'category'    (category, role, hand)                       -- mirrors baselineA_metrics.csv
          'verb'        (category, role, hand, verb): A/B pairs of that verb, C pairs touching it
          'verb_pooled' (ALL_pooled, role, hand, verb): same, pooled over categories
          'role'        (ALL_pooled, role, hand='ALL'): pooled over categories and hands
          'ALL_pooled'  (ALL_pooled, role, hand)
          'ALL_macro'   (ALL_macro, role, hand): mean over categories with all three pair types
Columns mirror baselineA_metrics.csv (mean_A/B/C, ci_lo/hi, cross_instance_invariance = B/C,
geometry_penalty = B/A, *_ci_lo/hi, *_n_boot_valid, n_categories) plus B strict variants
(B_cp: same_counterpart_cat, B_tr: same_triplet_ex_mesh, B_sj: same_subject) with their own
gp_*/cii_* ratios. All CIs are sequence-level bootstrap (resample sequences with replacement, keep
pairs whose both sequences are drawn, --n-boot draws, percentile 95%); draws are identical across
representations/distances of the same group (same seed, same sequence set).

Representations: <backend> x {l1, l2, cosine} on X_soft, <backend> x hard_l2 on X_hard, and the two
Baseline A rows carried over (baselineA_raw / chamfer_m, baselineA_scalednorm / chamfer).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import (add_common_args, encode_sequences, percentile_ci, resolve_backends,  # noqa: E402
                       safe_ratio, seq_bootstrap_draws, setup_logging)

log = setup_logging("step6")

SUBSETS = ["A", "B", "C", "B_cp", "B_tr", "B_sj"]
RATIOS = {  # name -> (numerator subset, denominator subset)
    "cii": ("B", "C"), "gp": ("B", "A"),
    "cii_cp": ("B_cp", "C"), "gp_cp": ("B_cp", "A"),
    "cii_tr": ("B_tr", "C"), "gp_tr": ("B_tr", "A"),
    "cii_sj": ("B_sj", "C"), "gp_sj": ("B_sj", "A"),
}
RATIO_COL = {"cii": "cross_instance_invariance", "gp": "geometry_penalty"}


def representations(table: pd.DataFrame, backends: list[str]) -> list[tuple[str, str, str]]:
    """(representation, distance, column) triples present in the distance table."""
    reps = []
    for b in backends:
        for dist in ("l1", "l2", "cosine"):
            col = f"{b}_{dist}"
            if col in table.columns:
                reps.append((b, dist, col))
        if f"{b}_hard_l2" in table.columns:
            reps.append((b, "hard_l2", f"{b}_hard_l2"))
    if "chamfer_raw_m" in table.columns:
        reps.append(("baselineA_raw", "chamfer_m", "chamfer_raw_m"))
    if "chamfer_scalednorm" in table.columns:
        reps.append(("baselineA_scalednorm", "chamfer", "chamfer_scalednorm"))
    return reps


def subset_masks(g: pd.DataFrame, verb: str | None) -> dict[str, np.ndarray]:
    pt = g.pair_type.values
    if verb is None:
        A, B, C = pt == "A", pt == "B", pt == "C"
    else:
        vi, vj = g.verb_i.values, g.verb_j.values
        A = (pt == "A") & (vi == verb)
        B = (pt == "B") & (vi == verb)
        C = (pt == "C") & ((vi == verb) | (vj == verb))
    return {"A": A, "B": B, "C": C,
            "B_cp": B & (g.same_counterpart_cat.values == 1),
            "B_tr": B & (g.same_triplet_ex_mesh.values == 1),
            "B_sj": B & (g.same_subject.values == 1)}


def boot_group(g: pd.DataFrame, reps: list[tuple[str, str, str]], n_boot: int, seed: int,
               verb: str | None = None, chunk: int = 250) -> list[dict]:
    """Point estimates + bootstrap CIs for every representation on one group of pairs."""
    seqs, (ia, ib) = encode_sequences(g.seq_i.values, g.seq_j.values)
    n_seq = len(seqs)
    masks = subset_masks(g, verb)
    inset = seq_bootstrap_draws(n_seq, n_boot, seed)
    D = {col: g[col].values.astype(np.float64) for _, _, col in reps}
    # bootstrap means: {subset: {col: (n_boot,)}}
    boot = {s: {col: np.full(n_boot, np.nan) for col in D} for s in SUBSETS}
    for s, m in masks.items():
        if not m.any():
            continue
        sa, sb = ia[m], ib[m]
        Dm = {col: d[m] for col, d in D.items()}
        for c0 in range(0, n_boot, chunk):
            sub = inset[c0:c0 + chunk]
            keep = (sub[:, sa] & sub[:, sb]).astype(np.float32)  # (chunk, n_pairs_subset)
            for col, d in Dm.items():
                fin = np.isfinite(d)
                tot = keep @ np.where(fin, d, 0.0).astype(np.float32)
                cnt = keep @ fin.astype(np.float32)
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot[s][col][c0:c0 + chunk] = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)
    rows = []
    for rep, dist, col in reps:
        r = dict(representation=rep, distance=dist, n_sequences=n_seq)
        pe = {}
        for s in SUBSETS:
            m = masks[s]
            d = D[col][m]
            d = d[np.isfinite(d)]
            pe[s] = float(d.mean()) if d.size else np.nan
            r[f"n_pairs_{s}"] = int(d.size)
            r[f"mean_{s}"] = pe[s]
            r[f"ci_lo_{s}"], r[f"ci_hi_{s}"], _ = percentile_ci(boot[s][col])
        for name, (num, den) in RATIOS.items():
            val = float(safe_ratio(pe[num], pe[den]))
            lo, hi, nv = percentile_ci(safe_ratio(boot[num][col], boot[den][col]))
            if name in RATIO_COL:
                r[RATIO_COL[name]] = val
            else:
                r[name] = val
            r[f"{name}_ci_lo"], r[f"{name}_ci_hi"], r[f"{name}_n_boot_valid"] = lo, hi, nv
        rows.append(r)
    return rows


def run_groups(table: pd.DataFrame, reps, n_boot: int, seed: int, min_pairs_verb: int) -> pd.DataFrame:
    out = []

    def emit(rows, **tags):
        for r in rows:
            out.append(tags | r)

    # per (category, role, hand) and per verb inside it
    for (cat, role, hand), g in table.groupby(["category", "role", "hand"], sort=True):
        log.info("group %s/%s/%s: %d pairs", cat, role, hand, len(g))
        emit(boot_group(g, reps, n_boot, seed), scope="category", category=cat, role=role, hand=hand, verb="")
        for verb in sorted(set(g.verb_i) | set(g.verb_j)):
            m = subset_masks(g, verb)
            if m["A"].sum() + m["B"].sum() < min_pairs_verb:
                continue
            emit(boot_group(g, reps, n_boot, seed, verb=verb), scope="verb", category=cat, role=role, hand=hand, verb=verb)
    # pooled over categories per (role, hand), and per verb pooled
    for (role, hand), g in table.groupby(["role", "hand"], sort=True):
        log.info("pooled %s/%s: %d pairs", role, hand, len(g))
        emit(boot_group(g, reps, n_boot, seed), scope="ALL_pooled", category="ALL_pooled", role=role, hand=hand, verb="")
        for verb in sorted(set(g.verb_i) | set(g.verb_j)):
            m = subset_masks(g, verb)
            if m["A"].sum() + m["B"].sum() < min_pairs_verb:
                continue
            emit(boot_group(g, reps, n_boot, seed, verb=verb), scope="verb_pooled", category="ALL_pooled",
                 role=role, hand=hand, verb=verb)
    # per role pooled over hands
    for role, g in table.groupby("role", sort=True):
        emit(boot_group(g, reps, n_boot, seed), scope="role", category="ALL_pooled", role=role, hand="ALL", verb="")
    met = pd.DataFrame(out)
    # macro rows: mean of per-category values over categories with all three pair types
    macro = []
    cat_rows = met[met.scope == "category"]
    for (rep, dist, role, hand), g in cat_rows.groupby(["representation", "distance", "role", "hand"], sort=True):
        ok = g[np.isfinite(g.cross_instance_invariance) & np.isfinite(g.geometry_penalty)]
        r = dict(scope="ALL_macro", category="ALL_macro", role=role, hand=hand, verb="", representation=rep,
                 distance=dist, n_sequences=int(ok.n_sequences.sum()), n_categories=len(ok))
        for s in SUBSETS:
            r[f"n_pairs_{s}"] = int(ok[f"n_pairs_{s}"].sum())
            r[f"mean_{s}"] = float(ok[f"mean_{s}"].mean()) if len(ok) else np.nan
        for name in RATIOS:
            colname = RATIO_COL.get(name, name)
            vals = ok[colname]
            r[colname] = float(vals[np.isfinite(vals)].mean()) if np.isfinite(vals).any() else np.nan
        macro.append(r)
    met = pd.concat([met, pd.DataFrame(macro)], ignore_index=True)
    met["n_categories"] = met.get("n_categories", np.nan)
    front = ["representation", "distance", "scope", "category", "role", "hand", "verb", "n_sequences"]
    cols = front + [c for c in met.columns if c not in front]
    return met[cols]


def headline(met: pd.DataFrame, axes=(("tool", "R"), ("target", "L"))) -> str:
    """geometry_penalty and cross_instance_invariance per representation for the primary axes."""
    lines = []
    for role, hand in axes:
        for scope in ("ALL_pooled", "ALL_macro"):
            sub = met[(met.scope == scope) & (met.role == role) & (met.hand == hand)]
            if sub.empty:
                continue
            lines.append(f"\n=== ({hand}, {role})  {scope} ===")
            t = sub[["representation", "distance", "n_pairs_A", "n_pairs_B", "n_pairs_C",
                     "mean_A", "mean_B", "mean_C", "geometry_penalty", "gp_ci_lo", "gp_ci_hi",
                     "cross_instance_invariance", "cii_ci_lo", "cii_ci_hi"]].copy()
            lines.append(t.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return "\n".join(lines)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--distances", type=Path, default=None, help="default <root>/contact_pair_distances.csv")
    ap.add_argument("--out", type=Path, default=None, help="default <root>/representation_metrics.csv")
    ap.add_argument("--min-pairs-verb", type=int, default=5, dest="min_pairs_verb",
                    help="skip per-verb rows with fewer A+B pairs than this")
    args = ap.parse_args()
    root = args.root
    backends = resolve_backends(root, args.backends)
    table = pd.read_csv(args.distances or root / "contact_pair_distances.csv")
    reps = representations(table, backends)
    log.info("%d pairs; representations: %s", len(table), [(r, d) for r, d, _ in reps])
    met = run_groups(table, reps, args.n_boot, args.seed, args.min_pairs_verb)
    out = args.out or root / "representation_metrics.csv"
    met.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(met))
    print(headline(met))


if __name__ == "__main__":
    main()
