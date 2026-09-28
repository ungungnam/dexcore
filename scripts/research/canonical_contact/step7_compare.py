"""Step 7 / Step 12: comparison tables across representations -> <root>/summary_tables.md (also printed).

Reads whatever exists among representation_metrics.csv (Step 6), nearest_neighbor_metrics.csv (Step 9),
probe_metrics.csv (Step 11), action_variance.csv (Step 8) and writes:
  1. Step-7 comparison on the primary axes (R, tool) and (L, target): Baseline A (raw, scalednorm),
     normalized, random_perm, aligned, dino (if present) -- mean A/B/C, geometry_penalty = B/A and
     cross_instance_invariance = B/C with sequence-bootstrap CIs, plus strict-B variants; ALL_pooled and ALL_macro.
  2. Per-category geometry_penalty / cross_instance_invariance for every representation (chosen distance).
  3. Step-12 decision numbers: the deltas that decide whether the canonical representation (aligned) beats the
     naive comparison (Baseline A) and the control (random_perm):
       D1 geometry penalty: GP(aligned) vs GP(baselineA_raw) and GP(normalized)      (lower = more instance-invariant)
       D2 function signal:  CII(aligned) vs CII(random_perm) and vs 1                (lower = same-verb pairs closer than cross-verb)
       D3 retrieval:        top-1 same-verb rate vs chance, and diff-mesh/same-verb share of top-1 neighbours
       D4 probes:           verb / mesh_id accuracy vs majority chance (GroupKFold by sequence)
       D5 action variance:  between/within ratio per backend
Usage: python scripts/step7_compare.py [--root R] [--distance l2] [--secondary cosine]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import add_common_args, setup_logging  # noqa: E402

log = setup_logging("step7")
REP_ORDER = ["baselineA_raw", "baselineA_scalednorm", "normalized", "random_perm", "aligned", "dino"]
AXES = [("tool", "R"), ("target", "L")]


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{x:.{nd}f}"


def ci(v, lo, hi, nd=3):
    return f"{fmt(v, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def rep_rank(r: str) -> int:
    return REP_ORDER.index(r) if r in REP_ORDER else len(REP_ORDER)


def select_rep_rows(met: pd.DataFrame, distance: str) -> pd.DataFrame:
    """Rows for the chosen distance for cache backends, plus the Baseline A rows (their own distance)."""
    base = met.representation.str.startswith("baselineA")
    sub = met[(met.distance == distance) | base].copy()
    sub["_rank"] = sub.representation.map(rep_rank)
    return sub.sort_values(["_rank", "representation"])


def step7_tables(met: pd.DataFrame, distance: str) -> str:
    parts = []
    for role, hand in AXES:
        for scope in ("ALL_pooled", "ALL_macro"):
            sub = select_rep_rows(met[(met.scope == scope) & (met.role == role) & (met.hand == hand)], distance)
            if sub.empty:
                continue
            rows = []
            for r in sub.itertuples(index=False):
                rows.append([r.representation, r.distance, int(r.n_pairs_A), int(r.n_pairs_B), int(r.n_pairs_C),
                             fmt(r.mean_A, 4), fmt(r.mean_B, 4), fmt(r.mean_C, 4),
                             ci(r.geometry_penalty, r.gp_ci_lo, r.gp_ci_hi),
                             ci(r.cross_instance_invariance, r.cii_ci_lo, r.cii_ci_hi),
                             ci(r.gp_cp, r.gp_cp_ci_lo, r.gp_cp_ci_hi), ci(r.cii_cp, r.cii_cp_ci_lo, r.cii_cp_ci_hi),
                             ci(r.gp_tr, r.gp_tr_ci_lo, r.gp_tr_ci_hi), ci(r.gp_sj, r.gp_sj_ci_lo, r.gp_sj_ci_hi)])
            parts.append(f"### ({hand}, {role}) - {scope}\n\n" + md_table(
                ["representation", "distance", "nA", "nB", "nC", "mean A", "mean B", "mean C",
                 "geometry_penalty B/A [95% CI]", "cross_instance_invariance B/C [95% CI]",
                 "GP strict counterpart", "CII strict counterpart", "GP strict triplet", "GP same subject"], rows))
    return "\n\n".join(parts)


def per_category_table(met: pd.DataFrame, distance: str, col: str, title: str) -> str:
    sub = select_rep_rows(met[met.scope == "category"], distance)
    if sub.empty:
        return ""
    reps = list(dict.fromkeys(sub.representation))
    piv = sub.pivot_table(index=["category", "role", "hand"], columns="representation", values=col, aggfunc="first")
    piv = piv.reindex(columns=[r for r in reps if r in piv.columns])
    rows = [[*idx, *[fmt(v) for v in vals]] for idx, vals in zip(piv.index, piv.values)]
    return f"### {title} ({distance})\n\n" + md_table(["category", "role", "hand", *piv.columns], rows)


def get(met, scope, role, hand, rep, distance, col):
    q = met[(met.scope == scope) & (met.role == role) & (met.hand == hand) & (met.representation == rep)]
    if not rep.startswith("baselineA"):
        q = q[q.distance == distance]
    return float(q[col].iloc[0]) if len(q) else np.nan


def decision_numbers(met, nn, probes, av, distance) -> str:
    parts = []
    for role, hand in AXES:
        rows = []
        if met is not None:
            for scope in ("ALL_pooled", "ALL_macro"):
                g = lambda rep, col: get(met, scope, role, hand, rep, distance, col)  # noqa: E731
                gp_al, gp_b, gp_bs, gp_n, gp_r = (g("aligned", "geometry_penalty"), g("baselineA_raw", "geometry_penalty"),
                                                  g("baselineA_scalednorm", "geometry_penalty"), g("normalized", "geometry_penalty"),
                                                  g("random_perm", "geometry_penalty"))
                ci_al, ci_r, ci_n, ci_b = (g("aligned", "cross_instance_invariance"), g("random_perm", "cross_instance_invariance"),
                                           g("normalized", "cross_instance_invariance"), g("baselineA_raw", "cross_instance_invariance"))
                rows += [
                    [scope, "D1 GP(aligned)", fmt(gp_al), f"CI [{fmt(g('aligned', 'gp_ci_lo'))}, {fmt(g('aligned', 'gp_ci_hi'))}]"],
                    [scope, "D1 GP(baselineA_raw) / GP(baselineA_scalednorm)", f"{fmt(gp_b)} / {fmt(gp_bs)}",
                     f"aligned/baselineA_raw = {fmt(gp_al / gp_b if gp_b else np.nan)}; aligned lower: {bool(gp_al < gp_b)}"],
                    [scope, "D1 GP(normalized) / GP(random_perm)", f"{fmt(gp_n)} / {fmt(gp_r)}",
                     f"aligned lower than normalized: {bool(gp_al < gp_n)}"],
                    [scope, "D2 CII(aligned)", fmt(ci_al), f"CI [{fmt(g('aligned', 'cii_ci_lo'))}, {fmt(g('aligned', 'cii_ci_hi'))}]; < 1: {bool(ci_al < 1)}"],
                    [scope, "D2 CII(random_perm) / CII(normalized) / CII(baselineA_raw)", f"{fmt(ci_r)} / {fmt(ci_n)} / {fmt(ci_b)}",
                     f"aligned - random_perm = {fmt(ci_al - ci_r)}; aligned lower: {bool(ci_al < ci_r)}"],
                ]
        if nn is not None:
            for scope in ("ALL_pooled", "ALL_macro"):
                for rep in ("aligned", "normalized", "random_perm", "dino"):
                    q = nn[(nn.scope == scope) & (nn.role == role) & (nn.hand == hand) & (nn.backend == rep)]
                    if q.empty:
                        continue
                    r = q.iloc[0]
                    rows.append([scope, f"D3 NN top-1 same verb ({rep})", ci(r.top1_verb, r.get("top1_verb_ci_lo", np.nan), r.get("top1_verb_ci_hi", np.nan)),
                                 f"chance {fmt(r.chance_verb)}; top-1 same mesh {fmt(r.top1_mesh_id)} (chance {fmt(r.chance_mesh_id)}); "
                                 f"diff-mesh/same-verb share {fmt(r.q_diff_mesh_same_verb)}, same-mesh/same-verb {fmt(r.q_same_mesh_same_verb)}"])
        if probes is not None:
            q = probes[(probes.scope == "ALL_pooled") & (probes.role == role) & (probes.hand == hand) & (probes.protocol == "groupkfold")]
            for r in q.itertuples(index=False):
                rows.append(["ALL_pooled", f"D4 probe {r.target} ({r.backend}, {r.features})", fmt(r.accuracy),
                             f"bal.acc {fmt(r.balanced_accuracy)}, macro-F1 {fmt(r.macro_f1)}, majority {fmt(r.chance_majority)}, "
                             f"uniform {fmt(r.chance_uniform)}, n_classes {int(r.n_classes)}"])
        if av is not None:
            q = av[(av.category == "ALL_macro") & (av.role == role) & (av.hand == hand)]
            for r in q.itertuples(index=False):
                rows.append(["ALL_macro", f"D5 between/within action variance ({r.backend})", fmt(r.ratio),
                             f"within {fmt(r.within_action_variance, 4)}, between {fmt(r.between_action_variance, 4)}, n_cat {int(r.n_categories)}"])
        if rows:
            parts.append(f"### ({hand}, {role})\n\n" + md_table(["scope", "quantity", "value", "context"], rows))
    return "\n\n".join(parts)


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__), with_backends=False)
    ap.add_argument("--distance", default="l2", help="primary distance for the comparison tables")
    ap.add_argument("--secondary", default="cosine", help="secondary distance (second set of tables)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root

    def maybe(name):
        p = root / name
        if p.exists():
            return pd.read_csv(p)
        log.warning("%s not found; section skipped", p)
        return None

    met, nn, probes, av = maybe("representation_metrics.csv"), maybe("nearest_neighbor_metrics.csv"), \
        maybe("probe_metrics.csv"), maybe("action_variance.csv")
    md = ["# Canonical contact study - summary tables", "",
          f"Root: `{root}`. Primary distance: `{args.distance}`; secondary: `{args.secondary}`. "
          "geometry_penalty = mean_B / mean_A (same verb: different mesh vs same mesh; 1 = instance-invariant). "
          "cross_instance_invariance = mean_B / mean_C (same verb different mesh vs different verb; < 1 = function "
          "dominates instance geometry). CIs: sequence-level bootstrap, 1000 draws, percentile 95%.", ""]
    if met is not None:
        for dist in (args.distance, args.secondary):
            md += [f"## Step 7 - representation comparison ({dist})", "", step7_tables(met, dist), ""]
        md += ["## Per-category geometry penalty", "", per_category_table(met, args.distance, "geometry_penalty", "geometry_penalty"), "",
               "## Per-category cross-instance invariance", "",
               per_category_table(met, args.distance, "cross_instance_invariance", "cross_instance_invariance"), ""]
    if nn is not None:
        sub = nn[nn.scope != "category"]
        rows = [[r.backend, r.scope, r.role, r.hand, int(r.n_queries), fmt(r.top1_verb), fmt(r.chance_verb), fmt(r.top5_any_verb),
                 fmt(r.top1_mesh_id), fmt(r.chance_mesh_id), fmt(r.top1_counterpart_cat), fmt(r.top1_triplet), fmt(r.top1_subject),
                 fmt(r.q_same_mesh_same_verb), fmt(r.q_same_mesh_diff_verb), fmt(r.q_diff_mesh_same_verb), fmt(r.q_diff_mesh_diff_verb)]
                for r in sub.itertuples(index=False)]
        md += ["## Step 9 - nearest-neighbour retrieval (cosine, excluding same sequence)", "",
               md_table(["backend", "scope", "role", "hand", "n", "top1 verb", "chance", "top5any verb", "top1 mesh", "chance",
                         "top1 counterpart", "top1 triplet", "top1 subject", "sm/sv", "sm/dv", "dm/sv", "dm/dv"], rows), ""]
    if probes is not None:
        sub = probes[probes.scope == "ALL_pooled"]
        rows = [[r.backend, r.role, r.hand, r.target, r.features, r.protocol, int(r.n_classes), int(r.n_train), int(r.n_test),
                 fmt(r.accuracy), fmt(r.balanced_accuracy), fmt(r.macro_f1), fmt(r.chance_majority), fmt(r.chance_uniform)]
                for r in sub.itertuples(index=False)]
        md += ["## Step 11 - linear probes (pooled across categories)", "",
               md_table(["backend", "role", "hand", "target", "features", "protocol", "n_classes", "n_train", "n_test",
                         "acc", "bal.acc", "macro-F1", "majority", "uniform"], rows), ""]
    if av is not None:
        sub = av[av.category == "ALL_macro"]
        rows = [[r.backend, r.role, r.hand, fmt(r.within_action_variance, 4), fmt(r.between_action_variance, 4), fmt(r.ratio),
                 fmt(r.ratio_seqw), int(r.n_categories)] for r in sub.itertuples(index=False)]
        md += ["## Step 8 - within vs between action variance (macro over categories)", "",
               md_table(["backend", "role", "hand", "within", "between", "ratio", "ratio (seq-weighted)", "n_cat"], rows), ""]
    md += ["## Step 12 - decision numbers", "", decision_numbers(met, nn, probes, av, args.distance), ""]
    text = "\n".join(md)
    out = args.out or root / "summary_tables.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
