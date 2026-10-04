#!/usr/bin/env python
"""Side-by-side comparison of the dynamic-contact study on TACO, ARCTIC and OakInk2.

Reads each dataset's dynamic_contact outputs (exp1/ratios.csv, lag_curves.csv, transition_stats.csv,
exp2/lag_slopes.csv, prediction_ablation.csv, exp3/per_fold.csv, representation_ablation.csv,
verifier_verdicts.json) and writes
  /result/uhnam/dexcore/reports/dynamic_contact_crossdataset.md   (+ figures/crossdataset_*.png)
Every number is a macro mean over that dataset's groups unless stated. Group counts, the second
identity axis (mesh for TACO / OakInk2, subject for ARCTIC) and the second hold-out split follow
each dataset's dc_common configuration.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DS = {
    "TACO": dict(root=Path("/result/uhnam/dexcore/taco/40_representation_study/time_decomp/dynamic_contact"), ax2="mesh", split2="mesh"),
    "ARCTIC": dict(root=Path("/result/uhnam/dexcore/arctic/30_dynamic_contact"), ax2="subject", split2="subject"),
    "OakInk2": dict(root=Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact"), ax2="mesh", split2="mesh"),
    "OakInk2-ext": dict(root=Path("/result/uhnam/dexcore/oakink2/11_dynamic_contact_extended"), ax2="mesh", split2="mesh"),
}
REP = Path("/result/uhnam/dexcore/reports")
FIG = REP / "figures"


def md(df, fmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def exp1(name, cfg):
    p = cfg["root"] / "exp1" / "ratios.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p); r = r[r.stat == "mean"]
    ax2 = cfg["ax2"]
    rows = []
    for ratio in ["time/take", f"time/{ax2}", f"{ax2}/take", "subject/take_same_subject", "action/take", "time/floor"]:
        d = r[r.ratio == ratio]
        if len(d) == 0:
            continue
        for met in ["raw", "mass_abs", "mass_log", "pattern_L2", "centroid"]:
            v = d[d.metric == met].value
            rows.append(dict(dataset=name, ratio=ratio, metric=met, macro=float(v.mean()), n_groups=len(v), groups_gt_1=int((v > 1).sum())))
    return pd.DataFrame(rows)


def exp2(name, cfg):
    p = cfg["root"] / "lag_curves.csv"
    if not p.exists():
        return None
    CU = pd.read_csv(p); L = pd.read_csv(cfg["root"] / "exp1" / "four_axes_long.csv")
    L = L[(L.variant == "all") & (L.stat == "mean") & (L.axis == "take") & (L.tol == 0.10)]
    TS = pd.read_csv(cfg["root"] / "transition_stats.csv"); SL = pd.read_csv(cfg["root"] / "exp2" / "lag_slopes.csv")
    rows = []
    for g in CU.group.unique():
        rec = dict(dataset=name, group=g)
        for met in ["raw", "mass_abs", "pattern_L2", "centroid"]:
            ref = L[(L.group == g) & (L.metric == met)].value
            for lag in (8, 32):
                v = CU[(CU.group == g) & (CU.variant == "touching") & (CU.metric == met) & (CU.stat == "mean") & (CU.lag_frames == lag)].value
                rec[f"{met}_lag{lag}_over_take"] = float(v.iloc[0] / ref.iloc[0]) if len(v) and len(ref) else np.nan
            s = SL[(SL.group == g) & (SL.metric == met)]
            rec[f"{met}_slope_1_8"] = float(s.slope_1_8.iloc[0]) if len(s) else np.nan
        for k in (5, 10, 20):
            rec[f"top{k}"] = float(TS[(TS.group == g) & (TS.quantity == f"share_raw_top{k}") & (TS.stat == "mean")].value.iloc[0])
            rec[f"null{k}"] = float(TS[(TS.group == g) & (TS.quantity == f"null_top{k}")].value.iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def exp3(name, cfg):
    p = cfg["root"] / "prediction_ablation.csv"
    if not p.exists():
        return None
    A = pd.read_csv(p); rows = []
    for split in ("take", cfg["split2"]):
        d = A[A.split == split]
        m = d.pivot_table(index="model", values=["raw", "mass_abs", "pattern_L2", "centroid", "r2_mass"], aggfunc="mean")
        for model in ["base_cat_mean", "base_mesh_mean", "mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_Dc_coarse_window", "mlp_D_full_window", "oracle_seq_mean", "base_prev_frame"]:
            if model in m.index:
                rows.append(dict(dataset=name, split=split, split_kind="take" if split == "take" else "second", model=model, **m.loc[model].to_dict()))
    return pd.DataFrame(rows)


def exp4(name, cfg):
    p = cfg["root"] / "representation_ablation.csv"
    if not p.exists():
        return None
    A = pd.read_csv(p); rows = []
    O = A[A.kind == "oracle"]; ref = O[O.representation == "static_seq_mean"].groupby("group").raw.mean()
    for rep in ["static_window_mean", "factor_rank1_lsq", "factor_rank2_lsq", "sparse_hold_K2", "sparse_hold_K3", "sparse_hold_K4", "sparse_hold_K8"]:
        d = O[O.representation == rep].set_index("group")
        rows.append(dict(dataset=name, kind="oracle", split="none", representation=rep, raw_rel=float((d.raw / ref.loc[d.index]).mean()),
                         temporal_rel=float((d.temporal / O[O.representation == "static_seq_mean"].set_index("group").temporal.loc[d.index]).mean())))
    P = A[A.kind == "predicted"]
    for split in ("take", cfg["split2"]):
        d = P[P.split == split]; refm = d[d.representation == "base_mesh_mean"].set_index("group").raw
        for rep in ["pred_static_A", "pred_sparse_K1", "pred_factor_AxC", "pred_factor_CxC", "pred_sparse_K2", "pred_sparse_K4", "pred_dense_C", "oracle_seq_mean"]:
            dd = d[d.representation == rep].set_index("group")
            if len(dd):
                rows.append(dict(dataset=name, kind="predicted", split=split, representation=rep, raw_rel=float((dd.raw / refm.loc[dd.index]).mean()), temporal_rel=np.nan))
    return pd.DataFrame(rows)


def exp4w(name, cfg):
    p = cfg["root"] / "exp4" / "window_level.csv"
    if not p.exists():
        return None
    A = pd.read_csv(p)
    m = A.groupby("representation")[["raw_rel", "temporal_rel"]].mean().reset_index(); m["dataset"] = name
    return m


def main():
    REP.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    parts = {k: [] for k in ("exp1", "exp2", "exp3", "exp4", "exp4w")}
    for name, cfg in DS.items():
        for k, fn in (("exp1", exp1), ("exp2", exp2), ("exp3", exp3), ("exp4", exp4), ("exp4w", exp4w)):
            d = fn(name, cfg)
            if d is not None:
                parts[k].append(d)
    E1 = pd.concat(parts["exp1"]) if parts["exp1"] else pd.DataFrame()
    E2 = pd.concat(parts["exp2"]) if parts["exp2"] else pd.DataFrame()
    E3 = pd.concat(parts["exp3"]) if parts["exp3"] else pd.DataFrame()
    E4 = pd.concat(parts["exp4"]) if parts["exp4"] else pd.DataFrame()
    E4W = pd.concat(parts["exp4w"]) if parts["exp4w"] else pd.DataFrame()
    for k, d in (("exp1", E1), ("exp2", E2), ("exp3", E3), ("exp4", E4), ("exp4w", E4W)):
        d.to_csv(REP / f"crossdataset_{k}.csv", index=False)
    out = ["# Dynamic contact study across TACO, ARCTIC and OakInk2\n",
           "Macro means over each dataset's groups; source CSVs `reports/crossdataset_exp{1,2,3,4,4w}.csv`, built from each dataset's `dynamic_contact` outputs. "
           "Per-dataset reports: `taco/40_representation_study/time_decomp/dynamic_contact/dynamic_contact_report.md`, `arctic/30_dynamic_contact/dynamic_contact_report_arctic.md`, `oakink2/10_dynamic_contact/dynamic_contact_report_oakink2.md`.\n"]
    narrative = REP / "crossdataset_narrative.md"
    if narrative.exists():
        out.append(narrative.read_text())
    if len(E1):
        t = E1.pivot_table(index=["ratio", "metric"], columns="dataset", values="macro").reset_index()
        g = E1.pivot_table(index=["ratio", "metric"], columns="dataset", values="groups_gt_1").reset_index()
        n = E1.groupby("dataset").n_groups.max().to_dict()
        out.append("## Experiment 1 — pair-distance ratios (macro over groups)\n\nGroups: " + ", ".join(f"{k} {v}" for k, v in n.items()) + "\n\n" + md(t))
        out.append("\nNumber of groups with ratio > 1:\n\n" + md(g, "{:.0f}"))
    if len(E2):
        agg = E2.groupby("dataset").agg({c: "mean" for c in E2.columns if c not in ("dataset", "group")}).reset_index()
        out.append("## Experiment 2 — lag structure (macro over groups)\n\n" + md(agg.T.reset_index().rename(columns={"index": "quantity"}), "{:.3f}"))
    if len(E3):
        for sk in ("take", "second"):
            t = E3[E3.split_kind == sk].pivot_table(index="model", columns="dataset", values=["raw", "mass_abs", "pattern_L2", "r2_mass"])
            out.append(f"## Experiment 3 — held-out {'take' if sk == 'take' else 'mesh (TACO, OakInk2) / subject (ARCTIC)'}\n\n" + md(t.reset_index().set_axis([" ".join(map(str, c)).strip() for c in t.reset_index().columns], axis=1)))
    if len(E4):
        t = E4.pivot_table(index=["kind", "split", "representation"], columns="dataset", values="raw_rel").reset_index()
        out.append("## Experiment 4 — representation ablation (raw L2 relative to the static take mean [oracle] or the mesh mean [predicted])\n\n" + md(t))
    if len(E4W):
        t = E4W.pivot_table(index="representation", columns="dataset", values=["raw_rel", "temporal_rel"])
        t = t.reset_index().set_axis([" ".join(map(str, c)).strip() for c in t.reset_index().columns], axis=1)
        out.append("## Experiment 4 at 64-frame window granularity (oracle fits relative to the window mean; the granularity-free comparison)\n\n" + md(t))
    (REP / "dynamic_contact_crossdataset.md").write_text("\n\n".join(out))
    print("\n\n".join(out)[:6000])


if __name__ == "__main__":
    main()
