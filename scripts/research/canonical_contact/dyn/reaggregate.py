#!/usr/bin/env python
"""Add medians and pooled rows to the Experiment-3 / -4 aggregates (mean + CI were there already).
pooled = mean over ALL test takes of every group of the take's error divided by its group's reference
(Exp 3: category-mean error of the same split; Exp 4: static take-mean error), so groups with different
raw scales can be pooled; macro = unweighted mean over the 8 groups of the same normalised quantity."""
import numpy as np
import pandas as pd

import dc_common as C

MET3 = ["raw", "mass_abs", "mass_log", "pattern_L2", "centroid"]
MET4 = ["raw", "mass_abs", "pattern_L2", "centroid", "temporal"]

# ---- Exp 3
PS = pd.read_csv(C.OUT / "exp3" / "per_sequence.csv")
A = pd.read_csv(C.OUT / "prediction_ablation.csv")
A = A[[c for c in A.columns if not c.endswith("_median")]]
med = PS.groupby(["group", "split", "model"])[MET3].median().add_suffix("_median").reset_index()
A = A.merge(med, on=["group", "split", "model"], how="left")
ref = PS[PS.model == "base_cat_mean"].groupby(["group", "split"])[MET3].mean()
N = PS.merge(ref.add_suffix("_ref").reset_index(), on=["group", "split"])
for m in MET3:
    N[m + "_rel"] = N[m] / N[m + "_ref"]
pooled = N.groupby(["split", "model"])[[m + "_rel" for m in MET3]].agg(["mean", "median"])
pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
macro = N.groupby(["group", "split", "model"])[[m + "_rel" for m in MET3]].mean().groupby(["split", "model"]).mean().add_suffix("_macro")
P3 = pooled.join(macro).reset_index()
P3.to_csv(C.OUT / "exp3" / "pooled_vs_macro.csv", index=False)
A.to_csv(C.OUT / "prediction_ablation.csv", index=False)
# ---- Exp 4
PS = pd.read_csv(C.OUT / "exp4" / "per_sequence.csv")
A = pd.read_csv(C.OUT / "representation_ablation.csv")
A = A[[c for c in A.columns if not c.endswith("_median")]]
med = PS.groupby(["group", "kind", "split", "representation"])[MET4].median().add_suffix("_median").reset_index()
A = A.merge(med, on=["group", "kind", "split", "representation"], how="left")
refo = PS[(PS.kind == "oracle") & (PS.representation == "static_seq_mean")].groupby("group")[MET4].mean()
N = PS.merge(refo.add_suffix("_ref").reset_index(), on="group")
for m in MET4:
    N[m + "_rel"] = N[m] / N[m + "_ref"]
pooled = N.groupby(["kind", "split", "representation"])[[m + "_rel" for m in MET4]].agg(["mean", "median"])
pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
macro = N.groupby(["group", "kind", "split", "representation"])[[m + "_rel" for m in MET4]].mean().groupby(["kind", "split", "representation"]).mean().add_suffix("_macro")
P4 = pooled.join(macro).reset_index()
P4.to_csv(C.OUT / "exp4" / "pooled_vs_macro.csv", index=False)
A.to_csv(C.OUT / "representation_ablation.csv", index=False)
pd.set_option("display.width", 250)
print(P3[P3.model.isin(["mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_D_full_window", "base_mesh_mean", "oracle_seq_mean"])][["split", "model", "raw_rel_mean", "raw_rel_median", "raw_rel_macro", "mass_abs_rel_mean", "mass_abs_rel_macro", "pattern_L2_rel_mean", "pattern_L2_rel_macro"]].round(3).to_string(index=False))
print(P4[(P4.kind == "oracle") & P4.representation.isin(["factor_rank1_lsq", "factor_rank2_lsq", "sparse_hold_K2", "sparse_hold_K4", "sparse_hold_K8"])][["representation", "raw_rel_mean", "raw_rel_median", "raw_rel_macro", "temporal_rel_mean", "temporal_rel_macro"]].round(3).to_string(index=False))
print(P4[(P4.kind == "predicted")][["split", "representation", "raw_rel_mean", "raw_rel_median", "raw_rel_macro"]].round(3).to_string(index=False))
