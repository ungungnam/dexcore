#!/usr/bin/env python
"""Markdown tables for dynamic_contact_report.md, each stamped with the CSV it came from.
Writes tables/*.md. Pooled = pair-weighted over all groups after dividing each group's distances
by its own phase-matched cross-take level (so categories with different raw scales can be
pooled); macro = unweighted mean of the per-group ratios.
"""
import numpy as np
import pandas as pd

import dc_common as C

T = C.OUT / "tables"
GROUPS = [C.gname(*g) for g in C.GROUPS]
SHORT = {g: g.replace("__target__", " target/").replace("__tool__", " tool/").replace("__obj__", " ") for g in GROUPS}
SPLIT2 = C.SPLIT2


def md(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) and not np.isnan(v) else ("" if isinstance(v, float) and np.isnan(v) else str(v)) for v in r.values) + " |")
    return "\n".join(lines)


def exp1_tables():
    out = []
    m = pd.read_csv(C.OUT / "metrics_four_axes.csv")
    L = pd.read_csv(C.OUT / "exp1" / "four_axes_long.csv")
    L = L[(L.variant == "all") & (L.stat == "mean")]
    mets = ["raw", "mass_abs", "mass_log", "pattern_L2", "pattern_L1", "pattern_cos", "centroid"]
    # per-group main table with CI
    rows = []
    for g in GROUPS:
        for axis in C.AXIS_NAMES:
            r = dict(group=SHORT[g], axis=axis)
            for met in mets:
                d = L[(L.group == g) & (L.axis == axis) & (L.metric == met)]
                if len(d):
                    r[met] = f"{d.value.iloc[0]:.3f} [{d.lo.iloc[0]:.3f}, {d.hi.iloc[0]:.3f}]"
            rows.append(r)
    out.append("### Table E1a. Raw pair distances per group and axis (mean over pairs, 95% sequence-bootstrap CI). Source: exp1/four_axes_long.csv (variant=all, stat=mean, tol=0.10)\n\n" + md(pd.DataFrame(rows)))
    # ratios per group
    R = pd.read_csv(C.OUT / "exp1" / "ratios.csv")
    rows = []
    for g in GROUPS:
        for ratio in [f"{a}/{b}" for a, b in C.RATIOS if b != "floor"]:
            r = dict(group=SHORT[g], ratio=ratio)
            for met in ("raw", "mass_abs", "pattern_L2", "centroid"):
                d = R[(R.group == g) & (R.ratio == ratio) & (R.metric == met) & (R.stat == "mean")]
                r[met] = f"{d.value.iloc[0]:.2f} [{d.lo.iloc[0]:.2f}, {d.hi.iloc[0]:.2f}]" if len(d) else ""
            rows.append(r)
    out.append("### Table E1b. Ratios of pair distances per group (paired sequence bootstrap, 95% CI). Source: exp1/ratios.csv (stat=mean)\n\n" + md(pd.DataFrame(rows)))
    # pooled and macro
    z = np.load(C.OUT / "exp1" / "pairs.npz", allow_pickle=True)
    rows = []
    for met in mets:
        r = dict(metric=met)
        ref = {g: float(np.nanmean(z[f"{g}__take__{met}"])) for g in GROUPS}
        for axis in ["time"] + C.CROSS_AXES:
            gs = [g for g in GROUPS if f"{g}__{axis}__{met}" in z.files]
            if not gs:
                continue
            pooled = np.concatenate([z[f"{g}__{axis}__{met}"] / ref[g] for g in gs])
            macro = np.mean([np.nanmean(z[f"{g}__{axis}__{met}"]) / ref[g] for g in gs])
            r[f"{axis} pooled"] = float(np.nanmean(pooled)); r[f"{axis} macro"] = float(macro)
        rows.append(r)
    out.append("### Table E1c. Pooled (pair-weighted) and macro (mean of groups) distances, each metric normalised by its own cross-take level. Source: exp1/pairs.npz\n\n" + md(pd.DataFrame(rows)))
    # zero transitions
    Z = pd.read_csv(C.OUT / "exp1" / "zero_transition_share.csv")
    zt = Z[Z.axis == "time"][["group", "n", "frac_both_zero", "frac_one_zero", "frac_both_nonzero", "d2share_one_zero", "raw_mean_all", "raw_mean_both_nonzero"]].copy()
    zt["group"] = zt.group.map(SHORT)
    out.append("### Table E1d. Zero-contact states among TIME pairs and the share of the squared time distance they carry. Source: exp1/zero_transition_share.csv\n\n" + md(zt))
    # sensitivity
    V = pd.read_csv(C.OUT / "verify" / "v1_v2_ordering_configs.csv")
    V = V[V.config.str.contains(r"raw\||mass_abs\||pattern_L2\||centroid\|", regex=True)]
    V = V[~V.config.str.contains("raw_L1")]
    ax2 = C.AX2
    out.append(f"### Table E1e. Ordering robustness: how many of the {len(GROUPS)} groups satisfy each inequality, and macro ratios, per configuration. Source: verify/v1_v2_ordering_configs.csv\n\n" +
               md(V[["config", "time_gt_take", f"time_gt_{ax2}", f"{ax2}_gt_take", "action_near_take", "macro_time_over_take", f"macro_time_over_{ax2}", f"macro_{ax2}_over_take"]]))
    inc = pd.read_csv(C.OUT / "exp1" / "incremental_rms.csv")
    inc["group"] = inc.group.map(SHORT)
    out.append("### Table E1f. Secondary: incremental RMS estimates sqrt(max(a^2-b^2,0)) under an approximate-orthogonality assumption (time=q(time,floor), take=q(take,floor), mesh=q(mesh,take), action=q(action,take)). Source: exp1/incremental_rms.csv\n\n" +
               md(inc[inc.metric.isin(["raw", "mass_abs", "pattern_L2"])]))
    (T / "exp1.md").write_text("\n\n".join(out))


def exp2_tables():
    out = []
    CU = pd.read_csv(C.OUT / "lag_curves.csv")
    for met in ("raw", "mass_abs", "pattern_L2", "centroid"):
        d = CU[(CU.variant == "touching") & (CU.metric == met) & (CU.stat == "mean")]
        p = d.pivot_table(index="group", columns="lag_frames", values="value").reindex(GROUPS)
        p.index = [SHORT[g] for g in p.index]; p.columns = [f"lag {c}f ({c/2:g} sampled, {c/30:.2f}s)" for c in p.columns]
        out.append(f"### Table E2a-{met}. Lag curve, {met}, touching frames, mean over takes. Source: lag_curves.csv (variant=touching, stat=mean)\n\n" + md(p.reset_index().rename(columns={'index': 'group'})))
    # lag at which the within-take distance reaches the phase-matched cross-take level (log-linear interpolation)
    L1 = pd.read_csv(C.OUT / "exp1" / "four_axes_long.csv"); L1 = L1[(L1.variant == "all") & (L1.stat == "mean")]
    rows = []
    for g in GROUPS:
        r = dict(group=SHORT[g])
        for met in ("raw", "mass_abs", "pattern_L2", "centroid"):
            d = CU[(CU.variant == "touching") & (CU.metric == met) & (CU.stat == "mean") & (CU.group == g)].sort_values("lag_frames")
            d = d[d.lag_frames <= 48]
            ref = L1[(L1.group == g) & (L1.axis == "take") & (L1.metric == met)].value.iloc[0]
            y = d.value.values / ref; x = d.lag_frames.values
            if (y >= 1).any():
                i = int(np.argmax(y >= 1)); 
                if i == 0:
                    r[met] = "<1"
                else:
                    lx = np.log(x[i - 1]) + (1 - y[i - 1]) / (y[i] - y[i - 1]) * (np.log(x[i]) - np.log(x[i - 1]))
                    r[met] = f"{np.exp(lx):.0f}"
            else:
                r[met] = f">48 (ratio {y[-1]:.2f} at 48)"
        rows.append(r)
    out.append("### Table E2e. Lag (original frames, 30 fps) at which the within-take distance reaches the phase-matched cross-take level, per metric. Source: lag_curves.csv + exp1/four_axes_long.csv\n\n" + md(pd.DataFrame(rows)))
    sl = pd.read_csv(C.OUT / "exp2" / "lag_slopes.csv"); sl["group"] = sl.group.map(SHORT)
    out.append("### Table E2b. Log-log slopes of the lag curves (1 = linear/ballistic growth, 0.5 = diffusive, 0 = flat). Source: exp2/lag_slopes.csv\n\n" + md(sl))
    TS = pd.read_csv(C.OUT / "transition_stats.csv")
    rows = []
    for g in GROUPS:
        r = dict(group=SHORT[g])
        for k in (5, 10, 20):
            d = TS[(TS.group == g) & (TS.quantity == f"share_raw_top{k}") & (TS.stat == "mean")]
            r[f"top{k}% share (raw)"] = f"{d.value.iloc[0]:.3f} [{d.lo.iloc[0]:.3f}, {d.hi.iloc[0]:.3f}]"
            r[f"top{k}% iid null"] = f"{TS[(TS.group == g) & (TS.quantity == f'null_top{k}')].value.iloc[0]:.3f}"
        for met in ("mass_abs", "pattern_L2"):
            d = TS[(TS.group == g) & (TS.quantity == f"share_{met}_top10") & (TS.stat == "mean")]
            r[f"top10% share ({met})"] = f"{d.value.iloc[0]:.3f}"
        d = TS[(TS.group == g) & (TS.quantity == "frac_big5_raw") & (TS.stat == "mean")]; r["frames > 5x median"] = f"{d.value.iloc[0]:.3f}"
        d = TS[(TS.group == g) & (TS.quantity == "share_big5_raw") & (TS.stat == "mean")]; r["their share"] = f"{d.value.iloc[0]:.3f}"
        d = TS[(TS.group == g) & (TS.quantity == "gini_raw") & (TS.stat == "mean")]; r["Gini of delta"] = f"{d.value.iloc[0]:.3f}"
        rows.append(r)
    out.append("### Table E2c. Concentration of frame-to-frame change (30 fps) within a take, mean over takes with 95% CI. Source: transition_stats.csv\n\n" + md(pd.DataFrame(rows)))
    dq = pd.read_csv(C.OUT / "exp2" / "delta_quantiles.csv"); dq["group"] = dq.group.map(SHORT)
    out.append("### Table E2d. Distribution of frame-to-frame change over all frames. Source: exp2/delta_quantiles.csv\n\n" + md(dq[dq.metric.isin(["raw", "mass_abs", "pattern_L2"])]))
    (T / "exp2.md").write_text("\n\n".join(out))


def exp3_tables():
    p = C.OUT / "prediction_ablation.csv"
    if not p.exists():
        return
    A = pd.read_csv(p)
    models = ["base_cat_mean", "base_mesh_mean", "mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_Dc_coarse_window", "mlp_D_full_window",
              "ridge_A_geometry", "ridge_B_current_state", "ridge_C_local_context", "ridge_Dc_coarse_window", "ridge_D_full_window", "base_prev_frame", "oracle_seq_mean"]
    out = []
    for split in ("take", SPLIT2):
        for met in ("raw", "mass_abs", "pattern_L2", "centroid"):
            d = A[(A.split == split) & (A.model.isin(models))]
            p_ = d.pivot_table(index="model", columns="group", values=met).reindex(models)
            p_ = p_[[g for g in GROUPS if g in p_.columns]]
            p_["macro"] = p_.mean(1)
            p_.columns = [SHORT.get(c, c) for c in p_.columns]
            out.append(f"### Table E3-{split}-{met}. Held-out {split}: {met} error per group (mean over test takes, 3 folds pooled). Source: prediction_ablation.csv\n\n" + md(p_.reset_index()))
        d = A[(A.split == split) & (A.model.isin(models))]
        p_ = d.pivot_table(index="model", columns="group", values="r2_mass").reindex(models); p_ = p_[[g for g in GROUPS if g in p_.columns]]; p_["macro"] = p_.mean(1); p_.columns = [SHORT.get(c, c) for c in p_.columns]
        out.append(f"### Table E3-{split}-r2mass. Held-out {split}: explained variance of the contact amount (1 - MSE/Var). Source: prediction_ablation.csv\n\n" + md(p_.reset_index()))
    act = C.OUT / "exp3" / "activation_summary.csv"
    if act.exists():
        out.append("### Table E3-activation. Zero-contact detection from predicted mass, pooled over groups and folds (post-hoc thresholds). Source: exp3/activation_summary.csv\n\n" + md(pd.read_csv(act)))
    (T / "exp3.md").write_text("\n\n".join(out))


def exp4_tables():
    p = C.OUT / "representation_ablation.csv"
    if not p.exists():
        return
    A = pd.read_csv(p)
    out = []
    O = A[A.kind == "oracle"]
    reps = ["static_seq_mean", "static_window_mean", "factor_rank1_mass", "factor_rank1_lsq", "factor_rank2_lsq", "factor_rank3_lsq", "factor_rank4_lsq"] + \
           [f"sparse_hold_K{k}" for k in (1, 2, 3, 4, 6, 8, 12, 16)] + [f"sparse_interp_K{k}" for k in (2, 4, 8, 16)] + [f"sparse_rule_k{k}" for k in (2, 3, 5)] + ["dense"]
    for met in ("raw", "mass_abs", "pattern_L2", "centroid", "temporal"):
        p_ = O.pivot_table(index="representation", columns="group", values=met).reindex(reps)[GROUPS]
        p_["macro"] = p_.mean(1); p_.columns = [SHORT.get(c, c) for c in p_.columns]
        out.append(f"### Table E4a-{met}. Oracle sufficiency: {met} error of each representation fitted to the take's own contact (mean over takes). Source: representation_ablation.csv (kind=oracle)\n\n" + md(p_.reset_index()))
    p_ = O.pivot_table(index="representation", columns="group", values="compression").reindex(reps)[GROUPS]; p_["macro"] = p_.mean(1); p_.columns = [SHORT.get(c, c) for c in p_.columns]
    out.append("### Table E4b. Complexity: parameters / (512 x T). Source: representation_ablation.csv\n\n" + md(p_.reset_index(), "{:.4f}"))
    ns = O[O.representation.str.startswith("sparse_rule")].pivot_table(index="representation", columns="group", values="n_states")[GROUPS]; ns.columns = [SHORT[c] for c in ns.columns]
    out.append("### Table E4c. Number of states produced by the change-point rule (new state when delta_t > k x take median). Source: representation_ablation.csv\n\n" + md(ns.reset_index(), "{:.1f}"))
    P = A[A.kind == "predicted"]
    preps = ["base_mesh_mean", "pred_static_A", "pred_factor_AxC", "pred_factor_CxC", "pred_factor_AxD", "pred_sparse_K1", "pred_sparse_K2", "pred_sparse_K4", "pred_sparse_K8",
             "pred_sparseD_K4", "pred_dense_B", "pred_dense_C", "pred_dense_D", "base_prev_frame", "oracle_seq_mean"]
    for split in ("take", SPLIT2):
        for met in ("raw", "mass_abs", "pattern_L2", "temporal"):
            p_ = P[P.split == split].pivot_table(index="representation", columns="group", values=met).reindex(preps)
            p_ = p_[[g for g in GROUPS if g in p_.columns]]
            p_["macro"] = p_.mean(1); p_.columns = [SHORT.get(c, c) for c in p_.columns]
            out.append(f"### Table E4d-{split}-{met}. Predicted representations, held-out {split}: {met} error. Source: representation_ablation.csv (kind=predicted)\n\n" + md(p_.reset_index()))
    (T / "exp4.md").write_text("\n\n".join(out))


if __name__ == "__main__":
    T.mkdir(exist_ok=True)
    exp1_tables(); exp2_tables(); exp3_tables(); exp4_tables()
    print("tables written:", sorted(p.name for p in T.glob("*.md")))
