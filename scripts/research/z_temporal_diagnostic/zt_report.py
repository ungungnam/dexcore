#!/usr/bin/env python
"""Assemble report.md: the final diagnosis first, then the 12 sections (narrative/<nn>_<name>.md).    python zt_report.py
Every number of the prose is injected from the result tables: the narrative files use <<key:format>> placeholders that are resolved
against a value dictionary built from the CSVs (keys listed in OUT/report_values.json), and {{tableN}} / {{figN}} placeholders for the
tables and figures. Nothing in the report is typed by hand from a table.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import zt_common as Z

NARR = Z.OUT / "narrative"
SECTIONS = ["motivation", "stage2_failure", "hypotheses", "data_encoder", "local_prediction", "temporal_geometry", "vs_persistence", "vs_stage2", "taco_vs_arctic", "representative_transitions",
            "final_interpretation", "next_step"]
TITLES = ["Motivation", "Stage-2 failure being diagnosed", "Competing hypotheses", "Data / frozen Stage-1 encoder", "Local prediction experiment", "Temporal geometry analysis", "Comparison against persistence",
          "Comparison against Stage-2 open-loop prediction", "TACO vs ARCTIC", "Representative failure transitions", "Final interpretation", "Recommended next step"]
METHOD_LAB = {"persistence": "persistence z_t", "mean": "train-mean z", "shrink": "per-dimension shrinkage of z_t (mean reversion)", "ar1": "linear map of z_t alone (64 × 64 ridge)", "linear": "linear (ridge) on z_t, G, τ", "probe_notau": "probe without trajectory", "probe": "**nonlinear probe**", "gt_z (decoder floor)": "GT z_{t+h} (decoder floor)",
              "dense persistence C_t (reference)": "dense persistence C_t (reference)"}


def f(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def val(r, k):
    return getattr(r, k) if isinstance(r, tuple) else r[k]


def ci(r, k, d=3):
    return f"{f(val(r, k), d)} [{f(val(r, k + '_lo'), d)}, {f(val(r, k + '_hi'), d)}]"


def pc(r, k, d=1):
    return f"{val(r, k) * 100:.{d}f} % [{val(r, k + '_lo') * 100:.{d}f}, {val(r, k + '_hi') * 100:.{d}f}]"


def md_table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]) + "\n"


def load(n):
    p = Z.OUT / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def values(L, P, G, Q, S, T, D, DIR, B, SP, X, TC, K):
    V = {}
    for r in X.itertuples(index=False):                                        # post-hoc splits (zt_posthoc.py): test by subject x position, validation by position
        for k in ("gain_rmse", "gain_rmse_lo", "gain_rmse_hi", "n_pairs", "n_takes", "rmse_persistence", "share_of_persistence_sse"):
            V[f"x_{r.dataset}_{r.h}_{r.method}_{r.eval_split}_{r.subject}_{r.position}_{k}"] = float(getattr(r, k))
    for r in TC.itertuples(index=False):                                       # paired contrast probe - probe without trajectory
        V[f"tc_{r.dataset}_{r.h}_{r.subject}_{r.position}"] = float(r.gain_probe_minus_notau); V[f"tc_{r.dataset}_{r.h}_{r.subject}_{r.position}_lo"] = float(r.lo); V[f"tc_{r.dataset}_{r.h}_{r.subject}_{r.position}_hi"] = float(r.hi)
    for r in K.itertuples(index=False):                                        # kinds of low-dC / high-dz transitions at h = 4
        for k, v in r._asdict().items():
            if isinstance(v, (int, float, np.floating)) and k != "h":
                V[f"bk_{r.dataset}_{r.kind}_{k}"] = float(v)
    for ds in Z.DATASETS:                                                      # the examples of Figure 5 (median member of each kind)
        p = Z.OUT / "figures" / f"fig5_selection_{ds}.csv"
        if p.exists():
            for r in pd.read_csv(p).itertuples(index=False):
                kk = "weak" if "weak-contact" in r.kind else "structural" if "structural" in r.kind else "neither" if "neither" in r.kind else "A" if r.kind.startswith("low ΔC / low Δz") else "D"
                for k in ("dC", "dz", "dT", "norm_C_t", "flips"):
                    V[f"f5_{ds}_{kk}_{k}"] = float(getattr(r, k))
    POS = {"t = 0 (first frame of the window)": "t0", "t = 1 .. 7": "t1_7", "t >= 8": "t8"}
    for r in SP.itertuples(index=False):
        key = POS.get(r.split, "seen" if r.split.startswith("subject seen") else ("unseen" if r.split.startswith("subject not") else None))
        if key:
            for k in ("gain_rmse", "gain_rmse_lo", "gain_rmse_hi", "rmse_persistence", "rmse_method", "share_of_persistence_sse", "n_pairs"):
                V[f"sp_{r.dataset}_{r.h}_{r.method}_{key}_{k}"] = float(getattr(r, k))
    for r in L.itertuples(index=False):
        m = {"gt_z (decoder floor)": "gtz", "dense persistence C_t (reference)": "densepers"}.get(r.method, r.method)
        for k, v in r._asdict().items():
            if isinstance(v, (int, float, np.floating)) and k not in ("h",):
                V[f"p_{r.dataset}_{r.h}_{m}_{k}"] = float(v)
    for df, pre in ((G, "g"), (Q, "q")):
        for r in df.itertuples(index=False):
            for k, v in r._asdict().items():
                if isinstance(v, (int, float, np.floating)) and k != "h":
                    V[f"{pre}_{r.dataset}_{r.h}_{k}"] = float(v)
    for r in S.itertuples(index=False):
        md = {"from_start": "start", "all_frames": "all", "stage2_all_63_frames": "ctx", "rescue_mean_h4_h8": "rmean"}[r.mode]
        for k, v in r._asdict().items():
            if isinstance(v, (int, float, np.floating)) and k != "h":
                V[f"s_{r.dataset}_{r.h}_{md}_{k}"] = float(v)
    for r in T.itertuples(index=False):
        for k, v in r._asdict().items():
            if isinstance(v, (int, float, np.floating)) and k != "h":
                V[f"t_{r.dataset}_{r.variant}_{r.h}_{k}"] = float(v)
    for r in DIR.itertuples(index=False):
        for k, v in r._asdict().items():
            if isinstance(v, (int, float, np.floating)):
                V[f"dir_{r.dataset}_{k}"] = float(v)
    for r in B.itertuples(index=False):
        for k in ("dC_mean", "dz_mean", "ratio_dz_over_dC", "dT_mean"):
            V[f"b_{r.dataset}_{r.h}_{r.decile}_{k}"] = float(getattr(r, k))
    for r in D.itertuples(index=False):
        V[f"ans_{r.dataset}_{Z_Q[r.question]}"] = r.answer
    for ds in Z.DATASETS:
        cj = Z.read_json(Z.cache_path(ds).with_suffix(".json"))
        V.update({f"n_{ds}_{k}_seq": cj["n_sequences"][k] for k in cj["n_sequences"]}); V.update({f"n_{ds}_{k}_takes": cj["n_takes"][k] for k in cj["n_takes"]})
        V[f"md5_{ds}"] = cj["md5"]; V[f"fresh_{ds}"] = cj["fresh_encoding_max_abs_diff_in_train_std"]; V[f"dtau_{ds}"] = cj["d_tau"]
        V[f"zsd_lo_{ds}"], V[f"zsd_hi_{ds}"] = cj["z_train_std_range"]
        # derived quantities used in the prose
        V[f"mean_spearman_{ds}"] = float(np.mean([V[f"g_{ds}_{h}_spearman_C_z"] for h in Z.HORIZONS])); V[f"mean_quadB_{ds}"] = float(np.mean([V[f"q_{ds}_{h}_B_lowC_highz"] for h in Z.HORIZONS]))
        V[f"rescue_{ds}"], V[f"rescue_lo_{ds}"], V[f"rescue_hi_{ds}"] = V[f"s_{ds}_0_rmean_rescue"], V[f"s_{ds}_0_rmean_rescue_lo"], V[f"s_{ds}_0_rmean_rescue_hi"]
        # Stage-2 reference numbers (read from the Stage-2 report tables / saved curves, not typed)
        s2o = Z.S2.OUT; mm = pd.read_csv(s2o / "main_metrics.csv"); bn = pd.read_csv(s2o / "bottleneck_metrics.csv"); pc_ = pd.read_csv(s2o / "paired_comparisons.csv")
        for m in ("B0", "B1"):
            V[f"s2_ec_{m}_{ds}"] = float(mm[(mm.dataset == ds) & (mm.model == m) & (mm.metric == "E_C")].value.iloc[0])
        V[f"s2_oracle_{ds}"] = float(bn[(bn.dataset == ds) & (bn.model == "B1")].E_C_oracle_z.iloc[0])
        V[f"s2_b1_vs_b0_{ds}"] = float(pc_[(pc_.dataset == ds) & (pc_.a == "B1") & (pc_.b == "B0") & (pc_.metric == "E_C")].rel_improvement.iloc[0])
        cv = np.load(Z.ds_out(ds) / "stage2_curves.npz"); V[f"s2_end_{ds}"] = float(np.sqrt((cv["stage2"][48:] ** 2).mean())); V[f"s2_f1_{ds}"] = float(cv["stage2"][0])
        V[f"s2_end_hold_{ds}"] = float(np.sqrt((cv["hold_z0"][48:] ** 2).mean())); V[f"s2_end_mean_{ds}"] = float(np.sqrt((cv["train_mean"][48:] ** 2).mean()))        # same last 15 frames
        for h in Z.HORIZONS:                                                       # how much of the all-frames rescue is persistence of a GT state h frames old
            a_, p_, pe_ = V[f"s_{ds}_{h}_all_rmse_stage2"], V[f"s_{ds}_{h}_all_rmse_probe"], V[f"s_{ds}_{h}_all_rmse_persistence"]
            V[f"pers_share_all_{ds}_{h}"] = (a_ - pe_) / (a_ - p_)
            V[f"shrink_share_{ds}_{h}"] = V[f"p_{ds}_{h}_shrink_gain_rmse"] / V[f"p_{ds}_{h}_probe_gain_rmse"] if V[f"p_{ds}_{h}_probe_gain_rmse"] > 0 else float("nan")
            V[f"hold_gain_s2_{ds}_{h}"] = 1 - V[f"s_{ds}_{h}_start_rmse_stage2"] / V[f"s_{ds}_{h}_start_rmse_hold_z0"]      # Stage 2 vs holding the GT z0 at frame h
            V[f"hold_gain_probe_{ds}_{h}"] = 1 - V[f"s_{ds}_{h}_start_rmse_probe"] / V[f"s_{ds}_{h}_start_rmse_hold_z0"]
        for h in Z.HORIZONS:
            V[f"ar_share_{ds}_{h}"] = V[f"p_{ds}_{h}_ar1_gain_rmse"] / V[f"p_{ds}_{h}_probe_gain_rmse"] if V[f"p_{ds}_{h}_probe_gain_rmse"] > 0 else float("nan")
            V[f"best_gain_{ds}_{h}"] = max(V[f"p_{ds}_{h}_{m}_gain_rmse"] for m in ("probe", "linear", "ar1", "shrink", "probe_notau"))
            V[f"ratio_move_{ds}_{h}"] = V[f"g_{ds}_{h}_mean_dz"] / V[f"g_{ds}_{h}_mean_dC"]
            V[f"tau_share_{ds}_{h}"] = V[f"p_{ds}_{h}_probe_gain_rmse"] - V[f"p_{ds}_{h}_probe_notau_gain_rmse"]
            V[f"err_ratio_{ds}_{h}"] = V[f"p_{ds}_{h}_probe_rmse"] / V[f"p_{ds}_{h}_mean_rmse"]
            V[f"shrink_share_of_ar_{ds}_{h}"] = V[f"p_{ds}_{h}_shrink_gain_rmse"] / V[f"p_{ds}_{h}_ar1_gain_rmse"] if V[f"p_{ds}_{h}_ar1_gain_rmse"] > 0 else float("nan")
            V[f"notau_minus_ar_{ds}_{h}"] = V[f"p_{ds}_{h}_probe_notau_gain_rmse"] - V[f"p_{ds}_{h}_ar1_gain_rmse"]          # geometry and / or nonlinearity (not separated)
            V[f"val_gain_rmse_{ds}_{h}"] = 1 - float(np.sqrt(V[f"t_{ds}_probe_{h}_best_val_mse"] / V[f"t_{ds}_probe_{h}_val_persistence_mse"]))
        V[f"mean_quadC_{ds}"] = float(np.mean([V[f"q_{ds}_{h}_C_highC_lowz"] for h in Z.HORIZONS]))
        for h in Z.HORIZONS:                                                       # share of the VALIDATION MSE gain that comes from pairs starting at t < 8
            ga, ge, se = V[f"x_{ds}_{h}_probe_val_all_all_gain_rmse"], V[f"x_{ds}_{h}_probe_val_all_t0_7_gain_rmse"], V[f"x_{ds}_{h}_probe_val_all_t0_7_share_of_persistence_sse"]
            V[f"val_early_share_{ds}_{h}"] = se * (1 - (1 - ge) ** 2) / (1 - (1 - ga) ** 2)
    return V


Z_Q = {"Is z_{t+1} predictable from z_t?": "h1", "Is z_{t+4} predictable?": "h4", "Is z_{t+8} predictable?": "h8", "Does the learned predictor beat persistence?": "beat",
       "Does delta_z track actual contact change?": "track", "Does current GT z rescue Stage-2 prediction?": "rescue", "Is the bottleneck representation or open-loop dynamics?": "case"}


def table1(T):
    rows = [["model", f"residual MLP around persistence: ẑ_t+h = z_t + f([z_t | G | τ_t | τ_t+h]); input linear → {Z.PROBE['depth']} pre-LayerNorm residual blocks (width {Z.PROBE['width']}, {Z.PROBE['expansion']}× GELU expansion, dropout {Z.PROBE['dropout']}) → LayerNorm → linear (zero-initialised)"],
            ["inputs", "current GT latent z_t (64, standardised with TRAIN mean / std); static descriptor G (1032-D); object-trajectory windows τ_local,t and τ_local,t+h (object states at −8 … +8 around each frame, stride 2: " + " / ".join(f"{Z.LABEL[ds]} 2 × {int(T[(T.dataset == ds)].d_in.iloc[0] - 64 - 1032) // 2}" for ds in Z.DATASETS) + " values)"],
            ["not given", "future contact, future z, hand state, R2, wrench"],
            ["parameters", " / ".join(f"{Z.LABEL[ds]} {T[(T.dataset == ds) & (T.variant == 'probe')].n_params.iloc[0] / 1e6:.2f} M" for ds in Z.DATASETS) + " (one model per horizon h = 1, 4, 8)"],
            ["loss", "mean squared error in standardised z (no structural, relational or smoothness term)"],
            ["optimisation", f"AdamW lr {Z.TRAIN['lr']:g}, weight decay {Z.TRAIN['wd']}, {Z.TRAIN['warmup']} warm-up steps then cosine to {Z.TRAIN['final_lr_frac']}×, batch {Z.TRAIN['batch']} pairs, ≤ {Z.TRAIN['max_steps']} steps, validation every {Z.TRAIN['eval_every']} steps on all validation pairs, early stopping after ≥ {Z.TRAIN['min_steps']} steps with patience {Z.TRAIN['patience']}; best-validation state; one seed"],
            ["baselines / references", "persistence ẑ = z_t; train-mean z; per-dimension shrinkage a_d z_t,d + b_d (plain mean reversion, least squares on train); linear map of z_t alone (full 64 × 64 ridge: the linear autonomous dynamics of the latent, no G, no τ); linear ridge regression on the probe's inputs (λ chosen on validation for both ridges); ablation probe without the trajectory"],
            ["steps run / selected step (probe, h = 1 / 4 / 8)", " ; ".join(f"{Z.LABEL[ds]} " + " / ".join(f"{int(T[(T.dataset == ds) & (T.variant == 'probe') & (T.h == h)].steps.iloc[0])} → {int(T[(T.dataset == ds) & (T.variant == 'probe') & (T.h == h)].best_step.iloc[0])}" for h in Z.HORIZONS) for ds in Z.DATASETS)]]
    return md_table(["item", "setting"], rows)


def table2(L):
    out = ""
    for ds in Z.DATASETS:
        rows = []
        for h in Z.HORIZONS:
            for m in ("persistence", "mean", "shrink", "ar1", "linear", "probe_notau", "probe"):
                r = L[(L.dataset == ds) & (L.h == h) & (L.method == m)].iloc[0]
                rows.append([h, METHOD_LAB[m], ci(r, "rmse"), ci(r, "r2"), f(r.r2_dim_mean, 3), ci(r, "cos"), "–" if m in ("persistence", "mean") else ci(r, "cos_delta"), ci(r, "dec_E_C") if not np.isnan(r.get("dec_E_C", np.nan)) else "–",
                             ci(r, "dec_d_gtz") if not np.isnan(r.get("dec_d_gtz", np.nan)) else "–"])
            for m in ("gt_z (decoder floor)", "dense persistence C_t (reference)"):
                r = L[(L.dataset == ds) & (L.h == h) & (L.method == m)]
                if len(r):
                    rows.append([h, METHOD_LAB[m], "–", "–", "–", "–", "–", ci(r.iloc[0], "dec_E_C"), "0" if m.startswith("gt_z") else "–"])
        n = {h: int(L[(L.dataset == ds) & (L.h == h) & (L.method == "probe")].n_pairs.iloc[0]) for h in Z.HORIZONS}
        out += f"\n**{Z.LABEL[ds]}** (test pairs: {n[1]} / {n[4]} / {n[8]} for h = 1 / 4 / 8; 95 % take-cluster bootstrap intervals)\n\n" + md_table(
            ["h", "predictor", "RMSE_z", "R² (explained variance)", "mean per-dim R²", "cos(ẑ, z)", "cos(predicted change, true change)", "decoded E_C vs GT map (frozen D_z)", "‖D_z(ẑ) − D_z(z_GT)‖"], rows)
    return out


def table3(P):
    rows = []
    for ds in Z.DATASETS:
        for h in Z.HORIZONS:
            for m in ("shrink", "ar1", "linear", "probe_notau", "probe"):
                r = P[(P.dataset == ds) & (P.h == h) & (P.method == m)].iloc[0]
                rows.append([Z.LABEL[ds], h, METHOD_LAB[m], f(r.rmse_persistence), f(r.rmse_method), pc(r, "gain_rmse"), pc(r, "gain_mse")])
    return md_table(["dataset", "h", "predictor", "RMSE persistence", "RMSE predictor", "Gain_h = 1 − RMSE / RMSE_persistence", "MSE version (skill)"], rows)


def table4(G):
    rows = [[Z.LABEL[r.dataset], r.h, int(r.n_test), ci(r, "pearson_C_z", 2), ci(r, "spearman_C_z", 2), f(r.pearson_log_C_z, 2), ci(r, "spearman_T_z", 2), ci(r, "spearman_C_T", 2), f(r.partial_spearman_z_C_given_T, 2), f(r.partial_spearman_z_T_given_C, 2),
             f(r.mean_dC), f(r.mean_dz), f(r.mean_dz / r.mean_dC, 2), f(r.acf_z, 3), f(r.persistence_r2_z, 3), f(r.persistence_r2_C, 3)] for r in G.itertuples()]
    return md_table(["dataset", "h", "test transitions", "Pearson(ΔC, Δz)", "Spearman(ΔC, Δz)", "Pearson of logs", "Spearman(ΔT, Δz)", "Spearman(ΔC, ΔT)", "partial Spearman(Δz, ΔC | ΔT)", "partial Spearman(Δz, ΔT | ΔC)",
                     "mean ΔC", "mean Δz", "mean Δz / mean ΔC", "lag-h autocorrelation of z", "persistence R² of z", "persistence R² of C"], rows)


def table5(Q):
    rows = [[Z.LABEL[r.dataset], r.h, pc(r, "A_lowC_lowz"), pc(r, "B_lowC_highz"), pc(r, "C_highC_lowz"), pc(r, "D_highC_highz"), f"{r.indep_B_at_test_marginals * 100:.1f} % / {r.indep_C_at_test_marginals * 100:.1f} %",
             f"{r.P_highz_given_lowC * 100:.1f} %", f"{r.P_lowz_given_highC * 100:.1f} %",
             f"{r.frac_lowC * 100:.0f} / {r.frac_highC * 100:.0f} / {r.frac_lowz * 100:.0f} / {r.frac_highz * 100:.0f} %", f"{r.B_lowC_highz_teacher_rank:.2f} vs {r.A_lowC_lowz_teacher_rank:.2f}", f"{r.B_lowC_highz_flip_frac * 100:.0f} % vs {r.A_lowC_lowz_flip_frac * 100:.0f} %",
             f"{int(r.B_n)} ({int(r.B_n_takes)} takes)", f"{r.B_lowC_highz_weak_contact_frac * 100:.0f} % vs {r.all_weak_contact_frac * 100:.0f} %", f"{r.B_flip_frac * 100:.0f} % / {r.B_highT_frac * 100:.0f} %", f"{r.B_none_frac * 100:.0f} %",
             "–" if np.isnan(getattr(r, "B_lowC_highz_articulating_frac", np.nan)) else f"{r.B_lowC_highz_articulating_frac * 100:.0f} % vs {r.all_articulating_frac * 100:.0f} %"] for r in Q.itertuples()]
    return md_table(["dataset", "h", "A low ΔC / low Δz", "B low ΔC / high Δz", "C high ΔC / low Δz", "D high ΔC / high Δz", "B / C expected under independence at the test marginals", "P(high Δz | low ΔC)", "P(low Δz | high ΔC)",
                     "test share below / above the train thresholds (low C / high C / low z / high z)", "mean train-quantile rank of the teacher step in B vs A", "transitions with a participation flip in B vs A",
                     "B: transitions (takes)", "B starting from a weak-contact frame (‖C_t‖ below the train 10 % quantile) vs all test transitions", "B with a participation flip / with a teacher step above the train 70 % quantile",
                     "B with none of the three", "ARCTIC: B transitions in which the object articulates (step above the train 70 % quantile) vs all"], rows)


def table6(S):
    rows = []
    for ds in Z.DATASETS:
        for mode, lab in (("from_start", "target frame f = h; probe from the GT z₀"), ("all_frames", "all target frames f ≥ h; probe from the GT z_{f−h}")):
            for h in Z.HORIZONS:
                r = S[(S.dataset == ds) & (S.h == h) & (S["mode"] == mode)].iloc[0]
                rows.append([Z.LABEL[ds], lab, h, ci(r, "rmse_stage2"), ci(r, "rmse_probe"), f(r.rmse_persistence), pc(r, "rescue", 0), f"{f(r.r2_stage2, 2)} / {f(r.r2_probe, 2)} / {f(r.r2_persistence, 2)}", pc(r, "stage2_gain_over_persistence", 0)])
        r = S[(S.dataset == ds) & (S["mode"] == "stage2_all_63_frames")].iloc[0]
        rows.append([Z.LABEL[ds], "context: Stage 2 over all 63 frames", "–", ci(r, "rmse_stage2"), "–", f"{f(r.rmse_hold_z0)} (hold z₀)", "–", f"{f(r.r2_stage2, 2)} / – / –", "–"])
    return md_table(["dataset", "comparison", "h", "Stage-2 B1 RMSE (open loop from s₀)", "local probe RMSE", "persistence RMSE from the same GT state", "rescue = 1 − RMSE_probe / RMSE_Stage-2", "R²: Stage 2 / probe / persistence",
                     "Stage-2 gain over that persistence"], rows)


def table7(D):
    rows = []
    for q in D.question.unique():
        a = {ds: D[(D.dataset == ds) & (D.question == q)].iloc[0] for ds in Z.DATASETS}
        rows.append([q] + [f"**{a[ds].answer}**" for ds in Z.DATASETS] + [" / ".join(f"{Z.LABEL[ds]}: {a[ds].evidence}" for ds in Z.DATASETS)])
    return md_table(["Question", "TACO", "ARCTIC", "Evidence"], rows)


def table_bins(B):
    out = ""
    for ds in Z.DATASETS:
        rows = [[int(r.decile), int(r.n), f(r.dC_mean), f(r.dz_mean), f"{f(r.dz_q25)} – {f(r.dz_q75)}", f(r.ratio_dz_over_dC, 2), f(r.dT_mean)] for r in B[(B.dataset == ds) & (B.h == 4)].itertuples()]
        out += f"\n**{Z.LABEL[ds]}**, h = 4\n\n" + md_table(["train decile of ΔC", "test transitions", "mean ΔC", "mean Δz", "Δz interquartile range", "mean Δz / mean ΔC", "mean teacher step ΔT"], rows)
    return out


def table_dir(DIR):
    rows = [[Z.LABEL[r.dataset], int(r.n_transitions), ci(r, "cos_dz_nn", 2), ci(r, "cos_dz_random", 2), ci(r, "cos_dC_nn", 2), ci(r, "cos_dC_random", 2), f"{r.frac_nn_similar_dense_change * 100:.0f} %", f(r.cos_dz_nn_given_similar_dC, 2),
             f(r.cos_dz_nn_given_dissimilar_dC, 2), f(r.spearman_cos_dz_cos_dC, 2)] for r in DIR.itertuples()]
    return md_table(["dataset", "moving test transitions (h = 4)", "cos(Δz_i, Δz_j), j = nearest neighbour in [C_t, C_t+h] (other take, same mesh)", "same, random partner", "cos(ΔC_i, ΔC_j), nearest neighbour", "random partner",
                     "neighbours with cos(ΔC) ≥ 0.5", "cos(Δz) given cos(ΔC) ≥ 0.5", "given cos(ΔC) < 0.5", "Spearman of cos(Δz) and cos(ΔC) over neighbour pairs"], rows)


def table_splits(SP):
    out = ""
    for ds in Z.DATASETS:
        s = SP[SP.dataset == ds]; rows = []
        for kind in dict.fromkeys(s.split_kind):
            for sp in dict.fromkeys(s[s.split_kind == kind].split):
                for h in Z.HORIZONS:
                    g = {m: s[(s.split == sp) & (s.h == h) & (s.method == m)].iloc[0] for m in ("probe", "linear", "ar1", "shrink")}
                    rows.append([kind, sp, h, int(g["probe"].n_pairs), f"{g['probe'].share_of_persistence_sse * 100:.0f} %", f(g["probe"].rmse_persistence), f(g["probe"].rmse_method), pc(g["probe"], "gain_rmse"),
                                 f"{g['linear'].gain_rmse * 100:.1f} %", f"{g['ar1'].gain_rmse * 100:.1f} %", f"{g['shrink'].gain_rmse * 100:.1f} %"])
        out += f"\n**{Z.LABEL[ds]}**\n\n" + md_table(["split", "pairs", "h", "test pairs", "share of the persistence squared error", "RMSE persistence", "RMSE probe", "probe gain over persistence", "linear (ridge) gain", "linear map of z_t alone", "shrinkage"], rows)
    return out


def table_posthoc(X):
    """Table 13: the probe's gain by window position crossed with the ARCTIC subject split, and on the validation takes by position."""
    lab_pos = {"all": "all pairs", "t0": "t = 0", "t1_7": "t = 1 .. 7", "t0_7": "t = 0 .. 7", "t8": "t >= 8"}
    lab_sub = {"all": "all takes", "seen": "subjects seen in training", "unseen": "subject not in training (s10)"}
    out = ""
    for ds in Z.DATASETS:
        x = X[X.dataset == ds]; rows = []
        for ev, subs in (("test", ("all", "seen", "unseen") if ds == "arctic" else ("all",)), ("val", ("all",))):
            for sb in subs:
                for ps in ("all", "t0", "t1_7", "t8"):
                    g = {(m, h): x[(x.eval_split == ev) & (x.subject == sb) & (x.position == ps) & (x.method == m) & (x.h == h)] for m in ("probe", "linear") for h in Z.HORIZONS}
                    if not len(g[("probe", 4)]):
                        continue
                    r4 = g[("probe", 4)].iloc[0]
                    rows.append(["test" if ev == "test" else "validation", lab_sub[sb], lab_pos[ps], f"{int(r4.n_pairs)} ({int(r4.n_takes)})", f"{r4.share_of_persistence_sse * 100:.0f} %"] + [pc(g[("probe", h)].iloc[0], "gain_rmse") for h in Z.HORIZONS]
                                + [" / ".join(f"{g[('linear', h)].iloc[0].gain_rmse * 100:.1f}" for h in Z.HORIZONS) + " %" if len(g[("linear", 4)]) else "–"])
        out += f"\n**{Z.LABEL[ds]}**\n\n" + md_table(["pairs from", "takes", "window position", "pairs at h = 4 (takes)", "share of the persistence squared error at h = 4", "probe gain, h = 1", "probe gain, h = 4", "probe gain, h = 8",
                                                     "linear (ridge) gain, h = 1 / 4 / 8"], rows)
    return out


def table_tau(TC):
    lab_pos = {"all": "all pairs", "t0": "t = 0", "t1_7": "t = 1 .. 7", "t0_7": "t = 0 .. 7", "t8": "t >= 8"}
    rows = []
    for ds in Z.DATASETS:
        for ps in ("all", "t0_7", "t8"):
            g = {h: TC[(TC.dataset == ds) & (TC.subject == "all") & (TC.position == ps) & (TC.h == h)].iloc[0] for h in Z.HORIZONS}
            rows.append([Z.LABEL[ds], lab_pos[ps]] + [f"{g[h].gain_probe_minus_notau * 100:+.1f} [{g[h].lo * 100:+.1f}, {g[h].hi * 100:+.1f}]" for h in Z.HORIZONS])
    return md_table(["dataset", "window position", "h = 1", "h = 4", "h = 8"], rows)


def table_train(T):
    rows = [[Z.LABEL[r.dataset], r.variant, r.h, f"{r.n_params / 1e6:.2f} M", int(r.steps), int(r.best_step), r.stopped_by, f(r.val_persistence_mse, 4), f(r.best_val_mse, 4), f"{r.val_gain_mse * 100:.1f} %", f"{r.seconds:.0f} s"] for r in T.itertuples()]
    return md_table(["dataset", "probe", "h", "parameters", "steps", "selected step", "stopped by", "validation MSE of persistence", "best validation MSE", "validation MSE gain", "wall time"], rows)


def sanity_table():
    p = Z.OUT / "sanity_summary.json"
    if not p.exists():
        return "(sanity summary not available)\n"
    J = json.loads(p.read_text()); keys = sorted({k for ds in J for k in J[ds]})
    return md_table(["#", "check", "TACO", "ARCTIC", "how"], [[k.split("_", 1)[0], k.split("_", 1)[1].replace("_", " ")] + [J.get(ds, {}).get(k, {}).get("status", "–") for ds in Z.DATASETS] + [next((J[ds][k].get("how", "") for ds in J if k in J[ds]), "")] for k in keys])


def main():
    L, P, G, Q, S, T, D, DIR, B, SP, X, TC, K = (load(n) for n in ("local_prediction_metrics.csv", "persistence_comparison.csv", "temporal_geometry_metrics.csv", "quadrant_metrics.csv", "stage2_comparison.csv", "probe_training.csv",
                                                                    "decision_summary.csv", "directional_consistency.csv", "temporal_geometry_bins.csv", "local_prediction_splits.csv", "posthoc_splits.csv",
                                                                    "trajectory_contribution.csv", "quadrant_B_kinds.csv"))
    V = values(L, P, G, Q, S, T, D, DIR, B, SP, X, TC, K)
    Z.write_json(Z.OUT / "report_values.json", V)
    cap = lambda n, title, body: f"**Table {n} — {title}**\n\n{body}"
    tables = {"table1": cap(1, "Probe architecture / training setup", table1(T)), "table2": cap(2, "Local z prediction at h = 1, 4, 8 (test)", table2(L)), "table3": cap(3, "Improvement over persistence (test)", table3(P)),
              "table4": cap(4, "ΔC / Δz correlation (test transitions; ΔT = Stage-1 teacher step, exact R2 + wrench)", table4(G)), "table5": cap(5, "Temporal-geometry quadrant fractions (test; thresholds = train 30 % / 70 % quantiles)", table5(Q)),
              "table6": cap(6, "Local prediction vs Stage-2 open-loop prediction (same test sequences and target frames)", table6(S)), "table7": cap(7, "Final diagnostic decision (pre-registered rule; seven questions)", table7(D)),
              "table12": cap(12, "The probe's gain over persistence by position in the window (and, on ARCTIC, by whether the subject was seen in training)", table_splits(SP)), "table8": cap(8, "Probe training summary", table_train(T)),
              "table13": cap(13, "Post-hoc: the probe's gain over persistence by window position crossed with the ARCTIC subject split, and on the validation takes (not part of the rule)", table_posthoc(X)),
              "table14": cap(14, "Post-hoc: what the object trajectory adds — gain of the probe minus gain of the probe without trajectory, percentage points, paired take-cluster bootstrap (one seed per probe)", table_tau(TC)),
              "table9": cap(9, "Conditional mean of the latent step in train deciles of the dense step", table_bins(B)), "table10": cap(10, "Directional consistency (optional check)", table_dir(DIR)), "sanity": cap(11, "Sanity checks", sanity_table())}
    figs = {f"fig{i}": f"![Figure {i}](figures/{n})" for i, n in [(1, "fig1_local_predictability.png"), (2, "fig2_local_vs_stage2.png"), (3, "fig3_deltaC_vs_deltaz.png"), (4, "fig4_quadrants.png"), (6, "fig6_gain_by_window_position.png")]}
    figs["fig5"] = "\n\n".join(f"![Figure 5 {Z.LABEL[ds]}](figures/fig5_transitions_{ds}.png)" for ds in Z.DATASETS if (Z.OUT / "figures" / f"fig5_transitions_{ds}.png").exists())
    missing = set()

    def fill(txt):
        for k, v in {**tables, **figs}.items():
            txt = txt.replace("{{" + k + "}}", v)

        def sub(m):
            key, fmt = m.group(1), m.group(2)
            if key not in V:
                missing.add(key); return f"??{key}??"
            v = V[key]
            return format(v, fmt) if fmt else str(v)
        return re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", sub, txt)
    head = fill((NARR / "00_head.md").read_text()) if (NARR / "00_head.md").exists() else "# z temporal diagnostic\n"
    body = head + "\n"
    for i, (sec, title) in enumerate(zip(SECTIONS, TITLES), 1):
        p = NARR / f"{i:02d}_{sec}.md"; txt = fill(p.read_text()) if p.exists() else "_(section not written)_\n"
        body += f"\n## {i}. {title}\n\n{txt}\n"
    for name, title in (("90_sanity.md", "Appendix A. Sanity checks"), ("91_limitations.md", "Appendix B. Limitations"), ("92_reproduction.md", "Appendix C. Files and reproduction")):
        if (NARR / name).exists():
            body += f"\n## {title}\n\n{fill((NARR / name).read_text())}\n"
    (Z.OUT / "report.md").write_text(body)
    print("report.md:", len(body.splitlines()), "lines; unresolved placeholders:", body.count("{{"), "; missing values:", sorted(missing))


if __name__ == "__main__":
    main()
