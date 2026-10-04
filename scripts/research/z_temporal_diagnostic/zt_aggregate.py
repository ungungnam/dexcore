#!/usr/bin/env python
"""Merge the per-dataset tables, apply the pre-registered decision rule (zt_common.DECISION) and write the config.    python zt_aggregate.py
Writes OUT/local_prediction_metrics.csv, persistence_comparison.csv, temporal_geometry_metrics.csv, quadrant_metrics.csv,
temporal_geometry_bins.csv, directional_consistency.csv (merges), probe_training.csv, decision_summary.csv, experiment_config.json,
and runs zt_posthoc.py (posthoc_splits.csv, trajectory_contribution.csv, quadrant_B_kinds.csv: interpretation only, not rule inputs).
(stage2_comparison.csv is written by zt_stage2.py.)
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import torch

import zt_common as Z

QUESTIONS = ["Is z_{t+1} predictable from z_t?", "Is z_{t+4} predictable?", "Is z_{t+8} predictable?", "Does the learned predictor beat persistence?", "Does delta_z track actual contact change?",
             "Does current GT z rescue Stage-2 prediction?", "Is the bottleneck representation or open-loop dynamics?"]
# Rule outcomes, worded as what the rule checked. The causal reading attached to each case in the plan is quoted as a reading, not as a result.
CASE_TEXT = {"A": "Rule outcome: the probe's window-averaged gain over persistence is clear at h = 4 and 8 and the rescue from the GT z₀ is ≥ 10 %. Pre-registered reading of Case A: 'local dynamics predictable, the problem is "
                  "whole-trajectory open-loop prediction'; that explanation was not tested here (no rollout, no horizon beyond 8)",
             "B": "Rule outcome: persistence is strong at h = 1 and the probe's window-averaged gain is below 10 % at every horizon. Pre-registered reading of Case B: 'locally smooth, little learnable dynamics beyond "
                  "persistence'; this is a statement about the window average and does not hold for pairs starting at the first frame of a window (Table 12)",
             "C": "Rule outcome: Δz does not follow ΔC. Pre-registered reading of Case C: 'temporally poor latent geometry'",
             "D": "Rule outcome: clear gains at h = 4 and 8, rescue ≥ 25 % and strong alignment. Pre-registered reading of Case D: 'z predictable and well behaved, the problem is the Stage-2 formulation / optimisation'",
             "mixed": "Rule outcome: no single pre-registered case applies"}


def decide(ds, L, P, G, Q, S):
    R = Z.DECISION; rows = []
    pr = {h: L[(L.dataset == ds) & (L.h == h) & (L.method == "probe")].iloc[0] for h in Z.HORIZONS}
    pe = {h: L[(L.dataset == ds) & (L.h == h) & (L.method == "persistence")].iloc[0] for h in Z.HORIZONS}
    li = {h: L[(L.dataset == ds) & (L.h == h) & (L.method == "linear")].iloc[0] for h in Z.HORIZONS}
    nt = {h: L[(L.dataset == ds) & (L.h == h) & (L.method == "probe_notau")].iloc[0] for h in Z.HORIZONS}
    g = {h: G[(G.dataset == ds) & (G.h == h)].iloc[0] for h in Z.HORIZONS}; q = {h: Q[(Q.dataset == ds) & (Q.h == h)].iloc[0] for h in Z.HORIZONS}
    s0 = {h: S[(S.dataset == ds) & (S.h == h) & (S["mode"] == "from_start")].iloc[0] for h in Z.HORIZONS}; sa = {h: S[(S.dataset == ds) & (S.h == h) & (S["mode"] == "all_frames")].iloc[0] for h in Z.HORIZONS}
    lvl = lambda r2: "yes" if r2 >= R["r2_yes"] else ("partial" if r2 >= R["r2_partial"] else "no")
    for h, qn in zip(Z.HORIZONS, QUESTIONS[:3]):
        rows.append(dict(dataset=ds, question=qn, answer=lvl(pr[h].r2),
                         evidence=f"probe R² {pr[h].r2:.3f} [{pr[h].r2_lo:.3f}, {pr[h].r2_hi:.3f}], RMSE {pr[h].rmse:.3f}; persistence R² {pe[h].r2:.3f}, RMSE {pe[h].rmse:.3f}; train-mean RMSE {L[(L.dataset == ds) & (L.h == h) & (L.method == 'mean')].iloc[0].rmse:.3f}"))
    gl = {}
    for h in Z.HORIZONS:
        gl[h] = "clear" if (pr[h].gain_rmse >= R["gain_clear"] and pr[h].gain_rmse_lo > 0) else ("marginal" if (pr[h].gain_rmse >= R["gain_marginal"] and pr[h].gain_rmse_lo > 0) else "no")
    beat = " / ".join(gl[h] for h in Z.HORIZONS)                                  # the rule defines the label per horizon; no aggregation across horizons
    rows.append(dict(dataset=ds, question=QUESTIONS[3], answer=beat,
                     evidence="; ".join(f"h={h}: gain {pr[h].gain_rmse * 100:.1f} % [{pr[h].gain_rmse_lo * 100:.1f}, {pr[h].gain_rmse_hi * 100:.1f}] ({gl[h]}; linear {li[h].gain_rmse * 100:.1f} %, without trajectory {nt[h].gain_rmse * 100:.1f} %)" for h in Z.HORIZONS)))
    sp = float(np.mean([g[h].spearman_C_z for h in Z.HORIZONS])); qb = float(np.mean([q[h].B_lowC_highz for h in Z.HORIZONS]))
    track = "yes" if (sp >= R["spearman_yes"] and qb <= R["quadrant_few"]) else ("partial" if sp >= R["spearman_partial"] else "no")
    ib = [float(q[h].indep_B_at_test_marginals) for h in Z.HORIZONS]
    rows.append(dict(dataset=ds, question=QUESTIONS[4], answer=track,
                     evidence="Spearman(ΔC, Δz) " + " / ".join(f"{g[h].spearman_C_z:.2f}" for h in Z.HORIZONS) + " (h = 1 / 4 / 8); low-ΔC / high-Δz transitions " + " / ".join(f"{q[h].B_lowC_highz * 100:.1f} %" for h in Z.HORIZONS)
                     + f" (expected under independence at the test marginals: {min(ib) * 100:.1f}–{max(ib) * 100:.1f} %; 9 % on train by construction); high-ΔC / low-Δz " + " / ".join(f"{q[h].C_highC_lowz * 100:.1f} %" for h in Z.HORIZONS)))
    rm = S[(S.dataset == ds) & (S["mode"] == "rescue_mean_h4_h8")].iloc[0]; rescue = float(rm.rescue)
    res = "yes" if rescue >= R["rescue_yes"] else ("partial" if rescue >= R["rescue_partial"] else "no")
    rows.append(dict(dataset=ds, question=QUESTIONS[5], answer=res,
                     evidence=f"rule input: mean rescue at h = 4, 8 from the initial state {rescue * 100:+.1f} % [{rm.rescue_lo * 100:+.1f}, {rm.rescue_hi * 100:+.1f}] | from the initial state — " + "; ".join(f"h={h}: Stage-2 RMSE {s0[h].rmse_stage2:.3f} vs probe from GT z₀ {s0[h].rmse_probe:.3f} (rescue {s0[h].rescue * 100:+.0f} % [{s0[h].rescue_lo * 100:+.0f}, {s0[h].rescue_hi * 100:+.0f}])" for h in Z.HORIZONS)
                     + " | state h frames old, all target frames — " + "; ".join(f"h={h}: {sa[h].rmse_stage2:.3f} vs {sa[h].rmse_probe:.3f} ({sa[h].rescue * 100:+.0f} %)" for h in Z.HORIZONS)))
    predictable = gl[4] == "clear" and gl[8] == "clear" and pr[8].r2 >= R["r2_partial"]          # 'clear' includes the interval-excludes-zero condition of the rule
    pers_strong = pe[1].r2 >= R["persistence_strong_r2"]
    if sp < R["spearman_partial"] or (qb >= R["quadrant_many"] and pr[1].r2 < R["r2_yes"]):
        case = "C"
    elif predictable and rescue >= R["rescue_yes"] and sp >= R["spearman_yes"] and qb <= R["quadrant_few"]:
        case = "D"
    elif predictable and rescue >= R["rescue_partial"]:
        case = "A"
    elif pers_strong and all(pr[h].gain_rmse < R["gain_clear"] for h in Z.HORIZONS):
        case = "B"
    else:
        case = "mixed"
    rows.append(dict(dataset=ds, question=QUESTIONS[6], answer=f"Case {case}" if case != "mixed" else "mixed", case=case,
                     evidence=f"{CASE_TEXT[case]}. Rule inputs: predictable beyond persistence at h = 4 and 8: {bool(predictable)} (gains {pr[4].gain_rmse * 100:.1f} % / {pr[8].gain_rmse * 100:.1f} %, R² at h = 8 {pr[8].r2:.2f}); "
                              f"rescue (mean of h = 4, 8 from the initial state) {rescue * 100:+.1f} % [{rm.rescue_lo * 100:+.1f}, {rm.rescue_hi * 100:+.1f}]; Spearman {sp:.2f}; low-ΔC / high-Δz {qb * 100:.1f} %; persistence R² at h = 1 {pe[1].r2:.2f}"))
    return rows


def main():
    py = sys.executable
    for script in ("zt_eval_local.py", "zt_geometry.py"):
        subprocess.run([py, str(Z.HERE / script), "--merge"], check=True)
    L, P, G, Q = (pd.read_csv(Z.OUT / n) for n in ("local_prediction_metrics.csv", "persistence_comparison.csv", "temporal_geometry_metrics.csv", "quadrant_metrics.csv"))
    S = pd.read_csv(Z.OUT / "stage2_comparison.csv")
    trows, runs = [], {}
    for ds in Z.DATASETS:
        for v in Z.VARIANTS:
            for h in Z.HORIZONS:
                p = Z.ckpt_path(ds, v, h)
                if not p.exists():
                    continue
                ck = torch.load(p, map_location="cpu", weights_only=False)
                trows.append(dict(dataset=ds, variant=v, h=h, n_params=ck["n_params"], d_in=ck["d_in"], steps=ck["steps"], best_step=ck["best_step"], stopped_by=ck["stopped_by"], best_val_mse=ck["best_val"],
                                  val_persistence_mse=ck["val_persistence"], val_gain_mse=1 - ck["best_val"] / ck["val_persistence"], seconds=ck["seconds"]))
                runs[f"{ds}/{v}_h{h}"] = dict(ckpt=str(p), md5=Z.md5(p), best_step=int(ck["best_step"]), steps=int(ck["steps"]), n_params=int(ck["n_params"]))
    pd.DataFrame(trows).to_csv(Z.OUT / "probe_training.csv", index=False)
    subprocess.run([py, str(Z.HERE / "zt_posthoc.py")], check=True, stdout=subprocess.DEVNULL)          # post-hoc splits (not rule inputs): posthoc_splits.csv, trajectory_contribution.csv, quadrant_B_kinds.csv
    D = pd.DataFrame([r for ds in Z.DATASETS for r in decide(ds, L, P, G, Q, S)]); D.to_csv(Z.OUT / "decision_summary.csv", index=False)
    Z.write_json(Z.OUT / "experiment_config.json", dict(
        horizons=Z.HORIZONS, probe=Z.PROBE, variants=Z.VARIANTS, train=Z.TRAIN, ridge_lambdas=Z.RIDGE_LAMBDAS, quantiles=dict(low=Z.QUANT_LOW, high=Z.QUANT_HIGH), n_random_pairs=Z.N_RANDOM_PAIRS,
        decision_rule=Z.DECISION, n_boot=Z.N_BOOT, n_boot_correlations=300, seed=Z.SEED, runs=runs, cases={r.dataset: r.case for r in D[D.question == QUESTIONS[6]].itertuples()},
        frozen_encoder={ds: Z.read_json(Z.cache_path(ds).with_suffix(".json")) for ds in Z.DATASETS},
        stage2_inputs={ds: dict(b1_predictions=str(Z.stage2_preds_path(ds)), md5=Z.md5(Z.stage2_preds_path(ds)), b1_checkpoint=str(Z.S2.ckpt_path(ds, Z.S2.run_name("B1"))), b1_checkpoint_md5=Z.md5(Z.S2.ckpt_path(ds, Z.S2.run_name("B1")))) for ds in Z.DATASETS},
        code_md5={p.name: Z.md5(p) for p in sorted(Z.HERE.glob("*.py")) + sorted(Z.HERE.glob("*.sh"))}))
    print(D[["dataset", "question", "answer"]].to_string(index=False))


if __name__ == "__main__":
    main()
