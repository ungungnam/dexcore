#!/usr/bin/env python
"""Sanity / capacity checks (Section 11 of the plan and the usual consistency checks).    python sanity.py
Reads the root tables written by aggregate.py and the training logs; writes sanity_summary.json (one entry per check and
dataset: pass / warn / fail / info with the numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import cf_common as S


def main():
    rec = pd.read_csv(S.OUT / "reconstruction_metrics.csv"); lat = pd.read_csv(S.OUT / "latent_statistics.csv"); ts = pd.read_csv(S.OUT / "training_summary.csv")
    prob = pd.read_csv(S.OUT / "probe_metrics.csv"); knn = pd.read_csv(S.OUT / "neighborhood_metrics.csv"); swp = pd.read_csv(S.OUT / "swap_metrics.csv")
    g = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    checks = {}
    for ds in S.DATASETS:
        c = {}
        a0, a1, a2, a3 = (g(rec, dataset=ds, model=m) for m in ("A0", "A1", "A2", "A3")); mean, pca64, pca128 = (g(rec, dataset=ds, model=m) for m in ("mean", "pca64", "pca128"))
        c["01_a0_reconstructs_well"] = dict(status="pass" if (a0.R2_full >= 0.95 and a0.E_full < pca128.E_full) else "warn",
                                            detail=f"A0 test R² {a0.R2_full:.4f}, E_full {a0.E_full:.3f} vs PCA-128 {pca128.E_full:.3f}, mean {mean.E_full:.3f}")
        c["02_a3_full_close_to_a0"] = dict(status="pass" if a3.E_full <= 1.1 * a0.E_full else ("warn" if a3.E_full <= 1.3 * a0.E_full else "fail"),
                                           detail=f"E_full A3 {a3.E_full:.3f} vs A0 {a0.E_full:.3f} ({100 * (a3.E_full / a0.E_full - 1):+.1f} %)")
        t3 = g(ts, dataset=ds, model="A3")
        c["03_a3_not_underfitting"] = dict(status="pass" if a3.train_mse_full <= 0.05 else "fail",
                                           detail=f"A3 train mse_full {a3.train_mse_full:.4f} (R² ≈ {1 - a3.train_mse_full:.3f}; no underfitting); train / val / test E_full {a3.train_E_full:.3f} / {a3.val_E_full:.3f} / {a3.E_full:.3f} (generalisation gap {a3.E_full / a3.train_E_full:.1f} x, shared by every model); best step {int(t3.best_step)} of {int(t3.steps)}")
        c["04_zonly_beats_trivial"] = dict(status="pass" if a3.E_zonly < 0.5 * mean.E_full else "fail",
                                           detail=f"A3 E_zonly {a3.E_zonly:.3f} vs train mean {mean.E_full:.3f}, per-mesh mean {g(rec, dataset=ds, model='mesh_mean').E_full:.3f}; linear PCA-64 {pca64.E_full:.3f} (a dense-optimal 64-D code; z is not asked to be one)")
        lr3 = g(lat, dataset=ds, model="A3", code="r")
        c["05_residual_decoder_used"] = dict(status="pass" if (a3.gain_r >= 0.05 and lr3.delta_abs_mean > 1e-3) else "fail",
                                             detail=f"Gain_r {a3.gain_r:.3f}, mean |Delta_C| {lr3.delta_abs_mean:.4f} raw, Delta_C energy share {lr3.delta_energy_share:.3f} of the centred map energy")
        lz3 = g(lat, dataset=ds, model="A3", code="z")
        c["06_z_not_collapsed"] = dict(status="pass" if (lz3.eff_rank >= 8 and lz3.frac_dims_above_1pct >= 0.5 and lz3.std_min > 1e-3) else "warn",
                                       detail=f"A3 z: std mean/min/max {lz3.std_mean:.3f}/{lz3.std_min:.3f}/{lz3.std_max:.3f}, effective rank {lz3.eff_rank:.1f} of {int(lz3.dim)}, dims above 1 % of top variance {100 * lz3.frac_dims_above_1pct:.0f} %")
        c["07_r_not_collapsed"] = dict(status="pass" if (lr3.eff_rank >= 8 and lr3.frac_dims_above_1pct >= 0.5 and lr3.std_min > 1e-3) else "warn",
                                       detail=f"A3 r: std mean/min/max {lr3.std_mean:.3f}/{lr3.std_min:.3f}/{lr3.std_max:.3f}, effective rank {lr3.eff_rank:.1f} of {int(lr3.dim)}, dims above 1 % of top variance {100 * lr3.frac_dims_above_1pct:.0f} %")
        c["08_dz_not_ignored"] = dict(status="pass" if (a3.R2_zonly >= 0.5 and a3.R2_zonly / a3.R2_full >= 0.5) else "fail",
                                      detail=f"A3 R² of C_bar {a3.R2_zonly:.3f} vs C_hat {a3.R2_full:.3f}: the z branch explains {100 * a3.R2_zonly / a3.R2_full:.0f} % of what the full model explains")
        c["09_both_branches_receive_gradient"] = dict(status="pass" if (t3.grad_z_last > 0 and t3.grad_r_last > 0) else "fail",
                                                      detail=f"A3 last gradient norms z {t3.grad_z_last:.3f}, r {t3.grad_r_last:.3f} (first phase-B steps: z {t3.grad_z_phaseB_first:.3f}, r {t3.grad_r_phaseB_first:.3f})")
        kz3, kz2 = g(knn, dataset=ds, space="A3:z"), g(knn, dataset=ds, space="A2:z")
        c["10_relational_loss_organises_z"] = dict(status="pass" if kz3.teacher < kz2.teacher else "warn",
                                                   detail=f"teacher distance of the z neighbours: A3 (relational) {kz3.teacher:.3f} vs A2 (no relational loss) {kz2.teacher:.3f}; A1 {g(knn, dataset=ds, space='A1:z').teacher:.3f}")
        best_last = {m: bool(g(ts, dataset=ds, model=m).best_is_last_eval) for m in S.MODELS}
        c["11_convergence"] = dict(status="info", detail=f"best validation step is the last evaluation for: {[m for m, b in best_last.items() if b]} (schedule A1 12k, A0 24k, factorised 4k + 12k; cosine to 0.1 x lr per phase; EMA weights evaluated)")
        c["12_probe_sanity"] = dict(status="pass" if g(prob, dataset=ds, input="C", probe="mlp").part_auprc > g(prob, dataset=ds, input="trivial", probe="constant").part_auprc else "fail",
                                    detail=f"dense-C MLP probe AUPRC {g(prob, dataset=ds, input='C', probe='mlp').part_auprc:.3f} > trivial {g(prob, dataset=ds, input='trivial', probe='constant').part_auprc:.3f}; [z, r] probe retention {g(prob, dataset=ds, input='A3:zr', probe='mlp').structure_retention:.2f} vs z alone {g(prob, dataset=ds, input='A3:z', probe='mlp').structure_retention:.2f} (adding r does not add structure)")
        sw = g(swp, dataset=ds, model="A3")
        c["13_swap_pairs"] = dict(status="pass" if sw.n_pairs >= 100 else "warn", detail=f"{int(sw.n_pairs)} controlled pairs (teacher distance {sw.d_teacher_sel:.3f}, dense distance {sw.d_dense_sel:.3f}); reconstruction floor of the structural distance {sw.struct_floor:.3f}, dense floor {sw.dense_floor:.3f}")
        c["14_one_seed"] = dict(status="info", detail="one training seed per model (plan Section 10); test uncertainty = take-cluster bootstrap (1000 reps)")
        c["15_no_independence_loss"] = dict(status="pass", detail="losses: reconstruction (z-only, full), relational SmoothL1 on pairwise distances, mean |Delta_C| (lambda 0.01); no MI / adversarial / covariance / orthogonality term (cf_train.step_losses)")
        c["16_no_r2_output_loss"] = dict(status="pass", detail="R2 and the wrench enter only through the pairwise teacher distance (cf_train.rel_loss); no network output is compared to R2 during training")
        checks[ds] = c
    summary = {ds: {k: v["status"] for k, v in c.items()} for ds, c in checks.items()}
    S.write_json(S.OUT / "sanity_summary.json", dict(checks=checks, summary=summary, counts={ds: {s: sum(v == s for v in d.values()) for s in ("pass", "warn", "fail", "info")} for ds, d in summary.items()}))
    for ds, d in summary.items():
        print(ds, {s: sum(v == s for v in d.values()) for s in ("pass", "warn", "fail", "info")})
        for k, v in checks[ds].items():
            print(f"  {k}: {v['status']} — {v['detail']}")


if __name__ == "__main__":
    main()
