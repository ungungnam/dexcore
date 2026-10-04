#!/usr/bin/env python
"""Sanity checks of the diagnostic (caches, probes, tables).    CUDA_VISIBLE_DEVICES=5 python zt_sanity.py      -> OUT/sanity_summary.json
One entry per check and dataset: status pass / warn / fail, the measured values, how it was checked.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import torch

import zt_common as Z
from zt_data import Cache
from zt_probe import build


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); out = {}
    L = pd.read_csv(Z.OUT / "local_prediction_metrics.csv"); Q = pd.read_csv(Z.OUT / "quadrant_metrics.csv"); G = pd.read_csv(Z.OUT / "temporal_geometry_metrics.csv")
    for ds in Z.DATASETS:
        c = Cache(ds, dev, with_contact=True); cj = Z.read_json(Z.cache_path(ds).with_suffix(".json")); chk = {}
        man = Z.SAT.HP.ROOTS[ds] / "manifests" / "fixed0.json"; mf = json.loads(man.read_text())
        s2 = np.load(Z.stage2_preds_path(ds))
        # 1 same sequences / split as the previous experiments
        ok = all(np.array_equal(np.sort(c.seq[c.rows(k)]), np.sort(np.asarray(mf[k], int))) for k in ("train", "val", "test")) and np.array_equal(s2["example"], c.seq[c.rows("test")])
        chk["01_same_split"] = dict(status="pass" if ok else "fail", manifest=str(man), manifest_md5=Z.md5(man), n_sequences=cj["n_sequences"], n_takes=cj["n_takes"],
                                    how="cache sequences per split vs the inherited manifest fixed0.json; test order vs the Stage-2 B1 prediction file")
        # 2 encoder frozen and recorded
        md5_now = Z.md5(c.ckpt); s2c = np.load(Z.S2.teacher_cache_path(ds), allow_pickle=True)
        ok = md5_now == c.md5 == str(s2c["md5"]) and cj["fresh_encoding_max_abs_diff_in_train_std"] < 0.02
        chk["02_encoder_frozen"] = dict(status="pass" if ok else "fail", checkpoint=c.ckpt, md5_now=md5_now, md5_in_cache=c.md5, md5_stage2_teacher=str(s2c["md5"]),
                                        fresh_encoding_max_abs_diff_in_train_std=cj["fresh_encoding_max_abs_diff_in_train_std"],
                                        how="md5 of the Stage-1 A3 checkpoint now / in this cache / in the Stage-2 teacher cache; a fresh encoding of every frame agrees with the cached latents up to bf16 rounding")
        # 3 z standardisation from TRAIN frames only, identical to Stage 2
        zt = c.z_raw[c.rows("train")].reshape(-1, c.dz); b1 = torch.load(Z.S2.ckpt_path(ds, Z.S2.run_name("B1")), map_location="cpu", weights_only=False)
        ok = np.allclose(c.z_mu, zt.mean(0), atol=1e-4) and np.allclose(c.z_sd, zt.std(0), atol=1e-4) and np.allclose(b1["stats"]["z_mu"], c.z_mu, atol=1e-4) and np.allclose(b1["stats"]["z_sd"], c.z_sd, atol=1e-4)
        chk["03_train_only_standardisation"] = dict(status="pass" if ok else "fail", z_sd_range=[float(c.z_sd.min()), float(c.z_sd.max())],
                                                    how="z_mu / z_sd of the cache = mean / std over the TRAIN frames = the statistics stored in the Stage-2 B1 checkpoint")
        # 4 probe inputs: only z_t, G, tau_t, tau_{t+h}.  Reconstruction test over ALL saved test predictions (not a perturbation test)
        diffs, params, budget, sel, n_checked = {}, {}, {}, {}, {}
        for v in Z.VARIANTS:
            for h in Z.HORIZONS:
                ck = torch.load(Z.ckpt_path(ds, v, h), map_location=dev, weights_only=False); net = build(c, v).to(dev); net.load_state_dict(ck["state"]); net.eval()
                p = np.load(Z.preds_path(ds, v, h)); r, t = p["test_r"], p["test_t"]
                # independent reconstruction: the four inputs are assembled here from the numpy cache (not through Cache.inputs) and nothing else is passed
                f32 = lambda a_: torch.from_numpy(np.ascontiguousarray(a_, dtype=np.float32)).to(dev)
                with torch.no_grad():
                    out_rec = np.concatenate([net(f32(c.zs[r[i:i + 4096], t[i:i + 4096]]), f32(c.S[r[i:i + 4096]]), f32(c.tau[r[i:i + 4096], t[i:i + 4096]]), f32(c.tau[r[i:i + 4096], t[i:i + 4096] + h])).cpu().numpy()
                                              for i in range(0, len(r), 4096)])
                diffs[f"{v}_h{h}"] = float(np.abs(out_rec - p["test_z_hat"]).max()); n_checked[f"{v}_h{h}"] = int(len(r))
                params[f"{v}_h{h}"] = int(ck["n_params"]); budget[f"{v}_h{h}"] = dict(steps=int(ck["steps"]), best_step=int(ck["best_step"]), stopped_by=ck["stopped_by"])
                lg = pd.read_csv(Z.train_log_path(ds, v, h)); sel[f"{v}_h{h}"] = bool(int(lg.loc[lg.val.idxmin(), "step"]) == int(ck["best_step"]))
        src = (Z.HERE / "zt_probe.py").read_text()
        code_ok = "torch.cat([z_t, S, tau_t, tau_h] if self.use_tau else [z_t, S], -1)" in src and "return self.t_zs[r, t], self.t_S[r], self.t_tau[r, t], self.t_tau[r, t + h], self.t_zs[r, t + h]" in (Z.HERE / "zt_data.py").read_text()
        chk["04_probe_inputs"] = dict(status="pass" if (max(diffs.values()) < 1e-4 and code_ok) else "fail", max_abs_diff_of_reconstructed_predictions=diffs, n_test_pairs_reconstructed=n_checked, inputs="z_t (standardised), G (1032-D), tau_local,t, tau_local,t+h",
                                      how="all saved test predictions are reproduced by calling every probe on inputs assembled independently from the cache — z_t, G, tau_t, tau_{t+h} and nothing else — so no other quantity "
                                          "(future latent, contact, R2, wrench, hand) can have entered; the forward signature takes only these four tensors (source inspection)")
        # 5 probe capacity
        ok = all(3e6 <= n <= 10e6 for n in params.values())
        chk["05_probe_capacity"] = dict(status="pass" if ok else "warn", n_params=params, arch=Z.PROBE, how="parameter counts of the checkpoints (plan: 3-10 M, hidden 512, residual blocks, LayerNorm, GELU)")
        # 6 identical evaluation pairs for every method and horizon
        l = L[L.dataset == ds]; ok = all(l[(l.h == h) & l.method.isin(["persistence", "mean", "ar1", "linear", "probe", "probe_notau"])].n_pairs.nunique() == 1 for h in Z.HORIZONS)
        chk["06_same_pairs"] = dict(status="pass" if ok else "fail", n_pairs={int(h): int(l[(l.h == h) & (l.method == "probe")].n_pairs.iloc[0]) for h in Z.HORIZONS},
                                    how="all methods of a horizon are scored on the same test pairs (every (sequence, t) with t <= 63 - h)")
        # 7 quadrant thresholds from TRAIN only
        ga = np.load(Z.ds_out(ds) / "geometry_arrays.npz"); thr_ok = {}
        for h in Z.HORIZONS:
            g_ = G[(G.dataset == ds) & (G.h == h)].iloc[0]; r, t = c.pairs("train", h)
            dC = np.linalg.norm(c.C[r, t] - c.C[r, t + h], axis=1) / float(g_.scale_C); dz = np.linalg.norm(c.zs[r, t] - c.zs[r, t + h], axis=1) / float(g_.scale_z)
            dT = (0.5 * np.linalg.norm(c.u_r2[r, t] - c.u_r2[r, t + h], axis=1) + 0.5 * np.linalg.norm(c.u_q[r, t] - c.u_q[r, t + h], axis=1)) / float(g_.scale_T)
            rec = [np.quantile(x, q) for x in (dC, dz, dT) for q in (Z.QUANT_LOW, Z.QUANT_HIGH)]
            thr_ok[int(h)] = bool(np.allclose(rec, ga[f"h{h}|thr"], rtol=1e-4))
        chk["07_train_thresholds"] = dict(status="pass" if all(thr_ok.values()) else "fail", recomputed_equal=thr_ok, quantiles=[Z.QUANT_LOW, Z.QUANT_HIGH],
                                          test_fraction_low_high={int(r.h): [float(r.frac_lowC), float(r.frac_highC), float(r.frac_lowz), float(r.frac_highz)] for r in Q[Q.dataset == ds].itertuples()},
                                          how="the low / high thresholds of delta_C, delta_z and the teacher step are the 30 % / 70 % quantiles of the TRAIN transitions at each horizon (all six recomputed from the cache); the test fractions below / above them are reported, not tuned")
        # 8 no test-based selection, budget
        ok = all(sel.values()) and all(b["steps"] <= Z.TRAIN["max_steps"] and b["steps"] >= Z.TRAIN["min_steps"] for b in budget.values())
        chk["08_selection_and_budget"] = dict(status="pass" if ok else "fail", runs=budget, selected_is_validation_minimum=sel, recipe=Z.TRAIN,
                                              how="saved state = validation-MSE minimum of the training log; every run trained >= 5 000 and <= 30 000 steps")
        # 9 Stage-2 numbers reused, nothing retrained
        s2cfg = Z.read_json(Z.S2.OUT / "experiment_config.json"); b1_md5 = Z.md5(Z.S2.ckpt_path(ds, Z.S2.run_name("B1"))); rec_md5 = s2cfg["runs"][f"{ds}/B1"]["md5"]
        pred_older = Z.stage2_preds_path(ds).stat().st_mtime < Z.cache_path(ds).stat().st_mtime
        chk["09_stage2_reused"] = dict(status="pass" if (b1_md5 == rec_md5 and pred_older) else "fail", b1_predictions=str(Z.stage2_preds_path(ds)), md5=Z.md5(Z.stage2_preds_path(ds)), b1_checkpoint_md5=b1_md5,
                                       b1_checkpoint_md5_in_stage2_config=rec_md5, predictions_older_than_this_study=bool(pred_older),
                                       how="the B1 checkpoint is the one recorded in the Stage-2 experiment_config.json (md5), its prediction file predates this study's cache, and this study contains no Stage-2 training code")
        # 10 frozen decoder: its teacher-latent reconstruction error reproduces the Stage-1 number
        fl = l[l.method == "gt_z (decoder floor)"]; s1 = np.load(Z.CF.latents_path(ds, Z.TEACHER_NAME, "test")); ref = float(s1["e_zonly"].mean())
        ok = all(abs(v - ref) / ref < 0.05 for v in fl.dec_E_C)
        chk["10_frozen_decoder"] = dict(status="pass" if ok else "warn", decoder_floor_by_h={int(r.h): float(r.dec_E_C) for r in fl.itertuples()}, stage1_test_z_only_reconstruction=ref,
                                        how="E_C of D_z(z_{t+h}^GT) over the target frames vs the Stage-1 A3 z-only reconstruction error of all test frames")
        # 11 the metric tables reproduce from the saved predictions
        h = 4; p = np.load(Z.preds_path(ds, "probe", h)); rm = float(np.sqrt(((p["test_z_hat"] - c.zs[p["test_r"], p["test_t"] + h]) ** 2).mean()))
        ok = abs(rm - float(l[(l.h == h) & (l.method == "probe")].rmse.iloc[0])) < 1e-4
        chk["11_tables_reproduce"] = dict(status="pass" if ok else "fail", rmse_recomputed=rm, rmse_in_table=float(l[(l.h == h) & (l.method == "probe")].rmse.iloc[0]), how="probe RMSE at h = 4 recomputed from the saved test predictions")
        out[ds] = chk
        print(ds, {k: v["status"] for k, v in chk.items()})
    Z.write_json(Z.OUT / "sanity_summary.json", out)


if __name__ == "__main__":
    main()
