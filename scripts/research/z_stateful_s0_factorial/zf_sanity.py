#!/usr/bin/env python
"""Sanity checks of the 2 x 2 study, evaluated on the caches, checkpoints, logs, predictions and metrics.
    CUDA_VISIBLE_DEVICES=4 python zf_sanity.py --dataset taco      (one dataset per process; a few GPU forward passes)
    python zf_sanity.py --merge                                     (-> OUT/sanity_summary.json)
Writes OUT/sanity_<ds>.json: one entry per check with status pass / fail / warn, the measured values and how it was checked.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

import zf_common as Z
import zf_infer as I
import zf_models as MD
from s2_data import assert_causal_conditioning

S2, ZJ = Z.S2, Z.ZJ
RECIPE = ("lr", "wd", "warmup", "batch", "grad_clip", "ema", "max_steps", "final_lr_frac", "eval_every", "min_steps", "patience", "n_dec_frames", "lambda_z")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=Z.DATASETS); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        Z.write_json(Z.OUT / "sanity_summary.json", {ds: json.loads((Z.OUT / f"sanity_{ds}.json").read_text()) for ds in Z.DATASETS if (Z.OUT / f"sanity_{ds}.json").exists()}); return
    ds = a.dataset; dev = torch.device("cuda"); c = {}
    torch.cuda.set_per_process_memory_fraction(min(1.0, 10000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    models = [m for m in Z.Z_MODELS if Z.model_ckpt(ds, m).exists()]; here = [m for m in models if m in Z.TRAINED_HERE]
    cks = {m: torch.load(Z.model_ckpt(ds, m), map_location="cpu", weights_only=False) for m in models}
    d0 = torch.load(S2.ckpt_path(ds, S2.run_name("B0")), map_location="cpu", weights_only=False)
    data = I.load_data(ds, dev, "M00"); assert_causal_conditioning(data)
    tck_path = S2.teacher_ckpt_path(ds); stage1 = torch.load(tck_path, map_location="cpu", weights_only=False)["ema_state"]
    preds = {m: np.load(Z.model_preds(ds, m)) for m in models + ["D0"] if Z.model_preds(ds, m).exists()}
    mets = {m: np.load(Z.model_metrics(ds, m), allow_pickle=True) for m in models + ["D0"] if Z.model_metrics(ds, m).exists()}
    logs = {m: pd.read_csv(Z.model_train_log(ds, m)) for m in models}
    nets = {m: MD.load(ds, m, data, dev)[0] for m in models}
    # 1 teacher frozen; no Stage-1 encoder in any model; the initial-state module is not a copy of it
    tmd5 = Z.md5(tck_path); cache_json = S2.read_json(S2.teacher_cache_path(ds).with_suffix(".json"))
    md5_ok = cache_json["md5"] == tmd5 and all(ck["teacher"]["md5"] == tmd5 and ck["teacher"]["ckpt_md5_at_training"] == tmd5 for ck in cks.values())
    no_enc = all(not any(k.startswith(("E_z", "E_r")) for k in ck["state"]) for ck in cks.values())
    cos = {}
    for m in models:
        if Z.MODELS[m]["stateful"]:
            a_, b_ = [], []
            for k, v in cks[m]["state"].items():
                if k.startswith("init.blocks.") and ("E_z." + k[5:]) in stage1 and stage1["E_z." + k[5:]].shape == v.shape:
                    a_.append(v.float().flatten()); b_.append(stage1["E_z." + k[5:]].float().flatten())
            if a_:
                x, y = torch.cat(a_), torch.cat(b_); cos[m] = float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-12))
    c["01_teacher_frozen_no_stage1_encoder"] = dict(status="pass" if (md5_ok and no_enc and all(abs(v) < 0.2 for v in cos.values())) else "fail", md5_now=tmd5, md5_at_cache=cache_json["md5"], no_E_z_tensors_in_models=bool(no_enc),
                                                 cosine_of_init_module_weights_with_stage1_encoder=cos,
                                                 how="md5 of the Stage-1 A3 file now = at caching = at training; no E_z / E_r tensors in any state dict; the initial-state module's block weights are uncorrelated with the Stage-1 encoder's (trained from scratch)")
    # 2 teacher z and future GT never enter generation
    n = torch.from_numpy(data.idx["test"][:6]).to(dev); diffs = {}
    for m in models:
        P0 = I.generate(nets[m], data, n)["C_hat"].cpu().numpy()
        C_keep, z_keep, Cn_keep = data.C[n].clone(), data.z_star[n].clone(), data.fd.C[n].clone()
        g = torch.Generator(device=dev); g.manual_seed(0)
        data.C[n, 1:] = torch.rand(C_keep[:, 1:].shape, device=dev, generator=g); data.z_star[n] = torch.randn(z_keep.shape, device=dev, generator=g) * 5
        data.fd.C[n, 1:] = torch.randn(Cn_keep[:, 1:].shape, device=dev, generator=g).to(Cn_keep.dtype)        # the standardised copy of the future maps as well
        P1 = I.generate(nets[m], data, n)["C_hat"].cpu().numpy()
        data.C[n] = C_keep; data.z_star[n] = z_keep; data.fd.C[n] = Cn_keep
        diffs[m] = float(np.abs(P0[:, 1:] - P1[:, 1:]).max())
    ok2 = all(d == 0.0 for d in diffs.values())
    c["02_generation_uses_only_s0_G_tau"] = dict(status="pass" if ok2 else "fail", max_abs_change_after_replacing_teacher_z_and_future_gt_by_noise=diffs, tau_offsets=(data.fd.offsets - Z.PAD).tolist(),
                                              how="six test sequences regenerated after their teacher latents (all frames, frame 0 included) and their GT frames 1..63 (raw and standardised copies) were replaced by noise: the trajectories, the stateful rollouts "
                                                  "included, must not change by a single bit (so the rollout runs on the model's own latents and z_hat_0 comes from s_0 and G)")
    # 3 the stateful models were trained by free rollout (no teacher forcing)
    src = (Z.HERE / "zf_models.py").read_text() + (Z.HERE / "zf_train.py").read_text()
    code = "z = z + dz.float(); out.append(z)" in src and "z_hat, z0 = MD.predict_latents(net, data, n)" in src and "z_star" not in src.split("def predict_latents")[1].split("def decode_raw")[0]
    insp = {m: [dict(step=r["step"], grad_init=r.get("gnorm_init"), grad_transition=r.get("gnorm_transition")) for r in cks[m].get("inspect", [])] for m in here if Z.MODELS[m]["stateful"]}
    c["03_rollout_training_without_teacher_forcing"] = dict(status="pass" if (code and ok2 and all(all((r["grad_init"] or 0) > 0 and (r["grad_transition"] or 0) > 0 for r in v) for v in insp.values())) else "fail",
                                                         gradient_inspection=insp, bptt=Z.TRAIN["bptt"],
                                                         how="predict_latents takes only (s_0, G, tau) and feeds each step's own output to the next (source + check 2); the loss is back-propagated through all 63 steps "
                                                             "(non-zero gradients on the initial-state module and on the transition at the inspection steps)")
    # 4 the s_0-preserving decoder: frame 0 is s_0, and an untrained residual decoder returns s_0 for every frame
    d4 = {}
    for m in ("M01", "M11"):
        fresh = MD.build(m, data, stage1_state=stage1).to(dev).eval(); nn = n[:2]
        with torch.no_grad():
            z, _ = I.predict_z(fresh, data, nn); Cf = I.decode_seq(fresh, data, nn, z)
        d4[m] = float((Cf - data.C[nn, :1]).abs().max()); del fresh
    s0_ok = all(bool(z["frame0_equals_s0"]) and np.allclose(z["s0"], data.C[torch.from_numpy(z["example"]).to(dev), 0].cpu().numpy(), atol=1e-6) for z in preds.values())
    c["04_s0_preserving_decoder"] = dict(status="pass" if (all(v == 0.0 for v in d4.values()) and s0_ok) else "fail", max_abs_deviation_from_s0_at_initialisation=d4, frame0_is_gt_s0_in_all_prediction_files=bool(s0_ok),
                                       how="a freshly initialised residual model decodes every frame to exactly s_0 (zero-initialised output layer); the stored s_0 of every prediction file is the GT frame 0")
    # 5 z standardisation uses train statistics only
    zc = np.load(S2.teacher_cache_path(ds), allow_pickle=True); zt = zc["z"][data.idx["train"]].reshape(-1, zc["z"].shape[-1])
    d5 = {m: float(max(np.abs(cks[m]["stats"]["z_mu"] - zt.mean(0)).max(), np.abs(cks[m]["stats"]["z_sd"] - zt.std(0)).max())) for m in models}
    c["05_train_only_z_standardisation"] = dict(status="pass" if all(v < 1e-4 for v in d5.values()) else "fail", max_abs_diff_vs_train_statistics=d5, how="z_mu / z_sd in every checkpoint vs the mean / std of the cached teacher over the TRAIN frames")
    # 6 identical split and conditioning
    gt = np.load(S2.metrics_path(ds, "GT"), allow_pickle=True)
    same_ex = all(np.array_equal(z["example"], data.idx["test"]) for z in list(mets.values()) + list(preds.values()) + [gt])
    st = [ck["stats"] for ck in cks.values()] + [d0["stats"]]
    same_stats = all(np.allclose(s["sat"]["hier"]["mu"], st[0]["sat"]["hier"]["mu"]) and s["sat"]["hier"]["s"] == st[0]["sat"]["hier"]["s"] and abs(s["sat"]["sigma_r"] - st[0]["sat"]["sigma_r"]) < 1e-6 * st[0]["sat"]["sigma_r"]
                     and np.allclose(s["sat"]["hier"]["muS"], st[0]["sat"]["hier"]["muS"]) for s in st)
    dims = {m: (ck["d_traj"], ck["d_static"]) for m, ck in cks.items()}; dims["D0"] = (d0["d_traj"], d0["d_static"])
    c["06_identical_split_and_conditioning"] = dict(status="pass" if (same_ex and same_stats and len(set(dims.values())) == 1) else "fail", n_train=len(data.idx["train"]), n_val=len(data.idx["val"]), n_test=len(data.idx["test"]),
                                                 n_test_takes=int(len(np.unique(data.take[data.idx["test"]]))), test_examples_identical=bool(same_ex), normalisation_identical_incl_D0=bool(same_stats), d_traj_d_static=dims,
                                                 how="example arrays of every prediction / metrics file equal the fixed test index; contact / G normalisation and sigma_r identical in every checkpoint and in D0's")
    # 7 identical metric code
    rc = {n_: Z.read_json(ZJ.ds_out(ds) / "metrics" / f"recheck_{n_}.json") for n_ in (S2.run_name("B0"), S2.run_name("B1")) if (ZJ.ds_out(ds) / "metrics" / f"recheck_{n_}.json").exists()}
    import zf_evaluate as ZE, zj_evaluate as ZJE
    worst = max((max(v["max_abs_diff_per_example"].values()) for v in rc.values()), default=np.nan)
    c["07_identical_metric_code"] = dict(status="pass" if (ZE.run is ZJE.run and len(rc) == 2 and worst < 1e-4) else "fail", same_function_object=bool(ZE.run is ZJE.run), previous_recheck_worst_abs_diff=float(worst),
                                        how="zf_evaluate.py scores with the function of zj_evaluate.py, which re-scored the Stage-2 prediction files of B0 and B1 and reproduced their stored per-example metrics")
    # 8 capacity
    tot = {m: ck["n_params"]["total"] for m, ck in cks.items()}; dec = {m: ck["n_params"].get("D_z") for m, ck in cks.items()}
    spread = max(tot.values()) / min(tot.values()) - 1; dspread = max(dec.values()) / min(dec.values()) - 1
    c["08_comparable_capacity"] = dict(status="pass" if (spread <= 0.15 and dspread <= 0.01) else "fail", total_params=tot, decoder_params=dec, D0_total=d0["n_params"]["total"], total_spread=float(spread), decoder_spread=float(dspread),
                                      how="parameter counts in the checkpoints: totals of the four z models within 15 %, the two decoders within 1 %")
    # 9 budget and recipe
    bud = {m: dict(steps=int(ck["steps"]), best_step=int(ck["best_step"]), stopped_by=ck["stopped_by"], hours=round(ck["seconds"] / 3600, 2)) for m, ck in cks.items()}
    rec = {m: {k: ck["cfg"].get(k) for k in RECIPE} for m, ck in cks.items()}
    same_rec = len({json.dumps(v, sort_keys=True) for v in rec.values()}) == 1
    ok9 = same_rec and all(bud[m]["stopped_by"] in ("patience", "max_steps") and bud[m]["steps"] >= Z.TRAIN["min_steps"] for m in models)
    c["09_budget_and_recipe"] = dict(status="pass" if ok9 else "fail", budgets=bud, identical_recipe_incl_reused_M00=bool(same_rec), recipe=rec.get("M00"),
                                    how="every z model, the reused M00 included, has the same optimiser / schedule / batch / stopping settings in its checkpoint and ended by the patience rule after >= 25 000 steps or at 100 000")
    # 10 no test-based selection
    sel = {m: dict(selected_step=int(cks[m]["best_step"]), argmin_val_L_C_step=int(logs[m].loc[logs[m]["val_L_C"].idxmin(), "step"]), test_columns=[cc for cc in logs[m].columns if "test" in cc]) for m in models}
    c["10_no_test_selection"] = dict(status="pass" if all(v["selected_step"] == v["argmin_val_L_C_step"] and not v["test_columns"] for v in sel.values()) else "fail", selection=sel,
                                    how="the saved state is the EMA state at the minimum VALIDATION L_C; the training logs hold no test quantity; the best z model is chosen by validation E_C")
    # 11 numerically valid
    rng = {m: dict(raw_min=float(z["raw_min"]), raw_max=float(z["raw_max"]), n_nonfinite=int(z["n_nonfinite"])) for m, z in preds.items()}
    c["11_numerically_valid"] = dict(status="pass" if all(v["n_nonfinite"] == 0 and v["raw_min"] > -0.5 and v["raw_max"] < 1.5 for v in rng.values()) else "fail", ranges=rng, how="raw range and non-finite count of every prediction file")
    # 12 diagnostics consistent with the generated trajectories
    sp = Z.ds_out(ds) / "diag" / "summary.json"
    if sp.exists():
        sm = Z.read_json(sp); ec = {m: sm[f"{m}_pred_path_vs_stored_E_C"] for m in models if f"{m}_pred_path_vs_stored_E_C" in sm}; zd = {m: sm[f"{m}_zhat_vs_preds_file_max_abs"] for m in models if f"{m}_zhat_vs_preds_file_max_abs" in sm}
        c["12_diagnostic_consistency"] = dict(status="pass" if (all(abs(v[0] - v[1]) < 5e-3 * max(v[1], 1e-9) for v in ec.values()) and all(v < 5e-2 for v in zd.values())) else "warn", E_C_recomputed_vs_stored=ec, z_hat_max_abs_diff=zd,
                                             how="the diagnostic recomputes every model's latents and decoded trajectory; both agree with the stored prediction files")
    Z.write_json(Z.OUT / f"sanity_{ds}.json", c)
    print(json.dumps({k: v["status"] for k, v in c.items()}, indent=1))


if __name__ == "__main__":
    main()
