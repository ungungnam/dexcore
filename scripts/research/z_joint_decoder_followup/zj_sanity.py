#!/usr/bin/env python
"""The 12 sanity checks of the plan (Section 19) + three extra ones, evaluated on the caches, checkpoints, logs, predictions and metrics.
    CUDA_VISIBLE_DEVICES=5 python zj_sanity.py --dataset taco      (one dataset per process; a few GPU forward passes)
    python zj_sanity.py --merge                                     (-> OUT/sanity_summary.json)
Writes OUT/sanity_<ds>.json: one entry per check with status pass / fail / warn / info, the measured values and how it was checked.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

import zj_common as Z
import zj_infer as I
import zj_models as MD
from s2_data import assert_causal_conditioning

S2 = Z.S2


def dec_change(state, stage1, prefix="D_z."):
    """Relative L2 distance of the loaded Stage-1 decoder weights from their Stage-1 values; norm of the extra blocks' FiLM layers."""
    num = den = 0.0; extra = 0.0; n_extra = 0
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        if k in stage1:
            num += float((v.float() - stage1[k].float()).pow(2).sum()); den += float(stage1[k].float().pow(2).sum())
        elif ".ada.1." in k:
            extra += float(v.float().pow(2).sum()); n_extra += 1
    return dict(rel_change=(num / max(den, 1e-12)) ** 0.5, extra_blocks_film_norm=extra ** 0.5, n_extra_film_tensors=n_extra)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=Z.DATASETS); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        Z.write_json(Z.OUT / "sanity_summary.json", {ds: json.loads((Z.OUT / f"sanity_{ds}.json").read_text()) for ds in Z.DATASETS if (Z.OUT / f"sanity_{ds}.json").exists()}); return
    ds = a.dataset; dev = torch.device("cuda"); c = {}
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8000 * 2 ** 20 / torch.cuda.get_device_properties(0).total_memory))
    models = [m for m in ("M0", "M1", "M2", "M3", "M0r") if Z.model_ckpt(ds, m).exists()]
    here = [m for m in models if m in Z.TRAINED_HERE]; zmods = [m for m in models if Z.MODELS[m]["joint"]]
    cks = {m: torch.load(Z.model_ckpt(ds, m), map_location="cpu", weights_only=False) for m in models}
    data = I.load_data(ds, dev, "M1"); assert_causal_conditioning(data)
    tck_path = S2.teacher_ckpt_path(ds); stage1 = torch.load(tck_path, map_location="cpu", weights_only=False)["ema_state"]
    preds = {m: np.load(Z.model_preds(ds, m)) for m in models if Z.model_preds(ds, m).exists()}
    mets = {m: np.load(Z.model_metrics(ds, m), allow_pickle=True) for m in models if Z.model_metrics(ds, m).exists()}
    logs = {m: pd.read_csv(Z.model_train_log(ds, m)) for m in models}
    # 1 teacher frozen
    tmd5 = Z.md5(tck_path); cache_json = S2.read_json(S2.teacher_cache_path(ds).with_suffix(".json"))
    md5_ok = cache_json["md5"] == tmd5 and all(ck["teacher"]["md5"] == tmd5 and ck["teacher"]["ckpt_md5_at_training"] == tmd5 for ck in cks.values())
    no_enc = all(not any(k.startswith(("E_z", "E_r")) for k in ck["state"]) for ck in cks.values())
    c["01_teacher_frozen"] = dict(status="pass" if (md5_ok and no_enc) else "fail", teacher_ckpt=str(tck_path), md5_now=tmd5, md5_at_cache=cache_json["md5"], md5_at_training={m: ck["teacher"]["ckpt_md5_at_training"] for m, ck in cks.items()},
                                 no_encoder_weights_in_models=bool(no_enc), how="md5 of the Stage-1 A3 file now = at caching = at every training; the model state dicts hold no E_z / E_r tensors (the teacher exists only as the cached targets z*)")
    # 2 + 5 teacher z and future GT contact never enter inference: replace both by noise and regenerate
    n = torch.from_numpy(data.idx["test"][:6]).to(dev); diffs = {}; nets = {}
    for m in models:
        nets[m] = MD.load(ds, m, data, dev)[0]
        P0 = I.generate(nets[m], data, n)["C_hat"].cpu().numpy()
        C_keep, z_keep = data.C[n].clone(), data.z_star[n].clone()
        g = torch.Generator(device=dev); g.manual_seed(0)
        data.C[n, 1:] = torch.rand(C_keep[:, 1:].shape, device=dev, generator=g); data.z_star[n] = torch.randn(z_keep.shape, device=dev, generator=g) * 5
        P1 = I.generate(nets[m], data, n)["C_hat"].cpu().numpy()
        data.C[n] = C_keep; data.z_star[n] = z_keep
        diffs[m] = float(np.abs(P0[:, 1:] - P1[:, 1:]).max())
    ok25 = all(d == 0.0 for d in diffs.values())
    c["02_teacher_z_never_in_inference"] = dict(status="pass" if ok25 else "fail", max_abs_change_after_replacing_teacher_z_and_future_gt_by_noise=diffs,
                                               how="six test sequences regenerated after their teacher z* (all frames) and their GT frames 1..63 were replaced by noise: the generated trajectories must not change by a single bit")
    # 3 the M2 / M3 decoder is actually updated
    dc = {m: dec_change(cks[m]["state"], stage1) for m in zmods}
    logged = {m: float(logs[m].loc[logs[m].step == cks[m]["best_step"], "dec_rel_change"].iloc[0]) for m in zmods if "dec_rel_change" in logs[m]}
    upd = all(dc[m]["rel_change"] > 1e-3 for m in zmods if m in here) and all(dc[m]["extra_blocks_film_norm"] > 0 for m in zmods if m in here)
    c["03_decoder_updated"] = dict(status="pass" if upd else "fail", change_vs_stage1_at_selected_state=dc, logged_at_selected_step=logged,
                                  how="relative L2 distance of the loaded decoder weights from the Stage-1 weights at the selected state (M1 = Stage-2 B1 shown for comparison); the extra blocks' zero-initialised FiLM layers must have moved (norm > 0)")
    # 4 the predicted z is the primary decoder input in training
    src = (Z.HERE / "zj_train.py").read_text()
    code_ok = "rho_pred = to_rho(MD.decode(net, g(z_hat), geo, Sg, ch))" in src and 'loss = cfg["beta_pred"] * L_C + cfg["lambda_z"] * L_z' in src and 'loss = loss + cfg["beta_gt"] * L_gt' in src
    insp = {m: cks[m].get("inspect", []) for m in here if Z.MODELS[m]["joint"]}
    reach = {m: [dict(step=r["step"], dense_term_grad_on_temporal_network=r.get("gnorm_L_C_trunk"), dense_term_grad_on_decoder=r.get("gnorm_L_C_dec"), teacher_term_grad_on_decoder=r.get("gnorm_L_C_gt_dec")) for r in rows] for m, rows in insp.items()}
    grad_ok = all(all((r.get("gnorm_L_C_trunk") or 0) > 0 and (r.get("gnorm_L_C_dec") or 0) > 0 for r in rows) for rows in insp.values())
    dom = all(all((r.get("gnorm_L_C_dec") or 0) > (r.get("gnorm_L_C_gt_dec") or 0) for r in rows) for m, rows in insp.items() if Z.MODELS[m]["dual"])
    w = {m: dict(beta_pred=cks[m]["cfg"]["beta_pred"], beta_gt=cks[m]["cfg"]["beta_gt"] if Z.MODELS[m]["dual"] else 0.0, teacher_frames_per_sequence=cks[m]["cfg"]["n_gt_frames"] if Z.MODELS[m]["dual"] else 0,
                 predicted_frames_per_sequence=cks[m]["cfg"]["n_dec_frames"]) for m in insp}
    c["04_predicted_z_is_primary_decoder_input"] = dict(status="pass" if (code_ok and grad_ok and dom) else "fail", weights=w, gradient_inspection=reach, code_match=bool(code_ok),
                                                       how="zj_train.step_losses decodes g(z_hat) with weight beta_pred = 1 (M2: nothing else; M3: + 0.25 x the teacher path on 1 of 4 frames); at the inspection steps the dense term has a non-zero "
                                                           "gradient on the temporal network (it flows through z_hat) and, for M3, a larger gradient on the decoder than the teacher term")
    # 5 no future GT contact enters F_theta
    offs = (data.fd.offsets - Z.PAD).tolist()
    c["05_no_future_gt_in_temporal_network"] = dict(status="pass" if (ok25 and offs == list(range(-8, 9, 2))) else "fail", tau_offsets=offs, max_abs_change=diffs,
                                                   how="the regeneration test of check 2 (future GT frames replaced by noise); the only contact input is s_0 (frame 0); tau_local reads object states t-8..t+8")
    # 6 z standardisation uses train statistics only
    zc = np.load(S2.teacher_cache_path(ds), allow_pickle=True); tr = data.idx["train"]
    zt = zc["z"][tr].reshape(-1, zc["z"].shape[-1]); za = zc["z"][zc["has"]].reshape(-1, zc["z"].shape[-1])
    d6 = {m: float(max(np.abs(cks[m]["stats"]["z_mu"] - zt.mean(0)).max(), np.abs(cks[m]["stats"]["z_sd"] - zt.std(0)).max())) for m in zmods}
    c["06_train_only_z_standardisation"] = dict(status="pass" if all(v < 1e-4 for v in d6.values()) else "fail", max_abs_diff_vs_train_statistics=d6, all_split_vs_train_mean_shift=float(np.abs(za.mean(0) - zt.mean(0)).max()),
                                               how="z_mu / z_sd stored in every z-model checkpoint vs the mean / std of the cached teacher over the TRAIN frames only")
    # 7 split identical to the prior experiments
    man = S2.SAT.HP.ROOTS[ds] / "manifests" / "fixed0.json"
    gt = np.load(S2.metrics_path(ds, "GT"), allow_pickle=True)
    same_ex = all(np.array_equal(z["example"], data.idx["test"]) for z in list(mets.values()) + [gt]) and all(np.array_equal(z["example"], data.idx["test"]) for z in preds.values())
    c["07_identical_split"] = dict(status="pass" if same_ex else "fail", manifest=str(man), manifest_md5=Z.md5(man), n_train=len(data.idx["train"]), n_val=len(data.idx["val"]), n_test=len(data.idx["test"]),
                                  n_test_takes=int(len(np.unique(data.take[data.idx["test"]]))), test_examples_identical_in_all_files=bool(same_ex),
                                  how="the fixed manifest of the previous studies; the example arrays of every prediction / metrics file (M0 and M1 are the Stage-2 files themselves) equal the test index")
    # 8 conditioning identical to M0
    st = [ck["stats"] for ck in cks.values()]
    same_stats = all(np.allclose(s["sat"]["hier"]["mu"], st[0]["sat"]["hier"]["mu"]) and s["sat"]["hier"]["s"] == st[0]["sat"]["hier"]["s"] and abs(s["sat"]["sigma_r"] - st[0]["sat"]["sigma_r"]) < 1e-6 * st[0]["sat"]["sigma_r"]
                     and np.allclose(s["sat"]["hier"]["muS"], st[0]["sat"]["hier"]["muS"]) for s in st)
    s0_ok = all(bool(z["frame0_equals_s0"]) and np.allclose(z["s0"], data.C[torch.from_numpy(z["example"]).to(dev), 0].cpu().numpy(), atol=1e-6) for z in preds.values())
    dims = {m: (ck["d_traj"], ck["d_static"]) for m, ck in cks.items()}; bb = {m: ck["n_params"]["backbone"] for m, ck in cks.items()}
    c["08_conditioning_matches_M0"] = dict(status="pass" if (same_stats and s0_ok and len(set(dims.values())) == 1 and len(set(bb.values())) == 1) else "fail", s0_is_gt_frame0=bool(s0_ok), d_traj_d_static=dims,
                                          backbone_params=bb, normalisation_statistics_identical=bool(same_stats),
                                          how="stored s_0 of every prediction file = GT frame 0; identical input dimensions, backbone parameter count and normalisation statistics (contact mu / s, G mean, sigma_r) in every checkpoint")
    # 9 identical metric extraction code
    rc = {n_: Z.read_json(Z.ds_out(ds) / "metrics" / f"recheck_{n_}.json") for n_ in (S2.run_name("B0"), S2.run_name("B1")) if (Z.ds_out(ds) / "metrics" / f"recheck_{n_}.json").exists()}
    worst = max((max(v["max_abs_diff_per_example"].values()) for v in rc.values()), default=np.nan)
    c["09_identical_metric_code"] = dict(status="pass" if (len(rc) == 2 and worst < 1e-4) else "fail", recheck=rc, worst_abs_diff=float(worst),
                                        how="zj_evaluate.py (the script that scores M2 / M3) re-scored the Stage-2 prediction files of B0 and B1 and reproduced their stored per-example metrics")
    # 10 capacity not substantially smaller than B0
    tot = {m: ck["n_params"]["total"] for m, ck in cks.items()}
    c["10_capacity_vs_B0"] = dict(status="pass" if all(tot[m] >= 0.95 * tot["M0"] for m in zmods) else "fail", total_params=tot, decoder_params={m: cks[m]["n_params"].get("D_z") for m in zmods},
                                 ratio_to_M0={m: tot[m] / tot["M0"] for m in models}, how="parameter counts stored in the checkpoints")
    # 11 no early manual stop
    bud = {m: dict(steps=int(ck["steps"]), best_step=int(ck["best_step"]), stopped_by=ck["stopped_by"], min_steps=int(ck["cfg"]["min_steps"]), patience=int(ck["cfg"]["patience"]), hours=round(ck["seconds"] / 3600, 2)) for m, ck in cks.items()}
    ok11 = all(bud[m]["stopped_by"] in ("patience", "max_steps") and bud[m]["steps"] >= Z.TRAIN["min_steps"] for m in here)
    c["11_no_early_manual_stop"] = dict(status="pass" if ok11 else "fail", budgets=bud, reused_runs_note="M1 = Stage-2 B1: TACO stopped by its own rule at 20 000 steps; ARCTIC stopped by hand at 18 000 (disclosed in the Stage-2 report)",
                                       how="every run trained in this study ended by the patience rule after >= 25 000 steps or at the 100 000-step limit (stopped_by / steps in the checkpoint)")
    # 12 no test-based selection
    sel = {}
    for m in here:
        lg = logs[m]; sel[m] = dict(selected_step=int(cks[m]["best_step"]), argmin_val_L_C_step=int(lg.loc[lg["val_L_C"].idxmin(), "step"]), test_columns_in_log=[cc for cc in lg.columns if "test" in cc])
    ok12 = all(v["selected_step"] == v["argmin_val_L_C_step"] and not v["test_columns_in_log"] for v in sel.values())
    c["12_no_test_selection"] = dict(status="pass" if ok12 else "fail", selection=sel, how="the saved state is the EMA state at the minimum VALIDATION L_C of the training log; the logs hold no test quantity; test files were written after training ended")
    # 13 (extra) D_phi at initialisation is the Stage-1 decoder
    if "M2" in cks:
        fresh = MD.JointModel(data.d_traj(), data.d_static(), data.d_geo(), data.z_mu, data.z_sd, stage1_state=stage1).to(dev).eval()
        m1 = MD.build("M1", data, stage1_state=stage1).to(dev).eval()
        nn = n[:2]; zz = data.teacher_std(nn)
        d13 = float((I.decode_seq(fresh, data, nn, zz) - I.decode_seq(m1, data, nn, zz)).abs().max())
        c["13_extra_blocks_start_as_identity"] = dict(status="pass" if d13 < 1e-4 else "fail", max_abs_diff_raw_contact=d13, how="the 6-block decoder with freshly initialised extra blocks vs the 4-block Stage-1 decoder on the same teacher latents")
        del fresh, m1
    # 14 (extra) numerically valid predictions
    rng = {m: dict(raw_min=float(z["raw_min"]), raw_max=float(z["raw_max"]), frac_below_0=float(z["frac_below_0"]), frac_above_1=float(z["frac_above_1"]), n_nonfinite=int(z["n_nonfinite"])) for m, z in preds.items()}
    c["14_numerically_valid"] = dict(status="pass" if all(v["n_nonfinite"] == 0 and v["raw_min"] > -0.5 and v["raw_max"] < 1.5 for v in rng.values()) else "fail", ranges=rng, how="raw range and non-finite count of every prediction file (unclamped, as in the previous studies)")
    # 15 (extra) the diagnostics used the latents of the generated trajectories
    sp = Z.ds_out(ds) / "diag" / "summary.json"
    if sp.exists():
        sm = Z.read_json(sp)
        zdiff = {m: sm[f"{m}_zhat_vs_preds_file_max_abs"] for m in zmods if f"{m}_zhat_vs_preds_file_max_abs" in sm}; ec = {m: sm[f"{m}_pred_path_vs_stored_E_C"] for m in zmods if f"{m}_pred_path_vs_stored_E_C" in sm}
        ok15 = all(v < 5e-2 for v in zdiff.values()) and all(abs(v[0] - v[1]) < 5e-3 * max(v[1], 1e-9) for v in ec.values())
        c["15_diagnostic_consistency"] = dict(status="pass" if ok15 else "warn", z_hat_recomputed_vs_stored_max_abs=zdiff, E_C_recomputed_vs_stored=ec,
                                             how="the diagnostic recomputes z_hat and the decoded trajectory of every z model; both agree with the stored prediction files (bf16 autocast, float16 storage)")
    Z.write_json(Z.OUT / f"sanity_{ds}.json", c)
    print(json.dumps({k: v["status"] for k, v in c.items()}, indent=1))


if __name__ == "__main__":
    main()
