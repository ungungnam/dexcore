#!/usr/bin/env python
"""The 15 sanity checks of the plan (Section 21), evaluated on the caches, checkpoints, predictions and metrics.
    CUDA_VISIBLE_DEVICES=5 python s2_sanity.py --dataset taco      (one dataset per process; a few GPU forward passes)
    python s2_sanity.py --merge                                     (-> OUT/sanity_summary.json)
Writes OUT/sanity_<ds>.json (one entry per check: status pass / fail / warn / info, the measured values, how it was checked).
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd
import torch

import s2_common as S
import s2_models as MD
from s2_data import S2Data, assert_causal_conditioning
from s2_generate import predict

HERE = S.HERE


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=S.DATASETS); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        S.write_json(S.OUT / "sanity_summary.json", {ds: json.loads((S.OUT / f"sanity_{ds}.json").read_text()) for ds in S.DATASETS if (S.OUT / f"sanity_{ds}.json").exists()})
        return
    ds = a.dataset; dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); c = {}
    names = {m: S.run_name(m) for m in S.MODELS}
    cks = {m: torch.load(S.ckpt_path(ds, n), map_location="cpu", weights_only=False) for m, n in names.items() if S.ckpt_path(ds, n).exists()}
    data = S2Data(ds, dev, need_masks=False, teacher=True); assert_causal_conditioning(data)
    preds = {m: np.load(S.preds_path(ds, n)) for m, n in names.items() if S.preds_path(ds, n).exists()}
    mets = {m: np.load(S.metrics_path(ds, n), allow_pickle=True) for m, n in names.items() if S.metrics_path(ds, n).exists()}
    gt = np.load(S.metrics_path(ds, "GT"), allow_pickle=True) if S.metrics_path(ds, "GT").exists() else None
    # 1 identical split: the same manifest as the previous study, the same test examples in every metrics file, identical normalisation statistics in every checkpoint
    man = S.SAT.HP.ROOTS[ds] / "manifests" / "fixed0.json"
    prev = np.load(S.sat_metrics_path(ds, S.SAT.run_name("D0", 0)), allow_pickle=True)
    same_ex = all(np.array_equal(z["example"], data.idx["test"]) for z in list(mets.values()) + ([gt] if gt is not None else [])) and np.array_equal(prev["example"], data.idx["test"])
    st = [ck["stats"] for ck in cks.values()]
    same_stats = all(np.allclose(s["sat"]["hier"]["mu"], st[0]["sat"]["hier"]["mu"]) and s["sat"]["hier"]["s"] == st[0]["sat"]["hier"]["s"] and abs(s["sat"]["sigma_r"] - st[0]["sat"]["sigma_r"]) < 1e-6 * st[0]["sat"]["sigma_r"]
                     and np.allclose(s["sat"]["hier"]["muS"], st[0]["sat"]["hier"]["muS"]) for s in st)
    c["01_identical_split"] = dict(status="pass" if (same_ex and same_stats) else "fail", manifest=str(man), manifest_md5=S.md5(man), n_train=len(data.idx["train"]), n_val=len(data.idx["val"]), n_test=len(data.idx["test"]),
                                  test_examples_identical_incl_previous_D0=bool(same_ex), checkpoint_statistics_identical=bool(same_stats), n_checkpoints=len(cks),
                                  sigma_r={m: float(ck["stats"]["sat"]["sigma_r"]) for m, ck in cks.items()},
                                  how="manifest of the hierarchical study (fixed0); example arrays of every metrics file and of the previous D0 compared to the test index; hier mu / s / muS and sigma_r of every checkpoint compared "
                                      "(sigma_r to a relative 1e-6: checkpoints written by s2_finalize.py recompute the train statistics on the CPU, which differs from the GPU value in the 8th digit)")
    # 2 identical s_0 / G / tau: the stored s_0 of every prediction file is the GT frame 0, frame 0 of every trajectory equals it, tau offsets are the causal window, G = fd.S from one statistics set
    s0_ok = all(bool(z["frame0_equals_s0"]) and np.allclose(z["s0"], data.C[torch.from_numpy(z["example"]).to(dev), 0].cpu().numpy(), atol=1e-6) for z in preds.values())
    offs = (data.fd.offsets - S.PAD).tolist()
    c["02_identical_conditioning"] = dict(status="pass" if (s0_ok and offs == list(range(-8, 9, 2))) else "fail", s0_is_gt_frame0=bool(s0_ok), tau_offsets=offs, d_traj=data.d_traj(), d_static=data.d_static(),
                                         how="s_0 arrays of the prediction files vs the GT frame 0; tau window offsets; G dimension")
    # 3 identical backbone capacity
    bb = {m: ck["n_params"]["backbone"] for m, ck in cks.items()}; tot = {m: ck["n_params"]["total"] for m, ck in cks.items()}
    spread = (max(tot.values()) - min(tot.values())) / min(tot.values()) if tot else np.nan
    c["03_backbone_capacity"] = dict(status="pass" if (len(set(bb.values())) == 1 and spread <= 0.15) else "fail", backbone_params=bb, total_params=tot, total_spread=float(spread),
                                    arch={k: v for k, v in S.BACKBONE.items()}, how="parameter counts recorded in the checkpoints; totals within 15 % (B0 head enlarged, Section 13)")
    # 4 teacher frozen: the Stage-1 checkpoint file is unchanged (md5 now = md5 at caching = md5 at training); no encoder weights inside the Stage-2 models
    tmd5 = S.md5(S.teacher_ckpt_path(ds)); cache_json = S.read_json(S.teacher_cache_path(ds).with_suffix(".json"))
    md5_ok = cache_json["md5"] == tmd5 and all(ck["teacher"]["md5"] == tmd5 and ck["teacher"]["ckpt_md5_at_training"] == tmd5 for ck in cks.values())
    no_enc = all(not any(k.startswith("E_z") or k.startswith("E_r") for k in ck["state"]) for ck in cks.values())
    c["04_teacher_frozen"] = dict(status="pass" if (md5_ok and no_enc) else "fail", teacher_ckpt=str(S.teacher_ckpt_path(ds)), md5_now=tmd5, md5_at_cache=cache_json["md5"], md5_at_training={m: ck["teacher"]["md5"] for m, ck in cks.items()},
                                 no_encoder_weights_in_stage2_models=bool(no_enc), how="md5 of the Stage-1 A3 file now vs at caching vs at training; Stage-2 state dicts contain no E_z / E_r keys")
    # 5 no future GT contact (and no teacher z) enters inference: perturb the future GT frames and the teacher of a few test sequences and regenerate
    n = torch.from_numpy(data.idx["test"][:6]).to(dev); diffs = {}
    for m, ck in cks.items():
        net = MD.build(m, data).to(dev); net.load_state_dict(ck["state"]); net.eval()
        P0 = predict(net, data, n)["C_hat"]
        C_keep, z_keep = data.C[n].clone(), data.z_star[n].clone()
        g = torch.Generator(device=dev); g.manual_seed(0)
        data.C[n, 1:] = torch.rand(C_keep[:, 1:].shape, device=dev, generator=g); data.z_star[n] = torch.randn(z_keep.shape, device=dev, generator=g) * 5
        P1 = predict(net, data, n)["C_hat"]
        data.C[n] = C_keep; data.z_star[n] = z_keep
        diffs[m] = float(np.abs(P0[:, 1:] - P1[:, 1:]).max())
        del net
    c["05_no_future_gt_at_inference"] = dict(status="pass" if all(d == 0.0 for d in diffs.values()) else "fail", max_abs_change_after_perturbing_future_gt_and_teacher=diffs,
                                            how="six test sequences regenerated after replacing their GT frames 1..63 and their teacher z* by noise: the trajectories must not change")
    # 6 teacher z from GT only
    c["06_teacher_from_gt"] = dict(status="pass" if cache_json["agreement_with_stage1_cache"]["test"]["max_abs_diff"] == 0 else "warn", encoded_with=str(np.load(S.teacher_cache_path(ds), allow_pickle=True)["encoded_with"]),
                                  agreement_with_stage1_cache=cache_json["agreement_with_stage1_cache"], how="s2_cache_teacher.py encodes the GT frames of the three splits with E_z only; identical to the Stage-1 latents cache")
    # 7 teacher normalisation uses train statistics only
    zc = np.load(S.teacher_cache_path(ds), allow_pickle=True); tr = data.idx["train"]
    zt = zc["z"][tr].reshape(-1, zc["z"].shape[-1]); za = zc["z"][zc["has"]].reshape(-1, zc["z"].shape[-1])
    ok7 = all(np.allclose(ck["stats"]["z_mu"], zt.mean(0), atol=1e-4) and np.allclose(ck["stats"]["z_sd"], zt.std(0), atol=1e-4) for m, ck in cks.items() if S.MODELS[m]["z"])
    c["07_train_only_z_normalisation"] = dict(status="pass" if ok7 else "fail", max_abs_diff_train_mean=float(max(np.abs(ck["stats"]["z_mu"] - zt.mean(0)).max() for m, ck in cks.items() if S.MODELS[m]["z"])) if any(S.MODELS[m]["z"] for m in cks) else None,
                                             all_split_vs_train_mean_shift=float(np.abs(za.mean(0) - zt.mean(0)).max()), how="z_mu / z_sd stored in the B1 / B2 checkpoints vs the mean / std of the cached teacher over the TRAIN frames")
    # 8 the decoder reads only predicted z (+ r) and allowed geometry: code inspection + check 5
    src = (HERE / "s2_models.py").read_text(); src_d = (HERE / "s2_data.py").read_text()
    geo_chan = re.search(r"torch\.cat\(\[x, Nn, self\.pstat\[n\], \(alpha \* S\.N_PTS\)\[\.\.\., None\], valid\.float\(\)\[\.\.\., None\]\], -1\)", src_d) is not None
    dec_inputs = "self._decoder(self.D_z, geo, torch.cat([Sg, z_raw], -1))" in src and "torch.cat([Sg, z_raw.detach(), r], -1), code=r)" in src
    c["08_decoder_inputs"] = dict(status="pass" if (geo_chan and dec_inputs and all(d == 0.0 for d in diffs.values())) else "fail", geometry_channels="x / l, outward normal, nearest, coverage, area, valid (no contact)",
                                 decoder_conditioning="D_z: [G | z_hat]; D_r: [G | sg(z_hat) | r_hat] + tokens [geo | sg(C_bar)]", code_match=bool(geo_chan and dec_inputs), how="source inspection of s2_models.decode_maps / s2_data.geo_tokens + check 5")
    # 9 no GT-r supervision in B2
    tsrc = (HERE / "s2_train.py").read_text(); cols = pd.read_csv(S.train_log_path(ds, names["B2"])).columns.tolist() if S.train_log_path(ds, names["B2"]).exists() else []
    loss_terms = sorted({cc[4:] for cc in cols if cc.startswith("val_L_")})
    c["09_no_gt_r_supervision"] = dict(status="pass" if ("r_star" not in tsrc and "r_hat" not in tsrc.split("def step_losses")[1].split("def evaluate")[0].replace("r_sel", "") and set(loss_terms) <= {"L_z", "L_full", "L_zonly", "L_res"}) else "fail",
                                      b2_loss_terms=loss_terms, cache_holds_r_trajectories=bool("r" in zc.files), cache_holds_only_r_train_mean=bool("r_train_mean" in zc.files and "r" not in zc.files),
                                      how="loss terms of the B2 training log; the teacher cache stores only the 64-D train-mean r (bias initialisation), no r trajectories; no r target in s2_train.step_losses")
    # 10 B2 z-only branch used
    if "B2" in cks:
        lg = pd.read_csv(S.train_log_path(ds, names["B2"])); row = lg[lg.step == cks["B2"]["best_step"]].iloc[0]
        zo = np.load(S.metrics_path(ds, names["B2"] + "_zonly"), allow_pickle=True) if S.metrics_path(ds, names["B2"] + "_zonly").exists() else None
        b1 = mets.get("B1")
        ratio = float(np.nanmean(zo["E_C"]) / np.nanmean(b1["E_C"])) if (zo is not None and b1 is not None) else np.nan
        c["10_b2_zonly_branch_used"] = dict(status="pass" if (row["val_L_zonly"] < 2 * row["val_L_full"] and (np.isnan(ratio) or ratio < 1.5)) else "warn", val_L_zonly_at_best=float(row["val_L_zonly"]), val_L_full_at_best=float(row["val_L_full"]),
                                           lambda_c=cks["B2"]["lambdas"]["c"], test_E_C_zonly_over_B1=ratio, how="validation L_zonly vs L_full at the selected step; test E_C of the z-only path of B2 relative to B1")
    # 11 residual correction nonzero but not dominating
    if S.OUT.joinpath("r_metrics.csv").exists():
        rm = pd.read_csv(S.OUT / "r_metrics.csv"); rm = rm[rm.dataset == ds]
        if len(rm):
            r = rm.iloc[0]
            c["11_residual_nonzero_not_dominating"] = dict(status="pass" if (r.delta_norm_mean > 1e-3 and r.delta_over_bar_ratio < 1.0) else "warn", delta_norm_mean=float(r.delta_norm_mean), bar_minus_s0_norm_mean=float(r.bar_minus_s0_norm_mean),
                                                           delta_over_bar_ratio=float(r.delta_over_bar_ratio), E_C_gain_from_r=float(r.E_C_gain_from_r), how="||Delta_t|| vs ||C_bar_t - s_0|| on the test trajectories (r_metrics.csv)")
    # 12 comparable budgets
    bud = {m: dict(steps=int(ck["steps"]), best_step=int(ck["best_step"]), stopped_by=ck["stopped_by"], hours=round(ck["seconds"] / 3600, 2)) for m, ck in cks.items()}
    same_cfg = len({json.dumps({k: v for k, v in ck["cfg"].items() if not k.startswith("lambda")}, sort_keys=True) for ck in cks.values()}) == 1
    manual = {m: b for m, b in bud.items() if b["stopped_by"].startswith("manual")}
    c["12_comparable_budgets"] = dict(status="pass" if (same_cfg and all(b["steps"] >= S.TRAIN["min_steps"] for b in bud.values())) else "warn", budgets=bud, same_recipe=bool(same_cfg),
                                     stopped_by_hand_before_min_steps={m: dict(steps=b["steps"], best_step=b["best_step"], checks_without_improvement=int(cks[m].get("checks_without_improvement", -1))) for m, b in manual.items()},
                                     how="steps / stopping reason of every run; identical recipe (lr, warm-up, batch, EMA, max steps, patience) in the checkpoints. warn: runs marked manual_stop were stopped "
                                         "by the user's decision before the 20 000-step minimum, 8-14 validation checks after their best step (4 000); the selected state is the same EMA state the rule would have saved unless a later step had set a new best")
    # 13 no test-based selection
    lgs = {m: pd.read_csv(S.train_log_path(ds, n)) for m, n in names.items() if S.train_log_path(ds, n).exists()}
    sel_ok = all(abs(float(lg.loc[lg.val.idxmin(), "step"]) - cks[m]["best_step"]) < 1 for m, lg in lgs.items()) and all(not any("test" in cc for cc in lg.columns) for lg in lgs.values())
    c["13_no_test_selection"] = dict(status="pass" if sel_ok else "fail", selected_steps={m: int(ck["best_step"]) for m, ck in cks.items()}, how="the saved checkpoint is the EMA state at the validation-objective minimum of the training log; the logs contain no test quantity")
    # 14 Stage-1 checkpoint recorded by hash / path
    c["14_stage1_recorded"] = dict(status="pass" if md5_ok else "fail", teacher=dict(path=str(S.teacher_ckpt_path(ds)), md5=tmd5, best_step=cache_json["best_step"], n_params=cache_json["n_params_teacher"]),
                                  decoders_initialised_from=str(S.teacher_ckpt_path(ds)), how="cache json + checkpoint metadata")
    # 15 decoded values numerically valid
    rng = {m: dict(raw_min=float(z["raw_min"]), raw_max=float(z["raw_max"]), frac_below_0=float(z["frac_below_0"]), frac_above_1=float(z["frac_above_1"]), n_nonfinite=int(z["n_nonfinite"])) for m, z in preds.items()}
    ok15 = all(v["n_nonfinite"] == 0 and v["raw_min"] > -0.5 and v["raw_max"] < 1.5 for v in rng.values())
    c["15_numerically_valid"] = dict(status="pass" if ok15 else "fail", ranges=rng, how="raw range, fraction outside [0, 1] and non-finite count of every prediction file (the previous D0 was also unclamped)")
    S.write_json(S.OUT / f"sanity_{ds}.json", c)
    print(json.dumps({k: v["status"] for k, v in c.items()}, indent=1))


if __name__ == "__main__":
    main()
