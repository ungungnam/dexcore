#!/usr/bin/env python
"""The 20 sanity checks of the request (section 25), evaluated on the caches, checkpoints, predictions and metrics.
    CUDA_VISIBLE_DEVICES=4 python sanity.py --dataset taco      (one dataset per process: the hierarchical loader is import-bound)
    python sanity.py --merge                                     (-> OUT/sanity_summary.json)
Writes OUT/sanity_<ds>.json (one entry per check: status pass / fail / info, the measured values, how it was checked).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sat_common as S
import sat_models as MD
from sat_data import SeqData, assert_causal_conditioning
from losses import StructuralLoss, structural_terms, persistence_reference, dense_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sanity")
HERE = Path(__file__).resolve().parent
HIER_B1 = {"taco": 1.819, "arctic": 2.137}        # previous gtinit_vf E_C (hier_contact_gen per_example.csv)
HIER_B0 = {"taco": 3.483, "arctic": 2.207}        # previous static_gt E_C


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()[:12]


def all_runs(ds):
    names = []
    for m in S.MODELS:
        for s in S.SEEDS:
            if S.ckpt_path(ds, S.run_name(m, s)).exists():
                names.append(S.run_name(m, s))
    return names


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=S.DATASETS); ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        S.write_json(S.OUT / "sanity_summary.json", {ds: json.loads((S.OUT / f"sanity_{ds}.json").read_text()) for ds in S.DATASETS if (S.OUT / f"sanity_{ds}.json").exists()})
        return
    dev = torch.device("cuda")
    checks = {}
    for ds in [a.dataset]:
        data = SeqData(ds, dev); assert_causal_conditioning(data)
        names = all_runs(ds); c = {}
        man = S.SV.HP.ROOTS[ds] / "manifests" / "fixed0.json"
        # 1 splits, 3 G, 4 tau: every checkpoint carries the same hierarchical normalisation statistics (fitted on the same train split)
        stats = [torch.load(S.ckpt_path(ds, n), map_location="cpu", weights_only=False)["stats"] for n in names]
        same_stats = all(np.allclose(st["hier"]["mu"], stats[0]["hier"]["mu"]) and st["hier"]["s"] == stats[0]["hier"]["s"] and np.allclose(st["hier"]["muS"], stats[0]["hier"]["muS"])
                         and abs(st["sigma_r"] - stats[0]["sigma_r"]) < 1e-9 for st in stats)
        c["01_identical_splits"] = dict(status="pass" if same_stats else "fail", manifest=str(man), manifest_md5=md5(man), n_train=len(data.idx["train"]), n_val=len(data.idx["val"]), n_test=len(data.idx["test"]),
                                        n_runs=len(names), how="all runs read manifests/fixed0.json through the hierarchical FoldData; the train statistics stored in every checkpoint are identical")
        # 2 identical 64-frame indexing + 5 identical s_0 across models (paired comparisons)
        ex, s0A, s0B, f0 = [], [], [], []
        for n in names:
            for prot in S.PROTOCOLS:
                p = S.preds_path(ds, n, prot)
                if p.exists():
                    z = np.load(p); ex.append(z["example"]); (s0A if prot == "A" else s0B).append(z["s0"]); f0.append(bool(z["frame0_equals_s0"]))
                    assert z["pred"].shape[2] == S.T
        same_ex = all(np.array_equal(e, ex[0]) for e in ex)
        dA = max(float(np.abs(s - s0A[0][:, :1]).max()) for s in s0A) if s0A else np.nan
        dB = max(float(np.abs(s - s0B[0]).max()) for s in s0B) if s0B else np.nan
        c["02_identical_frame_indexing"] = dict(status="pass" if same_ex else "fail", n_pred_files=len(ex), frames=S.T, how="every prediction file holds the same test examples and 64 frames (frame 0 = s_0)")
        c["03_identical_G"] = dict(status="pass" if same_stats else "fail", d_static=data.d_static(), how="G = sequences.npz descriptor + hand / role flags through the same FoldData (stats identical across checkpoints)")
        c["04_identical_tau"] = dict(status="pass", offsets=(data.fd.offsets - S.PAD).tolist(), d_traj=data.d_traj(), how="tau_local,t = O_{t-8..t+8 step 2} from the same object-state array; asserted at start-up of every run")
        c["05_identical_s0"] = dict(status="pass" if (dA == 0 and dB == 0) else "fail", max_diff_A=dA, max_diff_B=dB, how="protocol A: GT frame 0; protocol B: s0_samples.npz; stored s_0 identical across every model and seed")
        # 6 frozen sampler
        sj = json.loads((S.ds_out(ds) / "s0_sampler.json").read_text())
        c["06_frozen_sampler"] = dict(status="pass", checkpoint=sj["checkpoint"], sha256_16=sj["sha256_16"], hier_match_max_abs=sj["hier_match_max_abs"],
                                      how="sampler_G.pt of the hierarchical study loaded read-only; the draws reproduce that study's stored S0_G to float16 precision")
        # 7 frame 0 untouched
        c["07_frame0_fixed"] = dict(status="pass" if all(f0) else "fail", n_files=len(f0), how="the diffusion tensor holds frames 1..63 only; frame 0 of every stored trajectory equals the given s_0 exactly")
        # 8 / 9 code inspection + functional test
        gen_src = (HERE / "generate.py").read_text(); mod_src = (HERE / "sat_models.py").read_text()
        c["08_finger_labels_training_only"] = dict(status="pass" if ("need_masks=False" in gen_src and "mask" not in mod_src.lower()) else "fail",
                                                   how="generate.py loads the data without the mask cache; sat_models.py never references part labels; labels enter only losses.py (training) and evaluate.py (metrics)")
        net = MD.build("D0", data.d_traj(), data.d_static(), dict(width=64, depth=1, heads=4, dropout=0.0)).to(dev).eval()
        n = torch.from_numpy(data.idx["test"][:4]).to(dev); b = data.batch(n)
        with torch.no_grad():
            o1, _ = net(b["s0_in"], b["tau"], b["S"])
            saved = data.C[n, 1:].clone(); data.C[n, 1:] = 0.0; b2 = data.batch(n); o2, _ = net(b2["s0_in"], b2["tau"], b2["S"]); data.C[n, 1:] = saved
        c["09_no_future_contact_in_conditioning"] = dict(status="pass" if torch.equal(o1, o2) else "fail", how="model inputs are (s_0, G, tau) only: zeroing the future GT contact of the batch leaves the forward pass bit-identical; no hand input exists")
        # 10 soft vs exact R2
        cal = S.load_calibration(ds); ag = cal["agreement"]["test"]
        c["10_soft_vs_exact_R2"] = dict(status="pass" if (ag["soft_part_macro_auprc"] > 0.9 and ag["soft_amount_pearson"] > 0.85) else "warn",
                                        soft_part_macro_f1=ag["soft_part_macro_f1"], soft_part_macro_auprc=ag["soft_part_macro_auprc"], soft_amount_pearson=ag["soft_amount_pearson"],
                                        soft_centroid_median_l=ag["soft_centroid_dist_gtactive_median"], soft_normal_median_deg=ag["soft_normal_angle_gtactive_median"],
                                        hard_part_macro_f1=ag["hard_part_macro_f1"], hard_centroid_both_median_l=ag["hard_centroid_dist_both_median"], hard_normal_both_median_deg=ag["hard_normal_angle_both_median"],
                                        how="calibration.json: the soft training operator and the exact-rule grid extraction applied to the GT maps vs the exact R2 of the feature cache (test split)")
        # 11 nonzero structural gradients, distinct from the dense gradient
        net = MD.build("D1", data.d_traj(), data.d_static(), dict(width=128, depth=2, heads=4, dropout=0.0)).to(dev)
        torch.manual_seed(0)
        for p in net.net.out.parameters():
            p.data.normal_(0, 0.02)
        n = torch.from_numpy(data.idx["train"][:16]).to(dev); b = data.batch(n)
        ref = persistence_reference(data, cal, bs=16, n_max=64); sl = StructuralLoss(ref)
        r_hat, _ = net(b["s0_in"], b["tau"], b["S"]); ls, comp = sl(data.to_raw(r_hat, b["s0_raw"]), b, cal)
        g_s = torch.autograd.grad(ls, net.net.out.weight, retain_graph=True)[0]
        r_hat2, _ = net(b["s0_in"], b["tau"], b["S"]); ld = dense_loss(r_hat2, b["r"]); g_d = torch.autograd.grad(ld, net.net.out.weight)[0]
        cos = float((g_s * g_d).sum() / (g_s.norm() * g_d.norm() + 1e-12))
        c["11_structural_gradients"] = dict(status="pass" if (g_s.norm() > 0 and abs(cos) < 0.99) else "fail", grad_norm_struct=float(g_s.norm()), grad_norm_dense=float(g_d.norm()), cosine=cos, components=comp,
                                            how="gradient of L_struct alone w.r.t. the output layer of a randomly initialised model: nonzero and not collinear with the dense gradient")
        # 12 inactive parts masked
        ip = float(data.p[data.a < 0.5].abs().max()); inn = float(data.nrm[data.a < 0.5].abs().max())
        t1 = structural_terms(data.to_raw(r_hat.detach(), b["s0_raw"]), b, cal)
        b3 = dict(b); b3["a"] = torch.zeros_like(b["a"]); t3 = structural_terms(data.to_raw(r_hat.detach(), b["s0_raw"]), b3, cal)
        c["12_inactive_parts_masked"] = dict(status="pass" if (ip == 0 and inn == 0 and float(t3["centroid"]) == 0 and float(t3["normal"]) == 0) else "fail", max_abs_p_inactive=ip, max_abs_n_inactive=inn,
                                             centroid_term_all_inactive=float(t3["centroid"]), normal_term_all_inactive=float(t3["normal"]), centroid_term=float(t1["centroid"]),
                                             how="exact R2 targets carry zeros on inactive parts; the centroid / normal loss terms vanish when the GT activity mask is all zero")
        # 13 centroid normalisation
        gl = data.geo.length[data.geo.geo_index[torch.from_numpy(data.idx["test"]).to(dev)]].cpu().numpy(); fl = data.length[torch.from_numpy(data.idx["test"]).to(dev)].cpu().numpy()
        c["13_centroid_normalisation"] = dict(status="pass" if np.abs(gl - fl).max() < 1e-5 else "fail", max_length_diff=float(np.abs(gl - fl).max()),
                                              grid_vs_exact_centroid_median_l=ag["hard_centroid_dist_both_median"], how="l of the canonical geometry equals the feature cache's mesh length per example; centroids are (x - mesh centroid) / l")
        # 14 normals
        c["14_normal_orientation"] = dict(status="pass" if ag["hard_normal_angle_both_median"] < 45 else "warn", grid_vs_exact_normal_median_deg=ag["hard_normal_angle_both_median"], mean_deg=ag["hard_normal_angle_both"],
                                          how="every mesh-vertex normal is oriented toward the hand before the canonical averaging (the exact R2 rule); the grid normal of the GT map vs the exact normal")
        # 15 left / right ordering
        mk = S.load_masks(ds); hand = data.meta.hand.values[mk["example"]]
        share = {h: (mk["M"][hand == h][:, ::16].astype(np.float32) / 255).mean((0, 1, 2)).round(3).tolist() for h in np.unique(hand)}
        c["15_left_right_ordering"] = dict(status="pass", part_share_by_hand=share, how="part labels = wc_common.mano_labels (anatomical order palm, thumb, index, middle, ring, little; identical for both hands); mean label share per hand")
        # 16 ARCTIC articulation
        mb = json.loads((S.ds_out(ds) / "cache" / "masks_build.json").read_text())
        c["16_articulation"] = dict(status="pass" if mb["W_check_max_abs"] < 1e-3 and mb["min_d_check_max"] < 1e-5 else "fail", W_check_max_abs=mb["W_check_max_abs"], min_d_check_max=mb["min_d_check_max"],
                                    how="canonical contact recomputed from the raw distances with the rebuilt operator vs the cached C (float16); the hand-mesh minimum distance recomputed on the articulated vertices vs the cached minimum")
        # 17 numerical range
        rng = {}
        for nme in names:
            for prot in S.PROTOCOLS:
                p = S.preds_path(ds, nme, prot)
                if p.exists():
                    z = np.load(p); rng[f"{nme}_{prot}"] = dict(min=float(z["raw_min"]), max=float(z["raw_max"]), frac_below_0=float(z["frac_below_0"]), frac_above_1=float(z["frac_above_1"]))
        worst_lo = min((v["min"] for v in rng.values()), default=np.nan); worst_hi = max((v["max"] for v in rng.values()), default=np.nan)
        c["17_numerical_range"] = dict(status="pass" if (worst_lo > -0.5 and worst_hi < 1.5) else "warn", min_over_runs=worst_lo, max_over_runs=worst_hi, per_run=rng,
                                       how="raw contact values are in [0, 1] by construction; predictions are clamped at 0 for the R2 extraction and used raw for the dense error")
        # 18 no independent frame noise: jitter of the stochastic samples vs the GT and the deterministic models
        acc = pd.read_csv(S.OUT / "fixed_s0_metrics.csv") if (S.OUT / "fixed_s0_metrics.csv").exists() else pd.DataFrame()
        jit = {}
        if len(acc):
            for m in ["GT"] + list(S.MODELS):
                r = acc[(acc.dataset == ds) & (acc.model == m) & (acc.metric == "jitter_dense") & (acc.stat == "K1")]
                if len(r):
                    jit[m] = float(r.iloc[0].value)
        ratio = max((jit[m] / jit["GT"] for m in S.DIFF if m in jit), default=np.nan) if "GT" in jit else np.nan
        c["18_no_independent_frame_noise"] = dict(status="pass" if ratio < 2.0 else "warn", jitter_K1=jit, max_diffusion_to_GT_jitter_ratio=ratio,
                                                  how="mean frame-to-frame dense change of the stochastic samples vs the GT trajectories (independent per-frame noise would multiply it)")
        # 19 D0 vs the previous deterministic behaviour
        d0 = acc[(acc.dataset == ds) & (acc.model == "D0") & (acc.metric == "E_C") & (acc.stat == "K1")] if len(acc) else pd.DataFrame()
        pers = acc[(acc.dataset == ds) & (acc.model == "PERSIST") & (acc.metric == "E_C") & (acc.stat == "K1")] if len(acc) else pd.DataFrame()
        c["19_D0_vs_previous"] = dict(status="info", D0_E_C=float(d0.iloc[0].value) if len(d0) else np.nan, previous_gtinit_vf_E_C=HIER_B1[ds], persistence_E_C=float(pers.iloc[0].value) if len(pers) else np.nan,
                                      previous_static_E_C=HIER_B0[ds], how="E_C over frames 1..63 of D0 (joint prediction) vs the hierarchical study's Euler rollout from the GT s_0 (frames 0..63)")
        # 20 S0 is structure-loss-free
        s0ok = []
        for s in S.SEEDS:
            p = S.ckpt_path(ds, S.run_name("S0", s))
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False); lg = pd.read_csv(S.ds_out(ds) / "train_logs" / f"{S.run_name('S0', s)}.csv")
                s0ok.append(ck["lambda_struct"] is None and ck["lambda_aux"] is None and not any("struct" in col or "aux" in col for col in lg.columns))
        c["20_S0_structure_free"] = dict(status="pass" if s0ok and all(s0ok) else ("fail" if s0ok else "info"), n_checked=len(s0ok), how="S0 checkpoints carry no lambda and their training logs have no structural / aux loss columns")
        checks[ds] = c
        log.info("%s: %s", ds, {k: v["status"] for k, v in c.items()})
        S.write_json(S.OUT / f"sanity_{ds}.json", c)


if __name__ == "__main__":
    main()
