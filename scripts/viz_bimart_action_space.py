#!/usr/bin/env python
"""Is BimArt's hand-trajectory representation space well formed?

Distance to the GT window says little about a generative model -- a different but valid sample is
not an error. These three tests judge the model's OWN output instead, and two of them have an exact
reference that does not depend on matching GT:

  A. internal consistency.  The 1200-d action is 600 d of hand keypoints plus 600 d of dirvec, the
     offset from each keypoint to the nearest object surface point. The two halves are redundant by
     construction, so `kp + dirvec` MUST land on the object. For GT it lands exactly (that is how
     dirvec was defined), so any residual is pure model incoherence -- and it is precisely what the
     post-optimisation's projection loss exists to repair.
  B. diversity.  Several samples per window from the same conditioning. Collapse to one answer and
     the model is not really generative; wild spread and the conditioning is not binding.
  C. hand-shape validity.  The 100 points per hand are FPS-sampled MANO vertices, so their pairwise
     distances describe a hand. Compare that distribution to GT's: a broken shape shows up as an
     inflated or collapsed spread, independent of where the hand is placed.

  python scripts/viz_bimart_action_space.py --categories laptop --batches 8 --samples 5

Runs under the bimart conda env. Figures go to
$DEXCORE_RESULT_ROOT/analysis/bimart/action_space/<run>/plots.
"""
import pathlib as _pl
import sys as _sys

# scripts/ holds modules that shadow the stdlib (select.py), so it must not stay on sys.path.
_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import copy
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial import cKDTree

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("bimart_action_space")
N_KP = 100
PH = 64


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def split_action(a):
    """[T, 1200] -> keypoints [T, 200, 3] and dirvec [T, 200, 3], left hand first."""
    kp = a[:, :N_KP * 6].reshape(len(a), -1, 3)
    dv = a[:, N_KP * 6:].reshape(len(a), -1, 3)
    return kp, dv


def projection_residual(action, obj_verts_m):
    """A. mm from (kp + dirvec) to the nearest object vertex, per frame per keypoint."""
    kp, dv = split_action(action)
    proj = kp + dv
    out = np.empty(proj.shape[:2])
    for t in range(len(proj)):
        out[t] = cKDTree(obj_verts_m[t]).query(proj[t])[0]
    return out * 1000


def pairwise_spread(action):
    """C. median and max pairwise distance among one hand's 100 points, per frame, in mm."""
    kp, _ = split_action(action)
    med, span = [], []
    for t in range(len(kp)):
        for h in (slice(0, N_KP), slice(N_KP, 2 * N_KP)):
            d = np.linalg.norm(kp[t, h][:, None] - kp[t, h][None], axis=-1)
            iu = np.triu_indices(N_KP, 1)
            med.append(np.median(d[iu])); span.append(d[iu].max())
    return np.array(med) * 1000, np.array(span) * 1000


def offset_decomposition(kp_s, gt_kp):
    """B, extended. RMSE to GT is a scalar; the bias it summarises is a vector.

    Splits the discrepancy into a per-frame per-hand rigid translation and whatever survives
    removing it, and measures how consistently that translation points the same way. Resultant
    length R is |mean of the unit vectors|: 1 = always the same direction, 0 = random.
    """
    S, T = kp_s.shape[:2]
    hands = {"left": slice(0, N_KP), "right": slice(N_KP, 2 * N_KP)}
    o = {}
    for h, sl in hands.items():
        off = kp_s[:, :, sl].mean(axis=2) - gt_kp[None, :, sl].mean(axis=2)   # [S, T, 3]
        resid = kp_s[:, :, sl] - gt_kp[None, :, sl] - off[:, :, None]         # translation removed
        total = kp_s[:, :, sl] - gt_kp[None, :, sl]
        unit = off / (np.linalg.norm(off, axis=-1, keepdims=True) + 1e-12)
        o[h] = {
            "offset_mm": np.linalg.norm(off, axis=-1) * 1000,                 # [S, T]
            "total_rmse_mm": np.sqrt((total ** 2).sum(-1).mean(axis=-1)) * 1000,
            "resid_rmse_mm": np.sqrt((resid ** 2).sum(-1).mean(axis=-1)) * 1000,
            "R_frames": np.linalg.norm(unit.mean(axis=1), axis=-1),           # [S] across frames
            "R_samples": np.linalg.norm(unit.mean(axis=0), axis=-1),          # [T] across samples
        }
    return o


def surface_distance(kp, obj_verts_m):
    """mm from each keypoint to the nearest object vertex -- is the hand held too far off?"""
    out = np.empty(kp.shape[:2])
    for t in range(len(kp)):
        out[t] = cKDTree(obj_verts_m[t]).query(kp[t])[0]
    return out * 1000


def fig_bias_direction(records, out):
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.9))
    names = [r["short"] for r in records]
    x = np.arange(len(records))

    ax = axes[0]
    tot = [np.mean([r["offset"][h]["total_rmse_mm"] for h in ("left", "right")]) for r in records]
    res = [np.mean([r["offset"][h]["resid_rmse_mm"] for h in ("left", "right")]) for r in records]
    ax.bar(x - .2, tot, .4, color="#d62728", label="total discrepancy")
    ax.bar(x + .2, res, .4, color="#1f77b4", label="after removing a rigid shift")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("RMSE to GT (mm)")
    ax.set_title("is the bias just a translation?", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    frac = [100 * (1 - rr / tt) for rr, tt in zip(res, tot)]
    ax.bar(x, frac, .6, color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of the discrepancy explained")
    ax.set_title("how much a single shift accounts for", fontsize=11)
    ax.set_ylim(0, 100); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    rf = [np.mean([r["offset"][h]["R_frames"].mean() for h in ("left", "right")]) for r in records]
    rs = [np.mean([r["offset"][h]["R_samples"].mean() for h in ("left", "right")]) for r in records]
    ax.bar(x - .2, rf, .4, color="#9467bd", label="across frames")
    ax.bar(x + .2, rs, .4, color="#ff7f0e", label="across samples")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("resultant length R   (1 = one direction)")
    ax.set_title("does the shift point a consistent way?", fontsize=11)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[3]
    ax.bar(x - .2, [np.median(r["surf_pred"]) for r in records], .4, color="#d62728",
           label="prediction")
    ax.bar(x + .2, [np.median(r["surf_gt"]) for r in records], .4, color="#7f7f7f",
           label="ground truth")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("median keypoint-to-surface distance (mm)")
    ax.set_title("relative to the object: too far off?", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    fig.suptitle("B, extended -- the bias as a vector, not a scalar", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def fig_consistency(records, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    names = [r["short"] for r in records]
    x = np.arange(len(records))

    ax = axes[0]
    ax.bar(x - .2, [np.median(r["resid_pred"]) for r in records], .4, color="#d62728",
           label="prediction")
    ax.bar(x + .2, [np.median(r["resid_gt"]) for r in records], .4, color="#7f7f7f",
           label="ground truth (exact by construction)")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("median |kp + dirvec - surface|  (mm)")
    ax.set_title("A. internal consistency\nthe two halves of the action must agree", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    ax.boxplot([r["resid_pred"].ravel() for r in records], labels=names, showfliers=False)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("|kp + dirvec - surface|  (mm)")
    ax.set_title("A. spread over all frames and keypoints", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    for j, r in enumerate(records):
        ax.plot(np.median(r["resid_pred"], axis=1), lw=1.2, label=r["short"])
    ax.set_xlabel("window frame"); ax.set_ylabel("median residual (mm)")
    ax.set_title("A. does it drift over the window?", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Is the action's own geometry self-consistent?", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def fig_diversity(records, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    names = [r["short"] for r in records]
    x = np.arange(len(records))

    ax = axes[0]
    ax.bar(x, [r["spread_mm"] for r in records], .6, color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean sd across samples (mm)")
    ax.set_title("B. how far apart are repeated samples?", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    for j, r in enumerate(records):
        ax.plot(r["spread_per_frame"], lw=1.2, label=r["short"])
    ax.set_xlabel("window frame"); ax.set_ylabel("sd across samples (mm)")
    ax.set_title("B. spread over the window", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)

    ax = axes[2]
    for j, r in enumerate(records):
        ax.scatter([j] * len(r["gt_rmse_per_sample"]), r["gt_rmse_per_sample"], s=18, alpha=.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("RMSE to GT keypoints (mm)")
    ax.set_title("B. do some samples land much closer than others?", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("Does the same conditioning produce a distribution, or one answer?", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def fig_shape(records, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    names = [r["short"] for r in records]
    x = np.arange(len(records))

    for ax, key, lab in ((axes[0], "med", "median pairwise distance"),
                         (axes[1], "span", "hand span (max pairwise)")):
        ax.bar(x - .2, [np.mean(r[f"{key}_pred"]) for r in records], .4, color="#d62728",
               label="prediction")
        ax.bar(x + .2, [np.mean(r[f"{key}_gt"]) for r in records], .4, color="#7f7f7f",
               label="ground truth")
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(f"{lab} (mm)")
        ax.set_title(f"C. {lab}", fontsize=11)
        ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    allp = np.concatenate([r["med_pred"] for r in records])
    allg = np.concatenate([r["med_gt"] for r in records])
    bins = np.linspace(min(allp.min(), allg.min()), max(allp.max(), allg.max()), 60)
    ax.hist(allg, bins=bins, alpha=.6, color="#7f7f7f", label="ground truth", density=True)
    ax.hist(allp, bins=bins, alpha=.6, color="#d62728", label="prediction", density=True)
    ax.set_xlabel("median pairwise distance (mm)"); ax.set_ylabel("density")
    ax.set_title("C. distribution over all hands and frames", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25)

    fig.suptitle("Are the 100 predicted points still shaped like a hand?", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--config", default="config_files/bimart_inference.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(BIMART_ROOT)
    _sys.path.insert(0, str(BIMART_ROOT))

    from utils import data_util, model_util, yaml_util, mano_utils
    from dataset import motion_data
    from contact_prior.contact_inf_module import ContactInference
    import inference.guidance as guidance_mod
    from scripts.viz_bimart_denoising import (make_subset_dataset, run_contact_stage,
                                              run_motion_stage)
    from scripts.viz_bimart_conditioning import short_name

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    cfg = yaml_util.load_yaml(args.config)
    contact_cfg = yaml_util.load_yaml(cfg["test"]["contact_config_file"])
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/action_space" / run
    (out / "plots").mkdir(parents=True, exist_ok=True)
    LOG.info("writing to %s  (GPU %d)", out, args.gpu)

    stat_dict = data_util.load_stat_dict(cfg["data"]["stat_dict_path"], device)
    dataset = make_subset_dataset(motion_data, cfg, set(args.categories))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    mesh_dict = dataset.mesh_dict
    base = model_util.createNN(cfg, device)
    ckpt = os.path.join(cfg["save_root_dir"], cfg["exp_name"], cfg["train"]["pretrain_model_name"])
    ema_model, _, _, _, epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
        base, None, None, ckpt, ema_model=None)
    ema_model.eval()
    contact_model = ContactInference(mesh_dict, cfg=contact_cfg, motion_config=cfg)
    noise_scheduler = model_util.createScheduler(cfg)
    hand_index = np.load(cfg["data"]["hand_index_path"]).astype(np.int32)
    mano_layer = mano_utils.create_mano_layer()
    a_mean = stat_dict["action"]["mean"].cpu().numpy()
    a_std = stat_dict["action"]["std"].cpu().numpy()

    records = []
    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        name = batch["viz"]["filename"][-1]
        cat = batch["viz"]["category"][0]
        start = int(batch["viz"]["start_index"][0])
        file_idx = dataset.indices[b]["file_idx"]
        LOG.info("[%d/%d] %s", b + 1, args.batches, name)

        # object surface in metres, canonical space -- the frame dirvec was defined in
        scale = mesh_dict[cat]["scale"]
        obj_all = dataset.obj_process_data_all[file_idx]["obj_cano_verts_dense"]
        obj_verts_m = obj_all[start:start + PH] / scale

        bn = data_util.preprocess_batch(batch, stat_dict, device)
        gt_action = data_util.unnormalize_item(
            bn["action"].clone(), stat_dict["action"]["mean"],
            stat_dict["action"]["std"])[0].cpu().numpy()

        samples = []
        for s in range(args.samples):
            _, _, c_final, _ = run_contact_stage(contact_model, bn)
            bn = contact_model.postprocess_oneshot_contact(c_final, bn)
            m = run_motion_stage(cfg, ema_model, noise_scheduler, bn, guidance_mod, mano_layer,
                                 stat_dict, hand_index, mesh_dict[cat], device)
            samples.append(m["cfg"][-1] * (a_std + 1e-10) + a_mean)   # final x0_hat, unnormalised
        samples = np.stack(samples)                                    # [S, 64, 1200]

        kp_s = samples[:, :, :N_KP * 6].reshape(args.samples, PH, -1, 3)
        gt_kp = gt_action[:, :N_KP * 6].reshape(PH, -1, 3)
        med_p, span_p = pairwise_spread(samples[0])
        med_g, span_g = pairwise_spread(gt_action)

        r = {
            "name": name, "category": cat,
            "resid_pred": projection_residual(samples[0], obj_verts_m),
            "resid_gt": projection_residual(gt_action, obj_verts_m),
            "spread_per_frame": kp_s.std(axis=0).mean(axis=(1, 2)) * 1000,
            "gt_rmse_per_sample": [float(np.sqrt(((k - gt_kp) ** 2).sum(-1).mean())) * 1000
                                   for k in kp_s],
            "med_pred": med_p, "med_gt": med_g, "span_pred": span_p, "span_gt": span_g,
            "offset": offset_decomposition(kp_s, gt_kp),
            "surf_pred": surface_distance(kp_s[0], obj_verts_m),
            "surf_gt": surface_distance(gt_kp, obj_verts_m),
        }
        r["spread_mm"] = float(r["spread_per_frame"].mean())
        records.append(r)
        LOG.info("   consistency %.1f mm (GT %.2f) | diversity %.1f mm | span %.0f vs GT %.0f mm",
                 np.median(r["resid_pred"]), np.median(r["resid_gt"]), r["spread_mm"],
                 span_p.mean(), span_g.mean())

    for r in records:
        r["short"] = short_name(r["name"], [q["name"] for q in records])

    fig_consistency(records, out / "plots/01_internal_consistency.png")
    fig_diversity(records, out / "plots/02_diversity.png")
    fig_shape(records, out / "plots/03_hand_shape.png")
    fig_bias_direction(records, out / "plots/04_bias_direction.png")

    (out / "summary.json").write_text(json.dumps({
        "run": run, "samples_per_window": args.samples, "motion_ckpt_epoch": int(epoch),
        "windows": {r["short"]: {
            "consistency_median_mm": float(np.median(r["resid_pred"])),
            "consistency_p90_mm": float(np.percentile(r["resid_pred"], 90)),
            "consistency_gt_median_mm": float(np.median(r["resid_gt"])),
            "diversity_mm": r["spread_mm"],
            "gt_rmse_per_sample_mm": r["gt_rmse_per_sample"],
            "span_pred_mm": float(np.mean(r["span_pred"])),
            "span_gt_mm": float(np.mean(r["span_gt"])),
            "bias_total_rmse_mm": float(np.mean([r["offset"][h]["total_rmse_mm"].mean()
                                                 for h in ("left", "right")])),
            "bias_resid_after_shift_mm": float(np.mean([r["offset"][h]["resid_rmse_mm"].mean()
                                                        for h in ("left", "right")])),
            "shift_magnitude_mm": float(np.mean([r["offset"][h]["offset_mm"].mean()
                                                 for h in ("left", "right")])),
            "R_across_frames": float(np.mean([r["offset"][h]["R_frames"].mean()
                                              for h in ("left", "right")])),
            "R_across_samples": float(np.mean([r["offset"][h]["R_samples"].mean()
                                               for h in ("left", "right")])),
            "surface_dist_pred_mm": float(np.median(r["surf_pred"])),
            "surface_dist_gt_mm": float(np.median(r["surf_gt"])),
        } for r in records}}, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
