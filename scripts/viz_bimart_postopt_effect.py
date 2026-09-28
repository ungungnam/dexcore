#!/usr/bin/env python
"""Does BimArt's post-optimisation repair the action's self-contradiction -- and at what cost?

The 1200-d action encodes the hand twice: keypoint positions, and dirvec (the offset from each
keypoint to the nearest object surface point). They must agree, and for GT they agree exactly.
The raw diffusion output does not. BimArt's post-optimisation contains a projection loss that
targets exactly this residual -- but it holds dirvec FIXED and moves the MANO hand to satisfy it
(`joints_dirvec` is cloned once outside the loop; only MANO parameters are optimised).

So the optimiser does not resolve the contradiction -- it takes dirvec's side. If dirvec is the
wrong half, the hand is dragged somewhere worse. This measures, at three stages
(raw diffusion -> MANO fit -> post-optimisation):

  * the projection residual  |kp + dirvec - object surface|   (does it get repaired?)
  * the distance to the GT keypoints                          (at what cost?)
  * how far the hand moved                                    (how hard was it dragged?)

  python scripts/viz_bimart_postopt_effect.py --windows 0 5 --mano-steps 4000

Slow: the MANO fit is 4000 steps per window. Figures go to
$DEXCORE_RESULT_ROOT/analysis/bimart/postopt/<run>/plots.
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
LOG = logging.getLogger("bimart_postopt")
N_KP = 100
STAGES = ["raw diffusion", "after MANO fit", "after post-opt"]


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def proj_residual(joints, dirvec, obj_verts):
    """mm from (keypoint + dirvec) to the nearest object vertex, per frame per keypoint. World space."""
    proj = joints + dirvec
    out = np.empty(proj.shape[:2])
    for t in range(len(proj)):
        out[t] = cKDTree(obj_verts[t]).query(proj[t])[0]
    return out * 1000


def fig_stages(records, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    names = [r["name"][:22] for r in records]
    x = np.arange(len(STAGES))

    ax = axes[0]
    for j, r in enumerate(records):
        ax.plot(x, [np.median(r["resid"][s]) for s in STAGES], "o-", lw=1.8, ms=7, label=names[j])
    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=9)
    ax.set_ylabel("median |kp + dirvec - surface|  (mm)")
    ax.set_title("does the optimiser repair the contradiction?", fontsize=11)
    ax.axhline(0, c="#333333", ls="--", lw=1)
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axes[1]
    for j, r in enumerate(records):
        ax.plot(x, [r["gt_rmse"][s] for s in STAGES], "o-", lw=1.8, ms=7, label=names[j])
    ax.set_xticks(x); ax.set_xticklabels(STAGES, fontsize=9)
    ax.set_ylabel("RMSE to GT keypoints  (mm)")
    ax.set_title("at what cost? -- distance to the real hand", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    ax = axes[2]
    w = .35
    xs = np.arange(len(records))
    ax.bar(xs - w/2, [r["moved"]["fit"] for r in records], w, color="#7f7f7f",
           label="diffusion -> MANO fit")
    ax.bar(xs + w/2, [r["moved"]["opt"] for r in records], w, color="#d62728",
           label="MANO fit -> post-opt")
    ax.set_xticks(xs); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean keypoint displacement (mm)")
    ax.set_title("how far was the hand dragged?", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    fig.suptitle("The post-optimisation holds dirvec fixed and moves the hand to satisfy it",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--windows", nargs="+", type=int, default=[0, 5],
                    help="window indices; the default pairs a grab window with a use window")
    ap.add_argument("--mano-steps", type=int, default=4000)
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
    from inference.postprocess import postprocess_output, get_mano_fit
    from inference.optimization_refinement import PostOptimization
    from scripts.viz_bimart_denoising import (make_subset_dataset, run_contact_stage,
                                              run_motion_stage)

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    cfg = yaml_util.load_yaml(args.config)
    contact_cfg = yaml_util.load_yaml(cfg["test"]["contact_config_file"])
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/postopt" / run
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
    post_opt = PostOptimization(save_dir=str(out))

    records = []
    for b, batch in enumerate(loader):
        if b not in args.windows:
            continue
        name = batch["viz"]["filename"][-1]
        cat = batch["viz"]["category"][0]
        LOG.info("[window %d] %s", b, name)

        bn = data_util.preprocess_batch(batch, stat_dict, device)
        _, _, c_final, _ = run_contact_stage(contact_model, bn)
        bn = contact_model.postprocess_oneshot_contact(c_final, bn)
        m = run_motion_stage(cfg, ema_model, noise_scheduler, bn, guidance_mod, mano_layer,
                             stat_dict, hand_index, mesh_dict[cat], device)
        naction = data_util.unnormalize_item(
            torch.from_numpy(m["cfg"][-1])[None].to(device).float(),
            stat_dict["action"]["mean"], stat_dict["action"]["std"])
        unnorm = data_util.unnormalize_dict(bn, stat_dict)
        out_dict = postprocess_output(unnorm, naction, mesh_dict[cat],
                                      hand_keypoints=cfg["data"]["hand_keypoints"])

        obj_w = out_dict["obj_verts"][0]                                   # [T, V, 3] world
        dirvec = np.concatenate([out_dict["pred"]["left"]["dirvec"][0],
                                 out_dict["pred"]["right"]["dirvec"][0]], axis=1)
        j_raw = np.concatenate([out_dict["pred"]["left"]["joints"][0],
                                out_dict["pred"]["right"]["joints"][0]], axis=1)
        gt_kp = np.concatenate([out_dict["gt"]["left"]["verts"][0][:, hand_index],
                                out_dict["gt"]["right"]["verts"][0][:, hand_index]], axis=1)

        LOG.info("   MANO fit (%d steps)...", args.mano_steps)
        mano_param, lv, rv = get_mano_fit(out_dict, mano_layer, idxl=hand_index, idxr=hand_index,
                                          ph=cfg["data"]["pred_horizon"],
                                          mano_steps=args.mano_steps,
                                          hand_keypoints=cfg["data"]["hand_keypoints"])
        out_dict["pred"]["left"]["verts"] = lv
        out_dict["pred"]["right"]["verts"] = rv
        out_dict["mano_param"] = data_util.TensorDictToNp(mano_param)
        # fit_mano returns np.array(list) -> [B, T, 778, 3]; refine_noisy_motions later
        # writes back without the batch dim, so the two stages index differently
        j_fit = np.concatenate([lv[0][:, hand_index], rv[0][:, hand_index]], axis=1)

        LOG.info("   post-optimisation...")
        out_dict = post_opt.refine_noisy_motions(out_dict)
        j_opt = np.concatenate([out_dict["pred"]["left"]["joints"],
                                out_dict["pred"]["right"]["joints"]], axis=1)

        rmse = lambda a: float(np.sqrt(((a - gt_kp) ** 2).sum(-1).mean())) * 1000
        r = {"name": name, "window": b,
             "resid": {STAGES[0]: proj_residual(j_raw, dirvec, obj_w),
                       STAGES[1]: proj_residual(j_fit, dirvec, obj_w),
                       STAGES[2]: proj_residual(j_opt, dirvec, obj_w)},
             "gt_rmse": {STAGES[0]: rmse(j_raw), STAGES[1]: rmse(j_fit), STAGES[2]: rmse(j_opt)},
             "moved": {"fit": float(np.linalg.norm(j_fit - j_raw, axis=-1).mean()) * 1000,
                       "opt": float(np.linalg.norm(j_opt - j_fit, axis=-1).mean()) * 1000}}
        records.append(r)
        LOG.info("   residual  %s", {k: round(float(np.median(v)), 1) for k, v in r["resid"].items()})
        LOG.info("   GT RMSE   %s", {k: round(v, 1) for k, v in r["gt_rmse"].items()})
        LOG.info("   moved     %s", {k: round(v, 1) for k, v in r["moved"].items()})

    fig_stages(records, out / "plots/01_postopt_effect.png")
    (out / "summary.json").write_text(json.dumps({
        "run": run, "mano_steps": args.mano_steps,
        "windows": {r["name"]: {
            "residual_median_mm": {k: float(np.median(v)) for k, v in r["resid"].items()},
            "residual_p90_mm": {k: float(np.percentile(v, 90)) for k, v in r["resid"].items()},
            "gt_rmse_mm": r["gt_rmse"], "displacement_mm": r["moved"]} for r in records}},
        indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
