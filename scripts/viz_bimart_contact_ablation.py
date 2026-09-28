#!/usr/bin/env python
"""Whose error is the translation bias -- the contact module's, or the motion model's?

The hand trajectory comes out shifted from GT, and roughly 72% of that discrepancy is a rigid
translation. That measurement alone cannot say where the shift originates: stage 1 may be handing
stage 2 a wrong contact map, or stage 2 may misplace the hand however well it is conditioned.

Only a counterfactual separates them. The motion model is run twice per window under an otherwise
identical procedure, changing nothing but the contact map it is conditioned on:

    arm GT    conditioned on the ground-truth contact map   -> whatever bias remains is stage 2's own
    arm PRED  conditioned on stage 1's prediction           -> the system as it actually runs

The difference between the arms is stage 1's contribution. The GT arm's residual is stage 2's.
Stage 1 is re-sampled for every PRED sample, as the real pipeline does, so that arm's spread
includes stage 1's own variance.

  python scripts/viz_bimart_contact_ablation.py --categories laptop --batches 8 --samples 3

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/contact_ablation/<run>/plots.
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
LOG = logging.getLogger("bimart_contact_ablation")
N_KP = 100
PH = 64
ARMS = ["GT contact", "stage-1 contact"]


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def measure(kp_s, gt_kp, obj_verts_m):
    """Split the discrepancy into a rigid shift and what survives removing it, per hand."""
    hands = {"left": slice(0, N_KP), "right": slice(N_KP, 2 * N_KP)}
    tot, shift, resid = [], [], []
    for sl in hands.values():
        off = kp_s[:, :, sl].mean(axis=2) - gt_kp[None, :, sl].mean(axis=2)
        d = kp_s[:, :, sl] - gt_kp[None, :, sl]
        tot.append(np.sqrt((d ** 2).sum(-1).mean(axis=-1)) * 1000)
        shift.append(np.linalg.norm(off, axis=-1) * 1000)
        r = d - off[:, :, None]
        resid.append(np.sqrt((r ** 2).sum(-1).mean(axis=-1)) * 1000)
    surf = np.empty(kp_s.shape[1:3])
    for t in range(PH):
        surf[t] = cKDTree(obj_verts_m[t]).query(kp_s[0, t])[0]
    return {"total_mm": float(np.mean(tot)), "shift_mm": float(np.mean(shift)),
            "resid_mm": float(np.mean(resid)), "surface_mm": float(np.median(surf) * 1000)}


def fig_ablation(records, out):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.0))
    names = [r["short"] for r in records]
    x = np.arange(len(records))
    colour = {"GT contact": "#2ca02c", "stage-1 contact": "#d62728"}

    for ax, key, lab, title in (
            (axes[0], "total_mm", "RMSE to GT (mm)", "total discrepancy"),
            (axes[1], "shift_mm", "rigid shift (mm)", "the translation component"),
            (axes[2], "resid_mm", "residual after the shift (mm)",
             "what the shift does NOT explain")):
        for j, arm in enumerate(ARMS):
            ax.bar(x + (j - .5) * .4, [r[arm][key] for r in records], .4,
                   color=colour[arm], label=arm)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(lab); ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[3]
    attributable = [100 * (1 - r["GT contact"]["shift_mm"] / r["stage-1 contact"]["shift_mm"])
                    for r in records]
    ax.bar(x, attributable, .6, color="#1f77b4")
    ax.axhline(0, c="#333333", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of the shift removed by a correct contact map")
    ax.set_title("how much of the translation is stage 1's fault?", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("Same motion model, same procedure -- only the contact map changes", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--samples", type=int, default=3)
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
    out = result_root() / "analysis/bimart/contact_ablation" / run
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

        scale = mesh_dict[cat]["scale"]
        obj_verts_m = dataset.obj_process_data_all[file_idx]["obj_cano_verts_dense"][
            start:start + PH] / scale
        bn = data_util.preprocess_batch(batch, stat_dict, device)
        gt_contact = bn["obs"]["contact_points"].clone()
        gt_kp = data_util.unnormalize_item(
            bn["action"].clone(), stat_dict["action"]["mean"],
            stat_dict["action"]["std"])[0].cpu().numpy()[:, :N_KP * 6].reshape(PH, -1, 3)

        r = {"name": name, "category": cat}
        for arm in ARMS:
            kps = []
            for s in range(args.samples):
                if arm == "GT contact":
                    bn["obs"]["contact_points"] = gt_contact.clone()
                else:
                    bn["obs"]["contact_points"] = gt_contact.clone()  # shape source for stage 1
                    _, _, c_final, _ = run_contact_stage(contact_model, bn)
                    bn = contact_model.postprocess_oneshot_contact(c_final, bn)
                m = run_motion_stage(cfg, ema_model, noise_scheduler, bn, guidance_mod, mano_layer,
                                     stat_dict, hand_index, mesh_dict[cat], device)
                act = m["cfg"][-1] * (a_std + 1e-10) + a_mean
                kps.append(act[:, :N_KP * 6].reshape(PH, -1, 3))
            r[arm] = measure(np.stack(kps), gt_kp, obj_verts_m)
            LOG.info("   %-16s total %6.1f | shift %6.1f | resid %5.1f | surface %6.1f mm",
                     arm, r[arm]["total_mm"], r[arm]["shift_mm"], r[arm]["resid_mm"],
                     r[arm]["surface_mm"])
        records.append(r)

    for r in records:
        r["short"] = short_name(r["name"], [q["name"] for q in records])
    fig_ablation(records, out / "plots/01_contact_ablation.png")
    (out / "summary.json").write_text(json.dumps({
        "run": run, "samples_per_arm": args.samples, "motion_ckpt_epoch": int(epoch),
        "windows": {r["short"]: {a: r[a] for a in ARMS} for r in records}}, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
