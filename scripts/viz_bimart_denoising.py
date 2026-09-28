#!/usr/bin/env python
"""BimArt's two-stage denoising on ARCTIC, captured step by step across several test windows.

Both stages are DDPM with prediction_type="sample", so the network output at every step IS the
current estimate of the clean signal -- x0_hat is readable directly, no epsilon algebra. This
script runs the real inference path (contact prior -> motion model, with cost guidance and CFG)
over N test windows, records x_t and x0_hat at all 50 steps, and plots how each stage converges.

MANO fitting and the post-optimisation are skipped: they run after diffusion and cost ~5 min a
sample, and nothing in them changes the denoising trajectory being studied.

  python scripts/viz_bimart_denoising.py --categories laptop --batches 6
  python scripts/viz_bimart_denoising.py --categories laptop microwave --batches 8 --gpu 1

Runs under the bimart conda env:
  /home/uhnam/miniconda3/envs/bimart/bin/python scripts/viz_bimart_denoising.py

Arrays and figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/denoising/<run>/.
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
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("bimart_denoising")

N_BPS2 = 1024        # slots per hand (512 top + 512 bottom)
N_KP = 100           # FPS-sampled MANO vertices per hand
# BimArt's viz.norm_high (0.15 m) is a highlight ceiling, not the data range: GT distances run
# past 0.6 m. Colour scales here come from the data, or these figures saturate to solid black.
CLOSE_M = 0.02       # what "in contact" means when a binary read is wanted


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


# ----------------------------------------------------------------- capture-enabled model wrappers
def make_subset_dataset(motion_data_mod, cfg, categories):
    """MotionDataset, restricted to a few categories.

    The upstream class hard-codes all 11 categories and loads every test file into RAM (~180 MB
    each). Overriding the split keeps the working set small without touching third_party.
    """
    class SubsetMotionDataset(motion_data_mod.MotionDataset):
        def get_train_test_split(self):
            super().get_train_test_split()
            keep = lambda p: Path(p).parts[-3] in categories
            for split in ("train", "test"):
                for key in ("hand_process_files", "obj_process_files"):
                    self.data_files[split][key] = [p for p in self.data_files[split][key] if keep(p)]
            LOG.info("restricted to %s: %d test files", categories,
                     len(self.data_files["test"]["obj_process_files"]))
    return SubsetMotionDataset(cfg, split="test", return_aux_info=True)


class RecordingGuidance:
    """inference.guidance.Guidance, with the contact-discrepancy loss recorded per step."""

    def __init__(self, guidance_mod, **kw):
        self.g = guidance_mod.Guidance(**kw)
        self.losses = []

    def step(self, diff_pred):
        with torch.enable_grad():
            diff_pred.requires_grad_(True)
            from utils import data_util
            unnorm = data_util.unnormalize_item(
                diff_pred, self.g.stat_dict["action"]["mean"], self.g.stat_dict["action"]["std"])
            lkp, rkp = self.g.get_left_and_right_hand_kp(unnorm)
            loss = (self.g.cm_discrepancy(lkp, self.g.cm_pred[:, :N_BPS2])
                    + self.g.cm_discrepancy(rkp, self.g.cm_pred[:, N_BPS2:])) * self.g.w_cm
            grad = torch.autograd.grad(loss, diff_pred)[0]
            scale = 1.0 / (torch.linalg.norm(grad) + 1e-7) * self.g.guidance_scale
            out = diff_pred - scale * grad
        self.losses.append(float(loss.detach()))
        return out


# --------------------------------------------------------------------------------- the two stages
@torch.no_grad()
def run_contact_stage(contact_model, batch_norm):
    """Stage 1. Returns (x0_hat[K,T,2048], x_t[K,T,2048], final, timesteps) in contact-model norm."""
    data = contact_model.prepare_oneshot_contact_data(batch_norm)
    x = torch.randn(data["action"].shape, device=contact_model.device).float()
    contact_model.noise_scheduler.set_timesteps(contact_model.cfg["num_timesteps"])
    x0_hats, xts, steps = [], [], []
    for k in contact_model.noise_scheduler.timesteps:
        xts.append(x[0].cpu().numpy().copy())
        x0 = contact_model.model(sample=x, timestep=k, obj_feat=data["obs"]["obj_feat"],
                                 global_cond=data["obs"])
        x0_hats.append(x0[0].cpu().numpy().copy())
        steps.append(int(k))
        x = contact_model.noise_scheduler.step(model_output=x0, timestep=k, sample=x).prev_sample
    return np.stack(x0_hats), np.stack(xts), x, np.array(steps)


def run_motion_stage(cfg, ema_model, noise_scheduler, batch_norm, guidance_mod, mano_layer,
                     stat_dict, hand_index, mesh_viz_dict, device):
    """Stage 2. Captures the conditional, guided and CFG-combined estimate at every step."""
    B = batch_norm["action"].shape[0]
    x = torch.randn((B, cfg["data"]["pred_horizon"], cfg["data"]["action_dim"]),
                    device=device).float()
    noise_scheduler.set_timesteps(cfg["train"]["num_diffusion_iters"])
    guide = RecordingGuidance(guidance_mod, cfg=cfg, mano_layer=mano_layer, stat_dict=stat_dict,
                              hand_index=hand_index, test_batch=batch_norm, device=device,
                              w_cm=1, guidance_scale=1, mesh_viz_dict=mesh_viz_dict)
    rec = {k: [] for k in ("x_t", "cond", "guided", "cfg")}
    steps = []
    s = cfg["test"]["cfg_guide_strength"]
    for k in noise_scheduler.timesteps:
        rec["x_t"].append(x[0].detach().cpu().numpy().copy())
        with torch.no_grad():
            cond = ema_model(x, k, obj_feat=batch_norm["obs"]["object"],
                             contact_cond=batch_norm["obs"]["contact_points"],
                             global_states=batch_norm["obs"]["global_states"], contact_on_prob=1.0)
        rec["cond"].append(cond[0].detach().cpu().numpy().copy())
        guided = guide.step(cond.detach())
        rec["guided"].append(guided[0].detach().cpu().numpy().copy())
        with torch.no_grad():
            uncond = ema_model(x, k, obj_feat=batch_norm["obs"]["object"],
                               global_states=batch_norm["obs"]["global_states"], contact_on_prob=0)
            final = (1 + s) * guided - s * uncond
            rec["cfg"].append(final[0].detach().cpu().numpy().copy())
            x = noise_scheduler.step(model_output=final, timestep=k, sample=x).prev_sample
        steps.append(int(k))
    out = {k: np.stack(v) for k, v in rec.items()}
    out["guidance_loss"] = np.array(guide.losses)
    out["timesteps"] = np.array(steps)
    out["final"] = x[0].detach().cpu().numpy().copy()
    return out


# ------------------------------------------------------------------------------------------ plots
def _dist_to_gt(x0, gt):
    """MSE against the GT window. A sample need not match it -- this is a reference, not an error."""
    return ((x0 - gt[None]) ** 2).mean(axis=(1, 2))


def _self_convergence(x0):
    """RMS change between consecutive x0_hat estimates: when the sample stops moving."""
    return np.sqrt(((x0[1:] - x0[:-1]) ** 2).mean(axis=(1, 2)))


def _fit_view(ax, pts):
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_axis_off()


def fig_contact_maps(rec, gt, steps, name, out):
    """Stage 1's left-hand contact estimate unfolding over the 50 steps, at the data's own scale."""
    picks = [0, 9, 19, 29, 39, 44, 47, 49]
    vmax = float(np.percentile(gt[:, :N_BPS2], 97))
    fig, axes = plt.subplots(2, 5, figsize=(19, 6.8))
    axes = axes.ravel()
    for ax, i in zip(axes, picks):
        ax.imshow(rec[i][:, :N_BPS2].T, aspect="auto", origin="lower", cmap="magma_r",
                  vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"step {i+1}/50   (t={steps[i]})", fontsize=9)
        ax.axhline(512, c="#00ffff", lw=.8, ls="--")
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[len(picks)]
    im = ax.imshow(gt[:, :N_BPS2].T, aspect="auto", origin="lower", cmap="magma_r",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title("ground truth", fontsize=9, color="#d62728")
    ax.axhline(512, c="#00ffff", lw=.8, ls="--")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=.046, label="distance to hand (m)")
    axes[-1].set_axis_off()
    fig.suptitle(f"Stage 1 contact prior, x0_hat per denoising step -- {name}\n"
                 f"x: 64 frames   y: 1024 left-hand BPS slots (cyan = top/bottom split)   "
                 f"colour 0 to {vmax:.2f} m (GT p97)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_contact_on_mesh(rec, gt, bps_inds, obj_verts_m, name, out):
    """Stage 1 painted on the object -- distance field on top, thresholded contact below."""
    picks = [0, 19, 34, 44, 49]
    vmax = float(np.percentile(gt[0, :N_BPS2], 97))
    pts = obj_verts_m[bps_inds[0, :N_BPS2]]
    fig = plt.figure(figsize=(18, 6.2))
    for j, i in enumerate(picks + ["gt"]):
        vals = (gt if i == "gt" else rec[i])[0, :N_BPS2]
        lab = "ground truth" if i == "gt" else f"step {i+1}/50"
        col = "#d62728" if i == "gt" else "black"

        ax = fig.add_subplot(2, len(picks) + 1, j + 1, projection="3d")
        ax.scatter(*obj_verts_m.T, s=.6, c="#ececec", alpha=.3)
        p = ax.scatter(*pts.T, s=8, c=vals, cmap="magma_r", vmin=0, vmax=vmax)
        ax.set_title(lab, fontsize=9, color=col)
        _fit_view(ax, obj_verts_m)

        ax = fig.add_subplot(2, len(picks) + 1, len(picks) + 2 + j, projection="3d")
        ax.scatter(*obj_verts_m.T, s=.6, c="#ececec", alpha=.3)
        near = vals < CLOSE_M
        if near.any():
            ax.scatter(*pts[near].T, s=14, c="#d62728")
        ax.set_title(f"< {CLOSE_M*1000:.0f} mm:  {near.sum()} slots", fontsize=9, color=col)
        _fit_view(ax, obj_verts_m)

    fig.subplots_adjust(hspace=-.22, top=.86, bottom=.02, left=.02, right=.90)
    fig.colorbar(p, ax=fig.axes, fraction=.011, label="left-hand distance (m)")
    fig.suptitle(f"Stage 1 contact prior on the object surface, window frame 0 -- {name}\n"
                 f"top: distance field (0 to {vmax:.2f} m)   bottom: thresholded contact",
                 fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_motion_3d(x0_un, xt_un, gt_un, obj_verts_m, name, out, frame=32):
    """Stage 2 in the space it predicts. Top row is the noisy iterate x_t, bottom is the estimate
    x0_hat -- with prediction_type='sample' the estimate is a plausible hand from step one, while
    x_t is still mostly noise, so the two rows look nothing alike until late."""
    picks = [0, 14, 29, 39, 44, 47, 49]
    kp = lambda a: a[frame, :N_KP * 6].reshape(-1, 3)
    ref = np.vstack([kp(x0_un[i]) for i in picks] + [kp(gt_un), obj_verts_m])
    fig = plt.figure(figsize=(20, 6.0))
    n = len(picks) + 1
    for j, i in enumerate(picks + ["gt"]):
        for row, src in ((0, xt_un), (1, x0_un)):
            ax = fig.add_subplot(2, n, row * n + j + 1, projection="3d")
            ax.scatter(*obj_verts_m.T, s=.6, c="#dddddd", alpha=.35)
            pts = kp(gt_un) if i == "gt" else kp(src[i])
            ax.scatter(*pts[:N_KP].T, s=5, c="#1f77b4", alpha=.9)
            ax.scatter(*pts[N_KP:].T, s=5, c="#d62728", alpha=.9)
            if row == 0:
                ax.set_title("ground truth" if i == "gt" else f"step {i+1}/50", fontsize=9,
                             color="#d62728" if i == "gt" else "black")
            _fit_view(ax, ref)
    fig.subplots_adjust(hspace=-.20, top=.86, bottom=.02, left=.12, right=.98)
    fig.text(.085, .68, "noisy iterate\n$x_t$", fontsize=10, ha="center", weight="bold")
    fig.text(.085, .26, "estimate\n$\\hat{x}_0$", fontsize=10, ha="center", weight="bold")
    fig.suptitle(f"Stage 2 motion model at window frame {frame} -- {name}   "
                 f"(blue = left hand, red = right hand, grey = object, canonical space, metres)",
                 fontsize=12)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_convergence(records, out):
    """Every window on one set of axes: what each stage does over its 50 steps."""
    fig = plt.figure(figsize=(19, 9.2))
    gs = GridSpec(2, 4, figure=fig, hspace=.36, wspace=.30)
    cmap = plt.get_cmap("tab10")
    K = np.arange(1, 51)

    def panel(pos, title, ylab, log=True):
        ax = fig.add_subplot(gs[pos])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("denoising step"); ax.set_ylabel(ylab, fontsize=9)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=.25)
        return ax

    ax = panel((0, 0), "Stage 1: contact map vs. GT window", "MSE (normalised units)")
    for j, r in enumerate(records):
        ax.plot(K, r["contact_mse"], c=cmap(j % 10), lw=1.3, label=r["name"][:24])
    ax.legend(fontsize=6, ncol=1)

    ax = panel((0, 1), "Stage 2: hand feature vs. GT window", "MSE (normalised units)")
    for j, r in enumerate(records):
        ax.plot(K, r["motion_mse"], c=cmap(j % 10), lw=1.3)

    ax = panel((0, 2), "Stage 2 split:\nkeypoints (solid) vs. dirvec (dotted)", "MSE (normalised)")
    for j, r in enumerate(records):
        ax.plot(K, r["mse_kp"], c=cmap(j % 10), lw=1.3)
        ax.plot(K, r["mse_dirvec"], c=cmap(j % 10), lw=1.1, ls=":")

    ax = panel((0, 3), "self-convergence: |x0_hat(k) - x0_hat(k-1)|\n"
                       "stage 1 (solid) vs. stage 2 (dotted)", "RMS change")
    for j, r in enumerate(records):
        ax.plot(K[1:], r["contact_dx"], c=cmap(j % 10), lw=1.3)
        ax.plot(K[1:], r["motion_dx"], c=cmap(j % 10), lw=1.1, ls=":")

    ax = panel((1, 0), "cost guidance: predicted contact vs.\ncontact re-derived from the hands",
               "discrepancy (m$^2$)")
    for j, r in enumerate(records):
        ax.plot(K, r["guidance_loss"], c=cmap(j % 10), lw=1.3)

    ax = panel((1, 1), "how far each correction moves x0_hat\nguidance (solid) vs. CFG (dotted)",
               "RMS change")
    for j, r in enumerate(records):
        ax.plot(K, r["d_guidance"], c=cmap(j % 10), lw=1.3)
        ax.plot(K, r["d_cfg"], c=cmap(j % 10), lw=1.1, ls=":")

    ax = panel((1, 2), "Stage 2 iterate x_t (solid)\nvs. estimate x0_hat (dotted)", "std", log=False)
    for j, r in enumerate(records):
        ax.plot(K, r["xt_std"], c=cmap(j % 10), lw=1.3)
        ax.plot(K, r["x0_std"], c=cmap(j % 10), lw=1.1, ls=":")

    ax = panel((1, 3), "final contact: predicted vs. GT quantiles\n(dashed = agreement)",
               "predicted (m)", log=False)
    q = np.linspace(1, 99, 99)
    lo, hi = 0, 0
    for j, r in enumerate(records):
        gq = np.percentile(r["gt_contact_m"], q)
        pq = np.percentile(r["pred_contact_m"], q)
        ax.plot(gq, pq, c=cmap(j % 10), lw=1.3)
        hi = max(hi, gq.max(), pq.max())
    ax.plot([lo, hi], [lo, hi], c="#333333", ls="--", lw=1)
    ax.set_xlabel("ground truth (m)")

    fig.suptitle("BimArt denoising across %d ARCTIC test windows (50 DDPM steps, "
                 "prediction_type='sample')" % len(records), fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--batches", type=int, default=6, help="number of test windows to denoise")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--config", default="config_files/bimart_inference.yaml")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--detail-batches", nargs="+", type=int, default=[0, 5],
                    help="window indices to draw the per-window detail figures for; the default "
                         "pairs a grab window with a use window, which behave very differently")
    ap.add_argument("--save-arrays", action="store_true",
                    help="also dump the full per-step tensors (~1 GB per window)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(BIMART_ROOT)
    sys.path.insert(0, str(BIMART_ROOT))

    from utils import data_util, model_util, yaml_util, mano_utils
    from dataset import motion_data
    from contact_prior.contact_inf_module import ContactInference
    import inference.guidance as guidance_mod

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    cfg = yaml_util.load_yaml(args.config)
    contact_cfg = yaml_util.load_yaml(cfg["test"]["contact_config_file"])
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/denoising" / run
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
    LOG.info("motion model loaded from %s (epoch %s)", ckpt, epoch)
    contact_model = ContactInference(mesh_dict, cfg=contact_cfg, motion_config=cfg)
    noise_scheduler = model_util.createScheduler(cfg)
    hand_index = np.load(cfg["data"]["hand_index_path"]).astype(np.int32)
    mano_layer = mano_utils.create_mano_layer()

    records, meta = [], []
    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        name = batch["viz"]["filename"][-1]
        cat = batch["viz"]["category"][0]
        LOG.info("[%d/%d] %s", b + 1, args.batches, name)

        gt_contact_raw = copy.deepcopy(batch["obs"]["contact_points"])
        bn = data_util.preprocess_batch(batch, stat_dict, device)
        gt_motion = bn["action"][0].cpu().numpy().copy()
        # GT contact in the *contact model's* normalisation, for a like-for-like stage-1 curve
        gt_contact_c = data_util.normalize_item(
            gt_contact_raw.to(device).float(),
            contact_model.stat_dict["action"]["mean"],
            contact_model.stat_dict["action"]["std"])[0].cpu().numpy()

        c_x0, c_xt, c_final, c_steps = run_contact_stage(contact_model, bn)
        bn = contact_model.postprocess_oneshot_contact(c_final, bn)
        m = run_motion_stage(cfg, ema_model, noise_scheduler, bn, guidance_mod, mano_layer,
                             stat_dict, hand_index, mesh_dict[cat], device)

        # unnormalised contact, for the maps painted in metres
        cmean = contact_model.stat_dict["action"]["mean"].cpu().numpy()
        cstd = contact_model.stat_dict["action"]["std"].cpu().numpy()
        c_x0_m = c_x0 * (cstd + 1e-10) + cmean
        gt_c_m = gt_contact_raw[0].numpy()

        rec = {
            "name": name, "category": cat,
            "contact_mse": _dist_to_gt(c_x0, gt_contact_c),
            "motion_mse": _dist_to_gt(m["cfg"], gt_motion),
            "mse_kp": _dist_to_gt(m["cfg"][:, :, :N_KP * 6], gt_motion[:, :N_KP * 6]),
            "mse_dirvec": _dist_to_gt(m["cfg"][:, :, N_KP * 6:], gt_motion[:, N_KP * 6:]),
            "guidance_loss": m["guidance_loss"],
            "d_guidance": np.sqrt(((m["guided"] - m["cond"]) ** 2).mean(axis=(1, 2))),
            "d_cfg": np.sqrt(((m["cfg"] - m["guided"]) ** 2).mean(axis=(1, 2))),
            "xt_std": m["x_t"].std(axis=(1, 2)), "x0_std": m["cfg"].std(axis=(1, 2)),
            "contact_dx": _self_convergence(c_x0), "motion_dx": _self_convergence(m["cfg"]),
            "pred_contact_m": c_x0_m[-1], "gt_contact_m": gt_c_m,
        }
        records.append(rec)
        meta.append({k: (v if isinstance(v, str) else float(v[-1]))
                     for k, v in rec.items() if k in
                     ("name", "category", "contact_mse", "motion_mse", "guidance_loss")})

        if b in set(args.detail_batches):  # per-window detail figures
            file_idx = dataset.indices[b]["file_idx"]
            # the stored canonical mesh is unit-sphere normalised; /scale puts it back in metres,
            # which is the space the predicted hand keypoints live in
            obj_verts = dataset.obj_process_data_all[file_idx]["obj_cano_verts_dense"][
                int(batch["viz"]["start_index"][0])] / mesh_dict[cat]["scale"]
            bps_inds = batch["viz"]["bps_viz_index"][0].numpy()
            gt_action_un = data_util.unnormalize_item(
                torch.from_numpy(gt_motion).to(device),
                stat_dict["action"]["mean"], stat_dict["action"]["std"]).cpu().numpy()
            a_std = stat_dict["action"]["std"].cpu().numpy()
            a_mean = stat_dict["action"]["mean"].cpu().numpy()
            m_cfg_un = m["cfg"] * (a_std + 1e-10) + a_mean
            m_xt_un = m["x_t"] * (a_std + 1e-10) + a_mean
            fig_contact_maps(c_x0_m, gt_c_m, c_steps, name, out / f"plots/01_contact_maps_{name}.png")
            fig_contact_on_mesh(c_x0_m, gt_c_m, bps_inds, obj_verts, name,
                                out / f"plots/02_contact_on_mesh_{name}.png")
            fig_motion_3d(m_cfg_un, m_xt_un, gt_action_un, obj_verts, name,
                          out / f"plots/03_motion_keypoints_{name}.png")
            if args.save_arrays:
                np.savez_compressed(out / f"window_{b}_{name}.npz", contact_x0=c_x0,
                                    contact_xt=c_xt, gt_contact=gt_contact_c,
                                    gt_motion=gt_motion, **m)

    fig_convergence(records, out / "plots/04_convergence.png")
    (out / "summary.json").write_text(json.dumps(
        {"run": run, "config": args.config, "categories": args.categories,
         "windows": meta, "gpu": args.gpu, "motion_ckpt_epoch": int(epoch)}, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
