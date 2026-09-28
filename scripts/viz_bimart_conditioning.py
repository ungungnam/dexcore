#!/usr/bin/env python
"""What BimArt's transformers actually receive: the encoded conditioning latents, not the raw BPS.

Both stages inject conditioning by ADDITION into the token embedding, never by cross-attention:

    stage 2:  token(t) = PE(t) + input_process(x_t) + bps_encoder([obj || global]) + contact_encoder(c)
    stage 1:  token(t) = PE(t) + input_process(x_t) + bps_encoder([obj || global])

so the relative magnitude of those terms IS the conditioning budget -- a term that is small next to
the others cannot steer much, whatever it encodes. These figures measure that budget, show what the
512-d latents look like, and test two things the additive design depends on:

  * contact_encoder(c) must differ from the learned null_contact_token, or CFG has nothing to push
    against;
  * stage 1's contact error must survive the encoder, or the cascade's first stage does not matter.

  python scripts/viz_bimart_conditioning.py --categories laptop --batches 6
  python scripts/viz_bimart_conditioning.py --categories laptop microwave --batches 8 --gpu 1

Runs under the bimart conda env. Figures go to
$DEXCORE_RESULT_ROOT/analysis/bimart/conditioning/<run>/plots.
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
from matplotlib.gridspec import GridSpec

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("bimart_conditioning")
D = 512          # latent_cond_dim
PH = 64          # pred_horizon


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def rms(a, axis=-1):
    return np.sqrt((np.asarray(a) ** 2).mean(axis=axis))


def cosine(a, b, axis=-1):
    a, b = np.asarray(a), np.asarray(b)
    na = np.linalg.norm(a, axis=axis)
    nb = np.linalg.norm(b, axis=axis)
    return (a * b).sum(axis=axis) / (na * nb + 1e-12)


# ------------------------------------------------------------------ pulling the latents out of the
@torch.no_grad()
def motion_latents(model, bn, contact_norm, x_t, k):
    """Every additive term of a stage-2 token, each [PH, D], in the order the forward pass adds them."""
    obj = torch.cat([bn["obs"]["object"], bn["obs"]["global_states"]], dim=-1)
    z_obj = model.bps_encoder(obj)[0]
    z_contact = model.contact_encoder(contact_norm)[0]
    z_null = model.null_contact_token
    z_x = model.input_process(x_t.permute(1, 0, 2))[:, 0]           # input_process wants [T,B,F]
    pe = model.sequence_pos_encoder.pe[1:PH + 1, 0]                 # token 0 is the timestep embed
    t_emb = model.embed_timestep(torch.as_tensor([k], device=obj.device))[0, 0]
    return {k2: v.detach().cpu().numpy() for k2, v in
            dict(obj=z_obj, contact=z_contact, null=z_null, x=z_x, pe=pe,
                 timestep=t_emb[None].repeat(PH, 1)).items()}


@torch.no_grad()
def contact_latents(model, data, x_t, k):
    """The stage-1 equivalent. It has no contact branch -- object and global state are all it sees."""
    obj = torch.cat([data["obs"]["obj_feat"], data["obs"]["curr_global_states"]], dim=-1)
    z_obj = model.bps_encoder(obj)[0]
    z_x = model.input_process(x_t.permute(1, 0, 2))[:, 0]
    pe = model.sequence_pos_encoder.pe[1:PH + 1, 0]
    t_emb = model.embed_timestep(torch.as_tensor([k], device=obj.device))[0, 0]
    return {k2: v.detach().cpu().numpy() for k2, v in
            dict(obj=z_obj, x=z_x, pe=pe, timestep=t_emb[None].repeat(PH, 1)).items()}


# ------------------------------------------------------------------------------------------- plots
# The four terms that are genuinely summed into a FRAME token. The timestep embedding is not one
# of them -- it is concatenated as its own token at position 0 -- so it is drawn separately, for
# scale only, and left out of the share.
TERMS = [("x", "input_process($x_t$)", "#1f77b4"),
         ("obj", "bps_encoder([obj‖global])", "#2ca02c"),
         ("contact", "contact_encoder(c)", "#d62728"),
         ("pe", "positional encoding", "#9467bd")]
TSTEP = ("timestep", "timestep embedding (separate token)", "#8c564b")


def short_name(name, all_names):
    """Drop the prefix every window shares, so the labels stay distinguishable."""
    parts = [n.split("_") for n in all_names]
    keep = 0
    for i in range(min(len(q) for q in parts)):
        if len({q[i] for q in parts}) > 1:
            break
        keep = i + 1
    return "_".join(name.split("_")[keep:]) or name


def fig_budget(records, out):
    """Which additive term dominates the token -- the whole point of an additive design."""
    fig = plt.figure(figsize=(18, 8.6))
    gs = GridSpec(2, 3, figure=fig, hspace=.36, wspace=.26)
    r0 = records[0]

    ax = fig.add_subplot(gs[0, 0])
    for key, lab, c in TERMS:
        ax.plot(rms(r0["motion_early"][key]), c=c, lw=1.5, label=lab)
    ax.plot(rms(r0["motion_early"][TSTEP[0]]), c=TSTEP[2], lw=1.2, ls="--", label=TSTEP[1])
    ax.set_title("stage 2, step 1/50: per-frame RMS of each term", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("RMS over the 512 dims")
    ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for key, lab, c in TERMS:
        ax.plot(rms(r0["motion_late"][key]), c=c, lw=1.5, label=lab)
    ax.plot(rms(r0["motion_late"][TSTEP[0]]), c=TSTEP[2], lw=1.2, ls="--")
    ax.set_title("stage 2, step 50/50: the same terms at the end", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("RMS over the 512 dims")
    ax.set_yscale("log"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[0, 2])
    steps = r0["step_grid"]
    for key, lab, c in list(TERMS) + [TSTEP]:
        ax.plot(steps, [rms(s[key]).mean() for s in r0["motion_sweep"]], c=c, lw=1.5,
                marker="o", ms=3, ls="--" if key == TSTEP[0] else "-")
    ax.set_title("stage 2: how the budget shifts over denoising\n"
                 "(only input_process($x_t$) moves; the rest are fixed)", fontsize=11)
    ax.set_xlabel("denoising step"); ax.set_ylabel("mean RMS")
    ax.set_yscale("log"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[1, 0])
    for key, lab, c in [t for t in TERMS if t[0] != "contact"]:
        ax.plot(rms(r0["contact_early"][key]), c=c, lw=1.5, label=lab)
    ax.plot(rms(r0["contact_early"][TSTEP[0]]), c=TSTEP[2], lw=1.2, ls="--", label=TSTEP[1])
    ax.set_title("stage 1, step 1/50: no contact branch here", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("RMS over the 512 dims")
    ax.set_yscale("log"); ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1])
    w = .8 / len(records)
    allt = list(TERMS) + [TSTEP]
    xs = np.arange(len(allt))
    for j, r in enumerate(records):
        ax.bar(xs + j * w - .4 + w / 2,
               [rms(r["motion_late"][k2]).mean() for k2, _, _ in allt], w, label=r["short"])
    ax.set_xticks(xs); ax.set_xticklabels([t[1] for t in allt], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("mean RMS"); ax.set_yscale("log")
    ax.set_title("stage 2 at step 50, every window", fontsize=11)
    ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[1, 2])
    share = []
    for r in records:
        tot = sum(rms(r["motion_late"][k2]).mean() for k2, _, _ in TERMS)
        share.append([100 * rms(r["motion_late"][k2]).mean() / tot for k2, _, _ in TERMS])
    share = np.array(share)
    bottom = np.zeros(len(records))
    for i, (key, lab, c) in enumerate(TERMS):
        ax.bar(range(len(records)), share[:, i], .7, bottom=bottom, color=c, label=lab)
        bottom += share[:, i]
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels([r["short"] for r in records], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of the summed RMS")
    ax.set_title("conditioning budget of a frame token, as a share\n(timestep token excluded -- "
                 "it is not summed in)", fontsize=11)
    ax.legend(fontsize=7)

    fig.suptitle("BimArt conditioning budget -- every term is ADDED into the same 512-d token",
                 fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_latent_maps(rec, out):
    """The latents themselves, on a shared symmetric scale so magnitudes are comparable."""
    panels = [("x", "input_process($x_t$), step 1"), ("obj", "bps_encoder([obj‖global])"),
              ("contact", "contact_encoder(c), GT contact"), ("null", "null_contact_token"),
              ("pe", "positional encoding"), ("timestep", "timestep embedding (t=49)")]
    src = rec["motion_early"]
    v = max(np.abs(src[k]).max() for k, _ in panels)
    fig, axes = plt.subplots(2, 3, figsize=(17, 7.4))
    for ax, (key, title) in zip(axes.ravel(), panels):
        im = ax.imshow(src[key].T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-v, vmax=v,
                       interpolation="nearest")
        ax.set_title(f"{title}\nRMS {rms(src[key]).mean():.3f}", fontsize=10)
        ax.set_xlabel("window frame"); ax.set_ylabel("latent dim")
        fig.colorbar(im, ax=ax, fraction=.046)
    fig.suptitle(f"Stage 2 additive terms, {rec['name']} -- shared scale +/-{v:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_contact_vs_null(records, out):
    """CFG subtracts the null-token branch from the contact branch. If the two latents coincide,
    that subtraction is a no-op no matter what cfg_guide_strength says."""
    fig = plt.figure(figsize=(18, 8.4))
    gs = GridSpec(2, 3, figure=fig, hspace=.36, wspace=.26)
    r0 = records[0]
    c_gt, z_null = r0["motion_early"]["contact"], r0["motion_early"]["null"]

    ax = fig.add_subplot(gs[0, 0])
    v = max(np.abs(c_gt).max(), np.abs(z_null).max())
    im = ax.imshow((c_gt - z_null).T, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-v, vmax=v, interpolation="nearest")
    ax.set_title("contact_encoder(c) - null_contact_token\n(what CFG actually acts on)", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("latent dim")
    fig.colorbar(im, ax=ax, fraction=.046)

    ax = fig.add_subplot(gs[0, 1])
    for j, r in enumerate(records):
        ax.plot(cosine(r["motion_early"]["contact"], r["motion_early"]["null"]), lw=1.3,
                label=r["short"])
    ax.axhline(1, c="#333333", ls="--", lw=1)
    ax.set_title("cos(contact latent, null token) per frame\n1.0 would mean CFG is a no-op",
                 fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("cosine"); ax.grid(alpha=.25)
    ax.legend(fontsize=6, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    for j, r in enumerate(records):
        num = np.linalg.norm(r["motion_early"]["contact"] - r["motion_early"]["null"], axis=-1)
        den = np.linalg.norm(r["motion_early"]["contact"], axis=-1)
        ax.plot(num / (den + 1e-12), lw=1.3)
    ax.set_title("|contact - null| / |contact| per frame", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("relative gap"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(np.abs(c_gt - z_null).mean(axis=0), c="#d62728", lw=1.0)
    ax.set_title("per-dim mean |contact - null|", fontsize=11)
    ax.set_xlabel("latent dim"); ax.set_ylabel("mean |difference|"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[1, 1])
    for key, lab, c in (("obj", "bps_encoder output", "#2ca02c"),
                        ("contact", "contact_encoder output", "#d62728")):
        sd = np.concatenate([r["motion_early"][key] for r in records]).std(axis=0)
        ax.plot(np.sort(sd)[::-1], c=c, lw=1.4, label=f"{lab}  (dims with sd<1e-3: "
                                                      f"{int((sd < 1e-3).sum())})")
    ax.set_title("how many of the 512 dims carry signal\n(per-dim sd across all frames and windows)",
                 fontsize=11)
    ax.set_xlabel("dim, sorted"); ax.set_ylabel("sd"); ax.set_yscale("log")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(cosine(r0["motion_early"]["obj"], r0["motion_early"]["contact"]), c="#ff7f0e", lw=1.4,
            label="cos(obj latent, contact latent)")
    ax.plot(cosine(r0["motion_early"]["obj"], r0["motion_early"]["x"]), c="#1f77b4", lw=1.4,
            label="cos(obj latent, $x_t$ embedding)")
    ax.axhline(0, c="#333333", ls="--", lw=1)
    ax.set_title("do the added terms point in the same direction?", fontsize=11)
    ax.set_xlabel("window frame"); ax.set_ylabel("cosine"); ax.grid(alpha=.25); ax.legend(fontsize=8)

    fig.suptitle("The contact branch and the null token -- what classifier-free guidance has to work "
                 "with", fontsize=13)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_pred_vs_gt(records, out):
    """Does stage 1's contact error survive the encoder and reach stage 2's tokens?

    The null token sets the scale: it is the largest perturbation the contact branch can undergo,
    since it is what the branch becomes when conditioning is dropped entirely.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    names = [r["short"] for r in records]

    ax = axes[0]
    gap_pred, gap_null = [], []
    for r in records:
        gt, pr, nu = (r["motion_early"][k] for k in ("contact", "contact_pred", "null"))
        gap_pred.append(np.linalg.norm(pr - gt, axis=-1).mean())
        gap_null.append(np.linalg.norm(nu - gt, axis=-1).mean())
    x = np.arange(len(records))
    ax.bar(x - .2, gap_pred, .4, color="#d62728", label="|encode(stage-1 pred) - encode(GT)|")
    ax.bar(x + .2, gap_null, .4, color="#7f7f7f", label="|null token - encode(GT)|")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean L2 over frames")
    ax.set_title("stage-1 error in latent space,\nagainst the drop-conditioning baseline", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    frac = 100 * np.array(gap_pred) / (np.array(gap_null) + 1e-12)
    ax.bar(x, frac, .6, color="#d62728")
    ax.axhline(100, c="#333333", ls="--", lw=1, label="as far off as dropping conditioning")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of the null-token distance")
    ax.set_title("how much of the conditioning signal\nstage 1's error destroys", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    for j, r in enumerate(records):
        ax.plot(cosine(r["motion_early"]["contact_pred"], r["motion_early"]["contact"]), lw=1.3,
                label=r["short"])
    ax.axhline(1, c="#333333", ls="--", lw=1)
    ax.set_xlabel("window frame"); ax.set_ylabel("cosine")
    ax.set_title("cos(encode(pred), encode(GT)) per frame", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Does stage 1's contact error reach stage 2's tokens?", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)
    return {n: {"latent_gap": float(g), "null_gap": float(nu), "pct_of_null": float(f)}
            for n, g, nu, f in zip(names, gap_pred, gap_null, frac)}


def fig_latent_space(records, out):
    """Where the windows sit relative to each other, in the encoders' own 512-d spaces."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9))
    cmap = plt.get_cmap("tab10")
    for ax, key, title in zip(axes, ("obj", "contact", "x"),
                              ("bps_encoder output", "contact_encoder output (GT)",
                               "input_process($x_t$), step 1")):
        stacked = np.concatenate([r["motion_early"][key] for r in records])
        mu = stacked.mean(axis=0)
        u, s, vt = np.linalg.svd(stacked - mu, full_matrices=False)
        proj = (stacked - mu) @ vt[:2].T
        for j, r in enumerate(records):
            p = proj[j * PH:(j + 1) * PH]
            ax.plot(p[:, 0], p[:, 1], c=cmap(j % 10), lw=1.0, alpha=.85, label=r["short"])
            ax.scatter(p[0, 0], p[0, 1], c=[cmap(j % 10)], s=28, marker="o", zorder=3)
        var = 100 * (s[:2] ** 2).sum() / (s ** 2).sum()
        ax.set_title(f"{title}\nfirst 2 PCs hold {var:.0f}% of the variance", fontsize=10.5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=.25)
    axes[0].legend(fontsize=6, ncol=1)
    fig.suptitle("Each window's 64 frames traced through the conditioning encoders "
                 "(dot = first frame)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


# -------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--config", default="config_files/bimart_inference.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(BIMART_ROOT)
    _sys.path.insert(0, str(BIMART_ROOT))

    from utils import data_util, model_util, yaml_util
    from dataset import motion_data
    from contact_prior.contact_inf_module import ContactInference
    from scripts.viz_bimart_denoising import make_subset_dataset, run_contact_stage

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    cfg = yaml_util.load_yaml(args.config)
    contact_cfg = yaml_util.load_yaml(cfg["test"]["contact_config_file"])
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/conditioning" / run
    (out / "plots").mkdir(parents=True, exist_ok=True)
    LOG.info("writing to %s  (GPU %d)", out, args.gpu)

    stat_dict = data_util.load_stat_dict(cfg["data"]["stat_dict_path"], device)
    dataset = make_subset_dataset(motion_data, cfg, set(args.categories))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    base = model_util.createNN(cfg, device)
    ckpt = os.path.join(cfg["save_root_dir"], cfg["exp_name"], cfg["train"]["pretrain_model_name"])
    ema_model, _, _, _, epoch = model_util.load_model_optimizer_lrscheduler_checkpt(
        base, None, None, ckpt, ema_model=None)
    ema_model.eval()
    contact_model = ContactInference(dataset.mesh_dict, cfg=contact_cfg, motion_config=cfg)
    noise_scheduler = model_util.createScheduler(cfg)
    LOG.info("motion model epoch %s", epoch)

    STEP_GRID = [0, 9, 24, 39, 49]
    records = []
    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        name = batch["viz"]["filename"][-1]
        LOG.info("[%d/%d] %s", b + 1, args.batches, name)
        gt_contact = copy.deepcopy(batch["obs"]["contact_points"])
        bn = data_util.preprocess_batch(batch, stat_dict, device)
        gt_norm = bn["obs"]["contact_points"].clone()          # motion normalisation, as the model sees it

        # stage 1, so the predicted contact can be encoded alongside the GT one
        c_x0, _, c_final, _ = run_contact_stage(contact_model, bn)
        data_c = contact_model.prepare_oneshot_contact_data(bn)
        bn = contact_model.postprocess_oneshot_contact(c_final, bn)
        pred_norm = bn["obs"]["contact_points"].clone()

        # replay the stage-2 loop only far enough to read x_t at the grid points
        x = torch.randn((1, PH, cfg["data"]["action_dim"]), device=device).float()
        noise_scheduler.set_timesteps(cfg["train"]["num_diffusion_iters"])
        sweep, early, late = [], None, None
        for i, k in enumerate(noise_scheduler.timesteps):
            if i in STEP_GRID:
                lat = motion_latents(ema_model, bn, gt_norm, x, int(k))
                lat["contact_pred"] = ema_model.contact_encoder(pred_norm)[0].detach().cpu().numpy()
                sweep.append(lat)
                if i == 0:
                    early = lat
                if i == 49:
                    late = lat
            with torch.no_grad():
                pred = ema_model(x, k, obj_feat=bn["obs"]["object"],
                                 contact_cond=bn["obs"]["contact_points"],
                                 global_states=bn["obs"]["global_states"], contact_on_prob=1.0)
                x = noise_scheduler.step(model_output=pred, timestep=k, sample=x).prev_sample

        xc = torch.randn((1, PH, 2048), device=device).float()
        records.append({"name": name, "category": batch["viz"]["category"][0],
                        "motion_early": early, "motion_late": late, "motion_sweep": sweep,
                        "step_grid": [s + 1 for s in STEP_GRID],
                        "contact_early": contact_latents(contact_model.model, data_c, xc, 49)})

    all_names = [r["name"] for r in records]
    for r in records:
        r["short"] = short_name(r["name"], all_names)

    fig_budget(records, out / "plots/01_conditioning_budget.png")
    fig_latent_maps(records[0], out / "plots/02_latent_maps.png")
    fig_contact_vs_null(records, out / "plots/03_contact_vs_null.png")
    gaps = fig_pred_vs_gt(records, out / "plots/04_stage1_error_in_latent_space.png")
    fig_latent_space(records, out / "plots/05_latent_space.png")

    budget = {r["name"]: {k: float(rms(r["motion_late"][k]).mean()) for k, _, _ in TERMS}
              for r in records}
    (out / "summary.json").write_text(json.dumps(
        {"run": run, "categories": args.categories, "motion_ckpt_epoch": int(epoch),
         "budget_rms_step50": budget, "stage1_error_in_latent": gaps}, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
