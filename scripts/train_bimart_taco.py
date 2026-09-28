#!/usr/bin/env python
"""Train BimArt's contact or motion model on the TACO port.

  CUDA_VISIBLE_DEVICES=5 python scripts/train_bimart_taco.py --model contact

BimArt's own trainers instantiate the ARCTIC dataset classes directly, so the loop is rewritten
here rather than imported -- which means it has to match upstream deliberately rather than by
construction. It does, on every count that changes the result:

    per-ITEM diffusion timesteps      `(B,)`, not one shared draw per batch. The model broadcasts a
                                      scalar without complaint, so getting this wrong is silent; it
                                      leaves every sample in a batch at the same noise level and
                                      inflates the gradient variance.
    AdamW, weight_decay 1e-6
    cosine LR schedule, 100 warmup steps, stepped every batch
    EMA of the weights, power 0.75    motion model only, as upstream
    x0 prediction, MSE against the clean action

The model code itself is upstream's, unmodified; only the dimensions and the data differ.

ONE DELIBERATE ADDITION: A VALIDATION PASS. BimArt has none -- its split is four hardcoded file
indices per ARCTIC category and its motion config trains on `all` -- so training loss is the only
signal upstream ever sees, and nothing distinguishes the epoch-100 checkpoint from epoch-200.

Validation runs on TACO's `test_1`, which shares every triplet and every mesh with training and
differs only in which sequences were recorded. That is what a validation set is for: it detects
overfitting to the training distribution without spending `test_2`, `test_3` or `test_4`, which
hold out meshes, triplets and both, and which are the generalisation questions this port exists to
ask. Using any of those three for validation would make its own number meaningless.

The validation loss uses FIXED noise and FIXED timesteps, reseeded identically every epoch: a
diffusion loss resampled each time is dominated by which timesteps were drawn, and the curve would
be unreadable.
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
for _p in (_REPO, str(_pl.Path(_REPO) / "third_party/BimArt")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


def build(model_name, cfg, device):
    from model_arch.contact_transformer import ContactTransformer
    from model_arch.motion_transformer import MotionTransformer

    if model_name == "contact":
        return ContactTransformer(
            input_feats=cfg["num_bps_points"] * 4, latent_dim=cfg["latent_cond_dim"],
            ff_size=cfg["ff_size"], num_layers=cfg["num_layers"], num_heads=cfg["num_heads"],
            dropout=cfg["dropout"], activation=cfg["activation"],
            bps_input_dim=cfg["num_bps_points"] * cfg["num_features"] * 2 + cfg["global_state_dim"],
            pred_horizon=cfg["pred_horizon"],
            diffusion_step_embed_dim=cfg["latent_cond_dim"]).to(device)
    d, m = cfg["data"], cfg["model"]
    return MotionTransformer(
        input_feats=d["action_dim"], ff_size=m["ff_size"], num_layers=m["num_layers"],
        num_heads=m["num_heads"], dropout=m["dropout"], activation=m["activation"],
        latent_dim=m["latent_cond_dim"],
        bps_input_dim=d["bps_feature_dim"] * d["num_bps_points"] * 2 + d["global_state_dim"],
        pred_horizon=d["pred_horizon"], contact_input_dim=d["num_bps_points"] * 4,
        diffusion_step_embed_dim=m["diffusion_step_embed_dim"]).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("contact", "motion"), required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override the config; a larger batch means fewer optimiser steps, so "
                        "scale --lr with it or the run sees proportionally less training")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--tag", default=None, help="appended to the run directory and wandb name")
    p.add_argument("--val-split", default="test_1",
                   help="TACO split used for validation. test_1 shares every triplet and mesh with "
                        "train, so it measures overfitting without spending test_2/3/4. Empty "
                        "string disables validation.")
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--store", default="train_store",
                   help="store subdirectory of base_dir; train_store_fps holds the corrected "
                        "contact label")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("train")

    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from torch.utils.data import DataLoader

    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from utils import data_util

    cfg_path = args.config or f"configs/bimart_taco/train_{args.model}_config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    is_contact = args.model == "contact"
    d = cfg if is_contact else cfg["data"]
    tr = cfg if is_contact else cfg["train"]
    epochs = args.epochs or (cfg["num_epochs"] if is_contact else tr["num_epochs"])
    bs = args.batch_size or (cfg["batch_size"] if is_contact else tr["train_batch_size"])
    steps = cfg["num_timesteps"] if is_contact else tr["num_diffusion_iters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_cls = TacoContactDataset if is_contact else TacoMotionDataset
    ds = ds_cls(root=d["base_dir"], split=d["split"], pred_horizon=d["pred_horizon"],
                base_frame=d["base_frame"], store_dir=args.store, end_frame=d["end_frame"])
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.workers, drop_last=True,
                    persistent_workers=args.workers > 0,
                    prefetch_factor=6 if args.workers else None)
    val_dl = None
    if args.val_split:
        val_ds = ds_cls(root=d["base_dir"], split=args.val_split, pred_horizon=d["pred_horizon"],
                        base_frame=d["base_frame"], store_dir=args.store, end_frame=d["end_frame"])
        val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, drop_last=False)
    model = build(args.model, cfg, device)
    sched = DDPMScheduler(num_train_timesteps=steps,
                          beta_schedule=("squaredcos_cap_v2" if is_contact
                                         else cfg["model"]["beta"]),
                          clip_sample=False, prediction_type="sample")
    lr = args.lr or (cfg["lr"] if is_contact else tr["lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    warmup = 100 if is_contact else tr["warmup_steps"]
    lr_sched = get_scheduler(name="cosine", optimizer=opt, num_warmup_steps=warmup,
                             num_training_steps=len(dl) * epochs)
    # EMA is upstream's on the motion model only; the contact trainer does not use one
    ema = EMAModel(parameters=model.parameters(), power=0.75) if not is_contact else None
    stat = data_util.load_stat_dict(d["stat_dict_path"], device)

    tag = f"_{args.tag}" if args.tag else ""
    out = Path(args.out or (Path(cfg["save_root_dir"]) /
                            f"{cfg['exp_name']}{tag}_{datetime.now():%Y%m%d_%H%M%S}"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    log.info("%s: %d windows, bs %d, %d steps/epoch (%d total), lr %.1e, %.1fM params -> %s",
             args.model, len(ds), bs, len(dl), len(dl) * epochs, lr,
             sum(p.numel() for p in model.parameters()) / 1e6, out)

    use_wandb = (cfg["use_wandb"] if is_contact else tr["use_wandb"]) and not args.no_wandb
    run = None
    if use_wandb:
        import wandb
        project = cfg["wandb_pj_name"] if is_contact else tr["wandb_pj_name"]
        entity = cfg["wandb_entity"] if is_contact else tr["wandb_entity"]
        run = wandb.init(project=project, name=out.name, dir=str(out),
                         config={**cfg, "n_windows": len(ds), "steps_per_epoch": len(dl),
                                 "canonical_frame": "target", "dataset": "taco",
                                 "batch_size_used": bs, "lr_used": lr, "epochs_used": epochs,
                                 "total_steps": len(dl) * epochs},
                         **({"entity": entity} if entity and "credential" not in entity else {}))
        log.info("wandb: %s", getattr(run, "url", "(offline)"))

    def validate() -> float:
        """Mean MSE over the validation split, at fixed noise and fixed timesteps."""
        model.eval()
        gen = torch.Generator(device=device).manual_seed(1234)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for vb in val_dl:
                bv_in = {k: v for k, v in vb.items() if k not in ("aux", "viz")}
                bv = (data_util.preprocess_contact_batch(bv_in, device, stat, True) if is_contact
                      else data_util.preprocess_batch(bv_in, stat, device))
                av = bv["action"]
                tsv = torch.randint(0, steps, (av.shape[0],), device=device, generator=gen).long()
                nv = sched.add_noise(av, torch.randn(av.shape, device=device, generator=gen), tsv)
                if is_contact:
                    pv = model(nv, tsv, obj_feat=bv["obs"]["obj_feat"],
                               global_cond={"curr_global_states": bv["obs"]["curr_global_states"]})
                else:
                    # always condition on contact here; a 50% dropout would make the number a coin
                    # flip rather than a measurement
                    pv = model(nv, tsv, obj_feat=bv["obs"]["object"],
                               contact_cond=bv["obs"]["contact_points"], contact_on_prob=1.0,
                               global_states=bv["obs"]["global_states"])
                tot += torch.nn.functional.mse_loss(pv, av).item() * av.shape[0]
                cnt += av.shape[0]
        model.train()
        return tot / max(cnt, 1)

    if val_dl is not None:
        log.info("validation on %r: %d windows", args.val_split, len(val_dl.dataset))

    hist, best = [], float("inf")
    for epoch in range(epochs):
        model.train()
        t0, acc, n = time.time(), 0.0, 0
        for i, nb in enumerate(dl):
            batch = {k: v for k, v in nb.items() if k not in ("aux", "viz")}
            b = (data_util.preprocess_contact_batch(batch, device, stat, True) if is_contact
                 else data_util.preprocess_batch(batch, stat, device))
            a = b["action"]
            # one timestep PER ITEM, as upstream: a shared draw would put every sample in the
            # batch at the same noise level, and the model silently broadcasts a scalar
            ts = torch.randint(0, steps, (a.shape[0],), device=device).long()
            noisy = sched.add_noise(a, torch.randn_like(a), ts)
            if is_contact:
                pred = model(noisy, ts, obj_feat=b["obs"]["obj_feat"],
                             global_cond={"curr_global_states": b["obs"]["curr_global_states"]})
            else:
                pred = model(noisy, ts, obj_feat=b["obs"]["object"],
                             contact_cond=b["obs"]["contact_points"],
                             contact_on_prob=tr["contact_on_prob"],
                             global_states=b["obs"]["global_states"])
            loss = torch.nn.functional.mse_loss(pred, a)
            loss.backward(); opt.step(); opt.zero_grad(); lr_sched.step()
            if ema is not None:
                ema.step(model.parameters())
            acc += loss.item(); n += 1
            if args.log_every and (i + 1) % args.log_every == 0:
                log.info("  e%03d %4d/%d  loss %.5f  lr %.2e", epoch, i + 1, len(dl), acc / n,
                         lr_sched.get_last_lr()[0])
        mean_loss = acc / max(n, 1)
        hist.append({"epoch": epoch, "loss": mean_loss, "seconds": time.time() - t0,
                     "lr": lr_sched.get_last_lr()[0]})
        log.info("epoch %d/%d  loss %.5f  lr %.2e  %.0fs", epoch + 1, epochs, mean_loss,
                 lr_sched.get_last_lr()[0], hist[-1]["seconds"])
        metrics = {"Train/Loss": mean_loss, "Train/lr": lr_sched.get_last_lr()[0],
                   "Train/epoch_seconds": hist[-1]["seconds"]}
        if val_dl is not None and (epoch + 1) % args.val_every == 0:
            vloss = validate()
            hist[-1]["val_loss"] = vloss
            metrics["Val/Loss"] = vloss
            improved = vloss < best
            log.info("  val %.5f%s", vloss, "  (best)" if improved else "")
            if improved:
                best = vloss
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "ema_stat_dict": ema.state_dict() if ema is not None else None,
                            "val_loss": vloss}, out / "model_best.pth")
        if run is not None:
            run.log(metrics, step=epoch)
        (out / "loss_history.json").write_text(json.dumps(hist, indent=1))
        every = cfg["save_checkpt_epoch"] if is_contact else tr["save_checkpt_epoch"]
        if (epoch + 1) % every == 0 or epoch + 1 == epochs:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "ema_stat_dict": ema.state_dict() if ema is not None else None,
                        "optimizer": opt.state_dict(), "lr_sched": lr_sched.state_dict()},
                       out / f"model_epoch{epoch + 1}.pth")
            log.info("  saved checkpoint at epoch %d", epoch + 1)
    torch.save({"epoch": epochs, "model": model.state_dict(),
                "ema_stat_dict": ema.state_dict() if ema is not None else None,
                "optimizer": opt.state_dict(), "lr_sched": lr_sched.state_dict()},
               out / "model_final.pth")
    if run is not None:
        run.finish()
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()
