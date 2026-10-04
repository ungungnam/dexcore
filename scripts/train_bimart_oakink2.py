#!/usr/bin/env python
"""Train BimArt's contact or motion model on the OakInk2 port.

  CUDA_VISIBLE_DEVICES=4 python scripts/train_bimart_oakink2.py --model contact
  CUDA_VISIBLE_DEVICES=5 python scripts/train_bimart_oakink2.py --model motion

The loop is `train_bimart_taco.py`'s, which matches upstream on per-item timesteps, AdamW
(wd 1e-6), cosine warmup 100 stepped per batch, EMA (power 0.75, motion only) and x0 prediction.
What differs from the TACO trainer, each restoring upstream or following the port plan
(review/plan.md, "Restore upstream" and C "Taken with edits"):

    LR HORIZON, upstream's. Motion: `num_training_steps = len(train_dataset) * epochs` -- windows,
    not batches, so the cosine barely decays (train_motion_model.py:49-52). Contact:
    `len(dataloader) * epochs` (contact_prior_util.py:51-56). The TACO trainer used len(dl) for both.

    MODEL SELECTION ON EMA WEIGHTS for the motion model: upstream samples from the EMA copy, so the
    validation loss is measured on it. The contact model has no EMA upstream.

    VALIDATION on TaMF `val` (the authors' split; never test), stride-16 windows, 4 fixed
    timesteps and fixed noise per window, reseeded identically every epoch. Upstream has no
    validation; here it only decides when to stop: patience 10 evaluations without a relative
    improvement of 1e-3 over the best. The 200 epochs stay the schedule horizon.

    THE FIRST EVALUATION IS RUN UNDER A SECOND SEED too, so the noise floor of the validation loss
    is logged before patience is trusted.

Checkpoints go to /ckpt (best and last only, last carrying the optimiser for --resume); the
config, loss history and wandb files go to /result. As in the TACO trainer, a checkpoint's
"model" holds the RAW weights and "ema_stat_dict" the EMA copy: upstream saves the EMA weights as
"model", so load with `eval_bimart_taco.load_model(use_ema=True)`, which copies the EMA in.
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

VAL_SEED, FLOOR_SEED, VAL_TIMESTEPS = 1234, 4321, 4


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("contact", "motion"), required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--epochs", type=int, default=None, help="schedule horizon; default the config's 200")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--tag", default=None, help="appended to the run name")
    p.add_argument("--patience", type=int, default=10, help="evaluations without improvement")
    p.add_argument("--min-rel-improve", type=float, default=1e-3)
    p.add_argument("--resume", default=None, help="a model_last.pth to continue from")
    p.add_argument("--max-steps", type=int, default=None, help="stop each epoch after N steps (smoke)")
    p.add_argument("--max-val-batches", type=int, default=None, help="(smoke)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("train")

    from diffusers import DDPMScheduler
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel
    from torch.utils.data import DataLoader

    from scripts.train_bimart_taco import build
    from src.analysis.bimart.oakink2_data import OakInk2ContactDataset, OakInk2MotionDataset
    from utils import data_util

    cfg_path = args.config or f"{_REPO}/configs/bimart_oakink2/train_{args.model}_config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    is_contact = args.model == "contact"
    d = cfg if is_contact else cfg["data"]
    tr = cfg if is_contact else cfg["train"]
    epochs = args.epochs or tr["num_epochs"]
    bs = cfg["batch_size"] if is_contact else tr["train_batch_size"]
    steps = cfg["num_timesteps"] if is_contact else tr["num_diffusion_iters"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)                                 # as upstream's two trainers
    np.random.seed(0)

    ds_cls = OakInk2ContactDataset if is_contact else OakInk2MotionDataset
    kw = dict(root=d["base_dir"], pred_horizon=d["pred_horizon"], base_frame=d["base_frame"],
              nonpair_m=d["nonpair_m"])
    ds = ds_cls(split=d["split"], **kw)
    # upstream drops the last partial batch for the contact model only
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.workers, drop_last=is_contact,
                    persistent_workers=args.workers > 0, prefetch_factor=6 if args.workers else None)
    val_ds = ds_cls(split="val", **kw)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, drop_last=False)

    model = build(args.model, cfg, device)
    sched = DDPMScheduler(num_train_timesteps=steps,
                          beta_schedule="squaredcos_cap_v2" if is_contact else cfg["model"]["beta"],
                          clip_sample=False, prediction_type="sample")
    lr = tr["lr"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    warmup = 100 if is_contact else tr["warmup_steps"]
    horizon = (len(dl) if is_contact else len(ds)) * epochs          # upstream's, per model
    lr_sched = get_scheduler(name="cosine", optimizer=opt, num_warmup_steps=warmup,
                             num_training_steps=horizon)
    ema = EMAModel(parameters=model.parameters(), power=0.75) if not is_contact else None
    stat = data_util.load_stat_dict(d["stat_dict_path"], device)
    # the statistics must describe exactly the windows trained on
    meta = json.loads((Path(d["base_dir"]) / "stats_summary.json").read_text())
    if (meta["train_windows"], meta["nonpair_m"]) != (len(ds), d["nonpair_m"]):
        raise SystemExit(f"stats were computed on {meta['train_windows']} windows at nonpair "
                         f"{meta['nonpair_m']}, training has {len(ds)} at {d['nonpair_m']}")
    sched_key = {"horizon": horizon, "n_train_windows": len(ds), "epochs": epochs}

    # `ref` is the patience reference: it moves only on an improvement of at least
    # min_rel_improve, so ten small gains in a row still count as progress
    state = {"epoch": 0, "best": float("inf"), "ref": float("inf"), "bad_evals": 0, "hist": [],
             "floor": None, "sched": sched_key}
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        lr_sched.load_state_dict(ck["lr_sched"])
        if ema is not None:
            ema.load_state_dict(ck["ema_stat_dict"])
            ema.to(device)                                # load_state_dict leaves it on the CPU
        state = ck["state"]
        # the cosine's lambda is not in the scheduler's state, so it is rebuilt from these
        if state["sched"] != sched_key:
            raise SystemExit(f"resume changes the LR schedule: {state['sched']} -> {sched_key}")
        name = Path(args.resume).parent.name
    else:
        name = f"{cfg['exp_name']}{'_' + args.tag if args.tag else ''}_{datetime.now():%Y%m%d_%H%M%S}"
    ckpt_dir = Path(cfg["save_root_dir"]) / name
    run_dir = Path(cfg["run_root_dir"]) / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg))
    n_params = sum(q.numel() for q in model.parameters()) / 1e6
    log.info("%s: %d train windows (%d before the non-pair filter), %d val windows, bs %d, "
             "%d steps/epoch, lr %.1e, LR horizon %d steps, %.1fM params -> %s",
             args.model, len(ds), ds.n_windows_before_filter, len(val_ds), bs, len(dl), lr,
             horizon, n_params, ckpt_dir)

    run = None
    if tr["use_wandb"] and not args.no_wandb:
        import wandb
        run = wandb.init(project=tr["wandb_pj_name"], name=name, dir=str(run_dir), id=name, resume="allow",
                         config={**cfg, "dataset": "oakink2", "canonical_frame": "part1",
                                 "n_train_windows": len(ds), "n_val_windows": len(val_ds),
                                 "n_train_windows_before_filter": ds.n_windows_before_filter,
                                 "steps_per_epoch": len(dl), "lr_horizon_steps": horizon,
                                 "params_M": n_params, "ckpt_dir": str(ckpt_dir), "run_dir": str(run_dir),
                                 "patience": args.patience, "min_rel_improve": args.min_rel_improve,
                                 "val_timesteps_per_window": VAL_TIMESTEPS},
                         **({"entity": tr["wandb_entity"]} if tr.get("wandb_entity") else {}))
        log.info("wandb: %s", getattr(run, "url", "(offline)"))

    def forward(b, ts, noisy, contact_on_prob):
        if is_contact:
            return model(noisy, ts, obj_feat=b["obs"]["obj_feat"],
                         global_cond={"curr_global_states": b["obs"]["curr_global_states"]})
        return model(noisy, ts, obj_feat=b["obs"]["object"], contact_cond=b["obs"]["contact_points"],
                     contact_on_prob=contact_on_prob, global_states=b["obs"]["global_states"])

    def normalise(batch):
        batch = {k: v for k, v in batch.items() if k not in ("aux", "viz")}
        return (data_util.preprocess_contact_batch(batch, device, stat, True) if is_contact
                else data_util.preprocess_batch(batch, stat, device))

    def validate(seed: int) -> float:
        """Mean MSE over val windows x 4 timesteps, fixed noise; EMA weights for the motion model."""
        if ema is not None:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        model.eval()
        gen = torch.Generator(device=device).manual_seed(seed)
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i, vb in enumerate(val_dl):
                if args.max_val_batches and i >= args.max_val_batches:
                    break
                b = normalise(vb)
                a = b["action"]
                for _ in range(VAL_TIMESTEPS):
                    ts = torch.randint(0, steps, (a.shape[0],), device=device, generator=gen).long()
                    noisy = sched.add_noise(a, torch.randn(a.shape, device=device, generator=gen), ts)
                    # always condition on contact: a 50% dropout would make the number a coin flip
                    pred = forward(b, ts, noisy, 1.0)
                    tot += torch.nn.functional.mse_loss(pred, a).item() * a.shape[0]
                    cnt += a.shape[0]
        model.train()
        if ema is not None:
            ema.restore(model.parameters())
        return tot / max(cnt, 1)

    def save(path, extra=None):
        ck = {"epoch": state["epoch"] - 1, "model": model.state_dict(),
              "ema_stat_dict": ema.state_dict() if ema is not None else None, **(extra or {})}
        tmp = path.with_suffix(".tmp")
        torch.save(ck, tmp)
        tmp.replace(path)

    while state["epoch"] < epochs:
        epoch = state["epoch"]
        model.train()
        t0, acc, n = time.time(), 0.0, 0
        for i, nb in enumerate(dl):
            if args.max_steps and i >= args.max_steps:
                break
            b = normalise(nb)
            a = b["action"]
            ts = torch.randint(0, steps, (a.shape[0],), device=device).long()   # one per item
            pred = forward(b, ts, sched.add_noise(a, torch.randn_like(a), ts), tr["contact_on_prob"]
                           if not is_contact else None)
            loss = torch.nn.functional.mse_loss(pred, a)
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite training loss at epoch {epoch} step {i}")
            loss.backward(); opt.step(); opt.zero_grad(); lr_sched.step()
            if ema is not None:
                ema.step(model.parameters())
            acc += loss.item(); n += 1
            if args.log_every and (i + 1) % args.log_every == 0:
                log.info("  e%03d %5d/%d  loss %.5f  lr %.2e", epoch, i + 1, len(dl), acc / n,
                         lr_sched.get_last_lr()[0])
        h = {"epoch": epoch, "loss": acc / max(n, 1), "seconds": time.time() - t0,
             "lr": lr_sched.get_last_lr()[0]}
        h["val_loss"] = validate(VAL_SEED)
        if state["floor"] is None:
            state["floor"] = {"seed_1234": h["val_loss"], "seed_4321": validate(FLOOR_SEED)}
            log.info("  val noise floor at the first evaluation: %s", state["floor"])
            if run is not None:
                run.summary["val_noise_floor"] = state["floor"]
        if not np.isfinite(h["val_loss"]):
            raise SystemExit(f"non-finite validation loss at epoch {epoch}")
        if h["val_loss"] < state["ref"] * (1 - args.min_rel_improve):
            state["ref"] = h["val_loss"]
            state["bad_evals"] = 0
        else:
            state["bad_evals"] += 1
        if h["val_loss"] < state["best"]:
            state["best"] = h["val_loss"]
            state["best_epoch"] = epoch
        state["hist"].append(h)
        state["epoch"] = epoch + 1
        log.info("epoch %d/%d  loss %.5f  val %.5f%s  lr %.2e  %.0fs  (no-improve %d/%d)", epoch + 1, epochs,
                 h["loss"], h["val_loss"], "  (best)" if state.get("best_epoch") == epoch else "",
                 h["lr"], h["seconds"], state["bad_evals"], args.patience)
        if state.get("best_epoch") == epoch:
            save(ckpt_dir / "model_best.pth", {"val_loss": h["val_loss"]})
        save(ckpt_dir / "model_last.pth", {"optimizer": opt.state_dict(),
                                           "lr_sched": lr_sched.state_dict(), "state": state})
        if run is not None:
            run.log({"train/loss": h["loss"], "train/lr": h["lr"], "train/epoch": epoch,
                     "train/epoch_seconds": h["seconds"], "test/val_loss": h["val_loss"],
                     "test/best_val_loss": state["best"]}, step=epoch)
        (run_dir / "loss_history.json").write_text(json.dumps(state, indent=1, default=float))
        if state["bad_evals"] >= args.patience:
            log.info("stopping: %d evaluations without a %.1e relative improvement (best %.5f at epoch %d)",
                     args.patience, args.min_rel_improve, state["best"], state.get("best_epoch", -1) + 1)
            break
    if run is not None:
        run.summary.update({"best_val_loss": state["best"], "best_epoch": state.get("best_epoch"),
                            "epochs_run": state["epoch"], "stopped_early": state["epoch"] < epochs})
        run.finish()
    log.info("done: best val %.5f at epoch %d -> %s", state["best"], state.get("best_epoch", -1) + 1, ckpt_dir)


if __name__ == "__main__":
    main()
