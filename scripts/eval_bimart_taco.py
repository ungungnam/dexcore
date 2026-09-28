#!/usr/bin/env python
"""Evaluate the trained TACO models on each of the four official test splits.

  CUDA_VISIBLE_DEVICES=5 python scripts/eval_bimart_taco.py --contact <dir> --motion <dir>

TWO MEASUREMENTS, because they answer different questions:

    denoising loss   the training objective on held-out data -- one forward pass, fixed noise and
                     fixed timesteps so the splits are compared on identical conditions. Cheap
                     enough to run on every window.
    sampling error   the full 50-step DDPM loop from pure noise, which is what inference actually
                     does, scored in MILLIMETRES after unnormalising. A normalised MSE cannot say
                     whether a hand is in the right place; a millimetre can.

The motion model is scored twice, conditioned on GROUND TRUTH contact and on the contact model's
own prediction. The gap between them is what the two-stage pipeline costs, and it is invisible if
only one is reported.

THE SPLITS ARE GRADED, and reading them together is the point: `test_1` shares every triplet and
mesh with training, `test_2` holds out the tool meshes, `test_3` the triplets, `test_4` both. The
drop from test_1 to test_2 is the transfer question this whole port exists to ask.
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
from pathlib import Path

import numpy as np
import torch
import yaml

SPLITS = ("test_1", "test_2", "test_3", "test_4")
HAND_KP = 100


def load_model(kind, run_dir, cfg, device, prefer="model_best.pth", use_ema=True):
    """Load a checkpoint, preferring the EMA weights.

    Upstream trains an EMA copy of the motion model and samples from `self.ema_model`, never from
    the raw weights. The checkpoints here store both, so taking `ck["model"]` silently evaluates a
    different model than BimArt does. The contact trainer has no EMA, so there is nothing to
    prefer there and `ema_stat_dict` is None.
    """
    from scripts.train_bimart_taco import build

    path = Path(run_dir) / prefer
    if not path.exists():
        path = Path(run_dir) / "model_final.pth"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build(kind, cfg, device)
    model.load_state_dict(ck["model"])
    tag = path.name
    if use_ema and ck.get("ema_stat_dict") is not None:
        from diffusers.training_utils import EMAModel

        ema = EMAModel(parameters=model.parameters(), power=0.75)
        ema.load_state_dict(ck["ema_stat_dict"])
        ema.copy_to(model.parameters())
        tag += "+ema"
    model.eval()
    return model, tag, ck.get("epoch", -1) + 1, ck.get("val_loss")


CFG_GUIDE_STRENGTH = 0.5   # upstream config_files/bimart_inference.yaml


def sample(model, shape, sched, device, cond, kind, steps=50, seed=0, guide=None,
           cfg_strength=CFG_GUIDE_STRENGTH):
    """The full DDPM loop from noise -- what inference does.

    `guide`, when given, is applied to the model's prediction at every step, which is where BimArt
    puts it: the correction has to act on each partially denoised estimate, not once at the end.

    The motion model is sampled with CLASSIFIER-FREE GUIDANCE, as upstream does. Training drops the
    contact conditioning half the time (`contact_on_prob: 0.5`) precisely so that an unconditional
    branch exists to extrapolate away from:

        pred = (1 + s) * pred_with_contact - s * pred_without_contact

    Running only the conditional pass wastes that training and leaves the contact map underused.
    The order matches upstream: guidance acts on the conditional prediction, then the
    extrapolation. The contact model has no such dropout, so it is sampled plainly.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(shape, device=device, generator=g)
    sched.set_timesteps(steps)
    for t in sched.timesteps:
        ts = t.to(device).reshape(1)
        with torch.no_grad():
            if kind == "contact":
                pred = model(x, ts, obj_feat=cond["obj_feat"],
                             global_cond={"curr_global_states": cond["curr_global_states"]})
            else:
                pred = model(x, ts, obj_feat=cond["object"], contact_cond=cond["contact_points"],
                             contact_on_prob=1.0, global_states=cond["global_states"])
        if guide is not None:
            pred = guide.step(pred)
        if kind != "contact" and cfg_strength:
            with torch.no_grad():
                uncond = model(x, ts, obj_feat=cond["object"], contact_on_prob=0.0,
                               global_states=cond["global_states"])
            pred = (1.0 + cfg_strength) * pred - cfg_strength * uncond
        with torch.no_grad():
            x = sched.step(pred, t, x).prev_sample
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--out", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/evaluation")
    p.add_argument("--sample-windows", type=int, default=128,
                   help="windows per split for the 50-step sampling metric")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--no-guidance", action="store_true",
                   help="skip the contact-guidance stage, to measure what it is worth")
    p.add_argument("--no-ema", action="store_true",
                   help="sample from the raw weights instead of the EMA copy upstream uses")
    p.add_argument("--cfg-strength", type=float, default=CFG_GUIDE_STRENGTH,
                   help="classifier-free guidance strength; 0 disables it")
    p.add_argument("--tag", default="", help="suffix for the output directory")
    p.add_argument("--variant", default="", choices=("", "_fps", "_scene"),
                   help="'_fps' selects the corrected-contact-label configs and store")
    p.add_argument("--store", default=None,
                   help="store subdirectory; defaults to train_store{variant}")
    args = p.parse_args()
    V = args.variant
    STORE = args.store or ("train_store_fps" if V == "_fps" else "train_store")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("eval")

    import pandas as pd
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset

    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from src.analysis.bimart.guidance import ContactGuidance, sparse_object_vertices
    from utils import data_util

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ccfg = yaml.safe_load(open(f"configs/bimart_taco/train_contact_config{args.variant}.yaml"))
    mcfg = yaml.safe_load(open(f"configs/bimart_taco/train_motion_config{args.variant}.yaml"))
    cm, cname, cep, cval = load_model("contact", args.contact, ccfg, device,
                                      use_ema=not args.no_ema)
    mm, mname, mep, mval = load_model("motion", args.motion, mcfg, device,
                                      use_ema=not args.no_ema)
    log.info("sampling with cfg_strength=%.2f, ema=%s", args.cfg_strength, not args.no_ema)
    log.info("contact %s (epoch %d, val %.4f) | motion %s (epoch %d, val %.4f)",
             cname, cep, cval or -1, mname, mep, mval or -1)

    cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], device)
    sched = DDPMScheduler(num_train_timesteps=ccfg["num_timesteps"],
                          beta_schedule="squaredcos_cap_v2", clip_sample=False,
                          prediction_type="sample")
    a_mean = mstat["action"]["mean"]
    a_std = mstat["action"]["std"]
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]

    rows = []
    for split in SPLITS:
        cds = TacoContactDataset(root=ccfg["base_dir"], split=split,
                                 pred_horizon=ccfg["pred_horizon"], base_frame=ccfg["base_frame"], store_dir=STORE)
        mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split,
                                pred_horizon=mcfg["data"]["pred_horizon"],
                                base_frame=mcfg["data"]["base_frame"], store_dir=STORE)
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(cds), min(args.sample_windows, len(cds)), replace=False)
        rec = {"split": split, "windows": len(cds), "sampled": len(pick)}

        # ---- denoising loss, every window, fixed noise and timesteps
        for kind, ds, model, stat in (("contact", cds, cm, cstat), ("motion", mds, mm, mstat)):
            g = torch.Generator(device=device).manual_seed(1234)
            tot, n = 0.0, 0
            for b in DataLoader(ds, batch_size=args.batch_size, num_workers=2):
                bi = {k: v for k, v in b.items() if k not in ("aux", "viz")}
                nb = (data_util.preprocess_contact_batch(bi, device, stat, True) if kind == "contact"
                      else data_util.preprocess_batch(bi, stat, device))
                a = nb["action"]
                ts = torch.randint(0, ccfg["num_timesteps"], (a.shape[0],), device=device,
                                   generator=g).long()
                noisy = sched.add_noise(a, torch.randn(a.shape, device=device, generator=g), ts)
                with torch.no_grad():
                    pr = (model(noisy, ts, obj_feat=nb["obs"]["obj_feat"],
                                global_cond={"curr_global_states": nb["obs"]["curr_global_states"]})
                          if kind == "contact" else
                          model(noisy, ts, obj_feat=nb["obs"]["object"],
                                contact_cond=nb["obs"]["contact_points"], contact_on_prob=1.0,
                                global_states=nb["obs"]["global_states"]))
                tot += torch.nn.functional.mse_loss(pr, a).item() * a.shape[0]
                n += a.shape[0]
            rec[f"{kind}_denoise_mse"] = tot / max(n, 1)

        # ---- full sampling, on the subsample, scored in millimetres
        cerr, merr_gt, merr_pred, merr_guided = [], [], [], []
        for s in range(0, len(pick), args.batch_size):
            idx = pick[s:s + args.batch_size]
            cb = next(iter(DataLoader(Subset(cds, idx), batch_size=len(idx))))
            mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
            # the canonical object vertices guidance measures contact against
            obj_v = None
            if not args.no_guidance:
                obj_v = torch.stack([
                    sparse_object_vertices(mcfg["data"]["base_dir"],
                                           mds.index.iloc[mds.windows[i]["file_idx"]]["sequence_id"],
                                           mds.windows[i]["start"], mcfg["data"]["pred_horizon"],
                                           device)
                    for i in idx])
            cn = data_util.preprocess_contact_batch(
                {k: v for k, v in cb.items() if k != "aux"}, device, cstat, True)
            mn = data_util.preprocess_batch(
                {k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, device)

            # stage 1: contact from the object alone
            cs_hat = sample(cm, cn["action"].shape, sched, device, cn["obs"], "contact",
                            ccfg["num_timesteps"], args.seed)
            cerr.append(((cs_hat - cn["action"]) * c_std).abs().mean().item())
            # The contact model speaks in ITS normalisation; the motion model expects contact maps
            # in the motion stats' `contact_points` space. Upstream converts between the two
            # (postprocess_oneshot_contact) and so must this. The two happen to be identical for
            # this preprocessing run, making it a no-op -- but only by coincidence, and a silent
            # one if either stat file is ever regenerated on its own.
            cs_hat = (cs_hat * c_std + c_mean - mstat["contact_points"]["mean"]) \
                     / mstat["contact_points"]["std"]

            # stage 2, twice: ground-truth contact, then the stage-1 prediction
            kp_gt = (mn["action"] * a_std + a_mean)[..., : HAND_KP * 6]

            def score(a_hat):
                kp_hat = (a_hat * a_std + a_mean)[..., : HAND_KP * 6]
                d = (kp_hat - kp_gt).reshape(*kp_hat.shape[:-1], HAND_KP * 2, 3)
                return torch.linalg.norm(d, dim=-1).mean().item() * 1000.0

            for cond_contact, bucket, guided in ((mn["obs"]["contact_points"], merr_gt, False),
                                                 (cs_hat, merr_pred, False),
                                                 (cs_hat, merr_guided, True)):
                if guided and args.no_guidance:
                    continue
                cond = dict(mn["obs"]); cond["contact_points"] = cond_contact
                guide = (ContactGuidance(obj_v, cond_contact, mstat,
                                         guidance_scale=args.guidance_scale)
                         if guided else None)
                bucket.append(score(sample(mm, mn["action"].shape, sched, device, cond, "motion",
                                           ccfg["num_timesteps"], args.seed, guide=guide,
                                           cfg_strength=args.cfg_strength)))

        rec["contact_sample_mae_m"] = float(np.mean(cerr))
        rec["motion_kp_err_mm_gt_contact"] = float(np.mean(merr_gt))
        rec["motion_kp_err_mm_pred_contact"] = float(np.mean(merr_pred))
        if merr_guided:
            rec["motion_kp_err_mm_guided"] = float(np.mean(merr_guided))
        rows.append(rec)
        log.info("%s: contact mse %.4f | kp err  %.1f (gt) / %.1f (pred) / %s mm (guided)",
                 split, rec["contact_denoise_mse"], rec["motion_kp_err_mm_gt_contact"],
                 rec["motion_kp_err_mm_pred_contact"],
                 f"{rec['motion_kp_err_mm_guided']:.1f}" if merr_guided else "-")

    out = Path(args.out + args.tag); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "test_split_results.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"contact_run": args.contact, "motion_run": args.motion,
         "contact_ckpt": cname, "contact_epoch": cep, "motion_ckpt": mname, "motion_epoch": mep,
         "sample_windows": args.sample_windows, "seed": args.seed,
         "cfg_strength": args.cfg_strength, "ema": not args.no_ema}, indent=2))
    print("\n" + df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
