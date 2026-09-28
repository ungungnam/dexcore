#!/usr/bin/env python
"""Ask whether the motion model CAN place the recorded hand, or only declines to.

The training loss is an MSE against the recorded trajectory, so a 50 mm sampling error needs an
explanation. There are three candidates and this script separates them.

    conditional-mean probe   Feed the model a GT trajectory buried under noise at timestep t and
                             take ONE forward pass. Because the objective is x0-prediction, that
                             pass is an estimate of E[x0 | x_t, condition]. At t = T the input
                             carries no signal, so the answer is E[x0 | condition] alone -- the
                             best any sampler could do on average from this conditioning. If THAT
                             is already 50 mm from the recording, the object trajectory does not
                             determine the hand and no amount of training fixes it.
                             Reported against two controls: the error of the noisy input itself
                             (has the model done anything?) and the error of the dataset mean pose
                             (is the conditioning worth more than no conditioning?).

    spread (APD)             Sample N times from one condition with different noise. Compare the
                             average distance between samples to the average distance to the
                             recording. Comparable => the recording is one draw among many and the
                             model is right to spread. Much smaller => the samples agree with each
                             other and disagree with the recording, which is a wrong mode, not
                             honest uncertainty.

    oracle-of-N              The closest of the N samples. If that collapses while the mean does
                             not, the distribution covers the recording and only the choice fails.

Everything is conditioned on GROUND TRUTH contact, so the contact stage cannot be blamed.
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

HAND_KP = 100
KP_DIM = HAND_KP * 6  # 100 keypoints x 2 hands x 3 coords


def keypoints_m(a_norm, mean, std):
    """Normalised action -> keypoints in metres, shape [..., 200, 3]."""
    kp = (a_norm * std + mean)[..., :KP_DIM]
    return kp.reshape(*kp.shape[:-1], HAND_KP * 2, 3)


def err_mm(a_hat, a_gt, mean, std):
    d = keypoints_m(a_hat, mean, std) - keypoints_m(a_gt, mean, std)
    return torch.linalg.norm(d, dim=-1).mean(dim=(-1, -2)) * 1000.0  # per batch item


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "test_1"])
    p.add_argument("--out", default="/result/uhnam/dexcore/bimart_taco/diagnosis")
    p.add_argument("--windows", type=int, default=96, help="windows per split")
    p.add_argument("--n-sample", type=int, default=6, help="draws per window for the spread test")
    p.add_argument("--batch-size", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--variant", default="", choices=("", "_fps", "_scene"),
                   help="'_fps' selects the corrected-contact-label configs and store")
    p.add_argument("--store", default=None,
                   help="store subdirectory; defaults to train_store{variant}")
    args = p.parse_args()
    V = args.variant
    STORE = args.store or ("train_store_fps" if V == "_fps" else "train_store")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("diag")

    import pandas as pd
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset

    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.bimart.datasets import TacoMotionDataset
    from utils import data_util

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mcfg = yaml.safe_load(open(f"configs/bimart_taco/train_motion_config{args.variant}.yaml"))
    mm, mname, mep, mval = load_model("motion", args.motion, mcfg, device)
    log.info("motion %s (epoch %d, val %s)", mname, mep, f"{mval:.4f}" if mval else "-")

    T = mcfg["train"]["num_diffusion_iters"]
    mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], device)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False, prediction_type="sample")
    mean, std = mstat["action"]["mean"], mstat["action"]["std"]
    probe_ts = sorted({0, 1, 2, 5, 10, 20, 30, 40, T - 1})

    rows, probes = [], []
    for split in args.splits:
        ds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split,
                              pred_horizon=mcfg["data"]["pred_horizon"],
                              base_frame=mcfg["data"]["base_frame"], store_dir=STORE)
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(ds), min(args.windows, len(ds)), replace=False)

        probe = {t: [] for t in probe_ts}
        noisy_ctl = {t: [] for t in probe_ts}
        to_gt, spread, oracle, meanpose = [], [], [], []

        for s in range(0, len(pick), args.batch_size):
            idx = pick[s:s + args.batch_size]
            b = next(iter(DataLoader(Subset(ds, idx), batch_size=len(idx))))
            nb = data_util.preprocess_batch({k: v for k, v in b.items()
                                            if k not in ("aux", "viz")}, mstat, device)
            a = nb["action"]
            cond = nb["obs"]
            B = a.shape[0]

            # control: predicting the dataset mean pose for every frame (z = 0)
            meanpose.append(err_mm(torch.zeros_like(a), a, mean, std).cpu())

            # ---- conditional-mean probe
            g = torch.Generator(device=device).manual_seed(999)
            noise = torch.randn(a.shape, device=device, generator=g)
            for t in probe_ts:
                ts = torch.full((B,), t, device=device, dtype=torch.long)
                x_t = sched.add_noise(a, noise, ts)
                with torch.no_grad():
                    pr = mm(x_t, ts, obj_feat=cond["object"],
                            contact_cond=cond["contact_points"], contact_on_prob=1.0,
                            global_states=cond["global_states"])
                probe[t].append(err_mm(pr, a, mean, std).cpu())
                noisy_ctl[t].append(err_mm(x_t, a, mean, std).cpu())

            # ---- spread: N full samplings of the same condition
            draws = torch.stack([
                sample(mm, a.shape, sched, device, cond, "motion", T, seed=args.seed + 100 * k)
                for k in range(args.n_sample)])                      # [N,B,T,D]
            e = torch.stack([err_mm(draws[k], a, mean, std) for k in range(args.n_sample)])  # [N,B]
            to_gt.append(e.mean(0).cpu())
            oracle.append(e.min(0).values.cpu())
            pair = [err_mm(draws[i], draws[j], mean, std)
                    for i in range(args.n_sample) for j in range(i + 1, args.n_sample)]
            spread.append(torch.stack(pair).mean(0).cpu())
            log.info("%s  %d/%d windows", split, min(s + args.batch_size, len(pick)), len(pick))

        cat = lambda L: torch.cat(L).numpy()
        rec = {"split": split, "windows": len(ds), "sampled": len(pick),
               "n_sample": args.n_sample,
               "err_to_gt_mm": float(cat(to_gt).mean()),
               "spread_apd_mm": float(cat(spread).mean()),
               "oracle_of_n_mm": float(cat(oracle).mean()),
               "mean_pose_mm": float(cat(meanpose).mean()),
               "cond_mean_mm": float(cat(probe[T - 1]).mean())}
        rec["spread_over_err"] = rec["spread_apd_mm"] / rec["err_to_gt_mm"]
        rows.append(rec)
        for t in probe_ts:
            probes.append({"split": split, "t": t,
                           "model_pred_mm": float(cat(probe[t]).mean()),
                           "noisy_input_mm": float(cat(noisy_ctl[t]).mean())})
        log.info("%s: to_gt %.1f | apd %.1f | oracle %.1f | cond-mean %.1f | mean-pose %.1f mm",
                 split, rec["err_to_gt_mm"], rec["spread_apd_mm"], rec["oracle_of_n_mm"],
                 rec["cond_mean_mm"], rec["mean_pose_mm"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df, dp = pd.DataFrame(rows), pd.DataFrame(probes)
    df.to_csv(out / "spread.csv", index=False)
    dp.to_csv(out / "conditional_mean_probe.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"motion_run": args.motion, "motion_ckpt": mname, "motion_epoch": mep,
         "windows": args.windows, "n_sample": args.n_sample, "seed": args.seed}, indent=2))
    print("\n=== spread ===\n" + df.to_string(index=False))
    print("\n=== conditional-mean probe (mm) ===\n" + dp.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
