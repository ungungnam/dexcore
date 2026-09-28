#!/usr/bin/env python
"""Split the hand error between the two stages by measuring, not by subtracting.

Distances do not add linearly, so `error(own contact) - error(recorded contact)` is not the
contact stage's share -- it depends on whether the two displacements point the same way. This
samples the motion model TWICE from the SAME noise, changing only the contact map:

    x_gt    motion sampled on the RECORDED contact map
    x_pred  motion sampled on the contact model's own prediction, identical seed

Because the seed is shared, `x_pred - x_gt` is exactly what swapping the contact map did to the
hand, with the sampling noise cancelled. Then, per keypoint and frame,

    v_motion  = x_gt   - recorded     what stage 2 gets wrong given a perfect contact map
    v_contact = x_pred - x_gt         what stage 1 costs on top
    v_total   = x_pred - recorded     and v_total = v_motion + v_contact exactly

so the two shares are commensurable, and their mean cosine says whether they reinforce
(cos > 0, distances roughly add) or cancel (cos < 0).
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
KP_DIM = HAND_KP * 6


def kp(a_norm, mean, std):
    v = (a_norm * std + mean)[..., :KP_DIM]
    return v.reshape(*v.shape[:-1], HAND_KP * 2, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/experiments/"
                                        "contact_model_taco_20260923_151838")
    p.add_argument("--motion", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/experiments/"
                                       "motion_model_taco_20260923_151845")
    p.add_argument("--splits", nargs="+", default=["test_1", "test_2", "test_3", "test_4"])
    p.add_argument("--windows", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/attribution")
    p.add_argument("--variant", default="", choices=("", "_fps", "_scene"),
                   help="'_fps' selects the corrected-contact-label configs and store")
    p.add_argument("--store", default=None,
                   help="store subdirectory; defaults to train_store{variant}")
    args = p.parse_args()
    V = args.variant
    STORE = args.store or ("train_store_fps" if V == "_fps" else "train_store")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("attr")

    import pandas as pd
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset

    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from utils import data_util

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ccfg = yaml.safe_load(open(f"configs/bimart_taco/train_contact_config{args.variant}.yaml"))
    mcfg = yaml.safe_load(open(f"configs/bimart_taco/train_motion_config{args.variant}.yaml"))
    cm, cname, *_ = load_model("contact", args.contact, ccfg, device)
    mm, mname, *_ = load_model("motion", args.motion, mcfg, device)
    log.info("contact %s | motion %s", cname, mname)

    T = ccfg["num_timesteps"]
    cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], device)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False, prediction_type="sample")
    mean, std = mstat["action"]["mean"], mstat["action"]["std"]
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]

    rows = []
    for split in args.splits:
        cds = TacoContactDataset(root=ccfg["base_dir"], split=split,
                                 pred_horizon=ccfg["pred_horizon"], base_frame=ccfg["base_frame"], store_dir=STORE)
        mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split,
                                pred_horizon=mcfg["data"]["pred_horizon"],
                                base_frame=mcfg["data"]["base_frame"], store_dir=STORE)
        rng = np.random.default_rng(args.seed)
        pick = rng.choice(len(cds), min(args.windows, len(cds)), replace=False)
        acc = {k: [] for k in ("motion", "contact", "total", "cos")}

        for s in range(0, len(pick), args.batch_size):
            idx = pick[s:s + args.batch_size]
            cb = next(iter(DataLoader(Subset(cds, idx), batch_size=len(idx))))
            mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
            cn = data_util.preprocess_contact_batch(
                {k: v for k, v in cb.items() if k != "aux"}, device, cstat, True)
            mn = data_util.preprocess_batch(
                {k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, device)

            c_hat = sample(cm, cn["action"].shape, sched, device, cn["obs"], "contact",
                           T, args.seed)
            c_hat = (c_hat * c_std + c_mean - mstat["contact_points"]["mean"]) \
                    / mstat["contact_points"]["std"]

            cond_gt = dict(mn["obs"])
            cond_pr = dict(mn["obs"]); cond_pr["contact_points"] = c_hat
            # SAME seed for both, so the only difference is the contact conditioning
            x_gt = sample(mm, mn["action"].shape, sched, device, cond_gt, "motion", T, args.seed)
            x_pr = sample(mm, mn["action"].shape, sched, device, cond_pr, "motion", T, args.seed)

            rec = kp(mn["action"], mean, std)
            v_m = kp(x_gt, mean, std) - rec          # stage 2's own error
            v_c = kp(x_pr, mean, std) - kp(x_gt, mean, std)   # what stage 1 added
            v_t = kp(x_pr, mean, std) - rec

            n_m = torch.linalg.norm(v_m, dim=-1)
            n_c = torch.linalg.norm(v_c, dim=-1)
            cos = (v_m * v_c).sum(-1) / (n_m * n_c).clamp_min(1e-9)
            acc["motion"].append(n_m.mean(dim=(-1, -2)).cpu() * 1000)
            acc["contact"].append(n_c.mean(dim=(-1, -2)).cpu() * 1000)
            acc["total"].append(torch.linalg.norm(v_t, dim=-1).mean(dim=(-1, -2)).cpu() * 1000)
            acc["cos"].append(cos.mean(dim=(-1, -2)).cpu())
            log.info("%s %d/%d", split, min(s + args.batch_size, len(pick)), len(pick))

        cat = lambda k: torch.cat(acc[k]).numpy()
        r = {"split": split, "windows": len(pick),
             "motion_mm": float(cat("motion").mean()),
             "contact_mm": float(cat("contact").mean()),
             "total_mm": float(cat("total").mean()),
             "cos": float(cat("cos").mean())}
        r["share_contact"] = r["contact_mm"] / (r["motion_mm"] + r["contact_mm"])
        rows.append(r)
        log.info("%s: motion %.1f | contact %.1f | total %.1f | cos %.3f",
                 split, r["motion_mm"], r["contact_mm"], r["total_mm"], r["cos"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "stage_attribution.csv", index=False)
    (out / "meta.json").write_text(json.dumps(
        {"contact_run": args.contact, "motion_run": args.motion,
         "windows": args.windows, "seed": args.seed}, indent=2))
    print("\n" + df.round(3).to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
