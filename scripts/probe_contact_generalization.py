#!/usr/bin/env python
"""Per-window, per-frame, per-hand error decomposition for EVERY window of a split.

Built for one question: WHY does the contact stage generalise worse than the motion stage? The
candidate explanations make different predictions, and separating them needs the error broken
down finer than a split mean:

    by novelty     which of verb / tool / target / mesh / triplet the window has never seen
    by frame       does the error grow through the 64-frame window (temporal modelling)
    by component   a whole-hand translation (the hand is in the wrong place) versus the residual
                   after removing it (the hand is shaped wrong) -- a "strategy" error is the second
    by drift       how much the recorded contact map actually changes inside the window

Same construction as attribute_bimart_taco.py: the motion model is sampled twice from ONE seed,
on the recorded contact map and on the contact model's own prediction, so their difference is the
contact stage's contribution with the sampling noise cancelled.

One row per (window, hand, frame). Distances in millimetres.
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
C_HALF = 1024  # contact map: left hand [:1024], right hand [1024:]; each = tool 512 + target 512


def kp(a_norm, mean, std):
    v = (a_norm * std + mean)[..., :KP_DIM]
    return v.reshape(*v.shape[:-1], HAND_KP * 2, 3)


def decompose(v):
    """v: [B,T,K,3] -> per-frame mean norm, whole-hand translation norm, residual mean norm."""
    n = torch.linalg.norm(v, dim=-1).mean(-1)                 # [B,T]
    mu = v.mean(-2, keepdim=True)                              # [B,T,1,3]
    trans = torch.linalg.norm(mu.squeeze(-2), dim=-1)          # [B,T]
    resid = torch.linalg.norm(v - mu, dim=-1).mean(-1)         # [B,T]
    return n, trans, resid


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/experiments/"
                                        "contact_model_taco_20260923_151838")
    p.add_argument("--motion", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/experiments/"
                                       "motion_model_taco_20260923_151845")
    p.add_argument("--split", required=True)
    p.add_argument("--windows", type=int, default=0, help="0 = every window in the split")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/contact_probe")
    p.add_argument("--variant", default="", choices=("", "_fps", "_scene"),
                   help="'_fps' selects the corrected-contact-label configs and store")
    p.add_argument("--store", default=None,
                   help="store subdirectory; defaults to train_store{variant}")
    args = p.parse_args()
    V = args.variant
    STORE = args.store or ("train_store_fps" if V == "_fps" else "train_store")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("probe")

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
    T = ccfg["num_timesteps"]
    cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], device)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False, prediction_type="sample")
    mean, std = mstat["action"]["mean"], mstat["action"]["std"]
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]
    cp_mean, cp_std = mstat["contact_points"]["mean"], mstat["contact_points"]["std"]

    split = args.split
    cds = TacoContactDataset(root=ccfg["base_dir"], split=split,
                             pred_horizon=ccfg["pred_horizon"], base_frame=ccfg["base_frame"], store_dir=STORE)
    mds = TacoMotionDataset(root=mcfg["data"]["base_dir"], split=split,
                            pred_horizon=mcfg["data"]["pred_horizon"],
                            base_frame=mcfg["data"]["base_frame"], store_dir=STORE)
    n_all = len(cds)
    if args.windows and args.windows < n_all:
        pick = np.random.default_rng(args.seed).choice(n_all, args.windows, replace=False)
    else:
        pick = np.arange(n_all)
    log.info("%s: %d of %d windows", split, len(pick), n_all)

    frames = np.arange(mcfg["data"]["pred_horizon"])
    rows = []
    for s in range(0, len(pick), args.batch_size):
        idx = pick[s:s + args.batch_size]
        cb = next(iter(DataLoader(Subset(cds, idx), batch_size=len(idx))))
        mb = next(iter(DataLoader(Subset(mds, idx), batch_size=len(idx))))
        cn = data_util.preprocess_contact_batch(
            {k: v for k, v in cb.items() if k != "aux"}, device, cstat, True)
        mn = data_util.preprocess_batch(
            {k: v for k, v in mb.items() if k not in ("aux", "viz")}, mstat, device)

        c_hat = sample(cm, cn["action"].shape, sched, device, cn["obs"], "contact", T, args.seed)
        c_hat = (c_hat * c_std + c_mean - cp_mean) / cp_std
        c_gt = mn["obs"]["contact_points"]

        cond_gt = dict(mn["obs"])
        cond_pr = dict(mn["obs"]); cond_pr["contact_points"] = c_hat
        x_gt = sample(mm, mn["action"].shape, sched, device, cond_gt, "motion", T, args.seed)
        x_pr = sample(mm, mn["action"].shape, sched, device, cond_pr, "motion", T, args.seed)

        rec = kp(mn["action"], mean, std)
        K_gt, K_pr = kp(x_gt, mean, std), kp(x_pr, mean, std)
        v_m, v_c, v_t = K_gt - rec, K_pr - K_gt, K_pr - rec

        # contact maps back to metres, per hand
        cg_m = (c_gt * cp_std + cp_mean)
        ch_m = (c_hat * cp_std + cp_mean)

        B = rec.shape[0]
        for h, (sl, csl) in enumerate(((slice(0, HAND_KP), slice(0, C_HALF)),
                                       (slice(HAND_KP, None), slice(C_HALF, None)))):
            n_m, tr_m, re_m = decompose(v_m[:, :, sl])
            n_c, tr_c, re_c = decompose(v_c[:, :, sl])
            n_t, _, _ = decompose(v_t[:, :, sl])
            a, b = v_m[:, :, sl], v_c[:, :, sl]
            cos = ((a * b).sum(-1) / (torch.linalg.norm(a, dim=-1) *
                                      torch.linalg.norm(b, dim=-1)).clamp_min(1e-9)).mean(-1)
            # translation-only cosine: do the two whole-hand shifts point the same way?
            am, bm = a.mean(-2), b.mean(-2)
            cos_tr = (am * bm).sum(-1) / (torch.linalg.norm(am, dim=-1) *
                                          torch.linalg.norm(bm, dim=-1)).clamp_min(1e-9)
            gmin = cg_m[:, :, csl].min(-1).values                          # [B,T] metres
            drift = (cg_m[:, :, csl] - cg_m[:, :1, csl]).abs().mean(-1)    # vs window start
            cerr = (ch_m[:, :, csl] - cg_m[:, :, csl]).abs().mean(-1)      # contact-map error
            # is the predicted map even touching where the recorded one touches?
            touch_gt = (cg_m[:, :, csl] < 0.01)
            touch_pr = (ch_m[:, :, csl] < 0.01)
            inter = (touch_gt & touch_pr).sum(-1).float()
            union = (touch_gt | touch_pr).sum(-1).float().clamp_min(1)
            iou = inter / union
            cols = {"motion_mm": n_m, "motion_trans_mm": tr_m, "motion_resid_mm": re_m,
                    "contact_mm": n_c, "contact_trans_mm": tr_c, "contact_resid_mm": re_c,
                    "total_mm": n_t, "cos": cos, "cos_trans": cos_tr,
                    "gt_min_dist_mm": gmin * 1000, "gt_contact_drift_mm": drift * 1000,
                    "contact_map_err_mm": cerr * 1000, "touch_iou": iou,
                    "gt_touch_frac": touch_gt.float().mean(-1)}
            cols = {k: (v * (1000 if k.endswith("_mm") and not k.startswith(("gt_", "contact_map"))
                             else 1)).cpu().numpy() for k, v in cols.items()}
            for bi in range(B):
                w = mds.windows[int(idx[bi])]
                row = mds.index.iloc[w["file_idx"]]
                meta = {"sequence_id": row["sequence_id"], "start": int(w["start"]),
                        "triplet": row["triplet"], "verb": row["verb"],
                        "tool_cat": row["tool_cat"], "target_cat": row["target_cat"],
                        "tool_mesh": int(row["tool_mesh"]), "target_mesh": int(row["target_mesh"])}
                for t in frames:
                    rows.append({"split": split, "window": int(idx[bi]), "hand": "LR"[h],
                                "frame": int(t), **meta,
                                **{k: float(v[bi, t]) for k, v in cols.items()}})
        log.info("%s %d/%d", split, min(s + args.batch_size, len(pick)), len(pick))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = out / f"{split}.parquet"
    try:
        df.to_parquet(path, index=False)
    except Exception:
        path = out / f"{split}.csv"; df.to_csv(path, index=False)
    (out / f"{split}_meta.json").write_text(json.dumps(
        {"contact_run": args.contact, "motion_run": args.motion, "contact_ckpt": cname,
         "motion_ckpt": mname, "windows": int(len(pick)), "of": int(n_all), "seed": args.seed},
        indent=2))
    g = df.groupby("hand")[["motion_mm", "contact_mm", "total_mm", "contact_trans_mm",
                            "contact_resid_mm", "cos"]].mean()
    print("\n" + g.round(2).to_string())
    print(f"\n{len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
