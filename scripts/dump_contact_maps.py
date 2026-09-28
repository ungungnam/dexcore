#!/usr/bin/env python
"""Save the contact model's predicted map, and the recorded one, for every window of a split.

The per-window probe kept only summary statistics of the predicted contact map. Deciding whether a
wrong prediction is a *different plausible grasp region* (a strategy error) or *nowhere anyone
grasps* (a localisation error) needs the maps themselves, compared against every recorded grasp of
the same object in training. Arrays are float16 in metres; [N, 64, 2048] with the store's layout
(left hand [:1024] = tool 512 + target 512, right hand [1024:]).
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
import logging
from pathlib import Path

import numpy as np
import torch
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--contact", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label/experiments/"
                                        "contact_model_taco_20260923_151838")
    p.add_argument("--split", required=True)
    p.add_argument("--windows", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
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
    log = logging.getLogger("dump")

    import pandas as pd
    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset

    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.bimart.datasets import TacoContactDataset
    from utils import data_util

    device = "cuda"
    ccfg = yaml.safe_load(open(f"configs/bimart_taco/train_contact_config{args.variant}.yaml"))
    cm, cname, *_ = load_model("contact", args.contact, ccfg, device)
    T = ccfg["num_timesteps"]
    cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    sched = DDPMScheduler(num_train_timesteps=T, beta_schedule="squaredcos_cap_v2",
                          clip_sample=False, prediction_type="sample")
    c_mean, c_std = cstat["action"]["mean"], cstat["action"]["std"]

    ds = TacoContactDataset(root=ccfg["base_dir"], split=args.split,
                            pred_horizon=ccfg["pred_horizon"], base_frame=ccfg["base_frame"], store_dir=STORE)
    n = len(ds)
    pick = (np.random.default_rng(args.seed).choice(n, args.windows, replace=False)
            if args.windows and args.windows < n else np.arange(n))
    log.info("%s: %d of %d windows", args.split, len(pick), n)

    preds, gts, meta = [], [], []
    for s in range(0, len(pick), args.batch_size):
        idx = pick[s:s + args.batch_size]
        b = next(iter(DataLoader(Subset(ds, idx), batch_size=len(idx))))
        nb = data_util.preprocess_contact_batch({k: v for k, v in b.items() if k != "aux"},
                                                device, cstat, True)
        c_hat = sample(cm, nb["action"].shape, sched, device, nb["obs"], "contact", T, args.seed)
        preds.append(((c_hat * c_std + c_mean)).half().cpu().numpy())
        gts.append(((nb["action"] * c_std + c_mean)).half().cpu().numpy())
        for bi in range(len(idx)):
            w = ds.windows[int(idx[bi])]; row = ds.index.iloc[w["file_idx"]]
            meta.append({"window": int(idx[bi]), "sequence_id": row["sequence_id"],
                         "start": int(w["start"]), "triplet": row["triplet"], "verb": row["verb"],
                         "tool_cat": row["tool_cat"], "target_cat": row["target_cat"],
                         "tool_mesh": int(row["tool_mesh"]), "target_mesh": int(row["target_mesh"])})
        log.info("%s %d/%d", args.split, min(s + args.batch_size, len(pick)), len(pick))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{args.split}_contact_maps.npz", pred=np.concatenate(preds),
             gt=np.concatenate(gts))
    pd.DataFrame(meta).to_csv(out / f"{args.split}_contact_maps_index.csv", index=False)
    print(f"{args.split}: pred/gt {np.concatenate(preds).shape} -> {out}")


if __name__ == "__main__":
    main()
