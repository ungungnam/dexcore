#!/usr/bin/env python
"""Does the contact prior fail on `grab` because it saw too little of it, or because it cannot?

Two explanations for the contact prior's much larger error on `grab` windows survived the earlier
diagnosis, and they are confounded: `grab` is 20% of the training data AND occupies a different
region of the conditioning space. Training-set performance separates them, because a model that has
memorised a window has certainly seen enough of it:

    train `grab` accurate, test `grab` poor   ->  a data / generalisation problem, fixable by
                                                  rebalancing or more data
    train `grab` poor as well                 ->  the conditioning cannot determine the answer.
                                                  Where a hand grips a carried object is not a
                                                  function of object geometry, so no amount of data
                                                  closes it.

`use` windows are measured the same way as a control: whatever gap train-vs-test shows for `use` is
the generalisation gap this model has in general, and `grab`'s gap has to be read against it.

Runs on ObjectContactData -- the contact model's own pipeline -- so the motion->contact
renormalisation bridge is not a confound here.

  python scripts/viz_bimart_train_vs_test.py --categories laptop --per-group 10

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/train_vs_test/<run>/plots.
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

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("bimart_train_vs_test")
GROUPS = [("train", "grab"), ("train", "use"), ("test", "grab"), ("test", "use")]


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def kind(name):
    return "grab" if "_grab_" in name else "use"


def fig_train_vs_test(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
    labels = [f"{s}\n{k}" for s, k in GROUPS]
    colour = ["#d62728" if k == "grab" else "#2ca02c" for _, k in GROUPS]
    data = [[r["mse"] for r in rows if (r["split"], r["kind"]) == g] for g in GROUPS]

    ax = axes[0]
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colour):
        patch.set_facecolor(c); patch.set_alpha(.55)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.uniform(-.08, .08, len(d)), d,
                   s=18, color="#333333", alpha=.7, zorder=3)
    ax.set_ylabel("contact MSE (normalised)")
    ax.set_title("the decisive comparison", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    means = [float(np.mean(d)) if d else np.nan for d in data]
    ax.bar(range(len(GROUPS)), means, .6, color=colour)
    for i, m in enumerate(means):
        ax.text(i, m, f" {m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean contact MSE")
    gap_g = means[2] / means[0] if means[0] else np.nan
    gap_u = means[3] / means[1] if means[1] else np.nan
    ax.set_title(f"generalisation gap:  grab x{gap_g:.1f}   use x{gap_u:.1f}", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    for g, c in zip(GROUPS, colour):
        sel = [r for r in rows if (r["split"], r["kind"]) == g]
        ax.scatter([r["gt_absz"] for r in sel], [r["mse"] for r in sel], s=34, color=c,
                   marker="o" if g[0] == "train" else "^",
                   label=f"{g[0]} {g[1]}", alpha=.8)
    ax.set_xlabel("how atypical the GT contact is  (mean |z|)")
    ax.set_ylabel("contact MSE (normalised)")
    ax.set_title("circles = train, triangles = test", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25)

    fig.suptitle("Has the contact prior seen too little `grab`, or can it not represent it?",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--per-group", type=int, default=10)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--config", default="config_files/contact_inference.yaml")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(BIMART_ROOT)
    _sys.path.insert(0, str(BIMART_ROOT))

    from utils import data_util, yaml_util
    from dataset import contact_data
    from contact_prior.contact_prior_util import load_contact_module, load_noise_scheduler

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    ccfg = yaml_util.load_yaml(args.config)
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/train_vs_test" / run
    (out / "plots").mkdir(parents=True, exist_ok=True)
    LOG.info("writing to %s  (GPU %d)", out, args.gpu)

    cats = set(args.categories)

    class SubContact(contact_data.ObjectContactData):
        def __get_file_paths__(self):
            super().__get_file_paths__()
            for s in ("train", "test"):
                self.data_files[s]["process_files"] = [
                    p for p in self.data_files[s]["process_files"] if Path(p).parts[-3] in cats]

    stat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    model, _, _ = load_contact_module(ccfg, device)
    model.eval()
    sched = load_noise_scheduler(ccfg)

    rows = []
    for split in ("train", "test"):
        ds = SubContact(split=split, base_dir=ccfg["base_dir"], end_frame=ccfg["end_frame"],
                        pred_horizon=ccfg["pred_horizon"], return_aux_info=True)
        LOG.info("%s split: %d files, %d windows", split,
                 len(ds.data_files[split]["process_files"]), len(ds.indices))
        taken = {"grab": 0, "use": 0}
        for i in range(len(ds)):
            k = kind(ds.indices[i]["filename"])
            if taken[k] >= args.per_group:
                if all(v >= args.per_group for v in taken.values()):
                    break
                continue
            item = ds[i]
            batch = {"action": torch.from_numpy(np.asarray(item["action"]))[None],
                     "obs": {k2: torch.from_numpy(np.asarray(v2))[None]
                             for k2, v2 in item["obs"].items()},
                     "aux": {}}
            nb = data_util.preprocess_contact_batch(batch, device, stat, ccfg["normalize_data"])
            gt_z = nb["action"][0].cpu().numpy()
            x = torch.randn(nb["action"].shape, device=device).float()
            sched.set_timesteps(ccfg["num_timesteps"])
            with torch.no_grad():
                for t in sched.timesteps:
                    x0 = model(sample=x, timestep=t, obj_feat=nb["obs"]["obj_feat"],
                               global_cond=nb["obs"])
                    x = sched.step(model_output=x0, timestep=t, sample=x).prev_sample
            pred_z = x[0].cpu().numpy()
            rows.append({"split": split, "kind": k, "filename": item["aux"]["filename"],
                         "mse": float(((pred_z - gt_z) ** 2).mean()),
                         "gt_absz": float(np.abs(gt_z).mean()),
                         "pred_absz": float(np.abs(pred_z).mean())})
            taken[k] += 1
            LOG.info("  %-5s %-4s %-34s mse %6.2f  GT|z| %.2f  pred|z| %.2f", split, k,
                     rows[-1]["filename"][:34], rows[-1]["mse"], rows[-1]["gt_absz"],
                     rows[-1]["pred_absz"])
        del ds

    fig_train_vs_test(rows, out / "plots/01_train_vs_test.png")
    summary = {}
    for s, k in GROUPS:
        sel = [r["mse"] for r in rows if (r["split"], r["kind"]) == (s, k)]
        summary[f"{s}_{k}"] = {"n": len(sel), "mean_mse": float(np.mean(sel)) if sel else None,
                               "median_mse": float(np.median(sel)) if sel else None}
    (out / "summary.json").write_text(json.dumps(
        {"run": run, "per_group": args.per_group, "groups": summary, "rows": rows}, indent=2))
    LOG.info("summary: %s", {k: round(v["mean_mse"], 2) for k, v in summary.items() if v["mean_mse"]})
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
