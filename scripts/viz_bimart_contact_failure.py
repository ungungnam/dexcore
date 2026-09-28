#!/usr/bin/env python
"""Why does the contact prior fail so much harder on `grab` windows than on `use` windows?

Three mechanical explanations, all measurable without retraining:

  H1 training balance.  `grab` sequences may simply be rarer than `use` ones, leaving the prior
     undertrained on them.
  H2 out-of-distribution conditioning.  The contact model sees only BPS geometry and a 7-d global
     state (rotation, translation relative to the window start, scale). During a `grab` the object
     is carried, so that translation grows large; during a `use` it mostly sits and articulates.
     Normalised by dataset-wide statistics, `grab` conditioning may land far outside the range the
     model was trained on.
  H3 regression to the mean.  When conditioning cannot determine the answer, a model trained with
     an MSE objective hedges toward the dataset mean. If a window's true contact map is itself far
     from that mean, hedging is automatically a large error. Measured as |z|: how many standard
     deviations the GT contact sits from the training mean, against the same for the prediction.

  python scripts/viz_bimart_contact_failure.py --categories laptop --batches 8

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/contact_failure/<run>/plots.
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
import collections
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
LOG = logging.getLogger("bimart_contact_failure")
GS = {"rotation": slice(0, 3), "rel. translation": slice(3, 6), "scale": slice(6, 7)}


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def kind(name):
    return "grab" if "_grab_" in name else ("use" if "_use_" in name else "other")


def fig_failure(records, balance, out):
    fig, axes = plt.subplots(1, 4, figsize=(21, 5.0))
    names = [r["short"] for r in records]
    x = np.arange(len(records))
    col = ["#d62728" if r["kind"] == "grab" else "#2ca02c" for r in records]

    ax = axes[0]
    ks = sorted(balance)
    ax.bar(range(len(ks)), [balance[k]["windows"] for k in ks], .6,
           color=["#d62728" if k == "grab" else "#2ca02c" for k in ks])
    for i, k in enumerate(ks):
        ax.text(i, balance[k]["windows"], f"\n{balance[k]['files']} files", ha="center",
                va="top", fontsize=8, color="white")
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
    ax.set_ylabel("training windows")
    ax.set_title("H1. is `grab` under-represented?", fontsize=11)
    ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    w = .8 / len(GS)
    for j, (g, sl) in enumerate(GS.items()):
        ax.bar(x + j * w - .4 + w / 2, [r["gs_absz"][g] for r in records], w, label=g)
    ax.axhline(3, c="#333333", ls="--", lw=1, label="3 sd")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean |z| of the global state")
    ax.set_title("H2. is the conditioning out of distribution?", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[2]
    ax.bar(x - .2, [r["gt_absz"] for r in records], .4, color="#7f7f7f", label="GT contact")
    ax.bar(x + .2, [r["pred_absz"] for r in records], .4, color="#d62728", label="prediction")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("mean |z| from the training mean")
    ax.set_title("H3. does the model hedge toward the mean?", fontsize=11)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = axes[3]
    for r, c in zip(records, col):
        ax.scatter(r["gt_absz"], r["contact_mse"], s=70, color=c)
        ax.annotate(r["short"], (r["gt_absz"], r["contact_mse"]), fontsize=6,
                    xytext=(4, 3), textcoords="offset points")
    gx = np.array([r["gt_absz"] for r in records])
    gy = np.array([r["contact_mse"] for r in records])
    if len(gx) > 2:
        k = np.polyfit(gx, gy, 1)
        xs = np.linspace(gx.min(), gx.max(), 20)
        ax.plot(xs, np.polyval(k, xs), c="#333333", ls="--", lw=1,
                label=f"r = {np.corrcoef(gx, gy)[0,1]:.2f}")
        ax.legend(fontsize=8)
    ax.set_xlabel("how atypical the GT contact is  (mean |z|)")
    ax.set_ylabel("contact MSE (normalised)")
    ax.set_title("H3. atypical windows are the ones that fail\n(red = grab, green = use)",
                 fontsize=11)
    ax.grid(alpha=.25)

    fig.suptitle("Why the contact prior fails harder on `grab`", fontsize=13)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    LOG.info("wrote %s", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--batches", type=int, default=8)
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
    from scripts.viz_bimart_conditioning import short_name

    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    cfg = yaml_util.load_yaml(args.config)
    contact_cfg = yaml_util.load_yaml(cfg["test"]["contact_config_file"])
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/contact_failure" / run
    (out / "plots").mkdir(parents=True, exist_ok=True)
    LOG.info("writing to %s  (GPU %d)", out, args.gpu)

    stat_dict = data_util.load_stat_dict(cfg["data"]["stat_dict_path"], device)
    dataset = make_subset_dataset(motion_data, cfg, set(args.categories))

    # H1: training balance, over ALL categories, counted the way the loader counts
    full = motion_data.MotionDataset.__new__(motion_data.MotionDataset)
    full.base_dir = cfg["data"]["base_dir"]
    full.get_train_test_split()
    balance = collections.defaultdict(lambda: {"files": 0, "windows": 0})
    for p in full.data_files["train"]["hand_process_files"]:
        k = kind(Path(p).name)
        balance[k]["files"] += 1
    # A file contributes one training window per frame of its usable span. Sequence length comes
    # from the raw ARCTIC object file ([T, 7], a few KB) -- the processed features are 160 MB each
    # and loading every one of them just to read a length would be absurd.
    for p in full.data_files["train"]["obj_process_files"]:
        stem = Path(p).name.split("_processed_obj_features")[0]
        raw = Path("data/arctic_raw/raw_seqs") / Path(p).parent.name / f"{stem}.object.npy"
        if not raw.exists():
            LOG.warning("no raw sequence for %s", stem)
            continue
        T = len(np.load(raw, allow_pickle=True))
        balance[kind(stem)]["windows"] += max(
            0, T - cfg["data"]["pred_horizon"] - 2 * cfg["data"]["base_frame"])
    balance = {k: dict(v) for k, v in balance.items()}
    LOG.info("training balance: %s", balance)

    base = model_util.createNN(cfg, device)
    contact_model = ContactInference(dataset.mesh_dict, cfg=contact_cfg, motion_config=cfg)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    cmean = contact_model.stat_dict["action"]["mean"]
    cstd = contact_model.stat_dict["action"]["std"]

    records = []
    for b, batch in enumerate(loader):
        if b >= args.batches:
            break
        name = batch["viz"]["filename"][-1]
        gt_raw = batch["obs"]["contact_points"].clone()
        bn = data_util.preprocess_batch(batch, stat_dict, device)
        gs = bn["obs"]["global_states"][0].cpu().numpy()          # already z-scored
        gt_z = data_util.normalize_item(gt_raw.to(device).float(), cmean, cstd)[0].cpu().numpy()
        c_x0, _, c_final, _ = run_contact_stage(contact_model, bn)
        pred_z = c_final[0].cpu().numpy()

        r = {"name": name, "kind": kind(name),
             "gs_absz": {g: float(np.abs(gs[:, sl]).mean()) for g, sl in GS.items()},
             "gt_absz": float(np.abs(gt_z).mean()),
             "pred_absz": float(np.abs(pred_z).mean()),
             "contact_mse": float(((pred_z - gt_z) ** 2).mean())}
        records.append(r)
        LOG.info("[%d] %-24s %s | gs|z| %s | GT|z| %.2f  pred|z| %.2f  mse %.2f",
                 b, name, r["kind"], {k: round(v, 1) for k, v in r["gs_absz"].items()},
                 r["gt_absz"], r["pred_absz"], r["contact_mse"])

    for r in records:
        r["short"] = short_name(r["name"], [q["name"] for q in records])
    fig_failure(records, balance, out / "plots/01_why_contact_fails.png")
    (out / "summary.json").write_text(json.dumps(
        {"run": run, "training_balance": balance,
         "windows": {r["short"]: {k: v for k, v in r.items() if k != "short"}
                     for r in records}}, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
