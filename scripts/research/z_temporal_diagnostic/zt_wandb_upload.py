#!/usr/bin/env python
"""Log the training curves of the twelve probes to Weights & Biases from the saved logs (the probes train in about two minutes each and
were not logged live).    WANDB_API_KEY=... python zt_wandb_upload.py
One run per (dataset, variant, horizon) in project dexcore-z-temporal-diagnostic: config = recipe + probe architecture + paths,
history = train / validation MSE and learning rate per validation step, summary = best step, best validation MSE, persistence MSE."""
from __future__ import annotations

import pandas as pd
import torch
import wandb

import zt_common as Z


def main():
    for ds in Z.DATASETS:
        for v in Z.VARIANTS:
            for h in Z.HORIZONS:
                ck = torch.load(Z.ckpt_path(ds, v, h), map_location="cpu", weights_only=False); lg = pd.read_csv(Z.train_log_path(ds, v, h))
                run = wandb.init(project="dexcore-z-temporal-diagnostic", name=f"{ds}_{v}_h{h}", id=f"{ds}_{v}_h{h}", resume="allow", reinit=True, dir=str(Z.OUT / "wandb"),
                                 config=dict(dataset=ds, variant=v, horizon=h, seed=ck["seed"], **ck["cfg"], probe=ck["probe"], n_params=ck["n_params"], d_in=ck["d_in"], teacher_md5=ck["teacher_md5"],
                                             ckpt=str(Z.ckpt_path(ds, v, h)), preds=str(Z.preds_path(ds, v, h)), out=str(Z.ds_out(ds)), uploaded_from="saved train log"))
                for r in lg.itertuples():
                    run.log({"train/loss": r.train, "val/loss": r.val, "train/lr": r.lr}, step=int(r.step))
                run.summary.update(dict(best_val_mse=ck["best_val"], best_step=ck["best_step"], steps=ck["steps"], stopped_by=ck["stopped_by"], val_persistence_mse=ck["val_persistence"]))
                run.finish()


if __name__ == "__main__":
    main()
