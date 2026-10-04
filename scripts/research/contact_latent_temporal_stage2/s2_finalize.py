#!/usr/bin/env python
"""Write the best / final checkpoints and the training logs of a run from its resumable `last` checkpoint.
    python s2_finalize.py --dataset taco --model B2 --stopped-by manual_stop      the run was stopped by hand: same files as a normal end
                                                                                  of training, and the "done <name>:" line is appended to its log
    python s2_finalize.py --dataset arctic --model B1 --snapshot                  training continues: a PRELIMINARY copy of the current best
                                                                                  state (stopped_by = "snapshot ..."); overwritten when the run ends
The selected state is the EMA state of the best validation step stored in the last checkpoint (the same state s2_train.py would save);
nothing is retrained and no test quantity is read.  Runs on the CPU (no GPU contention with the running jobs).
"""
from __future__ import annotations

import argparse
import logging
import re
import time

import numpy as np
import pandas as pd
import torch

import s2_common as S
import s2_models as MD
from s2_data import S2Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2_finalize")
LOGS = S.OUT / "logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--model", required=True, choices=list(S.MODELS))
    ap.add_argument("--seed", type=int, default=S.SEED); ap.add_argument("--tag", default="")
    ap.add_argument("--snapshot", action="store_true"); ap.add_argument("--stopped-by", default="manual_stop")
    a = ap.parse_args(); ds, model = a.dataset, a.model; name = S.run_name(model, a.seed, a.tag)
    last = S.ckpt_path(ds, name, "last")
    for _ in range(10):                                                      # the training process rewrites the file every validation
        try:
            st = torch.load(last, map_location="cpu", weights_only=False); break
        except Exception as e:                                               # noqa: BLE001
            log.info("last checkpoint not readable (%s); retrying", type(e).__name__); time.sleep(30)
    dev = torch.device("cpu")
    data = S2Data(ds, dev, need_masks=False, teacher=True)
    tck_path = S.teacher_ckpt_path(ds)
    net = MD.build(model, data); pc = net.param_counts()
    cfg = dict(S.TRAIN); lam = dict(z=cfg["lambda_z"], c=cfg["lambda_c"], res=cfg["lambda_res"])
    train_log = LOGS / f"train_{ds}_{name}.log"
    txt = train_log.read_text() if train_log.exists() else ""
    m = re.search(r"val at init ([0-9.]+)", txt); v0 = float(m.group(1)) if m else np.nan
    step = int(st["step"]); stopped_by = f"snapshot at step {step} (training continues)" if a.snapshot else a.stopped_by
    meta = dict(model=model, dataset=ds, seed=a.seed, name=name, cfg=cfg, backbone=S.BACKBONE, lambdas=lam, stats=data.stats, n_params=pc, d_traj=data.d_traj(), d_static=data.d_static(),
                teacher=dict(data.teacher_info, ckpt_md5_at_training=S.md5(tck_path)), stage1_decoder_init=S.MODELS[model]["z"], best_val=float(st["best"]), best_step=int(st["best_step"]), steps=step,
                stopped_by=stopped_by, converged=not a.snapshot, val_init=v0, val_init_components={}, seconds=float(st["elapsed"]), inspect=st["inspect_rows"],
                checks_without_improvement=int(st["bad"]), written_by="s2_finalize.py")
    ck = S.ckpt_path(ds, name); ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=st["best_state"], **meta), ck)
    if not a.snapshot:
        torch.save(dict(ema_state=st["ema_state"], raw_state=st["state"], **meta), S.ckpt_path(ds, name, "final"))
    S.train_log_path(ds, name).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(st["rows"]).to_csv(S.train_log_path(ds, name), index=False); pd.DataFrame(st["train_rows"]).to_csv(S.train_log_path(ds, name + "_train"), index=False)
    pd.DataFrame(st["inspect_rows"]).to_csv(S.train_log_path(ds, name + "_gradinspect"), index=False)
    line = f"{time.strftime('%H:%M:%S')} done {name}: best val {st['best']:.4f} at step {st['best_step']} of {step} ({stopped_by}), {st['elapsed']:.0f} s -> {ck}"
    if not a.snapshot:
        with open(train_log, "a") as f:
            f.write(line + "\n")
    log.info("%s%s", "[snapshot] " if a.snapshot else "", line)


if __name__ == "__main__":
    main()
