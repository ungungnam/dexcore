#!/usr/bin/env python
"""Write a best-format checkpoint from a run's resumable `last` checkpoint WITHOUT touching the run (CPU only).
    python zj_snapshot.py --dataset taco --model M2 --dest <dir>          a preliminary copy for dry runs of the analysis chain (ZJ_CKPT=<dir>)
    python zj_snapshot.py --dataset taco --model M2 --finalize            a run that ended without writing its files (e.g. killed after its last validation):
                                                                          the same files zj_train.py writes, in place, and the "done" line in its log
The state written is the EMA state at the best validation L_C stored in the last checkpoint (and the state at the best objective).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

import zj_common as Z
import zj_models as MD
from s2_data import S2Data

S2 = Z.S2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--model", required=True, choices=list(Z.TRAINED_HERE)); ap.add_argument("--seed", type=int, default=Z.SEED); ap.add_argument("--tag", default="")
    ap.add_argument("--dest", default=""); ap.add_argument("--finalize", action="store_true"); ap.add_argument("--src-ckpt", default="/ckpt/uhnam/dexcore/z_joint_decoder_followup"); ap.add_argument("--stopped-by", default="patience")
    a = ap.parse_args(); ds, model = a.dataset, a.model; name = Z.run_name(model, a.seed, a.tag)
    assert bool(a.dest) != a.finalize, "give --dest (dry-run copy) or --finalize (in place)"
    last = Path(a.src_ckpt) / ds / f"{name}_last.pt"
    for _ in range(10):                                                        # the training process replaces the file at every validation (atomic rename)
        try:
            st = torch.load(last, map_location="cpu", weights_only=False); break
        except Exception:                                                      # noqa: BLE001
            time.sleep(20)
    data = S2Data(ds, torch.device("cpu"), need_masks=False, teacher=True); joint = Z.MODELS[model]["joint"]
    cfg = dict(Z.TRAIN, extra_blocks=Z.DECODER["extra_blocks"] if joint else 0)
    net = MD.build(model, data); pc = net.param_counts(); step = int(st["step"])
    meta = dict(model=model, dataset=ds, seed=a.seed, name=name, cfg=cfg, backbone=Z.BACKBONE, stats=data.stats, n_params=pc, d_traj=data.d_traj(), d_static=data.d_static(),
                teacher=dict(data.teacher_info, ckpt_md5_at_training=Z.md5(S2.teacher_ckpt_path(ds))), stage1_decoder_init=joint, selection="validation L_C of the predicted-z path" if joint else "validation L_C",
                best_val=st["best"]["sel"], best_step=st["best_step"]["sel"], best_val_total=st["best"]["total"], best_step_total=st["best_step"]["total"], steps=step,
                stopped_by=a.stopped_by if a.finalize else f"snapshot at step {step} (training continues)", converged=a.finalize, val_init=st.get("val_init"), val_init_components=st.get("val_init_components", {}),
                seconds=float(st["elapsed"]), inspect=st["inspect_rows"], decoder_change_final={}, written_by="zj_snapshot.py")
    root = Z.CKPT if a.finalize else Path(a.dest)
    ck = root / ds / f"{name}.pt"; ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=st["best_state"]["sel"], state_total=st["best_state"]["total"], **meta), ck)
    out = Z.OUT if a.finalize else Path(a.dest).parent / "out"
    lp = out / ds / "train_logs" / f"{name}.csv"; lp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(st["rows"]).to_csv(lp, index=False); pd.DataFrame(st["train_rows"]).to_csv(lp.with_name(name + "_train.csv"), index=False); pd.DataFrame(st["inspect_rows"]).to_csv(lp.with_name(name + "_gradinspect.csv"), index=False)
    line = f"{time.strftime('%H:%M:%S')} done {name}: best val L_C {st['best']['sel']:.4f} at step {st['best_step']['sel']} (objective {st['best']['total']:.4f} at step {st['best_step']['total']}) of {step} ({meta['stopped_by']}), {st['elapsed']:.0f} s -> {ck}"
    if a.finalize:
        torch.save(dict(ema_state=st["ema_state"], raw_state=st["state"], **meta), Z.ckpt_path(ds, name, "final"))
        with open(Z.OUT / "logs" / f"train_{ds}_{name}.log", "a") as f:
            f.write(line + "\n")
    print(("[finalized] " if a.finalize else "[snapshot] ") + line)


if __name__ == "__main__":
    main()
