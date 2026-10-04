#!/usr/bin/env python
"""Metrics of one prediction file of this study with the evaluation code of the previous studies (identical definitions and extraction code
for every model; the function is the one zj_evaluate.py / s2_evaluate.py call).
    CUDA_VISIBLE_DEVICES=4 python zf_evaluate.py --dataset taco --name M11_seed0 [--part val]
Writes <ds>/metrics/<name>_A[_val].npz (+ _events.csv) in the Stage-2 format."""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

import zf_common as Z
from s2_evaluate import sat_evaluate_module
from sat_data import SeqData
from zj_evaluate import SUMMARY, run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zf_evaluate")
S2 = Z.S2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--name", required=True); ap.add_argument("--part", default="test", choices=["test", "val"]); ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    E = sat_evaluate_module(); data = SeqData(ds, dev, need_masks=True); te = data.idx[a.part]
    z = np.load(Z.preds_path(ds, a.name, a.part)); P_all, example = z["pred"].astype(np.float32), z["example"]
    assert np.array_equal(example, te), "prediction file does not match the split"
    outs, curs, r2s = run(E, data, P_all, example, a.chunk)
    a_true = data.a[torch.from_numpy(example).to(dev)].cpu().numpy()
    Ev = S2.SAT.load_events(ds) if a.part == "test" else pd.DataFrame()
    rows = E.event_rows(data, Ev, {int(e): i for i, e in enumerate(example)}, r2s, curs, a_true, Z.PROTOCOL, a.name) if len(Ev) else []
    out = Z.metrics_path(ds, a.name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, example=example, take=data.take[example], K=P_all.shape[1], protocol=Z.PROTOCOL, name=a.name, **outs, **{f"curve_{k}": v.astype(np.float32) for k, v in curs.items()},
                        r2_a=r2s["a"].astype(np.int8), r2_m=r2s["m"].astype(np.float16), r2_p=r2s["p"].astype(np.float16), r2_n=r2s["n"].astype(np.float16), r2_a_soft=r2s["a_soft"].astype(np.float16),
                        q=r2s["q"].astype(np.float16), a_true=a_true.astype(np.int8))
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + "_events.csv"), index=False)
    log.info("%s %s %s (%d examples): %s (%.0f s)", ds, a.name, a.part, len(example), {k: round(float(np.nanmean(outs[k][:, 0])), 3) for k in SUMMARY}, time.time() - t0)


if __name__ == "__main__":
    main()
