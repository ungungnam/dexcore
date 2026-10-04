#!/usr/bin/env python
"""Metrics of one prediction file of this study with the evaluation code of the previous studies (the function Stage 2 called:
structure_aware_temporal_generation/evaluate.py: evaluate_chunk).  Identical metric definitions and extraction code for every model.
    CUDA_VISIBLE_DEVICES=5 python zj_evaluate.py --dataset taco --name M2_seed0 [--part val]
    CUDA_VISIBLE_DEVICES=5 python zj_evaluate.py --dataset taco --name B1_seed0 --recheck     re-evaluate a Stage-2 prediction file and
                                                compare with its stored metrics (sanity check: same code, same numbers); writes nothing but a json
Writes <ds>/metrics/<name>_A[_val].npz (+ _events.csv) in the Stage-2 format.
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch

import zj_common as Z
from s2_evaluate import sat_evaluate_module
from sat_data import SeqData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("zj_evaluate")
S2 = Z.S2
SUMMARY = ("E_C", "E_C_last16", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos", "jitter_dense", "jitter_r2")


def run(E, data, P_all, example, chunk):
    dev = data.device; outs, curs, r2s = {}, {}, {}
    for i in range(0, len(example), chunk):
        n = torch.from_numpy(example[i:i + chunk]).to(dev); P = torch.from_numpy(P_all[i:i + chunk]).to(dev)
        o, c, r = E.evaluate_chunk(data, n, P, Z.PROTOCOL)
        for dst, src in ((outs, o), (curs, c), (r2s, r)):
            for k, v in src.items():
                dst.setdefault(k, []).append(v.cpu().numpy())
    cat = lambda d: {k: np.concatenate(v) for k, v in d.items()}
    return cat(outs), cat(curs), cat(r2s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=Z.DATASETS); ap.add_argument("--name", required=True)
    ap.add_argument("--part", default="test", choices=["test", "val"]); ap.add_argument("--chunk", type=int, default=8); ap.add_argument("--recheck", action="store_true")
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    E = sat_evaluate_module()
    data = SeqData(ds, dev, need_masks=True)
    te = data.idx[a.part]
    z = np.load(S2.preds_path(ds, a.name, a.part) if a.recheck else Z.preds_path(ds, a.name, a.part))
    P_all, example = z["pred"].astype(np.float32), z["example"]
    assert np.array_equal(example, te), "prediction file does not match the split"
    outs, curs, r2s = run(E, data, P_all, example, a.chunk)
    if a.recheck:
        old = np.load(S2.metrics_path(ds, a.name, a.part), allow_pickle=True)
        diff = {k: float(np.nanmax(np.abs(outs[k].astype(np.float64) - old[k].astype(np.float64)))) for k in SUMMARY}
        Z.write_json(Z.ds_out(ds) / "metrics" / f"recheck_{a.name}.json", dict(name=a.name, max_abs_diff_per_example=diff, n_examples=int(len(example)),
                                                                                how="Stage-2 prediction file re-evaluated by zj_evaluate.py and compared with the stored Stage-2 metrics file"))
        log.info("%s recheck %s: max |diff| per example %s", ds, a.name, {k: f"{v:.2e}" for k, v in diff.items()}); return
    a_true = data.a[torch.from_numpy(example).to(dev)].cpu().numpy()
    Ev = S2.SAT.load_events(ds) if a.part == "test" else pd.DataFrame()
    rows = E.event_rows(data, Ev, {int(e): i for i, e in enumerate(example)}, r2s, curs, a_true, Z.PROTOCOL, a.name) if len(Ev) else []
    out = Z.metrics_path(ds, a.name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, example=example, take=data.take[example], K=P_all.shape[1], protocol=Z.PROTOCOL, name=a.name,
                        **outs, **{f"curve_{k}": v.astype(np.float32) for k, v in curs.items()},
                        r2_a=r2s["a"].astype(np.int8), r2_m=r2s["m"].astype(np.float16), r2_p=r2s["p"].astype(np.float16), r2_n=r2s["n"].astype(np.float16),
                        r2_a_soft=r2s["a_soft"].astype(np.float16), q=r2s["q"].astype(np.float16), a_true=a_true.astype(np.int8))
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + "_events.csv"), index=False)
    log.info("%s %s %s (%d examples): %s (%d event rows, %.0f s)", ds, a.name, a.part, len(example), {k: round(float(np.nanmean(outs[k][:, 0])), 3) for k in SUMMARY}, len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
