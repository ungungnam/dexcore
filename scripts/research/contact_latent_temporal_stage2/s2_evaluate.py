#!/usr/bin/env python
"""Metrics of one prediction file with the previous temporal study's evaluation code (identical metric definitions).
    CUDA_VISIBLE_DEVICES=5 python s2_evaluate.py --dataset taco --name B1_seed0 [--part val]
    CUDA_VISIBLE_DEVICES=5 python s2_evaluate.py --dataset taco --name GT          (grid-extraction floor)
    CUDA_VISIBLE_DEVICES=5 python s2_evaluate.py --dataset taco --name PERSIST     (C_t = s_0)
    CUDA_VISIBLE_DEVICES=5 python s2_evaluate.py --dataset taco --name B2_seed0 --zonly   (the z-only path C_bar of B2, stored as B2_seed0_zonly)
Per example (frames 1..63): dense E_C (raw L2 per frame), E_C of the last 16 frames; exact-rule grid R2 vs the exact R2 of the feature
cache (participation Hamming / TP-FP-FN for the macro F1 / soft scores for the AUPRC, amount normalised L1, centroid distance / l, normal
angle), the 76-direction wrench (relative L1, cosine, Q error), temporal metrics (dense and R2 frame-to-frame change, total variation),
error-vs-time curves, the four event classes.  Writes <ds>/metrics/<name>_A[_val].npz (+ _events.csv).
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import time

import numpy as np
import pandas as pd
import torch

import s2_common as S
from sat_data import SeqData

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("s2_evaluate")


def sat_evaluate_module():
    spec = importlib.util.spec_from_file_location("sat_evaluate", S.SAT_DIR / "evaluate.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_predictions(ds, name, part, zonly):
    z = np.load(S.preds_path(ds, name, part))
    if zonly:
        assert "C_bar" in z, "z-only path only exists for B2 predictions"
        return z["C_bar"].astype(np.float32), z["example"]
    return z["pred"].astype(np.float32), z["example"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--name", required=True)
    ap.add_argument("--part", default="test", choices=["test", "val"]); ap.add_argument("--zonly", action="store_true"); ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args(); ds = a.dataset; t0 = time.time(); dev = torch.device("cuda")
    E = sat_evaluate_module()
    data = SeqData(ds, dev, need_masks=True)
    te = data.idx[a.part]
    if a.name == "GT":
        P_all = data.C[te][:, None].cpu().numpy(); example = te
    elif a.name == "PERSIST":
        P_all = data.C[te, 0][:, None, None].expand(-1, 1, S.T, -1).cpu().numpy(); example = te
    else:
        P_all, example = load_predictions(ds, a.name, a.part, a.zonly)
    assert np.array_equal(example, te), "prediction file does not match the split"
    out_name = a.name + ("_zonly" if a.zonly else "")
    B, K = P_all.shape[:2]
    outs, curs, r2s = {}, {}, {}
    for i in range(0, B, a.chunk):
        n = torch.from_numpy(example[i:i + a.chunk]).to(dev); P = torch.from_numpy(P_all[i:i + a.chunk]).to(dev)
        o, c, r = E.evaluate_chunk(data, n, P, S.PROTOCOL)
        for k, v in o.items():
            outs.setdefault(k, []).append(v.cpu().numpy())
        for k, v in c.items():
            curs.setdefault(k, []).append(v.cpu().numpy())
        for k, v in r.items():
            r2s.setdefault(k, []).append(v.cpu().numpy())
    outs = {k: np.concatenate(v) for k, v in outs.items()}; curs = {k: np.concatenate(v) for k, v in curs.items()}; r2s = {k: np.concatenate(v) for k, v in r2s.items()}
    a_true = data.a[torch.from_numpy(example).to(dev)].cpu().numpy()
    Ev = S.SAT.load_events(ds) if a.part == "test" else pd.DataFrame()
    rows = E.event_rows(data, Ev, {int(e): i for i, e in enumerate(example)}, r2s, curs, a_true, S.PROTOCOL, out_name) if len(Ev) else []
    out = S.metrics_path(ds, out_name, a.part); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, example=example, take=data.take[example], K=K, protocol=S.PROTOCOL, name=out_name,
                        **outs, **{f"curve_{k}": v.astype(np.float32) for k, v in curs.items()},
                        r2_a=r2s["a"].astype(np.int8), r2_m=r2s["m"].astype(np.float16), r2_p=r2s["p"].astype(np.float16), r2_n=r2s["n"].astype(np.float16),
                        r2_a_soft=r2s["a_soft"].astype(np.float16), q=r2s["q"].astype(np.float16), a_true=a_true.astype(np.int8))
    pd.DataFrame(rows).to_csv(out.with_name(out.stem + "_events.csv"), index=False)
    summ = {k: float(np.nanmean(outs[k][:, 0])) for k in ("E_C", "E_C_last16", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos", "jitter_dense", "jitter_r2")}
    log.info("%s %s %s (%d examples): %s (%d event rows, %.0f s)", ds, out_name, a.part, B, {k: round(v, 3) for k, v in summ.items()}, len(rows), time.time() - t0)


if __name__ == "__main__":
    main()
