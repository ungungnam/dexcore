#!/usr/bin/env python
"""Did the structural output loss optimise what it measures? The four soft training terms (normalised by the persistence
references of the training set, as in the loss) evaluated on the stored TEST predictions of every run, next to the exact-rule
metrics.    CUDA_VISIBLE_DEVICES=4 python diag_soft_terms.py --dataset taco
Writes <ds>/soft_terms_test.csv (one row per run and protocol-A sample 0)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

import sat_common as S
from sat_data import SeqData
from losses import structural_terms, persistence_reference, STRUCT_TERMS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True)
    a = ap.parse_args(); dev = torch.device("cuda")
    data = SeqData(a.dataset, dev); cal = data.cal
    ref = persistence_reference(data, cal)
    te = torch.from_numpy(data.idx["test"]).to(dev)
    rows = []
    names = [("PERSIST", None)] + [(S.run_name(m, s), m) for m in S.MODELS for s in S.SEEDS if S.preds_path(a.dataset, S.run_name(m, s), "A").exists()]
    for name, model in names:
        P = None if name == "PERSIST" else torch.from_numpy(np.load(S.preds_path(a.dataset, name, "A"))["pred"][:, 0].astype(np.float32)).to(dev)
        acc = {k: 0.0 for k in STRUCT_TERMS}; cnt = 0
        with torch.no_grad():
            for i in range(0, len(te), 32):
                n = te[i:i + 32]; b = data.batch(n)
                C_hat = b["s0_raw"][:, None].expand(-1, S.TF, -1) if P is None else P[i:i + 32, 1:]
                t = structural_terms(C_hat, b, cal)
                for k in STRUCT_TERMS:
                    acc[k] += float(t[k]) * len(n)
                cnt += len(n)
        r = dict(dataset=a.dataset, run=name, model=model or "PERSIST", **{f"soft_{k}": acc[k] / cnt for k in STRUCT_TERMS}, **{f"soft_{k}_norm": acc[k] / cnt / ref[k] for k in STRUCT_TERMS})
        r["soft_struct_norm"] = float(np.mean([r[f"soft_{k}_norm"] for k in STRUCT_TERMS]))
        z = np.load(S.metrics_path(a.dataset, name, "A"), allow_pickle=True)
        for k in ("E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1"):
            r[f"exact_{k}"] = float(np.nanmean(z[k][:, 0]))
        rows.append(r); print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    pd.DataFrame(rows).to_csv(S.ds_out(a.dataset) / "soft_terms_test.csv", index=False)


if __name__ == "__main__":
    main()
