#!/usr/bin/env python
"""Choose lambda_struct (from the D1 sweep) and lambda_aux (from the D2 sweep) ONCE per dataset on the VALIDATION set.
    python select_lambda.py --dataset taco --stage struct      (after the D1 sweep runs were generated + evaluated on val)
    python select_lambda.py --dataset taco --stage aux         (after the D2 sweep)
Criterion: the mean over {dense E_C, participation Hamming, amount L1, centroid distance, normal angle, wrench rel. L1} of the
validation error divided by the persistence (C_t = s_0) validation error — one scalar per candidate; the smallest wins.
Writes / updates <ds>/lambda_choice.json and <ds>/lambda_selection.csv.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import sat_common as S

METRICS = ("E_C", "part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1")


def val_means(ds, name):
    z = np.load(S.metrics_path(ds, name, "A_val"), allow_pickle=True)
    return {k: float(np.nanmean(z[k][:, 0])) for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--stage", required=True, choices=["struct", "aux"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ds = a.dataset
    ref = val_means(ds, "PERSIST")
    p = S.ds_out(ds) / "lambda_choice.json"
    choice = json.loads(p.read_text()) if p.exists() else {}
    rows = []
    if a.stage == "struct":
        cands = [(S.run_name("D1", a.seed, lam), dict(lambda_struct=lam)) for lam in S.LAMBDA_STRUCT_GRID]
        base = [("D0", S.run_name("D0", a.seed))]
    else:
        ls = choice["lambda_struct"]
        cands = [(S.run_name("D2", a.seed, ls, lam), dict(lambda_struct=ls, lambda_aux=lam)) for lam in S.LAMBDA_AUX_GRID]
        base = [("D1", S.run_name("D1", a.seed, ls))]
    for label, name in base:
        if not S.metrics_path(ds, name, "A_val").exists():          # the baseline is informative only (not a sweep run)
            continue
        m = val_means(ds, name); rows.append(dict(dataset=ds, stage=a.stage, candidate=label, name=name, score=np.mean([m[k] / ref[k] for k in METRICS]), **{f"{k}_ratio": m[k] / ref[k] for k in METRICS}, **m))
    for name, lam in cands:
        m = val_means(ds, name)
        rows.append(dict(dataset=ds, stage=a.stage, candidate=json.dumps(lam), name=name, score=np.mean([m[k] / ref[k] for k in METRICS]), **{f"{k}_ratio": m[k] / ref[k] for k in METRICS}, **m, **lam))
    df = pd.DataFrame(rows)
    best = df[df.candidate.str.startswith("{")].sort_values("score").iloc[0]
    if a.stage == "struct":
        choice["lambda_struct"] = float(best.lambda_struct)
    else:
        choice["lambda_aux"] = float(best.lambda_aux)
    choice[f"criterion_{a.stage}"] = "mean over (E_C, part Hamming, amount L1, centroid, normal, wrench rel L1) of val error / persistence val error; seed 0; protocol A on the validation sequences"
    choice[f"scores_{a.stage}"] = {r.name: float(r.score) for r in df.itertuples()}
    p.write_text(json.dumps(choice, indent=2))
    old = pd.read_csv(S.ds_out(ds) / "lambda_selection.csv") if (S.ds_out(ds) / "lambda_selection.csv").exists() else pd.DataFrame()
    pd.concat([old[old.stage != a.stage] if len(old) else old, df], ignore_index=True).to_csv(S.ds_out(ds) / "lambda_selection.csv", index=False)
    print(df[["candidate", "score"] + [f"{k}_ratio" for k in METRICS]].round(4).to_string())
    print(f"{ds} {a.stage}: chosen {best.candidate} (score {best.score:.4f}) -> {p}")


if __name__ == "__main__":
    main()
