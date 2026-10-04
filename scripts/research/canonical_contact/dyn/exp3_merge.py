#!/usr/bin/env python
"""Merge sharded Experiment-3 outputs (per_sequence_shard*.csv, per_fold_shard*.csv) and aggregate."""
import glob

import pandas as pd

import dc_common as C
from exp3_predictability import OUT, aggregate

PS = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "per_sequence_shard*of*.csv")))], ignore_index=True)
PF = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(OUT / "per_fold_shard*of*.csv")))], ignore_index=True)
PS.to_csv(OUT / "per_sequence.csv", index=False); PF.to_csv(OUT / "per_fold.csv", index=False)
print("merged", PS.group.nunique(), "groups,", len(PF), "fold rows")
aggregate(PS, PF)
