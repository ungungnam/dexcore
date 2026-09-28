#!/usr/bin/env python
"""Aggregate the per-(window,hand,frame) probe rows of all three runs to window x hand level.

The three runs are the same port with different stage-1 contact labels:
  orig  = both objects scaled by the TARGET's factor (tool overflows the BPS basis: the defect)
  fps   = label gathered at per-mesh farthest-point samples (wrong fix: breaks input-label registration)
  scene = one scale over tool+target, label still gathered at the BPS indices (upstream's design)
Window level is the unit every later comparison uses; sequence_id is the bootstrap cluster.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("/result/uhnam/dexcore/bimart_taco_scene/contact_probe/gen/reverify")
RUNS = {"orig": "/result/uhnam/dexcore/bimart_taco/contact_probe",
        "fps": "/result/uhnam/dexcore/bimart_taco/contact_probe_fps",
        "scene": "/result/uhnam/dexcore/bimart_taco_scene/contact_probe"}
KEEP = ["split", "sequence_id", "start", "hand", "frame", "verb", "tool_cat", "target_cat",
        "tool_mesh", "target_mesh", "triplet", "motion_mm", "motion_trans_mm", "motion_resid_mm",
        "contact_mm", "contact_trans_mm", "contact_resid_mm", "total_mm", "cos",
        "gt_min_dist_mm", "contact_map_err_mm", "touch_iou", "gt_touch_frac"]
NUM = ["motion_mm", "motion_trans_mm", "motion_resid_mm", "contact_mm", "contact_trans_mm",
       "contact_resid_mm", "total_mm", "cos", "gt_min_dist_mm", "contact_map_err_mm",
       "touch_iou", "gt_touch_frac"]
GRP = ["split", "sequence_id", "start", "hand", "verb", "tool_cat", "target_cat", "tool_mesh",
       "target_mesh", "triplet"]


def agg(path: Path) -> pd.DataFrame:
    """Mean over the 64 frames of a window, plus the frame-bucket means used for the G6 profile."""
    parts = []
    for ch in pd.read_csv(path, usecols=KEEP, chunksize=2_000_000):
        ch["bucket"] = (ch.frame // 8).astype(int)
        parts.append(ch.groupby(GRP + ["bucket"], observed=True)[NUM].mean().reset_index())
    d = pd.concat(parts, ignore_index=True)
    # a chunk boundary can split one (window,bucket); average the pieces (equal frame counts)
    d = d.groupby(GRP + ["bucket"], observed=True)[NUM].mean().reset_index()
    return d


rows = []
for run, root in RUNS.items():
    for f in sorted(Path(root).glob("*.csv")):
        if f.stem not in ("train", "test_1", "test_2", "test_3", "test_4"):
            continue
        d = agg(f)
        d.insert(0, "run", run)
        rows.append(d)
        print(f"{run}/{f.stem}: {len(d)} (window,hand,bucket) rows, "
              f"{d.sequence_id.nunique()} sequences", flush=True)
b = pd.concat(rows, ignore_index=True)
b.to_pickle(OUT / "bucket_level.pkl")
w = b.groupby(["run"] + GRP, observed=True)[NUM].mean().reset_index()
w.to_pickle(OUT / "window_level.pkl")
print("bucket rows", len(b), "-> window rows", len(w))
print(w.groupby(["run", "split"]).size().to_string())
