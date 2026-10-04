#!/usr/bin/env python
"""Control: OakInk2 interaction-only takes vs the same segments widened to approach + retreat.
Compares Experiment-1 ratios, zero-transition shares and lag-32 levels over the groups common to both
runs. Writes oakink2/11_dynamic_contact_extended/control_vs_interaction_only.csv and prints the summary."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

A = Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact")
B = Path("/result/uhnam/dexcore/oakink2/11_dynamic_contact_extended")
ga = set("__".join(g) for g in json.load(open(A / "cache/groups.json")))
gb = set("__".join(g) for g in json.load(open(B / "cache/groups.json")))
common = sorted(ga & gb)
rows = []
for name, root in (("interaction_only", A), ("extended", B)):
    r = pd.read_csv(root / "exp1/ratios.csv"); r = r[(r.stat == "mean") & r.group.isin(common)]
    z = pd.read_csv(root / "exp1/zero_transition_share.csv"); z = z[(z.axis == "time") & z.group.isin(common)]
    L = pd.read_csv(root / "exp1/four_axes_long.csv"); L = L[(L.variant == "all") & (L.stat == "mean") & L.group.isin(common)]
    V = pd.read_csv(root / "verify/v1_v2_ordering_configs.csv") if (root / "verify/v1_v2_ordering_configs.csv").exists() else None
    for ratio in ("time/take", "time/mesh", "mesh/take", "action/take", "subject/take", "time/floor"):
        for met in ("raw", "mass_abs", "mass_log", "pattern_L2", "centroid"):
            v = r[(r.ratio == ratio) & (r.metric == met)].value
            if len(v):
                rows.append(dict(run=name, quantity=f"{ratio} [{met}]", macro=float(v.mean()), n_groups=len(v), groups_gt_1=int((v > 1).sum())))
    rows.append(dict(run=name, quantity="one_zero share of time pairs", macro=float(z.frac_one_zero.mean()), n_groups=len(z), groups_gt_1=np.nan))
    rows.append(dict(run=name, quantity="d2 share of one_zero pairs", macro=float(z.d2share_one_zero.mean()), n_groups=len(z), groups_gt_1=np.nan))
    for axis in ("floor", "time", "take"):
        v = L[(L.axis == axis) & (L.metric == "raw")].value
        rows.append(dict(run=name, quantity=f"raw mean distance, {axis}", macro=float(v.mean()), n_groups=len(v), groups_gt_1=np.nan))
    thr = json.load(open(root / "cache/zero_threshold.json"))["zero_mass"]
    rows.append(dict(run=name, quantity="zero-mass threshold", macro=thr, n_groups=np.nan, groups_gt_1=np.nan))
    ix = pd.read_csv(root / "takes_index.csv"); rows.append(dict(run=name, quantity="median take frames (30 Hz)", macro=float(ix.n_frames.median()), n_groups=len(ix), groups_gt_1=np.nan))
D = pd.DataFrame(rows)
P = D.pivot_table(index="quantity", columns="run", values=["macro", "groups_gt_1", "n_groups"])
P.to_csv(B / "control_vs_interaction_only.csv")
pd.set_option("display.width", 220)
print(f"common groups: {len(common)}")
print(P["macro"].join(P["groups_gt_1"].add_prefix("gt1_")).round(3).to_string())
