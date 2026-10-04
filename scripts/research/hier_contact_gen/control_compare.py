#!/usr/bin/env python
"""OakInk2 control: interaction-only (20_hier_contact_gen) vs approach/retreat restored
(21_hier_contact_gen_extended), on the groups the two runs share. Per group and pooled, held-out
take. Writes /result/uhnam/dexcore/reports/hier_control_oakink2.csv and .md."""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

sys_path = "/home/uhnam/workspace/dexcore/scripts/research/hier_contact_gen"
import sys; sys.path.insert(0, sys_path)
import hc_common as H

A = "/result/uhnam/dexcore/oakink2/20_hier_contact_gen"; B = "/result/uhnam/dexcore/oakink2/21_hier_contact_gen_extended"
ROWS = [("static_gt", 0), ("gtinit_vf", 0), ("samplerG_vf", 1), ("samplerG_vf", 10), ("samplerGT_vf", 1), ("samplerGT_vf", 10), ("dense", 0), ("samplerG_static", 10)]


def load(root):
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{root}/eval/take*/per_example.csv"))], ignore_index=True)
    return R


def main():
    Ra, Rb = load(A), load(B)
    common = sorted(set(Ra.group) & set(Rb.group))
    Ra, Rb = Ra[Ra.group.isin(common)], Rb[Rb.group.isin(common)]
    out = []
    for m, K in ROWS:
        for name, R in (("interaction-only", Ra), ("approach+retreat", Rb)):
            g = R[(R.model == m) & (R.K == K)]
            row = dict(model=m, K=K, run=name, n=len(g), n_groups=g.group.nunique())
            for met in ("E_C", "E_dC", "dmag_pred", "dmag_true", "pattern_L2", "mass_abs", "s0_err", "E_C_last16"):
                pt, lo, hi = H.cluster_bootstrap(g[met].values, g.take_key.values, n_boot=500)
                row[met], row[f"{met}_lo"], row[f"{met}_hi"] = pt, lo, hi
                row[f"{met}_macro"] = float(g.groupby("group")[met].mean().mean())
            out.append(row)
    D = pd.DataFrame(out); D.to_csv("/result/uhnam/dexcore/reports/hier_control_oakink2.csv", index=False)
    S = D.set_index(["model", "K", "run"])
    lines = [f"OakInk2 control on the {len(common)} shared groups (held-out take; `reports/hier_control_oakink2.csv`)", "",
             "| variant | E_C interaction-only | E_C approach+retreat | E_ΔC io / ar | pred Δ io / ar | true Δ io / ar | S0 err io / ar |", "|---|---|---|---|---|---|---|"]
    for m, K in ROWS:
        a, b = S.loc[(m, K, "interaction-only")], S.loc[(m, K, "approach+retreat")]
        lines.append(f"| {m} K={K} | {a.E_C:.3f} [{a.E_C_lo:.3f}, {a.E_C_hi:.3f}] | {b.E_C:.3f} [{b.E_C_lo:.3f}, {b.E_C_hi:.3f}] | {a.E_dC:.3f} / {b.E_dC:.3f} | {a.dmag_pred:.3f} / {b.dmag_pred:.3f} | {a.dmag_true:.3f} / {b.dmag_true:.3f} | {a.s0_err:.3f} / {b.s0_err:.3f} |")
    g = lambda run, m, K: S.loc[(m, K, run), "E_C"]
    for run in ("interaction-only", "approach+retreat"):
        lines.append(f"\n{run}: VF removes {100 * (g(run, 'static_gt', 0) - g(run, 'gtinit_vf', 0)) / g(run, 'static_gt', 0):.1f} % of the static error; "
                     f"B2 best-of-10 − B1 = {g(run, 'samplerG_vf', 10) - g(run, 'gtinit_vf', 0):+.3f}; B3 − B2 (best-of-10) = {g(run, 'samplerGT_vf', 10) - g(run, 'samplerG_vf', 10):+.3f}")
    open("/result/uhnam/dexcore/reports/hier_control_oakink2.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
