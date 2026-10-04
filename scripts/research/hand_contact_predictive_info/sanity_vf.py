#!/usr/bin/env python
"""Sanity check 4: does F0 behave like the existing contact-only vector field on the one-step
target?  python sanity_vf.py --dataset taco
Teacher-forced one-step prediction Delta^_1 = v_phi(C_t, G, tau_local,t) of the existing VF
checkpoints (vf.pt: one-step recipe; vf_unroll8.pt: the rollout recipe of every table) on the
same test pairs, compared with the F0 diagnostic model's h = 1 head (seed mean) per frame class.
Writes OUT/<ds>/results/f0_vs_existing_vf.csv
"""
from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd
import torch

import hp_common as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sanity_vf")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=P.DATASETS)
    a = ap.parse_args(); ds = a.dataset
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fd = P.fold(ds, dev)
    import hc_common as H  # noqa: E402  (after fold() put the hier dir first on sys.path with DC_DATASET set)
    from rollout_eval import load  # noqa: E402
    n_te, t_te = fd.frame_pairs("test")
    L = pd.read_csv(P.ds_out(ds) / "delta_contact_predictions.csv")
    L1 = L[(L.h == 1) & (L.cond == "F0")].reset_index(drop=True)
    rows = []
    for tag in ("", "_unroll8"):
        p = H.CKPT / "fixed0" / f"vf{tag}.pt"
        if not p.exists():
            continue
        net, ck = load("vf", fd, dev, tag)
        with torch.no_grad():
            pred = torch.cat([net(fd.C[n_te[i:i + 4096], t_te[i:i + 4096]], fd.S[n_te[i:i + 4096]], fd.context(n_te[i:i + 4096], t_te[i:i + 4096])) for i in range(0, len(n_te), 4096)])
            tgt = fd.C[n_te, t_te + 1] - fd.C[n_te, t_te]
            e = ((pred - tgt).norm(dim=1) * fd.stats["s"]).cpu().numpy()
        assert len(e) == len(L1)
        for cls in ("all", "non_spike", "spike", "persistent_spatial", "persistent_mixed", "onset", "release", "transient", "amount"):
            m = np.ones(len(L1), bool) if cls == "all" else ((L1.frame_class != "non_spike").values if cls == "spike" else (L1.frame_class == cls).values)
            if m.sum() == 0:
                continue
            rows.append(dict(dataset=ds, frame_class=cls, n_frames=int(m.sum()), existing_vf=f"vf{tag}", E1_existing_vf=float(e[m].mean()), E1_F0=float(L1.E.values[m].mean()),
                             E1_zero=float(L1.E0.values[m].mean()), best_val_existing=float(ck.get("best_val", np.nan)), selection_existing=ck.get("selection", "")))
    out = pd.DataFrame(rows); out.to_csv(P.ds_out(ds) / "results" / "f0_vs_existing_vf.csv", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
