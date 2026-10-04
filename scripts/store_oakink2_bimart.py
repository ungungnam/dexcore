#!/usr/bin/env python
"""Step 5 of the BimArt -> OakInk2 port: the flat training store and the normalisation statistics.

  python scripts/store_oakink2_bimart.py

Builds <port>/train_store/*.npy + offsets.csv from <port>/sequences, then computes the statistics
on TRAIN windows only (after the B5 non-pair filter), window-weighted, with each model's own
translation reference -- see src/analysis/bimart/oakink2_data.py. Writes
<port>/assets/oakink2_{contact,motion}_norm_stats.pkl and <port>/stats_summary.json.
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/result/uhnam/dexcore/oakink2/30_bimart_port")
    p.add_argument("--skip-store", action="store_true", help="reuse an existing train_store")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("store")

    import pandas as pd

    from src.analysis.bimart import oakink2_data as D

    port = Path(args.port)
    index = pd.read_csv(port / "sequence_index.csv")
    if not args.skip_store:
        D.build_store(port, index=index)
    store = D.open_store(port / "train_store")
    for k, v in store.items():
        if k != "nonpair" and not np.isfinite(v).all():
            raise SystemExit(f"non-finite values in store/{k}")

    stats = D.compute_stats(port)
    paths = D.save_stats(stats, port / "assets")

    # checks from review/plan.md step 6
    gm, gc = stats["motion"]["global_states"], stats["contact"]["curr_global_states"]
    tr_m, tr_c = gm["std"][3:6], gc["std"][0, 3:6]
    floored = stats["meta"]["floored_channels"]
    checks = {
        "translation_mean_motion_m": gm["mean"][3:6].round(5).tolist(),
        "translation_std_motion_m": tr_m.round(4).tolist(),
        "translation_mean_contact_m": gc["mean"][0, 3:6].round(5).tolist(),
        "translation_std_contact_m": tr_c.round(4).tolist(),
        "translation_std_cm_scale": bool(((tr_m > 0.005) & (tr_m < 0.5)).all()),
        "motion_and_contact_references_differ": bool(not np.allclose(tr_m, tr_c)),
        "floored_channels": floored,
        "stat_keys_contact": sorted(stats["contact"]), "stat_keys_motion": sorted(stats["motion"]),
        "shapes_motion": {k: list(v["mean"].shape) for k, v in stats["motion"].items()},
        "shapes_contact": {k: list(v["mean"].shape) for k, v in stats["contact"].items()},
    }
    summary = {"written": datetime.now().isoformat(timespec="seconds"), **stats["meta"],
               "rows": int(len(store["action"])), "checks": checks,
               "stat_paths": {k: str(v) for k, v in paths.items()}}
    (port / "stats_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("%s", json.dumps(summary, indent=2))
    if floored["bps"] or floored["action"]:
        log.warning("some bps/action channels are constant over train windows: %s", floored)


if __name__ == "__main__":
    main()
