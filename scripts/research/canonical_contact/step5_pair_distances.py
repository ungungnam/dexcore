"""Step 5: canonical-space distances for every pair in contact_pairs.csv.

Output: <root>/contact_pair_distances.csv -- one row per pair (key columns + pair attributes) with
  <backend>_l1, <backend>_l2, <backend>_cosine   (on X_soft)
  <backend>_hard_l2                               (on X_hard)
for every backend, plus Baseline A's chamfer_raw_m / chamfer_scalednorm merged in, so ONE table
holds every representation's distance for every pair. Pairs whose samples are missing from a
backend's cache (or filtered by --min-touch) get NaN for that backend.

Usage: python scripts/step5_pair_distances.py [--root R] [--backends ...] [--min-touch 0.2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import (PAIR_ATTRS, PAIR_KEY, add_common_args, list_cache_files, load_cache,  # noqa: E402
                       pair_distances, read_csv_comments, read_pairs, resolve_backends, sample_key_index,
                       setup_logging)

log = setup_logging("step5")


def distances_for_backend(root: Path, backend: str, pairs: pd.DataFrame, min_touch: float) -> pd.DataFrame:
    cols = {f"{backend}_l1": np.nan, f"{backend}_l2": np.nan, f"{backend}_cosine": np.nan,
            f"{backend}_hard_l2": np.nan}
    out = pd.DataFrame(cols, index=pairs.index)
    files = list_cache_files(root, backend)
    if not files:
        log.warning("backend %s: no cache files found", backend)
        return out
    n_hit = 0
    for cat, role, path in files:
        sel = (pairs.category == cat) & (pairs.role == role)
        if not sel.any():
            continue
        c = load_cache(path, min_touch=min_touch)
        idx = sample_key_index(c["meta"])
        sub = pairs[sel]
        ia = np.array([idx.get((s, int(w), h), -1) for s, w, h in zip(sub.seq_i, sub.window_i, sub.hand)])
        ib = np.array([idx.get((s, int(w), h), -1) for s, w, h in zip(sub.seq_j, sub.window_j, sub.hand)])
        ok = (ia >= 0) & (ib >= 0)
        if not ok.all():
            log.warning("%s/%s__%s: %d/%d pairs have a sample missing from the cache", backend, cat, role,
                        int((~ok).sum()), len(sub))
        if not ok.any():
            continue
        d = pair_distances(c["X_soft"], ia[ok], ib[ok])
        rows = sub.index[ok]
        out.loc[rows, f"{backend}_l1"] = d["l1"]
        out.loc[rows, f"{backend}_l2"] = d["l2"]
        out.loc[rows, f"{backend}_cosine"] = d["cosine"]
        if c["X_hard"] is not None:
            out.loc[rows, f"{backend}_hard_l2"] = pair_distances(c["X_hard"], ia[ok], ib[ok], ("l2",))["l2"]
        n_hit += int(ok.sum())
        log.info("%s/%s__%s: %d pairs", backend, cat, role, int(ok.sum()))
    log.info("backend %s: distances for %d/%d pairs", backend, n_hit, len(pairs))
    return out


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--out", type=Path, default=None, help="default <root>/contact_pair_distances.csv")
    args = ap.parse_args()
    root = args.root
    backends = resolve_backends(root, args.backends)
    pairs = read_pairs(root)
    log.info("%d pairs; backends %s", len(pairs), backends)
    attrs = [c for c in PAIR_ATTRS if c in pairs.columns]
    table = pairs[PAIR_KEY + attrs].copy()
    for b in backends:
        table = pd.concat([table, distances_for_backend(root, b, pairs, args.min_touch)], axis=1)
    # Baseline A distances (same key columns)
    bpath = root / "baselineA_distances.csv"
    if bpath.exists():
        ba = read_csv_comments(bpath)
        for c in ("window_i", "window_j"):
            ba[c] = ba[c].astype(int)
        ba = ba[PAIR_KEY + ["chamfer_raw_m", "chamfer_scalednorm"]].drop_duplicates(PAIR_KEY)
        table = table.merge(ba, on=PAIR_KEY, how="left", validate="one_to_one")
        log.info("merged baseline A distances for %d pairs", int(table.chamfer_raw_m.notna().sum()))
    else:
        log.warning("no baselineA_distances.csv at %s", bpath)
    out = args.out or root / "contact_pair_distances.csv"
    table.to_csv(out, index=False)
    log.info("wrote %s (%d rows, %d cols)", out, len(table), table.shape[1])


if __name__ == "__main__":
    main()
