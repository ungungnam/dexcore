#!/usr/bin/env python
"""Step 1 of the BimArt -> OakInk2 port: the segment manifest.

Every OakInk2 primitive segment that handles exactly two objects, with its frame range, split,
window counts and name-decided part roles. Nothing here reads the annotation pickles; the ranges
come from the program and program-extension files, the split from TaMF.

  python scripts/manifest_oakink2_bimart.py

Writes <out>/segments.csv and <out>/manifest_summary.json, and asserts the counts the 2026-09-30
review measured, so a silent change in the data or the parsing fails loudly.
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

#: Counts measured by the review (plan.md M1, M1b, M3, M7). Any mismatch means the inputs changed.
EXPECTED = {"segments": 1128, "recordings": 458, "part_groups": 608, "distinct_pairs": 520,
            "split_segments": {"train": 767, "val": 67, "test": 294},
            "split_recordings": {"train": 305, "val": 30, "test": 123},
            "train_windows": 316915, "frames": 534573, "segments_with_windows": 1114}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/result/uhnam/dexcore/oakink2/30_bimart_port/manifest")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("manifest")

    import pandas as pd

    from src.analysis.bimart import oakink2 as O

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(O.segments())
    df["pair"] = df["obj_a"] + "|" + df["obj_b"]
    df["name_part0"] = [O.name_roles(a, b, g) for a, b, g in zip(df.name_a, df.name_b, df.part_group)]
    df["segment_id"] = [f"{r}#{i}" for i, r in zip(df.groupby("recording").cumcount(), df.recording)]

    got = {"segments": len(df), "recordings": df.recording.nunique(),
           "part_groups": int(df.part_group.sum()), "distinct_pairs": int((~df.part_group).sum()),
           "split_segments": df.split.value_counts().to_dict(),
           "split_recordings": df.groupby("split").recording.nunique().to_dict(),
           "train_windows": int(df.loc[df.split == "train", "n_train_windows"].sum()),
           "frames": int(df.n_frames.sum()), "segments_with_windows": int((df.n_train_windows > 0).sum())}
    bad = {k: (got[k], v) for k, v in EXPECTED.items() if got[k] != v}
    if bad:
        raise SystemExit(f"manifest does not match the review's counts: {bad}")

    # a pair's role must not depend on which segment is looked at; names are per object, so this
    # holds by construction, but the travel-decided pairs are resolved later and must be too
    by_pair = df.dropna(subset=["name_part0"]).groupby("pair").name_part0.nunique()
    assert (by_pair == 1).all(), f"pairs with conflicting name roles: {list(by_pair[by_pair > 1].index)}"
    assert not df.groupby("recording").split.nunique().gt(1).any(), "a recording sits in two splits"
    # the extended range (B5) needs an approach/retreat entry; without one the segment would
    # silently fall back to its interaction-only range
    assert df.has_extension.all(), f"segments without an extension entry: {df.loc[~df.has_extension, 'segment_id'].tolist()[:5]}"

    df.to_csv(out / "segments.csv", index=False)
    summary = {"written": datetime.now().isoformat(timespec="seconds"), **got,
               "pairs": int(df.pair.nunique()), "objects": int(len(set(df.obj_a) | set(df.obj_b))),
               "segments_named_roles": int(df.name_part0.notna().sum()),
               "segments_travel_roles": int(df.name_part0.isna().sum()),
               "test_windows_upstream_rule": df.groupby("split").n_test_windows.sum().to_dict(),
               "segments_zero_test_windows": df[df.n_test_windows == 0].split.value_counts().to_dict()}
    (out / "manifest_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("%d segments, %d pairs, %d objects -> %s", len(df), summary["pairs"],
             summary["objects"], out)
    log.info("roles: %d by name, %d by travel (resolved in the cache step)",
             summary["segments_named_roles"], summary["segments_travel_roles"])


if __name__ == "__main__":
    main()
