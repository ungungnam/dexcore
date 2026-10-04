#!/usr/bin/env python
"""Step 2 of the BimArt -> OakInk2 port: pose cache, mesh dictionary, pair roles and pair scales.

Two passes, because the scale depends on the roles and the roles depend on the whole pair:

  pass 1  per recording: read the annotation pickle once, keep the hands and every object's pose at
          the frames the two-object segments use (30 Hz), write <out>/cache/<recording>.npz, and
          measure each pair object's vertex-centroid travel per segment.
  roles   per pair: the names decide where they can (manifest `name_part0`); otherwise the object
          that travels LESS over all of the pair's segments is part1, the canonical frame.
  pass 2  per segment, from the small cache: the largest radius each part reaches from part1's
          origin in part1's frame. Per pair, the one scale is 0.85 / (largest radius over every
          segment of the pair), so both parts are inside the basis in every frame (B3).

  python scripts/cache_oakink2_bimart.py --workers 12

Writes <out>/cache/*.npz, <out>/assets/oakink2_mesh_dict.npy, <out>/pairs.csv,
<out>/segments_geom.csv and <out>/cache_summary.json. Pass 1 reads ~30 GB of pickles.
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
import os
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger("cache")
_G = {}          # read-only state inherited by forked workers


def _pass1(task):
    """One recording -> cache file + per-segment centroid travel of both pair objects."""
    rec, segs = task
    from src.analysis.bimart import oakink2 as O

    md, out = _G["md"], _G["out"]
    path = out / "cache" / f"{rec}.npz"
    try:
        if path.exists():
            with np.load(path, allow_pickle=False) as z:
                cache = {k: z[k] for k in z.files}
        else:
            frames = sorted({f for s in segs for f in range(s["lo"], s["hi"], O.STEP)})
            cache = O.load_recording(rec, frames)
            np.savez_compressed(path, **cache)
        bad = [k for k, v in cache.items() if v.dtype.kind == "f" and not np.isfinite(v).all()]
        if bad:
            raise ValueError(f"non-finite values in {bad}")
        # the program's per-segment obj_list must name objects the annotation actually tracks
        absent = sorted({o for s in segs for o in (s["obj_a"], s["obj_b"])} - set(cache["obj_ids"].tolist()))
        if absent:
            raise ValueError(f"segment objects not in the recording's obj_list: {absent}")
        rows = []
        for s in segs:
            rows.append({"segment_id": s["segment_id"],
                         "travel_a": O.centroid_travel(cache, md, s["obj_a"], s["lo"], s["hi"]),
                         "travel_b": O.centroid_travel(cache, md, s["obj_b"], s["lo"], s["hi"]),
                         "frames_cached": int(len(O.frame_index(cache, s["lo"], s["hi"])))})
        return rec, rows, None
    except Exception as e:                          # noqa: BLE001 -- reported, then counted
        return rec, [], f"{type(e).__name__}: {e}"


def _pass2(task):
    """One recording's segments -> the radius each part reaches in part1's frame."""
    rec, segs = task
    from src.analysis.bimart import oakink2 as O

    md, out = _G["md"], _G["out"]
    with np.load(out / "cache" / f"{rec}.npz", allow_pickle=False) as z:
        cache = {k: z[k] for k in z.files}
    rows = []
    for s in segs:
        r0, r1 = O.part_radius(cache, md, s["part0"], s["part1"], s["lo"], s["hi"])
        rows.append({"segment_id": s["segment_id"], "r0_max": float(r0.max()), "r1": r1,
                     "r_union": float(max(r0.max(), r1))})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/result/uhnam/dexcore/oakink2/30_bimart_port")
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    import multiprocessing as mp

    import pandas as pd

    from src.analysis.bimart import oakink2 as O
    from src.analysis.bimart.features import BASIS_RADIUS

    out = Path(args.port)
    (out / "cache").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    seg = pd.read_csv(out / "manifest/segments.csv")

    # ---- every mesh the recordings can touch: pair objects, and the others for the B5 filter
    md_path = out / "assets/oakink2_mesh_dict.npy"
    if md_path.exists():
        md = np.load(md_path, allow_pickle=True).item()
    else:
        desc = json.load(open(O.DESC))
        md = {}
        for d in sorted(p for p in O.MESHES.iterdir() if p.is_dir()):
            try:
                v, f = O.load_mesh(d.name)
            except FileNotFoundError:
                continue
            md[d.name] = {"verts": v, "faces": f, "name": desc.get(d.name, {}).get("obj_name", d.name)}
        np.save(md_path, md, allow_pickle=True)
    missing = sorted((set(seg.obj_a) | set(seg.obj_b)) - set(md))
    if missing:
        raise SystemExit(f"pair objects without a mesh: {missing}")
    nv = np.array([len(m["verts"]) for m in md.values()])
    log.info("mesh dict: %d meshes, %d-%d verts (median %d)", len(md), nv.min(), nv.max(), np.median(nv))
    _G.update(md=md, out=out)

    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    ctx = mp.get_context("fork")
    nproc = max(1, min(args.workers, avail))

    # ---- pass 1
    tasks = [(rec, g.to_dict("records")) for rec, g in seg.groupby("recording")]
    trav, errors = [], {}
    with ctx.Pool(processes=nproc) as pool:
        for i, (rec, rows, err) in enumerate(pool.imap_unordered(_pass1, tasks), 1):
            trav += rows
            if err:
                errors[rec] = err
                log.warning("%s: %s", rec, err)
            if i % 25 == 0 or i == len(tasks):
                log.info("pass 1: %d/%d recordings", i, len(tasks))
    if errors:
        raise SystemExit(f"{len(errors)} recordings failed in pass 1: {errors}")
    seg = seg.merge(pd.DataFrame(trav), on="segment_id", how="left")
    short = seg[seg.frames_cached != seg.n_frames]
    if len(short):
        raise SystemExit(f"{len(short)} segments are missing cached frames")

    # ---- roles, fixed per pair
    pairs = []
    for pair, g in seg.groupby("pair"):
        a, b = g.obj_a.iloc[0], g.obj_b.iloc[0]
        ta, tb = float(g.travel_a.sum()), float(g.travel_b.sum())
        named = g.name_part0.dropna().unique()
        if len(named):
            rule, part0 = "name", (a if named[0] == "a" else b)
        else:
            rule, part0 = "travel", (a if ta > tb else b)      # the one that travels more moves
        part1 = b if part0 == a else a
        pairs.append({"pair": pair, "part0": part0, "part1": part1, "role_rule": rule,
                      "name_part0": md[part0]["name"], "name_part1": md[part1]["name"],
                      "part_group": bool(g.part_group.iloc[0]), "segments": len(g),
                      "travel_part0": ta if part0 == a else tb, "travel_part1": tb if part0 == a else ta})
    pairs = pd.DataFrame(pairs)
    seg = seg.merge(pairs[["pair", "part0", "part1", "role_rule"]], on="pair", how="left")

    # ---- pass 2
    tasks = [(rec, g.to_dict("records")) for rec, g in seg.groupby("recording")]
    geom = []
    with ctx.Pool(processes=nproc) as pool:
        for i, rows in enumerate(pool.imap_unordered(_pass2, tasks), 1):
            geom += rows
            if i % 50 == 0 or i == len(tasks):
                log.info("pass 2: %d/%d recordings", i, len(tasks))
    seg = seg.merge(pd.DataFrame(geom), on="segment_id", how="left")

    # ---- one scale per pair (B3); the per-segment scale is kept only as a comparison column
    rmax = seg.groupby("pair").r_union.max().rename("r_pair")
    pairs = pairs.merge(rmax, left_on="pair", right_index=True)
    pairs["scale"] = BASIS_RADIUS / pairs.r_pair
    seg = seg.merge(pairs[["pair", "scale"]], on="pair")
    seg["scale_segment"] = BASIS_RADIUS / seg.r_union
    seg["part1_radius_in_basis"] = seg.r1 * seg.scale
    seg["union_radius_in_basis"] = seg.r_union * seg.scale
    assert (seg.union_radius_in_basis <= BASIS_RADIUS + 1e-9).all(), "a part leaves the basis"
    pairs["part1_radius_in_basis"] = pairs.merge(
        seg.groupby("pair").r1.first(), left_on="pair", right_index=True).r1 * pairs.scale

    pairs.to_csv(out / "pairs.csv", index=False)
    seg.to_csv(out / "segments_geom.csv", index=False)
    csize = sum(f.stat().st_size for f in (out / "cache").glob("*.npz"))
    summary = {
        "written": datetime.now().isoformat(timespec="seconds"),
        "recordings_cached": int(seg.recording.nunique()), "cache_gb": round(csize / 1e9, 2),
        "meshes": len(md), "pairs": len(pairs),
        "pair_roles": pairs.role_rule.value_counts().to_dict(),
        "scale_ratio_segment_over_pair": {
            grp: {q: round(float(v), 2) for q, v in
                  (seg[seg.part_group == pg].scale_segment / seg[seg.part_group == pg].scale)
                  .quantile([.5, .9, 1.0]).items()}
            for grp, pg in (("part_groups", True), ("distinct_pairs", False))},
        "part1_radius_in_basis": {
            grp: {q: round(float(v), 3) for q, v in
                  seg[seg.part_group == pg].part1_radius_in_basis.quantile([0, .1, .5]).items()}
            for grp, pg in (("part_groups", True), ("distinct_pairs", False))},
    }
    (out / "cache_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
