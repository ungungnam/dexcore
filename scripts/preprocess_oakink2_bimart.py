#!/usr/bin/env python
"""Step 4 of the BimArt -> OakInk2 port: every two-object segment into BimArt's feature format.

  python scripts/preprocess_oakink2_bimart.py --workers 24

Reads the pose cache, the mesh dictionary and `segments_geom.csv` written by
`cache_oakink2_bimart.py`, and writes one archive per segment with `oakink2.segment_features`:

    obj_cano_bps (T,1024,3) f32    part0 slots [0:512] per frame, part1 [512:1024] once, repeated
    obj_cano_bps_inds (T,1024) i32 into the concatenated [part0, part1] vertex axis
    contact (T,2048) f32           dense hand-to-vertex distance gathered at the indices, lh | rh
    global_abs (T,7) f64           part1 rotvec, part1 ABSOLUTE translation (m), pair scale
    kp, dirvec (T,600) f32         100 MANO vertices per hand and their offsets, part1 frame
    nonpair_dist (T,2,n) f32       each hand's closest approach to every other tracked object (B5)

The label is gathered here, once, as the TACO store does; the dense per-vertex distance and the
world hand vertices are kept only for a seeded 5% of segments, so the label checks can be repeated
on the full run without storing ~7x the data.

Writes <port>/sequences/*.npz, <port>/sequence_index.csv and <port>/preprocess_summary.json.
Existing archives are kept, so an interrupted run resumes.
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

log = logging.getLogger("preprocess")
_G = {}          # read-only state inherited by forked workers
DENSE_FRACTION = 0.05
DENSE_SEED = 0


def seq_file(segment_id: str) -> str:
    return f"{segment_id.replace('#', '__')}.npz"


def _worker(task):
    """One recording's segments -> one archive each. The cache is read once per recording."""
    rec, segs = task
    from src.analysis.bimart import oakink2 as O

    if "layers" not in _G:                          # built per worker: torch does not survive fork
        import torch
        torch.set_num_threads(1)
        _G["layers"] = O.mano_layers("cpu")
    md, out = _G["md"], _G["out"]
    rows, errors, cache = [], {}, None
    for s in segs:
        path = out / "sequences" / seq_file(s["segment_id"])
        try:
            if not path.exists():
                if cache is None:
                    with np.load(out / "cache" / f"{rec}.npz", allow_pickle=False) as z:
                        cache = {k: z[k] for k in z.files}
                f = O.segment_features(cache, md, s["part0"], s["part1"], s["lo"], s["hi"], s["scale"],
                                       _G["basis"], _G["hand_index"], _G["layers"],
                                       keep_dense=s["dense"], nonpair=True)
                bad = [k for k, v in f.items() if v.dtype.kind == "f" and not k.startswith("nonpair")
                       and not np.isfinite(v).all()]      # nonpair is +inf with no other object
                if bad:
                    raise ValueError(f"non-finite values in {bad}")
                tmp = path.with_suffix(".tmp.npz")
                np.savez_compressed(tmp, **f)
                os.replace(tmp, path)
            with np.load(path, allow_pickle=False) as z:
                rows.append({"segment_id": s["segment_id"], "file": path.name,
                             "n_frames_saved": int(len(z["kp"])),
                             "n_nonpair_objects": int(z["n_nonpair_objects"]),
                             "dense_saved": "contact_dense_lh" in z.files})
        except Exception as e:                      # noqa: BLE001 -- reported, then counted
            errors[s["segment_id"]] = f"{type(e).__name__}: {e}"
    return rows, errors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/result/uhnam/dexcore/oakink2/30_bimart_port")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--limit", type=int, default=None, help="the N shortest recordings, for a smoke run")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    import multiprocessing as mp

    import pandas as pd

    from src.analysis.bimart import features as F

    out = Path(args.port)
    (out / "sequences").mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(out / "segments_geom.csv")
    pairs = pd.read_csv(out / "pairs.csv")[["pair", "name_part0", "name_part1"]]
    seg = seg.drop(columns=["name_part0"]).merge(pairs, on="pair", how="left")

    # the dense subset is drawn once over every segment, seeded, so a resumed run picks the same
    rng = np.random.default_rng(DENSE_SEED)
    dense_ids = set(rng.choice(seg.segment_id.to_numpy(), size=int(round(DENSE_FRACTION * len(seg))),
                               replace=False))
    seg["dense"] = seg.segment_id.isin(dense_ids)

    _G.update(md=np.load(out / "assets/oakink2_mesh_dict.npy", allow_pickle=True).item(), out=out,
              basis=F.load_basis(), hand_index=F.load_hand_index())
    cols = ["segment_id", "part0", "part1", "lo", "hi", "scale", "dense"]
    tasks = [(rec, g[cols].to_dict("records")) for rec, g in seg.groupby("recording")]
    tasks.sort(key=lambda t: -sum(s["hi"] - s["lo"] for s in t[1]))   # longest first: less tail
    if args.limit:
        tasks = tasks[-args.limit:]

    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    nproc = max(1, min(args.workers, avail))
    log.info("%d recordings, %d segments (%d dense), %d workers", len(tasks),
             sum(len(t[1]) for t in tasks), sum(s["dense"] for t in tasks for s in t[1]), nproc)
    rows, errors = [], {}
    with mp.get_context("fork").Pool(processes=nproc) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(_worker, tasks), 1):
            rows += r
            errors.update(e)
            for sid, msg in e.items():
                log.warning("%s: %s", sid, msg)
            if i % 20 == 0 or i == len(tasks):
                log.info("  %d/%d recordings, %d segments written", i, len(tasks), len(rows))
    if errors:
        (out / "preprocess_errors.json").write_text(json.dumps(errors, indent=1))
        raise SystemExit(f"{len(errors)} segments failed; see preprocess_errors.json")

    index = seg.merge(pd.DataFrame(rows), on="segment_id", how="inner")
    short = index[index.n_frames_saved != index.n_frames]
    if len(short):
        raise SystemExit(f"{len(short)} archives have the wrong frame count, e.g. {short.segment_id.tolist()[:3]}")
    if (index.dense != index.dense_saved).any():
        raise SystemExit("the dense subset on disk does not match the seeded draw; delete those archives")
    index = index.drop(columns=["n_frames_saved", "dense_saved"]).rename(columns={"segment_id": "sequence_id"})
    if args.limit:
        log.info("smoke run: %d segments, index not written", len(index))
        return
    index.to_csv(out / "sequence_index.csv", index=False)

    size = sum(f.stat().st_size for f in (out / "sequences").glob("*.npz"))
    (out / "preprocess_summary.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "n_sequences": int(len(index)), "n_frames": int(index.n_frames.sum()),
        "split_sequences": index.split.value_counts().to_dict(),
        "dense_subset": {"fraction": DENSE_FRACTION, "seed": DENSE_SEED, "n": int(index.dense.sum())},
        "sequences_gb": round(size / 1e9, 2),
        "canonical_frame": "part1", "global_abs": "[part1 rotvec, part1 absolute translation m, pair scale]",
        "bps_points": 512, "bps_feature_dim": 3, "action_dim": 1200, "contact_dim": 2048,
        "frame_rate_hz": 30, "mocap_step": 4,
    }, indent=2))
    log.info("done: %d segments, %.1f GB -> %s", len(index), size / 1e9, out / "sequences")


if __name__ == "__main__":
    main()
