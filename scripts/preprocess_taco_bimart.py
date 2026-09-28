#!/usr/bin/env python
"""Preprocess TACO into BimArt's feature format (everything up to, but not including, training).

  python scripts/preprocess_taco_bimart.py --workers 16

Writes one .npz per sequence plus the mesh dictionary, the normalisation statistics and an index.
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

#: Frames dropped from the head of each sequence. BimArt uses 100 to skip ARCTIC's T-pose lead-in;
#: TACO has no T-pose and a median length of 148, so 100 would discard most of the release.
DEFAULT_BASE_FRAME = 8
PRED_HORIZON = 64


def _worker(task):
    root, ref, out_dir, base_frame = task
    from src.analysis.action_structure import contact as C
    from src.analysis.bimart import features as F, meshes as M
    from src.analysis.loaders import taco

    g = _worker.__dict__
    if "md" not in g:
        g["md"] = M.load(g["mesh_path"])
        g["B"] = F.load_basis()
        g["HI"] = F.load_hand_index()
        g["L"] = {s: C.mano_layer(s) for s in ("left", "right")}
    try:
        traj = taco.load(ref, root=root)
        out = F.sequence_features(traj, g["md"], g["B"], g["HI"], g["L"])
    except Exception as e:                          # noqa: BLE001 -- reported, then skipped
        logging.warning("%s: %s: %s", ref.sequence_id, type(e).__name__, e)
        return None
    n = out["n_frames"]
    usable = max(0, n - base_frame - PRED_HORIZON)
    path = Path(out_dir) / f"{ref.sequence_id.replace('/', '__')}.npz"
    np.savez_compressed(
        path, obj_cano_bps=out["obj_cano_bps"], obj_cano_bps_inds=out["obj_cano_bps_inds"],
        global_states=out["global_states"], kp=out["kp"], dirvec=out["dirvec"],
        contact_left=out["contact_dict"]["left"]["dist"],
        contact_right=out["contact_dict"]["right"]["dist"])
    return {"sequence_id": ref.sequence_id, "triplet": out["triplet"], "verb": out["verb"],
            "tool_cat": out["tool_cat"], "target_cat": out["target_cat"],
            "tool_mesh": out["tool_mesh"], "target_mesh": out["target_mesh"],
            "n_frames": n, "n_windows": usable, "file": path.name}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None)
    p.add_argument("--out", default="/result/uhnam/dexcore/taco/10_bimart_gen1_original_label")
    p.add_argument("--mesh-dict", default=None)
    p.add_argument("--base-frame", type=int, default=DEFAULT_BASE_FRAME)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("bimart_pre")

    import multiprocessing as mp

    import pandas as pd

    from src.analysis.bimart import meshes as M
    from src.analysis.loaders import taco

    out = Path(args.out)
    (out / "sequences").mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    mesh_path = Path(args.mesh_dict) if args.mesh_dict else out / "assets/taco_mesh_dict.npy"
    if not mesh_path.exists():
        log.info("building mesh dictionary -> %s", mesh_path)
        M.save(M.build(root=args.root), mesh_path)

    root = Path(args.root or taco.default_root())
    refs = list(taco.index(root))
    if args.limit:
        refs = refs[: args.limit]
    tasks = [(root, r, out / "sequences", args.base_frame) for r in refs]
    _worker.__dict__["mesh_path"] = str(mesh_path)

    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    ctx = mp.get_context("fork")
    rows = []
    with ctx.Pool(processes=max(1, min(args.workers, avail))) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            if r is not None:
                rows.append(r)
            if i % 100 == 0 or i == len(tasks):
                log.info("  %d/%d", i, len(tasks))

    index = pd.DataFrame(rows)
    index.to_csv(out / "sequence_index.csv", index=False)
    kept = index[index["n_windows"] > 0]
    log.info("%d/%d sequences, %d usable windows, %d sequences too short",
             len(index), len(tasks), int(kept["n_windows"].sum()), int((index["n_windows"] == 0).sum()))
    (out / "preprocess_summary.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "n_sequences": int(len(index)), "n_windows": int(kept["n_windows"].sum()),
        "base_frame": args.base_frame, "pred_horizon": PRED_HORIZON,
        "canonical_frame": "tool", "global_state_dim": 8,
        "bps_points": 512, "bps_feature_dim": 3, "action_dim": 1200,
        "contact_dim_per_hand": int(np.load(out / "sequences" / rows[0]["file"])["contact_left"].shape[1]),
    }, indent=2))
    print(f"\n{len(index)} sequences -> {out}")
    print(index.groupby("verb")["n_windows"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
