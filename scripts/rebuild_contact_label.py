#!/usr/bin/env python
"""Rebuild the store and the normalisation statistics with the corrected contact label.

The dense per-vertex contact is already archived in sequences/*.npz, so nothing needs
re-preprocessing: only the 1024 indices the label is read at change (see
src/analysis/bimart/fps_contact.py for why). The BPS input, the hand action and the global states
are copied across byte-for-byte, which keeps this a controlled change -- the contact label and the
motion model's contact conditioning are the only things that differ from the previous run.

  python scripts/rebuild_contact_label.py --out-store train_store_fps --tag _fps
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/result/uhnam/dexcore/bimart_taco")
    p.add_argument("--out-store", default="train_store_fps")
    p.add_argument("--tag", default="_fps", help="suffix for the stat files")
    p.add_argument("--check", type=int, default=24, help="sequences to verify against dense truth")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("rebuild")

    import pandas as pd

    from src.analysis.bimart import fps_contact as F
    from src.analysis.bimart import meshes as M
    from src.analysis.bimart import stats as S

    root = Path(args.root)
    md = M.load(root / "assets" / "taco_mesh_dict.npy")
    index = pd.read_csv(root / "sequence_index.csv")

    # ---- 1. the sample sets, one per mesh
    tbl_path = root / "assets" / f"taco_fps_index{args.tag}.npy"
    if tbl_path.exists():
        table = F.load_table(tbl_path); log.info("fps table loaded (%d meshes)", len(table))
    else:
        log.info("building fps table for %d meshes", len(md))
        table = F.build_table(md)
        np.save(tbl_path, table, allow_pickle=True); log.info("-> %s", tbl_path)

    # ---- 2. the store: contact re-gathered, everything else copied
    out_dir = root / args.out_store
    out_dir.mkdir(parents=True, exist_ok=True)
    total = int(index["n_frames"].sum())
    ARRAYS = {"bps": 3072, "contact": 2048, "action": 1200, "global": 8}
    mm = {k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+", dtype=np.float32,
                                       shape=(total, d)) for k, d in ARRAYS.items()}
    rows, at, checks = [], 0, []
    for i, r in index.iterrows():
        inds = F.gather_indices(table, md, r["tool_mesh"], r["target_mesh"])
        with np.load(root / "sequences" / r["file"]) as z:
            n = len(z["kp"])
            mm["bps"][at:at + n] = z["obj_cano_bps"].reshape(n, -1)
            mm["contact"][at:at + n] = F.gather(z, inds)
            mm["action"][at:at + n] = np.concatenate([z["kp"], z["dirvec"]], axis=-1)
            mm["global"][at:at + n] = z["global_states"]
            if len(checks) < args.check:
                tk = F.mesh_key(md, r["tool_mesh"]); n_tool = len(md[tk]["verts_original"])
                bind = z["obj_cano_bps_inds"]; rr = np.arange(n)[:, None]
                for hand, c in (("L", z["contact_left"]), ("R", z["contact_right"])):
                    old = c[rr, bind]; new = c[:, inds]
                    checks.append({"sequence_id": r["sequence_id"], "hand": hand,
                                   "target_cat": r["target_cat"],
                                   "dense_tool": float((c[:, :n_tool].min(1) < .01).mean()),
                                   "old_tool": float((old[:, :512].min(1) < .01).mean()),
                                   "new_tool": float((new[:, :512].min(1) < .01).mean()),
                                   "dense_targ": float((c[:, n_tool:].min(1) < .01).mean()),
                                   "old_targ": float((old[:, 512:].min(1) < .01).mean()),
                                   "new_targ": float((new[:, 512:].min(1) < .01).mean())})
        rows.append({"sequence_id": r["sequence_id"], "start": at, "n_frames": n})
        at += n
        if (i + 1) % 250 == 0 or i + 1 == len(index):
            log.info("  store %d/%d (%d rows)", i + 1, len(index), at)
    for v in mm.values():
        v.flush()
    off = index.merge(pd.DataFrame(rows), on="sequence_id", suffixes=("", "_store"))
    off.to_csv(out_dir / "offsets.csv", index=False)
    log.info("store at %s: %d rows", out_dir, at)

    C = pd.DataFrame(checks)
    C.to_csv(out_dir / "label_check.csv", index=False)
    log.info("label vs dense truth on %d sequence-hands: tool |old-dense| %.4f -> |new-dense| %.4f"
             " | target %.4f -> %.4f", len(C),
             (C.old_tool - C.dense_tool).abs().mean(), (C.new_tool - C.dense_tool).abs().mean(),
             (C.old_targ - C.dense_targ).abs().mean(), (C.new_targ - C.dense_targ).abs().mean())

    # ---- 3. statistics, from the same corrected label
    log.info("statistics over %d sequences", len(index))
    acc = {k: S._Running() for k in ("contact_action", "obj_feat", "global_states", "motion_action")}
    for i, r in index.iterrows():
        inds = F.gather_indices(table, md, r["tool_mesh"], r["target_mesh"])
        with np.load(root / "sequences" / r["file"]) as z:
            n = len(z["kp"])
            acc["contact_action"].update(F.gather(z, inds))
            acc["obj_feat"].update(z["obj_cano_bps"].reshape(n, -1))
            acc["global_states"].update(z["global_states"])
            acc["motion_action"].update(np.concatenate([z["kp"], z["dirvec"]], axis=-1))
        if (i + 1) % 500 == 0 or i + 1 == len(index):
            log.info("  stats %d/%d", i + 1, len(index))
    f = {k: v.finish() for k, v in acc.items()}

    def cp(k):
        return {"mean": f[k]["mean"].copy(), "std": f[k]["std"].copy()}

    gs = {"mean": f["global_states"]["mean"].reshape(-1), "std": f["global_states"]["std"].reshape(-1)}
    contact = {"action": cp("contact_action"), "prev_contact": cp("contact_action"),
               "obj_feat": cp("obj_feat"), "curr_global_states": dict(gs),
               "prev_global_states": dict(gs), "prev_obj_feat": cp("obj_feat")}
    motion = {"action": cp("motion_action"), "contact_points": cp("contact_action"),
              "object": cp("obj_feat"), "global_states": dict(gs)}
    for name, d in (("contact", contact), ("motion", motion)):
        path = root / "assets" / f"taco_{name}_norm_stats{args.tag}.pkl"
        with open(path, "wb") as fh:
            pickle.dump(d, fh)
        log.info("-> %s", path)

    old = pickle.load(open(root / "assets" / "taco_motion_norm_stats.pkl", "rb"))
    a, b = np.asarray(old["contact_points"]["mean"]).ravel(), motion["contact_points"]["mean"].ravel()
    log.info("contact-label mean shifted by %.4f m on average (max %.4f); std ratio mean %.3f",
             np.abs(a - b).mean(), np.abs(a - b).max(),
             float((motion["contact_points"]["std"].ravel() /
                    np.asarray(old["contact_points"]["std"]).ravel()).mean()))
    print(f"\nstore   {out_dir}\nstats   {root}/assets/taco_{{contact,motion}}_norm_stats{args.tag}.pkl")


if __name__ == "__main__":
    main()
