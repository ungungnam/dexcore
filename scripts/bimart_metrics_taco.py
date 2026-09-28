#!/usr/bin/env python
"""Score the TACO port on BimArt's OWN metrics, so the two are directly comparable.

BimArt does not report a position error. It is a generative model: given one object trajectory
many grasps are physically valid, so being 80 mm from the recorded hand says little. What it
reports instead, and what its pretrained ARCTIC checkpoint scores, is

    Penetration_1cm   % of frames where some hand vertex is more than 1 cm inside the object
    Accel             mean acceleration magnitude of the hand vertices, x100 -- jitter
    Contact           % of MOVING-object frames where some hand vertex is within 5 mm of the
                      surface, counted only while the object is actually moving
    Articulation      ARCTIC-only: contact with the articulated part. No TACO analogue.

The metric classes are imported from upstream rather than reimplemented, so a difference in the
numbers is a difference in the data and not in the definition. `pysdf` needs a closed mesh to give
a reliable sign, so TACO's two objects are evaluated one at a time and combined the way the metric
combines frames.
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
for _p in (_REPO, str(_pl.Path(_REPO) / "third_party/BimArt")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import logging
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="/result/uhnam/dexcore/bimart_taco")
    p.add_argument("--splits", nargs="*", default=["test_1", "test_2"])
    p.add_argument("--key", default="pred", choices=("pred", "gt"))
    p.add_argument("--infer-dir", default="inference",
                   help="subdirectory of --root holding the predictions, e.g. a larger re-run")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("metrics")

    import pandas as pd

    from src.analysis.bimart import meshes as M
    from src.analysis.bimart.guidance import _canonical_objects
    from src.analysis.loaders import taco
    from utils.metric_util import (ContactPercentageMetric, JitterMetric,
                                   PenetrationDepthMetric)

    root = Path(args.root)
    md = M.load(root / "assets" / "taco_mesh_dict.npy")
    refs = {r.sequence_id: r for r in taco.index()}
    rows = []
    for split in args.splits:
        z = np.load(root / args.infer_dir / split / "predictions.npz", allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        jit = JitterMetric()
        # one metric object per object role: pysdf needs one closed mesh at a time
        pen = {r: PenetrationDepthMetric(thres=0.01) for r in ("tool", "target")}
        con = {r: ContactPercentageMetric() for r in ("tool", "target")}
        watertight = {"tool": 0, "target": 0, "n": 0}

        for i, m in enumerate(meta):
            seq, start = m["sequence_id"], m["start"]
            hands = np.concatenate([z[f"{i}_{args.key}_left"], z[f"{i}_{args.key}_right"]], axis=1)
            jit.update(hands)

            traj = taco.load(refs[seq], with_hands=False)
            objv = _canonical_objects(root, seq)[start:start + len(hands)]
            n_tool = len(md[traj.tool.name]["verts_original"])
            watertight["n"] += 1
            for role, name, sl in (("tool", traj.tool.name, slice(0, n_tool)),
                                   ("target", traj.target.name, slice(n_tool, None))):
                faces = md[name]["faces"]
                import trimesh
                mesh = trimesh.Trimesh(md[name]["verts_original"], faces, process=False)
                if not mesh.is_watertight:
                    continue
                watertight[role] += 1
                verts = np.ascontiguousarray(objv[:, sl])
                pen[role].update(hands, verts, faces)
                con[role].update(hands, verts, faces)

        rec = {"split": split, "key": args.key, "windows": len(meta),
               "Accel": jit.compute(),
               "Penetration_1cm_tool": pen["tool"].compute(),
               "Penetration_1cm_target": pen["target"].compute(),
               "Contact_tool": con["tool"].compute(), "Contact_target": con["target"].compute(),
               "watertight_tool": watertight["tool"], "watertight_target": watertight["target"],
               "sequences": watertight["n"]}
        # a frame counts as in contact if EITHER object is touched, which is what the single-object
        # metric measures on ARCTIC
        c_hit = con["tool"].con + con["target"].con
        c_tot = con["tool"].total + con["target"].total
        rec["Contact_either"] = round(100.0 * c_hit / c_tot, 3) if c_tot else None
        p_hit = pen["tool"].pn + pen["target"].pn
        p_tot = pen["tool"].total + pen["target"].total
        rec["Penetration_1cm_either"] = round(100.0 * p_hit / p_tot, 4) if p_tot else None
        rows.append(rec)
        log.info("%s [%s]: accel %.4f | pen %.2f%% | contact %.1f%% (watertight %d/%d tool, %d target)",
                 split, args.key, rec["Accel"], rec["Penetration_1cm_either"] or -1,
                 rec["Contact_either"] or -1, watertight["tool"], watertight["n"],
                 watertight["target"])

    df = pd.DataFrame(rows)
    out = Path(args.out or (root / "evaluation" / f"bimart_metrics_{args.key}.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
