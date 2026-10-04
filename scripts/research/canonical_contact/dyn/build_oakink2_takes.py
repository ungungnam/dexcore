#!/usr/bin/env python
"""OakInk2 -> per-take contact distances (stage 1 of the OakInk2 cache).

A take = one (sequence, hand, object) inside one program primitive segment (program_info key =
(lh_range, rh_range), each None or (start, end) in mocap frames at 120 Hz; frames are taken every 4th = 30 Hz). For every
frame the hand is posed with the repo's MANO layer (manopth, centre_idx 0, quaternions in wxyz —
verified: 0.03 cm hand-object minimum during a 'grip' segment) and the object scan (metres,
object frame) is placed with obj_transf; contact = per object vertex, distance to the nearest hand
vertex [m], exactly TACO's definition. Writes
  /result/uhnam/dexcore/oakink2/10_dynamic_contact/takes/<seq>__<lh|rh>__<obj>__<s>_<e>.npz
      contact (F, N) float16 [m, clipped at 0.5], frames (F,) mocap ids, obj_transf (F, 4, 4)
  .../takes_index.csv  one row per take (sequence, actor, hand, object, obj_name, category,
      primitive, interaction_mode, start, end, n_frames, n_verts, file)
Object meshes with more than MAX_VERTS vertices are subsampled (random, fixed seed) once per
object; the kept vertex ids are stored in .../meshes/<obj>.npz with the vertices.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/home/uhnam/workspace/dexcore")
from src.analysis.action_structure import contact as CT   # noqa: E402

ANNO = Path("/data/uhnam/oakink2/anno_preview")
PROG = Path("/data/uhnam/oakink2/extracted/program/program_info")
DESC = Path("/data/uhnam/oakink2/extracted/object_raw/obj_desc.json")
MESHES = Path("/result/uhnam/dexcore/oakink2/00_assets/object_raw/align_ds")
OUT = Path(os.environ.get("OAKINK2_OUT", "/result/uhnam/dexcore/oakink2/10_dynamic_contact"))
EXT = Path("/data/uhnam/oakink2/extracted/program_extension/extension")
EXTENDED = os.environ.get("OAKINK2_EXTENDED", "0") == "1"      # 1: approach start .. retreat end
MESH_CACHE = Path("/result/uhnam/dexcore/oakink2/10_dynamic_contact/meshes")
STEP, MAX_VERTS = 4, 5000
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("oakink2_takes")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def category(name: str) -> str:
    n = name.lower()
    for key in ["bottle cap", "bottle body", "teapot cap", "teapot body", "whiteboard pen cap", "whiteboard pen body",
                "toothpaste tube", "cap of toothpaste", "test tube holder", "test tube rack", "test tube", "microwave oven gate lever",
                "microwave oven gate", "microwave oven cavity", "laptop base", "laptop lid", "lamp base", "alcohol burner", "alcohol lamp",
                "glass rod", "asbestos mesh", "power socket", "whiteboard pen", "whiteboard eraser", "conical flask", "mug", "cup", "wineglass",
                "bottle", "beaker", "bowl", "plate", "teapot", "bread", "apple", "pear", "tongs", "scissors", "book", "paper", "drawer",
                "toothpaste", "knife", "spoon", "fork", "kettle", "lid", "box", "usb", "stapler", "plug", "pen", "brush", "spatula", "pan",
                "pot", "jar", "can", "flower", "vase", "donut", "mouse", "rod", "sponge", "ignitor", "scale", "tripod", "gamecontroler"]:
        if key in n:
            return key
    return n


def load_mesh(obj):
    p = MESH_CACHE / f"{obj}.npz"
    if p.exists():
        z = np.load(p); return z["verts"]
    files = sorted((MESHES / obj).glob("*.ply")) + sorted((MESHES / obj).glob("*.obj"))
    if not files:
        return None
    m = trimesh.load(files[0], force="mesh")
    V = np.asarray(m.vertices, dtype=np.float64)
    keep = np.arange(len(V))
    if len(V) > MAX_VERTS:
        keep = np.sort(np.random.default_rng(0).choice(len(V), MAX_VERTS, replace=False))
    np.savez_compressed(p, verts=V[keep].astype(np.float32), keep=keep, n_full=len(V), source=str(files[0]))
    return V[keep]


def mano_verts(layer, pc, tsl, betas):
    """pc (F,16,4) wxyz quaternions, tsl (F,3), betas (10,) -> (F,778,3) world metres."""
    q = pc[..., [1, 2, 3, 0]]
    aa = Rotation.from_quat(q.reshape(-1, 4)).as_rotvec().reshape(len(pc), 16, 3)
    th = torch.tensor(np.concatenate([aa[:, 0], aa[:, 1:].reshape(len(pc), 45)], 1), dtype=torch.float32, device=DEV)
    out = []
    with torch.no_grad():
        for i in range(0, len(th), 2048):
            v, _, _ = layer(th[i:i + 2048], torch.tensor(np.repeat(betas[None], len(th[i:i + 2048]), 0), dtype=torch.float32, device=DEV))
            out.append(v.cpu().numpy())
    return np.concatenate(out) / 1000.0 + tsl[:, None, :]


def main(limit=None, shard=0, n_shards=1):
    (OUT / "takes").mkdir(parents=True, exist_ok=True); MESH_CACHE.mkdir(exist_ok=True)
    desc = json.load(open(DESC))
    layers = {s: CT.mano_layer(s, DEV) for s in ("left", "right")}
    rows = []
    done = set(p.name for p in (OUT / "takes").glob("*.npz"))
    files = sorted(PROG.glob("*.json"))[:limit][shard::n_shards]
    index_path = OUT / (f"takes_index_shard{shard}of{n_shards}.csv" if n_shards > 1 else "takes_index.csv")
    t0 = time.time()
    for i, pf in enumerate(files):
        seq = pf.stem
        actor = seq.split("++")[0].split("__")[1]
        prog = json.load(open(pf))
        ext = json.load(open(EXT / pf.name)) if EXTENDED and (EXT / pf.name).exists() else {}
        segs = []
        for k, v in prog.items():
            # key = (lh_range, rh_range), each None or (start, end) in mocap frames
            try:
                lh_rng, rh_rng = ast.literal_eval(k)
            except Exception:
                log.warning("%s: unparsed segment key %r", seq, k); continue
            for hand, rng in (("lh", lh_rng), ("rh", rh_rng)):
                if rng is None:
                    continue
                s, e = int(rng[0]), int(rng[1])
                if EXTENDED and k in ext:      # widen to the annotated approach / retreat of this hand
                    hi = 0 if hand == "lh" else 1
                    ap, rt = ext[k].get("approach", [None, None])[hi], ext[k].get("retreat", [None, None])[hi]
                    if ap: s = int(ap[0])
                    if rt: e = int(rt[1])
                for obj in (v.get(f"obj_list_{hand}") or []):
                    segs.append((s, e, hand, obj, v.get(f"primitive_{hand}"), v.get("interaction_mode")))
        if not segs:
            continue
        need = [t for t in segs if f"{seq}__{t[2]}__{t[3]}__{t[0]}_{t[1]}.npz" not in done]
        if not need:                      # every take of this sequence exists: index it from the files
            for t in segs:
                fn = f"{seq}__{t[2]}__{t[3]}__{t[0]}_{t[1]}.npz"
                try:
                    with np.load(OUT / "takes" / fn) as z:
                        nf, nv = int(len(z["frames"])), int(z["contact"].shape[1])
                except Exception:
                    continue
                rows.append(dict(sequence=seq, actor=actor, hand=t[2], object=t[3], obj_name=desc.get(t[3], {}).get("obj_name", t[3]),
                                 category=category(desc.get(t[3], {}).get("obj_name", t[3])), primitive=t[4], interaction_mode=t[5],
                                 start=t[0], end=t[1], n_frames=nf, n_verts=nv, file=fn))
            continue
        a = pickle.load(open(ANNO / f"{seq}.pkl", "rb"))
        n_mocap = len(a["mocap_frame_id_list"])
        for (s, e, hand, obj, prim, mode) in segs:
            name = f"{seq}__{hand}__{obj}__{s}_{e}.npz"
            V = load_mesh(obj)
            if V is None:
                log.warning("%s: no mesh for %s", seq, obj); continue
            if obj not in a["obj_transf"]:
                log.warning("%s: no obj_transf for %s", seq, obj); continue
            frames = np.arange(s, min(e, n_mocap), STEP)
            frames = np.array([f for f in frames if f in a["obj_transf"][obj] and f in a["raw_mano"]])
            if len(frames) < 8:
                continue
            if name not in done:
                key = hand
                pc = np.stack([a["raw_mano"][f][f"{key}__pose_coeffs"].numpy()[0] for f in frames])
                tsl = np.stack([a["raw_mano"][f][f"{key}__tsl"].numpy()[0] for f in frames])
                betas = a["raw_mano"][frames[0]][f"{key}__betas"].numpy()[0]
                hv = mano_verts(layers["left" if hand == "lh" else "right"], pc, tsl, betas)
                Tm = np.stack([a["obj_transf"][obj][f] for f in frames]).astype(np.float64)
                Vw = np.einsum("tij,nj->tni", Tm[:, :3, :3], V) + Tm[:, None, :3, 3]
                d = np.empty((len(frames), len(V)), np.float32)
                for t in range(len(frames)):
                    d[t] = cKDTree(hv[t]).query(Vw[t])[0]
                np.savez_compressed(OUT / "takes" / name, contact=np.minimum(d, 0.5).astype(np.float16), frames=frames,
                                    obj_transf=Tm.astype(np.float32))
            rows.append(dict(sequence=seq, actor=actor, hand=hand, object=obj, obj_name=desc.get(obj, {}).get("obj_name", obj),
                             category=category(desc.get(obj, {}).get("obj_name", obj)), primitive=prim, interaction_mode=mode,
                             start=s, end=e, n_frames=len(frames), n_verts=len(V), file=name))
        if i % 20 == 0:
            log.info("%d/%d sequences, %d takes, %.0f s", i + 1, len(files), len(rows), time.time() - t0)
            pd.DataFrame(rows).to_csv(index_path, index=False)
    pd.DataFrame(rows).to_csv(index_path, index=False)
    log.info("done: %d takes", len(rows))


if __name__ == "__main__":
    # usage: build_oakink2_takes.py [limit|all] [shard n_shards]
    lim = None if len(sys.argv) < 2 or sys.argv[1] == "all" else int(sys.argv[1])
    sh, ns = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    main(lim, sh, ns)
