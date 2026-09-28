#!/usr/bin/env python
"""Build an ARCTIC contact-map store in the layout the TACO viz scripts already read.

`viz_contact_maps.py` and `viz_contact_mesh.py` were written for the TACO port, but nothing in
their figures is TACO-specific: both read a 2048-vector whose halves are the two hands and whose
quarters are two objects, painted on two meshes. ARCTIC has exactly that shape -- the two
"objects" are the two PARTS of one articulated mesh:

    TACO   [0:512] L->tool   [512:1024] L->target   [1024:1536] R->tool   [1536:2048] R->target
    ARCTIC [0:512] L->top    [512:1024] L->bottom   [1024:1536] R->top    [1536:2048] R->bottom

So this writes ARCTIC into that layout rather than duplicating the plotting code. `tool` means the
articulated (top) part and `target` the static (bottom) one throughout the store; the ARCTIC driver
relabels them for display. ARCTIC's verb -- `grab` or `use` -- lands in the store's `verb` column,
which is what makes the grab/use split legible in every figure.

Two index remappings are needed and neither is cosmetic:

  * `obj_cano_bps_inds` holds GLOBAL mesh vertex ids. The viz code expects part-local ids for the
    first 512 slots and `n_top + part-local` for the rest, so vertices are reordered to
    [top..., bottom...] and the indices remapped into that order.
  * dense contact is stored per global vertex; it is reordered the same way, so
    `contact[:n_top]` is the top part and `contact[n_top:]` the bottom, as the viz code assumes.

Predictions come from the contact model's OWN pipeline (`ObjectContactData`), not through the
motion model's renormalisation bridge, so the numbers here are the prior's own output.

  python scripts/dump_contact_maps_arctic.py --categories laptop microwave --split test
  python scripts/dump_contact_maps_arctic.py --categories laptop --split train --windows 60

Writes to <root>/{assets,sequences,contact_probe} (default /result/uhnam/dexcore/bimart_arctic).
"""
import pathlib as _pl
import sys as _sys

# scripts/ holds modules that shadow the stdlib (select.py), so it must not stay on sys.path.
_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
if _REPO not in _sys.path:
    _sys.path.insert(0, _REPO)

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("dump_arctic")
N_BPS = 512
TOP, BOTTOM = 0, 1


def part_mesh(verts, faces, parts, pid):
    """One part as a standalone mesh: its vertices, and the faces wholly inside it, reindexed."""
    keep = np.where(parts == pid)[0]
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    f = faces[(remap[faces] >= 0).all(axis=1)]
    return verts[keep], remap[f], keep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop"])
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--windows", type=int, default=0, help="cap on windows dumped (0 = all)")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--config", default="config_files/contact_inference.yaml")
    ap.add_argument("--root", default="/result/uhnam/dexcore/bimart_arctic")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.chdir(BIMART_ROOT)
    _sys.path.insert(0, str(BIMART_ROOT))

    import pandas as pd
    from utils import data_util, yaml_util
    from dataset import contact_data
    from contact_prior.contact_prior_util import load_contact_module, load_noise_scheduler

    root = Path(args.root)
    (root / "assets").mkdir(parents=True, exist_ok=True)
    (root / "sequences").mkdir(parents=True, exist_ok=True)
    (root / "contact_probe").mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0); np.random.seed(0)
    ccfg = yaml_util.load_yaml(args.config)
    cats = set(args.categories)

    class SubContact(contact_data.ObjectContactData):
        def __get_file_paths__(self):
            super().__get_file_paths__()
            for s in ("train", "test"):
                self.data_files[s]["process_files"] = [
                    p for p in self.data_files[s]["process_files"] if Path(p).parts[-3] in cats]

    ds = SubContact(split=args.split, base_dir=ccfg["base_dir"], end_frame=ccfg["end_frame"],
                    pred_horizon=ccfg["pred_horizon"], return_aux_info=True)
    files = ds.data_files[args.split]["process_files"]
    LOG.info("%s split: %d files, %d windows", args.split, len(files), len(ds.indices))

    # ---------------------------------------------------------------- meshes, one entry per part
    orig = data_util.load_mesh_dict()                       # metres, as contact was computed in
    mesh_dict, reorder = {}, {}
    for cat in sorted(cats):
        verts, faces = np.asarray(orig[cat]["verts"]), np.asarray(orig[cat]["faces"])
        parts = np.asarray(orig[cat]["parts"]).squeeze()
        keeps = []
        for pid, role in ((TOP, "top"), (BOTTOM, "bottom")):
            v, f, keep = part_mesh(verts, faces, parts, pid)
            mesh_dict[f"{cat}_{role}"] = {
                "verts": v, "verts_original": v, "faces": f,
                "scale": 1.0, "centroid": np.zeros(3), "category": cat, "role": role}
            keeps.append(keep)
            LOG.info("  %-22s %5d verts  %5d faces", f"{cat}_{role}", len(v), len(f))
        order = np.concatenate(keeps)                        # [top..., bottom...]
        inv = np.empty(len(verts), dtype=np.int64)
        inv[order] = np.arange(len(order))                   # global id -> position in that order
        reorder[cat] = {"order": order, "inv": inv, "n_top": len(keeps[0])}
    np.save(root / "assets/arctic_mesh_dict.npy", mesh_dict, allow_pickle=True)

    # ------------------------------------------------------------------- per-sequence dense store
    seq_rows, seq_of_file = [], {}
    for f_i, path in enumerate(files):
        d = ds.process_data_all[f_i]
        cat = d["category"]
        stem = Path(path).name.split("_processed_obj_features")[0]
        subj = Path(path).parent.name
        seq_id = f"{stem}/{subj}"
        verb = "grab" if "_grab_" in stem else "use"
        r = reorder[cat]
        inds = d["obj_cano_bps_inds"].astype(np.int64)
        out = root / "sequences" / f"{stem}__{subj}.npz"
        np.savez_compressed(
            out,
            contact_left=d["contact_dict"]["left"]["dist"][:, r["order"]].astype(np.float32),
            contact_right=d["contact_dict"]["right"]["dist"][:, r["order"]].astype(np.float32),
            obj_cano_bps_inds=r["inv"][inds].astype(np.int32))
        seq_of_file[f_i] = seq_id
        seq_rows.append({"sequence_id": seq_id, "triplet": f"({verb}, {cat}, {subj})",
                         "verb": verb, "tool_cat": cat, "target_cat": cat,
                         "tool_mesh": f"{cat}_top", "target_mesh": f"{cat}_bottom",
                         "n_frames": int(len(inds)), "file": out.name, "split": args.split})
        LOG.info("[%d/%d] %s  T=%d  verb=%s", f_i + 1, len(files), seq_id, len(inds), verb)

    # ------------------------------------------------------------------------- windows: GT + pred
    stat = data_util.load_stat_dict(ccfg["stat_dict_path"], device)
    model, _, _ = load_contact_module(ccfg, device)
    model.eval()
    sched = load_noise_scheduler(ccfg)
    mean, std = stat["action"]["mean"], stat["action"]["std"]

    n = len(ds.indices) if args.windows <= 0 else min(args.windows, len(ds.indices))
    gt_all = np.empty((n, ccfg["pred_horizon"], 2048), dtype=np.float16)
    pred_all = np.empty_like(gt_all)
    rows = []
    for w in range(n):
        item = ds[w]
        batch = {"action": torch.from_numpy(np.asarray(item["action"]))[None],
                 "obs": {k: torch.from_numpy(np.asarray(v))[None] for k, v in item["obs"].items()},
                 "aux": {}}
        nb = data_util.preprocess_contact_batch(batch, device, stat, ccfg["normalize_data"])
        x = torch.randn(nb["action"].shape, device=device).float()
        sched.set_timesteps(ccfg["num_timesteps"])
        with torch.no_grad():
            for t in sched.timesteps:
                x0 = model(sample=x, timestep=t, obj_feat=nb["obs"]["obj_feat"], global_cond=nb["obs"])
                x = sched.step(model_output=x0, timestep=t, sample=x).prev_sample
        pred_m = data_util.unnormalize_item(x, mean, std)[0].cpu().numpy()
        gt_all[w] = np.asarray(item["action"], dtype=np.float32)        # already metres
        pred_all[w] = pred_m
        meta = ds.indices[w]
        seq_id = seq_of_file[meta["file_idx"]]
        base = next(r for r in seq_rows if r["sequence_id"] == seq_id)
        rows.append({"sequence_id": seq_id, "window": w, "start": int(meta["subsequence_idx"]),
                     "triplet": base["triplet"], "verb": base["verb"],
                     "tool_mesh": base["tool_mesh"], "target_mesh": base["target_mesh"]})
        if (w + 1) % 10 == 0 or w == n - 1:
            LOG.info("  window %d/%d", w + 1, n)

    for r in seq_rows:
        r["n_windows"] = sum(1 for q in rows if q["sequence_id"] == r["sequence_id"])
    # One index serves every split, so merge rather than overwrite -- dumping `train` after
    # `test` must not strand the test episodes without their sequence rows.
    idx_path = root / "sequence_index.csv"
    new_idx = pd.DataFrame(seq_rows)
    if idx_path.exists():
        old_idx = pd.read_csv(idx_path)
        old_idx = old_idx[~old_idx["sequence_id"].isin(new_idx["sequence_id"])]
        new_idx = pd.concat([old_idx, new_idx], ignore_index=True)
    new_idx.to_csv(idx_path, index=False)
    np.savez_compressed(root / f"contact_probe/{args.split}_contact_maps.npz",
                        gt=gt_all, pred=pred_all)
    pd.DataFrame(rows).to_csv(root / f"contact_probe/{args.split}_contact_maps_index.csv",
                              index=False)
    LOG.info("wrote %d windows to %s", n, root / "contact_probe")


if __name__ == "__main__":
    main()
