#!/usr/bin/env python
"""Run all four BimArt inference stages on TACO and score each one.

  CUDA_VISIBLE_DEVICES=0 python scripts/infer_bimart_taco.py --split test_2 --windows 8

    1  contact sampling      object -> contact map, 50-step DDPM
    2  contact guidance      each denoising step is pulled toward that contact map
    3  MANO fitting          a real hand fitted to the 100 predicted surface points per side
    4  optimisation refine   the fitted hand moved onto the object: projection, penetration, jitter

Scoring every stage separately is the point. Stages 3 and 4 cannot move a hand that stage 2 put in
the wrong place -- they fit and polish -- so reporting only the final number hides where the error
actually comes from.
"""
import pathlib as _pl
import sys as _sys

_HERE = str(_pl.Path(__file__).resolve().parent)
_REPO = str(_pl.Path(__file__).resolve().parent.parent)
_sys.path[:] = [p for p in _sys.path if p not in ("", _HERE)]
for _p in (_REPO, str(_pl.Path(_REPO) / "third_party/BimArt")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from src.analysis.action_structure.contact import _install_numpy_aliases
from src.analysis.bimart.p3d_compat import install as _install_p3d

# chumpy (needed to unpickle MANO) and pytorch3d (imported by BimArt's utils) both have to be
# satisfied before anything from `utils/` is imported.
_install_numpy_aliases()
_install_p3d()

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

HAND_KP = 100


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--contact", default="/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/experiments/"
                                        "contact_model_taco_20260923_151838")
    p.add_argument("--motion", default="/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/experiments/"
                                       "motion_model_taco_20260923_151845")
    p.add_argument("--split", default="test_2")
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--mano-steps", type=int, default=1500)
    p.add_argument("--refine-steps", type=int, default=100)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/inference")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("infer")

    import contextlib
    import io

    from diffusers import DDPMScheduler
    from torch.utils.data import DataLoader, Subset

    from scripts.eval_bimart_taco import load_model, sample
    from src.analysis.action_structure import contact as C
    from src.analysis.bimart import refine as RF
    from src.analysis.bimart.datasets import TacoContactDataset, TacoMotionDataset
    from src.analysis.bimart.guidance import (ContactGuidance, _canonical_objects,
                                              sparse_object_vertices)
    from utils import data_util, mano_utils

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ccfg = yaml.safe_load(open("configs/bimart_taco/train_contact_config.yaml"))
    mcfg = yaml.safe_load(open("configs/bimart_taco/train_motion_config.yaml"))
    root = mcfg["data"]["base_dir"]
    horizon = mcfg["data"]["pred_horizon"]

    with contextlib.redirect_stdout(io.StringIO()):
        cm, cname, cep, _ = load_model("contact", args.contact, ccfg, dev)
        mm, mname, mep, _ = load_model("motion", args.motion, mcfg, dev)
        cstat = data_util.load_stat_dict(ccfg["stat_dict_path"], dev)
        mstat = data_util.load_stat_dict(mcfg["data"]["stat_dict_path"], dev)
    sched = DDPMScheduler(num_train_timesteps=ccfg["num_timesteps"],
                          beta_schedule="squaredcos_cap_v2", clip_sample=False,
                          prediction_type="sample")
    a_mean, a_std = mstat["action"]["mean"], mstat["action"]["std"]
    hand_index = np.load(mcfg["data"]["hand_index_path"]).astype(np.int32).reshape(-1)
    # upstream's create_mano_layer hardcodes a RELATIVE model path, so it only works from the
    # BimArt directory; the layer is built here with an absolute one instead
    import smplx
    mano_root = str(_pl.Path(_REPO) / "third_party/BimArt/assets/mano_v1_2")
    layer = {s_: smplx.create(mano_root, "mano", use_pca=False, is_rhand=(s_ == "right"),
                              num_pca_comps=45, flat_hand_mean=False).to(dev)
             for s_ in ("right", "left")}
    layers = {s: C.mano_layer(s) for s in ("left", "right")}

    cds = TacoContactDataset(root=root, split=args.split, pred_horizon=horizon, base_frame=8)
    mds = TacoMotionDataset(root=root, split=args.split, pred_horizon=horizon, base_frame=8)
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(cds), min(args.windows, len(cds)), replace=False)

    def kp_err(a_norm, kp_gt):
        kp = (a_norm * a_std + a_mean)[..., : HAND_KP * 6]
        d = (kp - kp_gt).reshape(*kp.shape[:-1], HAND_KP * 2, 3)
        return torch.linalg.norm(d, dim=-1).mean().item() * 1000.0

    out_dir = Path(args.out) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    records, dumps = [], []
    for w in pick:
        row = mds.index.iloc[mds.windows[w]["file_idx"]]
        seq, start = row["sequence_id"], mds.windows[w]["start"]
        cb = next(iter(DataLoader(Subset(cds, [w]), batch_size=1)))
        mb = next(iter(DataLoader(Subset(mds, [w]), batch_size=1)))
        cn = data_util.preprocess_contact_batch({k: v for k, v in cb.items() if k != "aux"},
                                                dev, cstat, True)
        mn = data_util.preprocess_batch({k: v for k, v in mb.items() if k not in ("aux", "viz")},
                                        mstat, dev)
        kp_gt = (mn["action"] * a_std + a_mean)[..., : HAND_KP * 6]
        obj_sparse = sparse_object_vertices(root, seq, start, horizon, dev)

        # ---- 1) contact, 2) motion with and without guidance
        cs_hat = sample(cm, cn["action"].shape, sched, dev, cn["obs"], "contact",
                        ccfg["num_timesteps"], args.seed)
        cond = dict(mn["obs"]); cond["contact_points"] = cs_hat
        a_plain = sample(mm, mn["action"].shape, sched, dev, cond, "motion",
                         ccfg["num_timesteps"], args.seed)
        guide = ContactGuidance(obj_sparse, cs_hat, mstat, guidance_scale=args.guidance_scale)
        a_guided = sample(mm, mn["action"].shape, sched, dev, cond, "motion",
                          ccfg["num_timesteps"], args.seed, guide=guide)
        rec = {"sequence_id": seq, "start": int(start), "split": args.split,
               "verb": row["verb"], "tool": row["tool_cat"], "target": row["target_cat"],
               "stage2_plain_mm": kp_err(a_plain, kp_gt),
               "stage2_guided_mm": kp_err(a_guided, kp_gt)}

        # ---- 3) MANO fit to the guided keypoints (canonical frame throughout)
        un = (a_guided * a_std + a_mean).squeeze(0)
        kp = un[:, : HAND_KP * 6].reshape(horizon, HAND_KP * 2, 3)
        dv = un[:, HAND_KP * 6: HAND_KP * 12].reshape(horizon, HAND_KP * 2, 3)
        kpl, kpr = kp[:, :HAND_KP], kp[:, HAND_KP:]
        with contextlib.redirect_stdout(io.StringIO()):
            params, vl, vr = mano_utils.fit_mano(kpl[None], kpr[None], layer, ph=horizon,
                                                 idxl=hand_index, idxr=hand_index,
                                                 hand_keypoints=HAND_KP, steps=args.mano_steps)
        vl_t = torch.as_tensor(vl[0], dtype=torch.float32, device=dev)
        vr_t = torch.as_tensor(vr[0], dtype=torch.float32, device=dev)
        rec["stage3_fit_mm"] = float(
            (torch.linalg.norm(torch.cat([vl_t[:, hand_index], vr_t[:, hand_index]], 1)
                               - kp, dim=-1).mean()) * 1000)

        # ---- ground-truth hand vertices, for a mesh-level error
        from src.analysis.loaders import taco
        ref = next(r for r in taco.index() if r.sequence_id == seq)
        traj = taco.load(ref)
        from src import geometry as G
        from src.analysis.bimart import features as F
        R_t, p_t = G.wxyz_to_R(traj.target.quat), traj.target.pos
        gt = {}
        for side in ("left", "right"):
            skin, _ = C.hand_geometry(traj.hands[side], layers[side])
            gt[side] = torch.as_tensor(
                F.to_canonical(skin, R_t, p_t)[start:start + horizon],
                dtype=torch.float32, device=dev)
        rec["stage3_vert_mm"] = float(
            ((torch.linalg.norm(vl_t - gt["left"], dim=-1).mean()
              + torch.linalg.norm(vr_t - gt["right"], dim=-1).mean()) / 2) * 1000)

        # ---- 4) optimisation refinement
        obj_full = torch.as_tensor(_canonical_objects(root, seq)[start:start + horizon],
                                   dtype=torch.float32, device=dev)
        post = RF.PostOptimization(layer, hand_index, steps=args.refine_steps, device=dev)
        with contextlib.redirect_stdout(io.StringIO()):
            ref_out = post.run(params[0], obj_full,
                               {"left": dv[:, :HAND_KP], "right": dv[:, HAND_KP:]})
        rl, rr = ref_out["left_verts"], ref_out["right_verts"]
        rec["stage4_vert_mm"] = float(
            ((torch.linalg.norm(rl - gt["left"], dim=-1).mean()
              + torch.linalg.norm(rr - gt["right"], dim=-1).mean()) / 2) * 1000)
        rec["refine_proj_first"] = ref_out["history"][0]["proj"]
        rec["refine_proj_last"] = ref_out["history"][-1]["proj"]

        records.append(rec)
        dumps.append({"sequence_id": seq, "start": int(start),
                      "pred_left": rl.cpu().numpy().astype(np.float32),
                      "pred_right": rr.cpu().numpy().astype(np.float32),
                      "gt_left": gt["left"].cpu().numpy().astype(np.float32),
                      "gt_right": gt["right"].cpu().numpy().astype(np.float32)})
        log.info("%s@%d (%s): stage2 %.1f -> guided %.1f | fit %.1f | verts %.1f -> refined %.1f mm",
                 seq[:34], start, row["verb"], rec["stage2_plain_mm"], rec["stage2_guided_mm"],
                 rec["stage3_fit_mm"], rec["stage3_vert_mm"], rec["stage4_vert_mm"])

    import pandas as pd
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "stage_results.csv", index=False)
    np.savez_compressed(out_dir / "predictions.npz",
                        **{f"{i}_{k}": v for i, d in enumerate(dumps)
                           for k, v in d.items() if isinstance(v, np.ndarray)},
                        meta=json.dumps([{"sequence_id": d["sequence_id"], "start": d["start"]}
                                         for d in dumps]))
    cols = [c for c in df.columns if c.endswith("_mm")]
    print(f"\n=== {args.split}, {len(df)} windows ===")
    print(df[cols].mean().round(1).to_string())
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
