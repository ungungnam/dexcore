#!/usr/bin/env python
"""Step 3 of the BimArt -> OakInk2 port: a 60-segment pilot and the nine checks that gate the rest.

The pilot runs the real feature code (src/analysis/bimart/oakink2.py::segment_features) on a
stratified sample -- 30 part-group and 30 distinct-pair segments, forced to include cutting, pouring,
the alcohol-burner cap, a mesh under 1,000 vertices, a test pair unseen in training, and the
distinct-pair segment whose part1 is compressed most by the per-pair scale -- and measures, with the
TACO port's own definitions where one exists:

  1  MANO      the hands actually reach the objects (wrong conventions leave them hovering)
  2  basis     every part stays inside the 0.85 basis radius in every frame
  3  coverage  distinct vertices the 512 slots of each part land on, per 64-frame window
  4  label     |contact fraction from the 512 gathered slots - from every vertex| per window,
               hand and part (TACO scene run: 0.015)
  5  held      windows the gathered label calls untouched (<0.1) while the truth is held (>0.5)
               (TACO scene run: 0 of 2,801)
  6  geometry  canonical keypoints map back to the world hand; keypoint + dirvec lands on the object
  7  offset    translation of part1 at a window's first frame relative to the frame before it
  8  others    windows in which a hand comes within 5 mm of a tracked object outside the pair (B5)
  9  rotvec    rate of large jumps in part1's axis-angle rotation, against ARCTIC's

Checks 3 and 4 are also run under the per-SEGMENT scale, as a comparison only: the design is one
scale per pair, and the pilot is where that choice would show a coverage problem.

  python scripts/pilot_oakink2_bimart.py --workers 8

Writes <port>/pilot/{features/*.npz, pilot_windows.csv, pilot_segments.csv, pilot_report.json}.
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

log = logging.getLogger("pilot")
TOUCH_M = 0.01           # TACO's contact threshold (truncation.py)
WIN_STRIDE = 16
TACO_TRUNC = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/contact_probe/gen/reverify/"
                  "truncation_per_window.csv")
ARCTIC_RAW = Path("/home/uhnam/workspace/dexcore/third_party/BimArt/data/arctic_raw/raw_seqs")
ARCTIC_STORE = Path("/result/uhnam/dexcore/bimart_arctic/sequences")
_G = {}


# ------------------------------------------------------------------------------------ sample
def pick(seg, n_per_group=30, seed=1):
    """30 part-group + 30 distinct-pair segments with windows, forced strata first."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    md = _G["md"]
    seg = seg[seg.n_train_windows > 0].copy()
    small = {o for o, m in md.items() if len(m["verts"]) < 1000}
    train_pairs = set(seg[seg.split == "train"].pair)
    forced = {
        "cut": seg.primitive.str.contains("cut", case=False, na=False),
        "pour": seg.primitive.str.contains("pour", case=False, na=False),
        "burner": seg.name_a.str.contains("alcohol burner", na=False) | seg.name_b.str.contains("alcohol burner", na=False),
        "small_mesh": seg.obj_a.isin(small) | seg.obj_b.isin(small),
        "unseen_test_pair": (seg.split == "test") & ~seg.pair.isin(train_pairs),
    }
    chosen, why = [], {}
    for grp, pg in (("part_group", True), ("distinct_pair", False)):
        g = seg[seg.part_group == pg]
        take = []
        for name, mask in forced.items():
            cand = g[mask.loc[g.index] & ~g.segment_id.isin(take)]
            if len(cand):
                sid = cand.segment_id.iloc[rng.integers(len(cand))]
                take.append(sid); why[sid] = name
        if not pg:                                  # the hardest case for the per-pair scale
            sid = g.sort_values("part1_radius_in_basis").segment_id.iloc[0]
            if sid not in take:
                take.append(sid); why[sid] = "min_part1_in_basis"
        rest = g[~g.segment_id.isin(take)]
        extra = rest.segment_id.iloc[rng.permutation(len(rest))[:n_per_group - len(take)]].tolist()
        take += extra
        why.update({s: "random" for s in extra})
        chosen += take
    out = seg[seg.segment_id.isin(chosen)].copy()
    out["stratum"] = out.segment_id.map(why)
    return out


# -------------------------------------------------------------------------------- per segment
def _one(row):
    from src.analysis.bimart import features as F
    from src.analysis.bimart import oakink2 as O

    md, port = _G["md"], _G["port"]
    if "layers" not in _G:                          # built per worker: torch does not survive fork
        import torch
        torch.set_num_threads(1)
        _G["layers"] = O.mano_layers("cpu")
    try:
        with np.load(port / "cache" / f"{row['recording']}.npz", allow_pickle=False) as z:
            cache = {k: z[k] for k in z.files}
        feats = O.segment_features(cache, md, row["part0"], row["part1"], row["lo"], row["hi"],
                                   row["scale"], _G["basis"], _G["hand_index"], _G["layers"],
                                   keep_dense=True, nonpair=True)
        feats["inds_segment_scale"] = O.bps_indices(cache, md, row["part0"], row["part1"], row["lo"],
                                                    row["hi"], row["scale_segment"], _G["basis"]).astype(np.int32)
        c0, v1 = O.canonical_parts(cache, md, row["part0"], row["part1"], row["lo"], row["hi"])
        idx = O.frame_index(cache, row["lo"], row["hi"])
        R1, t1 = O.obj_pose(cache, row["part1"], idx)
        fn = port / "pilot/features" / f"{row['segment_id'].replace('#', '__')}.npz"
        np.savez_compressed(fn, **feats)
        return row["segment_id"], _measure(row, feats, c0, v1, R1, t1), None
    except Exception as e:                          # noqa: BLE001
        import traceback
        return row["segment_id"], None, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"


def _measure(row, f, c0, v1, R1, t1):
    """Per-window rows for checks 2-8, plus per-segment numbers for checks 1, 6 and 9."""
    from scipy.spatial import cKDTree

    T = len(f["frames"])
    n0 = int(f["n_part0"])
    lab = f["contact"]
    dense = {"lh": f["contact_dense_lh"], "rh": f["contact_dense_rh"]}
    inds_pair, inds_seg = f["obj_cano_bps_inds"].astype(np.int64), f["inds_segment_scale"].astype(np.int64)
    rt = np.arange(T)[:, None]
    parts = {"part0": (slice(0, 512), slice(0, n0)), "part1": (slice(512, 1024), slice(n0, None))}

    # 1 -- closest approach of each active hand to either part, over the segment and on contact frames
    seg_rec = {}
    active = {"lh": row["hands"] in ("lh", "bh"), "rh": row["hands"] in ("rh", "bh")}
    for side in ("lh", "rh"):
        m = dense[side].min(1)                       # (T,) nearest object vertex to this hand
        seg_rec[f"{side}_active"] = active[side]
        seg_rec[f"{side}_min_mm"] = float(m.min() * 1000)
        on = m < TOUCH_M
        seg_rec[f"{side}_contact_frames"] = int(on.sum())
        seg_rec[f"{side}_median_on_contact_mm"] = float(np.median(m[on]) * 1000) if on.any() else np.nan

    # 2 -- basis radius, from the BPS itself: the selected point is basis + delta. Both parts are
    # queried with the same 512-point basis, so it is tiled once per part.
    pts = np.concatenate([_G["basis"], _G["basis"]])[None] + f["obj_cano_bps"]
    seg_rec["max_selected_radius"] = float(np.linalg.norm(pts, axis=2).max())
    seg_rec["max_union_radius_scaled"] = float(max(np.linalg.norm(c0, axis=2).max(), np.linalg.norm(v1, axis=1).max())
                                               * row["scale"])

    # 6 -- geometry
    kp = f["kp"].reshape(T, 200, 3)
    world_kp = np.einsum("tij,tnj->tni", R1, kp) + t1[:, None, :]
    ref = np.concatenate([f["hand_world_lh"][:, _G["hand_index"]], f["hand_world_rh"][:, _G["hand_index"]]], axis=1)
    seg_rec["kp_roundtrip_max_m"] = float(np.abs(world_kp - ref).max())
    obj_all = np.concatenate([c0, np.repeat(v1[None], T, 0)], axis=1)
    land = kp + f["dirvec"].reshape(T, 200, 3)
    seg_rec["kp_dirvec_to_surface_max_m"] = float(max(cKDTree(obj_all[t]).query(land[t])[0].max() for t in range(T)))

    # 9 -- part1 rotation-vector jumps between consecutive 30 Hz frames
    rv = f["global_abs"][:, :3]
    seg_rec["rotvec_jumps"] = int((np.linalg.norm(np.diff(rv, axis=0), axis=1) > 1.0).sum())
    seg_rec["rotvec_steps"] = int(T - 1)

    # windows
    from src.analysis.bimart import oakink2 as O
    wins = []
    end = T - O.PRED_HORIZON - O.BASE_FRAME
    for s in range(O.BASE_FRAME, max(O.BASE_FRAME, end) + 1, WIN_STRIDE):
        if s + O.PRED_HORIZON > T:
            break
        w = slice(s, s + O.PRED_HORIZON)
        rec = {"segment_id": row["segment_id"], "start": s}
        for side, off in (("lh", 0), ("rh", 1024)):
            for pn, (sl, vs) in parts.items():
                d_frac = float((dense[side][w, vs].min(1) < TOUCH_M).mean())
                g_pair = lab[w, off + sl.start: off + sl.stop]
                g_seg = dense[side][w][np.arange(O.PRED_HORIZON)[:, None], inds_seg[w, sl]]
                gp, gs = float((g_pair.min(1) < TOUCH_M).mean()), float((g_seg.min(1) < TOUCH_M).mean())
                rec[f"dense_{side}_{pn}"] = d_frac
                rec[f"gath_{side}_{pn}"] = gp
                rec[f"gathseg_{side}_{pn}"] = gs
        rec["uniq_part0"] = int(len(np.unique(inds_pair[w, :512])))
        rec["uniq_part1"] = int(len(np.unique(inds_pair[w, 512:])))
        rec["uniq_part0_segscale"] = int(len(np.unique(inds_seg[w, :512])))
        rec["uniq_part1_segscale"] = int(len(np.unique(inds_seg[w, 512:])))
        # 7 -- the motion model's reference is the frame before the window
        if s >= 1:
            rec["start_offset_mm"] = float(np.linalg.norm(f["global_abs"][s, 3:6] - f["global_abs"][s - 1, 3:6]) * 1000)
        # 8 -- a hand within 5 mm of a tracked object outside the pair, anywhere in the window
        if "nonpair_min_dist" in f:
            rec["nonpair_touch"] = bool((f["nonpair_min_dist"][w] < O.NONPAIR_TOUCH_M).any())
        wins.append(rec)
    return {"segment": seg_rec, "windows": wins}


# ------------------------------------------------------------------------------ references
def arctic_reference():
    """ARCTIC's per-window unique part0 vertices (from the re-indexed store) and its object
    rotation-vector jump rate (from the raw sequences, 30 Hz), with the pilot's definitions."""
    import glob

    uniq, jumps, steps = [], 0, 0
    for f in sorted(glob.glob(str(ARCTIC_STORE / "*.npz")))[:60]:
        with np.load(f) as z:
            ind = z["obj_cano_bps_inds"]
        for s in range(8, len(ind) - 64 - 8, 64):
            uniq.append(len(np.unique(ind[s:s + 64, :512])))
    for f in sorted(glob.glob(str(ARCTIC_RAW / "*/*.object.npy"))):
        rv = np.load(f, allow_pickle=True)[:, 1:4]
        d = np.linalg.norm(np.diff(rv, axis=0), axis=1)
        jumps += int((d > 1.0).sum()); steps += len(d)
    return {"arctic_uniq_part0_median": float(np.median(uniq)) if uniq else None,
            "arctic_uniq_part0_windows": len(uniq),
            "arctic_rotvec_jump_rate": jumps / max(steps, 1), "arctic_rotvec_steps": steps}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/result/uhnam/dexcore/oakink2/30_bimart_port")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    import multiprocessing as mp

    import pandas as pd

    from src.analysis.bimart import features as F
    from src.analysis.bimart import oakink2 as O

    port = Path(args.port)
    (port / "pilot/features").mkdir(parents=True, exist_ok=True)
    seg = pd.read_csv(port / "segments_geom.csv")
    _G.update(md=np.load(port / "assets/oakink2_mesh_dict.npy", allow_pickle=True).item(), port=port,
              basis=F.load_basis(), hand_index=F.load_hand_index())

    sample = pick(seg, seed=args.seed)
    sample.to_csv(port / "pilot/pilot_segments.csv", index=False)
    log.info("pilot: %d segments (%s)", len(sample), sample.stratum.value_counts().to_dict())

    ctx = mp.get_context("fork")
    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    seg_rows, win_rows, errors = [], [], {}
    with ctx.Pool(processes=max(1, min(args.workers, avail))) as pool:
        for i, (sid, res, err) in enumerate(pool.imap_unordered(_one, sample.to_dict("records")), 1):
            if err:
                errors[sid] = err
                log.warning("%s failed: %s", sid, err.splitlines()[0])
            else:
                seg_rows.append({"segment_id": sid, **res["segment"]})
                win_rows += res["windows"]
            if i % 10 == 0 or i == len(sample):
                log.info("  %d/%d", i, len(sample))
    if errors:
        (port / "pilot/errors.json").write_text(json.dumps(errors, indent=2))
        raise SystemExit(f"{len(errors)} pilot segments failed; see pilot/errors.json")

    S = sample.merge(pd.DataFrame(seg_rows), on="segment_id")
    W = pd.DataFrame(win_rows).merge(sample[["segment_id", "part_group", "split", "pair", "stratum"]], on="segment_id")
    S.to_csv(port / "pilot/pilot_segments.csv", index=False)
    W.to_csv(port / "pilot/pilot_windows.csv", index=False)

    # ---------------------------------------------------------------- the nine checks
    ref = arctic_reference()
    taco = pd.read_csv(TACO_TRUNC) if TACO_TRUNC.exists() else None
    taco_s = taco[taco.run == "scene"] if taco is not None else None
    grp = {"part_groups": W.part_group, "distinct_pairs": ~W.part_group}
    report = {"written": datetime.now().isoformat(timespec="seconds"), "segments": len(S), "windows": len(W),
              "window_stride": WIN_STRIDE, "touch_threshold_m": TOUCH_M, "references": {**ref}}
    if taco_s is not None:
        report["references"].update({"taco_scene_label_err_mean": float(taco_s.label_err.mean()),
                                     "taco_scene_uniq_part0_median": float(taco_s.uniq_tool_verts.median()),
                                     "taco_scene_untouched_while_held": int(((taco_s.dense_R_tool > .5) & (taco_s.gath_R_tool < .1)).sum()),
                                     "taco_scene_windows": int(len(taco_s))})

    # 1 MANO
    act = []
    for side in ("lh", "rh"):
        a = S[S[f"{side}_active"]]
        act.append(a[[f"{side}_min_mm", f"{side}_median_on_contact_mm"]].set_axis(["min_mm", "on_contact_mm"], axis=1))
    act = pd.concat(act)
    report["1_mano"] = {"active_hand_segments": len(act),
                        "min_mm_median": float(act.min_mm.median()), "min_mm_p90": float(act.min_mm.quantile(.9)),
                        "frac_min_below_1mm": float((act.min_mm < 1).mean()),
                        "frac_min_below_2mm": float((act.min_mm < 2).mean()),
                        "on_contact_mm_median": float(act.on_contact_mm.median())}
    # 2 basis
    report["2_basis"] = {"max_selected_radius": float(S.max_selected_radius.max()),
                         "max_union_radius_scaled": float(S.max_union_radius_scaled.max()),
                         "pass": bool(S.max_union_radius_scaled.max() <= F.BASIS_RADIUS + 1e-6)}
    # 3 coverage, 4 label, 5 held -- per group, both scales
    for name, m in grp.items():
        Wg = W[m]
        r3, r4, r5 = {}, {}, {}
        for tag, pre, u0, u1 in (("pair_scale", "gath", "uniq_part0", "uniq_part1"),
                                 ("segment_scale", "gathseg", "uniq_part0_segscale", "uniq_part1_segscale")):
            r3[tag] = {"uniq_part0_median": float(Wg[u0].median()), "uniq_part1_median": float(Wg[u1].median()),
                       "uniq_part1_min": int(Wg[u1].min())}
            errs, held = {}, {}
            for side in ("lh", "rh"):
                for pn in ("part0", "part1"):
                    e = (Wg[f"{pre}_{side}_{pn}"] - Wg[f"dense_{side}_{pn}"]).abs()
                    errs[f"{side}_{pn}"] = round(float(e.mean()), 4)
                    held[f"{side}_{pn}"] = int(((Wg[f"dense_{side}_{pn}"] > .5) & (Wg[f"{pre}_{side}_{pn}"] < .1)).sum())
            r4[tag] = {"mean_abs_err": errs, "overall": round(float(np.mean(list(errs.values()))), 4)}
            r5[tag] = {"untouched_while_held": held, "total": int(sum(held.values())), "windows": int(len(Wg))}
        report.setdefault("3_coverage", {})[name] = r3
        report.setdefault("4_label", {})[name] = r4
        report.setdefault("5_held", {})[name] = r5
        seg_g = S[S.part_group == (name == "part_groups")]
        report["3_coverage"][name]["part1_radius_in_basis"] = {
            "pair_scale_min": float(seg_g.part1_radius_in_basis.min()),
            "pair_scale_median": float(seg_g.part1_radius_in_basis.median()),
            "segment_scale_median": float((seg_g.r1 * seg_g.scale_segment).median())}
    # 6 geometry
    report["6_geometry"] = {"kp_roundtrip_max_m": float(S.kp_roundtrip_max_m.max()),
                            "kp_dirvec_to_surface_max_m": float(S.kp_dirvec_to_surface_max_m.max())}
    # 7 offset
    report["7_start_offset_mm"] = {"median": float(W.start_offset_mm.median()),
                                   "p90": float(W.start_offset_mm.quantile(.9))}
    # 8 others
    report["8_nonpair_touch"] = {"frac_windows": float(W.nonpair_touch.mean()),
                                 "by_group": {n: float(W[m].nonpair_touch.mean()) for n, m in grp.items()}}
    # 9 rotvec
    report["9_rotvec_jump_rate"] = {"oakink2": float(S.rotvec_jumps.sum() / max(S.rotvec_steps.sum(), 1)),
                                    "arctic": ref["arctic_rotvec_jump_rate"]}
    (port / "pilot/pilot_report.json").write_text(json.dumps(report, indent=2))
    log.info("report -> %s", port / "pilot/pilot_report.json")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
