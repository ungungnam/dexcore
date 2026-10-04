#!/usr/bin/env python
"""Step 4b of the BimArt -> OakInk2 port: the pilot's checks on the full preprocessed run.

  python scripts/check_oakink2_bimart.py --workers 24

The gate is restated the way the pilot's independent verification asked
(pilot/verification/verdict.md §3.3), so the numbers compare like with like:

    coverage  ARCTIC over ALL its files at stride 16 (the pilot's 214 came from the first 60 files,
              three categories), and unique part0 slots per FRAME as well as per 64-frame window
    label     |gathered - dense| contact fraction on windows WITH contact as well as on all
              windows (windows without contact score 0 and dilute the mean), plus a per-segment
              screen of the contact-conditioned error
    held      "dense > 0.5 and gathered < 0.1" at 10 mm, and a variant that needs dense contact
              within 2 mm, so a hand hovering 5-10 mm over a surface is not counted as holding it
    flips     part1's geodesic rotation step and translation step between 30 Hz frames; the
              rotvec difference only measures the representation's wrap at pi
    B5        windows dropped per split and pair type, those dropped only because of a part-tree
              sibling of a pair member, and the primitives that lose the most

Label, held, MANO and kp checks need the dense subset (5%, seeded, see preprocess); the rest run
on every segment. Contact thresholds are TACO's (10 mm), as in the pilot.
Writes <port>/full_check/{report.json, windows.csv, segments.csv}.
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

log = logging.getLogger("check")
_G = {}
TOUCH_M = 0.01
HOLD_M = 0.002
NONPAIR_M = 0.005
FLIP_RAD = 1.0
H, BASE, STRIDE = 64, 8, 16
ARCTIC_STORE = Path("/result/uhnam/dexcore/bimart_arctic/sequences")
ARCTIC_RAW = Path("/home/uhnam/workspace/dexcore/third_party/BimArt/data/arctic_raw/raw_seqs")
TACO_CONTACT_LABEL_ERR = 0.0157     # TACO gen3 scene, contact-conditioned (pilot verification, a1.py)


def family(obj: str) -> set:
    """The object and every part in its part-tree group."""
    tree, parent = _G["tree"], _G["parent"]
    p = parent.get(obj, obj)
    return {p, obj} | set(tree.get(p, []))


def _windows_hit(near: np.ndarray, starts: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(near.astype(np.int64))])
    return (c[starts + H] - c[starts]) > 0


def _one(row):
    from scipy.spatial.transform import Rotation

    from src.analysis.bimart import oakink2_data as D

    try:
        with np.load(_G["port"] / "sequences" / row["file"], allow_pickle=False) as z:
            f = {k: z[k] for k in z.files}
        T = len(f["kp"])
        n0 = int(f["n_part0"])
        inds = f["obj_cano_bps_inds"].astype(np.int64)
        seg = {"sequence_id": row["sequence_id"], "T": T}

        # flips: part1 geodesic step and translation step
        R = Rotation.from_rotvec(f["global_abs"][:, :3])
        geo = (R[1:] * R[:-1].inv()).magnitude() if T > 1 else np.zeros(0)
        dt = np.linalg.norm(np.diff(f["global_abs"][:, 3:6], axis=0), axis=1)
        seg.update(steps=int(max(T - 1, 0)), geo_flips=int((geo > FLIP_RAD).sum()),
                   geo_max_rad=float(geo.max()) if len(geo) else 0.0,
                   trans_step_max_mm=float(dt.max() * 1000) if len(dt) else 0.0,
                   rotvec_wrap_jumps=int((np.linalg.norm(np.diff(f["global_abs"][:, :3], axis=0), axis=1) > 1).sum()))

        # per-frame coverage (every 10th frame)
        seg["uniq_part0_per_frame"] = float(np.median([len(np.unique(r)) for r in inds[::10, :512]]))

        # B5 at the split's own window rule (stride 1 for train), total and sibling-only
        kind = {"train": "train", "val": "val", "test": "test"}[row["split"]]
        starts = D.window_starts(T, kind, H, BASE)
        ids = list(f["nonpair_obj_ids"])
        near_all = f["nonpair_dist"].min(axis=(1, 2), initial=np.inf) < NONPAIR_M
        fam = family(row["part0"]) | family(row["part1"])
        oth = [k for k, o in enumerate(ids) if o not in fam]
        near_oth = (f["nonpair_dist"][:, :, oth].min(axis=(1, 2), initial=np.inf) < NONPAIR_M) if oth \
            else np.zeros(T, bool)
        hit_all, hit_oth = _windows_hit(near_all, starts), _windows_hit(near_oth, starts)
        seg.update(split_windows=int(len(starts)), dropped=int(hit_all.sum()),
                   dropped_sibling_only=int((hit_all & ~hit_oth).sum()))

        # stride-16 windows for coverage and, on the dense subset, the label checks
        dense = "contact_dense_lh" in f
        wins = []
        rt = np.arange(H)[:, None]
        for s in D.window_starts(T, "val", H, BASE, STRIDE):
            w = slice(s, s + H)
            r = {"sequence_id": row["sequence_id"], "start": int(s),
                 "uniq_part0": int(len(np.unique(inds[w, :512]))),
                 "uniq_part1": int(len(np.unique(inds[w, 512:]))),
                 "nonpair": bool(near_all[w].any())}
            if dense:
                for side, off in (("lh", 0), ("rh", 1024)):
                    d = f[f"contact_dense_{side}"][w]
                    for pn, sl, vs in (("part0", slice(0, 512), slice(0, n0)),
                                       ("part1", slice(512, 1024), slice(n0, None))):
                        dm = d[:, vs].min(1)
                        r[f"dense_{side}_{pn}"] = float((dm < TOUCH_M).mean())
                        r[f"dense2_{side}_{pn}"] = float((dm < HOLD_M).mean())
                        r[f"gath_{side}_{pn}"] = float((f["contact"][w, off + sl.start: off + sl.stop].min(1)
                                                        < TOUCH_M).mean())
                        assert np.array_equal(f["contact"][w, off + sl.start: off + sl.stop],
                                              d[rt, inds[w, sl]])     # label == dense at the slots
            wins.append(r)

        if dense:
            # MANO: nearest object vertex to each active hand, on its contact frames
            for side in ("lh", "rh"):
                m = f[f"contact_dense_{side}"].min(1)
                seg[f"{side}_active"] = row["hands"] in (side, "bh")
                seg[f"{side}_min_mm"] = float(m.min() * 1000)
                seg[f"{side}_on_contact_median_mm"] = float(np.median(m[m < TOUCH_M]) * 1000) if (m < TOUCH_M).any() else np.nan
            # kp round trip: canonical keypoints back to world against the stored world hands
            Rm = R.as_matrix()
            kp = f["kp"].reshape(T, 200, 3)
            world = np.einsum("tij,tnj->tni", Rm, kp) + f["global_abs"][:, None, 3:6]
            ref = np.concatenate([f["hand_world_lh"][:, _G["hand_index"]], f["hand_world_rh"][:, _G["hand_index"]]], 1)
            seg["kp_roundtrip_max_m"] = float(np.abs(world - ref).max())
        seg["dense"] = dense
        return seg, wins, None
    except Exception as e:                          # noqa: BLE001
        import traceback
        return None, [], f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"


def arctic_reference():
    """ARCTIC with the same definitions: every file, stride 16, per window and per frame; and the
    object's real (geodesic) rotation jumps, which the raw unnormalised rotvec cannot show."""
    import glob

    from scipy.spatial.transform import Rotation

    win, frame = [], []
    for fn in sorted(glob.glob(str(ARCTIC_STORE / "*.npz"))):
        with np.load(fn) as z:
            ind = z["obj_cano_bps_inds"]
        for s in range(BASE, len(ind) - H - BASE, STRIDE):
            win.append(len(np.unique(ind[s:s + H, :512])))
        frame.append(np.median([len(np.unique(r)) for r in ind[::10, :512]]))
    steps, flips = 0, 0
    for fn in sorted(glob.glob(str(ARCTIC_RAW / "*/*.object.npy"))):
        R = Rotation.from_rotvec(np.load(fn, allow_pickle=True)[:, 1:4].astype(float))
        g = (R[1:] * R[:-1].inv()).magnitude()
        steps += len(g)
        flips += int((g > FLIP_RAD).sum())
    return {"files": len(frame), "uniq_part0_per_window_median": float(np.median(win)),
            "uniq_part0_per_frame_median": float(np.median(frame)), "windows": len(win),
            "geodesic_flip_rate": flips / max(steps, 1), "geodesic_flips": flips, "steps": steps}


def summarise(seg, win, idx):
    import pandas as pd

    out = {}
    W = win.merge(idx[["sequence_id", "part_group", "split"]], on="sequence_id")
    S = seg.merge(idx[["sequence_id", "part_group", "split", "pair", "primitive", "name_part0", "name_part1"]],
                  on="sequence_id")
    grp = {"part_groups": True, "distinct_pairs": False}

    # coverage
    out["coverage"] = {g: {"uniq_part0_per_window_median": float(W[W.part_group == v].uniq_part0.median()),
                           "uniq_part1_per_window_median": float(W[W.part_group == v].uniq_part1.median()),
                           "uniq_part1_min": int(W[W.part_group == v].uniq_part1.min()),
                           "uniq_part0_per_frame_median": float(S[S.part_group == v].uniq_part0_per_frame.median())}
                       for g, v in grp.items()}

    # label and held, on the dense subset
    D = W.dropna(subset=["dense_lh_part0"]) if "dense_lh_part0" in W else W.iloc[0:0]
    cells = [(s, p) for s in ("lh", "rh") for p in ("part0", "part1")]
    lab = {}
    for g, v in grp.items():
        d = D[D.part_group == v]
        err = {f"{s}_{p}": (d[f"gath_{s}_{p}"] - d[f"dense_{s}_{p}"]).abs() for s, p in cells}
        con = {f"{s}_{p}": d[f"dense_{s}_{p}"] > 0 for s, p in cells}
        allerr = pd.concat(err.values())
        allcon = pd.concat(con.values())
        held10 = {k: int(((d[f"dense_{k}"] > 0.5) & (d[f"gath_{k}"] < 0.1)).sum()) for k in err}
        held2 = {k: int(((d[f"dense2_{k}"] > 0.5) & (d[f"gath_{k}"] < 0.1)).sum()) for k in err}
        lab[g] = {"windows": int(len(d)), "segments": int(d.sequence_id.nunique()),
                  "mean_abs_err": round(float(allerr.mean()), 4),
                  "mean_abs_err_on_contact": round(float(allerr[allcon].mean()), 4),
                  "per_cell_on_contact": {k: round(float(e[con[k]].mean()), 4) for k, e in err.items()},
                  "held_untouched_10mm": held10, "held_untouched_2mm": held2}
    lab["taco_reference_on_contact"] = TACO_CONTACT_LABEL_ERR
    # per-segment screen: contact-conditioned error over every (window, hand, part) cell
    rows = []
    for sid, d in D.groupby("sequence_id"):
        e = np.concatenate([(d[f"gath_{s}_{p}"] - d[f"dense_{s}_{p}"]).abs()[d[f"dense_{s}_{p}"] > 0].to_numpy()
                            for s, p in cells])
        if len(e):
            rows.append({"sequence_id": sid, "err_on_contact": float(e.mean()), "cells": int(len(e))})
    scr = pd.DataFrame(rows).merge(S[["sequence_id", "name_part0", "name_part1", "primitive"]], on="sequence_id") \
        if rows else pd.DataFrame()
    lab["segments_over_0.1"] = scr[scr.err_on_contact > 0.1].sort_values("err_on_contact", ascending=False) \
        .round(3).to_dict("records") if len(scr) else []
    lab["segment_err_on_contact_quantiles"] = scr.err_on_contact.quantile([.5, .9, .99]).round(4).to_dict() \
        if len(scr) else {}
    out["label"] = lab

    # MANO and geometry on the dense subset
    Sd = S[S.dense]
    mins = pd.concat([Sd.loc[Sd[f"{s}_active"] == True, f"{s}_min_mm"] for s in ("lh", "rh")])   # noqa: E712
    onc = pd.concat([Sd.loc[Sd[f"{s}_active"] == True, f"{s}_on_contact_median_mm"] for s in ("lh", "rh")])  # noqa: E712
    out["mano"] = {"active_hands": int(len(mins)), "min_mm_median": round(float(mins.median()), 3),
                   "frac_min_below_1mm": round(float((mins < 1).mean()), 4),
                   "on_contact_median_mm": round(float(onc.median()), 3),
                   "kp_roundtrip_max_m": float(Sd.kp_roundtrip_max_m.max())}

    # flips
    fl = S[S.geo_flips > 0]
    out["flips"] = {"steps": int(S.steps.sum()), "geodesic_flips": int(S.geo_flips.sum()),
                    "rate": float(S.geo_flips.sum() / max(S.steps.sum(), 1)),
                    "segments": fl[["sequence_id", "name_part1", "geo_flips", "geo_max_rad", "trans_step_max_mm"]]
                    .round(3).to_dict("records"),
                    "rotvec_wrap_jumps": int(S.rotvec_wrap_jumps.sum())}

    # B5
    b5 = {}
    for sp in ("train", "val", "test"):
        s = S[S.split == sp]
        b5[sp] = {"windows": int(s.split_windows.sum()), "dropped": int(s.dropped.sum()),
                  "frac": round(float(s.dropped.sum() / max(s.split_windows.sum(), 1)), 4),
                  "sibling_only": int(s.dropped_sibling_only.sum()),
                  "by_group": {g: round(float(s[s.part_group == v].dropped.sum()
                                               / max(s[s.part_group == v].split_windows.sum(), 1)), 4)
                               for g, v in grp.items()}}
    tr = S[S.split == "train"].groupby("primitive")[["split_windows", "dropped"]].sum()
    tr["frac"] = tr.dropped / tr.split_windows.clip(lower=1)
    b5["train_primitives_most_lost"] = tr[tr.split_windows > 0].sort_values("dropped", ascending=False) \
        .head(12).assign(frac=lambda x: x.frac.round(3)).reset_index().to_dict("records")
    out["b5_nonpair"] = b5
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/result/uhnam/dexcore/oakink2/30_bimart_port")
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    import multiprocessing as mp

    import pandas as pd

    from src.analysis.bimart import features as F
    from src.analysis.bimart import oakink2 as O

    port = Path(args.port)
    out = port / "full_check"
    out.mkdir(exist_ok=True)
    idx = pd.read_csv(port / "sequence_index.csv")
    tree = json.load(open(O.PART_TREE))
    _G.update(port=port, hand_index=F.load_hand_index(), tree=tree,
              parent={c: p for p, cs in tree.items() for c in cs})

    avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    segs, wins, errors = [], [], {}
    cols = ["sequence_id", "file", "split", "hands", "part0", "part1"]
    with mp.get_context("fork").Pool(processes=max(1, min(args.workers, avail))) as pool:
        for i, (s, w, e) in enumerate(pool.imap_unordered(_one, idx[cols].to_dict("records"), chunksize=4), 1):
            if e:
                errors[len(errors)] = e
                log.warning("%s", e.splitlines()[0])
            else:
                segs.append(s)
                wins += w
            if i % 200 == 0 or i == len(idx):
                log.info("  %d/%d segments", i, len(idx))
    if errors:
        raise SystemExit(f"{len(errors)} segments failed: {list(errors.values())[:2]}")
    seg, win = pd.DataFrame(segs), pd.DataFrame(wins)
    seg.to_csv(out / "segments.csv", index=False)
    win.to_csv(out / "windows.csv", index=False)

    log.info("ARCTIC reference ...")
    report = {"written": datetime.now().isoformat(timespec="seconds"),
              "segments": int(len(seg)), "dense_segments": int(seg.dense.sum()),
              "windows_stride16": int(len(win)), "thresholds_m": {"touch": TOUCH_M, "hold": HOLD_M,
                                                                  "nonpair": NONPAIR_M},
              "flip_rad": FLIP_RAD, "arctic": arctic_reference(), **summarise(seg, win, idx)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    log.info("report -> %s", out / "report.json")
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
