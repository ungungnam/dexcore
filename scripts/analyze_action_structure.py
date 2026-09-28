#!/usr/bin/env python
"""Does TACO's tool-target rigid motion carry action structure, or object structure?

Rigid object poses only -- no hand signal, no mesh geometry, no learned model. Labels are used for
grouping and evaluation, never as input.

  python scripts/analyze_action_structure.py --workers 1
  python scripts/analyze_action_structure.py --limit 300 --no-umap      # a fast smoke run

Results go to $DEXCORE_RESULT_ROOT/analysis/taco/action_structure/<run>.
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
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="TACO root (default: the loader's own)")
    p.add_argument("--out", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--limit", type=int, default=None,
                   help="a random sample of N episodes (a smoke run); seeded by --seed")
    p.add_argument("--no-umap", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("action_structure")

    import pandas as pd
    from src.analysis.action_structure import chunks as CH
    from src.analysis.action_structure import compact as CP
    from src.analysis.action_structure import confound as CF
    from src.analysis.action_structure import plots as PL
    from src.analysis.action_structure import representations as REP
    from src.analysis.action_structure import screw as SC
    from src.analysis.action_structure import stats as ST
    from src.analysis.action_structure import structure as SR

    run = args.run_name or f"action_structure_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out) if args.out else result_root() / "analysis/taco/action_structure" / run
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "tensors").mkdir(exist_ok=True)
    if not args.no_plots:
        (out / "plots").mkdir(exist_ok=True)

    # ---------------------------------------------------------------- chunking + representations
    t0 = time.time()
    cs = CH.build(root=args.root, limit=args.limit, seed=args.seed)
    meta, w = cs.meta, cs.episode_weights
    log.info("%d chunks from %d episodes in %.1fs", len(cs), meta["episode_id"].nunique(),
             time.time() - t0)
    meta.to_csv(out / "tables" / "chunk_metadata.csv", index=False)

    reps = REP.build_all(cs)
    for name, X in reps.items():
        np.save(out / "tensors" / f"{name}_frames.npy", X.astype(np.float32))
        log.info("  %-13s %s", name, X.shape)

    # ------------------------------------------------------- Part 3: descriptive statistics
    # Motion statistics and virtual-joint (screw) statistics go in ONE table: both describe the
    # same chunk, and splitting them would mean two grouped-statistics tables saying nothing to
    # each other.
    stats_df = ST.chunk_statistics(cs)
    screw_df = pd.DataFrame(SC.chunk_screw_features(cs))
    stats_df = pd.concat([stats_df, screw_df], axis=1)
    motion_cols = [c for c in ST.statistic_columns(stats_df) if c not in SC.SCREW_COLUMNS]
    screw_cols = list(SC.SCREW_COLUMNS)
    stat_cols = motion_cols + screw_cols
    stats_df.to_csv(out / "tables" / "chunk_statistics.csv", index=False)
    summary_rows = []
    for g in ("action", "tool_cat", "target_cat"):
        s = stats_df.groupby(g)[stat_cols].median().reset_index().rename(columns={g: "level"})
        s.insert(0, "grouping", g)
        summary_rows.append(s)
    pd.concat(summary_rows).to_csv(out / "tables" / "statistics_by_group.csv", index=False)

    # The descriptive statistics are themselves representations -- three of them, because the
    # question "does a compact, absolute-pose-free description do better?" is only answerable by
    # scoring the compact ones on exactly the same footing as the 1152-dimensional trajectories.
    def _matrix(cols):
        X = stats_df[cols].to_numpy(dtype=np.float64)
        return np.nan_to_num(X, nan=np.nanmedian(X, axis=0))

    joint_cols = CP.columns(stat_cols)
    for name, cols in (("summary_stats", motion_cols), ("screw", screw_cols),
                       ("summary+screw", stat_cols), ("joint_motion", joint_cols),
                       ("joint_motion_euclid", joint_cols)):
        reps[name] = _matrix(cols)
        REP.CHANNELS[name] = 0
        np.save(out / "tensors" / f"{name}_frames.npy", reps[name].astype(np.float32))
        log.info("  %-19s %s", name, reps[name].shape)

    # A representation is a feature set AND the distance it is compared with. `joint_motion` is
    # defined with cosine; `joint_motion_euclid` is the same features under the default metric, so
    # the table shows what the metric alone is worth.
    metrics = {"joint_motion": CP.METRIC}

    # ------------------------------------------------------------ Part 1: confound audit
    audit = CF.audit(meta, weights=w)
    audit.to_csv(out / "tables" / "confound_association.csv", index=False)
    cover = CF.per_action_coverage(meta)
    cover.to_csv(out / "tables" / "action_object_coverage.csv", index=False)
    for a, b in (("action", "tool_cat"), ("action", "target_cat")):
        CF.contingency(meta, a, b).to_csv(out / "tables" / f"contingency_{a}_x_{b}.csv")
    ep = meta.drop_duplicates("episode_id")
    CF.audit(ep).to_csv(out / "tables" / "confound_association_episode_level.csv", index=False)
    CF.per_action_coverage(ep).to_csv(out / "tables" / "action_object_coverage_episode.csv",
                                      index=False)
    log.info("confound audit written")

    # ------------------------------------------------------------ Part 4: structure analysis
    rows, neighbours = [], {}
    for name, X in reps.items():
        t0 = time.time()
        metric = metrics.get(name, "euclidean")
        Z = REP.zscore(X, REP.CHANNELS.get(name, 0))
        np.save(out / "tensors" / f"{name}_flat_z.npy", Z.astype(np.float32))
        D = SR.pairwise_distances(Z, metric)

        km = SR.kmeans_scores(Z, meta, weights=w, seed=args.seed)
        sep = SR.separation(D, meta, weights=w)
        geo = SR.geometry_controlled(D, meta, level="mesh", seed=args.seed)
        geo.update(SR.geometry_controlled(D, meta, level="category", seed=args.seed))
        rows.append({"representation": name, "dim": int(Z.shape[1]), "metric": metric,
                     **{k: v for k, v in km.items() if k != "clusters"}, **sep, **geo})

        SR.confusion(km["clusters"], meta["action"]).to_csv(
            out / "tables" / f"confusion_{name}.csv")
        neighbours[name] = SR.neighbour_examples(D, meta, seed=args.seed)

        if not args.no_plots:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=args.seed).fit_transform(Z)
            PL.embedding_panel(xy, meta, out / "plots" / f"pca_{name}.png", f"PCA - {name}")
            if not args.no_umap:
                import umap
                u = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=args.seed,
                              n_components=2).fit_transform(Z)
                PL.embedding_panel(u, meta, out / "plots" / f"umap_{name}.png", f"UMAP - {name}")
            PL.confusion_heatmap(SR.confusion(km["clusters"], meta["action"]),
                                 out / "plots" / f"confusion_{name}.png",
                                 f"k-means vs action - {name}")
        log.info("%-14s done in %.1fs  (action NMI %.3f, geom P %.3f)", name, time.time() - t0,
                 km["action_nmi"], geo.get("mesh_p_closer", float("nan")))
        del D

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tables" / "representation_summary.csv", index=False)
    (out / "tables" / "neighbour_examples.json").write_text(
        json.dumps(neighbours, indent=2, ensure_ascii=False))

    # ------------------------------------------------------------------------- Part 3 plots
    if not args.no_plots:
        head = ["rel_trans_path_m", "rel_rot_path_rad", "rel_participation_ratio",
                "tool_target_dir_cos",
                # the virtual-joint block. `screw_axis_concentration` is the sign-invariant
                # replacement for `rel_axis_stability`, which reads near zero for any back-and-
                # forth motion and so cannot answer whether a joint axis is fixed.
                "screw_axis_concentration", "screw_dir_concentration",
                "screw_pitch_m_per_rad", "screw_reversal_fraction"]
        PL.stat_overview(stats_df, stat_cols, out / "plots" / "statistics_overview.png")
        PL.chunk_index_effect(stats_df, head, out / "plots" / "statistics_by_chunk_index.png")
        for s in head:
            for g in ("action", "tool_cat", "target_cat"):
                PL.stat_by_group(stats_df, s, g, out / "plots" / f"stat_{s}_by_{g}.png")
        log.info("wrote %d plots", len(list((out / "plots").glob("*.png"))))

    (out / "summary.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "n_chunks": int(len(cs)), "n_episodes": int(meta["episode_id"].nunique()),
        "n_actions": int(meta["action"].nunique()), "chunk_len": CH.CHUNK_LEN,
        "representations": list(reps), "seed": args.seed,
        "limit": args.limit, "umap": not args.no_umap,
    }, indent=2))

    cols = ["representation", "dim", "metric", "action_nmi", "tool_cat_nmi", "within_over_between",
            "nn_action_acc", "mesh_p_closer", "category_p_closer"]
    print(f"\n{len(cs)} chunks / {meta['episode_id'].nunique()} episodes -> {out}\n")
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
