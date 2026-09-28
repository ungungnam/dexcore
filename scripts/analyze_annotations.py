#!/usr/bin/env python
"""How TACO's annotation fields relate to each other -- the dataset's design, before any motion.

  python scripts/analyze_annotations.py

Results go to $DEXCORE_RESULT_ROOT/analysis/taco/annotations/<run>.
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


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap(tab, path, title, fmt="{:.0f}", cmap="magma_r", vmax=None):
    plt = _plt()
    M = tab.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(1.6 + 0.52 * M.shape[1], 1.6 + 0.38 * M.shape[0]))
    im = ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    ax.set_xticks(range(M.shape[1]), tab.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(M.shape[0]), tab.index, fontsize=7)
    if M.shape[0] * M.shape[1] <= 400:
        thr = (vmax or M.max()) * 0.55
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] > 0:
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=5.5,
                            color="white" if M[i, j] > thr else "0.2")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(tab.columns.name or "", fontsize=8)
    ax.set_ylabel(tab.index.name or "", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="TACO root (default: the loader's own)")
    p.add_argument("--meta", default=None, help="a chunk/episode metadata csv, instead of indexing")
    p.add_argument("--out", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("annotations")

    import pandas as pd
    from src.analysis import annotations as AN
    from src.analysis.action_structure.confound import contingency

    if args.meta:
        meta = pd.read_csv(args.meta)
    else:
        from src.analysis.loaders import taco
        refs = taco.index(args.root)
        meta = pd.DataFrame([{
            "episode_id": r.sequence_id, "triplet": r.triplet, "action": r.action.verb,
            "tool_cat": r.action.tool, "target_cat": r.action.target} for r in refs])
        root = Path(args.root or taco.default_root())
        meshes = []
        for r in refs:                                   # mesh ids are the pose filenames
            d = root / "Object_Poses" / r.triplet / r.sequence
            got = {"tool_mesh": "", "target_mesh": ""}
            for fn in sorted(os.listdir(d)) if d.is_dir() else []:
                role, _, oid = fn[:-4].partition("_")
                if role in ("tool", "target"):
                    got[f"{role}_mesh"] = oid
            meshes.append(got)
        meta = pd.concat([meta, pd.DataFrame(meshes)], axis=1)

    ep = AN.episode_table(meta)
    tri = AN.triplet_table(ep)
    log.info("%d episodes, %d triplets, %d dates", len(ep), len(tri), ep["date"].nunique())

    run = args.run_name or f"annotations_{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(args.out) if args.out else result_root() / "analysis/taco/annotations" / run
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)

    ep.to_csv(out / "tables" / "episodes.csv", index=False)
    tri.to_csv(out / "tables" / "triplets.csv", index=False)

    # ---- association, at both units
    am_ep = AN.association_matrix(ep); am_ep.to_csv(out / "tables" / "association_episode.csv", index=False)
    am_tr = AN.association_matrix(tri, ("verb", "tool_cat", "target_cat"))
    am_tr.to_csv(out / "tables" / "association_triplet.csv", index=False)
    AN.occupancy(ep).to_csv(out / "tables" / "label_space_occupancy.csv", index=False)
    AN.mesh_structure(ep).to_csv(out / "tables" / "mesh_structure.csv", index=False)

    for key, against in (("verb", "tool_cat"), ("verb", "target_cat"), ("tool_cat", "verb"),
                         ("verb", "date"), ("triplet", "date")):
        AN.exclusivity(ep, key, against).to_csv(
            out / "tables" / f"exclusivity_{key}_over_{against}.csv", index=False)

    # ---- contingency tables and their pictures
    for a, b in (("verb", "tool_cat"), ("verb", "target_cat"), ("tool_cat", "target_cat"),
                 ("verb", "date")):
        tab = contingency(ep, a, b)
        tab.to_csv(out / "tables" / f"contingency_{a}_x_{b}.csv")
        heatmap(tab, out / "plots" / f"contingency_{a}_x_{b}.png",
                f"episodes per ({a}, {b})")
        pr = AN.field_profile(ep, a, b)
        heatmap(pr, out / "plots" / f"profile_{a}_over_{b}.png",
                f"{a}'s distribution over {b} (row-normalised)", fmt="{:.2f}", vmax=1.0)

    for key, against in (("verb", "tool_cat"), ("verb", "target_cat")):
        sim = AN.profile_similarity(ep, key, against)
        sim.to_csv(out / "tables" / f"similarity_{key}_by_{against}.csv")
        heatmap(sim, out / "plots" / f"similarity_{key}_by_{against}.png",
                f"{key} similarity by {against} profile (cosine)", fmt="{:.2f}",
                cmap="viridis", vmax=1.0)

    # ---- the association matrix itself, as a picture
    piv = am_ep.pivot(index="target", columns="predictor", values="u_target_given_predictor")
    heatmap(piv.fillna(0.0), out / "plots" / "association_u_matrix.png",
            "U(row | column): share of the row's entropy the column removes",
            fmt="{:.2f}", cmap="magma_r", vmax=1.0)

    (out / "summary.json").write_text(json.dumps({
        "written": datetime.now().isoformat(timespec="seconds"),
        "n_episodes": int(len(ep)), "n_triplets": int(len(tri)),
        "n_verbs": int(ep["verb"].nunique()), "n_tool_cat": int(ep["tool_cat"].nunique()),
        "n_target_cat": int(ep["target_cat"].nunique()),
        "n_tool_mesh": int(ep["tool_mesh"].nunique()),
        "n_target_mesh": int(ep["target_mesh"].nunique()),
        "n_dates": int(ep["date"].nunique()),
    }, indent=2))
    log.info("wrote %d tables and %d plots to %s",
             len(list((out / "tables").glob("*.csv"))), len(list((out / "plots").glob("*.png"))), out)

    top = am_ep.sort_values("u_target_given_predictor", ascending=False)
    print(f"\n{len(ep)} episodes / {len(tri)} triplets -> {out}\n")
    print("strongest directed associations (U = share of the target's entropy the predictor removes):")
    print(top.head(12)[["target", "predictor", "u_target_given_predictor", "nmi"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
