#!/usr/bin/env python
"""How BimArt represents an ARCTIC object: part-wise BPS, and the contact map it carries.

BimArt never feeds the object mesh to its transformers. It resamples the mesh onto a fixed
512-point basis set, once per articulated part, and everything downstream -- geometry feature and
contact map alike -- lives on those 1024 slots. These figures show what that resampling keeps, what
it drops, and the one asymmetry that is easy to miss: the top part's slot->vertex assignment is
recomputed every frame, the bottom part's is fixed at frame 0.

  python scripts/viz_bimart_object_repr.py --categories laptop microwave
  python scripts/viz_bimart_object_repr.py --categories laptop --frame 400

Runs under the bimart conda env, from the BimArt checkout (its asset paths are relative):
  /home/uhnam/miniconda3/envs/bimart/bin/python scripts/viz_bimart_object_repr.py

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/object_repr/<run>/plots.
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
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

BIMART_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "BimArt"
LOG = logging.getLogger("bimart_object_repr")

TOP, BOTTOM = 0, 1
N_BPS = 512          # basis points, queried once per part -> 1024 slots
# BimArt's viz.norm_high (0.15 m) is a highlight ceiling, not the data range -- contact
# distances reach past 0.6 m, so fixed 0.15 m scales saturate these figures to solid black.
CONTACT_PCT = 97     # colour ceiling, as a percentile of the data being drawn


def result_root() -> Path:
    """Where analysis artefacts go. $DEXCORE_RESULT_ROOT overrides the server convention."""
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def load_processed(category: str, data_dir: Path):
    """The first processed obj-feature file for a category, with its sequence name."""
    hits = sorted(glob.glob(str(data_dir / category / "*" / "*processed_obj_features.npy")))
    if not hits:
        raise FileNotFoundError(f"no processed obj features for {category!r} under {data_dir}")
    # "use" sequences articulate the object; "grab" ones barely move it, which hides the whole
    # point of a part-wise representation
    used = [h for h in hits if "_use_" in Path(h).name] or hits
    path = Path(used[0])
    LOG.info("loading %s", path)
    return np.load(path, allow_pickle=True).item(), f"{path.parent.name}/{path.stem}"


def _equal_aspect(ax, pts, elev=20, azim=-62):
    c = (pts.max(0) + pts.min(0)) / 2
    r = (pts.max(0) - pts.min(0)).max() / 2 * 0.62  # crop the empty margin the basis cloud adds
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.view_init(elev=elev, azim=azim)


def fig_bps_geometry(data, mesh, bps_points, frame, category, out):
    """The resampling itself: 512 basis points, queried separately against each part."""
    verts = data["obj_cano_verts_dense"][frame]     # unit-sphere mesh, canonical space
    parts = np.asarray(mesh["parts"]).squeeze()
    inds = data["obj_cano_bps_inds"][frame]          # 1024 global vertex indices
    feat = data["obj_cano_bps"][frame]               # 1024 x 3 offset vectors

    fig = plt.figure(figsize=(16.5, 5.4))
    show = np.random.default_rng(0).choice(N_BPS, 110, replace=False)  # segments, thinned

    panels = [
        ("Object, coloured by part", None),
        (f"Top-part query -> slots 0:{N_BPS}", slice(0, N_BPS)),
        (f"Bottom-part query -> slots {N_BPS}:1024", slice(N_BPS, 2 * N_BPS)),
    ]
    for k, (title, sl) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        if sl is None:
            for pid, colour, name in ((TOP, "#1f77b4", "top (articulated)"),
                                      (BOTTOM, "#ff7f0e", "bottom (static)")):
                m = parts == pid
                ax.scatter(*verts[m].T, s=1.2, c=colour, alpha=.45,
                           label=f"{name}  n={m.sum()}")
            ax.legend(loc="upper left", fontsize=8, markerscale=6)
        else:
            ax.scatter(*verts.T, s=.7, c="#cccccc", alpha=.25)
            sel = verts[inds[sl]]
            ax.scatter(*bps_points.T, s=5, c="#2ca02c", alpha=.55, label="BPS basis (512)")
            ax.scatter(*sel.T, s=7, c="#d62728", alpha=.85, label="selected vertex")
            for i in show:  # basis point -> its nearest vertex; the stored feature is this vector
                ax.plot(*np.stack([bps_points[i], sel[i]]).T, c="#555555", lw=.45, alpha=.55)
            ax.legend(loc="upper left", fontsize=8, markerscale=2)
            title += f"\nmean |feature| = {np.linalg.norm(feat[sl], axis=-1).mean():.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        _equal_aspect(ax, np.vstack([verts, bps_points]))

    fig.suptitle(f"{category} -- part-wise BPS resampling, canonical space, frame {frame}"
                 f"  (articulation {data['obj_world_state'][frame, 0]:.3f} rad)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_slot_stability(data, category, out):
    """A slot is a place in space, not a point on the mesh -- for the top part.

    The top-part tree is rebuilt every frame, so slot i tracks whatever surface is nearest basis
    point i at time t. The bottom-part tree is built once at frame 0 and broadcast, so its slots
    are genuinely fixed vertices.
    """
    inds = data["obj_cano_bps_inds"]                 # T x 1024
    arti = data["obj_world_state"][:, 0]
    T = len(inds)
    rng = np.random.default_rng(0)

    fig = plt.figure(figsize=(15, 8.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 2.1, 1.5], hspace=.42, wspace=.22)

    ax = fig.add_subplot(gs[0, :])
    ax.plot(arti, c="#333333", lw=1.2)
    ax.set_ylabel("articulation\n(rad)", fontsize=9)
    ax.set_title(f"{category} -- does a BPS slot keep pointing at the same vertex?", fontsize=12)
    ax.set_xlim(0, T - 1)
    ax.grid(alpha=.25)

    for col, (sl, name) in enumerate((
            (slice(0, N_BPS), "top part (KD-tree rebuilt every frame)"),
            (slice(N_BPS, 2 * N_BPS), "bottom part (KD-tree built once, at frame 0)"))):
        ax = fig.add_subplot(gs[1, col])
        for i in rng.choice(N_BPS, 25, replace=False):
            ax.plot(inds[:, sl][:, i], lw=.8, alpha=.75)
        ax.set_title(f"{name}\nvertex index held by 25 sampled slots", fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("global mesh vertex index")
        ax.set_xlim(0, T - 1)
        ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[2, :])
    for sl, name, colour in ((slice(0, N_BPS), "top", "#1f77b4"),
                             (slice(N_BPS, 2 * N_BPS), "bottom", "#ff7f0e")):
        drift = (inds[:, sl] != inds[0, sl]).mean(axis=1) * 100
        ax.plot(drift, c=colour, lw=1.4, label=f"{name}: slots no longer on their frame-0 vertex")
    ax.set_xlabel("frame")
    ax.set_ylabel("% of slots reassigned")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=9)
    ax.grid(alpha=.25)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_feature_and_contact_maps(data, category, out):
    """The two tensors that actually reach the transformers, side by side on the same 1024 slots."""
    feat = np.linalg.norm(data["obj_cano_bps"], axis=-1)      # T x 1024
    left = np.stack([data["contact_dict"]["left"]["dist"][t, data["obj_cano_bps_inds"][t]]
                     for t in range(len(feat))])               # T x 1024
    right = np.stack([data["contact_dict"]["right"]["dist"][t, data["obj_cano_bps_inds"][t]]
                      for t in range(len(feat))])

    cmax = float(np.percentile(np.concatenate([left, right]), CONTACT_PCT))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for ax, (m, title, cmap, vmax) in zip(axes, (
            (feat, "BPS geometry  |nearest_vertex - basis|", "viridis", None),
            (left, "contact map, left hand  (slots 0:1024)", "magma_r", cmax),
            (right, "contact map, right hand  (slots 1024:2048)", "magma_r", cmax))):
        im = ax.imshow(m.T, aspect="auto", origin="lower", cmap=cmap, vmax=vmax,
                       interpolation="nearest")
        ax.axhline(N_BPS, c="#00ffff", lw=1.1, ls="--")
        ax.text(len(m) * .01, N_BPS * 1.5, "bottom-part slots", c="#00ffff", fontsize=8)
        ax.text(len(m) * .01, N_BPS * .12, "top-part slots", c="#00ffff", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame")
        ax.set_ylabel("BPS slot")
        fig.colorbar(im, ax=ax, fraction=.046, label="metres")

    fig.suptitle(f"{category} -- every slot carries geometry and contact in register "
                 f"(contact colour 0 to {cmax:.2f} m, p{CONTACT_PCT} of the data)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)


def fig_sparsification_cost(data, mesh, frame, category, out):
    """What the 1024-slot bottleneck costs: dense GT -> 1024 samples -> heat-extended back."""
    import potpourri3d as pp3d

    verts = data["obj_cano_verts_dense"][frame]
    inds = data["obj_cano_bps_inds"][frame]
    dense = data["contact_dict"]["left"]["dist"][frame]        # V, true per-vertex distance
    sparse = dense[inds]                                       # 1024, what the model sees
    recon = pp3d.PointCloudHeatSolver(verts).extend_scalar(list(inds), list(sparse))
    err = np.abs(recon - dense)

    cmax = float(np.percentile(dense, CONTACT_PCT))
    emax = float(np.percentile(err, 99))
    fig = plt.figure(figsize=(17, 5.2))
    for k, (vals, title, cmap, vmax) in enumerate((
            (dense, "GT dense contact (all %d verts)" % len(verts), "magma_r", cmax),
            (None, "the 1024 sampled slots", None, cmax),
            (recon, "heat-extended back to dense", "magma_r", cmax),
            (err, "|reconstruction - GT|", "inferno", emax))):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        if vals is None:
            ax.scatter(*verts.T, s=.7, c="#dddddd", alpha=.3)
            p = ax.scatter(*verts[inds].T, s=9, c=sparse, cmap="magma_r", vmin=0, vmax=vmax)
        else:
            p = ax.scatter(*verts.T, s=2.2, c=vals, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        _equal_aspect(ax, verts)
        fig.colorbar(p, ax=ax, fraction=.04, label="m")

    near = dense < 0.02  # vertices a hand is genuinely close to
    fig.suptitle(
        f"{category} frame {frame}, left hand -- sparsification cost:  "
        f"mean |err| {err.mean()*1000:.1f} mm, "
        f"on near-contact verts (<20 mm) {err[near].mean()*1000:.1f} mm "
        f"over {near.sum()} verts", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)
    return {"frame": int(frame), "mean_abs_err_mm": float(err.mean() * 1000),
            "near_contact_verts": int(near.sum()),
            "near_contact_err_mm": float(err[near].mean() * 1000) if near.any() else None}


def fig_coverage(mesh_dict, data_dir, out):
    """512 basis points against parts that differ in size by 6x across categories."""
    rows = []
    for cat in sorted(mesh_dict):
        hits = sorted(glob.glob(str(data_dir / cat / "*" / "*processed_obj_features.npy")))
        if not hits:
            continue
        parts = np.asarray(mesh_dict[cat]["parts"]).squeeze()
        inds = np.load(hits[0], allow_pickle=True).item()["obj_cano_bps_inds"]
        rows.append({
            "category": cat,
            "top_verts": int((parts == TOP).sum()), "bottom_verts": int((parts == BOTTOM).sum()),
            "top_unique_f0": int(len(np.unique(inds[0, :N_BPS]))),
            "bottom_unique": int(len(np.unique(inds[0, N_BPS:]))),
            "top_unique_all_frames": int(len(np.unique(inds[:, :N_BPS]))),
        })
        LOG.info("coverage %s: %s", cat, rows[-1])

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    x = np.arange(len(rows))
    cats = [r["category"] for r in rows]

    ax = axes[0]
    ax.bar(x - .2, [r["top_verts"] for r in rows], .4, label="top part vertices", color="#1f77b4")
    ax.bar(x + .2, [r["bottom_verts"] for r in rows], .4, label="bottom part vertices",
           color="#ff7f0e")
    ax.axhline(N_BPS, c="#d62728", ls="--", lw=1.3, label="512 basis points")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("vertices (log)")
    ax.set_title("part size vs. the fixed 512-point budget", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    ax = axes[1]
    ax.bar(x - .2, [100 * r["top_unique_f0"] / r["top_verts"] for r in rows], .4,
           label="top: unique verts hit at frame 0", color="#1f77b4")
    ax.bar(x + .2, [100 * r["bottom_unique"] / r["bottom_verts"] for r in rows], .4,
           label="bottom: unique verts hit (fixed)", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("% of the part's vertices sampled")
    ax.set_title("sampling density is wildly uneven across categories", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", out)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--categories", nargs="+", default=["laptop", "microwave"])
    ap.add_argument("--frame", type=int, default=None,
                    help="canonical frame to draw; default picks the most-contacted frame")
    ap.add_argument("--data-dir", type=Path, default=BIMART_ROOT / "data/arctic_processed_data")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.chdir(BIMART_ROOT)  # BimArt resolves its assets relatively
    sys.path.insert(0, str(BIMART_ROOT))
    from utils import data_util

    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/object_repr" / run / "plots"
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("writing to %s", out)

    bps_points = np.load("assets/bps_normalized_part_based.npy")
    mesh_dict = data_util.load_unit_mesh_dict()
    LOG.info("basis points %s, extent [%.3f, %.3f]", bps_points.shape,
             bps_points.min(), bps_points.max())

    summary = {"run": run, "bps_points": list(bps_points.shape), "categories": {}}
    for cat in args.categories:
        data, seq = load_processed(cat, args.data_dir)
        LOG.info("%s: %s, T=%d", cat, seq, len(data["obj_world_state"]))
        # default frame: the object open AND a hand near it, so the part split and the contact
        # are both visible. Articulation alone picks frames with the hands already withdrawn.
        if args.frame is not None:
            frame = args.frame
        else:
            T = len(data["obj_cano_bps_inds"])
            arti = data["obj_world_state"][:T, 0]
            nearest = data["contact_dict"]["left"]["dist"][:T].min(axis=1)
            open_enough = arti >= np.percentile(arti, 75)
            frame = int(np.where(open_enough, nearest, np.inf).argmin())
            LOG.info("%s: frame %d (articulation %.3f rad, nearest left vertex %.3f m)",
                     cat, frame, arti[frame], nearest[frame])

        fig_bps_geometry(data, mesh_dict[cat], bps_points, frame, cat,
                         out / f"01_bps_geometry_{cat}.png")
        fig_slot_stability(data, cat, out / f"02_slot_stability_{cat}.png")
        fig_feature_and_contact_maps(data, cat, out / f"03_feature_contact_maps_{cat}.png")
        cost = fig_sparsification_cost(data, mesh_dict[cat], frame, cat,
                                       out / f"04_sparsification_cost_{cat}.png")
        summary["categories"][cat] = {"sequence": seq, "frames": int(len(data["obj_world_state"])),
                                      "drawn_frame": frame, "sparsification": cost}
        del data

    summary["coverage"] = fig_coverage(mesh_dict, args.data_dir, out / "05_bps_coverage.png")
    (out.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
