#!/usr/bin/env python
"""Contact map painted on the OBJECT MESH, recorded against predicted, for a few sampled episodes.

`viz_contact_maps.py` plots the contact vector as a slot x frame image, which is faithful but
abstract: BPS slot 300 is a place in the canonical frame and nothing about the picture says which
part of the brush that is. This script puts the same numbers on the geometry -- one panel per
(hand, object) pair, the mesh coloured by distance to the nearest hand vertex, so a wrong
prediction is legible as "it grips the bowl's rim instead of its wall".

THREE COLUMNS, NOT TWO, because two would conflate two different errors:

    recorded, dense        TACO's own label at full resolution -- every one of the ~4000 vertices
                           carries its own distance. This is the ground truth, unabridged.
    recorded, 512 slots    the SAME label, but gathered at the 512 BPS slots the model is trained
                           on and interpolated back. The gap between this and the dense column is
                           what the representation costs, before any model is involved.
    predicted, 512 slots   BimArt's contact prior, through the identical gather-and-interpolate
                           path. Comparing it against the MIDDLE column is the like-for-like read;
                           comparing against the left one also charges it for the label's coarseness.

The prediction exists only at 512 slots per (hand, object), so painting it on ~4000 vertices is an
interpolation -- inverse-distance over the three nearest sampled vertices. That is a display
choice, and it is applied identically to the middle column so the two stay comparable.

SLOT -> VERTEX. `obj_cano_bps_inds[t]` is (1024,): the first 512 are tool vertex ids, the next 512
are target vertex ids offset by the tool's vertex count, matching how `features.py` concatenates
them. The slot's vertex CHANGES between frames -- the basis is anchored to a place, not to the
object -- so the scatter is redone per frame and the per-episode summary takes the minimum over
frames at the VERTEX level, not at the slot level.

EPISODES ARE DRAWN FROM THE UPPER HALF by recorded contact, at evenly spaced quantiles, and every
figure prints the episode's quantile. An episode nobody touches renders four blank meshes and says
nothing, which is why the bottom of the range is skipped; within the half that is kept, nothing is
chosen for flattery. `--sequences` overrides the choice outright.

GIFS (`--gif`) answer the two things a still cannot. `time` plays the window frame by frame, so a
grip that drifts is visible as drift rather than as two different pictures; `turntable` spins the
per-episode summary, so contact on the far side of the mesh is not simply missing. Both are written
next to the stills.

  python scripts/viz_contact_mesh.py --split test_1 --episodes 4
  python scripts/viz_contact_mesh.py --split test_3 --sequences "(cut, knife, bowl)/20231020_261"
  python scripts/viz_contact_mesh.py --split test_1 --episodes 2 --gif both --no-frames

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/contact_mesh/<run>/plots.
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
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from scripts import viz_contact_maps as _vm
from scripts.viz_contact_maps import set_parts as vm_set_parts
from scripts.viz_contact_maps import (CAT, CMAP_DIST, INK, INK_2, INK_MUTED, N_BPS, SURFACE,
                                      TOUCH_M, result_root, style)

LOG = logging.getLogger("viz_contact_mesh")

DEFAULT_ROOT = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
#: (label, hand, object) for the four halves of the contact vector, in the store's order.
def set_parts(a: str, b: str, source: str | None = None) -> None:
    """Rename the two objects for display, here and in `viz_contact_maps`. Keys are unchanged."""
    global PAIRS
    PAIRS = ((f"L→{a}", "left", "tool"), (f"L→{b}", "left", "target"),
             (f"R→{a}", "right", "tool"), (f"R→{b}", "right", "target"))
    vm_set_parts(a, b, source)


PAIRS = (("L→tool", "left", "tool"), ("L→target", "left", "target"),
         ("R→tool", "right", "tool"), ("R→target", "right", "target"))
#: Two orbits, so contact hidden on the far side of the mesh is still shown.
VIEWS = ((20, 45), (20, 225))
#: A light direction in view space, for the shading that keeps the mesh readable as a solid.
LIGHT = np.array([.4, .5, .75])


# ------------------------------------------------------------------------------------ geometry
def face_colors(verts, faces, vals, dmax_mm):
    """Per-face colour from the mean of its vertices, dimmed by a cheap directional shade so the
    mesh reads as a solid. The shade multiplies brightness only; the hue stays the colormap's."""
    tri = verts[faces]
    rgba = CMAP_DIST(Normalize(0, dmax_mm, clip=True)(vals[faces].mean(1)))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = .62 + .38 * np.clip(n @ (LIGHT / np.linalg.norm(LIGHT)), 0, 1)
    rgba[:, :3] *= shade[:, None]
    return tri, rgba


_ZOOM_CACHE = {}


def fit_zoom(mesh_key, verts, faces, view, panel_in, cap=1.7, safety=.94):
    """The largest zoom that still keeps the whole SILHOUETTE inside a panel of this shape.

    matplotlib fits the data BOX to the axes at zoom=1, but a mesh never fills its box -- a kettle
    leaves the box corners empty, a spatula seen at an angle more so -- and how much slack that
    leaves depends on the mesh, the view AND the panel's aspect together. One fixed zoom therefore
    either crops the tall meshes or wastes the panel on the flat ones: at 1.35 a spatula still has
    room while a kettle is already cut off. Measuring it on a throwaway render of the same panel
    shape settles it, and the answer is cached per (mesh, view, panel), so a 32-frame gif pays for
    it once per panel rather than once per frame.
    """
    w_in, h_in = panel_in
    key = (mesh_key, tuple(np.round(view, 2)), round(w_in, 2), round(h_in, 2))
    if key in _ZOOM_CACHE:
        return _ZOOM_CACHE[key]

    fig = plt.figure(figsize=(max(w_in, .4), max(h_in, .4)), dpi=48)
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, verts, faces, np.zeros(len(verts)), 1.0, view, zoom=1.0)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.canvas.draw()
    ink = (np.asarray(fig.canvas.buffer_rgba())[:, :, :3] < 245).any(-1)
    plt.close(fig)

    rows, cols = np.any(ink, axis=1), np.any(ink, axis=0)
    if not rows.any() or not cols.any():
        _ZOOM_CACHE[key] = 1.0
        return 1.0
    fy = (np.flatnonzero(rows)[-1] - np.flatnonzero(rows)[0] + 1) / ink.shape[0]
    fx = (np.flatnonzero(cols)[-1] - np.flatnonzero(cols)[0] + 1) / ink.shape[1]
    z = float(np.clip(min(1 / max(fx, 1e-6), 1 / max(fy, 1e-6)) * safety, 1.0, cap))
    _ZOOM_CACHE[key] = z
    return z


def panel_inches(fig, spec):
    """The gridspec cell's size in inches, which is what `fit_zoom` calibrates against."""
    box = spec.get_position(fig)
    w, h = fig.get_size_inches()
    return box.width * w, box.height * h


def draw_mesh(ax, verts, faces, vals, dmax_mm, view, zoom=None, title=None):
    tri, rgba = face_colors(verts, faces, vals, dmax_mm)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=rgba, edgecolors="none", linewidths=0))
    lo, hi = verts.min(0), verts.max(0)
    c = (lo + hi) / 2
    h = np.maximum((hi - lo) / 2 * 1.04, (hi - lo).max() * .04)   # floor keeps flat meshes visible
    ax.set_xlim(c[0] - h[0], c[0] + h[0])
    ax.set_ylim(c[1] - h[1], c[1] + h[1])
    ax.set_zlim(c[2] - h[2], c[2] + h[2])
    ax.set_box_aspect(tuple(h / h.max()), zoom=1.0 if zoom is None else zoom)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=8.5, pad=-2)


def scatter_to_vertices(n_verts, ids, vals):
    """Per-vertex minimum of `vals` written at `ids`; NaN where no slot ever landed."""
    out = np.full(n_verts, np.inf)
    np.minimum.at(out, ids, vals)
    out[np.isinf(out)] = np.nan
    return out


def fill_gaps(verts, vals, k=3):
    """Inverse-distance interpolation from the vertices that carry a value to those that do not."""
    from scipy.spatial import cKDTree

    known = ~np.isnan(vals)
    if known.all() or not known.any():
        return np.nan_to_num(vals, nan=float(np.nanmax(vals)) if known.any() else 0.0)
    kk = min(k, int(known.sum()))
    d, i = cKDTree(verts[known]).query(verts[~known], k=kk)
    d = np.atleast_2d(d.T).T; i = np.atleast_2d(i.T).T
    w = 1.0 / np.maximum(d, 1e-9)
    out = vals.copy()
    out[~known] = (w * vals[known][i]).sum(1) / w.sum(1)
    return out


# ---------------------------------------------------------------------------------------- data
class Episode:
    """One sequence: its meshes, TACO's dense contact, and the prediction for its windows."""

    def __init__(self, row, windows, gt_all, pred_all, mesh_dict, root):
        from src.analysis.bimart.fps_contact import mesh_key

        self.sequence_id = row["sequence_id"]
        self.row = row
        self.windows = windows                       # [(npz row, sequence start frame)]
        self.mesh_key = {"tool": mesh_key(mesh_dict, row["tool_mesh"]),
                         "target": mesh_key(mesh_dict, row["target_mesh"])}
        self.mesh = {k: mesh_dict[v] for k, v in self.mesh_key.items()}
        self.n_tool = len(self.mesh["tool"]["verts_original"])
        with np.load(Path(root) / "sequences" / row["file"]) as z:
            self.dense = {"left": z["contact_left"], "right": z["contact_right"]}
            self.inds = z["obj_cano_bps_inds"]       # (T, 1024)
        self.pred = {w: pred_all[w].astype(np.float32) for w, _ in windows}
        self.gt = {w: gt_all[w].astype(np.float32) for w, _ in windows}

    def n_verts(self, obj):
        return len(self.mesh[obj]["verts_original"])

    def slot_ids(self, obj, t):
        """Vertex ids, in the object's OWN numbering, that the 512 slots landed on at frame t."""
        if obj == "tool":
            return self.inds[t, :N_BPS].astype(np.int64)
        return self.inds[t, N_BPS:].astype(np.int64) - self.n_tool

    @staticmethod
    def slot_slice(hand, obj):
        i = [p[1:] for p in PAIRS].index((hand, obj))
        return slice(i * N_BPS, (i + 1) * N_BPS)

    def dense_frame(self, hand, obj, t):
        d = self.dense[hand][t]
        return d[:self.n_tool] if obj == "tool" else d[self.n_tool:]

    def painted(self, source, hand, obj, frames, w):
        """Per-vertex metres for one (hand, object) pair over `frames`, closest approach kept.

        `source` is "dense" (TACO's label at full resolution) or "slots" / "pred" (gathered at the
        512 BPS slots, then interpolated -- the model's own resolution).
        """
        verts = self.mesh[obj]["verts_original"]
        if source == "dense":
            return np.minimum.reduce([self.dense_frame(hand, obj, t) for t in frames])
        arr = self.pred[w] if source == "pred" else self.gt[w]
        sl = self.slot_slice(hand, obj)
        start = dict(self.windows)[w]
        out = np.full(self.n_verts(obj), np.inf)
        for t in frames:
            np.minimum.at(out, self.slot_ids(obj, t), arr[t - start, sl])
        out[np.isinf(out)] = np.nan
        return fill_gaps(verts, out)


def build_episodes(split, probe, root, n_episodes, explicit, seed=0,
                   mesh_file="assets/taco_mesh_dict.npy"):
    import pandas as pd

    idx = pd.read_csv(probe / f"{split}_contact_maps_index.csv")
    seq_index = pd.read_csv(Path(root) / "sequence_index.csv").set_index("sequence_id")
    with np.load(probe / f"{split}_contact_maps.npz") as z:
        pred_all, gt_all = z["pred"], z["gt"]

    touched = (gt_all[:, :, :].astype(np.float32) < TOUCH_M).mean(axis=(1, 2)) \
        if len(gt_all) < 400 else None
    if touched is None:                              # chunk the big splits
        touched = np.empty(len(gt_all))
        for s in range(0, len(gt_all), 64):
            touched[s:s + 64] = (gt_all[s:s + 64].astype(np.float32) < TOUCH_M).mean(axis=(1, 2))
    # Row i of the npz is row i of the index csv. The `window` column is the DATASET window id,
    # which only coincides with the row when the dump took every window -- never key on it.
    idx = idx.reset_index(drop=True)
    idx["row"] = np.arange(len(idx))
    idx["touch"] = touched[idx["row"].values]

    per_seq = idx.groupby("sequence_id")["touch"].max().sort_values()
    if explicit:
        names = [s for s in explicit if s in per_seq.index]
        missing = [s for s in explicit if s not in per_seq.index]
        if missing:
            LOG.warning("not in %s: %s", split, missing)
    else:
        q = np.linspace(len(per_seq) * .45, len(per_seq) - 1, n_episodes).round().astype(int)
        names = [per_seq.index[i] for i in dict.fromkeys(q)]

    mesh_dict = np.load(Path(root) / mesh_file, allow_pickle=True).item()
    out = []
    for name in names:
        sub = idx[idx["sequence_id"] == name]
        wins = [(int(r["row"]), int(r["start"])) for _, r in sub.iterrows()]
        row = seq_index.loc[name].to_dict(); row["sequence_id"] = name
        ep = Episode(row, wins, gt_all, pred_all, mesh_dict, root)
        ep.best = int(sub.loc[sub["touch"].idxmax(), "row"])
        ep.touch = float(sub["touch"].max())
        ep.rank = float((per_seq.values < ep.touch).mean())
        out.append(ep)
        ep.window_id = int(sub.loc[sub["touch"].idxmax(), "window"])
        LOG.info("episode %s: %d windows, best row %d (window %d), GT touch %.3f (quantile %.2f)",
                 name, len(wins), ep.best, ep.window_id, ep.touch, ep.rank)
    return out


# ------------------------------------------------------------------------------------- figures
def _suptitle(fig, ep, split, extra):
    r = ep.row
    fig.suptitle(
        f"Contact on the object mesh — recorded against predicted\n"
        f"{ep.sequence_id}  ·  {r['triplet']}  ·  {PAIRS[0][0][2:]} mesh {r['tool_mesh']} / "
        f"{PAIRS[1][0][2:]} mesh {r['target_mesh']}  ·  split `{split}`\n{extra}",
        fontsize=12.5, y=.985)


def _colorbar(fig, rect, dmax_mm, label, label_top=False):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(0, dmax_mm), CMAP_DIST), cax=cax,
                      orientation="horizontal", extend="max")
    if label_top:
        cax.xaxis.set_label_position("top")
        cax.xaxis.set_ticks_position("bottom")
    cb.set_label(label, fontsize=8.5, labelpad=6 if label_top else 4)
    cb.outline.set_visible(False)


def fig_summary(ep, split, out, dmax_mm):
    """The whole window at once: closest approach per vertex, so contact regions accumulate."""
    w = ep.best
    start = dict(ep.windows)[w]
    frames = list(range(start, start + ep.gt[w].shape[0]))
    cols = [("recorded, dense", "dense"), ("recorded, 512 slots", "slots"),
            ("predicted, 512 slots", "pred")]

    fig = plt.figure(figsize=(17.5, 10.6))
    gs = GridSpec(len(PAIRS), 6, figure=fig, hspace=.02, wspace=.01,
                  left=.042, right=.997, top=.885, bottom=.085)
    stats = {}
    for r, (name, hand, obj) in enumerate(PAIRS):
        mesh = ep.mesh[obj]
        painted = {src: ep.painted(src, hand, obj, frames, w) for _, src in cols}
        stats[name] = {src: {"touched_verts": int((v < TOUCH_M).sum()),
                             "verts": int(len(v)), "min_mm": float(v.min() * 1000)}
                       for src, v in painted.items()}
        for c, (label, src) in enumerate(cols):
            for vi, view in enumerate(VIEWS):
                spec = gs[r, vi * 3 + c]
                ax = fig.add_subplot(spec, projection="3d")
                ax.set_facecolor(SURFACE)
                z = fit_zoom(ep.mesh_key[obj], mesh["verts_original"], mesh["faces"], view,
                             panel_inches(fig, spec))
                draw_mesh(ax, mesh["verts_original"], mesh["faces"], painted[src] * 1000,
                          dmax_mm, view, zoom=z)
                if r == 0:
                    ax.set_title(f"{label}\nview {vi + 1}", fontsize=9, pad=-4, color=INK)
        pct = stats[name]["dense"]["touched_verts"] / stats[name]["dense"]["verts"] * 100
        box = gs[r, 0].get_position(fig)
        fig.text(.022, box.y0 + box.height / 2,
                 f"{name}\n{pct:.1f}% touched",
                 ha="center", va="center", fontsize=11, color=INK, rotation=90,
                 linespacing=1.7)

    _colorbar(fig, [.32, .030, .36, .015], dmax_mm,
              f"closest approach over the window (mm) — dark = touching, clipped at {dmax_mm:.0f}")
    _suptitle(fig, ep, split,
              f"window {ep.window_id}, sequence frames {frames[0]}–{frames[-1]}  ·  "
              f"GT touch fraction {ep.touch:.3f}, at quantile {ep.rank:.2f} of this split")
    path = out / f"{split}_{_slug(ep.sequence_id)}_summary.png"
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path.name)
    return stats


def fig_frames(ep, split, out, dmax_mm, n_frames=6):
    """The two liveliest pairs, frame by frame, recorded above predicted."""
    w = ep.best
    start = dict(ep.windows)[w]
    T = ep.gt[w].shape[0]
    frames = [start + int(t) for t in np.linspace(0, T - 1, n_frames).round()]

    order = sorted(PAIRS, key=lambda p: -(ep.painted("dense", p[1], p[2],
                                                     list(range(start, start + T)),
                                                     w) < TOUCH_M).mean())
    pairs = order[:2]

    fig = plt.figure(figsize=(2.9 * n_frames, 8.8))
    gs = GridSpec(4, n_frames, figure=fig, hspace=.03, wspace=.01,
                  left=.052, right=.997, top=.855, bottom=.105)
    for pi, (name, hand, obj) in enumerate(pairs):
        mesh = ep.mesh[obj]
        for ri, (label, src) in enumerate(((f"recorded ({_vm.SOURCE})", "dense"),
                                           ("predicted (BimArt)", "pred"))):
            row = pi * 2 + ri
            z = fit_zoom(ep.mesh_key[obj], mesh["verts_original"], mesh["faces"], VIEWS[0],
                         panel_inches(fig, gs[row, 0]))
            for c, t in enumerate(frames):
                ax = fig.add_subplot(gs[row, c], projection="3d")
                ax.set_facecolor(SURFACE)
                v = ep.painted(src, hand, obj, [t], w) * 1000
                draw_mesh(ax, mesh["verts_original"], mesh["faces"], v, dmax_mm, VIEWS[0],
                          zoom=z)
                if row == 0:
                    ax.set_title(f"frame {t}", fontsize=9.5, pad=-2, color=INK)
            box = gs[row, 0].get_position(fig)
            fig.text(.030, box.y0 + box.height / 2, f"{name}\n{label}", ha="center",
                     va="center", fontsize=10.5, color=INK, rotation=90, linespacing=1.6)

    _colorbar(fig, [.34, .038, .32, .016], dmax_mm,
              f"distance to nearest hand vertex (mm) — dark = touching, clipped at {dmax_mm:.0f}")
    _suptitle(fig, ep, split,
              f"window {ep.window_id}  ·  the two pairs TACO touches most  ·  "
              f"predicted rows are interpolated from 512 slots at that single frame")
    path = out / f"{split}_{_slug(ep.sequence_id)}_frames.png"
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path.name)


# ----------------------------------------------------------------------------------------- gifs
def _frame_image(fig, colors=128):
    """The figure as a palettised PIL frame. GIF carries 256 colours, and the blue ramp plus the
    surface fits well inside that, so quantising here keeps the file small without visible banding.
    """
    from PIL import Image

    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(),
                           fig.canvas.buffer_rgba(), "raw", "RGBA", 0, 1).convert("RGB")
    return img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)


def _save_gif(frames, path, fps):
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(round(1000 / fps)), optimize=True, disposal=2)
    LOG.info("wrote %s  (%d frames, %.1f MB)", path.name, len(frames),
             path.stat().st_size / 1e6)


def _pairs_by_contact(ep, w, frames, k=2):
    """The k (hand, object) pairs TACO touches most in this window."""
    return sorted(PAIRS, key=lambda p: -(ep.painted("dense", p[1], p[2], frames, w)
                                         < TOUCH_M).mean())[:k]


def gif_time(ep, split, out, dmax_mm, stride, fps, n_pairs=2):
    """The window played frame by frame: recorded beside predicted, for the liveliest pairs."""
    w = ep.best
    start = dict(ep.windows)[w]
    T = ep.gt[w].shape[0]
    allf = list(range(start, start + T))
    pairs = _pairs_by_contact(ep, w, allf, n_pairs)
    steps = allf[::max(1, stride)]

    fig = plt.figure(figsize=(5.1 * n_pairs, 5.7), dpi=92)
    gs = GridSpec(2, n_pairs, figure=fig, hspace=.02, wspace=.01,
                  left=.03, right=.995, top=.835, bottom=.125)
    axes = {(r, c): fig.add_subplot(gs[r, c], projection="3d")
            for r in range(2) for c in range(n_pairs)}
    for (r, c), ax in axes.items():
        ax.set_facecolor(SURFACE)
    _colorbar(fig, [.30, .055, .40, .022], dmax_mm,
              f"distance to nearest hand vertex (mm) — dark = touching, clipped at {dmax_mm:.0f}",
              label_top=True)

    zooms = {c: fit_zoom(ep.mesh_key[obj], ep.mesh[obj]["verts_original"],
                         ep.mesh[obj]["faces"], VIEWS[0], panel_inches(fig, gs[0, c]))
             for c, (_, _, obj) in enumerate(pairs)}
    images = []
    for t in steps:
        for c, (name, hand, obj) in enumerate(pairs):
            mesh = ep.mesh[obj]
            for r, (label, src) in enumerate(((f"recorded ({_vm.SOURCE})", "dense"),
                                              ("predicted (BimArt)", "pred"))):
                ax = axes[(r, c)]
                ax.clear()
                draw_mesh(ax, mesh["verts_original"], mesh["faces"],
                          ep.painted(src, hand, obj, [t], w) * 1000, dmax_mm, VIEWS[0],
                          zoom=zooms[c])
                ax.set_title(f"{name}  ·  {label}", fontsize=9.5, pad=-2, color=INK)
        fig.suptitle(f"{ep.sequence_id}  ·  {ep.row['triplet']}  ·  split `{split}`\n"
                     f"frame {t}  of  {allf[0]}–{allf[-1]}", fontsize=11.5, y=.975)
        images.append(_frame_image(fig))
    plt.close(fig)
    _save_gif(images, out / f"{split}_{_slug(ep.sequence_id)}_time.gif", fps)


def gif_turntable(ep, split, out, dmax_mm, n_views, fps):
    """The episode summary, spun a full turn, so contact facing away is not read as absent."""
    w = ep.best
    start = dict(ep.windows)[w]
    frames = list(range(start, start + ep.gt[w].shape[0]))
    rows = ((f"recorded ({_vm.SOURCE})", "dense"), ("predicted (BimArt)", "pred"))
    painted = {(name, src): ep.painted(src, hand, obj, frames, w) * 1000
               for name, hand, obj in PAIRS for _, src in rows}

    fig = plt.figure(figsize=(14.5, 6.6), dpi=92)
    gs = GridSpec(2, len(PAIRS), figure=fig, hspace=.02, wspace=.01,
                  left=.02, right=.995, top=.835, bottom=.135)
    axes = {(r, c): fig.add_subplot(gs[r, c], projection="3d")
            for r in range(2) for c in range(len(PAIRS))}
    for ax in axes.values():
        ax.set_facecolor(SURFACE)
    _colorbar(fig, [.33, .058, .34, .020], dmax_mm,
              f"closest approach over the window (mm) — dark = touching, clipped at {dmax_mm:.0f}",
              label_top=True)

    images = []
    for az in np.linspace(0, 360, n_views, endpoint=False):
        for c, (name, hand, obj) in enumerate(PAIRS):
            mesh = ep.mesh[obj]
            for r, (label, src) in enumerate(rows):
                ax = axes[(r, c)]
                ax.clear()
                view = (24, float(az))
                draw_mesh(ax, mesh["verts_original"], mesh["faces"], painted[(name, src)],
                          dmax_mm, view,
                          zoom=fit_zoom(ep.mesh_key[obj], mesh["verts_original"], mesh["faces"],
                                        view, panel_inches(fig, gs[r, c])))
                ax.set_title(f"{name}  ·  {label}", fontsize=9.5, pad=-2, color=INK)
        fig.suptitle(f"{ep.sequence_id}  ·  {ep.row['triplet']}  ·  split `{split}`\n"
                     f"closest approach over window frames {frames[0]}–{frames[-1]}",
                     fontsize=11.5, y=.975)
        images.append(_frame_image(fig))
    plt.close(fig)
    _save_gif(images, out / f"{split}_{_slug(ep.sequence_id)}_turntable.gif", fps)


def _slug(seq_id):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in seq_id).strip("_")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_1")
    ap.add_argument("--variant", default="", choices=("", "_fps"))
    ap.add_argument("--probe", default=None)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--episodes", type=int, default=4,
                    help="sampled at even quantiles of the upper half by contact")
    ap.add_argument("--sequences", nargs="+", default=None)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--dmax-mm", type=float, default=50.0,
                    help="distance the ramp saturates at; tighter than the slot-image default "
                         "because a grasp region is what the mesh view is for")
    ap.add_argument("--no-frames", action="store_true")
    ap.add_argument("--gif", default="off", choices=("off", "time", "turntable", "both"))
    ap.add_argument("--gif-stride", type=int, default=2,
                    help="keep every Nth window frame in the `time` gif")
    ap.add_argument("--gif-views", type=int, default=36, help="azimuths in the `turntable` gif")
    ap.add_argument("--gif-fps", type=float, default=10.0)
    ap.add_argument("--parts", nargs=2, metavar=("A", "B"), default=None,
                    help="display names for the two objects (ARCTIC: top bottom)")
    ap.add_argument("--source", default=None, help="who recorded the label, for figure labels")
    ap.add_argument("--mesh-file", default="assets/taco_mesh_dict.npy")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    if args.parts or args.source:
        set_parts(*(args.parts or (PAIRS[0][0][2:], PAIRS[1][0][2:])), source=args.source)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    style()

    probe = Path(args.probe) if args.probe else (
        result_root() / f"bimart_taco/contact_probe{args.variant}")
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/contact_mesh" / run
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    LOG.info("reading %s", probe)
    LOG.info("writing %s", out)

    eps = build_episodes(args.split, probe, args.root, args.episodes, args.sequences,
                         mesh_file=args.mesh_file)
    summary = {"run": run, "split": args.split, "probe": str(probe),
               "touch_threshold_m": TOUCH_M, "dmax_mm": args.dmax_mm, "gif": args.gif,
               "episodes": {}}
    for ep in eps:
        summary["episodes"][ep.sequence_id] = {
            "window": ep.window_id, "npz_row": ep.best, "gt_touch_frac": ep.touch, "quantile": ep.rank,
            "triplet": ep.row["triplet"], "verb": ep.row["verb"],
            "tool_mesh": str(ep.row["tool_mesh"]), "target_mesh": str(ep.row["target_mesh"]),
            "pairs": fig_summary(ep, args.split, plots, args.dmax_mm),
        }
        if not args.no_frames:
            fig_frames(ep, args.split, plots, args.dmax_mm, args.frames)
        if args.gif in ("time", "both"):
            gif_time(ep, args.split, plots, args.dmax_mm, args.gif_stride, args.gif_fps)
        if args.gif in ("turntable", "both"):
            gif_turntable(ep, args.split, plots, args.dmax_mm, args.gif_views, args.gif_fps)

    (out / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2, default=float))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
