#!/usr/bin/env python
"""Contact map, recorded against predicted: TACO's label next to what BimArt's contact prior emits.

The contact map is the distance, in metres, from each sampled OBJECT point to the nearest HAND
vertex -- so LOW means touching. One window is a [64 frames, 2048 points] array laid out the way
the store writes it:

    [   0: 512]  left hand  -> tool         [ 512:1024]  left hand  -> target
    [1024:1536]  right hand -> tool         [1536:2048]  right hand -> target

The BPS slots are anchored to a PLACE, not to a vertex -- basis point j sits at a fixed spot in the
canonical frame and records whichever object vertex is nearest it -- so a row of the heatmap is a
fixed location and the 512 basis positions can be scattered in space directly (`--spatial`).

Three views, all from the dumps `scripts/dump_contact_maps.py` writes:

  per window   GT | prediction | signed error | touch agreement, per hand, over the window's frames.
  spatial      the same window's contact painted on the 512 basis positions, GT above prediction.
  aggregate    the whole split: per-slot mean, distance distribution, touch IoU, per-block rates.

`--windows` names windows explicitly; otherwise they are drawn at evenly spaced QUANTILES of GT
touch fraction, so the set spans weak and strong contact instead of being picked for flattery.

    python scripts/viz_contact_maps.py --split test_1
    python scripts/viz_contact_maps.py --split test_2 --n-windows 8 --variant _fps
    python scripts/viz_contact_maps.py --split test_1 --windows 12 40 77

Figures go to $DEXCORE_RESULT_ROOT/analysis/bimart/contact_maps/<run>/plots.
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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

LOG = logging.getLogger("viz_contact_maps")

#: A hand's half of the vector, and the tool/target split inside it.
C_HALF = 1024
N_BPS = 512
#: `probe_contact_generalization.py` calls a slot touched below this, in metres.
TOUCH_M = 0.01
BLOCKS = (("left", 0), ("right", C_HALF))
#: Display names for the two objects the 2048-vector covers, and for whoever recorded the label.
#: TACO's two objects are a tool and a target; ARCTIC's are the two PARTS of one articulated mesh,
#: so its driver rebinds these through `set_parts()`. The vector layout is identical either way.
PARTS = ("tool", "target")
SOURCE = "TACO"
BLOCK_NAMES = ("L\u2192tool", "L\u2192target", "R\u2192tool", "R\u2192target")


def set_parts(a: str, b: str, source: str | None = None) -> None:
    """Rename the two objects for display. Does not touch the layout, only the labels."""
    global PARTS, BLOCK_NAMES, SOURCE
    PARTS = (a, b)
    BLOCK_NAMES = (f"L\u2192{a}", f"L\u2192{b}", f"R\u2192{a}", f"R\u2192{b}")
    if source:
        SOURCE = source

# ------------------------------------------------------------------------------------- palette
# Sequential magnitude: one blue hue, light -> dark. Distance is plotted with the ramp REVERSED so
# that touching (near zero) is the dark end and far recedes toward the surface.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
# Diverging polarity: the blue pole, a neutral GRAY midpoint, and a red pole stepped to match.
RED = ["#fbd6d2", "#f4a9a3", "#e8736d", "#e34948", "#c22e2d", "#96201f", "#6b1414"]
GRAY_MID = "#f0efec"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8982"
# Categorical slots 1-3, validated all-pairs in light mode.
CAT = {"both": "#2a78d6", "gt_only": "#eb6834", "pred_only": "#1baf7a"}

CMAP_DIST = LinearSegmentedColormap.from_list("dist_blue", BLUE[::-1])
#: Equal step count per arm: blue dark -> light, the gray midpoint, red light -> dark.
_BLUE_ARM = [BLUE[::-1][i] for i in (0, 2, 4, 6, 8, 10, 12)]
CMAP_ERR = LinearSegmentedColormap.from_list("err_div", _BLUE_ARM + [GRAY_MID] + RED)
CMAP_TOUCH = ListedColormap([SURFACE, CAT["both"], CAT["gt_only"], CAT["pred_only"]])


def result_root() -> Path:
    return Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": INK_MUTED, "axes.linewidth": .6,
        "grid.color": INK_MUTED, "grid.alpha": .25, "grid.linewidth": .5,
        "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
        "xtick.direction": "out", "ytick.direction": "out",
    })


# ---------------------------------------------------------------------------------------- data
def load(probe: Path, split: str):
    import pandas as pd

    npz = probe / f"{split}_contact_maps.npz"
    csv = probe / f"{split}_contact_maps_index.csv"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} -- run scripts/dump_contact_maps.py --split {split}")
    with np.load(npz) as z:
        pred, gt = z["pred"], z["gt"]
    idx = pd.read_csv(csv) if csv.exists() else None
    LOG.info("%s: %s windows of %d frames x %d slots", split, len(gt), gt.shape[1], gt.shape[2])
    return gt, pred, idx


def touch_frac(gt) -> np.ndarray:
    """Fraction of (frame, slot) entries the recorded map calls touched, per window."""
    out = np.empty(len(gt), dtype=np.float64)
    for s in range(0, len(gt), 64):
        out[s:s + 64] = (gt[s:s + 64].astype(np.float32) < TOUCH_M).mean(axis=(1, 2))
    return out


def pick_windows(gt, n: int, explicit=None):
    """Explicit ids, or n windows at evenly spaced quantiles of GT touch fraction."""
    tf = touch_frac(gt)
    if explicit:
        return list(explicit), tf
    n = min(n, len(gt))
    order = np.argsort(tf)
    qs = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [int(order[q]) for q in qs], tf


def window_stats(g, p):
    """Per-hand agreement numbers for one window. `g`, `p` are [T, 2048] float32 metres."""
    out = {}
    for side, o in BLOCKS:
        gh, ph = g[:, o:o + C_HALF], p[:, o:o + C_HALF]
        tg, tp = gh < TOUCH_M, ph < TOUCH_M
        inter = (tg & tp).sum(-1).astype(np.float64)
        union = np.maximum((tg | tp).sum(-1), 1).astype(np.float64)
        out[side] = {
            "mae_mm": float(np.abs(ph - gh).mean() * 1000),
            "gt_touch_frac": float(tg.mean()), "pred_touch_frac": float(tp.mean()),
            "touch_iou": float((inter / union).mean()),
            "iou_per_frame": inter / union,
            "gt_min_mm": gh.min(-1) * 1000, "pred_min_mm": ph.min(-1) * 1000,
            "gt_touch_per_frame": tg.mean(-1), "pred_touch_per_frame": tp.mean(-1),
        }
    return out


def window_label(idx, w: int) -> str:
    """`w` is the npz ROW. Dumps may subsample, so the dataset window id is a separate number."""
    if idx is None or w >= len(idx) or "window" not in idx:
        return f"row {w}"
    return f"row {w} (dataset window {int(idx.iloc[w]['window'])})"


def title_for(idx, w: int) -> str:
    if idx is None or w >= len(idx):
        return f"row {w}"
    r = idx.iloc[w]
    return (f"{r['sequence_id']}  ·  start frame {r['start']}  ·  {r['triplet']}"
            f"  ·  {PARTS[0]} mesh {r['tool_mesh']} / {PARTS[1]} mesh {r['target_mesh']}")


# ------------------------------------------------------------------------------------- figures
def _despine(ax):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)


def _map_axes(ax, title, show_y=True):
    """Slot axis with the two objects' boundary marked; frames on x."""
    ax.axhline(N_BPS, color=INK, lw=.8, ls="--", alpha=.55)
    ax.set_xlabel("frame in window")
    if show_y:
        ax.set_yticks([N_BPS / 2, N_BPS + N_BPS / 2])
        ax.set_yticklabels([f"{PARTS[0]}\n(512 slots)", f"{PARTS[1]}\n(512 slots)"], fontsize=8)
        ax.set_ylabel("BPS slot", labelpad=2)
    else:
        ax.set_yticks([])
    ax.set_title(title, fontsize=9.5)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_window(g, p, w, idx, tf, out, split, dmax_mm=100.0):
    """GT | prediction | signed error | touch agreement, per hand, for one window."""
    T = g.shape[0]
    st = window_stats(g, p)
    err_lim = max(5.0, float(np.percentile(np.abs(p - g) * 1000, 99)))

    fig = plt.figure(figsize=(19.5, 12.2))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[3.1, 3.1, .16, 1.5],
                  hspace=.46, wspace=.16, left=.055, right=.985, top=.885, bottom=.055)
    ext = [0, T, 0, C_HALF]
    im_d = im_e = None

    for row, (side, o) in enumerate(BLOCKS):
        gh = g[:, o:o + C_HALF].T * 1000                      # [1024 slots, T frames], mm
        ph = p[:, o:o + C_HALF].T * 1000
        s_ = st[side]

        ax = fig.add_subplot(gs[row, 0])
        im_d = ax.imshow(gh, aspect="auto", origin="lower", extent=ext, interpolation="nearest",
                         cmap=CMAP_DIST, norm=Normalize(0, dmax_mm))
        _map_axes(ax, f"recorded ({SOURCE})  ·  {side} hand", show_y=True)

        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(ph, aspect="auto", origin="lower", extent=ext, interpolation="nearest",
                  cmap=CMAP_DIST, norm=Normalize(0, dmax_mm))
        _map_axes(ax, f"predicted (BimArt)  ·  {side} hand", show_y=False)

        ax = fig.add_subplot(gs[row, 2])
        im_e = ax.imshow(ph - gh, aspect="auto", origin="lower", extent=ext,
                         interpolation="nearest", cmap=CMAP_ERR,
                         norm=TwoSlopeNorm(0, -err_lim, err_lim))
        _map_axes(ax, f"prediction − recorded  ·  MAE {s_['mae_mm']:.0f} mm", show_y=False)

        ax = fig.add_subplot(gs[row, 3])
        tg, tp = gh < TOUCH_M * 1000, ph < TOUCH_M * 1000
        code = np.where(tg & tp, 1, np.where(tg, 2, np.where(tp, 3, 0)))
        ax.imshow(code, aspect="auto", origin="lower", extent=ext, interpolation="nearest",
                  cmap=CMAP_TOUCH, norm=Normalize(0, 3))
        _map_axes(ax, f"touch below {TOUCH_M*1000:.0f} mm  ·  IoU {s_['touch_iou']:.2f}",
                  show_y=False)

    cb = fig.colorbar(im_d, cax=fig.add_subplot(gs[2, :2]), orientation="horizontal",
                      extend="max")
    cb.set_label("distance to nearest hand vertex (mm) — dark = touching, "
                 f"clipped at {dmax_mm:.0f}", fontsize=8.5)
    cb.outline.set_visible(False)

    cb = fig.colorbar(im_e, cax=fig.add_subplot(gs[2, 2]), orientation="horizontal",
                      extend="both")
    cb.set_label("prediction − recorded (mm):  blue = too close,  red = too far", fontsize=8.5)
    cb.outline.set_visible(False)

    ax = fig.add_subplot(gs[2, 3]); ax.axis("off")
    ax.legend(handles=[Patch(facecolor=CAT["both"], label="both agree"),
                       Patch(facecolor=CAT["gt_only"], label="recorded only (missed)"),
                       Patch(facecolor=CAT["pred_only"], label="predicted only (spurious)")],
              loc="center", ncol=3, fontsize=8.5, handlelength=1.1, columnspacing=1.2)

    # --- per-frame curves, so a window that is right on average but wrong in time shows it
    x = np.arange(T)
    ax = fig.add_subplot(gs[3, :2])
    for side, ls in (("left", "-"), ("right", "--")):
        ax.plot(x, st[side]["gt_touch_per_frame"] * 100, ls, color=CAT["gt_only"], lw=2,
                label=f"recorded, {side}")
        ax.plot(x, st[side]["pred_touch_per_frame"] * 100, ls, color=CAT["pred_only"], lw=2,
                label=f"predicted, {side}")
    ax.set_xlabel("frame in window"); ax.set_ylabel("slots touched (%)")
    ax.set_title("How much of the object is called touched?", fontsize=9.5)
    ax.grid(axis="y"); ax.legend(fontsize=7.5, ncol=2)
    _despine(ax)

    ax = fig.add_subplot(gs[3, 2])
    for side, ls in (("left", "-"), ("right", "--")):
        ax.plot(x, st[side]["gt_min_mm"], ls, color=CAT["gt_only"], lw=2,
                label=f"recorded, {side}")
        ax.plot(x, st[side]["pred_min_mm"], ls, color=CAT["both"], lw=2,
                label=f"predicted, {side}")
    ax.axhline(0, color=INK, lw=.9)
    ax.axhline(TOUCH_M * 1000, color=INK_MUTED, lw=.8, ls=":")
    ax.set_xlabel("frame in window"); ax.set_ylabel("closest slot (mm)")
    ax.set_title("Closest approach — below 0 is not a distance\n"
                 "(the sampler is unclipped, so it can emit one)", fontsize=9.5)
    ax.grid(axis="y"); ax.legend(fontsize=7.5, ncol=2)
    _despine(ax)

    ax = fig.add_subplot(gs[3, 3])
    for side, ls in (("left", "-"), ("right", "--")):
        ax.plot(x, st[side]["iou_per_frame"], ls, color=CAT["both"], lw=2,
                label=f"{side} hand")
    ax.set_ylim(-.02, 1.02)
    ax.set_xlabel("frame in window"); ax.set_ylabel("touch IoU")
    ax.set_title("Does it touch where TACO touches?", fontsize=9.5)
    ax.grid(axis="y"); ax.legend(fontsize=7.5)
    _despine(ax)

    q = float((tf < tf[w]).mean())
    fig.suptitle(
        f"Contact map — recorded against predicted\n{title_for(idx, w)}\n"
        f"{window_label(idx, w)}  ·  GT touch fraction {tf[w]:.3f}, "
        f"at quantile {q:.2f} of this split "
        f"(windows are drawn across the range, not chosen)",
        fontsize=12.5, y=.975)
    path = out / f"{split}_window_{w:05d}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s  (IoU L %.2f / R %.2f)", path.name,
             st["left"]["touch_iou"], st["right"]["touch_iou"])
    return {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)}
            for k, v in st.items()}


def fig_spatial(g, p, w, idx, basis, out, split, dmax_mm=100.0):
    """The same window painted on the 512 basis positions -- WHERE in the canonical frame."""
    gm = g.min(0) * 1000                       # closest approach over the window, per slot
    pm = p.min(0) * 1000
    lim = float(np.abs(basis).max()) * 1.08
    cols = [("left", "tool", 0), ("left", "target", N_BPS),
            ("right", "tool", C_HALF), ("right", "target", C_HALF + N_BPS)]

    fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.4))
    im = None
    for r, (name, arr) in enumerate(((f"recorded ({SOURCE})", gm), ("predicted (BimArt)", pm))):
        for c, (side, obj, o) in enumerate(cols):
            ax = axes[r, c]
            v = arr[o:o + N_BPS]
            order = np.argsort(-v)             # far first, so touching slots draw on top
            im = ax.scatter(basis[order, 0], basis[order, 2], c=v[order], s=26,
                            cmap=CMAP_DIST, norm=Normalize(0, dmax_mm),
                            linewidths=.3, edgecolors=SURFACE)
            n_touch = int((v < TOUCH_M * 1000).sum())
            disp = PARTS[0] if obj == "tool" else PARTS[1]
            ax.set_title(f"{name}\n{side} hand → {disp}  ·  {n_touch}/{N_BPS} slots touched",
                         fontsize=9.5)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xlabel("canonical x"); ax.set_ylabel("canonical z" if c == 0 else "")
            if c:
                ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.grid(alpha=.18)

    fig.subplots_adjust(left=.05, right=.93, top=.855, bottom=.075, hspace=.3, wspace=.12)
    cax = fig.add_axes([.94, .2, .010, .5])
    cb = fig.colorbar(im, cax=cax, extend="max")
    cb.set_label("closest approach over the window (mm)\ndark = touching", fontsize=8.5,
                 labelpad=8)
    cb.outline.set_visible(False)
    fig.suptitle(f"Where the contact sits, on the 512 fixed basis positions (x–z projection)\n"
                 f"{title_for(idx, w)}  ·  {window_label(idx, w)}", fontsize=12.5,
                 y=.975)
    path = out / f"{split}_window_{w:05d}_spatial.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path.name)


def fig_aggregate(gt, pred, split, out, dmax_mm=100.0, chunk=64):
    """The split as a whole: per-slot agreement, distance distribution, IoU spread, touch rates."""
    n, T, S = gt.shape
    g_sum = np.zeros(S); p_sum = np.zeros(S)
    edges = np.linspace(0, .40, 121)
    g_hist = np.zeros(len(edges) - 1); p_hist = np.zeros(len(edges) - 1)
    ious, blocks = [], {k: [0.0, 0.0] for k in BLOCK_NAMES}

    for s in range(0, n, chunk):
        gc = gt[s:s + chunk].astype(np.float32)
        pc = pred[s:s + chunk].astype(np.float32)
        g_sum += gc.sum(axis=(0, 1)); p_sum += pc.sum(axis=(0, 1))
        g_hist += np.histogram(gc, bins=edges)[0]
        p_hist += np.histogram(pc, bins=edges)[0]
        tg, tp = gc < TOUCH_M, pc < TOUCH_M
        for side, o in BLOCKS:
            inter = (tg[:, :, o:o + C_HALF] & tp[:, :, o:o + C_HALF]).sum(-1).astype(np.float64)
            union = np.maximum((tg[:, :, o:o + C_HALF] | tp[:, :, o:o + C_HALF]).sum(-1), 1)
            ious.append((inter / union).mean(-1))
        for i, k in enumerate(BLOCK_NAMES):
            sl = slice(i * N_BPS, (i + 1) * N_BPS)
            blocks[k][0] += float(tg[:, :, sl].sum()); blocks[k][1] += float(tp[:, :, sl].sum())

    denom = n * T
    g_mean, p_mean = g_sum / denom * 1000, p_sum / denom * 1000
    ious = np.concatenate(ious)
    per_block = denom * N_BPS

    fig = plt.figure(figsize=(19.5, 10.2))
    gs = GridSpec(2, 4, figure=fig, hspace=.4, wspace=.28,
                  left=.05, right=.98, top=.845, bottom=.075)

    # --- per-slot agreement, ONE FACET PER BLOCK. The slot index is an arbitrary ordering, so a
    # line across it would imply a continuity that is not there; the honest question is whether
    # the predicted per-slot mean tracks the recorded one, which is a scatter against identity.
    per_slot = {}
    for i, k in enumerate(BLOCK_NAMES):
        sl = slice(i * N_BPS, (i + 1) * N_BPS)
        gx, py = g_mean[sl], p_mean[sl]
        ax = fig.add_subplot(gs[0, i])
        lo = float(min(gx.min(), py.min())); hi = float(max(gx.max(), py.max()))
        pad = (hi - lo) * .06 + 1e-6
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK, lw=1, ls="--",
                label="identity")
        ax.scatter(gx, py, s=11, color=CAT["both"], alpha=.55, linewidths=0)
        r = float(np.corrcoef(gx, py)[0, 1]); bias = float((py - gx).mean())
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.set_xlabel("recorded mean (mm)")
        ax.set_ylabel("predicted mean (mm)" if i == 0 else "")
        ax.set_title(f"{k}   ·   r = {r:.3f},  bias {bias:+.1f} mm", fontsize=10)
        ax.grid(alpha=.2); ax.legend(fontsize=7.5, loc="upper left")
        _despine(ax)
        per_slot[k] = {"pearson_r": r, "bias_mm": bias,
                       "mean_abs_diff_mm": float(np.abs(py - gx).mean())}

    ax = fig.add_subplot(gs[1, 0])
    c = (edges[:-1] + edges[1:]) / 2 * 1000
    ax.plot(c, g_hist / g_hist.sum(), color=CAT["gt_only"], lw=2, label=f"recorded ({SOURCE})")
    ax.plot(c, p_hist / p_hist.sum(), color=CAT["both"], lw=2, label="predicted (BimArt)")
    ax.axvline(TOUCH_M * 1000, color=INK, lw=.8, ls=":",
               label=f"{TOUCH_M*1000:.0f} mm touch threshold")
    ax.set_xlabel("distance (mm)"); ax.set_ylabel("share of slot-frames")
    ax.set_title("Distance distribution — the recorded spike at 0 is contact", fontsize=10)
    ax.grid(axis="y"); ax.legend(fontsize=8)
    _despine(ax)

    ax = fig.add_subplot(gs[1, 1])
    ax.hist(ious, bins=40, range=(0, 1), color=CAT["both"], edgecolor=SURFACE, linewidth=.6)
    ax.axvline(float(ious.mean()), color=CAT["gt_only"], lw=2, label=f"mean {ious.mean():.3f}")
    ax.set_xlabel(f"touch IoU below {TOUCH_M*1000:.0f} mm (per window-hand)")
    ax.set_ylabel("window-hands")
    ax.set_title("Does it touch where TACO touches?", fontsize=10)
    ax.grid(axis="y"); ax.legend(fontsize=8)
    _despine(ax)

    ax = fig.add_subplot(gs[1, 2:])
    xs = np.arange(len(BLOCK_NAMES))
    gv = [blocks[k][0] / per_block * 100 for k in BLOCK_NAMES]
    pv = [blocks[k][1] / per_block * 100 for k in BLOCK_NAMES]
    ax.bar(xs - .21, gv, .40, color=CAT["gt_only"], label=f"recorded ({SOURCE})")
    ax.bar(xs + .21, pv, .40, color=CAT["pred_only"], label="predicted (BimArt)")
    for xi, (u, v) in enumerate(zip(gv, pv)):
        ax.text(xi - .21, u, f"{u:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
        ax.text(xi + .21, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(xs); ax.set_xticklabels(BLOCK_NAMES, fontsize=9)
    ax.set_ylabel("slots touched (%)")
    ax.set_title("Touch rate per hand–object block", fontsize=10)
    ax.grid(axis="y"); ax.legend(fontsize=8)
    _despine(ax)

    mae = float(np.abs(p_mean - g_mean).mean())
    fig.suptitle(f"Contact map, recorded against predicted — split `{split}`, "
                 f"{n} windows × {T} frames × {S} slots\n"
                 f"mean |per-slot mean difference| {mae:.1f} mm  ·  "
                 f"mean touch IoU {ious.mean():.3f}", fontsize=12.5, y=.955)
    path = out / f"aggregate_{split}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOG.info("wrote %s", path.name)
    return {"windows": int(n), "frames": int(T), "slots": int(S),
            "mean_abs_per_slot_mean_diff_mm": mae, "mean_touch_iou": float(ious.mean()),
            "median_touch_iou": float(np.median(ious)), "per_slot": per_slot,
            "touch_pct": {k: {"recorded": blocks[k][0] / per_block * 100,
                              "predicted": blocks[k][1] / per_block * 100}
                          for k in BLOCK_NAMES}}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test_1",
                    help="train, test_1 .. test_4 -- whichever was dumped")
    ap.add_argument("--variant", default="", choices=("", "_fps"),
                    help="'_fps' reads the corrected-contact-label dump")
    ap.add_argument("--probe", default=None, help="override the dump directory outright")
    ap.add_argument("--windows", type=int, nargs="+", default=None,
                    help="window ids; default is a quantile spread of GT touch fraction")
    ap.add_argument("--n-windows", type=int, default=6)
    ap.add_argument("--dmax-mm", type=float, default=100.0,
                    help="distance the sequential ramp saturates at")
    ap.add_argument("--no-spatial", action="store_true")
    ap.add_argument("--no-aggregate", action="store_true")
    ap.add_argument("--parts", nargs=2, metavar=("A", "B"), default=None,
                    help="display names for the two objects (ARCTIC: top bottom)")
    ap.add_argument("--source", default=None, help="who recorded the label, for figure labels")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()
    if args.parts or args.source:
        set_parts(*(args.parts or PARTS), source=args.source)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    style()

    probe = Path(args.probe) if args.probe else (
        result_root() / f"bimart_taco/contact_probe{args.variant}")
    run = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = result_root() / "analysis/bimart/contact_maps" / run
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    LOG.info("reading %s", probe)
    LOG.info("writing %s", out)

    gt, pred, idx = load(probe, args.split)
    picks, tf = pick_windows(gt, args.n_windows, args.windows)
    LOG.info("windows %s (GT touch fraction %s)", picks, [round(float(tf[w]), 3) for w in picks])

    basis = None
    if not args.no_spatial:
        from src.analysis.bimart import features
        basis = features.load_basis()
        LOG.info("basis %s", basis.shape)

    summary = {"run": run, "probe": str(probe), "split": args.split,
               "variant": args.variant, "touch_threshold_m": TOUCH_M,
               "dmax_mm": args.dmax_mm, "windows": {}}
    for w in picks:
        g = gt[w].astype(np.float32)
        p = pred[w].astype(np.float32)
        summary["windows"][str(w)] = {
            "npz_row": int(w),
            "dataset_window": (None if idx is None or "window" not in idx
                               else int(idx.iloc[w]["window"])),
            "gt_touch_frac": float(tf[w]),
            "quantile": float((tf < tf[w]).mean()),
            "meta": (None if idx is None or w >= len(idx)
                     else {k: (v.item() if hasattr(v, "item") else v)
                           for k, v in idx.iloc[w].to_dict().items()}),
            "hands": fig_window(g, p, w, idx, tf, plots, args.split, args.dmax_mm),
        }
        if basis is not None:
            fig_spatial(g, p, w, idx, basis, plots, args.split, args.dmax_mm)

    if not args.no_aggregate:
        summary["aggregate"] = fig_aggregate(gt, pred, args.split, plots, args.dmax_mm)

    (out / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=2, default=float))
    LOG.info("done -- %s", out)


if __name__ == "__main__":
    main()
