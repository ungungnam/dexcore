"""Qualitative figure "z_moves_with_contact" for the page block F3-diagB (z has a meaningful temporal geometry).

Per dataset ONE test sequence, chosen by a rule (below), drawn as two stacked parts:

* a chart over the 63 frame-to-frame transitions of the sequence: how far the recorded contact map moves,
  ``|C_t+1 - C_t|`` (blue), and how far the latent moves, ``|z_t+1 - z_t|`` (orange). Both are the quantities of
  the diagnostic at its one-frame horizon: L2 norm of the raw 512-point map, L2 norm of z standardised with
  the training statistics, each divided by the mean distance between two random training frames (the study's
  normalisers, read from its table). Next to it the same z path projected on the first two principal
  components of the TRAINING latents;
* the recorded contact map on the object, hand ghosted, at six frames: three frames before, just before and
  just after each of the two largest contact steps.

Selection rule. Per test sequence the Spearman correlation between its two 63-value step series; the sequence
drawn is the median one. With an even number of test sequences no sequence equals the median: the two middle
sequences are equally far from it and the smaller example index is taken.

Nothing is run: z is the cached encoding of the frozen Stage-1 encoder, everything is read from
``reports/z_temporal_diagnostic/<dataset>/cache/frames.npz``. Run as

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_z_moves_with_contact.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageFont
from scipy.stats import spearmanr

import qlib

FIG_ID = "z_moves_with_contact"
BLOCK = "F3-diagB"
TITLE = "The latent z moves when the contact moves"
DIAG = qlib.RESULT_ROOT / "reports/z_temporal_diagnostic"
METRICS = DIAG / "temporal_geometry_metrics.csv"      # normalisers and the dataset-level correlations
VALUES = DIAG / "report_values.json"                   # the report's mean over its three horizons
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
PAGE_QUOTE = {"taco": 0.85, "arctic": 0.81}            # Spearman printed in block F3-diagB of docs/index.html
HAND_WORD = {"L": "left", "R": "right"}

HORIZON = 1          # frame-to-frame steps: 63 transitions per 64-frame sequence
MIN_GAP = 8          # the second step is at least this many transitions from the first (two separate events)
CONTEXT = 3          # frames between the earlier context frame and the frame just before a step
PANEL = 300          # side of a rendered panel, px
DPI = 100
# One camera rule per dataset (a display choice; both are computed from the geometry, see camera_for):
#   bowl      the contact sits on the wall below the rim, which the broad-side view sees edge-on
#   scissors  a flat object: the direction that faces the contact looks at it edge-on
VIEW = {"taco": "facing the contact", "arctic": "broad side, seen from the side away from the hand"}

GUIDE = "#8a8983"    # frame markers and their connectors
BAND = "#ecebe6"     # the two selected steps
MUTED = "#6b6a64"    # the page's --ink-3


# ------------------------------------------------------------------------------ data
@dataclass(frozen=True)
class Cache:
    """What the figure reads of one dataset's diagnostic cache (frames.npz) and of the study's tables."""
    dataset: str
    example: np.ndarray        # (M,) example index of every cached sequence
    split: np.ndarray          # (M,) "train" / "val" / "test"
    contact: np.ndarray        # (M, 64, 512) float16 recorded canonical maps
    latent: np.ndarray         # (M, 64, 64) float32 z standardised with the training mean / std
    scale_contact: float       # mean |C_a - C_b| over random pairs of training frames (the study's normaliser)
    scale_latent: float        # the same for standardised z
    table: pd.Series           # row (dataset, h = HORIZON) of temporal_geometry_metrics.csv
    path: Path


def load_cache(dataset: str) -> Cache:
    path = DIAG / dataset / "cache/frames.npz"
    with np.load(path, allow_pickle=True) as z:
        example, split, contact = z["seq"], z["split"].astype(str), z["C"]
        latent = (z["z"].astype(np.float32) - z["z_mu"].astype(np.float32)) / z["z_sd"].astype(np.float32)
    table = pd.read_csv(METRICS).set_index(["dataset", "h"]).loc[(dataset, HORIZON)]
    return Cache(dataset, example, split, contact, latent, float(table.scale_C), float(table.scale_z), table, path)


def step_series(cache: Cache, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(contact step, latent step), each (len(rows), 63): the diagnostic's delta_C and delta_z at h = 1."""
    contact = cache.contact[rows].astype(np.float32)
    latent = cache.latent[rows]
    d_contact = np.linalg.norm(contact[:, 1:] - contact[:, :-1], axis=2) / cache.scale_contact
    d_latent = np.linalg.norm(latent[:, 1:] - latent[:, :-1], axis=2) / cache.scale_latent
    return d_contact, d_latent


def median_sequence(rho: np.ndarray, example: np.ndarray) -> tuple[int, dict[str, Any]]:
    """Position (into rho) of the median sequence and the evidence for the pick."""
    order = np.lexsort((example, rho))                     # ascending correlation, ties by example index
    n = len(order)
    middle = [int(order[n // 2])] if n % 2 else [int(order[n // 2 - 1]), int(order[n // 2])]
    pick = min(middle, key=lambda i: int(example[i]))
    rank = int(np.flatnonzero(order == pick)[0])
    near = order[max(rank - 3, 0):rank + 4]
    evidence = {
        "metric": "Spearman correlation between the 63 contact steps and the 63 latent steps of one test sequence",
        "n_candidates": n,
        "median": float(np.median(rho)),
        "middle_sequences": [{"example": int(example[i]), "spearman": float(rho[i])} for i in middle],
        "chosen_example": int(example[pick]),
        "chosen_spearman": float(rho[pick]),
        "chosen_rank_ascending_from_1": rank + 1,
        "distance_to_median": float(abs(rho[pick] - np.median(rho))),
        "neighbours_in_sorted_order": [{"rank": int(np.flatnonzero(order == i)[0]) + 1, "example": int(example[i]),
                                        "spearman": float(rho[i])} for i in near],
        "quantiles_over_test_sequences": {q: float(np.quantile(rho, v)) for q, v in
                                          (("min", 0), ("q10", .1), ("q25", .25), ("q50", .5), ("q75", .75), ("q90", .9), ("max", 1))},
        "full_ranking": "panels.npz: <dataset>_selection/example and <dataset>_selection/spearman, sorted ascending",
    }
    return pick, evidence


def largest_steps(contact_step: np.ndarray) -> list[int]:
    """The largest contact step and the largest one at least MIN_GAP transitions away, in time order.
    Step k is the transition from frame k to frame k + 1."""
    first = int(np.argmax(contact_step))
    far = np.flatnonzero(np.abs(np.arange(len(contact_step)) - first) >= MIN_GAP)
    second = int(far[np.argmax(contact_step[far])])
    return sorted((first, second))


def frames_around(step: int) -> list[int]:
    """CONTEXT frames before the step, just before it, just after it (frames step and step + 1 bracket it)."""
    context = step - CONTEXT if step - CONTEXT >= 0 else step + 1 + CONTEXT     # mirrored at the sequence start
    return sorted((context, step, step + 1))


def principal_plane(train_latent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean (64,), two components (2, 64), variance share of every component) of the training latents."""
    x = train_latent.reshape(-1, train_latent.shape[-1]).astype(np.float64)
    mean = x.mean(0)
    values, vectors = np.linalg.eigh(np.cov((x - mean).T))
    values, vectors = values[::-1], vectors[:, ::-1].T
    comps = vectors[:2] * np.sign(vectors[:2][np.arange(2), np.abs(vectors[:2]).argmax(1)])[:, None]   # fixed signs
    return mean, comps, values / values.sum()


# ------------------------------------------------------------------------------ one drawn sequence
@dataclass
class Drawn:
    dataset: str
    ex: qlib.Example
    row: int                    # row of the sequence in frames.npz
    contact: np.ndarray         # (64, 512) float32
    contact_step: np.ndarray    # (63,)
    latent_step: np.ndarray     # (63,)
    spearman: float
    steps: list[int]            # the two selected transitions
    frames: list[int]           # the six drawn frames
    path2d: np.ndarray          # (64, 2) z in the principal plane of the training latents
    kept_train: float           # share of the training latents' variance in that plane
    kept_sequence: float        # share of this sequence's own latent variance in that plane
    spearman_2d: float          # Spearman between the contact steps and the steps of the projected path
    evidence: dict[str, Any]
    checks: dict[str, Any]
    cache: Cache


def prepare(dataset: str) -> tuple[Drawn, dict[str, np.ndarray]]:
    cache = load_cache(dataset)
    test = np.flatnonzero(cache.split == "test")
    d_contact, d_latent = step_series(cache, test)
    rho = np.array([spearmanr(a, b)[0] for a, b in zip(d_contact, d_latent)])
    assert np.isfinite(rho).all(), "a test sequence has a constant step series"
    pick, evidence = median_sequence(rho, cache.example[test])
    row, n = int(test[pick]), int(cache.example[test[pick]])
    ex = qlib.load_example(dataset, n)
    contact = cache.contact[row].astype(np.float32)
    steps = largest_steps(d_contact[pick])
    frames = [f for k in steps for f in frames_around(k)]
    assert len(set(frames)) == 6 and frames == sorted(frames) and 0 <= frames[0] and frames[-1] < qlib.T, frames

    mean, comps, share = principal_plane(cache.latent[cache.split == "train"])
    latent = cache.latent[row].astype(np.float64)
    path2d = (latent - mean) @ comps.T
    d_path = np.linalg.norm(path2d[1:] - path2d[:-1], axis=1)

    # ---- cross-checks against what the study saved (reported, not hidden)
    pooled = float(spearmanr(d_contact.ravel(), d_latent.ravel())[0])
    with np.load(DIAG / dataset / "geometry_arrays.npz") as g:
        mine = g[f"h{HORIZON}|r"] == row
        assert np.array_equal(g[f"h{HORIZON}|t"][mine], np.arange(qlib.T - HORIZON))
        diff_contact = float(np.abs(g[f"h{HORIZON}|dC"][mine] - d_contact[pick]).max())
        diff_latent = float(np.abs(g[f"h{HORIZON}|dz"][mine] - d_latent[pick]).max())
    info = json.loads((DIAG / dataset / "cache/frames.json").read_text())
    checks = {
        "pooled_spearman_recomputed_here": pooled,
        "pooled_spearman_in_table": float(cache.table.spearman_C_z),
        "pooled_abs_difference": abs(pooled - float(cache.table.spearman_C_z)),
        "pooled_transitions_here": int(d_contact.size), "pooled_transitions_in_table": int(cache.table.n_test),
        "step_series_vs_geometry_arrays_max_abs_diff": {"contact": diff_contact, "latent": diff_latent},
        "cached_map_vs_sequences_npz_max_abs_diff": float(np.abs(contact - ex.C).max()),
        "n_test_sequences_here": int(len(test)), "n_test_sequences_in_frames_json": int(info["n_sequences"]["test"]),
        "table_rounds_to_the_page_quote": bool(round(float(cache.table.spearman_C_z), 2) == PAGE_QUOTE[dataset]),
    }
    order = np.argsort(rho, kind="stable")
    selection = {"example": cache.example[test][order], "spearman": rho[order]}
    drawn = Drawn(dataset, ex, row, contact, d_contact[pick], d_latent[pick], float(rho[pick]), steps, frames, path2d,
                  float(share[:2].sum()), float(path2d.var(0).sum() / latent.var(0).sum()),
                  float(spearmanr(d_path, d_contact[pick])[0]), evidence, checks, cache)
    return drawn, selection


# ------------------------------------------------------------------------------ 3-D panels
def vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """(V, 3) unit normals: face normals (area-weighted) summed on their vertices."""
    tri = verts[faces]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    out = np.zeros_like(verts)
    for j in range(3):
        np.add.at(out, faces[:, j], face_n)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def camera_for(d: Drawn, values: list[np.ndarray]) -> qlib.Camera:
    """One camera for the six panels of a sequence, fitted on the object and the hand at all six frames."""
    meshes = [d.ex.mesh(t) for t in d.frames]
    hands = [d.ex.hand(t)[0] for t in d.frames]
    object_all, hand_all = np.concatenate([v for v, _ in meshes]), np.concatenate(hands)
    if VIEW[d.dataset] == "facing the contact":
        # the direction most of the drawn contact faces: contact-weighted mean of the surface normals; "up" is the
        # object's thinnest principal axis on the camera's side (a bowl stands upright, seen from above)
        direction = sum((np.nan_to_num(val)[:, None] * vertex_normals(v, f)).sum(0) for (v, f), val in zip(meshes, values))
        thin = np.linalg.eigh(np.cov((object_all - object_all.mean(0)).T))[1][:, 0]
        up = thin if thin @ direction >= 0 else -thin
    else:
        # the object's broad side, seen from the side the hand is NOT on (the ghost hand stays behind the object)
        toward_hand, up = qlib.hand_side_view(object_all, hand_all)
        direction = -toward_hand
    return qlib.fit_camera([object_all, hand_all], direction, up)


def render_panels(d: Drawn, vmax: float, values: list[np.ndarray], camera: qlib.Camera) -> list[np.ndarray]:
    panels = []
    for t, val in zip(d.frames, values):
        verts, faces = d.ex.mesh(t)
        hand_verts, hand_faces = d.ex.hand(t)
        panels.append(qlib.render([qlib.mesh_item(verts, faces, rgb=qlib.contact_rgb(val, vmax)),
                                   qlib.hand_item(hand_verts, hand_faces, **qlib.GHOST)], camera, (PANEL, PANEL)))
    return panels


# ------------------------------------------------------------------------------ layout shared with qlib.grid
@dataclass(frozen=True)
class Layout:
    """Pixel layout of a qlib.grid of six panels with a row label (same formulas as qlib.grid, checked in locate)."""
    width: int
    x0: int          # left edge of the first panel
    panel: int
    gap: int
    pad: int
    font_px: int

    @property
    def line(self) -> int:
        return int(np.ceil(self.font_px * 1.32))

    def centre(self, j: int) -> float:
        return self.x0 + j * (self.panel + self.gap) + self.panel / 2


def locate(grid_image: Image.Image, first_panel: np.ndarray, n_col: int) -> Layout:
    """The layout of ``grid_image``; raises if the first panel is not where the formulas put it."""
    info = grid_image.info["qlib_layout"]
    w = info["panel"][0]
    gap = max(6, w // 50)
    pad = 2 * gap
    x0 = info["width"] - pad - (n_col * w + (n_col - 1) * gap)
    pixels = np.asarray(grid_image)
    if not any(np.array_equal(pixels[y:y + w, x0:x0 + w], first_panel) for y in range(pixels.shape[0] - w + 1)):
        raise RuntimeError("qlib.grid placed the panels differently from the layout assumed for the chart")
    return Layout(info["width"], x0, w, gap, pad, info["font_px"])


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> str:
    """Greedy word wrap of ``text`` to ``width`` px (DejaVu Sans metrics, the font matplotlib draws with)."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if font.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return "\n".join(lines + [line])


# ------------------------------------------------------------------------------ the chart band
def chart_band(lay: Layout, d: Drawn) -> np.ndarray:
    """The band above a row of panels: title and legend, the two step series over the frame axis (blue contact,
    orange latent, one shared y axis), the z path in the principal plane, and connectors from the six marked
    frames to the six panels below. Returns (height, lay.width, 3) uint8."""
    f_px, line, w = lay.font_px, lay.line, lay.panel
    width = lay.width
    lw = 2.0 * width / 1100.0                       # 2 px lines when the figure is shown 1100 px wide
    y_title = lay.pad
    y_axes = y_title + line + lay.gap + round(1.75 * f_px)      # below the title row and the frame-number row
    plot_h, fan_h = w, round(0.28 * w)
    height = y_axes + plot_h + fan_h
    axes_right = lay.x0 + 5 * w + 4 * lay.gap - f_px             # the time axis spans panels 1 to 5
    inset_left = lay.x0 + 5 * (w + lay.gap)

    fig = Figure(figsize=(width / DPI + 1e-6, height / DPI + 1e-6), dpi=DPI, facecolor="white")
    canvas = FigureCanvasAgg(fig)
    pt = 72.0 / DPI                                 # px -> points
    font = {"family": "DejaVu Sans", "size": f_px * pt, "color": qlib.INK}

    def rect(left: float, top: float, w_px: float, h_px: float) -> list[float]:
        return [left / width, 1 - (top + h_px) / height, w_px / width, h_px / height]

    # ---- title and legend
    m = d.ex.meta
    fig.text(lay.pad / width, 1 - y_title / height, f"{LABEL[d.dataset]}", va="top", ha="left", weight="bold", **font)
    title_w = ImageFont.truetype(_bold_font_path(), f_px).getlength(LABEL[d.dataset] + "  ")
    subtitle = f"one test sequence: {HAND_WORD[m.hand]} hand, object: {m.category}"
    fig.text((lay.pad + title_w) / width, 1 - y_title / height, subtitle, va="top", ha="left", **font)
    handles = [Line2D([], [], color=qlib.BLUE, lw=lw * pt), Line2D([], [], color=qlib.ORANGE, lw=lw * pt),
               Patch(facecolor=BAND, edgecolor=qlib.RULE, lw=pt)]
    legend = fig.legend(handles, ["contact step", "latent step (z)", f"two largest contact steps, {MIN_GAP}+ frames apart"], loc="upper right",
                        bbox_to_anchor=((width - lay.pad) / width, 1 - (y_title - 0.1 * f_px) / height), ncol=3, frameon=False,
                        prop={"family": font["family"], "size": font["size"]}, labelcolor=qlib.INK, borderpad=0,
                        borderaxespad=0, handlelength=1.2, handletextpad=0.5, columnspacing=1.1)

    # ---- the two step series
    ax = fig.add_axes(rect(lay.x0, y_axes, axes_right - lay.x0, plot_h))
    x_lo, x_hi = -1.0, float(qlib.T)
    mid = np.arange(qlib.T - 1) + 0.5               # step k (frame k -> k + 1) is drawn between the two frames
    top = 1.12 * float(max(d.contact_step.max(), d.latent_step.max()))
    for k in d.steps:
        ax.axvspan(k, k + 1, color=BAND, lw=0, zorder=0)
    for f in d.frames:
        ax.plot([f, f], [0, top], color=GUIDE, lw=0.5 * lw * pt, zorder=1, solid_capstyle="butt")
    ax.plot(mid, d.latent_step, color=qlib.ORANGE, lw=lw * pt, zorder=3, solid_joinstyle="round")
    ax.plot(mid, d.contact_step, color=qlib.BLUE, lw=lw * pt, zorder=4, solid_joinstyle="round")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(0, top)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(0, qlib.T, 10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
    ax.tick_params(axis="both", length=0.22 * f_px * pt, width=pt * lw / 2, color=qlib.RULE, pad=0.25 * f_px * pt,
                   labelsize=f_px * pt, labelcolor=qlib.INK)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family("DejaVu Sans")
    for side in ("left", "top", "bottom"):
        ax.spines[side].set_color(qlib.RULE)
        ax.spines[side].set_linewidth(pt * lw / 2)
    ax.spines["right"].set_visible(False)
    label_y = 1 - (y_axes - 0.47 * f_px) / height                        # baseline row of the frame numbers
    fig.text((lay.x0 - 0.5 * f_px) / width, label_y, "frame", va="bottom", ha="right", **font)
    measure = ImageFont.truetype(qlib.FONT_PATHS[0], f_px)
    fig.text(lay.pad / width, 1 - (y_axes + plot_h / 2) / height,
             wrap("Size of one step (1 = distance between two unrelated training frames)", measure, lay.x0 - lay.pad - 3.6 * f_px),
             va="center", ha="left", linespacing=1.32, **font)

    # ---- the z path in the principal plane of the training latents
    px = fig.add_axes(rect(inset_left, y_axes, w, plot_h))
    xy = d.path2d
    px.plot(xy[:, 0], xy[:, 1], color=qlib.ORANGE, lw=0.5 * lw * pt, zorder=2, solid_joinstyle="round")
    px.scatter(xy[:, 0], xy[:, 1], s=(1.3 * lw * pt) ** 2, color=qlib.ORANGE, zorder=3, linewidths=0)
    for k in d.steps:
        px.plot(xy[k:k + 2, 0], xy[k:k + 2, 1], color=qlib.DARK, lw=1.5 * lw * pt, zorder=4, solid_capstyle="round")
    px.scatter(xy[d.frames, 0], xy[d.frames, 1], s=(2.6 * lw * pt) ** 2, facecolor="white", edgecolor=qlib.INK,
               linewidths=0.6 * lw * pt, zorder=5)
    span = 0.5 * 1.5 * float(np.ptp(xy, axis=0).max())
    centre = 0.5 * (xy.min(0) + xy.max(0))
    px.set_xlim(centre[0] - span, centre[0] + span)
    px.set_ylim(centre[1] - span, centre[1] + span)
    px.set_xticks([])
    px.set_yticks([])
    for spine in px.spines.values():
        spine.set_color(qlib.RULE)
        spine.set_linewidth(pt * lw / 2)
    # only the first frame is named (the last one usually lies inside a cluster of frames); the word runs towards
    # the middle of the box and sits on the outer side of the path, so it stays inside
    above = xy[0, 1] >= xy[:, 1].mean()
    px.annotate("start", xy[0], xytext=(0, 0.75 * f_px * pt * (1 if above else -1)), textcoords="offset points",
                ha="left" if xy[0, 0] < centre[0] else "right", va="bottom" if above else "top", zorder=6, **font)
    fig.text((inset_left + w / 2) / width, label_y, "path of z, 2-D view", va="bottom", ha="center", **font)

    # ---- connectors: marked frame on the time axis -> the panel below
    y0, y1 = y_axes + plot_h, height
    for j, f in enumerate(d.frames):
        xa = lay.x0 + (f - x_lo) / (x_hi - x_lo) * (axes_right - lay.x0)
        xs = [xa, xa, lay.centre(j), lay.centre(j)]
        ys = [y0, y0 + 0.2 * fan_h, y1 - 0.2 * fan_h, y1]
        fig.add_artist(Line2D([x / width for x in xs], [1 - y / height for y in ys], color=GUIDE, lw=0.5 * lw * pt,
                              solid_joinstyle="round", solid_capstyle="butt"))

    canvas.draw()
    box = legend.get_window_extent()
    if box.x0 < lay.pad + title_w + measure.getlength(subtitle + "  "):
        raise RuntimeError("the legend runs into the title; shorten one of them")
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    assert image.shape == (height, width, 3), (image.shape, height, width)
    return image


def _bold_font_path() -> str:
    path = Path(qlib.FONT_PATHS[0]).with_name("DejaVuSans-Bold.ttf")
    if path.exists():
        return str(path)
    import matplotlib
    return str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf")


def stack(images: list[Image.Image | np.ndarray]) -> Image.Image:
    """Images of one width, one under the other."""
    arrays = [np.asarray(i) for i in images]
    assert len({a.shape[1] for a in arrays}) == 1, [a.shape for a in arrays]
    return Image.fromarray(np.concatenate(arrays, axis=0))


# ------------------------------------------------------------------------------ meta
def numbers_for(d: Drawn) -> list[dict[str, Any]]:
    ds, name = d.dataset, LABEL[d.dataset]
    frames_file = f"reports/z_temporal_diagnostic/{ds}/cache/frames.npz"
    here = f"computed in this script from {frames_file} (C, z, z_mu, z_sd)"
    table = "reports/z_temporal_diagnostic/temporal_geometry_metrics.csv"
    row = f"row dataset={ds}, h={HORIZON}"
    seq = f"cache row {d.row} (seq == {d.ex.meta.example})"
    values = json.loads(VALUES.read_text())
    out = [
        dict(label="Spearman correlation between the 63 contact steps and the 63 latent steps of the drawn sequence",
             value=d.spearman, dataset=name, source=here, locator=seq + f"; also panels.npz {ds}_series/contact_step, latent_step"),
        dict(label="median of that per-sequence correlation over the test sequences", value=d.evidence["median"], dataset=name,
             source=here, locator=f"rows with split == test; panels.npz {ds}_selection/spearman"),
        dict(label="number of test sequences (candidates of the selection)", value=d.evidence["n_candidates"], dataset=name,
             source=frames_file, locator=f"split == test; equals n_sequences.test of {ds}/cache/frames.json"),
        dict(label="smallest and largest per-sequence correlation over the test sequences",
             value=[d.evidence["quantiles_over_test_sequences"]["min"], d.evidence["quantiles_over_test_sequences"]["max"]],
             dataset=name, source=here, locator=f"panels.npz {ds}_selection/spearman, first and last value"),
        dict(label="dataset-level Spearman over ALL test transitions pooled, one-frame horizon (what the page quotes; a different statistic)",
             value=float(d.cache.table.spearman_C_z), dataset=name, source=table, locator=row + ", column spearman_C_z"),
        dict(label="the same dataset-level Spearman, recomputed in this script from the cache", value=d.checks["pooled_spearman_recomputed_here"],
             dataset=name, source=here, locator="all (test sequence, t) with t <= 62"),
        dict(label="the report's mean of the dataset-level Spearman over its horizons of 1, 4 and 8 frames",
             value=float(values[f"mean_spearman_{ds}"]), dataset=name, source="reports/z_temporal_diagnostic/report_values.json",
             locator=f"key mean_spearman_{ds}"),
        dict(label="number of pooled test transitions behind the dataset-level value", value=int(d.cache.table.n_test), dataset=name,
             source=table, locator=row + ", column n_test"),
        dict(label="number of transitions in the drawn sequence", value=int(len(d.contact_step)), dataset=name, source=here, locator=seq),
        dict(label="unit of the contact step: mean distance between the maps of two random training frames", value=d.cache.scale_contact,
             dataset=name, source=table, locator=row + ", column scale_C"),
        dict(label="unit of the latent step: mean distance between the standardised latents of two random training frames",
             value=d.cache.scale_latent, dataset=name, source=table, locator=row + ", column scale_z"),
    ]
    for rank, k in enumerate(sorted(d.steps, key=lambda s: -d.contact_step[s]), 1):
        out.append(dict(label=f"selected contact step {rank} (largest first): from frame {k} to frame {k + 1}; [contact step, latent step at the same transition]",
                        value=[float(d.contact_step[k]), float(d.latent_step[k])], dataset=name, source=here,
                        locator=f"panels.npz {ds}_series/contact_step[{k}], latent_step[{k}]"))
    k_lat = int(np.argmax(d.latent_step))
    out += [
        dict(label=f"largest latent step of the drawn sequence (from frame {k_lat} to frame {k_lat + 1})", value=float(d.latent_step[k_lat]),
             dataset=name, source=here, locator=f"panels.npz {ds}_series/latent_step[{k_lat}]"),
        dict(label="mean contact step, mean latent step and their ratio in the drawn sequence",
             value=[float(d.contact_step.mean()), float(d.latent_step.mean()), float(d.latent_step.mean() / d.contact_step.mean())],
             dataset=name, source=here, locator=seq),
        dict(label="share of the training latents' variance kept by the 2-D view (first two principal components)", value=d.kept_train,
             dataset=name, source=here, locator="rows with split == train, all 64 frames, standardised z"),
        dict(label="share of the drawn sequence's own latent variance kept by the 2-D view", value=d.kept_sequence, dataset=name,
             source=here, locator=seq),
        dict(label="Spearman between the contact steps and the steps of the 2-D projected path (drawn sequence)", value=d.spearman_2d,
             dataset=name, source=here, locator=f"panels.npz {ds}_zpath/xy"),
        dict(label="frames drawn", value=list(d.frames), dataset=name, source=here, locator="rule: CONTEXT frames before, just before, just after each selected step"),
        dict(label="largest change of a drawn surface value between consecutive pictures (five pairs, in the order drawn)",
             value=[round(float(np.nanmax(np.abs(d.ex.to_vertices(d.contact[b]) - d.ex.to_vertices(d.contact[a])))), 3)
                    for a, b in zip(d.frames[:-1], d.frames[1:])],
             dataset=name, source=here, locator=f"panels.npz {ds}_frameNN/values of consecutive drawn frames"),
        dict(label="closest distance between the hand and the object at each drawn frame, cm",
             value=[round(float(-qlib.SOFT_SCALE * np.log(d.ex.dense(t).max()) * 100), 2) for t in d.frames], dataset=name,
             source="dense hand-to-vertex distances of the take (taco/30_bimart_gen3_scene_scale or arctic/20_bimart_contact_probe, sequences/)",
             locator=f"sequence {d.ex.meta.sequence_id}, hand {d.ex.meta.hand}, take frames {[d.ex.take_frame(t) for t in d.frames]}"),
    ]
    return out


# ------------------------------------------------------------------------------ main
def main() -> None:
    qlib.require_headless()
    drawn: dict[str, Drawn] = {}
    selection: dict[str, dict[str, np.ndarray]] = {}
    for ds in LABEL:
        drawn[ds], selection[ds] = prepare(ds)
        d = drawn[ds]
        print(f"{LABEL[ds]}: example {d.ex.meta.example} ({d.ex.meta.sequence_id}, {HAND_WORD[d.ex.meta.hand]} hand on {d.ex.meta.category}); "
              f"per-sequence Spearman {d.spearman:.3f} = median of {d.evidence['n_candidates']} test sequences ({d.evidence['median']:.3f}); "
              f"the page quotes {PAGE_QUOTE[ds]:.2f}, the Spearman over all {int(d.cache.table.n_test)} test transitions pooled "
              f"({float(d.cache.table.spearman_C_z):.3f} in the table, {d.checks['pooled_spearman_recomputed_here']:.3f} recomputed): a different statistic. "
              f"Steps {d.steps}, frames {d.frames}.")

    values = {ds: [d.ex.to_vertices(d.contact[t]) for t in d.frames] for ds, d in drawn.items()}
    vmax = float(np.ceil(max(np.nanmax(v) for vs in values.values() for v in vs) * 10) / 10)   # one scale for the figure
    not_reached = any((~d.ex.covered).any() for d in drawn.values())
    bar = qlib.colorbar("contact", 0.0, vmax, "recorded contact drawn on the surface (0 = none, 1 = the hand touches)",
                        nodata="surface the contact map does not reach" if not_reached else None)

    parts: list[Image.Image | np.ndarray] = []
    cameras: dict[str, qlib.Camera] = {}
    panels: dict[str, Any] = {}
    layout: Layout | None = None
    for i, (ds, d) in enumerate(drawn.items()):
        cameras[ds] = camera_for(d, values[ds])
        row = render_panels(d, vmax, values[ds], cameras[ds])
        grid = qlib.grid([row], row_labels=[f"Recorded contact on the object ({d.ex.meta.category}) at the six marked frames; hand drawn as a ghost"],
                         col_labels=[f"frame {t}" for t in d.frames], colorbars=[bar] if i == len(drawn) - 1 else ())
        layout = locate(grid, row[0], len(row))
        parts += [chart_band(layout, d), grid]
        panels[f"{ds}_series"] = {"contact_step": d.contact_step, "latent_step": d.latent_step, "frames": np.array(d.frames),
                                  "selected_steps": np.array(d.steps)}
        panels[f"{ds}_zpath"] = {"xy": d.path2d}
        panels[f"{ds}_selection"] = selection[ds]
        panels[f"{ds}_mesh"] = {"faces": d.ex.mesh(0)[1], "hand_faces": d.ex.hand(0)[1], "covered": d.ex.covered}
        for t, val, image in zip(d.frames, values[ds], row):
            panels[f"{ds}_frame{t:02d}"] = {"values": val, "canonical": d.contact[t], "verts": d.ex.mesh(t)[0],
                                            "hand_verts": d.ex.hand(t)[0], "image": image}

    figure = stack(parts)
    assert layout is not None
    cap = ImageFont.truetype(qlib.FONT_PATHS[0], layout.font_px).getbbox("H")
    figure.info["qlib_layout"] = {"width": figure.width, "height": figure.height, "panel": [PANEL, PANEL], "font_px": layout.font_px,
                                  "cap_px": cap[3] - cap[1], "cap_px_at_1100": round((cap[3] - cap[1]) * 1100.0 / figure.width, 2),
                                  "composition": "per dataset: chart band (matplotlib, same font size) above a qlib.grid row"}

    t, a = drawn["taco"], drawn["arctic"]
    caption = (
        f"One test sequence per dataset, selected as the median sequence of its test set by the rank correlation between its two "
        f"step series: a {HAND_WORD[t.ex.meta.hand]} hand on a {t.ex.meta.category} (TACO, {t.spearman:.2f}) and a "
        f"{HAND_WORD[a.ex.meta.hand]} hand on {a.ex.meta.category} (ARCTIC, {a.spearman:.2f}). "
        f"Chart: for each of the {len(t.contact_step)} frame-to-frame transitions, how far the recorded contact map moves (blue) and how far "
        "the latent z moves (orange), both in units of the distance between two unrelated training frames. "
        f"Pictures: the recorded contact on the object, with the hand as a ghost, {CONTEXT} frames before, just before and just after "
        f"the largest contact step of the sequence and the largest one at least {MIN_GAP} frames away from it (shaded in the chart); "
        "a single step is a small change of the picture. "
        "In both sequences the orange line peaks where the blue line peaks and is low where the contact hardly changes. "
        "The square shows the same z path projected on the two directions along which the training latents vary most, which keep "
        f"about a third of their variance ({t.kept_train:.0%} on TACO, {a.kept_train:.0%} on ARCTIC), so distances in it are only indicative. "
        f"The {PAGE_QUOTE['taco']:.2f} and {PAGE_QUOTE['arctic']:.2f} quoted in the text are a different statistic: the correlation over all test "
        "transitions of a dataset taken together.")
    meta = {
        "id": FIG_ID, "block": BLOCK, "title": TITLE,
        "panels": {
            "layout": "two stacked blocks, TACO above ARCTIC; each block is a chart band above one row of six rendered panels",
            "chart": "x: frame of the 64-frame sequence (axis on top); a step from frame k to k + 1 is drawn at k + 0.5. Blue: size of the "
                     "contact step. Orange: size of the latent step. One shared y axis, both in units of the mean distance between two "
                     "random training frames. Shaded: the two selected contact steps. Thin grey lines: the six drawn frames, each "
                     "connected to its picture below.",
            "square": "the 64 latents of the sequence projected on the first two principal components of the standardised TRAINING "
                      "latents; dots are frames, rings the six drawn frames, dark segments the two selected steps, 'start' names "
                      "frame 0; no axes (arbitrary orientation, equal scale on both directions)",
            "pictures": "columns: the six frames in time order (three per selected step: 3 frames before, just before, just after). "
                        "The object mesh is coloured by the recorded canonical contact map of that frame; the recorded hand is a ghost.",
        },
        "examples": [{"dataset": LABEL[ds], "example": d.ex.meta.example, "sequence_id": d.ex.meta.sequence_id, "object": d.ex.meta.category,
                      "mesh_id": d.ex.meta.mesh_id, "role": d.ex.meta.role, "hand": HAND_WORD[d.ex.meta.hand], "split": d.ex.meta.split,
                      "cache_row": d.row, "first_take_frame": d.ex.meta.t0, "frames": d.frames, "selected_steps": d.steps}
                     for ds, d in drawn.items()],
        "selection_rule": "Per dataset, among all test sequences: compute the Spearman correlation between the sequence's 63 contact steps "
                          "and its 63 latent steps, and draw the median sequence. Both test sets have an even number of sequences, so no "
                          "sequence equals the median; the two middle sequences are equally far from it and the one with the smaller "
                          "example index is drawn. Frames: the largest contact step and the largest one at least "
                          f"{MIN_GAP} transitions away from it; for each, the frame {CONTEXT} before, the frame just before and the frame just after.",
        "selection_evidence": {LABEL[ds]: d.evidence for ds, d in drawn.items()},
        "page_quote": {LABEL[ds]: {"quoted_on_page": PAGE_QUOTE[ds], "table_value_h1": float(d.cache.table.spearman_C_z),
                                   "per_sequence_value_of_the_drawn_sequence": d.spearman,
                                   "note": "the page's value is the Spearman over all test transitions of the dataset pooled; the value of "
                                           "the drawn sequence is computed within one sequence. They are different statistics."}
                       for ds, d in drawn.items()},
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (Example.to_vertices); the drawn map is the recorded 512-point map of the frame",
            "colour_scales": {"contact": {"range": [0.0, vmax], "stops": "grey #dcdbd5 at 0, orange #eb6834 at half, dark #7a2408 at the top",
                                          "rule": "largest drawn per-vertex value over both datasets, rounded up to one decimal",
                                          "largest_drawn_value": {LABEL[ds]: float(max(np.nanmax(v) for v in values[ds])) for ds in drawn}},
                              "not_reached": qlib.NO_DATA},
            "chart": {"contact_step": qlib.BLUE, "latent_step": qlib.ORANGE, "selected_steps_band": BAND, "frame_markers": GUIDE,
                      "y_axis": "shared, starts at 0; no second axis"},
            "camera": {LABEL[ds]: {"rule": VIEW[ds], **asdict(cameras[ds]), "fitted_on": "object and hand at all six frames"} for ds in drawn},
            "hand": "recorded MANO hand of the contacting side, ghost (colour #cfcec8, opacity 0.35, front faces only)",
            "frame_rule_parameters": {"min_gap_between_steps": MIN_GAP, "context_frames": CONTEXT},
        },
        "numbers": [n for d in drawn.values() for n in numbers_for(d)] + [
            dict(label="top of the contact colour scale", value=vmax, dataset="both", source="computed in this script",
                 locator="largest drawn per-vertex value, rounded up to one decimal"),
            dict(label="frames between the earlier picture and the picture just before a selected step", value=CONTEXT, dataset="both",
                 source="constant CONTEXT of this script (display rule)", locator="frames_around"),
            dict(label="smallest distance, in transitions, between the two selected contact steps", value=MIN_GAP, dataset="both",
                 source="constant MIN_GAP of this script (display rule)", locator="largest_steps")],
        "cross_checks": {LABEL[ds]: d.checks for ds, d in drawn.items()},
        "inference": "none: data only. z is the cached encoding of the frozen Stage-1 encoder, read from "
                     "reports/z_temporal_diagnostic/<dataset>/cache/frames.npz (the encoder checkpoint is named in cache/frames.json and "
                     "was not loaded here). Newly computed: the per-sequence correlations, the two step series of the drawn sequences "
                     "(equal to the study's saved transition table), the principal components of the training latents. Device: CPU.",
        "caveats": [
            "One sequence per dataset. It is the median sequence by the per-sequence correlation, so it is typical in that one number; "
            "it says nothing about the spread (see quantiles_over_test_sequences).",
            "The per-sequence correlation is not the number on the page: the page quotes the Spearman over all test transitions of a "
            "dataset pooled, which also contains differences between sequences.",
            "z is computed by the encoder from the true frames, so the figure shows how the representation of recorded contact moves, "
            "not a prediction.",
            f"The figure shows that the two steps rise and fall together, not that they are equal: in these units the mean latent step "
            f"is {t.latent_step.mean() / t.contact_step.mean():.1f} times the mean contact step on the TACO sequence and "
            f"{a.latent_step.mean() / a.contact_step.mean():.1f} times on the ARCTIC sequence.",
            "A single frame-to-frame step is a small change of the picture, also at the selected steps: the pictures just before and "
            "just after a step differ only locally (see the numbers 'largest change of a drawn surface value').",
            f"The 2-D view keeps {t.kept_train:.0%} (TACO) and {a.kept_train:.0%} (ARCTIC) of the variance of the training latents, and "
            f"{t.kept_sequence:.0%} and {a.kept_sequence:.0%} of the drawn sequences' own latent variance; a step that is long in the 64 "
            f"dimensions can look short there. The rank correlation between the contact steps and the steps of the projected path is "
            f"{t.spearman_2d:.2f} (TACO) and {a.spearman_2d:.2f} (ARCTIC), lower than in 64 dimensions.",
            "The two rows use different camera rules (bowl: the direction the contact faces; scissors: the broad side, from the side "
            "away from the hand); within a row the camera is fixed.",
            "One colour scale for both rows, set by the largest drawn value (TACO); the ARCTIC contact stays in its lower two thirds.",
            "The pictures draw the 512-point canonical map through the display rule, which smooths: on thin parts the contact shows on "
            "both faces. ARCTIC: the map is defined on the closed object, so contact near the joint is drawn on both blades.",
        ],
        "suggested_caption": caption,
        "suggested_alt": "For one TACO and one ARCTIC test sequence, a chart in which the size of the latent step follows the size of the "
                         "contact step over 63 transitions, above six renders of the object coloured by the recorded contact around the "
                         "two largest contact steps.",
        "helpers_outside_qlib": "chart_band (matplotlib chart at the grid's font size), locate / stack (place the chart above a qlib.grid row), "
                                "camera_for (two geometric view rules)",
        "command": qlib.RUN_COMMAND.replace("<script>", "docs/build/qual/q_z_moves_with_contact.py"),
    }
    paths = qlib.save(FIG_ID, figure, panels, meta)
    for key, path in paths.items():
        print(f"{key}: {path}")
    print("cross-checks:", json.dumps({LABEL[ds]: d.checks for ds, d in drawn.items()}, indent=1))


if __name__ == "__main__":
    main()
