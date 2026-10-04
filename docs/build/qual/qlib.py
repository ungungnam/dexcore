"""Shared helpers for the qualitative figures of the project page: real 3-D renders of an object
mesh, the hand that touches it and the contact on the object's surface.

Run every script that imports this module on CPU with the software renderer, exactly as

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/<script>.py

``require_headless()`` refuses anything else (without the VTK variable the renderer may open a
context on a GPU).

Data. ``load_example(dataset, n)`` returns one sequence of the hierarchical contact study
(``taco/50_hier_contact_gen`` or ``arctic/40_hier_contact_gen``): ONE hand on ONE object over 64
frames. All geometry is in the contacted object's own rigid frame, in metres (TACO: the mesh frame
of the tool or the target; ARCTIC: the frame in which the bottom part is fixed and the top part
turns by the articulation angle). Sequence frame ``t`` is take frame ``t0 + t``, clamped to the take
like every cache of the studies.

Display rule (one rule for every map on the page). A contact map lives on 512 canonical points. The
study made it from the recorded per-vertex soft contact ``s = exp(-d / 2 cm)`` with its operator ``W``
(512 x V, rows sum to 1, Gaussian of sigma 0.075 within 0.15 canonical units): ``C = W s``. A map is
drawn on the mesh with the same operator transposed and normalised:

    value(vertex u) = sum_k W[k, u] C[k] / sum_k W[k, u]

This smooths (it is not an inverse of ``W``), so a recorded map and a model output must both go
through it before they are compared. Two properties of ``W`` show in the pictures:

* A vertex that no canonical point reaches has no value: the map cannot represent contact there.
  ``to_vertices`` returns NaN for it and the colour functions draw it in ``NO_DATA``. ARCTIC meshes
  are reached completely; 39 of the 79 TACO instance meshes are not (``Example.covered``), because the
  canonical points come from the category's template mesh.
* ARCTIC: ``W`` is built on the closed rest pose, so a canonical point near the closing surfaces
  averages vertices of both parts. Contact on one part is then drawn faintly on the other part too,
  also in frames where the object is open and that part is far from the hand.

Colour scales. ``contact_rgb``: neutral grey (0) through orange to dark red-brown (``vmax``).
``diverging_rgb``: blue (less), neutral grey (no change), orange (more). Colours are piecewise
linear in sRGB between the stops below; the renderer then shades them, so a colour bar shows the
unshaded scale.

Output. ``save(fig_id, ...)`` writes ``<result root>/reports/project_page_qualitative/<fig_id>/``
(``figure.png``, ``panels.npz``, ``meta.json``) and the page asset ``docs/assets/img/qual_<fig_id>.webp``.
Nothing else is written; every input is opened read-only, and importing the module does nothing but
define names. The TACO hand is computed with the repository's own loader and MANO layer
(``src.analysis``), imported on first use.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, features
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------------ paths
REPO = Path(__file__).resolve().parents[3]
RESULT_ROOT = Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))
OUT_ROOT = RESULT_ROOT / "reports/project_page_qualitative"
ASSET_DIR = REPO / "docs/assets/img"
HIER = {"taco": RESULT_ROOT / "taco/50_hier_contact_gen", "arctic": RESULT_ROOT / "arctic/40_hier_contact_gen"}
TACO_GEN3 = RESULT_ROOT / "taco/30_bimart_gen3_scene_scale"          # mesh dict, dense hand-to-vertex distances
TACO_WINDOW_INDEX = RESULT_ROOT / "taco/40_representation_study/window_index.csv"
ARCTIC_PROBE = RESULT_ROOT / "arctic/20_bimart_contact_probe"        # mesh dict, dense distances
ARCTIC_PROC = REPO / "third_party/BimArt/data/arctic_processed_data"  # hand vertices, object pose per frame
HAND_CACHE = {ds: RESULT_ROOT / f"reports/hand_contact_predictive_info/{ds}/cache/hand_cache.npz" for ds in HIER}
MANO_PARTS = RESULT_ROOT / "reports/wrench_counterfactual/cache/mano_vertex_parts.npz"
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",)
RUN_COMMAND = ('CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow '
               "PYVISTA_OFF_SCREEN=true python <script>")

# ------------------------------------------------------------------------------ constants of the studies
T = 64                         # frames per sequence
PAD = 8                        # the caches hold frames t in [-PAD, T + PAD)
RADIUS, SIGMA = 0.15, 0.075    # canonical operator W (canonical units: object radius = 1)
SOFT_SCALE = 0.02              # soft contact = exp(-distance / 2 cm)
SIDE = {"L": "left", "R": "right"}
DISPLAY_RULE = "value(u) = sum_k W[k,u] C[k] / sum_k W[k,u]; W = the study's canonical operator; NaN where sum_k W[k,u] = 0"

# ------------------------------------------------------------------------------ colours
INK = "#1f1e1b"
RULE = "#c3c2b7"               # outlines of colour bars and swatches
ZERO = "#dcdbd5"               # no contact / no change
ORANGE, BLUE, DARK = "#eb6834", "#2a78d6", "#7a2408"
NO_DATA = "#e7e1f1"            # vertex the 512 canonical points do not reach (a hue of its own: shading alone would hide it)
HAND_SKIN = "#e8d2c0"
GHOST = {"color": "#cfcec8", "opacity": 0.35}      # mesh_item(verts, faces, **GHOST)
PART_NAMES = ("palm", "thumb", "index", "middle", "ring", "little")     # label order of the studies
PART_HEX = ("#6b6a63", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd0")

# ------------------------------------------------------------------------------ layout
MIN_CAP_PX_AT_1100 = 11.0      # capital letters are at least this tall when the figure is shown 1100 px wide
ROW_LABEL_FRACTION = 0.15      # share of the figure width given to the row labels
ASSET_MAX_WIDTH = 2200
ASSET_QUALITY = 90
VIEW_ANGLE = 25.0              # vertical field of view, degrees
CREASE_ANGLE = 40.0            # degrees; sharper edges of an object mesh are not smoothed over
DISC_THICKNESS = 0.002         # metres


def _rgb(color: str | Sequence[float]) -> np.ndarray:
    """'#rrggbb' or an RGB triple in [0, 1] -> float (3,)."""
    if isinstance(color, str):
        return np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0
    return np.asarray(color, float).reshape(3)


PART_COLORS = np.stack([_rgb(c) for c in PART_HEX])      # (6, 3), PART_COLORS[label] colours MANO part labels


def _ramp(x: np.ndarray, stops: Sequence[tuple[float, str]]) -> np.ndarray:
    """Piecewise-linear colour scale over x in [0, 1]; NaN -> NO_DATA."""
    x = np.asarray(x, float)
    pos = [p for p, _ in stops]
    cols = np.stack([_rgb(c) for _, c in stops])
    out = np.stack([np.interp(np.clip(np.nan_to_num(x), 0, 1), pos, cols[:, j]) for j in range(3)], -1)
    out[np.isnan(x)] = _rgb(NO_DATA)
    return out


def contact_rgb(values: np.ndarray, vmax: float) -> np.ndarray:
    """Contact values -> RGB in [0, 1], shape values.shape + (3,). 0 grey, vmax / 2 orange, vmax dark red-brown."""
    return _ramp(np.asarray(values, float) / vmax, [(0.0, ZERO), (0.5, ORANGE), (1.0, DARK)])


def diverging_rgb(values: np.ndarray, vabs: float) -> np.ndarray:
    """Signed differences -> RGB in [0, 1]. -vabs blue (less), 0 grey, +vabs orange (more)."""
    return _ramp(0.5 + 0.5 * np.asarray(values, float) / vabs, [(0.0, BLUE), (0.5, ZERO), (1.0, ORANGE)])


# ------------------------------------------------------------------------------ environment
def require_headless() -> None:
    """Raise unless this process renders with OSMesa, sees no CUDA device and writes no bytecode."""
    env, bad = os.environ, []
    if env.get("VTK_DEFAULT_OPENGL_WINDOW") != "vtkOSOpenGLRenderWindow":
        bad.append("VTK_DEFAULT_OPENGL_WINDOW is not vtkOSOpenGLRenderWindow")
    if env.get("PYVISTA_OFF_SCREEN", "").lower() != "true":
        bad.append("PYVISTA_OFF_SCREEN is not true")
    if env.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        bad.append("CUDA_VISIBLE_DEVICES is not empty")
    elif "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        bad.append("torch sees a CUDA device")
    if env.get("PYTHONDONTWRITEBYTECODE", "") == "":
        bad.append("PYTHONDONTWRITEBYTECODE is not set")
    if bad:
        raise RuntimeError("qualitative figures run headless on CPU only (" + "; ".join(bad) + "). Run as: " + RUN_COMMAND)


# ------------------------------------------------------------------------------ data tables (read-only, cached)
@lru_cache(maxsize=None)
def _tables(dataset: str) -> dict[str, Any]:
    root = HIER[dataset]
    with np.load(root / "sequences.npz") as z:
        tb = {k: z[k] for k in ("C", "G_radius", "canonical_cats", "canonical_points")}
    tb["meta"] = pd.read_csv(root / "sequences_meta.csv", dtype={"mesh_id": str, "subject": str, "sequence_id": str})
    return tb


@lru_cache(maxsize=None)
def _mesh_dict(dataset: str) -> dict:
    path = TACO_GEN3 / "assets/taco_mesh_dict.npy" if dataset == "taco" else ARCTIC_PROBE / "assets/arctic_mesh_dict.npy"
    return np.load(path, allow_pickle=True).item()


@lru_cache(maxsize=None)
def _dense_index(dataset: str) -> pd.DataFrame:
    """sequence_id -> file of the dense distances (and, TACO, the number of tool vertices)."""
    idx = pd.read_csv((TACO_GEN3 if dataset == "taco" else ARCTIC_PROBE) / "sequence_index.csv").set_index("sequence_id")[["file"]]
    if dataset == "taco":
        idx = idx.join(pd.read_csv(TACO_WINDOW_INDEX).drop_duplicates("sequence_id").set_index("sequence_id")["n_tool"])
    return idx


@lru_cache(maxsize=None)
def _hand_cache(dataset: str) -> np.ndarray:
    with np.load(HAND_CACHE[dataset]) as z:
        assert np.array_equal(z["example"], np.arange(len(z["example"])))
        return z["hand"]                                   # (N, T + 2 PAD, 100, 3), object frame


@lru_cache(maxsize=None)
def _mano() -> dict[str, np.ndarray]:
    with np.load(MANO_PARTS) as z:
        return {"part": z["part"].astype(np.int64), "faces": z["faces"].astype(np.int64)}


def mano_parts() -> np.ndarray:
    """(778,) part label per MANO vertex (index into PART_NAMES), the same for both hands."""
    return _mano()["part"]


def mano_faces(side: str) -> np.ndarray:
    """(1538, 3) MANO triangles; the left hand is the right hand's list with the winding reversed."""
    f = _mano()["faces"]
    return f if side == "right" else f[:, ::-1].copy()


def gauss_operator(v_norm: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The studies' canonical operator W (512 x V): per canonical point a normalised Gaussian over the
    mesh vertices within RADIUS (same code as structure_aware_temporal_generation/build_masks.py)."""
    tree = cKDTree(v_norm)
    w_op = np.zeros((len(points), len(v_norm)), np.float32)
    for k, nb in enumerate(tree.query_ball_point(points, RADIUS)):
        if nb:
            nb = np.asarray(nb)
            w = np.exp(-0.5 * ((v_norm[nb] - points[k]) ** 2).sum(1) / SIGMA ** 2)
            w_op[k, nb] = w / w.sum()
    return w_op


# ------------------------------------------------------------------------------ one example
@dataclass(frozen=True)
class Meta:
    dataset: str
    example: int
    sequence_id: str
    category: str
    hand: str            # "L" or "R": the hand whose contact this example records
    role: str            # TACO: "tool" or "target"; ARCTIC: "obj"
    mesh_id: str
    t0: int              # take frame of sequence frame 0
    split: str
    group: str


class Example:
    """One sequence of the hierarchical study; see the module docstring for frames and units."""

    def __init__(self, dataset: str, n: int) -> None:
        tb = _tables(dataset)
        row = tb["meta"].iloc[n]
        self.meta = Meta(dataset, int(n), row.sequence_id, row.category, row.hand, row.role, row.mesh_id, int(row.t0),
                         row.split_set, row.group)
        self.C: np.ndarray = tb["C"][n].astype(np.float32)                   # (64, 512) recorded canonical map
        md = _mesh_dict(dataset)
        if dataset == "taco":
            parts = [md[row.mesh_id]]
        else:
            # vertex order of the cached contact; the template's few triangles that join the two parts are left
            # out of these face lists (they would stretch when the object opens)
            parts = [md[f"{row.category}_top"], md[f"{row.category}_bottom"]]
        offsets = np.cumsum([0] + [len(p["verts_original"]) for p in parts])
        self._rest = np.concatenate([np.asarray(p["verts_original"], float) for p in parts])
        self._faces = np.concatenate([np.asarray(p["faces"], np.int64) + o for p, o in zip(parts, offsets)])
        self._n_top = int(offsets[1]) if dataset == "arctic" else 0          # articulated vertices come first
        centre = self._rest.mean(0)
        radius = np.linalg.norm(self._rest - centre, axis=1).max()
        assert abs(radius - float(tb["G_radius"][row.g_index])) < 1e-6, "mesh does not match the study's geometry"
        points = tb["canonical_points"][list(tb["canonical_cats"]).index(row.category)].astype(np.float64)
        self.W: np.ndarray = gauss_operator((self._rest - centre) / radius, points)   # (512, V)
        reach = self.W.sum(0)
        self.covered: np.ndarray = reach > 0                                 # (V,) vertices the map can represent
        self._back = self.W / np.where(self.covered, reach, 1.0)

    # ---- geometry
    def take_frame(self, t: int) -> int:
        """Take frame of sequence frame t (clamped to the recorded take, as in the studies' caches)."""
        return int(np.clip(self.meta.t0 + t, 0, len(self._dist) - 1))

    def articulation(self, t: int) -> float:
        """ARCTIC: opening angle of the top part at frame t, radians (0 for TACO)."""
        return float(self._take["state"][self.take_frame(t), 0]) if self.meta.dataset == "arctic" else 0.0

    def mesh(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(verts (V, 3), faces (F, 3)) at frame t; ARCTIC: the top part turned by rotvec (0, 0, -articulation)."""
        verts = self._rest.copy()
        if self._n_top:
            rot = Rotation.from_rotvec((0.0, 0.0, -self.articulation(t))).as_matrix()
            verts[:self._n_top] = verts[:self._n_top] @ rot.T
        return verts, self._faces

    def hand(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(verts (778, 3), faces (1538, 3)) of the contacting MANO hand at frame t."""
        return self._take["hand"][self.take_frame(t)], mano_faces(SIDE[self.meta.hand])

    def hand_points(self, t: int) -> np.ndarray:
        """(100, 3) cached surface points of the contacting hand, t in [-8, 72)."""
        if not -PAD <= t < T + PAD:
            raise IndexError(f"hand points are cached for t in [{-PAD}, {T + PAD}), got {t}")
        return _hand_cache(self.meta.dataset)[self.meta.example, t + PAD].astype(np.float64)

    def dense(self, t: int) -> np.ndarray:
        """(V,) recorded per-vertex contact exp(-distance to the hand / 2 cm) at frame t."""
        return np.exp(-self._dist[self.take_frame(t)].astype(np.float64) / SOFT_SCALE)

    def to_vertices(self, c512: np.ndarray) -> np.ndarray:
        """Any canonical map (..., 512) -> per-vertex values (..., V) by DISPLAY_RULE (NaN where not covered)."""
        c512 = np.asarray(c512, np.float32)
        if c512.shape[-1] != self.W.shape[0]:
            raise ValueError(f"expected (..., {self.W.shape[0]}) values, got {c512.shape}")
        out = (c512 @ self._back).astype(np.float64)
        out[..., ~self.covered] = np.nan
        return out

    def test_row(self, example_array: np.ndarray) -> int:
        """Row of this example in a saved prediction file, given the file's ``example`` array."""
        rows = np.flatnonzero(np.asarray(example_array) == self.meta.example)
        if len(rows) != 1:
            raise KeyError(f"example {self.meta.example} occurs {len(rows)} times in the prediction file")
        return int(rows[0])

    # ---- per-take data, loaded on first use
    @cached_property
    def _dist(self) -> np.ndarray:
        """(take frames, V) recorded distance from the contacting hand to every mesh vertex, metres."""
        m, ds = self.meta, self.meta.dataset
        idx = _dense_index(ds).loc[m.sequence_id]
        with np.load((TACO_GEN3 if ds == "taco" else ARCTIC_PROBE) / "sequences" / idx.file) as z:
            d = z[f"contact_{SIDE[m.hand]}"]
        if ds == "taco":                                   # columns: tool vertices, then target vertices
            d = d[:, :int(idx.n_tool)] if m.role == "tool" else d[:, int(idx.n_tool):]
        assert d.shape[1] == len(self._rest), (d.shape, self._rest.shape)
        return d

    @cached_property
    def _take(self) -> dict[str, np.ndarray]:
        """hand (take frames, 778, 3) in the object frame; ARCTIC also state (take frames, 7)."""
        m, side = self.meta, SIDE[self.meta.hand]
        if m.dataset == "arctic":
            stem, subject = m.sequence_id.split("/")
            base = ARCTIC_PROC / m.category / subject
            state = np.load(base / f"{stem}_processed_obj_features.npy", allow_pickle=True).item()["obj_world_state"].astype(np.float64)
            world = np.load(base / f"{stem}_processed_hand_features.npy", allow_pickle=True).item()[f"{side}_hand_verts"]
            rot, pos = Rotation.from_rotvec(state[:, 1:4]).as_matrix(), state[:, 4:7]   # state = [articulation, rotvec, translation]
            assert len(world) == len(state) == len(self._dist)
            return {"hand": np.einsum("tvj,tjk->tvk", world - pos[:, None], rot), "state": state}
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from src import geometry
        from src.analysis.action_structure import contact
        from src.analysis.loaders import taco
        triplet, sequence = m.sequence_id.rsplit("/", 1)
        traj = taco.load(taco.SequenceRef(triplet, sequence, taco.parse_triplet(triplet)), with_hands=True)
        world, _ = contact.hand_geometry(traj.hands[side], contact.mano_layer(side))   # MANO on CPU
        track = traj.tool if m.role == "tool" else traj.target
        assert int(track.name) == int(m.mesh_id), (track.name, m.mesh_id)
        rot, pos = geometry.wxyz_to_R(track.quat), track.pos
        assert len(world) == len(self._dist)
        return {"hand": np.einsum("tvj,tjk->tvk", world - pos[:, None], rot)}          # R^T (x - p)


def load_example(dataset: str, n: int) -> Example:
    """Example ``n`` (row of sequences_meta.csv) of ``dataset`` in {"taco", "arctic"}."""
    require_headless()
    if dataset not in HIER:
        raise ValueError(f"dataset must be one of {sorted(HIER)}, got {dataset!r}")
    return Example(dataset, n)


# ------------------------------------------------------------------------------ rendering
@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]
    focal_point: tuple[float, float, float]
    up: tuple[float, float, float]
    view_angle: float     # vertical, degrees
    aspect: float         # width / height the camera was fitted for


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def fit_camera(points: Sequence[np.ndarray], direction: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0),
               margin: float = 0.08, aspect: float = 1.0, view_angle: float = VIEW_ANGLE) -> Camera:
    """The camera that looks at ``points`` (a list of (N, 3) arrays) from ``direction`` (scene -> camera) and
    just contains all of them, with ``margin`` of the half-frame left free. Fit it once on everything a row of
    panels will show and reuse it, so the panels are comparable."""
    pts = np.concatenate([np.asarray(p, float).reshape(-1, 3) for p in points])
    back = _unit(direction)
    up = np.asarray(up, float) - (np.asarray(up, float) @ back) * back
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("up is parallel to direction")
    up = _unit(up)
    right = np.cross(up, back)
    tan_y = np.tan(np.radians(view_angle) / 2)
    tan_x = tan_y * aspect
    focal = 0.5 * (pts.min(0) + pts.max(0))
    for step in range(7):
        q = pts - focal
        x, y, z = q @ right, q @ up, q @ back
        dist = float((z + (1 + margin) * np.maximum(np.abs(x) / tan_x, np.abs(y) / tan_y)).max())   # all points inside
        if step < 6:                                      # move the focal point to centre the projected bounding box
            px, py = x / (dist - z), y / (dist - z)
            focal = focal + 0.5 * dist * ((px.max() + px.min()) * right + (py.max() + py.min()) * up)
    return Camera(tuple(focal + dist * back), tuple(focal), tuple(up), float(view_angle), float(aspect))


def hand_side_view(object_verts: np.ndarray, hand_verts: np.ndarray, tilt: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """A (direction, up) pair for fit_camera, computed from the geometry alone: look onto the object's broad side
    (along its thinnest principal axis) from the side the hand is on, raised by ``tilt`` towards the middle axis,
    with the longest axis horizontal. A display choice; any other direction can be passed to fit_camera."""
    centre = object_verts.mean(0)
    _, axes = np.linalg.eigh(np.cov((object_verts - centre).T))       # columns: thinnest, middle, longest axis
    to_hand = hand_verts.mean(0) - centre
    thin, mid = (a if a @ to_hand >= 0 else -a for a in (axes[:, 0], axes[:, 1]))
    direction = _unit(thin + tilt * mid)
    return direction, _unit(mid - (mid @ direction) * direction)


def mesh_item(verts: np.ndarray, faces: np.ndarray, rgb: np.ndarray | None = None,
              color: str | Sequence[float] | None = None, opacity: float = 1.0, sharp_edges: bool = True) -> dict[str, Any]:
    """A triangle mesh with per-vertex colours ``rgb`` (V, 3) in [0, 1], or one ``color`` (default: ZERO).
    ``sharp_edges`` keeps creases above CREASE_ANGLE crisp (objects); a translucent mesh shows its front faces only."""
    if rgb is None:
        rgb = np.tile(_rgb(ZERO if color is None else color), (len(verts), 1))
    return {"kind": "mesh", "verts": np.asarray(verts, float), "faces": np.asarray(faces, np.int64),
            "rgb": np.asarray(rgb, float), "opacity": float(opacity), "sharp_edges": bool(sharp_edges)}


def hand_item(verts: np.ndarray, faces: np.ndarray, rgb: np.ndarray | None = None,
              color: str | Sequence[float] | None = HAND_SKIN, opacity: float = 1.0) -> dict[str, Any]:
    """The MANO hand: a mesh_item shaded smoothly everywhere (its coarse triangles are not creases), skin by default."""
    return mesh_item(verts, faces, rgb, color, opacity, sharp_edges=False)


def points_item(xyz: np.ndarray, color: str | Sequence[float], radius: float = 0.004) -> dict[str, Any]:
    """Spheres of ``radius`` metres."""
    return {"kind": "points", "xyz": np.asarray(xyz, float).reshape(-1, 3), "color": _rgb(color), "radius": float(radius)}


def arrows_item(origins: np.ndarray, directions: np.ndarray, color: str | Sequence[float], scale: float = 1.0) -> dict[str, Any]:
    """Arrows from ``origins`` along ``directions``, of length |direction| * scale metres."""
    return {"kind": "arrows", "origins": np.asarray(origins, float).reshape(-1, 3),
            "directions": np.asarray(directions, float).reshape(-1, 3), "color": _rgb(color), "scale": float(scale)}


def discs_item(centres: np.ndarray, normals: np.ndarray, radii: np.ndarray | float, color: str | Sequence[float]) -> dict[str, Any]:
    """Flat discs (DISC_THICKNESS thick) centred at ``centres``, facing ``normals``, of ``radii`` metres."""
    centres = np.asarray(centres, float).reshape(-1, 3)
    return {"kind": "discs", "centres": centres, "normals": np.asarray(normals, float).reshape(-1, 3),
            "radii": np.broadcast_to(np.asarray(radii, float), (len(centres),)), "color": _rgb(color)}


def _polydata(item: dict[str, Any]) -> Any:
    """Item -> pyvista PolyData with a per-point uint8 'rgb' array (None when the item is empty)."""
    import pyvista as pv
    kind = item["kind"]
    if kind == "mesh":
        faces = item["faces"]
        poly = pv.PolyData(item["verts"], np.hstack([np.full((len(faces), 1), 3), faces]))
        rgb = item["rgb"]
    else:
        if kind == "points":
            ball = pv.Sphere(radius=item["radius"], theta_resolution=16, phi_resolution=16)
            parts = [ball.translate(p, inplace=False) for p in item["xyz"]]
        elif kind == "arrows":
            lengths = np.linalg.norm(item["directions"], axis=1) * item["scale"]
            parts = [pv.Arrow(start=o, direction=d, scale=float(s), tip_length=0.3, tip_radius=0.11, shaft_radius=0.04,
                              tip_resolution=24, shaft_resolution=24)
                     for o, d, s in zip(item["origins"], item["directions"], lengths) if s > 0]
        elif kind == "discs":
            parts = [pv.Cylinder(center=c, direction=n, radius=float(r), height=DISC_THICKNESS, resolution=48).triangulate()
                     for c, n, r in zip(item["centres"], item["normals"], item["radii"]) if r > 0]
        else:
            raise ValueError(f"unknown item kind {kind!r}")
        if not parts:
            return None
        poly = parts[0]
        for part in parts[1:]:
            poly = poly.append_polydata(part)
        rgb = np.tile(item["color"], (poly.n_points, 1))
    poly.point_data.clear()
    poly.point_data["rgb"] = (np.clip(rgb, 0, 1) * 255).round().astype(np.uint8)
    return poly


def render(items: Sequence[dict[str, Any]], camera: Camera, size: tuple[int, int] = (480, 480)) -> np.ndarray:
    """Draw ``items`` from ``camera`` into a (height, width, 3) uint8 image on white; ``size`` = (width, height)."""
    require_headless()
    import pyvista as pv
    if abs(size[0] / size[1] - camera.aspect) > 1e-3:
        raise ValueError(f"camera was fitted for aspect {camera.aspect:g}, the panel has {size[0] / size[1]:g}")
    pl = pv.Plotter(off_screen=True, window_size=[int(size[0]), int(size[1])], lighting="three lights")
    try:
        if pl.render_window.GetClassName() != "vtkOSOpenGLRenderWindow":
            raise RuntimeError("VTK did not select the OSMesa window: " + pl.render_window.GetClassName())
        pl.set_background("white")
        for item in items:
            poly = _polydata(item)
            if poly is None:
                continue
            opacity = item.get("opacity", 1.0)
            pl.add_mesh(poly, scalars="rgb", rgb=True, smooth_shading=True, split_sharp_edges=item.get("sharp_edges", True),
                        feature_angle=CREASE_ANGLE, opacity=opacity, culling="back" if opacity < 1 else None,
                        ambient=0.35, diffuse=0.65, specular=0.06, specular_power=20, show_scalar_bar=False)
        if any(item.get("opacity", 1.0) < 1 for item in items):
            pl.enable_depth_peeling(8)
        pl.camera.position, pl.camera.focal_point, pl.camera.up = camera.position, camera.focal_point, camera.up
        pl.camera.view_angle = camera.view_angle
        pl.reset_camera_clipping_range()
        pl.enable_anti_aliasing("ssaa")
        image = pl.screenshot(None, return_img=True)
    finally:
        pl.close()
    return np.ascontiguousarray(image[..., :3])


# ------------------------------------------------------------------------------ layout
@dataclass(frozen=True)
class Colorbar:
    kind: str                 # "contact", "diverging" or "parts"
    lo: float
    hi: float
    label: str
    nodata: str | None        # text for a NO_DATA swatch, or None


def colorbar(kind: str, lo: float = 0.0, hi: float = 1.0, label: str = "", nodata: str | None = None) -> Colorbar:
    """A legend entry for grid(): "contact" (lo = 0, hi = vmax), "diverging" (lo = -vabs, hi = +vabs) or
    "parts" (the six hand-part colours; lo and hi are ignored)."""
    if kind not in ("contact", "diverging", "parts"):
        raise ValueError(f"unknown colour bar kind {kind!r}")
    return Colorbar(kind, float(lo), float(hi), label, nodata)


def _font(px: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, px)
    import matplotlib                                      # ships the same DejaVu Sans
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), px)


def _cap_height(font: ImageFont.FreeTypeFont) -> int:
    box = font.getbbox("H")
    return box[3] - box[1]


def _font_for_width(width: int) -> ImageFont.FreeTypeFont:
    """Smallest DejaVu Sans whose capitals are MIN_CAP_PX_AT_1100 tall once ``width`` is shown as 1100 px."""
    px = 8
    while _cap_height(_font(px)) < MIN_CAP_PX_AT_1100 * width / 1100.0:
        px += 1
    return _font(px)


def _wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).split("\n"):
        line = ""
        for word in paragraph.split():
            if font.getlength(word) > width:
                raise ValueError(f"label word {word!r} is wider than its {width:.0f} px slot; shorten the label")
            trial = f"{line} {word}".strip()
            if font.getlength(trial) <= width:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def _number(v: float, signed: bool = False) -> str:
    return (f"{v:+g}" if signed and v != 0 else f"{v:g}").replace("-", "\N{MINUS SIGN}")


def _legend_block(bar: Colorbar, font: ImageFont.FreeTypeFont, line: int, bar_w: int, max_w: int) -> Image.Image:
    """One legend entry as an image: the label, then the scale with its end values (or the part swatches)."""
    box = _cap_height(font)
    swatches = [(n, c) for n, c in zip(PART_NAMES, PART_HEX)] if bar.kind == "parts" else []
    if bar.nodata:
        swatches.append((bar.nodata, NO_DATA))
    sw_w = sum(box + box // 2 + int(font.getlength(n)) + box for n, _ in swatches)
    ramp_w = 0 if bar.kind == "parts" else bar_w + box
    width = int(min(max_w, max(ramp_w + sw_w, font.getlength(bar.label)))) + 2
    if ramp_w + sw_w > width:
        raise ValueError("legend entry is wider than the figure; shorten the labels")
    lines = _wrap(bar.label, font, width) if bar.label else []
    top = len(lines) * line
    img = Image.new("RGB", (width, top + (line if bar.kind == "parts" else 2 * line) + box // 2), "white")
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(lines):
        draw.text((0, i * line), text, font=font, fill=INK)
    x = 0
    if bar.kind != "parts":
        signed = bar.kind == "diverging"
        values = np.linspace(bar.lo, bar.hi, bar_w)
        rgb = contact_rgb(values - bar.lo, bar.hi - bar.lo) if not signed else diverging_rgb(values, max(abs(bar.lo), abs(bar.hi)))
        strip = np.repeat((rgb * 255).round().astype(np.uint8)[None], box, 0)
        img.paste(Image.fromarray(strip), (0, top + box // 4))
        draw.rectangle((0, top + box // 4, bar_w - 1, top + box // 4 + box - 1), outline=RULE)
        y = top + line + box // 4
        draw.text((0, y), _number(bar.lo, signed), font=font, fill=INK)
        hi = _number(bar.hi, signed)
        draw.text((bar_w - font.getlength(hi), y), hi, font=font, fill=INK)
        if signed:
            draw.text((bar_w / 2 - font.getlength("0") / 2, y), "0", font=font, fill=INK)
        x = ramp_w
    for name, colour in swatches:
        draw.rectangle((x, top + box // 4, x + box - 1, top + box // 4 + box - 1), fill=colour, outline=RULE)
        draw.text((x + box + box // 2, top + box // 4 + box / 2), name, font=font, fill=INK, anchor="lm")
        x += box + box // 2 + int(font.getlength(name)) + box
    return img


def grid(rows: Sequence[Sequence[np.ndarray]], row_labels: Sequence[str] | None = None, col_labels: Sequence[str] | None = None,
         colorbars: Sequence[Colorbar] = (), footnote: str | None = None) -> Image.Image:
    """Lay out rows of equally sized RGB panels with row labels (left), column labels (top), legend entries and a
    footnote (bottom) on white. Text is DejaVu Sans in INK, sized by MIN_CAP_PX_AT_1100. Labels wrap; a word that
    cannot fit raises instead of being clipped. ``image.info["qlib_layout"]`` records the sizes."""
    panels = [[np.asarray(p) for p in row] for row in rows]
    h, w = panels[0][0].shape[:2]
    if any(p.shape != (h, w, 3) or p.dtype != np.uint8 for row in panels for p in row):
        raise ValueError("every panel must be a uint8 RGB array of the same size")
    n_col = max(len(row) for row in panels)
    if (row_labels is not None and len(row_labels) != len(panels)) or (col_labels is not None and len(col_labels) != n_col):
        raise ValueError("one label per row / per column is required")
    gap = max(6, w // 50)
    pad = 2 * gap
    grid_w = n_col * w + (n_col - 1) * gap
    width = int(round((grid_w + 2 * pad) / (1 - ROW_LABEL_FRACTION))) if row_labels else grid_w + 2 * pad
    font = _font_for_width(width)
    line = int(np.ceil(font.size * 1.32))
    x0 = width - pad - grid_w                               # left edge of the panels
    row_text = [_wrap(t, font, x0 - pad - gap) for t in row_labels] if row_labels else [[] for _ in panels]
    col_text = [_wrap(t, font, w) for t in col_labels] if col_labels else []
    head_h = max(len(t) for t in col_text) * line + gap if col_text else 0
    row_h = [max(h, len(t) * line) for t in row_text]
    blocks = [_legend_block(b, font, line, int(0.2 * width), width - 2 * pad) for b in colorbars]
    legend_rows: list[list[Image.Image]] = []
    for block in blocks:                                    # flow the legend entries, wrapping to a new line
        if legend_rows and sum(b.width + 2 * pad for b in legend_rows[-1]) + block.width <= width - 2 * pad:
            legend_rows[-1].append(block)
        else:
            legend_rows.append([block])
    legend_h = sum(max(b.height for b in r) + gap for r in legend_rows) + (pad if legend_rows else 0)
    foot = _wrap(footnote, font, width - 2 * pad) if footnote else []
    foot_h = len(foot) * line + (pad if foot else 0)
    height = pad + head_h + sum(row_h) + gap * (len(panels) - 1) + legend_h + foot_h + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for j, text in enumerate(col_text):                     # column labels: centred, bottom-aligned
        for i, s in enumerate(text):
            y = pad + head_h - gap - (len(text) - i) * line
            draw.text((x0 + j * (w + gap) + w / 2, y), s, font=font, fill=INK, anchor="ma")
    y = pad + head_h
    for row, text, rh in zip(panels, row_text, row_h):
        for i, s in enumerate(text):                        # row label: left-aligned, centred on the row
            draw.text((pad, y + (rh - len(text) * line) / 2 + i * line), s, font=font, fill=INK)
        for j, panel in enumerate(row):
            canvas.paste(Image.fromarray(panel), (x0 + j * (w + gap), y + (rh - h) // 2))
        y += rh + gap
    y += pad - gap
    for blocks_row in legend_rows:
        x = pad
        for block in blocks_row:
            canvas.paste(block, (x, y))
            x += block.width + 2 * pad
        y += max(b.height for b in blocks_row) + gap
    y += pad if legend_rows else 0
    for i, s in enumerate(foot):
        draw.text((pad, y + i * line), s, font=font, fill=INK)
    cap = _cap_height(font)
    canvas.info["qlib_layout"] = {"width": width, "height": height, "panel": [w, h], "font_px": font.size, "cap_px": cap,
                                  "cap_px_at_1100": round(cap * 1100.0 / width, 2)}
    return canvas


# ------------------------------------------------------------------------------ output
def _jsonable(o: Any) -> Any:
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def save(fig_id: str, image: Image.Image, panels: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Path]:
    """Write the figure and what it was drawn from; return the four paths.

    ``panels`` maps a panel key to the array drawn there, or to a dict of named arrays (stored as "key/name").
    ``meta`` is free-form and JSON-serialisable: example ids, the selection rule, frames, cameras, colour limits.
    """
    if not re.fullmatch(r"[A-Za-z0-9_]+", fig_id):
        raise ValueError(f"figure id must be letters, digits and underscores, got {fig_id!r}")
    arrays: dict[str, np.ndarray] = {}
    for key, value in panels.items():
        for name, array in (value.items() if isinstance(value, Mapping) else [("", value)]):
            arrays[f"{key}/{name}" if name else str(key)] = np.asarray(array)
    out = OUT_ROOT / fig_id
    out.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    paths = {"figure": out / "figure.png", "panels": out / "panels.npz", "meta": out / "meta.json"}
    image.save(paths["figure"], optimize=True)
    np.savez_compressed(paths["panels"], **arrays)
    asset = image
    if image.width > ASSET_MAX_WIDTH:
        asset = image.resize((ASSET_MAX_WIDTH, round(image.height * ASSET_MAX_WIDTH / image.width)), Image.LANCZOS)
    if features.check("webp"):
        paths["asset"] = ASSET_DIR / f"qual_{fig_id}.webp"
        asset.save(paths["asset"], "WEBP", quality=ASSET_QUALITY, method=6)
    else:
        paths["asset"] = ASSET_DIR / f"qual_{fig_id}.png"
        asset.save(paths["asset"], optimize=True)
    record = {"fig_id": fig_id, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "outputs": paths,
              "master_size": list(image.size), "asset_size": list(asset.size), "layout": image.info.get("qlib_layout"),
              "display_rule": DISPLAY_RULE, "soft_contact": f"exp(-distance / {SOFT_SCALE} m)",
              "colours": {"zero": ZERO, "orange": ORANGE, "dark": DARK, "blue": BLUE, "no_data": NO_DATA,
                          "parts": dict(zip(PART_NAMES, PART_HEX))},
              "panels": {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()}, "meta": dict(meta)}
    paths["meta"].write_text(json.dumps(record, indent=2, default=_jsonable, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths
