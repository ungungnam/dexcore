"""Headless mesh rendering helpers (open3d OffscreenRenderer, EGL).

IMPORTANT: do not import torch in the same process -- open3d's renderer and CUDA do not coexist.

render_mesh_with_values(verts, faces, values, view, size) -> PIL.Image
    values in [0,1] per vertex; fixed colormap, grey for values == 0.
compose_grid(images, labels, ncols, ...) -> PIL.Image
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import open3d as o3d
from open3d.visualization import rendering
from PIL import Image, ImageDraw, ImageFont

GREY = np.array([0.78, 0.78, 0.78])
# Fixed colormap: yellow -> orange -> red (0 is grey, handled separately).
_CMAP_STOPS = np.array([[1.00, 0.95, 0.55], [1.00, 0.60, 0.10], [0.80, 0.05, 0.05]])

# Named camera views: unit direction from target towards the eye (z is up).
VIEWS = {
    "iso":   np.array([0.62, -0.95, 0.42]),
    "top":   np.array([0.05, -0.05, 1.00]),
    "front": np.array([0.0, -1.0, 0.25]),
    "side":  np.array([1.0, 0.0, 0.25]),
}


def value_colormap(values: np.ndarray, zero_eps: float = 1e-6) -> np.ndarray:
    """(N,) in [0,1] -> (N,3) RGB. Exactly-zero (<= eps) vertices are grey."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0.0, 1.0)
    t = v * (len(_CMAP_STOPS) - 1)
    i0 = np.clip(np.floor(t).astype(int), 0, len(_CMAP_STOPS) - 2)
    w = (t - i0)[:, None]
    rgb = _CMAP_STOPS[i0] * (1 - w) + _CMAP_STOPS[i0 + 1] * w
    rgb[v <= zero_eps] = GREY
    return rgb


def normalise_frame(verts: np.ndarray) -> np.ndarray:
    """Centre on the bounding-box centre and scale to unit radius. No rotation."""
    v = np.asarray(verts, dtype=np.float64)
    lo, hi = v.min(0), v.max(0)
    v = v - (lo + hi) / 2
    r = np.linalg.norm(v, axis=1).max()
    return v / max(r, 1e-9)


_RENDERERS: dict = {}


def _get_renderer(size):
    key = tuple(size)
    if key not in _RENDERERS:
        r = rendering.OffscreenRenderer(size[0], size[1])
        r.scene.set_background([1, 1, 1, 1])
        r.scene.scene.set_sun_light([-0.4, -0.5, -0.8], [1, 1, 1], 60000)
        r.scene.scene.enable_sun_light(True)
        r.scene.scene.set_indirect_light_intensity(25000)
        _RENDERERS[key] = r
    return _RENDERERS[key]


def render_mesh_with_values(verts, faces, values, view="iso", size=(512, 512),
                            normalise=True, fov=40.0, dist=2.9, up=(0, 0, 1)) -> Image.Image:
    """Render one mesh with per-vertex scalar values in [0,1] as vertex colours.

    verts (N,3), faces (F,3), values (N,). `view` is a name in VIEWS or a 3-vector direction.
    With normalise=True the mesh is centred and scaled to unit radius (no rotation) so the same
    camera works for every instance.
    """
    v = normalise_frame(verts) if normalise else np.asarray(verts, dtype=np.float64)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v),
                                  o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)))
    m.compute_vertex_normals()
    m.vertex_colors = o3d.utility.Vector3dVector(value_colormap(values))
    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (1.0, 1.0, 1.0, 1.0)
    mat.base_roughness = 0.7
    r = _get_renderer(size)
    r.scene.clear_geometry()
    r.scene.add_geometry("mesh", m, mat)
    d = VIEWS[view] if isinstance(view, str) else np.asarray(view, dtype=np.float64)
    d = d / np.linalg.norm(d)
    ctr = v.mean(0) * 0 if normalise else (v.min(0) + v.max(0)) / 2
    rad = 1.0 if normalise else np.linalg.norm(v - ctr, axis=1).max()
    r.setup_camera(fov, ctr.tolist(), (ctr + d * rad * dist).tolist(), list(up))
    return Image.fromarray(np.asarray(r.render_to_image()))


def _font(px):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            pass
    return ImageFont.load_default()


def compose_grid(images, labels=None, ncols=4, row_titles=None, title=None,
                 label_px=18, pad=8, bg=(255, 255, 255)) -> Image.Image:
    """Tile PIL images into a grid with optional per-cell labels (below), per-row titles
    (left margin) and a figure title (top)."""
    images = list(images)
    n = len(images)
    nrows = int(np.ceil(n / ncols))
    cw = max(im.width for im in images)
    ch = max(im.height for im in images)
    lab_h = label_px + 6 if labels else 0
    left = int(label_px * 1.6) if row_titles else 0
    top = int(label_px * 2.2) if title else 0
    W = left + ncols * (cw + pad) + pad
    H = top + nrows * (ch + lab_h + pad) + pad
    out = Image.new("RGB", (W, H), bg)
    dr = ImageDraw.Draw(out)
    f = _font(label_px)
    for k, im in enumerate(images):
        rr, cc = divmod(k, ncols)
        x = left + pad + cc * (cw + pad)
        y = top + pad + rr * (ch + lab_h + pad)
        out.paste(im, (x, y))
        if labels:
            txt = labels[k]
            tw = dr.textlength(txt, font=f)
            dr.text((x + (cw - tw) / 2, y + ch + 2), txt, fill=(20, 20, 20), font=f)
    if row_titles:
        for rr, t in enumerate(row_titles):
            y = top + pad + rr * (ch + lab_h + pad)
            tmp = Image.new("RGB", (ch, int(label_px * 1.6)), bg)
            d2 = ImageDraw.Draw(tmp)
            tw = d2.textlength(t, font=f)
            d2.text(((ch - tw) / 2, 4), t, fill=(20, 20, 20), font=f)
            out.paste(tmp.rotate(90, expand=True), (0, y))
    if title:
        ft = _font(int(label_px * 1.3))
        tw = dr.textlength(title, font=ft)
        dr.text(((W - tw) / 2, 8), title, fill=(20, 20, 20), font=ft)
    return out
