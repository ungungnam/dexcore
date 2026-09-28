#!/usr/bin/env python
"""What the time axis and the mesh axis do, drawn on the object surface in ONE shared frame.

The earlier version rendered every instance re-centred and rescaled to unit radius, which hides
the very thing the mesh row is supposed to show: the instances differ in proportion, so 'the same
grip' sits at a different place in the shared frame. Here every mesh is mapped into the category's
canonical frame (the frame the vectors are compared in) and drawn with ONE fixed camera, so the
panels are directly comparable. A black marker is the contact centroid.

No torch in this process (open3d's EGL renderer and CUDA do not coexist).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import pandas as pd
from open3d.visualization import rendering
from PIL import Image

ROOT = Path("/result/uhnam/dexcore/canonical_contact")
sys.path.insert(0, str(ROOT / "scripts"))
import render_util as RU
from render_util import compose_grid, value_colormap

from src.analysis.canonical import backends as BR

SEQ = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
MESH_DICT = SEQ / "assets/taco_mesh_dict.npy"
CAT, ROLE, HAND = "spatula", "tool", "R"
BINS, VMAX, SIZE, N_MESH = 5, 0.55, (400, 400), 5
VIEW = np.array([0.35, -0.95, 0.55])
OUT = ROOT / "figures"


def render_shared(verts, faces, values, centroid, rad, size=SIZE):
    """Like render_util.render_mesh_with_values but with a camera fixed by the caller, so every
    panel shares one frame, and with a marker at the contact centroid."""
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(verts, float)),
                                  o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    m.compute_vertex_normals()
    m.vertex_colors = o3d.utility.Vector3dVector(value_colormap(values))
    mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
    mat.base_color = (1.0, 1.0, 1.0, 1.0); mat.base_roughness = 0.7
    r = RU._get_renderer(size)
    r.scene.clear_geometry()
    r.scene.add_geometry("mesh", m, mat)
    if centroid is not None:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=rad * 0.045)
        ball.translate(centroid); ball.compute_vertex_normals()
        ball.paint_uniform_color([0.05, 0.05, 0.05])
        mk = rendering.MaterialRecord(); mk.shader = "defaultLit"
        r.scene.add_geometry("centroid", ball, mk)
    d = VIEW / np.linalg.norm(VIEW)
    r.setup_camera(40.0, [0, 0, 0], (d * rad * 2.9).tolist(), [0, 0, 1])
    return Image.fromarray(np.asarray(r.render_to_image()))


def accumulate():
    wi = pd.read_csv(ROOT / "window_index.csv")
    files = pd.read_csv(SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    rows = wi[(wi[f"{ROLE}_cat"] == CAT) & (wi[f"touch_{HAND}_{ROLE}"] >= 0.2)]
    acc: dict = {}
    for mesh, g in rows.groupby(f"{ROLE}_mesh"):
        key = f"{int(mesh):03d}"
        for seq, gg in g.groupby("sequence_id"):
            with np.load(SEQ / "sequences" / files[seq]) as z:
                d = z["contact_right" if HAND == "R" else "contact_left"]
            n_tool = int(gg.n_tool.iloc[0])
            d = d[:, :n_tool] if ROLE == "tool" else d[:, n_tool:]
            starts = sorted(gg.start.astype(int))
            lo, hi = starts[0], starts[-1] + 63
            for s in starts:
                fr = np.arange(s, min(s + 64, len(d)))
                b = np.clip(((fr - lo) / max(hi - lo, 1) * BINS).astype(int), 0, BINS - 1)
                soft = np.exp(-d[fr] / 0.02)
                for k in range(BINS):
                    m = b == k
                    if m.any():
                        cur = acc.setdefault((key, k), [np.zeros(soft.shape[1]), 0])
                        cur[0] += soft[m].sum(0); cur[1] += int(m.sum())
    return {k: v[0] / v[1] for k, v in acc.items()}, rows


def main():
    OUT.mkdir(exist_ok=True)
    md = np.load(MESH_DICT, allow_pickle=True).item()
    be = BR.load("normalized", str(ROOT / "canonical_backend"))
    means, rows = accumulate()
    counts = rows.groupby(f"{ROLE}_mesh").sequence_id.nunique().sort_values(ascending=False)
    meshes = [f"{int(m):03d}" for m in counts.index[:N_MESH]]
    main_mesh = meshes[0]
    can = {m: be.to_canonical(CAT, m, md[m]["verts_original"]) for m in meshes}
    rad = max(float(np.linalg.norm(v, axis=1).max()) for v in can.values())

    def centroid(mesh, k):
        w = means[(mesh, k)]
        return (can[mesh] * w[:, None]).sum(0) / max(w.sum(), 1e-9)

    imgs, labels = [], []
    for k in range(BINS):
        v = means[(main_mesh, k)]
        imgs.append(render_shared(can[main_mesh], md[main_mesh]["faces"],
                                  np.clip(v / VMAX, 0, 1), centroid(main_mesh, k), rad))
        labels.append(f"phase {k/BINS:.1f}-{(k+1)/BINS:.1f}   contact {v.sum():.0f}")
    mid = BINS // 2
    for m in meshes:
        v = means[(m, mid)]
        imgs.append(render_shared(can[m], md[m]["faces"], np.clip(v / VMAX, 0, 1),
                                  centroid(m, mid), rad))
        labels.append(f"mesh {m}   contact {v.sum():.0f}")
    fig = compose_grid(imgs, labels=labels, ncols=BINS,
                       row_titles=[f"time (mesh {main_mesh} fixed)", "mesh (phase 0.4-0.6 fixed)"],
                       title=f"{CAT}, right hand, all panels in the shared canonical frame with one camera: "
                             f"time changes HOW MUCH, the mesh changes WHERE (dot = contact centroid)",
                       label_px=17)
    p = OUT / "figF_time_vs_mesh.png"
    fig.save(p)
    c = np.stack([centroid(m, mid) for m in meshes])
    print("meshes:", meshes)
    print("centroid spread between meshes at mid phase:",
          float(np.linalg.norm(c - c.mean(0), axis=1).mean()).__round__(4))
    t = np.stack([centroid(main_mesh, k) for k in range(BINS)])
    print("centroid travel over the take (same mesh):", float(np.linalg.norm(t[-1] - t[0])).__round__(4))
    print("saved", p, fig.size)


if __name__ == "__main__":
    main()
