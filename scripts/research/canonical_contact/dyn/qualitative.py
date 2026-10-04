#!/usr/bin/env python
"""Qualitative sequence examples, selected automatically from exp2/per_sequence.csv.

Selection rule (per-take statistics on touching frames, ranked WITHIN each group by dividing by
the group median so no category dominates; takes must have >= 64 frames):
  stable      lowest  delta_raw_mean / group-median  among takes with mass_mean above the group
              median and no zero frames (real contact that does not change)
  dynamic     highest D32_raw / group-median (largest change over a 1-s lag)
  on_off      n_on_off >= 2, mass_max at or above the group's median mass_mean (the 'on' state is
              real contact) and the most balanced zero / non-zero split (min(f, 1-f) largest)
  migration   largest centroid_net / group-median among takes whose mass stays within a factor
              2 (mass_max / mass_min < 2) and that have no zero frames: the contact moves
              without the amount changing
Each example: contact mass over time, centroid coordinates over time, and the per-vertex soft
contact rendered on the mesh (canonical frame, one camera) at 6 frames spread over the take.
No torch in this process (open3d EGL renderer).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dc_common as C
sys.path.insert(0, str(C.STUDY / "scripts"))
import render_util as RU                                  # noqa: E402
from fig_common import CAT_VIEW, CAT_DIST                 # noqa: E402
from src.analysis.canonical import backends as BR         # noqa: E402

OUT = C.OUT / "qualitative"
N_FRAMES = 6


def select(PS):
    PS = PS[PS.n_frames >= 64].copy()
    for col in ("delta_raw_mean", "D32_raw", "centroid_net"):
        PS[col + "_rel"] = PS[col] / PS.groupby("group")[col].transform("median")
    PS["mass_rel"] = PS.mass_mean / PS.groupby("group").mass_mean.transform("median")
    ex = {}
    c = PS[(PS.mass_rel >= 1) & (PS.frac_zero == 0)]
    ex["stable"] = c.loc[c.delta_raw_mean_rel.idxmin()]
    ex["dynamic"] = PS.loc[PS.D32_raw_rel.idxmax()]
    c = PS[(PS.n_on_off >= 2) & (PS.mass_max >= PS.groupby("group").mass_mean.transform("median"))].copy()
    c["bal"] = np.minimum(c.frac_zero, 1 - c.frac_zero)
    ex["on_off"] = c.loc[c.bal.idxmax()] if len(c) else None
    c = PS[(PS.mass_max / PS.mass_min < 2) & (PS.frac_zero == 0)]
    ex["migration"] = c.loc[c.centroid_net_rel.idxmax()] if len(c) else None
    return ex


def render(verts, faces, values, centroid, view, dist, rad, size=(360, 360)):
    import open3d as o3d
    from open3d.visualization import rendering
    from PIL import Image
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(verts, float)),
                                  o3d.utility.Vector3iVector(np.asarray(faces, np.int32)))
    m.compute_vertex_normals()
    m.vertex_colors = o3d.utility.Vector3dVector(RU.value_colormap(values))
    mat = rendering.MaterialRecord(); mat.shader = "defaultLit"; mat.base_color = (1, 1, 1, 1); mat.base_roughness = 0.7
    r = RU._get_renderer(size); r.scene.clear_geometry(); r.scene.add_geometry("mesh", m, mat)
    if centroid is not None:
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=rad * 0.045)
        ball.translate(centroid); ball.compute_vertex_normals(); ball.paint_uniform_color([0.05, 0.05, 0.05])
        mk = rendering.MaterialRecord(); mk.shader = "defaultLit"; r.scene.add_geometry("centroid", ball, mk)
    d = np.asarray(view, float); d = d / np.linalg.norm(d)
    r.setup_camera(40.0, [0, 0, 0], (d * rad * dist).tolist(), [0, 0, 1])
    return np.asarray(r.render_to_image())


def main():
    OUT.mkdir(exist_ok=True)
    PS = pd.read_csv(C.OUT / "exp2" / "per_sequence.csv")
    ex = select(PS)
    thr = C.zero_threshold()
    be = BR.load("normalized", str(C.BACKEND_DIR))
    md = np.load(C.SEQ / "assets/taco_mesh_dict.npy", allow_pickle=True).item()
    files = pd.read_csv(C.SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    wi = pd.read_csv(C.STUDY / "window_index.csv").drop_duplicates("sequence_id").set_index("sequence_id")
    rows = []
    for kind, r in ex.items():
        if r is None:
            rows.append(dict(kind=kind, found=False)); continue
        cat, role, hand = r.group.split("__")
        X, M, P = C.load_group(cat, role, hand, stride=1)
        idx = M.index[M.sequence_id.values == r.sequence_id].values
        idx = idx[np.argsort(M.frame.values[idx])]
        Y = X[idx]; fr = M.frame.values[idx]; m = C.mass(Y); mu = C.centroid(Y, P)
        zero = C.is_zero(m, M.n_hard.values[idx], thr)
        mesh = str(M.mesh_id.values[idx[0]])
        # raw per-vertex soft contact for the rendered frames
        with np.load(C.SEQ / "sequences" / files[r.sequence_id]) as z:
            d = z["contact_left" if hand == "L" else "contact_right"]
        n_tool = int(wi.loc[r.sequence_id, "n_tool"])
        d = d[:, :n_tool] if role == "tool" else d[:, n_tool:]
        verts = be.to_canonical(cat, mesh, md[mesh]["verts_original"])
        rad = float(np.linalg.norm(verts, axis=1).max())
        view, dist = CAT_VIEW.get(cat, [0.62, -0.95, 0.42]), CAT_DIST.get(cat, 2.9)
        sel = np.linspace(0, len(fr) - 1, N_FRAMES).round().astype(int)
        fig = plt.figure(figsize=(3.0 * N_FRAMES, 8.5))
        gs = fig.add_gridspec(3, N_FRAMES, height_ratios=[1.0, 1.0, 1.6])
        t = (fr - fr[0]) / C.FPS
        ax = fig.add_subplot(gs[0, :]); ax.plot(t, m, color="k"); ax.axhline(thr, color="r", ls="--", lw=0.8, label="zero-contact threshold")
        for s in sel: ax.axvline(t[s], color="0.6", lw=0.6)
        ax.set_ylabel("contact mass  sum X"); ax.set_title(f"{kind}: {r.sequence_id}  [{cat}/{role}/{hand}, mesh {mesh}]  "
                                                          f"frac_zero={r.frac_zero:.2f}  D32_raw={r.D32_raw:.2f}  centroid_net={r.centroid_net:.3f}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax = fig.add_subplot(gs[1, :])
        for i, lab in enumerate("xyz"):
            ax.plot(t[~zero], mu[~zero, i], label=f"centroid {lab}")
        ax.set_ylabel("centroid (canonical units)"); ax.set_xlabel("time in take [s]"); ax.legend(fontsize=8, ncol=3)
        for j, s in enumerate(sel):
            soft = np.exp(-d[fr[s]] / 0.02)
            img = render(verts, md[mesh]["faces"], np.clip(soft / 0.55, 0, 1), None if zero[s] else mu[s], view, dist, rad)
            ax = fig.add_subplot(gs[2, j]); ax.imshow(img); ax.axis("off")
            ax.set_title(f"t={t[s]:.2f}s  mass={m[s]:.0f}" + ("  (zero)" if zero[s] else ""), fontsize=9)
        fig.tight_layout()
        p = OUT / f"qual_{kind}.png"; fig.savefig(p, dpi=110); plt.close(fig)
        rows.append(dict(kind=kind, found=True, group=r.group, sequence_id=r.sequence_id, mesh_id=mesh, verb=r.verb,
                         n_frames=int(r.n_frames), frac_zero=r.frac_zero, n_on_off=int(r.n_on_off), mass_mean=r.mass_mean,
                         mass_min=r.mass_min, mass_max=r.mass_max, delta_raw_mean=r.delta_raw_mean, D32_raw=r.D32_raw,
                         centroid_net=r.centroid_net, centroid_path=r.centroid_path, figure=str(p.name)))
        print(kind, r.sequence_id, r.group)
    pd.DataFrame(rows).to_csv(C.OUT / "qualitative_examples.csv", index=False)


if __name__ == "__main__":
    main()
