"""Figure B: the Figure A samples shown in the shared canonical space.
For each selected (category, verb) the 4 meshes' SEQUENCE-level X_soft vectors are splatted onto
the category TEMPLATE mesh (aligned canonical frame), row 1 backend 'aligned', row 2 'dino',
column 5 = mean over the 4 samples. Same camera in every panel of a category.
Run (no torch): PYTHONPATH=/home/uhnam/workspace/dexcore python scripts/figB_canonical_frames.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import FIG, Template, cat_camera, dino_fallback, load_cache, seq_row
from render_util import compose_grid, render_mesh_with_values

SIZE = (400, 400)
VMAX = 0.7          # fixed colour scale for every panel: value/VMAX clipped to [0,1]; 0 = grey
BACKENDS = ["aligned", "dino"]


def main():
    sel = pd.read_csv(FIG / "figA_selection.csv")
    fb = dino_fallback()
    all_ims, all_labels, all_rows = [], [], []
    for (cat, role, hand, verb), g in sel.groupby(["category", "role", "hand", "verb"], sort=False):
        T = Template(cat)
        view, dist = cat_camera(cat)
        ims, labels, row_titles = [], [], []
        for be in BACKENDS:
            z = load_cache(be, cat, role, seq=True)
            xs = []
            for _, r in g.iterrows():
                i = seq_row(z, r.sequence_id, hand)
                assert z["mesh_id"][i] == f"{r.mesh_id:03d}"
                x = z["X_soft"][i].astype(np.float64)
                xs.append(x)
                ims.append(render_mesh_with_values(T.verts, T.faces, np.clip(T.splat(x) / VMAX, 0, 1),
                                                   view=view, size=SIZE, normalise=False, dist=dist))
                lab = f"{be}: mesh {r.mesh_id:03d} | {r.sequence_id.split('/')[-1]}"
                if be == "dino":
                    lab += f" | fb {fb[(cat, f'{r.mesh_id:03d}')]:.2f}"
                labels.append(lab)
                print(cat, verb, be, r.mesh_id, f"max={x.max():.3f} mass={x.sum():.1f}")
            mean = np.mean(xs, 0)
            ims.append(render_mesh_with_values(T.verts, T.faces, np.clip(T.splat(mean) / VMAX, 0, 1),
                                               view=view, size=SIZE, normalise=False, dist=dist))
            labels.append(f"{be}: mean of 4 (max {mean.max():.2f})")
            row_titles.append(be)
        title = (f"{cat}/{role} '{verb}' ({hand}) on template {T.template_id} -- X_soft seq-level, "
                 f"scale 0..{VMAX} (grey = 0)")
        compose_grid(ims, labels, ncols=5, row_titles=row_titles, title=title, label_px=15).save(
            FIG / f"figB_{cat}_{verb.replace(' ', '_')}.png")
        all_ims += ims; all_labels += labels
        all_rows += [f"{cat} '{verb}' {be}" for be in BACKENDS]
    compose_grid(all_ims, all_labels, ncols=5, row_titles=all_rows, label_px=15,
                 title=f"Figure B: Figure A samples in the shared canonical space (template mesh, X_soft, scale 0..{VMAX})"
                 ).save(FIG / "figB_all.png")


if __name__ == "__main__":
    main()
