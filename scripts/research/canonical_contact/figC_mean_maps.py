"""Figure C: canonical MEAN contact map per verb on the category template, from
figures_data/mean_maps_<backend>_<category>_<role>_<hand>.npz (step 8; verbs, mean_maps (V,K),
n_sequences_per_verb). One column per verb, rows = backend (aligned, dino). Same camera per
category, fixed colour scale per figure (printed in the title).
Run (no torch): PYTHONPATH=/home/uhnam/workspace/dexcore python scripts/figC_mean_maps.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_common import FIG, FIGDATA, SIX, Template, cat_camera
from render_util import compose_grid, render_mesh_with_values

SIZE = (360, 360)
BACKENDS = ["aligned", "dino"]
MIN_SEQ = 2          # verbs with fewer sequences are dropped from the figure (noted in the title)


def main():
    for cat, role, hand in SIX:
        maps = {be: np.load(FIGDATA / f"mean_maps_{be}_{cat}_{role}_{hand}.npz", allow_pickle=True)
                for be in BACKENDS}
        verbs = [str(v) for v in maps["aligned"]["verbs"]]
        nseq = {v: int(n) for v, n in zip(verbs, maps["aligned"]["n_sequences_per_verb"])}
        for be in BACKENDS:
            assert [str(v) for v in maps[be]["verbs"]] == verbs, (cat, be)
        keep = [v for v in verbs if nseq[v] >= MIN_SEQ]
        keep.sort(key=lambda v: -nseq[v])
        dropped = [v for v in verbs if v not in keep]
        vmax = max(float(maps[be]["mean_maps"][verbs.index(v)].max()) for be in BACKENDS for v in keep)
        vmax = float(np.ceil(vmax * 20) / 20)      # round up to 0.05
        T = Template(cat)
        view, dist = cat_camera(cat)
        ims, labels = [], []
        for be in BACKENDS:
            for v in keep:
                x = maps[be]["mean_maps"][verbs.index(v)].astype(np.float64)
                ims.append(render_mesh_with_values(T.verts, T.faces, np.clip(T.splat(x) / vmax, 0, 1),
                                                   view=view, size=SIZE, normalise=False, dist=dist))
                labels.append(f"{be}: '{v}'  n_seq={nseq[v]}  max={x.max():.2f}")
        title = (f"{cat}/{role} ({hand}) verb-mean X_soft on template {T.template_id}, scale 0..{vmax:.2f}"
                 + (f"  [dropped n_seq<{MIN_SEQ}: {', '.join(dropped)}]" if dropped else ""))
        compose_grid(ims, labels, ncols=len(keep), row_titles=BACKENDS, title=title, label_px=14).save(
            FIG / f"figC_{cat}_{role}_{hand}.png")
        print(cat, role, hand, "verbs:", {v: nseq[v] for v in keep}, "vmax", vmax)


if __name__ == "__main__":
    main()
