"""Step 10 Figure A: sequence-level hard contact maps on instance meshes in their OWN frames
(centred, unit radius, NOT rotated) -- shows the raw cross-instance inconsistency.
Run: PYTHONPATH=/home/uhnam/workspace/dexcore python scripts/figA_instance_frames.py  (no torch)."""
import sys, csv
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path("/result/uhnam/dexcore/canonical_contact")
sys.path.insert(0, str(ROOT / "scripts"))
from render_util import render_mesh_with_values, compose_grid

MESH_DICT = "/result/uhnam/dexcore/bimart_taco/assets/taco_mesh_dict.npy"
# (category, role, hand, verb) -- primary hand: R for tool, L for target
PLAN = [("spatula", "tool", "R", "scrape off"),
        ("knife", "tool", "R", "cut"),
        ("bowl", "target", "L", "put in"),
        ("kettle", "tool", "R", "pour in some")]
N_MESH = 4
VIEW, SIZE = "iso", (420, 420)


def choose(samples):
    """For each planned category pick N_MESH distinct meshes with touch_frac>=0.5 samples; per mesh
    take the sequence with the highest mean window touch_frac (ties -> sequence_id order)."""
    rows = []
    for cat, role, hand, verb in PLAN:
        d = samples[(samples.category == cat) & (samples.role == role) & (samples.hand == hand)
                    & (samples.verb == verb) & (samples.touch_frac >= 0.5)]
        per_mesh = d.groupby("mesh_id").sequence_id.nunique().sort_values(ascending=False)
        meshes = sorted(per_mesh.index[:N_MESH].tolist())
        if len(meshes) < N_MESH:
            print(f"WARNING {cat}/{verb}: only {len(meshes)} distinct meshes with contact")
        for m in meshes:
            dm = d[d.mesh_id == m]
            seq_score = dm.groupby("sequence_id").touch_frac.mean().sort_values(ascending=False)
            best = sorted(seq_score[seq_score == seq_score.iloc[0]].index)[0]
            stem = dm[dm.sequence_id == best].file.iloc[0]
            rows.append(dict(category=cat, role=role, hand=hand, verb=verb, mesh_id=int(m),
                             sequence_id=best, file=stem,
                             seq_touch_mean=float(seq_score.iloc[0]),
                             split=dm[dm.sequence_id == best].split.iloc[0]))
    return pd.DataFrame(rows)


def main():
    samples = pd.read_csv(ROOT / "samples_index.csv")
    sel = choose(samples)
    fig = ROOT / "figures"; fig.mkdir(exist_ok=True)
    sel.to_csv(fig / "figA_selection.csv", index=False)
    meshes = np.load(MESH_DICT, allow_pickle=True).item()
    rows_img, row_titles = [], []
    for (cat, role, hand, verb), g in sel.groupby(["category", "role", "hand", "verb"], sort=False):
        ims, labels = [], []
        for _, r in g.iterrows():
            md = meshes[f"{r.mesh_id:03d}"]
            verts, faces = np.asarray(md["verts_original"]), np.asarray(md["faces"])
            dw = np.load(ROOT / "dense_window_cache" / r.file)
            vals = dw[f"seq_hard_{hand}_{role}"].astype(np.float32)
            assert len(vals) == len(verts), (r.file, len(vals), len(verts))
            ims.append(render_mesh_with_values(verts, faces, vals, view=VIEW, size=SIZE))
            labels.append(f"mesh {r.mesh_id:03d} | {r.sequence_id.split('/')[-1]}")
            print(cat, verb, r.mesh_id, r.sequence_id, f"maxhard={vals.max():.2f} frac>0.5={(vals>0.5).mean():.3f}")
        rt = f"{cat}/{role} - '{verb}' ({hand} hand)"
        compose_grid(ims, labels, ncols=N_MESH, title=rt + "  [instance frames, seq hard contact]").save(
            fig / f"figA_{cat}_{verb.replace(' ', '_')}.png")
        rows_img += ims; row_titles.append(f"{cat} '{verb}'")
    labels_all = [f"mesh {r.mesh_id:03d} | {r.sequence_id.split('/')[-1]}" for _, r in sel.iterrows()]
    compose_grid(rows_img, labels_all, ncols=N_MESH, row_titles=row_titles,
                 title="Figure A: sequence-level hard contact on instance meshes (own frame, no rotation)").save(
        fig / "figA_all.png")
    print(sel)


if __name__ == "__main__":
    main()
