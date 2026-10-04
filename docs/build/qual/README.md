# Qualitative figures

Scripts that render the page's qualitative figures: the real object mesh, the recorded MANO hand and
the contact on the object's surface, from recorded data and from saved or freshly computed outputs of
the trained models. Nothing is drawn by hand and nothing is trained here.

`qlib.py` is the shared module (data access, colour scales, rendering, layout, output). Figure
scripts go next to it, one per figure id.

## Run

CPU only, with the software renderer (pyvista / VTK on OSMesa). From the repository root:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexmachina
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    PYVISTA_OFF_SCREEN=true python docs/build/qual/<script>.py
```

`qlib.require_headless()` raises if any of the four variables is missing. Without
`VTK_DEFAULT_OPENGL_WINDOW` the renderer may open a context on a GPU, so there is no other way to
run these scripts. A panel takes about 0.1 to 0.5 s; loading one TACO take (MANO on CPU) a few seconds.

## What the scripts need

Unlike the chart modules in `../figs/`, these cannot be rebuilt from the repository alone. They read

- the result tree, `$DEXCORE_RESULT_ROOT` (default `/result/uhnam/dexcore`): the sequences of the
  hierarchical study (`taco/50_hier_contact_gen`, `arctic/40_hier_contact_gen`), the meshes and dense
  hand-to-vertex distances (`taco/30_bimart_gen3_scene_scale`, `arctic/20_bimart_contact_probe`) and
  the caches and saved predictions under `reports/`;
- the raw TACO hand and object poses (`$DEXCORE_TACO_ROOT`, default `/backups/uhnam/TACO`) and the
  MANO model files, for the TACO hand; the processed ARCTIC files under
  `third_party/BimArt/data/arctic_processed_data`, for the ARCTIC hand and object pose;
- for figures that run a model: its checkpoint under `/ckpt/uhnam/dexcore`, loaded read-only.

Every input is opened read-only.

## Display rule

All geometry is in the contacted object's own rigid frame, in metres. A contact map of the studies is
512 values on canonical points, made from the recorded per-vertex contact `s = exp(-d / 2 cm)` with
the study's operator `W` (512 x vertices): `C = W s`. Every map on the page, recorded or generated,
is drawn on the mesh with the same operator transposed and normalised:

```
value(vertex u) = sum_k W[k, u] C[k] / sum_k W[k, u]
```

This is a smoothing, not an inverse, so a recorded map and a model output are compared only after
both went through it (`Example.to_vertices`). The recorded per-vertex contact itself is
`Example.dense`. Two properties of `W` are visible in the pictures and belong in a caption when they
matter:

- A vertex that no canonical point reaches has no value and is drawn in pale violet (`NO_DATA`).
  ARCTIC meshes are reached completely. 39 of the 79 TACO instance meshes are not, because the
  canonical points come from the category's template mesh (`Example.covered` is the mask).
- ARCTIC: `W` is defined on the closed rest pose, so a canonical point near the closing surfaces
  mixes vertices of both parts. Contact on one part is drawn faintly on the other part as well, also
  when the object is open.

## Colour scales

| scale | colours |
|---|---|
| contact, `contact_rgb(values, vmax)` | grey `#dcdbd5` at 0, orange `#eb6834` at `vmax / 2`, dark red-brown `#7a2408` at `vmax` |
| difference, `diverging_rgb(values, vabs)` | blue `#2a78d6` at `-vabs` (less), grey `#dcdbd5` at 0, orange `#eb6834` at `+vabs` (more) |
| not reached by the map | pale violet `#e7e1f1` |
| hand | skin `#e8d2c0`; ghost: `#cfcec8` at 35 % opacity |
| hand parts | palm `#6b6a63`, thumb `#2a78d6`, index `#eb6834`, middle `#1baf7a`, ring `#eda100`, little `#8e5bd0` |

Colours are linear in sRGB between the stops. The renderer shades them, so the colour bar of a figure
shows the unshaded scale. Background white, labels DejaVu Sans in `#1f1e1b`, sized so that capital
letters are at least 11 px tall when the figure is shown 1100 px wide.

## Outputs

`qlib.save(fig_id, image, panels, meta)` writes

```
<result root>/reports/project_page_qualitative/<fig_id>/figure.png    master image
<result root>/reports/project_page_qualitative/<fig_id>/panels.npz    every array that was drawn, keyed by panel
<result root>/reports/project_page_qualitative/<fig_id>/meta.json     examples, selection rule, frames, cameras, colour limits
docs/assets/img/qual_<fig_id>.webp                                    page asset, at most 2200 px wide
```

Only the `.webp` is part of the repository.

## Rules for a figure script

- The example is chosen by a rule the script computes and records in `meta.json`, never by eye. If
  the chosen example renders badly, change the camera, not the example.
- TACO and ARCTIC never share a panel and are never pooled.
- One camera per row (`fit_camera` on everything the row shows), so its panels are comparable.
- Do not call the studies' own scripts that rewrite their outputs; load arrays and checkpoints directly.
