# Project page

**What Should a Temporal Contact Generator Actually Model?** — the research argument of this
repository on one page: motivation, three findings, and the current hypothesis.

This is the detailed working page. Since 5 October 2026 it is no longer served on the web: the
project's GitHub Pages address shows the concise public page built in `public_page/`. Preview this
page locally (see below).

## Run it

The page is static HTML, CSS and JavaScript with no build step and no external dependency.

```bash
python -m http.server -d docs 8000        # then open http://localhost:8000
```

Opening `docs/index.html` directly from disk also works: the chart data is loaded as a script
(`data/figures.js`), not fetched.

## What is in this folder

```
index.html                 the page: copy, diagrams (inline SVG), evidence blocks
assets/css/style.css       one light theme, system fonts
assets/js/charts.js        draws every chart from data/figures.js (bar, line, scatter; tooltip; data table)
assets/img/                the ten qualitative figures, rendered by build/qual (see evidence/images.json)
data/figures.json          every figure the modules return (69), with its module and source files
data/figures.js            the 31 figures the page shows, as a script
data/sources/              snapshots of the result tables the figures read (so they can be rebuilt here)
data/qual/                 metadata of each qualitative figure: selection rule and its evidence, numbers, caveats
evidence/*.json            verified evidence entries, one file per source report
evidence_manifest.json     claim -> report -> result files -> figure / table source (generated)
evidence_index.md          the same mapping as a table (generated)
content_outline.md         the argument: main and sub evidence under each finding
build/build_figures.py     result tables -> data/figures.{json,js}
build/build_manifest.py    index.html + evidence/ + figures.json -> evidence_manifest.json, evidence_index.md
build/figs/*.py            one module per source report: reads the tables, returns chart data
build/qual/*.py            one script per qualitative figure (3-D renders; needs the result tree, CPU only)
build/build_qualitative.py registers those figures: data/qual/, evidence/images.json, evidence/zq_qualitative.json
```

## Where each evidence block comes from

Every evidence block on the page carries a code (`data-evidence="F1-A"` in `index.html`) and a small
source label. `evidence_index.md` lists, per block, the reports, the number of verified claims and
numbers with the evidence file that holds them, and every figure with the module that builds it and
the result files it reads. In short:

| block(s) | finding | source report (relative to the result root) | figure module |
|---|---|---|---|
| `MOTIVATION` | Motivation | `taco/30_bimart_gen3_scene_scale/`, `arctic/10_bimart_upstream_analysis/` | `build/figs/m0_motivation.py` |
| `F1-A` | 1 · What varies? | `taco/40_representation_study/time_decomp/dynamic_contact/`, `arctic/30_dynamic_contact/`, `oakink2/10_dynamic_contact/`, `reports/dynamic_contact_crossdataset.md` | `build/figs/f1a_dynamic_contact.py` |
| `F1-B` | 1 | `reports/hier_contact_gen_report.md` | `build/figs/f1b_hier_contact_gen.py` |
| `F1-C`, `F1-subA` | 1 | `reports/temporal_contact_events/` | `build/figs/f1c_temporal_events.py` |
| `F1-subB` | 1 | `reports/hand_contact_predictive_info/` | `build/figs/f1s_hand_information.py` |
| `F1-D` | 1 | `reports/wrench_counterfactual/` | `build/figs/f1d_wrench.py` |
| `F2-A`, `F2-B`, `F2-C`, `F2-subA`, `F2-subB` | 2 · What matters? | `reports/structure_variance_boundary/` | `build/figs/f2_structure_variance.py` |
| `F3-A` | 3 · How to model it? | `reports/structure_aware_temporal_generation/` | `build/figs/f3a_structure_aware.py` |
| `F3-B` | 3 | `reports/contact_factorization_stage1/` | `build/figs/f3b_stage1_factorization.py` |
| `F3-C`, `F3-diagA` | 3 | `reports/contact_latent_temporal_stage2/` | `build/figs/f3c_stage2_latent_temporal.py` |
| `F3-diagB`, `F3-diagC` | 3 | `reports/z_temporal_diagnostic/` | `build/figs/f3d_z_temporal_diagnostic.py` |
| `F3-subA`, `F3-subB` | 3 (supporting evidence) | `reports/z_joint_decoder_followup/` | `build/figs/f3s_joint_decoder.py` |
| `F1-PRIMER`, `F2-PRIMER` | qualitative primers (what a contact map is; the 48 numbers on a grasp) | recorded data; `reports/structure_variance_boundary/` feature cache | `build/qual/q_contact_map_primer.py`, `build/qual/q_structure_on_a_grasp.py` |
| `HYP` | Current hypothesis (in progress) | `scripts/research/z_stateful_s0_factorial/` (design only) | none: no result exists |

The result root is the server's result tree (`/result/uhnam/dexcore`, or `$DEXCORE_RESULT_ROOT`). The
reports themselves are not in this repository; the scripts that produced them are under
`scripts/research/`.

## How the figures are generated

No plotted value is typed by hand. Each module in `build/figs/` declares the result files it reads
(`SOURCES`) and returns plain chart data read from them at run time:

```bash
python docs/build/build_figures.py                    # rebuild data/figures.json and data/figures.js
python docs/build/build_figures.py --only f1d_wrench  # print one module's figures, write nothing
python docs/build/build_manifest.py                   # rebuild evidence_manifest.json and evidence_index.md
```

`build_figures.py` checks every figure against a small contract (required fields, a non-empty note,
one panel per dataset, every source declared with a locator and actually read, a direction for the y
axis, finite values of the right length) and fails, before writing anything, when the page references
a figure that no module returns. That the note says what the error bars are is a convention of the
modules, not a check. When the result tree is present, each source table of at most 3 MB is copied to
`data/sources/`; without the result tree the snapshots are read instead, so all 69 figures can be
rebuilt from this repository alone. Two sources are larger than the limit
(`reports/temporal_contact_events/{taco,arctic}/frames_test.csv`, read by the timeline figure
`f1c_timeline`); their snapshots hold the header and the rows of the one plotted sequence only.

`charts.js` draws what the data holds and nothing else: it rounds for display, never rescales, pools or
reorders values, keeps TACO, ARCTIC and OakInk2 in separate panels, and gives every chart a "Data
table" with the values, the module's note and the source files. The four series colours
(`#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`, in that fixed order) pass a colour-vision-deficiency
separation check on the page's light surface; two of them are below 3:1 contrast, which is why every
chart also carries labels or a table.

## How the evidence was checked

For each source report one pass extracted the claims and their numbers from the result tables, and a
second, independent pass re-read every number in its source file, re-ran the figure module and compared
each plotted value with the table. `evidence/*.json` holds the entries after that check, with the
corrections applied and listed (`verification`); `evidence_manifest.json` joins them with the page.
Where the report prose and a table disagree, the table value is used and the disagreement is recorded
in the entry's caveats. The finished page was then audited twice, section by section, against the
source tables (every printed number re-read from its table; wording checked against the entries'
`do_not_claim` lists); `evidence/zz_supplement.json` holds the numbers that those audits added to the
page.

Rules the page follows:

- every number comes from an existing result table or report. No experiment was re-run and nothing was
  trained for the page; the qualitative figures run the existing trained models on a few examples
  (inference only, on CPU) and say so;
- TACO, ARCTIC and OakInk2 are never pooled, and a result that exists for one dataset is not stated
  for another;
- a null stays a null; best-of-K and "teacher latent" numbers are labelled as oracles;
- wrench capability is a modelled proxy for functional grasp capability, not intent;
  no null-space claim is made;
- `z` is described as structure-oriented (a preferential factorization), never as disentangled, and
  `r` as a realization residual, never as noise;
- the stateful-z experiment is shown as a hypothesis in progress, with no result.

## Images

The ten qualitative figures (`assets/img/qual_*.webp`) are 3-D renders made for the page by the
scripts in `build/qual/` (see its README): the real object mesh, the recorded MANO hand and the
contact on the object's surface. Seven use recorded data and saved model outputs only. Three run a
trained model on CPU for the drawn examples: the BimArt port's contact and motion stages (Motivation),
the Stage-1 model for a latent swap, and the z-mediated generator's decoder fed the latents of the
true frames. No model was trained.

Rules they follow: each example is chosen by a stated rule that the script computes (for example the
test sequence at the median error), never by appearance; the caption names the rule and says what
the example is not typical of; every number in a caption is in `data/qual/<id>.json` with its source;
each figure was checked by an independent pass (selection recomputed, drawn arrays compared with
their sources, numbers re-read). They are illustrations, not statistics: the results rest on the
tables. Unlike the charts, they cannot be rebuilt from this repository alone (they need the result
tree, the checkpoints and the raw datasets); `build/build_qualitative.py` re-registers them from the
metadata snapshots.

```bash
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \
    PYVISTA_OFF_SCREEN=true python docs/build/qual/q_<id>.py     # render one figure
python docs/build/build_qualitative.py && python docs/build/build_manifest.py
```

The renders show objects of TACO and ARCTIC and the MANO hand model; only images are published, no
mesh, pose or model parameters.
