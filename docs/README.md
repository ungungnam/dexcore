# Project page

**What Should a Temporal Contact Generator Actually Model?** — the research argument of this
repository on one page: motivation, three findings, and the current hypothesis.

Live: <https://ungungnam.github.io/dexcore/> (GitHub Pages serves this `docs/` folder from `main`).

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
assets/img/                two qualitative figures cropped from the reports (see evidence/images.json)
data/figures.json          every figure the modules return (69), with its module and source files
data/figures.js            the 31 figures the page shows, as a script
data/sources/              snapshots of the result tables the figures read (so they can be rebuilt here)
evidence/*.json            verified evidence entries, one file per source report
evidence_manifest.json     claim -> report -> result files -> figure / table source (generated)
evidence_index.md          the same mapping as a table (generated)
content_outline.md         the argument: main and sub evidence under each finding
build/build_figures.py     result tables -> data/figures.{json,js}
build/build_manifest.py    index.html + evidence/ + figures.json -> evidence_manifest.json, evidence_index.md
build/figs/*.py            one module per source report: reads the tables, returns chart data
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

- every number comes from an existing result table or report; nothing was re-run for the page;
- TACO, ARCTIC and OakInk2 are never pooled, and a result that exists for one dataset is not stated
  for another;
- a null stays a null; best-of-K and "teacher latent" numbers are labelled as oracles;
- wrench capability is a modelled proxy for functional grasp capability, not intent;
  no null-space claim is made;
- `z` is described as structure-oriented (a preferential factorization), never as disentangled, and
  `r` as a realization residual, never as noise;
- the stateful-z experiment is shown as a hypothesis in progress, with no result.

## Images

The two qualitative figures are crops of figures that the experiments already produced; nothing was
redrawn. `evidence/images.json` records the source file and the crop of each.
