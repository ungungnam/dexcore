# dexcore

Cross-object functional transfer of human hand-object demonstrations, and a study of what a temporal
hand-object contact generator should model.

**Project page: <https://ungungnam.github.io/dexcore/>** — the research argument (motivation, three
findings, current hypothesis) with the evidence behind each claim. Its source is in [`docs/`](docs/).

## What this repository contains

| part | where | what |
|---|---|---|
| Transfer pipeline | `src/`, `scripts/*.py`, `configs/`, `eval/`, `tests/` | selection → transfer → reconstruction of hand-object demonstrations, the BimArt end-to-end baseline, metrics and a viewer. Documented in the rest of this README |
| Dataset analysis and BimArt ports | `src/analysis/`, `scripts/*bimart*`, `configs/bimart_*/` | TACO / ARCTIC loaders and statistics; BimArt trained on TACO and on OakInk2 |
| Contact-generation research | [`scripts/research/`](scripts/research/README.md) | the experiment scripts behind the project page, one directory per study, catalogued in `scripts/research/README.md` |
| Project page | [`docs/`](docs/README.md) | the page, its figure-building scripts and the evidence manifest |

Things to know before reading the research scripts:

- They are **mirrors** of scripts that live beside their outputs on the development server, and they
  carry that server's absolute paths (`/result/uhnam/dexcore/...`). They are the record of how each
  number was produced; they do not run unchanged elsewhere.
- Datasets (TACO, ARCTIC, OakInk2), checkpoints, generated demonstrations and result files are not in
  this repository. The small result tables that the project page plots are snapshotted under
  `docs/data/sources/`.
- `scripts/research/z_stateful_s0_factorial/` belongs to an experiment that is still running; it is
  a snapshot of work in progress and has no result.

### Third-party code and attribution

`third_party/` is not tracked. The checkouts used on the development server:

| checkout | upstream | commit | used for |
|---|---|---|---|
| `BimArt` | <https://github.com/RosettaWYzhang/BimArt> | `bec093cd` | model code and pretrained checkpoints of the baseline; the ports in `src/analysis/bimart/` and `scripts/train_bimart_*.py` re-implement its preprocessing and training loop around the unmodified model code |
| `dexmachina` | <https://github.com/MandiZhao/dexmachina> | `adae5bf6` | demo format, object assets, MANO hands, ADD-AUC |
| `CorDex-Grasp` | <https://github.com/hxy-123/CorDex-Grasp> | `b19ea7ff` | reference only (see `src/transfer/cordex.py`) |
| `video_to_data` | <https://github.com/nvidia-isaac/video_to_data> | `5654c50e` | CHORD adapter (stub) |
| `gears`, `SSCF`, `CAMS` | <https://github.com/kzhou23/gears>, <https://github.com/Daisy-1227/SSCF>, <https://github.com/cams-hoi/CAMS> | `bef39220`, `49aec64a`, `7de39619` | read for reference; not imported by the code |

BimArt's code is released under CC BY-NC 4.0 and DexMachina's under MIT. Treat the BimArt ports in this
repository as subject to BimArt's non-commercial terms. three.js (MIT) is vendored under
`eval/visualization/static/vendor/` with its licence. This repository does not yet carry a licence of
its own.

## The transfer pipeline

A dense demonstration is factorized into a **dense object-state trajectory** and a **sparse set of
hand-object interaction states**:

```
D  ->  ( tau_O^{1:T} ,  K_HO )
```

and cross-object transfer is three independently replaceable stages:

```
D_s  --S-->  K_s  --Phi-->  K_t  --I-->  D_hat_t
```

| stage | question | module |
|---|---|---|
| **S** Selection | which frames of the demonstration need to be retained? | `src/selection/` |
| **Phi** Transfer | how is a sparse interaction state grounded onto a target geometry? | `src/transfer/` |
| **I** Reconstruction | how is dense hand motion recovered between transferred states? | `src/reconstruction/` |

`tau_O` is never thinned. Selection removes hand states; the object trajectory stays dense and, for
a different target, is produced by the canonical-frame object transfer.

## Status

| component | state |
|---|---|
| `AllSelector`, `UniformSelector`, `ReconstructionAwareSelector` | working |
| `IdentityTransfer` | working |
| `PalmFirstFingerTransfer` | **not implemented** — dexcore's own method, to be written in `src/` |
| `CorDexTransfer`, `ShapeGenTransfer` | working — thin wrappers over the exp3 reference implementations |
| `BimArtSynthesizer` | working — an END-TO-END baseline: `tau_O` in, a demonstration out, no selection/transfer/reconstruction |
| `LinearKeypointReconstructor` (object-relative interpolation + 51-DoF MANO IK) | working |
| penetration / collision / reconstruction / smoothness metrics | working |
| cross-object `tau_O` (canonical-frame transfer) | working — native, `src/data/target.py` |
| DexMachina export + ADD-AUC readback | working |
| CHORD adapter | stub (repository not available locally) |
| `src/analysis` — TACO + ARCTIC loaders, action-conditioned features, distribution report | working — see [Dataset analysis](#dataset-analysis) |
| MANO joint positions in `src/analysis` | **not available** — `manopth`/`smplx` cannot load the MANO pickles in the `dexmachina` env (chumpy vs numpy>=1.24). The analysis uses the hand ROOT and the 45-dim finger pose, which both datasets store directly |

Milestone 1 — `demo -> Uniform(10%) -> IdentityTransfer -> object-relative interpolation -> MANO IK
-> metrics` — runs end to end. Experiment 2 (cross-object) needs the two items marked above.

## Setup

dexcore is self-contained Python. It reads **assets** through a search path and drives **external
code** through checkouts under `third_party/` — neither is ever copied into the repository.

```bash
conda activate dexmachina                 # numpy, torch, trimesh, scipy, pyyaml, matplotlib
python -m unittest discover -s tests -v   # 47 tests
```

### Asset search path

An ordered list of roots, checked in order (`src/paths.py`):

1. `<repo>/assets` — objects you bake yourself, and demos dexcore generates
2. `third_party/dexmachina/dexmachina/assets` — the upstream ARCTIC objects, demos and MANO hands

A locally baked object therefore **shadows** an upstream one of the same name, which is what you
want while iterating on a target; `paths.object_root_of()` reports which root supplied an object so
the shadowing is visible rather than silent. Generated demos are installed into the repo-local root,
never into the `third_party` checkout — a modified checkout is one nobody can reason about.

Override with `$DEXCORE_ASSET_PATH` (colon-separated, highest priority first). To use targets baked
in a DexMachina working tree without copying them:

```bash
export DEXCORE_ASSET_PATH=$PWD/assets:$HOME/workspace/dexmachina/dexmachina/assets:$PWD/third_party/dexmachina/dexmachina/assets
```

### Third-party code

| checkout | used for |
|---|---|
| `third_party/dexmachina` | upstream: demo format, object assets, MANO hands, ADD-AUC (`eval/compute_add.py`) |
| `third_party/CorDex-Grasp` | the CorDex **paper** repo — a learned robot-hand grasp synthesiser. Not used as a transfer operator; see `src/transfer/cordex.py` on why it cannot be one |
| `third_party/video_to_data/robotic_grounding` | CHORD |

The `cordex` and `shapegen` operators are the **exp3 reference implementations** of `func_warp` and
`geom_warp`, which live in an unpublished DexMachina working tree (the upstream checkout has no
`eval/experiments/`). Point `$DEXCORE_DEXMACHINA_SRC` at that tree; each operator locates it, calls
into it by path, and reports clearly when it is absent. Nothing is vendored.

CPU is the default for IK. The GPUs on this box are shared and routinely saturated, where creating a
CUDA context blocks for minutes; a full 889-frame two-hand IK solve takes ~14 s on CPU. Ask for GPU
explicitly with `CUDA_VISIBLE_DEVICES=<free device>` and `reconstructor_options.ik_device=cuda`.

## Use

```bash
# synthesize
python scripts/synthesize.py --sequence box_use_01 --selector uniform --ratio 0.1 \
    --transfer identity --reconstructor linear_keypoint

# evaluate a SAVED demo, independently of how it was made
python scripts/evaluate.py -i outputs/<run>/box_use_01.npy \
    --metrics penetration collision --reference-sequence box_use_01

# stages in isolation
python scripts/select.py      --sequence box_use_01 --selector reconstruction_aware --ratio 0.05
python scripts/transfer.py    --sparse outputs/<run>/selected_keyframes.npy \
                              --source-sequence box_use_01 --target-object box_s110
python scripts/reconstruct.py --sparse outputs/<run>/selected_keyframes.npy \
                              --source-sequence box_use_01

# Experiment 1 sweep
python scripts/experiment1.py --sequence box_use_01 --budgets 0.25 0.1 0.05 0.01
```

Config files merge left to right and `--set` overrides everything:

```bash
python scripts/synthesize.py -c configs/experiment1_redundancy.yaml \
    --set selector_options.ratio=0.05 --set reconstructor_options.ik_iters=600
```

## Data representation

dexcore reads and writes **DexMachina's own format**, verified bit-exact on round trip. Exporting a
generated demonstration for training is a file copy, not a conversion.

```
params["obj_trans"]                  (F,3)     object ROOT position, m, world
params["obj_quat"]                   (F,4)     root orientation, wxyz, sign-unrolled on load
params["obj_arti"]                   (F,)      the single revolute joint, rad
world_coord["joints.{side}"]         (F,21,3)  MANO keypoints, world
world_coord["contact_links_{side}"]  (F,16,4)  per LINK [x,y,z,part_id]; ZERO ROW = no contact
world_coord["contacts.{side}"]       (F,50,4)  contact points on the object + part id
configs_{side}                       {name:(F,)}  51-DoF MANO URDF pose
```

Part ids: `1 = top/lid/moving`, `2 = bottom/base`. Sides are always `("left","right")`. 60 fps.

### Three facts that are easy to get wrong

1. **A zero contact row means "no contact", not "no data".** `contact_links` is dense — all 16 rows
   exist every frame. Contact is `norm(xyz) > 0`.
2. **`obj_quat` sign-flips freely** across the double cover. It is unrolled on load; any difference
   or interpolation before that is garbage.
3. **The 21 keypoints are not the URDF link origins.** The demo's keypoints come from ARCTIC's MANO
   fit and the URDF is a generic hand; they sit ~13 mm apart. Everything that moves a keypoint
   carries it at its own fixed offset in its owner link's frame (`src/data/mano.py`).

## What came from DexMachina, and how

The demo representation and the MANO/IK machinery were **refactored into dexcore** — the numbers
must match DexMachina's or an exported demo is silently corrupt:

| dexcore | source | notes |
|---|---|---|
| `src/data/demo.py` | exp4 `keyframe_selection/reference.py` | loader + contact/quaternion decoding, extended with provenance and a lossless writer |
| `src/data/mano.py` | exp3 `mano_fk.py` | batched differentiable URDF FK; rebuilt to avoid in-place autograd nodes (~100× faster backward) |
| `src/reconstruction/ik.py` | exp3 `hand_ik.py` | contact-target IK reworked into keypoint-target IK; objective rescaled (see below) |
| `src/data/object.py`, `src/data/target.py` | exp6 `object_model.py`, `chain_map.py` | canonical frame, articulation phase, cross-object `tau_O` — rewritten natively |
| `eval/downstream/dexmachina.py` | `eval/compute_add.py`, exp3 `install_demos.py` | export conventions + ADD-AUC readback |

The two **baselines** are not refactored and not copied: `src/transfer/{cordex,shapegen}.py` drive
the exp3 reference implementations in place, so their numbers come from the same code that produced
them elsewhere. Each file is self-contained — locating the tree, the calling convention it needs
(including the repo-root-relative cwd it was written for), and the dexcore contract are all in the
one file, so a baseline reads top to bottom.

## Design decisions worth knowing

**A scaled object is a different object.** `box_s110` is a separately baked asset, not "the box,
bigger". `IdentityTransfer` refuses any target whose name differs, and there is deliberately no
name-prefix rule that would collapse `box_s110` back into `box`. Doing nothing to a different
geometry is a no-op baseline, and has to be asked for by name
(`transfer_options.allow_different_object=true`).

**Interpolation happens in the object's frame.** A hand holding a moving object is nearly static in
the object frame and travels a long arc in the world frame. For a keypoint fixed on the object,
object-relative interpolation is exact to 1e-17 where world-frame interpolation drifts 3.7 mm over
the same span. This is why `tau_O` has to stay dense. Note the frame used is the object **root**
(base part), so a keypoint riding the lid is not static in it — exact at keyframes, first-order
between them, and visible for a lid grasp across a large articulation change.

**The IK objective is dimensionless.** In square metres the keypoint term is ~1e-18 at ground truth
while the smoothness term is ~4e-4; with comparable weights IK optimises smoothness and ignores the
keypoints entirely. The keypoint term is divided by `keypoint_scale_m**2` so it reads as "error in
units of 1 mm" and every weight is comparable.

**Smoothness penalises deviation from the warm start's velocity, not absolute velocity.** That makes
the term exactly zero at `q = q_init`, so `All -> Identity -> Reconstruct` reproduces the source
**exactly** (measured: max 0.057 mm over 889 frames) instead of being dragged off it. It also
targets the right thing — jitter IK *introduced*, not motion the human performed.

**Failures are reported, never repaired.** A keyframe that cannot be grounded is recorded in
`TransferDiagnostics` with a reason; unconverged IK frames are flagged; cross-object stages raise
with the exact modules to vendor. A plausible stand-in would run end to end and answer a different
question — DexMachina measured exactly that, where a translation-only fit preserved *less* contact
than not fitting at all (74.9% vs 100.3%).

## Metrics, and their baselines

Compare against the **source demonstration's own numbers**, not against zero. The generic MANO URDF
is not the ARCTIC MANO fit, and the object collision meshes are decimated, so a real human demo
scores non-zero on both geometric metrics:

| metric | on the source demo (`box_use_01`) | meaning |
|---|---|---|
| hand-object penetration | up to ~5 mm | the floor; only excess over this is attributable to a reconstruction |
| self-collision (min inter-link gap) | 0.5–0.7 mm, 1.5%/6.5% of frames under 1 mm | ditto |
| inter-hand gap | ~257 mm | the hands genuinely do not touch in this clip |

Recomputed contacts also **do not reproduce** `process_arctic.py`'s contact set — they agree on ~78%
of link-frames and flag ~1.6× more (IoU 0.70 left / 0.59 right). That one starts from ARCTIC's
object-side annotation; this one is purely geometric ("any link surface within 1 cm"). Geometric
re-derivation is nonetheless the only defensible default, because a different target object has no
annotation to carry and may not admit the source's contacts at all.

`reconstruction` error is **same-object only** and refuses a reference with a different object —
across a geometry change there is no ground-truth hand trajectory, which is exactly why the
cross-object experiment leans on downstream ADD-AUC instead.

## Viewer

A local web viewer for generated demonstrations: pick or upload a `.npy`, scrub a time slider, and
see the per-frame 3D hand-object scene.

```bash
python eval/visualization/serve.py          # http://127.0.0.1:8000
# over ssh:  ssh -L 8000:127.0.0.1:8000 <host>
```

Upload works on the files that actually exist, because a processed demo is a **pickled numpy dict**
that no browser can read — the server is the numpy process that converts it
(`eval/visualization/scene.py`). It binds to localhost and will only read from `outputs/` and the
DexMachina processed-demo tree; any other path is refused.

**The browser does no kinematics.** The server sends, per frame, the already-composed world
transform of each object part and each of the 32 hand links. There is no hinge, no FK, no
articulation angle and no `wxyz` quaternion in the viewer, so a convention bug cannot originate
there. The one crossing point — dexcore's `wxyz` to three.js's `xyzw` — happens once, in `scene.py`.
Verified against an independent recomputation: object part poses and hand link poses agree to
~1e-7 (float32), keypoints and contact masks bit-exact.

Shows: object parts (lid articulated), 16 posed hand link meshes per side, 21 keypoints, the MANO
skeleton, per-link contact highlighting, and an optional fingertip trail. **Selected keyframes are
drawn as ticks on the timeline** and flagged in the HUD, which is the fastest way to see whether a
selector put its budget where the motion is. Space toggles play, arrow keys step one frame.

**Overlay** loads a second demonstration ghosted on top of the first — source against
reconstruction, or a reconstruction against its own sparse keyframe set. The HUD then reports the
live per-frame keypoint divergence (`Δ mean / max`, mm), which is the number the comparison exists
to show. The two are aligned by **absolute source frame, never by row**: a 200-frame dense
reconstruction and its 20-frame keyframe set are both valid overlays for each other, and row 5 of
one is not row 5 of the other. Frames the overlay does not cover say so rather than showing a stale
pose.

**The frame window is in absolute source frames**, the same numbers as a DexMachina clip string
(`box-30-230-s01-u01`) — not indices into the particular file. Asking for 30–230 means the same
instants whether the file is the 889-frame source or a 200-frame demo cut from it, and on a demo
that already *is* that window it is a no-op. This works because a demo dexcore writes carries a
`dexcore_demo` block recording the source frames it covers, so a generated clip does not forget it
came from frames 30–230, and a sparse `selected_keyframes.npy` still knows which object it is for.

three.js is vendored under `static/vendor/` rather than loaded from a CDN — a viewer that breaks
when the box loses network is not usable as a research tool.

## Adding a variant

Register it; nothing in `src/pipeline.py` changes.

```python
from src.selection.base import Selector, register

@register
class MySelector(Selector):
    name = "my_selector"
    def frame_indices(self, demo):
        return np.array([...])          # provenance, budget and endpoints are handled by the base
```

Same for `Transfer` (`src/transfer/base.py`) and `Reconstructor` (`src/reconstruction/base.py`).

## Adding a whole METHOD

The three stages are dexcore's factorization, not a law. A method that produces a demonstration
some other way — an end-to-end generative model, say — has no selector to register, so it is
registered one level up, as a `Synthesizer`:

```python
from src.synthesis.base import Synthesizer, SynthesisResult, register

@register
class MySynthesizer(Synthesizer):
    name = "my_method"
    consumes = frozenset({"source_clip"})     # NOT "source_hand"

    def synthesize(self, inp) -> SynthesisResult:
        ...                                    # inp.trajectory, inp.target_object
```

`consumes` is **enforced, not documented**: reading an input the method did not declare raises.
The difference between the two families is exactly which inputs they consume — "this model is
conditioned on the object trajectory alone" is the claim a comparison rests on, and a claim that
lives only in a comment is one a later edit breaks silently. `hand_states_used` reports how many
source hand states the method actually consumed (the selection budget for `staged`, 0 for an
end-to-end model, T for a dense per-frame transfer), which is what lets a method with no budget
be plotted against one that has one.

`src/pipeline.py` is the driver for both: it loads the source, resolves the objects, builds the
dense `tau_O`, and hands the same input to whichever synthesizer the config names.

```bash
python scripts/synthesize.py --sequence box_use_01 --synthesizer staged --ratio 0.1
```

### The end-to-end baseline

`bimart` generates the hand motion with BimArt (CVPR'25), a learned diffusion prior over bimanual
articulated-object interaction, and then fits dexcore's own URDF hand to it and recomputes contacts
with the same code the staged pipeline uses — so the only difference between the two rows of a
table is what produced the hand motion.

```bash
export DEXCORE_ASSET_PATH=$PWD/assets:$HOME/workspace/dexmachina/dexmachina/assets:$PWD/third_party/dexmachina/dexmachina/assets
CUDA_VISIBLE_DEVICES=<free> python scripts/synthesize.py --sequence box_use_01 \
    --frame-start 30 --frame-end 230 --target-object pm100141_calibrated --synthesizer bimart
```

It reaches the reference through `$DEXCORE_DEXMACHINA_SRC` like the two transfer baselines, and it
is handed a file containing **only the object trajectory** — the human hand states are absent from
its input, not merely unread, which is what makes `hand_states_used = 0` a fact rather than a
claim. Two things about it are worth knowing before reading its numbers:

* **It generates in 64-frame windows.** That is the model's designed horizon; a longer clip is
  covered by consecutive stride-64 windows stitched in world space, which is the published model
  used outside its designed range. The window count is recorded in the diagnostics.
* **It has no per-instance object size.** BimArt's conditioning is a unit-normalised BPS plus a
  per-category scale constant, so it generates for the ARCTIC-native object size. The residual is
  absorbed by the IK and by the contact recompute against the true target geometry.

**Measured on `pm100141_calibrated` (frames 30-230), against `cordex` and `shapegen` on the same
`tau_O`.** Format is identical across all three — same keys, shapes and dtypes, 51-DoF configs —
and `tau_O` agrees to 0.00000 mm. The hand does not:

| | cordex | shapegen | bimart |
|---|---|---|---|
| penetration max (mm) | 6.89 | 11.89 | 30.41 |
| inter-hand min gap (mm) | 294.9 | 304.1 | **0.17** |
| keypoint accel, left (m/s²) | 3.96 | 3.66 | 13.41 |
| joint-limit violation (rad) | 0.00 | 0.00 | 0.37 |

The two hands overlap: 10.9 cm wrist-to-wrist against 45.3 cm in the source demonstration, with
their nearest keypoints 1.1 cm apart on 192 of 200 frames. It is not a scale error (the generated
hands measure 15.4 cm wrist-to-fingertip against the source's 14.1 cm) and not the IK (residual
3.2 mm) — the generated placement itself puts both hands on one side of the object.

**These numbers are out-of-distribution and must be labelled as such.** `pm100141_calibrated` is a
PartNet-Mobility instance BimArt never saw; the run reports `out_of_distribution` in its own
diagnostics. Attributing the collapse to the method rather than to the geometry needs an
in-distribution control, and there the reference constrains what is possible:

* `exp3_demo_gen_baselines/paths.py` hardcodes `SRC_PROCESSED`/`SRC_OBJ_DIR` to the ARCTIC box —
  "the trajectory every target is transferred FROM". Every target is reachable, but the SOURCE is
  always `box_use_01`, subject `s01`.
* `box_use_01_s01` is in BimArt's **training** split (its test split has `box_use_01_s10`), so a
  box-to-box control is contaminated and its numbers flatter the model.
* The one sequence that is both in BimArt's test split and in dexcore's asset tree is
  `notebook_use_01_s01`. Reaching it needs the reference's trajectory builder to accept a source
  object other than the box.

## Dataset analysis

`src/analysis/` is a separate lane from the pipeline: it reads the SOURCE datasets and measures
them. Nothing in `src/synthesis`, `src/selection`, `src/transfer` or `src/reconstruction` imports
it, and it imports none of them.

| dataset | where | what it labels | sequences |
|---|---|---|---|
| **TACO** | `$DEXCORE_TACO_ROOT`, default `/backups/uhnam/TACO` | a `(verb, tool, target)` triplet, 15 verbs | 2317 |
| **ARCTIC** | `$DEXCORE_ARCTIC_ROOT`, default `/data/uhnam/ARCTIC/data` | `grab` / `use` + the object, 11 objects | 301 |

Both load into ONE type, `analysis.schema.HOITrajectory`: a list of object tracks (each optionally
articulated) plus per-side hand tracks, positions in **metres**, rotations as **wxyz** quaternions,
articulation in **radians**. The loaders normalise on the way in — ARCTIC's millimetre object
translation and TACO's `(T,4,4)` matrices never reach a metric. So every feature is written once.

```bash
conda activate dexmachina                 # torch is needed to unpickle TACO's hand annotations

python scripts/analyze_actions.py --dataset taco --workers 16
python scripts/analyze_actions.py --dataset taco --verbs brush dust smear     # one question
python scripts/analyze_actions.py --dataset arctic --group tool_name          # 2 verbs only
```

Results go to `$DEXCORE_RESULT_ROOT/analysis/<dataset>/<run>` (default `/result/uhnam/dexcore`),
one timestamped run directory per invocation: `features.csv` (one row per sequence),
`group_summary.csv`, `discriminability.csv`, `confounds.csv`, `summary.json`, `plots/`.

**Two things to know before reading the output.**

*The ranking is Kruskal-Wallis epsilon-squared, not a classifier.* The question is which physical
quantities the action label is informative about. With 2317 sequences every p-value is tiny; the
effect size is what says whether a feature matters.

*TACO confounds the verb with the tool by construction* — a `brush` is almost always done with a
brush — so `confounds.csv` scores every feature under BOTH groupings. Over the full dataset the
hand-to-tool distance features top the by-verb ranking (eps^2 0.42-0.43) but score HIGHER by tool
(0.47-0.51): they are reading tool geometry, because a feature measured against the tool's mesh
ORIGIN puts a knife's handle a knife-length from the hand. The features that track the action more
than the tool are the tool's angular speed, the tool-target distance range, the tilt range and the
autocorrelation period (ratios 1.19-1.40). Only the verbs that share a tool can separate the two
properly, which is what `--verbs` is for.

## Layout

```
configs/       selection/ transfer/ reconstruction/ eval/ + experiment configs
src/
  data/        demo.py object.py mano.py hand.py trajectory.py target.py contacts.py
  selection/   base.py all.py uniform.py reconstruction_aware.py
  transfer/    base.py identity.py palm_first_finger.py
  reconstruction/ base.py interpolate.py ik.py linear_keypoint.py
  synthesis/   base.py staged.py bimart.py  a whole METHOD: tau_O -> a target demo
  metrics/     penetration.py collision.py reconstruction.py
  pipeline.py  config.py geometry.py paths.py viz.py cli.py
eval/          run_eval.py                     the two evaluation levels, dispatched
  cv/          report.py                       A: metrics on a saved demonstration
  downstream/  dexmachina.py chord.py          B: external frameworks (ADD-AUC)
  visualization/ serve.py scene.py static/     side-by-side viewer (three.js vendored)
  analysis/    schema.py features.py distribution.py report.py sweep.py loaders/{taco,arctic}.py
scripts/       synthesize.py evaluate.py select.py transfer.py reconstruct.py experiment1.py
               analyze_actions.py             action-conditioned analysis of a source dataset
tests/         test_dexcore.py test_synthesis.py
               test_analysis.py               synthetic; passes with no dataset mounted
               test_analysis_datasets.py      the real TACO/ARCTIC files; skips when absent
```
