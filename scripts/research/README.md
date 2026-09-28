# Research scripts mirrored out of the result tree

These 79 files are a **mirror**. Each one also lives beside the artifacts it produced, under
`/result/uhnam/dexcore/`, and that is where it must be *run* from: several of them self-locate with
`Path(__file__).resolve().parent` or `sys.path.insert(0, ROOT / "scripts")`, and their `ROOT`
constants name absolute result paths. Moving the originals would break those bindings.

They are copied here because until this commit they existed **nowhere else**: the repository had no
commits, so "the results are regenerable because the code is in git" was false. If a script here and
its original ever disagree, the original under `/result` is the one that produced the numbers.

| directory | origin | what it does |
| --- | --- | --- |
| `canonical_contact/` | `canonical_contact/scripts/` | the category-canonical contact study: cache builders (`build_canonical_cache`, `build_frame_cache`), steps 4–11, the figure scripts, and the later time-vs-mesh decomposition (`time_vs_mesh`, `time_probe`, `phase_sensitivity`, `what_changes`, `four_axes`) |
| `canonical_contact_verify/` | `canonical_contact/verify/` | the four adversarial verification passes of that study (`v1`–`v5`, `step9_refute`, `function_separation/`) |
| `taco_gen1_probes/` | `bimart_taco/contact_probe/{verify,gen/verify}/` | hypothesis probes run against the original (defective-label) generation: G3 shift regressions, G5 memorisation sweeps, C4 blocks/regressions |
| `taco_gen3_reverify/` | `bimart_taco_scene/contact_probe/gen/reverify/` | the re-verification of every hypothesis on the scene-scale run (window aggregation, stage comparison, truncation, matched bowl pairs, retrieval) |
| `taco_oneoff_analysis/` | `bimart_taco_scene/contact_probe/gen/scripts_orig/` | one-off analyses from the original investigation, archived out of a scratch directory: novelty tagging, subject recovery, seed nulls, k-NN oracles, label-fix checks |

Run them with the repo on `PYTHONPATH` and `scripts/` **not** first on `sys.path`
(`scripts/select.py` shadows the stdlib `select` module):

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate dexmachina
    PYTHONPATH=/home/uhnam/workspace/dexcore python /result/uhnam/dexcore/<original path>/<script>.py

## Note added 2026-09-28 — the gen1/gen2 scripts are a record, not a runnable pipeline

`taco_gen1_probes/` and `taco_oneoff_analysis/` hard-code paths such as
`/result/uhnam/dexcore/bimart_taco/train_store/`. That material was deleted on 2026-09-28 (the
generations it belonged to were superseded; see `/result/uhnam/dexcore/INDEX.md`). The paths are
left **as they were** on purpose: rewriting them would make the record of how those numbers were
produced untrue. To re-run any of them, re-create the generation first
(`preprocess_taco_bimart.py`, then training).

`canonical_contact/` scripts that read the gen1 `sequences/` were repointed at
`taco/30_bimart_gen3_scene_scale`, because those scripts use only the dense contact arrays, which
are bit-identical between the two generations.
