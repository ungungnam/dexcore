Files in this folder: `report.md`, `report_values.json` (every number used in the prose), `experiment_config.json` (configuration, pre-registered decision rule, checkpoint paths and md5, code md5), `local_prediction_metrics.csv`, `local_prediction_splits.csv` (by window position / subject), `persistence_comparison.csv`, `temporal_geometry_metrics.csv`, `temporal_geometry_bins.csv`, `quadrant_metrics.csv`, `directional_consistency.csv`, `stage2_comparison.csv`, `probe_training.csv`, `decision_summary.csv`, `posthoc_splits.csv` / `trajectory_contribution.csv` / `quadrant_B_kinds.csv` (post-hoc, from `zt_posthoc.py`), `sanity_summary.json`, `figures/` (Figures 1–6 and the selection tables of Figure 5), `logs/`, `narrative/` (the prose templates read by `zt_report.py` / `zt_dashboard.py`; a copy is kept next to the scripts in `narrative_templates/`), `dashboard.html`, `scripts/` (mirror of the code with `MD5SUMS.txt`), and per dataset `<ds>/cache/frames.npz` (C_t, z_t, G, τ_t, teacher descriptors for train / val / test), `<ds>/preds/` (probe predictions), `<ds>/train_logs/`, `<ds>/local_arrays.npz`, `<ds>/geometry_arrays.npz`, `<ds>/stage2_curves.npz`. Probe checkpoints: `/ckpt/uhnam/dexcore/z_temporal_diagnostic/<ds>/<probe|probe_notau>_h<h>_seed0.pt`. The probe training curves are also logged to Weights & Biases (project `dexcore-z-temporal-diagnostic`, uploaded from the saved logs by `zt_wandb_upload.py`).

Reproduction: `bash scripts/research/z_temporal_diagnostic/pipeline.sh`. The script changes into its own directory, copies `narrative_templates/` into `<out>/narrative/` where a file is missing, and then runs the stages below one by one, stopping at the first failure (one dataset per python process; the GPU id is an example; logs go to `logs/`). Its content:

    set -eu; cd "$(dirname "$0")"
    PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
    O=/result/uhnam/dexcore/reports/z_temporal_diagnostic; L=$O/logs; mkdir -p $L $O/narrative
    for f in narrative_templates/*; do [ -e "$O/narrative/$(basename "$f")" ] || cp "$f" "$O/narrative/"; done      # prose templates, never overwritten
    for ds in taco arctic; do CUDA_VISIBLE_DEVICES=5 $PY zt_cache.py --dataset $ds > $L/cache_$ds.log 2>&1; done          # frozen Stage-1 A3 encoder -> frame cache
    bash run_probes.sh 5 taco & P1=$!; bash run_probes.sh 5 arctic & P2=$!; wait $P1; wait $P2                              # Experiment A: 6 probes per dataset -> $L/probe_<ds>.log (non-zero exit if a probe failed)
    for ds in taco arctic; do
      CUDA_VISIBLE_DEVICES=5 $PY zt_eval_local.py --dataset $ds > $L/eval_local_$ds.log 2>&1                                # Experiment A metrics (test pairs)
      $PY zt_geometry.py --dataset $ds > $L/geometry_$ds.log 2>&1                                                           # Experiment B
    done
    $PY zt_stage2.py > $L/stage2.log 2>&1                                                                                   # comparison with the saved Stage-2 B1 predictions
    $PY zt_aggregate.py > $L/aggregate.log 2>&1                                                                             # merges, post-hoc splits (zt_posthoc.py), decision rule, config
    CUDA_VISIBLE_DEVICES=5 $PY zt_sanity.py > $L/sanity.log 2>&1
    CUDA_VISIBLE_DEVICES=5 $PY zt_figures.py > $L/figures.log 2>&1
    $PY zt_report.py > $L/report.log 2>&1
    $PY zt_dashboard.py > $L/dashboard.log 2>&1
    $PY zt_wandb_upload.py > $L/wandb_upload.log 2>&1 || true                                                              # optional: training curves of the probes -> Weights & Biases
