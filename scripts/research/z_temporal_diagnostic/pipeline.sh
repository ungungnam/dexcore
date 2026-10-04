#!/bin/bash
# Exact reproduction of the z temporal diagnostic (one seed). GPU id is an example; one dataset per python process.   bash pipeline.sh
# The prose templates read by zt_report.py / zt_dashboard.py live in OUT/narrative (a copy is kept in ./narrative_templates).
# Every stage is its own command, so `set -e` stops the pipeline at the first failure (an `a && b` list would not).
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
