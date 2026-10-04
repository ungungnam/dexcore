#!/bin/bash
# Exact reproduction of the Stage-2 experiment (one seed). GPU ids are examples; one dataset per python process.
#   bash pipeline.sh
set -eu; cd "$(dirname "$0")"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
# 1. teacher latents z*_t from the frozen Stage-1 A3 encoder (train / val / test)
for ds in taco arctic; do CUDA_VISIBLE_DEVICES=5 $PY s2_cache_teacher.py --dataset $ds; done
# 2. the six main runs (the launcher waits for free GPU memory, retries with --resume after an OOM; --wandb inside)
#    The reported runs were launched with launch.sh; arctic B1 / taco B2 / arctic B2 were stopped by hand at 18k / 14k / 12k steps
#    (user decision, best step 4000 in all of them) and written out with
#        python s2_finalize.py --dataset <ds> --model <B1|B2> --stopped-by manual_stop
#    A full reproduction lets them run to the 20k-step / 15-check rule.
bash launch_v2.sh 5 taco B2 & bash launch_v2.sh 5 taco B1 & bash launch_v2.sh 4 arctic B2 & bash launch_v2.sh 4 arctic B1 & bash launch_v2.sh 7 taco B0 & bash launch_v2.sh 6 arctic B0 & wait
# 3. fixed-s_0 test trajectories, metrics (previous study's evaluation code), references, tables, sanity checks, figures
bash orchestrate.sh 5 4            # (waits for the runs, then generate -> evaluate -> aggregate -> sanity -> figures)
# 4. bottleneck inspection (teacher-z* decoding floor, Table 11), final tables / figures / report / dashboard
for ds in taco arctic; do CUDA_VISIBLE_DEVICES=5 $PY s2_diag_bottleneck.py --dataset $ds; done; $PY s2_diag_bottleneck.py --merge
$PY s2_aggregate.py && CUDA_VISIBLE_DEVICES=5 $PY s2_figures.py && $PY s2_report.py && $PY s2_dashboard.py
