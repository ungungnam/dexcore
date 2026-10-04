#!/bin/bash
# Exact reproduction of the 2 x 2 factorial study (one seed).  One dataset per python process; every GPU command is memory-capped.     bash pipeline.sh
set -eu; cd "$(dirname "$0")"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
O=/result/uhnam/dexcore/reports/z_stateful_s0_factorial; L=$O/logs; mkdir -p $L $O/narrative
# 0. prerequisites (not redone here): the Stage-2 study (teacher cache, D0 = B0 files, GT / persistence metrics) and the joint-decoder follow-up (M00 = its M2 files).
# 1. the one design decision made before training (validation data only): supervision of the initial-state module  -> $O/init_aug_decision.json
#    (four 2 000-step pilots; the decision file records the readings and the rule)
for ds in taco arctic; do for a in 0 3; do CUDA_VISIBLE_DEVICES=4 $PY zf_pilot_init.py --dataset $ds --aug $a --steps 2000 > $L/pilot_${ds}_aug$a.log 2>&1; done; done
$PY zf_pilot_decide.py --step 2000
# 2. training: M01, M10, M11 on both datasets through the shared-GPU queue (GPUs 4 / 5 first; a stateful run gets a GPU to itself).
#    Single runs can be started directly:   CUDA_VISIBLE_DEVICES=4 $PY zf_train.py --dataset taco --model M11 --wandb --mem-cap-mib 34000 --no-dec-ckpt
printf 'taco M10\ntaco M11\narctic M10\narctic M11\ntaco M01\narctic M01\n' > $L/queue.txt
$PY zf_queue.py > $L/queue_stdout.log 2>&1
# 3. per dataset: trajectories (test + validation), metrics with the previous studies' code, the partial D0r row, diagnostics, sanity checks
bash zf_post.sh 4 taco
bash zf_post.sh 5 arctic
# 4. tables, figures, report, dashboard (CPU)
$PY zf_sanity.py --merge
$PY zf_aggregate.py > $L/aggregate.log 2>&1
CUDA_VISIBLE_DEVICES="" $PY zf_figures.py > $L/figures.log 2>&1
for f in narrative_templates/*; do [ -e "$O/narrative/$(basename "$f")" ] || cp "$f" "$O/narrative/"; done
$PY zf_report.py
$PY zf_dashboard.py
