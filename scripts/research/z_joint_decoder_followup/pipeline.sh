#!/bin/bash
# Exact reproduction of the z joint-decoder follow-up (one seed).  One dataset per python process.  GPU ids are examples: every GPU
# command is memory-capped and may share a GPU with other jobs.            bash pipeline.sh
set -eu; cd "$(dirname "$0")"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
O=/result/uhnam/dexcore/reports/z_joint_decoder_followup; L=$O/logs; mkdir -p $L $O/narrative
# 0. prerequisites (not redone here): the Stage-2 study provides the teacher cache (s2_cache_teacher.py), the reused checkpoints / prediction /
#    metric files of B0 (= M0) and B1 (= M1) and the GT / persistence reference metrics.
# 1. training: M2 (required), M3 (optional), M0r (budget-parity check) on both datasets through the shared-GPU queue.
#    The queue reads $L/queue.txt, starts a job when a GPU in 2..7 has room for its memory cap, retries with --resume after an OOM, and
#    exits when every listed job is done.  Single runs can be started directly:
#        CUDA_VISIBLE_DEVICES=5 $PY zj_train.py --dataset taco --model M2 --wandb --mem-cap-mib 9500
printf 'taco M2\narctic M2\ntaco M3\narctic M3\ntaco M0r\narctic M0r\n' > $L/queue.txt
$PY zj_queue.py > $L/queue_stdout.log 2>&1
# 2. per dataset: test / validation trajectories, metrics (the previous studies' evaluation code), re-check of the Stage-2 metric files,
#    z diagnostics + decoder-mismatch diagnostic, sanity checks
bash zj_post.sh 5 taco
bash zj_post.sh 5 arctic
# 3. held-out adaptation probe (post-hoc; run because its pre-registered trigger fired: see zj_common.DECISION["heldout_probe_trigger"])
for ds in taco arctic; do CUDA_VISIBLE_DEVICES=5 $PY zj_probe.py --dataset $ds --model M2 > $L/probe_$ds.log 2>&1; CUDA_VISIBLE_DEVICES=5 $PY zj_probe.py --dataset $ds --model M2 --shrink >> $L/probe_shrink.log 2>&1; done
# 4. tables, figures, report, dashboard (CPU)
$PY zj_sanity.py --merge
$PY zj_aggregate.py > $L/aggregate.log 2>&1
CUDA_VISIBLE_DEVICES="" $PY zj_figures.py > $L/figures.log 2>&1
for f in narrative_templates/*; do [ -e "$O/narrative/$(basename "$f")" ] || cp "$f" "$O/narrative/"; done
$PY zj_report.py
$PY zj_dashboard.py
