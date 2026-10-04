#!/bin/bash
# After the 2026-10-03 reboot: waits for the two relaunched runs, then runs the full evaluation of both datasets and the tables.
set -u; cd "$(dirname "$0")"
L=/result/uhnam/dexcore/reports/contact_factorization_stage1/logs; S=$L/orchestrate.log; PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
say() { echo "$(date '+%H:%M:%S') $*" >> $S; }
say "orchestrator 2 started (after reboot)"
until grep -q "done A2_seed0:" $L/train_taco_A2_seed0.log 2>/dev/null && grep -q "done A2_z16_seed0:" $L/train_arctic_A2_z16_seed0.log 2>/dev/null; do sleep 120; done
say "both relaunched runs done; full evaluation"
bash run_eval.sh 4 taco >> $S 2>&1 &
bash run_eval.sh 3 arctic >> $S 2>&1 &
wait
say "evaluation done; tables / figures"
$PY aggregate.py > $L/aggregate.log 2>&1 && $PY sanity.py > $L/sanity.log 2>&1 && $PY figures.py > $L/figures.log 2>&1 && say "tables done" || say "TABLES FAILED"
say "ORCHESTRATION_DONE"
