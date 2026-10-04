#!/bin/bash
# Survives session restarts (nohup). Waits for the first-phase runs, launches run_final.sh, waits for all 14 runs, runs the two
# evaluation chains and the table / figure / report step. Status lines -> logs/orchestrate.log
set -u
cd "$(dirname "$0")"
L=/result/uhnam/dexcore/reports/contact_factorization_stage1/logs; S=$L/orchestrate.log
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
say() { echo "$(date '+%H:%M:%S') $*" >> $S; }
n_done() { c=0; for d in taco arctic; do for m in "$@"; do grep -q "done ${m}_seed0:" $L/train_${d}_${m}_seed0.log 2>/dev/null && c=$((c+1)); done; done; echo $c; }
say "orchestrator started"
until [ $(n_done A0 A2 A3) -ge 6 ]; do sleep 60; done
say "first phase done; launching run_final.sh"
bash run_final.sh >> $S 2>&1
until [ $(n_done A0 A1 A2 A3 A1_z16 A2_z16 A3_z16) -ge 14 ]; do
  alive=$(pgrep -u uhnam -f "cf_train.py --dataset" | wc -l); say "waiting: $(n_done A0 A1 A2 A3 A1_z16 A2_z16 A3_z16)/14 done, $alive training processes"; sleep 300
done
say "all 14 trainings done; evaluation"
bash run_eval.sh 4 taco >> $S 2>&1 &
bash run_eval.sh 2 arctic >> $S 2>&1 &
wait
say "evaluation done; tables / figures / report"
$PY aggregate.py > $L/aggregate.log 2>&1 && $PY sanity.py > $L/sanity.log 2>&1 && $PY figures.py > $L/figures.log 2>&1 && say "tables done" || say "TABLES FAILED"
say "ORCHESTRATION_DONE"
