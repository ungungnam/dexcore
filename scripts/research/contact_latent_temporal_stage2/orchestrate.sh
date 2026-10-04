#!/bin/bash
# Restart-proof orchestration: waits for the six main runs, then generates / evaluates the test trajectories of both datasets (one GPU
# each), then the tables, sanity checks, figures, report and dashboard.   nohup bash orchestrate.sh <gpu_taco> <gpu_arctic> &
set -u; cd "$(dirname "$0")"
G1=${1:-5}; G2=${2:-4}
O=/result/uhnam/dexcore/reports/contact_latent_temporal_stage2; L=$O/logs; LOG=$L/orchestrate.log
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> $LOG; }
done_run() { grep -q "done ${2}_seed0:" $L/train_${1}_${2}_seed0.log 2>/dev/null; }
say "orchestrator started (gpus $G1 / $G2)"
for ds in taco arctic; do for m in B0 B1 B2; do until done_run $ds $m; do sleep 300; done; say "$ds $m finished"; done; done
eval_ds() {  # <gpu> <dataset>
  local gpu=$1 ds=$2
  for m in B0 B1 B2; do
    CUDA_VISIBLE_DEVICES=$gpu $PY s2_generate.py --dataset $ds --name ${m}_seed0 >> $L/eval_$ds.log 2>&1 || say "GENERATE FAILED $ds $m"
    CUDA_VISIBLE_DEVICES=$gpu $PY s2_evaluate.py --dataset $ds --name ${m}_seed0 >> $L/eval_$ds.log 2>&1 || say "EVALUATE FAILED $ds $m"
  done
  CUDA_VISIBLE_DEVICES=$gpu $PY s2_evaluate.py --dataset $ds --name B2_seed0 --zonly >> $L/eval_$ds.log 2>&1 || say "EVALUATE FAILED $ds B2 zonly"
  for ref in GT PERSIST; do CUDA_VISIBLE_DEVICES=$gpu $PY s2_evaluate.py --dataset $ds --name $ref >> $L/eval_$ds.log 2>&1 || say "EVALUATE FAILED $ds $ref"; done
  say "$ds evaluation done"
}
eval_ds $G1 taco & eval_ds $G2 arctic & wait
say "aggregate"
$PY s2_aggregate.py > $L/aggregate.log 2>&1 || say "AGGREGATE FAILED"
CUDA_VISIBLE_DEVICES=$G1 $PY s2_sanity.py --dataset taco > $L/sanity_taco.log 2>&1 || say "SANITY taco FAILED"
CUDA_VISIBLE_DEVICES=$G2 $PY s2_sanity.py --dataset arctic > $L/sanity_arctic.log 2>&1 || say "SANITY arctic FAILED"
$PY s2_sanity.py --merge >> $L/sanity_taco.log 2>&1
CUDA_VISIBLE_DEVICES=$G1 $PY s2_figures.py > $L/figures.log 2>&1 || say "FIGURES FAILED"
say "ORCHESTRATION_DONE"
