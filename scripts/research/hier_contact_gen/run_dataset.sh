#!/bin/bash
# One dataset, the one fixed split: train sampler_G, sampler_GT and vf in parallel on one GPU, then
# evaluate B0-B3 by free rollout.        run_dataset.sh <dataset> <gpu>
set -u
DS=$1; GPU=$2
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
LOG=/result/uhnam/dexcore/reports/hier_logs/$DS; mkdir -p $LOG
cd "$(dirname "$0")"
export DC_DATASET=$DS CUDA_VISIBLE_DEVICES=$GPU
pids=()
for m in sampler_G sampler_GT vf; do
  $PY train.py --split fixed --fold 0 --model $m > $LOG/train_$m.log 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail -eq 0 ] || { echo "training failed for $DS, see $LOG/train_*.log"; exit 1; }
$PY rollout_eval.py --split fixed --fold 0 > $LOG/eval.log 2>&1 || { echo "evaluation failed for $DS"; exit 1; }
if [ "$DS" = "taco" ]; then $PY rollout_eval.py --split fixed --fold 0 --part test_extra > $LOG/eval_extra.log 2>&1 || echo "extra evaluation failed"; fi
$PY aggregate.py > $LOG/aggregate.log 2>&1
$PY figures.py > $LOG/figures.log 2>&1
echo "dataset $DS finished $(date +%H:%M:%S)"
