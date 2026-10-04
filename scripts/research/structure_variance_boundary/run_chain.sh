#!/bin/bash
# One Experiment B chain (dataset, seed) on one GPU, skipping runs whose metrics exist.    run_chain.sh taco 2 1
set -u
ds=$1; seed=$2; gpu=$3
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
LOG=/result/uhnam/dexcore/reports/structure_variance_boundary/logs; mkdir -p $LOG
cd "$(dirname "$0")"
for head in structured dense; do
  for r in R0 R1 R2 R3 R4 R5 CONTACT_FULL FULL; do
    if [ -f /result/uhnam/dexcore/reports/structure_variance_boundary/$ds/temporal/metrics/${r}_${head}_seed${seed}.npz ]; then continue; fi
    CUDA_VISIBLE_DEVICES=$gpu $PY train_temporal.py --dataset $ds --rep $r --seed $seed --head $head > $LOG/temporal_${ds}_${r}_${head}_seed${seed}.log 2>&1 || echo "FAILED $ds $r $head seed $seed"
  done
done
echo "chain $ds seed $seed (gpu $gpu) finished $(date +%H:%M:%S)"
