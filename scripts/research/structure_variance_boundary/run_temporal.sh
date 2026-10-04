#!/bin/bash
# Experiment B: 2 datasets x 8 representations x 2 heads x 3 seeds. One sequential chain per (dataset, seed),
# chains spread over the GPUs given.        run_temporal.sh "2 3 5"
set -u
GPUS=($1)
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
LOG=/result/uhnam/dexcore/reports/structure_variance_boundary/logs; mkdir -p $LOG
cd "$(dirname "$0")"
REPS="R0 R1 R2 R3 R4 R5 CONTACT_FULL FULL"
i=0
for ds in arctic taco; do
  for seed in 0 1 2; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
    (
      for head in structured dense; do
        for r in $REPS; do
          if [ -f /result/uhnam/dexcore/reports/structure_variance_boundary/$ds/temporal/metrics/${r}_${head}_seed${seed}.npz ]; then continue; fi
          CUDA_VISIBLE_DEVICES=$gpu $PY train_temporal.py --dataset $ds --rep $r --seed $seed --head $head > $LOG/temporal_${ds}_${r}_${head}_seed${seed}.log 2>&1 || echo "FAILED $ds $r $head seed $seed"
        done
      done
      echo "chain $ds seed $seed (gpu $gpu) finished $(date +%H:%M:%S)"
    ) &
  done
done
wait
echo "all chains finished $(date +%H:%M:%S)"
