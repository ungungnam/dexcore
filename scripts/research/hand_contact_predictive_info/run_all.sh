#!/bin/bash
# Every training of the diagnostic: 2 datasets x 8 conditions x 3 seeds, the Delta-C regressors
# and then the event classifiers. One sequential chain per (dataset, seed); the chains are spread
# over the GPUs given (two chains per GPU with three GPUs).      run_all.sh "4 5 6"
set -u
GPUS=($1)
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
LOG=/result/uhnam/dexcore/reports/hand_contact_predictive_info/logs; mkdir -p $LOG
cd "$(dirname "$0")"
CONDS="F0 F1 F2k8 F2k1 F2k4 F2rel F2shuf Ffut"
i=0
for ds in arctic taco; do
  for seed in 0 1 2; do
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
    (
      for c in $CONDS; do
        CUDA_VISIBLE_DEVICES=$gpu $PY train_delta.py --dataset $ds --cond $c --seed $seed > $LOG/delta_${ds}_${c}_seed${seed}.log 2>&1 || echo "FAILED delta $ds $c seed $seed"
      done
      for c in $CONDS; do
        CUDA_VISIBLE_DEVICES=$gpu $PY train_event.py --dataset $ds --cond $c --seed $seed > $LOG/event_${ds}_${c}_seed${seed}.log 2>&1 || echo "FAILED event $ds $c seed $seed"
      done
      echo "chain $ds seed $seed (gpu $gpu) finished $(date +%H:%M:%S)"
    ) &
  done
done
wait
echo "all chains finished $(date +%H:%M:%S)"
