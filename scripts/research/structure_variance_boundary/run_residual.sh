#!/bin/bash
# Experiment D: decoder + residual predictors (history / condmean / oracle) for one Z* per dataset, 3 seeds.
#     run_residual.sh "<taco Z*>" "<arctic Z*>" "<gpu taco> <gpu arctic>" ["<models>"]
set -u
ZT=$1; ZA=$2; GPUS=($3); MODELS=${4:-"history condmean oracle"}; SUF=$(echo "$MODELS" | tr " " "_")
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
LOG=/result/uhnam/dexcore/reports/structure_variance_boundary/logs; mkdir -p $LOG
cd "$(dirname "$0")"
(
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY residual.py --dataset taco --zstar $ZT --seed $seed --models $MODELS > $LOG/residual_taco_${ZT}_seed${seed}_${SUF}.log 2>&1 || echo "FAILED taco $ZT seed $seed"
  done
  echo "taco residual chain finished $(date +%H:%M:%S)"
) &
(
  for seed in 0 1 2; do
    CUDA_VISIBLE_DEVICES=${GPUS[1]} $PY residual.py --dataset arctic --zstar $ZA --seed $seed --models $MODELS > $LOG/residual_arctic_${ZA}_seed${seed}_${SUF}.log 2>&1 || echo "FAILED arctic $ZA seed $seed"
  done
  echo "arctic residual chain finished $(date +%H:%M:%S)"
) &
wait
echo "all residual chains finished $(date +%H:%M:%S)"
