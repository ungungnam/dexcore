#!/bin/bash
# One training with an OOM guard: waits until the GPU has >= NEED MiB free, retries up to 4 times when the run dies with a CUDA OOM
# (other users share the GPUs).   usage: bash launch_retry.sh <gpu> <dataset> <model> [seed]
set -u
GPU=$1; DS=$2; MODEL=$3; SEED=${4:-0}; EXTRA="${5:-}"      # EXTRA: further cf_train.py arguments (e.g. phase-B-only continuation)
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
export PYTHONPATH=/home/uhnam/workspace/dexcore
LOG=/result/uhnam/dexcore/reports/contact_factorization_stage1/logs/train_${DS}_${MODEL}_seed${SEED}.log
case $MODEL in A2*|A3*) NEED=18000;; *) NEED=10000;; esac
cd "$(dirname "$0")"
for attempt in 1 2 3 4; do
  for i in $(seq 1 400); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
    [ "$FREE" -ge "$NEED" ] && break
    sleep 60
  done
  echo "=== attempt $attempt on gpu $GPU ($(date), free ${FREE} MiB)" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $PY cf_train.py --dataset $DS --model $MODEL --seed $SEED $EXTRA >> $LOG 2>&1
  if grep -q "done ${MODEL}_seed${SEED}" $LOG; then exit 0; fi
  if tail -n 30 $LOG | grep -q "OutOfMemoryError"; then sleep 120; continue; fi
  exit 1
done
