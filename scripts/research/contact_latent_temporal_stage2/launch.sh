#!/bin/bash
# One training with an OOM guard: waits until the GPU has >= NEED MiB free, retries (resuming from the last checkpoint) when the run dies
# with a CUDA OOM (other users share the GPUs).   usage: bash launch.sh <gpu> <dataset> <model> [seed] [extra s2_train.py args]
set -u
GPU=$1; DS=$2; MODEL=$3; SEED=${4:-0}; EXTRA="${5:-}"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
export PYTHONPATH=/home/uhnam/workspace/dexcore
[ -f ~/api_keys/wandb_key.txt ] && export WANDB_API_KEY=$(cat ~/api_keys/wandb_key.txt)
export WANDB_DIR=/result/uhnam/dexcore/reports/contact_latent_temporal_stage2/wandb; mkdir -p $WANDB_DIR
TAG=$(echo "$EXTRA" | sed -n 's/.*--tag \([^ ]*\).*/\1/p')
NAME=${MODEL}_seed${SEED}${TAG}
LOG=/result/uhnam/dexcore/reports/contact_latent_temporal_stage2/logs/train_${DS}_${NAME}.log
case $MODEL in B2) NEED=16000;; B1) NEED=12000;; *) NEED=8000;; esac
cd "$(dirname "$0")"
RESUME=""
for attempt in 1 2 3 4 5 6; do
  for i in $(seq 1 600); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
    [ "$FREE" -ge "$NEED" ] && break
    sleep 60
  done
  echo "=== attempt $attempt on gpu $GPU ($(date), free ${FREE} MiB) $RESUME" >> $LOG
  CUDA_VISIBLE_DEVICES=$GPU $PY s2_train.py --dataset $DS --model $MODEL --seed $SEED --wandb $EXTRA $RESUME >> $LOG 2>&1
  if grep -q "done ${NAME}:" $LOG; then exit 0; fi
  if tail -n 40 $LOG | grep -q "OutOfMemoryError\|CUDA error\|CUBLAS_STATUS"; then RESUME="--resume"; sleep 180; continue; fi
  exit 1
done
