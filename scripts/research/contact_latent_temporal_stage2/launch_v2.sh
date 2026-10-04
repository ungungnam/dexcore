#!/bin/bash
# One training with an OOM guard (v2: per-attempt log slice, python exit status, guarded free-memory parse). Waits until the GPU has
# >= NEED MiB free, retries (resuming from the last checkpoint) when the attempt dies with a CUDA OOM.
#   usage: bash launch_v2.sh <gpu> <dataset> <model> [seed] [extra s2_train.py args]
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
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU 2>/dev/null | head -n 1)
    [ -n "$FREE" ] && [ "$FREE" -ge "$NEED" ] && break
    sleep 60
  done
  ATT=$(mktemp)
  echo "=== attempt $attempt on gpu $GPU ($(date), free ${FREE:-?} MiB) $RESUME" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU $PY s2_train.py --dataset $DS --model $MODEL --seed $SEED --wandb $EXTRA $RESUME > $ATT 2>&1; RC=$?
  cat $ATT >> $LOG
  if [ $RC -eq 0 ] && grep -q "done ${NAME}:" $ATT; then rm -f $ATT; exit 0; fi
  if grep -q "OutOfMemoryError\|CUDA error\|CUBLAS_STATUS" $ATT; then rm -f $ATT; RESUME="--resume"; sleep 180; continue; fi
  echo "=== attempt $attempt failed with exit code $RC (not an OOM); giving up" >> $LOG; rm -f $ATT; exit 1
done
