#!/bin/bash
# Post-training evaluation of one dataset on one GPU (generation -> metrics -> diagnostics -> sanity).  Memory-capped (shared GPUs).
#   bash zj_post.sh <gpu> <dataset>          skips a model whose checkpoint does not exist; safe to re-run
set -u; cd "$(dirname "$0")"
GPU=$1; DS=$2
O=${ZJ_OUT:-/result/uhnam/dexcore/reports/z_joint_decoder_followup}; L=$O/logs; C=${ZJ_CKPT:-/ckpt/uhnam/dexcore/z_joint_decoder_followup}; mkdir -p $L
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore; export CUDA_VISIBLE_DEVICES=$GPU
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $L/post_$DS.log; }
FAIL=0
run() { "$@" >> $L/post_$DS.log 2>&1 || { say "FAILED: $*"; FAIL=1; }; }
for m in M2 M3 M0r; do
  [ -f $C/$DS/${m}_seed0.pt ] || { say "skip $m (no checkpoint)"; continue; }
  for part in test val; do run $PY zj_generate.py --dataset $DS --model $m --part $part; run $PY zj_evaluate.py --dataset $DS --name ${m}_seed0 --part $part; done
  if [ $m != M0r ]; then
    run $PY zj_generate.py --dataset $DS --model $m --state state_total; run $PY zj_evaluate.py --dataset $DS --name ${m}_seed0_seltotal
  fi
  say "$m generated and evaluated"
done
for n in B0_seed0 B1_seed0; do run $PY zj_evaluate.py --dataset $DS --name $n --recheck; done
run $PY zj_diag.py --dataset $DS
run $PY zj_sanity.py --dataset $DS
say "POST_DONE $DS (fail=$FAIL)"; exit $FAIL
