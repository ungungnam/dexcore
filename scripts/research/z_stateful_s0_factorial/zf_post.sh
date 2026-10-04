#!/bin/bash
# Post-training evaluation of one dataset on one GPU: generation -> metrics (test and validation) -> diagnostics -> sanity.  Memory-capped; safe to re-run.
#   bash zf_post.sh <gpu> <dataset>          skips a model whose checkpoint does not exist
set -u; cd "$(dirname "$0")"
GPU=$1; DS=$2
O=${ZF_OUT:-/result/uhnam/dexcore/reports/z_stateful_s0_factorial}; L=$O/logs; C=${ZF_CKPT:-/ckpt/uhnam/dexcore/z_stateful_s0_factorial}; mkdir -p $L
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore; export CUDA_VISIBLE_DEVICES=$GPU
say() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a $L/post_$DS.log; }
FAIL=0
run() { "$@" >> $L/post_$DS.log 2>&1 || { say "FAILED: $*"; FAIL=1; }; }
for m in M01 M10 M11; do
  [ -f $C/$DS/${m}_seed0.pt ] || { say "skip $m (no checkpoint)"; continue; }
  for part in test val; do run $PY zf_generate.py --dataset $DS --model $m --part $part; run $PY zf_evaluate.py --dataset $DS --name ${m}_seed0 --part $part; done
  say "$m generated and evaluated"
done
[ -f $O/$DS/metrics/D0r_partial_A.npz ] || { run $PY zf_d0r.py --dataset $DS; run $PY zf_evaluate.py --dataset $DS --name D0r_partial; }
run $PY zf_diag.py --dataset $DS
run $PY zf_sanity.py --dataset $DS
say "POST_DONE $DS (fail=$FAIL)"; exit $FAIL
