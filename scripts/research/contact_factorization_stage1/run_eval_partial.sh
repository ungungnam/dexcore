#!/bin/bash
# Preliminary evaluation of the models that already finished (GPU), while the last trainings run.  usage: bash run_eval_partial.sh <gpu> <dataset>
set -u; GPU=$1; DS=$2; PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore CUDA_VISIBLE_DEVICES=$GPU
LOG=/result/uhnam/dexcore/reports/contact_factorization_stage1/logs; K=/ckpt/uhnam/dexcore/contact_factorization_stage1; cd "$(dirname "$0")"
for m in A0 A1 A2 A3 A1_z16 A2_z16 A3_z16; do [ -f $K/$DS/${m}_seed0.pt ] && { $PY encode.py --dataset $DS --model $m > $LOG/encode_${DS}_$m.log 2>&1 || echo "encode $m failed"; }; done
for s in eval_recon eval_neighbors eval_temporal eval_swap eval_probes; do $PY $s.py --dataset $DS > $LOG/${s}_${DS}_prelim.log 2>&1 || echo "$s failed"; done
echo PRELIM_DONE_$DS
