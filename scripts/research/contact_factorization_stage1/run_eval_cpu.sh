#!/bin/bash
# CPU fallback of run_eval.sh (GPU driver hang on 2026-10-03): encodes every finished model and runs the evaluation chain of one dataset.
# usage: bash run_eval_cpu.sh <dataset> [threads]
set -u
DS=$1; export OMP_NUM_THREADS=${2:-16} MKL_NUM_THREADS=${2:-16} CF_DEVICE=cpu PYTHONPATH=/home/uhnam/workspace/dexcore
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; LOG=/result/uhnam/dexcore/reports/contact_factorization_stage1/logs; K=/ckpt/uhnam/dexcore/contact_factorization_stage1
cd "$(dirname "$0")"
for m in A0 A1 A2 A3 A1_z16 A2_z16 A3_z16; do
  [ -f $K/$DS/${m}_seed0.pt ] || { echo "skip $m (no checkpoint)"; continue; }
  [ -f /result/uhnam/dexcore/reports/contact_factorization_stage1/$DS/latents/${m}_seed0_test.npz ] && [ /result/uhnam/dexcore/reports/contact_factorization_stage1/$DS/latents/${m}_seed0_test.npz -nt $K/$DS/${m}_seed0.pt ] && { echo "latents of $m up to date"; continue; }
  $PY encode.py --dataset $DS --model $m > $LOG/encode_${DS}_$m.log 2>&1 || echo "encode $m failed"
done
for s in eval_recon eval_probes eval_neighbors eval_swap eval_temporal; do $PY $s.py --dataset $DS > $LOG/${s}_$DS.log 2>&1 || echo "$s failed"; done
echo EVAL_CPU_DONE_$DS
