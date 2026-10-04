#!/bin/bash
# Experiment A: the six probes of one dataset (3 horizons x {probe, probe_notau}), one seed.   bash run_probes.sh <gpu> <dataset>
# Exits non-zero if any probe failed (the remaining probes still run).
set -u; cd "$(dirname "$0")"; GPU=$1; DS=$2; FAIL=0
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
L=/result/uhnam/dexcore/reports/z_temporal_diagnostic/logs; mkdir -p $L
for v in probe probe_notau; do for h in 1 4 8; do
  CUDA_VISIBLE_DEVICES=$GPU $PY zt_probe.py --dataset $DS --h $h --variant $v >> $L/probe_$DS.log 2>&1 || { echo "FAILED $DS $v h=$h" >> $L/probe_$DS.log; FAIL=1; }
done; done
echo "PROBES_DONE $DS (failed: $FAIL)" >> $L/probe_$DS.log
exit $FAIL
