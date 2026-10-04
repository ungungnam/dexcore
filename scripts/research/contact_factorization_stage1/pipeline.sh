#!/bin/bash
# Exact reproduction of the Stage-1 study (GPU ids as used; any free GPU works).
#   1. training (8 runs, one seed)            bash run_train.sh          (= launch_retry.sh <gpu> <ds> <model> for each)
#   2. evaluation per dataset                 bash run_eval.sh 4 taco; bash run_eval.sh 3 arctic
#   3. tables, decision, sanity, figures      python aggregate.py && python sanity.py && python figures.py
#   4. report                                  python report.py           (narrative/*.md + tables -> report.md)
# Environment: PYTHONPATH=/home/uhnam/workspace/dexcore, python = /home/uhnam/miniconda3/envs/dexmachina/bin/python.
# Inputs: hier_contact_gen sequences / split (/result/uhnam/dexcore/{taco/50,arctic/40}_hier_contact_gen), the structure_variance_boundary
# feature cache (exact R2 + wrench), the structure_aware_temporal_generation mask / geometry caches and calibration.json.
set -e
cd "$(dirname "$0")"
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
case ${1:-all} in
  train) bash run_train.sh ;;
  eval) bash run_eval.sh ${2:-4} taco; bash run_eval.sh ${3:-3} arctic ;;
  tables) $PY aggregate.py && $PY sanity.py && $PY figures.py && $PY report.py ;;
  all) bash run_train.sh; echo "wait for the trainings, then: bash pipeline.sh eval && bash pipeline.sh tables" ;;
esac
