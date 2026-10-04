#!/bin/bash
set -e
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
OUT=/result/uhnam/dexcore/oakink2/10_dynamic_contact
cd /result/uhnam/dexcore/taco/40_representation_study/scripts/dyn
export DC_DATASET=oakink2 CUDA_VISIBLE_DEVICES=4
$PY figures.py 2 3 > $OUT/logs/figs23.log 2>&1
$PY exp3_predictability.py > $OUT/logs/exp3.log 2>&1
$PY exp3_activation.py > $OUT/logs/exp3_activation.log 2>&1
$PY exp4_representation.py > $OUT/logs/exp4.log 2>&1
$PY exp4_window_level.py > $OUT/logs/exp4_window.log 2>&1
$PY reaggregate.py > $OUT/logs/reaggregate.log 2>&1
$PY figures.py 4 5 > $OUT/logs/figs45.log 2>&1
$PY verify.py > $OUT/logs/verify.log 2>&1
$PY report_tables.py > $OUT/logs/tables.log 2>&1
echo OAKINK2_CHAIN_DONE >> $OUT/logs/tables.log
