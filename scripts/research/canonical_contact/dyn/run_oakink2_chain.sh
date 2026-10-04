#!/bin/bash
# OakInk2: merge the shard indices, build the cache, then run the whole study chain.
set -e
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
OUT=/result/uhnam/dexcore/oakink2/10_dynamic_contact
cd /result/uhnam/dexcore/taco/40_representation_study/scripts/dyn
$PY - <<'PYEOF'
import pandas as pd, glob
OUT = "/result/uhnam/dexcore/oakink2/10_dynamic_contact"
parts = [pd.read_csv(f) for f in sorted(glob.glob(f"{OUT}/takes_index_shard*of*.csv"))]
ix = pd.concat(parts, ignore_index=True).drop_duplicates("file")
ix.to_csv(f"{OUT}/takes_index.csv", index=False)
print("merged takes:", len(ix), "sequences:", ix.sequence.nunique(), "hands:", ix.hand.value_counts().to_dict())
PYEOF
mkdir -p $OUT/{exp1,exp2,exp3,exp4,figures,verify,tables,logs}
$PY build_oakink2_cache.py > $OUT/logs/build_cache.log 2>&1
export DC_DATASET=oakink2 CUDA_VISIBLE_DEVICES=4
$PY exp1_threshold.py > $OUT/logs/exp1_threshold.log 2>&1
$PY exp1_amount_vs_pattern.py > $OUT/logs/exp1.log 2>&1
$PY exp1_amount_vs_pattern.py --all-windows > $OUT/logs/exp1_allwindows.log 2>&1
$PY exp2_lag_curves.py > $OUT/logs/exp2.log 2>&1
$PY figures.py 1 2 3 > $OUT/logs/figs123.log 2>&1
$PY exp3_predictability.py > $OUT/logs/exp3.log 2>&1
$PY exp3_activation.py > $OUT/logs/exp3_activation.log 2>&1
$PY exp4_representation.py > $OUT/logs/exp4.log 2>&1
$PY exp4_window_level.py > $OUT/logs/exp4_window.log 2>&1
$PY reaggregate.py > $OUT/logs/reaggregate.log 2>&1
$PY figures.py 4 5 > $OUT/logs/figs45.log 2>&1
$PY verify.py > $OUT/logs/verify.log 2>&1
$PY report_tables.py > $OUT/logs/tables.log 2>&1
echo OAKINK2_CHAIN_DONE >> $OUT/logs/tables.log
