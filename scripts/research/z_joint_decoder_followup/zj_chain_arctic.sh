#!/bin/bash
# Wait for the ARCTIC M2 run, then: post-processing -> probe (only if its pre-registered trigger fires) -> tables -> figures.
set -u; cd "$(dirname "$0")"
O=/result/uhnam/dexcore/reports/z_joint_decoder_followup; L=$O/logs; PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python; export PYTHONPATH=/home/uhnam/workspace/dexcore
GPU=${1:-5}
until grep -q "done M2_seed0:" $L/train_arctic_M2_seed0.log; do sleep 30; done
bash zj_post.sh $GPU arctic > /dev/null 2>&1
TRIG=$($PY - <<PYEOF
import numpy as np
d = np.load("$O/arctic/diag/diag.npz", allow_pickle=True)
r = float(np.sqrt(d["M2|zerr2|train"].mean() / d["M2|zerr2|val"].mean())); print("run" if r < 0.5 else "skip", round(r, 3))
PYEOF
)
echo "$(date '+%m-%d %H:%M:%S') probe trigger arctic: $TRIG" >> $L/post_arctic.log
case "$TRIG" in run*) CUDA_VISIBLE_DEVICES=$GPU $PY zj_probe.py --dataset arctic --model M2 > $L/probe_arctic.log 2>&1;; esac
$PY zj_sanity.py --merge; CUDA_VISIBLE_DEVICES="" $PY zj_aggregate.py > $L/aggregate.log 2>&1; CUDA_VISIBLE_DEVICES="" $PY zj_figures.py > $L/figures.log 2>&1
echo "$(date '+%m-%d %H:%M:%S') CHAIN_ARCTIC_DONE" >> $L/post_arctic.log
