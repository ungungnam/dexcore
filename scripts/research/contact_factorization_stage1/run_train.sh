#!/bin/bash
# Final primary runs (one seed). Schedule: A1 12k; A0 24k (reconstruction control, trained longer); A2 / A3 4k (phase A) + 12k
# (phase B). Each run goes through launch_retry.sh (waits for free memory, retries on CUDA OOM: the GPUs are shared).
# History: first launch (20k / 8k + 16k) stopped for speed; second launch (12k / 4k + 10k) produced the A1 runs kept here, A0 runs
# whose validation error was still falling (re-run at 24k) and factorised runs whose residual branch never activated (zero-init
# output behind zero-init FiLM gates; fixed in cf_models.PointDecoder and re-run).   usage: bash run_train.sh [all|final]
cd "$(dirname "$0")"
L() { nohup bash launch_retry.sh "$@" > /dev/null 2>&1 & echo "gpu $1 $2 $3"; }
if [ "${1:-final}" = all ]; then L 4 taco A1; L 2 arctic A1; fi
L 4 taco A3;   L 4 taco A0
L 2 arctic A3; L 2 arctic A0
L 3 taco A2;   L 3 arctic A2
