#!/bin/bash
# Final launch after the first-phase runs finished:
#  * the finished A2 / A3 runs (|z| = 64; 4k phase A + 12k phase B with the first residual decoder, whose residual branch stayed
#    at zero) are kept as the PHASE-A checkpoints (<model>_seed0_phaseA.pt) and their z branch initialises a new 12k-step phase B
#    with the final residual decoder (A2_seed0.pt / A3_seed0.pt);
#  * the bottleneck variants A1_z16 / A2_z16 / A3_z16 are trained from scratch (12k; 4k + 12k).
# usage: bash run_final.sh
set -u
cd "$(dirname "$0")"
K=/ckpt/uhnam/dexcore/contact_factorization_stage1; R=/result/uhnam/dexcore/reports/contact_factorization_stage1
for d in taco arctic; do for m in A2 A3; do
  if [ -f $K/$d/${m}_seed0.pt ] && [ ! -f $K/$d/${m}_seed0_phaseA.pt ]; then
    mv $K/$d/${m}_seed0.pt $K/$d/${m}_seed0_phaseA.pt; mv $R/$d/train_logs/${m}_seed0.csv $R/$d/train_logs/${m}_seed0_phaseA.csv; mv $R/logs/train_${d}_${m}_seed0.log $R/logs/train_${d}_${m}_seed0_phaseA.log
  fi
done; done
L() { nohup bash launch_retry.sh "$@" > /dev/null 2>&1 & echo "gpu $1 $2 $3 ${5:-}"; }
PB="--init-z-from A3_seed0_phaseA --phase-b-only"; PB2="--init-z-from A2_seed0_phaseA --phase-b-only"
L 4 taco A3 0 "$PB";     L 4 taco A3_z16
L 2 arctic A3 0 "$PB";   L 2 arctic A3_z16
L 3 taco A2 0 "$PB2";    L 3 arctic A2 0 "$PB2"
L 5 taco A1_z16;         L 6 arctic A1_z16
L 3 taco A2_z16;         L 3 arctic A2_z16          # wait for free memory on GPU 3 (after the A2 continuations)
