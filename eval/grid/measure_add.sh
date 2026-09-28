#!/usr/bin/env bash
# measure_add.sh <clip_tag> <obj_name> <gpu>
#
# Give one finished run its ADD/AUC: roll the policy out, then score the object trajectory.
# train_eval does this as its step 4, but only for a run it launched itself -- a resumed run, or one
# whose training was moved between machines, never reaches that step, so it is done directly here.
#
# eval_rl_games needs very little memory (--num_envs 20), so this can share a card with training.
set -u
TAG=$1; OBJ=$2; GPU=$3
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
R=/result/uhnam/dexmachina/01_train_eval/rl_games/inspire_hand
cd /home/uhnam/workspace/dexmachina || exit 1

d=$(ls -d $R/*-u${TAG}_B* 2>/dev/null | head -1)
[ -n "$d" ] || { echo "FAIL $TAG: 런 디렉터리 없음"; exit 1; }
top=$(ls $d/nn/*ep_*.pth 2>/dev/null | grep -oE "ep_[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
[ -n "$top" ] || { echo "FAIL $TAG: 체크포인트 없음"; exit 1; }
# the shorter spelling of the two rl_games writes for the same epoch
ck=$(ls -S $d/nn/*ep_${top}_*.pth 2>/dev/null | tail -1)

echo "== $TAG ep$top  $(basename "$ck")"
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m dexmachina.rl.eval_rl_games \
    --checkpoint "$ck" --num_envs 20 > /tmp/add_${TAG}.log 2>&1 || { echo "FAIL $TAG: eval"; exit 1; }
np="$(dirname "$(dirname "$ck")")/$(basename "$ck" .pth)_eval/eval_ep0.npy"
[ -f "$np" ] || np=$(find "$d" -name eval_ep0.npy -newermt "-30 minutes" 2>/dev/null | head -1)
[ -f "$np" ] || { echo "FAIL $TAG: eval_ep0.npy 없음"; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU "$PY" -m dexmachina.eval.compute_add \
    --input "$np" --obj_name "$OBJ" --overwrite >> /tmp/add_${TAG}.log 2>&1 \
  && echo "OK   $TAG ep$top" || { echo "FAIL $TAG: compute_add"; exit 1; }
