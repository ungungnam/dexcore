#!/usr/bin/env bash
# Render every panel the comparison page needs, one subprocess per video (Genesis inits once per
# process, so batching inside one process is not an option).
#
#   MACHINE=mango GPU=3 ./build_videos.sh
#   MACHINE=sushi GPU=0 ./build_videos.sh
#
# Three panel kinds:
#   source  -- the ARCTIC box human demo, identical for every target (rendered once)
#   target  -- the generated reference installed at processed/s01/<target>_use_<tag>.npy
#   rollout -- eval_rl_games --record_video on the run's newest checkpoint
set -u
REPO=/home/uhnam/workspace/dexmachina
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
OUT=/home/uhnam/workspace/dexcore/visualize/videos
DRIVER=/home/uhnam/workspace/dexcore/visualize/render_demo_one.py
GPU="${GPU:-3}"
MACHINE="${MACHINE:-mango}"
LOG=$OUT/build_${MACHINE}.log
mkdir -p "$OUT"; : > "$LOG"
export PYOPENGL_PLATFORM=egl

# target_registry_name : demo_tag : strategy_label
if [ "$MACHINE" = "laptop" ]; then
  RUNS="lap11876_calibrated:lap876dx:dexcore
lap11876_calibrated:lap876b:bimart
lap11030_calibrated:lap030dx:dexcore
lap11030_calibrated:lap030b:bimart
lap10239_calibrated:lap239dx:dexcore
lap10239_calibrated:lap239b:bimart
lap10243_calibrated:lap243dx:dexcore
lap10243_calibrated:lap243b:bimart
lap10211_calibrated:lap211dx:dexcore
lap10211_calibrated:lap211b:bimart
lap10305_calibrated:lap305dx:dexcore
lap10305_calibrated:lap305b:bimart"
elif [ "$MACHINE" = "microwave" ]; then
  RUNS="mw7304_calibrated:mw304dx:dexcore
mw7304_calibrated:mw304b:bimart
mw7236_calibrated:mw236dx:dexcore
mw7236_calibrated:mw236b:bimart
mw7310_calibrated:mw310dx:dexcore
mw7310_calibrated:mw310b:bimart
mw7292_calibrated:mw292dx:dexcore
mw7292_calibrated:mw292b:bimart"
elif [ "$MACHINE" = "sushi" ]; then
  RUNS="pm100141_calibrated:pmk:ours_cordex
pm100141_calibrated:pmb:bimart
pm100658_calibrated:pm658dx:dexcore
pm100658_calibrated:pm658b:bimart"
else
  RUNS="pm102379_calibrated:pm379dx:dexcore
pm102379_calibrated:pm379b:bimart
pm100243_calibrated:pm243dx:dexcore
pm100243_calibrated:pm243b:bimart
pm100224_calibrated:pm224dx:dexcore
pm100224_calibrated:pm224b:bimart
pm100189_calibrated:pm189dx:dexcore
pm100189_calibrated:pm189b:bimart"
fi

say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

render_demo () {   # npy target tag
  local npy=$1 tgt=$2 tag=$3
  if [ -f "$OUT/$tag.mp4" ]; then say "SKIP demo $tag (있음)"; return; fi
  [ -f "$REPO/$npy" ] || { say "MISS demo $tag ($npy 없음)"; return; }
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPU timeout 1200 "$PY" "$DRIVER" \
      --npy "$npy" --target "$tgt" --out "$OUT" --tag "$tag" ) >> "$LOG" 2>&1 \
    && say "OK   demo $tag" || say "FAIL demo $tag"
}

render_rollout () { # run_dir_glob tag
  local glob=$1 tag=$2
  if [ -f "$OUT/$tag.mp4" ]; then say "SKIP rollout $tag (있음)"; return; fi
  local d ckpt
  d=$(ls -d /result/uhnam/dexmachina/01_train_eval/rl_games/inspire_hand/*${glob}* 2>/dev/null | head -1)
  [ -z "$d" ] && { say "MISS rollout $tag (런 없음)"; return; }
  ckpt=$(ls -t "$d"/nn/*.pth 2>/dev/null | head -1)
  [ -z "$ckpt" ] && { say "MISS rollout $tag (체크포인트 없음)"; return; }
  say "..   rollout $tag <- $(basename "$ckpt")"
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES=$GPU timeout 2400 "$PY" -m dexmachina.rl.eval_rl_games \
      --checkpoint "$ckpt" -B 1 --record_video ) >> "$LOG" 2>&1
  local v="$(dirname "$(dirname "$ckpt")")/$(basename "$ckpt" .pth)_eval/video.mp4"
  [ -f "$v" ] || v=$(find "$d" -name "video.mp4" -newermt '-45 minutes' 2>/dev/null | head -1)
  if [ -f "$v" ]; then cp "$v" "$OUT/$tag.mp4"; say "OK   rollout $tag"; else say "FAIL rollout $tag"; fi
}

# 1. source demo -- one per machine, same content
case "$MACHINE" in
  microwave) render_demo dexmachina/assets/arctic/processed/s01/microwave_use_01.npy microwave source_microwave_human ;;
  laptop)    render_demo dexmachina/assets/arctic/processed/s01/laptop_use_01.npy laptop source_laptop_human ;;
  *)         render_demo dexmachina/assets/arctic/processed/s01/box_use_01.npy box_s100 source_box_human ;;
esac

# 2+3. per run
for e in $RUNS; do
  tgt=${e%%:*}; rest=${e#*:}; tag=${rest%%:*}; strat=${rest##*:}
  short=$(echo "$tgt" | sed 's/_calibrated//')
  render_demo "dexmachina/assets/arctic/processed/s01/${tgt}_use_${tag}.npy" "$tgt" "${short}_${strat}_target"
  render_rollout "${tgt}_${strat}_" "${short}_${strat}_rollout"
done

rm -rf "$OUT"/_work_*
say "===== $MACHINE 완료 ====="
