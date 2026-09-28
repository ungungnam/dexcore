#!/usr/bin/env bash
# Measure ADD/AUC for every run in the grid that still lacks one, on whichever host holds it.
#
# Run this once training is finished. It is idempotent: a run whose newest epoch already has an
# add_stats.json is skipped, so re-running after a straggler completes only measures the straggler.
#
#   ./run_all_add.sh            # measure what is missing
#   ./run_all_add.sh --dry      # list what would be measured
set -u
DRY=${1:-}
G=/home/uhnam/workspace/dexcore/eval/grid
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
S=/tmp/grid_add
mkdir -p $S

# tag -> object name. The object is what compute_add scores the rollout against.
declare -A OBJ=(
  [pm379dx]=pm102379_calibrated [pm379b]=pm102379_calibrated
  [pm189dx]=pm100189_calibrated [pm189b]=pm100189_calibrated
  [pm224dx]=pm100224_calibrated [pm224b]=pm100224_calibrated
  [pm243dx]=pm100243_calibrated [pm243b]=pm100243_calibrated
  [pm658dx]=pm100658_calibrated [pm658b]=pm100658_calibrated
  [pmk]=pm100141_calibrated     [pmb]=pm100141_calibrated
  [lap876dx]=lap11876_calibrated [lap876b]=lap11876_calibrated
  [lap030dx]=lap11030_calibrated [lap030b]=lap11030_calibrated
  [lap239dx]=lap10239_calibrated [lap239b]=lap10239_calibrated
  [lap243dx]=lap10243_calibrated [lap243b]=lap10243_calibrated
  [lap211dx]=lap10211_calibrated [lap211b]=lap10211_calibrated
  [lap305dx]=lap10305_calibrated [lap305b]=lap10305_calibrated
)

"$PY" $G/add_audit.py mango > $S/mango.json
ssh -o BatchMode=yes sushi "$PY /home/uhnam/add_audit.py sushi" > $S/sushi.json

# a card with room for eval (--num_envs 20 is small, but leave headroom beside training)
pick () { ssh_pfx=$1
  cmd='nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits'
  [ -z "$ssh_pfx" ] && out=$(bash -lc "$cmd") || out=$(ssh -o BatchMode=yes "$ssh_pfx" "$cmd")
  echo "$out" | while IFS=, read -r i u t; do
    [ $(( ${t// /} - ${u// /} )) -ge 9000 ] && { echo "${i// /}"; return; }
  done | head -1
}

"$PY" - "$S" <<'PYEOF' > $S/todo.txt
import json, sys
S = sys.argv[1]
M = json.load(open(f"{S}/mango.json")); U = json.load(open(f"{S}/sushi.json"))
merged = {}
for src in (M, U):
    for k, v in src.items():
        if k not in merged or v["ep"] > merged[k]["ep"] or ("auc" in v and "auc" not in merged[k]):
            merged[k] = v
json.dump(merged, open(f"{S}/merged.json", "w"))
for k, v in merged.items():
    if "auc" not in v:
        print(f"{k} {v['host']} {v['ep']}")
PYEOF

n=$(wc -l < $S/todo.txt)
echo "측정 필요: ${n}건"
[ "$n" -eq 0 ] && { echo "모두 측정 완료 -- 비교표만 생성하면 됩니다"; exit 0; }
cat $S/todo.txt | sed 's/^/  /'
[ "$DRY" = "--dry" ] && exit 0

while read -r tag host ep; do
  obj=${OBJ[$tag]:-}
  [ -n "$obj" ] || { echo "SKIP $tag: 객체 이름 미등록"; continue; }
  if [ "$host" = mango ]; then
    g=$(pick ""); [ -n "$g" ] || { echo "SKIP $tag: mango 여유 GPU 없음"; continue; }
    $G/measure_add.sh "$tag" "$obj" "$g"
  else
    g=$(pick sushi); [ -n "$g" ] || { echo "SKIP $tag: sushi 여유 GPU 없음"; continue; }
    ssh -o BatchMode=yes sushi "/home/uhnam/measure_add.sh '$tag' '$obj' '$g'"
  fi
done < $S/todo.txt
echo "===== 측정 종료 -- compare.py 로 표를 만드세요 ====="
