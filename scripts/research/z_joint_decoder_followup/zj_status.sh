#!/bin/bash
# Compact status of the queue and of every run (last validation, best steps, speed).   bash zj_status.sh
O=${ZJ_OUT:-/result/uhnam/dexcore/reports/z_joint_decoder_followup}; L=$O/logs
PY=/home/uhnam/miniconda3/envs/dexmachina/bin/python
date '+%m-%d %H:%M:%S'
$PY - <<PYEOF
import json, re, glob, os
st = json.load(open("$L/queue_status.json"))
for k, v in st.items():
    ds, name = k.split("/"); log = f"$L/train_{ds}_{name}.log"; last = ""
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        m = re.findall(r"  step (\d+) train [\d.]+ val ([\d.]+) \| val L_C ([\d.]+) best ([\d.]+) \(step (\d+)\) \| objective best ([\d.]+) \(step (\d+)\).*?\((\d+) s\)", txt)
        d = re.findall(r"done %s: .*" % name, txt)
        if d: last = d[-1][:150]
        elif m:
            s = m[-1]; last = f"logged step {s[0]}: val L_C {s[2]} (best {s[3]} @ {s[4]}), objective best {s[5]} @ {s[6]}, {float(s[7]) / int(s[0]):.2f} s/step"
    print(f"{k:22s} {v['state']:8s} gpu {v['gpu']} attempts {v['attempts']} fails {v['fails']} | {last}")
PYEOF
tail -n 3 $L/queue.log | cut -c1-200
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
