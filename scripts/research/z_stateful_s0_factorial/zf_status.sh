#!/bin/bash
# Compact status of the queue and of every run.   bash zf_status.sh
O=${ZF_OUT:-/result/uhnam/dexcore/reports/z_stateful_s0_factorial}; L=$O/logs
date '+%m-%d %H:%M:%S'
/home/uhnam/miniconda3/envs/dexmachina/bin/python - <<PYEOF
import json, re, os
st = json.load(open("$L/queue_status.json"))
for k, v in st.items():
    ds, name = k.split("/"); log = f"$L/train_{ds}_{name}.log"; last = ""
    if os.path.exists(log):
        txt = open(log, errors="replace").read().split("the run restarts from step 1")[-1]
        m = re.findall(r"(\d\d:\d\d:\d\d)   step (\d+) train [\d.]+ val ([\d.]+) \| val L_C ([\d.]+) best ([\d.]+) \(step (\d+)\) \| objective best ([\d.]+) \(step (\d+)\).*?\((\d+) s\)", txt)
        d = re.findall(r"done %s: .*" % name, txt)
        if d: last = d[-1][:140]
        elif m:
            s = m[-1]; last = f"{s[0]} step {s[1]}: val L_C {s[3]} (best {s[4]} @ {s[5]}), objective best {s[6]} @ {s[7]}, {float(s[8]) / int(s[1]):.2f} s/step"
    print(f"{k:20s} {v['state']:8s} gpu {v['gpu']} att {v['attempts']} fails {v['fails']} | {last}")
PYEOF
tail -n 2 $L/queue.log | cut -c1-220
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' '; echo
# latest values logged to wandb (local summary files; validation is logged every 1 000 steps, the text log only every 5 000)
/home/uhnam/miniconda3/envs/dexmachina/bin/python - <<PYEOF
import json, glob, os
latest = {}
for d in sorted(glob.glob("$O/wandb/wandb/run-*")):
    latest[d.split("-")[-1]] = d
keep = ("_step", "val/L_C", "val/best_L_C", "val/L_z", "val/L_z0", "val/E_C", "train/L_z")
for rid, d in latest.items():
    p = d + "/files/wandb-summary.json"
    if os.path.exists(p) and (rid.endswith("_indep") or "M01" in rid):
        try:
            s = json.load(open(p))
        except Exception:
            continue
        print(f"  {rid:24s}", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k in keep})
PYEOF
