#!/usr/bin/env python
"""Per-run ADD/AUC state on ONE machine, as JSON. Merge the two hosts' output for the whole grid.

Matching is by EPOCH, not by filename: rl_games writes the same checkpoint under two spellings
(`rew_130.44226` and `rew__130.44226_`), so an eval directory keyed to one of them looks absent if
you compare paths. What matters is whether a measurement exists for the epoch the run has reached.
"""
import glob, json, os, re, sys

R = "/result/uhnam/dexmachina/01_train_eval/rl_games/inspire_hand"
TAGS = ["pm379dx","pm379b","pm189dx","pm189b","pm224dx","pm224b","pm243dx","pm243b",
        "pm658dx","pm658b","pmk","pmb","lap876dx","lap876b","lap030dx","lap030b",
        "lap239dx","lap239b","lap243dx","lap243b","lap211dx","lap211b","lap305dx","lap305b",
        "mw236dx","mw236b","mw292dx","mw292b","mw304dx","mw304b","mw310dx","mw310b"]

out = {}
for tag in TAGS:
    d = glob.glob(f"{R}/*-u{tag}_B*")
    if not d:
        continue
    d = d[0]
    eps = sorted({int(m.group(1)) for m in
                  (re.search(r"ep_(\d+)", p) for p in glob.glob(d + "/nn/*ep_*.pth")) if m})
    if not eps:
        continue
    top = eps[-1]
    rec = {"ep": top, "host": sys.argv[1], "dir": d}
    # any add_stats computed AT this epoch counts, whichever checkpoint spelling it hangs off
    for s in glob.glob(d + "/**/add_stats.json", recursive=True):
        m = re.search(r"ep_(\d+)", s)
        if m and int(m.group(1)) == top:
            j = json.load(open(s))
            rec["auc"] = j["auc"]["overall"]
            rec["add_cm"] = j["mean_add"]["overall"] * 100
            break
    if "auc" not in rec:
        cks = sorted(glob.glob(d + f"/nn/*ep_{top}_*.pth"), key=len)
        rec["ckpt"] = cks[0] if cks else None
    out[tag] = rec
print(json.dumps(out))
