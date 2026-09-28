#!/usr/bin/env python
"""The ADD/AUC comparison: our staged pipeline against BimArt, one row per target.

Reads the merged audit that run_all_add.sh leaves in /tmp/grid_add/merged.json.

Two things the table is careful about, because both would flatter one method over the other:

  matched epochs -- a run that stopped at 2510 has not been given the same budget as one that
    reached 5000, so the epoch is printed beside every number and rows where the two methods differ
    are marked. Only matched rows carry a verdict.
  failed rows -- when both methods leave the object roughly a metre from where the demonstration
    puts it, neither method worked, and the sign of a 0.02 AUC gap between two failures says
    nothing. Those rows are reported as a shared failure rather than as a win.
"""
import json, os, sys

MERGED = sys.argv[1] if len(sys.argv) > 1 else "/tmp/grid_add/merged.json"
FAIL_CM = 50.0          # beyond this the policy has lost the object; the comparison is moot

GRID = [("BOX", [("pm102379", 42, "pm379dx", "pm379b"), ("pm100189", 38, "pm189dx", "pm189b"),
                 ("pm100224", 19, "pm224dx", "pm224b"), ("pm100243", 17, "pm243dx", "pm243b"),
                 ("pm100658", 11, "pm658dx", "pm658b"), ("pm100141",  9, "pmk",     "pmb")]),
        ("LAPTOP", [("lap11876", 29, "lap876dx", "lap876b"), ("lap11030", 15, "lap030dx", "lap030b"),
                    ("lap10239", 12, "lap239dx", "lap239b"), ("lap10243",  8, "lap243dx", "lap243b"),
                    ("lap10211",  4, "lap211dx", "lap211b"), ("lap10305",  2, "lap305dx", "lap305b")]),
        ("MICROWAVE", [("mw7304", 46, "mw304dx", "mw304b"), ("mw7236", 34, "mw236dx", "mw236b"),
                       ("mw7310", 19, "mw310dx", "mw310b"), ("mw7292", 12, "mw292dx", "mw292b")])]

d = json.load(open(MERGED))
wins = losses = ties = 0
rows_matched = []

for section, rows in GRID:
    print(f"\n=== {section} ===")
    print("  %-9s %-4s | %-22s | %-22s | %s"
          % ("타깃", "조각", "우리 (AUC / ADD / ep)", "BimArt (AUC / ADD / ep)", "판정"))
    for target, pieces, dx, b in rows:
        a, c = d.get(dx, {}), d.get(b, {})
        def cell(x):
            if "auc" not in x:
                return f"{'미측정':<22}"
            return f"{x['auc']:.3f} / {x['add_cm']:5.1f}cm / ep{x['ep']:<5d}"
        verdict = "-"
        if "auc" in a and "auc" in c:
            same_ep = a["ep"] == c["ep"]
            both_lost = a["add_cm"] > FAIL_CM and c["add_cm"] > FAIL_CM
            gap = a["auc"] - c["auc"]
            if both_lost:
                verdict = "양쪽 실패"
            elif not same_ep:
                verdict = f"{'우리' if gap>0 else 'BimArt'} +{abs(gap):.3f} (에폭 불일치)"
            else:
                verdict = f"{'우리' if gap>0 else 'BimArt'} +{abs(gap):.3f}"
                rows_matched.append((target, gap))
                if gap > 0: wins += 1
                elif gap < 0: losses += 1
                else: ties += 1
        print("  %-9s %-4d | %-22s | %-22s | %s" % (target, pieces, cell(a), cell(c), verdict))

print("\n--- 동일 에폭 + 양쪽 성공한 행만 집계 ---")
print(f"  우리 승 {wins} / BimArt 승 {losses} / 동률 {ties}   (총 {len(rows_matched)}개 타깃)")
if rows_matched:
    g = [x[1] for x in rows_matched]
    print(f"  AUC 격차: 평균 {sum(g)/len(g):+.3f}, 최소 {min(g):+.3f}, 최대 {max(g):+.3f}")
print(f"\n  ADD {FAIL_CM:.0f}cm 초과는 물체를 놓친 것으로 보고 판정에서 제외했습니다.")
