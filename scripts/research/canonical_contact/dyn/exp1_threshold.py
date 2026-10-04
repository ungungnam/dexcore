#!/usr/bin/env python
"""Pick the 'effectively zero contact' mass threshold from the data and record it.

Mass = sum of the 512 soft values. exp(-d/0.02) has a long tail (a hand hovering 5 cm from the
whole surface still gives ~0.08 per point), so mass alone does not separate 'touching' from
'near'. This script tabulates mass against the hard-contact count (vertices within 1 cm) and the
hand-surface minimum distance over ALL frames (touching and non-touching windows), then chooses
the threshold conservatively: the largest mass at which 99% of frames have no vertex within 1 cm.
"""
import numpy as np
import pandas as pd

import dc_common as C

rows, allm, allh, alld = [], [], [], []
for cat, role, hand in C.GROUPS:
    X, M, P = C.load_group(cat, role, hand)
    m = C.mass(X)
    allm.append(m); allh.append(M.n_hard.values); alld.append(M.min_d.values)
    q = np.quantile(m, [0.01, 0.05, 0.1, 0.25, 0.5, 0.9])
    rows.append(dict(group=C.gname(cat, role, hand), n=len(m), frac_hard0=float((M.n_hard == 0).mean()),
                     frac_touch_win=float((M.touch_window >= C.TOUCH_MIN).mean()),
                     m_q01=q[0], m_q05=q[1], m_q10=q[2], m_q25=q[3], m_q50=q[4], m_q90=q[5],
                     m_med_hard0=float(np.median(m[M.n_hard == 0])) if (M.n_hard == 0).any() else np.nan,
                     m_med_hard=float(np.median(m[M.n_hard > 0]))))
D = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(D.round(2).to_string(index=False))
m, h, d = np.concatenate(allm), np.concatenate(allh), np.concatenate(alld)
print("\nmass histogram (log10 bins), all groups, split by n_hard == 0:")
edges = np.array([0, 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400])
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (m >= lo) & (m < hi)
    if sel.sum():
        print(f"  [{lo:6.1f},{hi:6.1f})  n={sel.sum():7d}  frac_hard0={(h[sel]==0).mean():.3f}  "
              f"median min_d={np.median(d[sel])*100:5.1f} cm")
# threshold: largest mass at which 99% of the frames below it have no vertex within 1 cm
cands = np.arange(0.5, 60, 0.5)
frac = np.array([(h[m < c] == 0).mean() if (m < c).sum() > 100 else 1.0 for c in cands])
thr = float(cands[np.where(frac >= 0.99)[0].max()])
print(f"\nchosen ZERO_MASS = {thr}  (99% of frames with mass below it have n_hard == 0)")
for c in (thr / 2, thr, 2 * thr):
    z = (m < c) & (h == 0)
    print(f"  thr {c:5.1f}: zero frames {z.mean()*100:5.2f}%   frames with n_hard==0 but mass>=thr: "
          f"{((h == 0) & (m >= c)).mean()*100:5.2f}%")
C.write_json(C.CACHE / "zero_threshold.json", dict(zero_mass=thr, rule="99% of frames below have n_hard==0",
                                                   hard_mm=C.HARD_MM))
D.to_csv(C.OUT / "exp1" / "mass_distribution.csv", index=False)
