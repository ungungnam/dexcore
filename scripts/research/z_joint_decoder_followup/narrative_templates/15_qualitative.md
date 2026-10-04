The sequences were picked by fixed rules on the per-sequence test errors (figures/fig6_selection_*.csv): the largest gain of M2
over M1, the largest loss of M2 against M0, and the sequence at the 90th percentile of M2's error. Rows are the true maps, M0, M1 and
M2 at seven frames; the bottom row shows participation, dense error, total contact amount, wrench error and frame-to-frame change.

**1. TACO, where M2 helps most against M1** (brush / roller / plate). The contact grows and moves to the upper edge after frame 16.
M0 keeps the initial pattern and drifts to the wrong side. M1 and M2 both put the new contact in the right region; M2's is closer
in the last 20 frames.

{{fig6_taco_helps}}

**2. ARCTIC, where M2 helps most against M1** (capsule machine, subject s01). M2 is closer than M1 and both remain far from M0.

{{fig6_arctic_helps}}

**3. ARCTIC, where M2 loses most against M0** (phone, subject s10, a subject not in the training set).

{{fig6_arctic_fails}}
