Three failure patterns recur. None of them is specific to M2: M1 shows the same ones in the same sequences.

- **The first generated frame does not continue s_0.** In the two ARCTIC examples of this section and the last, M2's dense error
  at frame 1 is already <<q_arctic_fails_e1_M2:.1f>> and <<q_arctic_typical_e1_M2:.1f>>, where M0's is <<q_arctic_fails_e1_M0:.1f>> and <<q_arctic_typical_e1_M0:.1f>>. The
  frame-to-frame change has a spike at the first frame. The latent path rebuilds the map from ẑ and G and has no access to the
  initial map itself.
- **The contact amount is wrong.** In the phone sequence M2's total contact peaks at <<q_arctic_fails_amt_peak_M2:.2f>> where the data
  peak at <<q_arctic_fails_amt_peak_gt:.2f>>, and over the last 16 frames it averages <<q_arctic_fails_amt_end_M2:.2f>> against <<q_arctic_fails_amt_end_gt:.2f>> in the
  data.
- **Contact appears in a plausible but wrong region.** With a wrong ẑ the decoder produces a clean, confident map of a different
  grasp. It does not blur. This is the same behaviour the perturbed-latent diagnostic measures.

**A representative large-error case** (ARCTIC, ketchup, subject s10; the 90th percentile of M2's error):

{{fig6_arctic_typical}}

Both ARCTIC failure examples are of the unseen subject. The selection rules did not look at the subject.
