The plan names a stateful or recurrent z-dynamics experiment as the next step if M2 does not improve on M1. This study did not test
it. What it shows is that the remaining error sits in the predicted z trajectory, which is the part a stateful model would change.
That makes it a reasonable next test on TACO and a weak one on ARCTIC. It is not implemented here.

**Why it is a reasonable next test.**

- The remaining bottleneck is the predicted z trajectory, and that is the one part this line of experiments has not changed.
- The diagnostic showed that on TACO a predictor that knows the current true z beats persistence at 4 and 8 frames. A model that
  carries a state could use that; the current one predicts every frame from (s_0, G, tau) without any z state. The diagnostic did
  not show that an iterative model would beat Stage 2.
- A stateful model that starts from z_0 = E_z(s_0) could reduce the first-frame deficit, the largest single loss against M0 on
  both datasets. The diagnostic measured that a probe given z_0 has <<zt_s_taco_1_start_rescue:.0%>> (TACO) and <<zt_s_arctic_1_start_rescue:.0%>> (ARCTIC) lower latent error
  than Stage 2 at the first frame, but only <<zt_s_taco_8_start_rescue:.0%>> and <<zt_s_arctic_8_start_rescue:.0%>> at frame 8. So z_0 alone is unlikely to close the gap over
  frames 1 to 16.

**Why to be cautious.**

- On ARCTIC the diagnostic found almost no predictable change of z beyond persistence, and M2 is behind M0 at every horizon. A
  stateful model has little to work with there.
- A rollout feeds its own errors back. The diagnostic measured direct prediction 1, 4 and 8 frames ahead from the true z_t, not a rollout.
- The argument of Section 18 applies to any z-mediated model: with ideal predictors, decoding alone cannot do better than direct
  prediction in dense error. The realistic target is the TACO pattern: equal dense error with better structure, without the
  early-frame loss.

**A concrete proposal for the user to decide on.**

1. Run the stateful model on TACO first, with z_0 = E_z(s_0) allowed at inference, and compare with M0 and M2 on the same test
   sequences under the rule used here.
2. Fix the success criteria in advance: no loss against M0 in frames 1 to 16, the structural advantage of M2 kept, and a lower
   test L_z than <<z_taco_M2_L_z_test:.2f>>.
3. Run ARCTIC only as a check. If the TACO criteria are not met, stop the z-mediated line for the final generator and keep z as a
   representation for analysis and structural evaluation.

Two cheaper options would also remove a limitation of this study without a new dynamics model: a decoder that additionally sees
s_0, and a decoder trained on cross-fitted held-out latents. Neither was part of the plan.
