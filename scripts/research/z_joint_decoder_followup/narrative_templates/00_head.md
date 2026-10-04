# z joint-decoder follow-up: did adapting the decoder rescue z-mediated temporal generation?

{{table8}}

**Short answer: no.** A larger decoder, a longer budget and selection on the final-task loss did not improve the z-mediated model by
the pre-registered rule. On TACO the dense error fell by <<p_taco_M2_vs_M1_E_C_rel:.2%>> against the previous model, just under the 2 % that the
rule requires. Its point estimate is now level with the direct model's (<<m_taco_M2_E_C:.3f>> against <<m_taco_M0_E_C:.3f>>), where M1's was
<<p_taco_M1_vs_M0_E_C_relchange:.1%>> higher, itself an unresolved difference, and the structural advantage is kept. On ARCTIC nothing changed
(<<m_arctic_M2_E_C:.3f>> against <<m_arctic_M1_E_C:.3f>>) and the model stays <<p_arctic_M2_vs_M0_E_C_relchange:.0%>> behind the direct model. The changes made
to the decoder did not move the error: the old and the new decoder reach the same dense error from the same predicted latent, within
half a percent, and the clean teacher latent still decodes three times better than the predicted one. A decoder trained on many
held-out predictions was not tested.

Four things to know before reading the rest.

1. **The previous model was already jointly trained.** The Stage-2 B1 fine-tuned its decoder on the predicted latent with L_C + L_z.
   M2 therefore differs from M1 in decoder capacity, optimisation budget and checkpoint selection, not in "joint versus frozen"
   (Section 2).
2. **The TACO verdict against M1 is a close call.** The difference is <<p_taco_M2_vs_M1_E_C_diff:+.3f>> [<<p_taco_M2_vs_M1_E_C_lo:+.3f>>, <<p_taco_M2_vs_M1_E_C_hi:+.3f>>]: its
   interval excludes zero and its size is <<p_taco_M2_vs_M1_E_C_rel:.2%>>, below the 2 % threshold. The rule says "similar" and the rule was not
   moved. Most of the difference appears between step 4 000 and the later selected step of the same run; it is not present at a
   matched step (Section 9).
3. **Joint training shows the decoder test-sized latent errors only in roughly its first thousand steps.** The training L_z is
   <<trainLz_taco_M2_1000:.2f>> and <<trainLz_arctic_M2_1000:.2f>> at step 1 000 and <<trainLz_taco_M2_2000:.2f>> and <<trainLz_arctic_M2_2000:.2f>> at step 2 000, against <<z_taco_M2_L_z_test:.2f>> and <<z_arctic_M2_L_z_test:.2f>> on test. At the selected state the
   predicted latent of the training sequences is almost the teacher latent (RMSE <<z_taco_M2_rmse_train:.2f>> and <<z_arctic_M2_rmse_train:.2f>>); on held-out
   sequences the error is <<zratio_taco_M2:.0f>> and <<zratio_arctic_M2:.0f>> times larger. Two small post-hoc probes that fit the decoder side to validation latents
   found no gain (Section 12).
4. **One seed.** Intervals are take-cluster bootstraps over the test takes. M3 (optional) and the budget-parity retraining of M0
   were stopped at the user's request before they finished; what their validation logs showed is in Sections 9 and 10.
