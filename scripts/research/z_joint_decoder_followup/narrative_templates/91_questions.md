**Q1. Was decoder / predicted-z distribution mismatch a meaningful cause of the previous Stage-2 failure?**
Not shown. The mismatch exists. M2's decoder was trained on predicted latents of the training sequences that are almost the
teacher latents (RMSE <<z_taco_M2_rmse_train:.2f>> and <<z_arctic_M2_rmse_train:.2f>>), and the validation error is <<zratio_taco_M2:.0f>> and <<zratio_arctic_M2:.0f>> times larger. M1's training latents were
less exact (<<z_taco_M1_rmse_train:.2f>> and <<z_arctic_M1_rmse_train:.2f>>; validation <<zratio_taco_M1:.1f>> and <<zratio_arctic_M1:.1f>> times larger). No evidence was found that this
mismatch caused the failure. A larger decoder, a longer budget and final-task selection changed the dense error by
<<p_taco_M2_vs_M1_E_C_rel:.2%>> (TACO, below the threshold) and <<p_arctic_M2_vs_M1_E_C_rel:.1%>> (ARCTIC). The two decoders are interchangeable on predicted latents.
Two small probes on held-out latents found no gain. A decoder trained on a large set of held-out predictions was not tested.

**Q2. Does jointly training the decoder on predicted z improve final contact generation?**
Not by the rule. TACO <<m_taco_M1_E_C:.3f>> to <<m_taco_M2_E_C:.3f>> (<<p_taco_M2_vs_M1_E_C_verdict>>, a close call; the gain is not present at a matched step);
ARCTIC <<m_arctic_M1_E_C:.3f>> to <<m_arctic_M2_E_C:.3f>> (<<p_arctic_M2_vs_M1_E_C_verdict>>).

**Q3. After decoder adaptation, does z-mediated generation outperform direct dense prediction?**
In dense error, no: equal on TACO (<<m_taco_M2_E_C:.3f>> against <<m_taco_M0_E_C:.3f>>), <<p_arctic_M2_vs_M0_E_C_relchange:.0%>> worse on ARCTIC. In structure
and wrench, yes on TACO (four of five metrics) and no on ARCTIC.

**Q4. Does the answer differ between TACO and ARCTIC?**
Yes. TACO is Case <<dec_taco_case>>: structure better than direct, dense equal, no gain over M1 by the rule. ARCTIC is Case
<<dec_arctic_case>>: no gain over M1 and worse than direct.

**Q5. If z still fails, is the remaining bottleneck primarily z trajectory prediction?**
Most likely yes. The error appears between the teacher latent and the predicted latent: <<d_taco_M2_oracle:.2f>> and <<d_arctic_M2_oracle:.2f>> against
<<d_taco_M2_pred:.2f>> and <<d_arctic_M2_pred:.2f>> through the same decoder. The latent R² on new sequences is <<z_taco_M2_r2_per_dim_mean:.2f>> and <<z_arctic_M2_r2_per_dim_mean:.2f>> and did not
change. The study cannot separate the latent error from the decoder's sensitivity to it, because a decoder trained on held-out
predictions was not tested. A second, smaller part is the first frames, where the latent path cannot continue s_0.

**Q6. Is a stateful / recurrent z-dynamics experiment justified as the next step?**
Proposed, not shown. On TACO it is a test aimed at the remaining bottleneck, to be run with success criteria fixed in advance. Two
cheaper options are also open: a decoder that sees s_0, and a decoder trained on cross-fitted held-out latents. On ARCTIC only as a
check. None of these was implemented.
