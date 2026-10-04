{{fig4}}

{{table5}}

**The latent prediction did not get better.** The test L_z of M2 is <<z_taco_M2_L_z_test:.3f>> against <<z_taco_M1_L_z_test:.3f>> for M1 on TACO and
<<z_arctic_M2_L_z_test:.3f>> against <<z_arctic_M1_L_z_test:.3f>> on ARCTIC; both paired differences are **<<p_taco_M2_vs_M1_L_z_verdict>>** and
**<<p_arctic_M2_vs_M1_L_z_verdict>>**. The mean per-dimension R² stays at <<z_taco_M2_r2_per_dim_mean:.2f>> and <<z_arctic_M2_r2_per_dim_mean:.2f>>. About half
of the variance of the teacher trajectory on TACO, and two thirds on ARCTIC, is not predicted. The error grows with the horizon
and flattens after frame 32 (Figure 4).

**On the training sequences the latent is predicted almost exactly.** With the same weights the RMSE on the training sequences is
<<z_taco_M2_rmse_train:.3f>> (TACO) and <<z_arctic_M2_rmse_train:.3f>> (ARCTIC), against <<z_taco_M2_rmse_val:.2f>> and <<z_arctic_M2_rmse_val:.2f>> on validation
and <<z_taco_M2_rmse_test:.2f>> and <<z_arctic_M2_rmse_test:.2f>> on test. The temporal network has about 68 M parameters and the training sets have
<<split_taco_n_train:,>> and <<split_arctic_n_train:,>> sequences. It fits them and does not carry that fit over. At M2's later selected step the gap is wider, not
narrower: M1's training RMSE at its selected step was <<z_taco_M1_rmse_train:.2f>> and <<z_arctic_M1_rmse_train:.2f>>.

**So which of the two happened: better z, or a more tolerant decoder?** Neither, in the sense the plan asked. The latent error is
unchanged and Section 12 shows that the decoder is not more tolerant. On TACO one thing did change. M2's latents are no closer to
the teacher than M1's, yet they decode to slightly better maps through either decoder (Section 12, swapped inputs). The size of
the latent error is the same; which errors remain is slightly different. That is the <<p_taco_M2_vs_M1_E_C_rel:.1%>> of Section 9.
