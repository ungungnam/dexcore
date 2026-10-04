- **Direct generation.** The deterministic dense model learns substantial structure. Supervising R2 outputs did not improve it and
  sequence diffusion added jitter without improving typical quality. More stochasticity or harder structural supervision is not the
  missing piece.
- **Stage 1.** A single-frame autoencoder gave a 64-D structure latent z that keeps the R2 and wrench information, organises
  structural neighbourhoods and controls structure in swaps. Its companion code r was not a persistent sequence identity. This
  experiment uses z only.
- **Stage 2.** Predicting through z did not beat direct prediction. TACO: similar dense error, better structure. ARCTIC: worse dense
  error, no structural benefit.
- **z temporal diagnostic.** The latent is not temporally pathological: its change tracks the change of contact (rank correlation
  <<zt_mean_spearman_taco:.2f>> and <<zt_mean_spearman_arctic:.2f>>). Given the true current z, a learned predictor beats persistence at 4 and 8 frames on TACO
  (<<zt_p_taco_4_probe_gain_rmse:.0%>> and <<zt_p_taco_8_probe_gain_rmse:.0%>>) and barely on ARCTIC (<<zt_p_arctic_4_probe_gain_rmse:.0%>> and <<zt_p_arctic_8_probe_gain_rmse:.0%>>).
- **Joint-decoder follow-up.** A larger decoder, a longer budget and final-task selection did not improve the z model by the rule
  (TACO <<zj_m_taco_M1_E_C:.3f>> to <<zj_m_taco_M2_E_C:.3f>>, ARCTIC <<zj_m_arctic_M1_E_C:.3f>> to <<zj_m_arctic_M2_E_C:.3f>>). The latent of the training sequences is fitted
  almost exactly (RMSE <<zj_z_taco_M2_rmse_train:.2f>> and <<zj_z_arctic_M2_rmse_train:.2f>>) and the latent of new sequences is not (<<zj_z_taco_M2_rmse_test:.2f>> and
  <<zj_z_arctic_M2_rmse_test:.2f>>). In the first eight frames the z model loses <<zj_early_gap_taco_M2:.2f>> and <<zj_early_gap_arctic_M2:.2f>> of dense error against the
  direct model, which starts from s_0.

That follow-up's model is this study's M00.
