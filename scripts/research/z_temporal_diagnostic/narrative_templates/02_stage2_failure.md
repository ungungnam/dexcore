The object being diagnosed is the latent path of Stage-2 B1, `(s₀, G, τ₁:₆₃) → ẑ₁:₆₃ → D_z → Ĉ₁:₆₃`, evaluated on the fixed test split with the GT initial map. Three facts from that report (the first two re-computed here from its saved predictions, Table 6) define the failure:

1. **The predicted latent explains a limited part of the teacher latent.** Over the 63 generated frames B1's standardised latent RMSE is <<s_taco_0_ctx_rmse_stage2:.3f>> on TACO and <<s_arctic_0_ctx_rmse_stage2:.3f>> on ARCTIC (explained variance <<s_taco_0_ctx_r2_stage2:.2f>> / <<s_arctic_0_ctx_r2_stage2:.2f>>).
2. **The error grows with the horizon**, from <<s2_f1_taco:.2f>> / <<s2_f1_arctic:.2f>> at the first generated frame to <<s2_end_taco:.2f>> / <<s2_end_arctic:.2f>> over the last 15 frames of the window.
3. **The decoder is not the problem.** With the teacher latent in place of the predicted one, the fine-tuned B1 decoder reaches a dense error of <<s2_oracle_taco:.2f>> / <<s2_oracle_arctic:.2f>>, against <<s2_ec_B1_taco:.2f>> / <<s2_ec_B1_arctic:.2f>> for the generated trajectories.

So the question is not "is z a bad representation" but which of two things made ẑ hard to predict.
