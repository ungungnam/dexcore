The two datasets entered this experiment with different prior evidence, and the plan asks for them to be read separately.

| Prior evidence | TACO | ARCTIC |
|---|---|---|
| z change tracks contact change (rank correlation) | <<zt_mean_spearman_taco:.2f>> | <<zt_mean_spearman_arctic:.2f>> |
| Gain of a learned local predictor over persistence, 4 / 8 frames ahead, from the true z_t | <<zt_p_taco_4_probe_gain_rmse:.1%>> / <<zt_p_taco_8_probe_gain_rmse:.1%>> | <<zt_p_arctic_4_probe_gain_rmse:.1%>> / <<zt_p_arctic_8_probe_gain_rmse:.1%>> |
| Previous z model against the direct model, dense error | <<zj_p_taco_M2_vs_M0_E_C_verdict>> (<<zj_p_taco_M2_vs_M0_E_C_rel:+.1%>>) | <<zj_p_arctic_M2_vs_M0_E_C_verdict>> (<<zj_p_arctic_M2_vs_M0_E_C_rel:+.1%>>) |
| Previous z model against the direct model, structure and wrench | <<zj_dec_taco_q3>> | <<zj_dec_arctic_q3>> |
| Test takes of a subject absent from training | none | 21 of <<split_arctic_n_test_takes>> |

- **TACO** has state-dependent latent dynamics that a local predictor can learn. The stateful hypothesis has its best chance here.
- **ARCTIC** is locally close to persistence. There the plan expects the s_0-preserving decoder to matter more than the stateful
  change, and allows for the direct model to stay ahead.
