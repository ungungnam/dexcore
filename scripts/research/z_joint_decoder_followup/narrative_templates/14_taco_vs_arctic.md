The plan expected the two datasets to differ and they do.

| | TACO | ARCTIC |
|---|---|---|
| M2 against M1, dense | <<p_taco_M2_vs_M1_E_C_rel:+.1%>>, <<p_taco_M2_vs_M1_E_C_verdict>> (close call) | <<p_arctic_M2_vs_M1_E_C_rel:+.1%>>, <<p_arctic_M2_vs_M1_E_C_verdict>> |
| M2 against M0, dense | <<p_taco_M2_vs_M0_E_C_rel:+.1%>>, <<p_taco_M2_vs_M0_E_C_verdict>> | <<p_arctic_M2_vs_M0_E_C_rel:+.1%>>, <<p_arctic_M2_vs_M0_E_C_verdict>> |
| M2 against M0, structure and wrench | <<dec_taco_q3>> | <<dec_arctic_q3>> |
| Latent R² (mean over dimensions) | <<z_taco_M2_r2_per_dim_mean:.2f>> | <<z_arctic_M2_r2_per_dim_mean:.2f>> |
| Teacher latent through the decoder | <<d_taco_M2_oracle:.2f>> | <<d_arctic_M2_oracle:.2f>> |
| Result case | <<dec_taco_case>> | <<dec_arctic_case>> |

- **TACO.** The diagnostic found learnable dynamics of z beyond persistence here, and the plan expected decoder adaptation to
  expose a clearer benefit. The benefit over the direct model is structural: four of five metrics over the whole sequence, arising
  from frame 17 onward (Table 7b). In the first 16 frames the structural metrics are not better than M0's, like the dense error.
  This benefit was already present in M1. The adaptation lowered the dense error by <<p_taco_M2_vs_M1_E_C_rel:.2%>> (similar by the rule); the
  point estimate is now level with M0's, where M1's was <<p_taco_M1_vs_M0_E_C_relchange:.1%>> higher with an interval that included zero.
- **ARCTIC.** The diagnostic found z smooth with little predictable change beyond persistence, and the plan expected that M2 might
  improve on M1 and still not beat M0. It did not improve on M1 and it does not beat M0. The latent is harder to predict here
  (R² <<z_arctic_M2_r2_per_dim_mean:.2f>>), and, as the diagnostic study reported, 21 of the <<split_arctic_n_test_takes>> test takes are of a subject that is not in the training set.

The two datasets agree on the mechanism. In both, the decoders are interchangeable on predicted latents, the latent error is
unchanged, and the teacher latent decodes about three times better than the predicted one.
