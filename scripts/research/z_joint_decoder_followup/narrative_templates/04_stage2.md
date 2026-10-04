Stage 2 compared, on one shared temporal backbone, direct dense prediction (B0 = M0), z-mediated prediction (B1 = M1) and a z + r variant.

- **Dense error.** M1 against M0 was <<m_taco_M1_E_C:.3f>> against <<m_taco_M0_E_C:.3f>> on TACO, verdict **<<p_taco_M1_vs_M0_E_C_verdict>>**, and
  <<m_arctic_M1_E_C:.3f>> against <<m_arctic_M0_E_C:.3f>> on ARCTIC, verdict **<<p_arctic_M1_vs_M0_E_C_verdict>>**.
- **Structure.** On TACO, M1 was better than M0 on participation (<<p_taco_M1_vs_M0_part_hamming_rel:+.1%>>), centroid (<<p_taco_M1_vs_M0_centroid_rel:+.1%>>),
  normal (<<p_taco_M1_vs_M0_normal_rel:+.1%>>) and wrench (<<p_taco_M1_vs_M0_q_rel_l1_rel:+.1%>>). On ARCTIC no structural metric was better and the amount was worse
  (<<p_arctic_M1_vs_M0_amount_l1_rel:+.1%>>).
- **The teacher latent decodes well.** Through M1's own fine-tuned decoder the teacher z of the true frames gives a dense error of
  <<s2b_taco_B1_E_C_oracle_z:.2f>> (TACO) and <<s2b_arctic_B1_E_C_oracle_z:.2f>> (ARCTIC). The predicted z gives <<s2b_taco_B1_E_C_pred:.2f>> and <<s2b_arctic_B1_E_C_pred:.2f>>.
- **The predicted latent is far from the teacher.** Its test R², averaged over the 64 dimensions, is <<z_taco_M1_r2_per_dim_mean:.2f>> (TACO) and <<z_arctic_M1_r2_per_dim_mean:.2f>> (ARCTIC).
- **The latent models cannot copy s_0.** The decoder emits an absolute map from z alone, so in the first eight frames M1's error is
  <<h_taco_M1_dense_1_8:.2f>> against <<h_taco_M0_dense_1_8:.2f>> for M0 on TACO, and <<h_arctic_M1_dense_1_8:.2f>> against <<h_arctic_M0_dense_1_8:.2f>> on ARCTIC.

The gap between the two decoder inputs is what motivates this follow-up. It shows that the information is in z and is lost when z is
predicted. It does not show whether a better-fitted decoder could recover part of it. This study keeps the last limitation on purpose:
the plan fixes the path (s_0, G, tau) -> z -> C, so M2's decoder still sees z and the geometry, not s_0.
