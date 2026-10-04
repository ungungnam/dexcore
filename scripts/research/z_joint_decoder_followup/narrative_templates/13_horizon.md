{{fig3}}

{{table7}}

**TACO.** Against M0 the picture is split by horizon. M2 is worse in frames 1 to 8 (<<h_taco_M2_dense_1_8:.2f>> against <<h_taco_M0_dense_1_8:.2f>>,
<<p_taco_M2_vs_M0_E_C_h1_8_rel:+.0%>>) and in frames 9 to 16 (<<p_taco_M2_vs_M0_E_C_h9_16_rel:+.1%>>), similar in frames 17 to 32, and better in frames
33 to 48 (<<p_taco_M2_vs_M0_E_C_h33_48_rel:+.1%>>) and 49 to 63 (<<p_taco_M2_vs_M0_E_C_h49_63_rel:+.1%>>). The error of the last 16 frames is
<<p_taco_M2_vs_M0_E_C_last16_rel:+.1%>> (<<p_taco_M2_vs_M0_E_C_last16_verdict>>). The two effects cancel in the sequence mean, which is why the dense error
equals M0's. Against M1 the last 16 frames are <<p_taco_M2_vs_M1_E_C_last16_rel:+.1%>> (<<p_taco_M2_vs_M1_E_C_last16_verdict>>).

**ARCTIC.** M2 is worse than M0 in every bin. The gap shrinks with the horizon, from <<p_arctic_M2_vs_M0_E_C_h1_8_rel:+.0%>> in frames 1 to 8
to <<p_arctic_M2_vs_M0_E_C_h49_63_rel:+.1%>> in frames 49 to 63, and never closes. The last 16 frames are <<p_arctic_M2_vs_M0_E_C_last16_rel:+.1%>>
(<<p_arctic_M2_vs_M0_E_C_last16_verdict>>). Against M1 only frames 1 to 8 moved (<<p_arctic_M2_vs_M1_E_C_h1_8_rel:+.1%>>).

**Structure by horizon.**

{{table7b}}

On TACO the structural advantage over M0 arises from frame 17 onward. In frames 1 to 8 participation is
<<p_taco_M2_vs_M0_part_h1_8_rel:+.1%>> (<<p_taco_M2_vs_M0_part_h1_8_verdict>>) and centroid <<p_taco_M2_vs_M0_centroid_h1_8_rel:+.1%>> (<<p_taco_M2_vs_M0_centroid_h1_8_verdict>>); in frames 17 to 32 they are
<<p_taco_M2_vs_M0_part_h17_32_rel:+.1%>> (<<p_taco_M2_vs_M0_part_h17_32_verdict>>) and <<p_taco_M2_vs_M0_centroid_h17_32_rel:+.1%>> (<<p_taco_M2_vs_M0_centroid_h17_32_verdict>>).

**The first frames.** The direct model starts from s_0 and adds a residual, so its error at short horizons is small. The latent
path has to rebuild the whole map from ẑ and the geometry. In frames 1 to 8 that costs <<early_gap_taco_M2:.2f>> (TACO) and <<early_gap_arctic_M2:.2f>>
(ARCTIC) of dense error against M0. The plan fixes the path (s_0, G, tau) -> z -> C, so this was kept. M2 narrowed it a little
against M1 and did not remove it.

**Temporal stability.**

{{table9}}

On TACO M2's frame-to-frame change matches M0's and is closer to the data than M1's. On ARCTIC M2 moves more than the data and
its deviation is larger than M1's and M0's.
