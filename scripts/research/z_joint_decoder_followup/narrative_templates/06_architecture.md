{{fig1}}

M2 is one network with two parts.

- **Temporal network F_θ.** The Stage-2 backbone, unchanged: one token per future frame built from the standardised s_0 and the local
  object trajectory, width 768, 8 transformer blocks, 12 heads, learned positions, FiLM conditioning on s_0 and G in every block,
  <<t_taco_M2_n_params_backbone:,.0f>> parameters. A linear head gives ẑ_t for all 63 future frames at once. There is no rollout and no recurrent state.
  F_θ encodes s_0 itself; there is no separate encoder of s_0 and no Stage-1 encoder anywhere in the model.
- **Contact decoder D_φ.** A transformer over the 512 canonical points. Each point token carries its position, normal and static
  per-point features of the frame; the pair (G, ẑ_t) conditions every block by FiLM. The first four blocks are the Stage-1 structure
  decoder, loaded from the A3 checkpoint. Two more blocks of the same kind follow. Their gates start at zero, so at initialisation D_φ
  is exactly the Stage-1 decoder (checked, Appendix A). Every decoder weight is trained; nothing is frozen.

The plan recommends a decoder width of 512 to 768. The width here stays 256 for two reasons. A wider decoder cannot load the Stage-1
weights, which the plan prefers. And the decoder runs over 512 point tokens per map, so its cost grows with the square of the width; at
width 512 one step would cost about four times as much on GPUs shared with other users. The teacher-latent floor in Section 12 shows
whether capacity is the binding limit.

The teacher z*_t = E_z(C_t, G_t) is computed once with the frozen Stage-1 encoder and cached. It is a training target only.

{{table1}}

M0 and M2 share the backbone, its inputs and the dense loss. Total size is <<t_taco_M2_n_params_total:,.0f>> parameters for M2 against
<<t_taco_M0_n_params_total:,.0f>> for M0 on TACO.
