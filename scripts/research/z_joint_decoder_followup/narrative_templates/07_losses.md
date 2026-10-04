Two terms and nothing else.

- **Latent trajectory loss.** L_z is the mean squared error between ẑ_t and the teacher z*_t over the 63 future frames, in coordinates
  standardised with the training-set mean and standard deviation of each latent dimension.
- **Final contact loss.** L_C is the mean squared error between the decoded map D_φ(ẑ_t, G_t) and the true map, in the residual units
  of the direct model, ρ = (C − s_0) / σ_r. M0 is trained with exactly this loss on its own output.
- **Total.** L = L_C + λ_z L_z with λ_z = 1.

The decoder's input in the dense term is the predicted ẑ. The gradient of L_C therefore reaches both the decoder and, through ẑ, the
temporal network (checked at five training steps, Appendix A).

Decoding 63 frames for 128 sequences through the point decoder at every step is too expensive. As in Stage 2 the dense term uses 4
random future frames per sequence per step, an unbiased estimate. L_z uses all 63 frames, and validation decodes all 63.

**M3** adds one term for the same decoder: 0.25 times L_C of the decoded teacher latent, D_φ(z*_t, G_t). The predicted-z term keeps
weight 1. To keep the step cost at +25 % the teacher term uses one of the four sampled frames per sequence.

No R2 loss, wrench loss, relational loss, residual loss, smoothness, velocity or event loss is used. There is no r, no diffusion and no
autoregressive rollout.
