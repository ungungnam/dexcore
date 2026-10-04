**What was tested.** Whether the Stage-2 z-mediated model lost to direct prediction because its decoder was not fitted to the
latents it actually receives.

**What was found.**

1. Giving the decoder more capacity, more time and final-task selection did not improve the model by the pre-registered rule.
   On TACO the dense error fell by <<p_taco_M2_vs_M1_E_C_rel:.2%>>, just under the threshold; on ARCTIC it did not move.
2. The decoder changes did not move the error on predicted latents. Old and new decoders reach the same dense error from the
   same predicted latent, within half a percent.
3. The latent prediction did not improve. Test R² is <<z_taco_M2_r2_per_dim_mean:.2f>> and <<z_arctic_M2_r2_per_dim_mean:.2f>>, as before.
4. The teacher latent still decodes far better than the predicted one: <<d_taco_M2_oracle:.2f>> against <<d_taco_M2_pred:.2f>> on TACO and
   <<d_arctic_M2_oracle:.2f>> against <<d_arctic_M2_pred:.2f>> on ARCTIC. Random latent errors of the same size reproduce most of that gap.
5. Fitting the decoder side to held-out latents did not help in the two small probes that were possible.

**What it means.** The Stage-1 z is not shown to be at fault, and none of the decoder changes tested here (capacity, budget,
selection, two small held-out probes) helped. What remains is the prediction of the z trajectory on new sequences. The temporal
network reproduces the training trajectories and predicts about half (TACO) or a third (ARCTIC) of the variance of new ones, and a
decoder that is faithful to its input passes that error on. The study cannot separate the latent error from the decoder's
sensitivity to it, because a decoder trained on held-out predictions was not tested.

There is also an argument, not a measurement, for a limit on the decoder side. The predicted latent is a function of (s_0, G, tau).
The decoder's other inputs are G and per-frame geometry tokens of the same object, so it has no information about the contact that
the inputs do not hold. An ideal decoder therefore cannot have a lower expected error than an ideal direct predictor of the same
inputs. For trained models this is not a guarantee: M2 is already better than M0 on the late TACO frames, and the decoder reads
point-level geometry that the direct model only sees through G. It does mean that decoder adaptation by itself is not expected to
beat a well-trained direct model in dense error. An advantage has to come from z being an easier or better-structured target. On
TACO that is what the structural metrics show. On ARCTIC there is no such advantage.

**Is z mediation justified for the final generator?** On TACO only for structure: equal dense error, better participation,
centroid, normal and wrench, better late frames, worse early frames. On ARCTIC no. This is the same answer as after Stage 2. The
decoder changes tested here (two more blocks at width 256, a longer budget, selection on the dense loss, one seed) did not change
it. A wider decoder and a decoder trained on held-out predictions remain untested.
