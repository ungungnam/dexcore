The z temporal diagnostic froze the Stage-1 encoder and asked whether z itself is a poor temporal representation.

- **z moves when contact moves.** The rank correlation between the dense step and the latent step is <<zt_mean_spearman_taco:.2f>> (TACO) and
  <<zt_mean_spearman_arctic:.2f>> (ARCTIC).
- **Pathological transitions are rare.** Transitions where contact barely changes and z jumps are <<zt_mean_quadB_taco:.1%>> and <<zt_mean_quadB_arctic:.1%>>
  of the test transitions.
- **On TACO a local predictor beats persistence.** Given the true z_t, a learned predictor of z four and eight frames ahead beats
  "hold z_t" by <<zt_p_taco_4_probe_gain_rmse:.1%>> and <<zt_p_taco_8_probe_gain_rmse:.1%>> in RMSE.
- **On ARCTIC z is smooth but adds little beyond persistence.** The same gains are <<zt_p_arctic_4_probe_gain_rmse:.1%>> and <<zt_p_arctic_8_probe_gain_rmse:.1%>>.

The diagnostic rejected "z has poor temporal geometry". It did not decide whether the Stage-2 failure came from the open-loop formulation or
from weak dynamics beyond persistence. This follow-up tests a third candidate that the diagnostic did not touch: the decoder.
