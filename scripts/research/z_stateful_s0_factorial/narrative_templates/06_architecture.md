**Whole-sequence z (M00, M01).** The Stage-2 backbone, unchanged: one token per future frame from the standardised s_0 and the
local object trajectory, width 768, 8 transformer blocks, 12 heads, FiLM on s_0 and G. A linear head gives ẑ_1:63 in one pass.

**Stateful z (M10, M11).** Two modules.

- *Initial state.* ẑ_0 = I(s_0, G). I is a point-token contact encoder over the 512 canonical points: each token carries the
  point's geometry at frame 0 and its s_0 value, and G conditions every block. It has the architecture of the Stage-1 encoder and
  is trained from scratch inside the model. No Stage-1 encoder weight is used, and the Stage-1 encoder is not part of inference.
- *Transition.* ẑ_t+1 = ẑ_t + F(ẑ_t, G, tau_t .. tau_t+4). F is a residual MLP of width 1 024 with 6 blocks; G modulates every
  block and the local trajectory features of frames t to t+4 (look-ahead k = 4, not swept) enter with the state. Each of these
  features is the same tau_local the whole-sequence models receive, itself a summary of object states from 8 frames before to 8
  frames after its frame, so the transition at step t sees object motion from t-8 to t+12 and nothing about hands or contact. The update is read from a
  zero-initialised layer, so the model starts as latent persistence. The same F is applied 63 times to its own output.

The plan allows "multi-head attention or a strong residual MLP" for the transition. A transformer over the tokens
[state, G, trajectory] of the same size was measured before training at 3.1 times the rollout time and 6 times the memory, because
63 sequential steps are dominated by per-operation overhead. The residual MLP was used. Its width is above the recommended 512 to
768 so that the stateful models stay within 15 % of the whole-sequence models' parameter count.

**Absolute decoder (M00, M10).** The decoder of the previous follow-up: a transformer over the 512 point tokens (the frame's
geometry), conditioned on [G, ẑ_t]; the four Stage-1 decoder blocks, loaded, plus two blocks that start as the identity.

**s_0-preserving decoder (M01, M11).** The same decoder with two changes. Every point token also carries the value of s_0 at that
point, so the decoder has explicit access to s_0, ẑ_t and G_t from its first layer. And its output is the change of the map:
Ĉ_t = s_0 + ΔĈ_t. The new input column and the output layer start at zero, so the model starts as contact persistence, Ĉ_t = s_0.
No residual-size penalty is used.

The teacher latents z*_t = E_z(C_t, G_t) come from the frozen Stage-1 encoder, are cached, and are training targets only.
