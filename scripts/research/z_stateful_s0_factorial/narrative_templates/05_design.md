{{fig1}}

{{table1}}

**What is held fixed.** Data, split, 64-frame sequences, the canonical 512-point map, s_0, G, tau, all normalisation, the frozen
Stage-1 A3 teacher and its cached latents, the decoder family and its size, the losses, the optimiser, the budget, the selection
rule, the seed, and the evaluation code.

**What changes.** Only the two factors.

- M00 is the previous follow-up's model, reused: same backbone, same 6-block decoder, same recipe and selection.
- M01 changes the decoder only. M10 changes the temporal part only. M11 changes both.
- D0, the external reference, is the direct dense model of Stage 2, reused.

**Capacity.** The whole-sequence models have <<t_taco_M00_n_params_total:,.0f>> parameters and the stateful models <<t_taco_M10_n_params_total:,.0f>>, a
difference of <<cap_gap:.1%>>. The two decoders differ by one input column (<<t_taco_M01_n_params_decoder:,.0f>> against <<t_taco_M00_n_params_decoder:,.0f>> parameters).

**Reading the design.** Each factor has two paired comparisons, one at each level of the other factor. The decoder effect is
M01 against M00 and M11 against M10. The stateful effect is M10 against M00 and M11 against M01. The main effects average the two,
and the interaction is their difference. All are computed on per-sequence values with a take-cluster bootstrap.
