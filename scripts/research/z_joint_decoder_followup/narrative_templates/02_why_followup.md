Stage 2 found that predicting through z did not beat direct dense prediction, yet the teacher latent decoded well. The diagnostic that
followed found nothing wrong with the temporal geometry of z. That leaves the way z was predicted and decoded as a suspect, and this
follow-up tests the decoding side.

**One fact about the previous model changes what this experiment can test.** The plan describes M1 as a model whose decoder was
"primarily a static GT-z reconstruction decoder". The Stage-2 code shows otherwise: B1 loaded the Stage-1 decoder and then trained it end
to end, together with the temporal network, on the predicted latent, with the loss L_C + 1.0 L_z. So M1 was already a jointly trained
model, and M2 cannot differ from it by "joint versus frozen". M2 differs from M1 in exactly three things, all declared before training.

| | M1 (Stage-2 B1) | M2 (this study) |
|---|---|---|
| Decoder capacity | Stage-1 decoder, 4 point-token blocks, <<t_taco_M1_n_params_decoder:,.0f>> parameters | the same 4 loaded blocks plus 2 new blocks, <<t_taco_M2_n_params_decoder:,.0f>> parameters |
| Optimisation budget | 80 k-step schedule, stop allowed after 20 k steps; TACO ran <<t_taco_M1_steps:,.0f>> steps, ARCTIC was stopped by hand at <<t_arctic_M1_steps:,.0f>> | 100 k-step schedule, no stop before 25 k steps, patience 15, no manual stop |
| Checkpoint selection | minimum of the validation objective L_C + L_z | minimum of the validation dense loss L_C, the criterion M0 uses |

Each of the three could plausibly have hidden a decoder problem in Stage 2.

- **Selection.** M1's selected state came from step <<t_taco_M1_best_step:,.0f>> on both datasets, inside the 5 000-step learning-rate warm-up. At that
  state the decoder had moved by <<decchg_taco_M1:.1%>> (TACO) and <<decchg_arctic_M1:.1%>> (ARCTIC) of its weight norm from the Stage-1 weights.
  The plan's description of M1 as a decoder close to the Stage-1 one can be checked against these two numbers.
- **Budget.** The direct model kept improving for <<t_taco_M0_best_step:,.0f>> (TACO) and <<t_arctic_M0_best_step:,.0f>> (ARCTIC) steps. No latent model was trained
  beyond 20 000.
- **Capacity.** The plan asks for a decoder that is not "intentionally small".

If M2 improves on M1, at least one of the three mattered. If it does not, none of them was the reason M1 lost to M0.
