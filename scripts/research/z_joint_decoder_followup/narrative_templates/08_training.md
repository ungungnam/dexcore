- **Recipe.** AdamW, learning rate 1e-4, weight decay 0.01, 5 000 warm-up steps, cosine decay over 100 000 steps, batch 128 sequences,
  gradient clipping 1.0, EMA 0.999, bf16 mixed precision. The EMA weights are validated every 1 000 steps on the whole validation set.
- **Budget.** At most 100 000 steps. No stop before 25 000 steps. After that a run stops only when neither the validation dense loss
  nor the full objective has improved for 15 checks. No run was stopped by hand for its result.
- **Selection.** The saved state is the EMA state at the minimum validation L_C. The state at the minimum of the full objective, M1's
  criterion, is saved too and reported as a sensitivity row. No test quantity was computed before training ended.
- **Seed.** One training seed, the seed of the reused M0 and M1. Uncertainty is a take-cluster bootstrap over the test takes.
- **M0.** The Stage-2 B0 checkpoint is reused. Its backbone, inputs, loss and recipe are identical; its schedule was 80 000 steps, of
  which TACO used all and ARCTIC <<t_arctic_M0_steps:,.0f>> before its own patience rule stopped it.
- **Shared GPUs.** Every run had a hard memory cap and shared its GPU with other users' jobs, so wall-clock times are not comparable
  between runs.
- **One restart.** The first launch was stopped by hand a few minutes in, before step 500, because a code review found a bug in the
  gradient-inspection routine that would have crashed every latent run at step 500. Nothing had been validated or saved. All runs
  reported here started from step 1 with the fixed code.

{{table2}}

{{fig7}}
