<<san_n_pass>> of <<san_n>> checks pass (not passing: <<san_not_pass>>). Checks 1 to 12 are the plan's; 13 to 15 were added.

{{table11}}

Notes.

- **Check 3.** At the selected state the decoder of M2 has moved by <<decchg_taco_M2:.1%>> (TACO) and <<decchg_arctic_M2:.1%>> (ARCTIC) of its weight
  norm from the Stage-1 weights, against <<decchg_taco_M1:.1%>> and <<decchg_arctic_M1:.1%>> for M1. These are the loaded Stage-1 weights only. Table 2 reports the change of all decoder weights, the two new blocks included,
  from their initial values (<<t_taco_M2_decoder_rel_change_at_best:.1%>> and <<t_arctic_M2_decoder_rel_change_at_best:.1%>>). The two new blocks left the identity.
- **Check 9.** The script that scores M2 re-scored the Stage-2 prediction files of B0 and B1 and reproduced every stored
  per-sequence metric exactly.
- **Check 11.** Both M2 runs stopped by the patience rule at 25 000 steps. No run of this study was stopped by hand for its result.
  One run (ARCTIC M3) was interrupted twice to move it to a less busy GPU and resumed from its last checkpoint.
