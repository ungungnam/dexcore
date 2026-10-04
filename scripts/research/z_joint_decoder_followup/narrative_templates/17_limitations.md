- **One seed.** Every model has one training seed. The intervals describe the test takes, not the training randomness. The TACO
  difference between M2 and M1 is close to the threshold, and a second seed could move it to either side. The rule for an extra seed
  concerned the comparison with M0 and did not fire (the difference is <<p_taco_M2_vs_M0_E_C_rel:+.1%>> and <<p_arctic_M2_vs_M0_E_C_rel:+.1%>>).
- **M1 and M2 differ in three things at once.** Capacity, budget and selection were changed together, as the plan describes M2.
  The sensitivity row shows that the TACO gain is absent at a matched step. It does not cleanly separate selection from capacity,
  and capacity and budget are not separated from each other. The control run with a 4-block decoder was reserved for the case that
  M2 beat M1 by the rule, and it did not.
- **The decoder width stayed 256.** The plan recommends 512 to 768. A wider decoder cannot load the Stage-1 weights and would cost
  about four times as much per step. The teacher-latent floor moved by <<p_taco_D_M2_oracle_vs_D_M1_oracle_E_C_rel:.1%>> and <<p_arctic_D_M2_oracle_vs_D_M1_oracle_E_C_rel:.1%>> with two more blocks, which makes capacity an
  unlikely limit, but a wider decoder was not run.
- **The dense loss uses 4 random frames per sequence per step.** It is an unbiased estimate of the all-frame loss. Validation and
  test decode all 63 frames.
- **The held-out probes are small.** They use 141 and 106 validation sequences. Variant A chose zero steps, most likely because it overfits; variant B has five parameters.
  A decoder trained on held-out predictions of the whole training set, obtained by cross-fitting the temporal network, was not run.
  It would be a new training procedure and the plan asked for one model.
- **The path was fixed.** The decoder sees ẑ, G and the geometry tokens of the object, not s_0. A decoder that also sees s_0, or a temporal network that starts from the
  encoded s_0, was not tested. The first-frame deficit is therefore part of every result here.
- **M0 was reused.** Its budget was 80 000 steps against 100 000 for M2. M2 used 25 000. <<m0r_limit_text>>
- **Shared GPUs.** Runs shared GPUs with other users and one run was moved between GPUs twice by resuming from its last checkpoint.
  Wall-clock times are not comparable. Results do not depend on the GPU.
