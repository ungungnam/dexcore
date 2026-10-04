- **Losses.** L_C is the mean squared error of the decoded map in the direct model's residual units. L_z is the mean squared error
  of ẑ_1:63 against the teacher latents in standardised coordinates. The stateful models add L_z0, the same error for ẑ_0. All
  weights are 1. Nothing else: no R2, wrench, smoothness, velocity, event, relational or residual-size term.
- **Rollout training.** The stateful models are trained on their own rollout. ẑ_0 comes from I(s_0, G), every later state from the
  model's previous output, and the loss is back-propagated through all 63 steps. There is no teacher forcing and no truncation.
- **Dense term.** As in the previous studies, L_C uses 4 random future frames per sequence per step, an unbiased estimate. L_z uses
  all frames and validation decodes all 63.
- **Recipe and budget.** AdamW 1e-4, weight decay 0.01, 5 000 warm-up steps, cosine decay over 100 000 steps, batch 128, gradient
  clipping 1.0, EMA 0.999, bf16 mixed precision with the rollout state kept in float32. Validation every 1 000 steps. No stop
  before 25 000 steps, then only when neither the validation dense loss nor the full objective improved for 15 checks.
- **Selection.** The EMA state at the minimum validation L_C. No test quantity was computed before training ended.
- **Seed.** One training seed, the seed of the reused M00 and D0.

**Two decisions made before training, on validation data only.**

- *Initial-state supervision.* L_z0 as written in the plan supervises one frame per training sequence. A pilot trained the
  initial-state module alone for 2 000 steps, once on frame 0 only and once with three extra frames per sequence as additional
  inputs. The validation RMSE of ẑ_0 was <<pilot_taco_aug0_val_rmse:.3f>> against <<pilot_taco_aug3_val_rmse:.3f>> on TACO and <<pilot_arctic_aug0_val_rmse:.3f>> against <<pilot_arctic_aug3_val_rmse:.3f>> on ARCTIC,
  ratios of <<pilot_taco_ratio:.2f>> and <<pilot_arctic_ratio:.2f>>. The rule fixed beforehand was to add the extra frames only above 1.5, so the plan's L_z0 on
  frame 0 was kept. The pilot was run twice. Its first run used a module whose random initial weights coincided with the Stage-1
  encoder's (see "Restarts"); it gave ratios of <<pilot_shared_taco_ratio:.2f>> and <<pilot_shared_arctic_ratio:.2f>> and the same decision. The numbers above are from the
  rerun with an independent initialisation.
- *Transition architecture.* See Section 6.

**Restarts.** Every run reported here was trained from step 1 with the final code. Three problems were corrected after a first
launch, each before any result was read.

- *Rollout precision.* A code review that ran in parallel with the launch found that the rollout state was being accumulated in
  bf16. The runs that had started were stopped before step 500 and the state was moved to float32.
- *GPU sharing.* Two stateful runs placed on one GPU slowed each other about sixfold, because the rollout issues thousands of
  small operations and GPU time-slicing penalises that. They were stopped before their first validation. From then on each
  stateful run had a GPU to itself.
- *Initial weights of the initial-state module.* A dry run of the sanity checks on the 2 000-step checkpoints of the two M10 runs
  showed that the module had started from the same random initial weights as the Stage-1 encoder, because both are built first,
  with the same seed and architecture. No trained weight had been copied, but the plan asks for a module learned independently of
  the Stage-1 encoder. The two runs were stopped and their checkpoints set aside; they are used nowhere in this report. The module
  is now drawn from its own random stream (cosine similarity with the Stage-1 encoder's initial weights 0.0002), the pilot was
  rerun, and M10 and M11 started again from step 1. M01 has no initial-state module and was not affected.

**GPUs.** The stateful runs used GPUs 4 and 5, one run per GPU. The two M01 runs shared GPU 7 with each other and with another
user's process. Every run had a hard memory cap. Wall-clock times are not comparable between runs.

{{table12}}

{{fig9}}
