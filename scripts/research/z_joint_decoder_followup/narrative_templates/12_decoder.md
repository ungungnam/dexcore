{{fig5}}

{{table6}}

{{table6b}}

Each trained decoder was fed four kinds of latent on the test sequences.

**A. Teacher latent (oracle).** M2's decoder turns the teacher latent of the true frames into maps with error <<d_taco_M2_oracle:.3f>>
(TACO) and <<d_arctic_M2_oracle:.3f>> (ARCTIC). M1's decoder gives <<d_taco_M1_oracle:.3f>> and <<d_arctic_M1_oracle:.3f>>. The floor moved little with two more
blocks and longer training: <<p_taco_D_M2_oracle_vs_D_M1_oracle_E_C_rel:+.1%>> on TACO (<<p_taco_D_M2_oracle_vs_D_M1_oracle_E_C_verdict>>) and <<p_arctic_D_M2_oracle_vs_D_M1_oracle_E_C_rel:+.1%>> on ARCTIC
(<<p_arctic_D_M2_oracle_vs_D_M1_oracle_E_C_verdict>>).

**B. Predicted latent.** The model's own predicted latent gives <<d_taco_M2_pred:.3f>> and <<d_arctic_M2_pred:.3f>>. The gap to the teacher
latent is <<gap_taco_M2:.2f>> and <<gap_arctic_M2:.2f>>, which is <<gapshare_taco_M2:.0%>> and <<gapshare_arctic_M2:.0%>> of the model's error.

**C. Perturbed teacher latent.** The teacher latent was perturbed at one level, the model's own test error. Gaussian noise with the
mean and covariance of that error gives <<d_taco_M2_noisy_gauss:.2f>> and <<d_arctic_M2_noisy_gauss:.2f>>. The error trajectory of another test
sequence gives <<d_taco_M2_noisy_perm:.2f>> and <<d_arctic_M2_noisy_perm:.2f>>. These are <<share_taco_M2_perm:.0%>> to <<share_taco_M2_gauss:.0%>> (TACO) and
<<share_arctic_M2_perm:.0%>> to <<share_arctic_M2_gauss:.0%>> (ARCTIC) of the predicted-latent error. Two readings follow.

- The decoder is not robust around the teacher latent. A latent error of the size the temporal network makes costs most of the
  dense error even when it is random and unrelated to the content.
- The model's own errors cost somewhat more than random errors of the same size. They are the smaller part of the story.

**S. Swapped decoders.** The same predicted latent was decoded by both decoders. On M1's latents M2's decoder gives
<<p_taco_D_M2_zhat_M1_vs_D_M1_zhat_M1_E_C_a:.3f>> against <<p_taco_D_M2_zhat_M1_vs_D_M1_zhat_M1_E_C_b:.3f>> for M1's own decoder (TACO) and
<<p_arctic_D_M2_zhat_M1_vs_D_M1_zhat_M1_E_C_a:.3f>> against <<p_arctic_D_M2_zhat_M1_vs_D_M1_zhat_M1_E_C_b:.3f>> (ARCTIC). On M2's latents the numbers are
<<p_taco_D_M2_zhat_M2_vs_D_M1_zhat_M2_E_C_a:.3f>> against <<p_taco_D_M2_zhat_M2_vs_D_M1_zhat_M2_E_C_b:.3f>> and <<p_arctic_D_M2_zhat_M2_vs_D_M1_zhat_M2_E_C_a:.3f>> against
<<p_arctic_D_M2_zhat_M2_vs_D_M1_zhat_M2_E_C_b:.3f>>. All four differences are under half a percent, and in each M2's decoder is the slightly worse one.
On predicted latents the two decoders are interchangeable. By the pre-registered test M2's decoder is **not** more robust to
predicted-latent error, and the small TACO gain travels with the latent, not with the decoder.

The same comparison on identical perturbed teacher latents gives a different picture. With perturbations matched to M1's error,
M2's decoder is lower than M1's by <<p_taco_D_M2_noisy_gauss_of_M1_vs_D_M1_noisy_gauss_of_M1_E_C_rel:.1%>> (Gaussian, <<p_taco_D_M2_noisy_gauss_of_M1_vs_D_M1_noisy_gauss_of_M1_E_C_verdict>>) and <<p_taco_D_M2_noisy_perm_of_M1_vs_D_M1_noisy_perm_of_M1_E_C_rel:.1%>> (another sequence's error,
<<p_taco_D_M2_noisy_perm_of_M1_vs_D_M1_noisy_perm_of_M1_E_C_verdict>>) on TACO, and by <<p_arctic_D_M2_noisy_gauss_of_M1_vs_D_M1_noisy_gauss_of_M1_E_C_rel:.1%>> (<<p_arctic_D_M2_noisy_gauss_of_M1_vs_D_M1_noisy_gauss_of_M1_E_C_verdict>>) and <<p_arctic_D_M2_noisy_perm_of_M1_vs_D_M1_noisy_perm_of_M1_E_C_rel:.2%>> (<<p_arctic_D_M2_noisy_perm_of_M1_vs_D_M1_noisy_perm_of_M1_E_C_verdict>>) on ARCTIC. Table 6b also lists the
perturbations matched to M2's error. Over all eight comparisons the difference is <<pert_min:.1%>> to <<pert_max:.1%>>, and <<pert_n_better>> of <<pert_n>> are better
by the rule. So the longer-trained, larger decoder tolerates random errors around the teacher latent a little better on TACO and in part on
ARCTIC, and that tolerance does not carry over to the model's own predicted latents.

### What the decoder met in training

{{fig8}}

In its first steps the temporal network was still under-fitted and the decoder was trained on latents with test-sized errors.
The training-batch L_z was <<trainLz_taco_M2_1000:.2f>> (TACO) and <<trainLz_arctic_M2_1000:.2f>> (ARCTIC) at step 1 000, against <<z_taco_M2_L_z_test:.2f>> and
<<z_arctic_M2_L_z_test:.2f>> on test. At step 2 000 it was <<trainLz_taco_M2_2000:.2f>> and <<trainLz_arctic_M2_2000:.2f>>, and at step 3 000 <<trainLz_taco_M2_3000:.2f>> and <<trainLz_arctic_M2_3000:.2f>>. After roughly the first
thousand steps the decoder saw latents with errors well below the test level, and from step 3 000 on almost only near-teacher
latents.

At the selected state, on 160 training sequences, the predicted latent gives a dense error of <<dtr_taco_M2_pred:.3f>> (TACO) and
<<dtr_arctic_M2_pred:.3f>> (ARCTIC), and the teacher latent gives <<dtr_taco_M2_oracle:.3f>> and <<dtr_arctic_M2_oracle:.3f>>. By then the decoder was no longer being
trained on errors of the size it receives at test time. "Trained on predicted z" therefore did not amount to "adapted to the
held-out predicted-z distribution". This applies to M1 as well, less strongly: its validation latent error is <<zratio_taco_M1:.1f>> and <<zratio_arctic_M1:.1f>> times its training error, and
on the training sequences its predicted latent decodes at <<dtr_taco_M1_pred:.2f>> and <<dtr_arctic_M1_pred:.2f>>.

### Two probes that give the decoder side the held-out distribution

The plan's premise can still be tested directly: freeze the temporal network, take its latents on the validation sequences, which
carry held-out errors, and fit the decoder side to them. This was fixed in advance as a conditional post-hoc diagnostic, to be run only
if the training-set latent RMSE was under half the validation value. It was <<z_taco_M2_train_over_val_rmse:.2f>> and
<<z_arctic_M2_train_over_val_rmse:.2f>> of it. The shrinkage variant (B) was added after the first variant chose zero steps, probably from overfitting.

{{table10}}

- **A. Fine-tune the whole decoder on validation latents.** Two-fold cross-fitting over the validation takes chose the number of
  steps. The out-of-fold error rose within the first 100 steps, from <<pr_taco_M2_oof_val_E_C_unadapted:.3f>> to <<pr_taco_M2_oof_val_E_C_step100:.3f>> on TACO and from
  <<pr_arctic_M2_oof_val_E_C_unadapted:.3f>> to <<pr_arctic_M2_oof_val_E_C_step100:.3f>> on ARCTIC, and never came back. The chosen number of steps is zero on both.
  The fitting folds hold <<pr_taco_M2_fit_fold_sizes>> (TACO) and <<pr_arctic_M2_fit_fold_sizes>> (ARCTIC) sequences. Overfitting of the 8 M-parameter
  decoder is the likely cause, but the fitting-fold error was not recorded and the fine-tuning recipe (learning rate 3e-5, no
  warm-up) was not varied. This probe shows no benefit, and it cannot bound what a decoder trained on many held-out predictions
  could do.
- **B. Fit only five numbers.** One shrinkage factor of the predicted latent per horizon bin, chosen on validation from the grid
  0.5 to 1.0, pulls the latent towards the training mean. It is the simplest hedge against an uncertain latent and has only five
  parameters, so overfitting is unlikely. The best factors are <<pr_taco_M2_shrink_factors>> on TACO and <<pr_arctic_M2_shrink_factors>> on ARCTIC. The test
  error moves by <<pr_taco_M2_shrink_test_diff:+.3f>> and <<pr_arctic_M2_shrink_test_diff:+.3f>>. A scalar shrinkage of the latent does not help. Other forms of
  hedging were not tried.

**Answer to the mismatch question.** There is a mismatch in the literal sense: at the selected state the decoder is trained on
near-teacher latents and it is tested on latents with large errors. This study did not remove it. Capacity, time and selection
did not change the decoder's dense error on predicted latents (swap: within half a percent) and moved the teacher-latent floor
little; on randomly perturbed teacher latents M2's decoder is <<pert_min:.1%>> to <<pert_max:.1%>> lower (<<pert_n_better>> of <<pert_n>> comparisons better by the rule). The two small held-out probes found no gain. Whether
a decoder trained on a large set of held-out predictions would help was not tested.
