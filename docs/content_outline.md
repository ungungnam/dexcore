# Content outline

The page is one argument, not an experiment log:

```
MOTIVATION  ->  FINDING 1: what varies?  ->  FINDING 2: what matters?  ->  FINDING 3: how to model what matters?  ->  CURRENT HYPOTHESIS
```

Block codes (`F1-A`, ...) are the `data-evidence` attributes in `index.html` and the keys of
`evidence_manifest.json`. Report paths are relative to the result root (`/result/uhnam/dexcore`).

## Motivation — How do we generate better hand motion?

**Message.** In the two-stage generator we studied (BimArt: contact stage, then motion stage), contact
prediction is a major source of the downstream hand error.

| block | evidence | datasets | source |
|---|---|---|---|
| `MOTIVATION` | Same motion model conditioned on the recorded contact map vs the contact stage's prediction: hand-keypoint error rises on all four TACO test splits (37.1 → 58.6, 48.8 → 86.5, 63.4 → 104.9, 50.9 → 79.8 mm); released BimArt on eight ARCTIC laptop windows shows the same direction | TACO, ARCTIC | `taco/30_bimart_gen3_scene_scale/evaluation/`, `arctic/10_bimart_upstream_analysis/contact_ablation/` |

No percentage "share of error due to contact" is quoted: the sources say the stage contributions do not add.

Ends with: *What should a temporal contact generator actually model?*

## Finding 1 — What varies?

**Claim.** A contact trajectory is an initial grasp mode plus a structured temporal evolution; the
evolution mixes structural change with reconfiguration that leaves the modelled wrench capability
nearly unchanged.

| block | role | evidence | datasets | source report |
|---|---|---|---|---|
| `F1-A` | main A | On TACO the contact amount changes more within a take than between takes (2.12) while the normalized pattern does not (1.04); the pattern changes less within a take than across meshes (TACO, time / mesh 0.69) or object instances (OakInk2, 0.36), and only slightly less than across subjects on ARCTIC (0.93; 6 of 22 groups above 1); identity ratios 1.48 (TACO, mesh / take), 1.16 (ARCTIC, subject / same-subject take), 1.69 (OakInk2, instance / take, 17 of 39 groups) | TACO, ARCTIC, OakInk2 | dynamic contact reports + `reports/dynamic_contact_crossdataset.md` |
| `F1-B` | main B | `p(C | G, τ) ≈ p(s0 | G) · p(C | s0, G, τ)`: TACO true s0 held 3.48 → evolved 1.82; best of ten sampled s0 1.86 (picked with the ground truth). ARCTIC and OakInk2: the evolution adds little beyond holding the true s0 (paired intervals contain zero); one sampled s0 adds +1.56 / +2.85 to the true-s0 rollout error of 2.14 / 1.24, so the initial map is the larger part of the error on OakInk2 but not on ARCTIC | TACO, ARCTIC, OakInk2 | `reports/hier_contact_gen_report.md` |
| `F1-C` | main C | Spike transitions (12.6 % / 5.7 % / 9.3 % of transitions) hold 70 % / 62 % / 72 % of the squared change; on spatial-dominant changes the evolved model is no better than holding the map (per-model intervals, no paired test); clearest gain on amount-dominant changes (TACO); most of the summed temporal error is on non-spike frames | TACO, ARCTIC, OakInk2 | `reports/temporal_contact_events/` |
| `F1-subA` | sub A | Hand speed relative to the object is associated with contact change at the same transition (Spearman 0.78 TACO with the corrected transform, 0.51 ARCTIC) | TACO, ARCTIC | `reports/temporal_contact_events/`, `reports/hand_contact_predictive_info/` |
| `F1-subB` | sub B | Current hand pose: no gain; hand history up to the current frame: small gain; future-hand oracle: larger gain; past hand frames add little to predicting when a persistent transition starts | TACO, ARCTIC | `reports/hand_contact_predictive_info/` |
| `F1-D` | main D | Persistent spatial events: median retention of the wrench profile 0.85 / 0.89, cosine 0.97 / 0.96; events that change the amount or the set of touching hand parts change capability more often | TACO, ARCTIC | `reports/wrench_counterfactual/` |

**Conclusion.** Temporal contact evolution contains both meaningful structural evolution and
function-preserving realization variation. → *Which part of the dense contact state actually matters?*

## Finding 2 — What matters?

**Claim.** Dense contact contains a compact structural state: R2 (participation, amount, centroid,
mean normal; 48 numbers per frame) is close to the smallest representation that keeps both the
modelled wrench capability and the prediction of future structured contact.

| block | role | evidence | datasets | source report |
|---|---|---|---|---|
| `F2-A` | main A | Ladder R0 … FULL: the wrench-probe error drops sharply at R2 and is flat from R2 to R5 | TACO, ARCTIC | `reports/structure_variance_boundary/` |
| `F2-B` | main B | The future structured contact is predicted from R2 with 18 % / 32 % lower error than from the full dense state; against the full dense state R2 is the smallest level saturated on both axes at the 2.5 % and 5 % tolerances (R1 on TACO at 10 %) | TACO, ARCTIC | same |
| `F2-C` | main C | The residual beyond R2 is 11–15 % (TACO) and 26–31 % (ARCTIC) of the dense variance; it is persistent, but its change is weakly predictable beyond persistence | TACO, ARCTIC | same |
| `F2-subA` | sub A | Coarse hand pose adds little | TACO, ARCTIC | same |
| `F2-subB` | sub B | Topology (spread, patch count) helps the wrench match slightly, not the temporal prediction | TACO, ARCTIC | same |

**Conclusion.** Under the tested probes the full dense map gave no advantage over the compact state for
the wrench capability or for future structured contact; the remaining dense realization is persistent
but weakly determined. → *Can this boundary be learned instead of manually
encoded?*

## Finding 3 — How should we model what matters?

**Claim.** The structural state can be learned as a latent z, but its dynamics remain the likely
bottleneck. One modelling sequence (steps A to D) with one supporting-evidence area, not several findings:

| step | block | evidence | datasets | source report |
|---|---|---|---|---|
| A | `F3-A` | Explicit R2 supervision (output loss, hidden head; the one surrogate tested) does not improve the dense generator; a diffusion future with s0 fixed adds dense jitter; sampling s0 gives more structural spread than sampling the future | TACO, ARCTIC | `reports/structure_aware_temporal_generation/` |
| B | `F3-B` | Learned z / r factorization: z keeps the probed structure, r completes the dense reconstruction, structure follows the z donor in 97 % / 98 % of swaps (preferential, not disentangled; detail follows r in 77 % / 44 %) | TACO, ARCTIC | `reports/contact_factorization_stage1/` |
| C | `F3-C` | z-mediated vs direct generation: TACO dense tie with better structure; ARCTIC 14 % worse dense error, no structural gain; predicted r adds nothing | TACO, ARCTIC | `reports/contact_latent_temporal_stage2/` |
| D | `F3-diagA` | The true z decodes at 0.51 / 0.71 against 1.73 / 2.16 for the predicted z | TACO, ARCTIC | same |
| D | `F3-diagB` | Δz follows ΔC (Spearman 0.85 / 0.81); low-ΔC / high-Δz transitions are rare (0.2 % / 0.6 %) | TACO, ARCTIC | `reports/z_temporal_diagnostic/` |
| D | `F3-diagC` | Local z prediction beats persistence on TACO (3.3 / 11.3 / 17.8 % at 1 / 4 / 8 frames); ARCTIC is persistence-dominated (−2.6 / 0.2 / 6.9 %) | TACO, ARCTIC | same |
| sub A | `F3-subA` | A larger decoder with a longer schedule does not help (dense error “similar” by the study's rule); old and new decoder agree within 0.5 % on the same predicted latent | TACO, ARCTIC | `reports/z_joint_decoder_followup/` |
| sub B | `F3-subB` | The latent trajectory is fitted on training sequences (RMSE 0.09 / 0.06) and not on held-out ones (validation / test 0.63 / 0.70 on TACO, 0.80 / 0.72 on ARCTIC) | TACO, ARCTIC | same |

**Conclusion.** A learned structural state z exists and is temporally meaningful; the evidence points
to how the latent trajectory is evolved and generalized from the initial observation. The test sets
differ too: the ARCTIC one includes a subject absent from training, the TACO one does not.
→ *How should the learned structural state evolve?*

## Current hypothesis (not a finding)

| block | content | status | source |
|---|---|---|---|
| `HYP` | A: carry z as a state (`z_t → z_t+1`) instead of predicting the whole trajectory in one pass. B: keep the known first map when decoding (`C_t = s0 + D_Δ(s0, z_t, G)`). 2 × 2 experiment M00 / M01 / M10 / M11 | in progress, no result reported | `scripts/research/z_stateful_s0_factorial/` |

## Qualitative figures

3-D renders of the real object, the recorded hand and the contact, one per block where a picture helps;
each example is chosen by a stated rule and is an illustration, not a statistic (`docs/build/qual/`).

| block | figure | datasets | needs |
|---|---|---|---|
| `MOTIVATION` | one test window: recorded vs predicted contact, and the hand points the motion stage generates from each | TACO | inference (BimArt port, contact + motion stage) |
| `F1-PRIMER` | what a contact map is: hand on object, map on the surface, the 512 stored values, six frames | TACO | recorded data |
| `F1-B` | initial grasp modes: recorded first contacts and ten sampled first maps on the scissors | ARCTIC | saved samples |
| `F1-B` | sample, then evolve, on the real object | TACO | saved predictions |
| `F1-C` | an amount-dominant, a spatial-dominant and a mixed change of the timeline sequence | TACO | saved predictions |
| `F1-D` | reconfigurations with the hand visible: a median-retention event and an event in which a part joins | TACO, ARCTIC | recorded data, event caches |
| `F2-PRIMER` | the 48 numbers of R2 drawn on a real grasp | TACO, ARCTIC | feature cache |
| `F3-B` | z-only map, what r adds, and a latent swap | TACO, ARCTIC | saved maps + inference (Stage-1 swap) |
| `F3-C` | recorded, direct, via the predicted z, via the true z | TACO, ARCTIC | saved predictions + inference (true-z decode) |
| `F3-diagB` | contact step and latent step over one sequence | TACO, ARCTIC | cached latents |

