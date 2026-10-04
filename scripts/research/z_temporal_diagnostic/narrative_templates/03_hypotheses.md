**H1 — open-loop / long-horizon problem.** z is locally predictable from the current latent state, but Stage 2 asked for the whole trajectory z₁:₆₃ from the initial map alone. If H1 holds, a predictor that is *given the current GT latent* z_t should predict z_{t+h} clearly better than persistence at short horizons, and much better than Stage 2 predicts the same frame.

**H2 — representation-dynamics problem.** z is structurally informative but its coordinates are not temporally well behaved: even given z_t the future is hard to predict, or the latent moves in ways that do not follow the contact. If H2 holds, a strong probe barely improves on persistence, or the latent step Δz is poorly aligned with the dense step ΔC (many transitions where the contact hardly changes but z jumps).

The two experiments map onto the two questions of the plan: Experiment A (local latent dynamics probe) answers "given the current GT z_t, how predictable is z_{t+h} for h = 1, 4, 8"; Experiment B (temporal geometry) answers "does z move when, and as much as, the contact moves". The comparison with Stage 2 (Section 8) asks whether access to a GT latent state *rescues* the prediction.

**Decision rule.** The thresholds were written into `zt_common.DECISION` before any probe was trained (the file was edited once afterwards, for the validation interval only):

* *predictable at h* — explained variance R² of the probe ≥ 0.50 (partial ≥ 0.25);
* *beats persistence at h* — Gain_h = 1 − RMSE / RMSE_persistence ≥ 10 % with a bootstrap interval excluding zero (*clear*); ≥ 2 % with an interval excluding zero (*marginal*); otherwise *no*. The label is per horizon; the decision table lists the three labels;
* *Δz tracks ΔC* — mean Spearman over the three horizons ≥ 0.60 and at most 3 % of the test transitions in the low-ΔC / high-Δz quadrant;
* *rescue* — 1 − RMSE of the probe started from the GT z₀ / RMSE of Stage 2 at the same target frame, averaged over h = 4 and 8: ≥ 25 % *yes*, ≥ 10 % *partial*, otherwise *no*.

Cases, tested in this order: **C** if the mean Spearman is below 0.30, or if at least 6 % of the transitions are low-ΔC / high-Δz and R² at h = 1 is below 0.5; **D** if the probe's gain is *clear* at h = 4 and h = 8 with R² ≥ 0.25 at h = 8 ("predictable beyond persistence"), the rescue is ≥ 25 % and the alignment is strong (Spearman ≥ 0.60, quadrant ≤ 3 %); **A** if it is predictable beyond persistence and the rescue is ≥ 10 % (no alignment condition); **B** if persistence is strong at h = 1 (R² ≥ 0.5) and the probe's gain is below 10 % at every horizon; otherwise *mixed*. Analyses that are not part of the rule — the split by window position and by subject, the linear and shrinkage references, the paired trajectory contrast, the composition of the exceptional quadrant — were added afterwards and are reported as interpretation, not as verdicts.

**Naming.** The plan's final question (Q5) offers "D — a mixture" as one of its answers. That is not the rule's Case D above (predictable beyond persistence, rescue ≥ 25 %, strong alignment). In this report "Case D" always means the rule's case, and the mixed answer to Q5 is written out as "a mixture".
