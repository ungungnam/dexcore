{{sanity}}

All eleven checks pass on both datasets (`sanity_summary.json`). What they do and do not verify:

* **Check 4** reproduces all saved test predictions of every probe (every test pair, not a sample) from inputs assembled independently from the cache — z_t, G, τ_t, τ_{t+h} and nothing else. Together with the forward signature (four tensors) this shows that no future latent, contact map, R2, wrench or hand quantity can have entered a prediction. It is a reconstruction test, not a perturbation test.
* **Check 7** recomputes all six low / high thresholds (ΔC, Δz, teacher step) from the training transitions.
* **Check 9** confirms that the Stage-2 B1 checkpoint is the one recorded in the Stage-2 configuration (md5) and that its prediction file predates this study; this study contains no Stage-2 training code.
* **Check 10** confirms that the frozen decoder used for the decoded-contact diagnostic reproduces the Stage-1 z-only reconstruction error.
