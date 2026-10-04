Can the z-mediated generator

    (s_0, G, tau)  ->  z_hat_1:63  ->  C_hat_1:63

do better when the temporal network and the contact decoder are trained together, so that the decoder is fitted to the z trajectories the
temporal network actually produces? Put differently: was part of the Stage-2 failure a mismatch between the decoder and the predicted
latent, rather than a weakness of the Stage-1 latent z itself?

The task is unchanged: one deterministic 64-frame contact trajectory per example from the initial contact map s_0, the static geometry
descriptor G and the object trajectory tau. Three models are compared on the same test sequences.

- **M0** predicts the dense maps directly. It is the Stage-2 B0 checkpoint.
- **M1** is the previous z-mediated model. It is the Stage-2 B1 checkpoint.
- **M2** is the jointly adapted z-mediated model trained in this study.
- **M3** is an optional variant of M2 whose decoder is also fed the teacher latent during training.

"Better", "worse" and "similar" follow one rule fixed before any result was read: a paired difference on identical test sequences counts
only if it is at least 2 % of the reference value and its 95 % take-cluster bootstrap interval excludes zero.
