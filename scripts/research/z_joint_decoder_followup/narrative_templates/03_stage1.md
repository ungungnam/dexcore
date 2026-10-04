Stage 1 trained a single-frame autoencoder that splits a dense map into a 64-D structure latent z and a 64-D realisation code r. The
A3 model of that study is the frozen teacher here. What it established about z:

- **z keeps the structural information.** A probe on z recovered <<s1_taco_keep_pct>> % (TACO) and <<s1_arctic_keep_pct>> % (ARCTIC) of what the same
  probe recovers from the dense map for participation, amount, centroid, normal and wrench.
- **z organises structure.** Nearest neighbours in z were closer in the R2 / wrench sense than nearest neighbours in the dense map.
- **z controls structure.** When latents of two frames were swapped, the decoded structure followed the z donor in <<s1_taco_swap_pct>> % and
  <<s1_arctic_swap_pct>> % of the swaps.
- **z is compact.** Decoding z alone reconstructs the map with error <<s1_taco_A3_E_zonly:.2f>> (TACO) and <<s1_arctic_A3_E_zonly:.2f>> (ARCTIC), against
  <<s1_taco_pca64_E_zonly:.2f>> and <<s1_arctic_pca64_E_zonly:.2f>> for a 64-D PCA of the map.

So z is a meaningful structure-oriented representation of a single frame. Stage 1 said nothing about how hard z is to predict over time.
