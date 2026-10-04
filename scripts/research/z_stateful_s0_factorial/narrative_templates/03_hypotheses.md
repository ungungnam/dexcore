**Hypothesis A: open-loop latent prediction is the problem.** The whole-sequence model must produce frame 40 of the latent
trajectory from s_0 and the object trajectory alone. The diagnostic showed that knowing the current latent helps, especially on
TACO. A model that carries its latent state forward could use that. It predicts that M10 beats M00 and M11 beats M01, and that the
gap between training and test latent error shrinks.

**Hypothesis B: decoding throws away s_0.** The absolute decoder has to rebuild the initial dense realisation from a 64-D latent
that does not keep every detail, although s_0 is given. Decoding only the change from s_0 keeps that detail. It predicts that M01
beats M00 and M11 beats M10, most of all in the first frames.

The four outcomes the plan distinguishes are: the stateful change is the main fix, the decoder change is the main fix, both help,
or neither. A fifth, that stateful modelling helps on TACO only, is a statement about the two datasets together.
