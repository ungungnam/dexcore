What was wrong with the previous z-mediated temporal contact generator? Two architectural assumptions were still untested.

- **How z evolves through time.** The previous model predicts the whole latent trajectory in one pass from (s_0, G, tau). The
  alternative is to carry a latent state forward, z_t to z_t+1.
- **How z is turned back into contact.** The previous decoder rebuilds every map from z_t and the geometry alone. The alternative is
  to keep the known initial map and decode only its change, C_t = s_0 + Delta_C_t.

The two choices are crossed in a 2 × 2 design, on TACO and on ARCTIC, with everything else fixed.

| | absolute decoding | s_0-preserving decoding |
|---|---|---|
| **whole-sequence z** | M00 | M01 |
| **stateful z** | M10 | M11 |

The task is unchanged: one deterministic 64-frame contact trajectory per example from (s_0, G, tau). The four z models are compared
with each other first and then with the direct dense model D0. "Better", "worse" and "similar" follow the rule of the previous
studies, fixed before any result was read: a paired difference on identical test sequences counts only if it is at least 2 % and its
95 % take-cluster bootstrap interval excludes zero.
