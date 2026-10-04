"""Contact factorization, Stage 1: shared definitions.

Question. Can a single-frame dense canonical contact map C (512-D) be factorised into a structure-oriented latent z and a
complementary realisation code r,   C_bar = D_z(z, G),  C_hat = C_bar + D_r(z, r, G),   such that z preferentially keeps the
R2 / wrench information, r adds what exact dense realisation needs, and r is temporally persistent when GT trajectories are
encoded frame by frame?  No temporal model, no independence loss, no explicit R2 output.

Inherited, not re-derived: the 64-frame sequences, the fixed take-level split, the canonical 512-D contact map and its
normalisation, the static descriptor G (hier_contact_gen); the exact R2 (participation / amount / centroid / normal) and the
76-D wrench profile q of the feature cache (structure_variance_boundary); the canonical geometry, the GT part-label masks and
the exact-rule grid operator for decoded maps (structure_aware_temporal_generation); the take-cluster bootstrap.

Model matrix (ONE training seed; TACO and ARCTIC separately):
  A0  standard autoencoder, one latent h (128 = |z| + |r|), reconstruction only              control
  A1  z-only structure autoencoder (|z| = 64), reconstruction + relational loss               how much z alone keeps
  A2  factorised z / r (64 / 64), same asymmetric architecture, lambda_rel = 0                 does the split alone create roles
  A3  factorised z / r + relational R2 + wrench teacher + weak residual penalty                 the proposed Stage-1 model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
for _p in (REPO / "scripts/research/structure_aware_temporal_generation", REPO / "scripts/research/structure_variance_boundary",
           REPO / "scripts/research/wrench_counterfactual", REPO / "scripts/research/hand_contact_predictive_info",
           REPO / "scripts/research/temporal_contact_events", REPO / "scripts/research/hier_contact_gen", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sat_common as SAT  # noqa: E402  (paths of the mask / geometry caches, calibration, exact R2, bootstrap)

OUT = Path("/result/uhnam/dexcore/reports/contact_factorization_stage1")
CKPT = Path("/ckpt/uhnam/dexcore/contact_factorization_stage1")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T, PAD = 64, 8
N_PTS = 512
N_PARTS, N_DIR, R2_DIM = 6, 76, 48
PARTS = SAT.PARTS
SEED = 0                                     # the one primary training seed
EXTRA_SEED = 1                               # only for the decisive models if the conclusion is ambiguous

MODELS = {"A0": dict(dz=128, dr=0, rel=False, label="standard autoencoder, one latent h (128)"),
          "A1": dict(dz=64, dr=0, rel=True, label="z-only structure autoencoder (64) + relational loss"),
          "A2": dict(dz=64, dr=64, rel=False, label="factorised z / r (64 / 64), no relational loss"),
          "A3": dict(dz=64, dr=64, rel=True, label="factorised z / r (64 / 64) + relational R2 + wrench teacher")}
# capacity variant (Section 5.1 of the plan: the conclusion depends on |z| — a 64-D z reconstructs the map almost entirely, so the
# bottleneck variant |z| = 16 (between PCA-8 and PCA-32 in linear terms, the size of the R2 content) is run for A1 / A2 / A3)
VARIANTS = {"A1_z16": dict(dz=16, dr=0, rel=True, label="bottleneck variant: z-only (16) + relational loss"),
            "A2_z16": dict(dz=16, dr=64, rel=False, label="bottleneck variant: factorised z / r (16 / 64), no relational loss"),
            "A3_z16": dict(dz=16, dr=64, rel=True, label="bottleneck variant: factorised z / r (16 / 64) + relational teacher")}
ALL_MODELS = {**MODELS, **VARIANTS}
FACTORISED = ("A2", "A3", "A2_z16", "A3_z16")

# architecture (point-token transformer with adaLN-zero FiLM on [G, latents]; the previous studies' block)
ARCH = dict(width=256, depth=4, heads=8, dropout=0.1, cond_width=256, n_pool=4)
# training
TRAIN = dict(batch=128, lr=3e-4, wd=0.01, warmup=500, ema=0.999, final_lr_frac=0.1,
             steps_single=12000,              # A0, A1
             steps_phase_a=4000,              # A2, A3: z branch alone
             steps_phase_b=12000,             # A2, A3: + r branch (z branch at 0.1 x lr for the first z_slow_steps, then joint)
             steps_override={"A0": 24000},    # the reconstruction control is trained longer (its validation error was still falling at 12k)
             z_freeze_steps=3000,             # phase B: z branch frozen, then z_slow_steps at 0.1 x lr, then joint
             z_slow_steps=3000,
             lambda_rel=1.0, lambda_coarse=1.0, lambda_delta=0.01,
             res_scale=10.0,                  # gain of the residual input channel C~ - C_bar~ of E_r (residual std ~ 0.1)
             eval_every=1000, log_every=100,
             balance_power=0.5)               # P(take, group) ∝ n_frames^0.5, then a frame uniformly (balanced sequence sampling)
TEACHER = dict(w_r2=0.5, w_q=0.5)            # d_teacher = 0.5 d_R2 / median + 0.5 d_q / median (train medians)
PROBE = dict(hidden=256, lr=1e-3, wd=1e-4, batch=1024, max_epochs=60, patience=6)
KNN_K = 10
N_BOOT = 1000
HORIZONS = (1, 4, 8, 16, 32)

# pre-registered decision thresholds (Section 18 of the plan), fixed before any result was looked at
DECISION = dict(
    recon_close_to_a0=0.10,        # E_full(A3) <= (1 + 0.10) E_full(A0)
    gain_r_min=0.25,               # (E_zonly - E_full) / E_zonly >= 0.25 (and the paired CI excludes 0)
    z_keeps_min=0.70,              # z probe retains >= 70 % of the dense-C probe's gain over the trivial predictor (R2 + wrench mean)
    z_over_r_margin=0.20,          # z probe retention exceeds r probe retention by >= 0.20 (mean over the structural quantities)
    knn_z_vs_random_max=0.70,      # teacher distance of the z neighbours <= 0.70 x the random-pair distance
    swap_struct_frac_min=0.70,     # fraction of pairs where swap_AB is structurally closer to the z donor
    swap_detail_frac_min=0.60,     # fraction of pairs where the detail of swap_AB is closer to the r donor
    r_acf_h8_min=0.50,             # autocorrelation of r at h = 8 (absolute floor; uninformative alone: the dense map itself is persistent)
    r_acf_vs_h_margin=0.05,        # acf_r(8) >= acf_h(8) - 0.05: r is not less persistent than the plain autoencoder latent h (A0)
    r_init_gain_min=0.10,          # R^2 gain of r_{t+h} from r_t beyond the current structure z_{t+h}, h = 8
    r_slower_than_dense=1.00,      # displacement ratio of r at h = 8 <= that of the dense residual C - C_bar it encodes
)
# The reference-relative persistence criteria (vs h and vs the dense residual) were fixed after the untrained smoke run showed
# that absolute autocorrelations are high for every function of the contact map (dense C: 0.96 at h = 8), before any trained
# model was evaluated.


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=SAT.HP._json_default))


def read_json(path):
    return json.loads(Path(path).read_text())


def run_name(model, seed=SEED):
    return f"{model}_seed{seed}"


def ckpt_path(ds, name):
    return CKPT / ds / f"{name}.pt"


def latents_path(ds, name, split):
    return ds_out(ds) / "latents" / f"{name}_{split}.npz"


def train_log_path(ds, name):
    return ds_out(ds) / "train_logs" / f"{name}.csv"


cluster_bootstrap = SAT.cluster_bootstrap
paired_cluster_bootstrap = SAT.paired_cluster_bootstrap
ratio_cluster_bootstrap = SAT.ratio_cluster_bootstrap


def model_table(param_counts=None):
    rows = []
    for m, c in ALL_MODELS.items():
        rows.append(dict(model=m, latent_z=c["dz"] if m != "A0" else 0, latent_r=c["dr"], latent_h=c["dz"] if m == "A0" else 0,
                         total_latent=c["dz"] + c["dr"], relational_loss=c["rel"], z_only_reconstruction_loss=m != "A0",
                         residual_branch=c["dr"] > 0, residual_penalty=c["dr"] > 0, description=c["label"],
                         n_params=(param_counts or {}).get(m, np.nan)))
    return pd.DataFrame(rows)
