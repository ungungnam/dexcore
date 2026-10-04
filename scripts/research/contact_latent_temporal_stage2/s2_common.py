"""Contact latent temporal generation, Stage 2: shared definitions.

Question. Is the Stage-1 structure-oriented latent z a better intermediate representation for temporal contact generation
p(C_1:T | s_0, G, tau) (T = 64, 512-D canonical maps) than predicting the dense trajectory directly?  Three models on ONE shared
temporal backbone (same inputs s_0, G, tau_1:T; no diffusion; no new disentanglement / temporal / event losses):
  B0  (s_0, G, tau) -> rho_hat_1:T                      direct dense residual prediction (the previous D0, scaled up)
  B1  (s_0, G, tau) -> z_hat_1:T -> D_z -> C_hat        z-mediated (teacher z*_t = E_z^Stage1(C_t^GT, G_t), cached; E_z never used at inference)
  B2  (s_0, G, tau) -> (z_hat, r_hat)_1:T -> D_z + D_r  z + r mediated (r reconstruction-driven, no supervision against the Stage-1 r)

Inherited, not re-derived: the 64-frame sequences, the fixed take-level split, the canonical map and its normalisation, G, tau_local,
the residual parameterisation rho_t = (C_t - s_0) / sigma_r (structure_aware_temporal_generation = SAT), the exact R2 / wrench
evaluation and the take-cluster bootstrap (SAT evaluate / aggregate), the Stage-1 A3 checkpoint (frozen teacher E_z; D_z / D_r used as
decoder initialisations and fine-tuned end to end).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
HERE = Path(__file__).resolve().parent
SAT_DIR = REPO / "scripts/research/structure_aware_temporal_generation"
CF_DIR = REPO / "scripts/research/contact_factorization_stage1"
for _p in (CF_DIR, SAT_DIR, REPO / "scripts/research/structure_variance_boundary", REPO / "scripts/research/wrench_counterfactual",
           REPO / "scripts/research/hand_contact_predictive_info", REPO / "scripts/research/temporal_contact_events", REPO / "scripts/research/hier_contact_gen", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sat_common as SAT  # noqa: E402
import cf_common as CF  # noqa: E402

OUT = Path("/result/uhnam/dexcore/reports/contact_latent_temporal_stage2")
CKPT = Path("/ckpt/uhnam/dexcore/contact_latent_temporal_stage2")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T, PAD = 64, 8
TF = T - 1                                   # generated frames 1..63
N_PTS = 512
SEED = 0                                     # the one primary training seed
EXTRA_SEED = 1                               # only for the decisive pair if the comparison is ambiguous
TEACHER_MODEL = "A3"                         # Stage-1 full factorisation model (one seed)
TEACHER_NAME = CF.run_name(TEACHER_MODEL, CF.SEED)
DZ, DR = CF.MODELS[TEACHER_MODEL]["dz"], CF.MODELS[TEACHER_MODEL]["dr"]

MODELS = {"B0": dict(z=False, r=False, label="direct dense: backbone -> MLP head -> rho_hat_1:T, C_hat_t = s_0 + sigma_r rho_hat_t"),
          "B1": dict(z=True, r=False, label="z-mediated: backbone -> z_hat_1:T -> D_z (Stage-1 init, fine-tuned) -> C_hat_t"),
          "B2": dict(z=True, r=True, label="z + r mediated: backbone -> (z_hat, r_hat)_1:T -> C_bar_t = D_z(z_hat), C_hat_t = C_bar_t + D_r(z_hat, r_hat)")}
PAIRS = [("B1", "B0", "does a structure-oriented latent intermediate help?"), ("B2", "B1", "does a complementary realisation pathway help beyond z?"),
         ("B2", "B0", "factorised vs direct")]

# shared temporal backbone (DiT-style adaLN-zero FiLM blocks over the 63 future frames; identical for B0 / B1 / B2)
BACKBONE = dict(width=768, depth=8, heads=12, mlp_ratio=4, dropout=0.1, cond_width=256,
                ckpt_dec=True,               # activation checkpointing of the decoder blocks during training (memory; no effect on the function computed)
                b0_head_hidden=2048)         # B0 output head 768 -> 2048 -> 2048 -> 512 (parameter fairness vs the B1 / B2 decoders, Section 13)
# training (identical for the three models)
TRAIN = dict(lr=1e-4, wd=0.01, warmup=5000, batch=128, grad_clip=1.0, ema=0.999, max_steps=80000, final_lr_frac=0.05,
             eval_every=1000, min_steps=20000, patience=15, log_every=100,
             n_dec_frames=4,                 # B1 / B2: the dense losses are evaluated on 4 random future frames per sequence per step (unbiased estimate;
                                             # decoding all 63 x 128 maps through the point-token decoders per step is ~16 x more expensive); L_z uses all 63 frames
             lambda_z=1.0, lambda_c=0.5, lambda_res=0.01,
             grad_inspect_steps=(500, 1000, 2000, 3000, 5000))   # per-term gradient norms logged at these steps (Section 12 inspection)
LAMBDA_Z_CHECK = (0.3, 1.0, 3.0)             # the ONE allowed check (TACO B1 only, Section 12) — run only if validation clearly suggests lambda_z is off
N_BOOT = 1000
PROTOCOL = "A"                               # fixed GT s_0 (the previous fixed-s_0 protocol); the sampled-s_0 protocol B is optional and not run
HORIZON_BINS = ((1, 8), (9, 16), (17, 32), (33, 48), (49, 63))

# metrics (SAT evaluate.py names) and their roles
DENSE = ["E_C", "E_C_last16"]
STRUCT = ["part_hamming", "amount_l1", "centroid", "normal"]
WRENCH = ["q_rel_l1", "q_cos", "q_Qerr"]
TEMPORAL = ["jitter_dense", "jitter_r2", "tv_dense"]
HIGHER_BETTER = {"q_cos", "part_macro_f1", "part_macro_auprc"}
MAIN_METRICS = ["E_C", "E_C_last16", "part_hamming", "part_macro_f1", "amount_l1", "centroid", "normal", "q_rel_l1", "q_cos", "jitter_dense", "jitter_r2"]
STRUCT_DECISION = ["part_hamming", "amount_l1", "centroid", "normal", "q_rel_l1"]   # the five structural / wrench metrics of the decision rule
# pre-registered decision rule (fixed before any Stage-2 result was looked at)
DECISION = dict(
    rel_min=0.02,                  # a paired difference counts as "better" / "worse" only if |relative change| >= 2 % AND the take-cluster bootstrap CI excludes 0
    struct_majority=3,             # "structure improves" = at least 3 of the 5 structural / wrench metrics better and none worse
    z_quality_tol=0.10,            # B2 keeps the z quality of B1 if its test L_z is at most 10 % higher
    stability_metrics=("jitter_dev_dense", "jitter_dev_r2", "E_C_last16"),   # long-horizon stability: deviation of the frame-to-frame change from the GT's and the last-16-frame error
)


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=SAT.HP._json_default))


def read_json(path):
    return json.loads(Path(path).read_text())


def md5(path, n=12):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:n]


def run_name(model, seed=SEED, tag=""):
    return f"{model}_seed{seed}{tag}"


def ckpt_path(ds, name, which="best"):
    return CKPT / ds / (f"{name}.pt" if which == "best" else f"{name}_{which}.pt")


def teacher_cache_path(ds):
    return ds_out(ds) / "cache" / "teacher_z.npz"


def teacher_ckpt_path(ds):
    return CF.ckpt_path(ds, TEACHER_NAME)


def preds_path(ds, name, part="test"):
    return ds_out(ds) / "preds" / f"{name}_{PROTOCOL}{'' if part == 'test' else '_' + part}.npz"


def metrics_path(ds, name, part="test"):
    return ds_out(ds) / "metrics" / f"{name}_{PROTOCOL}{'' if part == 'test' else '_' + part}.npz"


def train_log_path(ds, name):
    return ds_out(ds) / "train_logs" / f"{name}.csv"


def sat_metrics_path(ds, name):
    """Metrics file of the previous study (the D0 reference rows): SAT evaluate.py output, protocol A."""
    return SAT.metrics_path(ds, name, "A")


cluster_bootstrap = SAT.cluster_bootstrap
paired_cluster_bootstrap = SAT.paired_cluster_bootstrap
ratio_cluster_bootstrap = SAT.ratio_cluster_bootstrap


def model_table(param_counts=None):
    rows = []
    for m, c in MODELS.items():
        pc = (param_counts or {}).get(m, {})
        rows.append(dict(model=m, intermediate=("none (dense rho)" if not c["z"] else ("z (64)" if not c["r"] else "z (64) + r (64)")),
                         output_head=("MLP 768-2048-2048-512 (zero-initialised: starts as persistence)" if m == "B0" else ("linear 768-64 (z)" if m == "B1" else "linear 768-64 (z) + linear 768-64 (r)")),
                         decoder=("—" if m == "B0" else ("D_z (Stage-1 A3 init, fine-tuned)" if m == "B1" else "D_z + D_r (Stage-1 A3 init, fine-tuned)")),
                         losses=("L_C" if m == "B0" else ("L_C + lambda_z L_z" if m == "B1" else "L_full + lambda_c L_zonly + lambda_z L_z + lambda_res L_res")),
                         description=c["label"], **{f"n_params_{k}": v for k, v in pc.items()}))
    return pd.DataFrame(rows)
