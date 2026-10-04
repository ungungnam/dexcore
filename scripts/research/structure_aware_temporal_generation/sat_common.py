"""Structure-aware temporal contact generation: shared definitions.

Two questions on the direct dense generator p(C_1:T | s_0, G, tau) (T = 64 frames, 512-D canonical contact maps):
  (1) does supervising R2-level structure (per-part participation / amount / centroid / normal, GT finger labels used only
      inside the training loss) improve the generated dense trajectories, and
  (2) once s_0 is fixed, is the useful future evolution deterministic (Case A) or must the trajectory stay stochastic (B/C)?

Inherited, not re-derived: the 64-frame sequences, G, tau_local, the fixed split and the normalisation (hier_contact_gen), the
frozen initial sampler p(s_0 | G) (hier_contact_gen_ckpt/<ds>/fixed0/sampler_G.pt), the exact R2 / wrench profiles / object
length (structure_variance_boundary feature cache), the MANO finger labels, meshes and take loaders (wrench_counterfactual),
the support-function wrench model (wrench_lp), the test event labels (structure_variance_boundary events_test.csv), the
take-cluster bootstrap (temporal_contact_events).

Model matrix (3 seeds each, TACO and ARCTIC separately):
  D0 / D1 / D2   deterministic   C_1:T = s_0 + sigma_r F(s_0, G, tau)            dense | + output R2 loss | + hidden R2 aux loss
  S0 / S1 / S2   sequence diffusion over the 64-frame residual (frames 1..63 as ONE sample; frame 0 = s_0 is never generated)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
for _p in (REPO / "scripts/research/structure_variance_boundary", REPO / "scripts/research/wrench_counterfactual",
           REPO / "scripts/research/hand_contact_predictive_info", REPO / "scripts/research/temporal_contact_events",
           REPO / "scripts/research/hier_contact_gen", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import sv_common as SV  # noqa: E402
import wc_common as W  # noqa: E402
import hp_common as HP  # noqa: E402
import wrench_lp as L  # noqa: E402

OUT = Path("/result/uhnam/dexcore/reports/structure_aware_temporal_generation")
CKPT = Path("/ckpt/uhnam/dexcore/structure_aware_temporal_generation")
HIER_CKPT = Path("/result/uhnam/dexcore/hier_contact_gen_ckpt")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T, PAD = 64, 8
TF = T - 1                                   # generated frames 1..63
K_SAMPLES = 10
SEEDS = (0, 1, 2)
PARTS = W.PARTS
N_PARTS = 6
N_DIR = 76
MU = SV.MU
N_CONE = L.N_CONE
TAU_C, TAU_N = 0.05, 0.5                     # soft participation: point-level and count-level temperatures (fixed, never tuned)
LAMBDA_STRUCT_GRID = (0.3, 1.0, 3.0)
LAMBDA_AUX_GRID = (0.3, 1.0)
MODELS = {"D0": dict(family="det", struct=False, aux=False, label="deterministic, dense loss only"),
          "D1": dict(family="det", struct=True, aux=False, label="deterministic, dense + output R2 loss"),
          "D2": dict(family="det", struct=True, aux=True, label="deterministic, dense + output R2 loss + hidden R2 aux loss"),
          "S0": dict(family="diff", struct=False, aux=False, label="sequence diffusion, diffusion loss only"),
          "S1": dict(family="diff", struct=True, aux=False, label="sequence diffusion + output R2 loss (on x0 estimate)"),
          "S2": dict(family="diff", struct=True, aux=True, label="sequence diffusion + output R2 loss + hidden R2 aux loss")}
DET, DIFF = ("D0", "D1", "D2"), ("S0", "S1", "S2")
PROTOCOLS = ("A", "B")                       # A: fixed GT s_0;  B: the 10 sampled s_0 of the frozen sampler
EVENT_CLASSES = ["persistent_spatial", "persistent_mixed", "onset", "release"]
EVENT_LABEL = {"persistent_spatial": "persistent spatial", "persistent_mixed": "persistent mixed", "onset": "onset / regrasp", "release": "release"}
R2_DIM = 48                                  # [a (6) | m (6) | p (18) | n (18)]
HIST_MAJ = 3                                 # causal majority window of the participation rule


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=HP._json_default))


def mask_path(ds):
    return ds_out(ds) / "cache" / "masks.npz"


def geometry_path(ds):
    return ds_out(ds) / "cache" / "canonical_geometry.npz"


def calibration_path(ds):
    return ds_out(ds) / "calibration.json"


def s0_path(ds):
    return ds_out(ds) / "s0_samples.npz"


def load_calibration(ds):
    return json.loads(calibration_path(ds).read_text())


def run_name(model, seed, lam_struct=None, lam_aux=None):
    """Checkpoint / prediction stem of one run; the lambdas are part of the name only during the selection sweep."""
    s = f"{model}_seed{seed}"
    if lam_struct is not None:
        s += f"_ls{lam_struct:g}"
    if lam_aux is not None:
        s += f"_la{lam_aux:g}"
    return s


def ckpt_path(ds, name):
    return CKPT / ds / f"{name}.pt"


def preds_path(ds, name, protocol):
    return ds_out(ds) / "preds" / f"{name}_{protocol}.npz"


def metrics_path(ds, name, protocol):
    return ds_out(ds) / "metrics" / f"{name}_{protocol}.npz"


def chosen_lambdas(ds):
    """The once-chosen lambda_struct / lambda_aux of the dataset (written by select_lambda.py)."""
    p = ds_out(ds) / "lambda_choice.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ------------------------------------------------------------------------------ data access
def load_raw(ds):
    """C (N, 64, 512) f16, O (N, 80, D), meta, manifest (numpy)."""
    return HP.load_raw(ds)


def load_features(ds):
    return SV.load_features(ds)


def load_events(ds):
    return SV.load_events(ds, "test")


def load_masks(ds):
    z = np.load(mask_path(ds))
    return {k: z[k] for k in z.files}


def load_geometry(ds):
    z = np.load(geometry_path(ds), allow_pickle=True)
    return {k: z[k] for k in z.files}


def exact_r2(F, frames=slice(PAD, PAD + T)):
    """Exact R2 of the feature cache on the sequence frames: a (N, 64, 6) causal majority of n_hard >= 3, m (N, 64, 6),
    p (N, 64, 6, 3), nrm (N, 64, 6, 3) (inactive parts zeroed), q (N, 64, 76)."""
    a = SV.causal_majority(F["n_hard"] >= SV.N_MIN_VERTS)[:, frames]
    mask = a[..., None]
    return dict(a=a.astype(np.float32), m=F["m"][:, frames].astype(np.float32), p=(F["p"][:, frames] * mask).astype(np.float32),
                nrm=(F["nrm"][:, frames] * mask).astype(np.float32), q=F["q"].astype(np.float32))


# ------------------------------------------------------------------------------ statistics
cluster_bootstrap = SV.cluster_bootstrap
paired_cluster_bootstrap = SV.paired_cluster_bootstrap
ratio_cluster_bootstrap = SV.ratio_cluster_bootstrap


def model_matrix():
    rows = []
    for m, c in MODELS.items():
        rows.append(dict(model=m, family="deterministic" if c["family"] == "det" else "sequence diffusion",
                         base_loss="dense MSE on the residual r = (C_t - s_0) / sigma_r" if c["family"] == "det" else "v-prediction MSE (DDPM, cosine schedule, 1000 steps)",
                         output_structural_loss=c["struct"], hidden_aux_loss=c["aux"], inference="(s_0, G, tau) -> C_1:63 (one pass)" if c["family"] == "det"
                         else "(s_0, G, tau, noise) -> C_1:63 (DDIM 100 steps, eta = 0), K = 10 samples", description=c["label"]))
    return pd.DataFrame(rows)
