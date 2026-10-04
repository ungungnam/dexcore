"""z temporal diagnostic: shared definitions.

Question. WHY was the Stage-1 structure latent z hard to predict temporally in Stage 2?  Two hypotheses:
  H1  open-loop / long-horizon problem: z is locally predictable from the current z state, but the whole trajectory z_1:T cannot be
      predicted from s_0 alone;
  H2  representation-dynamics problem: even given the current GT z_t the future z_{t+h} is hard to predict / the coordinates of z are
      not temporally well behaved.
Two experiments on the EXISTING representation (the Stage-1 A3 encoder is frozen, nothing is retrained, no generator is built):
  A   local latent dynamics probe   P_h(z_t, G, tau_{t:t+h}) -> z_{t+h},  h = 1, 4, 8   vs persistence / mean / linear
  B   temporal geometry             delta_C(t, h) vs delta_z(t, h): correlation, binned means, train-quantile quadrants
plus the comparison with the saved Stage-2 B1 open-loop predictions (no retraining).

Inherited, not re-derived: sequences, split, canonical maps, G, tau_local, exact R2 / wrench, the Stage-1 teacher descriptors and the
frozen A3 checkpoint (contact_factorization_stage1), the teacher cache and the B1 predictions (contact_latent_temporal_stage2).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/uhnam/workspace/dexcore")
HERE = Path(__file__).resolve().parent
S2_DIR = REPO / "scripts/research/contact_latent_temporal_stage2"
for _p in (S2_DIR,):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import s2_common as S2  # noqa: E402  (adds the Stage-1 / previous-study paths; SAT, CF)

CF, SAT = S2.CF, S2.SAT
OUT = Path("/result/uhnam/dexcore/reports/z_temporal_diagnostic")
CKPT = Path("/ckpt/uhnam/dexcore/z_temporal_diagnostic")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T, PAD = 64, 8
DZ = S2.DZ
HORIZONS = (1, 4, 8)
SEED = 0
N_BOOT = 1000
TEACHER_NAME = S2.TEACHER_NAME

# Experiment A: one strong nonlinear probe per horizon (residual MLP around persistence), + diagnostics
PROBE = dict(width=512, depth=4, expansion=4, dropout=0.1)            # ~9 M parameters (3-10 M band of the plan)
VARIANTS = {"probe": dict(tau=True, label="nonlinear probe P_h(z_t, G, tau_t, tau_{t+h})"),          # PRIMARY
            "probe_notau": dict(tau=False, label="ablation: nonlinear probe without the object trajectory, P_h(z_t, G)")}
TRAIN = dict(lr=1e-4, wd=0.01, warmup=500, batch=1024, max_steps=30000, min_steps=5000, patience=10, eval_every=100, final_lr_frac=0.1)   # validation every 100 steps: the first launch (every 500) put the optimum at its first check
RIDGE_LAMBDAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)                # optional linear diagnostic: ridge on [z_t | G | tau], lambda chosen on validation
# Experiment B
QUANT_LOW, QUANT_HIGH = 0.30, 0.70                                    # low = bottom 30 %, high = top 30 % of the TRAIN distribution (fixed before any test result)
N_RANDOM_PAIRS = 200000                                               # train random-pair distances that normalise delta_C / delta_z / delta_T
DIRECTION_H = 4                                                       # optional directional-consistency check (one horizon only)

# Pre-registered decision rule (written before any probe was trained or any test quantity was computed).
DECISION = dict(
    r2_yes=0.50, r2_partial=0.25,          # "z_{t+h} predictable": explained variance of the standardised target on test
    gain_clear=0.10, gain_marginal=0.02,   # "beats persistence": Gain_h = (RMSE_persistence - RMSE_probe) / RMSE_persistence, bootstrap CI must exclude 0
    spearman_yes=0.60, spearman_partial=0.30,   # "delta_z tracks contact change": Spearman(delta_C, delta_z) on test, averaged over the three horizons
    quadrant_many=0.06, quadrant_few=0.03,      # low-delta_C / high-delta_z fraction of test transitions (independence would give 0.09)
    rescue_yes=0.25, rescue_partial=0.10,       # "current GT z rescues Stage-2": 1 - RMSE(probe from GT z_0 at horizon h) / RMSE(Stage-2 B1 at frame h), mean over h = 4, 8
    persistence_strong_r2=0.50,                 # "persistence is strong" at h = 1: R^2 of the persistence predictor
)
# Case assignment (Section 11 of the plan), evaluated per dataset:
#   C  if  Spearman < spearman_partial  or  (low-C / high-z fraction >= quadrant_many  and  R^2 of the probe at h = 1 < r2_yes)
#   D  elif  predictable (probe gain >= gain_clear at h = 4 and h = 8, R^2 >= r2_partial at h = 8)  and  rescue >= rescue_yes  and  Spearman >= spearman_yes and low-C / high-z <= quadrant_few
#   A  elif  predictable  and  rescue >= rescue_partial
#   B  elif  persistence strong at h = 1  and  probe gain < gain_clear at every horizon
#   else "mixed" (the elements are listed)


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=SAT.HP._json_default))


def read_json(path):
    return json.loads(Path(path).read_text())


def md5(path, n=12):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()[:n]


def cache_path(ds):
    return ds_out(ds) / "cache" / "frames.npz"


def ckpt_path(ds, variant, h):
    return CKPT / ds / f"{variant}_h{h}_seed{SEED}.pt"


def preds_path(ds, variant, h):
    return ds_out(ds) / "preds" / f"{variant}_h{h}.npz"


def train_log_path(ds, variant, h):
    return ds_out(ds) / "train_logs" / f"{variant}_h{h}.csv"


def stage2_preds_path(ds):
    return S2.preds_path(ds, S2.run_name("B1"))


cluster_bootstrap = SAT.cluster_bootstrap


def boot_stat(per_cluster, fn, n_boot=N_BOOT, seed=0):
    """Take-cluster bootstrap of a statistic fn(sums) of per-cluster sums: per_cluster (n_clusters, k) array."""
    rng = np.random.default_rng(seed); n = len(per_cluster); reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        reps.append(fn((w[:, None] * per_cluster).sum(0)))
    v = fn(per_cluster.sum(0))
    return float(v), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def cluster_sums(cols, clusters):
    """Stack columns (each (n,)) into per-cluster sums (n_clusters, k)."""
    u, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return np.stack([np.bincount(inv, weights=np.asarray(c, float), minlength=len(u)) for c in cols], 1)
