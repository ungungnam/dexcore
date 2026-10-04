"""z joint-decoder follow-up to the Stage-2 latent temporal generation study: shared definitions.

Question. Was the Stage-2 z-mediated failure caused partly by the decoding / optimisation formulation rather than by the Stage-1 z
itself?  Keep  (s_0, G, tau) -> z_hat_1:63 -> C_hat_1:63  (deterministic, whole sequence at once, no rollout, no r, no diffusion, no new
latent, no new loss) and train the temporal network F_theta and the contact decoder D_phi jointly, so that the decoder's primary input
during training is the predicted z_hat, with enough decoder capacity and enough optimisation time.

Models (ONE seed; TACO and ARCTIC separately; everything else inherited from Stage 2: sequences, split, canonical 512-D maps, s_0, G,
tau_local, normalisation, residual loss units, exact R2 / wrench evaluation, take-cluster bootstrap, the frozen Stage-1 A3 teacher):
  M0  direct dense baseline            = the Stage-2 B0 checkpoint, reused (same backbone and recipe; see M0_REUSE)
  M1  previous z-mediated formulation  = the Stage-2 B1 checkpoint, reused
  M2  jointly adapted z-mediated model   F_theta -> z_hat -> D_phi(z_hat, G) ;  L = L_C(D_phi(z_hat)) + lambda_z L_z
  M3  (optional) dual-input decoder      as M2 with  L = beta_pred L_C(D_phi(z_hat)) + beta_gt L_C(D_phi(z*)) + lambda_z L_z

What M1 already was (read from the Stage-2 code, s2_train.py / s2_models.py): B1 trained the temporal network AND its decoder (the Stage-1
D_z, loaded and fine-tuned end to end) on the predicted z_hat with L_C + 1.0 L_z.  M2 therefore does NOT differ from M1 by "joint vs
frozen".  It differs in exactly three pre-declared things:
  (a) decoder capacity: the Stage-1 D_z (4 point-token blocks, width 256; loaded) + 2 extra identity-initialised blocks (6 blocks);
  (b) optimisation budget: 100 k-step schedule, no stop before 25 k steps, patience 15 (M1: 80 k / 20 k, three runs stopped by hand);
  (c) checkpoint selection on the validation value of the FINAL-TASK loss L_C (the criterion of M0), not on L_C + lambda_z L_z (M1).
The state selected by M1's criterion (L_C + L_z) is saved as well and evaluated as the sensitivity row "M2_seltotal".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path("/home/uhnam/workspace/dexcore")
HERE = Path(__file__).resolve().parent
S2_DIR = REPO / "scripts/research/contact_latent_temporal_stage2"
if str(S2_DIR) not in sys.path:
    sys.path.insert(0, str(S2_DIR))
import s2_common as S2  # noqa: E402  (also puts the Stage-1 / previous-study / evaluation code on sys.path)

OUT = Path(os.environ.get("ZJ_OUT", "/result/uhnam/dexcore/reports/z_joint_decoder_followup"))      # ZJ_OUT / ZJ_CKPT: dry runs of the analysis chain on
CKPT = Path(os.environ.get("ZJ_CKPT", "/ckpt/uhnam/dexcore/z_joint_decoder_followup"))               # snapshot checkpoints (zj_snapshot.py) in a scratch directory
DATASETS = S2.DATASETS
LABEL = S2.LABEL
T, PAD, TF, N_PTS = S2.T, S2.PAD, S2.TF, S2.N_PTS
SEED = 0                                     # the one primary training seed (the seed of the reused B0 / B1)
EXTRA_SEED = 1                               # M2 only (and M0 if needed), only if the M2-vs-M0 dense comparison is ambiguous (DECISION["ambiguous"])
PROTOCOL = "A"                               # fixed GT s_0 (primary protocol; initial-contact sampling is not part of this study)
N_BOOT = 1000
HORIZON_BINS = S2.HORIZON_BINS

# name of this study -> (architecture key of s2_models.Stage2Model, reused Stage-2 run or None)
MODELS = {
    "M0": dict(arch="B0", reuse="B0", joint=False, dual=False, label="direct dense baseline: backbone -> MLP head -> rho_hat_1:63, C_hat_t = s_0 + sigma_r rho_hat_t (Stage-2 B0, reused)"),
    "M1": dict(arch="B1", reuse="B1", joint=True, dual=False, label="previous z-mediated formulation: backbone -> z_hat -> Stage-1 D_z (4 blocks, fine-tuned), selected on L_C + L_z (Stage-2 B1, reused)"),
    "M2": dict(arch="B1", reuse=None, joint=True, dual=False, label="jointly adapted: backbone -> z_hat -> D_phi (Stage-1 D_z + 2 identity-initialised blocks, fully fine-tuned), L_C(D_phi(z_hat)) + lambda_z L_z"),
    "M3": dict(arch="B1", reuse=None, joint=True, dual=True, label="dual-input decoder (optional): M2 + beta_gt L_C(D_phi(z*)); the predicted-z term dominates"),
    "M0r": dict(arch="B0", reuse=None, joint=False, dual=False, label="budget-parity check: M0 retrained under this study's 100 k-step protocol (conditional, see M0_REUSE)"),
}
TRAINED_HERE = ("M2", "M3", "M0r")
REQUIRED = ("M0", "M1", "M2")
Z_MODELS = ("M1", "M2", "M3")
DECODER = dict(extra_blocks=2)               # identity-initialised adaLN-zero blocks appended to the loaded Stage-1 D_z (width 256, 8 heads): 4 + 2 = 6 blocks
BACKBONE = S2.BACKBONE                       # width 768, 8 blocks, 12 heads, MLP x4, learned positions, adaLN-zero FiLM on (s_0, G); ~68 M parameters

# training (Section 11 of the plan).  Everything not listed is the Stage-2 recipe.
TRAIN = dict(lr=1e-4, wd=0.01, warmup=5000, batch=128, grad_clip=1.0, ema=0.999, max_steps=100000, final_lr_frac=0.05,
             eval_every=1000, min_steps=25000, patience=15, log_every=100,
             n_dec_frames=4,                 # dense terms on 4 random future frames per sequence per step (unbiased; all 63 frames at validation), L_z on all 63
             n_gt_frames=1,                  # M3: the teacher-z dense term on 1 of those 4 frames per sequence (unbiased; keeps the step cost at +25 %)
             dec_chunk=128,                  # maps per decoder call in training / validation (memory only: the runs share GPUs with other users)
             lambda_z=1.0, beta_pred=1.0, beta_gt=0.25,
             select="L_C",                   # checkpoint = EMA state at the minimum VALIDATION dense loss of the predicted-z path (the final task)
             grad_inspect_steps=(500, 1000, 2000, 3000, 5000))
# early stopping: only after min_steps AND only when neither the selection loss (val L_C) nor the full objective improved for `patience` checks

M0_REUSE = dict(
    reused="Stage-2 B0 checkpoint",
    why="identical backbone, inputs, loss and recipe (AdamW 1e-4, wd 0.01, 5 k warm-up, cosine, batch 128, clip 1.0, EMA 0.999, patience 15); "
        "its budget was 80 k steps (TACO used all 80 k, best at 79 k; ARCTIC stopped by patience at 48 k, best at 33 k)",
    parity_check="M0r = the same model retrained under this study's 100 k / 25 k protocol; lowest priority in the GPU queue; the M2-vs-M0 verdict is re-read against it when it exists")

# pre-registered decision rule (fixed before any M2 / M3 result was looked at; the Stage-2 rule where it applies)
DENSE = "E_C"
STRUCT_DECISION = S2.STRUCT_DECISION         # part_hamming, amount_l1, centroid, normal, q_rel_l1
LONG_HORIZON = ("E_C_last16", "jitter_dev_dense", "jitter_dev_r2")
HIGHER_BETTER = S2.HIGHER_BETTER
DECISION = dict(
    rel_min=0.02,                  # "better" / "worse": |relative change| >= 2 % AND the paired take-cluster bootstrap 95 % CI excludes 0; otherwise "similar"
    struct_majority=3,             # "structure improves": >= 3 of the 5 structural / wrench metrics better and none worse
    # adaptation axis (M2 vs M1) and direct axis (M2 vs M0), each read on the dense E_C first, then on the structural block:
    #   improves / beats   dense better
    #   structure_only     dense similar, structure improves
    #   none / similar     dense similar, structure does not improve
    #   worse              dense worse
    # cases (all that apply are listed, e.g. "C+D"):
    #   A  M2 better than M1 AND better than M0 (dense)
    #   B  M2 better than M1 (dense or structure), M2 vs M0: dense similar and no structural majority
    #   C  M2 vs M0: dense similar, structure improves (whether or not M2 improves on M1)
    #   D  M2 vs M1: dense not better and no structural majority  (decoder adaptation did not help)
    #   E  M2 worse than M0 (dense)
    z_better="test L_z of M2 lower than M1's by the same rule (paired, >= 2 %, CI excludes 0)",
    robust="same predicted z into both decoders: D_M2 gives a lower dense error than D_M1 on M1's z_hat AND on M2's z_hat (yes), on one of them (partial), on neither (no)",
    ambiguous="extra seed for M2 only if the M2-vs-M0 dense difference is >= 2 % in either direction while its CI includes 0",
    control_4blk="only if M2 is better than M1: one control run with the 4-block decoder under the M2 protocol, to separate capacity from budget / selection",
    heldout_probe_trigger=0.5,     # the held-out adaptation probe (zj_probe.py) is run only if RMSE(z_hat - z*) on TRAIN at the selected M2 step is below this fraction of the VALIDATION value,
                                   # i.e. only if the decoder did not see held-out-like z errors during joint training
)
NOISE = dict(seed=0, n_draws=1,              # decoder-mismatch diagnostic C: ONE perturbation level, matched to the model's own test prediction error
             primary="gaussian: eps_t ~ N(mu_bin, Sigma_bin), the mean and covariance of the model's test errors z_hat - z* in the horizon bin of t",
             secondary="permuted: the error trajectory z_hat - z* of ANOTHER test sequence (same frames) added to the teacher z* (exact error distribution, independent of the content)")


def ds_out(ds):
    return OUT / ds


def run_name(model, seed=SEED, tag=""):
    return f"{model}_seed{seed}{tag}"


def ckpt_path(ds, name, which="best"):
    return CKPT / ds / (f"{name}.pt" if which == "best" else f"{name}_{which}.pt")


def preds_path(ds, name, part="test"):
    return ds_out(ds) / "preds" / f"{name}_{PROTOCOL}{'' if part == 'test' else '_' + part}.npz"


def metrics_path(ds, name, part="test"):
    return ds_out(ds) / "metrics" / f"{name}_{PROTOCOL}{'' if part == 'test' else '_' + part}.npz"


def train_log_path(ds, name):
    return ds_out(ds) / "train_logs" / f"{name}.csv"


def reused_run(model):
    """Stage-2 run name behind a reused model (M0 -> B0_seed0, M1 -> B1_seed0), else None."""
    r = MODELS[model]["reuse"]
    return None if r is None else S2.run_name(r)


def model_ckpt(ds, model, seed=SEED):
    return S2.ckpt_path(ds, reused_run(model)) if MODELS[model]["reuse"] else ckpt_path(ds, run_name(model, seed))


def model_preds(ds, model, part="test", seed=SEED):
    return S2.preds_path(ds, reused_run(model), part) if MODELS[model]["reuse"] else preds_path(ds, run_name(model, seed), part)


def model_metrics(ds, model, part="test", seed=SEED):
    return S2.metrics_path(ds, reused_run(model), part) if MODELS[model]["reuse"] else metrics_path(ds, run_name(model, seed), part)


def model_train_log(ds, model, seed=SEED):
    return S2.train_log_path(ds, reused_run(model)) if MODELS[model]["reuse"] else train_log_path(ds, run_name(model, seed))


write_json, read_json, md5 = S2.write_json, S2.read_json, S2.md5
cluster_bootstrap, paired_cluster_bootstrap = S2.cluster_bootstrap, S2.paired_cluster_bootstrap
