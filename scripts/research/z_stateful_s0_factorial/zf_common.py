"""2 x 2 factorial follow-up on the z-mediated temporal contact generator: shared definitions.

Two remaining architectural hypotheses, tested together on TACO and ARCTIC (one seed, same data / split / teacher / evaluation as before):
  AXIS A  latent temporal formulation   A0 whole-sequence (open-loop) z prediction      A1 stateful z evolution (learned z_0, rollout)
  AXIS B  contact decoding formulation  B0 absolute decoding  C_t = D_abs(z_t, G_t)      B1 s_0-preserving  C_t = s_0 + D_delta(s_0, z_t, G_t)

              absolute C      s_0-residual C
  whole z       M00              M01
  stateful z    M10              M11

M00 is the jointly trained whole-sequence model of the previous follow-up (its M2 checkpoint, reused: same backbone, same 6-block
decoder, same recipe and selection as the three models trained here).  D0, the external reference, is the direct dense model of
Stage 2 (B0, reused).  Nothing else changes: no r, no diffusion, no new latent, no Stage-1 training, no loss other than
L_C + lambda_z L_z (+ lambda_z0 L_z0 for the stateful models).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path("/home/uhnam/workspace/dexcore")
HERE = Path(__file__).resolve().parent
ZJ_DIR = REPO / "scripts/research/z_joint_decoder_followup"
if str(ZJ_DIR) not in sys.path:
    sys.path.insert(0, str(ZJ_DIR))
import zj_common as ZJ  # noqa: E402  (also puts the Stage-2 / Stage-1 / evaluation code on sys.path)

S2 = ZJ.S2
OUT = Path(os.environ.get("ZF_OUT", "/result/uhnam/dexcore/reports/z_stateful_s0_factorial"))        # ZF_OUT / ZF_CKPT: dry runs of the analysis chain
CKPT = Path(os.environ.get("ZF_CKPT", "/ckpt/uhnam/dexcore/z_stateful_s0_factorial"))
DATASETS = S2.DATASETS
LABEL = S2.LABEL
T, PAD, TF, N_PTS = S2.T, S2.PAD, S2.TF, S2.N_PTS
SEED = 0
EXTRA_SEED = 1                               # only for the relevant pair if a decisive comparison is ambiguous (DECISION["ambiguous"])
PROTOCOL = "A"                               # fixed GT s_0
N_BOOT = 1000
HORIZON_BINS = ((1, 16), (17, 32), (33, 48), (49, 63))      # the plan's early / middle / late / late bins
EARLY_FRAMES = (1, 4, 8, 16)

# model -> (stateful, residual decoder, reused checkpoint)
TRANSITION_CHOICE = dict(
    chosen="residual MLP, width 1024, 6 blocks, 55.7 M parameters (the plan: '4-6 Transformer / residual blocks ... multi-head attention or strong residual MLP')",
    why="measured before training on one empty A6000, batch 128, 63-step rollout with full back-propagation: a 6-block width-768 transformer over the 7 tokens "
        "[state | G | tau_t..t+4] (51.6 M) took 1.50 s and 14.4 GB per rollout, the residual MLP 0.48 s and 2.4 GB; width 1024 (above the recommended 512-768) "
        "keeps the stateful models within 15 % of the whole-sequence models' parameter count",
)
MODELS = {
    "M00": dict(stateful=False, residual=False, reuse="zj:M2", label="whole-sequence z + absolute decoder (the previous follow-up's M2, reused)"),
    "M01": dict(stateful=False, residual=True, reuse=None, label="whole-sequence z + s_0-preserving decoder: C_t = s_0 + D_delta(s_0, z_hat_t, G_t)"),
    "M10": dict(stateful=True, residual=False, reuse=None, label="stateful z (z_hat_0 = I(s_0, G), z_hat_t+1 = z_hat_t + F_state(z_hat_t, G, tau_t:t+4)) + absolute decoder"),
    "M11": dict(stateful=True, residual=True, reuse=None, label="stateful z + s_0-preserving decoder"),
}
Z_MODELS = ("M00", "M01", "M10", "M11")
TRAINED_HERE = ("M01", "M10", "M11")
DIRECT = "D0"                                # Stage-2 B0 (reused prediction / metric files)

BACKBONE = S2.BACKBONE                       # whole-sequence temporal network: width 768, 8 blocks, 12 heads (67.9 M)
STATE = dict(width=1024, depth=6, dropout=0.1, cond_width=256,            # transition: residual MLP (4 x expansion) on [state | tau_t .. tau_t+k], FiLM on G in every block
             k=4,                            # look-ahead of the local trajectory context (the plan's primary setting; not swept)
             init_width=256, init_depth=4, init_heads=8, init_pool=4,    # I_theta: point-token contact encoder of (geometry tokens at frame 0 | s_0), FiLM on G; trained from scratch
             init_aug_frames=0,              # the plan's L_z0 on frame 0 only (decided by the pilot, INIT_AUG / init_aug_decision.json)
             ckpt_steps=False)               # activation checkpointing of the rollout steps (memory only)
INIT_AUG = dict(
    what="option considered: compute L_z0 on frame 0 and, for the initial-state module only, also on a few random later frames of the same training "
         "sequence (each encoded by I_theta as if it were an initial frame).  NOT used: see OUT/init_aug_decision.json.",
    why="frame 0 alone gives the module 1 312 / 1 005 training maps; the pilot zf_pilot_init.py checked whether that generalises",
    rule="augmentation is used iff the held-out RMSE of z_hat_0 with frame-0-only supervision exceeds 1.5 x the RMSE with augmentation in the pilot",
    threshold=1.5,                           # applied by zf_pilot_decide.py to the validation RMSE at the pilot's last step, on either dataset
)
DECODER = dict(extra_blocks=2)               # both decoders: the Stage-1 D_z blocks (loaded) + 2 identity-initialised blocks, width 256; D_delta has one extra
                                             # input channel per point (the standardised s_0; zero-initialised weights) and a zero-initialised output layer (starts at C_t = s_0)
TRAIN = dict(ZJ.TRAIN, lambda_z0=1.0, bptt="full")          # the previous follow-up's recipe (AdamW 1e-4, wd 0.01, 5 k warm-up, cosine to 100 k, batch 128, clip 1.0, EMA 0.999,
                                                            # validation every 1 k, no stop before 25 k, patience 15, dense terms on 4 random frames, selection on validation L_C)

DENSE = "E_C"
STRUCT_DECISION = S2.STRUCT_DECISION         # part_hamming, amount_l1, centroid, normal, q_rel_l1
HIGHER_BETTER = S2.HIGHER_BETTER
DECISION = dict(
    rel_min=0.02,                  # "better" / "worse": |relative change| >= 2 % AND the paired take-cluster bootstrap 95 % CI excludes 0; otherwise "similar"
    struct_majority=3,             # "structure improves": >= 3 of the 5 structural / wrench metrics better and none worse
    stateful_contact="M10 vs M00 and M11 vs M01 on dense E_C: both better -> yes; one better and none worse -> partial; any worse and none better -> worse; else no",
    decoder_contact="M01 vs M00 and M11 vs M10 on dense E_C, same reading",
    stateful_generalisation="test L_z (frames 1..63) of M10 vs M00 and M11 vs M01 by the same rule, and the gap test - train of the z RMSE",
    early_fix="residual vs absolute at frames 1-16 (both pairs better -> the decoder helps early); 'fixes' additionally requires the residual model not to be worse than D0 in frames 1-16",
    complementary="M11 better than both M01 and M10 on dense E_C -> yes; better than exactly one -> partial; else no (the interaction contrast is reported with its CI)",
    best_z="the z model with the lowest VALIDATION dense error E_C (never the test error) is compared with D0",
    cases="A stateful is the main fix | B s_0 preservation is the main fix | C both help (M11 better than M00, M01 and M10) | D stateful helps TACO only (cross-dataset) | E nothing beats M00 or D0",
    ambiguous="one extra seed for the relevant pair only if a decisive dense comparison differs by >= 2 % while its CI includes 0",
)


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


def model_ckpt(ds, model, seed=SEED):
    return ZJ.model_ckpt(ds, "M2") if MODELS[model]["reuse"] else ckpt_path(ds, run_name(model, seed))


def model_preds(ds, model, part="test", seed=SEED):
    if model == DIRECT:
        return S2.preds_path(ds, S2.run_name("B0"), part)
    return ZJ.model_preds(ds, "M2", part) if MODELS[model]["reuse"] else preds_path(ds, run_name(model, seed), part)


def model_metrics(ds, model, part="test", seed=SEED):
    if model == DIRECT:
        return S2.metrics_path(ds, S2.run_name("B0"), part)
    return ZJ.model_metrics(ds, "M2", part) if MODELS[model]["reuse"] else metrics_path(ds, run_name(model, seed), part)


def model_train_log(ds, model, seed=SEED):
    if model == DIRECT:
        return S2.train_log_path(ds, S2.run_name("B0"))
    return ZJ.model_train_log(ds, "M2") if MODELS[model]["reuse"] else train_log_path(ds, run_name(model, seed))


write_json, read_json, md5 = S2.write_json, S2.read_json, S2.md5
cluster_bootstrap, paired_cluster_bootstrap = S2.cluster_bootstrap, S2.paired_cluster_bootstrap
