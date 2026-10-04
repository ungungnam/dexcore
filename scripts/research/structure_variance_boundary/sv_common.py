"""Structure / variance boundary of the hand-contact state: shared definitions.

Question. Given the dense state X_t = {C_t (512-D canonical contact vector), H_t (100 MANO surface points in the
object frame)}, what is the smallest structured representation Z_t = f(C_t, H_t) that keeps (A) the modelled grasp
wrench capability and (B) the predictability of the grasp's future structured evolution? Everything finer than Z_t is
a candidate within-structure variation.

Inherited, not re-derived: the 64-frame sequences, C_t, G, tau_local and the fixed split (hier_contact_gen), the
hand cache H_t and the temporal recipe (hand_contact_predictive_info), the contact-to-finger assignment, patches,
normals, wrench profiles and event selection (wrench_counterfactual), the take-level bootstrap (temporal_contact_events).

Representation ladder (per analysed hand; parts palm / thumb / index / middle / ring / little):
  R0  participation a_k (6)                                    a_k = causal 3-frame majority of [>= 3 mesh vertices within 1 cm]
  R1  + contact amount m_k (6)                                 area-weighted soft mass sum_v exp(-d_v / 2 cm) A_v / A_total
  R2  + centroid p_k / l, mean toward-hand normal n_k (36)     hard-contact (< 1 cm) vertices, object frame, l = max radius
  R3  + RMS spread s_k / l, patch count c_k (12)               patches: wrench report's primary setting (1 cm / merge 1 cm)
  R4  R2 + H_coarse (42)                                       wrist / l, palm frame, fingertips / l, MCP->tip directions
  R5  R3 + H_coarse
  CONTACT_FULL  C_t (512);   FULL  C_t + H_t (812)
Inactive parts (a_k = 0) carry zeros in every geometric block. Feature frames are stored for t in [-8, 72) (clamped
at the take boundaries like the hand cache) so an 8-frame causal window exists at every t in [0, 64).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
for _p in (REPO / "scripts/research/wrench_counterfactual", REPO / "scripts/research/hand_contact_predictive_info",
           REPO / "scripts/research/temporal_contact_events", REPO / "scripts/research/hier_contact_gen", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hp_common as HP  # noqa: E402
import tce_common as K  # noqa: E402
import wc_common as W  # noqa: E402

OUT = Path("/result/uhnam/dexcore/reports/structure_variance_boundary")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
T, PAD, NF = 64, 8, 80                      # sequence frames, padding, stored frames per sequence (t in [-8, 72))
HIST = 8                                    # causal window t-7 .. t
HORIZONS = (1, 4, 8)
SEEDS = (0, 1, 2)
PARTS = W.PARTS                             # palm, thumb, index, middle, ring, little
N_PARTS = 6
N_MIN_VERTS = 3                             # participation: >= 3 mesh vertices within 1 cm
CONTACT_THR = W.CONTACT_THR["primary"]      # 1 cm
STORE_THR = W.STORE_THR                     # 2 cm (soft mass)
MERGE_R = W.MERGE_R["primary"]
MU = W.MU_PRIMARY
BUDGET = W.BUDGET_PRIMARY
N_DIR = 76
HAND_COARSE_DIM = 42
N_HAND, HAND_DIM = HP.N_HAND, HP.HAND_DIM   # 100 points, 300-D
TIPS = [745, 317, 444, 556, 673]            # MANO fingertip vertices: thumb, index, middle, ring, little
MCP = [13, 1, 4, 10, 7]                     # MANO joints (wrist 0, index 1-3, middle 4-6, little 7-9, ring 10-12, thumb 13-15)
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]

BLOCKS = {"part": 6, "amount": 6, "geom": 36, "topo": 12, "hand": HAND_COARSE_DIM, "C": 512, "H": HAND_DIM}
BLOCK_LABEL = {"part": "participation a_k", "amount": "contact amount m_k", "geom": "centroid p_k + normal n_k",
               "topo": "spread s_k + patch count c_k", "hand": "coarse hand kinematics H_coarse",
               "C": "dense canonical contact C_t (512)", "H": "100 MANO surface points H_t (300)"}
LADDER = {"R0": ["part"], "R1": ["part", "amount"], "R2": ["part", "amount", "geom"], "R3": ["part", "amount", "geom", "topo"],
          "R4": ["part", "amount", "geom", "hand"], "R5": ["part", "amount", "geom", "topo", "hand"],
          "CONTACT_FULL": ["C"], "FULL": ["C", "H"]}
REPS = list(LADDER)
REP_DIM = {r: sum(BLOCKS[b] for b in bs) for r, bs in LADDER.items()}
REP_LABEL = {"R0": "R0 participation", "R1": "R1 + amount", "R2": "R2 + centroid / normal", "R3": "R3 + spread / patches",
             "R4": "R4 = R2 + coarse hand", "R5": "R5 = R3 + coarse hand", "CONTACT_FULL": "dense contact", "FULL": "dense contact + hand"}
INCREMENTS = [("R0", "R1", "amount"), ("R1", "R2", "geom"), ("R2", "R3", "topo"), ("R2", "R4", "hand"), ("R3", "R5", "hand"),
              ("R5", "FULL", "dense detail"), ("CONTACT_FULL", "FULL", "H (100 points)")]
TARGETS = {"T1": "future participation a_{t+h}", "T2": "future amount m_{t+h}", "T3": "future centroid / normal (active parts)",
           "T4": "future wrench profile q_{t+h}", "T5C": "future dense contact C_{t+h}", "T5H": "future hand H_{t+h}"}
SATURATION_THRESHOLDS = (0.025, 0.05, 0.10)


def ds_out(ds):
    return OUT / ds


def write_json(path, obj):
    HP.write_json(path, obj)


def feature_path(ds):
    return ds_out(ds) / "cache" / "features.npz"


# ------------------------------------------------------------------------------ feature cache access
def load_features(ds):
    z = np.load(feature_path(ds), allow_pickle=True)
    return {k: z[k] for k in z.files}


def causal_majority(a_raw):
    """(N, NF, 6) raw indicator -> causal 3-frame majority over stored frames {i-2, i-1, i} (never a later frame)."""
    a = a_raw.astype(np.int8)
    prev1 = np.concatenate([a[:, :1], a[:, :-1]], 1)
    prev2 = np.concatenate([a[:, :1], a[:, :1], a[:, :-2]], 1)
    return ((a + prev1 + prev2) >= 2).astype(np.float32)


def block_arrays(F, hand100=None):
    """Raw (un-normalised) feature blocks, (N, NF, d) float32, with the geometric blocks masked by the causal
    participation. hand100: (N, NF, 100, 3) from the hand cache for the H block (optional)."""
    a = causal_majority(F["n_hard"] >= N_MIN_VERTS)                                   # (N, NF, 6)
    mask = a[..., None]
    p = F["p"] * mask; n = F["nrm"] * mask; s = F["s"] * a; c = F["c"].astype(np.float32) * a
    out = {"part": a, "amount": F["m"].astype(np.float32),
           "geom": np.concatenate([p.reshape(*p.shape[:2], -1), n.reshape(*n.shape[:2], -1)], -1).astype(np.float32),
           "topo": np.concatenate([s, c], -1).astype(np.float32), "hand": F["hand"].astype(np.float32),
           "C": F["C_pad"].astype(np.float32)}
    if hand100 is not None:
        out["H"] = hand100.reshape(len(hand100), NF, -1).astype(np.float32)
    return out


def representation(blocks, name):
    return np.concatenate([blocks[b] for b in LADDER[name]], -1)


def fit_norm(X, idx_train, frames=slice(PAD, PAD + T)):
    """Per-dimension mean / sd on the training sequences' frames t in [0, 64) (sd < 1e-6 -> 1: constant dims)."""
    x = X[idx_train][:, frames].reshape(-1, X.shape[-1]).astype(np.float64)
    mu = x.mean(0); sd = x.std(0); sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


# ------------------------------------------------------------------------------ events (same selection as the wrench report)
def select_events(ds, split="test"):
    """Every event of one split with pre = start and post = end + 1 frames, firm flags and the class of the wrench
    report (persistent_spatial / persistent_mixed / onset / release / transient / amount). Written to
    <ds>/events_<split>.csv. Test events are checked against the wrench report's events_selected.csv."""
    E = HP.load_events(ds); Fr = HP.load_frames(ds)
    D = HP.load_raw(ds); meta = D["meta"]; C = D["C"]
    te = E[E.split_set == split].copy().reset_index(drop=True)
    firm = Fr.pivot(index="example", columns="t", values="firm_t")
    rows = []
    for r in te.itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        pre, post = s, e + 1
        firm_row = firm.loc[r.example]
        is_firm = lambda t: bool(firm_row[t]) if t <= 62 else bool(r.firm_after)
        m = meta.iloc[r.example]
        dC = float(np.linalg.norm(C[r.example, post].astype(np.float32) - C[r.example, pre].astype(np.float32)))
        rows.append(dict(dataset=ds, event_id=r.event_id, example=int(r.example), sequence_id=r.sequence_id, take_key=r.take_key, group=r.group,
                         category=m.category, role=m.role, hand=m.hand, mesh_id=str(m.mesh_id), t0=int(m.t0), split=split,
                         cls=W.classify(te.iloc[[r.Index]])[0], event_category=r.event_category, start=s, end=e, pre_frame=pre, post_frame=post,
                         firm_pre=is_firm(pre), firm_post=is_firm(post), firm_through=all(is_firm(t) for t in range(pre, post + 1)),
                         duration=int(r.duration), peak_d=float(r.peak_d),
                         persistence_h4=float(r.persistence_h4) if pd.notna(r.persistence_h4) else np.nan,
                         amount_ratio=float(r.amount_ratio) if pd.notna(r.amount_ratio) else np.nan, dC=dC))
    S = pd.DataFrame(rows)
    S["primary"] = (S.cls == "persistent_spatial") & S.firm_pre & S.firm_post & S.firm_through
    ds_out(ds).mkdir(parents=True, exist_ok=True)
    S.to_csv(ds_out(ds) / f"events_{split}.csv", index=False)
    return S


def load_events(ds, split="test"):
    p = ds_out(ds) / f"events_{split}.csv"
    if not p.exists():
        return select_events(ds, split)
    return pd.read_csv(p, dtype={"sequence_id": str, "take_key": str, "event_id": str, "mesh_id": str})


# ------------------------------------------------------------------------------ statistics
def cluster_bootstrap(values, clusters, n_boot=1000, seed=0, stat="mean"):
    return K.cluster_bootstrap(values, clusters, n_boot=n_boot, seed=seed, stat=stat)


def paired_cluster_bootstrap(a, b, clusters, n_boot=1000, seed=0):
    """mean(a - b) with a take-cluster bootstrap CI (paired per frame / event)."""
    return K.cluster_bootstrap(np.asarray(a, float) - np.asarray(b, float), clusters, n_boot=n_boot, seed=seed)


def ratio_cluster_bootstrap(num, den, clusters, n_boot=1000, seed=0):
    """sum(num) / sum(den) over takes resampled with replacement (a relative gap of two error sums)."""
    num = np.asarray(num, float); den = np.asarray(den, float); clusters = np.asarray(clusters)
    ok = ~(np.isnan(num) | np.isnan(den)); num, den, clusters = num[ok], den[ok], clusters[ok]
    u, inv = np.unique(clusters, return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(u)); sd = np.bincount(inv, weights=den, minlength=len(u))
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        reps.append((w * sn).sum() / max((w * sd).sum(), 1e-12))
    return float(num.sum() / max(den.sum(), 1e-12)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def spearman_cluster(x, y, clusters, n_boot=500, seed=0):
    return K.spearman_cluster(x, y, clusters, n_boot=n_boot, seed=seed)[:3]


def representation_definitions():
    """Table 1 content."""
    rows = []
    for r in REPS:
        rows.append(dict(representation=r, label=REP_LABEL[r], blocks=" + ".join(LADDER[r]), dim=REP_DIM[r],
                         window_dim=REP_DIM[r] * HIST, components="; ".join(BLOCK_LABEL[b] for b in LADDER[r])))
    return pd.DataFrame(rows)
