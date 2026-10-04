"""Hand-contact predictive-information diagnostic (oracle study): shared definitions.

Question. Does CAUSAL hand information -- the current hand state H_t or a short hand history
H_{t-k:t}, expressed in the object frame -- carry predictive information about FUTURE contact
evolution beyond what the current contact vector field already sees (C_t, G, tau_local,t)?

Everything else is inherited, not re-derived: the 64-frame sequences, the 512-D canonical contact
vectors C_t, the geometry descriptor G, the object trajectories (hier_contact_gen: sequences.npz,
manifests/fixed0.json, the FoldData normalisation) and the temporal-event definitions (spike =
d_t above the train Q90, merged runs, onset / release / amount / spatial / mixed categories,
persistence; temporal_contact_events/tce_common.py, tce_analysis.py), applied here to EVERY
sequence of the fixed split (the earlier analysis labelled the test set only).

Hand representation (per dataset, documented in cache/hand_representation.json):
  TACO    the 100 MANO surface points BimArt predicts (assets/part_fps_hand_index_100.npy), of
          the contacting hand, in the frame of the contacted object: the gen3 sequences store
          them in the target frame (`kp`); tool-role sequences re-express them in the tool frame
          with the cached tool pose (tool_pos, tool_rot6d = first two COLUMNS of R_rel).
  ARCTIC  the same 100 sampled MANO vertices of the contacting hand in the object's canonical
          (rigid, articulation left in) frame: `*_processed_hand_features.npy`
          `{left,right}_hand_sampled_verts_cano`.
  Both: metres, 300-D per frame, stored for t in [-8, 72) (clamped at the take boundaries exactly
  like the object states), so a causal window at t = 0 uses the real pre-onset approach.

Conditions (the only thing that changes between them is the conditioning information):
  F0      C_t + G + tau_local,t                              (what the current vector field sees)
  F1      F0 + H_t                                           current hand state
  F2k{k}  F0 + H_{t-k..t}, k in {1, 4, 8}                    causal hand history (primary k = 8)
  F2rel   F0 + (H_j - H_{j-1}), j = t-7..t                   hand MOTION only (representation ablation)
  F2shuf  F2k8 with the hand history of a random OTHER sequence of the same split (capacity control)
  Ffut    F0 + H_{t-8..t+8}                                  NON-CAUSAL ORACLE UPPER BOUND (never mixed
                                                             with the causal results)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/uhnam/workspace/dexcore")
HIER = REPO / "scripts/research/hier_contact_gen"
TCE = REPO / "scripts/research/temporal_contact_events"
for _p in (str(REPO), str(TCE), str(HIER)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import tce_common as K  # noqa: E402
from tce_analysis import categorize, merge_events  # noqa: E402

OUT = Path("/result/uhnam/dexcore/reports/hand_contact_predictive_info")
DATASETS = ("taco", "arctic")
LABEL = {"taco": "TACO", "arctic": "ARCTIC"}
ROOTS = {d: K.ROOTS[d] for d in DATASETS}
T, PAD, N_HAND, HAND_DIM = 64, 8, 100, 300
HORIZONS = (1, 4, 8)
EVENT_HORIZONS = (4, 8)
TRANSIENT_THR = 0.5
HARD_MM = 0.01
SEEDS = (0, 1, 2)
CONDITIONS = {
    "F0": dict(hand=None, k=0, fut=0, shuffle=False, label="F0: C_t + G + τ"),
    "F1": dict(hand="abs", k=0, fut=0, shuffle=False, label="F1: + H_t"),
    "F2k1": dict(hand="abs", k=1, fut=0, shuffle=False, label="F2 (k=1)"),
    "F2k4": dict(hand="abs", k=4, fut=0, shuffle=False, label="F2 (k=4)"),
    "F2k8": dict(hand="abs", k=8, fut=0, shuffle=False, label="F2: + H_{t-8..t}"),
    "F2rel": dict(hand="rel", k=8, fut=0, shuffle=False, label="F2rel: motion only"),
    "F2shuf": dict(hand="abs", k=8, fut=0, shuffle=True, label="F2shuf: shuffled hand (control)"),
    "Ffut": dict(hand="abs", k=8, fut=8, shuffle=False, label="Ffut: NON-CAUSAL oracle (t−8..t+8)"),
    "Ffut1": dict(hand="abs", k=8, fut=1, shuffle=False, label="Ffut1: NON-CAUSAL oracle (t−8..t+1)"),   # one future frame: the hand displacement of the current step only
}
CAUSAL = [c for c, v in CONDITIONS.items() if v["fut"] == 0]
PRIMARY = ("F0", "F1", "F2k8")
HISTORY_ORDER = ("F1", "F2k1", "F2k4", "F2k8")
# persistent contact-mode transitions: the positives of the event-prediction task
POSITIVE_RULE = "persistent spatial / persistent mixed (persistence_h4 >= 0.5, or the event-level persistence when the h=4 window leaves the sequence) + release, onset+release and onset (regrasp) events"
FRAME_CLASSES = ["all", "non_spike", "spike", "persistent_spatial", "persistent_mixed", "onset", "release", "transient", "amount"]
CLASS_LABEL = {"all": "all valid frames", "non_spike": "non-spike", "spike": "spike", "persistent_spatial": "persistent spatial",
               "persistent_mixed": "persistent mixed", "onset": "onset", "release": "release", "transient": "transient (low-persistence)",
               "amount": "amount-dominant"}


def ds_out(ds):
    return OUT / ds


def hand_cache_path(ds):
    return ds_out(ds) / "cache" / "hand_cache.npz"


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ------------------------------------------------------------------------------ data access
def load_raw(ds):
    """Raw arrays of the dataset (numpy): C (N,64,512) f16, O (N,80,D), meta, manifest, thr."""
    return K.load(ds)


def fold(ds, device, stats=None):
    """The hierarchical study's FoldData (its normalisation, tensors on the device). One dataset
    per process: hc_common reads DC_DATASET at import."""
    os.environ["DC_DATASET"] = ds
    if str(HIER) != sys.path[0]:
        sys.path.insert(0, str(HIER))
    from data import FoldData  # noqa: E402  (scripts/research/hier_contact_gen/data.py)
    return FoldData("fixed", 0, device, stats=stats)


def load_hand_cache(ds):
    z = np.load(hand_cache_path(ds))
    return {k: z[k] for k in z.files}


def load_events(ds):
    E = pd.read_csv(ds_out(ds) / "cache" / "events_all.csv", dtype={"sequence_id": str, "take_key": str, "event_id": str, "event_category": str})
    E["event_category"] = E.event_category.fillna("")
    return E


def load_frames(ds):
    F = pd.read_csv(ds_out(ds) / "cache" / "frames_all.csv", dtype={"sequence_id": str, "take_key": str, "event_id": str, "event_category": str})
    F["event_id"] = F.event_id.fillna(""); F["event_category"] = F.event_category.fillna("")
    return F


# ------------------------------------------------------------------------------ hand windows (index arithmetic)
def window_offsets(cond):
    """Offsets (relative to t, in frames) of the hand frames a condition reads; None for F0.
    'abs': frames t-k .. t+fut;  'rel': differences H_{j}-H_{j-1} for j = t-k+1 .. t+fut, i.e. the
    difference array index j-1 = t-k .. t+fut-1 (D[j] = H[j+1] - H[j])."""
    c = CONDITIONS[cond]
    if c["hand"] is None:
        return None
    if c["hand"] == "abs":
        return list(range(-c["k"], c["fut"] + 1))
    return list(range(-c["k"], c["fut"]))


def hand_dim(cond):
    o = window_offsets(cond)
    return 0 if o is None else len(o) * HAND_DIM


def latest_hand_frame(cond):
    """The latest hand frame (relative to t) a condition can see: <= 0 for every causal condition."""
    c = CONDITIONS[cond]
    if c["hand"] is None:
        return -np.inf
    o = window_offsets(cond)
    return max(o) + (1 if c["hand"] == "rel" else 0)


def assert_causal():
    for c in CAUSAL:
        assert latest_hand_frame(c) <= 0, (c, latest_hand_frame(c))
    assert latest_hand_frame("Ffut") == PAD and latest_hand_frame("Ffut1") == 1


# ------------------------------------------------------------------------------ event labels on every sequence
def label_events(ds, C_all, n_hard, meta, man, thr, q90=None):
    """The temporal-event labelling of tce_analysis.py applied to every sequence. Returns
    (frames DataFrame, events DataFrame, q90)."""
    tr = man["train"]
    if q90 is None:
        d_train = np.concatenate([K.decompose(C_all[i].astype(np.float32), thr)["d"] for i in tr])
        q90 = float(np.quantile(d_train, 0.90))
    split_of = meta.split_set.values
    rows, ev_rows = [], []
    for n in range(len(meta)):
        C = C_all[n].astype(np.float32); r = meta.iloc[n]
        dec = K.decompose(C, thr)
        m = C.sum(1); firm = K.firm_mask(m, n_hard[n], thr)
        onset = (~firm[:-1]) & firm[1:]; release = firm[:-1] & (~firm[1:])
        p4 = K.persistence(C, 4)
        spike = dec["d"] > q90
        ev_id = np.array([""] * (T - 1), dtype=object); ev_cat = np.array([""] * (T - 1), dtype=object)
        ev_trans = np.zeros(T - 1, bool); ev_pos = np.zeros(T - 1, bool); ev_peak = np.zeros(T - 1, np.float32)
        recs = []
        for e_i, (s, e) in enumerate(merge_events(spike, 0)):
            seg = slice(s, e + 1)
            valid = dec["valid"][seg]
            A_sum = float(np.nansum(dec["A"][seg][valid])); S_sum = float(np.nansum(dec["S"][seg][valid]))
            pk = int(s + np.argmax(dec["d"][seg]))
            net = float(np.linalg.norm(C[e + 1] - C[s]))
            recs.append(dict(dataset=ds, example=n, sequence_id=r.sequence_id, take_key=r.take_key, group=r.group, split_set=split_of[n],
                             event_id=f"{n}_{e_i}", start_frame=int(s), end_frame=int(e), peak_frame=pk, duration=int(e - s + 1),
                             peak_d=float(dec["d"][seg].max()), integrated_d=float(dec["d"][seg].sum()), energy=float((dec["d"][seg] ** 2).sum()),
                             contact_mass_before=float(dec["m0"][s]), contact_mass_after=float(dec["m1"][e]),
                             amount_component=A_sum, spatial_component=S_sum,
                             amount_ratio=A_sum / (A_sum + S_sum + K.EPS) if valid.any() else np.nan, n_valid=int(valid.sum()),
                             onset_flag=bool(onset[seg].any()), release_flag=bool(release[seg].any()),
                             persistence_h4=float(p4[pk]), persistence_event=net / (dec["d"][seg].sum() + K.EPS),
                             firm_before=bool(firm[s]), firm_after=bool(firm[e + 1])))
        if recs:
            E = pd.DataFrame(recs)
            E["event_category"] = categorize(E, 0.7, 0.3)
            E["transient_flag"] = E.persistence_h4 < TRANSIENT_THR                      # NaN -> False (as before)
            E["persistent_flag"] = np.where(E.persistence_h4.notna(), E.persistence_h4 >= TRANSIENT_THR, E.persistence_event >= TRANSIENT_THR)
            E["positive_event"] = (E.event_category.isin(["spatial", "mixed"]) & E.persistent_flag) | E.event_category.isin(["release", "onset+release", "onset"])
            for ev in E.itertuples():
                sl = slice(ev.start_frame, ev.end_frame + 1)
                ev_id[sl] = ev.event_id; ev_cat[sl] = ev.event_category; ev_trans[sl] = ev.transient_flag; ev_pos[sl] = ev.positive_event; ev_peak[sl] = ev.peak_d
            ev_rows.extend(E.to_dict("records"))
        rows.append(pd.DataFrame(dict(dataset=ds, example=n, sequence_id=r.sequence_id, take_key=r.take_key, group=r.group, split_set=split_of[n],
                                      t=np.arange(T - 1), d=dec["d"], m_t=dec["m0"], m_t1=dec["m1"], valid_decomp=dec["valid"], A=dec["A"], S=dec["S"],
                                      firm_t=firm[:-1], onset=onset, release=release, pers_h4=p4[:-1], spike=spike,
                                      event_id=ev_id, event_category=ev_cat, transient_flag=ev_trans, positive_event=ev_pos, event_peak_d=ev_peak)))
    F = pd.concat(rows, ignore_index=True)
    E = pd.DataFrame(ev_rows)
    return F, E, q90


def frame_classes(F, h):
    """Frame class of every (example, t) for horizon h from the transitions t .. t+h-1 inside
    Delta_h C_t: 'non_spike' when none of them is a spike; otherwise the class of the dominant
    (largest peak d) event overlapping the window: transient (low-persistence) first, then
    onset / release (onset+release -> release) / amount / persistent spatial / persistent mixed.
    Returns an object array (n_rows,) aligned with F (F sorted by example, t)."""
    n_ex = F.example.nunique()
    assert len(F) == n_ex * (T - 1)
    spike = F.spike.values.reshape(n_ex, T - 1)
    peak = F.event_peak_d.values.reshape(n_ex, T - 1)
    cat = F.event_category.values.reshape(n_ex, T - 1)
    trans = F.transient_flag.values.reshape(n_ex, T - 1)
    out = np.full((n_ex, T - 1), "non_spike", dtype=object)
    for t in range(T - 1):
        lo, hi = t, min(t + h, T - 1)
        w_spike = spike[:, lo:hi]
        any_spike = w_spike.any(1)
        pk = np.where(w_spike, peak[:, lo:hi], -1.0)
        j = pk.argmax(1)
        c = cat[np.arange(n_ex), lo + j]; tr = trans[np.arange(n_ex), lo + j]
        lab = np.where(tr, "transient", np.where(c == "onset", "onset", np.where(np.isin(c, ["release", "onset+release"]), "release",
                       np.where(c == "amount", "amount", np.where(c == "spatial", "persistent_spatial", np.where(c == "mixed", "persistent_mixed", "other"))))))
        out[:, t] = np.where(any_spike, lab, "non_spike")
    return out.reshape(-1)


def event_labels(F, E, h):
    """y_t^(h) = 1 iff a positive (persistent contact-mode transition) event STARTS at a transition
    s with t < s <= t + h; the causal inputs of frame t (C_t, hand frames <= t) therefore end
    strictly before the event's first transition. Valid for t <= (T-2) - h. Returns (y, valid),
    both (n_ex, T-1) bool."""
    n_ex = F.example.nunique()
    starts = np.zeros((n_ex, T - 1), bool)
    if len(E):
        pos = E[E.positive_event]
        starts[pos.example.values, pos.start_frame.values] = True
    y = np.zeros((n_ex, T - 1), bool); valid = np.zeros((n_ex, T - 1), bool)
    for t in range(T - 1 - h):
        y[:, t] = starts[:, t + 1:t + h + 1].any(1); valid[:, t] = True
    return y, valid


# ------------------------------------------------------------------------------ statistics
cluster_bootstrap = K.cluster_bootstrap
cluster_bootstrap_ratio = K.cluster_bootstrap_ratio


def paired_cluster_bootstrap(a, b, clusters, n_boot=1000, seed=0):
    """Mean of (a - b) with a take-level cluster bootstrap CI (paired by frame)."""
    return K.cluster_bootstrap(np.asarray(a, float) - np.asarray(b, float), clusters, n_boot=n_boot, seed=seed)


def score_bootstrap(fn, y, p, clusters, n_boot=500, seed=0):
    """fn(y, p) (e.g. average precision) with a cluster bootstrap over takes."""
    y = np.asarray(y); p = np.asarray(p); clusters = np.asarray(clusters)
    u, inv = np.unique(clusters, return_inverse=True)
    groups = [np.where(inv == g)[0] for g in range(len(u))]
    rng = np.random.default_rng(seed); reps = []
    for _ in range(n_boot):
        sel = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
        if y[sel].any() and (~y[sel]).any():
            reps.append(fn(y[sel], p[sel]))
    return float(fn(y, p)), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))
