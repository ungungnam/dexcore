#!/usr/bin/env python
"""Stage 1: the first-firm-contact sequence set of one dataset (DC_DATASET).

For every (take, group) the per-frame contact vectors are taken from the dynamic-contact cache
(ARCTIC, OakInk2: the cache holds every frame of every full 64-frame window) or recomputed from the
raw per-vertex distances with the cached operator (TACO: the cache started at frame 8 and dropped
the tail, which would exclude a quarter of the onsets). Firm contact, episode onsets and the
T-frame sequences follow hc_common. Writes
  OUT/sequences.npz        C (N,T,512) f16, O (N,T+16,D) f32 object states for t in [-8,T+8),
                           g_index (N,) -> geometry row, hand (N,) 0/1, role (N,) 0/1,
                           G_nearest/G_coverage (n_mesh,512), G_radius (n_mesh,), G_extent (n_mesh,3),
                           G_cat, G_mesh (n_mesh,), canonical points per category
  OUT/sequences_meta.csv   one row per example: ids, onset frame, episode index, split_set
  OUT/manifests/fixed0.json  train / val / test example indices of the one fixed take-level split
  OUT/config/sequence_construction.json   the exact rule used
"""
from __future__ import annotations

import json
import logging
import re
import time

import numpy as np
import pandas as pd

import hc_common as H
from hc_common import C

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("build_seq")


def taco_frames():
    """Yield (group, sequence_id, mesh, subject, verb, split, X (n,512), n_hard (n,), frames)."""
    from src.analysis.canonical import backends as BR
    from build_full_cache import weight_matrix
    be = BR.load("normalized", str(C.BACKEND_DIR))
    wi = pd.read_csv(C.STUDY / "window_index.csv").drop_duplicates("sequence_id")
    files = pd.read_csv(C.SEQ / "sequence_index.csv").set_index("sequence_id")["file"]
    for cat, role, hand in C.GROUPS:
        rows = wi[wi[f"{role}_cat"] == cat]
        known = set(be.mesh_ids(cat))
        for mesh, g in rows.groupby(f"{role}_mesh"):
            mesh = f"{int(mesh):03d}"
            if mesh not in known:
                continue
            W = weight_matrix(be, cat, mesh)
            for _, r in g.iterrows():
                with np.load(C.SEQ / "sequences" / files[r.sequence_id]) as z:
                    d = z[f"contact_{'left' if hand == 'L' else 'right'}"]
                n_tool = int(r.n_tool)
                d = d[:, :n_tool] if role == "tool" else d[:, n_tool:]
                X = (np.exp(-d / 0.02).astype(np.float32) @ W.T).astype(np.float32)
                yield ((cat, role, hand), r.sequence_id, mesh, str(int(r.subject)), r.verb, r.split, X,
                       (d < C.HARD_MM).sum(1), np.arange(len(d)))


def cache_frames():
    for cat, role, hand in C.GROUPS:
        X, M, _ = C.load_group(cat, role, hand, stride=1)
        for sid, idx in M.groupby("sequence_id").indices.items():
            idx = idx[np.argsort(M.frame.values[idx])]
            f = M.frame.values[idx]
            # keep the longest contiguous run of frames (windows are contiguous from frame 0)
            breaks = np.where(np.diff(f) != 1)[0]
            if len(breaks):
                seg = np.split(np.arange(len(f)), breaks + 1)
                keep = max(seg, key=len); idx, f = idx[keep], f[keep]
            r = M.iloc[idx[0]]
            yield ((cat, role, hand), sid, str(r.mesh_id), str(r.subject), str(r.verb), str(r.split), X[idx],
                   M.n_hard.values[idx], f)


def take_key(sid):
    """The split unit. TACO / ARCTIC: the sequence (both hands, tool and target share it). OakInk2:
    the RECORDING (everything before __lh__/__rh__), so that the other hand's segment and the
    other object parts of the same recording, which share the object trajectory / hand motion,
    can never sit on the other side of the split."""
    if H.DATASET.startswith("oakink2"):
        return re.match(r"^(.*)__(lh|rh)__", sid).group(1)
    return sid


def main():
    t_start = time.time()
    thr = C.zero_threshold()
    states = H.ObjectStates()
    geo = H.geometry_table()
    gkeys = sorted(geo)
    gidx = {k: i for i, k in enumerate(gkeys)}
    H.OUT.mkdir(parents=True, exist_ok=True); (H.OUT / "manifests").mkdir(exist_ok=True); (H.OUT / "config").mkdir(exist_ok=True)
    Cs, Os, meta = [], [], []
    n_take = n_firm = n_onset = 0
    src = taco_frames() if H.DATASET == "taco" else cache_frames()
    for (cat, role, hand), sid, mesh, subj, verb, split, X, n_hard, frames in src:
        n_take += 1
        m = X.sum(1)
        firm = H.firm_mask(m, n_hard, thr)
        if not firm.any():
            continue
        n_firm += 1
        onsets = H.episode_onsets(firm)
        n_onset += len(onsets)
        for ei, o in enumerate(onsets):
            if o + H.T > len(X):
                continue
            Cs.append(X[o:o + H.T].astype(np.float16))
            Os.append(states.get(sid, frames[o] + np.arange(-H.CTX_PAD, H.T + H.CTX_PAD)))
            meta.append(dict(sequence_id=sid, take_key=take_key(sid), group=C.gname(cat, role, hand), category=cat, role=role,
                             hand=hand, mesh_id=mesh, subject=subj, verb=verb, orig_split=split, t0=int(frames[o]),
                             n_frames_take=int(len(X)), episode_index=ei, n_episodes=len(onsets),
                             first_onset=int(ei == 0), g_index=gidx[(cat, mesh)],
                             firm_frac=float(firm[o:o + H.T].mean()), leaves_contact=int(not firm[o:o + H.T].all()),
                             mass0=float(m[o]), mass_mean=float(m[o:o + H.T].mean())))
    meta = pd.DataFrame(meta)
    Cs = np.stack(Cs); Os = np.stack(Os).astype(np.float32)
    log.info("%s: %d (take, group) pairs, %d with firm contact, %d onsets, %d sequences kept (%.0fs)",
             H.DATASET, n_take, n_firm, n_onset, len(meta), time.time() - t_start)
    # ---- one fixed train / val / test split at the take level (a take never crosses splits).
    # TACO and ARCTIC: the dataset's own labels (TACO train vs test_1..4, ARCTIC train vs test);
    # OakInk2 has none, so a seeded 80/10/10 split of the takes, stratified by group. Validation =
    # 10 % of the training takes (seeded), used only for early stopping.
    takes = meta.drop_duplicates("take_key").set_index("take_key")
    rng = np.random.default_rng(0)
    extra_takes = set()
    if H.DATASET == "taco":
        # test_1 = the dataset's seen-everything test (all groups, meshes and subjects occur in train);
        # test_2..4 hold out meshes / actions / both and two whole groups have no training take at all,
        # so they are kept apart as a labelled 'test_extra' set (evaluated, reported separately).
        test_takes = set(takes.index[takes.orig_split == "test_1"])
        extra_takes = set(takes.index[takes.orig_split.isin(["test_2", "test_3", "test_4"])])
        train_pool = sorted(takes.index[takes.orig_split == "train"])
        split_rule = "dataset labels: train -> train/val, test_1 -> test (test_2..4 -> test_extra)"
    elif H.DATASET == "arctic":
        test_takes = set(takes.index[takes.orig_split != "train"])
        train_pool = sorted(takes.index[takes.orig_split == "train"])
        split_rule = "dataset labels: train -> train/val, test -> test"
    else:
        # recordings in a seeded random order; test = the first recordings until 10 % of the sequences
        recs = np.array(sorted(meta.take_key.unique())); recs = recs[rng.permutation(len(recs))]
        n_per = meta.groupby("take_key").size()
        test_takes, acc = set(), 0
        for r in recs:
            if acc >= 0.10 * len(meta):
                break
            test_takes.add(r); acc += int(n_per[r])
        train_pool = sorted(set(recs) - test_takes)
        split_rule = "seeded 80/10/10 by recording (seed 0)"
    train_pool = np.array(train_pool)
    val_takes = set(rng.choice(train_pool, max(1, int(round(0.1 * len(train_pool)))), replace=False))
    meta["split_set"] = np.where(meta.take_key.isin(test_takes), "test", np.where(meta.take_key.isin(extra_takes), "test_extra",
                                 np.where(meta.take_key.isin(val_takes), "val", "train")))
    meta.to_csv(H.OUT / "sequences_meta.csv", index=False)
    cano = {cat: np.load(C.CACHE / "frames_full" / f"{C.gname(*g)}.npz", allow_pickle=True)["canonical_points"]
            for g in C.GROUPS for cat in [g[0]]}
    np.savez_compressed(H.SEQ_FILE, C=Cs, O=Os, g_index=meta.g_index.values, hand=(meta.hand.values == "R").astype(np.int64),
                        role=(meta.role.values == "target").astype(np.int64),
                        G_nearest=np.stack([geo[k]["nearest"] for k in gkeys]), G_coverage=np.stack([geo[k]["coverage"] for k in gkeys]),
                        G_radius=np.array([geo[k]["radius_m"] for k in gkeys], np.float32), G_extent=np.stack([geo[k]["extent_m"] for k in gkeys]),
                        G_cat=np.array([k[0] for k in gkeys]), G_mesh=np.array([k[1] for k in gkeys]),
                        canonical_cats=np.array(sorted(cano)), canonical_points=np.stack([cano[c] for c in sorted(cano)]),
                        state_names=np.array(states.names))
    tr, va, te, tx = (meta.split_set.values == k for k in ("train", "val", "test", "test_extra"))
    man = dict(split="fixed", fold=0, rule=split_rule, split_unit="recording" if H.DATASET.startswith("oakink2") else "sequence",
               train=np.where(tr)[0].tolist(), val=np.where(va)[0].tolist(), test=np.where(te)[0].tolist(), test_extra=np.where(tx)[0].tolist(),
               n_train_takes=int(meta.take_key[tr].nunique()), n_val_takes=int(meta.take_key[va].nunique()), n_test_takes=int(meta.take_key[te].nunique()))
    json.dump(man, open(H.OUT / "manifests" / "fixed0.json", "w"))
    seen_groups = set(meta.group[tr]); seen_mesh = set(zip(meta.category[tr], meta.mesh_id[tr]))
    log.info("fixed split (%s): train %d / val %d / test %d (+%d extra) sequences (%d / %d / %d %ss); test groups %d of %d, all seen in train: %s; test meshes unseen in train: %d",
             split_rule, tr.sum(), va.sum(), te.sum(), tx.sum(), man["n_train_takes"], man["n_val_takes"], man["n_test_takes"], man["split_unit"],
             meta.group[te].nunique(), meta.group.nunique(), set(meta.group[te]) <= seen_groups,
             len(set(zip(meta.category[te], meta.mesh_id[te])) - seen_mesh))
    H.write_json(H.OUT / "config" / "sequence_construction.json", dict(
        dataset=H.DATASET, T=H.T, fps=H.FPS, stride=1, zero_mass=thr, hard_mm=C.HARD_MM, firm_rule="n_hard>0 and mass>=zero_mass",
        firm_persist_frames=H.FIRM_PERSIST, episode_gap_frames=H.EPISODE_GAP, local_context_offsets=H.LOCAL_OFFSETS,
        context_pad=H.CTX_PAD, contact_source="raw distances + cached operator" if H.DATASET == "taco" else "dynamic-contact cache",
        n_take_group_pairs=n_take, n_with_firm_contact=n_firm, n_onsets=n_onset, n_sequences=len(meta),
        n_first_onset_sequences=int(meta.first_onset.sum()), n_takes=int(meta.take_key.nunique()),
        state_dim=states.D, state_names=states.names, split=split_rule, split_unit=man["split_unit"], n_train_seq=int(tr.sum()), n_val_seq=int(va.sum()),
        n_test_seq=int(te.sum()), n_test_extra_seq=int(tx.sum()), n_train_takes=man["n_train_takes"], n_val_takes=man["n_val_takes"], n_test_takes=man["n_test_takes"],
        test_groups_all_seen_in_train=bool(set(meta.group[te]) <= seen_groups), n_test_meshes_unseen=len(set(zip(meta.category[te], meta.mesh_id[te])) - seen_mesh)))
    log.info("done: %d sequences, %d takes, %d first-onset; %.0f%% leave firm contact within T; written to %s",
             len(meta), meta.take_key.nunique(), meta.first_onset.sum(), 100 * meta.leaves_contact.mean(), H.OUT)


if __name__ == "__main__":
    main()
