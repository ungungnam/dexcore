#!/usr/bin/env python
"""Step 1: hand states in the object frame for EVERY sequence of the fixed split, firm-contact
counts, the temporal-event labels on every split, and the coordinate / alignment sanity checks.
    python build_hand_cache.py --dataset taco
Writes OUT/<ds>/cache/{hand_cache.npz, events_all.csv, frames_all.csv, hand_representation.json}
and OUT/<ds>/sanity/{hand_alignment.json, hand_alignment.png}.

hand_cache.npz: hand (N, 80, 100, 3) f32 metres in the contacted object's frame for t in [-8, 72)
(clamped at the take boundaries), n_hard (N, 64) int, min_d (N, 64) m, example (N,).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import hp_common as P
from hp_common import K

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hand_cache")
SANITY_EVERY = 10          # every 10th sequence is checked against the object mesh


def rot_from_6d(x, layout):
    """(T,6) -> (T,3,3). layout 'cols': the stored order is the first two COLUMNS of R flattened
    row-major (R00,R01,R10,R11,R20,R21) as build_object_states.py writes it; 'rows': the first three
    and last three entries taken as the two axes (the reading used by tce_common.py)."""
    a, b = (x[:, [0, 2, 4]], x[:, [1, 3, 5]]) if layout == "cols" else (x[:, :3], x[:, 3:])
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b - (a * b).sum(1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], 2)


def to_frame(x, R, p):
    """x (T,K,3) in the parent frame -> R^T (x - p) in the child frame given by (R, p) per frame."""
    return np.einsum("tji,tkj->tki", R, x - p[:, None])


def taco(D):
    meta, O, names = D["meta"], D["O"], D["names"]
    seq_root = Path("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale")
    files = pd.read_csv(seq_root / "sequence_index.csv").set_index("sequence_id")["file"]
    wi = pd.read_csv("/result/uhnam/dexcore/taco/40_representation_study/window_index.csv").drop_duplicates("sequence_id").set_index("sequence_id")
    mesh_dict = np.load(seq_root / "assets/taco_mesh_dict.npy", allow_pickle=True).item()
    ip = [names.index(k) for k in ("tool_pos_x", "tool_pos_y", "tool_pos_z")]
    ir = [names.index(f"tool_rot6d_{i}") for i in range(6)]
    N = len(meta)
    H = np.zeros((N, P.T + 2 * P.PAD, P.N_HAND, 3), np.float32); nh = np.zeros((N, P.T), np.int32); mind = np.zeros((N, P.T), np.float32)
    alt_speed = np.full((N, P.T - 1), np.nan, np.float32)
    checks = []
    for f, rows in meta.groupby(files.loc[meta.sequence_id].values).groups.items():
        with np.load(seq_root / "sequences" / f) as z:
            kp_all = z["kp"].reshape(len(z["kp"]), 2, P.N_HAND, 3); d_both = {"L": z["contact_left"], "R": z["contact_right"]}
        n = len(kp_all)
        for i in rows:
            r = meta.iloc[i]
            fr = np.clip(int(r.t0) + np.arange(-P.PAD, P.T + P.PAD), 0, n - 1)
            h = kp_all[fr, 0 if r.hand == "L" else 1].copy()                     # target frame, metres
            h_alt = None
            if r.role == "tool":
                o = O[i]                                                              # (80, 30), same clamped frames
                h_alt = to_frame(h, rot_from_6d(o[:, ir], "rows"), o[:, ip])
                h = to_frame(h, rot_from_6d(o[:, ir], "cols"), o[:, ip])
                alt_speed[i] = np.linalg.norm(np.diff(h_alt[P.PAD:P.PAD + P.T], axis=0), axis=2).mean(1)
            H[i] = h
            d = d_both[r.hand][int(r.t0):int(r.t0) + P.T]; n_tool = int(wi.loc[r.sequence_id].n_tool)
            dd = d[:, :n_tool] if r.role == "tool" else d[:, n_tool:]
            nh[i] = (dd < P.HARD_MM).sum(1); mind[i] = dd.min(1)
            if i % SANITY_EVERY == 0:
                tree = cKDTree(mesh_dict[r.mesh_id]["verts_original"])
                q = tree.query(h[P.PAD:P.PAD + P.T].reshape(-1, 3))[0].reshape(P.T, P.N_HAND).min(1)
                q_alt = tree.query(h_alt[P.PAD:P.PAD + P.T].reshape(-1, 3))[0].reshape(P.T, P.N_HAND).min(1) if h_alt is not None else np.full(P.T, np.nan)
                checks.append(pd.DataFrame(dict(example=i, role=r.role, t=np.arange(P.T), min_d_contact=mind[i], min_d_hand100=q, min_d_hand100_alt=q_alt)))
    return H, nh, mind, pd.concat(checks, ignore_index=True), alt_speed, dict(
        source="gen3 sequences kp (100 MANO surface points per hand, BimArt part_fps_hand_index_100) in the TARGET frame",
        tool_role="re-expressed in the tool frame: R_rel^T (x - p_rel) with tool_pos / tool_rot6d (first two columns of R_rel) from the cached object states",
        mesh_check="every %dth sequence: min distance of the 100 hand points to the contacted object's own mesh vertices (verts_original) vs the cached per-vertex contact minimum" % SANITY_EVERY)


def arctic(D):
    meta = D["meta"]
    hand_root = P.REPO / "third_party/BimArt/data/arctic_processed_data"
    probe = Path("/result/uhnam/dexcore/arctic/20_bimart_contact_probe")
    files = pd.read_csv(probe / "sequence_index.csv").set_index("sequence_id")["file"]
    N = len(meta)
    H = np.zeros((N, P.T + 2 * P.PAD, P.N_HAND, 3), np.float32); nh = np.zeros((N, P.T), np.int32); mind = np.zeros((N, P.T), np.float32)
    checks = []
    for sid, rows in meta.groupby("sequence_id").groups.items():
        stem, subj = sid.split("/"); cat = meta.iloc[rows[0]].category
        hf = np.load(hand_root / cat / subj / f"{stem}_processed_hand_features.npy", allow_pickle=True).item()
        verts = {"L": hf["left_hand_sampled_verts_cano"].astype(np.float32), "R": hf["right_hand_sampled_verts_cano"].astype(np.float32)}
        with np.load(probe / "sequences" / files[sid]) as z:
            d_both = {"L": z["contact_left"], "R": z["contact_right"]}
        n = len(verts["L"]); assert len(d_both["L"]) == n, (sid, n, len(d_both["L"]))
        obj = None
        if any(i % SANITY_EVERY == 0 for i in rows):
            obj = np.load(hand_root / cat / subj / f"{stem}_processed_obj_features.npy", allow_pickle=True).item()["obj_cano_verts_dense"]
        for i in rows:
            r = meta.iloc[i]
            fr = np.clip(int(r.t0) + np.arange(-P.PAD, P.T + P.PAD), 0, n - 1)
            H[i] = verts[r.hand][fr]
            d = d_both[r.hand][int(r.t0):int(r.t0) + P.T]
            nh[i] = (d < P.HARD_MM).sum(1); mind[i] = d.min(1)
            if i % SANITY_EVERY == 0:
                q = np.array([cKDTree(obj[int(r.t0) + t]).query(H[i, P.PAD + t])[0].min() for t in range(P.T)])
                checks.append(pd.DataFrame(dict(example=i, role="obj", t=np.arange(P.T), min_d_contact=mind[i], min_d_hand100=q, min_d_hand100_alt=np.nan)))
    return H, nh, mind, pd.concat(checks, ignore_index=True), None, dict(
        source="BimArt *_processed_hand_features.npy {left,right}_hand_sampled_verts_cano (100 MANO surface points) in the object's canonical rigid frame (articulation left in)",
        mesh_check="every %dth sequence: min distance of the 100 hand points to the object's per-frame canonical vertices (obj_cano_verts_dense) vs the cached per-vertex contact minimum" % SANITY_EVERY)


def alignment_stats(ch, key="min_d_hand100"):
    ok = ch[key].notna()
    a, b = ch.min_d_contact.values[ok], ch[key].values[ok]
    hard = a < P.HARD_MM
    return dict(n_frames=int(ok.sum()), n_sequences=int(ch.example[ok].nunique()), pearson=float(np.corrcoef(a, b)[0, 1]),
                median_abs_diff_mm=float(np.median(np.abs(a - b)) * 1000), p95_abs_diff_mm=float(np.percentile(np.abs(a - b), 95) * 1000),
                frac_hand100_below_contact=float((b < a - 1e-6).mean()),
                hard_frames=int(hard.sum()), hard_frac_hand100_within_2cm=float((b[hard] < 0.02).mean()) if hard.any() else np.nan,
                hard_median_hand100_mm=float(np.median(b[hard]) * 1000) if hard.any() else np.nan,
                nonhard_median_hand100_mm=float(np.median(b[~hard]) * 1000) if (~hard).any() else np.nan)


def sanity_figure(ds, ch, H, mind, meta, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for role, col in (("target", "tab:blue"), ("tool", "tab:orange"), ("obj", "tab:green")):
        c = ch[ch.role == role]
        if len(c):
            ax[0].scatter(c.min_d_contact * 100, c.min_d_hand100 * 100, s=3, alpha=0.3, color=col, label=f"{role} (n={len(c)})")
    if ch.min_d_hand100_alt.notna().any():
        c = ch[ch.min_d_hand100_alt.notna()]
        ax[0].scatter(c.min_d_contact * 100, c.min_d_hand100_alt * 100, s=3, alpha=0.2, color="tab:red", label="tool, wrong 6D layout")
    lim = max(ch.min_d_contact.max(), np.nanmax(ch.min_d_hand100)) * 100
    ax[0].plot([0, lim], [0, lim], "k--", lw=0.8); ax[0].set_xlabel("min hand distance from cached contact [cm]"); ax[0].set_ylabel("min distance of the 100 hand points to the mesh [cm]")
    ax[0].set_title(f"{P.LABEL[ds]}: hand / object frame check"); ax[0].legend(fontsize=8)
    # two example frames: hard-contact frame with the smallest hand100 distance, and a non-contact frame
    for k, (ttl, pick) in enumerate((("hard-contact frame", ch[ch.min_d_contact < P.HARD_MM].sort_values("min_d_hand100").iloc[0] if (ch.min_d_contact < P.HARD_MM).any() else None),
                                     ("no-contact frame", ch[ch.min_d_contact > 0.05].sort_values("min_d_hand100").iloc[-1] if (ch.min_d_contact > 0.05).any() else None))):
        if pick is None:
            continue
        i, t = int(pick.example), int(pick.t)
        h = H[i, P.PAD + t]
        if ds == "taco":
            md = np.load("/result/uhnam/dexcore/taco/30_bimart_gen3_scene_scale/assets/taco_mesh_dict.npy", allow_pickle=True).item()
            V = md[meta.iloc[i].mesh_id]["verts_original"]
        else:
            r = meta.iloc[i]; stem, subj = r.sequence_id.split("/")
            V = np.load(P.REPO / "third_party/BimArt/data/arctic_processed_data" / r.category / subj / f"{stem}_processed_obj_features.npy", allow_pickle=True).item()["obj_cano_verts_dense"][int(r.t0) + t]
        rng = np.random.default_rng(0); V = V[rng.choice(len(V), min(2000, len(V)), replace=False)]
        ax[k + 1].scatter(V[:, 0] * 100, V[:, 1] * 100, s=2, color="0.6", label="object vertices"); ax[k + 1].scatter(h[:, 0] * 100, h[:, 1] * 100, s=8, color="tab:red", label="hand points")
        ax[k + 1].set_aspect("equal"); ax[k + 1].set_xlabel("x [cm]"); ax[k + 1].set_ylabel("y [cm]")
        ax[k + 1].set_title(f"{ttl}: ex {i} t={t} ({meta.iloc[i].group}); contact min {pick.min_d_contact * 100:.1f} cm, hand100 min {pick.min_d_hand100 * 100:.1f} cm", fontsize=8); ax[k + 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=P.DATASETS)
    a = ap.parse_args(); ds = a.dataset; t_start = time.time()
    P.assert_causal()
    out = P.ds_out(ds); (out / "cache").mkdir(parents=True, exist_ok=True); (out / "sanity").mkdir(exist_ok=True); (out / "figures").mkdir(exist_ok=True)
    D = P.load_raw(ds); meta, man, thr = D["meta"], D["man"], D["thr"]
    H, nh, mind, checks, alt_speed, rep = (taco if ds == "taco" else arctic)(D)
    log.info("%s: hand states %s, %d sanity frames (%.0fs)", ds, H.shape, len(checks), time.time() - t_start)
    np.savez(P.hand_cache_path(ds), hand=H, n_hard=nh, min_d=mind, example=np.arange(len(meta)))
    # ---- alignment checks
    san = {"all": alignment_stats(checks)}
    for role in checks.role.unique():
        san[role] = alignment_stats(checks[checks.role == role])
    if checks.min_d_hand100_alt.notna().any():
        san["tool_wrong_6d_layout"] = alignment_stats(checks[checks.role == "tool"], "min_d_hand100_alt")
    checks.to_csv(out / "sanity" / "hand_alignment_frames.csv", index=False)
    sanity_figure(ds, checks, H, mind, meta, out / "sanity" / "hand_alignment.png")
    # ---- events on every split (same rules as the earlier test-only analysis; must reproduce it)
    F, E, q90 = P.label_events(ds, D["C"], nh, meta, man, thr)
    prev_ev = pd.read_csv(K.OUT / ds / "events.csv")
    mine_te = E[E.split_set == "test"]
    key = lambda e: set(zip(e.example.astype(int), e.start_frame.astype(int), e.end_frame.astype(int), e.event_category.astype(str)))
    same = key(mine_te) == key(prev_ev)
    prev_q90 = float(json.load(open(K.OUT / ds / "sanity.json"))["q90"])
    san["event_labels"] = dict(q90=q90, q90_previous=prev_q90, n_events_all=int(len(E)), n_events_test=int(len(mine_te)), n_events_test_previous=int(len(prev_ev)),
                               test_events_identical_to_previous=bool(same), n_positive_events=int(E.positive_event.sum()),
                               positive_by_split=E.groupby("split_set").positive_event.sum().to_dict(), events_by_split=E.groupby("split_set").size().to_dict(),
                               positive_rule=P.POSITIVE_RULE)
    assert abs(q90 - prev_q90) < 1e-6 and same, san["event_labels"]
    # ---- hand speed + kinematics per frame, Spearman with d_t on the test frames
    hs = np.linalg.norm(np.diff(H[:, P.PAD:P.PAD + P.T], axis=1), axis=3).mean(2)             # (N, 63) m/frame
    F["hand_rel_speed"] = hs.reshape(-1)
    kin = K.kinematics(ds, D, np.arange(len(meta)))
    for k in K.KIN_AVAILABLE[ds]:
        if k != "hand_rel_speed":
            F[k] = kin[k][:, :-1].reshape(-1)
    te = (F.split_set == "test").values
    rho = K.spearman_cluster(F.hand_rel_speed.values[te], F.d.values[te], F.take_key.values[te], n_boot=200)
    san["spearman_d_vs_hand_speed_test"] = dict(rho=rho[0], lo=rho[1], hi=rho[2], n=rho[3])
    # agreement with the earlier test-frame table (firm state, spikes, hand speed)
    prev = pd.read_csv(K.OUT / ds / "frames_test.csv", usecols=["example", "t", "firm_t", "hand_rel_speed", "is_spike"]).rename(
        columns={"firm_t": "firm_t_prev", "hand_rel_speed": "hand_rel_speed_prev", "is_spike": "spike_prev"})
    mf = F[te].merge(prev, on=["example", "t"])
    ok = mf.hand_rel_speed_prev.notna()
    san["agreement_with_previous_test_table"] = dict(n_frames=int(len(mf)), firm_state=float((mf.firm_t == mf.firm_t_prev).mean()), spike=float((mf.spike == mf.spike_prev).mean()),
                                                     hand_speed_pearson=float(np.corrcoef(mf.hand_rel_speed[ok], mf.hand_rel_speed_prev[ok])[0, 1]),
                                                     hand_speed_max_abs_diff=float(np.abs(mf.hand_rel_speed[ok] - mf.hand_rel_speed_prev[ok]).max()))
    if alt_speed is not None:
        F["hand_rel_speed_wrong6d"] = np.where(np.isnan(alt_speed.reshape(-1)), hs.reshape(-1), alt_speed.reshape(-1))
        rho_alt = K.spearman_cluster(F.hand_rel_speed_wrong6d.values[te], F.d.values[te], F.take_key.values[te], n_boot=200)
        san["spearman_d_vs_hand_speed_test_with_wrong_6d_layout"] = dict(rho=rho_alt[0], lo=rho_alt[1], hi=rho_alt[2])
    F.to_csv(out / "cache" / "frames_all.csv", index=False)
    E.to_csv(out / "cache" / "events_all.csv", index=False)
    # ---- class balance of the event-prediction labels
    bal = {}
    for h in P.EVENT_HORIZONS:
        y, valid = P.event_labels(F, E, h)
        for part in ("train", "val", "test"):
            idx = man[part]
            bal[f"h{h}_{part}"] = dict(n_valid=int(valid[idx].sum()), n_pos=int(y[idx][valid[idx]].sum()), prevalence=float(y[idx][valid[idx]].mean()))
    san["event_label_balance"] = bal
    rep.update(dict(dataset=ds, n_sequences=int(len(meta)), n_points=P.N_HAND, dims_per_frame=P.HAND_DIM, frames_stored="t in [-8, 72), clamped at the take boundaries",
                    units="metres", conditions={c: dict(P.CONDITIONS[c], offsets=P.window_offsets(c), input_dim=P.hand_dim(c), latest_frame_seen=P.latest_hand_frame(c)) for c in P.CONDITIONS},
                    hand_extent_m=dict(mean_abs=float(np.abs(H).mean()), p99_abs=float(np.percentile(np.abs(H), 99)))))
    P.write_json(out / "cache" / "hand_representation.json", rep)
    P.write_json(out / "sanity" / "hand_alignment.json", san)
    log.info("%s: alignment %s; events %d (test identical: %s); rho(d, hand speed) test %.3f; balance %s; done in %.0fs", ds,
             {k: round(v["pearson"], 3) for k, v in san.items() if isinstance(v, dict) and "pearson" in v}, len(E), same, rho[0], bal, time.time() - t_start)


if __name__ == "__main__":
    main()
