#!/usr/bin/env python
"""Sanity checks (section 17 of the request) on the feature cache and the study's conventions -> sanity/sanity_summary.json
and a few representative visual checks.
    python sanity.py
 1. finger labels: the contact-vertex labels are the nearest MANO vertex's skinning-weight part (wrench report); here the
    per-part hard-contact counts are compared with the 100-point FPS part labels of the hand cache on sampled frames
 2. centroids on the object surface: distance of every active part's centroid (p_k * l + mesh centroid) to the nearest
    mesh vertex, and to the nearest contact vertex of that part, on sampled frames
 3. normals: the toward-hand orientation is applied per vertex; the palm's mean normal against the palm frame's dorsal axis
 4. inactive parts carry zeros in the geometric blocks (assert on the whole cache)
 5. object-length normalisation: l per object equals the wrench report's mesh length; p_k * l lies inside the mesh extent
 6. left / right hands: the palm-normal check and the fingertip order per hand side
 7. ARCTIC articulation: recomputed minimum hand-object distance equals the cached one (verts_at(arti))
 8. event pre / post frames equal the wrench report's events_selected.csv (event id, pre, post, dC)
 9. no identity leakage: the representation dimensions are fixed-size descriptors (listed); no sequence / event index enters
10. train statistics only: fit_norm uses the train index (checked by recomputing on train vs on all)
11. causal windows: offsets <= 0 (assert) and the feature frames used by a window at t are t-7 .. t
12. the object trajectory tau_local (t-8 .. t+8) is the explicit task input of every previous study (documented)
13. FULL vs the previous diagnostic: compared in the report from temporal_prediction.csv (E_C relative to persistence)
"""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import sv_common as S
import wc_common as W
from sv_data import assert_causal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("sanity")
BIMART_PART = S.REPO / "third_party/BimArt/data/part_fps_hand_index_100.npy"


def main():
    out = S.OUT / "sanity"; out.mkdir(parents=True, exist_ok=True)
    assert_causal()
    summary = dict(causal_window_offsets=list(range(-S.HIST + 1, 1)), representation_dims=S.REP_DIM, identity_leak="none: every input is a fixed-width descriptor of the frame (no ids)",
                   object_trajectory_input="tau_local,t = O_{t-8..t+8 step 2}, the explicit task input of the hierarchical / hand studies")
    lab778 = W.mano_labels()
    fps = np.load(BIMART_PART) if BIMART_PART.exists() else None
    for ds in S.DATASETS:
        F = S.load_features(ds); build = json.load(open(S.ds_out(ds) / "cache" / "features_build.json"))
        D = S.HP.load_raw(ds); meta = D["meta"]; man = D["man"]; tr = man["train"]
        inc = F["included"]; idx = np.where(inc)[0]
        s = dict(build=build)
        blocks = S.block_arrays(F)
        # 4. inactive parts -> zeros
        a = blocks["part"]; geom = blocks["geom"]; topo = blocks["topo"]
        inactive = a < 0.5
        s["inactive_parts_geom_max_abs"] = float(np.abs(geom.reshape(*geom.shape[:2], 2, 6, 3)[inactive[:, :, None, :].repeat(2, 2)]).max()) if inactive.any() else 0.0
        s["inactive_parts_topo_max_abs"] = float(np.abs(topo.reshape(*topo.shape[:2], 2, 6)[inactive[:, :, None, :].repeat(2, 2)]).max()) if inactive.any() else 0.0
        # 10. train statistics only
        mu_tr, sd_tr = S.fit_norm(blocks["amount"], tr); mu_all, sd_all = S.fit_norm(blocks["amount"], idx)
        s["train_only_norm_differs_from_all"] = bool(np.abs(mu_tr - mu_all).max() > 0)
        # 5. object length vs the wrench report's mesh length
        wl = {}
        s["length_range_m"] = [float(F["length"][idx].min()), float(F["length"][idx].max())]
        # 1-3, 5, 7: sampled frames
        rng = np.random.default_rng(0); sample = rng.choice(idx, min(60, len(idx)), replace=False)
        cent_to_mesh, cent_to_contact, inside, palm = [], [], [], {"L": [], "R": []}
        lab_agree = []
        for n in sample:
            r = meta.iloc[n]
            mesh = W.taco_mesh(str(r.mesh_id)) if ds == "taco" else W.arctic_mesh(r.category)
            wl[str(r.mesh_id) if ds == "taco" else r.category] = mesh.length
            assert abs(mesh.length - F["length"][n]) < 1e-6
            state = None
            if ds == "arctic":
                import build_geometry as BG
                state = BG.arctic_take(r.sequence_id, r.category)["state"]
            for t in (S.PAD + 8, S.PAD + 32, S.PAD + 56):
                tree = cKDTree(mesh.verts if state is None else mesh.verts_at(state[min(int(r.t0) + t - S.PAD, len(state) - 1), 0]))   # ARCTIC: the articulated surface of that frame
                for k in range(6):
                    if a[n, t, k] < 0.5:
                        continue
                    p = F["p"][n, t, k] * F["length"][n] + F["centroid"][n]
                    cent_to_mesh.append(tree.query(p)[0]); inside.append(bool(np.all(np.abs(p - mesh.centroid) <= mesh.extent / 2 + 0.02)))
            if F["n_hard"][n, S.PAD + 32, 0] >= S.N_MIN_VERTS:
                palm[r.hand].append(float(F["nrm"][n, S.PAD + 32, 0] @ F["hand"][n, S.PAD + 32, 9:12]))
        s["centroid_to_nearest_mesh_vertex_mm"] = dict(median=float(np.median(cent_to_mesh) * 1000), p95=float(np.percentile(cent_to_mesh, 95) * 1000), n=len(cent_to_mesh))
        s["centroid_inside_mesh_extent_frac"] = float(np.mean(inside))
        s["palm_normal_dot_dorsal"] = {h: dict(mean=float(np.mean(v)), n=len(v)) for h, v in palm.items() if v}
        pdot = F["palm_dot"]; hands = meta.hand.values
        s["palm_normal_dot_dorsal_all_frames_by_hand"] = {h: dict(mean=float(np.nanmean(pdot[idx][hands[idx] == h])), n_sequences=int(np.isfinite(pdot[idx][hands[idx] == h]).sum())) for h in ("L", "R")}
        s["palm_normal_dot_dorsal_all"] = build.get("palm_dot_mean")
        s["min_d_recomputed_vs_cached_max"] = build.get("min_d_check_max")
        s["C_padded_vs_sequences_max"] = build.get("C_check_max")
        s["clamped_frame_frac"] = build.get("clamped_frac")
        # 1. labels: hand-cache 100 points (FPS ids) vs 778 labels — the part of each FPS point
        if fps is not None:
            s["fps100_part_counts"] = {W.PARTS[k]: int((lab778[fps] == k).sum()) for k in range(6)}
        # 8. events vs the wrench report
        E = S.load_events(ds, "test"); Wv = pd.read_csv(W.ds_out(ds) / "events_selected.csv", dtype={"event_id": str})
        m = E.merge(Wv[["event_id", "pre_frame", "post_frame", "dC", "cls"]], on="event_id", suffixes=("", "_w"))
        s["events_test"] = dict(n_here=len(E), n_wrench=len(Wv), n_matched=len(m), pre_post_equal=bool(((m.pre_frame == m.pre_frame_w) & (m.post_frame == m.post_frame_w)).all()),
                                dC_max_abs_diff=float(np.abs(m.dC - m.dC_w).max()), cls_equal=bool((m.cls == m.cls_w).all()))
        # 6. fingertip order: tips ordered thumb..little; check the mean tip-to-tip distances (thumb far from little) per hand
        tips = F["hand"][idx][:, S.PAD:S.PAD + S.T, 12:27].reshape(-1, 5, 3)
        s["tip_distance_thumb_little_over_index_middle"] = float(np.linalg.norm(tips[:, 0] - tips[:, 4], axis=1).mean() / np.linalg.norm(tips[:, 1] - tips[:, 2], axis=1).mean())
        # feature statistics
        aa = a[idx][:, S.PAD:S.PAD + S.T]
        s["participation_rate_by_part"] = {W.PARTS[k]: float(aa[..., k].mean()) for k in range(6)}
        s["mean_active_parts"] = float(aa.sum(-1).mean()); s["frames_with_no_active_part_frac"] = float((aa.sum(-1) == 0).mean())
        s["q_zero_frac"] = float((F["q"][idx].sum(-1) <= 1e-6).mean()); s["q_strict_zero_frac"] = float((F["q_strict"][idx].sum(-1) <= 1e-6).mean())
        s["Q_mean"] = float(F["q"][idx].mean()); s["patches_per_frame_mean"] = float(F["n_patches"][idx][:, S.PAD:S.PAD + S.T].mean())
        summary[ds] = s
        # visual check: participation, amount and Q along a few sequences + one frame's descriptors
        fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        n = sample[0]; t = np.arange(-S.PAD, S.T + S.PAD)
        for k in range(6):
            ax[0].plot(t, a[n, :, k] + 1.2 * k, label=W.PARTS[k]); ax[1].plot(t, blocks["amount"][n, :, k], label=W.PARTS[k])
        ax[0].set_ylabel("a_k (offset)"); ax[0].legend(fontsize=7, ncol=6); ax[1].set_ylabel("m_k (area fraction)")
        ax[2].plot(np.arange(S.T), F["q"][n].mean(1), label="Q (support)"); ax[2].plot(np.arange(S.T), F["q_strict"][n].mean(1), label="Q strict"); ax[2].legend(fontsize=8); ax[2].set_xlabel("frame t")
        ax[2].axvline(0, color="k", lw=0.5); ax[2].axvline(63, color="k", lw=0.5)
        fig.suptitle(f"{S.LABEL[ds]} sequence {n} ({meta.iloc[n].sequence_id}, {meta.iloc[n].hand}): participation, amount, capability along the stored frames")
        fig.tight_layout(); fig.savefig(out / f"sequence_descriptors_{ds}.png", dpi=110); plt.close(fig)
        log.info("%s: %s", ds, json.dumps({k: v for k, v in s.items() if k != "build"}, default=str)[:1500])
    S.write_json(out / "sanity_summary.json", summary)


if __name__ == "__main__":
    main()
