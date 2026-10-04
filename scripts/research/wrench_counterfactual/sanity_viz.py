#!/usr/bin/env python
"""Sanity figures and numbers for the geometry / patch / wrench pipeline.  python sanity_viz.py
  <ds>/sanity/finger_assignment_<event>.png   hand vertices coloured by MANO part, object vertices (grey),
                                               contact vertices coloured by their assigned finger, patch
                                               centroids with normals (three orthographic views)
  <ds>/sanity/prepost_<event>.png              pre / post frames of persistent-spatial events (>= 10 per
                                               dataset) with patches
  <ds>/sanity/transport_<event>.png            ARCTIC: pre-event patches attached to the top part follow
                                               the articulation to s + 8 / s + 16
  sanity_summary.json                          finger-assignment agreement with the 100-point FPS labels,
                                               normal orientation statistics, patch counts, synthetic grasp
                                               map, positive-control direction (release / onset capacity)
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wc_common as W
import wrench_lp as L
from wrench_analysis import geom_dict, load_geometry, mesh_of

COLORS = {0: "0.55", 1: "tab:red", 2: "tab:blue", 3: "tab:green", 4: "tab:orange", 5: "tab:purple"}
VIEWS = ((0, 1, "x", "y"), (0, 2, "x", "z"), (1, 2, "y", "z"))


def draw_frame(ax_row, g, k, mesh, patches, hand_lab, title):
    verts = mesh.verts if f"tr_{k}_{k}_pos" not in g else None
    ids = g[f"{k}_ids"]; pos = g[f"{k}_pos"]; lab = g[f"{k}_label"]; sel = g[f"{k}_dist"] < W.CONTACT_THR["primary"]
    hand = g[f"hand_{k}"].astype(float)
    rng = np.random.default_rng(0)
    V = mesh.verts if mesh.parts is None else mesh.verts_at(g["arti_pre"] if k == "pre" else g["arti_post"])
    Vs = V[rng.choice(len(V), min(2500, len(V)), replace=False)]
    for ax, (i, j, nx, ny) in zip(ax_row, VIEWS):
        ax.scatter(Vs[:, i] * 100, Vs[:, j] * 100, s=1, color="0.85")
        for p in range(6):
            m = hand_lab == p
            ax.scatter(hand[m, i] * 100, hand[m, j] * 100, s=2, color=COLORS[p], alpha=0.35)
        for p in range(6):
            m = sel & (lab == p)
            if m.any():
                ax.scatter(pos[m, i] * 100, pos[m, j] * 100, s=14, color=COLORS[p], edgecolor="k", lw=0.3, label=W.PARTS[p] if ax is ax_row[0] else None)
        for pc in patches:
            c = pc["centroid"] * 100; n = pc["normal"] * 2.5
            ax.plot([c[i], c[i] + n[i]], [c[j], c[j] + n[j]], color=COLORS[pc["part"]], lw=2)
            ax.scatter([c[i]], [c[j]], s=60, marker="*", color=COLORS[pc["part"]], edgecolor="k", lw=0.5, zorder=5)
        ax.set_aspect("equal"); ax.set_xlabel(f"{nx} [cm]"); ax.set_ylabel(f"{ny} [cm]")
    ax_row[0].set_title(title, fontsize=9, loc="left"); ax_row[0].legend(fontsize=7, loc="best")


def figure_event(ds, ev, kind, out):
    g = load_geometry(ds, ev.event_id); mesh = mesh_of(ds, ev); lab = W.mano_labels()
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    p_pre = W.build_patches(geom_dict(g, "pre"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
    p_post = W.build_patches(geom_dict(g, "post"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
    draw_frame(ax[0], g, "pre", mesh, p_pre, lab, f"PRE frame {g['frames']['pre']}: {len(p_pre)} patches, fingers {sorted({W.PARTS[p['part']] for p in p_pre})}")
    draw_frame(ax[1], g, "post", mesh, p_post, lab, f"POST frame {g['frames']['post']}: {len(p_post)} patches, fingers {sorted({W.PARTS[p['part']] for p in p_post})}")
    fig.suptitle(f"{W.LABEL[ds]} {ev.cls} event {ev.event_id} ({ev.sequence_id}, {ev.group}); ‖C_post − C_pre‖ = {ev.dC:.2f}; stars = patch centroids, bars = patch normals (into the hand)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / f"{kind}_{ev.event_id}.png", dpi=110); plt.close(fig)
    return g, p_pre, p_post


def transport_figure(ds, ev, out):
    g = load_geometry(ds, ev.event_id); mesh = mesh_of(ds, ev)
    keys = [k for k in ("pre", "post", "s8", "s16") if f"tr_pre_{k}_pos" in g]
    fig, ax = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.2))
    ax = np.atleast_1d(ax)
    sel = g["pre_dist"] < W.CONTACT_THR["primary"]
    for a, k in zip(ax, keys):
        arti = g["arti_pre"] if k == "pre" else (g["arti_post"] if k == "post" else None)
        fr = g["frames"][k]
        V = mesh.verts_at(arti) if arti is not None else None
        if V is None:                                                    # articulation at s+K is not stored: use transported positions only
            V = mesh.verts
        rng = np.random.default_rng(0); Vs = V[rng.choice(len(V), 2000, replace=False)]
        a.scatter(Vs[:, 0] * 100, Vs[:, 2] * 100, s=1, color="0.85")
        pos = g[f"tr_pre_{k}_pos"][sel]; lab = g["pre_label"][sel]
        for p in range(6):
            m = lab == p
            if m.any():
                a.scatter(pos[m, 0] * 100, pos[m, 2] * 100, s=12, color=COLORS[p], edgecolor="k", lw=0.3)
        top = mesh.parts == 0 if mesh.parts is not None else None
        if top is not None:
            a.scatter(V[top][::5, 0] * 100, V[top][::5, 2] * 100, s=1, color="tab:cyan", alpha=0.4)
        a.set_aspect("equal"); a.set_title(f"pre patches at frame {fr} ({k})", fontsize=9); a.set_xlabel("x [cm]"); a.set_ylabel("z [cm]")
    fig.suptitle(f"{W.LABEL[ds]} {ev.event_id}: pre-event contact vertices transported through the object trajectory (cyan = articulated top part)", fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"transport_{ev.event_id}.png", dpi=110); plt.close(fig)


def main():
    summary = {}
    ok, rows = L.synthetic_check(verbose=False)
    summary["synthetic_grasp_map"] = dict(passed=bool(ok), cases=[dict(name=n, got=float(a), expected=float(b)) for n, a, b in rows])
    lab = W.mano_labels(); fps = np.load(W.BIMART / "assets/part_fps_hand_index_100.npy").reshape(-1)
    summary["mano_labels"] = dict(vertices_per_part=dict(zip(W.PARTS, np.bincount(lab, minlength=6).tolist())), fps100_per_part=dict(zip(W.PARTS, np.bincount(lab[fps], minlength=6).tolist())))
    for ds in W.DATASETS:
        out = W.ds_out(ds) / "sanity"; out.mkdir(parents=True, exist_ok=True)
        S = W.load_selected(ds)
        rng = np.random.default_rng(1)
        prim = S[S.primary]
        picks = prim.sample(min(12, len(prim)), random_state=1)
        stats = dict(n_primary=int(len(prim)), n_figures=int(len(picks)), flip_frac=[], n_patches_pre=[], n_patches_post=[], patch_verts=[], nearest_label_vs_fps=[])
        for ev in picks.itertuples():
            g, p_pre, p_post = figure_event(ds, ev, "prepost", out)
            for k, ps in (("pre", p_pre), ("post", p_post)):
                sel = g[f"{k}_dist"] < W.CONTACT_THR["primary"]
                if sel.any():
                    stats["flip_frac"].append(float(1 - g[f"{k}_outward"][sel].mean()))
                stats[f"n_patches_{k}"].append(len(ps)); stats["patch_verts"] += [p["n"] for p in ps]
                # the finger label of a contact vertex from the nearest of the 100 FPS points (the earlier studies' hand representation) vs the 778-vertex label
                hand = g[f"hand_{k}"].astype(float); pos = g[f"{k}_pos"][sel]
                if sel.any():
                    from scipy.spatial import cKDTree
                    nn = cKDTree(hand[fps]).query(pos)[1]
                    stats["nearest_label_vs_fps"].append(float((lab[fps][nn] == g[f"{k}_label"][sel]).mean()))
        # finger-assignment figures for a few events of every class
        for cls in ("persistent_spatial", "persistent_mixed", "onset", "release"):
            sub = S[S.cls == cls]
            for ev in sub.sample(min(2, len(sub)), random_state=2).itertuples():
                figure_event(ds, ev, "finger_assignment", out)
        if ds == "arctic":
            for ev in prim.sample(min(4, len(prim)), random_state=3).itertuples():
                transport_figure(ds, ev, out)
        gs = json.load(open(out / "geometry_sanity.json"))
        summary[ds] = dict(n_events=gs["n_events"], n_errors=gs["n_errors"], min_d_agreement_max=gs["min_d_abs_diff_max"], hand_outward_mean=gs["frac_hand_outward_mean"],
                           contact_vertices_median_pre_post=[gs["n_contact_pre_median"], gs["n_contact_post_median"]], label_hist=gs["label_hist"],
                           persistent_spatial_figures=int(len(picks)), normal_flipped_toward_hand_frac_mean=float(np.mean(stats["flip_frac"])) if stats["flip_frac"] else None,
                           patches_per_frame_mean=float(np.mean(stats["n_patches_pre"] + stats["n_patches_post"])), vertices_per_patch_median=float(np.median(stats["patch_verts"])) if stats["patch_verts"] else None,
                           finger_label_agreement_778_vs_fps100=float(np.mean(stats["nearest_label_vs_fps"])) if stats["nearest_label_vs_fps"] else None)
    W.write_json(W.OUT / "sanity_summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "synthetic_grasp_map"}, indent=1, default=str)[:3000])
    print("synthetic:", summary["synthetic_grasp_map"]["passed"])


if __name__ == "__main__":
    main()
