#!/usr/bin/env python
"""Step 4: figures A-F.   python figures.py
  A  persistent spatial: contact-map change vs relative wrench-capability change (and retention), per dataset
  B  distribution of pre->post retention, per dataset and class
  C  comparison by event class (persistent spatial / mixed / onset / release / transient): relative change,
     retention, cosine, with take-bootstrap intervals
  D  manipulation-conditioned directional capacity pre (transported) vs post along the future-motion
     directions (force / torque / support), K = 8 and 16
  E  sensitivity to the friction coefficient (and patch / budget settings)
  F  representative events: large contact change with unchanged capability, and clear post improvement
     (pre/post hand + object + patches + capability profiles), drawn from the extremes of the primary class
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import wc_common as W
from sanity_viz import COLORS, draw_frame
from wrench_analysis import geom_dict, load_geometry, mesh_of

FIG = W.OUT / "figures"
CLS = ["persistent_spatial", "persistent_mixed", "onset", "release", "transient"]
CLS_SHORT = {"persistent_spatial": "persistent\nspatial", "persistent_mixed": "persistent\nmixed", "onset": "onset /\nregrasp", "release": "release", "transient": "transient\n(diagnostic)", "amount": "amount"}
CCOL = {"persistent_spatial": "tab:red", "persistent_mixed": "tab:orange", "onset": "tab:green", "release": "tab:blue", "transient": "0.6", "amount": "tab:brown"}


def events():
    return pd.read_csv(W.OUT / "events.csv", dtype={"event_id": str, "take_key": str, "sequence_id": str, "mesh_id": str, "subject": str})


def fig_a():
    E = events(); thr = json.load(open(W.OUT / "thresholds.json"))
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for j, ds in enumerate(W.DATASETS):
        d = E[(E.dataset == ds) & (E.class_report == "persistent_spatial")]
        for i, (m, lab) in enumerate((("rel_change_Q", "relative change of mean capability  (Q_post − Q_pre) / Q_pre"), ("R_pre_to_post", "pre→post retention  mean_u min(h_pre / h_post, 1)"))):
            a = ax[i, j]
            a.scatter(d.dC, d[m], s=18, color="tab:red", alpha=0.6, edgecolor="k", lw=0.3)
            if ds in thr:
                a.axvline(thr[ds]["dC_q50"], color="0.4", ls="--", lw=0.8, label=f"train median ‖ΔC‖ = {thr[ds]['dC_q50']:.2f}")
                if m == "rel_change_Q":
                    a.axhspan(-thr[ds]["abs_rel_change_Q_q50"], thr[ds]["abs_rel_change_Q_q50"], color="tab:green", alpha=0.08, label=f"|rel. change| ≤ train median ({thr[ds]['abs_rel_change_Q_q50']:.2f})")
                    a.axhline(0, color="k", lw=0.8)
            a.set_xlabel("contact-map change ‖C_post − C_pre‖₂"); a.set_ylabel(lab, fontsize=9)
            a.set_title(f"{W.LABEL[ds]}: persistent spatial events (n = {len(d)})", fontsize=10); a.legend(fontsize=8)
            if m == "rel_change_Q":
                a.set_ylim(-1.05, max(2.0, np.nanpercentile(d[m], 97) * 1.1))
    fig.tight_layout(); fig.savefig(FIG / "figA_contact_change_vs_wrench_change.png", dpi=140); plt.close(fig)


def fig_b():
    E = events()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for j, ds in enumerate(W.DATASETS):
        d = E[E.dataset == ds]
        bins = np.linspace(0, 1, 21)
        for cls in ("persistent_spatial", "persistent_mixed", "onset", "release"):
            v = d[d.class_report == cls].R_pre_to_post.dropna()
            if len(v):
                ax[j].hist(v, bins=bins, histtype="step", lw=2, color=CCOL[cls], label=f"{cls.replace('_', ' ')} (n={len(v)}, median {v.median():.2f})", density=True)
        ax[j].set_xlabel("pre→post retention R (1 = the pre grasp already has the post grasp's capability)"); ax[j].set_ylabel("density"); ax[j].set_title(W.LABEL[ds]); ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "figB_retention_distribution.png", dpi=140); plt.close(fig)


def fig_c():
    S = pd.read_csv(W.OUT / "summary_by_event_type.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for k, (m, lab) in enumerate((("rel_change_Q_median", "median relative change of Q (post vs pre)"), ("R_pre_to_post_median", "median pre→post retention"), ("cosine_median", "median profile cosine (pre, post)"))):
        a = ax[k]
        for j, ds in enumerate(W.DATASETS):
            d = S[S.dataset == ds].set_index("class_report").reindex(CLS)
            x = np.arange(len(CLS)) + (j - 0.5) * 0.35
            a.bar(x, d[m], 0.35, color=[CCOL[c] for c in CLS], alpha=0.9 if j == 0 else 0.5, edgecolor="k", lw=0.5, label=W.LABEL[ds],
                  yerr=[d[m] - d[f"{m}_lo"], d[f"{m}_hi"] - d[m]], capsize=2)
            for xi, c in zip(x, CLS):
                if not np.isnan(d.loc[c, "n_events"]):
                    a.text(xi, a.get_ylim()[0], f"n={int(d.loc[c, 'n_events'])}", ha="center", va="bottom", fontsize=6, rotation=90)
        a.set_xticks(np.arange(len(CLS))); a.set_xticklabels([CLS_SHORT[c] for c in CLS], fontsize=8); a.set_ylabel(lab); a.axhline(0 if k == 0 else 1, color="k", lw=0.8)
        a.set_title("darker = TACO, lighter = ARCTIC (take-bootstrap 95 % CI)", fontsize=9); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "figC_by_event_class.png", dpi=140); plt.close(fig)


def fig_d():
    E = events()
    keys = [("force_K8", "force along Δv (K=8)"), ("torque_K8", "torque along Δω (K=8)"), ("force_K16", "force along Δv (K=16)"), ("torque_K16", "torque along Δω (K=16)"), ("support", "support (against gravity)")]
    fig, ax = plt.subplots(2, len(keys), figsize=(4 * len(keys), 8))
    for i, ds in enumerate(W.DATASETS):
        d = E[(E.dataset == ds) & (E.class_report == "persistent_spatial")]
        for j, (k, lab) in enumerate(keys):
            a = ax[i, j]
            if f"B_{k}_pre" not in d:
                continue
            x, y = d[f"B_{k}_pre"], d[f"B_{k}_post"]
            ok = x.notna() & y.notna()
            a.scatter(x[ok], y[ok], s=16, color="tab:red", alpha=0.6, edgecolor="k", lw=0.3)
            m = max(float(np.nanmax(x)), float(np.nanmax(y)), 1e-3); a.plot([0, m], [0, m], "k--", lw=0.8)
            a.set_xlabel("pre grasp (transported to the post frame)"); a.set_ylabel("post grasp"); a.set_title(f"{W.LABEL[ds]}: {lab}\nmedian post/pre {np.nanmedian((y[ok] + 1e-6) / (x[ok] + 1e-6)):.2f}, post > pre in {np.mean(y[ok] > x[ok] + 1e-6) * 100:.0f} %", fontsize=9)
    fig.suptitle("Manipulation-conditioned directional capacity (support-function h, unit finger budgets) — persistent spatial events", fontsize=11)
    fig.tight_layout(); fig.savefig(FIG / "figD_manipulation_directions.png", dpi=140); plt.close(fig)


def fig_e():
    Smu = pd.read_csv(W.OUT / "sensitivity_mu.csv"); Sp = pd.read_csv(W.OUT / "sensitivity_patch.csv"); Sb = pd.read_csv(W.OUT / "sensitivity_budget.csv")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for j, ds in enumerate(W.DATASETS):
        d = Smu[(Smu.dataset == ds) & (Smu.class_report == "persistent_spatial")].sort_values("mu")
        ax[0].errorbar(d.mu + (j - 0.5) * 0.01, d.R_pre_to_post_median, yerr=[d.R_pre_to_post_median - d.R_pre_to_post_median_lo, d.R_pre_to_post_median_hi - d.R_pre_to_post_median], marker="o", capsize=3, label=f"{W.LABEL[ds]} retention")
        ax[0].errorbar(d.mu + (j - 0.5) * 0.01, d.rel_change_Q_median + 1, yerr=[d.rel_change_Q_median - d.rel_change_Q_median_lo, d.rel_change_Q_median_hi - d.rel_change_Q_median], marker="s", ls="--", capsize=3, label=f"{W.LABEL[ds]} Q_post/Q_pre (1 + median rel. change)")
        for a, S_, name in ((ax[1], Sp, "patch"), (ax[2], Sb, "budget")):
            d = S_[(S_.dataset == ds) & (S_.class_report == "persistent_spatial")]
            order = d.setting.tolist()
            x = np.arange(len(order)) + (j - 0.5) * 0.3
            a.bar(x, d.R_pre_to_post_median, 0.3, yerr=[d.R_pre_to_post_median - d.R_pre_to_post_median_lo, d.R_pre_to_post_median_hi - d.R_pre_to_post_median], capsize=2, label=f"{W.LABEL[ds]} retention", alpha=0.85 if j == 0 else 0.5, color="tab:red")
            a.set_xticks(np.arange(len(order))); a.set_xticklabels(order, fontsize=8, rotation=20); a.set_ylim(0, 1.05); a.set_title(f"{name} setting (persistent spatial)", fontsize=10); a.legend(fontsize=8)
    ax[0].set_xlabel("friction coefficient μ"); ax[0].set_title("friction sensitivity (persistent spatial)", fontsize=10); ax[0].legend(fontsize=7); ax[0].axhline(1, color="k", lw=0.8)
    fig.tight_layout(); fig.savefig(FIG / "figE_sensitivity.png", dpi=140); plt.close(fig)


def fig_f(n_per=2):
    E = events(); lab = W.mano_labels()
    U_names = None; sheet = []
    for ds in W.DATASETS:
        d = E[(E.dataset == ds) & (E.class_report == "persistent_spatial")].dropna(subset=["R_pre_to_post"])
        thr = json.load(open(W.OUT / "thresholds.json")).get(ds, {})
        big = d[d.dC >= thr.get("dC_q50", d.dC.median())]
        equiv = big.sort_values("R_pre_to_post", ascending=False).head(n_per)              # large contact change, capability retained
        improve = big.sort_values("rel_change_Q", ascending=False).head(n_per)             # large contact change, post clearly better
        cap = np.load(W.ds_out(ds) / "capacity" / "primary.npz", allow_pickle=True); ids = list(cap["event_id"]); U_names = list(cap["direction"])
        for kind, sub in (("equivalent", equiv), ("improved", improve)):
            for ev in sub.itertuples():
                g = load_geometry(ds, ev.event_id); mesh = mesh_of(ds, ev)
                p_pre = W.build_patches(geom_dict(g, "pre"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
                p_post = W.build_patches(geom_dict(g, "post"), mesh.adj, W.CONTACT_THR["primary"], W.MERGE_R["primary"])
                fig = plt.figure(figsize=(16, 12)); gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.8])
                ax0 = [fig.add_subplot(gs[0, k]) for k in range(3)]; ax1 = [fig.add_subplot(gs[1, k]) for k in range(3)]
                draw_frame(ax0, g, "pre", mesh, p_pre, lab, f"PRE frame {g['frames']['pre']}: {len(p_pre)} patches, {'+'.join(sorted({W.PARTS[p['part']] for p in p_pre}))}")
                draw_frame(ax1, g, "post", mesh, p_post, lab, f"POST frame {g['frames']['post']}: {len(p_post)} patches, {'+'.join(sorted({W.PARTS[p['part']] for p in p_post}))}")
                a = fig.add_subplot(gs[2, :])
                i = ids.index(ev.event_id); qp, qq = cap["q_pre"][i], cap["q_post"][i]
                order = np.argsort(-qq)
                a.plot(qq[order], color="tab:blue", lw=2, label=f"post profile h(u), mean {qq.mean():.2f}")
                a.plot(qp[order], color="tab:red", lw=2, label=f"pre profile h(u), mean {qp.mean():.2f}")
                a.set_xlabel("direction (sorted by the post capability)"); a.set_ylabel("directional capacity h(u)")
                a.set_title(f"capability profiles over the 76 shared directions — retention R = {ev.R_pre_to_post:.2f}, relative change {ev.rel_change_Q:+.2f}, cosine {ev.cosine:.2f}", fontsize=10); a.legend(fontsize=8)
                fig.suptitle(f"{W.LABEL[ds]} — {kind}: persistent spatial event {ev.event_id} ({ev.sequence_id}, {ev.group}); ‖ΔC‖ = {ev.dC:.2f}; fingers: "
                             + ", ".join(f"{p} {getattr(ev, 'fc_' + p)}" for p in W.PARTS if getattr(ev, 'fc_' + p) != 'absent'), fontsize=10)
                fig.tight_layout(); name = f"figF_{ds}_{kind}_{ev.event_id}.png"; fig.savefig(FIG / name, dpi=110); plt.close(fig)
                sheet.append(dict(dataset=ds, kind=kind, event_id=ev.event_id, sequence_id=ev.sequence_id, group=ev.group, dC=ev.dC, R=ev.R_pre_to_post, rel_change_Q=ev.rel_change_Q, cosine=ev.cosine,
                                  fingers_pre=ev.fingers_pre, fingers_post=ev.fingers_post, figure=name, **{f"fc_{p}": getattr(ev, "fc_" + p) for p in W.PARTS}))
    pd.DataFrame(sheet).to_csv(FIG / "figF_examples.csv", index=False)
    return pd.DataFrame(sheet)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_a(); fig_b(); fig_c(); fig_d(); fig_e(); print(fig_f().to_string())


if __name__ == "__main__":
    main()
