#!/usr/bin/env python
"""Step 5: figures A-F from the result tables of both datasets.  python figures.py
  A  Delta-C error of F0 / F1 / F2 at h = 1, 4, 8 (all frames) with the zero-change reference
  B  the same restricted to persistent spatial / persistent mixed / release frames
  C  amount-change vs spatial-change error for F0 / F1 / F2 (h = 4, 8)
  D  event-prediction AUPRC (and AUROC) at h = 4, 8, with the chance level (prevalence)
  E  error vs hand-history length (current only, 1, 4, 8 previous frames) + motion-only, shuffled
     control and the non-causal oracle, on all frames and on the persistent-transition frames
  F  event-aligned qualitative examples (GT contact map, hand history, ||Delta_4 C||, predictions of
     F0 / F1 / F2, object variables, event start), helpful and unhelpful cases
Writes OUT/figures/fig{A..E}_*.png and OUT/<ds>/figures/figF_*.png (+ a contact sheet).
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hp_common as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("figures")
COL = {"F0": "0.35", "F1": "tab:blue", "F2k8": "tab:red", "F2k1": "#f4a582", "F2k4": "#d6604d", "F2rel": "tab:purple", "F2shuf": "tab:olive", "Ffut": "tab:green", "Ffut1": "#a1d99b"}
FIG = P.OUT / "figures"


def load_all(name):
    parts = []
    for ds in P.DATASETS:
        p = P.ds_out(ds) / "results" / f"{name}.csv"
        if p.exists() and p.stat().st_size > 1:
            parts.append(pd.read_csv(p))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bars(ax, df, conds, xcats, xkey, ykey, lo, hi, ref=None, ref_label="zero-change", width=0.8):
    n = len(conds); w = width / n
    for i, c in enumerate(conds):
        d = df[df.cond == c].set_index(xkey).reindex(xcats)
        x = np.arange(len(xcats)) + (i - (n - 1) / 2) * w
        ax.bar(x, d[ykey].values, w, color=COL.get(c, "0.5"), label=P.CONDITIONS[c]["label"], yerr=[d[ykey].values - d[lo].values, d[hi].values - d[ykey].values], capsize=2, error_kw=dict(lw=0.8))
    if ref is not None:
        d = df[df.cond == conds[0]].set_index(xkey).reindex(xcats)
        ax.plot(np.arange(len(xcats)), d[ref].values, "k_", ms=18, mew=1.5, label=ref_label)
    ax.set_xticks(np.arange(len(xcats)))


def fig_a():
    R = load_all("overall_delta_results")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for j, ds in enumerate(P.DATASETS):
        d = R[(R.dataset == ds)]
        bars(ax[0, j], d, list(P.PRIMARY), list(P.HORIZONS), "h", "E", "E_lo", "E_hi", ref="E0")
        ax[0, j].set_xticklabels([f"h = {h}" for h in P.HORIZONS]); ax[0, j].set_ylabel("E_Δ(h) = ‖Δ̂_h C − Δ_h C‖₂ (raw units)")
        ax[0, j].set_title(f"{P.LABEL[ds]}: Δ-contact error, all valid test frames\n(seed mean; take-bootstrap 95 % CI)", fontsize=10); ax[0, j].legend(fontsize=8)
        bars(ax[1, j], d, list(P.PRIMARY), list(P.HORIZONS), "h", "improvement", "improvement_lo", "improvement_hi")
        ax[1, j].axhline(0, color="k", lw=0.8); ax[1, j].set_xticklabels([f"h = {h}" for h in P.HORIZONS]); ax[1, j].set_ylabel("improvement over zero change  1 − ΣE/ΣE0")
        ax[1, j].set_title(f"{P.LABEL[ds]}: relative improvement over the zero-change predictor")
    fig.tight_layout(); fig.savefig(FIG / "figA_delta_error.png", dpi=140); plt.close(fig)


def fig_b():
    R = load_all("event_conditioned_results")
    cls = ["persistent_spatial", "persistent_mixed", "release", "onset", "transient", "non_spike"]
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    for j, ds in enumerate(P.DATASETS):
        for i, h in enumerate((4, 8)):
            d = R[(R.dataset == ds) & (R.h == h) & R.frame_class.isin(cls)]
            present = [c for c in cls if c in set(d.frame_class)]
            bars(ax[i, j], d, list(P.PRIMARY), present, "frame_class", "E", "E_lo", "E_hi", ref="E0")
            short = {"persistent_spatial": "persistent\nspatial", "persistent_mixed": "persistent\nmixed", "transient": "transient", "non_spike": "non-spike", "release": "release", "onset": "onset"}
            ax[i, j].set_xticklabels([f"{short[c]}\n(n={int(d[(d.cond == 'F0') & (d.frame_class == c)].n_frames.iloc[0])})" for c in present], fontsize=8)
            ax[i, j].set_ylabel(f"E_Δ(h={h})"); ax[i, j].set_title(f"{P.LABEL[ds]}, h = {h}: Δ-contact error by frame class (dominant event inside Δ_h)", fontsize=10); ax[i, j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figB_event_conditioned.png", dpi=140); plt.close(fig)


def fig_c():
    R = load_all("event_conditioned_results")
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for j, ds in enumerate(P.DATASETS):
        for i, h in enumerate((4, 8)):
            a = ax[i, j]
            for cls, mk in (("all", "o"), ("persistent_spatial", "s"), ("persistent_mixed", "^"), ("release", "D")):
                d = R[(R.dataset == ds) & (R.h == h) & (R.frame_class == cls)]
                for c in P.PRIMARY:
                    r = d[d.cond == c]
                    if len(r):
                        a.scatter(r.eA, r.eS, marker=mk, s=70, color=COL[c], edgecolor="k", lw=0.5, label=P.CONDITIONS[c]["label"] if cls == "all" else None)
                pts = d[d.cond.isin(P.PRIMARY)].sort_values("cond", key=lambda s: s.map({c: k for k, c in enumerate(P.PRIMARY)}))
                if len(pts) > 1:
                    a.plot(pts.eA, pts.eS, "-", color="0.6", lw=0.8)
                    a.annotate(P.CLASS_LABEL[cls], (pts.eA.iloc[-1], pts.eS.iloc[-1]), fontsize=7, xytext=(4, 4), textcoords="offset points")
            a.set_xlabel("amount-change error  e_A = |Δm̂ − Δm|·‖P̄‖"); a.set_ylabel("spatial-change error  e_S = m̄·‖P̂_{t+h} − P_{t+h}‖")
            a.set_title(f"{P.LABEL[ds]}, h = {h} (markers: frame class; colour: condition)"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figC_amount_vs_spatial.png", dpi=140); plt.close(fig)


def fig_d():
    R = load_all("transition_prediction_results")
    if not len(R):
        return
    conds = [c for c in ("F0", "F1", "F2k8", "F2rel", "F2shuf", "Ffut1", "Ffut") if c in set(R.cond)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, ds in enumerate(P.DATASETS):
        d = R[R.dataset == ds]
        if not len(d):
            continue
        xcats = list(P.EVENT_HORIZONS)
        bars(ax[j], d, conds, xcats, "h", "auprc", "auprc_lo", "auprc_hi", ref="prevalence", ref_label="chance (prevalence)")
        ax[j].set_xticklabels([f"h = {h}\n(AUROC F0 {d[(d.cond == 'F0') & (d.h == h)].auroc.iloc[0]:.2f}, F2 {d[(d.cond == 'F2k8') & (d.h == h)].auroc.iloc[0]:.2f})" for h in xcats], fontsize=8)
        ax[j].set_ylabel("AUPRC (seed mean, take-bootstrap 95 % CI)"); ax[j].set_title(f"{P.LABEL[ds]}: does a persistent contact-mode transition start within h frames?"); ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "figD_event_prediction.png", dpi=140); plt.close(fig)


def fig_e():
    R = load_all("history_length_ablation")
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for j, ds in enumerate(P.DATASETS):
        for i, h in enumerate((4, 8)):
            a = ax[i, j]
            for cls, ls, mk in (("all", "-", "o"), ("persistent_spatial", "--", "s"), ("persistent_mixed", ":", "^"), ("release", "-.", "D")):
                d = R[(R.dataset == ds) & (R.h == h) & (R.frame_class == cls)].set_index("cond")
                if not len(d):
                    continue
                ks = [c for c in P.HISTORY_ORDER if c in d.index]
                x = [P.CONDITIONS[c]["k"] for c in ks]; y = d.loc[ks, "E"].values; lo = d.loc[ks, "E_lo"].values; hi = d.loc[ks, "E_hi"].values
                base = d.loc["F0", "E"] if "F0" in d.index else np.nan
                a.errorbar(x, y / base, yerr=[(y - lo) / base, (hi - y) / base], ls=ls, marker=mk, color="tab:red", capsize=2, label=f"F2 (abs. states), {P.CLASS_LABEL[cls]}")
                for c, xx, col in (("F2rel", 10, COL["F2rel"]), ("F2shuf", 11.5, COL["F2shuf"]), ("Ffut1", 13, COL["Ffut1"]), ("Ffut", 14.5, COL["Ffut"])):
                    if c in d.index:
                        a.errorbar([xx], [d.loc[c, "E"] / base], yerr=[[(d.loc[c, "E"] - d.loc[c, "E_lo"]) / base], [(d.loc[c, "E_hi"] - d.loc[c, "E"]) / base]], marker=mk, color=col, capsize=2, ls="none",
                                   label=P.CONDITIONS[c]["label"] if cls == "all" else None)
            a.axhline(1.0, color="0.35", lw=1.2, label="F0 (no hand)")
            a.axvline(9, color="0.8", lw=0.8, ls=":")
            a.set_xticks([0, 1, 4, 8, 10, 11.5, 13, 14.5]); a.set_xticklabels(["H_t only", "k=1", "k=4", "k=8", "motion\nonly", "shuffled\n(control)", "oracle\n+1 ⚠", "oracle\n+8 ⚠"], fontsize=8)
            a.set_xlabel("hand history length k (frames before t)"); a.set_ylabel("E_Δ relative to F0"); a.set_title(f"{P.LABEL[ds]}, h = {h}"); a.legend(fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / "figE_history_length.png", dpi=140); plt.close(fig)


def fig_f(ds, n_examples=4):
    """Event-aligned qualitative examples: the persistent spatial / mixed / release events where the
    causal hand history helps most and least (F2k8 vs F0, E at h = 4, averaged over the 4 frames
    before the event start), from the seed-0 predictions."""
    out = P.ds_out(ds); D = P.load_raw(ds); meta, man = D["meta"], D["man"]; te = man["test"]
    z = {c: np.load(out / "preds" / f"delta_{c}_seed0.npz") for c in P.PRIMARY}
    C = D["C"][te].astype(np.float32); O = D["O"][te, P.PAD:P.PAD + P.T]; names = D["names"]
    H = P.load_hand_cache(ds)["hand"][te]                                        # (n, 80, 100, 3)
    E = P.load_events(ds); E = E[(E.split_set == "test") & E.positive_event & (E.start_frame >= 6) & (E.start_frame <= 50)].copy()
    pos = {n: i for i, n in enumerate(te)}
    j4 = list(P.HORIZONS).index(4); h = 4
    gt4 = C[:, h:] - C[:, :-h]
    err = {c: np.linalg.norm(z[c]["pred"][:, :P.T - h, j4].astype(np.float32) - gt4, axis=2) for c in P.PRIMARY}   # (n, 60)
    E["gain"] = [float(np.mean(err["F0"][pos[r.example], r.start_frame - 4:r.start_frame] - err["F2k8"][pos[r.example], r.start_frame - 4:r.start_frame])) for r in E.itertuples()]
    E = E.sort_values("gain", ascending=False)
    picks = list(E.head(n_examples // 2).itertuples()) + list(E.tail(n_examples // 2).itertuples())
    kin = P.K.kinematics(ds, D, te)
    sheet = []
    for ev in picks:
        i = pos[ev.example]; s = ev.start_frame; lo, hi = max(0, s - 12), min(P.T - 1, s + 12)
        fig, ax = plt.subplots(2, 3, figsize=(17, 8))
        # (1) GT contact map over time
        a = ax[0, 0]; im = a.imshow(C[i].T, aspect="auto", cmap="magma", origin="lower", interpolation="nearest"); a.axvline(s, color="w", ls="--"); a.set_xlabel("frame t"); a.set_ylabel("canonical point")
        a.set_title(f"GT contact map C_t ({meta.iloc[ev.example].group}, ex {ev.example})\n{ev.event_category} event starts at t = {s}", fontsize=9); plt.colorbar(im, ax=a, fraction=0.03)
        # (2) hand history: centroid + spread of the 100 points in the object frame, and the hand-relative speed
        a = ax[0, 1]; cen = H[i, P.PAD:P.PAD + P.T].mean(1) * 100
        for k, lab in enumerate("xyz"):
            a.plot(np.arange(P.T), cen[:, k], label=f"hand centroid {lab} [cm]")
        a2 = a.twinx(); a2.plot(np.arange(P.T - 1) + 0.5, np.linalg.norm(np.diff(H[i, P.PAD:P.PAD + P.T], axis=0), axis=2).mean(1) * 1000, color="k", lw=0.8, label="hand speed rel. object [mm/frame]")
        a.axvline(s, color="r", ls="--"); a.axvspan(s - 8, s, color="r", alpha=0.08, label="causal window at t = s (k=8)"); a.set_xlabel("frame t"); a.set_title("hand in the object frame (causal history shaded)", fontsize=9); a.legend(fontsize=7, loc="upper left"); a2.legend(fontsize=7, loc="upper right")
        # (3) GT ||Delta_4 C|| and errors of F0/F1/F2
        a = ax[0, 2]; tt = np.arange(P.T - h)
        a.plot(tt, np.linalg.norm(gt4[i], axis=1), color="k", lw=1.5, label="‖Δ₄C_t‖ GT")
        for c in P.PRIMARY:
            a.plot(tt, err[c][i], color=COL[c], lw=1.2, label=f"E_Δ(4) {c}")
        a.axvline(s, color="r", ls="--"); a.set_xlim(lo, hi); a.set_xlabel("frame t (prediction made at t, target C_{t+4} − C_t)"); a.set_title(f"4-frame change and prediction error\n(F0 − F2 error over the 4 frames before the event: {ev.gain:+.3f})", fontsize=9); a.legend(fontsize=7)
        # (4) predicted vs GT Delta_4 at t = s-1 (the last causal frame before the event) as vectors over the canonical points
        a = ax[1, 0]; t_ = max(0, s - 1)
        a.plot(gt4[i, t_], color="k", lw=1, label=f"GT Δ₄C at t={t_}")
        for c in P.PRIMARY:
            a.plot(z[c]["pred"][i, t_, j4].astype(np.float32), color=COL[c], lw=0.8, alpha=0.8, label=c)
        a.set_xlabel("canonical point index"); a.set_ylabel("Δ₄ C"); a.set_title("predicted vs GT 4-frame change at the last frame before the event", fontsize=9); a.legend(fontsize=7)
        # (5) object trajectory variables
        a = ax[1, 1]
        for k in P.K.KIN_AVAILABLE[ds]:
            if k in kin and not np.all(np.isnan(kin[k][i])):
                v = kin[k][i]; a.plot(np.arange(P.T), v / (np.nanmax(v) + 1e-9), label=f"{P.K.KIN_LABEL[k].split(' [')[0]} (max {np.nanmax(v):.3g})")
        a.axvline(s, color="r", ls="--"); a.set_xlabel("frame t"); a.set_ylabel("normalised to max"); a.set_title("object trajectory variables"); a.legend(fontsize=7)
        # (6) hand points + contact at s-4, s-1, s+3 (top view; contact-weighted canonical points)
        a = ax[1, 2]
        for k, (tf, col) in enumerate(((s - 4, "tab:blue"), (s - 1, "tab:orange"), (min(P.T - 1, s + 3), "tab:red"))):
            hp = H[i, P.PAD + tf] * 100
            a.scatter(hp[:, 0], hp[:, 1], s=6, color=col, alpha=0.7, label=f"hand t={tf}")
        a.set_aspect("equal"); a.set_xlabel("x [cm]"); a.set_ylabel("y [cm]"); a.set_title("hand points in the object frame\nbefore / at / after the event (top view)", fontsize=9); a.legend(fontsize=7)
        fig.suptitle(f"{P.LABEL[ds]} — {'helpful' if ev.gain > 0 else 'unhelpful'} case: {ev.event_category} event (persistence h4 {ev.persistence_h4:.2f}), sequence {ev.sequence_id}", fontsize=11)
        fig.tight_layout(); name = f"figF_{ds}_{'help' if ev.gain > 0 else 'nohelp'}_{ev.example}_{s}.png"; fig.savefig(out / "figures" / name, dpi=120); plt.close(fig)
        sheet.append(dict(dataset=ds, example=ev.example, sequence_id=ev.sequence_id, group=meta.iloc[ev.example].group, event_category=ev.event_category, start_frame=s,
                          persistence_h4=ev.persistence_h4, gain_F0_minus_F2_h4=ev.gain, figure=name))
    pd.DataFrame(sheet).to_csv(out / "figures" / "figF_examples.csv", index=False)
    return pd.DataFrame(sheet)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_a(); fig_b(); fig_c(); fig_d(); fig_e()
    for ds in P.DATASETS:
        (P.ds_out(ds) / "figures").mkdir(exist_ok=True)
        fig_f(ds)
    log.info("figures written to %s", FIG)


if __name__ == "__main__":
    main()
