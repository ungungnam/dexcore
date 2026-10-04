#!/usr/bin/env python
"""Figures A-G for one dataset:  python tce_figures.py --dataset taco
Reads OUT/<ds>/{frames_test.csv, events.csv, event_summary.csv, model_error_by_event.csv, event_aligned.npz}
and the experiment root's sequences.npz / preds for the qualitative panels. Writes OUT/<ds>/figures/*.png (+ .pdf).
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tce_common as K

CAT_ORDER = ["onset", "release", "onset+release", "amount", "spatial", "mixed", "near_zero"]
CAT_COL = {"onset": "C2", "release": "C3", "onset+release": "C8", "amount": "C0", "spatial": "C1", "mixed": "C4", "near_zero": "0.6", "transient": "C5", "non_spike": "0.7", "spike": "k", "all": "0.4"}
MODEL_STYLE = {"e_B0_static": ("B0 static", "k"), "e_B1_gtinit_vf": ("B1 GT-init + VF", "C0"), "e_B2_samplerG_vf_K10": ("B2 p(S0|G)+VF best-of-10", "C1"),
               "e_B2_samplerG_vf_K1": ("B2 K=1", "C1"), "e_B3_samplerGT_vf_K10": ("B3 best-of-10", "C2"), "e_B4_bimart_K10_own": ("B4 BimArt best-of-10 (own support)", "C5"),
               "e_B4_bimart_K10_full": ("B4 BimArt best-of-10 (vs full GT)", "C6")}


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=130); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def fig_A(ds, F, san, fig_dir):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    d = F.d.values
    ax[0].hist(d, bins=120, color="0.6", alpha=0.9, log=True); ax[0].axvline(san["q90"], color="C3", ls="--", label=f"train Q90 = {san['q90']:.3f}"); ax[0].axvline(san["q95"], color="C1", ls=":", label=f"train Q95 = {san['q95']:.3f}")
    ax[0].set(xlabel="d_t = ||C_{t+1} − C_t||  (raw, test transitions)", ylabel="count (log)", title=f"{K.LABEL[ds]}: distribution of temporal change (test)"); ax[0].legend(fontsize=8)
    xs = np.sort(d); ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="k", label="test ECDF")
    ax[1].axvline(san["q90"], color="C3", ls="--"); ax[1].axvline(san["q95"], color="C1", ls=":")
    ax[1].axhline(1 - san["frac_spike_frames_q90"], color="C3", ls="--", lw=0.6); ax[1].set(xlabel="d_t", ylabel="ECDF", title=f"test frames above train Q90: {100 * san['frac_spike_frames_q90']:.1f} %, carrying {100 * san['spike_energy_share_of_all']:.0f} % of Σ d_t²")
    ax[1].set_xscale("log"); ax[1].legend(fontsize=8)
    fig.tight_layout(); save(fig, fig_dir / "figA_change_distribution")


def fig_B(ds, E, fig_dir):
    fig, ax = plt.subplots(figsize=(6, 5.2))
    for c in CAT_ORDER:
        g = E[E.event_category == c]
        if len(g):
            ax.scatter(g.amount_component, g.spatial_component, s=14, alpha=0.7, color=CAT_COL[c], label=f"{c} (n={len(g)})")
    m = max(E.amount_component.max(), E.spatial_component.max()); ax.plot([0, m], [0, m], "k--", lw=0.6)
    for r_, ls in ((0.7, ":"), (0.3, ":")):
        ax.plot([0, m], [0, m * (1 - r_) / r_], color="0.5", ls=ls, lw=0.6)
    ax.set(xlabel="aggregated amount component  Σ_t ‖Δm_t P̄_t‖", ylabel="aggregated spatial component  Σ_t ‖m̄_t ΔP_t‖", title=f"{K.LABEL[ds]}: amount vs spatial per spike event (onset/release classified first)")
    ax.legend(fontsize=7); fig.tight_layout(); save(fig, fig_dir / "figB_amount_vs_spatial")


def fig_C(ds, S, fig_dir):
    S = S.set_index("category").reindex([c for c in CAT_ORDER + ["transient"] if c in S.category.values])
    x = np.arange(len(S)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - w / 2, S.frequency, w, color="0.55", label="event frequency (share of events)")
    ax.bar(x + w / 2, S.energy_share, w, color="C3", label="change energy (share of Σ d_t² over spike frames)")
    ax.errorbar(x + w / 2, S.energy_share, yerr=[S.energy_share - S.energy_lo, S.energy_hi - S.energy_share], fmt="none", ecolor="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(S.index); ax.set(ylabel="fraction", title=f"{K.LABEL[ds]}: event composition — frequency vs change energy (take-bootstrap 95 % CI)"); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, fig_dir / "figC_event_composition")


def fig_D(ds, E, fig_dir):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, (col, lab) in enumerate((("persistence_h2", "persistence at the peak, h = 2"), ("persistence_h4", "persistence at the peak, h = 4"), ("persistence_event", "event-level net displacement / path length"))):
        bins = np.linspace(0, 1, 26)
        ax[j].hist(E[col].dropna(), bins=bins, color="0.75", label=f"all events (n={E[col].notna().sum()})")
        for c in ("amount", "spatial", "mixed"):
            g = E[E.event_category == c][col].dropna()
            if len(g) > 3:
                ax[j].hist(g, bins=bins, histtype="step", lw=1.4, color=CAT_COL[c], label=f"{c} (n={len(g)})")
        oo = E[E.event_category.isin(["onset", "release", "onset+release"])][col].dropna()
        if len(oo) > 3:
            ax[j].hist(oo, bins=bins, histtype="step", lw=1.4, color="C2", ls="--", label=f"onset/release (n={len(oo)})")
        ax[j].set(xlabel=lab, ylabel="events"); ax[j].legend(fontsize=7)
    fig.suptitle(f"{K.LABEL[ds]}: persistence of spike events (1 = consistent direction, 0 = cancels out)", fontsize=10)
    fig.tight_layout(); save(fig, fig_dir / "figD_persistence")


def fig_E(ds, E, aligned, fig_dir):
    sigs = [s for s in K.KIN_AVAILABLE[ds] if s in aligned]
    if not sigs:
        return
    groups = {"amount": E.event_category.values == "amount", "spatial": E.event_category.values == "spatial", "mixed": E.event_category.values == "mixed",
              "onset/release": np.isin(E.event_category.values, ["onset", "release", "onset+release"])}
    fig, axes = plt.subplots(1, len(sigs) + 1, figsize=(3.4 * (len(sigs) + 1), 3.4))
    tt = np.arange(-8, 9)
    for j, s in enumerate(sigs + ["d"]):
        ax = axes[j]; W = aligned[s]
        for c, sel in groups.items():
            if sel.sum() < 5:
                continue
            w = W[sel]; mu = np.nanmean(w, 0)
            # take-cluster bootstrap on the mean curve
            takes = E.take_key.values[sel]; u, inv = np.unique(takes, return_inverse=True); rng = np.random.default_rng(0); reps = []
            for _ in range(200):
                wt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u))[inv].astype(float)
                reps.append(np.nansum(w * wt[:, None], 0) / np.maximum((wt[:, None] * ~np.isnan(w)).sum(0), 1e-9))
            lo, hi = np.nanpercentile(reps, 2.5, 0), np.nanpercentile(reps, 97.5, 0)
            col = CAT_COL["onset"] if c == "onset/release" else CAT_COL[c]
            ax.plot(tt, mu, color=col, label=f"{c} (n={sel.sum()})"); ax.fill_between(tt, lo, hi, color=col, alpha=0.15)
        ax.axvline(0, color="k", lw=0.6); ax.set(xlabel="frames from event peak", title=K.KIN_LABEL.get(s, "d_t (contact change)"), xticks=[-8, -4, 0, 4, 8])
        ax.title.set_fontsize(8)
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle(f"{K.LABEL[ds]}: event-aligned kinematics (mean, take-bootstrap 95 % band)", fontsize=10)
    fig.tight_layout(); save(fig, fig_dir / "figE_event_aligned_kinematics")


def fig_F(ds, M, fig_dir):
    classes = [c for c in ["non_spike", "spike", "onset", "release", "amount", "spatial", "mixed", "transient"] if c in M.frame_class.values]
    M = M.set_index("frame_class")
    models = [m for m in MODEL_STYLE if m in M.columns and m not in ("e_B2_samplerG_vf_K1", "e_B3_samplerGT_vf_K10", "e_B4_bimart_K10_full")]
    x = np.arange(len(classes)); w = 0.8 / len(models)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw=dict(width_ratios=[2, 1]))
    for i, m in enumerate(models):
        v = M.loc[classes, m].values; lo = M.loc[classes, f"{m}_lo"].values; hi = M.loc[classes, f"{m}_hi"].values
        ax[0].bar(x + (i - len(models) / 2 + 0.5) * w, v, w, color=MODEL_STYLE[m][1], label=MODEL_STYLE[m][0], yerr=[v - lo, hi - v], error_kw=dict(lw=0.6))
    ax[0].set_xticks(x); ax[0].set_xticklabels(classes); ax[0].set(ylabel="temporal error e_t = ‖ΔĈ_t − ΔC_t‖ (raw)", title=f"{K.LABEL[ds]}: temporal error by frame class (take-bootstrap 95 % CI)"); ax[0].legend(fontsize=7)
    # share of each model's total temporal error by class
    share_cols = [f"{m}_share_of_total" for m in models]
    cls2 = [c for c in classes if c not in ("spike", "transient")]
    bottom = np.zeros(len(models))
    for c in cls2:
        vals = np.array([M.loc[c, f"{m}_share_of_total"] for m in models])
        ax[1].bar(np.arange(len(models)), vals, bottom=bottom, color=CAT_COL.get(c, "0.5"), label=c); bottom += vals
    ax[1].set_xticks(np.arange(len(models))); ax[1].set_xticklabels([MODEL_STYLE[m][0].split(" (")[0] for m in models], rotation=20, fontsize=7); ax[1].set(ylabel="share of Σ_t e_t", title="where each model's temporal error sits"); ax[1].legend(fontsize=7)
    fig.tight_layout(); save(fig, fig_dir / "figF_model_error_by_event")


def project(P):
    Pc = P - P.mean(0); U, s, Vt = np.linalg.svd(Pc, full_matrices=False)
    return Pc @ Vt[:2].T


def fig_G(ds, F, E, D, fig_dir, transient_thr):
    """Representative events: for each category the event nearest the median peak_d (representative) and the event with the
    largest VF temporal error (failure); plus the lowest-persistence event."""
    P2 = {c: project(D["P"][j]) for j, c in enumerate(D["cats"])}
    preds = D["preds"]; bim = D["bimart"]; te = D["man"]["test"]; pos = {n: i for i, n in enumerate(te)}
    picks = []
    inner = E[(E.peak_frame >= 4) & (E.peak_frame <= 58)]          # events with context on both sides (the edge events are still in the tables)
    for c in ["amount", "spatial", "mixed", "onset", "release"]:
        g = inner[inner.event_category == c]
        if len(g) == 0:
            g = E[E.event_category == c]
        if len(g) == 0:
            continue
        picks.append((c + " (representative)", g.iloc[(g.peak_d - g.peak_d.median()).abs().argsort().iloc[0]]))
        picks.append((c + " (VF failure: max e_B1)", g.loc[g.e_B1_gtinit_vf.idxmax()]))
    g = inner.dropna(subset=["persistence_h4"])
    if len(g):
        picks.append(("lowest persistence h4 (transient / jitter-like)", g.loc[g.persistence_h4.idxmin()]))
    for k, (label, ev) in enumerate(picks):
        n = int(ev.example); i = pos[n]; Fs = F[F.example == n]
        C = D["C"][n].astype(np.float32); pk = int(ev.peak_frame); row = D["meta"].iloc[n]
        frames = list(dict.fromkeys([max(0, int(ev.start_frame) - 4), int(ev.start_frame), pk, min(63, int(ev.end_frame) + 1), min(63, int(ev.end_frame) + 5)]))
        seqs = [("GT", C), ("B1 GT-init + VF", preds["gtinit_vf"][i].astype(np.float32)), ("B2 best-of-10", preds["samplerG_vf__K10"][i].astype(np.float32))]
        if bim is not None:
            seqs.append(("B4 BimArt best-of-10 (X_r)", bim["bimart_K10"][i].astype(np.float32)))
        ncol = max(len(frames), 4)
        fig = plt.figure(figsize=(2.2 * ncol + 4, 2.0 * len(seqs) + 3.2))
        gs = fig.add_gridspec(len(seqs) + 2, ncol, height_ratios=[1] * len(seqs) + [1.1, 1.1])
        vmax = max(C.max(), 1e-3); pp = P2[row.category]
        for r_, (lab, seq) in enumerate(seqs):
            for c_, t in enumerate(frames):
                ax = fig.add_subplot(gs[r_, c_]); ax.scatter(pp[:, 0], pp[:, 1], c=np.clip(seq[t], 0, vmax), s=5, cmap="inferno", vmin=0, vmax=vmax); ax.set_aspect("equal"); ax.axis("off")
                if r_ == 0:
                    ax.set_title(f"t = {t}" + (" (peak)" if t == pk else ""), fontsize=8)
                if c_ == 0:
                    ax.text(-0.05, 0.5, lab, transform=ax.transAxes, fontsize=7, ha="right", va="center")
        ax1 = fig.add_subplot(gs[len(seqs), :]); t = Fs.t.values
        ax1.plot(t, Fs.d, "k", label="d_t"); ax1.plot(t, Fs.A, color="C0", lw=0.9, label="A_t amount"); ax1.plot(t, Fs.S, color="C1", lw=0.9, label="S_t spatial")
        ax1.plot(t, Fs.e_B1_gtinit_vf, color="C0", ls="--", lw=0.8, label="e_t B1"); ax1.plot(t, Fs.e_B2_samplerG_vf_K10, color="C1", ls="--", lw=0.8, label="e_t B2")
        if bim is not None:
            ax1.plot(t, Fs.e_B4_bimart_K10_own, color="C5", ls="--", lw=0.8, label="e_t B4 (own support)")
        ax1.axvspan(ev.start_frame - 0.5, ev.end_frame + 0.5, color="C3", alpha=0.15); ax1.set(ylabel="raw units", xlim=(0, 62)); ax1.legend(fontsize=6, ncol=6)
        ax2 = fig.add_subplot(gs[len(seqs) + 1, :]); ax2.plot(t, Fs.m_t, "k", label="m_t (mass)"); ax2.set(ylabel="mass", xlabel="transition t", xlim=(0, 62))
        ax2.fill_between(t, 0, Fs.m_t.max() * Fs.firm_t.astype(float), color="C2", alpha=0.08, label="firm contact")
        ax3 = ax2.twinx()
        for s, col in zip(K.KIN_AVAILABLE[ds], ("C4", "C5", "C6", "C7", "C8")):
            v = Fs[s].values
            if np.isfinite(v).any():
                ax3.plot(t, v / (np.nanmax(v) + 1e-9), color=col, lw=0.8, label=s)
        ax3.set(ylabel="kinematics (norm. to max)"); ax2.legend(fontsize=6, loc="upper left"); ax3.legend(fontsize=6, loc="upper right")
        fig.suptitle(f"{K.LABEL[ds]} — {label}: {row.group}, take {str(row.sequence_id)[:38]}, event frames {ev.start_frame}–{ev.end_frame}, peak d {ev.peak_d:.2f}, "
                     f"A {ev.amount_component:.2f} S {ev.spatial_component:.2f} r {ev.amount_ratio:.2f}, persistence h2 {ev.persistence_h2:.2f} | e_t over event: B0 {ev.e_B0_static:.2f} B1 {ev.e_B1_gtinit_vf:.2f} B2 {ev.e_B2_samplerG_vf_K10:.2f}"
                     + (f" B4 {ev.e_B4_bimart_K10_own:.2f}" if bim is not None else ""), fontsize=8)
        fig.tight_layout(); save(fig, fig_dir / f"figG_{k:02d}_{label.split(' ')[0]}_{'fail' if 'failure' in label else 'rep'}")
        plt.close("all")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args(); ds = a.dataset
    out = K.OUT / ds; fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)
    F = pd.read_csv(out / "frames_test.csv"); E = pd.read_csv(out / "events.csv"); S = pd.read_csv(out / "event_summary.csv")
    M = pd.read_csv(out / "model_error_by_event.csv"); san = json.load(open(out / "sanity.json")); al = np.load(out / "event_aligned.npz", allow_pickle=True)
    aligned = {k: al[k] for k in al.files if k not in ("event_id", "event_category")}
    fig_A(ds, F, san, fig_dir); fig_B(ds, E, fig_dir); fig_C(ds, S, fig_dir); fig_D(ds, E, fig_dir); fig_E(ds, E, aligned, fig_dir); fig_F(ds, M, fig_dir)
    D = K.load(ds); fig_G(ds, F, E, D, fig_dir, san.get("transient_thr"))
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
