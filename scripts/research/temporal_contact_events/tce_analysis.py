#!/usr/bin/env python
"""Steps 2-7 of the temporal contact-change analysis for one dataset:
    python tce_analysis.py --dataset taco [--transient-thr 0.3]
Writes OUT/<ds>/{frames_test.csv, events.csv, event_summary.csv, model_error_by_event.csv,
kinematic_associations.csv, threshold_sensitivity.csv, event_aligned.npz, sanity.json}.
Thresholds (spike Q90/Q95 of d_t) come from the TRAIN sequences; everything else is on TEST.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

import tce_common as K

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tce")
R_THRESHOLDS = {"0.7/0.3": (0.7, 0.3), "0.6/0.4": (0.6, 0.4), "0.8/0.2": (0.8, 0.2)}
CATS = ["onset", "release", "amount", "spatial", "mixed", "near_zero"]


def categorize(events, hi, lo):
    cat = np.where(events.onset_flag & events.release_flag, "onset+release", np.where(events.onset_flag, "onset", np.where(events.release_flag, "release", "")))
    r = events.amount_ratio.values
    rest = np.where(events.n_valid.values == 0, "near_zero", np.where(r >= hi, "amount", np.where(r <= lo, "spatial", "mixed")))
    return np.where(cat == "", rest, cat)


def merge_events(spike, gap=0):
    """spike (T-1,) bool -> list of (start, end) transition indices, merging runs separated by <= gap non-spike frames."""
    ev, start, last = [], None, None
    for t, s in enumerate(spike):
        if s:
            if start is None:
                start = t
            elif t - last > gap + 1:
                ev.append((start, last)); start = t
            last = t
    if start is not None:
        ev.append((start, last))
    return ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--transient-thr", type=float, default=0.5,
                                                                                                  help="descriptive low-persistence flag: persistence at the peak with h=4 below this (no dataset shows a natural separation, so this is a labelled cut, not a category)")
    ap.add_argument("--n-boot", type=int, default=500)
    a = ap.parse_args(); ds = a.dataset
    t_start = time.time()
    D = K.load(ds); out = K.OUT / ds; out.mkdir(parents=True, exist_ok=True)
    meta, man, thr = D["meta"], D["man"], D["thr"]
    C_all = D["C"]
    # ---------------------------------------------------------------- Step 2: train thresholds + identity check
    tr = man["train"]; te = man["test"]
    d_train = np.concatenate([K.decompose(C_all[i].astype(np.float32), thr)["d"] for i in tr])
    q90, q95 = float(np.quantile(d_train, 0.90)), float(np.quantile(d_train, 0.95))
    rng = np.random.default_rng(0); worst = 0.0; n_checked = 0
    for i in rng.choice(te, 20, replace=False):
        dec = K.decompose(C_all[i].astype(np.float32), thr)
        if dec["valid"].any():
            worst = max(worst, float(np.nanmax(dec["resid"][dec["valid"]] / (dec["d"][dec["valid"]] + K.EPS)))); n_checked += int(dec["valid"].sum())
    log.info("%s: train d_t Q90 %.4f Q95 %.4f (n=%d transitions); decomposition identity max relative residual %.2e over %d valid transitions",
             ds, q90, q95, len(d_train), worst, n_checked)
    assert worst < 1e-4, "decomposition identity violated"
    # ---------------------------------------------------------------- per-frame test table
    meta_te = meta.iloc[te].reset_index(drop=True)
    P = D["preds"]; assert (P["example"] == te).all()
    B = D["bimart"]
    if B is not None:
        assert (B["example"] == te).all()
    nh, hand_speed = K.hard_counts_and_hand(ds, meta.iloc[te], D)
    kin = K.kinematics(ds, D, te); kin["hand_rel_speed"] = np.concatenate([hand_speed, np.full((len(te), 1), np.nan, np.float32)], 1)
    rows, per_seq = [], []
    for i, n in enumerate(te):
        C = C_all[n].astype(np.float32); r = meta_te.iloc[i]
        dec = K.decompose(C, thr)
        m = C.sum(1); firm = K.firm_mask(m, nh[i], thr)
        onset = (~firm[:-1]) & firm[1:]; release = firm[:-1] & (~firm[1:])
        p2, p4 = K.persistence(C, 2), K.persistence(C, 4)
        d3, d5 = K.smoothed_d(C, 3), K.smoothed_d(C, 5)
        df = pd.DataFrame(dict(dataset=ds, example=n, sequence_id=r.sequence_id, take_key=r.take_key, group=r.group, category_obj=r.category,
                               t=np.arange(K.T - 1), d=dec["d"], m_t=dec["m0"], m_t1=dec["m1"], dm=dec["dm"], A=dec["A"], S=dec["S"], r=dec["r"],
                               valid_decomp=dec["valid"], firm_t=firm[:-1], firm_t1=firm[1:], onset=onset, release=release,
                               pers_h2=p2[:-1], pers_h4=p4[:-1], d_s3=d3, d_s5=d5, spike90=dec["d"] > q90, spike95=dec["d"] > q95))
        for k in K.KIN_AVAILABLE[ds]:
            df[k] = kin[k][i, :-1]
        # model temporal errors e_t = ||dC^ - dC|| and absolute errors ||C^_t - C_t|| (at frame t of the transition)
        dC = np.diff(C, axis=0)
        for mname, key in K.MODELS.items():
            pr = P[key][i].astype(np.float32)
            df[f"e_{mname}"] = np.linalg.norm(np.diff(pr, axis=0) - dC, axis=1); df[f"abs_{mname}"] = np.linalg.norm(pr - C, axis=1)[:-1]
        if B is not None:
            gr = B["gt_restricted"][i].astype(np.float32); dGr = np.diff(gr, axis=0)
            for mname, key in K.BIMART_MODELS.items():
                pr = B[key][i].astype(np.float32)
                df[f"e_{mname}_own"] = np.linalg.norm(np.diff(pr, axis=0) - dGr, axis=1); df[f"abs_{mname}_own"] = np.linalg.norm(pr - gr, axis=1)[:-1]
                df[f"e_{mname}_full"] = np.linalg.norm(np.diff(pr, axis=0) - dC, axis=1); df[f"abs_{mname}_full"] = np.linalg.norm(pr - C, axis=1)[:-1]
            df["d_restricted"] = np.linalg.norm(dGr, axis=1)
        rows.append(df)
    F = pd.concat(rows, ignore_index=True)
    # ---------------------------------------------------------------- Step 3-5: events (canonical: Q90, adjacency merge)
    ev_rows, aligned = [], {}
    HW = 8
    for q_name, q in (("q90", q90), ("q95", q95)):
        for gap in (0, 1):
            n_ev = 0
            for (n, sid), g in F.groupby(["example", "sequence_id"], sort=False):
                spike = (g.d.values > q)
                for e_i, (s, e) in enumerate(merge_events(spike, gap)):
                    n_ev += 1
                    if not (q_name == "q90" and gap == 0):
                        continue
                    seg = g.iloc[s:e + 1]
                    C = C_all[n].astype(np.float32)
                    pk = int(seg.t.values[np.argmax(seg.d.values)])
                    valid = seg.valid_decomp.values
                    A_sum, S_sum = float(np.nansum(seg.A.values[valid])), float(np.nansum(seg.S.values[valid]))
                    net = float(np.linalg.norm(C[e + 1] - C[s]))
                    rec = dict(dataset=ds, example=n, sequence_id=sid, take_key=seg.take_key.iloc[0], group=seg.group.iloc[0], event_id=f"{n}_{e_i}",
                               start_frame=int(s), end_frame=int(e), peak_frame=pk, duration=int(e - s + 1), peak_d=float(seg.d.max()),
                               integrated_d=float(seg.d.sum()), energy=float((seg.d.values ** 2).sum()),
                               contact_mass_before=float(seg.m_t.iloc[0]), contact_mass_after=float(seg.m_t1.iloc[-1]),
                               amount_component=A_sum, spatial_component=S_sum, amount_ratio=A_sum / (A_sum + S_sum + K.EPS) if valid.any() else np.nan,
                               n_valid=int(valid.sum()), onset_flag=bool(seg.onset.any()), release_flag=bool(seg.release.any()),
                               persistence_h2=float(F.loc[seg.index[np.argmax(seg.d.values)], "pers_h2"]), persistence_h4=float(F.loc[seg.index[np.argmax(seg.d.values)], "pers_h4"]),
                               persistence_event=net / (seg.d.sum() + K.EPS), firm_before=bool(seg.firm_t.iloc[0]), firm_after=bool(seg.firm_t1.iloc[-1]))
                    for k in K.KIN_AVAILABLE[ds]:
                        rec[f"kin_{k}"] = float(np.nanmean(g[k].values[s:e + 2 if e + 2 <= len(g) else len(g)]))
                    for c in [c for c in F.columns if c.startswith("e_") or c.startswith("abs_")]:
                        rec[c] = float(seg[c].mean())
                    ev_rows.append(rec)
                    # event-aligned windows around the peak: kinematics + d + e_t (NaN outside the sequence)
                    for k in K.KIN_AVAILABLE[ds] + ["d"] + [c for c in F.columns if c.startswith("e_")]:
                        w = np.full(2 * HW + 1, np.nan, np.float32)
                        lo_, hi_ = max(0, pk - HW), min(len(g) - 1, pk + HW)
                        w[lo_ - pk + HW:hi_ - pk + HW + 1] = g[k].values[lo_:hi_ + 1]
                        aligned.setdefault(k, []).append(w)
            if q_name == "q90" and gap == 0:
                n_events_canonical = n_ev
            log.info("  %s spikes (%s, merge gap %d): %d events", ds, q_name, gap, n_ev)
            ev_counts = ev_rows  # noqa
    E = pd.DataFrame(ev_rows)
    E["event_category"] = categorize(E, 0.7, 0.3)
    # transient flag (after inspection; default: none)
    E["transient_flag"] = (E.persistence_h4 < a.transient_thr) if a.transient_thr is not None else False   # NaN (edge) -> False
    # map event ids back to frames
    F["event_id"] = ""; F["event_category"] = ""; F["transient_flag"] = False
    fidx = {(n, t): i for i, (n, t) in enumerate(zip(F.example.values, F.t.values))}
    for ev in E.itertuples():
        for t in range(ev.start_frame, ev.end_frame + 1):
            i = fidx[(ev.example, t)]
            F.at[i, "event_id"] = ev.event_id; F.at[i, "event_category"] = ev.event_category; F.at[i, "transient_flag"] = bool(ev.transient_flag)
    F["is_spike"] = F.event_id != ""
    F.to_csv(out / "frames_test.csv", index=False)
    E.to_csv(out / "events.csv", index=False)
    np.savez_compressed(out / "event_aligned.npz", event_id=E.event_id.values, event_category=E.event_category.values,
                        **{k: np.stack(v) for k, v in aligned.items()})
    # ---------------------------------------------------------------- Step 5: composition (frequency, energy, magnitude) + sensitivity
    tot_energy_all = float((F.d.values ** 2).sum()); tot_mag_all = float(F.d.sum())
    spike_energy = float((F.d.values[F.is_spike] ** 2).sum()); spike_mag = float(F.d.values[F.is_spike].sum())
    comp = []
    cats_present = list(dict.fromkeys(list(E.event_category.unique())))
    for c in cats_present + (["transient"] if a.transient_thr is not None else []):
        sel = (E.transient_flag if c == "transient" else (E.event_category == c)).values
        fr = F[F.event_id.isin(E.event_id[sel])]
        n_ev = int(sel.sum())
        en_pt, en_lo, en_hi = K.cluster_bootstrap_ratio((F.d.values ** 2) * F.event_id.isin(E.event_id[sel]).values, (F.d.values ** 2) * F.is_spike.values, F.take_key.values, a.n_boot)
        comp.append(dict(dataset=ds, category=c, n_events=n_ev, frequency=n_ev / max(len(E), 1), energy_share=en_pt, energy_lo=en_lo, energy_hi=en_hi,
                         magnitude_share=float(fr.d.sum() / max(spike_mag, 1e-12)), energy_share_of_all_frames=float((fr.d.values ** 2).sum() / tot_energy_all),
                         n_frames=len(fr), mean_peak_d=float(E.peak_d[sel].mean()) if n_ev else np.nan, median_peak_d=float(E.peak_d[sel].median()) if n_ev else np.nan,
                         mean_duration=float(E.duration[sel].mean()) if n_ev else np.nan, mean_amount=float(E.amount_component[sel].mean()) if n_ev else np.nan,
                         mean_spatial=float(E.spatial_component[sel].mean()) if n_ev else np.nan, median_persistence_h2=float(E.persistence_h2[sel].median()) if n_ev else np.nan,
                         median_persistence_h4=float(E.persistence_h4[sel].median()) if n_ev else np.nan, median_persistence_event=float(E.persistence_event[sel].median()) if n_ev else np.nan,
                         low_persistence_frac=float(E.transient_flag[sel].mean()) if n_ev else np.nan,
                         low_persistence_frac_defined=float((E.persistence_h4[sel] < a.transient_thr).sum() / max(E.persistence_h4[sel].notna().sum(), 1)) if n_ev else np.nan))
    S = pd.DataFrame(comp); S.to_csv(out / "event_summary.csv", index=False)
    sens = []
    for name, (hi, lo) in R_THRESHOLDS.items():
        cat = categorize(E, hi, lo)
        for c in ["onset", "release", "onset+release", "amount", "spatial", "mixed", "near_zero"]:
            sel = cat == c
            sens.append(dict(dataset=ds, spike_q="q90", r_thresholds=name, category=c, n_events=int(sel.sum()), frequency=float(sel.mean()) if len(E) else np.nan,
                             energy_share=float((F.d.values[F.event_id.isin(E.event_id[sel])] ** 2).sum() / max(spike_energy, 1e-12))))
    # Q95 variant: recompute events quickly (canonical categories) for the frequency/energy comparison
    ev95 = []
    for (n, sid), g in F.groupby(["example", "sequence_id"], sort=False):
        for s, e in merge_events(g.d.values > q95, 0):
            seg = g.iloc[s:e + 1]; valid = seg.valid_decomp.values
            A_sum, S_sum = float(np.nansum(seg.A.values[valid])), float(np.nansum(seg.S.values[valid]))
            ev95.append(dict(onset_flag=bool(seg.onset.any()), release_flag=bool(seg.release.any()), n_valid=int(valid.sum()),
                             amount_ratio=A_sum / (A_sum + S_sum + K.EPS) if valid.any() else np.nan, energy=float((seg.d.values ** 2).sum())))
    E95 = pd.DataFrame(ev95)
    if len(E95):
        E95["cat"] = categorize(E95, 0.7, 0.3)
        for c in ["onset", "release", "onset+release", "amount", "spatial", "mixed", "near_zero"]:
            sel = E95.cat == c
            sens.append(dict(dataset=ds, spike_q="q95", r_thresholds="0.7/0.3", category=c, n_events=int(sel.sum()), frequency=float(sel.mean()),
                             energy_share=float(E95.energy[sel].sum() / max(E95.energy.sum(), 1e-12))))
    pd.DataFrame(sens).to_csv(out / "threshold_sensitivity.csv", index=False)
    # ---------------------------------------------------------------- Step 6: kinematic associations
    assoc = []
    frame_cats = {"non_spike": ~F.is_spike.values, "spike": F.is_spike.values}
    for c in cats_present:
        frame_cats[c] = (F.event_category == c).values
    if a.transient_thr is not None:
        frame_cats["transient"] = F.transient_flag.values
    for k in K.KIN_AVAILABLE[ds]:
        x = F[k].values
        rho, lo, hi, n = K.spearman_cluster(x, F.d.values, F.take_key.values, n_boot=200)
        base = dict(dataset=ds, signal=k, spearman_d=rho, spearman_lo=lo, spearman_hi=hi, n_frames=n,
                    cliffs_delta_spike_vs_non=K.cliffs_delta(x[F.is_spike.values], x[~F.is_spike.values]))
        for c, sel in frame_cats.items():
            v = x[sel]
            pt, plo, phi = K.cluster_bootstrap(v, F.take_key.values[sel], n_boot=a.n_boot, stat="median")
            base[f"median_{c}"] = pt; base[f"median_{c}_lo"] = plo; base[f"median_{c}_hi"] = phi
            base[f"mean_{c}"] = float(np.nanmean(v)) if len(v) else np.nan
        assoc.append(base)
    pd.DataFrame(assoc).to_csv(out / "kinematic_associations.csv", index=False)
    # ---------------------------------------------------------------- Step 7: model error by event type
    err = []
    ecols = [c for c in F.columns if c.startswith("e_")]; acols = [c for c in F.columns if c.startswith("abs_")]
    for c, sel in list(frame_cats.items()) + [("all", np.ones(len(F), bool))]:
        row = dict(dataset=ds, frame_class=c, n_frames=int(sel.sum()), n_takes=int(F.take_key[sel].nunique()),
                   d_mean=float(F.d.values[sel].mean()) if sel.any() else np.nan, energy_share_all=float((F.d.values[sel] ** 2).sum() / tot_energy_all))
        for col in ecols + acols:
            pt, lo, hi = K.cluster_bootstrap(F[col].values[sel], F.take_key.values[sel], n_boot=a.n_boot)
            row[col], row[f"{col}_lo"], row[f"{col}_hi"] = pt, lo, hi
        # share of each model's total temporal error (sum of e_t) that falls in this class
        for col in ecols:
            row[f"{col}_share_of_total"] = float(F[col].values[sel].sum() / max(F[col].sum(), 1e-12))
        err.append(row)
    pd.DataFrame(err).to_csv(out / "model_error_by_event.csv", index=False)
    # ---------------------------------------------------------------- sanity summary
    san = dict(dataset=ds, n_test_sequences=int(len(te)), n_takes=int(meta_te.take_key.nunique()), n_transitions=int(len(F)),
               n_train_transitions=int(len(d_train)), q90=q90, q95=q95, decomposition_max_rel_residual=worst,
               n_valid_decomp=int(F.valid_decomp.sum()), frac_valid_decomp=float(F.valid_decomp.mean()),
               n_spike_frames_q90=int(F.spike90.sum()), frac_spike_frames_q90=float(F.spike90.mean()), n_events_q90=int(len(E)),
               n_events_q90_gap1=None, events_per_sequence=float(len(E) / len(te)), n_spike_frames_q95=int(F.spike95.sum()),
               spike_energy_share_of_all=spike_energy / tot_energy_all, spike_magnitude_share_of_all=spike_mag / tot_mag_all,
               n_onset_transitions=int(F.onset.sum()), n_release_transitions=int(F.release.sum()),
               frac_frames_firm=float(F.firm_t.mean()),
               smoothing_energy_ratio_s3=float((F.d_s3.values ** 2).sum() / tot_energy_all), smoothing_energy_ratio_s5=float((F.d_s5.values ** 2).sum() / tot_energy_all),
               smoothing_spike_energy_ratio_s3=float((F.d_s3.values[F.is_spike] ** 2).sum() / max(spike_energy, 1e-12)),
               smoothing_spike_energy_ratio_s5=float((F.d_s5.values[F.is_spike] ** 2).sum() / max(spike_energy, 1e-12)),
               persistence_h2_quantiles={q: float(np.nanquantile(E.persistence_h2, q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)} if len(E) else {},
               persistence_event_quantiles={q: float(np.nanquantile(E.persistence_event, q)) for q in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9)} if len(E) else {},
               category_counts=E.event_category.value_counts().to_dict(), kinematics_available=K.KIN_AVAILABLE[ds], transient_thr=a.transient_thr,
               models=list(K.MODELS) + (list(K.BIMART_MODELS) if B is not None else []))
    json.dump(san, open(out / "sanity.json", "w"), indent=2, default=float)
    pd.set_option("display.width", 250)
    log.info("%s: %d test seq, %d transitions, %d spike frames (%.1f%%), %d events (%.2f/seq), spike energy share %.2f, valid-decomp %.0f%%, onset %d release %d",
             ds, len(te), len(F), F.spike90.sum(), 100 * F.spike90.mean(), len(E), len(E) / len(te), spike_energy / tot_energy_all, 100 * F.valid_decomp.mean(), F.onset.sum(), F.release.sum())
    print(S[["category", "n_events", "frequency", "energy_share", "magnitude_share", "median_peak_d", "mean_duration", "median_persistence_h2", "median_persistence_event"]].round(3).to_string())
    print("persistence_h2 quantiles:", {k: round(v, 3) for k, v in san["persistence_h2_quantiles"].items()})
    print("persistence_event quantiles:", {k: round(v, 3) for k, v in san["persistence_event_quantiles"].items()})
    print("smoothing: energy ratio s3 %.3f s5 %.3f (all frames); on spike frames s3 %.3f s5 %.3f" % (san["smoothing_energy_ratio_s3"], san["smoothing_energy_ratio_s5"], san["smoothing_spike_energy_ratio_s3"], san["smoothing_spike_energy_ratio_s5"]))
    log.info("done in %.0fs -> %s", time.time() - t_start, out)


if __name__ == "__main__":
    main()
