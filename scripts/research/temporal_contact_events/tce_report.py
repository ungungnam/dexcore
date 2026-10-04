#!/usr/bin/env python
"""Unified tables + the report skeleton:  python tce_report.py
Reads OUT/<ds>/{sanity.json, events.csv, event_summary.csv, model_error_by_event.csv, kinematic_associations.csv,
threshold_sensitivity.csv, frames_test.csv} for the three datasets and writes OUT/{summary_table.csv,
event_type_table.csv, threshold_sensitivity_all.csv, temporal_contact_event_report.md}. Narrative sections come
from OUT/narrative/*.md when present.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import tce_common as K

DS = ["taco", "arctic", "oakink2"]
CATS = ["onset", "release", "onset+release", "amount", "spatial", "mixed", "near_zero"]
MODEL_COLS = [("e_B0_static", "B0 static"), ("e_B1_gtinit_vf", "B1 GT-init + VF"), ("e_B2_samplerG_vf_K10", "B2 p(S0|G)+VF best-of-10"),
              ("e_B2_samplerG_vf_K1", "B2 K=1"), ("e_B3_samplerGT_vf_K10", "B3 best-of-10"), ("e_B4_bimart_K10_own", "B4 BimArt best-of-10 (own support)"),
              ("e_B4_bimart_K10_full", "B4 BimArt best-of-10 (vs full GT)")]


def narr(name):
    p = K.OUT / "narrative" / f"{name}.md"
    return p.read_text().strip() + "\n" if p.exists() else f"_(narrative {name} not written)_\n"


def load(ds):
    o = K.OUT / ds
    return dict(san=json.load(open(o / "sanity.json")), E=pd.read_csv(o / "events.csv"), S=pd.read_csv(o / "event_summary.csv"),
                M=pd.read_csv(o / "model_error_by_event.csv"), A=pd.read_csv(o / "kinematic_associations.csv"), TS=pd.read_csv(o / "threshold_sensitivity.csv"))


def summary_table(D):
    rows = []
    for ds, x in D.items():
        s, S = x["san"], x["S"].set_index("category")
        f = lambda c, col: float(S.loc[c, col]) if c in S.index else 0.0
        onoff_f = sum(f(c, "frequency") for c in ("onset", "release", "onset+release")); onoff_e = sum(f(c, "energy_share") for c in ("onset", "release", "onset+release"))
        rows.append({"dataset": K.LABEL[ds], "test sequences": s["n_test_sequences"], "takes/recordings": s["n_takes"], "test transitions": s["n_transitions"],
                     "valid-decomposition transitions (%)": round(100 * s["frac_valid_decomp"], 1), "train Q90 of d_t": round(s["q90"], 3), "train Q95": round(s["q95"], 3),
                     "spike frames (% of test transitions)": round(100 * s["frac_spike_frames_q90"], 1), "merged spike events": s["n_events_q90"],
                     "events per sequence": round(s["events_per_sequence"], 2), "spike share of Σd_t² (all frames)": round(s["spike_energy_share_of_all"], 3),
                     "spike share of Σd_t": round(s["spike_magnitude_share_of_all"], 3),
                     "onset/release: freq / energy": f"{onoff_f:.2f} / {onoff_e:.2f}", "amount: freq / energy": f"{f('amount', 'frequency'):.2f} / {f('amount', 'energy_share'):.2f}",
                     "spatial: freq / energy": f"{f('spatial', 'frequency'):.2f} / {f('spatial', 'energy_share'):.2f}", "mixed: freq / energy": f"{f('mixed', 'frequency'):.2f} / {f('mixed', 'energy_share'):.2f}",
                     "low-persistence events (h4 < 0.5): freq / energy": f"{f('transient', 'frequency'):.2f} / {f('transient', 'energy_share'):.2f}",
                     "median persistence h2 / h4 (spike events)": f"{s['persistence_h2_quantiles']['0.5']:.2f} / {float(x['E'].persistence_h4.median()):.2f}",
                     "Σd_t² kept after 3-frame smoothing (all / spike frames)": f"{s['smoothing_energy_ratio_s3']:.2f} / {s['smoothing_spike_energy_ratio_s3']:.2f}",
                     "Σd_t² kept after 5-frame smoothing (all / spike frames)": f"{s['smoothing_energy_ratio_s5']:.2f} / {s['smoothing_spike_energy_ratio_s5']:.2f}"})
    return pd.DataFrame(rows)


def event_type_table(D):
    rows = []
    for ds, x in D.items():
        S = x["S"].set_index("category"); M = x["M"].set_index("frame_class"); A = x["A"].set_index("signal"); E = x["E"]
        for c in CATS + ["transient"]:
            if c not in S.index or S.loc[c, "n_events"] == 0:
                continue
            r = {"dataset": K.LABEL[ds], "event type": c if c != "transient" else "low-persistence (h4 < 0.5)", "events": int(S.loc[c, "n_events"]), "frequency": round(S.loc[c, "frequency"], 3),
                 "energy share": round(S.loc[c, "energy_share"], 3), "mean / median peak d_t": f"{S.loc[c, 'mean_peak_d']:.2f} / {S.loc[c, 'median_peak_d']:.2f}",
                 "mean duration (frames)": round(S.loc[c, "mean_duration"], 2), "mean amount comp. A": round(S.loc[c, "mean_amount"], 2), "mean spatial comp. S": round(S.loc[c, "mean_spatial"], 2),
                 "median persistence h2 / h4": f"{S.loc[c, 'median_persistence_h2']:.2f} / {S.loc[c, 'median_persistence_h4']:.2f}",
                 "low-persistence fraction": round(S.loc[c, "low_persistence_frac_defined"], 2) if "low_persistence_frac_defined" in S.columns else np.nan}
            for sig in K.KIN_AVAILABLE[ds]:
                if sig in A.index and f"median_{c}" in A.columns:
                    r[f"median {sig} (event / non-spike)"] = f"{A.loc[sig, f'median_{c}']:.4f} / {A.loc[sig, 'median_non_spike']:.4f}"
            mc = c
            if mc in M.index:
                for col, lab in MODEL_COLS:
                    if col in M.columns:
                        r[f"{lab} e_t"] = f"{M.loc[mc, col]:.3f} [{M.loc[mc, col + '_lo']:.3f}, {M.loc[mc, col + '_hi']:.3f}]"
            rows.append(r)
        for c in ("non_spike", "spike", "all"):
            r = {"dataset": K.LABEL[ds], "event type": c, "events": "", "frequency": "", "energy share": round(M.loc[c, "energy_share_all"], 3)}
            for col, lab in MODEL_COLS:
                if col in M.columns:
                    r[f"{lab} e_t"] = f"{M.loc[c, col]:.3f} [{M.loc[c, col + '_lo']:.3f}, {M.loc[c, col + '_hi']:.3f}]"
            rows.append(r)
    return pd.DataFrame(rows)


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c).replace("|", "\\|") for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(out)


def error_share_table(D):
    rows = []
    for ds, x in D.items():
        M = x["M"].set_index("frame_class")
        for col, lab in MODEL_COLS:
            if col not in M.columns:
                continue
            r = {"dataset": K.LABEL[ds], "model": lab}
            for c in ["non_spike", "onset", "release", "amount", "spatial", "mixed", "transient"]:
                if c in M.index:
                    r[f"{'low-persistence (h4 < 0.5)' if c == 'transient' else c} share of Σe_t"] = round(M.loc[c, f"{col}_share_of_total"], 3)
            r["mean e_t (all)"] = round(M.loc["all", col], 3); r["mean e_t non-spike"] = round(M.loc["non_spike", col], 3); r["mean e_t spike"] = round(M.loc["spike", col], 3)
            rows.append(r)
    return pd.DataFrame(rows)


def kin_table(D):
    rows = []
    for ds, x in D.items():
        A = x["A"]
        for r in A.itertuples():
            row = {"dataset": K.LABEL[ds], "signal": r.signal, "Spearman(d_t, signal) [95 % CI]": f"{r.spearman_d:+.3f} [{r.spearman_lo:+.3f}, {r.spearman_hi:+.3f}]",
                   "Cliff's δ spike vs non-spike (≤4000-frame subsample)": round(r.cliffs_delta_spike_vs_non, 2), "median non-spike": f"{r.median_non_spike:.4f}", "median spike": f"{r.median_spike:.4f}"}
            for c in ("onset", "release", "amount", "spatial", "mixed", "transient"):
                if f"median_{c}" in A.columns and not np.isnan(getattr(r, f"median_{c}")):
                    row[f"median {'low-persistence (h4 < 0.5)' if c == 'transient' else c}"] = f"{getattr(r, f'median_{c}'):.4f}"
            rows.append(row)
    return pd.DataFrame(rows)


def sens_table(D):
    rows = []
    for ds, x in D.items():
        TS = x["TS"]
        for (q, rt), g in TS.groupby(["spike_q", "r_thresholds"]):
            g = g.set_index("category")
            f = lambda c, col: float(g.loc[c, col]) if c in g.index else 0.0
            rows.append({"dataset": K.LABEL[ds], "spike threshold": q, "r thresholds (amount ≥ / spatial ≤)": rt, "events": int(g.n_events.sum()),
                         "on/off freq": round(f("onset", "frequency") + f("release", "frequency") + f("onset+release", "frequency"), 2),
                         "amount freq": round(f("amount", "frequency"), 2), "spatial freq": round(f("spatial", "frequency"), 2), "mixed freq": round(f("mixed", "frequency"), 2),
                         "on/off energy": round(f("onset", "energy_share") + f("release", "energy_share") + f("onset+release", "energy_share"), 2),
                         "amount energy": round(f("amount", "energy_share"), 2), "spatial energy": round(f("spatial", "energy_share"), 2), "mixed energy": round(f("mixed", "energy_share"), 2)})
    return pd.DataFrame(rows)


def main():
    D = {ds: load(ds) for ds in DS if (K.OUT / ds / "sanity.json").exists()}
    T1 = summary_table(D); T1.to_csv(K.OUT / "summary_table.csv", index=False)
    T2 = event_type_table(D); T2.to_csv(K.OUT / "event_type_table.csv", index=False)
    T3 = error_share_table(D); T3.to_csv(K.OUT / "model_error_share_table.csv", index=False)
    T4 = kin_table(D); T4.to_csv(K.OUT / "kinematic_associations_all.csv", index=False)
    T5 = sens_table(D); T5.to_csv(K.OUT / "threshold_sensitivity_all.csv", index=False)
    L = ["# Temporal contact-change events: what constitutes a large temporal contact change, and which of it the models miss", "",
         "Analysis of the existing hierarchical contact-generation experiments (fixed split, cached sequences and predictions; no retraining). "
         "Scripts: `scripts/research/temporal_contact_events/`; outputs: `/result/uhnam/dexcore/reports/temporal_contact_events/<dataset>/`.", "",
         "## Conclusions", "", narr("conclusions"),
         "## 1. Motivation and definitions", "", narr("definitions"),
         "## 2. Protocol and sanity checks", "", narr("protocol"),
         "## 3. Unified results", "", "### Table 1 — datasets side by side (`summary_table.csv`)", "", md_table(T1.set_index("dataset").T.reset_index().rename(columns={"index": "quantity"})), "",
         "### Table 2 — dataset × event type (`event_type_table.csv`)", "", md_table(T2), "",
         "### Table 3 — where each model's temporal error sits (`model_error_share_table.csv`)", "", md_table(T3), "",
         "### Table 4 — kinematic associations (`kinematic_associations_all.csv`; Spearman with take-cluster bootstrap CI; medians per frame class)", "", md_table(T4), "",
         "### Table 5 — threshold sensitivity (`threshold_sensitivity_all.csv`)", "", md_table(T5), "",
         "## 4. Dataset comparison", "", narr("comparison"),
         "## 5. Model-error decomposition", "", narr("model_errors"),
         "## 6. Representative qualitative events", "", narr("qualitative"),
         "## 7. Limitations", "", narr("limitations"),
         "## 8. Implication for the next temporal model", "", narr("implication"),
         "## 9. Files", "", narr("files")]
    (K.OUT / "temporal_contact_event_report.md").write_text("\n".join(L))
    print("written", K.OUT / "temporal_contact_event_report.md")


if __name__ == "__main__":
    main()
