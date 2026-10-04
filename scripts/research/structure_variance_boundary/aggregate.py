#!/usr/bin/env python
"""Aggregation of Experiments A–D into the deliverable tables:
  functional_probe.csv          per dataset x representation: q-reconstruction metrics (mean over seeds, take CI, seed sd) and
                                the ratio of the relative-L1 error sum to FULL's (paired take bootstrap)
  temporal_prediction.csv       Experiment B metrics (both datasets) + the ratios to FULL
  incremental_gain.csv          R_i -> R_j: functional and temporal error ratios (paired take bootstrap)
  event_representation_change.csv, event_summary tables (copied / merged from Experiment C)
  residual_prediction.csv       Experiment D: residual error ratios per block, model, horizon
  saturation_summary.csv        the saturation rule at 2.5 / 5 / 10 %, the smallest saturated representation per dataset
  representation_definitions.json
    python aggregate.py
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

import sv_common as S

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aggregate")
N_BOOT = 1000
PRIMARY = "mlpG"            # primary functional probe: representation + object descriptor G, validation-chosen regularisation
T1_PROXY = "per-frame mean |p(a_k) - a_k| over the 6 parts (a proper score of the participation probabilities)"


# ------------------------------------------------------------------------------ per-frame error tables
def functional_frames(ds, probe=PRIMARY, target="q"):
    """rep -> seed-averaged per-frame rel L1 (and the frame keys) on the test frames in contact."""
    out = {}; keys = None
    for rep in S.REPS + ["MEAN"]:
        files = sorted((S.ds_out(ds) / "functional").glob(f"probe_{rep}_{'const' if rep == 'MEAN' else probe}{'_strict' if target == 'q_strict' else ''}_seed*.npz"))
        if not files:
            continue
        arrs = [np.load(f, allow_pickle=True) for f in files]
        if keys is None:
            keys = (arrs[0]["take"].astype(str), arrs[0]["example"], arrs[0]["t"])
        out[rep] = {m: np.nanmean(np.stack([z[m] for z in arrs]), 0) for m in ("rel_l1", "cosine", "nrmse", "Q_err", "overlap", "abs_err")}
        out[rep]["_n_seeds"] = len(files)
    return out, keys


def temporal_frames(ds, head="structured"):
    """rep -> seed-averaged per-frame errors (N_pairs, 3) for the composite and the paired ratios."""
    out = {}; keys = None
    for rep in S.REPS:
        files = sorted((S.ds_out(ds) / "temporal" / "metrics").glob(f"{rep}_{head}_seed*.npz"))
        if not files:
            continue
        arrs = [np.load(f, allow_pickle=True) for f in files]
        z0 = arrs[0]
        if keys is None:
            keys = (z0["take"].astype(str), z0["example"], z0["t"], z0["valid"] > 0)
        e = {}
        if head == "structured":
            e["T1_l1"] = np.nanmean(np.stack([np.abs(z["a_prob"].astype(np.float32) - z["a_true"].astype(np.float32)).mean(2) for z in arrs]), 0)
            e["T1_l1_persist"] = np.abs(z0["a_now"].astype(np.float32) - z0["a_true"].astype(np.float32)).mean(2)
            for m in ("T2_rel_l1", "T3_cent", "T3_ang", "T4_rel_l1", "T4_Qerr", "mse_m", "mse_geom", "mse_q"):
                e[m] = np.nanmean(np.stack([z[m] for z in arrs]), 0)
            for m in ("T2_rel_l1_persist", "T3_cent_persist", "T3_ang_persist", "T4_rel_l1_persist", "mse_m_persist", "mse_geom_persist", "mse_q_persist"):
                e[m] = z0[m]
        else:
            for m in ("E_C", "E_H", "mse_C", "mse_H"):
                e[m] = np.nanmean(np.stack([z[m] for z in arrs]), 0)
            for m in ("E0_C", "E0_H", "mse_C_persist", "mse_H_persist"):
                e[m] = z0[m]
        e["_n_seeds"] = len(files)
        out[rep] = e
    return out, keys


def take_sums(vals, take, valid=None):
    u, inv = np.unique(take, return_inverse=True)
    ok = np.isfinite(vals) if valid is None else (np.isfinite(vals) & valid)
    return np.bincount(inv[ok], weights=vals[ok], minlength=len(u)), np.bincount(inv[ok], minlength=len(u)).astype(float), u


def boot_weights(n_takes, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(0, n_takes, n_takes), minlength=n_takes).astype(float) for _ in range(n_boot)]


def ratio_reps(sa, ca, sb, cb, W):
    """bootstrap replicates of (mean_a / mean_b) with shared take weights; index 0 = the point estimate."""
    pt = (sa.sum() / ca.sum()) / (sb.sum() / cb.sum())
    reps = np.array([((w * sa).sum() / (w * ca).sum()) / ((w * sb).sum() / (w * cb).sum()) for w in W])
    return np.concatenate([[pt], reps])


# ------------------------------------------------------------------------------ Experiment A
def aggregate_functional():
    rows, ratio_tab = [], {}
    for ds in S.DATASETS:
        raw = pd.read_csv(S.ds_out(ds) / "functional" / "functional_probe.csv")
        for (rep, probe, target), g in raw.groupby(["rep", "probe", "target"]):
            r = dict(dataset=ds, rep=rep, probe=probe, target=target, dim=int(g.dim.iloc[0]), n_seeds=len(g), n_frames=int(g.n_frames.iloc[0]))
            for m in ("rel_l1", "cosine", "nrmse", "Q_err", "overlap", "r2", "abs_err"):
                r[m] = float(g[m].mean()); r[f"{m}_seed_sd"] = float(g[m].std(ddof=0))
            rows.append(r)
        for target in ("q", "q_strict"):
            fr, keys = functional_frames(ds, PRIMARY, target)
            if not fr or "FULL" not in fr:
                continue
            take = keys[0]; W = boot_weights(len(np.unique(take)))
            sums = {rep: {m: take_sums(fr[rep][m], take)[:2] for m in ("rel_l1", "abs_err", "Q_err")} for rep in fr}
            for rep in fr:
                row = dict(dataset=ds, rep=rep, probe=f"{PRIMARY}_seedavg", target=target, dim=S.REP_DIM.get(rep, 0), n_seeds=fr[rep]["_n_seeds"], n_frames=len(take))
                for m in ("rel_l1", "cosine", "nrmse", "Q_err", "overlap"):
                    pt, lo, hi = S.cluster_bootstrap(fr[rep][m], take, n_boot=N_BOOT)
                    row.update({m: pt, f"{m}_lo": lo, f"{m}_hi": hi})
                rows.append(row)
                for m in ("rel_l1", "abs_err", "Q_err"):
                    rr = ratio_reps(*sums[rep][m], *sums["FULL"][m], W)
                    ratio_tab[(ds, target, rep, m)] = rr
                    rows.append(dict(dataset=ds, rep=rep, probe="ratio_to_FULL", target=target, dim=S.REP_DIM.get(rep, 0), metric=m, ratio=float(rr[0]), ratio_lo=float(np.percentile(rr[1:], 2.5)), ratio_hi=float(np.percentile(rr[1:], 97.5))))
    df = pd.DataFrame(rows); df.to_csv(S.OUT / "functional_probe.csv", index=False)
    return df, ratio_tab


# ------------------------------------------------------------------------------ Experiment B
def composite_reps(e_a, e_b, take, valid, W, j):
    """bootstrap replicates of the composite T1–T4 error ratio a / b at horizon index j (index 0 = point)."""
    parts = []
    for m in ("T1_l1", "T2_rel_l1", "T4_rel_l1"):
        sa, ca, _ = take_sums(e_a[m][:, j], take, valid[:, j]); sb, cb, _ = take_sums(e_b[m][:, j], take, valid[:, j])
        parts.append(ratio_reps(sa, ca, sb, cb, W))
    t3 = []
    for m in ("T3_cent", "T3_ang"):
        sa, ca, _ = take_sums(e_a[m][:, j], take, valid[:, j]); sb, cb, _ = take_sums(e_b[m][:, j], take, valid[:, j])
        t3.append(ratio_reps(sa, ca, sb, cb, W))
    parts.append(0.5 * (t3[0] + t3[1]))
    return np.mean(parts, 0), dict(T1=parts[0], T2=parts[1], T4=parts[2], T3=parts[3])


def aggregate_temporal():
    frames_all = {}
    tabs = []
    for ds in S.DATASETS:
        p = S.ds_out(ds) / "temporal" / "temporal_prediction.csv"
        if p.exists():
            tabs.append(pd.read_csv(p))
        fr, keys = temporal_frames(ds, "structured")
        frd, keysd = temporal_frames(ds, "dense")
        frames_all[ds] = (fr, keys, frd, keysd)
    df = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    ratio_rows = []
    comp_tab = {}
    for ds, (fr, keys, frd, keysd) in frames_all.items():
        if not fr or "FULL" not in fr:
            continue
        take, ex, tt, valid = keys; W = boot_weights(len(np.unique(take)), seed=1)
        for rep in fr:
            if rep == "FULL":
                continue
            comps = []
            for j, h in enumerate(S.HORIZONS):
                c, parts = composite_reps(fr[rep], fr["FULL"], take, valid, W, j)
                comps.append(c); comp_tab[(ds, rep, h)] = c
                ratio_rows.append(dict(dataset=ds, rep=rep, vs="FULL", metric="composite_T1_T4", h=h, ratio=float(c[0]), lo=float(np.percentile(c[1:], 2.5)), hi=float(np.percentile(c[1:], 97.5))))
                for tg, rr in parts.items():
                    ratio_rows.append(dict(dataset=ds, rep=rep, vs="FULL", metric=f"{tg}_ratio", h=h, ratio=float(rr[0]), lo=float(np.percentile(rr[1:], 2.5)), hi=float(np.percentile(rr[1:], 97.5))))
            c = np.mean(comps, 0); comp_tab[(ds, rep, 0)] = c
            ratio_rows.append(dict(dataset=ds, rep=rep, vs="FULL", metric="composite_T1_T4", h=0, ratio=float(c[0]), lo=float(np.percentile(c[1:], 2.5)), hi=float(np.percentile(c[1:], 97.5))))
        # dense diagnostic ratios to FULL (E_C, E_H) and the T1 proxy / persistence reference rows
        if frd and "FULL" in frd:
            taked, _, _, validd = keysd
            for rep in frd:
                for m in ("E_C", "E_H"):
                    for j, h in enumerate(S.HORIZONS):
                        sa, ca, _ = take_sums(frd[rep][m][:, j], taked, validd[:, j]); sb, cb, _ = take_sums(frd["FULL"][m][:, j], taked, validd[:, j])
                        rr = ratio_reps(sa, ca, sb, cb, W)
                        ratio_rows.append(dict(dataset=ds, rep=rep, vs="FULL", metric=f"{m}_ratio", h=h, ratio=float(rr[0]), lo=float(np.percentile(rr[1:], 2.5)), hi=float(np.percentile(rr[1:], 97.5))))
                        s0, c0, _ = take_sums(frd[rep]["E0_" + m[-1]][:, j], taked, validd[:, j])
                        rr = ratio_reps(sa, ca, s0, c0, W)
                        ratio_rows.append(dict(dataset=ds, rep=rep, vs="PERSIST", metric=f"{m}_ratio", h=h, ratio=float(rr[0]), lo=float(np.percentile(rr[1:], 2.5)), hi=float(np.percentile(rr[1:], 97.5))))
        for rep in fr:
            for j, h in enumerate(S.HORIZONS):
                pt, lo, hi = S.cluster_bootstrap(fr[rep]["T1_l1"][:, j][valid[:, j]], take[valid[:, j]], n_boot=N_BOOT)
                ratio_rows.append(dict(dataset=ds, rep=rep, vs="", metric="T1_l1", h=h, ratio=pt, lo=lo, hi=hi))
                for m in ("T2_rel_l1", "T3_cent", "T3_ang", "T4_rel_l1", "T1_l1"):
                    sa, ca, _ = take_sums(fr[rep][m][:, j], take, valid[:, j]); s0, c0, _ = take_sums(fr[rep][m + "_persist"][:, j], take, valid[:, j])
                    rr = ratio_reps(sa, ca, s0, c0, W)
                    ratio_rows.append(dict(dataset=ds, rep=rep, vs="PERSIST", metric=f"{m}_ratio", h=h, ratio=float(rr[0]), lo=float(np.percentile(rr[1:], 2.5)), hi=float(np.percentile(rr[1:], 97.5))))
    ratios = pd.DataFrame(ratio_rows)
    if len(df):
        df.to_csv(S.OUT / "temporal_prediction.csv", index=False)
    ratios.to_csv(S.OUT / "temporal_ratios.csv", index=False)
    return df, ratios, frames_all, comp_tab


# ------------------------------------------------------------------------------ incremental gains
def incremental(frames_all):
    rows = []
    for ds, (fr, keys, frd, keysd) in frames_all.items():
        if not fr:
            continue
        take, ex, tt, valid = keys; W = boot_weights(len(np.unique(take)), seed=2)
        fun, fkeys = functional_frames(ds, PRIMARY, "q")
        Wf = boot_weights(len(np.unique(fkeys[0])), seed=3) if fun else None
        for a, b, block in S.INCREMENTS:
            r = dict(dataset=ds, frm=a, to=b, added=block, dim_from=S.REP_DIM[a], dim_to=S.REP_DIM[b])
            if fun and a in fun and b in fun:
                sa, ca, _ = take_sums(fun[a]["rel_l1"], fkeys[0]); sb, cb, _ = take_sums(fun[b]["rel_l1"], fkeys[0])
                rr = ratio_reps(sb, cb, sa, ca, Wf)                     # to / from: < 1 = improvement
                r.update(functional_ratio=float(rr[0]), functional_lo=float(np.percentile(rr[1:], 2.5)), functional_hi=float(np.percentile(rr[1:], 97.5)),
                         functional_rel_l1_from=float(sa.sum() / ca.sum()), functional_rel_l1_to=float(sb.sum() / cb.sum()))
            if a in fr and b in fr:
                for j, h in enumerate(S.HORIZONS):
                    c, parts = composite_reps(fr[b], fr[a], take, valid, W, j)
                    r[f"temporal_ratio_h{h}"] = float(c[0]); r[f"temporal_lo_h{h}"] = float(np.percentile(c[1:], 2.5)); r[f"temporal_hi_h{h}"] = float(np.percentile(c[1:], 97.5))
                    for tg, rr in parts.items():
                        r[f"{tg}_ratio_h{h}"] = float(rr[0])
                cs = np.mean([composite_reps(fr[b], fr[a], take, valid, W, j)[0] for j in range(len(S.HORIZONS))], 0)
                r.update(temporal_ratio=float(cs[0]), temporal_lo=float(np.percentile(cs[1:], 2.5)), temporal_hi=float(np.percentile(cs[1:], 97.5)))
            if frd and a in frd and b in frd:
                taked, _, _, validd = keysd
                for m in ("E_C", "E_H"):
                    vals = []
                    for j in range(len(S.HORIZONS)):
                        sa, ca, _ = take_sums(frd[a][m][:, j], taked, validd[:, j]); sb, cb, _ = take_sums(frd[b][m][:, j], taked, validd[:, j])
                        vals.append(ratio_reps(sb, cb, sa, ca, W))
                    cs = np.mean(vals, 0)
                    r[f"dense_{m}_ratio"] = float(cs[0]); r[f"dense_{m}_lo"] = float(np.percentile(cs[1:], 2.5)); r[f"dense_{m}_hi"] = float(np.percentile(cs[1:], 97.5))
            rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(S.OUT / "incremental_gain.csv", index=False)
    return df


# ------------------------------------------------------------------------------ Experiment C (merge)
def aggregate_events():
    tabs = {}
    for name in ("event_representation_change", "event_correlations", "event_probes", "event_quadrants", "event_by_finger_kind"):
        parts = [pd.read_csv(S.ds_out(ds) / "events" / f"{name}.csv") for ds in S.DATASETS if (S.ds_out(ds) / "events" / f"{name}.csv").exists()]
        if parts:
            tabs[name] = pd.concat(parts, ignore_index=True); tabs[name].to_csv(S.OUT / f"{name}.csv", index=False)
    return tabs


# ------------------------------------------------------------------------------ Experiment D
def aggregate_residual():
    rows = []
    for ds in S.DATASETS:
        d = S.ds_out(ds) / "residual"
        if not d.exists():
            continue
        infos = [json.load(open(p)) for p in sorted(d.glob("decoder_*_seed*.json")) if "_smoke" not in p.stem]
        for zstar in sorted({i["zstar"] for i in infos}):
            zi = [i for i in infos if i["zstar"] == zstar]
            for blk in ("C", "H"):
                rows.append(dict(dataset=ds, zstar=zstar, model="decoder", block=blk, h=0, metric="decoder_r2_test", value=float(np.mean([i["decoder_r2_test"][blk] for i in zi])),
                                 lo=np.nan, hi=np.nan, seed_sd=float(np.std([i["decoder_r2_test"][blk] for i in zi])), n_seeds=len(zi)))
            files = {}
            for p in sorted(d.glob(f"metrics_{zstar}_*_seed*.npz")):
                if "_smoke" in p.stem:
                    continue
                model = p.stem[len(f"metrics_{zstar}_"):].rsplit("_seed", 1)[0]
                files.setdefault(model, []).append(np.load(p, allow_pickle=True))
            if "zero" not in files:
                continue
            z0 = files["zero"][0]; take = z0["take"].astype(str); valid = z0["valid"] > 0; W = boot_weights(len(np.unique(take)), seed=4)
            files["last"] = [{"mse_C": z0["last_C"], "mse_H": z0["last_H"]}]        # the last-residual baseline as a model
            for model, arrs in files.items():
                for blk in ("C", "H"):
                    mse = np.nanmean(np.stack([z[f"mse_{blk}"] for z in arrs]), 0)
                    for base, key in (("zero", f"zero_{blk}"), ("last", f"last_{blk}")):
                        for j, h in enumerate(S.HORIZONS):
                            sa, ca, _ = take_sums(mse[:, j], take, valid[:, j]); sb, cb, _ = take_sums(z0[key][:, j], take, valid[:, j])
                            rr = ratio_reps(sa, ca, sb, cb, W)
                            per_seed = [float(np.nansum(z[f"mse_{blk}"][:, j][valid[:, j]]) / np.nansum(z0[key][:, j][valid[:, j]])) for z in arrs]
                            rows.append(dict(dataset=ds, zstar=zstar, model=model, block=blk, h=h, metric=f"mse_ratio_to_{base}", value=float(rr[0]), lo=float(np.percentile(rr[1:], 2.5)),
                                             hi=float(np.percentile(rr[1:], 97.5)), seed_sd=float(np.std(per_seed)), n_seeds=len(arrs)))
                    if model == "zero":
                        for j, h in enumerate(S.HORIZONS):
                            for key, nm in ((f"zero_{blk}", "residual_energy"), (f"last_{blk}", "last_residual_mse")):
                                pt, lo, hi = S.cluster_bootstrap(z0[key][:, j][valid[:, j]], take[valid[:, j]], n_boot=N_BOOT)
                                rows.append(dict(dataset=ds, zstar=zstar, model="zero", block=blk, h=h, metric=nm, value=pt, lo=lo, hi=hi, seed_sd=0.0, n_seeds=1))
    df = pd.DataFrame(rows); df.to_csv(S.OUT / "residual_prediction.csv", index=False)
    return df


# ------------------------------------------------------------------------------ saturation
def saturation(fun_ratio, comp_tab):
    """The rule of the request (gap to FULL inside the bootstrap interval OR below 2.5 / 5 / 10 %), plus the same rule against the
    BEST representation of each axis (the one with the lowest point estimate), for the case where FULL is not the best."""
    rows = []
    for ds in S.DATASETS:
        arrs = {}
        for rep in S.REPS:
            fa = fun_ratio.get((ds, "q", rep, "rel_l1")); cb = comp_tab.get((ds, rep, 0))
            if rep == "FULL":
                fa = np.ones(N_BOOT + 1); cb = np.ones(N_BOOT + 1)
            if fa is not None and cb is not None:
                arrs[rep] = (np.asarray(fa, float), np.asarray(cb, float))
        if not arrs:
            continue
        best_f = min(arrs, key=lambda r: arrs[r][0][0]); best_t = min(arrs, key=lambda r: arrs[r][1][0])
        for rep, (fa, cb) in arrs.items():
            r = dict(dataset=ds, rep=rep, dim=S.REP_DIM[rep], best_functional=best_f, best_temporal=best_t)
            for kind, a, best in (("functional", fa, arrs[best_f][0]), ("temporal", cb, arrs[best_t][1])):
                g = a - 1; gb = a / best - 1
                r.update({f"{kind}_gap": float(g[0]), f"{kind}_gap_lo": float(np.percentile(g[1:], 2.5)), f"{kind}_gap_hi": float(np.percentile(g[1:], 97.5)),
                          f"{kind}_gap_best": float(gb[0]), f"{kind}_gap_best_lo": float(np.percentile(gb[1:], 2.5)), f"{kind}_gap_best_hi": float(np.percentile(gb[1:], 97.5))})
            for thr in S.SATURATION_THRESHOLDS:
                for suf in ("", "_best"):
                    fs = (r[f"functional_gap{suf}"] < thr) or (r[f"functional_gap{suf}_lo"] <= 0)
                    ts = (r[f"temporal_gap{suf}"] < thr) or (r[f"temporal_gap{suf}_lo"] <= 0)
                    r[f"functional_saturated_{thr}{suf}"] = bool(fs); r[f"temporal_saturated_{thr}{suf}"] = bool(ts); r[f"saturated_{thr}{suf}"] = bool(fs and ts)
            rows.append(r)
    df = pd.DataFrame(rows)
    choice = []
    for ds in S.DATASETS:
        d = df[df.dataset == ds].sort_values("dim")
        for thr in S.SATURATION_THRESHOLDS:
            for suf in ("", "_best"):
                for kind in ("functional", "temporal", ""):
                    col = f"{kind}_saturated_{thr}{suf}" if kind else f"saturated_{thr}{suf}"
                    sat = d[d[col]]
                    choice.append(dict(dataset=ds, threshold=thr, reference="FULL" if not suf else "best", criterion=kind or "both",
                                       smallest_saturated=sat.rep.iloc[0] if len(sat) else "none", dim=int(sat.dim.iloc[0]) if len(sat) else np.nan))
    df.to_csv(S.OUT / "saturation_summary.csv", index=False)
    pd.DataFrame(choice).to_csv(S.OUT / "saturation_choice.csv", index=False)
    return df, pd.DataFrame(choice)


def main():
    S.OUT.mkdir(parents=True, exist_ok=True)
    defs = S.representation_definitions()
    S.write_json(S.OUT / "representation_definitions.json", dict(blocks=S.BLOCKS, block_labels=S.BLOCK_LABEL, ladder=S.LADDER, dims=S.REP_DIM, window_frames=S.HIST,
                                                                 participation="n_hard >= %d vertices within 1 cm, causal 3-frame majority" % S.N_MIN_VERTS,
                                                                 amount="area-weighted soft mass sum exp(-d/0.02) A_v / A_total over the part's vertices (<= 2 cm)",
                                                                 geometry="centroid / l and mean toward-hand normal of the hard-contact vertices; RMS spread / l; patch count (1 cm / merge 1 cm)",
                                                                 hand_coarse="wrist / l, palm frame (toward fingers, toward thumb side, dorsal), fingertips / l, MCP->tip directions",
                                                                 T1_proxy_for_paired_gaps=T1_PROXY, table=defs.to_dict("records")))
    fun, fun_ratio = aggregate_functional(); log.info("functional: %d rows", len(fun))
    tdf, ratios, frames_all, comp_tab = aggregate_temporal(); log.info("temporal: %d rows, %d ratio rows", len(tdf), len(ratios))
    inc = incremental(frames_all); log.info("incremental: %d rows", len(inc))
    ev = aggregate_events(); log.info("events: %s", {k: len(v) for k, v in ev.items()})
    res = aggregate_residual(); log.info("residual: %d rows", len(res))
    sat, choice = saturation(fun_ratio, comp_tab); log.info("saturation:\n%s", choice.to_string())


if __name__ == "__main__":
    main()
