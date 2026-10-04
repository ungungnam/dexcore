#!/usr/bin/env python
"""Step 4: per-frame metrics, aggregate tables and bootstrap statistics for one dataset.
    python evaluate.py --dataset taco [--n-boot 1000]

Delta-C regression (raw contact units; the seed-mean of every per-frame metric over the 3 seeds):
  E_delta(h)   = ||Delta^_h C_t - Delta_h C_t||_2                       h in (1, 4, 8)
  E0(h)        = ||Delta_h C_t||_2                                       the zero-change predictor
  improvement  = 1 - sum E / sum E0                                      (take-cluster bootstrap CI)
  cosine       = <Delta^, Delta> / (|Delta^| |Delta|)  on frames with ||Delta_h C_t|| above the TRAIN
                 median of ||Delta_h|| (sufficiently non-zero change)
  e_A, e_S     amount / spatial-pattern error where m_t, m_{t+h} >= zero threshold (and the predicted
                 mass >= threshold / 2): e_A = |Delta m^ - Delta m| * ||P-bar||,  e_S = m-bar * ||P^_{t+h} - P_{t+h}||
                 with P = C / m, P-bar / m-bar the GT means of the two frames (the decomposition of the
                 temporal-event analysis: Delta C = Delta m * P-bar + m-bar * Delta P)
Frame classes per horizon (hp_common.frame_classes): from the transitions inside Delta_h.
Event prediction: AUPRC / AUROC of the seed-mean over seeds (CI: take bootstrap recomputing the
seed-mean score), per positive subtype, recall at precision >= 0.5.
Writes OUT/<ds>/{delta_contact_predictions.csv, transition_predictions.csv} and
OUT/<ds>/results/{overall_delta_results, event_conditioned_results, transition_prediction_results,
incremental_information, history_length_ablation, seed_variation, f0_vs_existing_vf}.csv/json
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import hp_common as P

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("evaluate")
PAIRS = [("F1", "F0"), ("F2k8", "F1"), ("F2k8", "F0"), ("F2k4", "F1"), ("F2k1", "F1"), ("F2rel", "F1"), ("F2rel", "F0"), ("F2rel", "F2k8"),
         ("F2shuf", "F0"), ("F2k8", "F2shuf"), ("F2rel", "F2shuf"), ("Ffut", "F2k8"), ("Ffut", "F2rel"), ("Ffut", "F0"), ("Ffut1", "F2rel"), ("Ffut", "Ffut1")]
ABL_CLASSES = ["all", "non_spike", "spike", "persistent_spatial", "persistent_mixed", "release", "onset", "transient"]


def per_frame_metrics(C, pred, thr, cos_thr):
    """C (n,64,512) raw GT, pred (n,63,3,512) raw Delta^ -> dict of (n,63,3) arrays (NaN where invalid)."""
    n = len(C); nh = len(P.HORIZONS)
    E = np.full((n, P.T - 1, nh), np.nan, np.float32); E0 = E.copy(); cos = E.copy(); eA = E.copy(); eS = E.copy()
    m = C.sum(2)
    for j, h in enumerate(P.HORIZONS):
        tv = P.T - 1 - h + 1                                              # valid t: 0 .. 63-h
        Ct = C[:, :tv]; Ch = C[:, h:h + tv]; D = Ch - Ct; Dp = pred[:, :tv, j]
        E[:, :tv, j] = np.linalg.norm(Dp - D, axis=2); E0[:, :tv, j] = np.linalg.norm(D, axis=2)
        num = (Dp * D).sum(2); den = np.linalg.norm(Dp, axis=2) * E0[:, :tv, j] + 1e-9
        c = num / den; c[E0[:, :tv, j] <= cos_thr[j]] = np.nan; cos[:, :tv, j] = c
        mt, mh = m[:, :tv], m[:, h:h + tv]; Chat = Ct + Dp; mhat = Chat.sum(2)
        ok = (mt >= thr) & (mh >= thr) & (mhat >= 0.5 * thr)
        Pt = Ct / np.maximum(mt, 1e-6)[..., None]; Ph = Ch / np.maximum(mh, 1e-6)[..., None]; Phat = Chat / np.maximum(mhat, 1e-6)[..., None]
        a = np.abs((mhat - mt) - (mh - mt)) * np.linalg.norm(0.5 * (Pt + Ph), axis=2)
        s = 0.5 * (mt + mh) * np.linalg.norm(Phat - Ph, axis=2)
        a[~ok] = np.nan; s[~ok] = np.nan; eA[:, :tv, j] = a; eS[:, :tv, j] = s
    return dict(E=E, E0=E0, cos=cos, eA=eA, eS=eS)


def ci_row(vals, clusters, n_boot, seed=0):
    pt, lo, hi = P.cluster_bootstrap(vals, clusters, n_boot=n_boot, seed=seed)
    return pt, lo, hi


def recall_at_precision(y, p, prec=0.5):
    pr, rc, _ = precision_recall_curve(y, p)
    ok = pr[:-1] >= prec
    return float(rc[:-1][ok].max()) if ok.any() else 0.0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=P.DATASETS); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); ds = a.dataset; t_start = time.time()
    out = P.ds_out(ds); res = out / "results"; res.mkdir(exist_ok=True)
    D = P.load_raw(ds); meta, man, thr = D["meta"], D["man"], D["thr"]
    te = man["test"]; C_all = D["C"]
    C = C_all[te].astype(np.float32)
    Fr = P.load_frames(ds); E_ev = P.load_events(ds)
    F_te = Fr[Fr.example.isin(te)].reset_index(drop=True)
    assert (F_te.example.values.reshape(len(te), P.T - 1)[:, 0] == te).all()
    take = meta.take_key.values[te]; group = meta.group.values[te]
    take_f = np.repeat(take, P.T - 1)
    # cosine threshold: train median of ||Delta_h C_t||
    Ctr = C_all[man["train"]].astype(np.float32)
    cos_thr = [float(np.median(np.linalg.norm(Ctr[:, h:] - Ctr[:, :-h], axis=2))) for h in P.HORIZONS]
    classes = {h: P.frame_classes(F_te, h) for h in P.HORIZONS}                  # (n_te*63,) object
    # ------------------------------------------------------------------ Delta-C regression
    conds = [c for c in P.CONDITIONS if (out / "preds" / f"delta_{c}_seed0.npz").exists()]
    seeds = [s for s in P.SEEDS if all((out / "preds" / f"delta_{c}_seed{s}.npz").exists() for c in conds)]
    log.info("%s: conditions %s, seeds %s", ds, conds, seeds)
    M = {}                                                                          # cond -> seed-mean metrics (n,63,3)
    per_seed_E = {}
    rows_long = []
    for c in conds:
        acc = None
        for s in seeds:
            z = np.load(out / "preds" / f"delta_{c}_seed{s}.npz"); assert (z["example"] == te).all()
            mt = per_frame_metrics(C, z["pred"].astype(np.float32), thr, cos_thr)
            per_seed_E[(c, s)] = mt["E"]
            acc = {k: v.copy() for k, v in mt.items()} if acc is None else {k: acc[k] + v for k, v in mt.items()}
        M[c] = {k: v / len(seeds) for k, v in acc.items()}
        for j, h in enumerate(P.HORIZONS):
            df = pd.DataFrame(dict(dataset=ds, example=np.repeat(te, P.T - 1), take_key=take_f, group=np.repeat(group, P.T - 1), t=np.tile(np.arange(P.T - 1), len(te)),
                                   h=h, cond=c, causal=P.CONDITIONS[c]["fut"] == 0, frame_class=classes[h],
                                   E=M[c]["E"][:, :, j].reshape(-1), E0=M[c]["E0"][:, :, j].reshape(-1), cos=M[c]["cos"][:, :, j].reshape(-1),
                                   eA=M[c]["eA"][:, :, j].reshape(-1), eS=M[c]["eS"][:, :, j].reshape(-1)))
            for s in seeds:
                df[f"E_seed{s}"] = per_seed_E[(c, s)][:, :, j].reshape(-1)
            rows_long.append(df[df.E.notna()])
    L = pd.concat(rows_long, ignore_index=True)
    L.to_csv(out / "delta_contact_predictions.csv", index=False, float_format="%.5g")
    # class masks per h on the long table are the frame_class column; 'all' = every valid frame
    def sel(df, cls):
        if cls == "all":
            return np.ones(len(df), bool)
        if cls == "spike":
            return (df.frame_class != "non_spike").values
        return (df.frame_class == cls).values
    overall, event_cond, seedvar = [], [], []
    for c in conds:
        for h in P.HORIZONS:
            d = L[(L.cond == c) & (L.h == h)]
            for cls in P.FRAME_CLASSES:
                m = sel(d, cls)
                if m.sum() == 0:
                    continue
                dd = d[m]; cl = dd.take_key.values
                E_pt, E_lo, E_hi = ci_row(dd.E.values, cl, a.n_boot)
                r_pt, r_lo, r_hi = P.cluster_bootstrap_ratio(dd.E.values, dd.E0.values, cl, n_boot=a.n_boot)
                q_pt, q_lo, q_hi = P.cluster_bootstrap_ratio(dd.E.values ** 2, dd.E0.values ** 2, cl, n_boot=a.n_boot)   # squared-error version (the training criterion)
                cs = ci_row(dd.cos.values, cl, a.n_boot) if dd.cos.notna().any() else (np.nan,) * 3
                row = dict(dataset=ds, cond=c, causal=P.CONDITIONS[c]["fut"] == 0, h=h, frame_class=cls, n_frames=int(m.sum()), n_takes=int(dd.take_key.nunique()),
                           E=E_pt, E_lo=E_lo, E_hi=E_hi, E0=float(dd.E0.mean()), improvement=1 - r_pt, improvement_lo=1 - r_hi, improvement_hi=1 - r_lo,
                           E2=float((dd.E.values ** 2).mean()), E2_0=float((dd.E0.values ** 2).mean()), improvement_sq=1 - q_pt, improvement_sq_lo=1 - q_hi, improvement_sq_hi=1 - q_lo,
                           cos=cs[0], cos_lo=cs[1], cos_hi=cs[2], n_cos=int(dd.cos.notna().sum()),
                           eA=float(np.nanmean(dd.eA)) if dd.eA.notna().any() else np.nan, eS=float(np.nanmean(dd.eS)) if dd.eS.notna().any() else np.nan, n_decomp=int(dd.eA.notna().sum()))
                for s in seeds:
                    row[f"E_seed{s}"] = float(dd[f"E_seed{s}"].mean())
                row["E_seed_std"] = float(np.std([row[f"E_seed{s}"] for s in seeds]))
                (overall if cls == "all" else event_cond).append(row)
    pd.DataFrame(overall).to_csv(res / "overall_delta_results.csv", index=False)
    pd.DataFrame(overall + event_cond).to_csv(res / "event_conditioned_results.csv", index=False)
    # ------------------------------------------------------------------ incremental information (paired per frame)
    inc = []
    for h in P.HORIZONS:
        base = {c: L[(L.cond == c) & (L.h == h)].reset_index(drop=True) for c in conds}
        for cls in P.FRAME_CLASSES:
            for ca, cb in PAIRS:
                if ca not in base or cb not in base:
                    continue
                m = sel(base[ca], cls)
                if m.sum() == 0:
                    continue
                A, B = base[ca][m], base[cb][m]
                assert (A.example.values == B.example.values).all() and (A.t.values == B.t.values).all()
                pt, lo, hi = P.paired_cluster_bootstrap(A.E.values, B.E.values, A.take_key.values, n_boot=a.n_boot)
                q_pt, q_lo, q_hi = P.paired_cluster_bootstrap(A.E.values ** 2, B.E.values ** 2, A.take_key.values, n_boot=a.n_boot)
                f0 = base["F0"][m].E.mean() if "F0" in base else np.nan
                f0sq = (base["F0"][m].E.values ** 2).mean() if "F0" in base else np.nan
                inc.append(dict(dataset=ds, h=h, frame_class=cls, pair=f"{ca}-{cb}", cond_a=ca, cond_b=cb, n_frames=int(m.sum()), diff=pt, diff_lo=lo, diff_hi=hi,
                                rel_to_F0_pct=100 * pt / f0 if f0 else np.nan, E_a=float(A.E.mean()), E_b=float(B.E.mean()),
                                significant=bool(hi < 0 or lo > 0),
                                diff_sq=q_pt, diff_sq_lo=q_lo, diff_sq_hi=q_hi, rel_sq_to_F0_pct=100 * q_pt / f0sq if f0sq else np.nan, significant_sq=bool(q_hi < 0 or q_lo > 0)))
    pd.DataFrame(inc).to_csv(res / "incremental_information.csv", index=False)
    # ------------------------------------------------------------------ history-length ablation (+ representation, control, oracle)
    abl = []
    for h in P.HORIZONS:
        for cls in ABL_CLASSES:
            for c in conds:
                d = L[(L.cond == c) & (L.h == h)]; m = sel(d, cls)
                if m.sum() == 0:
                    continue
                dd = d[m]; pt, lo, hi = ci_row(dd.E.values, dd.take_key.values, a.n_boot)
                abl.append(dict(dataset=ds, h=h, frame_class=cls, cond=c, k=P.CONDITIONS[c]["k"], hand=P.CONDITIONS[c]["hand"], causal=P.CONDITIONS[c]["fut"] == 0,
                                shuffle=P.CONDITIONS[c]["shuffle"], n_frames=int(m.sum()), E=pt, E_lo=lo, E_hi=hi, E0=float(dd.E0.mean()), hand_input_dim=P.hand_dim(c)))
    pd.DataFrame(abl).to_csv(res / "history_length_ablation.csv", index=False)
    # ------------------------------------------------------------------ event prediction
    econds = [c for c in P.CONDITIONS if (out / "preds" / f"event_{c}_seed0.npz").exists()]
    eseeds = [s for s in P.SEEDS if all((out / "preds" / f"event_{c}_seed{s}.npz").exists() for c in econds)]
    E_test = E_ev[E_ev.split_set == "test"]
    subtype = {}                                                                    # positive subtype start maps
    for name, cats in (("spatial_mixed", ["spatial", "mixed"]), ("release_regrasp", ["release", "onset+release", "onset"])):
        st = np.zeros((len(meta), P.T - 1), bool)
        pos = E_test[E_test.positive_event & E_test.event_category.isin(cats)]
        st[pos.example.values, pos.start_frame.values] = True
        subtype[name] = st[te]
    trans_rows, tp_long, einc = [], [], []
    probs = {}
    for c in econds:
        pr = []
        for s in eseeds:
            z = np.load(out / "preds" / f"event_{c}_seed{s}.npz"); assert (z["example"] == te).all()
            pr.append(z["prob"]); y_all, v_all = z["y"], z["valid"]
        probs[c] = np.stack(pr)                                                     # (seeds, n, 63, 2)
    if not econds:
        y_all = v_all = np.zeros((len(te), P.T - 1, len(P.EVENT_HORIZONS)), bool)
    for c in econds:
        for j, h in enumerate(P.EVENT_HORIZONS):
            v = v_all[:, :, j]; y = y_all[:, :, j][v]; cl = np.repeat(take, P.T - 1).reshape(len(te), P.T - 1)[v]
            pj = probs[c][:, :, :, j][:, v]                                          # (seeds, n_valid)
            mean_ap = lambda yy, pp: float(np.mean([average_precision_score(yy, pp[s]) for s in range(len(pp))]))
            mean_auc = lambda yy, pp: float(np.mean([roc_auc_score(yy, pp[s]) for s in range(len(pp))]))
            # bootstrap over takes, recomputing the seed-mean score; the score fn receives the seed axis through a closure on indices
            u, inv = np.unique(cl, return_inverse=True); groups = [np.where(inv == g)[0] for g in range(len(u))]
            rng = np.random.default_rng(0); reps_ap, reps_auc = [], []
            for _ in range(500):
                s_ = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
                if y[s_].any() and (~y[s_]).any():
                    reps_ap.append(mean_ap(y[s_], pj[:, s_])); reps_auc.append(mean_auc(y[s_], pj[:, s_]))
            row = dict(dataset=ds, cond=c, causal=P.CONDITIONS[c]["fut"] == 0, h=h, n_frames=int(v.sum()), n_pos=int(y.sum()), prevalence=float(y.mean()),
                       auprc=mean_ap(y, pj), auprc_lo=float(np.percentile(reps_ap, 2.5)), auprc_hi=float(np.percentile(reps_ap, 97.5)),
                       auroc=mean_auc(y, pj), auroc_lo=float(np.percentile(reps_auc, 2.5)), auroc_hi=float(np.percentile(reps_auc, 97.5)),
                       recall_at_p50=float(np.mean([recall_at_precision(y, pj[s]) for s in range(len(pj))])))
            for s in range(len(eseeds)):
                row[f"auprc_seed{eseeds[s]}"] = float(average_precision_score(y, pj[s])); row[f"auroc_seed{eseeds[s]}"] = float(roc_auc_score(y, pj[s]))
            # positive subtypes: one-vs-rest with the other positives removed
            for name, st in subtype.items():
                ys = np.zeros((len(te), P.T - 1), bool)
                for t in range(P.T - 1 - h):
                    ys[:, t] = st[:, t + 1:t + h + 1].any(1)
                ys = ys[v]; keep = ys | ~y
                row[f"auprc_{name}"] = mean_ap(ys[keep], pj[:, keep]) if ys[keep].any() else np.nan
                row[f"n_pos_{name}"] = int(ys.sum())
            trans_rows.append(row)
            tp_long.append(pd.DataFrame(dict(dataset=ds, example=np.repeat(te, P.T - 1).reshape(len(te), P.T - 1)[v], take_key=cl, t=np.tile(np.arange(P.T - 1), len(te)).reshape(len(te), P.T - 1)[v],
                                             h=h, cond=c, y=y, prob=pj.mean(0))))
    pd.DataFrame(trans_rows).to_csv(res / "transition_prediction_results.csv", index=False)
    if tp_long:
        pd.concat(tp_long, ignore_index=True).to_csv(out / "transition_predictions.csv", index=False, float_format="%.5g")
    # paired take-bootstrap differences of the seed-mean AUPRC
    for j, h in enumerate(P.EVENT_HORIZONS):
        if not econds:
            break
        v = v_all[:, :, j]; y = y_all[:, :, j][v]; cl = np.repeat(take, P.T - 1).reshape(len(te), P.T - 1)[v]
        u, inv = np.unique(cl, return_inverse=True); groups = [np.where(inv == g)[0] for g in range(len(u))]
        for ca, cb in PAIRS:
            if ca not in probs or cb not in probs:
                continue
            pa, pb = probs[ca][:, :, :, j][:, v], probs[cb][:, :, :, j][:, v]
            f = lambda yy, pp: float(np.mean([average_precision_score(yy, pp[s]) for s in range(len(pp))]))
            rng = np.random.default_rng(1); reps = []
            for _ in range(500):
                s_ = np.concatenate([groups[g] for g in rng.integers(0, len(u), len(u))])
                if y[s_].any() and (~y[s_]).any():
                    reps.append(f(y[s_], pa[:, s_]) - f(y[s_], pb[:, s_]))
            einc.append(dict(dataset=ds, h=h, metric="auprc", pair=f"{ca}-{cb}", cond_a=ca, cond_b=cb, diff=f(y, pa) - f(y, pb), diff_lo=float(np.percentile(reps, 2.5)), diff_hi=float(np.percentile(reps, 97.5)),
                             significant=bool(np.percentile(reps, 2.5) > 0 or np.percentile(reps, 97.5) < 0)))
    pd.DataFrame(einc).to_csv(res / "transition_incremental.csv", index=False)
    # ------------------------------------------------------------------ bookkeeping
    ck = {}
    for kind in ("delta", "event"):
        for c in P.CONDITIONS:
            for s in P.SEEDS:
                p = out / "ckpt" / f"{kind}_{c}_seed{s}.pt"
                if p.exists():
                    import torch
                    z = torch.load(p, map_location="cpu", weights_only=False)
                    ck[f"{kind}_{c}_seed{s}"] = {k: z[k] for k in z if k in ("best_val", "best_step", "steps", "stopped_by", "n_params", "n_hand_params", "d_hand", "d_in", "test_auprc", "test_auroc")}
    P.write_json(res / "training_summary.json", ck)
    P.write_json(res / "evaluation_config.json", dict(dataset=ds, conditions=conds, seeds=seeds, event_conditions=econds, event_seeds=eseeds, cos_threshold_train_median=dict(zip(map(str, P.HORIZONS), cos_thr)),
                                                       zero_mass=thr, n_test=len(te), n_takes=int(len(np.unique(take))), n_boot=a.n_boot, pairs=PAIRS, frame_classes=P.FRAME_CLASSES))
    o = pd.DataFrame(overall)
    pd.set_option("display.width", 250)
    print(o[o.cond.isin(["F0", "F1", "F2k8", "F2shuf", "Ffut"])].pivot(index="cond", columns="h", values=["E", "improvement", "cos"]).round(3).to_string())
    if trans_rows:
        print(pd.DataFrame(trans_rows)[["cond", "h", "prevalence", "auprc", "auprc_lo", "auprc_hi", "auroc"]].round(3).to_string())
    log.info("done %s in %.0fs", ds, time.time() - t_start)


if __name__ == "__main__":
    main()
