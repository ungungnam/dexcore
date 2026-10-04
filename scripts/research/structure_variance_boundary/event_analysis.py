#!/usr/bin/env python
"""Experiment C — event-level structure vs residual: for every persistent contact-change event (the temporal-event
labels, pre = start frame, post = end + 1, as in the wrench report) the dense contact change ||C_post - C_pre||, the
hand change (RMS displacement of the 100 surface points / l), the wrench-profile change (relative L1 distance,
relative Q change, retention) and the change of every representation block / ladder level, each block z-scored with
TRAIN statistics and summarised as the RMS over its dimensions (so blocks of different width are comparable).
    python event_analysis.py --dataset taco
Analyses on the TEST events (take-cluster bootstrap): Spearman correlations of the dense change with every block /
level change and of every block / level change with the wrench change; ridge probes fitted on the TRAIN events that
predict the wrench change from the cumulative block changes; a logistic probe for 'large wrench change' (above the
train median); train-quantile quadrants as a secondary view. Finger annotations (appeared / disappeared / slid /
stable) and the event class are attached for interpretation only.
Writes <ds>/events/{event_representation_change.csv, event_correlations.csv, event_probes.csv, event_quadrants.csv}.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

import sv_common as S
import wrench_lp as L

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("events")
COARSE = ["part", "amount", "geom", "topo", "hand"]
LEVELS = ["R0", "R1", "R2", "R3", "R4", "R5", "CONTACT_FULL", "FULL"]
SLIDE_M = 0.01
WRENCH = {"d_q": "relative L1 profile distance", "abs_rel_change_Q": "|relative Q change|", "one_minus_R": "1 - retention (pre->post)"}


def block_deltas(blocks, norm, ev):
    """RMS z-scored change of every block and level between the pre and post frames of one event."""
    n, pre, post = int(ev.example), int(ev.pre_frame) + S.PAD, int(ev.post_frame) + S.PAD
    d = {}
    z = {}
    for b, X in blocks.items():
        mu, sd = norm[b]
        z[b] = (X[n, post] - X[n, pre]) / sd
        d[f"d_{b}" if b not in ("C", "H") else f"dz_{b}"] = float(np.sqrt((z[b] ** 2).mean()))     # dense blocks: dz_ (d_C / d_H are the raw changes)
    for lv in LEVELS:
        zz = np.concatenate([z[b] for b in S.LADDER[lv]])
        d[f"d_{lv}"] = float(np.sqrt((zz ** 2).mean()))
    return d


def finger_annotation(blocks, F, ev):
    n, pre, post = int(ev.example), int(ev.pre_frame) + S.PAD, int(ev.post_frame) + S.PAD
    a_pre, a_post = blocks["part"][n, pre] > 0.5, blocks["part"][n, post] > 0.5
    p_pre, p_post = F["p"][n, pre] * F["length"][n], F["p"][n, post] * F["length"][n]
    app = int((a_post & ~a_pre).sum()); dis = int((a_pre & ~a_post).sum())
    both = a_pre & a_post
    slid = int((np.linalg.norm(p_post - p_pre, axis=1)[both] > SLIDE_M).sum()); stable = int(both.sum() - slid)
    kind = "finger_exchange" if (app or dis) else ("sliding" if slid else "stable_set")
    return dict(n_appeared=app, n_disappeared=dis, n_slid=slid, n_stable=stable, n_active_pre=int(a_pre.sum()), n_active_post=int(a_post.sum()), finger_kind=kind)


def build_table(ds, F, blocks, norm):
    rows = []
    for split in ("train", "test"):
        E = S.select_events(ds, split)
        for ev in E.itertuples():
            n, pre, post = int(ev.example), int(ev.pre_frame), int(ev.post_frame)
            q_pre, q_post = F["q"][n, pre], F["q"][n, post]
            pm = L.profile_metrics(q_pre, q_post)
            H_pre, H_post = blocks["H"][n, pre + S.PAD].reshape(100, 3), blocks["H"][n, post + S.PAD].reshape(100, 3)
            r = dict(dataset=ds, split=split, event_id=ev.event_id, example=n, take_key=ev.take_key, cls=ev.cls, event_category=ev.event_category,
                     category=ev.category, hand=ev.hand, pre_frame=pre, post_frame=post, firm_pre=ev.firm_pre, firm_post=ev.firm_post, firm_through=ev.firm_through,
                     primary=ev.primary, persistence_h4=ev.persistence_h4, amount_ratio=ev.amount_ratio,
                     d_C=float(np.linalg.norm(blocks["C"][n, post + S.PAD] - blocks["C"][n, pre + S.PAD])), dC_wrench=float(ev.dC),
                     d_H=float(np.sqrt(((H_post - H_pre) ** 2).sum(1).mean())) / float(F["length"][n]),
                     d_q=pm["l1_rel_distance"], rel_change_Q=pm["rel_change_Q"], abs_rel_change_Q=abs(pm["rel_change_Q"]), retention=pm["R_pre_to_post"],
                     one_minus_R=1.0 - pm["R_pre_to_post"] if np.isfinite(pm["R_pre_to_post"]) else np.nan, cosine_q=pm["cosine"], Q_pre=pm["Q_pre"], Q_post=pm["Q_post"])
            r.update(block_deltas(blocks, norm, ev)); r.update(finger_annotation(blocks, F, ev))
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True, choices=S.DATASETS); ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(); ds = a.dataset
    out = S.ds_out(ds) / "events"; out.mkdir(parents=True, exist_ok=True)
    F = S.load_features(ds); hand100 = S.HP.load_hand_cache(ds)["hand"]
    D = S.HP.load_raw(ds); tr = D["man"]["train"]
    blocks = S.block_arrays(F, hand100)
    norm = {}
    for b, X in blocks.items():
        mu, sd = S.fit_norm(X, tr)
        if b in ("C", "H"):
            sd = np.full_like(sd, float((X[tr][:, S.PAD:S.PAD + S.T] - mu).std()))
        norm[b] = (mu, sd)
    E = build_table(ds, F, blocks, norm)
    E.to_csv(out / "event_representation_change.csv", index=False)
    te = E[E.split == "test"].reset_index(drop=True); trn = E[E.split == "train"].reset_index(drop=True)
    log.info("%s: %d train / %d test events; |d_C - wrench dC| max %.2e", ds, len(trn), len(te), float(np.abs(te.d_C - te.dC_wrench).max()))
    subsets = {"all_persistent": te[te.cls.isin(["persistent_spatial", "persistent_mixed"])], "persistent_spatial": te[te.cls == "persistent_spatial"],
               "persistent_mixed": te[te.cls == "persistent_mixed"], "all_events": te}
    change_cols = [f"d_{b}" for b in COARSE] + [f"d_{lv}" for lv in LEVELS] + ["d_H"]
    # 1. correlations
    corr = []
    for name, sub in subsets.items():
        if len(sub) < 10:
            continue
        for col in change_cols:
            rho, lo, hi = S.spearman_cluster(sub.d_C.values, sub[col].values, sub.take_key.values)
            corr.append(dict(dataset=ds, subset=name, x="d_C", y=col, rho=rho, lo=lo, hi=hi, n=len(sub)))
            for w in WRENCH:
                ok = np.isfinite(sub[w].values)
                rho, lo, hi = S.spearman_cluster(sub[col].values[ok], sub[w].values[ok], sub.take_key.values[ok])
                corr.append(dict(dataset=ds, subset=name, x=col, y=w, rho=rho, lo=lo, hi=hi, n=int(ok.sum())))
        for w in WRENCH:
            ok = np.isfinite(sub[w].values)
            rho, lo, hi = S.spearman_cluster(sub.d_C.values[ok], sub[w].values[ok], sub.take_key.values[ok])
            corr.append(dict(dataset=ds, subset=name, x="d_C", y=w, rho=rho, lo=lo, hi=hi, n=int(ok.sum())))
    pd.DataFrame(corr).to_csv(out / "event_correlations.csv", index=False)
    # 2. probes fitted on the train events: cumulative blocks -> wrench change (ridge R^2), large change (logistic AUROC / AUPRC)
    probes = []
    cum = {"R0": ["d_part"], "R1": ["d_part", "d_amount"], "R2": ["d_part", "d_amount", "d_geom"], "R3": ["d_part", "d_amount", "d_geom", "d_topo"],
           "R4": ["d_part", "d_amount", "d_geom", "d_hand"], "R5": ["d_part", "d_amount", "d_geom", "d_topo", "d_hand"], "CONTACT_FULL": ["d_C"], "FULL": ["d_C", "d_H"],
           "R5+dense": ["d_part", "d_amount", "d_geom", "d_topo", "d_hand", "d_C", "d_H"]}
    for name, sub in subsets.items():
        if len(sub) < 20:
            continue
        sub_tr = trn[trn.cls.isin(sub.cls.unique())]
        for w in WRENCH:
            y_tr = sub_tr[w].values; y_te = sub[w].values; ok_tr = np.isfinite(y_tr); ok_te = np.isfinite(y_te)
            thr = float(np.nanmedian(y_tr))
            for lv, cols in cum.items():
                X_tr = np.log1p(sub_tr[cols].values); X_te = np.log1p(sub[cols].values)
                mu, sd = X_tr[ok_tr].mean(0), X_tr[ok_tr].std(0) + 1e-9
                X_tr = (X_tr - mu) / sd; X_te = (X_te - mu) / sd
                best = None
                for alpha in (0.1, 1.0, 10.0, 100.0):
                    m = Ridge(alpha=alpha).fit(X_tr[ok_tr], y_tr[ok_tr]); pred = m.predict(X_te[ok_te])
                    sse = (pred - y_te[ok_te]) ** 2; sst = (y_te[ok_te] - y_tr[ok_tr].mean()) ** 2
                    r2 = 1 - sse.sum() / sst.sum()
                    if best is None or r2 > best[0]:
                        best = (r2, alpha, sse, sst)
                r2, alpha, sse, sst = best
                _, lo, hi = S.ratio_cluster_bootstrap(sse, sst, sub.take_key.values[ok_te], n_boot=a.n_boot)
                clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_tr[ok_tr], y_tr[ok_tr] > thr)
                p = clf.predict_proba(X_te[ok_te])[:, 1]; yb = y_te[ok_te] > thr
                auroc = roc_auc_score(yb, p) if 0 < yb.mean() < 1 else np.nan; auprc = average_precision_score(yb, p) if 0 < yb.mean() < 1 else np.nan
                probes.append(dict(dataset=ds, subset=name, wrench=w, level=lv, features="+".join(cols), n_train=int(ok_tr.sum()), n_test=int(ok_te.sum()),
                                   ridge_r2=r2, ridge_r2_lo=1 - hi, ridge_r2_hi=1 - lo, ridge_alpha=alpha, logit_auroc=auroc, logit_auprc=auprc, thr_train_median=thr, positive_rate=float(yb.mean())))
    pd.DataFrame(probes).to_csv(out / "event_probes.csv", index=False)
    # 3. quadrants with TRAIN medians (secondary): large dense change but small structured + small wrench change; large wrench change missed by a level
    quad = []
    for name, sub in subsets.items():
        if len(sub) < 10:
            continue
        sub_tr = trn[trn.cls.isin(sub.cls.unique())]
        thr_C = float(sub_tr.d_C.median())
        for w in WRENCH:
            thr_w = float(np.nanmedian(sub_tr[w]))
            large_C = sub.d_C.values > thr_C; large_w = sub[w].values > thr_w; small_w = sub[w].values <= thr_w
            for col in [f"d_{lv}" for lv in LEVELS] + [f"d_{b}" for b in COARSE] + ["d_H"]:
                thr_i = float(sub_tr[col].median()); small_i = sub[col].values <= thr_i; large_i = ~small_i
                f1, lo1, hi1 = S.cluster_bootstrap((small_i & small_w)[large_C].astype(float), sub.take_key.values[large_C], n_boot=a.n_boot) if large_C.sum() else (np.nan,) * 3
                f2, lo2, hi2 = S.cluster_bootstrap(small_i[large_w].astype(float), sub.take_key.values[large_w], n_boot=a.n_boot) if large_w.sum() else (np.nan,) * 3
                f3, lo3, hi3 = S.cluster_bootstrap(large_i[large_w].astype(float), sub.take_key.values[large_w], n_boot=a.n_boot) if large_w.sum() else (np.nan,) * 3
                quad.append(dict(dataset=ds, subset=name, wrench=w, change=col, thr_C=thr_C, thr_w=thr_w, thr_change=thr_i, n_large_C=int(large_C.sum()), n_large_w=int(large_w.sum()),
                                 frac_variation_candidates=f1, lo1=lo1, hi1=hi1, frac_large_w_missed=f2, lo2=lo2, hi2=hi2, frac_large_w_captured=f3, lo3=lo3, hi3=hi3))
    pd.DataFrame(quad).to_csv(out / "event_quadrants.csv", index=False)
    # 4. by finger kind (interpretation only)
    by = te[te.cls.isin(["persistent_spatial", "persistent_mixed"])].groupby(["cls", "finger_kind"]).agg(n=("event_id", "size"), d_C=("d_C", "median"), d_q=("d_q", "median"),
                                                                                                    retention=("retention", "median"), d_R0=("d_R0", "median"), d_R2=("d_R2", "median"), d_R5=("d_R5", "median")).reset_index()
    by.to_csv(out / "event_by_finger_kind.csv", index=False)
    log.info("done %s: %d correlation rows, %d probe rows, %d quadrant rows", ds, len(corr), len(probes), len(quad))


if __name__ == "__main__":
    main()
