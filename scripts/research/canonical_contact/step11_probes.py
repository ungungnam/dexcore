"""Step 11: linear probes on canonical contact vectors.

For every backend, per (role, hand) pooled across categories AND per (category, role, hand):
  logistic regression (sklearn, C=1, max_iter=2000, lbfgs, standardised features fitted on the
  training fold) predicting  (A) verb  and  (B) mesh_id  from X_soft.
Evaluation protocols:
  * groupkfold : GroupKFold(5) grouped by SEQUENCE; sample order is sorted by
                 (category, sequence_id, window, hand) so folds are identical for every backend.
  * split      : official split, train -> test_1.
Feature variants:
  * X_soft            : the canonical contact vector only
  * X_soft+category   : X_soft plus one-hot category (pooled rows only; clearly labelled)
Classes with fewer than --min-class-seq sequences are dropped (they cannot appear in both train
and test). Reported: accuracy, balanced accuracy, macro-F1 (aggregated over folds), chance_majority
(accuracy of always predicting the training-fold majority class), chance_uniform = 1/n_classes,
n_classes, n_train / n_test, n_sequences.

Output: <root>/probe_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cc_common import add_common_args, list_cache_files, load_cache, resolve_backends, setup_logging  # noqa: E402

log = setup_logging("step11")
TARGETS = ["verb", "mesh_id"]
N_FOLDS = 5


def fit_predict(Xtr, ytr, Xte, seed: int, max_iter: int):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=max_iter, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def score(y_true, y_pred, y_major):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    return dict(accuracy=accuracy_score(y_true, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
                chance_majority=accuracy_score(y_true, y_major))


def filter_classes(meta: pd.DataFrame, target: str, min_class_seq: int) -> np.ndarray:
    n_seq = meta.groupby(target).sequence_id.nunique()
    ok = n_seq[n_seq >= min_class_seq].index
    return meta[target].isin(ok).values


def run_groupkfold(X, y, groups, seed, max_iter):
    gkf = GroupKFold(n_splits=min(N_FOLDS, len(np.unique(groups))))
    yt, yp, ym, ntr = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        pred = fit_predict(X[tr], y[tr], X[te], seed, max_iter)
        vals, cnt = np.unique(y[tr], return_counts=True)
        yt.append(y[te]); yp.append(pred); ym.append(np.full(len(te), vals[cnt.argmax()])); ntr.append(len(tr))
    if not yt:
        return None
    yt, yp, ym = map(np.concatenate, (yt, yp, ym))
    # n_train = mean training-fold size; n_test = total held-out predictions over the folds
    return score(yt, yp, ym) | dict(n_train=int(round(np.mean(ntr))), n_test=len(yt), n_folds=len(ntr))


def run_split(X, y, split, seed, max_iter):
    tr = split == "train"
    te = split == "test_1"
    if tr.sum() < 5 or te.sum() < 1 or len(np.unique(y[tr])) < 2:
        return None
    pred = fit_predict(X[tr], y[tr], X[te], seed, max_iter)
    vals, cnt = np.unique(y[tr], return_counts=True)
    return score(y[te], pred, np.full(te.sum(), vals[cnt.argmax()])) | dict(n_train=int(tr.sum()), n_test=int(te.sum()), n_folds=1)


def probe_group(X, meta, backend, scope, category, role, hand, feature_set, args) -> list[dict]:
    rows = []
    for target in TARGETS:
        keep = filter_classes(meta, target, args.min_class_seq)
        m = meta[keep].reset_index(drop=True)
        Xk = X[keep]
        y = m[target].values.astype(str)
        n_cls = len(np.unique(y))
        base = dict(backend=backend, scope=scope, category=category, role=role, hand=hand, target=target,
                    features=feature_set, n_samples=len(m), n_sequences=m.sequence_id.nunique(), n_classes=n_cls,
                    chance_uniform=1.0 / n_cls if n_cls else np.nan)
        if n_cls < 2 or len(m) < args.min_samples:
            log.info("skip %s/%s/%s/%s target=%s: %d classes, %d samples", backend, category, role, hand, target, n_cls, len(m))
            continue
        r = run_groupkfold(Xk, y, m.sequence_id.values, args.seed, args.max_iter)
        if r is not None:
            rows.append(base | dict(protocol="groupkfold") | r)
        r = run_split(Xk, y, m.split.values, args.seed, args.max_iter)
        if r is not None:
            rows.append(base | dict(protocol="split") | r)
    return rows


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__))
    ap.add_argument("--min-class-seq", type=int, default=2, dest="min_class_seq")
    ap.add_argument("--min-samples", type=int, default=10, dest="min_samples")
    ap.add_argument("--max-iter", type=int, default=2000, dest="max_iter")
    ap.add_argument("--no-per-category", action="store_true", dest="no_per_category")
    args = ap.parse_args()
    root = args.root
    rows = []
    for backend in resolve_backends(root, args.backends):
        # collect everything for this backend, sorted so that folds are identical across backends
        blocks = []
        for cat, role, path in list_cache_files(root, backend):
            c = load_cache(path, min_touch=args.min_touch)
            m = c["meta"].copy()
            m["category"] = cat
            m["role"] = role
            m["_X"] = list(range(len(m)))
            blocks.append((m, c["X_soft"]))
        if not blocks:
            continue
        meta = pd.concat([b[0].assign(_blk=i) for i, b in enumerate(blocks)], ignore_index=True)
        X = np.concatenate([b[1] for b in blocks], axis=0)
        # global row index into X
        offs = np.cumsum([0] + [len(b[0]) for b in blocks[:-1]])
        meta["_row"] = meta["_X"].values + offs[meta["_blk"].values]
        meta = meta.sort_values(["category", "sequence_id", "window", "hand"], kind="stable").reset_index(drop=True)
        X = X[meta["_row"].values]
        cats = sorted(meta.category.unique())
        onehot = (meta.category.values[:, None] == np.array(cats)[None, :]).astype(np.float32)
        for (role, hand), g in meta.groupby(["role", "hand"], sort=True):
            idx = g.index.values
            log.info("%s pooled %s/%s: %d samples", backend, role, hand, len(idx))
            rows += probe_group(X[idx], g.reset_index(drop=True), backend, "ALL_pooled", "ALL_pooled", role, hand, "X_soft", args)
            if len(cats) > 1:
                Xc = np.concatenate([X[idx], onehot[idx]], axis=1)
                rows += probe_group(Xc, g.reset_index(drop=True), backend, "ALL_pooled", "ALL_pooled", role, hand,
                                    "X_soft+category", args)
            if args.no_per_category:
                continue
            for cat, gc in g.groupby("category", sort=True):
                rows += probe_group(X[gc.index.values], gc.reset_index(drop=True), backend, "category", cat, role, hand,
                                    "X_soft", args)
    out = pd.DataFrame(rows)
    front = ["backend", "scope", "category", "role", "hand", "target", "features", "protocol", "n_samples", "n_sequences",
             "n_classes", "n_train", "n_test", "n_folds", "accuracy", "balanced_accuracy", "macro_f1", "chance_majority",
             "chance_uniform"]
    if len(out):
        out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]
    out.to_csv(root / "probe_metrics.csv", index=False)
    log.info("wrote probe_metrics.csv (%d rows)", len(out))
    if len(out):
        show = out[out.scope == "ALL_pooled"][["backend", "role", "hand", "target", "features", "protocol", "n_classes",
                                                "accuracy", "balanced_accuracy", "macro_f1", "chance_majority"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
