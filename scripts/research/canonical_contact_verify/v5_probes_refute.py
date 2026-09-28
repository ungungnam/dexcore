"""v5: try to refute the Step-11 probe claims (i)-(iii).

(a) reproduce pooled GroupKFold(5)-by-sequence probes (own folds, seed 0) for verb / mesh_id per backend,
    with sequence-level bootstrap CIs over the held-out predictions;
(b) leakage + trivial-solution controls for the mesh probe: coverage-only, ||X||-only, coverage-count-only,
    X restricted to points covered by every mesh of the category, X with per-(category,role,hand) mean removed;
(c) verb from object-side information alone (category / counterpart / mesh one-hots) vs X_soft (+ one-hots);
(d) same within single categories with >= 3 verbs;
(e) MLP capacity check on pooled (tool,R);
Outputs (all under verify/): probe_a_reproduce.csv, probe_b_mesh_controls.csv, probe_c_verb_objectside.csv,
probe_d_within_category.csv, probe_e_mlp.csv, probe_f_backend_paired.csv, v5_probes_refute.log
"""
from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

R = Path("/result/uhnam/dexcore/canonical_contact")
OUT = R / "verify"
sys.path.insert(0, str(R / "scripts"))
from cc_common import list_cache_files, load_cache  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    handlers=[logging.FileHandler(OUT / "v5_probes_refute.log", mode="w"), logging.StreamHandler()])
log = logging.getLogger("v5")
BACKENDS = ["normalized", "aligned", "dino", "random_perm"]
SEED, N_FOLDS, N_BOOT, MIN_TOUCH, MIN_CLASS_SEQ, MAX_ITER = 0, 5, 1000, 0.2, 2, 2000
warnings.simplefilter("ignore", ConvergenceWarning)


# ----------------------------------------------------------------------------- data
def load_backend(backend: str):
    blocks = []
    for cat, role, path in list_cache_files(R, backend):
        c = load_cache(path, min_touch=MIN_TOUCH)
        m = c["meta"].copy()
        m["category"] = cat
        m["role"] = role
        blocks.append((m, c["X_soft"], c["coverage"]))
    meta = pd.concat([b[0] for b in blocks], ignore_index=True)
    X = np.concatenate([b[1] for b in blocks]).astype(np.float32)
    C = np.concatenate([b[2] for b in blocks]).astype(bool)
    order = np.lexsort((meta.hand.values, meta.window.values, meta.sequence_id.values, meta.category.values))
    meta = meta.iloc[order].reset_index(drop=True)
    meta["mesh"] = meta.category + ":" + meta.mesh_id.astype(str)
    return meta, X[order], C[order]


def folds_by_sequence(seqs: np.ndarray, seed: int = SEED, k: int = N_FOLDS):
    """Own GroupKFold: shuffle the unique sequences with seed, deal them into k folds."""
    u = np.unique(seqs)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    fold_of = {u[p]: i % k for i, p in enumerate(perm)}
    f = np.array([fold_of[s] for s in seqs])
    return f


def keep_classes(y: np.ndarray, seqs: np.ndarray, min_seq: int = MIN_CLASS_SEQ):
    df = pd.DataFrame(dict(y=y, s=seqs))
    n = df.groupby("y").s.nunique()
    return np.isin(y, n[n >= min_seq].index.values)


def onehot(v: np.ndarray):
    u = np.unique(v)
    return (v[:, None] == u[None, :]).astype(np.float32)


# ----------------------------------------------------------------------------- probes
def fit_predict(Xtr, ytr, Xte, model="logreg"):
    sc = StandardScaler().fit(Xtr)
    if model == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=MAX_ITER, random_state=SEED)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(128,), early_stopping=True, max_iter=500, random_state=SEED)
        # sklearn's early-stopping scorer needs numeric labels
        classes, yint = np.unique(ytr, return_inverse=True)
        clf.fit(sc.transform(Xtr), yint)
        return classes[clf.predict(sc.transform(Xte))]
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def cv_predict(X, y, seqs, folds, model="logreg"):
    """Held-out prediction for every sample (each sample is in exactly one test fold). Also verifies
    that no sequence is in both train and test of any fold."""
    yp = np.empty_like(y)
    ymaj = np.empty_like(y)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        assert not (set(seqs[tr]) & set(seqs[te])), "sequence leak between train and test"
        yp[te] = fit_predict(X[tr], y[tr], X[te], model)
        vals, cnt = np.unique(y[tr], return_counts=True)
        ymaj[te] = vals[cnt.argmax()]
    return yp, ymaj


def boot_acc(correct: np.ndarray, seqs: np.ndarray, n_boot=N_BOOT, seed=SEED):
    """Sequence-level bootstrap of accuracy: resample sequences with replacement."""
    u, inv = np.unique(seqs, return_inverse=True)
    n_ok = np.bincount(inv, weights=correct.astype(float), minlength=len(u))
    n_all = np.bincount(inv, minlength=len(u)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(u), (n_boot, len(u)))
    W = np.stack([np.bincount(d, minlength=len(u)) for d in draws]).astype(float)
    acc = (W @ n_ok) / (W @ n_all)
    return acc


def summarize(y, yp, ymaj, seqs, **base):
    correct = (y == yp)
    b = boot_acc(correct, seqs)
    bm = boot_acc(y == ymaj, seqs)
    labels = np.unique(np.concatenate([y, yp]))
    return base | dict(
        n_samples=len(y), n_sequences=len(np.unique(seqs)), n_classes=len(np.unique(y)),
        accuracy=accuracy_score(y, yp), acc_lo=np.percentile(b, 2.5), acc_hi=np.percentile(b, 97.5),
        balanced_accuracy=balanced_accuracy_score(y, yp),
        macro_f1=f1_score(y, yp, average="macro", labels=labels, zero_division=0),
        chance_majority=accuracy_score(y, ymaj), chance_maj_lo=np.percentile(bm, 2.5), chance_maj_hi=np.percentile(bm, 97.5),
        chance_uniform=1.0 / len(np.unique(y)))


def run_probe(X, y, seqs, folds, model="logreg", **base):
    t = time.time()
    yp, ymaj = cv_predict(X, y, seqs, folds, model)
    r = summarize(y, yp, ymaj, seqs, **base)
    r["fit_s"] = time.time() - t
    log.info("%s  acc=%.3f [%.3f,%.3f] bal=%.3f maj=%.3f n=%d k=%d (%.0fs)",
             " ".join(f"{k}={v}" for k, v in base.items()), r["accuracy"], r["acc_lo"], r["acc_hi"],
             r["balanced_accuracy"], r["chance_majority"], r["n_samples"], r["n_classes"], r["fit_s"])
    return r, (y == yp)


# ----------------------------------------------------------------------------- main
def main():
    data = {b: load_backend(b) for b in BACKENDS}
    meta0 = data[BACKENDS[0]][0]
    for b in BACKENDS[1:]:
        m = data[b][0]
        assert len(m) == len(meta0) and (m.sequence_id.values == meta0.sequence_id.values).all() \
            and (m.window.values == meta0.window.values).all() and (m.hand.values == meta0.hand.values).all() \
            and (m.role.values == meta0.role.values).all() and (m.category.values == meta0.category.values).all(), b
    meta = meta0
    log.info("samples %d, sequences %d, meshes %d (mesh_id alone: %d)", len(meta), meta.sequence_id.nunique(),
             meta.mesh.nunique(), meta.mesh_id.nunique())
    # mesh id unique across categories?
    log.info("mesh_id -> #categories max = %d", meta.groupby("mesh_id").category.nunique().max())
    folds_all = folds_by_sequence(meta.sequence_id.values)
    meta["fold"] = folds_all
    groups = [("tool", "R"), ("target", "L"), ("tool", "L"), ("target", "R")]

    # per-backend, per-category common-coverage mask (points covered by every mesh of the category-role file)
    common = {}
    for b in BACKENDS:
        m, X, C = data[b]
        for (cat, role), g in m.groupby(["category", "role"]):
            common[(b, cat, role)] = C[g.index.values].all(0)

    rows_a, rows_b, rows_c, rows_d, rows_e = [], [], [], [], []
    correct_store = {}  # (section, backend, role, hand, target, features) -> per-sample correctness for paired diffs

    for role, hand in groups:
        gsel = (meta.role == role) & (meta.hand == hand)
        idx = np.where(gsel)[0]
        mg = meta.iloc[idx].reset_index(drop=True)
        seqs = mg.sequence_id.values
        folds = folds_all[idx]
        # object-side one-hots (backend independent)
        oh_cat = onehot(mg.category.values)
        oh_cp = onehot(mg.counterpart_cat.values)
        oh_mesh = onehot(mg.mesh.values)
        for target in ["verb", "mesh"]:
            y = mg[target].values.astype(str)
            keep = keep_classes(y, seqs)
            log.info("== (%s,%s) target=%s keep %d/%d", role, hand, target, keep.sum(), len(keep))
            yk, sk, fk = y[keep], seqs[keep], folds[keep]
            # (c) object-side-only verb probes (once)
            if target == "verb":
                for name, F in [("category_onehot", oh_cat), ("category+counterpart_onehot", np.hstack([oh_cat, oh_cp])),
                                ("mesh_onehot", oh_mesh), ("mesh+counterpart_onehot", np.hstack([oh_mesh, oh_cp]))]:
                    r, c = run_probe(F[keep], yk, sk, fk, section="c", backend="none", role=role, hand=hand, target=target,
                                     features=name)
                    rows_c.append(r); correct_store[("c", "none", role, hand, target, name)] = c
            for b in BACKENDS:
                m, X, C = data[b]
                Xg, Cg = X[idx][keep], C[idx][keep]
                # (a) reproduce
                r, c = run_probe(Xg, yk, sk, fk, section="a", backend=b, role=role, hand=hand, target=target, features="X_soft")
                rows_a.append(r); correct_store[("a", b, role, hand, target, "X_soft")] = c
                if target == "mesh":
                    # (b) controls
                    norm = np.linalg.norm(Xg, axis=1, keepdims=True)
                    cov_cnt = Cg.sum(1, keepdims=True).astype(np.float32)
                    cm = np.stack([common[(b, cat, role)] for cat in mg.category.values[keep]])
                    Xcommon = Xg * cm
                    # per-(category, role, hand) mean removed (role, hand fixed inside this group)
                    Xdm = Xg.copy()
                    for cat in np.unique(mg.category.values[keep]):
                        sel = mg.category.values[keep] == cat
                        Xdm[sel] -= Xg[sel].mean(0, keepdims=True)
                    Xcommon_dm = Xcommon.copy()
                    for cat in np.unique(mg.category.values[keep]):
                        sel = mg.category.values[keep] == cat
                        Xcommon_dm[sel] -= Xcommon[sel].mean(0, keepdims=True)
                    frac_common = float(np.mean([common[(b, cat, role)].mean() for cat in np.unique(mg.category.values[keep])]))
                    for name, F in [("coverage_only", Cg.astype(np.float32)), ("norm_only", np.hstack([norm, np.log(norm + 1e-6)])),
                                    ("coverage_count_only", cov_cnt),
                                    ("norm+coverage_count", np.hstack([norm, cov_cnt])),
                                    ("X_soft_common_points", Xcommon), ("X_soft_demeaned_cat", Xdm),
                                    ("X_soft_common_points_demeaned_cat", Xcommon_dm)]:
                        r, c = run_probe(F, yk, sk, fk, section="b", backend=b, role=role, hand=hand, target=target, features=name)
                        r["frac_points_common_mean"] = frac_common
                        rows_b.append(r); correct_store[("b", b, role, hand, target, name)] = c
                else:
                    # (c) X_soft + object-side one-hots
                    for name, F in [("X_soft+category_onehot", np.hstack([Xg, oh_cat[keep]])),
                                    ("X_soft+mesh_onehot", np.hstack([Xg, oh_mesh[keep]])),
                                    ("X_soft+mesh+counterpart_onehot", np.hstack([Xg, oh_mesh[keep], oh_cp[keep]]))]:
                        r, c = run_probe(F, yk, sk, fk, section="c", backend=b, role=role, hand=hand, target=target, features=name)
                        rows_c.append(r); correct_store[("c", b, role, hand, target, name)] = c
                # (e) MLP on pooled (tool, R)
                if (role, hand) == ("tool", "R"):
                    r, c = run_probe(Xg, yk, sk, fk, model="mlp", section="e", backend=b, role=role, hand=hand, target=target,
                                     features="X_soft(MLP128)")
                    rows_e.append(r); correct_store[("e", b, role, hand, target, "mlp")] = c
        pd.DataFrame(rows_a).to_csv(OUT / "probe_a_reproduce.csv", index=False)
        pd.DataFrame(rows_b).to_csv(OUT / "probe_b_mesh_controls.csv", index=False)
        pd.DataFrame(rows_c).to_csv(OUT / "probe_c_verb_objectside.csv", index=False)
        pd.DataFrame(rows_e).to_csv(OUT / "probe_e_mlp.csv", index=False)

    # (d) within single categories with >= 3 verbs (after class filtering), any (role, hand) with >= 30 samples
    for cat in ["spatula", "spoon", "plate", "bowl", "pan"]:
        for role, hand in groups:
            gsel = (meta.category == cat) & (meta.role == role) & (meta.hand == hand)
            idx = np.where(gsel)[0]
            if len(idx) < 30:
                continue
            mg = meta.iloc[idx].reset_index(drop=True)
            seqs, folds = mg.sequence_id.values, folds_all[idx]
            y = mg.verb.values.astype(str)
            keep = keep_classes(y, seqs)
            if len(np.unique(y[keep])) < 3 or keep.sum() < 30:
                continue
            yk, sk, fk = y[keep], seqs[keep], folds[keep]
            oh_mesh, oh_cp = onehot(mg.mesh.values[keep]), onehot(mg.counterpart_cat.values[keep])
            for name, F in [("mesh_onehot", oh_mesh), ("counterpart_onehot", oh_cp), ("mesh+counterpart_onehot", np.hstack([oh_mesh, oh_cp]))]:
                r, c = run_probe(F, yk, sk, fk, section="d", backend="none", category=cat, role=role, hand=hand, target="verb", features=name)
                r["n_meshes"] = len(np.unique(mg.mesh.values[keep]))
                rows_d.append(r)
            for b in BACKENDS:
                m, X, C = data[b]
                Xg = X[idx][keep]
                for name, F in [("X_soft", Xg), ("X_soft+mesh_onehot", np.hstack([Xg, oh_mesh])),
                                ("X_soft+mesh+counterpart_onehot", np.hstack([Xg, oh_mesh, oh_cp]))]:
                    r, c = run_probe(F, yk, sk, fk, section="d", backend=b, category=cat, role=role, hand=hand, target="verb", features=name)
                    r["n_meshes"] = len(np.unique(mg.mesh.values[keep]))
                    rows_d.append(r); correct_store[("d", b, cat, role, hand, name)] = c
                # mesh probe within category (for (f): per-category selectivity), plus the common-points control
                ym = mg.mesh.values.astype(str)
                keepm = keep_classes(ym, seqs)
                if len(np.unique(ym[keepm])) >= 2:
                    cm = common[(b, cat, role)]
                    for name, F in [("X_soft", X[idx][keepm]), ("X_soft_common_points", X[idx][keepm] * cm),
                                    ("coverage_only", C[idx][keepm].astype(np.float32))]:
                        r, c = run_probe(F, ym[keepm], seqs[keepm], folds[keepm], section="d", backend=b, category=cat, role=role,
                                         hand=hand, target="mesh", features=name)
                        r["n_meshes"] = len(np.unique(ym[keepm])); r["frac_points_common_mean"] = float(cm.mean())
                        rows_d.append(r)
            pd.DataFrame(rows_d).to_csv(OUT / "probe_d_within_category.csv", index=False)

    # (f) paired backend differences on identical samples/folds: acc(b1) - acc(b2), sequence bootstrap
    rows_f = []
    for role, hand in groups:
        gsel = (meta.role == role) & (meta.hand == hand)
        idx = np.where(gsel)[0]
        mg = meta.iloc[idx].reset_index(drop=True)
        for target in ["verb", "mesh"]:
            y = mg[target].values.astype(str)
            keep = keep_classes(y, mg.sequence_id.values)
            sk = mg.sequence_id.values[keep]
            for b1 in BACKENDS:
                for b2 in BACKENDS:
                    if b1 == b2:
                        continue
                    c1 = correct_store[("a", b1, role, hand, target, "X_soft")]
                    c2 = correct_store[("a", b2, role, hand, target, "X_soft")]
                    d = boot_acc(c1.astype(float), sk) - boot_acc(c2.astype(float), sk)  # same draws (same seed)
                    rows_f.append(dict(role=role, hand=hand, target=target, backend=b1, minus=b2,
                                       diff=float(c1.mean() - c2.mean()), diff_lo=np.percentile(d, 2.5), diff_hi=np.percentile(d, 97.5)))
    pd.DataFrame(rows_f).to_csv(OUT / "probe_f_backend_paired.csv", index=False)
    log.info("done")


if __name__ == "__main__":
    main()
