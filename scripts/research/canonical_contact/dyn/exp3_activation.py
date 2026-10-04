#!/usr/bin/env python
"""Zero-contact activation from the Experiment-3 predictions: the regression models never output
a mass below ZERO_MASS (=1.5), so the F1 of the zero class at the fixed threshold is 0 by
construction. This tabulates precision / recall / F1 of the zero class as a function of the
predicted-mass threshold (post hoc, on the test frames, so it is an upper bound), per split and
model, pooled over groups and folds; and the AUROC of predicted mass as a zero-contact score.
"""
import numpy as np
import pandas as pd

import dc_common as C

PRED = C.OUT / "exp3" / "preds"
MODELS = ["mlp_A_geometry", "mlp_B_current_state", "mlp_C_local_context", "mlp_D_full_window", "mlp_Dc_coarse_window"]
THRS = [1.5, 3, 5, 10, 20, 30]


def auroc(score, label):
    """AUROC of score for label==1 (higher score = more likely 1)."""
    o = np.argsort(score); r = np.empty(len(score)); r[o] = np.arange(1, len(score) + 1)
    n1 = label.sum(); n0 = len(label) - n1
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def main():
    thr = C.zero_threshold()
    rows = []
    for cat, role, hand in C.GROUPS:
        g = C.gname(cat, role, hand)
        X, M, P = C.load_group(cat, role, hand, stride=C.STRIDE_PAIRS)
        key = {(s, int(f)): i for i, (s, f) in enumerate(zip(M.sequence_id.values, M.frame.values))}
        mt_all = C.mass(X); zero_all = C.is_zero(mt_all, M.n_hard.values, thr)
        for split in ("take", C.SPLIT2):
            for k in range(3):
                p = PRED / f"{g}__{split}{k}.npz"
                if not p.exists():
                    continue
                z = np.load(p, allow_pickle=True)
                j = np.array([key[(s, int(f))] for s, f in zip(z["sequence_id"].astype(str), z["frame"])])
                zt = zero_all[j]
                for m in MODELS:
                    mp = C.mass(np.clip(z[m].astype(np.float32), 0, None))
                    rec = dict(group=g, split=split, fold=k, model=m, n=len(j), zero_frac=float(zt.mean()),
                               auroc=auroc(-mp, zt))
                    for t in THRS:
                        zp = mp < t
                        tp = (zt & zp).sum(); fp = (~zt & zp).sum(); fn = (zt & ~zp).sum()
                        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
                        rec[f"f1_thr{t}"] = 2 * pr * rc / max(pr + rc, 1e-12); rec[f"prec_thr{t}"] = pr; rec[f"rec_thr{t}"] = rc
                    rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(C.OUT / "exp3" / "activation_by_threshold.csv", index=False)
    pd.set_option("display.width", 250)
    S = R.groupby(["split", "model"]).agg(zero_frac=("zero_frac", "mean"), auroc=("auroc", "mean"),
                                         **{f"f1_thr{t}": (f"f1_thr{t}", "mean") for t in THRS}).reset_index()
    S.to_csv(C.OUT / "exp3" / "activation_summary.csv", index=False)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
