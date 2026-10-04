"""MOTIVATION block: hand error with the ground-truth contact map versus the predicted one.

BimArt is a two-stage generator: a contact stage predicts a contact map from the object, and a
motion stage generates the hands conditioned on that map. This module plots the motion stage's
hand-keypoint error under the two conditionings, one panel per dataset.

Source reports
    TACO    reports/bimart_taco_and_canonical_contact.md (Part A) and
            taco/30_bimart_gen3_scene_scale/scene_reverification.md (section 1). The table is the
            generation-3 evaluation written by scripts/eval_bimart_taco.py.
    ARCTIC  arctic/10_bimart_upstream_analysis (contact ablation of the released upstream BimArt,
            written by scripts/viz_bimart_contact_ablation.py).

The two panels use different metrics and sample sizes (mean keypoint distance on sampled windows
per split for TACO, per-hand RMSE on a handful of windows for ARCTIC). They are never pooled, and
their levels are not comparable with each other.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

TACO_EVAL = "taco/30_bimart_gen3_scene_scale/evaluation/test_split_results.csv"
ARCTIC_ABLATION = "arctic/10_bimart_upstream_analysis/contact_ablation/final/summary.json"
SOURCES = [TACO_EVAL, ARCTIC_ABLATION]

TACO_SPLITS = ["test_1", "test_2", "test_3", "test_4"]
TACO_GT_COLUMN = "motion_kp_err_mm_gt_contact"
TACO_PRED_COLUMN = "motion_kp_err_mm_pred_contact"
ARCTIC_GT_ARM = "GT contact"
ARCTIC_PRED_ARM = "stage-1 contact"

SERIES_GT = "GT contact"
SERIES_PRED = "predicted contact"


def _taco_panel(path: Path) -> tuple[dict, str]:
    """Generation-3 TACO evaluation: one bar pair per official test split."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["split"]: row for row in csv.DictReader(handle)}
    sampled = sorted({int(rows[split]["sampled"]) for split in TACO_SPLITS})
    panel = {
        "title": "TACO",
        "x": {"label": "test split (BimArt port, generation 3)", "categories": list(TACO_SPLITS)},
        "y": {"label": "hand keypoint error, mean distance (mm)", "direction": "lower_better"},
        "series": [
            {"name": SERIES_GT, "values": [float(rows[s][TACO_GT_COLUMN]) for s in TACO_SPLITS]},
            {"name": SERIES_PRED, "values": [float(rows[s][TACO_PRED_COLUMN]) for s in TACO_SPLITS]},
        ],
    }
    return panel, " / ".join(str(n) for n in sampled)


def _arctic_panel(path: Path) -> tuple[dict, int, int, int]:
    """Upstream BimArt contact ablation on ARCTIC: one bar pair per window, in file order."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    windows: dict[str, dict] = summary["windows"]
    names = list(windows)
    # a window name is "<sequence>_<start frame>"
    n_sequences = len({name.rsplit("_", 1)[0] for name in names})
    panel = {
        "title": "ARCTIC",
        "x": {"label": "laptop test window (released upstream BimArt)", "categories": names},
        "y": {"label": "hand keypoint error, RMSE (mm)", "direction": "lower_better"},
        "series": [
            {"name": SERIES_GT, "values": [float(windows[n][ARCTIC_GT_ARM]["total_mm"]) for n in names]},
            {"name": SERIES_PRED, "values": [float(windows[n][ARCTIC_PRED_ARM]["total_mm"]) for n in names]},
        ],
    }
    return panel, len(names), n_sequences, int(summary["samples_per_arm"])


def build(src: Callable[[str], Path]) -> list[dict]:
    taco, taco_sampled = _taco_panel(src(TACO_EVAL))
    arctic, n_windows, n_sequences, n_samples = _arctic_panel(src(ARCTIC_ABLATION))
    note = (
        "Position error of the generated hand keypoints in mm (lower is better) when the motion stage is "
        "conditioned on the ground-truth contact map versus on the contact stage's own prediction, with the "
        "same motion model and sampling procedure in both arms. "
        f"TACO: mean distance over 200 keypoints and 64 frames, {taco_sampled} sampled windows per split, "
        f"one sample each; ARCTIC: per-hand RMSE on {n_windows} windows from {n_sequences} laptop sequences, "
        f"{n_samples} samples per arm; the metrics differ, so the panels are not comparable in level, and "
        "there are no error bars because neither table gives an interval. "
        "The ground-truth arm is an oracle: its contact map is computed from the recorded hands. "
        "TACO: this project's port of BimArt, one training run; test_1 was the validation split used to "
        "select the checkpoints, so test_2 to test_4 are the unbiased splits. "
        "ARCTIC: the released checkpoints, with upstream's cost guidance, which also reads the arm's "
        "contact map; the two TACO columns are sampled without cost guidance."
    )
    figure = {
        "id": "m0_hand_error_gt_vs_predicted_contact",
        "type": "bar",
        "title": "Hand error with the ground-truth contact map versus the predicted one",
        "panels": [taco, arctic],
        "note": note,
        "sources": [
            {
                "path": TACO_EVAL,
                "locator": (
                    f"rows split = {', '.join(TACO_SPLITS)}; columns {TACO_GT_COLUMN} (GT contact) and "
                    f"{TACO_PRED_COLUMN} (predicted contact); column sampled = windows scored"
                ),
            },
            {
                "path": ARCTIC_ABLATION,
                "locator": (
                    f"windows.<window>.'{ARCTIC_GT_ARM}'.total_mm and windows.<window>.'{ARCTIC_PRED_ARM}'.total_mm, "
                    "all windows in file order; samples_per_arm"
                ),
            },
        ],
    }
    return [figure]
