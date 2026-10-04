"""Figure data for page block F1-subB: what hand information adds to predicting future contact change.

Source report: reports/hand_contact_predictive_info/hand_contact_predictive_info_report.md
(Table 1 overall error, Table 3 transition-start prediction, Table 4 paired differences,
Table 5 control and oracles). Datasets: TACO and ARCTIC; the study has no OakInk2 run.

Every plotted value is a cell of the per-dataset result tables written by
scripts/research/hand_contact_predictive_info/evaluate.py. One bar is not a single cell and is
marked where it is built: the tables hold no paired row "one-future-frame oracle vs no hand", so
that bar is the sum of the two paired rows that chain it (oracle vs motion only, motion only vs
no hand). Both rows are in % of the no-hand error on the same frames, so the sum is exact; it
equals the "oracle +1 frame" column of the report's Table 5.

The by-class figure shows every frame class of the table (the report's own tables leave the
amount-dominant class out); the classes are mutually exclusive and the note states how many
frames fall in none of them.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

Src = Callable[[str], Path]

REPORT_DIR = "reports/hand_contact_predictive_info"
DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]
TABLES = [
    "incremental_information.csv",
    "overall_delta_results.csv",
    "transition_prediction_results.csv",
    "transition_incremental.csv",
    "evaluation_config.json",
]
SOURCES = [f"{REPORT_DIR}/{ds}/results/{table}" for ds, _ in DATASETS for table in TABLES]

HORIZON = "4"  # the report's main horizon: contact change over 4 frames (30 Hz)
NO_HAND = ("no hand", "F0")  # the predictor that sees contact, geometry and object motion only
CONDITIONS = [  # (label on the page, condition id in the tables)
    ("current pose", "F1"),
    ("history (8 fr.)", "F2k8"),
    ("motion only", "F2rel"),
    ("shuffled (control)", "F2shuf"),
    ("future +1 (oracle)", "Ffut1"),
    ("future +8 (oracle)", "Ffut"),
]
# No direct paired row "Ffut1-F0" exists: chain it through the motion-only model.
CHAINED_PAIRS = {"Ffut1": ("Ffut1-F2rel", "F2rel-F0")}
FRAME_CLASSES = [  # (label on the page, frame_class in the tables)
    ("all frames", "all"),
    ("quiet", "non_spike"),
    ("transient", "transient"),
    ("persistent spatial", "persistent_spatial"),
    ("persistent mixed", "persistent_mixed"),
    ("amount-dominant", "amount"),
    ("onset", "onset"),
    ("release", "release"),
]
CLASS_SERIES = [CONDITIONS[1], CONDITIONS[2], CONDITIONS[5]]
ORACLE_HIGHLIGHT = {"category": CONDITIONS[5][0], "label": "non-causal: is given future hand frames"}
# What each label means (hp_common.CONDITIONS / window_offsets of the study's scripts). Definitions only, no results.
INPUT_DEFS = (
    "Inputs (ground-truth positions of 100 hand-surface points in the contacted object's frame, t = current frame): "
    "current pose = frame t; history (8 fr.) = frames t-8..t; motion only = the 8 frame-to-frame displacements up "
    "to t, without absolute position; shuffled = the history of another sequence (same size, no information); "
    "future +1 / +8 = the history plus 1 / 8 future frames."
)
# How a frame gets its class (hp_common.frame_classes and the temporal-event study's rules). Definitions only.
CLASS_DEFS = (
    "Frame class = the largest spike event overlapping the window (spike = a frame-to-frame contact change above the "
    "training 90th percentile): quiet = none; transient = an event of low persistence (net change over +-4 frames "
    "under half of the summed frame-to-frame change), whatever its type; otherwise onset / release of firm contact, "
    "else by the share of the change that is a change of total contact amount: amount-dominant (>= 0.7), persistent "
    "spatial (<= 0.3), persistent mixed (between)."
)


def _path(ds: str, table: str) -> str:
    return f"{REPORT_DIR}/{ds}/results/{table}"


def _rows(path: Path, **where: str) -> list[dict[str, str]]:
    """Rows of a CSV table whose columns equal the given strings."""
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if all(row[key] == value for key, value in where.items())]


def _one(rows: list[dict[str, str]], **where: str) -> dict[str, str]:
    hits = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one row for {where}, found {len(hits)}")
    return hits[0]


def _pairs_vs_no_hand(cond: str) -> tuple[str, ...]:
    return CHAINED_PAIRS.get(cond, (f"{cond}-{NO_HAND[1]}",))


def _sample(src: Src, ds: str) -> str:
    config = json.loads(src(_path(ds, "evaluation_config.json")).read_text(encoding="utf-8"))
    return f"{config['n_test']} test sequences from {config['n_takes']} takes, {len(config['seeds'])} seeds"


def _listing(parts: list[str]) -> str:
    return ", ".join(parts) if parts else "none"


def _error_change_vs_no_hand(src: Src) -> dict:
    """Bar: change of the 4-frame contact-change error against the no-hand predictor, per hand input."""
    panels, samples, not_significant, higher_error = [], [], [], []
    for ds, title in DATASETS:
        rows = _rows(src(_path(ds, "incremental_information.csv")), h=HORIZON, frame_class="all")
        values, unclear, worse = [], [], []
        for label, cond in CONDITIONS:
            cells = [_one(rows, pair=pair) for pair in _pairs_vs_no_hand(cond)]
            values.append(sum(float(cell["rel_to_F0_pct"]) for cell in cells))
            if len(cells) != 1:
                continue  # chained bar: no interval of its own
            if cells[0]["significant"] != "True":
                unclear.append(label)
            elif float(cells[0]["diff"]) > 0:
                worse.append(label)
        panels.append({
            "title": title,
            "x": {"label": "hand input added to the predictor", "categories": [label for label, _ in CONDITIONS]},
            "y": {"label": "error change vs no-hand predictor (% of its error)", "direction": "lower_better"},
            "series": [{"name": f"{HORIZON}-frame horizon", "values": values}],
            "highlight": ORACLE_HIGHLIGHT,
        })
        samples.append(f"{title} {rows[0]['n_frames']} test frames ({_sample(src, ds)})")
        not_significant.append(f"{title}: {_listing(unclear)}")
        higher_error.append(f"{title}: {_listing(worse)}")
    return {
        "id": "f1subb_delta_error_vs_no_hand",
        "type": "bar",
        "title": "Future contact change: what each hand input adds to a predictor without the hand",
        "panels": panels,
        "note": (
            "Paired change of the mean L2 error of the predicted contact change over the next "
            f"{HORIZON} frames, against the same predictor without hand input (it sees the current contact map, the "
            "object geometry and the object's motion from 8 frames back to 8 ahead), in % of that predictor's error "
            "(negative = lower error); the two 'future' bars are non-causal oracles. "
            f"{INPUT_DEFS} {'; '.join(samples)}. No error bars: the table gives the paired take-bootstrap 95 % "
            "interval of the seed-mean error in raw units only (no seed-to-seed variance); it includes 0 for "
            f"{'; '.join(not_significant)}, and lies above 0 (higher error than without the hand) for "
            f"{'; '.join(higher_error)}. 'future +1' is the sum of two paired rows and has no interval. How the "
            "no-hand predictor itself compares with predicting no change is stated in the text under the chart."
        ),
        "sources": [
            {
                "path": _path(ds, "incremental_information.csv"),
                "locator": (
                    f"rows h={HORIZON}, frame_class=all; column rel_to_F0_pct of pairs F1-F0, F2k8-F0, F2rel-F0, "
                    "F2shuf-F0, Ffut-F0; 'future +1' = rel_to_F0_pct(Ffut1-F2rel) + rel_to_F0_pct(F2rel-F0); "
                    "columns significant, n_frames for the note"
                ),
            }
            for ds, _ in DATASETS
        ] + [
            {"path": _path(ds, "evaluation_config.json"), "locator": "keys n_test, n_takes, seeds (note only)"}
            for ds, _ in DATASETS
        ],
    }


def _error_change_by_frame_class(src: Src) -> dict:
    """Bar: the same paired change, split by the kind of contact-change event inside the window."""
    panels, counts, not_significant, higher_error, unclassified = [], [], [], [], []
    for ds, title in DATASETS:
        rows = _rows(src(_path(ds, "incremental_information.csv")), h=HORIZON)
        series, unclear, worse, sizes, rest = [], [], [], [], 0
        for label, cond in CLASS_SERIES:
            cells = [_one(rows, frame_class=cls, pair=f"{cond}-{NO_HAND[1]}") for _, cls in FRAME_CLASSES]
            series.append({"name": label, "values": [float(cell["rel_to_F0_pct"]) for cell in cells]})
            classes = [cls_label for (cls_label, _), cell in zip(FRAME_CLASSES, cells) if cell["significant"] != "True"]
            if classes:
                unclear.append(f"{label} on {' / '.join(classes)}")
            classes = [
                cls_label for (cls_label, _), cell in zip(FRAME_CLASSES, cells)
                if cell["significant"] == "True" and float(cell["diff"]) > 0
            ]
            if classes:
                worse.append(f"{label} on {' / '.join(classes)}")
            # every pair of one frame class is evaluated on the same frames, so any series gives the class sizes
            sizes = [f"{cls_label} {cell['n_frames']}" for (cls_label, _), cell in zip(FRAME_CLASSES, cells)]
            rest = sum(int(cell["n_frames"]) * (1 if cls == "all" else -1) for (_, cls), cell in zip(FRAME_CLASSES, cells))
        panels.append({
            "title": title,
            "x": {"label": f"largest contact-change event inside the {HORIZON}-frame window", "categories": [label for label, _ in FRAME_CLASSES]},
            "y": {"label": "error change vs no-hand predictor (% of its error)", "direction": "lower_better"},
            "series": series,
        })
        counts.append(f"{title} {', '.join(sizes)}")
        not_significant.append(f"{title}: {_listing(unclear)}")
        higher_error.append(f"{title}: {_listing(worse)}")
        unclassified.append(f"{title} {rest}")
    return {
        "id": "f1subb_delta_error_by_frame_class",
        "type": "bar",
        "title": "Where hand information helps: error change by kind of contact change",
        "panels": panels,
        "note": (
            f"Paired change of the mean L2 error of the predicted {HORIZON}-frame contact change against the no-hand "
            "predictor, in % of that predictor's error on the same frames (negative = lower error); ground-truth hand: "
            "history (8 fr.) = positions at frames t-8..t, motion only = their 8 frame-to-frame displacements, "
            "future +8 = the history plus 8 future frames (non-causal oracle). "
            f"{CLASS_DEFS} Test frames: {'; '.join(counts)} (exclusive classes; frames in none of them: "
            f"{', '.join(unclassified)}). No error bars; the paired take-bootstrap 95 % interval includes 0 for "
            f"{'; '.join(not_significant)}; lies above 0 (higher error than without the hand) for "
            f"{'; '.join(higher_error)}; lies below 0 for every other bar."
        ),
        "sources": [
            {
                "path": _path(ds, "incremental_information.csv"),
                "locator": (
                    f"rows h={HORIZON}, pairs F2k8-F0, F2rel-F0, Ffut-F0, frame_class in all, non_spike, transient, "
                    "persistent_spatial, persistent_mixed, amount, onset, release; column rel_to_F0_pct; columns "
                    "significant, diff, n_frames for the note"
                ),
            }
            for ds, _ in DATASETS
        ],
    }


def _transition_start_auprc(src: Src) -> dict:
    """Bar: AUPRC for 'a persistent contact-mode transition starts within the next frames'."""
    conditions = [NO_HAND] + CONDITIONS
    panels, counts, paired = [], [], []
    for ds, title in DATASETS:
        rows = _rows(src(_path(ds, "transition_prediction_results.csv")), h=HORIZON)
        cells = [_one(rows, cond=cond) for _, cond in conditions]
        diffs = _rows(src(_path(ds, "transition_incremental.csv")), h=HORIZON, metric="auprc", cond_b=NO_HAND[1])
        higher, lower, unclear = [], [], []
        for label, cond in CONDITIONS:
            hits = [row for row in diffs if row["cond_a"] == cond]
            if not hits:
                continue
            if hits[0]["significant"] != "True":
                unclear.append(label)
            else:
                (higher if float(hits[0]["diff"]) > 0 else lower).append(label)
        panels.append({
            "title": title,
            "x": {"label": "hand input given to the classifier", "categories": [label for label, _ in conditions]},
            "y": {"label": f"AUPRC: transition starts within {HORIZON} frames", "direction": "higher_better"},
            "series": [{
                "name": f"{HORIZON}-frame horizon",
                "values": [float(cell["auprc"]) for cell in cells],
                "lo": [float(cell["auprc_lo"]) for cell in cells],
                "hi": [float(cell["auprc_hi"]) for cell in cells],
            }],
            "refs": [{"axis": "y", "value": float(cells[0]["prevalence"]), "label": "chance (share of positive frames)"}],
            "highlight": ORACLE_HIGHLIGHT,
        })
        counts.append(f"{title} {cells[0]['n_pos']} positive of {cells[0]['n_frames']} test frames ({_sample(src, ds)})")
        paired.append(f"{title} higher: {_listing(higher)}, lower: {_listing(lower)}, interval includes 0: {_listing(unclear)}")
    return {
        "id": "f1subb_transition_start_auprc",
        "type": "bar",
        "title": "Predicting that a persistent contact transition is about to start",
        "panels": panels,
        "note": (
            "Area under the precision-recall curve (mean of the seeds; higher = better; chance = share of positive "
            f"frames) for 'a persistent contact-mode transition starts within the next {HORIZON} frames': frame t is "
            "positive when a persistent spatial or persistent mixed event, or any release / onset (regrasp) event, "
            f"has its first large change at a step s -> s + 1 with t < s <= t + {HORIZON}. The same small classifier "
            f"is used for every input. {INPUT_DEFS} {'; '.join(counts)}. Error bars: take-bootstrap 95 % interval "
            "of each condition, not a paired test. Paired difference against 'no hand' (same bootstrap, no row for "
            f"'future +1'): {'; '.join(paired)}."
        ),
        "sources": [
            {
                "path": _path(ds, "transition_prediction_results.csv"),
                "locator": (
                    f"rows h={HORIZON}, cond F0, F1, F2k8, F2rel, F2shuf, Ffut1, Ffut; columns auprc, auprc_lo, "
                    "auprc_hi; reference line = column prevalence; n_pos, n_frames for the note"
                ),
            }
            for ds, _ in DATASETS
        ] + [
            {
                "path": _path(ds, "transition_incremental.csv"),
                "locator": f"rows h={HORIZON}, metric=auprc, cond_b=F0; columns diff, significant (note only)",
            }
            for ds, _ in DATASETS
        ] + [
            {"path": _path(ds, "evaluation_config.json"), "locator": "keys n_test, n_takes, seeds (note only)"}
            for ds, _ in DATASETS
        ],
    }


def _improvement_over_no_change(src: Src) -> dict:
    """Bar: every predictor against the trivial 'contact does not change' prediction."""
    conditions = [NO_HAND] + CONDITIONS
    panels, counts = [], []
    for ds, title in DATASETS:
        rows = _rows(src(_path(ds, "overall_delta_results.csv")), h=HORIZON, frame_class="all")
        cells = [_one(rows, cond=cond) for _, cond in conditions]
        panels.append({
            "title": title,
            "x": {"label": "hand input added to the predictor", "categories": [label for label, _ in conditions]},
            "y": {"label": "share of the no-change error removed (fraction)", "direction": "higher_better"},
            "series": [{
                "name": f"{HORIZON}-frame horizon",
                "values": [float(cell["improvement"]) for cell in cells],
                "lo": [float(cell["improvement_lo"]) for cell in cells],
                "hi": [float(cell["improvement_hi"]) for cell in cells],
            }],
            "highlight": ORACLE_HIGHLIGHT,
        })
        counts.append(f"{title} {cells[0]['n_frames']} test frames from {cells[0]['n_takes']} takes")
    return {
        "id": "f1subb_delta_error_vs_no_change",
        "type": "bar",
        "title": "The same predictors against the trivial prediction 'contact does not change'",
        "panels": panels,
        "note": (
            f"Share of the summed L2 error of predicting no contact change over {HORIZON} frames that each predictor "
            "removes (1 - sum of its error / sum of the no-change error; positive = better than predicting no change, "
            "negative = worse); mean of the seeds, ground-truth hand, inputs as in f1subb_delta_error_vs_no_hand. "
            f"{'; '.join(counts)}. Error bars: take-cluster bootstrap 95 % interval of each condition's ratio, not a "
            "paired test between conditions. This is the L2 error; the models were trained on the squared error, "
            "whose columns of the same table (improvement_sq) are not plotted."
        ),
        "sources": [
            {
                "path": _path(ds, "overall_delta_results.csv"),
                "locator": (
                    f"rows h={HORIZON} (frame_class=all), cond F0, F1, F2k8, F2rel, F2shuf, Ffut1, Ffut; columns "
                    "improvement, improvement_lo, improvement_hi; n_frames, n_takes for the note"
                ),
            }
            for ds, _ in DATASETS
        ],
    }


def build(src: Src) -> list[dict]:
    return [
        _error_change_vs_no_hand(src),
        _error_change_by_frame_class(src),
        _transition_start_auprc(src),
        _improvement_over_no_change(src),
    ]
