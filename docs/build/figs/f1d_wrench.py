"""F1-D block: many dense contact reconfigurations preserve the modelled wrench capability.

For every contact event of the fixed test split, the wrench-counterfactual study compares the
grasp at the frame before the event (PRE) with the grasp at the frame after it (POST) under one
contact-patch wrench model (unit force budget per finger, Coulomb friction, torque normalised by
the object size). The capability profile of a grasp is its support-function capacity along 76
shared force / torque directions; Q is the mean of that profile. This module plots

    1. contact-map change against the relative change of Q, per event (two scatters),
    2. the profile-similarity medians (retention both ways, cosine) by event class,
    3. the share of events with a small / a large capability change by event class,
    4. the capability change by the number of reconfigured hand parts.

Source report
    reports/wrench_counterfactual/wrench_counterfactual_report.md
    (events.csv = per-event table; summary_by_event_type.csv = Table 1;
    finger_transition_summary.csv = Table 5; thresholds.json = train quantiles of Table 2;
    correlation_dC_vs_wrench.csv = the Spearman correlations of section 6).

TACO and ARCTIC are separate panels everywhere. OakInk2 is not part of this study. Wrench
capability is a modelled proxy in normalised units, not a measured force.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

REPORT = "reports/wrench_counterfactual"
EVENTS = f"{REPORT}/events.csv"
SUMMARY = f"{REPORT}/summary_by_event_type.csv"
FINGERS = f"{REPORT}/finger_transition_summary.csv"
THRESHOLDS = f"{REPORT}/thresholds.json"
CORRELATION = f"{REPORT}/correlation_dC_vs_wrench.csv"
SOURCES = [EVENTS, SUMMARY, FINGERS, THRESHOLDS, CORRELATION]

# (value of the `dataset` column, panel title)
DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]

# (value of the `class_report` column, category label), in the order of the report's Table 1
CLASSES = [
    ("persistent_spatial", "Persistent spatial"),
    ("persistent_mixed", "Persistent mixed"),
    ("onset", "Onset / regrasp"),
    ("release", "Release"),
    ("transient", "Transient"),
    ("amount", "Amount-dominant"),
]
SPATIAL = "persistent_spatial"
MIXED = "persistent_mixed"

# (column suffix in finger_transition_summary.csv, category label)
RECONFIGURED = [("changed0", "0"), ("changed1", "1"), ("changed2plus", "2 or more")]

X_CONTACT_CHANGE = "contact-map change of the event, ‖C_post − C_pre‖₂"
Y_REL_CHANGE = "relative change of wrench capability, (Q_post − Q_pre) / Q_pre"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(text: str) -> float | None:
    """A CSV cell as a float; an empty cell (no event in that group) is a missing value."""
    return float(text) if text.strip() else None


def _by_class(rows: list[dict[str, str]], dataset: str) -> dict[str, dict[str, str]]:
    """Rows of one dataset keyed by the report's event class."""
    return {row["class_report"]: row for row in rows if row["dataset"] == dataset}


def _points(rows: list[dict[str, str]]) -> list[list[float]]:
    """(contact-map change, relative capability change) of every event in `rows`, in file order."""
    return [[float(row["dC"]), float(row["rel_change_Q"])] for row in rows]


def _events_of(events: list[dict[str, str]], dataset: str, class_report: str) -> list[dict[str, str]]:
    return [row for row in events if row["dataset"] == dataset and row["class_report"] == class_report]


def _train_refs(threshold: dict[str, float]) -> list[dict]:
    """Reference lines from the TRAIN persistent-spatial events (the report's Table 2 thresholds)."""
    return [
        {"axis": "y", "value": 0.0, "label": "no change"},
        {"axis": "x", "value": threshold["dC_q50"], "label": "train median contact change"},
        {"axis": "y", "value": threshold["abs_rel_change_Q_q50"], "label": "train median |relative change|"},
        {"axis": "y", "value": -threshold["abs_rel_change_Q_q50"], "label": "minus the train median |relative change|"},
    ]


def _scatter_by_class(events: list[dict[str, str]], correlation: list[dict[str, str]]) -> dict:
    """Persistent spatial and persistent mixed events: contact change against capability change."""
    panels, counts, rhos = [], [], {SPATIAL: [], MIXED: []}
    for dataset, title in DATASETS:
        spatial = _events_of(events, dataset, SPATIAL)
        mixed = _events_of(events, dataset, MIXED)
        panels.append({
            "title": title,
            "x": {"label": X_CONTACT_CHANGE},
            "y": {"label": Y_REL_CHANGE, "direction": "none"},
            "points": [
                {"name": "Persistent spatial", "xy": _points(spatial)},
                {"name": "Persistent mixed", "xy": _points(mixed)},
            ],
            "refs": [{"axis": "y", "value": 0.0, "label": "no change"}],
        })
        counts.append(f"{title} {len(spatial)} spatial and {len(mixed)} mixed")
        for class_report in (SPATIAL, MIXED):
            rho = next(
                float(row["spearman_with_dC"]) for row in correlation
                if row["dataset"] == dataset and row["class_report"] == class_report and row["metric"] == "rel_change_Q"
            )
            rhos[class_report].append(f"{rho:.2f} ({title})")
    note = (
        "One point per test-split event: x is the Euclidean distance between the 512-D contact maps of the frame "
        "before and the frame after the event, y the relative change of the modelled wrench capability Q (mean "
        "support-function capacity over 76 shared force / torque directions; 0 = unchanged, +1 = doubled; neither "
        "direction is better). Events with firm contact throughout: " + ", ".join(counts) + "; Spearman correlation "
        "of x with y: spatial " + ", ".join(rhos[SPATIAL]) + "; mixed " + ", ".join(rhos[MIXED]) + "."
    )
    return {
        "id": "f1d_contact_change_vs_capability_change",
        "type": "scatter",
        "title": "Contact-map change versus change of the modelled wrench capability, per event",
        "panels": panels,
        "note": note,
        "sources": [
            {
                "path": EVENTS,
                "locator": (
                    "rows with class_report = persistent_spatial or persistent_mixed, per dataset; "
                    "columns dC (x) and rel_change_Q (y); every row plotted"
                ),
            },
            {
                "path": CORRELATION,
                "locator": "rows metric = rel_change_Q, class_report = persistent_spatial / persistent_mixed; column spearman_with_dC (quoted in the note)",
            },
        ],
    }


def _scatter_by_finger_set(events: list[dict[str, str]], thresholds: dict[str, dict[str, float]]) -> dict:
    """Persistent spatial events only, grouped by whether the set of contacting hand parts changed."""
    panels, counts = [], []
    for dataset, title in DATASETS:
        spatial = _events_of(events, dataset, SPATIAL)
        added = [row for row in spatial if int(row["n_appeared"]) > 0]
        removed_only = [row for row in spatial if int(row["n_appeared"]) == 0 and int(row["n_disappeared"]) > 0]
        same = [row for row in spatial if int(row["n_appeared"]) == 0 and int(row["n_disappeared"]) == 0]
        panels.append({
            "title": title,
            "x": {"label": X_CONTACT_CHANGE},
            "y": {"label": Y_REL_CHANGE, "direction": "none"},
            "points": [
                {"name": "Same hand parts", "xy": _points(same)},
                {"name": "Part added", "xy": _points(added)},
                {"name": "Part removed only", "xy": _points(removed_only)},
            ],
            "refs": _train_refs(thresholds[dataset]),
        })
        counts.append(f"{title} {len(same)} / {len(added)} / {len(removed_only)}")
    note = (
        "Persistent spatial test events only, split by whether the set of hand parts touching the object (palm and "
        "five fingers) is the same before and after the event, gained at least one part, or only lost parts; x is the "
        "distance between the 512-D contact maps before and after, y the relative change of the modelled wrench "
        "capability Q (mean support-function capacity over 76 shared directions; 0 = unchanged; neither direction is "
        "better). Events per group (same / added / removed only): " + "; ".join(counts) + "; the reference lines are "
        "no change (y = 0) and medians of the train-split persistent spatial events (thresholds.json); the page "
        "draws the no-change line only."
    )
    return {
        "id": "f1d_spatial_events_by_finger_set",
        "type": "scatter",
        "title": "Persistent spatial events: capability change with and without a change of the contacting hand parts",
        "panels": panels,
        "note": note,
        "sources": [
            {
                "path": EVENTS,
                "locator": (
                    "rows class_report = persistent_spatial, per dataset; columns dC (x), rel_change_Q (y); groups "
                    "from n_appeared and n_disappeared: same = both 0, added = n_appeared > 0, removed only = "
                    "n_appeared = 0 and n_disappeared > 0"
                ),
            },
            {
                "path": THRESHOLDS,
                "locator": "<dataset>.dC_q50 (x reference) and <dataset>.abs_rel_change_Q_q50 (y references, plus and minus)",
            },
        ],
    }


def _series(rows: dict[str, dict[str, str]], column: str, name: str, interval: bool) -> dict:
    """One bar series over the event classes; `interval` adds the table's bootstrap bounds."""
    series: dict = {"name": name, "values": [_number(rows[key][column]) for key, _ in CLASSES]}
    if interval:
        series["lo"] = [_number(rows[key][f"{column}_lo"]) for key, _ in CLASSES]
        series["hi"] = [_number(rows[key][f"{column}_hi"]) for key, _ in CLASSES]
    return series


def _class_counts(summary: list[dict[str, str]]) -> str:
    """Events per class, in category order, for the notes."""
    parts = []
    for dataset, title in DATASETS:
        rows = _by_class(summary, dataset)
        parts.append(f"{title} " + " / ".join(rows[key]["n_events"] for key, _ in CLASSES))
    return "; ".join(parts)


def _bar_similarity(summary: list[dict[str, str]]) -> dict:
    """Median retention (both directions) and profile cosine by event class, with the table's intervals."""
    panels = []
    for dataset, title in DATASETS:
        rows = _by_class(summary, dataset)
        panels.append({
            "title": title,
            "x": {"label": "event class", "categories": [label for _, label in CLASSES]},
            # Not "1 = preserved": retention is one-sided, so a release scores pre→post = 1 with no grasp left.
            "y": {"label": "median over events (near 1 = profiles alike; retention is one-sided, see note)", "direction": "none"},
            "series": [
                _series(rows, "R_pre_to_post_median", "Retention pre→post", interval=True),
                _series(rows, "R_post_to_pre_median", "Retention post→pre", interval=True),
                _series(rows, "cosine_median", "Profile cosine", interval=True),
            ],
            "highlight": {"category": "Persistent spatial", "label": "primary class"},
        })
    note = (
        "Medians over test-split events of three similarity measures between the grasp's wrench-capability profile "
        "(support-function capacity along 76 shared force / torque directions) before and after an event; error bars "
        "are 95 % intervals of a bootstrap that resamples whole takes (1 000 replicates). Retention pre→post is the "
        "share of the post grasp's capability that the pre grasp already had, direction by direction, and post→pre "
        "the reverse: both are one-sided, so a release (no grasp afterwards) has pre→post = 1 and an onset from no "
        "contact has 0 by construction. Events per class (in axis order): " + _class_counts(summary) + "; the TACO "
        "transient class is a random subsample of " + _by_class(summary, "taco")["transient"]["n_events"] + " events."
    )
    return {
        "id": "f1d_retention_cosine_by_event_class",
        "type": "bar",
        "title": "How much of the wrench-capability profile is preserved, by event class",
        "panels": panels,
        "note": note,
        "sources": [
            {
                "path": SUMMARY,
                "locator": (
                    "one row per dataset x class_report (persistent_spatial, persistent_mixed, onset, release, "
                    "transient, amount); columns R_pre_to_post_median, R_post_to_pre_median, cosine_median with "
                    "their _lo / _hi bounds; n_events"
                ),
            },
        ],
    }


def _bar_shares(summary: list[dict[str, str]]) -> dict:
    """Share of events with a small capability change and with a large gain, by event class."""
    panels = []
    for dataset, title in DATASETS:
        rows = _by_class(summary, dataset)
        panels.append({
            "title": title,
            "x": {"label": "event class", "categories": [label for _, label in CLASSES]},
            "y": {"label": "share of events", "direction": "none"},
            "series": [
                _series(rows, "frac_abs_rel_change_le_0.25", "Change within ±25 %", interval=False),
                _series(rows, "frac_rel_change_gt_0.25", "Gain above 25 %", interval=False),
            ],
            "highlight": {"category": "Persistent spatial", "label": "primary class"},
        })
    note = (
        "Share of test-split events whose modelled wrench capability Q (mean support-function capacity over 76 "
        "shared force / torque directions) changes by at most 25 % in either direction, and share that gains more "
        "than 25 %; the rest loses more than 25 % (an onset from no contact counts as a gain, a release loses all). "
        "Events per class (in axis order): " + _class_counts(summary) + "; the TACO transient class is a random subsample of "
        + _by_class(summary, "taco")["transient"]["n_events"] + " events; the source table gives no interval for these shares."
    )
    return {
        "id": "f1d_share_capability_change_by_event_class",
        "type": "bar",
        "title": "How often the modelled wrench capability changes, by event class",
        "panels": panels,
        "note": note,
        "sources": [
            {
                "path": SUMMARY,
                "locator": (
                    "one row per dataset x class_report; columns frac_abs_rel_change_le_0.25 and "
                    "frac_rel_change_gt_0.25; n_events"
                ),
            },
        ],
    }


def _bar_reconfigured(fingers: list[dict[str, str]], column: str, figure_id: str, title: str, y_label: str, lead: str) -> dict:
    """One Table-5 quantity by the number of reconfigured hand parts, spatial and mixed classes."""
    panels, counts = [], []
    for dataset, panel_title in DATASETS:
        rows = _by_class(fingers, dataset)
        series = []
        for class_report, name in ((SPATIAL, "Persistent spatial"), (MIXED, "Persistent mixed")):
            series.append({"name": name, "values": [_number(rows[class_report][f"{column}_{suffix}"]) for suffix, _ in RECONFIGURED]})
            n_events = " / ".join(rows[class_report][f"n_events_{suffix}"] for suffix, _ in RECONFIGURED)
            counts.append(f"{panel_title} {name.lower()} {n_events}")
        panels.append({
            "title": panel_title,
            "x": {"label": "hand parts reconfigured in the event", "categories": [label for _, label in RECONFIGURED]},
            "y": {"label": y_label, "direction": "none"},
            "series": series,
        })
    note = (
        lead + " A hand part (palm or one of five fingers) counts as reconfigured when its contact-patch centroid "
        "moved more than 1 cm on the object, or when it touches the object only before or only after the event. "
        "Events per bar (0 / 1 / 2 or more): " + "; ".join(counts) + "; the source table gives no interval."
    )
    return {
        "id": figure_id,
        "type": "bar",
        "title": title,
        "panels": panels,
        "note": note,
        "sources": [
            {
                "path": FINGERS,
                "locator": (
                    "rows class_report = persistent_spatial and persistent_mixed, per dataset; columns "
                    f"{column}_changed0 / _changed1 / _changed2plus and n_events_changed0 / 1 / 2plus"
                ),
            },
        ],
    }


def build(src: Callable[[str], Path]) -> list[dict]:
    events = _read_csv(src(EVENTS))
    summary = _read_csv(src(SUMMARY))
    fingers = _read_csv(src(FINGERS))
    correlation = _read_csv(src(CORRELATION))
    thresholds = json.loads(src(THRESHOLDS).read_text(encoding="utf-8"))
    return [
        _scatter_by_class(events, correlation),
        _scatter_by_finger_set(events, thresholds),
        _bar_similarity(summary),
        _bar_shares(summary),
        _bar_reconfigured(
            fingers,
            column="median_abs_rel_change_Q",
            figure_id="f1d_capability_change_by_reconfigured_parts",
            title="Size of the capability change by the number of reconfigured hand parts",
            y_label="median |relative change of wrench capability|",
            lead=(
                "Median over test-split events of the absolute relative change of the modelled wrench capability Q "
                "(mean support-function capacity over 76 shared force / torque directions; 0 = unchanged)."
            ),
        ),
        _bar_reconfigured(
            fingers,
            column="median_retention",
            figure_id="f1d_retention_by_reconfigured_parts",
            title="Retention of the wrench-capability profile by the number of reconfigured hand parts",
            y_label="median retention pre→post (1 = fully retained)",
            lead=(
                "Median over test-split events of the pre→post retention: the share of the post grasp's "
                "support-function capacity, over 76 shared force / torque directions, that the pre grasp already had."
            ),
        ),
    ]
