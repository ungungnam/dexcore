"""Figure data for page blocks F1-C and F1-subA (temporal contact-change events).

F1-C    large contact changes are concentrated in few transitions, and the evolved model's
        temporal error by event class (static baseline B0 vs GT-initialised vector field B1).
F1-subA hand-object relative motion is associated with the contact change (rank correlation).

Source report: reports/temporal_contact_events/temporal_contact_event_report.md (analysis of the
cached hierarchical-generation sequences and predictions, nothing retrained). The corrected TACO
hand-relative-speed correlation comes from the follow-up study's sanity file
reports/hand_contact_predictive_info/taco/sanity/hand_alignment.json (the temporal-event tables
still hold the value computed with the wrong tool-frame rotation layout).

Every plotted value is read from the tables below at run time. The timeline figure reads
<dataset>/frames_test.csv, which is larger than the runner's 3 MB snapshot limit: the snapshot of
that table holds the header and the rows of the one plotted example sequence only.
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from typing import Callable

Src = Callable[..., Path]

TCE = "reports/temporal_contact_events"
HAND_FIX = "reports/hand_contact_predictive_info/taco/sanity/hand_alignment.json"

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC"), ("oakink2", "OakInk2")]

# Event types of the report, in its priority order (the single ARCTIC "near_zero" event is left out).
EVENT_TYPES = ["onset", "release", "onset+release", "amount", "spatial", "mixed"]

# Frame classes of the report's Figure F: (frame_class in the table, label on the page).
FRAME_CLASSES = [
    ("non_spike", "non-spike"),
    ("spike", "all spike"),
    ("amount", "amount"),
    ("onset", "onset"),
    ("release", "release"),
    ("mixed", "mixed"),
    ("spatial", "spatial"),
    ("transient", "low-persistence"),
]

# Models compared: (column prefix in model_error_by_event.csv, series name).
MODELS = [
    ("e_B0_static", "hold the first map (static baseline)"),
    ("e_B1_gtinit_vf", "evolved from the true first map"),
]

# Kinematic signals: (signal in kinematic_associations.csv, label on the page).
SIGNALS = [
    ("obj_lin_speed", "object linear speed"),
    ("obj_ang_speed", "object angular speed"),
    ("tool_rel_lin_speed", "tool-vs-target linear speed"),
    ("tool_rel_ang_speed", "tool-vs-target angular speed"),
    ("arti_rate", "articulation rate"),
    ("hand_rel_speed", "hand-relative speed"),
]
HAND_SIGNAL = "hand_rel_speed"

# Timeline examples: the test sequence that holds each dataset's representative spatial-dominant
# event of the report's Figure G (figG_02_spatial_rep) and is described in report section 6.
TIMELINE_EXAMPLES = [("taco", "TACO", "3023"), ("arctic", "ARCTIC", "1030")]

SOURCES = (
    [f"{TCE}/{ds}/sanity.json" for ds, _ in DATASETS]
    + [f"{TCE}/{ds}/event_summary.csv" for ds, _ in DATASETS]
    + [f"{TCE}/{ds}/model_error_by_event.csv" for ds, _ in DATASETS]
    + [f"{TCE}/{ds}/kinematic_associations.csv" for ds, _ in DATASETS]
    + [f"{TCE}/{ds}/frames_test.csv" for ds, _, _ in TIMELINE_EXAMPLES]
    + [HAND_FIX]
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(text: str | None) -> float | None:
    """Parse a table cell; empty or non-finite cells are missing values."""
    if text is None or text == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def by_key(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def spike_concentration(src: Src) -> dict:
    panels, counts = [], []
    for ds, name in DATASETS:
        sanity = read_json(src(f"{TCE}/{ds}/sanity.json"))
        counts.append(f"{name} {sanity['n_spike_frames_q90']} of {sanity['n_transitions']}")
        panels.append({
            "title": name,
            "x": {"label": "quantity summed over the test transitions",
                  "categories": ["transitions", "change magnitude (sum of d_t)", "change energy (sum of d_t squared)"]},
            "y": {"label": "share held by spike transitions (fraction)", "direction": "none"},
            "series": [{"name": "spike transitions (d_t above train Q90)",
                        "values": [sanity["frac_spike_frames_q90"],
                                   sanity["spike_magnitude_share_of_all"],
                                   sanity["spike_energy_share_of_all"]]}],
        })
    return {
        "id": "f1c_spike_concentration",
        "type": "bar",
        "title": "Share of the test transitions, change magnitude and change energy held by spike transitions",
        "panels": panels,
        "note": ("d_t is the L2 norm of the frame-to-frame change of the 512-point contact map; a spike is a test "
                 "transition whose d_t exceeds the 90th percentile of d_t over the same dataset's training transitions. "
                 "Bars are fractions of the test-set total (spike transitions: " + "; ".join(counts) + "); no direction is better. "
                 "The threshold is a training quantile, so the share of transitions is close to a tenth by definition; what the "
                 "figure shows is how much of the change those transitions hold, with and without squaring."),
        "sources": [{"path": f"{TCE}/{ds}/sanity.json",
                     "locator": "keys frac_spike_frames_q90, spike_magnitude_share_of_all, spike_energy_share_of_all "
                                "(n_spike_frames_q90, n_transitions for the note)"} for ds, _ in DATASETS],
    }


def event_composition(src: Src) -> dict:
    panels, counts = [], []
    for ds, name in DATASETS:
        rows = by_key(read_csv(src(f"{TCE}/{ds}/event_summary.csv")), "category")
        sanity = read_json(src(f"{TCE}/{ds}/sanity.json"))
        counts.append(f"{name} {sanity['n_events_q90']}")
        cells = [rows.get(event_type, {}) for event_type in EVENT_TYPES]
        panels.append({
            "title": name,
            "x": {"label": "event type", "categories": list(EVENT_TYPES)},
            "y": {"label": "share (fraction)", "direction": "none"},
            "series": [
                {"name": "share of spike events", "values": [number(c.get("frequency")) for c in cells]},
                {"name": "share of spike change energy",
                 "values": [number(c.get("energy_share")) for c in cells],
                 "lo": [number(c.get("energy_lo")) for c in cells],
                 "hi": [number(c.get("energy_hi")) for c in cells]},
            ],
        })
    return {
        "id": "f1c_event_composition",
        "type": "bar",
        "title": "Spike events by type: share of events and share of spike change energy",
        "panels": panels,
        "note": ("Adjacent spike transitions are merged into events (" + "; ".join(counts) + " test events), typed onset / "
                 "release when the event contains a firm-contact transition and otherwise amount-dominant, spatial-dominant "
                 "or mixed by the amount ratio of the exact amount / pattern decomposition (cuts 0.7 and 0.3). Energy = share "
                 "of the squared change summed over spike frames, error bars = take-level cluster-bootstrap 95 % interval "
                 "(energy only); one ARCTIC event with no valid decomposition (near_zero) is not shown."),
        "sources": [{"path": f"{TCE}/{ds}/event_summary.csv",
                     "locator": "rows category in {onset, release, onset+release, amount, spatial, mixed}; "
                                "columns frequency, energy_share, energy_lo, energy_hi"} for ds, _ in DATASETS]
                   + [{"path": f"{TCE}/{ds}/sanity.json", "locator": "key n_events_q90 (for the note)"} for ds, _ in DATASETS],
    }


def error_by_event_class(src: Src) -> dict:
    panels, not_shown, few_takes = [], [], []
    shown = {frame_class for frame_class, _ in FRAME_CLASSES}
    for ds, name in DATASETS:
        rows = by_key(read_csv(src(f"{TCE}/{ds}/model_error_by_event.csv")), "frame_class")
        cells = [rows.get(frame_class, {}) for frame_class, _ in FRAME_CLASSES]
        # Classes of the table that get no bar of their own (their frames are inside "all spike"), read from the table.
        hidden = [f"{frame_class} (n={row['n_frames']})" for frame_class, row in rows.items()
                  if frame_class not in shown and frame_class != "all"]
        if hidden:
            not_shown.append(f"{name} " + ", ".join(hidden))
        # Shown classes whose frames come from one or two takes: their bootstrap interval is degenerate.
        thin = [f"{label} ({cell['n_takes']})" for (_, label), cell in zip(FRAME_CLASSES, cells)
                if cell and int(cell["n_takes"]) <= 2]
        if thin:
            few_takes.append(f"{name} " + ", ".join(thin))
        categories = [f"{label} (n={cell['n_frames']})" if cell else label
                      for (_, label), cell in zip(FRAME_CLASSES, cells)]
        panels.append({
            "title": name,
            "x": {"label": "frame class (n = test transitions in the class)", "categories": categories},
            "y": {"label": "mean temporal error e_t (raw contact units)", "direction": "lower_better"},
            "series": [{"name": series_name,
                        "values": [number(c.get(column)) for c in cells],
                        "lo": [number(c.get(f"{column}_lo")) for c in cells],
                        "hi": [number(c.get(f"{column}_hi")) for c in cells]}
                       for column, series_name in MODELS],
        })
    return {
        "id": "f1c_error_by_event_class",
        "type": "bar",
        "title": "Temporal error of the static baseline and the evolved model, by frame class",
        "panels": panels,
        "note": ("e_t is the L2 error of the predicted frame-to-frame contact change (lower is better); for the static "
                 "baseline it equals the ground-truth change d_t, and the evolved model rolls the learned vector field forward "
                 "from the ground-truth first map (source-table codes: B0 = static baseline, B1 = evolved model). Spike frames are grouped by the type of their event (low-persistence, persistence "
                 "at h = 4 below 0.5, is an overlapping cut); error bars = take-level cluster-bootstrap 95 % interval per "
                 "model, not a paired test, and degenerate where a class comes from one or two takes"
                 + (" (number of takes: " + "; ".join(few_takes) + ")" if few_takes else "") + "."
                 + (" Spike-frame classes of the table without a bar of their own, counted in 'all spike': "
                    + "; ".join(not_shown) + "." if not_shown else "")),
        "sources": [{"path": f"{TCE}/{ds}/model_error_by_event.csv",
                     "locator": "rows frame_class in {non_spike, spike, amount, onset, release, mixed, spatial, transient}; "
                                "columns n_frames, e_B0_static, e_B1_gtinit_vf and their _lo / _hi "
                                "(n_takes and the remaining frame_class rows for the note)"} for ds, _ in DATASETS],
    }


def event_bands(rows: list[dict[str, str]]) -> list[dict]:
    """One band per spike event: transition indices first - 0.5 .. last + 0.5, labelled with the event type."""
    bands: list[dict] = []
    current = None
    for row in rows:
        event_id = row["event_id"] if row["is_spike"] == "True" else ""
        if event_id and event_id == current:
            bands[-1]["x1"] = int(row["t"]) + 0.5
        elif event_id:
            bands.append({"x0": int(row["t"]) - 0.5, "x1": int(row["t"]) + 0.5, "label": row["event_category"]})
        current = event_id
    return bands


def timeline(src: Src) -> dict | None:
    tables: dict[str, Path] = {}
    for ds, _, example in TIMELINE_EXAMPLES:
        try:
            tables[ds] = src(f"{TCE}/{ds}/frames_test.csv", keep_rows=("example", example))
        except FileNotFoundError:
            warnings.warn(f"f1c_timeline skipped: {TCE}/{ds}/frames_test.csv is in neither the result tree nor the snapshots")
    if len(tables) < len(TIMELINE_EXAMPLES):
        return None
    panels, described, events, sources = [], [], [], []
    for ds, name, example in TIMELINE_EXAMPLES:
        rows = sorted((row for row in read_csv(tables[ds]) if row["example"] == example), key=lambda row: int(row["t"]))
        sanity = read_json(src(f"{TCE}/{ds}/sanity.json"))
        described.append(f"{name} {rows[0]['sequence_id']}, {rows[0]['group']}")
        # The renderer shades the bands without printing their labels, so the event types go into the note.
        events.append(f"{name} " + ", ".join(f"{band['label']} {int(band['x0'] + 0.5)}-{int(band['x1'] - 0.5)}"
                                             for band in event_bands(rows)))
        panels.append({
            "title": name,
            "x": {"label": "transition t (frame t to t+1 of the 64-frame window)", "values": [int(row["t"]) for row in rows]},
            "y": {"label": "raw contact units", "direction": "none"},
            "series": [
                {"name": "ground-truth change d_t", "values": [number(row["d"]) for row in rows]},
                {"name": "evolved model: error of its predicted change e_t", "values": [number(row["e_B1_gtinit_vf"]) for row in rows]},
            ],
            "refs": [{"axis": "y", "value": sanity["q90"], "label": "spike threshold (train Q90 of d_t)"}],
            "bands": event_bands(rows),
        })
        sources.append({"path": f"{TCE}/{ds}/frames_test.csv",
                        "locator": f"rows example == {example}; columns t, d, e_B1_gtinit_vf, is_spike, event_id, "
                                   "event_category (bands), sequence_id, group (note)"})
        sources.append({"path": f"{TCE}/{ds}/sanity.json", "locator": "key q90 (reference line)"})
    return {
        "id": "f1c_timeline",
        "type": "line",
        "title": "One test sequence per dataset: contact change and the evolved model's temporal error",
        "panels": panels,
        "note": ("Single examples, not statistics: the test sequence holding the report's representative spatial-dominant "
                 "event (" + "; ".join(described) + "), shaded bands = spike events (type and transitions: "
                 + "; ".join(events) + "). d_t is also the static baseline's error, so the evolved model (B1 in the source table) on the d_t line means no "
                 "better than holding the map still, below it means part of the change was predicted and above it means a "
                 "larger error than holding the map still; the table stores the error of the predicted change, not its "
                 "magnitude."),
        "sources": sources,
    }


def kinematic_association(src: Src) -> dict:
    corrected = read_json(src(HAND_FIX))["spearman_d_vs_hand_speed_test"]
    panels, counts = [], []
    for ds, name in DATASETS:
        rows = by_key(read_csv(src(f"{TCE}/{ds}/kinematic_associations.csv")), "signal")
        categories, values, lo, hi = [], [], [], []
        for signal, label in SIGNALS:
            row = rows.get(signal)
            if row is None and signal != HAND_SIGNAL:
                continue                                    # signal not measured for this dataset
            categories.append(label)
            if row is None:                                 # OakInk2: no hand signal is cached
                cell = (None, None, None)
            elif ds == "taco" and signal == HAND_SIGNAL:    # corrected tool-frame transform
                cell = (corrected["rho"], corrected["lo"], corrected["hi"])
            else:
                cell = (number(row["spearman_d"]), number(row["spearman_lo"]), number(row["spearman_hi"]))
            values.append(cell[0])
            lo.append(cell[1])
            hi.append(cell[2])
        panel = {
            "title": name,
            "x": {"label": "per-frame kinematic signal", "categories": categories},
            "y": {"label": "Spearman rank correlation with d_t", "direction": "none"},
            "series": [{"name": "Spearman(d_t, signal)", "values": values, "lo": lo, "hi": hi}],
        }
        if ds == "taco":
            panel["refs"] = [{"axis": "y", "value": number(rows[HAND_SIGNAL]["spearman_d"]),
                              "label": "hand-relative speed as published (uncorrected tool frame)"}]
        counts.append(f"{name} {rows['obj_lin_speed']['n_frames']}")
        panels.append(panel)
    return {
        "id": "f1suba_kinematic_association",
        "type": "bar",
        "title": "Rank correlation between the contact change and kinematic signals of the same transition",
        "panels": panels,
        "note": ("Rank correlation over all test transitions (" + "; ".join(counts) + ") between the contact change d_t and "
                 "each kinematic signal of the same transition, error bars = take-level cluster-bootstrap 95 % interval: an "
                 "association at the same frame, not a prediction result. The TACO hand-relative bar uses the corrected "
                 "tool-frame transform (reference line = the value in the original tables); hand motion was not measured "
                 "on OakInk2, articulation only on ARCTIC, tool-vs-target motion only on TACO."),
        "sources": [{"path": f"{TCE}/{ds}/kinematic_associations.csv",
                     "locator": "all rows; columns signal, spearman_d, spearman_lo, spearman_hi, n_frames"} for ds, _ in DATASETS]
                   + [{"path": HAND_FIX, "locator": "key spearman_d_vs_hand_speed_test: rho, lo, hi "
                                                    "(replaces the TACO hand_rel_speed row)"}],
    }


def build(src: Src) -> list[dict]:
    figures = [spike_concentration(src), event_composition(src), error_by_event_class(src),
               timeline(src), kinematic_association(src)]
    return [figure for figure in figures if figure is not None]
