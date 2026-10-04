"""Figure data for page block F1-A: within-take and between-take contact variation.

Source study: the dynamic-contact study, Experiment 1 ("amount vs spatial pattern").
    TACO     taco/40_representation_study/time_decomp/dynamic_contact/dynamic_contact_report.md
    ARCTIC   arctic/30_dynamic_contact/dynamic_contact_report_arctic.md
    OakInk2  oakink2/10_dynamic_contact/dynamic_contact_report_oakink2.md
    summary  reports/dynamic_contact_crossdataset.md

What a plotted value is. Two frames of a 512-D contact map are compared with one metric
(amount = |difference of the summed soft contact|, pattern = L2 distance of the maps
normalised to sum 1). The pairs come from one "axis":
    time     same take, early vs late (phase gap > 0.5)
    take     other take, same mesh and verb, matched phase          <- the denominator
    mesh     other mesh (TACO) / other object instance (OakInk2), same verb, matched phase
    subject  other subject, same verb (ARCTIC)
    action   other verb
A ratio "X / take" is the mean pair distance along X divided by the mean cross-take pair
distance of the same metric, per group; the tables then average the per-group ratios
("macro"). 1 means "as different as two takes at the same phase".

ARCTIC has one mesh per object, so its identity axis is the subject and the table divides it
by the SAME-SUBJECT cross-take distance ("subject/take_same_subject"), not by the "take"
distance the time ratios use. The reference line of the ARCTIC panels is labelled accordingly,
and the notes quote the subject ratio over the any-subject "take" distance (the denominator of
the ARCTIC time bars), read from ARCTIC's verification table.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

MACRO_TABLE = "reports/crossdataset_exp1.csv"
GROUP_TABLES = {
    "TACO": "taco/40_representation_study/time_decomp/dynamic_contact/exp1/ratios.csv",
    "ARCTIC": "arctic/30_dynamic_contact/exp1/ratios.csv",
    "OakInk2": "oakink2/10_dynamic_contact/exp1/ratios.csv",
}
# ARCTIC only: subject / take with the any-subject "take" denominator of the time ratios.
ARCTIC_VERIFY_TABLE = "arctic/30_dynamic_contact/verify/v1_v2_ordering_configs.csv"
ARCTIC_VERIFY_CONFIG = "pattern_L2|mean|all|tol0.10"
SOURCES = [MACRO_TABLE, *GROUP_TABLES.values(), ARCTIC_VERIFY_TABLE]

DATASETS = ["TACO", "ARCTIC", "OakInk2"]
AMOUNT = "mass_abs"
LOG_AMOUNT = "mass_log"
PATTERN = "pattern_L2"

# The identity axis each dataset has: (ratio key in the tables, label of the bar).
# OakInk2's "mesh" axis is another object instance of the same category.
IDENTITY = {
    "TACO": ("mesh/take", "mesh / take"),
    "ARCTIC": ("subject/take_same_subject", "subject / same-subject take"),
    "OakInk2": ("mesh/take", "object instance / take"),
}
Y_LABEL = "pair distance / cross-take pair distance"
REF_TAKE = {"axis": "y", "value": 1.0, "label": "cross-take level"}
# The ARCTIC subject bars are divided by the same-subject cross-take distance, so the line at 1
# is not one common level in the ARCTIC bar panels.
REF_TAKE_ARCTIC = {"axis": "y", "value": 1.0,
                   "label": "cross-take level (same-subject takes for the subject bars)"}

MacroTable = dict[tuple[str, str, str], dict[str, str]]


def bar_ref(dataset: str) -> dict:
    """Reference line of a bar panel: the denominator level of that dataset's bars."""
    return REF_TAKE_ARCTIC if dataset == "ARCTIC" else REF_TAKE


def arctic_subject_over_take_note(path: Path) -> str:
    """Sentence giving ARCTIC's pattern subject / take ratio over the any-subject take distance."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["config"] == ARCTIC_VERIFY_CONFIG]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected one row with config {ARCTIC_VERIFY_CONFIG!r}, found {len(rows)}")
    row = rows[0]
    return (f"Over the any-subject cross-take distance that its time bars use, the ARCTIC subject ratio of the "
            f"pattern is {float(row['macro_subject_over_take']):.3f} "
            f"({row['subject_gt_take']} of {row['n_groups']} groups above 1).")


def read_macro(path: Path) -> MacroTable:
    """(dataset, ratio, metric) -> row of the cross-dataset table (macro, n_groups, groups_gt_1)."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["dataset"], row["ratio"], row["metric"]): row for row in csv.DictReader(handle)}


def read_group_ratios(path: Path, ratio: str, metric: str) -> dict[str, float]:
    """group -> mean-based ratio for one (ratio, metric) of a dataset's exp1/ratios.csv."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["group"]: float(row["value"]) for row in csv.DictReader(handle)
                if row["ratio"] == ratio and row["metric"] == metric and row["stat"] == "mean"}


def counts(table: MacroTable, dataset: str, cells: list[tuple[str, str]]) -> str:
    """'n groups (groups with ratio > 1)' for each bar, read from the table."""
    return ", ".join(f"{table[(dataset, ratio, metric)]['n_groups']} ({table[(dataset, ratio, metric)]['groups_gt_1']})"
                     for ratio, metric in cells)


ARCTIC_VERIFY_SOURCE = {"path": ARCTIC_VERIFY_TABLE,
                        "locator": f"note only (no plotted value): row config = {ARCTIC_VERIFY_CONFIG}; columns "
                                   "macro_subject_over_take, subject_gt_take, n_groups"}


def headline_figure(table: MacroTable, arctic_note: str) -> dict:
    """Three bars per dataset: time/take on the amount, time/take and identity/take on the pattern."""
    panels, count_notes = [], []
    for dataset in DATASETS:
        identity_ratio, identity_label = IDENTITY[dataset]
        cells = [("time/take", AMOUNT), ("time/take", PATTERN), (identity_ratio, PATTERN)]
        panels.append({
            "title": dataset,
            "x": {"label": "metric: numerator axis / denominator axis",
                  "categories": ["amount: time / take", "pattern: time / take", f"pattern: {identity_label}"]},
            "y": {"label": Y_LABEL, "direction": "none"},
            "series": [{"name": "macro mean of group ratios",
                        "values": [float(table[(dataset, ratio, metric)]["macro"]) for ratio, metric in cells]}],
            "refs": [bar_ref(dataset)],
        })
        count_notes.append(f"{dataset} {counts(table, dataset, cells)}")
    return {
        "id": "f1a_time_vs_identity_ratios",
        "type": "bar",
        "title": "Contact amount and normalised spatial pattern: within-take change and identity change, relative to two takes",
        "panels": panels,
        "note": ("Each bar is a ratio of mean pair distances in one metric (amount = |difference of summed soft contact|, "
                 "pattern = L2 between contact maps normalised to sum 1): early-vs-late frames of one take (time), or "
                 "another mesh (TACO), object instance (OakInk2) or subject (ARCTIC) at matched phase, over two takes "
                 "at matched phase; macro mean of per-group "
                 "ratios, 1 = as different as two takes, no direction is better, no error bars (the tables give "
                 "intervals per group only). Groups per bar (of which ratio > 1): " + "; ".join(count_notes) + "; the "
                 "ARCTIC third bar divides by the same-subject cross-take distance, not by the denominator of its "
                 "first two bars. " + arctic_note),
        "sources": [{"path": MACRO_TABLE,
                     "locator": "column macro (n_groups, groups_gt_1 for the note); rows dataset in {TACO, ARCTIC, OakInk2}, "
                                "(ratio, metric) = (time/take, mass_abs), (time/take, pattern_L2), and (mesh/take, pattern_L2) "
                                "for TACO / OakInk2 or (subject/take_same_subject, pattern_L2) for ARCTIC"},
                    ARCTIC_VERIFY_SOURCE],
    }


def axes_figure(table: MacroTable, arctic_note: str) -> dict:
    """Every variation source against the take level, on the amount, the log-amount and the pattern."""
    panels, group_notes = [], []
    for dataset in DATASETS:
        identity_ratio, identity_label = IDENTITY[dataset]
        ratios = ["time/take", identity_ratio, "action/take"]
        group_notes.append(f"{dataset} " + " / ".join(table[(dataset, ratio, PATTERN)]["n_groups"] for ratio in ratios))
        panels.append({
            "title": dataset,
            "x": {"label": "variation source / reference", "categories": ["time / take", identity_label, "action / take"]},
            "y": {"label": Y_LABEL, "direction": "none"},
            "series": [{"name": name, "values": [float(table[(dataset, ratio, metric)]["macro"]) for ratio in ratios]}
                       for name, metric in (("amount |Δmass|", AMOUNT), ("log-amount", LOG_AMOUNT), ("pattern L2", PATTERN))],
            "refs": [bar_ref(dataset)],
        })
    return {
        "id": "f1a_axes_amount_vs_pattern",
        "type": "bar",
        "title": "Variation sources relative to the cross-take level: amount, log-amount and normalised pattern",
        "panels": panels,
        "note": ("Macro mean over groups of the ratio of the mean pair distance along one axis to the matched-phase "
                 "cross-take distance of the same metric (log-amount = |difference of log(1 + mass)|); 1 = cross-take "
                 "level, no direction is better, no error bars. Groups per bar group (time / identity / action): "
                 + "; ".join(group_notes) + "; ARCTIC's identity bars divide by the same-subject cross-take distance. "
                 + arctic_note),
        "sources": [{"path": MACRO_TABLE,
                     "locator": "column macro; rows dataset in {TACO, ARCTIC, OakInk2}, ratio in {time/take, action/take, "
                                "mesh/take (TACO, OakInk2) or subject/take_same_subject (ARCTIC)}, "
                                "metric in {mass_abs, mass_log, pattern_L2}"},
                    ARCTIC_VERIFY_SOURCE],
    }


def per_group_figure(src: Callable[[str], Path]) -> dict:
    """One point per group: time/take on the amount (x) against time/take on the pattern (y)."""
    panels, sources = [], []
    for dataset in DATASETS:
        path = src(GROUP_TABLES[dataset])
        amount = read_group_ratios(path, "time/take", AMOUNT)
        pattern = read_group_ratios(path, "time/take", PATTERN)
        groups = sorted(set(amount) & set(pattern))
        points = [{"name": name, "xy": [[amount[g], pattern[g]] for g in groups if g.endswith(suffix)]}
                  for name, suffix in (("left hand", "__L"), ("right hand", "__R"))]
        panels.append({
            "title": dataset,
            "x": {"label": "time / take, amount"},
            "y": {"label": "time / take, normalised pattern", "direction": "none"},
            "points": [p for p in points if p["xy"]],
            "refs": [{"axis": "x", "value": 1.0, "label": "cross-take level"}, REF_TAKE],
        })
        sources.append({"path": GROUP_TABLES[dataset],
                        "locator": "column value; rows ratio = time/take, stat = mean, metric = mass_abs (x) and "
                                   "pattern_L2 (y); one point per group, split by the hand suffix of the group name"})
    return {
        "id": "f1a_per_group_time_over_take",
        "type": "scatter",
        "title": "Per group: early-vs-late change of the amount against that of the pattern",
        "panels": panels,
        "note": ("One point per group (TACO category x role x hand, where left hand = target and right hand = tool; ARCTIC "
                 "object x hand; OakInk2 category x hand): the early-vs-late pair distance within a take divided by the "
                 "matched-phase cross-take distance, for the amount (x) and for the normalised pattern (y). "
                 "1 = cross-take level on either axis; no direction is better; the per-group 95 % bootstrap intervals "
                 "of the table are not drawn."),
        "sources": sources,
    }


def build(src: Callable[[str], Path]) -> list[dict]:
    table = read_macro(src(MACRO_TABLE))
    arctic_note = arctic_subject_over_take_note(src(ARCTIC_VERIFY_TABLE))
    return [headline_figure(table, arctic_note), axes_figure(table, arctic_note), per_group_figure(src)]
