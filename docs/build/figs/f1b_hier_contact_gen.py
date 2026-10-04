"""Figure data for page block F1-B: "Initial mode sampling explains much of the multimodality".

Experiment: hierarchical ("sample then evolve") contact generation, one model set per dataset.
Source report: reports/hier_contact_gen_report.md (tables in section 3 and section 4).

Variants read here (names as in the result tables):
    static_gt        B0  the true first contact map s0, held for all 64 frames
    gtinit_vf        B1  the true s0, then the deterministic vector field ("evolution")
    samplerG_vf      B2  s0 sampled from p(s0 | G), then the same vector field
    samplerG_static      the sampled s0 held for all 64 frames
K = 1 is sample 0; K = 5 / 10 is the best of the first 5 / 10 samples, the one with the lowest
E_C against the ground truth (an oracle selection, not a deployable generator).

Three figures, one panel per dataset (TACO, ARCTIC, OakInk2 are never pooled):
    f1b_sample_then_evolve    bar   E_C of B0, B1 and B2 at K = 1 / 5 / 10, with 95 % intervals
    f1b_paired_differences    bar   paired per-sequence E_C differences, with 95 % intervals
    f1b_error_over_time       line  E_C per frame for B0, B1 and B2 at K = 1 / 10
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

REPORT = "reports/hier_contact_gen_report.md"
SPLIT = "fixed"

# dataset name on the page -> experiment directory under the result root
DATASETS: list[tuple[str, str]] = [
    ("TACO", "taco/50_hier_contact_gen"),
    ("ARCTIC", "arctic/40_hier_contact_gen"),
    ("OakInk2", "oakink2/20_hier_contact_gen"),
]
TABLES = ("aggregate.csv", "paired_differences.csv", "curves.csv")
SOURCES: list[str] = [f"{root}/results/{table}" for _, root in DATASETS for table in TABLES]

# bar label -> (model, K) row of aggregate.csv
BAR_ROWS: list[tuple[str, str, str]] = [
    ("True s0, held", "static_gt", "0"),
    ("True s0 + evolution", "gtinit_vf", "0"),
    ("Sampled s0 (K=1) + evolution", "samplerG_vf", "1"),
    ("Sampled s0 (best of 5) + evolution", "samplerG_vf", "5"),
    ("Sampled s0 (best of 10) + evolution", "samplerG_vf", "10"),
]

# bar label -> `comparison` row of paired_differences.csv (difference = a - b)
PAIRED_ROWS: list[tuple[str, str]] = [
    ("Evolve minus hold, true s0", "Q2 VF - static (GT init)"),
    ("Evolve minus hold, sampled s0 (K=1)", "VF vs static, sampled G K=1"),
    ("Sampled K=1 minus true s0, both evolved", "Q3 sampled(G,K=1) - GT init"),
    ("Best of 10 minus true s0, both evolved", "Q3 sampled(G,best10) - GT init"),
]

# line name -> (model, K) rows of curves.csv, metric "err"
CURVE_ROWS: list[tuple[str, str, str]] = [
    ("True s0, held", "static_gt", "0"),
    ("True s0 + evolution", "gtinit_vf", "0"),
    ("Sampled K=1 + evolution", "samplerG_vf", "1"),
    ("Sampled best of 10 + evolution", "samplerG_vf", "10"),
]

Src = Callable[[str], Path]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one_row(rows: list[dict[str, str]], **match: str) -> dict[str, str]:
    """The single row whose columns equal `match`; anything else is an error."""
    hits = [row for row in rows if all(row[key] == value for key, value in match.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {match}, found {len(hits)}")
    return hits[0]


def test_set_size(aggregate: list[dict[str, str]]) -> tuple[int, int]:
    """(test sequences, split units) of the test set, read from the static_gt row."""
    row = one_row(aggregate, split=SPLIT, model="static_gt", K="0")
    return int(row["n_examples"]), int(row["n_takes"])


def size_text(sizes: dict[str, tuple[int, int]]) -> str:
    unit = {"OakInk2": "recordings"}
    return "; ".join(f"{name} {n} test sequences from {takes} {unit.get(name, 'takes')}" for name, (n, takes) in sizes.items())


def sample_then_evolve(src: Src) -> dict:
    panels, sizes = [], {}
    for name, root in DATASETS:
        aggregate = read_rows(src(f"{root}/results/aggregate.csv"))
        sizes[name] = test_set_size(aggregate)
        rows = [one_row(aggregate, split=SPLIT, model=model, K=k) for _, model, k in BAR_ROWS]
        panels.append({
            "title": name,
            "x": {"label": "initial contact map s0 and what follows it", "categories": [label for label, _, _ in BAR_ROWS]},
            "y": {"label": "contact error E_C (raw contact units)", "direction": "lower_better"},
            "series": [{
                "name": "E_C",
                "values": [float(row["E_C"]) for row in rows],
                "lo": [float(row["E_C_lo"]) for row in rows],
                "hi": [float(row["E_C_hi"]) for row in rows],
            }],
        })
    return {
        "id": "f1b_sample_then_evolve",
        "type": "bar",
        "title": "Sample an initial contact map, then evolve it deterministically",
        "panels": panels,
        "note": (
            "E_C is the L2 distance between predicted and true 512-point contact maps, averaged over the 64 frames "
            f"that follow a contact-episode onset and over the test sequences ({size_text(sizes)}); lower is better, "
            "and raw units are not comparable across datasets. Error bars are 95 % cluster-bootstrap intervals over "
            "takes (OakInk2: recordings), 500 resamples. The first two bars start from the true first contact map, so "
            "they are diagnostic rows, not generators. 'Best of K' is the lowest-error sample of K, picked with the "
            "ground truth; s0 is sampled from the static object condition only (geometry descriptor plus hand and "
            "role flags, no object motion)."
        ),
        "sources": [
            {"path": f"{root}/results/aggregate.csv",
             "locator": "rows split=fixed, (model, K) = (static_gt, 0), (gtinit_vf, 0), (samplerG_vf, 1 / 5 / 10); columns E_C, E_C_lo, E_C_hi, n_examples, n_takes"}
            for _, root in DATASETS
        ],
    }


def paired_differences(src: Src) -> dict:
    panels, sizes = [], {}
    for name, root in DATASETS:
        sizes[name] = test_set_size(read_rows(src(f"{root}/results/aggregate.csv")))
        paired = read_rows(src(f"{root}/results/paired_differences.csv"))
        rows = [one_row(paired, split=SPLIT, comparison=comparison) for _, comparison in PAIRED_ROWS]
        panels.append({
            "title": name,
            "x": {"label": "comparison (first minus second, same test sequences)", "categories": [label for label, _ in PAIRED_ROWS]},
            "y": {"label": "paired difference in E_C (raw contact units)", "direction": "none"},
            "series": [{
                "name": "E_C difference",
                "values": [float(row["E_C"]) for row in rows],
                "lo": [float(row["E_C_lo"]) for row in rows],
                "hi": [float(row["E_C_hi"]) for row in rows],
            }],
            "refs": [{"axis": "y", "value": 0.0, "label": "no difference"}],
        })
    return {
        "id": "f1b_paired_differences",
        "type": "bar",
        "title": "Paired differences: what evolution adds and what sampling costs",
        "panels": panels,
        "note": (
            "Mean per-sequence difference of the contact error E_C between two variants on the same test sequences "
            f"({size_text(sizes)}); below zero the first-named variant has the lower error. Error bars are 95 % "
            "cluster-bootstrap intervals over takes (OakInk2: recordings), 500 resamples: an interval that contains "
            "zero is no measurable difference. At K=1 'evolve minus hold' uses the same sampled map on both sides; "
            "'best of 10' is the sample of ten with the lowest whole-rollout error, picked with the ground truth."
        ),
        "sources": [
            {"path": f"{root}/results/paired_differences.csv",
             "locator": "rows split=fixed, comparison = 'Q2 VF - static (GT init)', 'VF vs static, sampled G K=1', "
                        "'Q3 sampled(G,K=1) - GT init', 'Q3 sampled(G,best10) - GT init'; columns E_C, E_C_lo, E_C_hi"}
            for _, root in DATASETS
        ] + [
            {"path": f"{root}/results/aggregate.csv", "locator": "row split=fixed, model=static_gt, K=0; columns n_examples, n_takes (test-set size in the note)"}
            for _, root in DATASETS
        ],
    }


def error_over_time(src: Src) -> dict:
    panels, sizes = [], {}
    for name, root in DATASETS:
        sizes[name] = test_set_size(read_rows(src(f"{root}/results/aggregate.csv")))
        curves = [row for row in read_rows(src(f"{root}/results/curves.csv")) if row["split"] == SPLIT and row["metric"] == "err"]
        frames: list[int] | None = None
        series = []
        for label, model, k in CURVE_ROWS:
            rows = sorted((row for row in curves if row["model"] == model and row["K"] == k), key=lambda row: int(row["t"]))
            t = [int(row["t"]) for row in rows]
            if frames is None:
                frames = t
            elif t != frames:
                raise ValueError(f"{name}: frame axis of {model} K={k} differs from the first curve")
            series.append({"name": label, "values": [float(row["value"]) for row in rows]})
        panels.append({
            "title": name,
            "x": {"label": "frame since the contact-episode onset (30 Hz)", "values": frames},
            "y": {"label": "contact error per frame (raw contact units)", "direction": "lower_better"},
            "series": series,
        })
    return {
        "id": "f1b_error_over_time",
        "type": "line",
        "title": "Contact error along the 64-frame rollout",
        "panels": panels,
        "note": (
            "L2 distance between predicted and true 512-point contact maps at each frame, averaged over the test "
            f"sequences ({size_text(sizes)}); lower is better, raw units are not comparable across datasets, and the "
            "table gives no interval per frame. Frame 0 is the onset of a contact episode: the first firm-contact "
            "frame of a take, or of a later episode of the same take. 'Best of 10' is the sample with the lowest "
            "whole-rollout error, picked with the ground truth."
        ),
        "sources": [
            {"path": f"{root}/results/curves.csv",
             "locator": "rows split=fixed, metric=err, (model, K) = (static_gt, 0), (gtinit_vf, 0), (samplerG_vf, 1), (samplerG_vf, 10); columns t, value"}
            for _, root in DATASETS
        ] + [
            {"path": f"{root}/results/aggregate.csv", "locator": "row split=fixed, model=static_gt, K=0; columns n_examples, n_takes (test-set size in the note)"}
            for _, root in DATASETS
        ],
    }


def build(src: Src) -> list[dict]:
    return [sample_then_evolve(src), paired_differences(src), error_over_time(src)]
