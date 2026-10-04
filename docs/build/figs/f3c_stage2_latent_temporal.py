"""F3-C (can z serve as the temporal state?) and F3-diagA (the teacher z decodes well).

Stage 2 of the contact-latent study trains three temporal generators with the same backbone
architecture and the same inputs (s_0, G, tau) and scores their 63 generated frames:

    B0  direct dense          (s_0, G, tau) -> C_1:63, a residual around s_0
    B1  z-mediated            (s_0, G, tau) -> z_1:63 -> D_z -> C_1:63
    B2  (z, r)-mediated       (s_0, G, tau) -> (z, r)_1:63 -> D_z + D_r -> C_1:63

F3-C figures compare the three on the dense error, on the paired differences, on the structural /
wrench metrics and along the horizon. F3-diagA figures come from the report's bottleneck
inspection: the fine-tuned B1 decoder is fed with the teacher latents z*_t (the frozen Stage-1
encoder applied to the ground-truth frames) instead of the latents the temporal model predicted.

Source report
    reports/contact_latent_temporal_stage2/report.md (tables 2, 3, 9, 11; sections 8-12, 17, 18).

All values are test means of one training seed per model, TACO and ARCTIC in separate panels.
Intervals are the 95 % take-cluster bootstrap intervals stored in the tables (1000 resamples).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np

REPORT = "reports/contact_latent_temporal_stage2"
MAIN = f"{REPORT}/main_metrics.csv"
PAIRED = f"{REPORT}/paired_comparisons.csv"
HORIZON = f"{REPORT}/horizon_metrics.csv"
BOTTLENECK = f"{REPORT}/bottleneck_metrics.csv"
CURVES = f"{REPORT}/temporal_curves.npz"
BOTTLENECK_CURVES = {
    "taco": f"{REPORT}/taco/bottleneck_curves.npz",
    "arctic": f"{REPORT}/arctic/bottleneck_curves.npz",
}
SOURCES = [MAIN, PAIRED, HORIZON, BOTTLENECK, CURVES, *BOTTLENECK_CURVES.values()]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]

# model keys of the tables and the short names used on the page
B0, B1, B2 = "B0", "B1", "B2"
PERSISTENCE, GROUND_TRUTH = "PERSIST", "GT"
NAME_B0 = "direct"
NAME_B1 = "via z"
NAME_B2 = "via z + r"
NAME_B1_PREDICTED = "via z: predicted z"
NAME_B1_TEACHER = "via z: teacher z* (same decoder)"
NAME_STAGE1 = "Stage-1 autoencoder, z only"
NAME_PERSISTENCE = "hold s_0 (no model)"

DENSE_ERROR = "dense error E_C (L2 over the 512 points, raw contact units)"

# metric key, page label: the dense error and the five structural / wrench metrics of the report's rule
RELATIVE_METRICS = [
    ("E_C", "dense error E_C"),
    ("part_hamming", "participation Hamming"),
    ("amount_l1", "contact amount L1"),
    ("centroid", "centroid / l"),
    ("normal", "normal angle"),
    ("q_rel_l1", "wrench rel. L1"),
]
VERDICT_MARK = " *"  # appended to a metric label when the table's verdict is better / worse

Rows = list[dict[str, str]]


def _read_csv(path: Path) -> Rows:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _one(rows: Rows, **where: str) -> dict[str, str]:
    """The single row matching every column = value filter."""
    hits = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {where}, found {len(hits)}")
    return hits[0]


def _interval_series(name: str, rows: Rows, value: str = "value", lo: str = "ci_lo", hi: str = "ci_hi") -> dict:
    return {
        "name": name,
        "values": [float(row[value]) for row in rows],
        "lo": [float(row[lo]) for row in rows],
        "hi": [float(row[hi]) for row in rows],
    }


def _test_sizes(main: Rows) -> str:
    """'364 (TACO) / 186 (ARCTIC)': test sequences behind the dense error of B0."""
    return " / ".join(
        f"{_one(main, dataset=ds, model=B0, metric='E_C')['n_examples']} ({title})" for ds, title in DATASETS
    )


def _dense_error(main: Rows, sizes: str) -> dict:
    models = [(B0, NAME_B0), (B1, NAME_B1), (B2, NAME_B2)]
    panels = []
    for ds, title in DATASETS:
        rows = [_one(main, dataset=ds, model=model, metric="E_C") for model, _ in models]
        persistence = _one(main, dataset=ds, model=PERSISTENCE, metric="E_C")
        panels.append({
            "title": title,
            "x": {"label": "model", "categories": [name for _, name in models]},
            "y": {"label": DENSE_ERROR, "direction": "lower_better"},
            "series": [_interval_series("E_C, frames 1-63", rows)],
            "refs": [{"axis": "y", "value": float(persistence["value"]), "label": NAME_PERSISTENCE}],
        })
    return {
        "id": "f3c_dense_error_b0_b1_b2",
        "type": "bar",
        "title": "Dense error of the direct, z-mediated and (z, r)-mediated generators",
        "panels": panels,
        "note": (
            "Per-frame L2 distance between the generated and the true 512-point contact map, averaged over "
            f"the 63 generated frames and the test sequences (n = {sizes}); lower is better; one seed per model. "
            "Error bars are the 95 % take-cluster bootstrap interval of each model's own mean; they are not "
            "paired, so overlap does not decide the comparison; the paired differences (paired_comparisons.csv) are "
            "quoted in the text next to the chart. Source-table codes: B0 = direct, B1 = via z, B2 = via z + r."
        ),
        "sources": [{
            "path": MAIN,
            "locator": "rows metric = E_C, model in B0 / B1 / B2 (bars) and PERSIST (reference line), per dataset; "
                       "columns value, ci_lo, ci_hi, n_examples",
        }],
    }


def _dense_paired_difference(paired: Rows, sizes: str) -> dict:
    pairs = [(B1, B0, "B1 - B0"), (B2, B1, "B2 - B1")]
    panels = []
    for ds, title in DATASETS:
        rows = [_one(paired, dataset=ds, a=a, b=b, metric="E_C") for a, b, _ in pairs]
        panels.append({
            "title": title,
            "x": {"label": "paired comparison on identical test sequences", "categories": [name for _, _, name in pairs]},
            "y": {"label": "difference in dense error E_C (first minus second)", "direction": "lower_better"},
            "series": [_interval_series("mean paired difference", rows, value="diff")],
            "refs": [{"axis": "y", "value": 0.0, "label": "no difference"}],
        })
    return {
        "id": "f3c_dense_error_paired_diff",
        "type": "bar",
        "title": "Paired difference in dense error: does z help, does r add?",
        "panels": panels,
        "note": (
            "Mean per-sequence difference in dense error E_C between two models on the same test sequences "
            f"(n = {sizes}); below zero means the first-named model is better. Error bars are the paired 95 % "
            "take-cluster bootstrap interval; an interval that contains zero is a tie. One seed per model."
        ),
        "sources": [{
            "path": PAIRED,
            "locator": "rows metric = E_C with (a, b) = (B1, B0) and (B2, B1), per dataset; columns diff, ci_lo, ci_hi",
        }],
    }


def _relative_change(paired: Rows, sizes: str) -> dict:
    panels = []
    for ds, title in DATASETS:
        rows = [_one(paired, dataset=ds, a=B1, b=B0, metric=metric) for metric, _ in RELATIVE_METRICS]
        categories = [
            label + (VERDICT_MARK if row["verdict"] != "similar" else "")
            for (_, label), row in zip(RELATIVE_METRICS, rows)
        ]
        panels.append({
            "title": title,
            "x": {"label": "error metric", "categories": categories},
            "y": {"label": "relative improvement of the z-mediated over the direct generator (fraction of the direct error)", "direction": "higher_better"},
            "series": [{"name": "via z versus direct", "values": [float(row["rel_improvement"]) for row in rows]}],
            "refs": [{"axis": "y", "value": 0.0, "label": "no difference"}],
        })
    return {
        "id": "f3c_structure_rel_change_b1_vs_b0",
        "type": "bar",
        "title": "z-mediated versus direct generation on the dense, structural and wrench metrics",
        "panels": panels,
        "note": (
            "Relative change of the paired mean error when the generator predicts through z (B1 in the source table) "
            "instead of directly (B0): above zero the z-mediated generator is better, below zero it is worse; 0.05 = 5 %. All six metrics are "
            f"errors averaged over the 63 generated frames (n = {sizes}, one seed). A star marks a difference "
            "the report's rule counts as better or worse (paired 95 % interval excludes zero and the change is "
            "at least 2 %); unmarked metrics are ties. The table gives no interval for the relative change."
        ),
        "sources": [{
            "path": PAIRED,
            "locator": "rows a = B1, b = B0, metric in E_C / part_hamming / amount_l1 / centroid / normal / q_rel_l1, "
                       "per dataset; columns rel_improvement (plotted) and verdict (star)",
        }],
    }


def _horizon_figure(horizon: Rows, sizes: str, curve: str, models: list[tuple[str, str]],
                    figure_id: str, title: str, y_label: str, note: str) -> dict:
    panels = []
    for ds, panel_title in DATASETS:
        series, bins = [], None
        for model, name in models:
            rows = sorted(
                (row for row in horizon if row["dataset"] == ds and row["model"] == model and row["curve"] == curve),
                key=lambda row: int(row["h_lo"]),
            )
            model_bins = [f"{row['h_lo']}-{row['h_hi']}" for row in rows]
            # every series of a panel must sit on the same bins, otherwise values would be drawn under the wrong label
            if not rows or (bins is not None and model_bins != bins):
                raise ValueError(f"{figure_id}: horizon bins of {model} on {ds} are {model_bins}, expected {bins}")
            bins = model_bins
            series.append(_interval_series(name, rows))
        panels.append({
            "title": panel_title,
            "x": {"label": "generated frames t (horizon bin)", "categories": bins},
            "y": {"label": y_label, "direction": "lower_better"},
            "series": series,
        })
    model_keys = " / ".join(model for model, _ in models)
    return {
        "id": figure_id,
        "type": "line",
        "title": title,
        "panels": panels,
        "note": (
            f"{note} Test means (n = {sizes}, one seed); error bars are the 95 % take-cluster bootstrap interval "
            "of each curve's own bin mean. They are not paired, and the tables hold no paired test per bin, so "
            "overlapping bars do not decide which curve is lower in a bin."
        ),
        "sources": [{
            "path": HORIZON,
            "locator": f"rows curve = {curve}, model in {model_keys}, per dataset, bins ordered by h_lo; "
                       "columns h_lo, h_hi (bin), value, ci_lo, ci_hi",
        }],
    }


def _teacher_versus_predicted(main: Rows, bottleneck: Rows, sizes: str) -> dict:
    panels = []
    for ds, title in DATASETS:
        direct = _one(main, dataset=ds, model=B0, metric="E_C")
        latent = _one(bottleneck, dataset=ds, model=B1)
        bars = [  # page label, value, lower bound, upper bound
            (NAME_B0, direct["value"], direct["ci_lo"], direct["ci_hi"]),
            (NAME_B1_PREDICTED, latent["E_C_pred"], latent["E_C_pred_lo"], latent["E_C_pred_hi"]),
            (NAME_B1_TEACHER, latent["E_C_oracle_z"], latent["E_C_oracle_z_lo"], latent["E_C_oracle_z_hi"]),
            (NAME_STAGE1, latent["stage1_E_zonly"], latent["stage1_E_zonly_lo"], latent["stage1_E_zonly_hi"]),
        ]
        panels.append({
            "title": title,
            "x": {"label": "what produces the 63 frames", "categories": [bar[0] for bar in bars]},
            "y": {"label": DENSE_ERROR, "direction": "lower_better"},
            "series": [{
                "name": "E_C, frames 1-63",
                "values": [float(bar[1]) for bar in bars],
                "lo": [float(bar[2]) for bar in bars],
                "hi": [float(bar[3]) for bar in bars],
            }],
            "highlight": {"category": NAME_B1_TEACHER, "label": "same decoder, latents of the true frames"},
        })
    return {
        "id": "f3diaga_teacher_z_vs_predicted_z",
        "type": "bar",
        "title": "Decoding the teacher z versus decoding the predicted z",
        "panels": panels,
        "note": (
            "Dense error of the 63 frames (lower is better) when the fine-tuned decoder of the z-mediated generator (B1 "
            "in the source tables) receives the latents its temporal model predicted, versus the teacher latents z* "
            "that the frozen Stage-1 encoder computes from the true frames (analysis only, never available at "
            "inference). The first bar is the direct generator (B0); the last bar is the Stage-1 autoencoder's own "
            "z-only reconstruction of the same 63 frames of the test sequences, a different frame set from the unique "
            "test frames of f3b_reconstruction_error, so the two values of that model differ slightly. "
            f"Test means (n = {sizes}, one seed) with 95 % take-cluster bootstrap intervals."
        ),
        "sources": [
            {"path": MAIN, "locator": "rows model = B0, metric = E_C, per dataset; columns value, ci_lo, ci_hi, n_examples"},
            {
                "path": BOTTLENECK,
                "locator": "rows model = B1, per dataset; columns E_C_pred, E_C_oracle_z, stage1_E_zonly and their _lo / _hi",
            },
        ],
    }


def _floats(array: np.ndarray) -> list[float]:
    return [float(value) for value in array]


def _per_frame(src: Callable[[str], Path], sizes: str) -> list[dict]:
    """Dense error per generated frame, and the first generated frame alone."""
    line_panels, first_panels = [], []
    with np.load(src(CURVES)) as curves:
        for ds, title in DATASETS:
            with np.load(src(BOTTLENECK_CURVES[ds])) as floor:
                teacher = _floats(floor["B1|oracle_z"])          # frames 1..63
            # arrays of temporal_curves.npz are indexed by frame 0..63; frame 0 is s_0 itself
            direct = _floats(curves[f"{ds}|{B0}|dense"])[1:]
            latent = _floats(curves[f"{ds}|{B1}|dense"])[1:]
            persistence = _floats(curves[f"{ds}|{PERSISTENCE}|dense"])[1:]
            frames = list(range(1, len(direct) + 1))
            line_panels.append({
                "title": title,
                "x": {"label": "generated frame t", "values": frames},
                "y": {"label": DENSE_ERROR, "direction": "lower_better"},
                "series": [
                    {"name": NAME_B0, "values": direct},
                    {"name": NAME_B1_PREDICTED, "values": latent},
                    {"name": NAME_B1_TEACHER, "values": teacher},
                ],
            })
            first_panels.append({
                "title": title,
                "x": {"label": f"first generated frame (t = {frames[0]})",
                      "categories": [NAME_PERSISTENCE, NAME_B0, NAME_B1_PREDICTED, NAME_B1_TEACHER]},
                "y": {"label": DENSE_ERROR, "direction": "lower_better"},
                "series": [{"name": "E_C at the first generated frame",
                            "values": [persistence[0], direct[0], latent[0], teacher[0]]}],
            })
    per_frame = {
        "id": "f3diaga_dense_error_per_frame",
        "type": "line",
        "title": "Dense error per generated frame: direct, predicted z, teacher z",
        "panels": line_panels,
        "note": (
            "Test-mean dense error at each of the 63 generated frames (lower is better): the direct generator B0, "
            "the z-mediated generator B1 with its predicted latents, and the same B1 decoder fed with the teacher "
            f"latents z* of the true frames (analysis only). n = {sizes}, one seed; the source stores means only, "
            "so there are no error bars."
        ),
        "sources": [
            {"path": CURVES, "locator": "arrays <dataset>|B0|dense and <dataset>|B1|dense, elements [1..63] (index = frame)"},
            *[{"path": path, "locator": "array B1|oracle_z (index 0..62 = frame 1..63)"} for path in BOTTLENECK_CURVES.values()],
        ],
    }
    first_frame = {
        "id": "f3diaga_first_frame_error",
        "type": "bar",
        "title": "Dense error at the first generated frame",
        "panels": first_panels,
        "note": (
            "Test-mean dense error at the first generated frame (lower is better). Persistence repeats the given "
            "initial map s_0; B0 predicts a residual around s_0; B1 has to carry s_0 through the latent and a "
            "decoder that outputs an absolute map; the last bar is that decoder fed with the teacher latent of the "
            f"true frame. n = {sizes}, one seed; the source stores means only, so there are no error bars."
        ),
        "sources": [
            {"path": CURVES, "locator": "arrays <dataset>|PERSIST|dense, <dataset>|B0|dense, <dataset>|B1|dense, element [1] (frame 1)"},
            *[{"path": path, "locator": "array B1|oracle_z, element [0] (frame 1)"} for path in BOTTLENECK_CURVES.values()],
        ],
    }
    return [per_frame, first_frame]


def build(src: Callable[[str], Path]) -> list[dict]:
    main = _read_csv(src(MAIN))
    paired = _read_csv(src(PAIRED))
    horizon = _read_csv(src(HORIZON))
    bottleneck = _read_csv(src(BOTTLENECK))
    sizes = _test_sizes(main)
    dense_by_horizon = _horizon_figure(
        horizon, sizes, curve="dense", models=[(B0, NAME_B0), (B1, NAME_B1)],
        figure_id="f3c_dense_error_by_horizon",
        title="Dense error along the horizon: direct versus z-mediated",
        y_label=DENSE_ERROR,
        note="Dense error averaged within five bins of the 63 generated frames; lower is better.",
    )
    participation_by_horizon = _horizon_figure(
        horizon, sizes, curve="part",
        models=[(B0, NAME_B0), (B1, NAME_B1), (GROUND_TRUTH, "true map, same extraction (floor)")],
        figure_id="f3c_participation_by_horizon",
        title="Participation error along the horizon: direct versus z-mediated",
        y_label="participation Hamming error",
        note=(
            "Hamming distance between the participation pattern extracted from the generated map and the exact "
            "one, within five bins of the 63 generated frames; lower is better. The floor is the true map passed "
            "through the same extraction."
        ),
    )
    return [
        _dense_error(main, sizes),
        _dense_paired_difference(paired, sizes),
        _relative_change(paired, sizes),
        dense_by_horizon,
        participation_by_horizon,
        _teacher_versus_predicted(main, bottleneck, sizes),
        *_per_frame(src, sizes),
    ]
