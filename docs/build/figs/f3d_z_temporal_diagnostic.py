"""Blocks F3-diagB and F3-diagC: the temporal diagnostic of the Stage-1 structure latent z.

F3-diagB (z has meaningful temporal geometry)
    f3diagb_dz_vs_dc              latent step as a function of the contact step, train-decile bins
    f3diagb_lowc_highz_fraction   share of "contact hardly changes, z jumps" transitions
F3-diagC (local z dynamics exist, especially on TACO)
    f3diagc_persistence_vs_probe_rmse   prediction error of persistence and of the learned probe
    f3diagc_gain_over_persistence       the probe's gain over persistence (input of the decision rule)
    f3diagc_gain_by_window_position     the same gain split by where the pair starts (POST HOC)

Source report: reports/z_temporal_diagnostic/report.md (Sections 5-7; Tables 2-5, 9, 12, 13).
Frozen Stage-1 encoder, test split, TACO and ARCTIC analysed separately: one panel per dataset,
nothing pooled. Every plotted value, every threshold line and every number quoted in a note is read
from the report's tables at run time.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

REPORT = "reports/z_temporal_diagnostic"
BINS = f"{REPORT}/temporal_geometry_bins.csv"
GEOMETRY = f"{REPORT}/temporal_geometry_metrics.csv"
QUADRANTS = f"{REPORT}/quadrant_metrics.csv"
LOCAL = f"{REPORT}/local_prediction_metrics.csv"
GAIN = f"{REPORT}/persistence_comparison.csv"
POSTHOC = f"{REPORT}/posthoc_splits.csv"
CONFIG = f"{REPORT}/experiment_config.json"
VALUES = f"{REPORT}/report_values.json"  # the report's own derived values (horizon means used by the rule)
SOURCES = [BINS, GEOMETRY, QUADRANTS, LOCAL, GAIN, POSTHOC, CONFIG, VALUES]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]
HORIZONS = ["1", "4", "8"]
HORIZON_LABELS = [f"h = {h}" for h in HORIZONS]
BIN_HORIZON = "4"  # the horizon the report prints in its Table 9
POSITIONS = [("t0", "t = 0 (window start)"), ("t1_7", "t = 1..7"), ("t8", "t ≥ 8")]

Row = dict[str, str]
Src = Callable[[str], Path]


def _read(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(text: str) -> float | None:
    return float(text) if text != "" else None


def _one(rows: list[Row], **where: str) -> Row:
    """The single row whose columns equal the given values."""
    found = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(found) != 1:
        raise ValueError(f"expected one row for {where}, found {len(found)}")
    return found[0]


def _column(rows: list[Row], column: str, **where: str) -> list[float | None]:
    """One value per horizon (h = 1, 4, 8) of the row selected by ``where``."""
    return [_num(_one(rows, h=h, **where)[column]) for h in HORIZONS]


def _series(rows: list[Row], name: str, column: str, **where: str) -> dict:
    """A series over the three horizons with the table's interval (<column>_lo / <column>_hi)."""
    return {
        "name": name,
        "values": _column(rows, column, **where),
        "lo": _column(rows, f"{column}_lo", **where),
        "hi": _column(rows, f"{column}_hi", **where),
    }


def _per_horizon(rows: list[Row], column: str, fmt: str, **where: str) -> str:
    """'a / b / c' for h = 1 / 4 / 8, formatted from the table."""
    return " / ".join(format(float(_one(rows, h=h, **where)[column]), fmt) for h in HORIZONS)


def _pct(value: float) -> str:
    return f"{value:.0%}".replace("%", " %")


def _dz_vs_dc(src: Src, quantiles: dict) -> dict:
    bins, geometry, quadrants = _read(src(BINS)), _read(src(GEOMETRY)), _read(src(QUADRANTS))
    h = BIN_HORIZON
    panels, spearman, counts, shares, skewed = [], [], [], [], []
    for key, title in DATASETS:
        deciles = sorted((r for r in bins if r["dataset"] == key and r["h"] == h), key=lambda r: int(r["decile"]))
        metrics = _one(geometry, dataset=key, h=h)
        contact_step = [float(r["dC_mean"]) for r in deciles]
        # bins whose mean lies outside their own interquartile range (a skewed bin), named in the note
        skewed += [f"{title} decile {r['decile']}" for r in deciles
                   if not float(r["dz_q25"]) <= float(r["dz_mean"]) <= float(r["dz_q75"])]
        panels.append({
            "title": title,
            "x": {"label": f"contact step ΔC = ‖C_t − C_t+{h}‖ / random-pair distance (mean of each train decile)",
                  "values": contact_step},
            "y": {"label": f"latent step Δz = ‖z_t − z_t+{h}‖ / random-pair distance", "direction": "none"},
            "series": [
                {"name": "mean Δz in the bin",
                 "values": [float(r["dz_mean"]) for r in deciles],
                 "lo": [float(r["dz_q25"]) for r in deciles],
                 "hi": [float(r["dz_q75"]) for r in deciles]},
                {"name": "ΔC itself (y = x)", "values": list(contact_step)},
            ],
            "refs": [{"axis": "y", "value": float(metrics["thr_dz_high"]),
                      "label": f"high Δz (train {_pct(quantiles['high'])} quantile)"}],
            "bands": [{"x0": 0.0, "x1": float(metrics["thr_dC_low"]),
                       "label": f"low ΔC (below the train {_pct(quantiles['low'])} quantile)"}],
        })
        spearman.append(f"{title} {_per_horizon(geometry, 'spearman_C_z', '.2f', dataset=key)}")
        counts.append(f"{int(metrics['n_test'])} ({title})")
        shares.append(f"{title} {_per_horizon(quadrants, 'B_lowC_highz', '.1%', dataset=key)}".replace("%", " %"))
    skew_clause = f"; the mean lies outside that range in the skewed bin(s) {', '.join(skewed)}" if skewed else ""
    note = (
        f"Test transitions (t, t+{h}) grouped by the train deciles of the contact step ΔC: mean latent step Δz per "
        "bin, with the interquartile range of Δz in the bin drawn around it (the spread of the transitions, not a "
        f"confidence interval{skew_clause}). Both steps are L2 distances divided by the mean distance between random "
        f"training frames, no direction is better, n = {', '.join(counts)} transitions. The shaded vertical band "
        f"marks low ΔC (below the train {_pct(quantiles['low'])} quantile of ΔC) and the dashed line high Δz (the "
        f"train {_pct(quantiles['high'])} quantile of Δz). Spearman(ΔC, Δz) over all test transitions at "
        f"h = 1 / 4 / 8 is {'; '.join(spearman)}, and the individual transitions with low ΔC and high Δz "
        f"(thresholds of each horizon) are {'; '.join(shares)} of the test transitions."
    )
    return {
        "id": "f3diagb_dz_vs_dc",
        "type": "line",
        "title": f"The latent step follows the contact step (h = {h} frames)",
        "panels": panels,
        "note": note,
        "sources": [
            {"path": BINS, "locator": f"rows h = {h}, decile 1..10 per dataset; columns dC_mean (x and the y = x line), "
                                      "dz_mean (y), dz_q25 / dz_q75 (interquartile range around the line)"},
            {"path": GEOMETRY, "locator": f"rows h = {h}: thr_dC_low (band), thr_dz_high (dashed line), n_test; "
                                          "rows h = 1, 4, 8: spearman_C_z (note)"},
            {"path": QUADRANTS, "locator": "rows h = 1, 4, 8; column B_lowC_highz (shares quoted in the note)"},
            {"path": CONFIG, "locator": "quantiles.low / quantiles.high (labels of the band and the dashed line)"},
        ],
    }


def _lowc_highz_fraction(src: Src, rule: dict, quantiles: dict) -> dict:
    quadrants = _read(src(QUADRANTS))
    values = json.loads(src(VALUES).read_text(encoding="utf-8"))
    panels, counts, means = [], [], []
    for key, title in DATASETS:
        means.append(f"{title} {float(values[f'mean_quadB_{key}']):.1%}".replace("%", " %"))
        panels.append({
            "title": title,
            "x": {"label": "step between the two frames", "categories": list(HORIZON_LABELS)},
            "y": {"label": "share of test transitions (fraction)", "direction": "lower_better"},
            "series": [
                _series(quadrants, "low ΔC / high Δz, observed", "B_lowC_highz", dataset=key),
                {"name": "expected if independent",
                 "values": _column(quadrants, "indep_B_at_test_marginals", dataset=key)},
            ],
            "refs": [{"axis": "y", "value": float(rule["quadrant_few"]),
                      "label": f"rule: 'Δz tracks ΔC' needs at most {_pct(rule['quadrant_few'])}"}],
        })
        counts.append(f"{title} {_per_horizon(quadrants, 'n_test', '.0f', dataset=key)}")
    note = (
        f"Share of test transitions whose contact step is low (below the train {_pct(quantiles['low'])} quantile of "
        f"ΔC) while the latent step is high (above the train {_pct(quantiles['high'])} quantile of Δz), i.e. the "
        "contact hardly changes but z jumps; lower means fewer such transitions. The second bar is the share that "
        "independent steps would give at the test marginals; error bars are 95 % take-cluster bootstrap intervals; "
        f"test transitions at h = 1 / 4 / 8: {'; '.join(counts)}. The decision rule compares the mean of the three "
        f"horizons with the dashed limit: {'; '.join(means)}."
    )
    return {
        "id": "f3diagb_lowc_highz_fraction",
        "type": "bar",
        "title": "Transitions where the contact hardly changes but z jumps are rare",
        "panels": panels,
        "note": note,
        "sources": [
            {"path": QUADRANTS, "locator": "rows h = 1, 4, 8 per dataset; columns B_lowC_highz, B_lowC_highz_lo, "
                                           "B_lowC_highz_hi, indep_B_at_test_marginals, n_test"},
            {"path": CONFIG, "locator": "decision_rule.quadrant_few (dashed line); quantiles.low / quantiles.high (note)"},
            {"path": VALUES, "locator": "keys mean_quadB_taco, mean_quadB_arctic (mean over h = 1, 4, 8 quoted in the "
                                        "note; the rule's input)"},
        ],
    }


def _persistence_vs_probe(src: Src) -> dict:
    local = _read(src(LOCAL))
    panels, counts, scale = [], [], []
    for key, title in DATASETS:
        panels.append({
            "title": title,
            "x": {"label": "prediction horizon (frames ahead)", "categories": list(HORIZON_LABELS)},
            "y": {"label": "RMSE of the predicted z_t+h (train-standardised units)", "direction": "lower_better"},
            "series": [
                _series(local, "persistence (copy z_t)", "rmse", dataset=key, method="persistence"),
                _series(local, "learned probe", "rmse", dataset=key, method="probe"),
            ],
        })
        counts.append(f"{title} {_per_horizon(local, 'n_pairs', '.0f', dataset=key, method='probe')}")
        scale.append(f"{title} {_per_horizon(local, 'rmse', '.2f', dataset=key, method='mean')}")
    note = (
        "Error of predicting the latent h frames ahead when the true current latent z_t is given, on all test pairs "
        "of a window (lower is better; a single step, no rollout): 'persistence' copies z_t, the probe is a residual "
        "MLP on z_t, the static object descriptor and the object-trajectory windows around frames t and t+h (one "
        "probe per horizon, one seed). z is standardised with the training mean and "
        f"standard deviation (the train-mean predictor scores {'; '.join(scale)} at h = 1 / 4 / 8), error bars are "
        f"95 % take-cluster bootstrap intervals, and the test pairs at h = 1 / 4 / 8 are {'; '.join(counts)}."
    )
    return {
        "id": "f3diagc_persistence_vs_probe_rmse",
        "type": "bar",
        "title": "Predicting z a few frames ahead: copying the current latent versus a learned probe",
        "panels": panels,
        "note": note,
        "sources": [
            {"path": LOCAL, "locator": "rows method = persistence and method = probe, h = 1, 4, 8 per dataset; "
                                       "columns rmse, rmse_lo, rmse_hi, n_pairs; rows method = mean, column rmse (note)"},
        ],
    }


def _gain_refs(rule: dict) -> list[dict]:
    return [
        {"axis": "y", "value": float(rule["gain_clear"]), "label": f"'clear' ≥ {_pct(rule['gain_clear'])}"},
        {"axis": "y", "value": float(rule["gain_marginal"]), "label": f"'marginal' ≥ {_pct(rule['gain_marginal'])}"},
    ]


def _gain_over_persistence(src: Src, rule: dict) -> dict:
    gain = _read(src(GAIN))
    panels = []
    for key, title in DATASETS:
        panels.append({
            "title": title,
            "x": {"label": "prediction horizon (frames ahead)", "categories": list(HORIZON_LABELS)},
            "y": {"label": "gain over persistence, 1 − RMSE_probe / RMSE_persistence (fraction)",
                  "direction": "higher_better"},
            "series": [_series(gain, "learned probe, all pairs of a window", "gain_rmse", dataset=key, method="probe")],
            "refs": _gain_refs(rule),
        })
    note = (
        "How much the learned probe, given the true current latent z_t, lowers the RMSE of simply copying that "
        "latent, over all test pairs of a window (higher is better, 0 = no better than persistence, negative = "
        "worse). Error bars: 95 % take-cluster bootstrap intervals; the dashed lines are the thresholds of the "
        "decision rule, which according to the report was fixed before the probes were trained (a label also needs "
        "an interval that excludes zero); one seed per probe, single-step prediction, no horizon beyond 8 frames and "
        "no closed-loop rollout."
    )
    return {
        "id": "f3diagc_gain_over_persistence",
        "type": "bar",
        "title": "What a learned predictor adds to persistence (window average, the rule's input)",
        "panels": panels,
        "note": note,
        "sources": [
            {"path": GAIN, "locator": "rows method = probe, h = 1, 4, 8 per dataset; columns gain_rmse, gain_rmse_lo, "
                                      "gain_rmse_hi"},
            {"path": CONFIG, "locator": "decision_rule.gain_clear and decision_rule.gain_marginal (dashed lines)"},
        ],
    }


def _gain_by_window_position(src: Src, rule: dict) -> dict:
    rows = [r for r in _read(src(POSTHOC))
            if r["eval_split"] == "test" and r["method"] == "probe" and r["subject"] == "all"]
    panels, counts = [], []
    for key, title in DATASETS:
        panels.append({
            "title": title,
            "x": {"label": "prediction horizon (frames ahead)", "categories": list(HORIZON_LABELS)},
            "y": {"label": "gain over persistence, 1 − RMSE_probe / RMSE_persistence (fraction)",
                  "direction": "higher_better"},
            "series": [_series(rows, label, "gain_rmse", dataset=key, position=code) for code, label in POSITIONS],
            "refs": _gain_refs(rule)[:1],
        })
        per_position = ", ".join(
            f"{label} {_per_horizon(rows, 'n_pairs', '.0f', dataset=key, position=code)}" for code, label in POSITIONS)
        counts.append(f"{title}: {per_position}")
    note = (
        "POST HOC split, added after the test results had been seen and not an input of the decision rule: the "
        "probe's gain over persistence (higher is better, negative = worse than persistence) by the frame t at which "
        "the pair starts inside the window, which begins at a contact-episode onset; the same probes as in "
        "the window average, given the true current latent z_t. Error bars: 95 % take-cluster bootstrap intervals; "
        f"test pairs at h = 1 / 4 / 8 - {'; '.join(counts)}; one seed per probe, no closed-loop rollout."
    )
    return {
        "id": "f3diagc_gain_by_window_position",
        "type": "bar",
        "title": "Post hoc: the gain over persistence by where in the window the pair starts",
        "panels": panels,
        "note": note,
        "sources": [
            {"path": POSTHOC, "locator": "rows eval_split = test, method = probe, subject = all, position = t0 / t1_7 / "
                                         "t8, h = 1, 4, 8 per dataset; columns gain_rmse, gain_rmse_lo, gain_rmse_hi, "
                                         "n_pairs"},
            {"path": CONFIG, "locator": "decision_rule.gain_clear (dashed line)"},
        ],
    }


def build(src: Src) -> list[dict]:
    config = json.loads(src(CONFIG).read_text(encoding="utf-8"))
    rule, quantiles = config["decision_rule"], config["quantiles"]
    return [
        _dz_vs_dc(src, quantiles),
        _lowc_highz_fraction(src, rule, quantiles),
        _persistence_vs_probe(src),
        _gain_over_persistence(src, rule),
        _gain_by_window_position(src, rule),
    ]
