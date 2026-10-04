"""Figure data for page block F3-A: explicit structural supervision is not enough.

Source report: reports/structure_aware_temporal_generation/report.md (2026-10-01/02).
Six generators of a 64-frame contact-map trajectory were trained per dataset with 3 seeds:
D0 / D1 / D2 = one-pass deterministic model with the dense loss only / + R2 loss on the output /
+ hidden R2 auxiliary head; S0 / S1 / S2 = sequence diffusion with the same three supervision levels.

Figures (every value is read from the report's CSV tables; TACO and ARCTIC are separate panels):
  f3a_levels_*      the six models side by side, true initial contact s_0 given (protocol A)
  f3a_paired_*      relative change between two models, paired per test sequence, with the paired intervals
  f3a_diversity_*   spread of 10 trajectories: future diffusion with s_0 fixed versus sampled s_0
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

REPORT = "reports/structure_aware_temporal_generation"
FIXED = f"{REPORT}/fixed_s0_metrics.csv"
PAIRED = f"{REPORT}/paired_comparisons.csv"
DIVERSITY = f"{REPORT}/diversity_metrics.csv"
SOURCES = [FIXED, PAIRED, DIVERSITY]

Row = dict[str, str]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]

# x positions of the "levels" figures and the models behind each bar
LEVELS = ["dense loss only", "+ R2 output loss", "+ hidden R2 head"]
FAMILIES = [("deterministic", ["D0", "D1", "D2"]), ("diffusion, 1 sample", ["S0", "S1", "S2"])]

# x positions of the "paired" figures: (metric column value, label)
PAIRED_METRICS = [
    ("E_C", "dense error"),
    ("part_hamming", "participation"),
    ("amount_l1", "amount"),
    ("centroid", "centroid"),
    ("normal", "normal"),
    ("q_rel_l1", "wrench"),
]

# x positions of the "diversity" figures: (label, source column value, protocol, model)
DIVERSITY_ROWS = [
    ("future diffusion, s_0 fixed", "future (fixed s_0, stochastic futures)", "A", "S0"),
    ("sampled s_0, deterministic future", "initial (sampled s_0, one future each)", "B", "D0"),
    ("sampled s_0, diffusion future", "initial + future (sampled s_0, one stochastic future each)", "B", "S0"),
    ("sampled s_0 held", "initial, no evolution (sampled s_0 held)", "B", "PERSIST"),
]

FIXED_SIZE_LOCATOR = "columns n_examples and n_seeds of the D0 / E_C / K1 rows (test-set size quoted in the note)"

# A generated map carries no finger identity: evaluate.py extracts participation / amount / centroid / normal (and the
# wrench profile built from them) with the ground-truth part label of every canonical point at that frame.
PART_LABEL_NOTE = ("The extraction uses the real hand's part label of every surface point at that frame; no model "
                   "outputs which finger touches.")
PAIRED_PART_LABEL_NOTE = ("Participation, amount, centroid, normal and wrench are extracted from the generated map "
                          "with the real hand's part label of every surface point at that frame.")


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick(rows: list[Row], **where: str) -> Row:
    """The single row whose columns equal every given filter value."""
    hits = [row for row in rows if all(row[column] == value for column, value in where.items())]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one row for {where}, found {len(hits)}")
    return hits[0]


def column(rows: list[Row], name: str) -> list[float]:
    return [float(row[name]) for row in rows]


def test_sizes(fixed: list[Row]) -> str:
    """'n = ... (TACO) / ... (ARCTIC) test sequences, k training seeds ...', read from the protocol-A table."""
    rows = [pick(fixed, dataset=dataset, model="D0", metric="E_C", stat="K1") for dataset, _ in DATASETS]
    sizes = " / ".join(f"{row['n_examples']} ({title})" for row, (_, title) in zip(rows, DATASETS))
    return f"n = {sizes} test sequences, {rows[0]['n_seeds']} training seeds averaged per sequence"


def reference_text(fixed: list[Row], metric: str, references: list[tuple[str, str]]) -> str:
    """'label value [lo, hi] (TACO), value [lo, hi] (ARCTIC); ...': the reference lines with their own intervals."""
    parts = []
    for model, label in references:
        rows = [pick(fixed, dataset=dataset, model=model, metric=metric, stat="K1") for dataset, _ in DATASETS]
        values = ", ".join(f"{float(row['value']):.3f} [{float(row['ci_lo']):.3f}, {float(row['ci_hi']):.3f}] ({title})"
                           for row, (_, title) in zip(rows, DATASETS))
        parts.append(f"{label} {values}")
    return "; ".join(parts)


def level_figure(fixed: list[Row], fig_id: str, title: str, metric: str, y_label: str, direction: str,
                 references: list[tuple[str, str]], note: str) -> dict:
    """Six models (3 supervision levels x 2 families) on one protocol-A metric; references = (model row, label)."""
    panels = []
    for dataset, panel_title in DATASETS:
        series = []
        for family, models in FAMILIES:
            picked = [pick(fixed, dataset=dataset, model=model, metric=metric, stat="K1") for model in models]
            series.append({"name": family, "values": column(picked, "value"),
                           "lo": column(picked, "ci_lo"), "hi": column(picked, "ci_hi")})
        refs = [{"axis": "y", "label": label,
                 "value": float(pick(fixed, dataset=dataset, model=model, metric=metric, stat="K1")["value"])}
                for model, label in references]
        panels.append({"title": panel_title,
                       "x": {"label": "training supervision", "categories": LEVELS},
                       "y": {"label": y_label, "direction": direction},
                       "series": series, "refs": refs})
    reference_rows = " and ".join(model for model, _ in references)
    locator = (f"rows protocol = A, metric = {metric}, stat = K1, model in D0 D1 D2 (deterministic) and S0 S1 S2 "
               f"(diffusion, sample 0); columns value, ci_lo, ci_hi; reference lines = model {reference_rows}, same "
               f"metric and stat, columns value (drawn), ci_lo, ci_hi (quoted in the note); test-set size = columns "
               f"n_examples, n_seeds; one panel per dataset")
    return {"id": fig_id, "type": "bar", "title": title, "panels": panels,
            "note": f"{note} Error bars: 95 % take-cluster bootstrap interval of each model's own mean (not of the "
                    f"difference between two models). The reference lines are test means with their own 95 % "
                    f"interval, which is not drawn: {reference_text(fixed, metric, references)}. {test_sizes(fixed)}.",
            "sources": [{"path": FIXED, "locator": locator}]}


def paired_figure(paired: list[Row], fixed: list[Row], fig_id: str, title: str,
                  pairs: list[tuple[str, str, str]], note: str) -> dict:
    """Relative change (%) of model b against model a on six test metrics; pairs = (a, b, series name)."""
    panels = []
    for dataset, panel_title in DATASETS:
        series = []
        for a, b, name in pairs:
            picked = [pick(paired, dataset=dataset, protocol="A", a=a, b=b, stat="K1", metric=metric)
                      for metric, _ in PAIRED_METRICS]
            series.append({"name": name, "values": column(picked, "rel_change_pct"),
                           "lo": column(picked, "rel_lo"), "hi": column(picked, "rel_hi")})
        panels.append({"title": panel_title,
                       "x": {"label": "test metric", "categories": [label for _, label in PAIRED_METRICS]},
                       "y": {"label": "relative change of the test error (%)", "direction": "lower_better"},
                       "series": series})
    wanted = "; ".join(f"a = {a}, b = {b}" for a, b, _ in pairs)
    metrics = ", ".join(metric for metric, _ in PAIRED_METRICS)
    return {"id": fig_id, "type": "bar", "title": title, "panels": panels,
            "note": f"{note} {PAIRED_PART_LABEL_NOTE} Error bars: 95 % take-cluster bootstrap interval of the paired "
                    f"ratio; {test_sizes(fixed)}.",
            "sources": [{"path": PAIRED, "locator": f"rows protocol = A, stat = K1, pairs {wanted}, metric in {metrics}; "
                                                    f"columns rel_change_pct, rel_lo, rel_hi; one panel per dataset"},
                        {"path": FIXED, "locator": FIXED_SIZE_LOCATOR}]}


def diversity_figure(diversity: list[Row], fixed: list[Row], fig_id: str, title: str, metric: str, y_label: str) -> dict:
    """Mean pairwise distance between the 10 trajectories of one test sequence, by where the randomness enters."""
    panels = []
    for dataset, panel_title in DATASETS:
        picked = [pick(diversity, dataset=dataset, source=source, protocol=protocol, model=model, metric=metric)
                  for _, source, protocol, model in DIVERSITY_ROWS]
        panels.append({"title": panel_title,
                       "x": {"label": "where the 10 trajectories differ", "categories": [row[0] for row in DIVERSITY_ROWS]},
                       "y": {"label": y_label, "direction": "none"},
                       "series": [{"name": "mean pairwise distance", "values": column(picked, "value"),
                                   "lo": column(picked, "ci_lo"), "hi": column(picked, "ci_hi")}]})
    wanted = "; ".join(f"'{label}' = source '{source}', protocol {protocol}, model {model}"
                       for label, source, protocol, model in DIVERSITY_ROWS)
    note = ("Mean pairwise distance between the 10 trajectories generated for one test sequence, for the models "
            "trained with the dense loss only; the first bar uses the true s_0 and frames 1-63, the other three use "
            "10 sampled s_0 and frames 0-63, where frame 0 already differs between the draws. Error bars: 95 % "
            f"take-cluster bootstrap interval; {test_sizes(fixed)} (the held-s_0 bar involves no trained model).")
    return {"id": fig_id, "type": "bar", "title": title, "panels": panels, "note": note,
            "sources": [{"path": DIVERSITY, "locator": f"rows metric = {metric}; {wanted}; columns value, ci_lo, "
                                                       f"ci_hi; one panel per dataset"},
                        {"path": FIXED, "locator": FIXED_SIZE_LOCATOR}]}


def build(src: Callable[[str], Path]) -> list[dict]:
    fixed = read_rows(src(FIXED))
    paired = read_rows(src(PAIRED))
    diversity = read_rows(src(DIVERSITY))

    setting = "true initial contact s_0 given, frames 1-63, mean over test sequences, one sample for the diffusion models"
    paired_setting = "paired per test sequence (true s_0 given, frames 1-63)"
    one_sample = "one sample per sequence for the diffusion models"

    return [
        level_figure(fixed, "f3a_levels_dense_error", "Dense error of the six generators", "E_C",
                     "dense error (per-frame L2 distance to the real map)", "lower_better",
                     [("PERSIST", "hold s_0 (no model)")],
                     f"L2 distance between the generated and the real 512-point contact map, averaged over frames "
                     f"({setting}); lower is better, the line is the error of keeping s_0 unchanged."),
        level_figure(fixed, "f3a_levels_participation", "Participation error of the six generators", "part_hamming",
                     "participation error (fraction of the 6 hand parts wrong)", "lower_better",
                     [("GT", "extraction floor (real map)"), ("PERSIST", "hold s_0 (no model)")],
                     f"Fraction of the six hand parts whose touching / not-touching state, extracted from the "
                     f"generated map, disagrees with the real hand ({setting}); lower is better, the lines are the "
                     f"same extraction applied to the real map and the error of keeping s_0 unchanged. "
                     f"{PART_LABEL_NOTE}"),
        level_figure(fixed, "f3a_levels_wrench", "Wrench-profile error of the six generators", "q_rel_l1",
                     "wrench-profile error (relative L1)", "lower_better",
                     [("GT", "extraction floor (real map)"), ("PERSIST", "hold s_0 (no model)")],
                     f"Relative L1 error of the modelled wrench-capability profile (76 directions, a proxy for "
                     f"functional grasp capability) computed from the generated map against that of the real grasp "
                     f"({setting}); lower is better, the lines are the same extraction applied to the real map and "
                     f"the error of keeping s_0 unchanged. {PART_LABEL_NOTE}"),
        level_figure(fixed, "f3a_levels_frame_change", "Frame-to-frame change of the generated maps", "jitter_dense",
                     "L2 change of the dense map between consecutive frames", "none",
                     [("GT", "real trajectories")],
                     f"Mean L2 change of the 512-point contact map from one frame to the next ({setting}); the line "
                     f"is the same quantity for the real trajectories, so a bar far above it is frame-to-frame "
                     f"jitter and a bar below it a smoother-than-real trajectory."),
        paired_figure(paired, fixed, "f3a_paired_supervision_deterministic",
                      "Effect of R2 supervision on the deterministic generator",
                      [("D0", "D1", "+ R2 output loss vs dense only"), ("D1", "D2", "+ hidden R2 head vs output loss")],
                      f"Relative change of the test error when a structural training term is added to the "
                      f"deterministic generator, {paired_setting}; positive = the added term made the metric worse, "
                      f"0 = no change."),
        paired_figure(paired, fixed, "f3a_paired_supervision_diffusion",
                      "Effect of R2 supervision on the sequence-diffusion generator",
                      [("S0", "S1", "+ R2 output loss vs dense only"), ("S1", "S2", "+ hidden R2 head vs output loss")],
                      f"Relative change of the test error when a structural training term is added to the "
                      f"sequence-diffusion generator, {paired_setting}, {one_sample}; positive = the added term "
                      f"made the metric worse, 0 = no change."),
        paired_figure(paired, fixed, "f3a_paired_diffusion_vs_deterministic",
                      "One diffusion sample against the deterministic prediction",
                      [("D0", "S0", "dense loss only"), ("D1", "S1", "+ R2 output loss"), ("D2", "S2", "+ hidden R2 head")],
                      f"Relative change of the test error of the diffusion model against the deterministic model "
                      f"trained with the same supervision, {paired_setting}, {one_sample}; positive = the diffusion "
                      f"sample is worse, 0 = no change."),
        diversity_figure(diversity, fixed, "f3a_diversity_r2",
                         "Structural (R2) diversity: sampled s_0 versus future diffusion",
                         "D_Z", "distance between standardised R2 summaries (48 numbers)"),
        diversity_figure(diversity, fixed, "f3a_diversity_wrench",
                         "Wrench-profile diversity: sampled s_0 versus future diffusion",
                         "D_q", "symmetric relative L1 distance between wrench profiles"),
        diversity_figure(diversity, fixed, "f3a_diversity_dense",
                         "Dense diversity: sampled s_0 versus future diffusion",
                         "D_C", "per-frame L2 distance between dense maps"),
    ]
