"""Figure data for the page blocks F2-A, F2-B, F2-C, F2-subA and F2-subB (the representation ladder).

Source report: ``reports/structure_variance_boundary/structure_variance_report.md`` (structure / variance
boundary study; TACO and ARCTIC only, OakInk2 was not part of it). Every plotted value is read from the
report's root tables at run time:

    functional_probe.csv             Experiment A, wrench-profile probe per rung   f2a_wrench_error_by_rung
    incremental_gain.csv             error after / before adding one block         f2a_block_gain, f2suba_*, f2subb_*
    saturation_summary.csv           gaps to FULL and to the best rung             f2a_gap_to_best,
                                                                                   f2b_future_structure_gap_by_rung
    saturation_choice.csv            thresholds of the saturation rule             reference lines of f2a_gap_to_best
    temporal_ratios.csv              Experiment B, paired error ratios             f2b_future_structure_ratio_by_horizon,
                                                                                   f2b_r2_vs_persistence
    residual_prediction.csv          Experiment D, the residual left by R2         f2c_*
    temporal_prediction.csv          number of test takes and seeds                figure notes only
    representation_definitions.json  sizes of the rungs and blocks                 labels and notes only

One panel per dataset; nothing is pooled or averaged over datasets.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

Row = dict[str, str]
Src = Callable[[str], Path]

REPORT = "reports/structure_variance_boundary"
FUNCTIONAL = f"{REPORT}/functional_probe.csv"
GAIN = f"{REPORT}/incremental_gain.csv"
SATURATION = f"{REPORT}/saturation_summary.csv"
CHOICE = f"{REPORT}/saturation_choice.csv"
RATIOS = f"{REPORT}/temporal_ratios.csv"
TEMPORAL = f"{REPORT}/temporal_prediction.csv"
RESIDUAL = f"{REPORT}/residual_prediction.csv"
DEFINITIONS = f"{REPORT}/representation_definitions.json"
SOURCES = [FUNCTIONAL, GAIN, SATURATION, CHOICE, RATIOS, TEMPORAL, RESIDUAL, DEFINITIONS]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]
RUNGS = ["R0", "R1", "R2", "R3", "R4", "R5", "CONTACT_FULL", "FULL"]
HORIZONS = ["1", "4", "8"]
HORIZON_LABELS = ["h = 1", "h = 4", "h = 8"]
RUNG_AXIS = "representation (ordered by size; R4 = R2 + coarse hand, R5 = R3 + coarse hand)"
INTERVALS = "error bars are 95 % intervals of a bootstrap over the test takes"
TAKES_ONLY = {"path": TEMPORAL, "locator": "column n_takes (number of test takes, note only)"}
TAKES_SEEDS = {"path": TEMPORAL, "locator": "columns n_takes, n_seeds (note only)"}


def read_rows(path: Path) -> list[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def num(text: str) -> float | None:
    """A table cell as a float; an empty or non-finite cell is a missing value."""
    if text == "":
        return None
    value = float(text)
    return value if math.isfinite(value) else None


def one(rows: list[Row], **where: str) -> Row:
    """The single row in which every named column equals the given text."""
    hits = [row for row in rows if all(row[column] == text for column, text in where.items())]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} rows match {where}")
    return hits[0]


def series(name: str, rows: list[Row], value: str, lo: str | None = None, hi: str | None = None) -> dict:
    """One series from an ordered list of table rows; lo / hi only when the table has interval columns."""
    out: dict = {"name": name, "values": [num(row[value]) for row in rows]}
    if lo and hi:
        out["lo"] = [num(row[lo]) for row in rows]
        out["hi"] = [num(row[hi]) for row in rows]
    return out


def cells(name: str, row: Row, columns: list[tuple[str, str, str]]) -> dict:
    """One series whose x positions are (value, lo, hi) column triples of a single table row."""
    return {"name": name,
            "values": [num(row[value]) for value, _, _ in columns],
            "lo": [num(row[lo]) for _, lo, _ in columns],
            "hi": [num(row[hi]) for _, _, hi in columns]}


def per_dataset(rows: dict[str, Row], column: str) -> str:
    """A count column of one row per dataset as text, for example 'TACO 186 / ARCTIC 44'."""
    return " / ".join(f"{label} {int(float(rows[key][column]))}" for key, label in DATASETS)


def shared(rows: dict[str, Row], column: str) -> int:
    """A count column that must be the same for every dataset (number of seeds)."""
    values = {int(float(row[column])) for row in rows.values()}
    if len(values) != 1:
        raise ValueError(f"{column} differs between datasets: {sorted(values)}")
    return values.pop()


def build(src: Src) -> list[dict]:
    functional = read_rows(src(FUNCTIONAL))
    gain = read_rows(src(GAIN))
    saturation = read_rows(src(SATURATION))
    choice = read_rows(src(CHOICE))
    ratios = read_rows(src(RATIOS))
    temporal = read_rows(src(TEMPORAL))
    residual = read_rows(src(RESIDUAL))
    definitions = json.loads(src(DEFINITIONS).read_text(encoding="utf-8"))

    # Sizes quoted in labels and notes are read from the tables, not typed.
    probe_r2 = {ds: one(functional, dataset=ds, rep="R2", probe="mlpG_seedavg", target="q") for ds, _ in DATASETS}
    model_r2 = {ds: one(temporal, dataset=ds, rep="R2", metric="T4_rel_l1", h="8") for ds, _ in DATASETS}
    decoder_r2 = {ds: one(residual, dataset=ds, zstar="R2", model="decoder", block="C", metric="decoder_r2_test")
                  for ds, _ in DATASETS}
    frames = per_dataset(probe_r2, "n_frames")
    takes = per_dataset(model_r2, "n_takes")
    probe_seeds = shared(probe_r2, "n_seeds")
    model_seeds = shared(model_r2, "n_seeds")
    residual_seeds = shared(decoder_r2, "n_seeds")
    rung_dim: dict[str, int] = definitions["dims"]
    block_dim: dict[str, int] = definitions["blocks"]
    window: int = definitions["window_frames"]
    dims = ", ".join(f"{rep} {rung_dim[rep]}" for rep in RUNGS)
    highlight = {"category": "R2", "label": f"R2 ({rung_dim['R2']} numbers per frame)"}
    rung_sizes = {"path": DEFINITIONS, "locator": "key dims (numbers per frame of each rung: highlight label, note)"}
    thresholds = sorted({row["threshold"] for row in choice}, key=float)
    # Inputs shared by every temporal model (report section 5); 'causal' refers to the hand / contact window only.
    shared_inputs = ("every model also receives the object descriptor and the local object trajectory, which includes the "
                     "object's future states")
    # The participation error used in the paired ratios, as the definitions file words it (text before the bracket).
    participation_error = definitions["T1_proxy_for_paired_gaps"].split(" (")[0]
    # R5 → FULL is not an addition of blocks: FULL must hold none of R5's blocks for the sentence in the note to be true.
    ladder: dict[str, list[str]] = definitions["ladder"]
    if not set(ladder["R5"]).isdisjoint(ladder["FULL"]):
        raise ValueError("FULL shares blocks with R5: rewrite the note of f2a_block_gain")
    full_blocks = " + ".join(definitions["block_labels"][block] for block in ladder["FULL"])

    # ---------------------------------------------------------------- F2-A: wrench probe error by rung
    panels = []
    for ds, label in DATASETS:
        probe = [one(functional, dataset=ds, rep=rep, probe="mlpG_seedavg", target="q") for rep in RUNGS]
        ridge = [one(functional, dataset=ds, rep=rep, probe="ridgeG", target="q") for rep in RUNGS]
        mean = one(functional, dataset=ds, rep="MEAN", probe="mlpG_seedavg", target="q")
        panels.append({
            "title": label,
            "x": {"label": RUNG_AXIS, "categories": RUNGS},
            "y": {"label": "wrench-profile reconstruction error (relative L1)", "direction": "lower_better"},
            "series": [series("MLP probe", probe, "rel_l1", "rel_l1_lo", "rel_l1_hi"),
                       series("ridge probe", ridge, "rel_l1")],
            "refs": [{"axis": "y", "value": num(mean["rel_l1"]), "label": "train-mean profile"}],
            "highlight": highlight,
        })
    fig_wrench = {
        "id": "f2a_wrench_error_by_rung", "type": "line",
        "title": "Wrench capability recovered from each representation",
        "panels": panels,
        "note": ("Relative L1 error of a fixed small probe that reconstructs a frame's 76-direction wrench-capability profile "
                 "from that frame's representation plus the object descriptor (lower is better); test frames in contact: "
                 f"{frames} ({takes} test takes); MLP probe = mean of {probe_seeds} seeds, ridge probe = one fit; {INTERVALS} "
                 f"(MLP probe only). Numbers per frame: {dims}; the comparison shows what this fixed probe extracts, not the "
                 "information content."),
        "sources": [{"path": FUNCTIONAL,
                     "locator": "rows probe = mlpG_seedavg, target = q (rep MEAN = train-mean reference line): columns rel_l1, "
                                "rel_l1_lo, rel_l1_hi, n_frames; rows probe = ridgeG, target = q: column rel_l1"},
                    dict(TAKES_ONLY), dict(rung_sizes)],
    }

    # ---------------------------------------------------------------- F2-A: error ratio per added block
    steps = [("R0", "R1", "+ amount (R0→R1)"), ("R1", "R2", "+ centroid / normal (R1→R2)"),
             ("R2", "R3", "+ spread / patches (R2→R3)"), ("R2", "R4", "+ coarse hand (R2→R4)"),
             ("R5", "FULL", "+ dense detail (R5→FULL)")]
    panels = []
    dense_step_names = set()
    for ds, label in DATASETS:
        rows = [one(gain, dataset=ds, frm=frm, to=to) for frm, to, _ in steps]
        dense_step_names.add(rows[-1]["added"])
        panels.append({
            "title": label,
            "x": {"label": "step of the ladder (block added; the last step replaces R5 by the dense state)",
                  "categories": [name for _, _, name in steps]},
            "y": {"label": "error after / error before (below 1 = the step helps)", "direction": "lower_better"},
            "series": [series("wrench probe", rows, "functional_ratio", "functional_lo", "functional_hi"),
                       series("future structure", rows, "temporal_ratio", "temporal_lo", "temporal_hi")],
        })
    if len(dense_step_names) != 1:
        raise ValueError(f"the R5 → FULL step is named differently per dataset: {sorted(dense_step_names)}")
    fig_blocks = {
        "id": "f2a_block_gain", "type": "bar",
        "title": "What each added block buys",
        "panels": panels,
        "note": ("Error after / before adding one block (below 1 = it helps, lower is better), for the wrench probe and for the "
                 "composite future-structure error (future participation, amount, centroid / normal and wrench profile, mean "
                 f"over h = 1, 4, 8 frames). {takes} test takes; {INTERVALS} (paired); one fixed probe and one fixed temporal "
                 f"recipe for all rungs. The last step (the table's '{dense_step_names.pop()}') is not a pure addition: FULL is "
                 f"{full_blocks} and does not contain the compact blocks of R5 explicitly."),
        "sources": [{"path": GAIN,
                     "locator": "rows (frm, to) = (R0, R1), (R1, R2), (R2, R3), (R2, R4), (R5, FULL): columns functional_ratio, "
                                "functional_lo, functional_hi, temporal_ratio, temporal_lo, temporal_hi, added (name of the "
                                "last step, note)"},
                    dict(TAKES_ONLY),
                    {"path": DEFINITIONS, "locator": "keys ladder.R5, ladder.FULL, block_labels (content of FULL, note)"}],
    }

    # ---------------------------------------------------------------- F2-A caveat: gap to the best rung
    compact = ["R2", "R3", "R4", "R5"]
    panels = []
    best = []
    for ds, label in DATASETS:
        rows = [one(saturation, dataset=ds, rep=rep) for rep in compact]
        best.append(f"{label}: wrench {rows[0]['best_functional']}, future structure {rows[0]['best_temporal']}")
        panels.append({
            "title": label,
            "x": {"label": "representation", "categories": compact},
            "y": {"label": "error above the best representation (gap = ratio − 1)", "direction": "lower_better"},
            "series": [series("wrench probe", rows, "functional_gap_best", "functional_gap_best_lo", "functional_gap_best_hi"),
                       series("future structure", rows, "temporal_gap_best", "temporal_gap_best_lo", "temporal_gap_best_hi")],
            "refs": [{"axis": "y", "value": num(text), "label": f"{float(text) * 100:g} % threshold"} for text in thresholds],
            "highlight": highlight,
        })
    fig_best = {
        "id": "f2a_gap_to_best", "type": "bar",
        "title": "How far each compact representation is from the best one",
        "panels": panels,
        "note": ("Relative error gap to the representation with the lowest error on each axis (" + "; ".join(best) + "); "
                 f"0 = the best, lower is better; {takes} test takes; {INTERVALS} (paired). The lines are the report's saturation "
                 "thresholds: a representation counts as saturated when its gap is below the threshold or its interval reaches 0."),
        "sources": [{"path": SATURATION,
                     "locator": "rows rep = R2, R3, R4, R5: columns functional_gap_best(_lo, _hi), temporal_gap_best(_lo, _hi), "
                                "best_functional, best_temporal"},
                    {"path": CHOICE, "locator": "column threshold (distinct values, reference lines)"},
                    dict(TAKES_ONLY), dict(rung_sizes)],
    }

    # ---------------------------------------------------------------- F2-B: future-structure error by rung
    panels = []
    for ds, label in DATASETS:
        rows = [one(saturation, dataset=ds, rep=rep) for rep in RUNGS]
        panels.append({
            "title": label,
            "x": {"label": RUNG_AXIS, "categories": RUNGS},
            "y": {"label": "future-structure error relative to FULL (gap = ratio − 1)", "direction": "lower_better"},
            "series": [series("T1–T4 composite", rows, "temporal_gap", "temporal_gap_lo", "temporal_gap_hi")],
            "highlight": highlight,
        })
    fig_future = {
        "id": "f2b_future_structure_gap_by_rung", "type": "line",
        "title": "Predicting the future structured contact from each representation",
        "panels": panels,
        "note": ("Composite error of predicting the structured contact state 1, 4 and 8 frames ahead from a causal "
                 f"{window}-frame window of each representation: the mean error ratio to FULL over future participation (T1), "
                 "amount (T2), centroid / normal (T3) and wrench profile (T4) and over the three horizons, shown as ratio − 1 "
                 f"(0 = FULL, negative = lower error, lower is better). {model_seeds} seeds, {takes} test takes; {INTERVALS} "
                 "(paired); all rungs share one fixed temporal recipe, so FULL being worse is a statement about that recipe, "
                 f"not about the information in the dense state; {shared_inputs}. A comparison between inputs, not an "
                 "absolute forecasting result (see the comparison with persistence)."),
        "sources": [{"path": SATURATION,
                     "locator": "all 8 rep rows per dataset: columns temporal_gap, temporal_gap_lo, temporal_gap_hi"},
                    dict(TAKES_SEEDS),
                    {"path": DEFINITIONS, "locator": "keys dims, window_frames (highlight label, note)"}],
    }

    # ---------------------------------------------------------------- F2-B detail: the same composite per horizon
    not_full = RUNGS[:-1]
    panels = []
    for ds, label in DATASETS:
        panels.append({
            "title": label,
            "x": {"label": RUNG_AXIS, "categories": not_full},
            "y": {"label": "future-structure error / FULL's error (below 1 = lower than FULL)", "direction": "lower_better"},
            "series": [series(name, [one(ratios, dataset=ds, rep=rep, vs="FULL", metric="composite_T1_T4", h=h)
                                     for rep in not_full], "ratio", "lo", "hi") for h, name in zip(HORIZONS, HORIZON_LABELS)],
            "highlight": highlight,
        })
    fig_horizon = {
        "id": "f2b_future_structure_ratio_by_horizon", "type": "line",
        "title": "The same comparison per prediction horizon",
        "panels": panels,
        "note": ("The T1–T4 composite error as a ratio to FULL's error, separately for the prediction horizons h = 1, 4, 8 "
                 "frames (FULL is the denominator and therefore has no point); lower is better. "
                 f"{model_seeds} seeds, {takes} test takes; {INTERVALS} (paired); {shared_inputs}."),
        "sources": [{"path": RATIOS,
                     "locator": "rows vs = FULL, metric = composite_T1_T4, h = 1, 4, 8, rep = R0 … CONTACT_FULL: columns ratio, "
                                "lo, hi"},
                    dict(TAKES_SEEDS), dict(rung_sizes)],
    }

    # ---------------------------------------------------------------- F2-B caveat: the R2 model against persistence
    targets = [("T1_l1_ratio", "participation"), ("T2_rel_l1_ratio", "amount"), ("T3_cent_ratio", "centroid"),
               ("T3_ang_ratio", "normal"), ("T4_rel_l1_ratio", "wrench profile")]
    # The note says the R2 model is above persistence by macro AUPRC everywhere: stop if the table stops saying so.
    for ds, _ in DATASETS:
        for h in HORIZONS:
            model_auprc = num(one(temporal, dataset=ds, rep="R2", metric="auprc_macro", h=h)["value"])
            persist_auprc = num(one(temporal, dataset=ds, rep="PERSIST", metric="auprc_macro", h=h)["value"])
            if model_auprc is None or persist_auprc is None or model_auprc <= persist_auprc:
                raise ValueError(f"{ds} h={h}: R2 macro AUPRC is not above persistence; rewrite the note of f2b_r2_vs_persistence")
    panels = []
    for ds, label in DATASETS:
        panels.append({
            "title": label,
            "x": {"label": "future target predicted from the R2 window", "categories": [name for _, name in targets]},
            "y": {"label": "model error / persistence error (below 1 = better than persistence)", "direction": "lower_better"},
            "series": [series(name, [one(ratios, dataset=ds, rep="R2", vs="PERSIST", metric=metric, h=h) for metric, _ in targets],
                              "ratio", "lo", "hi") for h, name in zip(HORIZONS, HORIZON_LABELS)],
        })
    fig_persist = {
        "id": "f2b_r2_vs_persistence", "type": "bar",
        "title": "The R2 model against carrying the current value forward",
        "panels": panels,
        "note": ("Error of the model fed the R2 window divided by the error of persistence (the true current target carried "
                 "forward), per future target and horizon; below 1 the model beats persistence, lower is better. "
                 f"{takes} test takes; {INTERVALS} (paired); all models predict absolute future targets under one fixed recipe, "
                 "so the report takes the comparison between representations, not these absolute levels, as its evidence. "
                 f"Participation is scored by the {participation_error}, not by the macro AUPRC of the report's tables (a "
                 "ranking measure, by which the same model is above the persistence line at every horizon on both datasets); "
                 "for centroid and normal the model error is averaged over the parts in contact at t + h, the persistence "
                 "error over the parts in contact at both t and t + h, so these two ratios are not like for like."),
        "sources": [{"path": RATIOS,
                     "locator": "rows rep = R2, vs = PERSIST, metric = T1_l1_ratio, T2_rel_l1_ratio, T3_cent_ratio, T3_ang_ratio, "
                                "T4_rel_l1_ratio, h = 1, 4, 8: columns ratio, lo, hi"},
                    {"path": TEMPORAL,
                     "locator": "column n_takes (note); rows metric = auprc_macro, rep = R2 and PERSIST, h = 1, 4, 8: column "
                                "value (checked for the AUPRC sentence of the note, not plotted)"},
                    {"path": DEFINITIONS, "locator": "key T1_proxy_for_paired_gaps (definition of the participation error, note)"}],
    }

    # ---------------------------------------------------------------- F2-C: how much of the dense state R2 determines
    blocks = [("C", f"contact map ({block_dim['C']}-D)"), ("H", f"hand points ({block_dim['H']}-D)")]
    panels = []
    for ds, label in DATASETS:
        panels.append({
            "title": label,
            "x": {"label": "part of the dense state", "categories": [name for _, name in blocks]},
            "y": {"label": "share of variance reproduced from the representation alone (decoder test R²)", "direction": "none"},
            "series": [series(f"from {z}", [one(residual, dataset=ds, zstar=z, model="decoder", block=b, metric="decoder_r2_test")
                                             for b, _ in blocks], "value") for z in ("R2", "R3")],
        })
    fig_decoder = {
        "id": "f2c_dense_explained_by_r2", "type": "bar",
        "title": "How much of the dense state the compact representation determines",
        "panels": panels,
        "note": ("Test R² of a frame-wise MLP decoder that reconstructs the standardised dense state from one frame of R2 (or R3) "
                 "alone; the residual of the next figures is the remainder, 1 − R². "
                 f"Mean of {residual_seeds} decoder seeds, no interval in the table; the residual is what this decoder leaves, "
                 "not a canonical decomposition."),
        "sources": [{"path": RESIDUAL,
                     "locator": "rows model = decoder, metric = decoder_r2_test, zstar = R2 and R3, block = C and H: columns "
                                "value, n_seeds"},
                    {"path": DEFINITIONS, "locator": "key blocks (sizes of C and H, category labels)"}],
    }

    # ---------------------------------------------------------------- F2-C: predicting the future residual
    predictors = [("last", "last value (persistence)"), ("history", "causal, dense history"),
                  ("history_delta", "causal, + residual history"), ("oracle_delta", "non-causal oracle")]
    # Seeds behind the residual ratios, read from the plotted rows: the learned predictors, and the baseline (the residual
    # energy and the last-value error of the table come from fewer decoder seeds than the learned predictors).
    def residual_row(ds: str, block: str, model: str, metric: str, h: str) -> Row:
        return one(residual, dataset=ds, zstar="R2", block=block, model=model, metric=metric, h=h)

    def seed_count(models: tuple[str, ...]) -> int:
        """The number of seeds of the R2-residual ratio rows of these models; it must be the same in every row."""
        counts = {int(float(residual_row(ds, block, model, metric, h)["n_seeds"]))
                  for ds, _ in DATASETS for block in ("C", "H") for model in models
                  for metric in ("mse_ratio_to_zero", "mse_ratio_to_last") for h in HORIZONS}
        if len(counts) != 1:
            raise ValueError(f"seed counts of the residual rows of {models} differ: {sorted(counts)}")
        return counts.pop()

    predictor_seeds = seed_count(("history", "history_delta", "oracle_delta"))
    baseline_seed_count = seed_count(("last",))
    # What the table itself gives for the zero-residual predictor (it is not exactly 1 because of the baseline seeds).
    zero_reads = "; ".join(
        f"{label} " + " / ".join(f"{float(residual_row(ds, 'C', 'zero', 'mse_ratio_to_zero', h)['value']):.3f}" for h in HORIZONS)
        for ds, label in DATASETS)
    baseline_note = (f"The residual energy and the last-value error are taken from {baseline_seed_count} decoder "
                     f"seed{'' if baseline_seed_count == 1 else 's'}, the learned predictors are means of {predictor_seeds} "
                     "seeds with their own decoders")
    panels = []
    for ds, label in DATASETS:
        panels.append({
            "title": label,
            "x": {"label": "prediction horizon (frames)", "categories": HORIZON_LABELS},
            "y": {"label": "error of the predicted contact residual / residual energy", "direction": "lower_better"},
            "series": [series(name, [one(residual, dataset=ds, zstar="R2", block="C", model=model, metric="mse_ratio_to_zero", h=h)
                                     for h in HORIZONS], "value", "lo", "hi") for model, name in predictors],
        })
    fig_residual = {
        "id": "f2c_residual_predictability", "type": "bar",
        "title": "Predicting the future contact residual left by R2",
        "panels": panels,
        "note": ("Mean squared error of the predicted future contact-map residual (dense contact minus its decoding from R2) "
                 "divided by the residual's own energy (about 1 = predicting a zero residual, lower is better). 'Last value' "
                 "keeps the current residual; the causal predictors see the dense history without or with the residual's own "
                 "past; the oracle also receives the true future R2 (non-causal); all learned predictors also receive the "
                 f"object descriptor and the object trajectory including its future states. {takes} test takes; {INTERVALS} "
                 f"(paired). {baseline_note}, so the zero-residual predictor itself reads {zero_reads} at h = 1 / 4 / 8 "
                 "in the table instead of exactly 1."),
        "sources": [{"path": RESIDUAL,
                     "locator": "rows zstar = R2, block = C, metric = mse_ratio_to_zero, model = last, history, history_delta, "
                                "oracle_delta, h = 1, 4, 8: columns value, lo, hi, n_seeds; rows model = zero (same filter): "
                                "column value (note only)"},
                    dict(TAKES_ONLY)],
    }

    # ---------------------------------------------------------------- F2-C: the residual's change beyond persistence
    changes = [("C", "history_delta", "contact, causal"), ("C", "oracle_delta", "contact, oracle"),
               ("H", "history_delta", "hand, causal"), ("H", "oracle_delta", "hand, oracle")]
    panels = []
    for ds, label in DATASETS:
        last = one(residual, dataset=ds, zstar="R2", block="C", model="last", metric="mse_ratio_to_last", h="1")
        panels.append({
            "title": label,
            "x": {"label": "prediction horizon (frames)", "categories": HORIZON_LABELS},
            "y": {"label": "error / error of the last value (below 1 = the change is partly predicted)",
                  "direction": "lower_better"},
            "series": [series(name, [one(residual, dataset=ds, zstar="R2", block=b, model=model, metric="mse_ratio_to_last", h=h)
                                     for h in HORIZONS], "value", "lo", "hi") for b, model, name in changes],
            "refs": [{"axis": "y", "value": num(last["value"]), "label": "last value (persistence)"}],
        })
    fig_change = {
        "id": "f2c_residual_change_vs_last", "type": "bar",
        "title": "Is the residual's change predictable beyond persistence?",
        "panels": panels,
        "note": ("Error of predictors that start from the current residual and predict its change, divided by the error of "
                 "keeping the current residual; below 1 the change is partly predicted, lower is better. Causal = dense history + "
                 "residual history; oracle = the same plus the true future R2 (non-causal); both also receive the object descriptor and "
                 "the object trajectory, which includes the object's future states; the page draws the contact-map residual, the "
                 f"hand-point rows are in the same source table. {takes} test takes; {INTERVALS} (paired). {baseline_note}, so every ratio "
                 f"carries a small offset from the decoder seed (the zero-residual predictor itself reads {zero_reads} at "
                 "h = 1 / 4 / 8 instead of exactly 1) and ratios very close to 1 should not be read as a gain or a loss."),
        "sources": [{"path": RESIDUAL,
                     "locator": "rows zstar = R2, metric = mse_ratio_to_last, block = C and H, model = history_delta, "
                                "oracle_delta, h = 1, 4, 8: columns value, lo, hi, n_seeds; reference line = row model = last, "
                                "block = C, h = 1 (its n_seeds for the note)"},
                    dict(TAKES_ONLY)],
    }

    # ---------------------------------------------------------------- F2-subA / F2-subB: coarse hand and topology
    axes = [("functional_ratio", "functional_lo", "functional_hi"),
            ("temporal_ratio_h1", "temporal_lo_h1", "temporal_hi_h1"),
            ("temporal_ratio_h4", "temporal_lo_h4", "temporal_hi_h4"),
            ("temporal_ratio_h8", "temporal_lo_h8", "temporal_hi_h8"),
            ("temporal_ratio", "temporal_lo", "temporal_hi")]
    axis_labels = ["wrench probe", "future structure, h = 1", "future structure, h = 4", "future structure, h = 8",
                   "future structure, mean over h"]

    def gain_figure(fig_id: str, title: str, block: str, block_key: str, pairs: list[tuple[str, str]]) -> dict:
        """Error ratios of the ladder steps that add one block, for the wrench probe and the future-structure composite."""
        gain_panels = []
        for ds, label in DATASETS:
            gain_panels.append({
                "title": label,
                "x": {"label": "what is measured", "categories": axis_labels},
                "y": {"label": "error after / error before (below 1 = the added block helps)", "direction": "lower_better"},
                "series": [cells(f"{frm} → {to}", one(gain, dataset=ds, frm=frm, to=to), axes) for frm, to in pairs],
            })
        return {
            "id": fig_id, "type": "bar", "title": title, "panels": gain_panels,
            "note": (f"Error after / before adding {block} ({block_dim[block_key]} numbers per frame) to the representation "
                     "(below 1 = it helps, lower is better), for the wrench probe and for the composite future-structure error "
                     "(participation, amount, centroid / normal, wrench profile) at h = 1, 4, 8 frames and averaged over them. "
                     f"{takes} test takes; {INTERVALS} (paired)."),
            "sources": [{"path": GAIN,
                         "locator": "rows (frm, to) = " + ", ".join(f"({a}, {b})" for a, b in pairs) + ": columns "
                                    "functional_ratio, temporal_ratio_h1, temporal_ratio_h4, temporal_ratio_h8, temporal_ratio and "
                                    "their _lo / _hi columns"},
                        dict(TAKES_ONLY),
                        {"path": DEFINITIONS, "locator": f"key blocks.{block_key} (size of the added block, note)"}],
        }

    fig_hand = gain_figure("f2suba_coarse_hand_gain", "Adding the coarse hand pose",
                           "the coarse hand pose", "hand", [("R2", "R4"), ("R3", "R5")])
    fig_topology = gain_figure("f2subb_topology_gain", "Adding contact spread and patch count",
                               "contact spread and patch count", "topo", [("R2", "R3")])

    return [fig_wrench, fig_blocks, fig_best, fig_future, fig_horizon, fig_persist,
            fig_decoder, fig_residual, fig_change, fig_hand, fig_topology]
