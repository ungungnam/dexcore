"""Figure data for the page blocks F3-subA and F3-subB (z joint-decoder follow-up).

F3-subA: enlarging the contact decoder (with a longer budget and selection on the dense loss) left the
dense error of the z-mediated generator "similar" by the study's decision rule (M2 against M1; on TACO a
small resolved reduction below the rule's threshold), and both decoders reach the same dense error from
the same predicted z.
F3-subB: the temporal network reproduces the training z trajectories and predicts held-out ones much
less accurately; the dense error follows the latent it is given.

Source report: reports/z_joint_decoder_followup/report.md (tables written by zj_aggregate.py).
Models: M0 = direct dense prediction (Stage-2 B0, reused), M1 = previous z-mediated model (Stage-2 B1,
reused, already trained jointly on the predicted z), M2 = z-mediated model of this study (6-block decoder).
Every plotted value is read from the CSV / JSON tables at run time; nothing is pooled across datasets.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

REPORT = "reports/z_joint_decoder_followup"
MAIN = f"{REPORT}/main_metrics.csv"
PAIRED = f"{REPORT}/paired_comparisons.csv"
DECODER = f"{REPORT}/decoder_diagnostic.csv"
Z_METRICS = f"{REPORT}/z_metrics.csv"
CONFIG = f"{REPORT}/experiment_config.json"
SOURCES = [MAIN, PAIRED, DECODER, Z_METRICS, CONFIG]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]
Rows = list[dict[str, str]]

DENSE_Y = "Dense contact error E_C (L2 over 512 points, mean of frames 1-63)"
TRAIN_SPLIT = "train (160 sequences)"


def read_rows(path: Path) -> Rows:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(text: str | None) -> float | None:
    """A table cell as a float; an empty cell is a missing value."""
    return float(text) if text not in (None, "") else None


def pick(rows: Rows, **where: str) -> dict[str, str]:
    """The single row matching every column filter."""
    hits = [row for row in rows if all(row[key] == value for key, value in where.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {where}, found {len(hits)}")
    return hits[0]


def series(name: str, rows: list[dict[str, str]], value: str, lo: str | None = None, hi: str | None = None) -> dict:
    """One bar series from a list of table rows (one row per x category)."""
    out: dict = {"name": name, "values": [number(row[value]) for row in rows]}
    if lo and hi:
        out["lo"] = [number(row[lo]) for row in rows]
        out["hi"] = [number(row[hi]) for row in rows]
    return out


def sample_sizes(rows: Rows, column: str, **where: str) -> str:
    """'364 (TACO) and 186 (ARCTIC)' read from the table, for the figure notes."""
    return " and ".join(f"{pick(rows, dataset=key, **where)[column]} ({title})" for key, title in DATASETS)


def dense_error_models(main: Rows) -> dict:
    models = [("M0", "M0 direct"), ("M1", "M1 z, old decoder"), ("M2", "M2 z, new decoder")]
    panels = []
    for key, title in DATASETS:
        rows = [pick(main, dataset=key, model=model, metric="E_C") for model, _ in models]
        panels.append({
            "title": title,
            "x": {"label": "Generator", "categories": [label for _, label in models]},
            "y": {"label": DENSE_Y, "direction": "lower_better"},
            "series": [series("test dense error", rows, "value", "ci_lo", "ci_hi")],
        })
    n = sample_sizes(main, "n_examples", model="M2", metric="E_C")
    return {
        "id": "f3suba_dense_error_models",
        "type": "bar",
        "title": "Dense contact error of the direct model and the two z-mediated models",
        "panels": panels,
        "note": (
            f"Test dense error E_C, lower is better; mean over {n} test sequences, one training seed. "
            "Error bars are 95 % take-cluster bootstrap intervals of each model's own mean (unpaired); "
            "the paired differences on identical sequences are in f3suba_paired_dense_differences."
        ),
        "sources": [{"path": MAIN, "locator": "rows metric == E_C, model in {M0, M1, M2}; columns value, ci_lo, ci_hi, n_examples"}],
    }


def rule_threshold(config: dict) -> float:
    """Minimum relative size of a paired difference under the study's decision rule (a fraction)."""
    return float(config["decision_rule"]["rel_min"])


def paired_dense_differences(paired: Rows, config: dict) -> dict:
    pairs = [("M2", "M1", "M2 - M1"), ("M2", "M0", "M2 - M0"), ("M1", "M0", "M1 - M0")]
    percent = f"{rule_threshold(config) * 100:g} %"
    panels = []
    verdicts = []                                   # the rule's call for every bar, read from the verdict column
    for key, title in DATASETS:
        rows = [pick(paired, dataset=key, a=a, b=b, metric="E_C") for a, b, _ in pairs]
        panels.append({
            "title": title,
            "x": {"label": "Model pair (first minus second)", "categories": [label for _, _, label in pairs]},
            "y": {"label": "Paired difference in dense error E_C", "direction": "none"},
            "series": [series("paired difference", rows, "diff", "ci_lo", "ci_hi")],
        })
        verdicts.append(f"{title}: " + ", ".join(f"{label} {row['verdict']}" for (_, _, label), row in zip(pairs, rows)))
    n = sample_sizes(paired, "n", a="M2", b="M1", metric="E_C")
    return {
        "id": "f3suba_paired_dense_differences",
        "type": "bar",
        "title": "Paired differences in dense error between the models",
        "panels": panels,
        "note": (
            f"Mean per-sequence difference of the test dense error E_C on identical sequences (n = {n}); "
            "a negative value means the first-named model has the lower error. Error bars are 95 % paired "
            "take-cluster bootstrap intervals; the study's rule (fixed in advance, according to the report) calls a difference better or worse only "
            f"if it is at least {percent} of the second model's error and its interval excludes 0. "
            f"Verdicts of that rule for the first-named model: {'; '.join(verdicts)}."
        ),
        "sources": [
            {"path": PAIRED, "locator": "rows metric == E_C, (a, b) in {(M2, M1), (M2, M0), (M1, M0)}; columns diff, ci_lo, ci_hi, n, verdict"},
            {"path": CONFIG, "locator": "decision_rule.rel_min (the threshold quoted in the note)"},
        ],
    }


def structure_relative(paired: Rows, config: dict) -> dict:
    metrics = [("part_hamming", "participation"), ("amount_l1", "amount"), ("centroid", "centroid"),
               ("normal", "normal"), ("q_rel_l1", "wrench")]
    pairs = [("M2", "M0", "M2 vs M0"), ("M2", "M1", "M2 vs M1"), ("M1", "M0", "M1 vs M0")]
    threshold = rule_threshold(config)
    panels = []
    unresolved = []                                 # bars beyond a threshold line that the rule still calls "similar"
    for key, title in DATASETS:
        rows_by_pair = [(label, [pick(paired, dataset=key, a=a, b=b, metric=metric) for metric, _ in metrics]) for a, b, label in pairs]
        panels.append({
            "title": title,
            "x": {"label": "Structural / wrench error metric", "categories": [label for _, label in metrics]},
            "y": {"label": "Relative improvement of the first model (fraction of the second model's error)",
                  "direction": "higher_better"},
            "series": [series(label, rows, "rel_improvement") for label, rows in rows_by_pair],
            "refs": [{"axis": "y", "value": threshold, "label": "rule threshold (+)"},
                     {"axis": "y", "value": -threshold, "label": "rule threshold (-)"}],
        })
        names = [f"{label} {name}" for label, rows in rows_by_pair for (_, name), row in zip(metrics, rows)
                 if abs(number(row["rel_improvement"])) >= threshold and row["verdict"] == "similar"]
        unresolved.append(f"{title}: {', '.join(names) if names else 'none'}")
    return {
        "id": "f3suba_structure_relative",
        "type": "bar",
        "title": "Structural and wrench errors: relative change between the models",
        "panels": panels,
        "note": (
            "Paired relative improvement on the test sequences for the five error metrics of the decision rule "
            "(participation Hamming, contact amount L1, centroid / object size, normal angle, wrench relative L1); "
            "positive means the first-named model has the lower error. No interval is given for the relative value: "
            "a bar counts as better or worse only beyond the threshold lines AND when the bootstrap interval of the "
            "paired difference excludes 0 (column verdict). Bars beyond a threshold line that the rule nevertheless "
            f"calls similar, because that interval includes 0: {'; '.join(unresolved)}."
        ),
        "sources": [
            {"path": PAIRED, "locator": "rows metric in {part_hamming, amount_l1, centroid, normal, q_rel_l1}, (a, b) in {(M2, M0), (M2, M1), (M1, M0)}; column rel_improvement (verdict column holds the rule's call)"},
            {"path": CONFIG, "locator": "decision_rule.rel_min (threshold lines at +/- this value)"},
        ],
    }


def decoder_panels(decoder: Rows, categories: list[str], inputs_m1: list[tuple[str, str]], inputs_m2: list[tuple[str, str]],
                   x_label: str, intervals: bool) -> list[dict]:
    """Panels with one series per decoder; each input is an (input, split) pair of decoder_diagnostic.csv."""
    bounds = ("ci_lo", "ci_hi") if intervals else (None, None)
    panels = []
    for key, title in DATASETS:
        rows_m1 = [pick(decoder, dataset=key, decoder="M1", input=name, split=split) for name, split in inputs_m1]
        rows_m2 = [pick(decoder, dataset=key, decoder="M2", input=name, split=split) for name, split in inputs_m2]
        panels.append({
            "title": title,
            "x": {"label": x_label, "categories": categories},
            "y": {"label": DENSE_Y, "direction": "lower_better"},
            "series": [series("old decoder (4 blocks)", rows_m1, "E_C", *bounds),
                       series("new decoder (6 blocks)", rows_m2, "E_C", *bounds)],
        })
    return panels


def decoder_same_input(decoder: Rows) -> dict:
    panels = decoder_panels(
        decoder,
        categories=["teacher z*", "z predicted by the earlier model", "z predicted by the adapted model"],
        inputs_m1=[("oracle", "test"), ("pred", "test"), ("zhat_M2", "test")],
        inputs_m2=[("oracle", "test"), ("zhat_M1", "test"), ("pred", "test")],
        x_label="Latent fed to the decoder (the same latent for both decoders)",
        intervals=True,
    )
    return {
        "id": "f3suba_decoder_same_input",
        "type": "bar",
        "title": "The same latent through the old and the new decoder",
        "panels": panels,
        "note": (
            "Test dense error E_C (lower is better) when the trained decoder of the earlier z-mediated model (M1 in the "
            "source table, 4 blocks) and of the adapted model (M2, 6 blocks) decode the same latent: the teacher z* of "
            "the true frames, or the z predicted by the earlier or by the adapted model. "
            "Error bars are 95 % take-cluster bootstrap intervals of each mean; the teacher z* is a diagnostic "
            "input and is never available at generation time."
        ),
        "sources": [{"path": DECODER, "locator": "rows split == test; decoder M1: input in {oracle, pred, zhat_M2}; decoder M2: input in {oracle, zhat_M1, pred}; columns E_C, ci_lo, ci_hi"}],
    }


def decoder_perturbed_teacher(decoder: Rows) -> dict:
    inputs = [("oracle", "test"), ("noisy_gauss", "test"), ("noisy_perm", "test"), ("pred", "test")]
    panels = decoder_panels(
        decoder,
        categories=["teacher z*", "z* + matched Gaussian noise", "z* + another sequence's error", "predicted z (own model)"],
        inputs_m1=inputs,
        inputs_m2=inputs,
        x_label="Latent fed to the decoder",
        intervals=True,
    )
    return {
        "id": "f3suba_decoder_perturbed_teacher",
        "type": "bar",
        "title": "Decoder sensitivity: clean, perturbed and predicted latents",
        "panels": panels,
        "note": (
            "Test dense error E_C (lower is better) of each trained decoder fed the teacher z*, the teacher z* plus a "
            "perturbation of the size of its own model's test latent error (Gaussian with the error's mean and covariance "
            "per horizon bin, or the error trajectory of another test sequence), and its own model's predicted z. "
            "Error bars are 95 % take-cluster bootstrap intervals; the perturbations of the two decoders are matched to "
            "different models, so the two series do not receive identical perturbed latents."
        ),
        "sources": [{"path": DECODER, "locator": "rows split == test, decoder in {M1, M2}, input in {oracle, noisy_gauss, noisy_perm, pred}; columns E_C, ci_lo, ci_hi"}],
    }


def z_rmse_splits(z_metrics: Rows) -> dict:
    splits = [("rmse_train", "train"), ("rmse_val", "validation"), ("rmse_test", "test")]
    models = [("M1", "earlier model (old decoder)"), ("M2", "adapted model (new decoder)")]
    panels = []
    for key, title in DATASETS:
        rows = {model: pick(z_metrics, dataset=key, model=model) for model, _ in models}
        panels.append({
            "title": title,
            "x": {"label": "Sequences the trained model is applied to", "categories": [label for _, label in splits]},
            "y": {"label": "RMSE of predicted z against the teacher z* (standardised units)", "direction": "lower_better"},
            "series": [{"name": label, "values": [number(rows[model][column]) for column, _ in splits]} for model, label in models],
            "refs": [{"axis": "y", "value": number(rows["M2"]["rmse_train_mean"]), "label": "constant train-mean z, on test"}],
        })
    return {
        "id": "f3subb_z_rmse_splits",
        "type": "bar",
        "title": "Latent prediction error on training, validation and test sequences",
        "panels": panels,
        "note": (
            "Root mean squared error between the predicted latent and the teacher latent z* of the true frames, pooled over "
            "sequences, the 63 future frames and the 64 latent dimensions, in coordinates standardised with training-set "
            "statistics; lower is better, same weights (selected state) on every split. No interval is given in the source "
            "for these values; the reference line is the error of predicting the training-mean z on the test sequences."
        ),
        "sources": [{"path": Z_METRICS, "locator": "rows model in {M1, M2}; columns rmse_train, rmse_val, rmse_test; reference line: column rmse_train_mean"}],
    }


def dense_train_vs_test(decoder: Rows) -> dict:
    inputs = [("pred", TRAIN_SPLIT), ("oracle", TRAIN_SPLIT), ("pred", "test"), ("oracle", "test")]
    panels = decoder_panels(
        decoder,
        categories=["train: predicted z", "train: teacher z*", "test: predicted z", "test: teacher z*"],
        inputs_m1=inputs,
        inputs_m2=inputs,
        x_label="Sequences and latent fed to the decoder",
        intervals=True,
    )
    return {
        "id": "f3subb_dense_train_vs_test",
        "type": "bar",
        "title": "Decoded dense error on training and on test sequences, by decoder input",
        "panels": panels,
        "note": (
            "Dense error E_C (lower is better) of each trained decoder fed its own model's predicted z or the teacher z*, "
            "on a fixed random subset of 160 training sequences and on the test sequences. Error bars (test rows only) are "
            "95 % take-cluster bootstrap intervals; the source gives no interval for the training subset."
        ),
        "sources": [{"path": DECODER, "locator": f"rows decoder in {{M1, M2}}, input in {{pred, oracle}}, split in {{{TRAIN_SPLIT}, test}}; columns E_C, ci_lo, ci_hi (empty for the train rows)"}],
    }


def build(src: Callable[[str], Path]) -> list[dict]:
    main = read_rows(src(MAIN))
    paired = read_rows(src(PAIRED))
    decoder = read_rows(src(DECODER))
    z_metrics = read_rows(src(Z_METRICS))
    config = json.loads(src(CONFIG).read_text(encoding="utf-8"))
    return [
        dense_error_models(main),
        paired_dense_differences(paired, config),
        structure_relative(paired, config),
        decoder_same_input(decoder),
        decoder_perturbed_teacher(decoder),
        z_rmse_splits(z_metrics),
        dense_train_vs_test(decoder),
    ]
