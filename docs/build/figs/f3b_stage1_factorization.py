"""Figure data for page block F3-B: the learned z / r factorisation of the dense contact map (Stage 1).

Source report: reports/contact_factorization_stage1/report.md (Sections 7, 8, 10, 11 and the decision table).
One model family per dataset (TACO and ARCTIC trained and evaluated separately, one training seed):
A0 = plain autoencoder (one 128-D latent h), A2 = z / r split without the relational loss, A3 = z / r split with the
relational R2 + wrench loss on z (the proposed model; |z| = |r| = 64).

Figures (every plotted value is read from the report's root tables at run time):
  f3b_structure_retention              probe-recoverable structure information in the dense map, in z and in r
  f3b_structure_retention_by_quantity  the same retention, split into the five probed quantities
  f3b_relational_loss_ablation         retention of z and r with (A3) and without (A2) the relational loss
  f3b_reconstruction_error             test reconstruction error: plain autoencoder, linear PCA-128, z only, z + r
  f3b_swap_follows_donor               latent swap: share of swaps whose structure follows z / whose detail follows r
  f3b_code_autocorrelation             limit of the result: temporal autocorrelation of r against the residual it encodes
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Optional

REPORT = "reports/contact_factorization_stage1"
PROBE = f"{REPORT}/probe_metrics.csv"
RECON = f"{REPORT}/reconstruction_metrics.csv"
SWAP = f"{REPORT}/swap_metrics.csv"
TEMPORAL = f"{REPORT}/temporal_persistence.csv"
CONFIG = f"{REPORT}/experiment_config.json"
SOURCES = [PROBE, RECON, SWAP, TEMPORAL, CONFIG]

DATASETS = [("taco", "TACO"), ("arctic", "ARCTIC")]
Row = dict[str, str]


def read_rows(path: Path) -> list[Row]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def one(rows: list[Row], **filters: str) -> Row:
    """The single row matching every column = value filter (a missing or duplicated row is an error)."""
    hits = [row for row in rows if all(row[col] == value for col, value in filters.items())]
    if len(hits) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(hits)}")
    return hits[0]


def num(row: Row, column: str) -> Optional[float]:
    """Unrounded float of a cell; an empty cell is a missing value."""
    return float(row[column]) if row[column] != "" else None


def structure_retention(probe: list[Row]) -> dict:
    inputs = [("dense map C (512-D)", "C"), ("z (64-D)", "A3:z"), ("r (64-D)", "A3:r")]
    panels = []
    for ds, title in DATASETS:
        cells = [one(probe, dataset=ds, input=key, probe="mlp") for _, key in inputs]
        panels.append({
            "title": title,
            "x": {"label": "what the probe reads", "categories": [label for label, _ in inputs]},
            "y": {"label": "structure retention (1 = dense-map probe, 0 = trivial predictor)", "direction": "none"},
            "series": [{"name": "MLP probe", "values": [num(row, "structure_retention") for row in cells]}],
            "refs": [{"axis": "y", "value": num(cells[0], "structure_retention"), "label": "probe on the full dense map"}],
        })
    return {
        "id": "f3b_structure_retention", "type": "bar",
        "title": "Structure information recoverable by a probe: dense map, z and r",
        "panels": panels,
        "note": ("Structure retention of the proposed model A3 on the unique test frames (TACO 22,938; ARCTIC 11,560): a small MLP "
                 "probe predicts the R2 structure descriptor and the wrench profile from a frozen code, and (probe - trivial) / "
                 "(dense-map probe - trivial) is averaged over five quantities, so 1 = as good as probing the full map, 0 = as "
                 "good as predicting the training mean, below 0 = worse than that. Higher = more structure information "
                 "recoverable; no error bars (the table gives none), one training seed. z was trained with a relational loss "
                 "built from the same R2 + wrench descriptors that the probes predict (scored here on held-out test takes), "
                 "and the ratio is relative to a small probe on the 512-D map, not a measure of information content."),
        "sources": [{"path": PROBE, "locator": "rows probe = mlp, input in {C, A3:z, A3:r}; column structure_retention; one panel per dataset"}],
    }


def structure_retention_by_quantity(probe: list[Row]) -> dict:
    quantities = [("participation (AUPRC)", "ret_part_auprc"), ("amount (R²)", "ret_amount_r2"), ("centroid (error)", "ret_centroid"),
                  ("normal (angle)", "ret_normal"), ("wrench (explained variance)", "ret_wrench_ev")]
    codes = [("z", "A3:z"), ("r", "A3:r")]
    panels = []
    for ds, title in DATASETS:
        dense = one(probe, dataset=ds, input="C", probe="mlp")
        series = []
        for name, key in codes:
            row = one(probe, dataset=ds, input=key, probe="mlp")
            series.append({"name": name, "values": [num(row, column) for _, column in quantities]})
        panels.append({
            "title": title,
            "x": {"label": "probed quantity", "categories": [label for label, _ in quantities]},
            "y": {"label": "retention (1 = dense-map probe, 0 = trivial predictor)", "direction": "none"},
            "series": series,
            "refs": [{"axis": "y", "value": num(dense, "structure_retention"), "label": "probe on the full dense map"}],
        })
    return {
        "id": "f3b_structure_retention_by_quantity", "type": "bar",
        "title": "The same probe retention, per structural quantity",
        "panels": panels,
        "note": ("Per-quantity retention of the MLP probes on A3's z and r (unique test frames; TACO 22,938, ARCTIC 11,560): "
                 "(probe - trivial) / (dense-map probe - trivial) for part participation, contact amount, centroid, normal and "
                 "the wrench profile; the single retention number is the mean of these five. Higher = more recoverable, "
                 "negative = worse than the training-mean predictor; no error bars, one training seed."),
        "sources": [{"path": PROBE, "locator": "rows probe = mlp, input in {A3:z, A3:r}; columns ret_part_auprc, ret_amount_r2, ret_centroid, ret_normal, ret_wrench_ev; reference line = structure_retention of input C"}],
    }


def relational_loss_ablation(probe: list[Row]) -> dict:
    models = [("A3: with relational loss", "A3"), ("A2: without", "A2")]
    codes = [("z", "z"), ("r", "r")]
    panels = []
    for ds, title in DATASETS:
        dense = one(probe, dataset=ds, input="C", probe="mlp")
        series = []
        for name, model in models:
            values = [num(one(probe, dataset=ds, input=f"{model}:{code}", probe="mlp"), "structure_retention") for _, code in codes]
            series.append({"name": name, "values": values})
        panels.append({
            "title": title,
            "x": {"label": "code the probe reads", "categories": [label for label, _ in codes]},
            "y": {"label": "structure retention (1 = dense-map probe, 0 = trivial predictor)", "direction": "none"},
            "series": series,
            "refs": [{"axis": "y", "value": num(dense, "structure_retention"), "label": "probe on the full dense map"}],
        })
    return {
        "id": "f3b_relational_loss_ablation", "type": "bar",
        "title": "What the relational loss changes: structure retention of z and r with and without it",
        "panels": panels,
        "note": ("Structure retention (MLP probe, unique test frames, mean of five quantities) of the two codes of the z / r "
                 "model trained with the relational R2 + wrench loss on z (A3) and without it (A2); same architecture and "
                 "latent sizes (64 / 64). Higher = more structure information recoverable from that code; no error bars, "
                 "one training seed per model."),
        "sources": [{"path": PROBE, "locator": "rows probe = mlp, input in {A3:z, A3:r, A2:z, A2:r}; column structure_retention; reference line = input C"}],
    }


def reconstruction_error(recon: list[Row]) -> dict:
    bars = [("plain autoencoder (128-D)", "A0", "E_full"), ("linear PCA-128", "pca128", "E_full"),
            ("z only (64-D)", "A3", "E_zonly"), ("z + r (64 + 64-D)", "A3", "E_full")]
    panels = []
    for ds, title in DATASETS:
        cells = [(one(recon, dataset=ds, model=model), column) for _, model, column in bars]
        panels.append({
            "title": title,
            "x": {"label": "reconstruction", "categories": [label for label, _, _ in bars]},
            "y": {"label": "reconstruction error (mean per-frame L2 over 512 points, raw contact units)", "direction": "lower_better"},
            "series": [{"name": "test error",
                        "values": [num(row, column) for row, column in cells],
                        "lo": [num(row, f"{column}_lo") for row, column in cells],
                        "hi": [num(row, f"{column}_hi") for row, column in cells]}],
            "highlight": {"category": "z + r (64 + 64-D)", "label": "proposed model, full reconstruction"},
        })
    return {
        "id": "f3b_reconstruction_error", "type": "bar",
        "title": "Dense reconstruction error: z alone versus z + r",
        "panels": panels,
        "note": ("Mean L2 distance between the reconstructed and the true 512-point contact map of a frame (raw contact units, "
                 "lower is better) on the unique test frames (TACO 22,938 frames of 186 takes; ARCTIC 11,560 of 44 takes); "
                 "error bars are 95 % take-cluster bootstrap intervals (1000 resamples), i.e. test-set uncertainty with one "
                 "training seed. The plain autoencoder (A0 in the source table) is a weak control (its latent collapses), so linear "
                 "PCA-128 of the training frames, with the same total latent size, is the stricter reference. The plain "
                 "autoencoder, PCA-128 and z + r use 128 latent dimensions in total; the z-only bar uses the 64-D z alone "
                 "(z only and z + r are the proposed model, A3 in the source table)."),
        "sources": [{"path": RECON, "locator": "rows model in {A0, pca128, A3}; columns E_full (A0, pca128, A3) and E_zonly (A3) with their _lo / _hi; one panel per dataset"}],
    }


def swap_follows_donor(swap: list[Row], thresholds: dict) -> dict:
    bars = [("structure follows the z donor", "frac_struct_follows_z"), ("dense detail follows the r donor", "frac_detail_follows_r")]
    panels = []
    for ds, title in DATASETS:
        row = one(swap, dataset=ds, model="A3")
        panels.append({
            "title": title,
            "x": {"label": "latent swap: decode z of frame A with r of frame B", "categories": [label for label, _ in bars]},
            "y": {"label": "share of swaps", "direction": "higher_better"},
            "series": [{"name": "proposed model",
                        "values": [num(row, column) for _, column in bars],
                        "lo": [num(row, f"{column}_lo") for _, column in bars],
                        "hi": [num(row, f"{column}_hi") for _, column in bars]}],
            "refs": [{"axis": "y", "value": float(thresholds["swap_struct_frac_min"]), "label": "threshold fixed in advance, structure"},
                     {"axis": "y", "value": float(thresholds["swap_detail_frac_min"]), "label": "threshold fixed in advance, detail"}],
        })
    return {
        "id": "f3b_swap_follows_donor", "type": "bar",
        "title": "Latent swap test: what follows z and what follows r",
        "panels": panels,
        "note": ("Latent swap on pairs of test frames of the same object from different takes with similar structure but "
                 "different dense maps (TACO 1600 swaps from 800 pairs over 40 meshes; ARCTIC 1450 from 725 pairs over 11 "
                 "meshes): share of swaps whose decoded map is structurally (R2 + wrench distance) closer to the frame that gave "
                 "z, and share whose added correction is more similar (cosine) to the r donor's own residual than to the "
                 "host's. Higher = cleaner role separation; error bars are 95 % take-cluster bootstrap intervals (1000 "
                 "resamples), one training seed. The swapped map is, by design, the z donor's z-only map plus a residual "
                 "correction decoded with the other frame's r, so a high structure share is the expected outcome of the "
                 "architecture; the detail share is the test of whether the content of r transfers between frames."),
        "sources": [{"path": SWAP, "locator": "rows model = A3; columns frac_struct_follows_z, frac_detail_follows_r with their _lo / _hi; one panel per dataset"},
                    {"path": CONFIG, "locator": "decision_thresholds.swap_struct_frac_min and decision_thresholds.swap_detail_frac_min (reference lines)"}],
    }


def code_autocorrelation(temporal: list[Row]) -> dict:
    codes = [("r", "A3:r"), ("dense residual left by z", "A3:resid_z"), ("z", "A3:z"), ("plain autoencoder latent h", "A0:h")]
    panels = []
    for ds, title in DATASETS:
        rows = [row for row in temporal if row["dataset"] == ds and row["metric"] == "autocorrelation"]
        lags = sorted({int(row["h"]) for row in rows if row["code"] == codes[0][1]})
        series = [{"name": name, "values": [num(one(rows, code=key, h=str(lag)), "value") for lag in lags]} for name, key in codes]
        panels.append({
            "title": title,
            "x": {"label": "lag h (frames, 30 Hz)", "values": lags},
            "y": {"label": "autocorrelation along real test trajectories", "direction": "none"},
            "series": series,
        })
    return {
        "id": "f3b_code_autocorrelation", "type": "line",
        "title": "Limit: r is only as persistent as the dense residual it encodes",
        "panels": panels,
        "note": ("Autocorrelation at lag h of each code when the frozen encoders are applied frame by frame to the real test "
                 "trajectories (TACO 364, ARCTIC 186 sequences of 64 frames; nothing temporal was trained); the dense residual "
                 "is the part of the contact map that z alone does not reconstruct. Higher = the code changes more slowly; "
                 "the decision rule reads h = 8; the table gives no interval for this statistic, so there are no error "
                 "bars; one training seed. At the last lag (63) each 64-frame sequence contributes a single frame pair."),
        "sources": [{"path": TEMPORAL, "locator": "rows metric = autocorrelation, code in {A3:r, A3:resid_z, A3:z, A0:h}, every lag h; column value; one panel per dataset"}],
    }


def build(src: Callable[[str], Path]) -> list[dict]:
    probe = read_rows(src(PROBE))
    recon = read_rows(src(RECON))
    swap = read_rows(src(SWAP))
    temporal = read_rows(src(TEMPORAL))
    thresholds = json.loads(src(CONFIG).read_text(encoding="utf-8"))["decision_thresholds"]
    return [
        structure_retention(probe),
        structure_retention_by_quantity(probe),
        relational_loss_ablation(probe),
        reconstruction_error(recon),
        swap_follows_donor(swap, thresholds),
        code_autocorrelation(temporal),
    ]
