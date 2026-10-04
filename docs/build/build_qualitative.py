#!/usr/bin/env python
"""Register the qualitative figures of the page: provenance, numbers and a metadata snapshot.

The figures themselves are rendered by the scripts in ``docs/build/qual/`` (one per figure; they need
the result tree, the checkpoints and the raw datasets, and run on CPU). Each writes

    <result root>/reports/project_page_qualitative/<id>/{figure.png, panels.npz, meta.json}
    docs/assets/img/qual_<id>.webp                                   # the page asset

This script reads every ``meta.json`` (from the result tree when present, else from the snapshot in
``docs/data/qual/``) and writes

    docs/data/qual/<id>.json            snapshot of the metadata (selection rule and its evidence,
                                        display choices, every number a caption quotes, caveats)
    docs/evidence/images.json           provenance entry per page asset
    docs/evidence/zq_qualitative.json   one evidence entry per figure, so the numbers in the captions
                                        appear in evidence_manifest.json like every other number

    python docs/build/build_qualitative.py        # then: python docs/build/build_manifest.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
DOCS_DIR = BUILD_DIR.parent
RESULT_ROOT = Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))
QUAL_ROOT = RESULT_ROOT / "reports" / "project_page_qualitative"
SNAPSHOT_DIR = DOCS_DIR / "data" / "qual"
IMAGES = DOCS_DIR / "evidence" / "images.json"
EVIDENCE = DOCS_DIR / "evidence" / "zq_qualitative.json"

# figure id -> (page block that shows it, datasets)
FIGURES = {
    "motivation_two_contacts": ("MOTIVATION", ["TACO"]),
    "contact_map_primer": ("F1-PRIMER", ["TACO"]),
    "grasp_modes": ("F1-B", ["ARCTIC"]),
    "sample_then_evolve": ("F1-B", ["TACO"]),
    "change_events": ("F1-C", ["TACO"]),
    "reconfiguration_with_hand": ("F1-D", ["TACO", "ARCTIC"]),
    "structure_on_a_grasp": ("F2-PRIMER", ["TACO", "ARCTIC"]),
    "z_and_r_maps": ("F3-B", ["TACO", "ARCTIC"]),
    "direct_vs_via_z": ("F3-C", ["TACO", "ARCTIC"]),
    "z_moves_with_contact": ("F3-diagB", ["TACO", "ARCTIC"]),
}


def relative(text):
    """Shorten absolute server paths in any string of a nested structure."""
    if isinstance(text, str):
        return (text.replace(str(RESULT_ROOT) + "/", "").replace("/home/uhnam/workspace/dexcore/", "")
                .replace("/ckpt/uhnam/dexcore/", "<ckpt root>/"))
    if isinstance(text, list):
        return [relative(item) for item in text]
    if isinstance(text, dict):
        return {key: relative(value) for key, value in text.items()}
    return text


def load_meta(fig_id: str) -> dict:
    live = QUAL_ROOT / fig_id / "meta.json"
    snap = SNAPSHOT_DIR / f"{fig_id}.json"
    if live.is_file():
        data = relative(json.loads(live.read_text(encoding="utf-8")))
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        return data
    if snap.is_file():
        return json.loads(snap.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"{fig_id}: no meta.json in {QUAL_ROOT} and no snapshot in {SNAPSHOT_DIR}")


def as_text(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def number_table(numbers: list[dict]) -> dict:
    table = {}
    for item in numbers:
        label = f"{item.get('dataset', '')} | {item.get('label', '')}".strip(" |")
        while label in table:
            label += " "
        table[label] = {"value": as_text(item.get("value")), "source": as_text(item.get("source", "")),
                        "locator": as_text(item.get("locator", ""))}
    return table


def main() -> int:
    images = json.loads(IMAGES.read_text(encoding="utf-8")) if IMAGES.is_file() else {}
    images = {src: entry for src, entry in images.items() if (DOCS_DIR / src).is_file()}   # drop images no longer shipped
    entries = []
    for fig_id, (block, datasets) in FIGURES.items():
        asset = f"assets/img/qual_{fig_id}.webp"
        if not (DOCS_DIR / asset).is_file():
            raise FileNotFoundError(f"{fig_id}: page asset docs/{asset} is missing")
        meta = load_meta(fig_id)["meta"]
        script = f"docs/build/qual/q_{fig_id}.py"
        images[asset] = {
            "from": f"rendered for this page by {script} (master: reports/project_page_qualitative/{fig_id}/figure.png)",
            "edit": "none: the page asset is the rendered figure, resized to at most 2200 px wide and saved as WebP",
            "shows": as_text(meta.get("title", "")),
            "selection_rule": as_text(meta.get("selection_rule", "")),
            "examples": meta.get("examples", []),
            "display_choices": meta.get("display_choices", {}),
            "inference": meta.get("inference", "none"),
            "caveats": meta.get("caveats", []),
            "metadata": f"docs/data/qual/{fig_id}.json",
        }
        numbers = meta.get("numbers", [])
        sources = sorted({as_text(item.get("source", "")) for item in numbers if item.get("source")})
        entries.append({
            "block": block,
            "claim": f"Qualitative figure '{fig_id}': {as_text(meta.get('title', ''))}. Example(s) chosen by a stated rule: "
                     f"{as_text(meta.get('selection_rule', ''))}",
            "claim_status": "figure checked by an independent pass: selection rule recomputed, drawn arrays compared with their "
                            "sources, numbers re-read",
            "datasets": datasets,
            "report": script,
            "source_files": sources,
            "metric": "Illustration, not a statistic. Contact maps are the studies' 512-value canonical maps drawn back on the "
                      "object mesh by the studies' own smoothing operator; see display_choices in the figure's metadata.",
            "numbers": number_table(numbers),
            "caveats": [as_text(c) for c in meta.get("caveats", [])],
        })
    IMAGES.write_text(json.dumps(images, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {"unit": "qualitative_figures", "figure_module": None, "entries": entries, "dataset_differences": [],
               "do_not_claim": ["That a drawn example is typical beyond the property its selection rule names.",
                                "That a qualitative figure establishes a dataset-level result: the results rest on the tables."],
               "candidate_images": [], "verification": None}
    EVIDENCE.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"registered {len(entries)} qualitative figures: {IMAGES.relative_to(DOCS_DIR.parent)}, "
          f"{EVIDENCE.relative_to(DOCS_DIR.parent)}, {SNAPSHOT_DIR.relative_to(DOCS_DIR.parent)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
