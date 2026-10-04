#!/usr/bin/env python
"""Build the project page's figure data from the experiment result tables.

Every quantitative figure on the page is produced by one module in ``figs/``. A module
declares the result files it reads and returns chart data read straight from them:

    SOURCES = ["reports/<report>/<table>.csv", ...]      # relative to the result root

    def build(src):                                       # src(rel_path) -> pathlib.Path
        ...
        return [figure, ...]

A figure is a plain dict (see ``validate_figure`` for the exact contract):

    {"id": "f1a_variation", "type": "bar" | "line" | "scatter", "title": "...",
     "panels": [{"title": "TACO",
                 "x": {"label": "...", "categories": [...]} | {"label": "...", "values": [...]},
                 "y": {"label": "...", "direction": "lower_better" | "higher_better" | "none"},
                 "series": [{"name": "...", "values": [...], "lo": [...], "hi": [...]}],
                 "points": [{"name": "...", "xy": [[x, y], ...]}],          # scatter only
                 "refs": [{"axis": "y", "value": 1.0, "label": "..."}],     # optional
                 "highlight": {"category": "R2", "label": "..."}}],         # optional
     "note": "...", "sources": [{"path": "...", "locator": "..."}]}

One panel per dataset: values of different datasets are never pooled or averaged.

The result root is ``$DEXCORE_RESULT_ROOT`` (default ``/result/uhnam/dexcore``). When it is
present, each declared source no larger than 3 MB is copied to ``docs/data/sources/`` so the
figures can be rebuilt from the repository alone; when it is absent, the snapshots are read.
A module that uses only some rows of a larger CSV names them (``src(path, keep_rows=(column,
value))``) and the snapshot holds the header and those rows.

    python docs/build/build_figures.py                    # rebuild docs/data/figures.{json,js}
    python docs/build/build_figures.py --only f1b_hier    # print one module's figures, write nothing

``figures.json`` holds every figure the modules return. ``figures.js`` is what the page loads: only
the figures that docs/index.html references (``data-fig="<id>"``), so the page stays small.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

BUILD_DIR = Path(__file__).resolve().parent
DOCS_DIR = BUILD_DIR.parent
FIGS_DIR = BUILD_DIR / "figs"
SNAPSHOT_DIR = DOCS_DIR / "data" / "sources"
OUT_JSON = DOCS_DIR / "data" / "figures.json"
OUT_JS = DOCS_DIR / "data" / "figures.js"
PAGE = DOCS_DIR / "index.html"
RESULT_ROOT = Path(os.environ.get("DEXCORE_RESULT_ROOT", "/result/uhnam/dexcore"))
MAX_SNAPSHOT_BYTES = 3_000_000

FIGURE_TYPES = {"bar", "line", "scatter"}
DIRECTIONS = {"lower_better", "higher_better", "none"}
DATASET_PANELS = {"TACO", "ARCTIC", "OakInk2"}


class Sources:
    """Resolve a declared source path to a readable file (live result tree, else snapshot)."""

    def __init__(self, module_name: str, declared: list[str], snapshot: bool):
        self.module_name = module_name
        self.declared = set(declared)
        self.snapshot = snapshot
        self.used: set[str] = set()

    def __call__(self, rel_path: str, keep_rows: tuple[str, str] | None = None) -> Path:
        """keep_rows = (column, value): the only rows the caller uses of a CSV above the snapshot limit."""
        if rel_path not in self.declared:
            raise KeyError(f"{self.module_name}: '{rel_path}' is read but not declared in SOURCES")
        self.used.add(rel_path)
        live = RESULT_ROOT / rel_path
        snap = SNAPSHOT_DIR / rel_path
        if live.is_file():
            if self.snapshot and live.stat().st_size <= MAX_SNAPSHOT_BYTES:
                snap.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(live, snap)
            elif self.snapshot and keep_rows is not None:
                _snapshot_rows(live, snap, *keep_rows)
            return live
        if snap.is_file():
            return snap
        raise FileNotFoundError(f"{self.module_name}: {rel_path} is in neither {RESULT_ROOT} nor {SNAPSHOT_DIR}")


def _snapshot_rows(live: Path, snap: Path, column: str, value: str) -> None:
    """Write the header of a CSV and its rows with row[column] == value, fields unchanged."""
    snap.parent.mkdir(parents=True, exist_ok=True)
    with live.open(newline="", encoding="utf-8") as src_file, snap.open("w", newline="", encoding="utf-8") as out_file:
        reader = csv.reader(src_file)
        writer = csv.writer(out_file, lineterminator="\n")
        header = next(reader)
        index = header.index(column)
        writer.writerow(header)
        writer.writerows(row for row in reader if row[index] == value)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_values(where: str, values, length: int | None = None) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{where}: expected a list")
    if length is not None and len(values) != length:
        raise ValueError(f"{where}: {len(values)} values for {length} x positions")
    for value in values:
        if value is not None and not _is_number(value):
            raise ValueError(f"{where}: non-finite or non-numeric value {value!r}")


def validate_figure(fig: dict, declared: set[str]) -> None:
    """Raise ValueError when a figure breaks the contract the page's renderer relies on."""
    for key in ("id", "type", "title", "panels", "note", "sources"):
        if key not in fig:
            raise ValueError(f"figure {fig.get('id', '?')}: missing '{key}'")
    fid = fig["id"]
    if fig["type"] not in FIGURE_TYPES:
        raise ValueError(f"{fid}: unknown type {fig['type']!r}")
    if not fig["sources"]:
        raise ValueError(f"{fid}: a figure must name its source files")
    for source in fig["sources"]:
        if source.get("path") not in declared:
            raise ValueError(f"{fid}: source {source.get('path')!r} is not declared in SOURCES")
        if not source.get("locator"):
            raise ValueError(f"{fid}: source {source['path']} has no locator (rows / columns used)")
    if not str(fig["note"]).strip():
        raise ValueError(f"{fid}: empty note")
    if not fig["panels"]:
        raise ValueError(f"{fid}: no panels")
    titles = [panel.get("title") for panel in fig["panels"]]
    if len(set(titles)) != len(titles) or not set(titles) <= DATASET_PANELS:
        raise ValueError(f"{fid}: panels must be one per dataset out of {sorted(DATASET_PANELS)}, got {titles}")
    for index, panel in enumerate(fig["panels"]):
        where = f"{fid} panel {index}"
        for key in ("title", "x", "y"):
            if key not in panel:
                raise ValueError(f"{where}: missing '{key}'")
        if panel["y"].get("direction") not in DIRECTIONS:
            raise ValueError(f"{where}: y.direction must be one of {sorted(DIRECTIONS)}")
        if fig["type"] == "scatter":
            groups = panel.get("points") or []
            if not groups:
                raise ValueError(f"{where}: scatter panel without points")
            for group in groups:
                for xy in group["xy"]:
                    if len(xy) != 2 or not all(_is_number(v) for v in xy):
                        raise ValueError(f"{where}: bad point {xy!r} in group {group.get('name')!r}")
            continue
        x = panel["x"]
        positions = x.get("categories", x.get("values"))
        if not positions:
            raise ValueError(f"{where}: x needs 'categories' or 'values'")
        if not panel.get("series"):
            raise ValueError(f"{where}: no series")
        for series in panel["series"]:
            name = f"{where} series {series.get('name')!r}"
            _check_values(name, series.get("values"), len(positions))
            for bound in ("lo", "hi"):
                if bound in series and series[bound] is not None:
                    _check_values(f"{name} {bound}", series[bound], len(positions))
        highlight = panel.get("highlight")
        if highlight and highlight.get("category") not in positions:
            raise ValueError(f"{where}: highlight {highlight.get('category')!r} is not an x position")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"figs_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_module(path: Path, snapshot: bool) -> list[dict]:
    module = load_module(path)
    declared = list(getattr(module, "SOURCES", []))
    src = Sources(path.stem, declared, snapshot)
    figures = module.build(src)
    for fig in figures:
        validate_figure(fig, set(declared))
        fig["module"] = f"docs/build/figs/{path.name}"
    unused = set(declared) - src.used
    if unused:
        raise ValueError(f"{path.name}: declared but never read: {sorted(unused)}")
    return figures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", help="build one module (file stem) and print its figures; writes nothing")
    args = parser.parse_args()

    if args.only:
        figures = build_module(FIGS_DIR / f"{args.only}.py", snapshot=False)
        json.dump(figures, sys.stdout, indent=1, ensure_ascii=False)
        print()
        return 0

    figures: dict[str, dict] = {}
    for path in sorted(FIGS_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        for fig in build_module(path, snapshot=True):
            if fig["id"] in figures:
                raise ValueError(f"duplicate figure id {fig['id']} ({path.name})")
            figures[fig["id"]] = fig
        print(f"{path.name}: ok", file=sys.stderr)

    used = re.findall(r'data-fig="([^"]+)"', PAGE.read_text(encoding="utf-8")) if PAGE.is_file() else list(figures)
    absent = sorted(set(used) - set(figures))
    if absent:
        raise ValueError(f"{PAGE.name} references figures that no module returns: {absent}; nothing was written")
    payload = {"result_root": str(RESULT_ROOT), "figures": figures}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1, ensure_ascii=False)
    OUT_JSON.write_text(text + "\n", encoding="utf-8")
    page_payload = {"figures": {fid: figures[fid] for fid in dict.fromkeys(used)}}
    OUT_JS.write_text("window.DEXCORE_FIGURES = " + json.dumps(page_payload, ensure_ascii=False) + ";\n", encoding="utf-8")
    print(f"wrote {len(figures)} figures to {OUT_JSON.relative_to(DOCS_DIR.parent)}; "
          f"{len(page_payload['figures'])} of them to {OUT_JS.name} for the page", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
