#!/usr/bin/env python3
"""Build the data of the public page from the internal page, then check the result.

    python public_page/build/build_public.py

Steps
  1. Figures. For every entry of public_figures.py, copy the kept values of the internal figure
     (docs/data/figures.json) into a public figure with public labels, and write
     site/data/figures.js (window.DEXCORE_FIGURES = {figures: {...}}). Every copied value, error-bar
     bound and reference value is compared with the internal figure again after the file is written.
     Nothing internal is copied: no source list, no module, no internal note or id.
  2. Images. Byte copies of the three internal renders under their public names. A missing source
     image stops the build.
  3. Traceability. public_claims.json -> public_evidence_manifest.json (development only, never
     shipped). The build FAILS when
       a. a displayed number does not equal its full-precision value at the displayed rounding,
       b. a referenced internal evidence key or figure value does not exist or differs (for a
          displayed number and for a supporting value of a claim that shows no digit),
       c. a figure asset is missing,
       d. a number in the visible text of site/index.html does not lie inside the text of a claim
          that lists it and is not a listed structural number,
       e. a claim sentence or one of its listed tile strings is not on the page word for word, a
          displayed number is neither in its claim's text nor plotted in the chart the claim names, or
          a number typed into a chart label (a supporting value marked "where": "chart") is not written
          in the panel of its dataset or has no supporting value at all,
       f. a claim has no trace at all (no number, no supporting value and no evidence note)
     (d and e are skipped, with a message, while site/index.html does not exist).
  4. Checks. Runs check_public.py over site/ (writes exposure_checklist.md).

The build reads only files under docs/ and public_page/ (no result folder, no network) and writes
only site/data/figures.js, the three images, the manifest and the checklist. The two generated
documents carry no time stamp: building twice gives the same bytes. Nothing under docs/ is written.
Exit status is non-zero when any step fails.
"""
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

BUILD = Path(__file__).resolve().parent
PUBLIC = BUILD.parent
ROOT = PUBLIC.parent
DOCS = ROOT / "docs"
# The four locations below can be redirected on the command line (for tests in a scratch folder).
SITE = PUBLIC / "site"
CLAIMS = BUILD / "public_claims.json"
MANIFEST = PUBLIC / "public_evidence_manifest.json"
CHECKLIST = PUBLIC / "exposure_checklist.md"
MANUAL = BUILD / "manual_checks.json"

sys.path.insert(0, str(BUILD))
import check_public  # noqa: E402  (shared text extraction and the checks)
import public_figures  # noqa: E402  (the declarative figure spec)

DATASETS = ("TACO", "ARCTIC", "OakInk2")
DIRECTIONS = ("lower_better", "higher_better", "none")

# How a displayed number may derive from the full-precision internal value. Anything else must be shown as it is.
TRANSFORMS = {
    "identity": lambda x: x,                               # 0.8457 -> "0.85"
    "percent": lambda x: x * 100.0,                        # 0.4575 -> "46%"
    "abs": lambda x: abs(x),                               # -0.07 -> "0.07"
    "abs_percent": lambda x: abs(x) * 100.0,               # -0.1824 -> "18%" (said as "18% lower")
    "one_minus_percent": lambda x: (1.0 - x) * 100.0,      # 0.8927 explained -> "11%" left
    "minus_one_percent": lambda x: (x - 1.0) * 100.0,      # ratio 1.15 -> "+15%"
}


class BuildError(Exception):
    pass


def digits(strings):
    """All runs of digits (with decimals) in a collection of strings."""
    found = set()
    for s in strings:
        found.update(re.findall(r"[0-9]+(?:\.[0-9]+)?", s))
    return found


def strings_of(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from strings_of(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from strings_of(v)


def numbers_of(obj, path=""):
    """Paths of numeric values in a spec (there must be none)."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield path
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from numbers_of(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from numbers_of(v, f"{path}[{i}]")


# --------------------------------------------------------------------------------------------------
# 1. Figures
# --------------------------------------------------------------------------------------------------

def per_panel(value, panel):
    """A spec field that is either one list for all panels or {panel title: list}."""
    if isinstance(value, dict):
        if panel not in value:
            raise BuildError(f"no entry for panel {panel!r}")
        return value[panel]
    return value


def one(items, key, wanted, what):
    hits = [x for x in items if x.get(key) == wanted]
    if len(hits) != 1:
        raise BuildError(f"{what} {wanted!r}: found {len(hits)} times in the internal figure (expected once)")
    return hits[0]


def build_figure(pub_id, spec, internal):
    """(public figure, trace). The trace lists where every copied number comes from."""
    if not re.fullmatch(r"pub_[a-z0-9_]+", pub_id):
        raise BuildError("a public figure id must look like pub_...")
    typed = list(numbers_of({k: v for k, v in spec.items()}))
    if typed:
        raise BuildError(f"the spec holds numeric values at {typed}: no number may be typed in the spec")
    src_id = spec["source"]
    if src_id not in internal:
        raise BuildError(f"internal figure {src_id!r} does not exist")
    src = internal[src_id]
    if spec["direction"] not in DIRECTIONS:
        raise BuildError(f"direction {spec['direction']!r} is not one of {DIRECTIONS}")
    fig = {"type": src["type"], "title": spec["title"], "panels": [], "note": spec["note"]}
    trace = []
    for panel_title in spec["panels"]:
        sp = one(src["panels"], "title", panel_title, "panel")
        if sp["y"].get("direction") != spec["direction"]:
            raise BuildError(f"panel {panel_title}: direction {spec['direction']!r} differs from the internal figure's {sp['y'].get('direction')!r}")
        src_cats = sp["x"].get("categories")
        if not src_cats:
            raise BuildError(f"panel {panel_title}: the internal figure has no categories (only categorical figures are supported)")
        cats = per_panel(spec["categories"], panel_title)
        index = []
        for src_name, _pub in cats:
            if src_cats.count(src_name) != 1:
                raise BuildError(f"panel {panel_title}: category {src_name!r} found {src_cats.count(src_name)} times in the internal figure")
            index.append(src_cats.index(src_name))
        if index != sorted(index):
            raise BuildError(f"panel {panel_title}: the kept categories are not in the order of the internal figure")
        pub_cats = [pub for _src, pub in cats]
        if len(set(pub_cats)) != len(pub_cats):
            raise BuildError(f"panel {panel_title}: two categories share a public name")
        panel = {
            "title": panel_title,
            "x": {"label": spec["x_label"], "categories": pub_cats},
            "y": {"label": spec["y_label"], "direction": spec["direction"]},
            "series": [],
        }
        for src_name, pub_name in spec["series"]:
            ss = one(sp.get("series", []), "name", src_name, f"panel {panel_title}: series")
            out = {"name": pub_name, "values": [ss["values"][i] for i in index]}
            for bound in ("lo", "hi"):
                if bound in ss:
                    out[bound] = [ss[bound][i] for i in index]
            panel["series"].append(out)
            for k, i in enumerate(index):
                trace.append({"panel": panel_title, "series": pub_name, "category": pub_cats[k],
                              "from": {"figure": src_id, "panel": panel_title, "series": src_name, "category": src_cats[i]}})
        refs = per_panel(spec.get("refs") or [], panel_title)
        if refs:
            panel["refs"] = []
            for src_label, pub_label in refs:
                sr = one(sp.get("refs", []), "label", src_label, f"panel {panel_title}: reference line")
                panel["refs"].append({"axis": sr["axis"], "value": sr["value"], "label": pub_label})
                trace.append({"panel": panel_title, "ref": pub_label, "from": {"figure": src_id, "panel": panel_title, "ref": src_label}})
        hl = spec.get("highlight")
        if hl:
            src_hl = sp.get("highlight") or {}
            if src_hl.get("category") != hl["category"]:
                raise BuildError(f"panel {panel_title}: the internal figure highlights {src_hl.get('category')!r}, the spec {hl['category']!r}")
            names = dict(cats)
            if hl["category"] not in names:
                raise BuildError(f"panel {panel_title}: the highlighted category {hl['category']!r} is not kept")
            panel["highlight"] = {"category": names[hl["category"]], "label": hl["label"]}
        fig["panels"].append(panel)
    # no new number in a public string: every digit run of the public strings occurs in the internal figure's strings
    internal_digits = digits(strings_of({k: v for k, v in src.items() if k not in ("sources", "module")}))
    public_strings = [s for s in strings_of(fig)]
    new = sorted(digits(public_strings) - internal_digits)
    if new:
        raise BuildError(f"public strings hold number(s) {new} that the internal figure {src_id!r} does not state")
    return fig, trace


def same(a, b):
    """Identity of two plotted values: equal numbers with the same text form, or both missing."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b and repr(a) == repr(b)


def verify_figure(pub_id, fig, spec, internal):
    """Independent second pass: looks every public value up again in the internal figure. Returns the number checked."""
    src = internal[spec["source"]]
    allowed = check_public.DATA_KEYS
    if set(fig) - allowed["figure"]:
        raise BuildError(f"unexpected keys {sorted(set(fig) - allowed['figure'])}")
    if fig["type"] != src["type"]:
        raise BuildError("chart type differs from the internal figure")
    checked = 0
    if [p["title"] for p in fig["panels"]] != list(spec["panels"]):
        raise BuildError("panels differ from the spec")
    for panel in fig["panels"]:
        sp = one(src["panels"], "title", panel["title"], "panel")
        cats = per_panel(spec["categories"], panel["title"])
        back = {pub: s for s, pub in cats}
        series_back = {pub: s for s, pub in spec["series"]}
        if [s["name"] for s in panel["series"]] != [pub for _s, pub in spec["series"]]:
            raise BuildError(f"panel {panel['title']}: series differ from the spec")
        for s in panel["series"]:
            ss = one(sp["series"], "name", series_back[s["name"]], "series")
            for k, cat in enumerate(panel["x"]["categories"]):
                i = sp["x"]["categories"].index(back[cat])
                if not same(s["values"][k], ss["values"][i]):
                    raise BuildError(f"{panel['title']} / {s['name']} / {cat}: value {s['values'][k]!r} differs from the internal {ss['values'][i]!r}")
                checked += 1
                for bound in ("lo", "hi"):
                    if (bound in ss) != (bound in s):
                        raise BuildError(f"{panel['title']} / {s['name']}: error-bar bound {bound!r} present on one side only")
                    if bound in ss:
                        if not same(s[bound][k], ss[bound][i]):
                            raise BuildError(f"{panel['title']} / {s['name']} / {cat}: {bound} bound differs from the internal figure")
                        checked += 1
            if len(s["values"]) != len(panel["x"]["categories"]):
                raise BuildError(f"{panel['title']} / {s['name']}: number of values differs from the number of categories")
        refs = per_panel(spec.get("refs") or [], panel["title"])
        if len(panel.get("refs", [])) != len(refs):
            raise BuildError(f"panel {panel['title']}: reference lines differ from the spec")
        for r, (src_label, pub_label) in zip(panel.get("refs", []), refs):
            sr = one(sp.get("refs", []), "label", src_label, "reference line")
            if r["label"] != pub_label or r["axis"] != sr["axis"] or not same(r["value"], sr["value"]):
                raise BuildError(f"panel {panel['title']}: reference line {pub_label!r} differs from the internal figure")
            checked += 1
        if "highlight" in panel:
            if back[panel["highlight"]["category"]] != (sp.get("highlight") or {}).get("category"):
                raise BuildError(f"panel {panel['title']}: highlight is not on the level the internal figure highlights")
    return checked


def write_if_changed(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == text:
        return False
    path.write_text(text, encoding="utf-8")
    return True


def build_figures(log):
    src_path = ROOT / public_figures.SOURCE_FIGURES
    doc = json.loads(src_path.read_text(encoding="utf-8"))
    internal = doc["figures"]
    figures, traces, errors = {}, {}, []
    for pub_id, spec in public_figures.FIGURES.items():
        try:
            fig, trace = build_figure(pub_id, spec, internal)
            verify_figure(pub_id, fig, spec, internal)
            figures[pub_id], traces[pub_id] = fig, trace
        except (BuildError, KeyError, IndexError, TypeError) as err:
            errors.append(f"figure {pub_id}: {err!r}" if not isinstance(err, BuildError) else f"figure {pub_id}: {err}")
    if errors:
        return None, None, internal, doc, errors
    text = ("/* Chart data of the page. Generated file: do not edit by hand. */\n"
            "window.DEXCORE_FIGURES = " + json.dumps({"figures": figures}, ensure_ascii=False, indent=1) + ";\n")
    figures_js = SITE / "data" / "figures.js"
    changed = write_if_changed(figures_js, text)
    # read the written file back and compare it with the internal figures once more
    parsed, err = check_public.parse_data_file(figures_js.read_text(encoding="utf-8"))
    if err:
        return None, None, internal, doc, [f"site/data/figures.js: {err}"]
    total = 0
    for pub_id, spec in public_figures.FIGURES.items():
        try:
            total += verify_figure(pub_id, parsed["figures"][pub_id], spec, internal)
        except (BuildError, KeyError, IndexError, TypeError) as err:
            errors.append(f"site/data/figures.js, figure {pub_id}: {err}")
    log(f"figures: {len(figures)} public figures, {total} values, bounds and reference values identical to the internal figures "
        f"-> site/data/figures.js ({'written' if changed else 'unchanged'})")
    return figures, traces, internal, doc, errors


# --------------------------------------------------------------------------------------------------
# 2. Images
# --------------------------------------------------------------------------------------------------

def copy_images(log):
    errors, record = [], []
    for pub_name, src_name in public_figures.IMAGES.items():
        src = DOCS / "assets" / "img" / src_name
        dst = SITE / "assets" / "img" / pub_name
        if not src.is_file():
            errors.append(f"image: the internal image docs/assets/img/{src_name} is missing")
            continue
        data = src.read_bytes()
        dst.parent.mkdir(parents=True, exist_ok=True)
        changed = not (dst.is_file() and dst.read_bytes() == data)
        if changed:
            dst.write_bytes(data)
        if dst.read_bytes() != data:
            errors.append(f"image: site/assets/img/{pub_name} differs from its source after the copy")
        meta = DOCS / "data" / "qual" / (Path(src_name).stem[len("qual_"):] + ".json")
        record.append({"public_file": f"site/assets/img/{pub_name}", "copied_from": f"docs/assets/img/{src_name}",
                       "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "edit": "none (byte copy under a public name)",
                       "internal_metadata": meta.relative_to(ROOT).as_posix() if meta.is_file() else None})
        if not meta.is_file():
            errors.append(f"image: the metadata file {meta.relative_to(ROOT).as_posix()} of docs/assets/img/{src_name} is missing")
        log(f"image: docs/assets/img/{src_name} -> site/assets/img/{pub_name} ({'copied' if changed else 'unchanged'})")
    (SITE / ".nojekyll").touch()
    return record, errors


# --------------------------------------------------------------------------------------------------
# 3. Traceability
# --------------------------------------------------------------------------------------------------

NUMBER_IN_VALUE = re.compile(r"(?<![A-Za-z0-9_.])[-+−]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


def value_from_string(text, index, where):
    found = NUMBER_IN_VALUE.findall(text)
    if not found:
        raise BuildError(f"{where}: the internal value {text!r} holds no number")
    if index is None:
        if len(found) != 1:
            raise BuildError(f"{where}: the internal value {text!r} holds {len(found)} numbers; say which one with value_index (0 = first)")
        index = 0
    if not isinstance(index, int) or not 0 <= index < len(found):
        raise BuildError(f"{where}: value_index {index!r} is outside the {len(found)} numbers of {text!r}")
    return float(found[index].replace("−", "-"))


class Internal:
    """Read-only view of the internal page's evidence."""

    def __init__(self, figures_doc):
        self.figures = figures_doc["figures"]
        self.result_root = figures_doc.get("result_root", "")
        self.manifest = json.loads((DOCS / "evidence_manifest.json").read_text(encoding="utf-8"))
        self.blocks = {}
        for b in self.manifest["blocks"]:
            self.blocks.setdefault(b["block"], []).append(b)
        self.files = {}

    def json_file(self, rel):
        path = (ROOT / rel).resolve()
        if DOCS.resolve() not in path.parents:
            raise BuildError(f"{rel}: internal evidence must be a file under docs/")
        if not path.is_file():
            raise BuildError(f"{rel}: file not found")
        if path not in self.files:
            self.files[path] = json.loads(path.read_text(encoding="utf-8"))
        return self.files[path]

    def block_reports(self, block):
        return sorted({e.get("report") for b in self.blocks.get(block, []) for e in b.get("evidence", []) if e.get("report")})

    def block_figures(self, block):
        return {f["id"] for b in self.blocks.get(block, []) for f in b.get("figures", [])}

    def resolve(self, ref, block, where):
        """(value, facts) of one internal evidence reference; facts are what the claim is compared with."""
        if not isinstance(ref, dict):
            raise BuildError(f"{where}: internal_evidence must be an object")
        if "figure" in ref:
            fig = self.figures.get(ref["figure"])
            if fig is None:
                raise BuildError(f"{where}: internal figure {ref['figure']!r} does not exist")
            panel = one(fig["panels"], "title", ref.get("panel"), f"{where}: panel")
            if "ref" in ref:
                value = one(panel.get("refs", []), "label", ref["ref"], f"{where}: reference line")["value"]
            else:
                series = one(panel.get("series", []), "name", ref.get("series"), f"{where}: series")
                cats = panel["x"].get("categories") or panel["x"].get("values") or []
                if cats.count(ref.get("category")) != 1:
                    raise BuildError(f"{where}: category {ref.get('category')!r} not found once in the internal figure")
                bound = ref.get("bound", "values")
                if bound not in ("values", "lo", "hi") or bound not in series:
                    raise BuildError(f"{where}: the series has no {bound!r}")
                value = series[bound][cats.index(ref["category"])]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise BuildError(f"{where}: the internal figure holds no number there")
            return float(value), {
                "kind": "figure", "dataset": ref.get("panel"),
                "source_files": [s["path"] for s in fig.get("sources", [])],
                "locators": {s["path"]: s.get("locator", "") for s in fig.get("sources", [])},
                "reports": self.block_reports(block),
                "in_block": ref["figure"] in self.block_figures(block),
                "module": fig.get("module"),
            }
        if "evidence_file" in ref:
            doc = self.json_file(ref["evidence_file"])
            hits = []
            for i, entry in enumerate(doc.get("entries", [])):
                if ref.get("key") in entry.get("numbers", {}) and ref.get("entry", i) == i:
                    hits.append((i, entry))
            if len(hits) != 1:
                raise BuildError(f"{where}: key {ref.get('key')!r} found {len(hits)} times in {ref['evidence_file']} (expected once; "
                                 f"add \"entry\": <index> if several entries share it)")
            i, entry = hits[0]
            number = entry["numbers"][ref["key"]]
            value = value_from_string(str(number["value"]), ref.get("value_index"), where)
            key_dataset = ref["key"].split("|")[0].strip()
            return value, {
                "kind": "evidence", "dataset": key_dataset if key_dataset in DATASETS else None,
                "source_files": [number.get("source")], "locators": {number.get("source"): number.get("locator", "")},
                "reports": [entry.get("report")], "in_block": entry.get("block") == block,
                "entry": i, "internal_claim": entry.get("claim"), "metric": entry.get("metric"), "raw_value": number["value"],
            }
        if "json_file" in ref:
            node = self.json_file(ref["json_file"])
            for step in ref.get("path", []):
                try:
                    node = node[step]
                except (KeyError, IndexError, TypeError):
                    raise BuildError(f"{where}: path {ref.get('path')!r} does not exist in {ref['json_file']}")
            if isinstance(node, bool) or not isinstance(node, (int, float, str)):
                raise BuildError(f"{where}: {ref['json_file']} holds no number or text at {ref.get('path')!r}")
            value = float(node) if isinstance(node, (int, float)) else value_from_string(node, ref.get("value_index"), where)
            return value, {"kind": "json", "dataset": None, "source_files": None, "locators": {},
                           "reports": self.block_reports(block), "in_block": True, "raw_value": node}
        raise BuildError(f"{where}: internal_evidence needs \"figure\", \"evidence_file\" or \"json_file\"")


def check_displayed(displayed, full, transform, where):
    """Failure a: the displayed number must be the full-precision value at the displayed rounding."""
    if transform not in TRANSFORMS:
        raise BuildError(f"{where}: unknown transform {transform!r} (known: {', '.join(TRANSFORMS)})")
    m = re.fullmatch(r"\s*[≈~]?\s*([+\-−]?)\s*((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.([0-9]+))?)\s?(%?)\s*[A-Za-zµ°]*\s*", str(displayed))
    if not m:
        raise BuildError(f"{where}: displayed {displayed!r} is not one number (optionally with a sign, % or a unit)")
    sign, body, frac, pct = m.groups()
    shown = float(body.replace(",", "")) * (-1.0 if sign in ("-", "−") else 1.0)
    value = TRANSFORMS[transform](full)
    step = 10.0 ** (-len(frac or ""))
    if abs(shown - value) > 0.5 * step * (1 + 1e-9) + 1e-12:
        raise BuildError(f"{where}: displayed {displayed!r} is not {value!r} (full value {full!r}, transform {transform}) at the displayed rounding")
    return value


REQUIRED_CLAIM = ("id", "section", "public_claim", "datasets", "numbers", "figure_asset")
REQUIRED_NUMBER = ("displayed", "dataset", "metric", "full_value", "internal_block", "internal_evidence", "internal_report", "source_file")
# a supporting value: what backs a claim that shows no digit ("unchanged", "no reduction", "almost only").
# It is checked against the internal evidence like a displayed number, but it is not a number of the page.
REQUIRED_SUPPORT = ("shown_as", "dataset", "metric", "full_value", "internal_block", "internal_evidence", "internal_report", "source_file")


def asset_exists(asset, figures):
    if asset is None:
        return True
    if isinstance(asset, list):
        return all(asset_exists(a, figures) for a in asset)
    if not isinstance(asset, str):
        return False
    if re.fullmatch(r"pub_[a-z0-9_]+", asset):
        return figures is not None and asset in figures
    path = (SITE / asset).resolve()
    return SITE.resolve() in path.parents and path.is_file()


def verify_value(num, claim, internal, where):
    """Failure b for one displayed number or one supporting value. Returns (internal value, facts, locator)."""
    if num["dataset"] not in claim["datasets"]:
        raise BuildError(f"{where}: dataset {num['dataset']!r} is not among the claim's datasets")
    if num["internal_block"] not in internal.blocks:
        raise BuildError(f"{where}: internal block {num['internal_block']!r} does not exist in docs/evidence_manifest.json (failure b)")
    value, facts = internal.resolve(num["internal_evidence"], num["internal_block"], where)
    try:
        full = float(num["full_value"])
    except (TypeError, ValueError):
        raise BuildError(f"{where}: full_value {num['full_value']!r} is not a number")
    if full != value:
        raise BuildError(f"{where}: full_value {num['full_value']!r} differs from the internal value {value!r} (failure b)")
    if facts["dataset"] and facts["dataset"] != num["dataset"]:
        raise BuildError(f"{where}: dataset {num['dataset']!r} differs from the internal evidence's {facts['dataset']!r} (failure b)")
    if not facts["in_block"]:
        raise BuildError(f"{where}: the internal evidence is not part of block {num['internal_block']!r} (failure b)")
    if facts["source_files"] is not None and num["source_file"] not in facts["source_files"]:
        raise BuildError(f"{where}: source_file {num['source_file']!r} is not the internal source {facts['source_files']} (failure b)")
    if num["internal_report"] not in facts["reports"]:
        raise BuildError(f"{where}: internal_report {num['internal_report']!r} is not the report of the internal evidence {facts['reports']} (failure b)")
    locator = num.get("locator") or facts["locators"].get(num["source_file"], "")
    return value, facts, locator


def verified_entry(num, required, value, facts, locator):
    entry = {k: num[k] for k in required}
    entry["locator"] = locator
    for k in ("where", "note"):
        if k in num:
            entry[k] = num[k]
    entry["verified"] = {"internal_value": value, "internal_evidence_kind": facts["kind"]}
    for k in ("internal_claim", "raw_value", "module"):
        if facts.get(k) is not None:
            entry["verified"][k] = facts[k]
    return entry


def build_manifest(figures, traces, figures_doc, images, log):
    """Validates public_claims.json and writes public_evidence_manifest.json. Returns the list of failures."""
    failures, warnings = [], []
    if not CLAIMS.is_file():
        return [f"claims: {CLAIMS} not found"]
    try:
        doc = json.loads(CLAIMS.read_text(encoding="utf-8"))
    except ValueError as err:
        return [f"claims: public_claims.json is not valid JSON: {err}"]
    internal = Internal(figures_doc)
    out_claims, seen_ids = [], set()
    for ci, claim in enumerate(doc.get("claims", [])):
        cid = claim.get("id", f"#{ci}")
        missing = [k for k in REQUIRED_CLAIM if k not in claim]
        if missing:
            failures.append(f"claim {cid}: missing field(s) {missing}")
            continue
        if cid in seen_ids:
            failures.append(f"claim {cid}: id used twice")
        seen_ids.add(cid)
        if not claim["datasets"] or any(d not in DATASETS for d in claim["datasets"]):
            failures.append(f"claim {cid}: datasets must be a non-empty list out of {DATASETS}")
        if not asset_exists(claim["figure_asset"], figures):
            failures.append(f"claim {cid}: figure asset {claim['figure_asset']!r} is missing (failure c)")
        out = {k: claim[k] for k in ("id", "section", "public_claim", "datasets", "figure_asset")}
        if claim.get("shown"):
            out["shown"] = claim["shown"]
        out["numbers"] = []
        for ni, num in enumerate(claim["numbers"]):
            where = f"claim {cid}, number {ni + 1} ({num.get('displayed')!r})"
            missing = [k for k in REQUIRED_NUMBER if k not in num]
            if missing:
                failures.append(f"{where}: missing field(s) {missing}")
                continue
            try:
                value, facts, locator = verify_value(num, claim, internal, where)
                shown_value = check_displayed(num["displayed"], float(num["full_value"]), num.get("transform", "identity"), where + " (failure a)")
                if check_public.displayed_key(num["displayed"]) is None:
                    raise BuildError(f"{where}: displayed must hold exactly one number")
                if num.get("where", "text") not in ("text", "chart"):
                    raise BuildError(f"{where}: \"where\" must be \"text\" (the default) or \"chart\"")
                entry = verified_entry(num, REQUIRED_NUMBER, value, facts, locator)
                entry.update({"transform": num.get("transform", "identity"), "value_at_display": shown_value})
                out["numbers"].append(entry)
            except BuildError as err:
                failures.append(str(err))
        if claim.get("support"):
            out["support"] = []
            for si, sup in enumerate(claim["support"]):
                where = f"claim {cid}, supporting value {si + 1} ({sup.get('shown_as')!r})"
                missing = [k for k in REQUIRED_SUPPORT if k not in sup]
                if missing:
                    failures.append(f"{where}: missing field(s) {missing}")
                    continue
                try:
                    value, facts, locator = verify_value(sup, claim, internal, where)
                    if sup.get("where") not in (None, "chart"):
                        raise BuildError(f"{where}: \"where\" of a supporting value can only be \"chart\" (a number typed into a chart label)")
                    out["support"].append(verified_entry(sup, REQUIRED_SUPPORT, value, facts, locator))
                except BuildError as err:
                    failures.append(str(err))
        if str(claim.get("evidence_note", "")).strip():
            out["evidence_note"] = claim["evidence_note"]
        if not claim["numbers"] and not claim.get("support") and "evidence_note" not in out:
            failures.append(f"claim {cid}: no displayed number, no supporting value and no evidence_note: the claim has no trace (failure f)")
        out_claims.append(out)

    # d, e. the claims against the visible text of site/index.html
    index = SITE / "index.html"
    coverage = {"status": "skipped", "reason": "site/index.html does not exist yet"}
    if index.is_file():
        site = check_public.Site(SITE)
        try:
            uncovered, used, total = check_public.number_coverage(site.index.units, doc)
            problems = check_public.claims_on_page(site.index.units, doc, site.data)
            chart_problems, chart_total = check_public.chart_text_coverage(site.data, doc)
        except re.error as err:
            uncovered, used, total, problems, chart_problems, chart_total = [], set(), 0, [], [], 0
            failures.append(f"claims: a structural_numbers regex is not valid: {err}")
        coverage = {"status": "checked", "numbers_in_page_text": total, "displayed_numbers_matched": sorted(used), "uncovered": [],
                    "numbers_in_chart_text": chart_total}
        for u, a, b, key in uncovered:
            coverage["uncovered"].append({"line": u.line_at(a), "number": u.text[a:b].strip(), "context": check_public.excerpt(u.text, a, b)})
            failures.append(f"index.html:{u.line_at(a)}: number {u.text[a:b].strip()!r} does not lie inside the text of a claim that lists it "
                            f"and is not a structural number (failure d): {check_public.excerpt(u.text, a, b)}")
        for cid, text, why in problems:
            failures.append(f"claim {cid}: {why} (failure e)" + (f": {text}" if text else ""))
        for fid, number, context, why in chart_problems:
            failures.append(f"data/figures.js, {fid}: {number} {context}: {why}".replace("  ", " "))
        for claim in doc.get("claims", []):
            for num in claim.get("numbers", []):
                if num.get("where") == "chart":
                    warnings.append(f"claim {claim.get('id')}: {num.get('displayed')!r} is shown as a value label of a chart only (not in the page text)")
        log(f"claims: number coverage of site/index.html: {total} numbers in the text, {len(uncovered)} outside a claim that lists them; "
            f"{len(problems)} claim text problem(s); {chart_total} numbers in the chart titles, labels and notes, {len(chart_problems)} not listed")
    else:
        log("claims: number coverage SKIPPED: site/index.html does not exist yet (the check runs as soon as the page exists)")

    block_of_image = {}
    for b in internal.manifest["blocks"]:
        for img in b.get("images", []):
            name = img if isinstance(img, str) else (img.get("file") or img.get("path") or img.get("id") or "")
            block_of_image[Path(str(name)).name] = b["block"]
    for rec in images:
        rec["internal_block"] = block_of_image.get(Path(rec["copied_from"]).name)

    manifest = {
        "title": "Public evidence manifest: claims shown on the public page and where each comes from on the internal page",
        "status": "FAILED: see failures" if failures else "ok",
        "note": "DEVELOPMENT ONLY. This file holds internal paths and is never shipped: it lives outside site/.",
        "generated_by": "public_page/build/build_public.py (no time stamp: the same inputs give the same file)",
        "source_of_truth": {"page": "docs/index.html", "manifest": "docs/evidence_manifest.json", "figures": public_figures.SOURCE_FIGURES,
                            "result_root": internal.result_root},
        "how_to_read": "claims: one entry per claim of the public page. numbers: every displayed number with its full-precision value, the "
                       "internal evidence block, the internal report, the source result file and its locator. support: values that back a "
                       "claim shown without a digit ('unchanged', 'no reduction'), checked against the internal evidence in the same way; "
                       "a supporting value with where = chart backs a number typed into a label of a public chart (shown_as is that label). "
                       "evidence_note: what else backs the claim and the caveats that belong to it (free text, not machine-checked). "
                       "shown: the exact strings on the page that carry the numbers outside the claim sentence (stat tiles). "
                       "figures: every public chart with the internal figure it is derived from and that figure's source files. "
                       "images: every public image with the internal render it is a copy of, its evidence block and its metadata file. "
                       "Source and report paths are relative to result_root unless they start with docs/.",
        "failures": failures,
        "warnings": warnings,
        "claims": out_claims,
        "structural_numbers": doc.get("structural_numbers", []),
        "chart_text_numbers": doc.get("chart_text_numbers", {}),
        "number_coverage": coverage,
        "figures": [],
        "images": images,
    }
    for pub_id, spec in public_figures.FIGURES.items():
        src = figures_doc["figures"].get(spec["source"], {})
        blocks = sorted(b["block"] for b in internal.manifest["blocks"] if spec["source"] in {f["id"] for f in b.get("figures", [])})
        manifest["figures"].append({
            "public_figure": pub_id, "public_title": spec["title"], "shipped_in": "site/data/figures.js",
            "derived_from": spec["source"], "internal_blocks": blocks, "internal_module": src.get("module"),
            "internal_sources": src.get("sources", []), "panels_kept": list(spec["panels"]),
            "series": [{"internal": s, "public": p} for s, p in spec["series"]],
            "categories": ({k: [{"internal": s, "public": p} for s, p in v] for k, v in spec["categories"].items()}
                           if isinstance(spec["categories"], dict) else [{"internal": s, "public": p} for s, p in spec["categories"]]),
            "values": (traces or {}).get(pub_id, []),
        })
    changed = write_if_changed(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    n_numbers = sum(len(c["numbers"]) for c in out_claims)
    n_support = sum(len(c.get("support", [])) for c in out_claims)
    log(f"claims: {len(out_claims)} claims, {n_numbers} displayed numbers and {n_support} supporting values verified against the internal "
        f"evidence; {len(failures)} failure(s), {len(warnings)} note(s) -> public_evidence_manifest.json ({'written' if changed else 'unchanged'})")
    for w in warnings:
        log(f"  note: {w}")
    return failures


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quiet", action="store_true", help="print failures only")
    ap.add_argument("--site", help="write the site data into this folder instead of public_page/site (tests)")
    ap.add_argument("--claims", help="read this claims file instead of build/public_claims.json (tests)")
    ap.add_argument("--manifest", help="write the manifest to this file (tests)")
    ap.add_argument("--checklist", help="write the checklist to this file (tests)")
    ap.add_argument("--manual", help="read the record of the manual checks from this file (tests)")
    args = ap.parse_args()
    global SITE, CLAIMS, MANIFEST, CHECKLIST, MANUAL
    SITE = Path(args.site).resolve() if args.site else SITE
    CLAIMS = Path(args.claims).resolve() if args.claims else CLAIMS
    MANIFEST = Path(args.manifest).resolve() if args.manifest else MANIFEST
    CHECKLIST = Path(args.checklist).resolve() if args.checklist else CHECKLIST
    MANUAL = Path(args.manual).resolve() if args.manual else MANUAL

    def log(msg):
        if not args.quiet:
            print(msg)

    failures = []
    figures, traces, _internal, figures_doc, errs = build_figures(log)
    failures += errs
    images, errs = copy_images(log)
    failures += errs
    failures += build_manifest(figures, traces, figures_doc, images, log)
    for f in failures:
        print(f"BUILD FAILURE: {f}")
    code, _results = check_public.run(SITE, CHECKLIST, CLAIMS, args.quiet, MANUAL)
    ok = not failures and code == 0
    print(f"build_public: {'OK' if ok else 'FAILED'} ({len(failures)} build failure(s); checks {'passed' if code == 0 else 'failed'})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
