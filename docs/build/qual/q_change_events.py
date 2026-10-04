"""Figure "change_events" (page block F1-C): what an amount-dominant, a spatial-dominant and a mixed
change of the contact map look like on the object.

Three rows from ONE sequence, the TACO test sequence of the page's timeline chart (a right hand and a
knife, from a cutting take). Each row is one spike event that the timeline caption names. Columns:

    before | after | recorded change (after minus before) | the evolved model's map at the after frame

"Before" is the frame at which the event's first transition starts, "after" the frame at which its
last transition ends (transition t goes from frame t to frame t + 1, as in the event study).

No sequence is chosen here: it is read from the page's own chart module. The three events are fixed
by their transitions (DRAWN); they are three of the five events of that sequence the timeline caption
names, and the script checks in the event table why these three: the only amount-dominant event of
the sequence, the report's representative spatial-dominant event, and the larger of its mixed events.
No model is run: the recorded maps come from the study's sequences, the model's maps from its saved
predictions. Every number printed in the figure is read from the event table and recomputed from the
arrays.

The errors in the row labels score the predicted frame-to-frame CHANGE, not the map of the last
column (a state, which also carries the model's drift since the first frame). The labels, the
footnote in the figure and the caption say so, and the caption quotes the state distances as well.

Run (CPU, software renderer):

    CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow \\
        PYVISTA_OFF_SCREEN=true python docs/build/qual/q_change_events.py
"""
from __future__ import annotations

import ast
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qlib  # noqa: E402

log = logging.getLogger("q_change_events")

FIG_ID = "change_events"
BLOCK = "F1-C"
DATASET = "taco"
SCRIPT = Path(__file__).name

# ------------------------------------------------------------------------------ inputs (all read-only)
EVENTS_CSV = qlib.RESULT_ROOT / "reports/temporal_contact_events" / DATASET / "events.csv"
SEQUENCES_NPZ = qlib.HIER[DATASET] / "sequences.npz"
PREDS_NPZ = qlib.HIER[DATASET] / "eval/fixed0/preds.npz"
ZERO_MASS_JSON = qlib.RESULT_ROOT / "taco/40_representation_study/time_decomp/dynamic_contact/cache/zero_threshold.json"
PAGE_CHART_MODULE = qlib.REPO / "docs/build/figs/f1c_temporal_events.py"     # names the timeline sequence
PAGE_HTML = qlib.REPO / "docs/index.html"                                    # its caption names the events
MODEL_KEY = "gtinit_vf"            # saved rollout of the evolution model from the true first map
MODEL_CHECKPOINT = qlib.RESULT_ROOT / "hier_contact_gen_ckpt/taco/fixed0/vf_unroll8.pt"   # made that rollout; NOT loaded

# The events of the sequence that the page's timeline caption names:
# (first transition, last transition) -> (class in the event table, the caption's wording).
PAGE_NAMED = {(7, 8): ("amount", "amount-dominant event at transitions 7\N{EN DASH}8"),
              (26, 28): ("mixed", "mixed event at 26\N{EN DASH}28"),
              (34, 35): ("spatial", "spatial-dominant event at 34\N{EN DASH}35"),
              (39, 42): ("mixed", "large mixed event at 39\N{EN DASH}42"),
              (19, 20): ("spatial", "spatial event at 19\N{EN DASH}20")}
# The three of them this figure draws, fixed by the figure's brief: class -> (first transition, last transition).
DRAWN = {"amount": (7, 8), "spatial": (34, 35), "mixed": (39, 42)}
ROW_TITLE = {"amount": "Amount-dominant", "spatial": "Spatial-dominant", "mixed": "Mixed"}
AMOUNT_HI, SPATIAL_LO = 0.7, 0.3   # class limits of the event study on amount / (amount + spatial)
REPRESENTATIVE_PEAK = (4, 58)      # peak transitions the report allows for its representative events (tce_figures.py::fig_G)

# ------------------------------------------------------------------------------ display choices
PANEL = (800, 430)                 # width, height of one rendered panel in the master image, pixels
HAND_OPACITY = 0.6                 # translucent, so the contact under the hand stays visible
SCALE_STEP = 0.05                  # colour limits are the largest drawn value rounded up to this step
CHANGE_MARK = 0.02                 # a vertex "gains" / "loses" contact when its drawn change exceeds this
NEAR = 0.01                        # metres; hand vertices closer than this to an object vertex are counted as near
COL_LABELS = ("Before\n(hand, recorded contact)", "After\n(hand, recorded contact)",
              "Recorded change\n(after minus before)", "Evolved model\n(contact at the after frame)")
# Printed under the panels: what the two errors of the row labels are, and what they are not.
FOOTNOTE = ("Error of the change (the timeline chart's temporal error): distance between the predicted and the recorded "
            "frame-to-frame change, averaged over the event's transitions; hold = the map kept still. "
            "It does not score the map in the last column.")
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


@dataclass(frozen=True)
class Event:
    kind: str            # "amount", "spatial" or "mixed" (class in the event table)
    event_id: str
    first: int           # first transition of the event
    last: int            # last transition of the event
    before: int          # frame drawn as "before" (= first)
    after: int           # frame drawn as "after" (= last + 1)
    table: pd.Series     # the event's row of events.csv

    @property
    def name(self) -> str:
        return ROW_TITLE[self.kind].lower()


# ------------------------------------------------------------------------------ selection (nothing is chosen here)
def page_timeline_example(dataset: str) -> int:
    """The sequence the page's timeline chart uses, read from the chart module's source (not imported)."""
    tree = ast.parse(PAGE_CHART_MODULE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TIMELINE_EXAMPLES" for t in node.targets):
            for name, _, example in ast.literal_eval(node.value):
                if name == dataset:
                    return int(example)
    raise LookupError(f"TIMELINE_EXAMPLES has no entry for {dataset!r} in {PAGE_CHART_MODULE}")


def select(events: pd.DataFrame, example: int) -> tuple[list[Event], dict[str, Any]]:
    """Look the three drawn events up in the event table; return them and the evidence for the pick.

    Raises if the event table does not support what the selection rule says about them."""
    in_sequence = events[events.example == example].sort_values("start_frame")

    def lookup(first: int, last: int) -> pd.Series | None:
        hit = in_sequence[(in_sequence.start_frame == first) & (in_sequence.end_frame == last)]
        return hit.iloc[0] if len(hit) == 1 else None

    chosen: list[Event] = []
    for kind, (first, last) in DRAWN.items():
        row = lookup(first, last)
        if row is None or row.event_category != kind:
            raise LookupError(f"the event table has no {kind} event at transitions {first}-{last} of example {example}")
        chosen.append(Event(kind, str(row.event_id), first, last, first, last + 1, row))
    drawn = {e.event_id for e in chosen}
    by_kind = {e.kind: e for e in chosen}

    # Why this sequence is the page's example: it holds the report's representative spatial-dominant event, the
    # spatial-dominant test event whose largest transition change is nearest the class median.
    lo, hi = REPRESENTATIVE_PEAK
    spatial = events[(events.event_category == "spatial") & (events.peak_frame >= lo) & (events.peak_frame <= hi)]
    median = float(spatial.peak_d.median())
    distance = (spatial.peak_d - median).abs()
    report_pick = spatial.iloc[distance.argsort().iloc[0]]                 # the report's own expression (tce_figures.py::fig_G)
    order = distance.sort_values(kind="stable")
    tied = order[np.isclose(order, order.iloc[0], rtol=0.0, atol=1e-9)]
    ranked = [{"rank": int((order < order[j] - 1e-9).sum()) + 1, "event_id": str(events.event_id[j]), "example": int(events.example[j]),
               "largest_transition_change": float(events.peak_d[j]), "distance_to_median": float(order[j])}
              for j in order.index[:5]]
    # What the page's timeline caption names, and why these three of them are drawn (each reason checked in the table).
    html = PAGE_HTML.read_text(encoding="utf-8") if PAGE_HTML.exists() else ""
    named: list[dict[str, Any]] = []
    for (first, last), (kind, wording) in PAGE_NAMED.items():
        row = lookup(first, last)
        named.append({"wording": wording, "found_in_page_caption": wording in html, "transitions": [first, last],
                      "event_id": None if row is None else str(row.event_id),
                      "class_in_event_table": None if row is None else str(row.event_category),
                      "class_agrees_with_wording": bool(row is not None and row.event_category == kind),
                      "drawn": bool(row is not None and str(row.event_id) in drawn)})
        if not named[-1]["found_in_page_caption"]:
            log.warning("the page caption no longer contains %r", wording)
    of_class = {kind: in_sequence[in_sequence.event_category == kind] for kind in DRAWN}
    mixed = of_class["mixed"]
    reasons = {
        "amount": {"reason": "the only amount-dominant event of the sequence",
                   "n_events_of_class_in_sequence": int(len(of_class["amount"])),
                   "holds": bool(len(of_class["amount"]) == 1)},
        "spatial": {"reason": "the report's representative spatial-dominant event (see 'sequence')",
                    "n_events_of_class_in_sequence": int(len(of_class["spatial"])),
                    "event_the_reports_expression_returns": str(report_pick.event_id),
                    "holds": bool(str(report_pick.event_id) == by_kind["spatial"].event_id)},
        "mixed": {"reason": "the one the page caption calls the large mixed event: the larger of the sequence's mixed events, "
                            "by energy and by largest transition change",
                  "n_events_of_class_in_sequence": int(len(mixed)),
                  "candidates": [{"event_id": str(r.event_id), "transitions": [int(r.start_frame), int(r.end_frame)],
                                  "energy": float(r.energy), "largest_transition_change": float(r.peak_d),
                                  "drawn": str(r.event_id) in drawn} for r in mixed.itertuples()],
                  "holds": bool(str(mixed.loc[mixed.energy.idxmax()].event_id) == by_kind["mixed"].event_id
                                and str(mixed.loc[mixed.peak_d.idxmax()].event_id) == by_kind["mixed"].event_id)},
    }
    broken = [kind for kind, r in reasons.items() if not r["holds"]]
    if broken:
        raise LookupError(f"the event table does not support the stated reason for drawing the {', '.join(broken)} event")
    evidence = {
        "sequence": {
            "example": example,
            "named_in": f"{PAGE_CHART_MODULE.relative_to(qlib.REPO)} :: TIMELINE_EXAMPLES",
            "rule_of_the_page": "the test sequence that holds the report's representative spatial-dominant event: among the "
                                f"spatial-dominant test events with peak transition {lo} to {hi}, the one whose largest transition "
                                "change is nearest the median",
            "n_candidate_events": int(len(spatial)), "median_largest_transition_change": median,
            "nearest_to_median_ranked": ranked,
            "n_tied_at_the_smallest_distance": int(len(tied)),
            "tied_events": [str(events.event_id[j]) for j in tied.index],
            "event_the_reports_expression_returns": str(report_pick.event_id),
            "that_event_is_in_this_sequence": bool(int(report_pick.example) == example),
        },
        "events": {
            "rule": f"three of the {sum(n['found_in_page_caption'] for n in named)} events of this sequence that the timeline "
                    f"caption on the page names ({len(in_sequence)} spike events in the sequence), fixed by their transitions "
                    "and looked up in the event table: the only amount-dominant one, the spatial-dominant one that is the "
                    f"report's representative event ({by_kind['spatial'].event_id}), and the one the caption calls the large "
                    "mixed event; the other named events ("
                    + ", ".join(n["wording"] for n in named if not n["drawn"]) + ") are not drawn",
            "class_limits": {"amount-dominant": f"amount / (amount + spatial) >= {AMOUNT_HI}",
                             "spatial-dominant": f"amount / (amount + spatial) <= {SPATIAL_LO}", "mixed": "in between"},
            "named_in_page_caption": named,
            "n_named_in_page_caption": int(sum(n["found_in_page_caption"] for n in named)),
            "why_these_three": reasons,
            "n_events_in_sequence": int(len(in_sequence)),
            "all_events_of_the_sequence": [
                {"event_id": str(r.event_id), "class": str(r.event_category), "transitions": [int(r.start_frame), int(r.end_frame)],
                 "amount_part": float(r.amount_component), "spatial_part": float(r.spatial_component),
                 "amount_share": float(r.amount_ratio), "energy": float(r.energy), "firm_contact_before": bool(r.firm_before),
                 "hold_error": float(r.e_B0_static), "model_error": float(r.e_B1_gtinit_vf), "drawn": str(r.event_id) in drawn}
                for r in in_sequence.itertuples()],
        },
    }
    return chosen, evidence


def within_class(events: pd.DataFrame, event: Event) -> dict[str, Any]:
    """Where the event stands among all test events of its class (for the caveats)."""
    same = events[events.event_category == event.kind]
    return {"n_events_of_class": int(len(same)),
            "rank_by_largest_transition_change": int((same.peak_d > event.table.peak_d).sum()) + 1,
            "share_with_smaller_or_equal_largest_change": float((same.peak_d <= event.table.peak_d).mean()),
            "share_starting_in_firm_contact": float(same.firm_before.mean()),
            "this_event_starts_in_firm_contact": bool(event.table.firm_before),
            "this_event_flagged_short_lived": bool(event.table.transient_flag),
            "share_where_model_error_below_hold_error": float((same.e_B1_gtinit_vf < same.e_B0_static).mean())}


# ------------------------------------------------------------------------------ the event study's quantities, recomputed
def change_parts(maps: np.ndarray, zero_mass: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per transition of a (T, 512) map sequence: size of the change, its amount part and its spatial part.

    The event study's split (tce_common.decompose): with total m and distribution P = C / m,
    C' - C = (m' - m) mean(P) + mean(m) (P' - P); the norms of the two terms are the amount and the spatial part.
    Undefined (NaN) where either frame's total is below ``zero_mass``."""
    prev, nxt = maps[:-1], maps[1:]
    m0, m1 = prev.sum(1), nxt.sum(1)
    size = np.linalg.norm(nxt - prev, axis=1)
    ok = (m0 >= zero_mass) & (m1 >= zero_mass)
    amount = np.full(len(size), np.nan, np.float32)
    spatial = np.full(len(size), np.nan, np.float32)
    p0, p1 = prev[ok] / m0[ok, None], nxt[ok] / m1[ok, None]
    amount[ok] = np.linalg.norm((m1 - m0)[ok, None] * 0.5 * (p0 + p1), axis=1)
    spatial[ok] = np.linalg.norm(0.5 * (m0 + m1)[ok, None] * (p1 - p0), axis=1)
    return size, amount, spatial


def round_up(value: float, step: float = SCALE_STEP) -> float:
    return round(float(np.ceil(value / step - 1e-9)) * step, 6)


class Ledger:
    """Every number a caption could quote, with its source; and the checks of the table against the arrays."""

    def __init__(self) -> None:
        self.numbers: list[dict[str, Any]] = []
        self.checks: list[dict[str, Any]] = []

    def add(self, label: str, value: float, source: Path | str, locator: str) -> float:
        self.numbers.append({"label": label, "value": float(value), "dataset": "TACO", "source": str(source), "locator": locator})
        return float(value)

    def check(self, quantity: str, table: float, recomputed: float) -> None:
        gap = abs(float(table) - float(recomputed))
        agree = bool(gap <= 1e-3 * max(1.0, abs(float(table))))
        self.checks.append({"quantity": quantity, "event_table": float(table), "recomputed_from_arrays": float(recomputed),
                            "abs_difference": gap, "agree": agree})
        if not agree:
            log.warning("DISAGREEMENT %s: event table %.6f, recomputed %.6f", quantity, table, recomputed)

    def trace(self, text: str) -> list[dict[str, Any]]:
        """Every number written in ``text`` with the ledger entries it equals at the precision it is written in.
        Raises if a number has no entry: a caption may quote only what the ledger holds."""
        found: list[dict[str, Any]] = []
        for token in re.findall(r"\d+(?:\.\d+)?", text):
            digits = len(token.partition(".")[2])
            labels = [n["label"] for n in self.numbers if abs(round(n["value"], digits) - float(token)) < 1e-9]
            if not labels:
                raise RuntimeError(f"the number {token} of the caption is not in the ledger")
            found.append({"as_written": token, "n_entries": len(labels), "first_entry": labels[0]})
        return found


# ------------------------------------------------------------------------------ arrays and panels
@dataclass(frozen=True)
class Scene:
    """Everything the figure draws for the one sequence."""
    ex: qlib.Example
    recorded: np.ndarray                         # (64, 512) recorded canonical maps
    model: np.ndarray                            # (64, 512) saved maps of the model evolved from the true first map
    row: int                                     # row of the sequence in the saved predictions
    verts: np.ndarray                            # (V, 3) object vertices; rigid, the same at every frame
    faces: np.ndarray                            # (F, 3)
    hand_faces: np.ndarray                       # (1538, 3)
    drawn: dict[str, dict[str, np.ndarray]]      # event class -> before, after, change, model (V,); hand_before, hand_after (778, 3)
    contact_max: float                           # upper end of the contact colour scale (all rows)
    change_abs: float                            # limit of the change colour scale (all rows)


def load_scene(example: int, chosen: list[Event]) -> Scene:
    """Recorded maps, the model's saved maps, object and hands; per-vertex values by the page's display rule."""
    ex = qlib.load_example(DATASET, example)
    recorded = ex.C
    with np.load(PREDS_NPZ) as z:
        row = ex.test_row(z["example"])
        if not np.array_equal(z["gt"][row].astype(np.float32), recorded):
            raise RuntimeError("the ground truth stored with the saved predictions is not this sequence")
        model = z[MODEL_KEY][row].astype(np.float32)
    verts, faces = ex.mesh(0)
    if not ex.covered.all():
        log.warning("%d of %d vertices are not reached by the map (drawn in the no-data colour)", (~ex.covered).sum(), len(verts))
    drawn: dict[str, dict[str, np.ndarray]] = {}
    for e in chosen:
        before, after = ex.to_vertices(recorded[e.before]), ex.to_vertices(recorded[e.after])
        drawn[e.kind] = {"before": before, "after": after, "change": after - before, "model": ex.to_vertices(model[e.after]),
                         "hand_before": ex.hand(e.before)[0], "hand_after": ex.hand(e.after)[0]}
    contact_max = round_up(max(np.nanmax(d[k]) for d in drawn.values() for k in ("before", "after", "model")))
    change_abs = round_up(max(np.nanmax(np.abs(d["change"])) for d in drawn.values()))
    return Scene(ex, recorded, model, row, verts, faces, ex.hand(0)[1], drawn, contact_max, change_abs)


def render_rows(scene: Scene, chosen: list[Event]) -> tuple[list[list[np.ndarray]], dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """The twelve panels, the arrays drawn in them, and the camera (one for all panels)."""
    hands = [d[k] for d in scene.drawn.values() for k in ("hand_before", "hand_after")]
    direction, up = qlib.hand_side_view(scene.verts, np.concatenate(hands), tilt=0.0)
    camera = qlib.fit_camera([scene.verts] + hands, direction, up, margin=0.05, aspect=PANEL[0] / PANEL[1])

    def panel(values: np.ndarray, hand: np.ndarray | None = None, diverging: bool = False) -> np.ndarray:
        rgb = qlib.diverging_rgb(values, scene.change_abs) if diverging else qlib.contact_rgb(values, scene.contact_max)
        items = [qlib.mesh_item(scene.verts, scene.faces, rgb=rgb)]
        if hand is not None:
            items.append(qlib.hand_item(hand, scene.hand_faces, opacity=HAND_OPACITY))
        return qlib.render(items, camera, PANEL)

    rows: list[list[np.ndarray]] = []
    panels: dict[str, dict[str, np.ndarray]] = {"object": {"verts": scene.verts, "faces": scene.faces}}
    for e in chosen:
        d = scene.drawn[e.kind]
        rows.append([panel(d["before"], d["hand_before"]), panel(d["after"], d["hand_after"]),
                     panel(d["change"], diverging=True), panel(d["model"])])
        panels[f"{e.kind}_before"] = {"contact": d["before"], "hand": d["hand_before"], "map512": scene.recorded[e.before]}
        panels[f"{e.kind}_after"] = {"contact": d["after"], "hand": d["hand_after"], "map512": scene.recorded[e.after]}
        panels[f"{e.kind}_change"] = {"change": d["change"]}
        panels[f"{e.kind}_model"] = {"contact": d["model"], "map512": scene.model[e.after]}
    return rows, panels, {"direction": direction, "up": up, "camera": camera}


# ------------------------------------------------------------------------------ numbers
def measure(scene: Scene, chosen: list[Event], events: pd.DataFrame, evidence: dict[str, Any],
            ledger: Ledger) -> dict[str, dict[str, Any]]:
    """Fill the ledger: the event table's values, their recomputation from the arrays, and what is measured on the
    drawn arrays. Returns, per event class, what the caption quotes: the hand's closest distance (mm) and its number
    of near vertices at the before and the after frame, the two state distances at the after frame, the share of
    the class that starts in firm contact (percent) and the event's rank in its class."""
    ex, recorded, model = scene.ex, scene.recorded, scene.model
    zero_mass = float(json.loads(ZERO_MASS_JSON.read_text())["zero_mass"])
    size, amount, spatial = change_parts(recorded, zero_mass)
    model_error = np.linalg.norm(np.diff(model, axis=0) - np.diff(recorded, axis=0), axis=1)
    centre = scene.verts.mean(0)
    along = (scene.verts - centre) @ np.linalg.eigh(np.cov((scene.verts - centre).T))[1][:, 2]   # position along the knife, metres
    along = along * -np.sign(np.nansum(scene.drawn["mixed"]["after"] * along))                   # positive = the end without contact (blade)
    surface = cKDTree(scene.verts)
    computed = f"computed in {SCRIPT} from the drawn arrays"
    dense_source = f"{qlib.TACO_GEN3 / 'sequences'} (recorded hand-to-vertex distances, read by qlib.Example.dense)"

    n_hand = len(scene.drawn[chosen[0].kind]["hand_before"])
    ledger.add("mesh vertices of the hand model", n_hand, computed, "panels.npz */hand, first dimension")
    ledger.add("nearness threshold for counting hand vertices close to the object, mm", 1000 * NEAR, f"constant NEAR in {SCRIPT}",
               "NEAR, metres times 1000")
    ledger.add("position of the rear end of the knife along its long axis, mm (0 = mean of the mesh vertices, positive = "
               "towards the blade)", 1000 * along.min(), computed, "panels.npz object/verts, first principal axis, minimum")
    ledger.add("position of the tip of the blade along the knife's long axis, mm", 1000 * along.max(), computed,
               "panels.npz object/verts, first principal axis, maximum")

    facts: dict[str, dict[str, Any]] = {}
    for e in chosen:
        d, t, where = scene.drawn[e.kind], e.table, f"row event_id == {e.event_id}"
        span = slice(e.first, e.last + 1)
        # read from the event table
        ledger.add(f"{e.name}: first transition of the event", e.first, EVENTS_CSV, f"{where}, column start_frame")
        ledger.add(f"{e.name}: last transition of the event", e.last, EVENTS_CSV, f"{where}, column end_frame")
        ledger.add(f"{e.name}: frame drawn as before", e.before, EVENTS_CSV, f"{where}, column start_frame")
        ledger.add(f"{e.name}: frame drawn as after", e.after, EVENTS_CSV, f"{where}, column end_frame, plus 1")
        ledger.add(f"{e.name}: amount part of the change", t.amount_component, EVENTS_CSV, f"{where}, column amount_component")
        ledger.add(f"{e.name}: spatial part of the change", t.spatial_component, EVENTS_CSV, f"{where}, column spatial_component")
        ledger.add(f"{e.name}: amount share, amount / (amount + spatial)", t.amount_ratio, EVENTS_CSV, f"{where}, column amount_ratio")
        ledger.add(f"{e.name}: temporal error of holding the map still", t.e_B0_static, EVENTS_CSV, f"{where}, column e_B0_static")
        ledger.add(f"{e.name}: temporal error of the evolved model", t.e_B1_gtinit_vf, EVENTS_CSV, f"{where}, column e_B1_gtinit_vf")
        ledger.add(f"{e.name}: total contact before (sum of the 512 map values)", t.contact_mass_before, EVENTS_CSV,
                   f"{where}, column contact_mass_before")
        ledger.add(f"{e.name}: total contact after (sum of the 512 map values)", t.contact_mass_after, EVENTS_CSV,
                   f"{where}, column contact_mass_after")
        # the same values recomputed from the arrays
        ledger.check(f"{e.event_id} amount part", t.amount_component, np.nansum(amount[span]))
        ledger.check(f"{e.event_id} spatial part", t.spatial_component, np.nansum(spatial[span]))
        ledger.check(f"{e.event_id} temporal error of holding", t.e_B0_static, size[span].mean())
        ledger.check(f"{e.event_id} temporal error of the evolved model", t.e_B1_gtinit_vf, model_error[span].mean())
        ledger.check(f"{e.event_id} total contact before", t.contact_mass_before, recorded[e.before].sum())
        ledger.check(f"{e.event_id} total contact after", t.contact_mass_after, recorded[e.after].sum())
        # measured on what is drawn
        facts[e.kind] = {"closest_mm": [], "near": []}
        for tag, frame in (("before", e.before), ("after", e.after)):
            ledger.add(f"{e.name}: largest drawn contact value {tag} (frame {frame})", np.nanmax(d[tag]), computed,
                       f"panels.npz {e.kind}_{tag}/contact, maximum")
            facts[e.kind]["near"].append(ledger.add(
                f"{e.name}: hand vertices (of {len(d[f'hand_{tag}'])}) within {1000 * NEAR:.0f} mm of an object vertex, {tag} (frame {frame})",
                (surface.query(d[f"hand_{tag}"])[0] < NEAR).sum(), computed, f"panels.npz {e.kind}_{tag}/hand against object/verts"))
            facts[e.kind]["closest_mm"].append(ledger.add(
                f"{e.name}: closest distance between the hand and the object, {tag} (frame {frame}), mm",
                -1000 * qlib.SOFT_SCALE * np.log(ex.dense(frame).max()), dense_source,
                f"sequence {ex.meta.sequence_id}, take frame {ex.take_frame(frame)}, right hand, tool vertices, minimum"))
        gain, loss = d["change"] > CHANGE_MARK, d["change"] < -CHANGE_MARK
        ledger.add(f"{e.name}: most negative drawn change", np.nanmin(d["change"]), computed, f"panels.npz {e.kind}_change/change, minimum")
        ledger.add(f"{e.name}: most positive drawn change", np.nanmax(d["change"]), computed, f"panels.npz {e.kind}_change/change, maximum")
        ledger.add(f"{e.name}: share of mesh vertices that gain more than {CHANGE_MARK}", gain.mean(), computed,
                   f"panels.npz {e.kind}_change/change > {CHANGE_MARK}")
        ledger.add(f"{e.name}: share of mesh vertices that lose more than {CHANGE_MARK}", loss.mean(), computed,
                   f"panels.npz {e.kind}_change/change < -{CHANGE_MARK}")
        for word, mask in (("gain", gain), ("lose", loss)):
            if mask.any():
                ledger.add(f"{e.name}: mean position along the knife of the vertices that {word} more than {CHANGE_MARK}, mm "
                           "(0 = mean of the mesh vertices, positive = towards the blade)", 1000 * along[mask].mean(), computed,
                           f"panels.npz {e.kind}_change/change and object/verts")
                for end, value in (("rearmost", along[mask].min()), ("foremost", along[mask].max())):
                    ledger.add(f"{e.name}: {end} position along the knife of the vertices that {word} more than {CHANGE_MARK}, mm",
                               1000 * value, computed, f"panels.npz {e.kind}_change/change and object/verts")
        ledger.add(f"{e.name}: largest drawn value of the model's map at the after frame", np.nanmax(d["model"]), computed,
                   f"panels.npz {e.kind}_model/contact, maximum")
        ledger.add(f"{e.name}: total of the model's map at the after frame (sum of the 512 map values)", model[e.after].sum(),
                   PREDS_NPZ, f"{MODEL_KEY}[{scene.row}], frame {e.after}")
        state_hold = ledger.add(
            f"{e.name}: distance between the before map held still and the recorded map at the after frame, which is the size of "
            "the recorded change from before to after (norm over the 512 values)",
            np.linalg.norm(recorded[e.after] - recorded[e.before]), SEQUENCES_NPZ, f"C[{ex.meta.example}], frames {e.before} and {e.after}")
        ledger.add(f"{e.name}: size of the model's own change over the same frames (norm over the 512 values)",
                   np.linalg.norm(model[e.after] - model[e.before]), PREDS_NPZ, f"{MODEL_KEY}[{scene.row}], frames {e.before} and {e.after}")
        state_model = ledger.add(
            f"{e.name}: distance between the model's map and the recorded map at the after frame (norm over the 512 values)",
            np.linalg.norm(model[e.after] - recorded[e.after]), PREDS_NPZ, f"{MODEL_KEY}[{scene.row}] and gt[{scene.row}], frame {e.after}")
        ledger.add(f"{e.name}: distance between the model's map and the recorded map at the before frame (norm over the 512 values)",
                   np.linalg.norm(model[e.before] - recorded[e.before]), PREDS_NPZ,
                   f"{MODEL_KEY}[{scene.row}] and gt[{scene.row}], frame {e.before}")
        # where the event stands in its class
        standing = within_class(events, e)
        ledger.add(f"{e.name}: share of the class's test events whose largest transition change is not larger than this event's",
                   standing["share_with_smaller_or_equal_largest_change"], EVENTS_CSV, f"column peak_d, class {e.kind}")
        rank = ledger.add(f"{e.name}: rank of this event among the class's test events by largest transition change (1 = largest)",
                          standing["rank_by_largest_transition_change"], EVENTS_CSV, f"column peak_d, class {e.kind}")
        ledger.add(f"{e.name}: share of the class's test events that start in firm contact",
                   standing["share_starting_in_firm_contact"], EVENTS_CSV, f"column firm_before, class {e.kind}")
        firm_percent = ledger.add(f"{e.name}: share of the class's test events that start in firm contact, percent",
                                  100 * standing["share_starting_in_firm_contact"], EVENTS_CSV, f"column firm_before, class {e.kind}")
        facts[e.kind].update(state_hold=state_hold, state_model=state_model, rank=rank, firm_percent=firm_percent,
                             n_class=standing["n_events_of_class"])
    ledger.add("upper end of the contact colour scale", scene.contact_max, f"computed in {SCRIPT}",
               f"largest value drawn in the nine contact panels, rounded up to {SCALE_STEP}")
    ledger.add("limit of the change colour scale (plus and minus)", scene.change_abs, f"computed in {SCRIPT}",
               f"largest absolute change drawn in the three change panels, rounded up to {SCALE_STEP}")
    ledger.add("spike events in this sequence", (events.example == ex.meta.example).sum(), EVENTS_CSV,
               f"rows with example == {ex.meta.example}")
    ledger.add("events of this sequence that the page's timeline caption names", evidence["events"]["n_named_in_page_caption"],
               PAGE_HTML, "figcaption of the figure f1c_timeline; wordings in PAGE_NAMED")
    ledger.add("events of this sequence drawn in the figure (rows)", len(chosen), f"constant DRAWN in {SCRIPT}", "number of entries")
    for kind, n in events.event_category.value_counts().items():
        ledger.add(f"TACO test events of class '{kind}'", n, EVENTS_CSV, f"rows with event_category == {kind}")
    return facts


# ------------------------------------------------------------------------------ texts
def row_label(e: Event) -> str:
    """Seven short lines. The two errors stand under a heading that says what they score (the change, not a map)."""
    t = e.table
    return (f"{ROW_TITLE[e.kind]}\nframes {e.before} \N{RIGHTWARDS ARROW} {e.after}\n"
            f"amount part {t.amount_component:.2f}\nspatial part {t.spatial_component:.2f}\n"
            f"error of the change:\nhold {t.e_B0_static:.2f}\nmodel {t.e_B1_gtinit_vf:.2f}")


def spell(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n < len(NUMBER_WORDS) else str(n)


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def write_caption(chosen: list[Event], facts: dict[str, dict[str, Any]], n_hand_vertices: int, n_events_in_sequence: int) -> str:
    """Six sentences; every number in them is a ledger entry (checked by Ledger.trace)."""
    a, s, m = (next(e for e in chosen if e.kind == k) for k in ("amount", "spatial", "mixed"))
    state_model = [f"{facts[e.kind]['state_model']:.2f}" for e in chosen]
    state_hold = [f"{facts[e.kind]['state_hold']:.2f}" for e in chosen]
    return (
        f"One TACO test sequence, the one of the timeline chart (a right hand and a knife, from a cutting take): each row is one of "
        f"{spell(len(chosen))} of that sequence's {spell(n_events_in_sequence)} spike events (transitions "
        f"{a.first}\N{EN DASH}{a.last}, {s.first}\N{EN DASH}{s.last} and {m.first}\N{EN DASH}{m.last}), drawn at the frame before "
        f"its first transition and the frame after its last, with the hand translucent so that the contact under it stays visible. "
        f"Amount-dominant (frames {a.before} to {a.after}): the hand is near the handle without touching it (closest point "
        f"{facts['amount']['closest_mm'][0]:.0f} mm away, then {facts['amount']['closest_mm'][1]:.0f} mm) and moves away, so the total "
        f"contact (the sum of the map) falls from {a.table.contact_mass_before:.0f} to {a.table.contact_mass_after:.0f} and the map "
        f"fades over the whole handle, blue everywhere; unlike most amount-dominant test events "
        f"({facts['amount']['firm_percent']:.0f} % start in firm contact), this one happens before the grasp. "
        f"Spatial-dominant (frames {s.before} to {s.after}): the total barely changes ({s.table.contact_mass_before:.0f} to "
        f"{s.table.contact_mass_after:.0f}) while contact weakens on the rear part of the handle and grows towards the blade, blue "
        f"and orange side by side. "
        f"Mixed (frames {m.before} to {m.after}), one of the largest mixed events of the test set "
        f"({ordinal(int(facts['mixed']['rank']))} of {facts['mixed']['n_class']} by its largest single-transition change): more of "
        f"the hand "
        f"comes close to the handle ({facts['mixed']['near'][0]:.0f} then {facts['mixed']['near'][1]:.0f} of its {n_hand_vertices} "
        f"mesh vertices within {1000 * NEAR:.0f} mm), the total rises from {m.table.contact_mass_before:.0f} to "
        f"{m.table.contact_mass_after:.0f} and the pattern shifts as well. "
        f"Row labels give the amount and spatial parts of the recorded change, summed over the event's transitions, and the temporal "
        f"error (the error of the predicted frame-to-frame change) of holding the map still and of the model evolved from the true "
        f"first map, averaged over them. "
        f"These errors do not score the map in the last column, that model's map at the after frame, which also carries the model's "
        f"drift since the first frame: its distance to the recorded after map (second column) is {state_model[0]}, {state_model[1]} "
        f"and {state_model[2]} in the three rows, against {state_hold[0]}, {state_hold[1]} and {state_hold[2]} for the before map "
        f"held still."
    )


def write_caveats(chosen: list[Event], evidence: dict[str, Any], typicality: dict[str, dict[str, Any]],
                  facts: dict[str, dict[str, Any]], scene: Scene) -> list[str]:
    a, m = (next(e for e in chosen if e.kind == k) for k in ("amount", "mixed"))
    tie = evidence["sequence"]
    named = evidence["events"]
    return [
        "One sequence, three events: an illustration, not a statistic. The three are not all of the sequence's events: it has "
        f"{named['n_events_in_sequence']} spike events and the page's timeline caption names {named['n_named_in_page_caption']} of "
        "them; the named ones that are not drawn are "
        + " and ".join(f"the {n['wording']}" for n in named["named_in_page_caption"] if not n["drawn"]) + ".",
        "The maps are 512-point canonical maps drawn back on the mesh by a smoothing rule, so the handle is coloured as a whole; "
        "the figure shows where and how much contact there is, not finger outlines.",
        "'Contact' is the soft value exp(-distance / 2 cm). In the amount-dominant row the hand does not touch the knife: the "
        "event happens before the grasp and the event table marks it as not in firm contact, whereas "
        f"{typicality[a.event_id]['share_starting_in_firm_contact']:.0%} of the amount-dominant test events start in firm contact. "
        "This row is not typical of its class in that respect.",
        "The mixed event is one of the largest of its class (its largest transition change is at or above that of "
        f"{typicality[m.event_id]['share_with_smaller_or_equal_largest_change']:.0%} of the mixed test events) and the event table "
        "flags it as partly short-lived. The spatial-dominant event is the report's median-sized representative; "
        f"{tie['n_tied_at_the_smallest_distance']} events are equally near that median ({', '.join(tie['tied_events'])}) and the "
        "report's code returns this one.",
        "Amount and spatial are two parts of the map change, not two kinds of hand motion. The amount-dominant event also has a "
        "spatial part; the classes are set by the share amount / (amount + spatial).",
        "The last column is a state, while the errors in the row labels are about changes: the model's map at the after frame also "
        "carries what the model drifted since the first frame, and the column cannot show whether the model followed the change. "
        "Over each event's frames the model's own map changes much less than the recorded one (see numbers).",
        "In the amount-dominant row the labels and the picture point opposite ways, and both are right: the model's error of the "
        f"change ({a.table.e_B1_gtinit_vf:.2f}) is below that of holding ({a.table.e_B0_static:.2f}), yet its map at frame {a.after} "
        f"is further from the recorded map ({facts['amount']['state_model']:.2f}) than the before map held still "
        f"({facts['amount']['state_hold']:.2f}); its total is {float(scene.model[a.after].sum()):.1f} against the recorded "
        f"{a.table.contact_mass_after:.1f}. In the spatial-dominant row the state distances are "
        f"{facts['spatial']['state_model']:.2f} (model) and {facts['spatial']['state_hold']:.2f} (before map held still), in the "
        f"mixed row {facts['mixed']['state_model']:.2f} and {facts['mixed']['state_hold']:.2f}.",
        "Before and after are two to four frames apart, so the hand moves little between the first two columns; the change column "
        "carries the message.",
        "One camera has to hold the knife and the hand of all six frames, so the knife is small in its panel; the two "
        "object-only columns keep that camera so that all twelve panels stay comparable.",
    ]


# ------------------------------------------------------------------------------ figure
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    qlib.require_headless()

    events = pd.read_csv(EVENTS_CSV)
    example = page_timeline_example(DATASET)
    chosen, evidence = select(events, example)
    scene = load_scene(example, chosen)
    rows, panels, view = render_rows(scene, chosen)
    ledger = Ledger()
    facts = measure(scene, chosen, events, evidence, ledger)
    typicality = {e.event_id: within_class(events, e) for e in chosen}
    for e in chosen:
        t = e.table
        log.info("%s (%s, transitions %d-%d, frames %d -> %d): amount part %.2f, spatial part %.2f; error of the change: hold %.2f, "
                 "model %.2f; distance to the recorded after map: before map held still %.2f, model's map %.2f",
                 ROW_TITLE[e.kind], e.event_id, e.first, e.last, e.before, e.after, t.amount_component, t.spatial_component,
                 t.e_B0_static, t.e_B1_gtinit_vf, facts[e.kind]["state_hold"], facts[e.kind]["state_model"])

    image = qlib.grid(rows, row_labels=[row_label(e) for e in chosen], col_labels=list(COL_LABELS), colorbars=[
        qlib.colorbar("contact", 0.0, scene.contact_max, "Contact (0 = far from the hand, 1 = touching)"),
        qlib.colorbar("diverging", -scene.change_abs, scene.change_abs, "Change in contact: blue = less, orange = more")],
        footnote=FOOTNOTE)

    info = scene.ex.meta
    named = evidence["events"]
    by_kind = {e.kind: e for e in chosen}
    caption = write_caption(chosen, facts, len(scene.drawn["mixed"]["hand_after"]), named["n_events_in_sequence"])
    caption_numbers = ledger.trace(caption)
    meta = {
        "id": FIG_ID, "block": BLOCK,
        "title": "What an amount-dominant, a spatial-dominant and a mixed contact change look like (one TACO sequence)",
        "panels": {
            "rows": {ROW_TITLE[e.kind]: f"event {e.event_id}, transitions {e.first} to {e.last} of the sequence, drawn at frame "
                                        f"{e.before} (before) and frame {e.after} (after)" for e in chosen},
            "columns": {
                "Before": "the knife with the recorded contact map of the before frame, and the recorded right hand (translucent)",
                "After": "the same at the after frame",
                "Recorded change": "the after map minus the before map, per mesh vertex; object only",
                "Evolved model": "the saved map of the model evolved from the true first map, at the after frame; object only "
                                 "(the model outputs a map, not a hand)"},
            "row_label_numbers": "amount part and spatial part: the two parts of the recorded map change, SUMMED over the event's "
                                 "transitions. Error of the change, hold and model: the temporal error (norm of predicted change "
                                 "minus recorded change, per transition) of keeping the map unchanged and of the evolved model, "
                                 "AVERAGED over the event's transitions. All four are read from the event table. The two errors "
                                 "score the change, not the map drawn in the last column.",
            "footnote": FOOTNOTE,
        },
        "examples": [{"dataset": "TACO", "example": example, "sequence_id": info.sequence_id,
                      "object": f"{info.category} (mesh {info.mesh_id}, role {info.role})",
                      "hand": {"R": "right", "L": "left"}[info.hand], "split": info.split, "row_in_saved_predictions": scene.row,
                      "frames_drawn": [f for e in chosen for f in (e.before, e.after)], "t0": info.t0,
                      "events": [{"event_id": e.event_id, "class": e.kind, "transitions": [e.first, e.last], "before_frame": e.before,
                                  "after_frame": e.after, "take_frames": [scene.ex.take_frame(e.before), scene.ex.take_frame(e.after)]}
                                 for e in chosen]}],
        "selection_rule": "No new choice of sequence: it is the documented TACO example of the page's timeline chart (TIMELINE_EXAMPLES "
                          f"in docs/build/figs/f1c_temporal_events.py). The rows are three of the {named['n_named_in_page_caption']} "
                          f"events of that sequence the chart's caption names ({named['n_events_in_sequence']} spike events in the "
                          "sequence), fixed by their transitions: the only amount-dominant one "
                          f"({by_kind['amount'].first}-{by_kind['amount'].last}), the spatial-dominant one that is the report's "
                          f"representative event ({by_kind['spatial'].event_id}, {by_kind['spatial'].first}-{by_kind['spatial'].last}) "
                          "and the one the caption calls the large mixed event "
                          f"({by_kind['mixed'].first}-{by_kind['mixed'].last}); the caption's "
                          + " and ".join(n["wording"] for n in named["named_in_page_caption"] if not n["drawn"])
                          + " are not drawn. Before = the frame at which the event's first transition starts, after = the frame at "
                            "which its last transition ends (frames "
                          + ", ".join(f"{e.before} and {e.after}" for e in chosen) + ").",
        "selection_evidence": evidence,
        "display_choices": {
            "canonical_to_surface": qlib.DISPLAY_RULE + " (qlib.Example.to_vertices); used for the recorded maps, their difference "
                                                        "and the model's map",
            "contact_scale": {"range": [0.0, scene.contact_max], "colours": "grey (0) through orange (half) to dark red-brown (maximum)",
                              "rule": f"largest value drawn in the nine contact panels, rounded up to {SCALE_STEP}; one scale for all rows"},
            "change_scale": {"range": [-scene.change_abs, scene.change_abs], "colours": "blue (less), grey (no change), orange (more)",
                             "rule": f"largest absolute change drawn in the three change panels, rounded up to {SCALE_STEP}; "
                                     "one scale for all rows"},
            "camera": {"rule": "one camera for all twelve panels: qlib.hand_side_view(object, the six drawn hands, tilt 0), i.e. the "
                               "knife's broad side seen from the side the hand is on, long axis horizontal; fitted to the object and "
                               "the six drawn hands with a 5 % margin", **view},
            "hand": f"recorded MANO mesh of the contacting (right) hand, skin colour, opacity {HAND_OPACITY}; drawn in the before and "
                    "after columns only",
            "panel_pixels": list(PANEL),
        },
        "numbers": ledger.numbers,
        "cross_checks": ledger.checks,
        "typicality": typicality,
        "inference": {"summary": "none", "checkpoints_loaded": [], "device": "cpu", "seed": None,
                      "read_from_saved_predictions": f"{PREDS_NPZ} :: {MODEL_KEY}[{scene.row}], the study's saved rollout from the true "
                                                     f"first map (made by its evaluation with {MODEL_CHECKPOINT}, which is not loaded here)",
                      "newly_computed": "only the MANO hand meshes, from the recorded TACO hand poses (a forward pass of the hand "
                                        "model on CPU; no learned model of this project is run)"},
        "caveats": write_caveats(chosen, evidence, typicality, facts, scene),
        "suggested_caption": caption,
        "caption_numbers_traced_to_the_ledger": caption_numbers,
        "suggested_alt": "A knife and a translucent right hand before and after three contact changes of one sequence, each with a "
                         "blue and orange map of the change on the handle and the evolved model's contact map.",
    }
    paths = qlib.save(FIG_ID, image, panels, meta)
    for name, path in paths.items():
        log.info("%s: %s", name, path)
    log.info("layout: %s", image.info["qlib_layout"])
    log.info("cross-checks: %d of %d agree", sum(c["agree"] for c in ledger.checks), len(ledger.checks))


if __name__ == "__main__":
    main()
