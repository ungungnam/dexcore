"""Public chart spec: every chart of the public page is DERIVED from one figure of the internal page.

One entry per public chart. An entry says which internal figure it comes from (docs/data/figures.json),
which panels, series, categories and reference lines are kept, and what they are called in public.
build_public.py copies the plotted values from the internal figure; this file holds no plotted number
(it is rejected if it contains a numeric value), and every digit that occurs in a public string must
also occur in the strings of the internal figure it is derived from.

Entry fields
  source       id of the internal figure
  title        public title (the visible heading of the chart, and part of its accessible name)
  panels       internal panel titles to keep, in order (the dataset names; they stay as they are)
  series       [(internal series name, public name), ...] in the order they are drawn
  categories   [(internal category, public name), ...] in axis order; either one list for all panels
               or {panel title: list}
  x_label      public x-axis label (shown in the data table and in tooltips)
  y_label      public y-axis label
  direction    "lower_better" | "higher_better" | "none"; must equal the internal figure's direction
  refs         reference lines to keep: [(internal label, public label), ...] or {panel title: list};
               internal reference lines that are not listed are dropped
  highlight    None, or {"category": internal category, "label": public label}; the internal figure
               must highlight that same category
  note         public note under the data table: what is plotted, units, error bars, sample sizes.
               Plain words only: no internal code, no file, no path.
  attributes   documentation only: the attributes recommended for <figure class="viz" data-fig="...">

Strings of this file are shipped (title, labels, names, note). They are scanned by check_public.py.
"""

SOURCE_FIGURES = "docs/data/figures.json"

DATASETS_2 = ["TACO", "ARCTIC"]
DATASETS_3 = ["TACO", "ARCTIC", "OakInk2"]

# The ladder of contact descriptions of Observation 3, from the coarsest to the dense state.
# The levels are ordered by size, not nested: '+ spread & patch count' and 'compact + coarse hand' each add
# one block to the compact state ('+ position & direction'), 'compact + both' adds the two, and the two dense
# levels describe the same frame differently. The hand block of the fifth and sixth level is a COARSE hand
# pose; the last level adds hand surface points to the dense map (another hand description).
LADDER = [
    ("R0", "parts"),
    ("R1", "+ amount"),
    ("R2", "+ position & direction"),
    ("R3", "+ spread & patch count"),
    ("R4", "compact + coarse hand"),
    ("R5", "compact + both"),
    ("CONTACT_FULL", "dense map"),
    ("FULL", "dense map + hand surface points"),
]
LADDER_X_LABEL = "contact description of a frame, ordered by size"
LADDER_HIGHLIGHT = {"category": "R2", "label": "compact state: 48 numbers per frame"}
LADDER_LEVELS_NOTE = (
    "Levels, in axis order: which hand parts touch; plus how much contact each part makes; plus where on "
    "the object each part touches and which way that surface faces (the compact state, highlighted); the "
    "compact state plus the spread and the number of contact patches of each part; the compact "
    "state plus a coarse hand pose; the compact state plus both; the dense contact map; the dense contact "
    "map plus hand surface points. The levels are ordered by size, not nested, and the two dense levels "
    "describe the same frame differently: they do not add to the compact ones. "
)

# Three charts. The Motivation and the hold-versus-evolve comparison of Observation 1 are told in prose (their
# numbers are claims of build/public_claims.json), so no chart data is shipped for them.
FIGURES = {
    # ---------------------------------------------------------------- Observation 1
    "pub_obs1_variation": {
        "source": "f1a_time_vs_identity_ratios",
        "title": "Contact variation across time and across object or subject identity, relative to two takes",
        "panels": DATASETS_3,
        "series": [
            ("macro mean of group ratios", "ratio to the distance between two takes"),
        ],
        "categories": {
            "TACO": [
                ("amount: time / take", "amount over time"),
                ("pattern: time / take", "pattern over time"),
                ("pattern: mesh / take", "pattern across object meshes"),
            ],
            "ARCTIC": [
                ("amount: time / take", "amount over time"),
                ("pattern: time / take", "pattern over time"),
                ("pattern: subject / same-subject take", "pattern across subjects (vs two takes of one subject)"),
            ],
            "OakInk2": [
                ("amount: time / take", "amount over time"),
                ("pattern: time / take", "pattern over time"),
                ("pattern: object instance / take", "pattern across object instances"),
            ],
        },
        "x_label": "what is compared",
        "y_label": "contact variation, relative to two takes of the same object and action",
        "direction": "none",
        # One label for the three panels (one legend entry). The ARCTIC subjects bar says on its own axis
        # label that it is relative to two takes of one subject; the note gives its value against the other level.
        "refs": {
            "TACO": [("cross-take level", "two takes, same object and action")],
            "ARCTIC": [("cross-take level (same-subject takes for the subject bars)", "two takes, same object and action")],
            "OakInk2": [("cross-take level", "two takes, same object and action")],
        },
        "highlight": None,
        "note": (
            "Each bar is a ratio of two mean distances between pairs of contact maps. Amount: the absolute "
            "difference of the summed contact values. Pattern: the L2 distance between maps normalised to "
            "sum 1. The numerator compares early and late frames of one take (across time), or "
            "frames of another object mesh (TACO), subject (ARCTIC) or object instance (OakInk2) at the same "
            "phase of the motion. The denominator compares two takes of the same object and action at the "
            "same phase, so 1 means as different as two takes. No direction is better. Bars are means of "
            "per-group ratios; groups per bar: TACO 8, 8, 8; ARCTIC 22, 22, 22; OakInk2 39, 39, 17. There are "
            "no error bars: intervals exist per group only. Amount and pattern are different measures, each "
            "divided by its own take-to-take level. On ARCTIC the third bar is relative to two takes of the "
            "same subject, which is not the denominator of its first two bars; against that denominator it "
            "is 1.011. On TACO an object mesh is confounded with the people who used it. What a take contains "
            "differs: TACO takes include approach and release, ARCTIC takes are long and hold several "
            "episodes, OakInk2 segments are interaction only."
        ),
        "attributes": 'data-height="200" data-shared-y',
    },
    # ---------------------------------------------------------------- Observation 2
    "pub_obs2_capability_by_class": {
        # The counterpoint of Observation 2. Only the gain series and three kinds of event are kept: a release
        # loses all modelled capability (an empty bar in a gain-only chart), an onset gains from no contact by
        # construction, and the short-lived kind is a diagnostic subsample.
        "source": "f1d_share_capability_change_by_event_class",
        "title": "How often the modelled wrench capability rises by more than 25%, by kind of contact event",
        "panels": DATASETS_2,
        "series": [
            ("Gain above 25 %", "events that gain more than 25% capability"),
        ],
        # The number of test events of each kind is part of its axis label: equal bar widths would
        # otherwise make the three kinds look equally common. These six numbers are typed here, so each
        # has a supporting value in build/public_claims.json (claim obs2-counterpoint, "where": "chart"):
        # the build compares it with the internal evidence and checks the panel it stands in.
        "categories": {
            "TACO": [
                ("Persistent spatial", "contact moves (119)"),
                ("Persistent mixed", "moves, amount changes (435)"),
                ("Amount-dominant", "mainly amount changes (72)"),
            ],
            "ARCTIC": [
                ("Persistent spatial", "contact moves (26)"),
                ("Persistent mixed", "moves, amount changes (118)"),
                ("Amount-dominant", "mainly amount changes (40)"),
            ],
        },
        "x_label": "kind of contact event (number of test events)",
        "y_label": "share of events that gain more than 25% modelled capability",
        "direction": "none",
        "refs": [],
        "highlight": {"category": "Persistent spatial", "label": "reconfiguration"},
        "note": (
            "Share of test events whose modelled wrench capability (the mean capacity of a grasp model over "
            "76 force and torque directions) gains more than 25%, for three kinds of contact event. Contact "
            "moves: a persistent re-arrangement of where the hand touches at a roughly constant contact "
            "amount, inside a maintained grasp (the reconfigurations of this observation). Moves, amount "
            "changes: persistent events that also change the contact amount. Mainly amount changes: events "
            "that are mostly a change of the contact amount. Onsets, releases and short-lived changes are "
            "not shown. Events per kind, in axis order: TACO 119, 435, 72; ARCTIC 26, 118, 40. There are no "
            "error bars: no interval was computed for these shares, so this is a descriptive comparison."
        ),
        "attributes": 'data-height="150" data-format="pct" data-shared-y',
    },
    # ---------------------------------------------------------------- Observation 3
    "pub_obs3_future_ladder": {
        "source": "f2b_future_structure_gap_by_rung",
        "title": "Predicting the future structured contact from each contact description",
        "panels": DATASETS_2,
        "series": [
            ("T1–T4 composite", "error of predicting the future structured contact"),
        ],
        "categories": LADDER,
        "x_label": LADDER_X_LABEL,
        "y_label": "future-structure error relative to the full dense state, the last level (ratio − 1)",
        "direction": "lower_better",
        "refs": [],
        "highlight": LADDER_HIGHLIGHT,
        "note": (
            "Error of predicting the structured contact state 1, 4 and 8 frames ahead from the last 8 frames "
            "of each description, relative to the same prediction from the full dense state (the last "
            "level: the dense contact map plus hand surface points, 0 by definition). Plotted is the mean "
            "error ratio over the predicted quantities (which parts touch, contact amount, position and "
            "direction, wrench profile) and over the three horizons, minus 1: negative values are lower "
            "errors, and lower is better. "
            + LADDER_LEVELS_NOTE +
            "Means of 3 fits; test takes: TACO 186, ARCTIC 44; error bars are 95% paired bootstrap "
            "intervals over the test takes. All levels go through one fixed small prediction model, so the "
            "dense inputs doing worse is a statement about that model, not about the information in the "
            "dense state. Every model also receives an object descriptor and the local object trajectory, "
            "including the future states of the object. A comparison between inputs, not an absolute "
            "forecasting result."
        ),
        "attributes": 'data-height="190" data-format="pct" data-labels="none" data-shared-y',
    },
}

# Images of the public page: public file name (under site/assets/img/) -> internal file (under docs/assets/img/).
# They are byte copies of the internal renders; check_public.py holds the hash of each inspected version.
IMAGES = {
    "contact_trajectory.webp": "qual_contact_map_primer.webp",
    "reconfiguration_events.webp": "qual_reconfiguration_with_hand.webp",
    "compact_state.webp": "qual_structure_on_a_grasp.webp",
}
