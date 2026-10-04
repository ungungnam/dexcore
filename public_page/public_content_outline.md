# Public research page: content outline

The public page (`public_page/site/index.html`) tells one story and nothing else:

> Motivation → Observation 1 (contact variation has structure) → Observation 2 (dense contact change
> is not always functional change) → Observation 3 (a compact structural state appears to exist) →
> Current research direction (toward structured contact dynamics).

It is labelled "Ongoing research · Work in progress" and presents observations, an emerging
hypothesis and a research direction. It presents no method and no result of a method. Every number
on it is already on the internal page (`docs/`); the mapping is in `public_evidence_manifest.json`.

Length (rules H01 and H02 of `build/check_public.py`; the limits were set by the orchestrating session after
the audit and replace the 1,300 and 80 words of the build plan; the owner's confirmation is still due):

- Body text: 1,200 words (limit 1,200). The count separates every inline element, so each dataset
  chip, source label, stat tile and table cell counts as its own words, and the symbols of the
  formula count one by one; the text as a browser renders it (a symbol with its subscript is one
  word) has 1,190 words.
- Of the 1,200 words, 1,055 are running prose (headings, paragraphs, list items, captions) and 145
  are navigation, chips, source labels, stat tiles, table cells and diagram labels.
- Four collapsed technical notes of 72, 98, 80 and 97 words (limit 100 each).
- Nine figures: three pictures, three charts, three conceptual diagrams.

Reading time: about one minute by scrolling (headlines, figures, interpretations). A careful read,
estimated at 230 words per minute plus ten seconds per figure (9 × 10 s = 1.5 minutes):

- every word counted: 1,200 / 230 = 5.2 minutes, plus 1.5 = about 6.7 minutes;
- running prose only: 1,055 / 230 = 4.6 minutes, plus 1.5 = about 6.1 minutes.

The brief asks for approximately four to six minutes. The prose-only estimate is just above the
upper end of that range (6.1 against 6); with every chip, label, tile and table cell read as a
word the estimate is about three quarters of a minute above it. The collapsed technical notes are
in neither estimate.

## The four take-aways (hero, "In one minute")

A reader should leave with exactly these four ideas (owner's brief, section 18). Each line of the
hero links to its section.

1. Choosing a grasp realization and evolving it through time are different sources of contact
   variation: not the same problem. → Observation 1
2. Not every dense contact reconfiguration meaningfully changes the modelled wrench capability.
   → Observation 2
3. A compact structural state keeps much of what matters for modelled wrench capability and future
   structured contact. → Observation 3
4. We are investigating whether contact generation should be modelled as structured contact
   dynamics, not unrestricted dense trajectory generation. → Current research direction

## Pattern of every observation

Kicker ("Observation N") → the question → one headline → one main figure → at most two compact
cards → one collapsed technical note → one interpretation → one bridge sentence to the next section.
Every figure and card carries dataset chips. A plain-words source label stands under the main
figure of each observation, under the paragraph of the Motivation that gives the hand errors and
under the card of Observation 1, which tells another analysis (five labels in all).
Observation 3 has a concept picture before its main figure, because its concept has to be
explained before the evidence (brief, section 7).

## Sections

### Top bar and hero

| Element | Content |
|---|---|
| Top bar | "Contact dynamics", status "Ongoing research", anchors Motivation, Observations 1 2 3, Direction, Datasets |
| Badge | Ongoing research · Work in progress |
| Title | Understanding Structure in Human–Object Contact Dynamics |
| Subtitle | Studying how dense human–object contact trajectories decompose into grasp-level structure, temporal evolution, and realization variation. |
| Note | Observations from ongoing work, not a finished method or published result. |
| Diagram (inline SVG, conceptual) | Object motion → **Contact dynamics** (emphasised, "studied here") → Hand interaction; no further label |
| In one minute | the four take-aways above |

### Motivation: why contact?

- **Lead.** One way to generate hand motion goes through contact (object trajectory → contact →
  hand motion).
- **Figure** (image `assets/img/contact_trajectory.webp`, TACO): what a contact trajectory is:
  the contact map as 512 stored values and, at six of its 64 frames, on the object.
- **Text** (prose at reading width, no chart; chip TACO and the label "Source: two-stage generator
  study" under the first sentence). In the two-stage generator we studied (BimArt, CVPR 2025; our
  port to TACO), replacing the recorded contact with the contact stage's prediction raises the
  mean hand error on all four TACO test splits: 37.1 → 58.6, 48.8 → 86.5, 63.4 → 104.9 and
  50.9 → 79.8 mm. These results suggest that contact-stage errors can substantially affect
  downstream hand generation. No percentage share is quoted.
- **Technical note.** One model family; one training run of the TACO port; split 1 used for model
  selection; the recorded-contact input is an oracle and still leaves 37 to 63 mm; no share, no
  interval.
- **Ends with the question.** What information does a temporal contact generator actually need to
  model?
- Datasets: TACO (four test splits).

### Observation 1: contact variation has structure

- **Question.** What varies across a human contact trajectory?
- **Headline.** Choosing a grasp realization and evolving it through time are different sources of
  variation. (No grasp labels are used: "grasp realization" means which take, object or subject a
  frame comes from, and which first contact map a sequence starts from; the page says it in one
  short sentence under the headline.)
- **Main figure** (chart `pub_obs1_variation`; TACO, ARCTIC, OakInk2; one y axis): contact
  variation relative to two takes of the same object and action. Amount over time 2.12 / 1.10 /
  0.60; pattern over time 1.04 / 0.94 / 0.45; pattern across meshes 1.48 (TACO), across subjects
  1.16 (ARCTIC, relative to two takes of the same subject), across object instances 1.69 (OakInk2).
- **Text.** Within a take (one recording), the spatial pattern changes about as much as between
  two takes on TACO, slightly less on ARCTIC and clearly less on OakInk2, and less than across
  object meshes or instances. The contact amount differs about twice as much within a TACO take,
  which includes approach and release, as between takes; not within ARCTIC's long multi-episode
  takes (1.10) or OakInk2's interaction-only segments (0.60). The pattern does change over time:
  the statement is relative.
- **One card** (text, no chart; chips TACO, ARCTIC, OakInk2; label "Source: hold-versus-evolve
  comparison"): "This motivates a hierarchical view", with the conceptual factorization
  p(C_1:T | G, τ) ≈ p(s_0 | G) · p(C_1:T | s_0, G, τ): choose an initial contact realization
  (first contact map s_0), then model its evolution. On TACO, a learned model that evolves the
  true first map clearly beats holding it (error 3.48 → 1.82); on ARCTIC and OakInk2 the
  difference is not distinguishable from zero, although contact still changes there.
- **Technical note.** Means over groups; the ARCTIC subjects bar (1.01 against the level of its
  other bars); OakInk2's third bar covers fewer than half of its groups; diagnostic conditions, not
  generators; the two other pairs with their paired differences and 95% intervals (ARCTIC
  2.21 → 2.14, −0.07, −0.26 to +0.11; OakInk2 1.26 → 1.24, −0.02, −0.11 to +0.06); one model
  family, one training run; error units differ between datasets.
- **Interpretation.** Which contact a sequence starts from and how it evolves are different
  problems; how much the second adds differs with the interaction regime.
- **Bridge.** The evolution itself is not all of one kind.
- Datasets: TACO, ARCTIC, OakInk2 (never pooled).

### Observation 2: dense contact change is not always functional change

- **Question.** Does every visible contact reconfiguration change what the grasp can do?
- **Headline.** Large contact change does not necessarily mean functional change.
- **Lead.** The modelled wrench capability (a friction-cone grasp model along 76 directions) is a
  proxy for functional grasp capability; retention is the share of the later grasp's capability
  that the earlier one already had.
- **Main figure** (image `assets/img/reconfiguration_events.webp`; TACO, ARCTIC): four recorded
  events, before and after, with the capability profiles. Single examples chosen by a stated rule,
  typical in retention only: on the knife the hand moves 4 mm and a part joins (capability +18%;
  class median +4%); on the notebook it moves 29 mm (+14%; +2%).
- **Card 1** (stat tiles): within-grasp spatial reconfiguration keeps most modelled capability:
  median retention 0.85 (TACO, 119 events) and 0.89 (ARCTIC, 26 events; 95% interval 0.76 to
  0.96); median cosine of the capability profiles 0.97 and 0.96. Dataset difference: on TACO the
  size of the contact change says little about the capability change (rank correlation 0.19); on
  ARCTIC (0.55) 5 of the 11 larger changes gain more than 25%.
- **Card 2, the counterpoint** (chart `pub_obs2_capability_by_class`, the number of events of each
  kind on its axis): events that also change the contact amount gain more than 25% modelled
  capability more often (46% on both datasets, against 20% and 23%; mainly-amount events 61.1%
  and 47.5%). Among spatial reconfigurations such gains occur almost only when a hand part joins,
  which adds capability by construction in the model.
- **Technical note.** Simplified single-hand grasp model; medians over small classes of events;
  what "larger changes" means; retention is one-sided and the cosine ignores overall scale; a
  stricter measure gives 0.66 and 0.75.
- **Interpretation.** Dense contact trajectories mix structural change with realization variation
  that, in the typical event, leaves the modelled wrench capability nearly unchanged.
- **Bridge.** Can the structural part be isolated?
- Datasets: TACO, ARCTIC.

### Observation 3: a compact structural state appears to exist

- **Question.** Can the functionally meaningful part of dense contact be represented more
  compactly?
- **Headline.** A compact description keeps much of what matters: which hand parts touch, how much,
  roughly where, facing which way.
- **Concept picture** (image `assets/img/compact_state.webp`; TACO, ARCTIC): one recorded frame per
  dataset reduced to the compact state (48 numbers per frame).
- **Lead.** Each frame is described at eight levels; fixed small models recover its modelled
  wrench profile (itself computed from contact positions and directions) and predict the compact
  quantities and that profile 1, 4 and 8 frames ahead. A ranking of inputs, not absolute
  prediction.
- **Main figure** (chart `pub_obs3_future_ladder`, eight levels of description ordered by size, the
  compact level highlighted): the error of predicting future structured contact falls over the
  first three levels, to 18% (TACO) and 32% (ARCTIC) below the full dense state; richer compact
  levels change little.
- **Sentence** (no chart): wrench-profile error (relative L1): 0.35 → 0.18 (TACO), 0.51 → 0.18
  (ARCTIC) with coarse position and direction.
- **Card 1** (stat tiles): what richer compact levels add. Spread and patch count: wrench
  error −10% (TACO), −7% (ARCTIC). Coarse hand pose: future-structure error −2% on TACO (averaged
  over horizons), no reduction on ARCTIC.
- **Card 2** (text): the remaining dense detail is real. From the compact state alone a small
  fitted model leaves 11% (TACO) and 26% (ARCTIC) of the contact-map variance unexplained; that
  remainder persists from frame to frame and the tested predictors capture little of its change.
- **Technical note** (the only place where the internal level name appears): comparisons between
  inputs under fixed small models; nothing between 12 and 48 numbers was tried; not statements of
  absolute predictability; dense inputs score worse under these fixed models, which is no
  statement about the information they hold; the residual is what one reconstruction model leaves.
- **Interpretation, with the diagram** (inline SVG, conceptual): dense contact reads as a
  structured contact state plus realization detail.
- **Bridge.** How should such a structured state evolve over time?
- Datasets: TACO, ARCTIC.

### Current research direction: toward structured contact dynamics

- **Question.** Can human contact dynamics be modelled as the evolution of a compact structural
  state, while dense contact detail is treated as realization variation?
- **Diagram** (inline SVG, conceptual): initial contact realization → structured contact state →
  temporal structural evolution → dense contact trajectory → hand motion or retargeting.
- **Text.** We are currently investigating how such structural states should evolve over time and
  how realization detail should be preserved. This is a direction: no method and no result is
  claimed.
- Nothing else: no architecture, no training recipe, no experiment design, no status of running
  work. The text of this section is pinned by the manual check `direction-wording`.

### Datasets and scope, credits, contact

| Part | TACO | ARCTIC | OakInk2 |
|---|---|---|---|
| Motivation | used (its four test splits) | – | – |
| Observation 1 | used | used | used |
| Observations 2 and 3 | used | used | – |

- TACO shows the stronger effect of temporal evolution; on ARCTIC the predictors we tested rarely
  beat carrying the current contact forward, although contact does change there. We read this as
  contact dynamics depending on the interaction regime (the test sets also differ in
  composition), not as a failure of either dataset.
- Credits: TACO, ARCTIC, OakInk2, BimArt, MANO. Only rendered images are shown; no data is
  redistributed.
- Contact: the e-mail address and the GitHub profile decided by the owner. No name, no affiliation.
- Footer: Last updated October 2026.

## Figures and assets

| Asset | Kind | Derived from (internal) | Section |
|---|---|---|---|
| hero pipeline | inline SVG, conceptual | – | Hero |
| `assets/img/contact_trajectory.webp` | image, byte copy | internal render of the contact-map primer | Motivation |
| `pub_obs1_variation` | chart | time versus identity ratios | Observation 1 |
| `assets/img/reconfiguration_events.webp` | image, byte copy | internal render of four reconfiguration events | Observation 2 |
| `pub_obs2_capability_by_class` | chart | share of capability change by event class: gain series, three classes | Observation 2 |
| `assets/img/compact_state.webp` | image, byte copy | internal render of the compact state on a grasp | Observation 3 |
| `pub_obs3_future_ladder` | chart | future-structure gap by level | Observation 3 |
| dense = structured + detail | inline SVG, conceptual | – | Observation 3 |
| direction chain | inline SVG, conceptual | – | Direction |

Nine figures: three pictures, three charts, three conceptual diagrams. The hand errors of the
Motivation and the hold-versus-evolve comparison of Observation 1 have no chart: they are claims
with their numbers in `build/public_claims.json`.

The internal figure id, evidence block and source files of every chart, and the internal render,
evidence block and metadata file of every image, are in `public_evidence_manifest.json` (sections
`figures` and `images`). Every claim of the page is in its section `claims`, with its displayed
numbers, its supporting values (for a claim that shows no digit) and its evidence note.

## What is deliberately not on the page

The internal chronology of experiments; model, case and study codes; the learned latent state and
its realization code; supervision, losses and weights; architecture sizes; seeds, steps and
checkpoints; diagnostics and negative results that only motivated implementation changes; the
current experiment and any next-method plan; file paths, logs and hashes; links to the internal
page, its evidence manifest or the code repository. `build/check_public.py` enforces this list on
every build (see `exposure_checklist.md`); what a script cannot recognise (the same things said in
other words) is a recorded manual check (`build/manual_checks.json`).

## Where the page deviates from the build plan

The evidence packs and the audits corrected the plan in several places; the page follows them, and
where the plan and the owner's brief disagree the brief wins.

- "often goes through contact" → "one way ... goes through contact" (no claim about the literature).
- Motivation: the check on eight ARCTIC test windows (two sequences of one object, an illustration
  only) is no longer on the page. It was cut for length (brief, section 12); the Motivation rests
  on the four TACO test splits, and the scope table says so.
- Observation 1: the ARCTIC subjects bar is labelled as relative to two takes of one subject; "the
  amount is what changes" and "depends on" were replaced by descriptive wording that says what a
  take contains in each dataset; "simple deterministic model" became "a learned model that evolves
  the true first map"; the hold-versus-evolve sentence says that contact still changes where the
  difference is not distinguishable from zero (the null result is not absence of change).
- "One training seed" became "one training run" (the brief removes training seeds, section 9).
- Motivation and Observation 1: the plan's two charts (hand error; hold versus evolve) are not on
  the page; their numbers are told in prose (see "Tightening after the audit").
- Footer: "Last updated October 2026." (the plan's "Ongoing research." stands in the top bar and
  in the badge of the hero).
- Observation 2: the counterpoint chart keeps the gain series and three kinds of event (onset,
  release and short-lived changes are left out); the caption of the picture gives the two
  "typical" events' own capability changes next to the class medians, because they are typical in
  retention only; the dataset difference (rank correlations, and the ARCTIC larger changes) and a
  technical note were added.
- Observation 3: one chart instead of two (brief, section 13: one main figure). The wrench-profile
  result is a sentence, because the profile is computed from contact positions and directions, the
  quantities the compact level adds; the chart of the reproduced dense variance became one
  sentence with the two contact-map remainders (11% and 26%; the hand-point remainders are not
  shown). Level names "+ spread & patch count", "compact + coarse hand", "compact + both", "dense
  map + hand surface points"; "flat" became "richer compact levels change little"; card 1 shows
  the 10% and 7% further reduction instead of "richer descriptions add little"; the technical note
  avoids the word "decoder".
- A source label stands under the main figure of each observation, under the hand-error sentence
  of the Motivation and under the card of Observation 1, not under every figure and card (length).
- Numbers that the charts print as value labels are not repeated in the prose (word budget); they
  have claim entries marked `"where": "chart"`.

## Tightening after the audit (5 October 2026)

The re-verification found that the word count joined adjacent inline elements and that a careful
read took about 7.5 minutes. The orchestrating session decided the cuts (the owner's confirmation is still due); the page follows them.

Decided cuts
- Counting: every inline element is separated by a space; limits 1,200 words (body) and 100 words
  (each technical note).
- Motivation: the hand-error chart is gone; the four pairs of values are in the sentence, with the
  chip TACO and the source label under it; the closing question stands below the prose.
- Observation 1: the card "Does evolving the first contact help?" and its chart are gone. One card
  remains, with the formula and the hold-versus-evolve result in words; the two pairs that are not
  distinguishable from zero are in the technical note with their paired differences and intervals.
  "(one recording)" moved to the first mention of a take.
- Observation 2: the sentence "We compare it before and after each contact event." is gone; the
  picture caption therefore names the kind of event itself ("the median-retention within-grasp
  spatial reconfiguration").
- Observation 3: the statement about the dense inputs moved from the caption of the chart into the
  technical note; the wrench sentence names its measure (relative L1).
- Datasets: "TACO shows the stronger effect of temporal evolution" replaces the sentence about the
  learned temporal evolution (the brief's wording, section 11).
- Development side: the six numbers of events typed into the axis labels of the counterpoint chart
  are supporting values of the claim `obs2-counterpoint`; the build compares each with the internal
  evidence and checks that its label stands in the panel of its dataset.

Wording trimmed to reach 1,200 words (no caveat, no dataset difference and none of the four
take-aways was removed)
- Hero diagram: the line "where, how much and when the hand touches" under "Contact dynamics".
- Footer: "Ongoing research." (the top bar and the badge say it).
- Motivation, lead: "for a given object motion" (the chain that follows says it).
- Observation 1, lead: "No grasp labels are used: “grasp realization” means which take, object or
  subject a frame comes from, and which first contact map a sequence starts from." became "Without
  grasp labels, “grasp realization” is proxied by take, object or subject identity and the first
  contact map."; interpretation: "then".
- Observation 3: the caption of the concept picture ends after "48 numbers" (the picture itself
  says "a disc and an arrow for each touching part"); lead: "of detail", "then"; tile label:
  "contact spread" became "spread"; the sentence "The coarse hand pose leaves the wrench error
  unchanged." (the tile on the future-structure error still shows what the hand pose adds); the
  second sentence of the interpretation ("The first carries much of the modelled function and
  future structure; the second is real, persistent, weakly predictable."), which repeated the
  headline and the two cards; the diagram labels "per frame" (twice).
- Datasets: the table shows "●" for the Motivation instead of "four test splits" (the Motivation
  says it) and has no header over its first column; credits: "under their own licences" became
  "(their own licences)", "(CVPR 2025)" is given once, in the Motivation, and "Only rendered images
  are shown" became "Rendered images only".
