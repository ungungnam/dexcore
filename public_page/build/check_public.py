#!/usr/bin/env python3
"""Exposure, traceability and self-containment checks of the public page.

Scans EVERY file under public_page/site/ (HTML text, attributes, comments, processing instructions and
declarations, element and attribute names, the raw HTML source, the chart data file, CSS, JavaScript
comments / strings / code, image metadata and file names) and writes
public_page/exposure_checklist.md: one line per rule with PASS / FAIL / NOT EVALUATED YET, what was
scanned, and for a failure the offending file, line and text. Exit status is non-zero on any FAIL.

    python public_page/build/check_public.py                 # check site/, write the checklist
    python public_page/build/check_public.py --site DIR --out FILE --claims FILE --manual FILE   # for tests
    python public_page/build/check_public.py --pins          # digests to record in build/manual_checks.json

The rules implement section 5 of the build plan of the public page ("what must not appear anywhere in
site/"), the wording list of section 16 of the owner's brief, and the layout / accessibility limits of
sections 1, 2 and 4 of the plan. The two length limits (rules H01 and H02) were set by the orchestrating session
after the audit of the page; they replace the numbers of plan section 2. The forbidden vocabulary below
is explicit: every pattern carries the reason it is there. It was built from the plan and from a scan
of the internal page (docs/index.html) and of the labels of its figures (docs/data/figures.json) for
code-like names.

How text is scanned
  prose   visible HTML text (per block, so a phrase split by inline tags is still found; each block is
          read a second time with a space at every inline-tag boundary, so that a symbol written with
          a subscript is found too; the word counts of rules H01 and H02 use that spaced reading, so
          that every chip, source label and stat tile counts as its own words), text-like attributes
          (alt, title, aria-label, content, data-series ...), HTML comments, processing instructions,
          declarations and CDATA sections, the text of a <script> that is not JavaScript, the strings
          of the data file, comments and string literals of CSS and JavaScript, image metadata text
  code    identifier-like attributes (class, id, href, src, data-fig ...), attribute and element names,
          raw CSS and JavaScript, inline style attributes, file names, and the raw source of every HTML
          file (a second net for anything the HTML parser does not hand over)
Patterns marked EVERYWHERE run on both; patterns marked PROSE only on prose (single letters and common
English words would otherwise hit variable names). The words of the internal modelling work are
searched in identifiers as well: every identifier of the code kinds is split at '-', '_', '.' and
at camelCase boundaries, and its parts are compared with IDENTIFIER_WORDS.
Before any rule runs, invisible characters (soft hyphen, zero-width space and joiners, word joiner,
byte-order mark) are removed from every piece of text and reported by rule V09: they could split a
forbidden word so that no pattern matches while the reader still sees the word.

A script cannot recognise a description of the internal work in other words. That is a human check;
its record is build/manual_checks.json (rule M01).
"""
import sys

sys.dont_write_bytecode = True

import argparse
import bisect
import hashlib
import html
import json
import re
import struct
import zlib
from html.parser import HTMLParser
from pathlib import Path

BUILD = Path(__file__).resolve().parent
PUBLIC = BUILD.parent
ROOT = PUBLIC.parent
DOCS = ROOT / "docs"
SITE = PUBLIC / "site"
CHECKLIST = PUBLIC / "exposure_checklist.md"
CLAIMS = BUILD / "public_claims.json"
MANUAL = BUILD / "manual_checks.json"

# Length limits: set by the orchestrating session after the audit (5 October 2026; the owner's confirmation
# is still due). They replace the numbers of plan
# section 2 (1,300 and 80), which were set for a count that joined adjacent inline elements.
MAX_BODY_WORDS = 1200   # visible body text; technical notes and alt texts excluded. Counted with every inline
                        # element read as separated by a space, so a row of chips with its source label, a stat
                        # tile and a table cell each count as the words a reader sees
MAX_NOTE_WORDS = 100    # each collapsed technical note, its summary line included; the same count
# Body text that is not running prose. Rule H01 also reports the count without it (the reading-time estimate of
# public_content_outline.md gives both counts): the navigation, the rows of dataset chips and source labels,
# the stat tiles, the table, and the labels drawn inside the conceptual diagrams (inline SVG).
ASIDE_TAGS = {"nav", "table", "svg"}
ASIDE_CLASSES = {"foot", "stats"}   # <div class="foot">: chips and source label; <div class="stats">: stat tiles
TECH_NOTE_CLASS = "tech-note"   # <details class="tech-note">: the only place where R0-R5 and "decoder" may appear
INTUITION_CLASS = "intuition"   # the only place where the "null-space-like" intuition may appear

# --------------------------------------------------------------------------------------------------
# Explicit lists
# --------------------------------------------------------------------------------------------------

# Files that may be shipped (paths relative to site/). Anything else fails rule S01.
ALLOWED_SITE_FILES = [
    r"index\.html",
    r"\.nojekyll",
    r"assets/css/[\w.-]+\.css",
    r"assets/js/[\w.-]+\.js",
    r"assets/img/[\w.-]+\.(?:webp|png|jpe?g|svg)",
    r"data/figures\.js",
    r"favicon\.(?:ico|svg|png)",
]

# Outgoing links: plain <a href> to these public pages only (prefix match; the contact address is exact).
# URL checked on 2026-10-05: each of the first five answered HTTP 200.
OUTGOING_LINK_ALLOWLIST = [
    "https://taco2024.github.io/",                    # TACO dataset page
    "https://arctic.is.tue.mpg.de/",                  # ARCTIC dataset page
    "https://oakink.net/v2/",                         # OakInk2 dataset page
    "https://vcai.mpi-inf.mpg.de/projects/bimart/",   # BimArt project page (cited paper)
    "https://github.com/RosettaWYzhang/BimArt",       # BimArt public code (cited paper; linked by its project page)
    "https://mano.is.tue.mpg.de/",                    # MANO hand model
    "https://arxiv.org/abs/",                         # cited papers
    "https://openaccess.thecvf.com/",                 # cited papers (CVF open access)
]
CONTACT_URLS = ["https://github.com/ungungnam", "https://github.com/ungungnam/"]   # contact line, exact match
# The contact line shows exactly one e-mail address (the owner's decision of 5 October 2026, build plan
# sections 3, 5 and 7). Any other address, in the text or in a mailto: link, fails rules P03 and N03.
ALLOWED_EMAILS = ["01unghui@gmail.com"]
# Namespace names used by SVG / XML code. They are identifiers, not network requests.
XML_NAMESPACES = [
    "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink", "http://www.w3.org/1999/xhtml",
    "http://www.w3.org/XML/1998/namespace", "http://www.w3.org/1998/Math/MathML",
]

# Images whose PIXELS were inspected by eye for drawn text (titles, labels, legends): no internal code,
# model name, path or file name is drawn in them. A script cannot read drawn text, so an image that is
# new or has changed fails rule I02 until someone looks at it and records its hash here.
#   contact_trajectory.webp     inspected 2026-10-05: frame labels, "512 points", colour scale 0-1
#   reconfiguration_events.webp inspected 2026-10-05: dataset names, "typical event (median retention)",
#                               "event in which a part joins (median change)", "76 directions, sorted",
#                               hand-part legend
#   compact_state.webp          inspected 2026-10-05: "1. Hand on the object", "2. Contact on the surface,
#                               coloured by hand part", "3. Only the 48 numbers ...", "64 frames", legend
INSPECTED_IMAGES = {
    "assets/img/contact_trajectory.webp": "adb7ddedfbe261caf2b77cf1fa58f33d997ca34482dbc28513218c3c104f6bd1",
    "assets/img/reconfiguration_events.webp": "57f901f2ab3e4a4b3c7d90eed139217cde50f9308c5f95bc8857245f2088aa9d",
    "assets/img/compact_state.webp": "7e52555c1575b05504cbc8f94f576a8f46ff8edef40c734b56129ee22b8ef5d6",
}

# Keys the chart data file may hold. Anything else (sources, module, id, ...) fails rule S02.
DATA_KEYS = {
    "figure": {"type", "title", "panels", "note"},
    "panel": {"title", "x", "y", "series", "refs", "highlight"},
    "x": {"label", "categories"},
    "y": {"label", "direction"},
    "series": {"name", "values", "lo", "hi"},
    "ref": {"axis", "value", "label"},
    "highlight": {"category", "label"},
}

# "text-spaced": a text block read again with a space at each inline-tag boundary (z<sub>t</sub> -> "z t").
PROSE = {"text", "text-spaced", "attr-text", "comment", "js-comment", "css-comment", "js-string", "css-string",
         "data-string", "image-meta"}
# "raw": the source of an HTML file as it is on disk (second net); "attr-name" / "tag-name": names in the markup.
CODE = {"attr-id", "attr-name", "tag-name", "js-code", "css-code", "filename", "raw"}
EVERYWHERE = PROSE | CODE
# what a reader of the page sees (the wording rules and the lower-case forms of the internal codes look here)
READER = {"text", "text-spaced", "attr-text", "comment", "data-string", "css-string", "image-meta"}
# kinds whose identifiers are split into words and compared with IDENTIFIER_WORDS
IDENTIFIER_KINDS = {"attr-id", "attr-name", "tag-name", "js-code", "css-code", "filename"}
INVISIBLE = "\u00ad\u200b\u200c\u200d\u2060\ufeff"   # soft hyphen, zero-width space / non-joiner / joiner, word joiner, BOM

# ---- V01: internal model, case and study codes (plan section 5, first bullet; scan of the internal page)
V01_CODES = [
    # B0 B1 B2: direct / via-latent generators; D0-D2, S0-S2: generator levels; M0-M3 and M00 M01 M10 M11:
    # the follow-up and the factorial models; A0-A3: ablation arms; F0-F2: plan list. Upper case only, so
    # the formula symbol s0 (first contact map) stays legal.
    (r"(?<![A-Za-z0-9_#])(?:B[0-2]|D[0-2]|S[0-2]|M[0-3]|M(?:00|01|10|11)|A[0-3]|F[0-2])(?![A-Za-z0-9_])",
     "internal model / arm code", EVERYWHERE),
    # T1-T4: the four prediction targets of the composite score; Q90: the spike threshold quantile.
    (r"(?<![A-Za-z0-9_#])(?:T[1-4]|Q90)(?![A-Za-z0-9_])", "internal target / threshold code", EVERYWHERE),
    (r"\b[Cc]ases?[ -][A-E]\b", "internal outcome label (Case A ... Case E)", EVERYWHERE),
    (r"\b[Ss]tage[ -]?(?:[12]\b|(?i:one|two)\b|II?\b)", "internal study name (Stage 1 / Stage 2)", EVERYWHERE),
    (r"\b[Gg]en(?:eration)?[ -]?[1-3]\b", "internal generation number of the ported model", EVERYWHERE),
    # CONTACT_FULL and FULL: the two dense levels of the representation ladder.
    (r"\bCONTACT_FULL\b|(?<![A-Za-z0-9_])FULL(?![A-Za-z0-9_])", "internal level name", EVERYWHERE),
    (r"\bPCA-[0-9]+\b", "internal baseline name", EVERYWHERE),
    # the same codes written in lower case, in what a reader sees ('arms b1 and m11'). The formula symbol s0
    # stays legal; code kinds are left out (single lower-case letters with a digit are ordinary variable names).
    (r"(?<![A-Za-z0-9_#])(?:b[0-2]|d[0-2]|s[12]|m[0-3]|m(?:00|01|10|11)|a[0-3]|f[0-2]|t[1-4]|q90)(?![A-Za-z0-9_])",
     "internal model / arm / target code in lower case", READER),
]

# ---- V02: level names R0-R5 (allowed only inside the one technical note)
V02_LEVELS = r"(?<![A-Za-z0-9_#])R[0-5](?![A-Za-z0-9_])"
V02_LEVELS_LOWER = r"(?<![A-Za-z0-9_#])r[0-5](?![A-Za-z0-9_])"    # in what a reader sees

# ---- V03: identifiers of the internal page: figure ids, evidence blocks, section and file names
V03_IDS = [
    # figure id families of docs/data/figures.json and the evidence units: f1a_ f1b_ f1c_ f1d_ f1suba_ f1subb_
    # f1s_ f2a_ f2b_ f2c_ f2suba_ f2subb_ f2_ f3a_ f3b_ f3c_ f3d_ f3s_ f3diaga_ ... f3suba_ ..., m0_, hyp_,
    # qual_ (images), zq_ / zz_ (evidence files)
    (r"(?<![A-Za-z0-9_])(?:f[1-3](?:[a-ds]|sub[ab]|diag[a-c])?|m0|hyp|qual|zq|zz)_[a-z0-9_]+",
     "internal figure / evidence id", EVERYWHERE),
    (r"(?<![A-Za-z0-9_])q_[a-z][a-z0-9_]+", "internal render script name", EVERYWHERE),
    (r"\bF[1-3]-(?:PRIMER|[A-D]|sub[AB]|diag[A-C])\b", "internal evidence block id", EVERYWHERE),
    (r"(?<![A-Za-z0-9_-])HYP(?![A-Za-z0-9_])", "internal evidence block id", EVERYWHERE),
    (r"\b[Ff]inding[ -][1-3]\b", "section name of the internal page", EVERYWHERE),
    (r"\b(?:Main|Sub)[- ]evidence\b|\bMain diagnostic\b|\bStep [A-D]\b|\bHypothesis [AB]\b",
     "block naming of the internal page", EVERYWHERE),
    # file stems of the internal renders and their metadata
    (r"(?<![A-Za-z0-9_])(?:contact_map_primer|reconfiguration_with_hand|structure_on_a_grasp|"
     r"motivation_two_contacts|grasp_modes|sample_then_evolve|change_events|z_and_r_maps|direct_vs_via_z|"
     r"z_moves_with_contact)(?![A-Za-z0-9_])", "file name of an internal render", EVERYWHERE),
]

# ---- V04: internal symbols and code-like identifiers
V04_SYMBOLS = [
    # symbols and arm names found in the labels and source locators of the internal figures
    (r"(?<![A-Za-z0-9_])(?:E_C|Q_pre|Q_post|C_pre|C_post|z_t|d_t|e_t|H_t|a_k|RMSE_probe|RMSE_persistence|"
     r"near_zero|samplerG\w*|gtinit\w*|static_gt|mlpG\w*|ridgeG\w*|test_[1-4])(?![A-Za-z0-9_])",
     "internal symbol / arm name", EVERYWHERE),
    # recorded sequence ids such as grab_01_s01_100
    (r"(?<![A-Za-z0-9_])(?:grab|use)_[0-9]{2}_s[0-9]{2}(?:_[0-9]+)?(?![A-Za-z0-9_])", "recorded sequence id", EVERYWHERE),
]
# Any other snake_case token in prose is treated as an internal identifier, except these:
SNAKE_CASE = r"(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9:]+)+(?![A-Za-z0-9_])"
SNAKE_ALLOWED = [
    r"s_0", r"C_1(?::T)?", r"C_T", r"C_t",          # symbols of the public factorization formula
    r"lower_better", r"higher_better",              # axis directions the chart renderer reads
    r"pub_[a-z0-9_]+",                              # public figure ids
    r"DEXCORE_FIGURES",                             # the global the chart renderer reads
]

# ---- V05: the learned latent state and its realization code (plan section 5, second bullet)
V05_LATENT = [
    (r"(?i)\blatents?\b", "the learned latent state", PROSE),
    # z and r as stand-alone symbols (also z-mediated, z-only, z*, delta-z); lower case only
    (r"(?<![A-Za-z0-9_'’])Δ?[zr](?![A-Za-z0-9'’])", "symbol of the learned state (z) or its realization code (r)", PROSE),
    # the same symbols with an index, as the internal page writes them: z1, zt, zt+1, r1 (joined from z<sub>t</sub>)
    (r"(?<![A-Za-z0-9_'’])Δ?[zr](?:[0-9]+|t(?:[+−-][0-9]+)?)(?![A-Za-z0-9'’])", "indexed symbol of the learned state (z) or its realization code (r)", PROSE),
    # upper-case Z named as a state, code or symbol
    (r"(?i:\b(?:states?|codes?|symbols?|variables?|vectors?))\s+Z(?![A-Za-z0-9])", "symbol of the learned state (Z)", PROSE),
    (r"(?i)\b(?:auto[- ]?)?encoders?\b", "encoder of the learned state", PROSE),
    (r"(?i)\brealization code\b", "the realization code", PROSE),
]
V05_DECODER = r"(?i)\bdecoders?\b"   # allowed only inside the technical note (the plan words a caveat with it)

# ---- V06: training recipe and architecture (supervision, losses, weights, sizes, seeds / steps / checkpoints)
V06_RECIPE = [
    (r"(?i)\b(?:relational|teacher|structur(?:al|e)|explicit)[- ]supervision\b|\bsupervision (?:loss|signal|experiment)\b",
     "relational / teacher / structural supervision", PROSE),
    (r"(?i)\bteacher\b", "teacher signal", PROSE),
    (r"(?i)\b(?:relational|dense|structur\w*|output|reconstruction|training|independence|auxiliary|diffusion|"
     r"L1|L2|MSE)[- ]loss(?:es)?\b|\bloss(?:es)? (?:weights?|terms?|functions?|ablation)\b|\bloss on\b",
     "loss name", PROSE),
    (r"[λΛ]|(?i:\blambda\b)", "loss weight symbol", PROSE),
    # sizes such as 64-D or 128-dimensional; 2-D and 3-D (pictures) stay legal
    (r"\b(?![23]-D\b)[0-9]+-D\b|(?i:\b[0-9]+[- ]dimensional\b)", "architecture / code size", PROSE),
    (r"(?i)\b[0-9]+ (?:blocks|layers|heads|hidden units|channels|parameters)\b|\bhidden (?:size|dim\w*|head|layers?|units?)\b",
     "architecture size", PROSE),
    # the owner's brief (section 9, item 3) removes training seeds and training steps: the page says 'one
    # training run', so the word 'seed' itself is not used, and no step, iteration or epoch count is given
    (r"(?i)\bseeds?\b|\b[0-9]{1,3}(?:,[0-9]{3})+ steps\b|\b[0-9]{3,} steps\b|\b[0-9][0-9,]* training steps\b|"
     r"\b[0-9]+-step\b|\b(?:steps?|epochs?|iterations?)\s*[=:]\s*[0-9]|\bepochs?\b|"
     r"\b[0-9][0-9.,]*\s?k?\s(?:iterations?|epochs?)\b|\b[0-9][0-9.,]*\s?k\ssteps\b|\bstep\s[0-9]+\b",
     "training seed / step / iteration / epoch detail", PROSE),
    (r"(?i)\bloss\s[0-9.]+|\b(?:objective|loss)\sweights?\b|\bweight(?:ed)?\s[0-9]*\.[0-9]+", "loss value or weight", PROSE),
    (r"(?i)\b[0-9]+-layer\b|\bwidth\s[0-9]+\b|\b(?:transformers?|MLPs?|GRUs?|LSTMs?|U-?Nets?|ResNets?|attention heads?)\b",
     "architecture name or size", PROSE),
    # a dated run or log line ('Run of 2026-10-03: ...'); the page gives its date as a month and a year
    (r"(?<![0-9.])(?:19|20)[0-9]{2}-[0-9]{2}-[0-9]{2}(?![0-9])", "ISO date (run or log date)", PROSE),
    (r"(?i)\bcheckpoints?\b|\bckpt\b", "checkpoint", EVERYWHERE),
    (r"\bEMA\b|(?i:\blearning rate\b|\bbatch size\b|\bweight decay\b|\bclassifier-free\b|\bcfg[_ ]?strength\b|"
     r"\bhyper-?parameters?\b|\bfine-?tun\w*)", "training setting", PROSE),
    (r"(?i)\b(?:MLP|ridge|learned|wrench) probe\b", "internal name of an analysis arm", PROSE),
    # the brief (section 5): no internal sampler architecture; the internal page names a diffusion sampler and a vector field
    (r"(?i)\bsamplers?\b|\bvector[- ]field\b", "internal sampler / evolution model", PROSE),
]

# ---- V07: internal diagnostics, experiments and plans (plan section 5, second bullet; brief section 8)
V07_DIAGNOSTICS = [
    (r"(?i)\b(?:oracle|teacher)[- ](?:z|latent)\b", "oracle / teacher-latent diagnostic", PROSE),
    (r"(?i)\btrain\s*[/–-]\s*test (?:latent )?gap\b|\bgenerali[sz]ation gap\b", "train / test latent gap", PROSE),
    (r"(?i)\bdecoder[- ]mismatch\b|\bmismatch (?:study|investigation)\b", "decoder-mismatch study", PROSE),
    (r"(?i)\bstructure[- ]aware\b|\bstructural surrogate\b", "structural-supervision experiment", PROSE),
    (r"(?i)\bdiffusion (?:futures?|samples?|recipe)\b|\bfuture diffusion\b|\bsequence[- ]diffusion\b",
     "diffusion future experiment", PROSE),
    (r"(?i)\bstateful\b|\bwhole[- ]sequence\b", "stateful versus whole-sequence prediction", PROSE),
    (r"(?i)(?<![A-Za-z0-9.])2\s*[×x]\s*2(?![A-Za-z0-9.])|\btwo-by-two\b|\bfactorial\b", "the 2 x 2 experiment", PROSE),
    (r"(?i)\bresidual decoder\b|\b(?:first-map|s_?0)[- ]preserving\b|\bjoint[- ]decoder\b",
     "first-map-preserving residual decoder", PROSE),
    (r"(?i)\bz[- ]mediated\b|\bvia z\b", "latent-mediated generation", PROSE),
    (r"(?i)\bnext[- ](?:method|experiment)\b", "next-method plan", PROSE),
    (r"\b(?:TODO|FIXME|XXX|TBD)\b", "internal to-do marker", EVERYWHERE),
    # the internal modelling work said in other words (found by the exposure audit); none of these words is
    # needed by the public page
    (r"(?i)\bembeddings?\b|\bbottlenecks?\b", "learned-state vocabulary (embedding, bottleneck)", PROSE),
    (r"(?i)\bauto[- ]?regressive\b|\broll[- ]?outs?\b|\brecurrent\b|\bone pass\b|\bcarried forward\b",
     "how the internal model advances its state", PROSE),
    (r"(?i)\bheld[- ]out\b", "train / held-out comparison of the internal diagnostics", PROSE),
    (r"(?i)\bsupervis\w+", "supervision of the internal models", PROSE),
    (r"(?i)\bdiffusion models?\b", "diffusion experiment", PROSE),
    (r"(?i)\bwe (?:will|plan)\b|\bplanned\b|\bto[- ]do\b|\bnot yet run\b", "plan or to-do", PROSE),
    (r"(?i)\bfour[- ]arm\b|\btwo by two\b|\b(?:two|three|four)[- ]arm(?:ed)?\b", "design of the running experiment", PROSE),
]

# ---- identifiers: words of the internal modelling work inside names (id, class, attribute and element names,
# CSS selectors and custom properties, JavaScript identifiers and object keys, file names). Every identifier
# is split at '-', '_', '.', digits and camelCase boundaries; a part that equals one of these words fails
# the rule named with it. (The chart data file is not split: its strings are read as prose, its keys are
# whitelisted by rule S02.)
IDENTIFIER_WORDS = {
    "latent": "V05", "latents": "V05", "encoder": "V05", "encoders": "V05", "autoencoder": "V05",
    "decoder": "V05", "decoders": "V05",
    "teacher": "V06", "supervision": "V06", "supervised": "V06", "loss": "V06", "losses": "V06", "lambda": "V06",
    "sampler": "V06", "samplers": "V06", "epoch": "V06", "epochs": "V06", "seed": "V06", "seeds": "V06",
    "oracle": "V07", "stateful": "V07", "factorial": "V07", "rollout": "V07", "autoregressive": "V07",
    "bottleneck": "V07", "todo": "V07", "fixme": "V07",
}
# In the markup (id and class values, data-* attribute names, custom element names) a part that is exactly
# one of these letters is the symbol of the learned state or of its realization code (rule V05).
IDENTIFIER_LETTERS = {"z", "r"}

# ---- P01: server paths and internal directories (plan section 5, third bullet)
P01_PATHS = [
    (r"(?<![\w.])/(?:result|ckpt|home|data|backups|workspace|tmp|mnt|snap|usr|etc|var|opt|root)(?=/|\b)", "server path", EVERYWHERE),
    (r"(?<![\w/.-])(?:reports|scripts|configs|third_party|outputs)/", "internal directory", EVERYWHERE),
    (r"\b(?:taco|arctic|oakink2)/[0-9]{2}_", "numbered result directory", EVERYWHERE),
    (r"(?i)\bfile://|\blocalhost\b", "local address", EVERYWHERE),
]

# ---- P02: file names of data, code and logs; hashes
P02_FILES = [
    (r"(?i)[\w./-]*\w\.(?:csv|tsv|npz|npy|json|jsonl|pt|pth|ckpt|safetensors|pkl|h5|hdf5|parquet|ipynb|yaml|yml|toml|md|sh)\b(?![\w(])",
     "data / config / document file name", EVERYWHERE),
    (r"[\w./-]*\w\.py\b(?![\w(])", "code file name", EVERYWHERE),
    (r"[\w./-]*\w\.log\b(?!\s*\()", "log file name", EVERYWHERE),     # a call such as Math.log( is not a file
    (r"(?i)\bwandb\b", "experiment-tracker name", EVERYWHERE),
]
# a hexadecimal string of 7 to 64 characters with digits and letters: commit or file hash (CSS colours start with #)
P02_HASH = r"(?<![#\w.])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}(?!\w)"

# ---- P03: server or personal information (plan section 5, fourth bullet)
P03_PERSONAL = [
    (r"(?i)\buhnam\b", "server user name", EVERYWHERE),
    (r"(?i)\bmango\b", "server host name", EVERYWHERE),
    (r"(?<![\d.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![\d.])", "IP address", EVERYWHERE),
    # the owner's decision: the page gives no name and no affiliation (build plan, section 3)
    (r"(?i)\buniversit(?:y|ies)\b|\binstitutes?\b|\blaborator(?:y|ies)\b", "affiliation", PROSE),
    (r"(?i)(?<![0-9a-z])unghui(?![0-9a-z@])", "given name outside the contact address", EVERYWHERE),
    (r"(?i)\[\s*at\s*\]|\(\s*at\s*\)|\s+at\s+\S+\s+dot\s+\S+", "e-mail address written around the @ sign", PROSE),
]
# Names that must not appear but are not spelled out here either, because this file lives in a public
# repository: SHA-256 digests of the lower-cased name of an institution account, of an institution host
# label, and of an institution mail domain. Every word of the scanned text, every dotted name and every
# dotted suffix of it (a.b.c -> a.b.c, b.c, c) is hashed and compared with these digests.
P03_PRIVATE_DIGESTS = {
    "87faf4b0570809b9074b25c5d62470e4c7fcab98ef5f1e28dc14ad9c36644a19": "institution host label",
    "cca00904aab192730dac37105f0b2536dbf175b98e02e8d71a38f7cc8b96d299": "institution account name",
    "1097ccf3cf3095bdf95e2344010ad46f6b1ace625ca198ae48ee579469223175": "institution mail domain",
}
P03_EMAIL = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"
P03_USERNAME = r"(?i)ungungnam"   # legal only as the contact line github.com/ungungnam (no repository path)

# ---- N04: links or references to the internal page, docs/, the evidence manifest or the code repository
N04_INTERNAL = [
    (r"(?i)evidence_manifest|evidence_index|content_outline|public_claims|public_evidence|exposure_checklist",
     "development file of the project", EVERYWHERE),
    (r"(?<![\w/.-])docs/", "the internal page directory", EVERYWHERE),
    (r"\.\./", "parent-directory reference (leaves the site)", EVERYWHERE),
    (r"(?i)github\.com/ungungnam/[\w.-]+", "the code repository", EVERYWHERE),
    (r"(?i)\binternal (?:page|version|site|reports?|notes?|manifest)\b|(?<![\w.])/internal\b", "reference to the internal page", EVERYWHERE),
]
N04_REPO_NAME = r"(?i)dexcore"    # the repository name; legal only inside the identifier DEXCORE_FIGURES

# ---- W01: wording blacklist (brief section 16; plan section 4)
W01_WORDING = [
    (r"(?i)\bprov(?:e|es|ed|en|ing)\b|\bproofs?\b", "prove (say: we observe / our analysis suggests)"),
    (r"(?i)\bsolv(?:e|es|ed|ing)\b", "solve"),
    (r"(?i)\bfinal method\b", "final method"),
    (r"(?i)\bstate[- ]of[- ]the[- ]art\b|(?-i:\bSOTA\b)", "state-of-the-art"),
    (r"(?i)\bhuman intent(?:ions?)?\b", "human intent"),
    (r"(?i)\bnois(?:e|es|y)\b", "noise (say: realization variation)"),
    (r"(?i)\b(?:true )?task functionality\b|\btrue functionality\b", "true task functionality (say: modelled wrench capability)"),
    (r"(?i)\boptimal(?:ly|ity)?\b|\boptimum\b", "optimal"),
    (r"(?i)\bnecessary\b", "necessary ('not necessarily' is legal)"),
    (r"(?i)\bdisentangl\w*", "disentangled"),
    (r"(?i)\btrue null[- ]?space\b", "true null-space motion"),
    # finished-method tone (brief sections 1 and 16)
    (r"(?i)\bour (?:final |new |proposed )?(?:method|model|approach|framework|system)\b", "our method / model / approach"),
    (r"(?i)\boutperform\w*", "outperform"),
    (r"(?i)\bwe (?:show|demonstrate|achieve)\b", "we show / demonstrate / achieve"),
    (r"(?i)\bbest\b", "best (a ranking claim)"),
]
W02_NULLSPACE = r"(?i)\bnull[- ]?space\b"

# --------------------------------------------------------------------------------------------------
# Units of text
# --------------------------------------------------------------------------------------------------


class Unit:
    """One piece of scanned text: where it is, what kind it is, and whether it sits in a technical note."""

    __slots__ = ("file", "line", "kind", "text", "tech", "intuition", "what", "marks", "counted", "note_id", "nav", "section",
                 "spaced", "aside")

    def __init__(self, file, line, kind, text, tech=False, intuition=False, what="", marks=None, counted=False, note_id=None,
                 nav=False, section=None, spaced=None, aside=False):
        self.file, self.line, self.kind, self.text = file, line, kind, text
        self.tech, self.intuition, self.what = tech, intuition, what
        self.nav = nav            # True inside <nav>: section numbers of a navigation bar are labels, not measurements
        self.section = section    # id of the enclosing <section id="...">, if any (the manual checks pin a section's text)
        self.marks = marks        # [(offset, line)] for text assembled from several source lines
        self.counted = counted    # True for visible body text that counts towards the word budget
        self.note_id = note_id    # index of the enclosing technical note
        self.spaced = spaced      # a text block read with a space at every inline-tag boundary: what the words are counted on
        self.aside = aside        # True for body text that is not running prose (navigation, chip rows, stat tiles,
                                  # table cells, diagram labels)

    def line_at(self, pos):
        if self.marks:
            i = bisect.bisect_right([m[0] for m in self.marks], pos) - 1
            return self.marks[max(i, 0)][1]
        return self.line + self.text.count("\n", 0, pos)


def excerpt(text, start, end, width=48):
    a, b = max(0, start - width), min(len(text), end + width)
    s = ("…" if a > 0 else "") + text[a:b] + ("…" if b < len(text) else "")
    return re.sub(r"\s+", " ", s).strip()


BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "body", "br", "caption", "dd", "details", "dialog", "div", "dl",
    "dt", "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "head",
    "header", "hr", "html", "li", "main", "nav", "noscript", "ol", "option", "p", "pre", "section", "summary",
    "table", "tbody", "td", "tfoot", "th", "thead", "title", "tr", "ul", "button", "label", "legend", "select",
    "textarea", "template", "svg", "text", "desc", "g", "defs", "symbol", "foreignobject", "img", "figure",
}
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
# attributes whose value is read by people (or shown by the chart renderer)
TEXT_ATTRS = {"alt", "title", "aria-label", "aria-description", "aria-roledescription", "aria-valuetext",
              "placeholder", "content", "label", "summary", "value", "abbr"}
# data-* attributes that hold identifiers or numbers; every other data-* attribute is read as text
DATA_ID_ATTRS = {"data-fig", "data-height", "data-format", "data-labels", "data-shared-y"}
URL_ATTRS = {"src", "href", "poster", "data", "action", "formaction", "background", "xlink:href", "cite",
             "manifest", "ping", "longdesc", "srcset", "imagesrcset"}
NUM = r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?"
GEOMETRY_ATTRS = {
    "d": r"[MmLlHhVvCcSsQqTtAaZz0-9eE\s,.+-]*",
    "points": r"[0-9eE\s,.+-]*",
    "viewbox": r"[0-9eE\s,.+-]*",
    "transform": r"(?:\s*(?:matrix|translate|scale|rotate|skewX|skewY)\s*\([0-9eE\s,.+-]*\)\s*)*",
    "gradienttransform": r"(?:\s*(?:matrix|translate|scale|rotate|skewX|skewY)\s*\([0-9eE\s,.+-]*\)\s*)*",
    "stroke-dasharray": r"[0-9eE\s,.+-]*|none",
}
NUMERIC_ATTRS = {
    "x", "y", "x1", "x2", "y1", "y2", "cx", "cy", "r", "rx", "ry", "width", "height", "dx", "dy", "offset",
    "refx", "refy", "markerwidth", "markerheight", "stroke-width", "stroke-opacity", "fill-opacity", "opacity",
    "font-size", "rotate", "textlength", "stroke-miterlimit", "colspan", "rowspan", "tabindex", "span", "size",
    "start", "cols", "rows", "data-height", "stddeviation", "letter-spacing",
}
NUMERIC_VALUE = re.compile(r"\s*(?:" + NUM + r"(?:px|em|rem|%|pt|vh|vw)?|auto)\s*", re.I)


class PageParser(HTMLParser):
    """Splits an HTML (or SVG) file into units and records what the structural rules need."""

    def __init__(self, rel):
        super().__init__(convert_charrefs=True)
        self.rel = rel
        self.units = []
        self.stack = []            # frames: dict(tag, tech, intuition, raw, noscript, hidden_text, note_id, head)
        self.buf, self.marks, self.buf_line = [], [], None
        self.sbuf = []             # the same text with a space at every inline-tag boundary
        self.metas = []            # (line, attrs) of <meta>
        self.headings = []         # (level, line)
        self.imgs = []             # (line, attrs)
        self.svgs = []             # (line, attrs) of top-level inline <svg>
        self.refs = []             # (tag, attr, value, line)
        self.ids = {}              # id -> line
        self.notes = []            # technical notes: dict(line, tag)
        self.figs = []             # (line, data-fig)
        self.scripts = []          # (line, src or None)
        self.forbidden_tags = []   # (line, tag, why)
        self.svg_depth = 0

    # -- context helpers
    def top(self, key, default=None):
        return self.stack[-1][key] if self.stack else default

    def flush(self):
        if self.buf:
            text = "".join(self.buf)
            if text.strip():
                counted = not (self.top("tech", False) or self.top("noscript", False) or self.top("hidden_text", False)
                               or self.top("head", False))
                spaced = re.sub(r"\s+", " ", "".join(self.sbuf)).strip()
                self.units.append(Unit(self.rel, self.buf_line, "text", text, tech=self.top("tech", False),
                                       intuition=self.top("intuition", False), what="text", marks=list(self.marks),
                                       counted=counted, note_id=self.top("note_id"), nav=self.top("nav", False),
                                       section=self.top("section"), spaced=spaced, aside=self.top("aside", False)))
                if spaced != re.sub(r"\s+", " ", text).strip():
                    # second reading for the vocabulary rules: never used for numbers (the word counts read the same
                    # string from the text unit above, so nothing is counted twice)
                    self.units.append(Unit(self.rel, self.buf_line, "text-spaced", spaced, tech=self.top("tech", False),
                                           intuition=self.top("intuition", False), what="text, inline tags read as spaces",
                                           note_id=self.top("note_id"), nav=self.top("nav", False), section=self.top("section")))
        self.buf, self.marks, self.buf_line = [], [], None
        self.sbuf = []

    # -- parser callbacks
    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        if tag in BLOCK_TAGS:
            self.flush()
        else:
            self.sbuf.append(" ")
        self.units.append(Unit(self.rel, line, "tag-name", tag, what="element name"))
        a = {}
        for k, v in attrs:
            a.setdefault(k, v if v is not None else "")
            self.units.append(Unit(self.rel, line, "attr-name", k, what=f"attribute name of <{tag}>"))
        classes = set(a.get("class", "").split())
        boundary = TECH_NOTE_CLASS in classes or INTUITION_CLASS in classes
        if boundary:
            self.flush()    # the text of such an element is a unit of its own, also when the element is inline
        frame = {
            "boundary": boundary,
            "tag": tag,
            "tech": self.top("tech", False) or TECH_NOTE_CLASS in classes,
            "intuition": self.top("intuition", False) or INTUITION_CLASS in classes,
            # a <script> of another type (a template, JSON ...) is not code: its content is read as text
            "raw": ("script" if tag == "script" and is_javascript(a.get("type")) else "style" if tag == "style"
                    else "other-script" if tag == "script" else None),
            "noscript": self.top("noscript", False) or tag == "noscript",
            # the <title> / <desc> of an inline SVG and a <template> are not rendered as page text
            "hidden_text": self.top("hidden_text", False) or tag == "template" or (self.svg_depth > 0 and tag in ("title", "desc")),
            "note_id": self.top("note_id"),
            "head": self.top("head", False) or tag == "head",
            "nav": self.top("nav", False) or tag == "nav",
            "aside": self.top("aside", False) or tag in ASIDE_TAGS or bool(classes & ASIDE_CLASSES),
            "section": a["id"] if tag == "section" and a.get("id") else self.top("section"),
        }
        if TECH_NOTE_CLASS in classes and not self.top("tech", False):
            self.notes.append({"line": line, "tag": tag})
            frame["note_id"] = len(self.notes) - 1
        # structure
        if re.fullmatch(r"h[1-6]", tag):
            self.headings.append((int(tag[1]), line))
        if tag == "img":
            self.imgs.append((line, a))
        if tag == "svg":
            if self.svg_depth == 0:
                self.svgs.append((line, a))
        if tag == "figure" and "data-fig" in a:
            self.figs.append((line, a["data-fig"]))
        if tag == "script":
            self.scripts.append((line, a.get("src")))
        if tag in ("base", "iframe", "object", "embed", "frame", "frameset", "applet"):
            self.forbidden_tags.append((line, tag, "element that loads or re-bases external content"))
        if tag == "meta" and a.get("http-equiv", "").lower() == "refresh":
            self.forbidden_tags.append((line, "meta http-equiv=refresh", "redirect"))
        if tag == "meta":
            self.metas.append((line, a))
        if "id" in a:
            self.ids.setdefault(a["id"], line)
        # attributes
        for name, value in a.items():
            if name in URL_ATTRS:
                self.refs.append((tag, name, value, line))
            if name.startswith("xmlns") and value in XML_NAMESPACES:
                continue   # a namespace name is an identifier, not a request
            if name == "style":
                self.units.append(Unit(self.rel, line, "css-code", value, tech=frame["tech"], what="style attribute"))
                continue
            if name in GEOMETRY_ATTRS and re.fullmatch(GEOMETRY_ATTRS[name], value):
                continue
            if name in NUMERIC_ATTRS and NUMERIC_VALUE.fullmatch(value):
                continue
            is_text = name in TEXT_ATTRS or (name.startswith("data-") and name not in DATA_ID_ATTRS)
            if tag == "meta" and name == "content":
                # only the description and the sharing texts are read by people (not viewport, charset ...)
                meta_name = (a.get("name") or a.get("property") or "").lower()
                is_text = meta_name in ("description", "author", "keywords") or meta_name.startswith(("og:", "twitter:"))
            self.units.append(Unit(self.rel, line, "attr-text" if is_text else "attr-id", value, tech=frame["tech"],
                                   intuition=frame["intuition"], what=f"{name} attribute of <{tag}>", section=frame["section"]))
            if name.startswith("on"):
                self.units.append(Unit(self.rel, line, "js-code", value, what=f"{name} handler"))
        if tag == "svg":
            self.svg_depth += 1
        if tag not in VOID_TAGS:
            self.stack.append(frame)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        open_tags = [f["tag"] for f in self.stack]
        if tag in BLOCK_TAGS or (tag in open_tags and any(f["boundary"] for f in self.stack[len(open_tags) - 1 - open_tags[::-1].index(tag):])):
            self.flush()
        else:
            self.sbuf.append(" ")
        if tag in open_tags:
            while self.stack:
                f = self.stack.pop()
                if f["tag"] == "svg":
                    self.svg_depth = max(0, self.svg_depth - 1)
                if f["tag"] == tag:
                    break

    def handle_data(self, data):
        line = self.getpos()[0]
        raw = self.top("raw")
        if raw == "script":
            self.units.append(Unit(self.rel, line, "js-code", data, what="inline script"))
            return
        if raw == "style":
            self.units.append(Unit(self.rel, line, "css-code", data, what="inline style sheet"))
            return
        if raw == "other-script":
            self.units.append(Unit(self.rel, line, "comment", html.unescape(data), what="content of a <script> that is not JavaScript"))
            return
        # inline content is concatenated without added spaces, so that s<sub>0</sub> reads "s0"
        for i, piece in enumerate(data.split("\n")):
            if i:
                self.buf.append(" ")
                self.sbuf.append(" ")
            if self.buf_line is None:
                self.buf_line = line + i
            self.marks.append((sum(len(x) for x in self.buf), line + i))
            piece = re.sub(r"[ \t\r\f\v\u00a0]+", " ", piece)
            self.buf.append(piece)
            self.sbuf.append(piece)

    def handle_comment(self, data):
        # character references are not decoded inside a comment by the parser: a word written with them is decoded here
        self.units.append(Unit(self.rel, self.getpos()[0], "comment", html.unescape(data), what="HTML comment"))

    def handle_pi(self, data):
        self.units.append(Unit(self.rel, self.getpos()[0], "comment", html.unescape(data), what="processing instruction"))

    def handle_decl(self, decl):
        if decl.strip().lower() != "doctype html":
            self.units.append(Unit(self.rel, self.getpos()[0], "comment", html.unescape(decl), what="declaration"))

    def unknown_decl(self, data):
        # a CDATA section: inside an inline SVG it is rendered as page text
        line = self.getpos()[0]
        text = data[6:] if data.startswith("CDATA[") else data
        if self.svg_depth > 0:
            if self.buf_line is None:
                self.buf_line = line
            self.marks.append((sum(len(x) for x in self.buf), line))
            piece = re.sub(r"\s+", " ", text)
            self.buf.append(piece)
            self.sbuf.append(piece)
        else:
            self.units.append(Unit(self.rel, line, "comment", text, what="CDATA section / unknown declaration"))

    def close(self):
        super().close()
        self.flush()


def is_javascript(script_type):
    """True for a <script> whose type makes the browser run it (no type, a JavaScript type, a module)."""
    t = (script_type or "").split(";")[0].strip().lower()
    return t in ("", "module", "text/javascript", "application/javascript", "text/ecmascript", "application/ecmascript",
                 "text/jscript", "text/livescript", "application/x-javascript", "text/x-javascript")


def js_tokens(src):
    """Comments and string literals of a script: yields (kind, text, offset). Regular-expression literals are skipped."""
    i, n = 0, len(src)
    last_kind, last_text = None, ""     # the last token before the current position: "ident" or "punct"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = src.find("\n", i)
            j = n if j < 0 else j
            yield "js-comment", src[i + 2:j], i
            i = j
        elif c == "/" and nxt == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j
            yield "js-comment", src[i + 2:j], i
            i = j + 2
        elif c in "\"'`":
            j = i + 1
            while j < n and src[j] != c and (c == "`" or src[j] != "\n"):
                j += 2 if src[j] == "\\" else 1
            yield "js-string", src[i + 1:j], i
            i, last_kind, last_text = j + 1, "ident", "string"
        elif c == "/" and (last_kind is None or (last_kind == "punct" and last_text in "(,=:[!&|?{};+-*%<>~^")
                           or (last_kind == "ident" and last_text in ("return", "typeof", "case", "in", "of", "void", "delete"))):
            j, in_class = i + 1, False      # a regular-expression literal: skip to its closing slash
            while j < n and (src[j] != "/" or in_class) and src[j] != "\n":
                if src[j] == "\\":
                    j += 1
                elif src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                j += 1
            i, last_kind, last_text = j + 1, "ident", "regex"
        elif c.isalnum() or c in "_$":
            j = i
            while j < n and (src[j].isalnum() or src[j] in "_$"):
                j += 1
            last_kind, last_text, i = "ident", src[i:j], j
        else:
            if not c.isspace():
                last_kind, last_text = "punct", c
            i += 1


def css_tokens(src):
    """Comments and quoted strings of a style sheet: yields (kind, text, offset)."""
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == "/" and src[i + 1:i + 2] == "*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j
            yield "css-comment", src[i + 2:j], i
            i = j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c and src[j] != "\n":
                j += 2 if src[j] == "\\" else 1
            yield "css-string", src[i + 1:j], i
            i = j + 1
        else:
            i += 1


RAW_GEOMETRY = re.compile(r"""(\b(?:d|points|viewBox|transform|gradientTransform|stroke-dasharray)\s*=\s*)(["'])(.*?)\2""", re.S)


def raw_markup(source):
    """The source of an HTML file prepared for the second net: every character stays on its line. Geometry
    attributes (path data such as d="M3 0H16") are blanked, because their letters and digits are coordinates,
    and the slash of a closing tag is blanked, because </var> or </data> is not a path."""
    def blank(m):
        name = m.group(1).split("=")[0].strip().lower()
        rule = GEOMETRY_ATTRS.get(name)
        if rule and re.fullmatch(rule, m.group(3)):
            return m.group(1) + m.group(2) + re.sub(r"[^\n]", " ", m.group(3)) + m.group(2)
        return m.group(0)
    return re.sub(r"</(?=[A-Za-z])", "< ", RAW_GEOMETRY.sub(blank, source))


def line_of(text, offset, base=1):
    return base + text.count("\n", 0, offset)


def parse_data_file(text):
    """The object assigned to window.DEXCORE_FIGURES, or (None, error)."""
    m = re.search(r"window\.DEXCORE_FIGURES\s*=\s*", text)
    if not m:
        return None, "no assignment to window.DEXCORE_FIGURES"
    payload = text[m.end():].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    try:
        return json.loads(payload), None
    except ValueError as err:
        return None, f"the assigned value is not plain JSON: {err}"


# ---- image metadata ------------------------------------------------------------------------------

def image_chunks(path):
    """(kind, list of (chunk name, text or None, is_private_metadata)) of a raster image; raises ValueError if unreadable."""
    b = path.read_bytes()
    out = []
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        off = 12
        while off + 8 <= len(b):
            tag = b[off:off + 4].decode("latin-1")
            size = struct.unpack("<I", b[off + 4:off + 8])[0]
            body = b[off + 8:off + 8 + size]
            if tag in ("VP8 ", "VP8L", "VP8X", "ALPH", "ANIM", "ANMF"):
                out.append((tag, None, False))
            elif tag == "ICCP":
                out.append((tag, printable(body), False))
            else:
                out.append((tag, printable(body), True))     # EXIF, XMP and anything unknown
            off += 8 + size + (size & 1)
        return "webp", out
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        off = 8
        while off + 8 <= len(b):
            size = struct.unpack(">I", b[off:off + 4])[0]
            tag = b[off + 4:off + 8].decode("latin-1")
            body = b[off + 8:off + 8 + size]
            if tag in ("tEXt", "zTXt", "iTXt"):
                text = body
                if tag == "zTXt":
                    key, _, rest = body.partition(b"\0")
                    try:
                        text = key + b" " + zlib.decompress(rest[1:])
                    except zlib.error:
                        text = body
                elif tag == "iTXt":
                    key, _, rest = body.partition(b"\0")
                    if rest[:1] == b"\x01":
                        parts = rest[2:].split(b"\0", 2)
                        try:
                            text = key + b" " + zlib.decompress(parts[-1])
                        except (zlib.error, IndexError):
                            text = body
                out.append((tag, printable(text, 1), True))   # text chunk (author, software, title ...): strip it
            elif tag == "eXIf":
                out.append((tag, printable(body), True))
            elif tag in ("iCCP",):
                out.append((tag, printable(body), False))
            else:
                out.append((tag, None, False))
            off += 12 + size
        return "png", out
    if b[:2] == b"\xff\xd8":
        off = 2
        while off + 4 <= len(b) and b[off] == 0xFF:
            marker = b[off + 1]
            if marker == 0xDA:
                break
            size = struct.unpack(">H", b[off + 2:off + 4])[0]
            body = b[off + 4:off + 2 + size]
            name = f"APP{marker - 0xE0}" if 0xE0 <= marker <= 0xEF else ("COM" if marker == 0xFE else f"0x{marker:02X}")
            if marker == 0xFE:
                out.append((name, printable(body, 1), True))      # comment segment: strip it
            elif marker in (0xE1, 0xED):      # Exif / XMP, IPTC
                out.append((name, printable(body), True))
            elif 0xE0 <= marker <= 0xEF:
                out.append((name, printable(body), False))
            else:
                out.append((name, None, False))
            off += 2 + size
        return "jpeg", out
    raise ValueError("not a WebP, PNG or JPEG file")


def printable(b, minimum=6):
    """Readable strings inside a binary chunk."""
    return " | ".join(m.group().decode("latin-1") for m in re.finditer(rb"[\x20-\x7e]{%d,}" % minimum, b))


# ---- numbers in the visible text -------------------------------------------------------------------

NUMBER = re.compile(r"(?<![A-Za-z0-9_])([+\-−]?)((?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?)(\s?(?:%|percent\b|per cent\b))?")


def number_tokens(text):
    """Numbers of a text as (start, end, key): key is the digits without sign or thousands separators, plus '%'
    for a percentage. In a range such as 11–15% both ends are percentages."""
    toks = []
    for m in NUMBER.finditer(text):
        toks.append([m.start(), m.end(), m.group(2).replace(",", ""), bool(m.group(3))])
    for i in range(len(toks) - 1):
        gap = text[toks[i][1]:toks[i + 1][0]]
        if toks[i + 1][3] and not toks[i][3] and re.fullmatch(r"\s?(?:–|-|to)?\s?", gap) and (gap.strip() or text[toks[i + 1][0]] in "-−"):
            toks[i][3] = True
    return [(a, b, key + ("%" if pct else "")) for a, b, key, pct in toks]


def displayed_key(displayed):
    """The key of a displayed number of a claim ('37.1', '46%', '+18%' -> '18%'), or None if it is not one number."""
    toks = number_tokens(str(displayed))
    return toks[0][2] if len(toks) == 1 else None


def structural_spans(text, structural):
    spans = []
    for entry in structural:
        if "regex" in entry:
            pattern = entry["regex"]
        else:
            pattern = r"(?i)" + r"\s+".join(re.escape(w) for w in str(entry["text"]).split())
        for m in re.finditer(pattern, text):
            spans.append((m.start(), m.end()))
    return spans


def claim_strings(claim):
    """The exact strings a claim has on the page: its sentence, and the listed strings that carry its numbers
    outside the sentence (stat tiles: 'TACO 0.85 119 events')."""
    out = [str(claim.get("public_claim", ""))] + [str(x) for x in claim.get("shown", [])]
    return [x for x in out if x.strip()]


def text_pattern(text):
    """Matches a string word for word, whatever the white space between its words."""
    return re.compile(r"\s+".join(re.escape(w) for w in text.split()))


def claim_text_keys(claim):
    """Displayed numbers of a claim that are shown in its text (a number marked "where": "chart" is a bar label only)."""
    keys = set()
    for num in claim.get("numbers", []):
        if num.get("where", "text") != "chart":
            key = displayed_key(num.get("displayed", ""))
            if key:
                keys.add(key)
    return keys


def number_coverage(units, claims_doc):
    """Numbers of the visible page text without a claim entry or a structural allowance.
    A number is covered only INSIDE the text of a claim that lists it (the claim's sentence or one of its listed
    tile strings, found word for word in the same text block), so a number moved to another sentence, quantity
    or dataset is not covered by a claim that happens to display the same digits elsewhere.
    Returns (uncovered: [(unit, start, end, key)], used displayed keys, total number of tokens)."""
    structural = claims_doc.get("structural_numbers", [])
    claims = []
    for claim in claims_doc.get("claims", []):
        keys = claim_text_keys(claim)
        if keys:
            claims.append(([text_pattern(t) for t in claim_strings(claim)], keys))
    uncovered, used, total = [], set(), 0
    for u in units:
        if u.kind not in ("text", "attr-text") or u.nav:
            continue
        if u.kind == "attr-text" and not re.match(r"(?:alt|title|aria-label|aria-description|content) attribute", u.what):
            continue
        spans = structural_spans(u.text, structural)
        claim_spans = [(m.start(), m.end(), keys) for patterns, keys in claims for rx in patterns for m in rx.finditer(u.text)]
        for a, b, key in number_tokens(u.text):
            total += 1
            if any(s <= a and b <= e for s, e in spans):
                continue
            if any(s <= a and b <= e and key in keys for s, e, keys in claim_spans):
                used.add(key)
                continue
            uncovered.append((u, a, b, key))
    return uncovered, used, total


def panel_strings(panel):
    """The reader-visible strings of one panel of a figure of the data file (labels, category and series names)."""
    out = [panel.get("x", {}).get("label", ""), panel.get("y", {}).get("label", "")]
    out += list(panel.get("x", {}).get("categories", []))
    out += [sr.get("name", "") for sr in panel.get("series", [])]
    out += [r.get("label", "") for r in panel.get("refs", [])]
    out += [panel.get("highlight", {}).get("label", "")]
    return [x for x in out if isinstance(x, str) and x]


def claims_on_page(units, claims_doc, data):
    """Problems of the claims themselves, as [(claim id, text, why)]:
    a claim sentence or tile string that is not on the page word for word; a displayed number that is in none of
    its claim's strings; a chart-only number that the named public chart does not plot for that dataset; a
    supporting value marked "where": "chart" (a number typed into a chart label, such as the number of events
    of a kind) whose label is not written in the panel of its dataset or does not hold the supported value."""
    texts = [u.text for u in units if u.kind == "text"]
    figures = (data or {}).get("figures", {})
    bad = []
    for claim in claims_doc.get("claims", []):
        cid = claim.get("id", "?")
        strings = claim_strings(claim)
        if not strings:
            bad.append((cid, "", "the claim has no public_claim text"))
        for t in strings:
            rx = text_pattern(t)
            if not any(rx.search(text) for text in texts):
                bad.append((cid, t[:110], "claim text is not found word for word in one text block of the page"))
        in_text = set()
        for t in strings:
            in_text.update(key for _a, _b, key in number_tokens(t))
        assets = claim.get("figure_asset")
        assets = assets if isinstance(assets, list) else [assets]
        for num in claim.get("numbers", []):
            key = displayed_key(num.get("displayed", ""))
            if num.get("where", "text") == "chart":
                plotted = False
                for fid in assets:
                    for panel in figures.get(fid, {}).get("panels", []) if isinstance(fid, str) else []:
                        if panel.get("title") != num.get("dataset"):
                            continue
                        for series in panel.get("series", []):
                            for bound in ("values", "lo", "hi"):
                                if any(isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) == num.get("full_value")
                                       for v in series.get(bound, [])):
                                    plotted = True
                if not plotted:
                    bad.append((cid, str(num.get("displayed")), "chart-only number: its full value is not plotted for that dataset "
                                                                "in a public chart named by the claim"))
            elif key is None or key not in in_text:
                bad.append((cid, str(num.get("displayed")), "displayed number is in neither the claim sentence nor its listed tile "
                                                            "strings (stale entry, or mark it \"where\": \"chart\")"))
        for sup in claim.get("support", []):
            if sup.get("where") is None:
                continue
            label = str(sup.get("shown_as", ""))
            if sup.get("where") != "chart":
                bad.append((cid, label, "supporting value: \"where\" can only be \"chart\" (a number typed into a chart label)"))
                continue
            written = any(label in panel_strings(panel)
                          for fid in assets if isinstance(fid, str)
                          for panel in figures.get(fid, {}).get("panels", []) if panel.get("title") == sup.get("dataset"))
            if not written:
                bad.append((cid, label, "chart label of a supporting value: not written, word for word, in the panel of that dataset "
                                        "of a public chart named by the claim"))
            holds = False
            for _a, _b, key in number_tokens(label):
                try:
                    holds = holds or float(key.rstrip("%")) == float(sup.get("full_value"))
                except (TypeError, ValueError):
                    pass
            if not holds:
                bad.append((cid, label, "chart label of a supporting value: the label does not hold the supported value"))
    return bad


def figure_strings(fig):
    """The reader-visible strings of one figure of the data file."""
    out = [fig.get("title", ""), fig.get("note", "")]
    for panel in fig.get("panels", []):
        out += [panel.get("x", {}).get("label", ""), panel.get("y", {}).get("label", "")]
        out += list(panel.get("x", {}).get("categories", []))
        out += [sr.get("name", "") for sr in panel.get("series", [])]
        out += [r.get("label", "") for r in panel.get("refs", [])]
        out += [panel.get("highlight", {}).get("label", "")]
    return [x for x in out if isinstance(x, str) and x]


def chart_text_coverage(data, claims_doc):
    """Numbers written in the titles, labels and notes of the chart data file, against the list kept per figure
    in the claims file (chart_text_numbers). A number typed into an axis (category) label must also have a
    supporting value marked "where": "chart" for that label and dataset, so that its value is compared with the
    internal evidence and its panel with its dataset (rule T02 makes the second half of that check).
    Returns ([(figure, number, context, why)], total)."""
    structural = claims_doc.get("structural_numbers", [])
    listed = claims_doc.get("chart_text_numbers", {})
    backed = set()
    for claim in claims_doc.get("claims", []):
        assets = claim.get("figure_asset")
        for fid in assets if isinstance(assets, list) else [assets]:
            for sup in claim.get("support", []):
                if sup.get("where") == "chart" and isinstance(fid, str):
                    backed.add((fid, sup.get("dataset"), str(sup.get("shown_as", ""))))
    bad, total = [], 0
    for fid, fig in (data or {}).get("figures", {}).items():
        for panel in fig.get("panels", []):
            for cat in panel.get("x", {}).get("categories", []):
                if not isinstance(cat, str):
                    continue
                spans = structural_spans(cat, structural)
                typed = [key for a, b, key in number_tokens(cat) if not any(s0 <= a and b <= e0 for s0, e0 in spans)]
                if typed and (fid, panel.get("title"), cat) not in backed:
                    bad.append((fid, ", ".join(typed), f"{panel.get('title')}: {cat}",
                                "number typed into an axis label without a supporting value (\"where\": \"chart\") for that "
                                "label and dataset in the claims file"))
        allowed = {str(x) for x in listed.get(fid, {}).get("numbers", [])}
        seen = set()
        for text in figure_strings(fig):
            spans = structural_spans(text, structural)
            for a, b, key in number_tokens(text):
                total += 1
                if any(s0 <= a and b <= e0 for s0, e0 in spans):
                    continue
                if key not in allowed and key not in seen:
                    bad.append((fid, text[a:b].strip(), excerpt(text, a, b), "number in a chart title, label or note that is not listed "
                                                                             "for this figure in chart_text_numbers of the claims file"))
                seen.add(key)
        for key in sorted(allowed - seen):
            bad.append((fid, key, "", "number listed for this figure in chart_text_numbers but not written in its strings (stale entry)"))
    for fid in sorted(set(listed) - set((data or {}).get("figures", {}))):
        bad.append((fid, "", "", "chart_text_numbers names a figure that the data file does not hold"))
    return bad, total


# --------------------------------------------------------------------------------------------------
# Collecting the site
# --------------------------------------------------------------------------------------------------


class Site:
    def __init__(self, root):
        self.root = Path(root)
        self.files = sorted(p for p in self.root.rglob("*") if p.is_file() or p.is_symlink())
        self.rel = {p: p.relative_to(self.root).as_posix() for p in self.files}
        self.units = []
        self.pages = {}       # rel -> PageParser
        self.css_refs = []    # (rel, line, url)
        self.images = {}      # rel -> (kind, chunks) or error string
        self.data = None      # parsed chart data
        self.data_error = None
        self.unreadable = []
        self.invisible = []   # (file, line, excerpt, why): invisible characters found (rule V09)
        for p in self.files:
            rel = self.rel[p]
            self.units.append(Unit(rel, 1, "filename", rel, what="file name"))
            if p.is_symlink():
                continue
            suffix = p.suffix.lower()
            if suffix in (".html", ".htm", ".svg"):
                self.read_html(p, rel)
            elif suffix == ".css":
                self.read_css(rel, self.text(p), 1, "style sheet")
            elif suffix == ".js":
                self.read_js(p, rel)
            elif suffix in (".webp", ".png", ".jpg", ".jpeg"):
                try:
                    kind, chunks = image_chunks(p)
                    self.images[rel] = (kind, chunks)
                    for name, text, _private in chunks:
                        if text:
                            self.units.append(Unit(rel, 1, "image-meta", text, what=f"{name} chunk"))
                except (ValueError, struct.error, IndexError) as err:
                    self.images[rel] = str(err)
            elif p.stat().st_size:
                # anything else that has content is read as text so that the vocabulary rules still see it
                try:
                    self.units.append(Unit(rel, 1, "text", self.text(p), what="file content"))
                except UnicodeDecodeError:
                    self.unreadable.append(rel)
        self.index = self.pages.get("index.html")
        self.strip_invisible()

    @staticmethod
    def text(p):
        return p.read_text(encoding="utf-8")

    def strip_invisible(self):
        """Removes invisible characters from every unit and records where they were (rule V09)."""
        rx = re.compile("[" + INVISIBLE + "]")
        seen = set()
        for u in self.units:
            if not rx.search(u.text):
                continue
            for m in rx.finditer(u.text):
                line = u.line_at(m.start())
                key = (u.file, line, m.group())
                if key not in seen:
                    seen.add(key)
                    shown = rx.sub("", excerpt(u.text, m.start(), m.end(), 24))
                    self.invisible.append((u.file, line, shown, f"invisible character U+{ord(m.group()):04X} in the {u.what or u.kind}"))
            if u.marks:
                # keep the line marks valid: positions shift left by the number of removed characters before them
                cut = [m.start() for m in rx.finditer(u.text)]
                u.marks = [(off - bisect.bisect_left(cut, off), ln) for off, ln in u.marks]
            u.text = rx.sub("", u.text)
            if u.spaced is not None:
                u.spaced = rx.sub("", u.spaced)

    def read_html(self, p, rel):
        source = self.text(p)
        parser = PageParser(rel)
        parser.feed(source)
        parser.close()
        self.pages[rel] = parser
        self.units.append(Unit(rel, 1, "raw", raw_markup(source), what="raw HTML source"))
        for u in parser.units:
            if u.kind == "js-code":
                self.units.append(u)
                self.add_tokens(u, js_tokens(u.text))
            elif u.kind == "css-code":
                self.units.append(u)
                self.add_tokens(u, css_tokens(u.text))
                for m in re.finditer(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)|@import\s+['\"]([^'\"]+)['\"]", u.text):
                    self.css_refs.append((rel, u.line + u.text.count("\n", 0, m.start()), m.group(1) or m.group(2)))
            else:
                self.units.append(u)

    def add_tokens(self, unit, tokens):
        for kind, text, off in tokens:
            self.units.append(Unit(unit.file, line_of(unit.text, off, unit.line), kind, text, what=kind.replace("-", " ")))

    def read_css(self, rel, text, line, what):
        u = Unit(rel, line, "css-code", text, what=what)
        self.units.append(u)
        self.add_tokens(u, css_tokens(text))
        for m in re.finditer(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)|@import\s+['\"]([^'\"]+)['\"]", text):
            self.css_refs.append((rel, line_of(text, m.start(), line), m.group(1) or m.group(2)))

    def read_js(self, p, rel):
        text = self.text(p)
        u = Unit(rel, 1, "js-code", text, what="script")
        self.units.append(u)
        if rel == "data/figures.js":
            self.data, self.data_error = parse_data_file(text)
            head = text[:text.find("window.DEXCORE_FIGURES")] if "window.DEXCORE_FIGURES" in text else ""
            self.add_tokens(Unit(rel, 1, "js-code", head), js_tokens(head))
            if self.data is not None:
                def walk(o):
                    if isinstance(o, dict):
                        for k, v in o.items():
                            yield k
                            yield from walk(v)
                    elif isinstance(o, list):
                        for v in o:
                            yield from walk(v)
                    elif isinstance(o, str):
                        yield o
                seen = set()
                for s in walk(self.data):
                    if s in seen:
                        continue
                    seen.add(s)
                    pos = text.find(json.dumps(s, ensure_ascii=False))
                    self.units.append(Unit(rel, line_of(text, pos) if pos >= 0 else 1, "data-string", s, what="string of the data file"))
        else:
            self.add_tokens(u, js_tokens(text))

    def names(self, kinds=None):
        files = sorted({u.file for u in self.units if kinds is None or u.kind in kinds})
        return files


# --------------------------------------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------------------------------------

PASS, FAIL, PENDING = "PASS", "FAIL", "NOT EVALUATED YET"
NO_INDEX = "site/index.html does not exist yet"


class Result:
    def __init__(self, rule_id, title, status, scanned, findings=(), note=""):
        self.id, self.title, self.status, self.scanned = rule_id, title, status, scanned
        self.findings = list(findings)    # (file, line, text, why)
        self.note = note


def scan(site, patterns, exempt=None):
    """All matches of (regex, why, kinds) patterns in the units of the site, without duplicates."""
    found, seen = [], set()
    for regex, why, kinds in patterns:
        rx = re.compile(regex)
        for u in site.units:
            if u.kind not in kinds:
                continue
            for m in rx.finditer(u.text):
                if exempt and exempt(u, m):
                    continue
                line = u.line_at(m.start())
                key = (u.file, line, m.group(), why)
                if key in seen:
                    continue
                seen.add(key)
                found.append((u.file, line, excerpt(u.text, m.start(), m.end()), f'"{m.group()}": {why}'))
    return sorted(found)


def scanned_all(site):
    kinds = {}
    for u in site.units:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    files = len(site.files)
    missing = "" if site.index else f"; {NO_INDEX}"
    return (f"{files} files: {kinds.get('text', 0)} text blocks ({kinds.get('text-spaced', 0)} read a second time with inline tags "
            f"as spaces), {kinds.get('attr-text', 0) + kinds.get('attr-id', 0)} attribute values, "
            f"{kinds.get('attr-name', 0)} attribute and {kinds.get('tag-name', 0)} element names, "
            f"{kinds.get('comment', 0)} HTML comments and declarations, {kinds.get('raw', 0)} raw HTML source(s), "
            f"{kinds.get('data-string', 0)} data-file strings, "
            f"{kinds.get('css-code', 0)} CSS and {kinds.get('js-code', 0)} script sources with "
            f"{kinds.get('css-comment', 0) + kinds.get('js-comment', 0)} comments and "
            f"{kinds.get('css-string', 0) + kinds.get('js-string', 0)} strings, "
            f"{kinds.get('image-meta', 0)} image metadata chunks, {kinds.get('filename', 0)} file names{missing}")


def identifier_parts(text):
    """(offset, lower-cased part) of every identifier part of a piece of code or of a name."""
    for run in re.finditer(r"[A-Za-z][A-Za-z0-9]*", text):
        for part in re.finditer(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", run.group()):
            yield run.start() + part.start(), part.group().lower()


def identifier_scan(site, rule_id):
    """Words of the internal modelling work inside identifiers (see IDENTIFIER_WORDS), for one rule."""
    found, seen = [], set()
    for u in site.units:
        if u.kind not in IDENTIFIER_KINDS or u.file == "data/figures.js":
            continue
        markup_name = (u.kind == "tag-name" and "-" in u.text) or (u.kind == "attr-name" and u.text.startswith("data-")) \
            or (u.kind == "attr-id" and re.match(r"(?:id|class) attribute", u.what))
        for off, part in identifier_parts(u.text):
            why = None
            if IDENTIFIER_WORDS.get(part) == rule_id:
                why = f'"{part}" inside an identifier: word of the internal modelling work'
            elif rule_id == "V05" and markup_name and part in IDENTIFIER_LETTERS:
                why = f'"{part}" as a part of a name in the markup: symbol of the learned state or of its realization code'
            if why:
                line = u.line_at(off)
                key = (u.file, line, part)
                if key not in seen:
                    seen.add(key)
                    found.append((u.file, line, excerpt(u.text, off, off + len(part), 30), f"{why} ({u.what or u.kind})"))
    return found


def vocabulary_rule(rule_id, title, site, patterns, exempt=None, extra=()):
    found = scan(site, patterns, exempt) + list(extra)
    return Result(rule_id, title, FAIL if found else PASS, scanned_all(site), sorted(set(found)))


def rule_s01(site):
    allowed = [re.compile(p) for p in ALLOWED_SITE_FILES]
    bad = []
    for p in site.files:
        rel = site.rel[p]
        if p.is_symlink():
            bad.append((rel, 1, rel, "symbolic link"))
        elif not any(a.fullmatch(rel) for a in allowed):
            bad.append((rel, 1, rel, "file that the public site is not expected to ship"))
    for rel in site.unreadable:
        bad.append((rel, 1, rel, "file that could not be read as text"))
    for d in sorted(p for p in site.root.rglob("*") if p.is_dir()):
        if d.name.startswith(".") or d.name == "__pycache__":
            bad.append((d.relative_to(site.root).as_posix(), 1, d.name, "hidden or cache directory"))
    listing = ", ".join(site.rel[p] for p in site.files)
    return Result("S01", "Only the expected files are shipped (no manifest, checklist, source data, code or hidden file)",
                  FAIL if bad else PASS, f"{len(site.files)} files: {listing}", bad)


def rule_s02(site):
    title = "The chart data file holds public figures only (no source list, module, internal id or unknown key)"
    rel = "data/figures.js"
    if rel not in site.rel.values():
        return Result("S02", title, FAIL, "site/data/figures.js", [(rel, 1, "", "the data file is missing")])
    if site.data is None:
        return Result("S02", title, FAIL, rel, [(rel, 1, "", site.data_error)])
    bad = []
    data = site.data
    if set(data.keys()) != {"figures"}:
        bad.append((rel, 1, ", ".join(sorted(data.keys())), "top-level keys other than 'figures'"))
    figures = data.get("figures", {})

    def keys(obj, kind, where):
        if not isinstance(obj, dict):
            bad.append((rel, 1, where, f"{kind} is not an object"))
            return
        extra = set(obj) - DATA_KEYS[kind]
        if extra:
            bad.append((rel, 1, where, f"unexpected key(s) {sorted(extra)} in a {kind}"))

    n_values = 0
    for fid, fig in figures.items():
        if not re.fullmatch(r"pub_[a-z0-9_]+", fid):
            bad.append((rel, 1, fid, "figure id that is not a public id (pub_...)"))
        keys(fig, "figure", fid)
        for panel in fig.get("panels", []) if isinstance(fig, dict) else []:
            where = f"{fid} / {panel.get('title')}"
            keys(panel, "panel", where)
            keys(panel.get("x", {}), "x", where)
            keys(panel.get("y", {}), "y", where)
            for s in panel.get("series", []):
                keys(s, "series", where)
                n_values += len(s.get("values", []))
            for r in panel.get("refs", []):
                keys(r, "ref", where)
            if "highlight" in panel:
                keys(panel["highlight"], "highlight", where)
    try:
        sys.path.insert(0, str(BUILD))
        import public_figures
        want = set(public_figures.FIGURES)
        if set(figures) != want:
            bad.append((rel, 1, ", ".join(sorted(set(figures) ^ want)), "figure ids differ from the public figure spec"))
    except ImportError:
        pass
    finally:
        if str(BUILD) in sys.path:
            sys.path.remove(str(BUILD))
    return Result("S02", title, FAIL if bad else PASS, f"{rel}: {len(figures)} figures, {n_values} plotted values", bad)


def rule_v01(site):
    return vocabulary_rule("V01", "No internal model, case or study code (B0-B2, D0-D2, S0-S2, M0-M3, M00-M11, A0-A3, F0-F2, "
                           "T1-T4, Case A-E, Stage 1 / 2, internal level names; lower-case forms in what a reader sees)", site, V01_CODES)


def rule_v02(site):
    title = "Level names R0-R5 appear only inside the one technical note (class tech-note)"
    # the raw source is left out here: it holds the technical note too, and every other piece of the markup
    # reaches this rule through the parser (text, attributes, comments, declarations)
    found = scan(site, [(V02_LEVELS, "internal level name outside a technical note", EVERYWHERE - {"raw"}),
                        (V02_LEVELS_LOWER, "internal level name (lower case) outside a technical note", READER)],
                 exempt=lambda u, m: u.tech)
    notes_with = sorted({(u.file, u.note_id) for u in site.units
                         if u.kind == "text" and u.tech and u.note_id is not None and re.search(V02_LEVELS, u.text)})
    if len(notes_with) > 1:
        for file, nid in notes_with[1:]:
            line = site.pages[file].notes[nid]["line"]
            found.append((file, line, "", "level names in more than one technical note (the plan allows one)"))
    return Result("V02", title, FAIL if found else PASS, scanned_all(site) + f"; technical notes that use level names: {len(notes_with)}", found)


def rule_v03(site):
    return vocabulary_rule("V03", "No identifier of the internal page (figure ids, evidence block ids, section names, render file names)",
                           site, V03_IDS)


def rule_v04(site):
    allowed = re.compile("|".join(f"(?:{p})" for p in SNAKE_ALLOWED))
    generic = scan(site, [(SNAKE_CASE, "code-like identifier (snake_case) in text", PROSE)],
                   exempt=lambda u, m: bool(allowed.fullmatch(m.group())))
    # id and class values, attribute and element names: a snake_case name there is an internal identifier too
    generic += scan(site, [(SNAKE_CASE, "code-like identifier (snake_case) in the markup", {"attr-id", "attr-name", "tag-name"})],
                    exempt=lambda u, m: bool(allowed.fullmatch(m.group()))
                    or (u.kind == "attr-id" and not re.match(r"(?:id|class) attribute", u.what)))
    found = scan(site, V04_SYMBOLS)
    seen = {(f, l) for f, l, _t, _w in found}
    found += [g for g in generic if (g[0], g[1]) not in seen]
    return Result("V04", "No internal symbol or code-like identifier (E_C, Q_pre, test_1, sequence ids, any snake_case name in the text or in id / class values)",
                  FAIL if found else PASS, scanned_all(site), sorted(found))


def rule_v05(site):
    extra = scan(site, [(V05_DECODER, "decoder outside a technical note", PROSE)], exempt=lambda u, m: u.tech)
    extra += identifier_scan(site, "V05")
    return vocabulary_rule("V05", "No learned latent state or realization code (latent, z, r, encoder; decoder only in a technical note), "
                           "in the text or inside an identifier", site, V05_LATENT, extra=extra)


def rule_v06(site):
    return vocabulary_rule("V06", "No training recipe or architecture detail (supervision, loss names and weights, sizes, architecture "
                           "names, seeds, step / iteration / epoch counts, run dates, checkpoints), in the text or inside an identifier",
                           site, V06_RECIPE, extra=identifier_scan(site, "V06"))


def rule_v07(site):
    return vocabulary_rule("V07", "No internal diagnostic, experiment or plan (oracle / teacher latent, train-test gap, decoder mismatch, "
                           "diffusion future, stateful vs whole-sequence, 2 x 2, residual decoder, next method, plans, TODO), "
                           "in the text or inside an identifier", site, V07_DIAGNOSTICS, extra=identifier_scan(site, "V07"))


def rule_v09(site):
    title = "No invisible character that could split a word (soft hyphen, zero-width space or joiner, word joiner, byte-order mark)"
    return Result("V09", title, FAIL if site.invisible else PASS, scanned_all(site), sorted(set(site.invisible)),
                  note="such characters are removed before every other rule reads the text")


def rule_v08(site):
    title = "Cross-check with the internal page: none of its figure ids, image names, evidence files or source paths appears"
    fig_json = DOCS / "data" / "figures.json"
    if not fig_json.is_file():
        return Result("V08", title, PENDING, "docs/data/figures.json", note="the internal figure file was not found, so this cross-check did not run")
    names = {}
    internal = json.loads(fig_json.read_text(encoding="utf-8"))
    for fid, fig in internal.get("figures", {}).items():
        names[fid] = "internal figure id"
        if fig.get("module"):
            names[fig["module"]] = "internal build module"
        for s in fig.get("sources", []):
            names[s["path"]] = "internal source path"
    for folder, why in (("assets/img", "internal image name"), ("evidence", "internal evidence file"), ("data/qual", "internal render metadata")):
        d = DOCS / folder
        if d.is_dir():
            for p in d.iterdir():
                if "_" in p.stem:
                    names[p.stem] = why
    manifest = DOCS / "evidence_manifest.json"
    if manifest.is_file():
        for block in json.loads(manifest.read_text(encoding="utf-8")).get("blocks", []):
            if "-" in block.get("block", ""):
                names[block["block"]] = "internal evidence block id"
    found = []
    for name, why in sorted(names.items()):
        rx = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
        for u in site.units:
            for m in rx.finditer(u.text):
                found.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), f'"{name}": {why}'))
    return Result("V08", title, FAIL if found else PASS, f"{len(names)} internal names against {scanned_all(site)}", sorted(set(found)))


def rule_p01(site):
    return vocabulary_rule("P01", "No server path or internal directory (/result, /ckpt, /home, /data, reports/, scripts/ ...)", site, P01_PATHS)


def rule_p02(site):
    def float_piece(u, m):
        return bool(re.fullmatch(r"[0-9]+e[0-9]*", m.group()))
    extra = scan(site, [(P02_HASH, "hexadecimal string that looks like a hash", EVERYWHERE)], exempt=float_piece)
    return vocabulary_rule("P02", "No file name of data, code, logs or documents (*.csv, *.npz, *.json, *.pt, *.py, *.log, *.md ...) and no hash",
                           site, P02_FILES, extra=extra)


def rule_p03(site):
    def contact_only(u, m):
        tail = u.text[m.end():m.end() + 2]
        head = u.text[max(0, m.start() - 11):m.start()].lower()
        return head.endswith("github.com/") and not re.match(r"/[\w.-]", tail)
    extra = scan(site, [(P03_USERNAME, "account name outside the contact line", EVERYWHERE)], exempt=contact_only)
    extra += scan(site, [(P03_EMAIL, "e-mail address", EVERYWHERE)], exempt=lambda u, m: m.group() in ALLOWED_EMAILS)
    # names known by their digest only
    digest_of, seen = {}, set()
    for u in site.units:
        for m in re.finditer(r"[A-Za-z0-9]+(?:[.-][A-Za-z0-9]+)*", u.text):
            token = m.group().lower()
            dotted = token.split(".")
            candidates = {token, *re.split(r"[.-]", token), *(".".join(dotted[i:]) for i in range(1, len(dotted)))}
            for c in candidates:
                if c not in digest_of:
                    digest_of[c] = hashlib.sha256(c.encode("utf-8")).hexdigest()
                why = P03_PRIVATE_DIGESTS.get(digest_of[c])
                if why:
                    line = u.line_at(m.start())
                    if (u.file, line, digest_of[c]) not in seen:
                        seen.add((u.file, line, digest_of[c]))
                        extra.append((u.file, line, f"(a name of {len(c)} characters, not repeated here)", f"{why} (matched by digest)"))
    # the owner's decision: no name on the page, so no author meta element either
    for rel, page in site.pages.items():
        for line, a in page.metas:
            if (a.get("name") or a.get("property") or "").lower() in ("author", "article:author", "creator", "publisher"):
                extra.append((rel, line, f'<meta name="{a.get("name") or a.get("property")}">', "author / publisher meta element"))
    return vocabulary_rule("P03", "No server or personal information (host, user and account names, a personal name or affiliation, "
                           "e-mail and IP addresses, author meta element)", site, P03_PERSONAL, extra=extra)


URL_RX = re.compile(r"(?i)\b(?:https?|ftp|wss?):\/\/[^\s\"'<>)\\]+|(?<![:\w/])\/\/[a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s\"'<>)\\]*")


def is_external(url):
    return bool(re.match(r"(?i)[a-z][a-z0-9+.-]*:", url)) or url.startswith("//")


def rule_n01(site):
    title = "No resource is loaded from the network (scripts, styles, images, fonts, frames, CSS url(), script requests)"
    bad = []
    n_refs = 0
    for rel, page in site.pages.items():
        for line, tag, why in page.forbidden_tags:
            bad.append((rel, line, f"<{tag}>", why))
        for tag, attr, value, line in page.refs:
            if tag in ("a", "area") and attr == "href":
                continue
            n_refs += 1
            for url in ([v.split()[0] for v in value.split(",") if v.split()] if attr in ("srcset", "imagesrcset") else [value.strip()]):
                if is_external(url):
                    bad.append((rel, line, f'<{tag} {attr}="{url}">', "resource with an absolute or data: address"))
    for rel, line, url in site.css_refs:
        n_refs += 1
        if is_external(url.strip()):
            bad.append((rel, line, f"url({url})", "style sheet resource with an absolute or data: address"))
    # also code that could build an element or an address at run time (a script element with a composed src)
    api = re.compile(r"\b(?:fetch|importScripts)\s*\(|\bXMLHttpRequest\b|\bWebSocket\b|\bEventSource\b|\bsendBeacon\b|\bimport\s*\(|"
                     r"\bnew\s+Image\b|\.src\s*=|\bnavigator\.serviceWorker\b|"
                     r"\bcreateElement(?:NS)?\s*\([^)]*[\"'](?:script|link|iframe|img|image|object|embed|video|audio|source|use)[\"']|"
                     r"\bsetAttribute(?:NS)?\s*\([^)]*[\"'](?:src|href|xlink:href|srcset|data|action)[\"']|"
                     r"\.(?:href|srcset|action)\s*=(?!=)|\bdocument\.write(?:ln)?\b|\binsertAdjacentHTML\b|\b(?:inner|outer)HTML\s*=(?!=)|"
                     r"\blocation\s*(?:=(?!=)|\.(?:assign|replace)\s*\()|\bwindow\.open\s*\(|\beval\s*\(|\bnew\s+Function\b")
    for u in site.units:
        if u.kind == "js-code":
            for m in api.finditer(u.text):
                bad.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "script code that can request a resource"))
        if u.kind in ("js-code", "css-code", "js-string", "css-string"):
            for m in URL_RX.finditer(u.text):
                if m.group().rstrip(".,;") in XML_NAMESPACES:
                    continue
                bad.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "network address in code"))
    return Result("N01", title, FAIL if bad else PASS,
                  f"{n_refs} resource references in {len(site.pages)} HTML file(s) and the style sheets; every script and style source",
                  sorted(set(bad)))


def rule_n02(site):
    title = "Every referenced local file and anchor exists; references are relative and stay inside the site"
    bad, n = [], 0
    existing = set(site.rel.values())

    def local(rel_from, url, line, what):
        nonlocal n
        n += 1
        url = url.strip()
        if not url or is_external(url):
            return
        path, _, frag = url.partition("#")
        path = path.split("?")[0]
        if path.startswith("/"):
            bad.append((rel_from, line, url, f"{what}: root-relative address (the site must work from any folder)"))
            return
        base = Path(rel_from).parent
        target = rel_from if not path else (base / path).as_posix()
        norm, parts = [], target.split("/")
        for part in parts:
            if part == "..":
                if not norm:
                    bad.append((rel_from, line, url, f"{what}: leaves the site folder"))
                    return
                norm.pop()
            elif part not in (".", ""):
                norm.append(part)
        target = "/".join(norm)
        if path and target not in existing:
            bad.append((rel_from, line, url, f"{what}: file not found in the site"))
            return
        if frag and frag != "top" and target in site.pages and frag not in site.pages[target].ids:
            bad.append((rel_from, line, url, f"{what}: no element with id \"{frag}\""))

    for rel, page in site.pages.items():
        for tag, attr, value, line in page.refs:
            urls = [v.split()[0] for v in value.split(",") if v.split()] if attr in ("srcset", "imagesrcset") else [value]
            for url in urls:
                local(rel, url, line, f"<{tag} {attr}>")
        for u in page.units:
            if u.kind in ("attr-id", "css-code"):
                for m in re.finditer(r"url\(\s*#([^)\s]+)\s*\)", u.text):
                    n += 1
                    if m.group(1) not in page.ids:
                        bad.append((rel, u.line, m.group(), f"no element with id \"{m.group(1)}\""))
    for rel, line, url in site.css_refs:
        if not url.strip().startswith("#"):
            local(rel, url, line, "url() in a style sheet")
    return Result("N02", title, FAIL if bad else PASS, f"{n} local references in {len(site.pages)} HTML file(s) and the style sheets", sorted(set(bad)))


def link_allowed(url):
    u = url.strip()
    if u in CONTACT_URLS:
        return True
    return any(u.startswith(prefix) for prefix in OUTGOING_LINK_ALLOWLIST)


def rule_n03(site):
    title = "Outgoing links go only to the allowlisted public pages (datasets, cited papers, the contact address)"
    bad, n = [], 0
    for rel, page in site.pages.items():
        for tag, attr, value, line in page.refs:
            if not (tag in ("a", "area") and attr == "href"):
                continue
            url = value.strip()
            if not is_external(url):
                continue
            n += 1
            if re.match(r"(?i)mailto:", url):
                if url[7:].split("?")[0] not in ALLOWED_EMAILS:
                    bad.append((rel, line, url, "mail link to an address other than the contact address"))
            elif not link_allowed(url):
                bad.append((rel, line, url, "outgoing link that is not on the allowlist of this script"))
        for u in page.units:
            if u.kind in ("text", "attr-text", "comment"):
                for m in URL_RX.finditer(u.text):
                    n += 1
                    if not link_allowed(m.group().rstrip(".,;")):
                        bad.append((rel, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "address in the text that is not on the allowlist"))
    for u in site.units:
        if u.kind == "data-string":
            for m in URL_RX.finditer(u.text):
                n += 1
                bad.append((u.file, u.line, excerpt(u.text, m.start(), m.end()), "address in the chart data"))
    return Result("N03", title, FAIL if bad else PASS,
                  f"{n} outgoing addresses in links, text and chart data; allowlist of {len(OUTGOING_LINK_ALLOWLIST) + 1} entries", sorted(set(bad)))


def rule_n04(site):
    extra = scan(site, [(N04_REPO_NAME, "name of the code repository", EVERYWHERE)],
                 exempt=lambda u, m: u.text[m.start():m.start() + 15] == "DEXCORE_FIGURES")
    return vocabulary_rule("N04", "No link or reference to the internal page, docs/, the evidence manifest or the code repository",
                           site, N04_INTERNAL, extra=extra)


def html_prose(site):
    """Wording rules look at what a reader sees: page text, text attributes, comments and the chart strings."""
    return READER


def rule_w01(site):
    kinds = html_prose(site)
    found = scan(site, [(rx, why, kinds) for rx, why in W01_WORDING])
    n = sum(1 for u in site.units if u.kind in kinds)
    return Result("W01", "Wording: none of prove, solve, final method, state-of-the-art, human intent, noise, true task functionality, "
                  "optimal, necessary, disentangled, our method, outperform, we show, best", FAIL if found else PASS,
                  f"{n} pieces of reader-visible text (page text, text attributes, comments, chart strings)" + ("" if site.index else f"; {NO_INDEX}"),
                  found)


def rule_w02(site):
    title = "'Null-space' appears at most once, as 'null-space-like', inside an element of class intuition"
    if not site.index:
        return Result("W02", title, PENDING, "site/index.html", note=NO_INDEX)
    hits = []
    rx = re.compile(W02_NULLSPACE)
    for u in site.units:
        if u.kind in html_prose(site):
            for m in rx.finditer(u.text):
                hits.append((u, m))
    bad = []
    for u, m in hits:
        if not u.intuition:
            bad.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "outside an element of class intuition"))
        elif not re.match(r"(?i)null[- ]?space-like", u.text[m.start():]):
            bad.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "not worded as 'null-space-like'"))
    for u, m in hits[1:]:
        bad.append((u.file, u.line_at(m.start()), excerpt(u.text, m.start(), m.end()), "more than one occurrence"))
    return Result("W02", title, FAIL if bad else PASS, f"reader-visible text; occurrences: {len(hits)}", sorted(set(bad)))


def words(text):
    return [w for w in text.split() if re.search(r"[^\W_]", w)]


def counted_words(unit):
    """Number of words of a text block as a reader sees them: counted on the reading that has a space at every
    inline-tag boundary. Adjacent inline elements written without a space between them (the dataset chips and the
    source label of a row) are separate words; a word is anything between spaces that holds a letter or a digit.
    A symbol written with a subscript therefore counts as two words: the count is never below what is rendered."""
    return len(words(unit.spaced if unit.spaced is not None else unit.text))


def rule_w03(site):
    title = "The page is labelled as ongoing research / work in progress"
    if not site.index:
        return Result("W03", title, PENDING, "site/index.html", note=NO_INDEX)
    text = " ".join(u.text for u in site.index.units if u.kind == "text" and u.counted)
    bad = []
    if not re.search(r"(?i)ongoing research", text):
        bad.append(("index.html", 1, "", "the words 'Ongoing research' are not in the page text"))
    if not re.search(r"(?i)work in progress", text):
        bad.append(("index.html", 1, "", "the words 'Work in progress' are not in the page text"))
    return Result("W03", title, FAIL if bad else PASS, "visible text of index.html", bad)


def rule_h01(site):
    title = (f"Body text is at most {MAX_BODY_WORDS:,} words (visible text outside technical notes, every inline element "
             "counted as separate words; alt texts not counted)")
    if not site.index:
        return Result("H01", title, PENDING, "site/index.html", note=NO_INDEX)
    body = [u for u in site.index.units if u.kind == "text" and u.counted]
    n = sum(counted_words(u) for u in body)
    prose = sum(counted_words(u) for u in body if not u.aside)
    bad = [("index.html", 1, "", f"{n:,} words")] if n > MAX_BODY_WORDS else []
    return Result("H01", title, FAIL if bad else PASS,
                  f"index.html: {n:,} words of visible body text; {prose:,} of them are running prose (headings, paragraphs, "
                  f"list items and captions) and {n - prose:,} are navigation, chips, source labels, stat tiles, table cells and "
                  "diagram labels", bad)


def rule_h02(site):
    title = f"Each technical note is at most {MAX_NOTE_WORDS} words (its summary line included; counted as in H01)"
    if not site.index:
        return Result("H02", title, PENDING, "site/index.html", note=NO_INDEX)
    page = site.index
    counts = [0] * len(page.notes)
    for u in page.units:
        if u.kind == "text" and u.note_id is not None:
            counts[u.note_id] += counted_words(u)
    bad = []
    for note, n in zip(page.notes, counts):
        if n > MAX_NOTE_WORDS:
            bad.append(("index.html", note["line"], "", f"technical note of {n} words"))
        if note["tag"] != "details":
            bad.append(("index.html", note["line"], f"<{note['tag']}>", "a technical note must be a collapsed <details class=\"tech-note\">"))
    return Result("H02", title, FAIL if bad else PASS,
                  f"index.html: {len(counts)} technical note(s) of {', '.join(map(str, counts)) or '0'} words", bad)


def rule_h03(site):
    title = "Exactly one h1; heading levels do not skip"
    if not site.index:
        return Result("H03", title, PENDING, "site/index.html", note=NO_INDEX)
    hs = site.index.headings
    bad = []
    h1 = [line for level, line in hs if level == 1]
    if len(h1) != 1:
        bad.append(("index.html", h1[1] if len(h1) > 1 else 1, "", f"{len(h1)} h1 elements"))
    prev = 0
    for level, line in hs:
        if prev and level > prev + 1:
            bad.append(("index.html", line, f"h{level} after h{prev}", "heading level skipped"))
        prev = level
    return Result("H03", title, FAIL if bad else PASS, f"index.html: {len(hs)} headings", bad)


def rule_h04(site):
    title = "Every image has alt text; every inline SVG is either aria-hidden or has role=\"img\" and an aria-label"
    if not site.index:
        return Result("H04", title, PENDING, "site/index.html", note=NO_INDEX)
    bad = []
    for line, a in site.index.imgs:
        if not a.get("alt", "").strip():
            bad.append(("index.html", line, a.get("src", ""), "img without alt text"))
    for line, a in site.index.svgs:
        if a.get("aria-hidden") == "true":
            continue
        if a.get("role") != "img" or not a.get("aria-label", "").strip():
            bad.append(("index.html", line, "<svg>", "inline SVG without role=\"img\" and aria-label"))
    return Result("H04", title, FAIL if bad else PASS, f"index.html: {len(site.index.imgs)} images, {len(site.index.svgs)} inline SVGs", bad)


def rule_h05(site):
    title = "Every chart placeholder names a figure of the data file, every figure of the data file is shown, and the data file is loaded before the chart script"
    if not site.index:
        return Result("H05", title, PENDING, "site/index.html", note=NO_INDEX)
    page, bad = site.index, []
    figures = (site.data or {}).get("figures", {})
    for line, fid in page.figs:
        if fid not in figures:
            bad.append(("index.html", line, fid, "data-fig that is not in the data file"))
    srcs = [s[2:] if s.startswith("./") else s for _line, s in page.scripts if s]
    if page.figs:
        if "data/figures.js" not in srcs or not any(s.endswith("charts.js") for s in srcs):
            bad.append(("index.html", 1, ", ".join(srcs), "the page has charts but does not load data/figures.js and the chart script"))
        elif srcs.index("data/figures.js") > [i for i, s in enumerate(srcs) if s.endswith("charts.js")][0]:
            bad.append(("index.html", 1, ", ".join(srcs), "the chart script is loaded before the data file"))
    used = {fid for _line, fid in page.figs}
    for fid in sorted(set(figures) - used):
        bad.append(("data/figures.js", 1, fid, "figure of the data file that no chart of the page shows (data shipped without a reader)"))
    return Result("H05", title, FAIL if bad else PASS, f"index.html: {len(page.figs)} chart placeholders, {len(srcs)} script files; "
                  f"data file: {len(figures)} figures", bad)


def rule_i01(site):
    title = "Images carry no EXIF / XMP block and no text or comment chunk, and the text of their other metadata passes every rule above"
    bad, listing = [], []
    for rel, info in sorted(site.images.items()):
        if isinstance(info, str):
            bad.append((rel, 1, "", f"image could not be read: {info}"))
            continue
        kind, chunks = info
        listing.append(f"{rel} ({kind}: {', '.join(sorted({c[0].strip() for c in chunks}))})")
        for name, text, private in chunks:
            if private:
                bad.append((rel, 1, (text or "")[:120], f"{name.strip()} metadata block: strip it before shipping"))
    for p in site.files:
        rel = site.rel[p]
        if p.suffix.lower() in (".gif", ".avif", ".bmp", ".tif", ".tiff", ".heic"):
            bad.append((rel, 1, rel, "image type whose metadata this script cannot read"))
    return Result("I01", title, FAIL if bad else PASS, "; ".join(listing) or "no raster image", bad,
                  note="the text found in metadata chunks is scanned by the vocabulary, path and wording rules")


def rule_i02(site):
    title = "Every image is a version whose drawn text was inspected by eye (hash recorded in this script)"
    bad, n = [], 0
    for p in site.files:
        rel = site.rel[p]
        if p.suffix.lower() not in (".webp", ".png", ".jpg", ".jpeg", ".svg") or p.is_symlink():
            continue
        if p.suffix.lower() == ".svg":
            continue    # SVG text is read by the text rules
        n += 1
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        want = INSPECTED_IMAGES.get(rel)
        if want is None:
            bad.append((rel, 1, digest, "image that is not in the list of inspected images"))
        elif want != digest:
            bad.append((rel, 1, digest, "image changed since it was inspected: look at it again, then record this hash"))
    return Result("I02", title, FAIL if bad else PASS, f"{n} raster image(s) against {len(INSPECTED_IMAGES)} recorded hashes", bad)


def load_claims(claims_path):
    """(claims document, None) or (None, error text)."""
    if not Path(claims_path).is_file():
        return None, "the claims file was not found"
    try:
        return json.loads(Path(claims_path).read_text(encoding="utf-8")), None
    except ValueError as err:
        return None, f"the claims file is not valid JSON: {err}"


def rule_t01(site, claims_path):
    title = ("Every number in the visible text of the page lies inside the text of a claim that lists it, "
             "or is a listed structural number")
    if not site.index:
        return Result("T01", title, PENDING, "site/index.html", note=NO_INDEX)
    claims_doc, err = load_claims(claims_path)
    if err:
        return Result("T01", title, FAIL, str(claims_path), [("index.html", 1, "", err)])
    try:
        uncovered, used, total = number_coverage(site.index.units, claims_doc)
    except re.error as err:
        return Result("T01", title, FAIL, str(claims_path), [("index.html", 1, "", f"the claims file could not be used: {err}")])
    bad = [(u.file, u.line_at(a), excerpt(u.text, a, b),
            f'"{u.text[a:b].strip()}": number outside the text of a claim that lists it') for u, a, b, _key in uncovered]
    return Result("T01", title, FAIL if bad else PASS,
                  f"index.html: {total} numbers in the text, alt texts and labels; {len(used)} distinct displayed numbers matched to claims", bad)


def rule_t02(site, claims_path):
    title = ("Every claim of the claims file is on the page word for word (its sentence and its listed tile strings), "
             "every displayed number is in its claim's text or is plotted in the chart the claim names, and every number typed "
             "into a chart label that has a supporting value stands in the panel of its dataset")
    if not site.index:
        return Result("T02", title, PENDING, "site/index.html", note=NO_INDEX)
    claims_doc, err = load_claims(claims_path)
    if err:
        return Result("T02", title, FAIL, str(claims_path), [("index.html", 1, "", err)])
    problems = claims_on_page(site.index.units, claims_doc, site.data)
    bad = [("index.html", 1, text, f"claim {cid}: {why}") for cid, text, why in problems]
    n_claims = len(claims_doc.get("claims", []))
    n_numbers = sum(len(c.get("numbers", [])) for c in claims_doc.get("claims", []))
    n_labels = sum(1 for c in claims_doc.get("claims", []) for sup in c.get("support", []) if sup.get("where") == "chart")
    return Result("T02", title, FAIL if bad else PASS, f"{n_claims} claims with {n_numbers} displayed numbers and {n_labels} numbers "
                  "typed into chart labels against the text of index.html and the chart data file", bad)


def rule_t03(site, claims_path):
    title = ("Every number written in a title, label or note of the chart data file is listed for that figure in the claims file "
             "(or is a structural number), and a number typed into an axis label has a supporting value for that label and dataset")
    if site.data is None:
        return Result("T03", title, FAIL, "data/figures.js", [("data/figures.js", 1, "", site.data_error or "the data file is missing")])
    claims_doc, err = load_claims(claims_path)
    if err:
        return Result("T03", title, FAIL, str(claims_path), [("data/figures.js", 1, "", err)])
    try:
        problems, total = chart_text_coverage(site.data, claims_doc)
    except re.error as err:
        return Result("T03", title, FAIL, str(claims_path), [("data/figures.js", 1, "", f"the claims file could not be used: {err}")])
    bad = [("data/figures.js", 1, f"{fid}: {number} {context}".strip(), why) for fid, number, context, why in problems]
    return Result("T03", title, FAIL if bad else PASS,
                  f"data/figures.js: {total} numbers in the titles, labels and notes of {len(site.data.get('figures', {}))} figures", bad)


# ---- manual checks ---------------------------------------------------------------------------------
# What a script cannot judge is checked by a reader and recorded in build/manual_checks.json, with the name of
# who made the check (the owner, or the agent that prepared the page: the record says which). A text check is
# pinned to the text it was made on: when that text changes, the record is stale and rule M01 fails until the
# text has been read again and the new digest recorded (the same idea as rule I02 for the images).
MANUAL_CHECKS = {
    # id: (scope of the pinned text or None, what was checked)
    "page-text": ("page", "the whole page text and the chart titles, labels and notes were read: a reader leaves with the four "
                          "take-aways of the brief (section 18); no description of the internal modelling work, of a plan or of a "
                          "to-do in other words"),
    "direction-wording": ("#direction", "the section 'Current research direction' says only what the brief allows (section 8): the "
                                        "question, the conceptual diagram, one sentence on what is investigated, no method, no result"),
    "image-text": (None, "the text drawn inside the pictures was read: no internal code, name, path or file name (rule I02 pins the "
                         "inspected versions)"),
    "render": (None, "the page was looked at in a browser at 1280 px and at 390 px: figures legible, no horizontal page scroll"),
}


def pinned_text(site, scope):
    """The text a manual check is pinned to, normalised: one piece per line, single spaces."""
    if not site.index:
        return ""
    pieces = []
    for u in site.index.units:
        if u.kind not in ("text", "attr-text"):
            continue
        if scope.startswith("#") and u.section != scope[1:]:
            continue
        pieces.append(re.sub(r"\s+", " ", u.text).strip())
    if scope == "page":
        for fid, fig in sorted((site.data or {}).get("figures", {}).items()):
            pieces += [fid] + [re.sub(r"\s+", " ", x).strip() for x in figure_strings(fig)]
    return "\n".join(x for x in pieces if x)


def rule_m01(site, manual_path):
    title = "The checks that a script cannot make are recorded (what, when, by whom), and each text check was made on the text as it is now"
    if not site.index:
        return Result("M01", title, PENDING, "site/index.html", note=NO_INDEX)
    scanned = f"{shown_path(manual_path)}: {len(MANUAL_CHECKS)} required records"
    if not Path(manual_path).is_file():
        return Result("M01", title, FAIL, scanned, [("index.html", 1, "", f"{shown_path(manual_path)} was not found")])
    try:
        doc = json.loads(Path(manual_path).read_text(encoding="utf-8"))
    except ValueError as err:
        return Result("M01", title, FAIL, scanned, [("index.html", 1, "", f"{shown_path(manual_path)} is not valid JSON: {err}")])
    records = {c.get("id"): c for c in doc.get("checks", []) if isinstance(c, dict)}
    bad, lines = [], []
    for cid, (scope, what) in MANUAL_CHECKS.items():
        rec = records.get(cid)
        if rec is None:
            bad.append(("index.html", 1, cid, f"manual check not recorded: {what}"))
            continue
        if rec.get("result") != "pass":
            bad.append(("index.html", 1, cid, f"manual check recorded with result {rec.get('result')!r} (must be 'pass')"))
        if not re.fullmatch(r"20[0-9]{2}-[01][0-9]-[0-3][0-9]", str(rec.get("date", ""))) or not str(rec.get("by", "")).strip():
            bad.append(("index.html", 1, cid, "manual check without a date (YYYY-MM-DD) or without the name of who made it"))
        if scope:
            digest = hashlib.sha256(pinned_text(site, scope).encode("utf-8")).hexdigest()
            if rec.get("text_sha256") != digest:
                bad.append(("index.html", 1, digest, f"manual check {cid}: the text changed since it was read. Read it again "
                                                     f"({what}), then record this digest as text_sha256"))
        lines.append(f"{cid} ({rec.get('date')}, {rec.get('by')}: {rec.get('result')})")
    return Result("M01", title, FAIL if bad else PASS, scanned + ("; " + "; ".join(lines) if lines else ""), bad)


def all_rules(site, claims_path, manual_path=MANUAL):
    return [
        rule_s01(site), rule_s02(site),
        rule_v01(site), rule_v02(site), rule_v03(site), rule_v04(site), rule_v05(site), rule_v06(site), rule_v07(site), rule_v08(site),
        rule_v09(site),
        rule_p01(site), rule_p02(site), rule_p03(site),
        rule_n01(site), rule_n02(site), rule_n03(site), rule_n04(site),
        rule_w01(site), rule_w02(site), rule_w03(site),
        rule_h01(site), rule_h02(site), rule_h03(site), rule_h04(site), rule_h05(site),
        rule_i01(site), rule_i02(site),
        rule_t01(site, claims_path), rule_t02(site, claims_path), rule_t03(site, claims_path),
        rule_m01(site, manual_path),
    ]


def shown_path(path):
    """A path as it is written in the checklist: relative to the repository when it lies inside it."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_checklist(results, site, out):
    n_fail = sum(r.status == FAIL for r in results)
    n_pending = sum(r.status == PENDING for r in results)
    n_pass = sum(r.status == PASS for r in results)
    # no time stamp: the file is the same whenever the same site is checked, so a rebuild leaves the repository clean
    lines = [
        "# Exposure checklist of the public page",
        "",
        f"Written by `build/check_public.py`. Scanned folder: `{shown_path(site.root)}` ({len(site.files)} files).",
        "",
        f"**Result: {'FAIL' if n_fail else 'PASS'}** ({n_pass} rules passed, {n_fail} failed, {n_pending} not evaluated yet).",
        "",
    ]
    if not site.index:
        lines += [f"`{NO_INDEX}`: the rules marked NOT EVALUATED YET need it, and every PASS below covers only the files that exist now.", ""]
    lines += ["One line per rule: result, rule, what was scanned; under a failed rule, the offending file, line and text.", ""]
    for r in results:
        lines.append(f"- **{r.status}** · {r.id} · {r.title} · scanned: {r.scanned}" + (f" · {r.note}" if r.note else ""))
        for file, line, text, why in r.findings:
            shown = f" · `{text}`" if text else ""
            lines.append(f"    - `{file}:{line}`{shown} · {why}")
    lines += [
        "",
        "## What the script cannot check",
        "",
        "The vocabulary rules find the listed words, codes, names and paths. A description of the internal modelling work, "
        "of a plan or of a to-do **in other words** is not machine-checkable, and neither is the text drawn inside a picture. "
        "These are checked by a reader and recorded in `build/manual_checks.json` (rule M01): each record names who looked "
        "(the line of rule M01 above repeats it; a record made by an agent is a first reading, not the owner's sign-off), "
        "when, and for the two text checks the digest of the text that was read, so a later change of the text makes the "
        "record stale and fails the rule.",
        "",
    ]
    for cid, (scope, what) in MANUAL_CHECKS.items():
        lines.append(f"- {cid}: {what}." + (f" Pinned to the text of {'the whole page' if scope == 'page' else 'the section ' + scope}." if scope else ""))
    lines.append("")
    Path(out).write_text("\n".join(lines), encoding="utf-8")


def run(site_dir=SITE, out=CHECKLIST, claims_path=CLAIMS, quiet=False, manual_path=MANUAL):
    site = Site(site_dir)
    results = all_rules(site, claims_path, manual_path)
    write_checklist(results, site, out)
    n_fail = sum(r.status == FAIL for r in results)
    if not quiet:
        for r in results:
            print(f"  {r.status:<17} {r.id}  {r.title}")
            for file, line, text, why in r.findings[:12]:
                print(f"      {file}:{line}  {text[:110]}  <- {why}")
            if len(r.findings) > 12:
                print(f"      ... {len(r.findings) - 12} more in the checklist")
        n_pending = sum(r.status == PENDING for r in results)
        print(f"check_public: {'FAIL' if n_fail else 'PASS'} ({len(results) - n_fail - n_pending} passed, {n_fail} failed, "
              f"{n_pending} not evaluated yet) -> {shown_path(out)}")
    return (1 if n_fail else 0), results


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", default=str(SITE), help="folder of the static site (default: public_page/site)")
    ap.add_argument("--out", default=str(CHECKLIST), help="checklist file to write (default: public_page/exposure_checklist.md)")
    ap.add_argument("--claims", default=str(CLAIMS), help="claims file used by the traceability rules")
    ap.add_argument("--manual", default=str(MANUAL), help="record of the manual checks (default: build/manual_checks.json)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--pins", action="store_true", help="print the digests of the texts that the manual checks are pinned to, and stop")
    args = ap.parse_args()
    if args.pins:
        site = Site(args.site)
        for cid, (scope, _what) in MANUAL_CHECKS.items():
            if scope:
                print(f"{cid}: {hashlib.sha256(pinned_text(site, scope).encode('utf-8')).hexdigest()}")
        return 0
    code, _ = run(args.site, args.out, args.claims, args.quiet, args.manual)
    return code


if __name__ == "__main__":
    sys.exit(main())
