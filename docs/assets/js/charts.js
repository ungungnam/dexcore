/* Chart renderer for the project page.
 *
 * Every <figure class="viz" data-fig="<id>"> is filled from window.DEXCORE_FIGURES (written by
 * docs/build/build_figures.py). The renderer draws what the data file holds and nothing else:
 * it rounds for display only, never rescales, pools or reorders values.
 *
 * Optional attributes on the <figure>:
 *   data-panels="TACO,ARCTIC"   show only these panels, in this order
 *   data-series="a|b"           show only these series, in this order
 *   data-emphasis="name"        this series (or bar category) in colour, the rest in grey
 *   data-labels="all|none"      value labels on marks (default: only when a panel has few marks);
 *                               labels that would touch each other are never drawn
 *   data-format="pct"           display values as percentages (value x 100)
 *   data-height="220"           plot height in px
 *   data-shared-y               all panels use one y range (for panels that show the same quantity)
 *   data-refs="label|label"     draw only these reference lines
 */
(function () {
  "use strict";

  var SVG_NS = "http://www.w3.org/2000/svg";
  var SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4"];
  var MARKERS = ["circle", "square", "diamond", "triangle"];
  var DASHES = ["", "6 4", "2 3", "8 3 2 3"]; // line series after the first are dashed: identity never rests on colour alone
  var SLANTS = [35, 50, 65, 90]; // angles tried, in this order, for x labels that do not fit under their tick
  var FONT = 11.5;
  var CHAR_W = 6.1; // average glyph width at FONT px; a fallback only, label space is measured (textW)

  function svgEl(name, attrs, text) {
    var node = document.createElementNS(SVG_NS, name);
    if (attrs) Object.keys(attrs).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function htmlEl(name, className, text) {
    var node = document.createElement(name);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function isNum(v) { return typeof v === "number" && isFinite(v); }

  /* Display rounding only. */
  function fmt(v, pct) {
    if (!isNum(v)) return "–";
    if (pct) return fmtPlain(v * 100) + "%";
    return fmtPlain(v);
  }

  function fmtPlain(v) {
    var a = Math.abs(v);
    var s;
    if (a === 0) s = "0";
    else if (a >= 1000) s = Math.round(v).toLocaleString("en-US");
    else if (a >= 100) s = v.toFixed(0);
    else if (a >= 10) s = v.toFixed(1);
    else if (a >= 1) s = v.toFixed(2);
    else if (a >= 0.01) s = v.toFixed(3);
    else s = v.toPrecision(2);
    return s.replace("-", "−");
  }

  /* Axis ticks share one number of decimals, chosen from the tick step. */
  function tickFormatter(ticks, pct) {
    var step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
    if (pct) step *= 100;
    var d = 0;
    while (d < 6 && Math.abs(step * Math.pow(10, d) - Math.round(step * Math.pow(10, d))) > 1e-9) d++;
    return function (t) {
      var x = pct ? t * 100 : t;
      var str = Math.abs(x) >= 1000 ? Math.round(x).toLocaleString("en-US") : x.toFixed(d);
      return str.replace("-", "−") + (pct ? "%" : "");
    };
  }

  function fmtTable(v, pct) {
    if (!isNum(v)) return "–";
    var x = pct ? v * 100 : v;
    var s = Math.abs(x) >= 100 ? x.toFixed(1) : x.toPrecision(4);
    return s.replace("-", "−") + (pct ? "%" : "");
  }

  /* The same rounding as fmt, without padding zeros ("70%" rather than "70.0%"): for the values of reference lines. */
  function fmtShort(v, pct) {
    var s = fmt(v, pct);
    return s.indexOf(".") < 0 ? s : s.replace(/\.?0+(%?)$/, "$1");
  }

  /* An x position as it is written in a heading or a row header: whole numbers as they are. */
  function fmtX(v) {
    if (typeof v !== "number") return String(v);
    return Number.isInteger(v) ? String(v) : fmtPlain(v);
  }

  /* "Nice" axis ticks covering [lo, hi]. */
  function niceTicks(lo, hi, count) {
    if (lo === hi) { hi = lo + 1; }
    var span = hi - lo;
    var raw = span / Math.max(1, count);
    var pow = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / pow;
    var step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * pow;
    var start = Math.floor(lo / step) * step;
    var end = Math.ceil(hi / step) * step;
    var ticks = [];
    for (var v = start; v <= end + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
    return ticks;
  }

  function cssVar(el, name) { return getComputedStyle(el).getPropertyValue(name).trim(); }

  /* ---------- text measurement ---------- */
  // Label space is laid out from the rendered width of the text in its own class (the browser's font),
  // not from a per-character estimate. One hidden <svg> on the page does the measuring.
  var ruler = null;
  var rulerCache = {};
  function textW(text, cls) {
    text = String(text);
    var key = cls + "|" + text;
    if (rulerCache[key] !== undefined) return rulerCache[key];
    var w = 0;
    try {
      if (!ruler) {
        var box = svgEl("svg", { class: "viz-svg viz-ruler", width: 1, height: 1, "aria-hidden": "true" });
        ruler = svgEl("text", {});
        box.appendChild(ruler);
        document.body.appendChild(box);
      }
      ruler.setAttribute("class", cls);
      ruler.textContent = text;
      w = ruler.getComputedTextLength();
    } catch (err) { w = 0; }
    if (!(w > 0)) return text.length * CHAR_W * 1.2; // no layout available: a generous estimate, not cached
    rulerCache[key] = w;
    return w;
  }

  /* Greedy word wrap to a width in px, measured in the class the text is drawn in. */
  function wrapLabel(text, maxW, cls) {
    var words = String(text).split(/\s+/);
    var lines = [];
    var line = "";
    words.forEach(function (w) {
      if (!line) line = w;
      else if (textW(line + " " + w, cls) <= maxW) line += " " + w;
      else { lines.push(line); line = w; }
    });
    if (line) lines.push(line);
    return lines;
  }

  function widestLine(lines, cls) {
    return Math.max.apply(null, lines.map(function (t) { return textW(t, cls); }));
  }

  /* Label of the shaded band that holds x ("" when x lies in none). */
  function bandLabel(bands, x) {
    for (var i = 0; i < bands.length; i++) {
      if (bands[i].label && x >= bands[i].x0 && x <= bands[i].x1) return bands[i].label;
    }
    return "";
  }

  /* ---------- tooltip (one for the page) ---------- */
  var tip = null;
  function showTip(evtOrRect, rows, heading) {
    if (!tip) { tip = htmlEl("div", "viz-tip"); tip.setAttribute("role", "status"); document.body.appendChild(tip); }
    tip.textContent = "";
    if (heading) tip.appendChild(htmlEl("div", "viz-tip-head", heading));
    rows.forEach(function (r) {
      var row = htmlEl("div", "viz-tip-row");
      if (r.color) { var key = htmlEl("span", "viz-tip-key"); key.style.background = r.color; row.appendChild(key); }
      row.appendChild(htmlEl("strong", "", r.value));
      row.appendChild(htmlEl("span", "viz-tip-name", r.name));
      tip.appendChild(row);
    });
    tip.style.display = "block";
    var x, y;
    if (evtOrRect.clientX !== undefined) { x = evtOrRect.clientX; y = evtOrRect.clientY; }
    else { x = evtOrRect.left + evtOrRect.width / 2; y = evtOrRect.top; }
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = Math.min(window.innerWidth - w - 8, Math.max(8, x + 14));
    var top = y - h - 12;
    if (top < 8) top = y + 18;
    tip.style.left = left + window.scrollX + "px";
    tip.style.top = top + window.scrollY + "px";
  }
  function hideTip() { if (tip) tip.style.display = "none"; }

  // The panel whose mark holds the tooltip registers how to let go of it: a tap elsewhere, a scroll,
  // a redraw or a mark in another panel then clears the highlight together with the tooltip.
  var release = null;
  function releaseTip() { if (release) release(); hideTip(); }

  /* ---------- marks ---------- */
  function barPath(x, y0, y1, w) {
    // Rounded at the data end, square at the baseline.
    var r = Math.min(4, w / 2, Math.abs(y1 - y0));
    if (y1 <= y0) {
      return "M" + x + "," + y0 + "V" + (y1 + r) + "Q" + x + "," + y1 + " " + (x + r) + "," + y1 +
        "H" + (x + w - r) + "Q" + (x + w) + "," + y1 + " " + (x + w) + "," + (y1 + r) + "V" + y0 + "Z";
    }
    return "M" + x + "," + y0 + "V" + (y1 - r) + "Q" + x + "," + y1 + " " + (x + r) + "," + y1 +
      "H" + (x + w - r) + "Q" + (x + w) + "," + y1 + " " + (x + w) + "," + (y1 - r) + "V" + y0 + "Z";
  }

  function r1(v) { return Math.round(v * 10) / 10; }

  function marker(shape, cx, cy, r, attrs) {
    if (shape === "square") return svgEl("rect", Object.assign({ x: r1(cx - r), y: r1(cy - r), width: 2 * r, height: 2 * r }, attrs));
    if (shape === "diamond") return svgEl("path", Object.assign({ d: "M" + cx + "," + r1(cy - r - 1) + "L" + r1(cx + r + 1) + "," + cy + "L" + cx + "," + r1(cy + r + 1) + "L" + r1(cx - r - 1) + "," + cy + "Z" }, attrs));
    if (shape === "triangle") return svgEl("path", Object.assign({ d: "M" + cx + "," + r1(cy - r - 1) + "L" + r1(cx + r + 1) + "," + r1(cy + r) + "L" + r1(cx - r - 1) + "," + r1(cy + r) + "Z" }, attrs));
    return svgEl("circle", Object.assign({ cx: cx, cy: cy, r: r }, attrs));
  }

  /* Stroke of a line series: solid for the first, a dash pattern of its own for each later one. */
  function lineStroke(idx, color) {
    var attrs = { fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" };
    var dash = DASHES[idx % DASHES.length];
    if (dash) { attrs["stroke-dasharray"] = dash; attrs["stroke-linecap"] = "butt"; }
    return attrs;
  }

  /* Legend key: the mark as it is drawn (bar colour; line dash and marker; scatter marker). */
  function swatch(type, idx, color, surface, withMarker) {
    if (type !== "line" && type !== "scatter") {
      var sw = htmlEl("span", "viz-swatch");
      sw.style.background = color;
      return sw;
    }
    var w = type === "line" ? 28 : 12;
    var key = svgEl("svg", { width: w, height: 12, viewBox: "0 0 " + w + " 12", class: "viz-key", "aria-hidden": "true" });
    var shape = MARKERS[idx % MARKERS.length];
    if (type === "line") {
      key.appendChild(svgEl("line", Object.assign({ x1: 1, x2: w - 1, y1: 6, y2: 6 }, lineStroke(idx, color))));
      if (withMarker) key.appendChild(marker(shape, w / 2, 6, 4, { fill: color, stroke: surface, "stroke-width": 2 }));
    } else {
      key.appendChild(marker(shape, 6, 6, 4, { fill: color }));
    }
    return key;
  }

  /* True when two of the value labels, or a value label and another label, touch (boxes as rendered). */
  function labelsCollide(labels, others) {
    var boxes;
    try { boxes = labels.concat(others).map(function (t) { return t.getBBox(); }); } catch (err) { return false; }
    for (var i = 0; i < labels.length; i++) {
      var a = boxes[i];
      if (!a.width) return false; // not laid out (inside a closed <details>): checked again when it is shown
      for (var j = i + 1; j < boxes.length; j++) {
        var b = boxes[j];
        // 2 px around each box is its halo: two halos must not meet. The box is taller than the digits it holds.
        if (a.x - 2 < b.x + b.width + 2 && b.x - 2 < a.x + a.width + 2 && a.y + 1.5 < b.y + b.height - 1.5 && b.y + 1.5 < a.y + a.height - 1.5) return true;
      }
    }
    return false;
  }

  /* ---------- one panel ---------- */
  function renderPanel(host, fig, panel, ctx) {
    var width = Math.max(200, Math.floor(host.clientWidth));
    var pct = ctx.pct;
    var type = fig.type;
    var surface = cssVar(host, "--surface") || "#fcfcfb";
    var series = panel.series || [];
    if (ctx.seriesFilter) {
      series = ctx.seriesFilter.map(function (n) { return series.filter(function (s) { return s.name === n; })[0]; }).filter(Boolean);
    }
    var colors = {};
    series.forEach(function (s) {
      var idx = ctx.seriesOrder.indexOf(s.name);
      var c = cssVar(host, SERIES_VARS[idx % SERIES_VARS.length]);
      if (ctx.emphasis && ctx.emphasis !== s.name && ctx.seriesOrder.indexOf(ctx.emphasis) >= 0) c = cssVar(host, "--series-muted");
      colors[s.name] = c;
    });

    var cats = panel.x.categories || null;
    var xVals = panel.x.values || null;
    var isScatter = type === "scatter";
    var hlIndex = panel.highlight && cats ? cats.indexOf(panel.highlight.category) : -1;
    var bands = xVals ? (panel.bands || []) : [];

    /* y domain */
    var lo = Infinity, hi = -Infinity;
    function see(v) { if (isNum(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    var xLo = Infinity, xHi = -Infinity;
    if (isScatter) {
      (panel.points || []).forEach(function (g) { g.xy.forEach(function (p) { see(p[1]); if (p[0] < xLo) xLo = p[0]; if (p[0] > xHi) xHi = p[0]; }); });
    } else {
      series.forEach(function (s) { s.values.forEach(see); (s.lo || []).forEach(see); (s.hi || []).forEach(see); });
    }
    (panel.refs || []).forEach(function (r) { if (ctx.refFilter && ctx.refFilter.indexOf(r.label) < 0) return; if (r.axis === "x") { if (isScatter) { xLo = Math.min(xLo, r.value); xHi = Math.max(xHi, r.value); } } else see(r.value); });
    if (ctx.yDomain) { lo = ctx.yDomain[0]; hi = ctx.yDomain[1]; }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (type === "bar") { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    else if (!isScatter) { var pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.1 || 1; lo -= pad; hi += pad; if (lo < 0 && lo + pad >= 0) lo = 0; }
    // A scatter's y axis hugs its points: it runs from the round value just below the lowest point to the one just
    // above the highest, in steps of about 36 px and without padding (a padded range squeezes the band around 0).
    var yTicks = niceTicks(lo, hi, isScatter ? Math.max(4, Math.round(ctx.height / 36)) : 4);
    var y0 = yTicks[0], y1 = yTicks[yTicks.length - 1];

    var tickStrs = yTicks.map(tickFormatter(yTicks, pct));
    var mLeft = Math.ceil(widestLine(tickStrs, "viz-tick")) + 14;
    var mRight = 12;
    // notes above the plot: the highlight label (wrapped when long) or up to two rows of band labels
    var hlLines = hlIndex >= 0 && panel.highlight.label && !ctx.emphasis ? wrapLabel(panel.highlight.label, width - 8, "viz-note-label") : [];
    var bandRows = Math.min(2, bands.filter(function (b) { return b.label; }).length);
    var mTop = 20 + Math.max(0, hlLines.length - 1, bandRows - 1) * 12;
    var plotH = ctx.height;
    var plotW = width - mLeft - mRight;

    /* x layout: category labels are wrapped under their tick, and slanted when a word is wider than the tick spacing */
    var n = cats ? cats.length : (xVals ? xVals.length : 0);
    var band = n ? plotW / n : plotW;
    function tickClass(i) { return "viz-tick" + (i === hlIndex ? " viz-tick-strong" : ""); }
    var catLines = cats ? cats.map(function (c, i) { return wrapLabel(c, band - 6, tickClass(i)); }) : [];
    var xLabelLines = cats ? Math.max.apply(null, catLines.map(function (l) { return l.length; })) : 1;
    var widest = cats ? Math.max.apply(null, catLines.map(function (l, i) { return widestLine(l, tickClass(i)); })) : 0;
    var slanted = !!cats && (widest > band - 4 || xLabelLines > 4);
    if (ctx.layout) { slanted = !!cats && ctx.layout.slanted; xLabelLines = Math.max(xLabelLines, ctx.layout.lines); }
    var angle = 0, lens = null;
    if (slanted) {
      // A slanted label ends under its tick and runs down to the left, so the first ticks need room on their
      // left. The flattest angle is taken that keeps that room under a third of the width and the labels apart.
      lens = cats.map(function (c, i) { return textW(c, tickClass(i)); });
      var slantLeft = function (deg) {
        var cos = Math.cos(deg * Math.PI / 180), need = 0;
        lens.forEach(function (len, i) {
          var f = (i + 0.5) / n;
          need = Math.max(need, (len * cos + 8 - (width - mRight) * f) / (1 - f));
        });
        return Math.max(mLeft, Math.ceil(need));
      };
      var own = SLANTS[SLANTS.length - 1];
      for (var a = 0; a < SLANTS.length; a++) {
        var left = slantLeft(SLANTS[a]);
        if (left <= Math.max(mLeft, width * 0.34) && (width - left - mRight) / n * Math.sin(SLANTS[a] * Math.PI / 180) >= 12) { own = SLANTS[a]; break; }
      }
      angle = Math.max(own, ctx.layout ? ctx.layout.angle : 0);
      mLeft = slantLeft(angle);
      plotW = width - mLeft - mRight;
      band = plotW / n;
    }
    var sinA = Math.sin(angle * Math.PI / 180), cosA = Math.cos(angle * Math.PI / 180);
    var slantDrop = 5 + 11 * cosA; // from the axis down to the end of a slanted label
    var xTitle = panel.x.label ? wrapLabel(panel.x.label, width - 8, "viz-axis-label") : [];
    var xTitleH = xTitle.length ? 8 + xTitle.length * 14 : 0;
    var mBottom = slanted
      ? Math.ceil(slantDrop + Math.max.apply(null, lens) * sinA + 4) + xTitleH
      : 10 + xLabelLines * 14 + xTitleH;
    var height = mTop + plotH + mBottom;

    var svg = svgEl("svg", { width: width, height: height, viewBox: "0 0 " + width + " " + height, role: "img", tabindex: "0", class: "viz-svg" });
    svg.setAttribute("aria-label", (panel.title ? panel.title + ": " : "") + fig.title + ". " + panel.y.label + ". Arrow keys step through the values; the data table below lists them.");

    function sy(v) { return mTop + plotH - (v - y0) / (y1 - y0) * plotH; }
    var xTicks = null, sx;
    if (isScatter) {
      var xp = (xHi - xLo) * 0.05 || 1;
      xTicks = niceTicks(xLo - xp, xHi + xp, Math.max(3, Math.floor(plotW / 80)));
      var xa = xTicks[0], xb = xTicks[xTicks.length - 1];
      var xTickFmt = tickFormatter(xTicks, false);
      sx = function (v) { return mLeft + (v - xa) / (xb - xa) * plotW; };
    } else if (xVals && type === "line") {
      var vmin = Math.min.apply(null, xVals), vmax = Math.max.apply(null, xVals);
      var inset = Math.min(28, plotW * 0.08);
      sx = function (v) { return vmax === vmin ? mLeft + plotW / 2 : mLeft + inset + (v - vmin) / (vmax - vmin) * (plotW - 2 * inset); };
    } else {
      sx = function (i) { return mLeft + band * (i + 0.5); };
    }
    function xPos(i) { return (xVals && type === "line") ? sx(xVals[i]) : sx(i); }

    /* grid + y ticks */
    var gGrid = svgEl("g", { class: "viz-grid" });
    yTicks.forEach(function (t, i) {
      var y = Math.round(sy(t)) + 0.5;
      gGrid.appendChild(svgEl("line", { x1: mLeft, x2: mLeft + plotW, y1: y, y2: y, class: t === 0 ? "viz-axis" : "viz-gridline" }));
      gGrid.appendChild(svgEl("text", { x: mLeft - 8, y: y + 4, "text-anchor": "end", class: "viz-tick" }, tickStrs[i]));
    });
    svg.appendChild(gGrid);
    if (y0 !== 0 || type !== "bar") {
      svg.appendChild(svgEl("line", { x1: mLeft, x2: mLeft + plotW, y1: mTop + plotH + 0.5, y2: mTop + plotH + 0.5, class: "viz-axis" }));
    }

    /* x labels */
    if (cats) {
      catLines.forEach(function (lines, i) {
        if (slanted) {
          var tx = r1(xPos(i) + 3), ty = r1(mTop + plotH + slantDrop);
          svg.appendChild(svgEl("text", { x: tx, y: ty, "text-anchor": "end", transform: "rotate(-" + angle + " " + tx + " " + ty + ")", class: tickClass(i) }, cats[i]));
          return;
        }
        lines.forEach(function (ln, k) {
          svg.appendChild(svgEl("text", { x: xPos(i), y: mTop + plotH + 16 + k * 14, "text-anchor": "middle", class: tickClass(i) }, ln));
        });
      });
    } else if (xVals && type === "line") {
      if (xVals.length <= 6) {
        xVals.forEach(function (v, i) {
          svg.appendChild(svgEl("text", { x: xPos(i), y: mTop + plotH + 16, "text-anchor": "middle", class: "viz-tick" }, fmtX(v)));
        });
      } else {
        // many x positions: label round values of the axis, not every data point
        var lineTicks = niceTicks(Math.min.apply(null, xVals), Math.max.apply(null, xVals), Math.max(4, Math.floor(plotW / 90)));
        var lineFmt = tickFormatter(lineTicks, false);
        lineTicks.forEach(function (t) {
          // the axis runs past the first and the last point: keep every round value that falls on it
          if (sx(t) < mLeft - 0.5 || sx(t) > mLeft + plotW + 0.5) return;
          svg.appendChild(svgEl("line", { x1: sx(t), x2: sx(t), y1: mTop + plotH, y2: mTop + plotH + 4, class: "viz-axis" }));
          svg.appendChild(svgEl("text", { x: sx(t), y: mTop + plotH + 16, "text-anchor": "middle", class: "viz-tick" }, lineFmt(t)));
        });
      }
    } else if (isScatter) {
      xTicks.forEach(function (t) {
        var x = Math.round(sx(t)) + 0.5;
        svg.appendChild(svgEl("line", { x1: x, x2: x, y1: mTop, y2: mTop + plotH, class: t === 0 ? "viz-axis" : "viz-gridline" }));
        svg.appendChild(svgEl("text", { x: x, y: mTop + plotH + 16, "text-anchor": "middle", class: "viz-tick" }, xTickFmt(t)));
      });
    }
    xTitle.forEach(function (line, k) {
      var half = textW(line, "viz-axis-label") / 2;
      var cx = Math.min(Math.max(mLeft + plotW / 2, half + 2), width - half - 2);
      svg.appendChild(svgEl("text", { x: cx, y: height - 6 - (xTitle.length - 1 - k) * 14, "text-anchor": "middle", class: "viz-axis-label" }, line));
    });

    /* highlight band */
    if (hlIndex >= 0) {
      var hw = Math.min(band, 56);
      svg.insertBefore(svgEl("rect", { x: xPos(hlIndex) - hw / 2, y: mTop - 6, width: hw, height: plotH + 6, rx: 6, class: "viz-highlight" }), gGrid);
      hlLines.forEach(function (line, k) {
        var half = textW(line, "viz-note-label") / 2;
        var hlX = Math.min(Math.max(xPos(hlIndex), half + 4), width - half - 4);
        svg.appendChild(svgEl("text", { x: hlX, y: mTop - 10 - (hlLines.length - 1 - k) * 12, "text-anchor": "middle", class: "viz-note-label" }, line));
      });
    }
    /* shaded x ranges, each named above the plot where its name fits (it is also in the tooltip and the table) */
    var rowEnd = [-Infinity, -Infinity];
    bands.forEach(function (b) {
      var bx0 = Math.max(mLeft, sx(b.x0)), bx1 = Math.min(mLeft + plotW, sx(b.x1));
      svg.insertBefore(svgEl("rect", { x: bx0, y: mTop, width: Math.max(2, bx1 - bx0), height: plotH, class: "viz-band" }), gGrid);
      if (!b.label) return;
      var w = textW(b.label, "viz-note-label");
      if (w > plotW) return;
      var cx = Math.min(Math.max((bx0 + bx1) / 2, mLeft + w / 2), mLeft + plotW - w / 2);
      for (var row = 0; row < bandRows; row++) {
        if (cx - w / 2 < rowEnd[row] + 6) continue;
        svg.appendChild(svgEl("text", { x: r1(cx), y: mTop - 5 - row * 12, "text-anchor": "middle", class: "viz-note-label" }, b.label));
        rowEnd[row] = cx + w / 2;
        break;
      }
    });

    /* reference lines sit above the marks; the figure-level key names them */
    function drawRefs() {
      (panel.refs || []).forEach(function (r) {
        if (ctx.refFilter && ctx.refFilter.indexOf(r.label) < 0) return;
        if (r.axis === "x" && isScatter) {
          var rx = Math.round(sx(r.value)) + 0.5;
          svg.appendChild(svgEl("line", { x1: rx, x2: rx, y1: mTop, y2: mTop + plotH, class: "viz-ref" }));
        } else if (r.axis !== "x") {
          var ry = Math.round(sy(r.value)) + 0.5;
          svg.appendChild(svgEl("line", { x1: mLeft, x2: mLeft + plotW, y1: ry, y2: ry, class: "viz-ref" }));
        }
      });
    }

    /* marks */
    var marks = []; // {node, hit, heading, rows} for bars and line columns, {at, heading, rows} for scatter points
    var valueLabels = [];
    var ring = null;
    var gMarks = svgEl("g", {});
    svg.appendChild(gMarks);
    var gLabels = svgEl("g", {}); // value labels go above the reference lines, so that no line strikes through a number
    var markCount = isScatter ? 0 : series.length * n;
    var labelAll = ctx.labels === "all" || (ctx.labels !== "none" && markCount <= 10);
    if (ctx.layout && !ctx.layout.labels) labelAll = false;
    if (labelAll && ctx.labels !== "all" && n) {
      // room per label: one bar pitch (bars) or the smallest x step (lines)
      var room = band;
      if (type === "bar" && series.length > 1) room = Math.min(band * 0.78 / series.length, 26);
      if (type === "line") { room = plotW; for (var q = 1; q < n; q++) room = Math.min(room, Math.abs(xPos(q) - xPos(q - 1))); }
      var widestValue = 0;
      series.forEach(function (s) { s.values.forEach(function (v) { if (isNum(v)) widestValue = Math.max(widestValue, textW(fmt(v, pct), "viz-value")); }); });
      if (widestValue > room - 2 || (type === "line" && series.length > 1)) labelAll = false;
    }
    function valueLabel(x, y, text) {
      var node = svgEl("text", { x: r1(x), y: r1(y), "text-anchor": "middle", class: "viz-value viz-halo" }, text);
      gLabels.appendChild(node);
      valueLabels.push(node);
    }

    if (type === "bar") {
      var gap = 2;
      var groupW = Math.min(band * 0.78, series.length * 24 + (series.length - 1) * gap);
      var bw = (groupW - (series.length - 1) * gap) / series.length;
      series.forEach(function (s, si) {
        s.values.forEach(function (v, i) {
          if (!isNum(v)) return;
          var x = xPos(i) - groupW / 2 + si * (bw + gap);
          var color = colors[s.name];
          if (ctx.emphasis && series.length === 1 && cats) color = cats[i] === ctx.emphasis ? cssVar(host, "--series-1") : cssVar(host, "--series-muted");
          var bar = svgEl("path", { d: barPath(x, sy(0), sy(v), bw), fill: color, class: "viz-mark" });
          gMarks.appendChild(bar);
          var top = sy(v);
          var hasCI = s.lo && s.hi && isNum(s.lo[i]) && isNum(s.hi[i]);
          if (hasCI) {
            var cx = x + bw / 2;
            gMarks.appendChild(svgEl("line", { x1: cx, x2: cx, y1: sy(s.lo[i]), y2: sy(s.hi[i]), class: "viz-whisker" }));
            gMarks.appendChild(svgEl("line", { x1: cx - 3, x2: cx + 3, y1: sy(s.hi[i]), y2: sy(s.hi[i]), class: "viz-whisker" }));
            gMarks.appendChild(svgEl("line", { x1: cx - 3, x2: cx + 3, y1: sy(s.lo[i]), y2: sy(s.lo[i]), class: "viz-whisker" }));
            top = Math.min(top, sy(s.hi[i]));
          }
          if (labelAll && bw >= 14) {
            var ly = top - 5;
            if (v < 0) {
              ly = Math.max(sy(v), hasCI ? sy(s.lo[i]) : sy(v)) + 13;
              // under a bar that ends near the bottom of the plot the label would sit on the tick labels: it goes inside the bar
              if (ly > mTop + plotH - 3) ly = top - 5;
            }
            valueLabel(x + bw / 2, ly, fmt(v, pct));
          }
          var rows = [{ color: color, value: fmt(v, pct) + (hasCI ? "  [" + fmt(s.lo[i], pct) + ", " + fmt(s.hi[i], pct) + "]" : ""), name: series.length > 1 ? s.name : panel.y.label }];
          var hit = svgEl("rect", { x: x - gap / 2, y: mTop, width: bw + gap, height: plotH, fill: "transparent" });
          gMarks.appendChild(hit);
          marks.push({ node: bar, hit: hit, heading: cats ? cats[i] : fmtX(xVals[i]), rows: rows });
        });
      });
    } else if (type === "line") {
      series.forEach(function (s) {
        var color = colors[s.name];
        var idx = ctx.seriesOrder.indexOf(s.name);
        var pts = [];
        s.values.forEach(function (v, i) { if (isNum(v)) pts.push([xPos(i), sy(v), i]); });
        if (s.lo && s.hi) {
          var up = [], dn = [];
          s.values.forEach(function (v, i) { if (isNum(s.lo[i]) && isNum(s.hi[i])) { up.push(xPos(i) + "," + sy(s.hi[i])); dn.unshift(xPos(i) + "," + sy(s.lo[i])); } });
          if (up.length > 1) gMarks.appendChild(svgEl("polygon", { points: up.concat(dn).join(" "), fill: color, opacity: 0.12 }));
        }
        if (pts.length > 1) gMarks.appendChild(svgEl("polyline", Object.assign({ points: pts.map(function (p) { return p[0] + "," + p[1]; }).join(" ") }, lineStroke(idx, color))));
        if (pts.length <= 24) pts.forEach(function (p) {
          var m = marker(MARKERS[idx % MARKERS.length], p[0], p[1], 4, { fill: color, stroke: surface, "stroke-width": 2 });
          m.setAttribute("class", "viz-mark");
          gMarks.appendChild(m);
        });
        if (labelAll) pts.forEach(function (p) { valueLabel(p[0], p[1] - 10, fmt(s.values[p[2]], pct)); });
      });
      // one hit column per x position: the tooltip lists every series
      for (var i = 0; i < n; i++) (function (i) {
        var half = n > 1 ? Math.abs(xPos(Math.min(i + 1, n - 1)) - xPos(Math.max(i - 1, 0))) / (i === 0 || i === n - 1 ? 2 : 4) : plotW / 2;
        var hit = svgEl("rect", { x: xPos(i) - half, y: mTop, width: 2 * half, height: plotH, fill: "transparent" });
        gMarks.appendChild(hit);
        var rows = series.filter(function (s) { return isNum(s.values[i]); }).map(function (s) {
          var ci = s.lo && s.hi && isNum(s.lo[i]) ? "  [" + fmt(s.lo[i], pct) + ", " + fmt(s.hi[i], pct) + "]" : "";
          return { color: colors[s.name], value: fmt(s.values[i], pct) + ci, name: s.name };
        });
        var cross = svgEl("line", { x1: xPos(i), x2: xPos(i), y1: mTop, y2: mTop + plotH, class: "viz-cross" });
        cross.style.display = "none";
        svg.insertBefore(cross, gMarks);
        var shaded = cats ? "" : bandLabel(bands, xVals[i]);
        marks.push({ node: cross, hit: hit, heading: (panel.x.label ? panel.x.label + " " : "") + (cats ? cats[i] : fmtX(xVals[i])) + (shaded ? " · shaded: " + shaded : ""), rows: rows, cross: true });
      })(i);
    } else if (isScatter) {
      // drawn in reverse so that the first-listed group (the one the block is about) ends on top
      (panel.points || []).slice().reverse().forEach(function (g) {
        var idx = Math.max(0, ctx.seriesOrder.indexOf(g.name));
        var color = cssVar(host, SERIES_VARS[idx % SERIES_VARS.length]);
        var layer = svgEl("g", { fill: color, "fill-opacity": 0.5, stroke: color, "stroke-width": 0.75, "stroke-opacity": 0.9 });
        g.xy.forEach(function (p) {
          var px = r1(sx(p[0])), py = r1(sy(p[1]));
          layer.appendChild(marker(MARKERS[idx % MARKERS.length], px, py, 2.6));
          marks.push({ at: [px, py], x: p[0], heading: g.name, rows: [{ color: color, value: fmt(p[1], pct), name: panel.y.label }, { value: fmtPlain(p[0]), name: panel.x.label }] });
        });
        gMarks.appendChild(layer);
      });
      // the arrow keys walk the points from left to right
      marks.sort(function (p, q) { return p.x - q.x; });
      ring = svgEl("circle", { r: 6, fill: "none", class: "viz-ring" });
      ring.style.display = "none";
      svg.appendChild(ring);
    }

    drawRefs();
    svg.appendChild(gLabels);

    /* hover, tap and keyboard */
    var active = -1;
    function anchor(m) {
      if (!m.at) return m.hit.getBoundingClientRect();
      var box = svg.getBoundingClientRect(), k = box.width / width;
      return { left: box.left + m.at[0] * k, top: box.top + m.at[1] * k - 6, width: 0, height: 0 };
    }
    function activate(i, evt) {
      if (i < 0 && active < 0) return; // nothing of this panel is selected: a tooltip that is showing belongs to another panel
      if (i >= 0 && release && release !== off) release();
      if (active >= 0 && marks[active] && marks[active].node) {
        marks[active].node.classList.remove("is-active");
        if (marks[active].cross) marks[active].node.style.display = "none";
      }
      active = i;
      if (i < 0) {
        if (ring) ring.style.display = "none";
        if (release === off) { release = null; hideTip(); }
        return;
      }
      var m = marks[i];
      if (m.node) m.node.classList.add("is-active");
      if (m.cross) m.node.style.display = "";
      if (m.at) { ring.setAttribute("cx", m.at[0]); ring.setAttribute("cy", m.at[1]); ring.style.display = ""; }
      release = off;
      showTip(evt || anchor(m), m.rows, m.heading);
    }
    function off() { activate(-1); }
    // a finger leaves the mark as soon as it is lifted: the tooltip of a tap stays until the next tap or a scroll
    function leave(e) { if (e.pointerType !== "touch") activate(-1); }
    if (isScatter) {
      var overlay = svgEl("rect", { x: mLeft, y: mTop, width: plotW, height: plotH, fill: "transparent" });
      svg.appendChild(overlay);
      var nearest = function (e) {
        var box = svg.getBoundingClientRect(), k = width / (box.width || width);
        var mx = (e.clientX - box.left) * k, my = (e.clientY - box.top) * k, best = -1, bd = 900;
        for (var j = 0; j < marks.length; j++) {
          var d = (marks[j].at[0] - mx) * (marks[j].at[0] - mx) + (marks[j].at[1] - my) * (marks[j].at[1] - my);
          if (d < bd) { bd = d; best = j; }
        }
        if (best >= 0 || active >= 0) activate(best, e);
      };
      overlay.addEventListener("pointermove", nearest);
      overlay.addEventListener("pointerdown", nearest);
      overlay.addEventListener("pointerleave", leave);
    } else {
      marks.forEach(function (m, i) {
        var on = function (e) { activate(i, e); };
        m.hit.addEventListener("pointermove", on);
        m.hit.addEventListener("pointerdown", on);
        m.hit.addEventListener("pointerleave", leave);
      });
    }
    svg.addEventListener("keydown", function (e) {
      if (!marks.length) return;
      var last = marks.length - 1, to = -2;
      if (e.key === "ArrowRight" || e.key === "ArrowDown") to = active >= last ? 0 : active + 1;
      else if (e.key === "ArrowLeft" || e.key === "ArrowUp") to = active <= 0 ? last : active - 1;
      else if (e.key === "Escape") to = -1;
      // once a mark is selected these keys move within the chart (a scatter can hold hundreds of points); before that they scroll the page as usual
      else if (active >= 0 && e.key === "PageDown") to = Math.min(last, active + 10);
      else if (active >= 0 && e.key === "PageUp") to = Math.max(0, active - 10);
      else if (active >= 0 && e.key === "Home") to = 0;
      else if (active >= 0 && e.key === "End") to = last;
      if (to === -2) return;
      activate(to);
      if (to >= 0) e.preventDefault();
    });
    svg.addEventListener("blur", function () { activate(-1); });

    host.textContent = "";
    host.appendChild(svg);

    /* value labels stay only when none of them touches another label; the values remain in the tooltip and the data table */
    var labelled = valueLabels.length > 0;
    if (labelled && labelsCollide(valueLabels, Array.prototype.slice.call(svg.querySelectorAll(".viz-note-label")))) {
      svg.removeChild(gLabels);
      labelled = false;
    }
    return { slanted: !!slanted, lines: xLabelLines, angle: angle, labels: labelled };
  }

  /* ---------- table view ---------- */
  var tableCount = 0;
  function buildTable(fig, panels, ctx) {
    var wrap = htmlEl("div", "viz-table-wrap");
    panels.forEach(function (panel) {
      var table = htmlEl("table", "viz-table");
      var cap = htmlEl("caption", "", (panel.title ? panel.title + " — " : "") + panel.y.label);
      var thead = htmlEl("thead"), tr = htmlEl("tr");
      function colHead(text, cls) {
        var th = htmlEl("th", cls || "");
        th.setAttribute("scope", "col");
        // the first header cell holds the x-axis label, which can be a sentence: its own box keeps it from taking the width of the table
        if (tr.firstChild) th.textContent = text;
        else th.appendChild(htmlEl("span", "viz-table-xlabel", text));
        tr.appendChild(th);
        return th;
      }
      if (fig.type === "scatter") {
        // The plotted points themselves, in the order of the data file, inside a box that scrolls. The heading
        // stays above the box and spells out the two axes, so that the header row is one short line.
        var groups = panel.points || [], total = 0;
        groups.forEach(function (g) { total += g.xy.length; });
        var title = htmlEl("div", "viz-table-cap", (panel.title ? panel.title + " — " : "") + total + " plotted points (" + groups.map(function (g) { return g.xy.length + " " + g.name; }).join(", ") + ")");
        title.id = "viz-table-" + (++tableCount);
        title.appendChild(htmlEl("span", "", " x: " + panel.x.label + ". y: " + panel.y.label + "."));
        table.setAttribute("aria-labelledby", title.id);
        colHead("Group");
        colHead("x", "num").setAttribute("aria-label", "x: " + panel.x.label);
        colHead("y", "num").setAttribute("aria-label", "y: " + panel.y.label);
        thead.appendChild(tr); table.appendChild(thead);
        var tb = htmlEl("tbody");
        groups.forEach(function (g) {
          g.xy.forEach(function (p) {
            var r = htmlEl("tr");
            r.appendChild(htmlEl("td", "", g.name));
            r.appendChild(htmlEl("td", "num", fmtTable(p[0], false)));
            r.appendChild(htmlEl("td", "num", fmtTable(p[1], ctx.pct)));
            tb.appendChild(r);
          });
        });
        table.appendChild(tb);
        var box = htmlEl("div", "viz-table-scroll");
        box.setAttribute("tabindex", "0");
        box.setAttribute("role", "group");
        box.setAttribute("aria-labelledby", title.id);
        box.appendChild(table);
        wrap.appendChild(title); wrap.appendChild(box);
        return;
      }
      table.appendChild(cap);
      var series = panel.series;
      if (ctx.seriesFilter) series = ctx.seriesFilter.map(function (n) { return panel.series.filter(function (s) { return s.name === n; })[0]; }).filter(Boolean);
      var bands = panel.x.values ? (panel.bands || []).filter(function (b) { return b.label; }) : [];
      colHead(panel.x.label || "");
      series.forEach(function (s) { colHead(s.name, "num"); });
      if (bands.length) colHead("Shaded band");
      thead.appendChild(tr); table.appendChild(thead);
      var tbody = htmlEl("tbody");
      var xs = panel.x.categories || panel.x.values;
      xs.forEach(function (x, i) {
        var row = htmlEl("tr");
        var head = htmlEl("th", "", typeof x === "number" && !Number.isInteger(x) ? fmtTable(x, false) : String(x));
        head.setAttribute("scope", "row");
        row.appendChild(head);
        series.forEach(function (s) {
          var cell = htmlEl("td", "num", fmtTable(s.values[i], ctx.pct));
          // the interval is one unbreakable piece: in a narrow table it moves under its value as a whole
          if (s.lo && s.hi && isNum(s.lo[i]) && isNum(s.hi[i])) {
            cell.appendChild(document.createTextNode(" "));
            cell.appendChild(htmlEl("span", "viz-ci", "[" + fmtTable(s.lo[i], ctx.pct) + ", " + fmtTable(s.hi[i], ctx.pct) + "]"));
          }
          row.appendChild(cell);
        });
        if (bands.length) row.appendChild(htmlEl("td", "viz-table-note", bandLabel(bands, x)));
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
    });
    return wrap;
  }

  /* Key text of a reference line: its label and its own value (per panel where the panels differ). */
  function refText(label, vals) {
    var same = vals.every(function (v) { return v.text === vals[0].text; });
    // a label that already spells out its percentage ("'clear' ≥ 10 %") is left as it is
    var bare = label.replace(/\s+/g, ""), at = bare.indexOf(vals[0].text);
    if (same && /%$/.test(vals[0].text) && at >= 0 && !/[0-9.,]/.test(bare.charAt(at - 1))) return label;
    if (same) return label + ": " + vals[0].text;
    return label + ": " + vals.map(function (v) { return v.text + (v.panel ? " (" + v.panel + ")" : ""); }).join(", ");
  }

  /* ---------- one figure ---------- */
  function renderFigure(el) {
    var data = (window.DEXCORE_FIGURES || {}).figures || {};
    var fig = data[el.getAttribute("data-fig")];
    var body = el.querySelector(".viz-body");
    if (!body) { body = htmlEl("div", "viz-body"); el.insertBefore(body, el.firstChild); }
    body.textContent = "";
    if (!fig) { body.appendChild(htmlEl("p", "viz-missing", "Figure data not found: " + el.getAttribute("data-fig"))); return; }

    var panels = fig.panels;
    var want = el.getAttribute("data-panels");
    if (want) panels = want.split(",").map(function (t) { return fig.panels.filter(function (p) { return p.title === t.trim(); })[0]; }).filter(Boolean);
    if (!panels.length) { body.appendChild(htmlEl("p", "viz-missing", "No panel of this figure matches data-panels=\"" + want + "\".")); return; }
    var seriesFilter = el.getAttribute("data-series") ? el.getAttribute("data-series").split("|") : null;

    var seriesOrder = [];
    panels.forEach(function (p) {
      (p.series || p.points || []).forEach(function (s) { if (seriesOrder.indexOf(s.name) < 0 && (!seriesFilter || seriesFilter.indexOf(s.name) >= 0)) seriesOrder.push(s.name); });
    });
    if (seriesFilter) seriesOrder = seriesFilter.filter(function (n) { return seriesOrder.indexOf(n) >= 0; });

    var ctx = {
      pct: el.getAttribute("data-format") === "pct",
      labels: el.getAttribute("data-labels") || "auto",
      emphasis: el.getAttribute("data-emphasis"),
      height: parseInt(el.getAttribute("data-height") || "210", 10),
      seriesOrder: seriesOrder,
      seriesFilter: seriesFilter,
      refFilter: el.getAttribute("data-refs") ? el.getAttribute("data-refs").split("|") : null,
    };
    function refShown(r) { return !ctx.refFilter || ctx.refFilter.indexOf(r.label) >= 0; }

    if (el.hasAttribute("data-shared-y")) {
      var gLo = Infinity, gHi = -Infinity;
      var span = function (v) { if (isNum(v)) { if (v < gLo) gLo = v; if (v > gHi) gHi = v; } };
      panels.forEach(function (p) {
        (p.series || []).forEach(function (sr) {
          if (seriesFilter && seriesFilter.indexOf(sr.name) < 0) return;
          [sr.values, sr.lo || [], sr.hi || []].forEach(function (arr) { arr.forEach(span); });
        });
        (p.points || []).forEach(function (g) { g.xy.forEach(function (q) { span(q[1]); }); });
        (p.refs || []).forEach(function (r) { if (r.axis !== "x" && refShown(r)) span(r.value); });
      });
      if (gLo !== Infinity) ctx.yDomain = [gLo, gHi];
    }

    var dir = panels[0].y.direction;
    var sameY = panels.every(function (p) { return p.y.label === panels[0].y.label; });
    // with the percent format the axis no longer shows fractions
    function yLabel(text) { return ctx.pct ? text.replace(/\s*\(fraction\)/, "") : text; }
    var head = htmlEl("div", "viz-head");
    if (sameY) head.appendChild(htmlEl("div", "viz-ylabel", yLabel(panels[0].y.label)));
    if (dir === "lower_better") head.appendChild(htmlEl("span", "viz-dir", "↓ lower is better"));
    if (dir === "higher_better") head.appendChild(htmlEl("span", "viz-dir", "↑ higher is better"));
    body.appendChild(head);

    if (seriesOrder.length > 1) {
      var legend = htmlEl("ul", "viz-legend");
      var surface = cssVar(el, "--surface") || "#fcfcfb";
      // line markers are drawn only on series of at most 24 points (renderPanel)
      var withMarkers = panels.every(function (p) { return (p.x.categories || p.x.values || []).length <= 24; });
      seriesOrder.forEach(function (name, i) {
        var li = htmlEl("li");
        var color = cssVar(el, SERIES_VARS[i % SERIES_VARS.length]);
        if (ctx.emphasis && ctx.emphasis !== name && seriesOrder.indexOf(ctx.emphasis) >= 0) color = cssVar(el, "--series-muted");
        li.appendChild(swatch(fig.type, i, color, surface, withMarkers));
        li.appendChild(htmlEl("span", "", name));
        legend.appendChild(li);
      });
      body.appendChild(legend);
    }

    var refKeys = [], refVals = {};
    panels.forEach(function (p) {
      (p.refs || []).forEach(function (r) {
        if (!r.label || !refShown(r) || (r.axis === "x" && fig.type !== "scatter")) return;
        var key = (r.axis === "x" ? "x" : "y") + "|" + r.label;
        if (refKeys.indexOf(key) < 0) { refKeys.push(key); refVals[key] = []; }
        refVals[key].push({ panel: p.title, text: fmtShort(r.value, ctx.pct && r.axis !== "x") });
      });
    });
    if (refKeys.length) {
      var refLegend = htmlEl("ul", "viz-legend viz-reflegend");
      refKeys.forEach(function (key) {
        var li = htmlEl("li");
        var sample = svgEl("svg", { width: 18, height: 12, viewBox: "0 0 18 12", class: "viz-key", "aria-hidden": "true" });
        sample.appendChild(svgEl("line", key.charAt(0) === "x" ? { x1: 9.5, x2: 9.5, y1: 0, y2: 12, class: "viz-ref" } : { x1: 0, x2: 18, y1: 6.5, y2: 6.5, class: "viz-ref" }));
        li.appendChild(sample);
        li.appendChild(htmlEl("span", "", refText(key.slice(2), refVals[key])));
        refLegend.appendChild(li);
      });
      body.appendChild(refLegend);
    }

    var grid = htmlEl("div", "viz-panels viz-panels-" + panels.length);
    body.appendChild(grid);
    var hosts = panels.map(function (panel) {
      var cell = htmlEl("div", "viz-panel");
      if (panel.title) cell.appendChild(htmlEl("div", "viz-panel-title", panel.title));
      if (!sameY) cell.appendChild(htmlEl("div", "viz-panel-ylabel", yLabel(panel.y.label)));
      var host = htmlEl("div", "viz-plot");
      cell.appendChild(host);
      grid.appendChild(cell);
      return host;
    });
    // Panels of one figure share the x-label layout and either all carry value labels or none does:
    // drawn again with the common layout until every panel agrees with it.
    function draw() {
      releaseTip();
      ctx.layout = null;
      for (var pass = 0; pass < 4; pass++) {
        var m = hosts.map(function (host, i) { return renderPanel(host, fig, panels[i], ctx); });
        var layout = {
          slanted: m.some(function (x) { return x.slanted; }),
          lines: Math.max.apply(null, m.map(function (x) { return x.lines; })),
          angle: Math.max.apply(null, m.map(function (x) { return x.angle; })),
          labels: m.every(function (x) { return x.labels; }),
        };
        if (m.every(function (x) { return x.slanted === layout.slanted && x.lines === layout.lines && x.angle === layout.angle && x.labels === layout.labels; })) break;
        ctx.layout = layout;
      }
    }
    draw();
    el._redraw = draw;

    var details = htmlEl("details", "viz-data");
    details.appendChild(htmlEl("summary", "", "Data table"));
    details.appendChild(buildTable(fig, panels, ctx));
    if (fig.note) details.appendChild(htmlEl("p", "viz-data-note", fig.note));
    var src = htmlEl("p", "viz-data-note");
    src.appendChild(document.createTextNode("Read from: "));
    (fig.sources || []).forEach(function (s, i) {
      if (i) src.appendChild(document.createTextNode("; "));
      src.appendChild(htmlEl("code", "", s.path));
    });
    details.appendChild(src);
    body.appendChild(details);
  }

  /* ---------- wide diagrams ---------- */
  // The schematic diagrams keep their drawn size and scroll sideways in a narrow column; the cue that says so
  // (style.css) is shown exactly where a diagram is cut off.
  function markScrollers() {
    Array.prototype.forEach.call(document.querySelectorAll(".diagram-scroll"), function (d) {
      var clipped = d.scrollWidth > d.clientWidth + 1;
      d.classList.toggle("is-clipped", clipped);
      d.classList.toggle("is-fit", !clipped);
    });
  }

  function renderAll() {
    var figs = document.querySelectorAll("figure.viz[data-fig]");
    // one figure that cannot be drawn must not stop the others
    Array.prototype.forEach.call(figs, function (f) {
      try { renderFigure(f); }
      catch (err) { (f.querySelector(".viz-body") || f).appendChild(htmlEl("p", "viz-missing", "This chart could not be drawn: " + err.message)); }
    });
    function redraw(list) {
      Array.prototype.forEach.call(list, function (f) { if (f._redraw) { try { f._redraw(); } catch (err) { /* the chart keeps its last drawing */ } } });
    }
    markScrollers();
    var timer = null;
    window.addEventListener("resize", function () {
      clearTimeout(timer);
      timer = setTimeout(function () { redraw(figs); markScrollers(); }, 120);
    });
    document.addEventListener("toggle", function (e) {
      if (!e.target || !e.target.querySelectorAll) return;
      redraw(e.target.querySelectorAll("figure.viz"));
      markScrollers();
    }, true);
    document.addEventListener("scroll", releaseTip, { passive: true });
    // a tap outside a chart lets go of the tooltip that a tap opened
    document.addEventListener("pointerdown", function (e) {
      if (!e.target.closest || !e.target.closest(".viz-svg")) releaseTip();
    });
    // A closed <details> is not laid out, so it would be missing from a printout: open them all for printing,
    // draw the charts inside at their real width, and close them again afterwards.
    window.addEventListener("beforeprint", function () {
      Array.prototype.forEach.call(document.querySelectorAll("details:not([open])"), function (d) {
        d.setAttribute("data-print-opened", "");
        d.open = true;
      });
      redraw(figs);
      markScrollers();
    });
    window.addEventListener("afterprint", function () {
      Array.prototype.forEach.call(document.querySelectorAll("details[data-print-opened]"), function (d) {
        d.open = false;
        d.removeAttribute("data-print-opened");
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", renderAll);
  else renderAll();
})();
