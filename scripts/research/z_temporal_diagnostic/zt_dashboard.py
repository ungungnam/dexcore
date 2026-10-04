#!/usr/bin/env python
"""Small research dashboard (one HTML page, Korean): WHY did z-mediated Stage-2 generation fail?    python zt_dashboard.py
Numbers are injected from the result tables; the prose lives in narrative/dashboard_text.json and uses the same <<key:format>>
placeholders as the report (resolved against OUT/report_values.json).  -> OUT/dashboard.html (published as a Claude artifact)."""
from __future__ import annotations

import base64
import json
import re

import numpy as np
import pandas as pd

import zt_common as Z

L_ = {"taco": "TACO", "arctic": "ARCTIC"}
MC = {"persistence": "#9E9E9E", "mean": "#C9C9C9", "linear": "#8172B3", "probe_notau": "#64B5CD", "probe": "#DD8452", "stage2": "#4C72B0", "hold": "#C44E52"}
ML = {"persistence": "지속 (z_t 유지)", "linear": "선형 (릿지)", "probe_notau": "프로브, 궤적 없음", "probe": "비선형 프로브", "stage2": "Stage-2 B1 (s₀에서 개루프)", "hold": "GT z₀ 유지"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(items, vmax=None, fmt="{:.3f}", width=430, ref=None):
    """Horizontal bars on a SIGNED axis: negative values are drawn to the left of the zero line and intervals are never clipped."""
    vals = [v for _, v, *_ in items]; his = [c[1] for _, _, _, *c in items if c and not np.isnan(c[0])]; los = [c[0] for _, _, _, *c in items if c and not np.isnan(c[0])]
    vmax = vmax or max(max(vals + his + [r[1] for r in (ref or [])]), 1e-9) * 1.18; vmin = min(0.0, min(vals + los)) * 1.18
    h = 22 * len(items) + 10; left = 168; w = width - left - 62; sx = lambda v: left + (v - vmin) / (vmax - vmin) * w; x0 = sx(0.0)
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for i, (lab, v, col, *ci) in enumerate(items):
        y = 4 + 22 * i; xa, xb = min(x0, sx(v)), max(x0, sx(v))
        out.append(f'<text x="{left - 6}" y="{y + 14}" text-anchor="end" font-size="11" fill="var(--tx)">{esc(lab)}</text><rect x="{xa:.1f}" y="{y + 3}" width="{xb - xa:.1f}" height="15" rx="2" fill="{col}"/>')
        end = xb
        if ci and not np.isnan(ci[0]):
            out.append(f'<line x1="{sx(ci[0]):.1f}" x2="{sx(ci[1]):.1f}" y1="{y + 10.5}" y2="{y + 10.5}" stroke="var(--tx)" stroke-width="1.2"/>'); end = max(end, sx(ci[1]))
        out.append(f'<text x="{end + 5:.1f}" y="{y + 14}" font-size="11" fill="var(--tx)">{fmt.format(v)}</text>')
    if vmin < 0:
        out.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="2" y2="{h - 2}" stroke="var(--mt)" stroke-width="1"/>')
    for lab, v, col in (ref or []):
        x = sx(v); out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="2" y2="{h - 2}" stroke="{col}" stroke-dasharray="4 3"/><text x="{x + 3:.1f}" y="{h - 2}" font-size="9" fill="{col}">{esc(lab)}</text>')
    out.append("</svg>"); return "".join(out)


def pill_class(question_index, answer):
    """Colour of an answer pill. The 'beats persistence' answer lists three per-horizon labels; the rule uses h = 4 and 8, so its colour
    is the weaker of those two labels (clear -> green, marginal -> yellow, no -> red)."""
    if question_index == 3:
        lab = [x.strip() for x in answer.split("/")]; rel = lab[1:] if len(lab) == 3 else lab
        return "no" if "no" in rel else ("marginal" if "marginal" in rel else "clear")
    return answer.split()[0]


def lines(series, xs_list, width=440, height=210, ylim=(0, 1), xlim=None, xlabel="", ylabel="", points=None, band=None):
    left, bottom, top, right = 46, 30, 10, 12; w, h = width - left - right, height - top - bottom; xlim = xlim or (min(min(x) for x in xs_list), max(max(x) for x in xs_list))
    sx = lambda x: left + (x - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * w; sy = lambda y: top + (1 - (min(max(y, ylim[0]), ylim[1]) - ylim[0]) / max(ylim[1] - ylim[0], 1e-9)) * h
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for yt in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{left}" x2="{left + w}" y1="{sy(yt):.1f}" y2="{sy(yt):.1f}" stroke="var(--grid)"/><text x="{left - 4}" y="{sy(yt) + 4:.1f}" text-anchor="end" font-size="10" fill="var(--mt)">{yt:.2f}</text>')
    for xt in np.linspace(xlim[0], xlim[1], 5):
        out.append(f'<text x="{sx(xt):.1f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="var(--mt)">{xt:.2g}</text>')
    if band is not None:
        bx, lo, hi, col = band; pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(bx, hi)) + " " + " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(bx[::-1], lo[::-1]))
        out.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.18" stroke="none"/>')
    for (lab, ys, col, wd, dash), xs in zip(series, xs_list):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y) and xlim[0] <= x <= xlim[1])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    for (lab, x, y, col) in (points or []):
        out.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4.2" fill="{col}" stroke="var(--card)" stroke-width="1.2"/>')
    out.append(f'<text x="{left + w / 2:.0f}" y="{height - 1}" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(10,{top + h / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(ylabel)}</text></svg>')
    leg = [f'<span class="lg"><i style="background:{col}"></i>{esc(lab)}</span>' for lab, _, col, _, _ in series] + ([f'<span class="lg"><i style="background:{points[0][3]};border-radius:50%"></i>{esc(points[0][0])}</span>'] if points else [])
    return "".join(out) + f'<div class="legend">{" ".join(leg)}</div>'


def img(name):
    p = Z.OUT / "figures" / name
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{name}" loading="lazy">' if p.exists() else ""


def main():
    V = json.loads((Z.OUT / "report_values.json").read_text())
    raw = json.loads((Z.OUT / "narrative" / "dashboard_text.json").read_text()) if (Z.OUT / "narrative" / "dashboard_text.json").exists() else {}
    fill = lambda s: re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", lambda m: (format(V[m.group(1)], m.group(2)) if m.group(2) else str(V[m.group(1)])) if m.group(1) in V else f"??{m.group(1)}??", s)
    T = json.loads(fill(json.dumps(raw, ensure_ascii=False)))
    L, G, Q, S, D, B = (pd.read_csv(Z.OUT / n) for n in ("local_prediction_metrics.csv", "temporal_geometry_metrics.csv", "quadrant_metrics.csv", "stage2_comparison.csv", "decision_summary.csv", "temporal_geometry_bins.csv"))
    cfg = json.loads((Z.OUT / "experiment_config.json").read_text()); cases = cfg.get("cases", {})
    san = json.loads((Z.OUT / "sanity_summary.json").read_text()) if (Z.OUT / "sanity_summary.json").exists() else {}
    lm = lambda ds, h, m: L[(L.dataset == ds) & (L.h == h) & (L.method == m)].iloc[0]
    css = """
.marginal{background:var(--warn)}.clear{background:var(--ok)}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle}.dot.ok{background:var(--ok)}.dot.wn{background:var(--warn)}.dot.bd{background:var(--bad)}
:root{--bg:#f5f4ef;--card:#ffffff;--tx:#1c2230;--mt:#5d6676;--ac:#1f5fbf;--ok:#2e9e62;--warn:#d8a31a;--bad:#d1443c;--grid:#e5e2d9;--bd:#dcd9d0;color-scheme:light dark}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14171d;--card:#1d2128;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2c313a;--bd:#333944}}
:root[data-theme="dark"]{--bg:#14171d;--card:#1d2128;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2c313a;--bd:#333944}
body{background:var(--bg);color:var(--tx);font-family:"Noto Sans KR","Pretendard","Segoe UI",system-ui,sans-serif;margin:0;padding-block:24px;padding-inline:16px;line-height:1.5}
main{max-width:1080px;margin:0 auto}h1{font-size:1.45rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.12rem;margin:34px 0 10px;border-bottom:1px solid var(--bd);padding-bottom:4px}h3{font-size:.98rem;margin:12px 0 6px}
.sub{color:var(--mt);font-size:.9rem;margin-bottom:16px}.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}.verdict{font-size:1.3rem;font-weight:700}.case{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem;background:var(--ac)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;color:#fff;margin:2px 4px 2px 0}.yes{background:var(--ok)}.partial{background:var(--warn)}.no{background:var(--bad)}.mixed{background:var(--warn)}.Case{background:var(--ac)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:8px 0}
.kv div{border:1px solid var(--bd);border-radius:8px;padding:6px 8px}.kv .n{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}.kv .l{font-size:.75rem;color:var(--mt)}
table{border-collapse:collapse;font-size:.82rem;width:100%}th,td{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--mt);font-weight:600}
.tw{overflow-x:auto}.num{font-variant-numeric:tabular-nums}.legend{font-size:.76rem;color:var(--mt);margin-top:4px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:middle}
img{max-width:100%;border-radius:6px;border:1px solid var(--bd)}p{margin:6px 0}.ev{font-size:.82rem;color:var(--mt)}.note{font-size:.85rem;color:var(--mt)}ul{margin:4px 0 4px 18px;padding:0}
"""
    H = [f"<title>z Temporal Diagnostic</title><style>{css}</style><main>", "<h1>z를 거친 Stage-2 생성은 왜 실패했나</h1>",
         f"<div class='sub'>{esc(T.get('subtitle', ''))}</div>"]
    # ---- first screen: diagnosis cards
    H.append("<div class='hero'>")
    for ds in Z.DATASETS:
        c = cases.get(ds, "?"); d = D[D.dataset == ds]
        k = [(f"{V[f'p_{ds}_{h}_probe_gain_rmse'] * 100:.0f} %", f"지속 대비 이득 h={h} (윈도우 평균)") for h in Z.HORIZONS] + [(f"{V[f'sp_{ds}_4_probe_t0_gain_rmse'] * 100:.0f} %", "접촉 시작 프레임에서의 이득 (h=4)"),
             (f"{V[f'mean_spearman_{ds}']:.2f}", "ΔC–Δz 순위 상관"), (f"{V[f'mean_quadB_{ds}'] * 100:.1f} %", "ΔC 작고 Δz 큰 전이"), (f"{V[f'rescue_{ds}'] * 100:+.0f} %", "참 z₀의 구제 효과 (h=4, 8 평균)"),
             (f"{V[f's_{ds}_8_all_rescue'] * 100:+.0f} %", "8프레임 전 GT 상태의 구제 효과")]
        H.append(f"<div class='card'><div class='verdict'>{L_[ds]}: <span class='case'>{'Case ' + c if c != 'mixed' else '혼합'}</span></div><p>{esc(T.get('diagnosis', {}).get(ds, ''))}</p><div class='kv'>"
                 + "".join(f"<div><div class='n'>{a}</div><div class='l'>{esc(b)}</div></div>" for a, b in k) + "</div>"
                 + "".join(f"<span class='pill {pill_class(i, r.answer)}'>{esc(q)}: {esc(r.answer)}</span>" for i, (q, r) in enumerate(zip(["z_t+1 예측", "z_t+4 예측", "z_t+8 예측", "지속 초과 (h=1/4/8)", "Δz가 ΔC 추적", "참 z₀ 구제 (h=4·8 평균)", "규칙 판정"], d.itertuples())))
                 + "<p class='note'>알약은 사전 규칙의 답입니다. '지속 초과'의 색은 규칙이 쓰는 h=4, 8 중 약한 쪽을 따릅니다.</p></div>")
    H.append("</div>")
    H.append(f"<div class='card'><h3>한 줄 진단</h3><p>{esc(T.get('one_line', ''))}</p></div>")
    # ---- first screen charts: local predictability + delta_C vs delta_z
    H.append("<div class='grid2'>")
    for ds in Z.DATASETS:
        items = []
        for h in Z.HORIZONS:
            for m in ("persistence", "probe"):
                r = lm(ds, h, m); items.append((f"h={h} {ML[m]}", float(r.rmse), MC[m], float(r.rmse_lo), float(r.rmse_hi)))
        H.append(f"<div class='card'><h3>{L_[ds]} — 국소 예측 오차 (표준화 z의 RMSE ↓)</h3>{bars(items, ref=[('학습 평균 예측', float(lm(ds, 8, 'mean').rmse), '#888')])}"
                 f"<p class='note'>설명 분산 R²: " + ", ".join(f"h={h} 프로브 {lm(ds, h, 'probe').r2:.2f} / 지속 {lm(ds, h, 'persistence').r2:.2f}" for h in Z.HORIZONS) + "</p></div>")
    for ds in Z.DATASETS:
        b = B[(B.dataset == ds) & (B.h == 4)]; g = G[(G.dataset == ds) & (G.h == 4)].iloc[0]
        ymax = float(b.dz_q75.max()) * 1.1; xmax = float(b.dC_mean.max()) * 1.05
        H.append(f"<div class='card'><h3>{L_[ds]} — ΔC 구간별 Δz (h=4, 테스트)</h3>"
                 + lines([("Δz 평균 (띠: 사분위 범위)", b.dz_mean.values, MC["probe"], 2.2, ""), ("Δz = ΔC", np.array([0, xmax]), "#888", 1.0, "4 3")], [b.dC_mean.values, np.array([0, xmax])], ylim=(0, ymax), xlim=(0, xmax),
                         xlabel="ΔC (무작위 쌍 거리 대비)", ylabel="Δz (무작위 쌍 거리 대비)", band=(b.dC_mean.values, b.dz_q25.values, b.dz_q75.values, MC["probe"]))
                 + f"<p class='note'>Pearson {g.pearson_C_z:.2f}, Spearman {g.spearman_C_z:.2f}. 사분면: 둘 다 작음 {V[f'q_{ds}_4_A_lowC_lowz'] * 100:.1f} %, ΔC 작고 Δz 큼 {V[f'q_{ds}_4_B_lowC_highz'] * 100:.1f} %, ΔC 크고 Δz 작음 {V[f'q_{ds}_4_C_highC_lowz'] * 100:.1f} %, 둘 다 큼 {V[f'q_{ds}_4_D_highC_highz'] * 100:.1f} %.</p></div>")
    H.append("</div>")
    # ---- 1 question
    H.append("<h2>1. 무엇을 가리려 했나</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("question", [])) + "</div>")
    # ---- 2 local prediction
    H.append("<h2>2. 실험 A: 현재 GT z를 알면 미래 z가 예측되는가</h2><div class='grid2'>")
    for ds in Z.DATASETS:
        items = [(f"h={h} {ML[m]}", float(lm(ds, h, m).gain_rmse) * 100, MC[m], float(lm(ds, h, m).gain_rmse_lo) * 100, float(lm(ds, h, m).gain_rmse_hi) * 100) for h in Z.HORIZONS for m in ("linear", "probe_notau", "probe")]
        H.append(f"<div class='card'><h3>{L_[ds]} — 지속 대비 이득 (%) ↑</h3>{bars(items, fmt='{:.1f}', ref=[('명확 기준 10 %', 10.0, '#888')])}</div>")
    X = pd.read_csv(Z.OUT / "posthoc_splits.csv"); X = X[X.method == "probe"]
    def xrow(lab, ds, ev, sb, ps):
        g = [X[(X.dataset == ds) & (X.eval_split == ev) & (X.subject == sb) & (X.position == ps) & (X.h == h)].iloc[0] for h in Z.HORIZONS]
        cls = lambda r: "ok" if (r.gain_rmse >= 0.10 and r.gain_rmse_lo > 0) else ("wn" if (r.gain_rmse >= 0.02 and r.gain_rmse_lo > 0) else "bd")
        return f"<tr><td>{esc(lab)}</td>" + "".join(f"<td class='num'><span class='dot {cls(r)}'></span>{r.gain_rmse * 100:+.1f} % <span class='ev'>[{r.gain_rmse_lo * 100:+.1f}, {r.gain_rmse_hi * 100:+.1f}]</span></td>" for r in g) + "</tr>"
    rows_x = [("TACO 테스트, t = 0", "taco", "test", "all", "t0"), ("TACO 테스트, t = 1..7", "taco", "test", "all", "t1_7"), ("TACO 테스트, t ≥ 8", "taco", "test", "all", "t8"),
              ("ARCTIC 테스트, t = 0", "arctic", "test", "all", "t0"), ("ARCTIC 테스트, t = 1..7, 학습에 있는 피험자", "arctic", "test", "seen", "t1_7"), ("ARCTIC 테스트, t = 1..7, 학습에 없는 피험자", "arctic", "test", "unseen", "t1_7"),
              ("ARCTIC 테스트, t ≥ 8, 학습에 있는 피험자", "arctic", "test", "seen", "t8"), ("ARCTIC 테스트, t ≥ 8, 학습에 없는 피험자", "arctic", "test", "unseen", "t8"), ("ARCTIC 검증, t ≥ 8 (모두 학습 피험자)", "arctic", "val", "all", "t8")]
    xt = ("<h3>출발 위치별 프로브 이득 (지속 대비, 사후 분석)</h3><div class='tw'><table><tr><th>쌍</th><th>h = 1</th><th>h = 4</th><th>h = 8</th></tr>" + "".join(xrow(*r) for r in rows_x)
          + "</table></div><p class='note'>점 색: 초록 = 10 % 이상이고 구간이 0을 제외, 노랑 = 2 % 이상이고 구간이 0을 제외, 빨강 = 그 외. 규칙에 쓰이지 않는 사후 분할입니다.</p>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("local", [])) + xt + img("fig6_gain_by_window_position.png") + img("fig1_local_predictability.png") + "</div>")
    # ---- 3 vs stage 2
    H.append("<h2>3. Stage-2 개루프 예측과의 비교</h2><div class='grid2'>")
    for ds in Z.DATASETS:
        cv = np.load(Z.ds_out(ds) / "stage2_curves.npz"); f = np.arange(1, Z.T)
        pts = [("GT z₀에서 출발한 프로브 (h=1, 4, 8)", h, float(S[(S.dataset == ds) & (S.h == h) & (S["mode"] == "from_start")].rmse_probe.iloc[0]), "var(--tx)") for h in Z.HORIZONS]
        H.append(f"<div class='card'><h3>{L_[ds]} — 목표 프레임별 z 오차 (처음 16프레임)</h3>"
                 + lines([(ML["stage2"], cv["stage2"], MC["stage2"], 2.2, ""), (ML["hold"], cv["hold_z0"], MC["hold"], 1.3, "5 3")], [f, f], ylim=(0, 1.2), xlim=(1, 16), xlabel="목표 프레임 f", ylabel="표준화 RMSE", points=pts)
                 + "<p class='note'>" + " · ".join(f"h={h}: Stage-2 {V[f's_{ds}_{h}_start_rmse_stage2']:.2f} vs 프로브 {V[f's_{ds}_{h}_start_rmse_probe']:.2f} (구제 {V[f's_{ds}_{h}_start_rescue'] * 100:+.0f} %)" for h in Z.HORIZONS) + "</p></div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("stage2", [])) + img("fig2_local_vs_stage2.png") + "</div>")
    # ---- 4 geometry
    H.append("<h2>4. 실험 B: z의 움직임은 접촉의 움직임을 따르는가</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("geometry", [])) + img("fig3_deltaC_vs_deltaz.png") + img("fig4_quadrants.png") + "</div>")
    # ---- 5 examples
    H.append("<h2>5. 대표 전이</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("examples", [])) + "".join(img(f"fig5_transitions_{ds}.png") for ds in Z.DATASETS) + "</div>")
    # ---- 6 decision
    H.append("<h2>6. 최종 진단</h2><div class='card'><div class='tw'><table><tr><th>질문</th><th>TACO</th><th>ARCTIC</th><th>근거</th></tr>")
    for qi, q in enumerate(dict.fromkeys(D.question)):
        a = {ds: D[(D.dataset == ds) & (D.question == q)].iloc[0] for ds in Z.DATASETS}
        H.append(f"<tr><td>{esc(q)}</td>" + "".join(f"<td><span class='pill {pill_class(qi, a[ds].answer)}'>{esc(a[ds].answer)}</span></td>" for ds in Z.DATASETS) + "<td class='ev'>" + "<br>".join(f"<b>{L_[ds]}</b>: {esc(a[ds].evidence)}" for ds in a) + "</td></tr>")
    H.append("</table></div>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("decision", [])) + "</div>")
    if san:
        H.append("<div class='card'><h3>검증 요약</h3>" + "".join(f"<p><b>{L_[ds]}</b>: " + ", ".join(f"{k} {sum(1 for v in san[ds].values() if v['status'] == k)}" for k in ("pass", "warn", "fail")) + "</p>" for ds in san) + f"<p class='note'>{esc(T.get('sanity', ''))}</p></div>")
    H.append(f"<div class='card'><h3>파일</h3><p class='note'>{esc(T.get('files', ''))}</p></div></main>")
    (Z.OUT / "dashboard.html").write_text("\n".join(H)); print("dashboard.html written", (Z.OUT / "dashboard.html").stat().st_size // 1024, "KB; unresolved:", "\n".join(H).count("??"))


if __name__ == "__main__":
    main()
