#!/usr/bin/env python
"""Research dashboard (one HTML page, Korean) from the root tables: numbers are injected, the narrative strings live in
narrative/dashboard_text.json.    python s2_dashboard.py -> OUT/dashboard.html (then published as a Claude artifact)."""
from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd

import s2_common as S

TXT = json.loads((S.OUT / "narrative" / "dashboard_text.json").read_text()) if (S.OUT / "narrative" / "dashboard_text.json").exists() else {}
L = {"taco": "TACO", "arctic": "ARCTIC"}
MC = {"B0": "#4C72B0", "B1": "#DD8452", "B2": "#55A868", "B2_zonly": "#8FD19E", "D0_prev": "#9E9E9E", "PERSIST": "#C44E52", "GT": "#222222", "hold_z0": "#B07AA1", "train_mean": "#BBBBBB"}
ML = {"B0": "B0 직접 밀집", "B1": "B1 z→C", "B2": "B2 (z,r)→C", "B2_zonly": "B2 z만 경로", "D0_prev": "이전 D0 (25.7M, 3시드)", "PERSIST": "지속 (C_t = s₀)", "GT": "GT 추출 바닥", "hold_z0": "z*₀ 유지", "train_mean": "학습 평균 z"}


def load(n):
    p = S.OUT / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(items, vmax=None, fmt="{:.3f}", width=420, ref=None):
    vals = [v for _, v, *_ in items]; vmax = vmax or max(max(vals + [r[1] for r in (ref or [])]), 1e-9) * 1.15; h = 22 * len(items) + 10; left = 150; w = width - left - 70
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for i, (lab, v, col, *ci) in enumerate(items):
        y = 4 + 22 * i; bw = max(0.0, v) / vmax * w
        out.append(f'<text x="{left - 6}" y="{y + 14}" text-anchor="end" font-size="11" fill="var(--tx)">{esc(lab)}</text>')
        out.append(f'<rect x="{left}" y="{y + 3}" width="{bw:.1f}" height="15" rx="2" fill="{col}"/>')
        if ci and not np.isnan(ci[0]):
            x0, x1 = left + max(ci[0], 0) / vmax * w, left + ci[1] / vmax * w
            out.append(f'<line x1="{x0:.1f}" x2="{x1:.1f}" y1="{y + 10.5}" y2="{y + 10.5}" stroke="var(--tx)" stroke-width="1.2"/>')
        out.append(f'<text x="{left + bw + 5:.1f}" y="{y + 14}" font-size="11" fill="var(--tx)">{fmt.format(v)}</text>')
    if ref is not None:
        for lab, v, col in ref:
            x = left + v / vmax * w; out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="2" y2="{h - 2}" stroke="{col}" stroke-dasharray="4 3"/><text x="{x + 3:.1f}" y="{h - 2}" font-size="9" fill="{col}">{esc(lab)}</text>')
    out.append("</svg>"); return "".join(out)


def lines(series, xs, width=440, height=200, ylim=(0, 1), xlabel="", ylabel=""):
    left, bottom, top, right = 46, 28, 10, 10; w, h = width - left - right, height - top - bottom
    sx = lambda x: left + (x - xs[0]) / max(xs[-1] - xs[0], 1e-9) * w; sy = lambda y: top + (1 - (y - ylim[0]) / max(ylim[1] - ylim[0], 1e-9)) * h
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for yt in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{left}" x2="{left + w}" y1="{sy(yt):.1f}" y2="{sy(yt):.1f}" stroke="var(--grid)" stroke-width="1"/><text x="{left - 4}" y="{sy(yt) + 4:.1f}" text-anchor="end" font-size="10" fill="var(--mt)">{yt:.2f}</text>')
    for xt in np.linspace(xs[0], xs[-1], 5):
        out.append(f'<text x="{sx(xt):.1f}" y="{height - 10}" text-anchor="middle" font-size="10" fill="var(--mt)">{xt:.0f}</text>')
    for lab, ys, col, wd in series:
        pts = " ".join(f"{sx(x):.1f},{sy(min(max(y, ylim[0]), ylim[1])):.1f}" for x, y in zip(xs, ys) if np.isfinite(y))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}"/>')
    out.append(f'<text x="{left + w / 2:.0f}" y="{height - 0}" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(10,{top + h / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(ylabel)}</text></svg>')
    legend = " ".join(f'<span class="lg"><i style="background:{col}"></i>{esc(lab)}</span>' for lab, _, col, _ in series)
    return "".join(out) + f'<div class="legend">{legend}</div>'


def img(path):
    p = S.OUT / "figures" / path
    if not p.exists():
        return ""
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{path}" loading="lazy">'


def main():
    acc, P, zm, rm, ts, dec = (load(n) for n in ("main_metrics.csv", "paired_comparisons.csv", "z_metrics.csv", "r_metrics.csv", "training_summary.csv", "decision_summary.csv"))
    cfg = json.loads((S.OUT / "experiment_config.json").read_text()); cases = cfg.get("cases", {})
    curves = dict(np.load(S.OUT / "temporal_curves.npz"))
    san = json.loads((S.OUT / "sanity_summary.json").read_text()) if (S.OUT / "sanity_summary.json").exists() else {}
    g = lambda ds, m, met: acc[(acc.dataset == ds) & (acc.model == m) & (acc.metric == met)].iloc[0]
    has = lambda ds, m, met: ((acc.dataset == ds) & (acc.model == m) & (acc.metric == met)).any()
    pr = lambda ds, a, b, met: P[(P.dataset == ds) & (P.a == a) & (P.b == b) & (P.metric == met)].iloc[0]
    css = """
:root{--bg:#f6f5f1;--card:#ffffff;--tx:#1d2330;--mt:#5f6878;--ac:#1f5fbf;--ok:#2e9e62;--warn:#d8a31a;--bad:#d1443c;--grid:#e6e3da;--bd:#dedbd2;color-scheme:light dark}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181e;--card:#1e2229;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2d323b;--bd:#343a45}}
:root[data-theme="dark"]{--bg:#15181e;--card:#1e2229;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2d323b;--bd:#343a45}
body{background:var(--bg);color:var(--tx);font-family:"Noto Sans KR","Pretendard","Segoe UI",system-ui,sans-serif;margin:0;padding-block:24px;padding-inline:16px;line-height:1.5}
main{max-width:1100px;margin:0 auto}h1{font-size:1.5rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.15rem;margin:36px 0 10px;border-bottom:1px solid var(--bd);padding-bottom:4px}h3{font-size:1rem;margin:14px 0 6px}
.sub{color:var(--mt);font-size:.9rem;margin-bottom:18px}.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}.verdict{font-size:1.4rem;font-weight:700}.case{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem}
.A{background:var(--ok)}.B{background:#3a8f5a}.C{background:#2d7fbf}.D{background:var(--warn)}.E{background:var(--bad)}.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;color:#fff;margin:2px 4px 2px 0}
.yes{background:var(--ok)}.partial{background:var(--warn)}.no{background:var(--bad)}.worse{background:var(--bad)}.B0{background:#4C72B0}.B1{background:#DD8452}.B2{background:#55A868}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
table{border-collapse:collapse;font-size:.82rem;width:100%}th,td{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--mt);font-weight:600}
.tw{overflow-x:auto}.num{font-variant-numeric:tabular-nums}.legend{font-size:.78rem;color:var(--mt);margin-top:4px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:middle}
img{max-width:100%;border-radius:6px;border:1px solid var(--bd)}p{margin:6px 0}.ev{font-size:.82rem;color:var(--mt)}.note{font-size:.85rem;color:var(--mt)}ul{margin:4px 0 4px 18px;padding:0}.big{font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:.85rem}.better{color:var(--ok);font-weight:600}.similar{color:var(--mt)}
"""
    H = [f"<title>Contact Latent Temporal Stage 2</title><style>{css}</style><main>",
         "<h1>접촉 생성 2단계: 잠재 z를 거치면 시간적 접촉 생성이 좋아지는가?</h1>",
         "<div class='sub'>TACO·ARCTIC, 고정 GT s₀ 프로토콜, 동일 백본(68 M)·동일 입력(s₀, G, τ)의 B0 / B1 / B2 (2026-10-03, 시드 1개). 결정 규칙은 결과를 보기 전에 고정했고(experiment_config.json), 불확실성은 테이크 단위 부트스트랩 95 % 구간입니다.</div>"]
    # ---- first screen
    H.append("<div class='hero'>")
    for ds in S.DATASETS:
        c = cases.get(ds, "?"); d = dec[(dec.dataset == ds) & (dec.question != "case")]
        e = {m: g(ds, m, "E_C") for m in ("B0", "B1", "B2")}; best = d[d.question.str.startswith("Which")].iloc[0].answer if (d.question.str.startswith("Which")).any() else "?"
        p10, p21 = pr(ds, "B1", "B0", "E_C"), pr(ds, "B2", "B1", "E_C")
        H.append(f"<div class='card'><div class='verdict'>{L[ds]}: <span class='case {c}'>Case {c}</span> · 최선 <span class='pill {best.split()[0]}'>{esc(best.split()[0])}</span></div><p>{esc(TXT.get('verdict', {}).get(ds, ''))}</p>"
                 f"<div class='grid3'>" + "".join(f"<div><div class='note'>{ML[m]} E_C</div><div class='big' style='color:{MC[m]}'>{e[m].value:.3f}</div><div class='note'>[{e[m].ci_lo:.3f}, {e[m].ci_hi:.3f}]</div></div>" for m in ("B0", "B1", "B2")) + "</div>"
                 f"<ul class='ev'><li>B1 − B0 밀집 오차 {p10['diff']:+.3f} [{p10.ci_lo:+.3f}, {p10.ci_hi:+.3f}] ({p10.rel_improvement * 100:+.1f} %, {p10.verdict}); 참여 Hamming {pr(ds, 'B1', 'B0', 'part_hamming').rel_improvement * 100:+.1f} %, 렌치 {pr(ds, 'B1', 'B0', 'q_rel_l1').rel_improvement * 100:+.1f} %</li>"
                 f"<li>B2 − B1 밀집 오차 {p21['diff']:+.3f} [{p21.ci_lo:+.3f}, {p21.ci_hi:+.3f}] ({p21.rel_improvement * 100:+.1f} %, {p21.verdict}); 참여 Hamming {pr(ds, 'B2', 'B1', 'part_hamming').rel_improvement * 100:+.1f} %, 렌치 {pr(ds, 'B2', 'B1', 'q_rel_l1').rel_improvement * 100:+.1f} %</li>"
                 f"<li>이전 D0 {g(ds, 'D0_prev', 'E_C').value:.3f} → B0 {e['B0'].value:.3f}; 지속 {g(ds, 'PERSIST', 'E_C').value:.3f}</li></ul>"
                 + "".join(f"<span class='pill {r.answer.split()[0]}'>{esc(q)}</span>" for q, r in zip(["z→밀집", "z→구조·렌치", "z→장기 안정", "r→밀집", "r가 z 구조 보존", "최선 모델"], d.itertuples())) + "</div>")
    H.append("</div>")
    H.append(f"<div class='card'><h3>한 줄 결론</h3><p>{esc(TXT.get('one_line', ''))}</p></div>")
    # ---- 1 hypothesis
    H.append("<h2>1. 가설</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("hypothesis", [])) + "</div>")
    # ---- 2 architecture
    H.append("<h2>2. 세 모델의 구조</h2><div class='card'>" + img("fig1_schematic.png") + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("architecture", [])))
    if len(ts):
        H.append("<div class='tw'><table><tr><th>데이터셋</th><th>모델</th><th>파라미터</th><th>스텝</th><th>최적 스텝</th><th>종료</th><th>최적 검증 목적함수</th><th>소요</th></tr>" + "".join(
            f"<tr><td>{L[r.dataset]}</td><td>{r.model}</td><td class='num'>{r.n_params_total / 1e6:.1f} M</td><td class='num'>{int(r.steps)}</td><td class='num'>{int(r.best_step)}</td><td>{r.stopped_by}</td><td class='num'>{r.best_val:.4f}</td><td class='num'>{r.hours:.1f} h</td></tr>" for r in ts.itertuples()) + "</table></div>")
    H.append("</div>")
    # ---- 3 main metrics
    H.append("<h2>3. 주요 지표 (테스트, 고정 s₀)</h2><div class='grid2'>")
    for ds in S.DATASETS:
        for met, lab, fmt in (("E_C", "밀집 오차 E_C ↓", "{:.3f}"), ("part_hamming", "참여 Hamming ↓", "{:.3f}"), ("centroid", "중심 오차 / l ↓", "{:.3f}"), ("normal", "법선 각 오차 (°) ↓", "{:.1f}"), ("q_rel_l1", "렌치 상대 L1 ↓", "{:.3f}"), ("amount_l1", "접촉량 L1 ↓", "{:.3f}")):
            items = [(ML[m], float(g(ds, m, met).value), MC[m], float(g(ds, m, met).ci_lo), float(g(ds, m, met).ci_hi)) for m in ("B0", "B1", "B2", "B2_zonly") if has(ds, m, met)]
            ref = [(f"{ML[m]} {fmt.format(g(ds, m, met).value)}", float(g(ds, m, met).value), MC[m]) for m in ("D0_prev", "GT") if has(ds, m, met)]
            if met != "E_C" and has(ds, "PERSIST", met):
                ref.append((f"지속 {fmt.format(g(ds, 'PERSIST', met).value)}", float(g(ds, "PERSIST", met).value), MC["PERSIST"]))
            H.append(f"<div class='card'><h3>{L[ds]} — {lab}</h3>{bars(items, fmt=fmt, ref=ref)}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("main", [])) + img("fig2_main_metrics.png") + "</div>")
    # ---- 4 horizon
    H.append("<h2>4. 지평선에 따른 오차</h2><div class='grid2'>")
    for ds in S.DATASETS:
        ser = [(ML[m], curves[f"{ds}|{m}|dense"][1:], MC[m], 2.0 if m in ("B0", "B1", "B2") else 1.2) for m in ("B0", "B1", "B2", "D0_prev") if f"{ds}|{m}|dense" in curves]
        ymax = max(np.nanmax(s[1]) for s in ser) * 1.1
        H.append(f"<div class='card'><h3>{L[ds]} — 프레임별 밀집 오차 ‖Ĉ_t − C_t‖</h3>{lines(ser, np.arange(1, 64), ylim=(0, ymax), xlabel='프레임 t', ylabel='오차')}")
        ser2 = [(ML[m], curves[f'{ds}|{m}|part'][1:], MC[m], 2.0 if m in ('B0', 'B1', 'B2') else 1.2) for m in ("B0", "B1", "B2", "GT") if f"{ds}|{m}|part" in curves]
        H.append(f"<h3>프레임별 참여 Hamming</h3>{lines(ser2, np.arange(1, 64), ylim=(0, max(np.nanmax(s[1]) for s in ser2) * 1.1), xlabel='프레임 t', ylabel='Hamming')}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("horizon", [])) + img("fig3_horizon.png") + img("fig4_stability.png") + "</div>")
    # ---- 5 z diagnostics
    H.append("<h2>5. z 진단 (B1·B2의 ẑ vs 교사 z*)</h2><div class='grid2'>")
    for ds in S.DATASETS:
        ser = [(ML[m], curves[f"{ds}|{m}|z_err"], MC[m], 2.0 if m in ("B1", "B2") else 1.2) for m in ("B1", "B2", "hold_z0", "train_mean") if f"{ds}|{m}|z_err" in curves]
        z = zm[(zm.dataset == ds) & (zm.model.isin(["B1", "B2"]))]
        items = [(ML[r.model], float(r.L_z_test), MC[r.model], float(r.L_z_lo), float(r.L_z_hi)) for r in z.itertuples()]
        ref = [(f"z*₀ 유지 {z.iloc[0].L_z_hold_z0:.2f}", float(z.iloc[0].L_z_hold_z0), MC["hold_z0"]), (f"학습 평균 {z.iloc[0].L_z_train_mean:.2f}", float(z.iloc[0].L_z_train_mean), MC["train_mean"])] if len(z) else None
        rr = rm[rm.dataset == ds]
        extra = f"<p class='note'>B2의 r 경로: ΔC로 인한 E_C 이득 {rr.iloc[0].E_C_gain_from_r:+.3f} [{rr.iloc[0].gain_lo:+.3f}, {rr.iloc[0].gain_hi:+.3f}], ‖ΔC‖/‖C̄ − s₀‖ = {rr.iloc[0].delta_over_bar_ratio:.2f}, r̂ 유효 차원 {rr.iloc[0].r_eff_rank:.1f}, r̂ 자기상관(h=8) {rr.iloc[0].r_acf_h8:.2f}</p>" if len(rr) else ""
        H.append(f"<div class='card'><h3>{L[ds]} — 프레임별 z 오차 RMSE (표준화 좌표)</h3>{lines(ser, np.arange(1, 64), ylim=(0, max(np.nanmax(s[1]) for s in ser) * 1.1), xlabel='프레임 t', ylabel='RMSE')}<h3>테스트 L_z ↓</h3>{bars(items, fmt='{:.3f}', ref=ref)}{extra}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("z", [])) + "</div>")
    # ---- 6 qualitative
    H.append("<h2>6. 정성적 예시</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("qualitative", [])) + "".join(img(f"fig5{c}_{ds}.png") for ds in S.DATASETS for c in "abc") + "</div>")
    # ---- 7 decision
    H.append("<h2>7. 최종 결정</h2><div class='card'><div class='tw'><table><tr><th>질문</th><th>TACO</th><th>ARCTIC</th><th>근거</th></tr>")
    for q in [q for q in dict.fromkeys(dec.question) if q != "case"]:
        a = {ds: dec[(dec.dataset == ds) & (dec.question == q)].iloc[0] for ds in S.DATASETS if ((dec.dataset == ds) & (dec.question == q)).any()}
        H.append(f"<tr><td>{esc(q)}</td>" + "".join(f"<td><span class='pill {a[ds].answer.split()[0]}'>{esc(a[ds].answer)}</span></td>" if ds in a else "<td>–</td>" for ds in S.DATASETS) + "<td class='ev'>" + "<br>".join(f"<b>{L[ds]}</b>: {esc(a[ds].evidence)}" for ds in a) + "</td></tr>")
    H.append("</table></div>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("decision", [])) + "</div>")
    if san:
        H.append("<div class='card'><h3>검증 요약</h3>" + "".join(f"<p><b>{L[ds]}</b>: " + ", ".join(f"{k} {sum(1 for v in san[ds].values() if v['status'] == k)}" for k in ("pass", "warn", "info", "fail")) + "</p>" for ds in san) + f"<p class='note'>{esc(TXT.get('sanity', ''))}</p></div>")
    H.append(f"<div class='card'><h3>파일</h3><p class='note'>{esc(TXT.get('files', ''))}</p></div></main>")
    (S.OUT / "dashboard.html").write_text("\n".join(H)); print("dashboard.html written", (S.OUT / "dashboard.html").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
