#!/usr/bin/env python
"""Research dashboard (one HTML page, Korean) from the root tables: numbers are injected, the narrative strings live in
narrative/dashboard_text.json.    python dashboard.py -> <out>/dashboard.html (then published as a Claude artifact)."""
from __future__ import annotations

import base64
import json

import numpy as np
import pandas as pd

import cf_common as S

TXT = json.loads((S.OUT / "narrative" / "dashboard_text.json").read_text()) if (S.OUT / "narrative" / "dashboard_text.json").exists() else {}
L = {"taco": "TACO", "arctic": "ARCTIC"}


def load(n):
    p = S.OUT / n
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def bars(items, vmax=None, fmt="{:.3f}", width=420, lo_better=False, ref=None):
    """Horizontal bar chart as inline SVG. items: list of (label, value, color, lo, hi)."""
    vals = [v for _, v, *_ in items]; vmax = vmax or max(max(vals), 1e-9) * 1.15; h = 22 * len(items) + 8; left = 150; w = width - left - 70
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
            x = left + v / vmax * w; out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="2" y2="{h - 2}" stroke="{col}" stroke-dasharray="4 3"/><text x="{x + 3:.1f}" y="{h - 4}" font-size="9" fill="{col}">{esc(lab)}</text>')
    out.append("</svg>"); return "".join(out)


def lines(series, xs, width=440, height=200, ylim=(0, 1), xlabel="", ylabel=""):
    left, bottom, top, right = 42, 28, 10, 10; w, h = width - left - right, height - top - bottom
    sx = lambda x: left + (x - xs[0]) / (xs[-1] - xs[0]) * w; sy = lambda y: top + (1 - (y - ylim[0]) / (ylim[1] - ylim[0])) * h
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for yt in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{left}" x2="{left + w}" y1="{sy(yt):.1f}" y2="{sy(yt):.1f}" stroke="var(--grid)" stroke-width="1"/><text x="{left - 4}" y="{sy(yt) + 4:.1f}" text-anchor="end" font-size="10" fill="var(--mt)">{yt:.2f}</text>')
    for xt in np.linspace(xs[0], xs[-1], 5):
        out.append(f'<text x="{sx(xt):.1f}" y="{height - 10}" text-anchor="middle" font-size="10" fill="var(--mt)">{xt:.0f}</text>')
    for lab, ys, col, wd in series:
        pts = " ".join(f"{sx(x):.1f},{sy(min(max(y, ylim[0]), ylim[1])):.1f}" for x, y in zip(xs, ys))
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
    rec, pr, kn, sw, tp, dec, lat = (load(n) for n in ("reconstruction_metrics.csv", "probe_metrics.csv", "neighborhood_metrics.csv", "swap_metrics.csv", "temporal_persistence.csv", "decision_summary.csv", "latent_statistics.csv"))
    cfg = json.loads((S.OUT / "experiment_config.json").read_text()); cases = cfg.get("cases", {}); cases_v = cfg.get("cases_bottleneck_variant", {})
    if "model" in dec:
        dec_v = dec[dec.model == "A3_z16"]; dec = dec[dec.model == "A3"]
    else:
        dec_v = dec.iloc[0:0]
    san = json.loads((S.OUT / "sanity_summary.json").read_text()) if (S.OUT / "sanity_summary.json").exists() else {}
    g = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
    MC = {"A0": "#8a8f98", "A1": "#2e9e62", "A2": "#e4932d", "A3": "#d1443c"}
    css = """
:root{--bg:#f7f6f2;--card:#ffffff;--tx:#1d2330;--mt:#5f6878;--ac:#1f5fbf;--ok:#2e9e62;--warn:#d8a31a;--bad:#d1443c;--grid:#e6e3da;--bd:#dedbd2;color-scheme:light dark}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181e;--card:#1e2229;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2d323b;--bd:#343a45}}
:root[data-theme="dark"]{--bg:#15181e;--card:#1e2229;--tx:#e8eaee;--mt:#9aa3b2;--ac:#7aa7ef;--grid:#2d323b;--bd:#343a45}
body{background:var(--bg);color:var(--tx);font-family:"Noto Sans KR","Pretendard","Segoe UI",system-ui,sans-serif;margin:0;padding-block:24px;padding-inline:16px;line-height:1.5}
main{max-width:1100px;margin:0 auto}h1{font-size:1.5rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.15rem;margin:36px 0 10px;border-bottom:1px solid var(--bd);padding-bottom:4px}h3{font-size:1rem;margin:14px 0 6px}
.sub{color:var(--mt);font-size:.9rem;margin-bottom:18px}.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px}.verdict{font-size:1.5rem;font-weight:700}.case{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:.9rem}
.A{background:var(--ok)}.B{background:var(--warn)}.C{background:var(--bad)}.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;color:#fff;margin-right:4px}
.yes{background:var(--ok)}.partial{background:var(--warn)}.no{background:var(--bad)}.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}
table{border-collapse:collapse;font-size:.82rem;width:100%}th,td{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--mt);font-weight:600}
.tw{overflow-x:auto}.num{font-variant-numeric:tabular-nums}.legend{font-size:.78rem;color:var(--mt);margin-top:4px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:middle}
img{max-width:100%;border-radius:6px;border:1px solid var(--bd)}p{margin:6px 0}.ev{font-size:.82rem;color:var(--mt)}.note{font-size:.85rem;color:var(--mt)}ul{margin:4px 0 4px 18px;padding:0}
"""
    H = [f"<title>Contact Factorization Stage 1</title><style>{css}</style><main>",
         "<h1>접촉 맵 분해 1단계: 구조 잠재 z와 실현 코드 r</h1>",
         f"<div class='sub'>TACO·ARCTIC, 단일 프레임 표현 실험 (2026-10-02). 결정 임계값은 결과를 보기 전에 고정했고(experiment_config.json), 불확실성은 테이크 단위 부트스트랩 95 % 구간입니다.</div>"]
    # ---- first screen: did it work?
    H.append("<div class='hero'>")
    for ds in S.DATASETS:
        c = cases.get(ds, "?"); d = dec[dec.dataset == ds]
        a3, a0 = g(rec, dataset=ds, model="A3"), g(rec, dataset=ds, model="A0"); pz, pr_ = g(pr, dataset=ds, input="A3:z", probe="mlp"), g(pr, dataset=ds, input="A3:r", probe="mlp")
        s3 = g(sw, dataset=ds, model="A3"); acf = g(tp, dataset=ds, code="A3:r", metric="autocorrelation", h=8)
        H.append(f"<div class='card'><div class='verdict'>{L[ds]}: <span class='case {c}'>Case {c}</span></div><p>{esc(TXT.get('verdict', {}).get(ds, ''))}</p>"
                 f"<ul class='ev'><li>재구성: z만 {a3.E_zonly:.2f} → z+r {a3.E_full:.2f} (A0 {a0.E_full:.2f}), Gain_r {a3.gain_r:.2f}</li>"
                 f"<li>구조 보존(프로브): z {pz.structure_retention:.2f} vs r {pr_.structure_retention:.2f}</li>"
                 f"<li>교환: 구조가 z를 따르는 비율 {100 * s3.frac_struct_follows_z:.0f} %, 세부가 r을 따르는 비율 {100 * s3.frac_detail_follows_r:.0f} %</li>"
                 f"<li>r 지속성: 자기상관(h=8) {acf.value:.2f}, 과거 r의 추가 설명력 {g(tp, dataset=ds, code='A3:r', metric='r2_gain_past_beyond_z3', h=8).value:.2f}</li></ul>"
                 + "".join(f"<span class='pill {r.answer.split()[0]}'>{esc(q)}</span>" for q, r in zip(["z 구조", "r 밀집 이득", "z>r 집중", "이웃", "교환", "r 지속"], d.itertuples())) + "</div>")
    H.append("</div>")
    H.append(f"<div class='card'><h3>한 줄 결론</h3><p>{esc(TXT.get('one_line', ''))}</p></div>")
    # ---- 1 hypothesis
    H.append("<h2>1. 1단계 가설</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("hypothesis", [])) + "</div>")
    # ---- 2 architecture
    H.append("<h2>2. 구조</h2><div class='card'>" + img("fig1_architecture.png") + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("architecture", [])) + "</div>")
    # ---- 3 reconstruction
    H.append("<h2>3. 재구성 비교</h2><div class='grid2'>")
    for ds in S.DATASETS:
        r = rec[rec.dataset == ds]; items = []
        for m in S.MODELS:
            q = g(r, model=m); items.append((f"{m} z만 (C̄)" if m != "A0" else "A0 h", q.E_zonly, MC[m] + "99", q.E_zonly_lo, q.E_zonly_hi)); items.append((f"{m} 전체 (Ĉ)", q.E_full, MC[m], q.E_full_lo, q.E_full_hi))
        ref = [(f"PCA-64 {g(r, model='pca64').E_full:.2f}", g(r, model="pca64").E_full, "#777"), (f"PCA-128 {g(r, model='pca128').E_full:.2f}", g(r, model="pca128").E_full, "#333")]
        a3 = g(r, model="A3")
        H.append(f"<div class='card'><h3>{L[ds]} — 테스트 재구성 오차 ‖Ĉ − C‖</h3>{bars(items, ref=ref, fmt='{:.2f}')}<p class='note'>평균 맵 기준선 {g(r, model='mean').E_full:.2f}, 메시별 평균 {g(r, model='mesh_mean').E_full:.2f}. A3: Gain_r = {a3.gain_r:.2f} [{a3.gain_r_lo:.2f}, {a3.gain_r_hi:.2f}], r가 제거한 z-잔차 에너지 {100 * a3.explained_by_r:.0f} %, R²(C̄) {a3.R2_zonly:.3f} / R²(Ĉ) {a3.R2_full:.3f}; 이전 연구의 정확 R2 디코더 R² {g(r, model='R2_decoder_prev').R2_full:.3f}.</p></div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("reconstruction", [])) + img("fig2_reconstruction_examples.png") + "</div>")
    # ---- 4 information
    H.append("<h2>4. 구조 정보는 어디에 있나 (z vs r 프로브)</h2><div class='grid2'>")
    for ds in S.DATASETS:
        p = pr[(pr.dataset == ds) & (pr.probe == "mlp")]; cols = {"A3:z": "#2b6cd4", "A3:r": "#d1443c", "A3:zr": "#7b4fc4", "A2:z": "#9fb9e6", "A2:r": "#e9a19b", "A0:h": "#8a8f98", "C": "#222", "C_pca64": "#777"}
        tv = g(pr, dataset=ds, probe="constant")
        for k, lab, lo_b in (("structure_retention", "구조 보존률 (5개 양 평균; C 프로브 = 1, 평균 예측 = 0)", False), ("part_auprc", "참여 macro AUPRC ↑", False), ("wrench_ev", "렌치 설명 분산 ↑", False), ("normal", "법선 각 오차 (°) ↓", True)):
            items = [(inp, float(g(p, input=inp)[k]), cols[inp], float(g(p, input=inp).get(k + "_lo", np.nan)), float(g(p, input=inp).get(k + "_hi", np.nan))) for inp in cols if (p.input == inp).any()]
            ref = [(f"평균 예측 {tv[k]:.2f}", float(tv[k]), "#999")] if k != "structure_retention" else None
            H.append(f"<div class='card'><h3>{L[ds]} — {lab}</h3>{bars(items, fmt='{:.2f}', ref=ref)}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("information", [])) + "</div>")
    # ---- 5 neighbours
    H.append("<h2>5. 최근접 이웃 (같은 메시, 다른 테이크)</h2><div class='grid2'>")
    for ds in S.DATASETS:
        k = kn[kn.dataset == ds]; cols = {"A3:z": "#2b6cd4", "A3:r": "#d1443c", "A2:z": "#9fb9e6", "A1:z": "#2e9e62", "A0:h": "#8a8f98", "C": "#222", "teacher": "#1a9c6e", "random": "#bbb"}
        items = [(sp, float(g(k, space=sp).teacher), cols[sp], float(g(k, space=sp).teacher_lo), float(g(k, space=sp).teacher_hi)) for sp in cols if (k.space == sp).any()]
        items2 = [(sp, float(g(k, space=sp).recall), cols[sp], float(g(k, space=sp).recall_lo), float(g(k, space=sp).recall_hi)) for sp in cols if (k.space == sp).any()]
        H.append(f"<div class='card'><h3>{L[ds]} — 10개 이웃의 교사(R2+렌치) 거리 ↓</h3>{bars(items, fmt='{:.3f}')}<h3>교사 이웃과의 recall@10 ↑</h3>{bars(items2, fmt='{:.2f}')}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("neighbors", [])) + img("fig5_neighbors.png") + "</div>")
    # ---- 6 swap
    H.append("<h2>6. 잠재 교환 테스트</h2><div class='grid2'>")
    for ds in S.DATASETS:
        s3 = g(sw, dataset=ds, model="A3"); s2 = g(sw, dataset=ds, model="A2") if ((sw.dataset == ds) & (sw.model == "A2")).any() else None
        rows = [("구조 거리 → z 제공자 / → r 제공자 / 자기재구성 바닥", f"{s3.struct_to_zdonor:.3f} / {s3.struct_to_rdonor:.3f} / {s3.struct_floor:.3f}"), ("구조가 z를 따르는 비율", f"{100 * s3.frac_struct_follows_z:.0f} % [{100 * s3.frac_struct_follows_z_lo:.0f}, {100 * s3.frac_struct_follows_z_hi:.0f}]"),
                ("세부 코사인 → r 제공자 / → z 제공자", f"{s3.detail_cos_rdonor:.2f} / {s3.detail_cos_zdonor:.2f}"), ("세부가 r을 따르는 비율", f"{100 * s3.frac_detail_follows_r:.0f} % [{100 * s3.frac_detail_follows_r_lo:.0f}, {100 * s3.frac_detail_follows_r_hi:.0f}]"),
                ("r 교환으로 맵이 움직인 거리 / 쌍 거리", f"{s3.dense_change:.2f} / {s3.dense_AB:.2f}"), ("쌍 수 (양방향 교환)", f"{int(s3.n_pairs)} ({int(s3.n_swaps)})")]
        if s2 is not None:
            rows.append(("A2 (관계 손실 없음): 구조→z 비율 / 세부→r 비율", f"{100 * s2.frac_struct_follows_z:.0f} % / {100 * s2.frac_detail_follows_r:.0f} %"))
        H.append(f"<div class='card'><h3>{L[ds]} — A3</h3><div class='tw'><table>" + "".join(f"<tr><th>{esc(a)}</th><td class='num'>{esc(b)}</td></tr>" for a, b in rows) + "</table></div></div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("swap", [])) + img("fig6_swap.png") + "</div>")
    # ---- 7 temporal
    H.append("<h2>7. 시간 지속성 (GT 궤적을 프레임별로 인코딩)</h2><div class='grid2'>")
    for ds in S.DATASETS:
        cv = np.load(S.ds_out(ds) / "temporal_curves.npz"); codes = [("A3 r", "A3:r", "#d1443c", 2.2), ("A3 z", "A3:z", "#2b6cd4", 2.2), ("A0 h", "A0:h", "#8a8f98", 1.4), ("밀집 잔차 C−C̄", "A3:resid_z", "#7b4fc4", 1.4), ("밀집 C", "dense_C", "#222", 1.2), ("교사 R2+렌치", "teacher", "#1a9c6e", 1.2)]
        ser = [(lab, cv[f"{k}__acf"][:33], col, wd) for lab, k, col, wd in codes if f"{k}__acf" in cv]
        t = tp[tp.dataset == ds]; items = [(lab, float(g(t, code=k, metric="displacement_ratio", h=8).value), col, float(g(t, code=k, metric="displacement_ratio", h=8).lo), float(g(t, code=k, metric="displacement_ratio", h=8).hi)) for lab, k, col, _ in codes if (t.code == k).any()]
        gain = [(lab, float(g(t, code=k, metric="r2_gain_past_beyond_z3", h=8).value), col) for lab, k, col, _ in codes if ((t.code == k) & (t.metric == "r2_gain_past_beyond_z3") & (t.h == 8)).any()]
        H.append(f"<div class='card'><h3>{L[ds]} — 자기상관 vs 지연 h</h3>{lines(ser, np.arange(33), ylim=(0, 1), xlabel='h (프레임)', ylabel='자기상관')}<h3>변위 비율 h=8 (무작위 쌍 거리 대비) ↓</h3>{bars(items, fmt='{:.2f}')}<h3>x_t가 z_(t+8) 너머로 x_(t+8)을 설명하는 R² 이득</h3>{bars([(a, b, c) for a, b, c in gain], fmt='{:.2f}')}</div>")
    H.append("</div><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("temporal", [])) + "</div>")
    # ---- 8 decision
    H.append("<h2>8. 최종 결정</h2><div class='card'><div class='tw'><table><tr><th>질문</th><th>TACO</th><th>ARCTIC</th><th>근거</th></tr>")
    for q in list(dict.fromkeys(dec.question)):
        a = {ds: dec[(dec.dataset == ds) & (dec.question == q)].iloc[0] for ds in S.DATASETS if ((dec.dataset == ds) & (dec.question == q)).any()}
        H.append(f"<tr><td>{esc(q)}</td>" + "".join(f"<td><span class='pill {a[ds].answer.split()[0]}'>{esc(a[ds].answer)}</span></td>" if ds in a else "<td>–</td>" for ds in S.DATASETS) + "<td class='ev'>" + "<br>".join(f"<b>{L[ds]}</b>: {esc(a[ds].evidence)}" for ds in a) + "</td></tr>")
    H.append("</table></div>" + "".join(f"<p>{esc(p)}</p>" for p in TXT.get("decision", [])) + "</div>")
    if san:
        H.append("<div class='card'><h3>검증 요약</h3>" + "".join(f"<p><b>{L[ds]}</b>: " + ", ".join(f"{k} {v}" for k, v in san["counts"][ds].items()) + "</p>" for ds in san["counts"]) + f"<p class='note'>{esc(TXT.get('sanity', ''))}</p></div>")
    H.append(f"<div class='card'><h3>파일</h3><p class='note'>{esc(TXT.get('files', ''))}</p></div></main>")
    (S.OUT / "dashboard.html").write_text("\n".join(H)); print("dashboard.html written", (S.OUT / "dashboard.html").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
