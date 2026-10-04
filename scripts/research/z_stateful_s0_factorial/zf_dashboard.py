#!/usr/bin/env python
"""Small research dashboard (one HTML page, Korean): what was wrong with the previous z-mediated generator?    python zf_dashboard.py
Numbers are injected from the result tables; the prose lives in narrative/dashboard_text.json with the report's <<key:format>> placeholders
(resolved against OUT/report_values.json).  -> OUT/dashboard.html (published as a Claude artifact)."""
from __future__ import annotations

import base64
import json
import re

import numpy as np
import pandas as pd

import zf_common as Z
import zj_dashboard as ZD

esc, bars, lines = ZD.esc, ZD.bars, ZD.lines
L_ = {"taco": "TACO", "arctic": "ARCTIC"}
MC = {"M00": "#DD8452", "M01": "#C44E52", "M10": "#2e9e62", "M11": "#8172B3", "D0": "#4C72B0"}
ML = {"M00": "M00 전체 z · 절대 디코딩", "M01": "M01 전체 z · s₀ 보존", "M10": "M10 상태형 z · 절대 디코딩", "M11": "M11 상태형 z · s₀ 보존", "D0": "D0 직접 밀집 예측"}
VK = {"better": "더 좋음", "worse": "더 나쁨", "similar": "비슷함", "non-zero": "0과 다름", "zero within CI": "구간에 0 포함"}
VC = {"better": "b", "worse": "w", "similar": "s"}
QK = ["상태형 z가 일반화를 개선", "상태형 z가 최종 접촉을 개선", "s₀ 보존 디코딩이 접촉을 개선", "s₀ 보존이 초기 프레임을 해결", "두 변경이 상보적", "최선의 z 모델이 D0를 이김", "가장 지지되는 구조"]
QS = ["Does stateful z improve generalization?", "Does stateful z improve final contact?", "Does s_0-preserving decoding improve contact?", "Does s_0 preservation specifically fix early frames?",
      "Are the two changes complementary?", "Does the best z model beat direct D0?", "Which architecture is best supported?"]
KO = {"yes": "예", "no": "아니오", "partial": "부분적", "worse": "더 나쁨", "mixed": "엇갈림", "no (similar)": "아니오 (비슷함)", "no (worse)": "아니오 (더 나쁨)",
      "partial (better than absolute decoding, still worse than D0)": "부분적 (절대 디코딩보다 낫지만 D0보다 나쁨)", "direct dense D0": "직접 밀집 예측 D0", "whole-sequence z": "전체 시퀀스 z",
      "s_0-preserving z (whole-sequence)": "s₀ 보존 z (전체 시퀀스)", "stateful z": "상태형 z", "stateful + s_0-preserving z": "상태형 + s₀ 보존 z"}
CSS_EXTRA = """
.grid22{display:grid;grid-template-columns:auto 1fr 1fr;gap:6px;font-size:.84rem;margin:8px 0}.grid22 .h{color:var(--mt);font-weight:600;display:flex;align-items:center;justify-content:center;text-align:center;padding:2px 4px}
.cell{border:1px solid var(--bd);border-radius:8px;padding:8px 10px;min-width:0}.cell .n{font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}.cell .k{font-size:.74rem;color:var(--mt)}
.cell.b{border-color:var(--ok);box-shadow:inset 0 0 0 1px var(--ok)}.cell.w{border-color:var(--bad)}.cell.best{background:color-mix(in srgb,var(--ac) 10%,var(--card))}
.eff{font-size:.84rem;margin:2px 0}.mixed{background:var(--warn)}
"""


def ko(a):
    if a in KO:
        return KO[a]
    for k, v in KO.items():
        if a.startswith(k):
            return v + a[len(k):]
    return a


def pill(a):
    a = a.lower()
    return "yes" if a.startswith("yes") else ("partial" if (a.startswith("partial") or a.startswith("mixed") or "for structure" in a) else "no")


def img(name):
    p = Z.OUT / "figures" / name
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{esc(name)}" loading="lazy">' if p.exists() else ""


def main():
    V = json.loads((Z.OUT / "report_values.json").read_text())
    raw = json.loads((Z.OUT / "narrative" / "dashboard_text.json").read_text()) if (Z.OUT / "narrative" / "dashboard_text.json").exists() else {}
    miss = set()
    def sub(m):
        if m.group(1) not in V:
            miss.add(m.group(1)); return f"??{m.group(1)}??"
        return format(V[m.group(1)], m.group(2)) if m.group(2) else str(V[m.group(1)])
    T = json.loads(re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", sub, json.dumps(raw, ensure_ascii=False)))
    A = {ds: pd.read_csv(Z.OUT / f"{ds}_metrics.csv") for ds in Z.DATASETS}
    P, FX, LAT, EF, D, DC = (pd.read_csv(Z.OUT / n) for n in ("paired_comparisons.csv", "factorial_effects.csv", "latent_metrics.csv", "early_frame_metrics.csv", "decision_summary.csv", "direct_comparison.csv"))
    curves = dict(np.load(Z.OUT / "temporal_curves.npz"))
    acc = lambda ds, m, met: A[ds][(A[ds].model == m) & (A[ds].metric == met)].iloc[0]
    pr = lambda ds, a, b, met: P[(P.dataset == ds) & (P.a == a) & (P.b == b) & (P.metric == met)].iloc[0]
    fx = lambda ds, met, eff: FX[(FX.dataset == ds) & (FX.metric == met) & (FX.effect == eff)].iloc[0]
    H = [f"<title>z Stateful s0 Factorial</title><style>{ZD.CSS}{CSS_EXTRA}</style><main>", "<h1>이전 z 경유 생성기는 무엇이 문제였나</h1>", f"<div class='sub'>{esc(T.get('subtitle', ''))}</div>",
         f"<div class='answer'><b>{esc(T.get('answer_head', ''))}</b>{esc(T.get('answer_body', ''))}</div>", "<div class='hero'>"]
    for ds in Z.DATASETS:
        d = D[D.dataset == ds]; case = d[d.question == "case"].answer.iloc[0]; best = V[f"best_{ds}"]
        def cell(m):
            r = acc(ds, m, "E_C"); p = pr(ds, m, "D0", "E_C")
            return (f"<div class='cell {VC[p.verdict]}{' best' if m == best else ''}'><div class='k'>{m}{' · 검증 기준 최선' if m == best else ''}</div><div class='n'>{r.value:.3f}</div>"
                    f"<div class='k'>D0 대비 {-p.rel_change * 100:+.1f} % · {VK[p.verdict]}</div></div>")
        grid = ("<div class='grid22'><div></div><div class='h'>절대 디코딩</div><div class='h'>s₀ 보존 디코딩</div>"
                f"<div class='h'>전체<br>시퀀스 z</div>{cell('M00')}{cell('M01')}<div class='h'>상태형 z</div>{cell('M10')}{cell('M11')}</div>")
        effs = "".join(f"<div class='eff'>{lab}: <b class='num'>{fx(ds, 'E_C', e).rel_to_M00 * 100:+.1f} %</b> [{fx(ds, 'E_C', e).ci_lo:+.3f}, {fx(ds, 'E_C', e).ci_hi:+.3f}] · {VK[fx(ds, 'E_C', e).verdict]}</div>"
                       for e, lab in (("stateful", "상태형 주효과"), ("decoder", "디코더 주효과"), ("interaction", "상호작용")))
        bz = pr(ds, best, "D0", "E_C")
        pills = "".join(f"<span class='pill {pill(r.answer.iloc[0])}'>{esc(k)}: {esc(ko(r.answer.iloc[0]))}</span>" for k, q in zip(QK, QS) for r in [d[d.question == q]] if len(r))
        H.append(f"<div class='card'><div class='verdict'>{L_[ds]} <span class='case'>Case {esc(case)}</span></div><p>{esc(T.get('verdict', {}).get(ds, ''))}</p>"
                 f"<h3>2×2 결과: 테스트 밀집 오차 E_C (D0 = {acc(ds, 'D0', 'E_C').value:.3f})</h3>{grid}{effs}"
                 f"<p class='note'>칸의 % 는 D0 대비 오차 변화입니다. 음수가 z 모델의 오차가 낮다는 뜻입니다. 주효과의 % 는 M00 평균 대비이고 음수가 개선입니다.</p>"
                 f"<h3>최선의 z 모델 대 D0</h3><p><b>{best}</b> {bz.value_a:.3f} 대 D0 {bz.value_b:.3f}, 차이 {bz['diff']:+.3f} [{bz.ci_lo:+.3f}, {bz.ci_hi:+.3f}] · {VK[bz.verdict]}</p>{pills}</div>")
    H.append("</div>")
    H.append("<div class='card'><h3>한 줄 정리</h3>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("one_line", [])) + "</div>")
    def sec(title, paras=None, extra=""):
        H.append(f"<h2>{esc(title)}</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in (paras or [])) + extra + "</div>")
    sec("1. 가설", T.get("hypotheses", []))
    sec("2. 2×2 구조", T.get("design", []), img("fig1_design.png"))
    for num, ds in ((3, "taco"), (4, "arctic")):
        ms = [m for m in ("M00", "M01", "M10", "M11", "D0") if len(A[ds][A[ds].model == m])]
        items = [(ML[m], float(acc(ds, m, "E_C").value), MC[m], float(acc(ds, m, "E_C").ci_lo), float(acc(ds, m, "E_C").ci_hi)) for m in ms]
        ys = [curves[f"{ds}|{m}|dense"][1:] for m in ms]; ymax = float(max(y.max() for y in ys)) * 1.08
        rows = []
        for met, lab in (("part_hamming", "참여 (해밍)"), ("amount_l1", "접촉량 L1"), ("centroid", "중심 오차"), ("normal", "법선 각도"), ("q_rel_l1", "렌치 상대 L1"), ("E_C_h1_16", "E_C 프레임 1~16"), ("E_C_h49_63", "E_C 프레임 49~63")):
            cells = "".join((lambda r: f"<td class='num {VC[r.verdict]}'>{r.rel_improvement * 100:+.1f} % {VK[r.verdict]}</td>")(pr(ds, m, "D0", met)) for m in ("M00", "M01", "M10", "M11"))
            rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
        extra = ("<div class='grid2'><div><h3>밀집 오차 E_C (테스트)</h3>" + bars(items, left=190) + "</div><div><h3>프레임별 밀집 오차</h3>"
                 + lines([(ML[m], y, MC[m], 2.0, "5 3" if m == "D0" else "") for m, y in zip(ms, ys)], [np.arange(1, Z.T)] * len(ms), ylim=(0, ymax), xlabel="프레임 t", ylabel="‖Ĉ_t − C_t‖") + "</div></div>"
                 + f"<div class='tw'><table><tr><th>D0 대비 (양수 = z 모델이 더 좋음)</th><th>M00</th><th>M01</th><th>M10</th><th>M11</th></tr>{''.join(rows)}</table></div>")
        sec(f"{num}. {L_[ds]} 결과", T.get(ds, []), extra)
    zz = "<div class='grid2'>"
    for ds in Z.DATASETS:
        L = LAT[LAT.dataset == ds]; items = []
        for r in L.itertuples():
            for col, lab, c in (("rmse_train", "학습", "#9ECAE1"), ("rmse_val", "검증", "#4292C6"), ("rmse_test", "테스트", "#1f5fa8")):
                items.append((f"{r.model} {lab}", float(getattr(r, col)), c))
        ms = [m for m in Z.Z_MODELS if f"{ds}|{m}|z_err" in curves]; ymax = float(max(curves[f"{ds}|{m}|z_err"].max() for m in ms)) * 1.1
        zz += (f"<div><h3>{L_[ds]} — z RMSE: 학습 / 검증 / 테스트</h3>{bars(items, fmt='{:.2f}', left=110)}<h3>{L_[ds]} — 프레임별 z 오차 (테스트)</h3>"
               + lines([(m, curves[f"{ds}|{m}|z_err"], MC[m], 2.0, "") for m in ms], [np.arange(1, Z.T)] * len(ms), ylim=(0, ymax), xlabel="프레임 t", ylabel="RMSE(ẑ − z*)") + "</div>")
    zz += "</div>"
    sec("5. z 일반화", T.get("generalisation", []), zz)
    ee = "<div class='grid2'>"
    for ds in Z.DATASETS:
        e = EF[(EF.dataset == ds) & (EF.quantity == "E_C")]; items = []
        for t in Z.EARLY_FRAMES:
            for m in ("M00", "M01", "M10", "M11", "D0"):
                r = e[(e.model == m) & (e.frame == t)]
                if len(r):
                    items.append((f"t={t} {m}", float(r.iloc[0].value), MC[m], float(r.iloc[0].ci_lo), float(r.iloc[0].ci_hi)))
        ee += f"<div><h3>{L_[ds]} — 초기 프레임의 밀집 오차</h3>{bars(items, fmt='{:.2f}', left=90)}</div>"
    ee += "</div>"
    sec("6. 초기 프레임과 s₀ 보존", T.get("early", []), ee + img("fig6_early_frames.png"))
    rows = "".join(f"<tr><td>{esc(k)}</td>" + "".join((lambda r: f"<td><span class='pill {pill(r.answer.iloc[0])}'>{esc(ko(r.answer.iloc[0]))}</span></td>" if len(r) else "<td>—</td>")(D[(D.dataset == ds) & (D.question == q)]) for ds in Z.DATASETS) + "</tr>"
                   for k, q in zip(QK, QS))
    sec("7. 최종 판단", T.get("decision", []), f"<div class='tw'><table><tr><th>질문</th><th>TACO</th><th>ARCTIC</th></tr>{rows}</table></div>"
        + "<h3>한계</h3><ul>" + "".join(f"<li>{esc(p)}</li>" for p in T.get("limitations", [])) + "</ul><h3>다음 단계</h3>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("next", []))
        + "<h3>파일</h3><ul>" + "".join(f"<li>{esc(p)}</li>" for p in T.get("files", [])) + "</ul>")
    H.append("</main>")
    html = "\n".join(H); (Z.OUT / "dashboard.html").write_text(html)
    print(f"dashboard.html: {len(html) / 1024:.0f} KB; unresolved: {html.count('??')}; missing: {sorted(miss)}")


if __name__ == "__main__":
    main()
