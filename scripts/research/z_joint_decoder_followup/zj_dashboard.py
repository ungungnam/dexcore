#!/usr/bin/env python
"""Small research dashboard (one HTML page, Korean): did adapting the decoder rescue z-mediated temporal generation?    python zj_dashboard.py
Numbers are injected from the result tables; the prose lives in narrative/dashboard_text.json and uses the same <<key:format>>
placeholders as the report (resolved against OUT/report_values.json).  -> OUT/dashboard.html (published as a Claude artifact)."""
from __future__ import annotations

import base64
import json
import re

import numpy as np
import pandas as pd

import zj_common as Z

L_ = {"taco": "TACO", "arctic": "ARCTIC"}
MC = {"M0": "#4C72B0", "M1": "#DD8452", "M2": "#2e9e62", "M3": "#8172B3", "M0r": "#8FAADC"}
ML = {"M0": "M0 직접 밀집 예측", "M1": "M1 이전 z 경유", "M2": "M2 공동 학습·큰 디코더", "M3": "M3 이중 입력 디코더"}
IC = {"oracle": "#7FB77E", "pred": "#3b4252", "noisy_gauss": "#E0A458", "noisy_perm": "#C97C5D"}
IL = {"oracle": "교사 z* (정답 잠재)", "pred": "예측 ẑ (실제 모델)", "noisy_gauss": "z* + 같은 크기 가우스 잡음", "noisy_perm": "z* + 다른 시퀀스의 예측 오차"}
QK = ["디코더 적응이 M1을 개선", "M2가 M0을 밀집 오차로 이김", "M2가 M0을 구조·렌치로 이김", "먼 지평선 개선", "z 예측 자체 개선", "디코더가 예측 z 오차에 더 강건", "최종 생성기에 z 경유가 정당"]
QS = ["Does joint decoder adaptation improve over previous M1?", "Does M2 beat direct M0 in dense contact?", "Does M2 beat M0 in structural / wrench quality?", "Does M2 improve long-horizon behavior?",
      "Is z prediction itself improved?", "Is the decoder more robust to predicted-z error?", "Is z mediation justified for the final generator?"]
KO = {"yes": "예", "no": "아니오", "no (similar)": "아니오 (비슷함)", "no (worse)": "아니오 (더 나쁨)", "partial": "부분적", "worse": "더 나쁨", "structure only": "구조만",
      "for structure only (dense not better than M0)": "구조에 한해서만", "no (no final-task advantage)": "아니오 (최종 과제 이득 없음)", "no (worse than direct)": "아니오 (직접 예측보다 나쁨)"}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def ko(ans):
    if ans in KO:
        return KO[ans]
    return "; ".join(f"{p.split(':')[0].replace('vs', '대').strip()}: {KO.get(p.split(':', 1)[1].strip(), p.split(':', 1)[1].strip())}" if ":" in p else KO.get(p.strip(), p.strip()) for p in ans.split(";"))


def pill(ans):
    a = ans.lower()
    if a.startswith("yes") or a == "better":
        return "yes"
    if "partial" in a or "structure" in a:
        return "partial"
    if "vs" in a:
        return "yes" if a.count("yes") == 2 else ("partial" if "yes" in a else "no")
    return "no"


def bars(items, vmax=None, fmt="{:.3f}", width=430, ref=None, left=168):
    """Horizontal bars on a signed axis; intervals are never clipped."""
    vals = [v for _, v, *_ in items]; his = [c[1] for _, _, _, *c in items if c and not np.isnan(c[0])]; los = [c[0] for _, _, _, *c in items if c and not np.isnan(c[0])]
    vmax = vmax or max(max(vals + his + [r[1] for r in (ref or [])]), 1e-9) * 1.18; vmin = min(0.0, min(vals + los)) * 1.18
    h = 22 * len(items) + 10; w = width - left - 62; sx = lambda v: left + (v - vmin) / (vmax - vmin) * w; x0 = sx(0.0)
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


def lines(series, xs_list, width=440, height=210, ylim=(0, 1), xlim=None, xlabel="", ylabel=""):
    left, bottom, top, right = 46, 30, 10, 12; w, h = width - left - right, height - top - bottom; xlim = xlim or (min(min(x) for x in xs_list), max(max(x) for x in xs_list))
    sx = lambda x: left + (x - xlim[0]) / max(xlim[1] - xlim[0], 1e-9) * w; sy = lambda y: top + (1 - (min(max(y, ylim[0]), ylim[1]) - ylim[0]) / max(ylim[1] - ylim[0], 1e-9)) * h
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block" role="img">']
    for yt in np.linspace(ylim[0], ylim[1], 5):
        out.append(f'<line x1="{left}" x2="{left + w}" y1="{sy(yt):.1f}" y2="{sy(yt):.1f}" stroke="var(--grid)"/><text x="{left - 4}" y="{sy(yt) + 4:.1f}" text-anchor="end" font-size="10" fill="var(--mt)">{yt:.2f}</text>')
    for xt in np.linspace(xlim[0], xlim[1], 5):
        out.append(f'<text x="{sx(xt):.1f}" y="{height - 12}" text-anchor="middle" font-size="10" fill="var(--mt)">{xt:.0f}</text>')
    for (lab, ys, col, wd, dash), xs in zip(series, xs_list):
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys) if np.isfinite(y) and xlim[0] <= x <= xlim[1])
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{wd}"' + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
    out.append(f'<text x="{left + w / 2:.0f}" y="{height - 1}" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(xlabel)}</text>')
    out.append(f'<text transform="translate(10,{top + h / 2:.0f}) rotate(-90)" text-anchor="middle" font-size="10" fill="var(--mt)">{esc(ylabel)}</text></svg>')
    leg = [f'<span class="lg"><i style="background:{col}"></i>{esc(lab)}</span>' for lab, _, col, _, _ in series]
    return "".join(out) + f'<div class="legend">{" ".join(leg)}</div>'


def img(name):
    p = Z.OUT / "figures" / name
    return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" alt="{esc(name)}" loading="lazy">' if p.exists() else ""


CSS = """
/* layout: answer first, two dataset columns that stack on a phone, evidence sections below */
:root{--bg:#f3f5f4;--card:#ffffff;--tx:#18222b;--mt:#5b6873;--ac:#1d6a8f;--ok:#2e9e62;--warn:#c98a12;--bad:#c9483f;--grid:#e3e7e6;--bd:#d6dcdb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#12171b;--card:#1b2227;--tx:#e6ebee;--mt:#97a4ae;--ac:#6fb7da;--ok:#49b97c;--warn:#e0a83a;--bad:#e2685f;--grid:#2a333a;--bd:#323c44;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#12171b;--card:#1b2227;--tx:#e6ebee;--mt:#97a4ae;--ac:#6fb7da;--ok:#49b97c;--warn:#e0a83a;--bad:#e2685f;--grid:#2a333a;--bd:#323c44;color-scheme:dark}
body{background:var(--bg);color:var(--tx);font-family:"Noto Sans KR","Pretendard","Apple SD Gothic Neo","Segoe UI",system-ui,sans-serif;margin:0;padding-block:24px;padding-inline:16px;line-height:1.55}
main{max-width:1080px;margin:0 auto}h1{font-size:1.5rem;margin:0 0 4px;text-wrap:balance}h2{font-size:1.12rem;margin:34px 0 10px;border-bottom:1px solid var(--bd);padding-bottom:4px;text-wrap:balance}h3{font-size:.98rem;margin:10px 0 6px}
.sub{color:var(--mt);font-size:.9rem;margin-bottom:14px}.answer{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:16px 18px;font-size:1.02rem;max-width:78ch}
.answer b{display:block;font-size:1.22rem;margin-bottom:6px;text-wrap:balance}.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin:14px 0}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;min-width:0}.verdict{font-size:1.2rem;font-weight:700}.case{display:inline-block;padding:2px 10px;border-radius:999px;color:#fff;font-weight:600;font-size:.88rem;background:var(--ac)}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:.78rem;color:#fff;margin:2px 4px 2px 0}.yes{background:var(--ok)}.partial{background:var(--warn)}.no{background:var(--bad)}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px}.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(118px,1fr));gap:8px;margin:8px 0}
.kv div{border:1px solid var(--bd);border-radius:8px;padding:6px 8px}.kv .n{font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums}.kv .l{font-size:.75rem;color:var(--mt)}
table{border-collapse:collapse;font-size:.82rem;width:100%}th,td{border-bottom:1px solid var(--bd);padding:4px 6px;text-align:left;vertical-align:top}th{color:var(--mt);font-weight:600}
.tw{overflow-x:auto}.num{font-variant-numeric:tabular-nums}.legend{font-size:.76rem;color:var(--mt);margin-top:4px}.lg i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px 0 8px;vertical-align:middle}
img{max-width:100%;border-radius:6px;border:1px solid var(--bd)}p{margin:6px 0}.note{font-size:.84rem;color:var(--mt)}ul{margin:4px 0 4px 18px;padding:0}li{margin:3px 0}
.b{color:var(--ok);font-weight:600}.w{color:var(--bad);font-weight:600}.s{color:var(--mt)}
"""


def main():
    V = json.loads((Z.OUT / "report_values.json").read_text())
    raw = json.loads((Z.OUT / "narrative" / "dashboard_text.json").read_text()) if (Z.OUT / "narrative" / "dashboard_text.json").exists() else {}
    miss = set()
    def sub(m):
        if m.group(1) not in V:
            miss.add(m.group(1)); return f"??{m.group(1)}??"
        return format(V[m.group(1)], m.group(2)) if m.group(2) else str(V[m.group(1)])
    T = json.loads(re.sub(r"<<([A-Za-z0-9_]+)(?::([^>]+))?>>", sub, json.dumps(raw, ensure_ascii=False)))
    A, P, D, DD, ZM, H = (pd.read_csv(Z.OUT / n) for n in ("main_metrics.csv", "paired_comparisons.csv", "decision_summary.csv", "decoder_diagnostic.csv", "z_metrics.csv", "horizon_metrics.csv"))
    PR = pd.read_csv(Z.OUT / "heldout_probe.csv") if (Z.OUT / "heldout_probe.csv").exists() else pd.DataFrame()
    curves = dict(np.load(Z.OUT / "temporal_curves.npz"))
    acc = lambda ds, m, met: A[(A.dataset == ds) & (A.model == m) & (A.metric == met)].iloc[0]
    pr = lambda ds, a, b, met: P[(P.dataset == ds) & (P.a == a) & (P.b == b) & (P.metric == met)].iloc[0]
    models = lambda ds: [m for m in ("M0", "M1", "M2", "M3") if len(A[(A.dataset == ds) & (A.model == m)])]
    vcls = {"better": "b", "worse": "w", "similar": "s"}; vko = {"better": "더 좋음", "worse": "더 나쁨", "similar": "비슷함"}
    Hh = [f"<title>z Joint Decoder Follow-up</title><style>{CSS}</style><main>", "<h1>디코더를 맞추면 z 경유 생성이 살아나는가</h1>", f"<div class='sub'>{esc(T.get('subtitle', ''))}</div>",
          f"<div class='answer'><b>{esc(T.get('answer_head', ''))}</b>{esc(T.get('answer_body', ''))}</div>"]
    # ---- first screen: verdict cards
    Hh.append("<div class='hero'>")
    for ds in Z.DATASETS:
        d = D[D.dataset == ds]; case = d[d.question == "case"].answer.iloc[0]
        e21, e20 = pr(ds, "M2", "M1", "E_C"), pr(ds, "M2", "M0", "E_C")
        kv = [(f"{acc(ds, m, 'E_C').value:.3f}", f"{m} 밀집 오차 E_C") for m in models(ds)] + [(f"{-e21.rel_change * 100:+.1f} %", f"M2의 M1 대비 E_C 변화 ({vko[e21.verdict]})"), (f"{-e20.rel_change * 100:+.1f} %", f"M2의 M0 대비 E_C 변화 ({vko[e20.verdict]})")]
        pills = "".join(f"<span class='pill {pill(r.answer.iloc[0])}'>{esc(k)}: {esc(ko(r.answer.iloc[0]))}</span>" for k, q in zip(QK, QS) for r in [d[d.question == q]] if len(r))
        Hh.append(f"<div class='card'><div class='verdict'>{L_[ds]} <span class='case'>Case {esc(case)}</span></div><p>{esc(T.get('verdict', {}).get(ds, ''))}</p><div class='kv'>"
                  + "".join(f"<div><div class='n'>{a}</div><div class='l'>{esc(b)}</div></div>" for a, b in kv) + f"</div>{pills}<p class='note'>E_C 변화의 부호: 음수가 M2의 오차가 낮다는 뜻입니다.</p></div>")
    Hh.append("</div>")
    # ---- first screen: main metrics
    Hh.append("<div class='grid2'>")
    for ds in Z.DATASETS:
        items = [(ML[m], float(acc(ds, m, "E_C").value), MC[m], float(acc(ds, m, "E_C").ci_lo), float(acc(ds, m, "E_C").ci_hi)) for m in models(ds)]
        rows = []
        for met, lab in (("part_hamming", "참여 (해밍)"), ("amount_l1", "접촉량 L1"), ("centroid", "중심 오차"), ("normal", "법선 각도"), ("q_rel_l1", "렌치 상대 L1"), ("E_C_last16", "마지막 16프레임 E_C")):
            cells = "".join((lambda r: f"<td class='num {vcls[r.verdict]}'>{r.rel_improvement * 100:+.1f} % {vko[r.verdict]}</td>")(pr(ds, a, b, met)) for a, b in (("M2", "M1"), ("M2", "M0"), ("M1", "M0")))
            rows.append(f"<tr><td>{lab}</td>{cells}</tr>")
        Hh.append(f"<div class='card'><h3>{L_[ds]} — 밀집 접촉 오차 E_C (테스트, 낮을수록 좋음)</h3>{bars(items, left=176)}"
                  f"<div class='tw'><table><tr><th>구조·렌치 지표</th><th>M2 대 M1</th><th>M2 대 M0</th><th>M1 대 M0 (Stage 2)</th></tr>{''.join(rows)}</table></div>"
                  "<p class='note'>양수는 앞 모델이 더 좋다는 뜻입니다. 2 % 이상이고 신뢰구간이 0을 벗어날 때만 좋음·나쁨으로 셉니다.</p></div>")
    # ---- first screen: decoder mismatch
    for ds in Z.DATASETS:
        d = DD[(DD.dataset == ds) & (DD.split == "test")]; items = []
        for m in [x for x in ("M1", "M2", "M3") if (d.decoder == x).any()]:
            for inp in ("oracle", "pred", "noisy_gauss", "noisy_perm"):
                r = d[(d.decoder == m) & (d.input == inp)]
                if len(r):
                    r = r.iloc[0]; items.append((f"{m} 디코더 · {IL[inp]}", float(r.E_C), IC[inp], float(r.ci_lo), float(r.ci_hi)))
        Hh.append(f"<div class='card'><h3>{L_[ds]} — 디코더 불일치 진단 (같은 디코더, 다른 잠재 입력)</h3>{bars(items, fmt='{:.2f}', left=250, width=520)}<p class='note'>{esc(T.get('decoder_note', {}).get(ds, ''))}</p></div>")
    Hh.append("</div>")
    Hh.append(f"<div class='card' style='margin-top:14px'><h3>최종 해석</h3>" + "".join(f"<p>{esc(p)}</p>" for p in T.get("interpretation", [])) + "</div>")
    # ---- sections
    def sec(title, paras=None, extra=""):
        Hh.append(f"<h2>{esc(title)}</h2><div class='card'>" + "".join(f"<p>{esc(p)}</p>" for p in (paras or [])) + extra + "</div>")
    sec("1. 무엇이 문제였고 무엇을 했나", T.get("what", []), img("fig1_architecture.png"))
    sec("2. M1은 이미 공동 학습이었다: M2가 실제로 바꾼 세 가지", T.get("three_changes", []))
    sec("3. 학습 경과", T.get("training", []), img("fig7_training_curves.png"))
    # horizon charts
    hz = "<div class='grid2'>"
    for ds in Z.DATASETS:
        ms = models(ds); ys = [curves[f"{ds}|{m}|dense"][1:] for m in ms]; ymax = float(max(y.max() for y in ys)) * 1.08
        hz += f"<div><h3>{L_[ds]} — 프레임별 밀집 오차</h3>" + lines([(ML[m], y, MC[m], 2.0, "") for m, y in zip(ms, ys)], [np.arange(1, Z.T)] * len(ms), ylim=(0, ymax), xlabel="프레임 t", ylabel="‖Ĉ_t − C_t‖") + "</div>"
    hz += "</div>"
    sec("4. 지평선별 오차", T.get("horizon", []), hz)
    zz = "<div class='grid2'>"
    for ds in Z.DATASETS:
        ms = [m for m in ("M1", "M2", "M3") if f"{ds}|{m}|z_err" in curves]
        ser = [(f"{m} 테스트", curves[f"{ds}|{m}|z_err"], MC[m], 2.0, "") for m in ms] + [(f"{m} 학습 시퀀스", curves[f"{ds}|{m}|z_err_train"], MC[m], 1.2, "4 3") for m in ms]
        ymax = float(max(s[1].max() for s in ser)) * 1.1
        zz += f"<div><h3>{L_[ds]} — z 예측 오차 (표준화 RMSE)</h3>" + lines(ser, [np.arange(1, Z.T)] * len(ser), ylim=(0, ymax), xlabel="프레임 t", ylabel="RMSE(ẑ − z*)") + "</div>"
    zz += "</div>"
    sec("5. z 예측은 나아졌나", T.get("z_prediction", []), zz)
    sec("6. 학습 때 디코더가 본 입력과 테스트 때 받는 입력", T.get("train_vs_test", []), img("fig8_train_vs_heldout.png"))
    if len(PR):
        rows = "".join(f"<tr><td>{L_[r.dataset]}</td><td class='num'>{r.test_E_C_unadapted:.3f} → {r.test_E_C_adapted:.3f} ({r.test_gain_rel * 100:+.1f} %)</td><td class='num'>{r.s_star}</td>"
                       f"<td class='num'>{r.vs_M0_diff:+.3f} [{r.vs_M0_lo:+.3f}, {r.vs_M0_hi:+.3f}]</td><td class='num'>{r.test_E_C_teacher_z_unadapted:.3f} → {r.test_E_C_teacher_z_adapted:.3f}</td></tr>" for r in PR.itertuples())
        sec("7. 보류 데이터의 예측 z에 디코더를 직접 맞추면", T.get("probe", []),
            f"<div class='tw'><table><tr><th>데이터셋</th><th>테스트 E_C (적응 전 → 후)</th><th>미세조정 스텝</th><th>적응 후 − M0 [95 % 구간]</th><th>교사 z 입력 E_C (전 → 후)</th></tr>{rows}</table></div>")
    sec("8. 정성 예시", T.get("qualitative", []), "".join(img(f"fig6_{ds}_{k}.png") for ds, k in (("taco", "helps"), ("arctic", "fails"), ("arctic", "typical"))))
    sec("9. 한계", None, "<ul>" + "".join(f"<li>{esc(p)}</li>" for p in T.get("limitations", [])) + "</ul>")
    sec("10. 다음 단계", T.get("next", []))
    sec("11. 파일", None, "<ul>" + "".join(f"<li>{esc(p)}</li>" for p in T.get("files", [])) + "</ul>")
    Hh.append("</main>")
    html = "\n".join(Hh); (Z.OUT / "dashboard.html").write_text(html)
    print(f"dashboard.html: {len(html) / 1024:.0f} KB; unresolved: {html.count('??')}; missing: {sorted(miss)}")


if __name__ == "__main__":
    main()
