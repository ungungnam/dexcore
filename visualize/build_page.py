"""Assemble visualize/index.html -- the target x method grid.

One card per TARGET, the two methods side by side inside it, so the comparison that matters
(same object, same source demo, different reference synthesis) is one glance rather than a scroll.
Each method column carries two panels:

  레퍼런스   the generated demo on the target object -- what training imitates
  롤아웃     the trained policy driving the real hand under physics

Each card names the human demo it was synthesised FROM. There are three -- box, laptop and
microwave -- and a target is always transferred from its own category's demo, so the sources are
shown together at the top rather than repeated on every card.

Numbers come from each run's add_stats.json and its newest checkpoint; run status comes from the
training log's terminal marker. Anything missing degrades to a labelled placeholder rather than a
broken panel, so the page is meaningful while runs are still training.

  python build_page.py [--videos videos] [--out index.html]
"""
import argparse
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_ROOT = "/result/uhnam/dexmachina/01_train_eval/rl_games/inspire_hand"
TRAIN_LOGS = "/home/uhnam/workspace/dexcore/outputs/pm_sweep_0240/train"

# target, collision pieces, (tag, strategy, machine) for ours then bimart. Ordered by piece count
# descending: that axis is what drove the OOM story, so keeping it monotonic makes the grid readable.
GRID = [
    ("mw7304_calibrated",   46, ("mw304dx", "dexcore",     "mango"), ("mw304b",  "bimart", "mango")),
    ("mw7236_calibrated",   34, ("mw236dx", "dexcore",     "mango"), ("mw236b",  "bimart", "mango")),
    ("mw7310_calibrated",   19, ("mw310dx", "dexcore",     "mango"), ("mw310b",  "bimart", "mango")),
    ("mw7292_calibrated",   12, ("mw292dx", "dexcore",     "sushi"), ("mw292b",  "bimart", "sushi")),
    ("pm102379_calibrated", 42, ("pm379dx", "dexcore",     "mango"), ("pm379b",  "bimart", "mango")),
    ("pm100189_calibrated", 38, ("pm189dx", "dexcore",     "mango"), ("pm189b",  "bimart", "mango")),
    ("pm100224_calibrated", 19, ("pm224dx", "dexcore",     "mango"), ("pm224b",  "bimart", "mango")),
    ("pm100243_calibrated", 17, ("pm243dx", "dexcore",     "mango"), ("pm243b",  "bimart", "mango")),
    ("pm100658_calibrated", 11, ("pm658dx", "dexcore",     "sushi"), ("pm658b",  "bimart", "sushi")),
    ("pm100141_calibrated",  9, ("pmk",     "ours_cordex", "sushi"), ("pmb",     "bimart", "sushi")),
    ("lap11876_calibrated", 29, ("lap876dx", "dexcore",    "mango"), ("lap876b", "bimart", "mango")),
    ("lap11030_calibrated", 15, ("lap030dx", "dexcore",    "mango"), ("lap030b", "bimart", "mango")),
    ("lap10239_calibrated", 12, ("lap239dx", "dexcore",    "sushi"), ("lap239b", "bimart", "sushi")),
    ("lap10243_calibrated",  8, ("lap243dx", "dexcore",    "sushi"), ("lap243b", "bimart", "sushi")),
    ("lap10211_calibrated",  4, ("lap211dx", "dexcore",    "sushi"), ("lap211b", "bimart", "sushi")),
    ("lap10305_calibrated",  2, ("lap305dx", "dexcore",    "sushi"), ("lap305b", "bimart", "sushi")),
]

# Which human demo each row was synthesised FROM. The page used to assume one source for the whole
# grid, which was true while every target came from the ARCTIC box; laptop and microwave targets are
# transferred from their own category's demo, and showing a box above them would misstate the input.
SOURCE = {"mw": ("source_microwave_human.mp4", "ARCTIC microwave (사람)"),
          "lap": ("source_laptop_human.mp4", "ARCTIC laptop (사람)"),
          "pm": ("source_box_human.mp4", "ARCTIC box (사람)")}

PRETTY = {"dexcore": "우리 + cordex", "ours_cordex": "우리 + cordex", "bimart": "BimArt"}
SIDE = {"dexcore": "ours", "ours_cordex": "ours", "bimart": "bimart"}


def find_run_dir(target, strategy):
    hits = sorted(glob.glob(os.path.join(LOG_ROOT, f"*{target}_{strategy}_*")))
    return hits[0] if hits else None


def run_status(target, strategy, epoch):
    """('완주'|'조기종료'|'진행중', css class).

    The training log is authoritative when present: rl_games prints "MAX EPOCHS NUM!" on a clean
    finish and "Reward too low" when it gives up; a log with NEITHER marker belongs to a run that is
    still going. Runs trained on the other machine have no local log, so they fall back to the
    checkpoint's epoch -- and a bare inspire_hand.pth (no ep_N) is a rolling mid-training save.
    """
    log = os.path.join(TRAIN_LOGS, f"{target}_{strategy}.log")
    if os.path.exists(log):
        try:
            with open(log, errors="ignore") as f:
                tail = f.read()[-400000:]
            if "MAX EPOCHS NUM!" in tail:
                return "완주", "ok"
            if "Reward too low" in tail:
                return "조기종료", "early"
            return "진행중", "live"
        except OSError:
            pass
    if epoch is None:
        return "진행중", "live"
    return ("완주", "ok") if epoch >= 5000 else ("조기종료", "early")


def run_facts(target, strategy):
    """{epoch, reward, auc, add_cm, auc_epoch} -- every field independently optional."""
    d = find_run_dir(target, strategy)
    if d is None:
        return {}
    out = {}
    ckpts = sorted(glob.glob(os.path.join(d, "nn", "*.pth")), key=os.path.getmtime)
    if ckpts:
        name = os.path.basename(ckpts[-1])
        m = re.search(r"ep_(\d+)", name)
        if m:
            out["epoch"] = int(m.group(1))
        m = re.search(r"rew__?(-?[\d.]+?)_?\.pth$", name)
        if m:
            try:
                out["reward"] = float(m.group(1))
            except ValueError:
                pass
    stats = sorted(glob.glob(os.path.join(d, "**", "add_stats.json"), recursive=True),
                   key=os.path.getmtime)
    if stats:
        try:
            s = json.load(open(stats[-1]))
            out["auc"] = s["auc"]["overall"]
            out["add_cm"] = s["mean_add"]["overall"] * 100.0
            m = re.search(r"ep_(\d+)", stats[-1])
            if m:
                out["auc_epoch"] = int(m.group(1))
        except Exception:
            pass
    return out


def panel(rel, videos_dir, label):
    if not os.path.exists(os.path.join(videos_dir, rel)):
        return (f'<figure class="panel"><figcaption>{label}</figcaption>'
                f'<div class="ph">아직 없음</div></figure>')
    return (f'<figure class="panel"><figcaption>{label}</figcaption>'
            f'<video src="videos/{rel}" muted loop playsinline preload="metadata"></video></figure>')


def method_col(short, target, tag, strategy, machine, videos_dir):
    f = run_facts(target, strategy)
    status, scls = run_status(target, strategy, f.get("epoch"))

    badges = [f'<span class="b st {scls}">{status}</span>']
    if "epoch" in f:
        badges.append(f'<span class="b">ep {f["epoch"]}</span>')
    elif status == "진행중":
        badges.append('<span class="b">중간 스냅샷</span>')
    if "reward" in f:
        badges.append(f'<span class="b">rew {f["reward"]:.1f}</span>')
    if "auc" in f:
        stale = f.get("auc_epoch") not in (None, f.get("epoch"))
        extra = f' <em>ep{f["auc_epoch"]}</em>' if stale else ""
        badges.append(f'<span class="b auc{" stale" if stale else ""}">'
                      f'AUC {f["auc"]:.3f}{extra}</span>')
        badges.append(f'<span class="b">ADD {f["add_cm"]:.1f}cm</span>')
    else:
        badges.append('<span class="b none">AUC 미측정</span>')

    return f"""
          <div class="method {SIDE[strategy]}">
            <div class="mhead">
              <span class="mname">{PRETTY[strategy]}</span>
              <span class="badges">{''.join(badges)}</span>
            </div>
            <div class="pair">
              {panel(f'{short}_{strategy}_target.mp4', videos_dir, '레퍼런스')}
              {panel(f'{short}_{strategy}_rollout.mp4', videos_dir, '롤아웃')}
            </div>
          </div>""", f.get("auc")


CATEGORY = {"mw": ("MICROWAVE", "ARCTIC microwave 시연 → PartNet-Mobility Microwave"),
            "pm": ("BOX", "ARCTIC box 시연 → PartNet-Mobility Box"),
            "lap": ("LAPTOP", "ARCTIC laptop 시연 → PartNet-Mobility Laptop")}


def cat_of(short):
    return "mw" if short.startswith("mw") else ("lap" if short.startswith("lap") else "pm")


def build(videos_dir):
    """Cards grouped under their category, in the GRID's order within each group.

    A target is only comparable to others transferred from the SAME human demo, so the categories
    are separated rather than interleaved: a box row and a microwave row sitting next to each other
    invites reading their AUCs as one series when they do not share an input.
    """
    groups, order = {}, []
    for target, pieces, ours, bim in GRID:
        short = target.replace("_calibrated", "")
        key = cat_of(short)
        if key not in groups:
            groups[key] = []
            order.append(key)
        col_o, auc_o = method_col(short, target, *ours, videos_dir)
        col_b, auc_b = method_col(short, target, *bim, videos_dir)

        verdict = ""
        if auc_o is not None and auc_b is not None:
            d = auc_o - auc_b
            who = "우리" if d > 0 else "BimArt"
            cls = "win-ours" if d > 0 else "win-bimart"
            verdict = f'<span class="verdict {cls}">{who} +{abs(d):.3f} AUC</span>'
        groups[key].append((auc_o, auc_b, f"""
      <section class="card" data-cat="{key}" data-target="{short}">
        <header>
          <h2>{short}</h2>
          <span class="pieces">{pieces}조각</span>
          {verdict}
        </header>
        <div class="methods">{col_o}{col_b}</div>
      </section>"""))

    out = []
    for key in order:
        rows = groups[key]
        name, sub = CATEGORY[key]
        svid, slabel = SOURCE[key]
        paired = [(a, b) for a, b, _ in rows if a is not None and b is not None]
        wins = sum(1 for a, b in paired if a > b)
        tally = (f"{len(rows)}타깃 · 비교가능 {len(paired)} · 우리 {wins}승 {len(paired)-wins}패"
                 if paired else f"{len(rows)}타깃 · 비교 가능한 행 없음")
        out.append(f"""
      <section class="catblock" data-cat="{key}">
        <div class="cathead">
          <video src="videos/{svid}" muted loop playsinline preload="metadata"></video>
          <div>
            <h2 class="catname">{name}</h2>
            <p class="catsub">{sub}</p>
            <p class="cattally">{tally}</p>
          </div>
        </div>
{"".join(c for _, _, c in rows)}
      </section>""")
    return TEMPLATE.replace("{{CARDS}}", "\n".join(out))


TEMPLATE = """<!doctype html>
<html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>레퍼런스 합성 그리드</title>
<style>
:root{
  --bg:#f7f7f5; --fg:#1b1b1a; --muted:#6b6b66; --line:#e0e0da; --card:#fff; --sunk:#f2f2ef;
  --ours:#1f6f4a; --bimart:#a8622c; --warn:#a33;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#141413; --fg:#ecebe6; --muted:#9a9a93; --line:#2c2c29; --card:#1c1c1a; --sunk:#171716;
  --ours:#5fbf8f; --bimart:#d9925a; --warn:#e08585;
}}
:root[data-theme="dark"]{
  --bg:#141413; --fg:#ecebe6; --muted:#9a9a93; --line:#2c2c29; --card:#1c1c1a; --sunk:#171716;
  --ours:#5fbf8f; --bimart:#d9925a; --warn:#e08585;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
.wrap{max-width:1400px;margin:0 auto;padding:30px 20px 80px}
h1{font-size:23px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 20px;font-size:14px;max-width:70ch}

/* the shared input, shown once */
.source{display:flex;gap:16px;align-items:center;background:var(--card);
  border:1px solid var(--line);border-radius:12px;padding:14px;margin-bottom:6px}
.source video{width:150px;aspect-ratio:1/1;border-radius:8px;background:#000;
  border:1px solid var(--line);object-fit:cover;flex:none;cursor:pointer}
.source .txt{min-width:0}
.source h3{margin:0 0 4px;font-size:15px}
.source p{margin:0;color:var(--muted);font-size:13px}

.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;
  position:sticky;top:0;z-index:5;padding:12px 0;background:var(--bg);border-bottom:1px solid var(--line)}
button{font:inherit;font-size:13px;padding:5px 12px;border:1px solid var(--line);
  border-radius:999px;background:var(--card);color:var(--fg);cursor:pointer}
button.on{border-color:var(--fg)}
.spacer{flex:1}

.catblock{margin-top:34px}
.cathead{display:flex;gap:14px;align-items:center;padding:12px 4px 4px;
  border-bottom:2px solid var(--line);margin-bottom:4px}
.cathead video{width:120px;aspect-ratio:1/1;object-fit:cover;border-radius:8px;
  background:#000;border:1px solid var(--line);flex:none;cursor:pointer}
.catname{font-size:19px;margin:0;letter-spacing:.02em}
.catsub{margin:2px 0 0;color:var(--muted);font-size:13px}
.cattally{margin:4px 0 0;font-size:12px;color:var(--fg);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:14px 16px 16px;margin-top:16px}
.card header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:10px}
h2{font-size:17px;margin:0;font-weight:600;letter-spacing:-.01em}
.pieces{font-size:12px;color:var(--muted);border:1px solid var(--line);
  border-radius:6px;padding:1px 7px}
.src{font-size:12px;color:var(--muted)}
.verdict{font-size:12px;font-weight:600;margin-left:auto;
  border:1px solid currentColor;border-radius:999px;padding:2px 10px}
.verdict.win-ours{color:var(--ours)}
.verdict.win-bimart{color:var(--bimart)}

.methods{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.methods{grid-template-columns:1fr}}
.method{background:var(--sunk);border:1px solid var(--line);border-radius:10px;
  padding:10px;border-top:3px solid var(--line);min-width:0}
.method.ours{border-top-color:var(--ours)}
.method.bimart{border-top-color:var(--bimart)}
.mhead{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
.mname{font-size:13px;font-weight:600}
.method.ours .mname{color:var(--ours)}
.method.bimart .mname{color:var(--bimart)}
.badges{display:flex;gap:5px;flex-wrap:wrap}
.b{font-size:11px;color:var(--muted);border:1px solid var(--line);
  border-radius:5px;padding:1px 6px;white-space:nowrap;background:var(--card)}
.b.auc{color:var(--fg);font-weight:600}
.b.auc.stale{color:var(--warn);border-color:var(--warn)}
.b.auc em{font-style:normal;font-weight:400;opacity:.75}
.b.none{opacity:.55}
.b.st{font-weight:600;color:var(--fg)}
.b.st.ok{border-color:var(--ours);color:var(--ours)}
.b.st.early{border-color:var(--bimart);color:var(--bimart)}
.b.st.live{border-color:var(--muted);color:var(--muted)}

.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.panel{margin:0;min-width:0;display:flex;flex-direction:column;gap:5px}
figcaption{font-size:11px;color:var(--muted);order:2}
video{width:100%;aspect-ratio:1/1;object-fit:cover;background:#000;
  border-radius:7px;border:1px solid var(--line);cursor:pointer;order:1}
.ph{aspect-ratio:1/1;border:1px dashed var(--line);border-radius:7px;display:flex;
  align-items:center;justify-content:center;color:var(--muted);font-size:12px;order:1}

footer{margin-top:34px;color:var(--muted);font-size:13px;
  border-top:1px solid var(--line);padding-top:16px;max-width:80ch}
footer code{background:var(--card);border:1px solid var(--line);
  border-radius:4px;padding:1px 5px;font-size:12px}
</style></head><body>
<div class="wrap">
<h1>레퍼런스 합성 그리드</h1>
<p class="sub">타깃 하나에 두 합성 방법을 나란히. 왼쪽 열이 우리 파이프라인, 오른쪽이 BimArt이고,
각 열은 <b>합성된 레퍼런스</b>와 그것으로 학습한 <b>정책 롤아웃</b>입니다.</p>

<div class="bar">
  <button data-f="all" class="on">전체</button>
  <button data-f="mw">MICROWAVE</button>
  <button data-f="pm">BOX</button>
  <button data-f="lap">LAPTOP</button>
  <span class="spacer"></span>
  <button id="play">전체 재생</button>
  <button id="pause">전체 정지</button>
  <span class="spacer"></span>
  <button id="theme">테마</button>
</div>

{{CARDS}}

<footer>
데모 두 종은 <code>demo_gen.render_demo</code>의 기구학 재생, 롤아웃은
<code>eval_rl_games --record_video</code>의 물리 시뮬레이션입니다. 상태는 학습 로그의 종료 마커
(<code>MAX EPOCHS NUM!</code> / <code>Reward too low</code>)로 판정하며, 마커가 없으면 진행중입니다 —
<b>진행중</b> 런의 롤아웃은 중간 스냅샷이라 최종 성능이 아닙니다. AUC가 체크포인트와 다른 에폭에서
계산됐으면 붉게 표시됩니다. <code>build_videos.sh</code>로 영상을, 이 스크립트로 페이지를 다시 만듭니다.
</footer>
</div>
<script>
const vids = () => [...document.querySelectorAll('video')];
document.querySelectorAll('.bar button[data-f]').forEach(b => b.onclick = () => {
  document.querySelectorAll('.bar button[data-f]').forEach(x => x.classList.remove('on'));
  b.classList.add('on');
  const f = b.dataset.f;
  document.querySelectorAll('.catblock').forEach(c => {
    c.style.display = (f === 'all' || c.dataset.cat === f) ? '' : 'none';
  });
});
document.getElementById('play').onclick  = () => vids().forEach(v => v.play().catch(() => {}));
document.getElementById('pause').onclick = () => vids().forEach(v => v.pause());
document.getElementById('theme').onclick = () => {
  const r = document.documentElement;
  const dark = r.getAttribute('data-theme') === 'dark'
    || (!r.hasAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
  r.setAttribute('data-theme', dark ? 'light' : 'dark');
  try { localStorage.setItem('theme', r.getAttribute('data-theme')); } catch (e) {}
};
try { const t = localStorage.getItem('theme'); if (t) document.documentElement.setAttribute('data-theme', t); } catch (e) {}
vids().forEach(v => v.onclick = () => v.paused ? v.play().catch(() => {}) : v.pause());
// a card's four panels play together, and only while on screen
const io = new IntersectionObserver(es => es.forEach(e => {
  e.target.querySelectorAll('video').forEach(v =>
    e.isIntersecting ? v.play().catch(() => {}) : v.pause());
}), {threshold: 0.2});
document.querySelectorAll('.card, .source').forEach(c => io.observe(c));
</script>
</body></html>
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", default=os.path.join(HERE, "videos"))
    p.add_argument("--out", default=os.path.join(HERE, "index.html"))
    a = p.parse_args()
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(build(a.videos))
    print(f"wrote {a.out}  ({len(glob.glob(os.path.join(a.videos, '*.mp4')))} videos present)")


if __name__ == "__main__":
    main()
