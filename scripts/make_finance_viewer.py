#!/usr/bin/env python
"""Render data/finance_regrade.jsonl as a single self-contained HTML comparison page.

One card per answered episode: the question, the gold answer and the model's answer side by
side with their numeric figures highlighted, the judge's verdict under both parsers, and the
judge's own rationale. The point is to let a reader check the grading rather than take the
aggregate on trust -- every finance arm scores 0, and that claim should be inspectable.

Usage: uv run --no-project python scripts/make_finance_viewer.py [--out finance_answers.html]
"""
import argparse
import html
import json
import re
from collections import Counter

NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*%?")

CSS = """
:root{
  --bg:#fbfbfa; --surface:#fff; --ink:#1b1b1a; --ink2:#4a4a48; --ink3:#77776f;
  --line:#e4e4e0; --hit:#1f6f5c; --hit-bg:#e2f1ec; --miss:#8a5a00; --miss-bg:#fdf1dc;
  --no:#9b2c2c; --no-bg:#fbe9e9; --yes:#1f6f5c; --yes-bg:#e2f1ec;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16171a; --surface:#1e2024; --ink:#ececeb; --ink2:#b9b9b5; --ink3:#8b8b85;
  --line:#31333a; --hit:#7fd8bf; --hit-bg:#16342c; --miss:#e8b566; --miss-bg:#3a2d15;
  --no:#f0918f; --no-bg:#3a1e1e; --yes:#7fd8bf; --yes-bg:#16342c;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:22px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--ink3);font-size:13px;margin-bottom:24px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:22px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:8px;
  padding:10px 14px;min-width:120px}
.stat b{display:block;font-size:20px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}
.stat span{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:20px;align-items:center}
button{font:inherit;font-size:13px;padding:5px 12px;border-radius:999px;cursor:pointer;
  border:1px solid var(--line);background:var(--surface);color:var(--ink2)}
button[aria-pressed=true]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;
  padding:18px 20px;margin-bottom:14px}
.meta{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:10px}
.tag{font-size:11px;padding:2px 8px;border-radius:5px;border:1px solid var(--line);
  color:var(--ink2);font-weight:600;letter-spacing:.02em}
.v{font-size:11px;padding:2px 8px;border-radius:5px;font-weight:700}
.v.no{background:var(--no-bg);color:var(--no)} .v.yes{background:var(--yes-bg);color:var(--yes)}
.warn{background:var(--miss-bg);color:var(--miss);font-size:11px;padding:2px 8px;
  border-radius:5px;font-weight:600}
.q{font-weight:600;margin:2px 0 14px;line-height:1.45}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:820px){.cols{grid-template-columns:1fr}}
.col h4{margin:0 0 6px;font-size:11px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--ink3);font-weight:700}
.col div.body{font-size:13.5px;color:var(--ink2);white-space:pre-wrap;word-break:break-word}
mark.hit{background:var(--hit-bg);color:var(--hit);padding:0 3px;border-radius:3px;font-weight:600}
mark.miss{background:var(--miss-bg);color:var(--miss);padding:0 3px;border-radius:3px;font-weight:600}
details{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
summary{cursor:pointer;font-size:12px;color:var(--ink3);font-weight:600}
details .body{font-size:13px;color:var(--ink2);white-space:pre-wrap;margin-top:8px}
.legend{font-size:12px;color:var(--ink3);margin:-8px 0 20px}
.none{color:var(--ink3);font-style:italic}
"""

JS = """
const btns=[...document.querySelectorAll('[data-filter]')];
btns.forEach(b=>b.onclick=()=>{
  btns.forEach(x=>x.setAttribute('aria-pressed', x===b));
  const f=b.dataset.filter;
  document.querySelectorAll('.card').forEach(c=>{
    c.style.display = (f==='all'||c.dataset.arm===f||c.dataset.set===f) ? '' : 'none';
  });
});
"""


def mark_numbers(text: str, found: set[str]) -> str:
    """Highlight numeric tokens: green if it also appears in the counterpart, amber if not."""
    def norm(tok):
        t = tok.replace(",", "").rstrip("%").rstrip(".")
        try:
            return f"{float(t):g}"
        except ValueError:
            return None

    out, last = [], 0
    for m in NUM.finditer(text or ""):
        out.append(html.escape(text[last:m.start()]))
        n = norm(m.group())
        cls = "hit" if (n and n in found) else "miss"
        out.append(f'<mark class="{cls}">{html.escape(m.group())}</mark>')
        last = m.end()
    out.append(html.escape((text or "")[last:]))
    return "".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/finance_regrade.jsonl")
    ap.add_argument("--out", default="finance_answers.html")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.src)]
    ans = [r for r in recs if r.get("answered")]
    arms = sorted({r["arm"] for r in ans})
    n_unparsed = sum(1 for r in ans if not r.get("old_parsed"))
    n_flip = sum(1 for r in ans if r.get("verdict_old") != r.get("verdict_new"))
    n_yes = sum(1 for r in ans if r.get("verdict_new") == "yes")
    cov = [len(r["gold_numbers_found"]) / len(r["gold_numbers"])
           for r in ans if r.get("gold_numbers")]
    mean_cov = 100 * sum(cov) / len(cov) if cov else 0

    cards = []
    for r in sorted(ans, key=lambda x: (x["arm"], x["set"])):
        gold_n = set(r["gold_numbers"])
        resp_n = set(r["gold_numbers_found"])
        # gold: green where the model reproduced it. model: green where gold contains it.
        def norm_all(t):
            s = set()
            for tok in NUM.findall(t or ""):
                tt = tok.replace(",", "").rstrip("%").rstrip(".")
                try:
                    s.add(f"{float(tt):g}")
                except ValueError:
                    pass
            return s
        model_nums = norm_all(r["response"])
        v = r.get("verdict_new", "no")
        warn = '<span class="warn">judge text had no <code>correct:</code> line</span>' \
               if not r.get("old_parsed") else ""
        cards.append(f"""
<div class="card" data-arm="{html.escape(r['arm'])}" data-set="{html.escape(r['set'])}">
  <div class="meta">
    <span class="tag">{html.escape(r['arm'])}</span>
    <span class="tag">{html.escape(r['set'])} set</span>
    <span class="v {v}">judge: {v}</span>
    <span class="tag">{len(r['gold_numbers_found'])}/{len(r['gold_numbers'])} gold figures reproduced</span>
    <span class="tag">{r.get('n_assistant_turns','?')} turns</span>
    {warn}
  </div>
  <div class="q">{html.escape(r['question'])}</div>
  <div class="cols">
    <div class="col"><h4>Ground truth</h4><div class="body">{mark_numbers(r['gold'], model_nums)}</div></div>
    <div class="col"><h4>Model answer</h4><div class="body">{mark_numbers(r['response'], gold_n)}</div></div>
  </div>
  <details><summary>Judge rationale</summary>
    <div class="body">{html.escape(r.get('raw_judge') or '(none recorded)')}</div></details>
</div>""")

    btns = "".join(f'<button data-filter="{a}">{a}</button>' for a in arms)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Act-PRM · finance rollout answers vs ground truth</title><style>{CSS}</style></head><body>
<div class="wrap">
<h1>Finance rollout — model answers vs ground truth</h1>
<div class="sub">Every finance arm scores 0/10 (fair) and 0/29 (hard). This page shows each
answer the grader actually saw, so the null can be checked rather than taken on trust.
Re-graded offline with the corrected verdict parser; no re-rollout.</div>

<div class="stats">
  <div class="stat"><b>{len(recs)}</b><span>episodes</span></div>
  <div class="stat"><b>{len(ans)}</b><span>reached an answer</span></div>
  <div class="stat"><b>{n_yes}</b><span>judged correct</span></div>
  <div class="stat"><b>{mean_cov:.0f}%</b><span>gold figures reproduced</span></div>
  <div class="stat"><b>{n_unparsed}/{len(ans)}</b><span>old parser defaulted</span></div>
  <div class="stat"><b>{n_flip}</b><span>verdicts changed</span></div>
</div>

<div class="bar"><button data-filter="all" aria-pressed="true">all</button>{btns}
  <button data-filter="fair">fair set</button><button data-filter="hard">hard set</button></div>
<div class="legend">Numbers are highlighted <mark class="hit">green</mark> when the figure appears
on both sides and <mark class="miss">amber</mark> when it appears on only one — so a gold answer
full of amber is one whose key quantities the model never produced.</div>

{''.join(cards)}
</div><script>{JS}</script></body></html>"""

    with open(args.out, "w") as f:
        f.write(doc)
    print(f"wrote {args.out}  ({len(ans)} answers, {len(recs)} episodes)")
    print("  arms:", Counter(r["arm"] for r in ans).most_common())


if __name__ == "__main__":
    main()
