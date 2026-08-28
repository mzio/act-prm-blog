#!/usr/bin/env python
"""Overlay Stage-1 EM reward curves to isolate the optimizer.

The finding this chart exists for: optim.get_optimizer defaults to name="sgd" and
main_pytorch never passed one, so every Stage-1 EM and Stage-2 SFT run in this project
trained with plain SGD. The Tinker reference uses Adam at the SAME lr 4e-5 and its EM reward
climbs ~0.55 -> ~0.88 over 38 steps. SGD's update is lr*grad -- with LoRA grads ~1e-3 that is
~4e-8/step -- while Adam's is roughly lr*sign(grad) and scale-invariant.

Curves are drawn on a shared axis so the comparison is direct; retail is available for both
optimizers, which makes the optimizer the only variable.

Usage: uv run --no-project python scripts/plot_optimizer_compare.py [--out optimizer_compare.html]
"""
import argparse
import glob
import json
import os
import statistics

SERIES = [
    ("retail · AdamW · lr 4e-5", "#2d6ca8",
     "logs/act_prm_tau2_retail/hf_qwen3_4b_instruct/retail_s1em_policy_adamw-*/"),
    ("retail · SGD · lr 4e-5", "#9b2c2c",
     "logs/act_prm_tau2_retail/hf_qwen3_4b_instruct/act-prm-*gc=act_prm-tc=pg*swb=0-*/"),
    ("airline · SGD · lr 4e-5 (100b)", "#c26a3d",
     "logs/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s1em_policy_ap32-*nb=100*/"),
    ("airline · SGD · lr 3e-3", "#8a6d3b",
     "logs/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s1em_policy_lr3e3-*/"),
]


def load(pat):
    ds = [d for d in glob.glob(pat) if os.path.exists(d + "metrics.jsonl")]
    if not ds:
        return None
    d = max(ds, key=os.path.getmtime)
    rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
    tr = sorted([r for r in rows if r.get("train/try_0/final_reward") is not None],
                key=lambda r: r.get("progress/batch", 0))
    if len(tr) < 5:
        return None
    return [(r.get("progress/batch", i), r["train/try_0/final_reward"]) for i, r in enumerate(tr)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="optimizer_compare.html")
    args = ap.parse_args()

    data = [(lab, col, load(pat)) for lab, col, pat in SERIES]
    data = [(l, c, p) for l, c, p in data if p]
    xmax = max(max(x for x, _ in p) for _, _, p in data)
    ymin = min(min(v for _, v in p) for _, _, p in data)
    ymax = max(max(v for _, v in p) for _, _, p in data)
    pad = (ymax - ymin) * 0.12
    lo, hi = ymin - pad, ymax + pad
    X0, Y0, W, H = 84, 60, 620, 300

    def px(x): return X0 + W * x / xmax
    def py(v): return Y0 + H - H * (v - lo) / (hi - lo)

    svg = []
    for f in (0, .25, .5, .75, 1):
        yy = Y0 + H - H * f
        svg.append(f'<line x1="{X0}" y1="{yy:.1f}" x2="{X0+W}" y2="{yy:.1f}" class="g"/>')
        svg.append(f'<text x="{X0-8}" y="{yy+4:.1f}" class="ax" text-anchor="end">{lo+(hi-lo)*f:.2f}</text>')
    for x in range(0, xmax + 1, max(10, (xmax // 10 // 10) * 10 or 10)):
        svg.append(f'<text x="{px(x):.1f}" y="{Y0+H+18}" class="ax" text-anchor="middle">{x}</text>')
    svg.append(f'<text x="{X0+W/2}" y="{Y0+H+38}" class="ax" text-anchor="middle">EM training batch</text>')
    svg.append(f'<text transform="translate({X0-52},{Y0+H/2}) rotate(-90)" class="ax" '
               f'text-anchor="middle">reward  p(x | s, z)</text>')

    rows_html = []
    for lab, col, pts in data:
        path = "M" + " L".join(f"{px(x):.1f},{py(v):.1f}" for x, v in pts)
        svg.append(f'<path d="{path}" fill="none" stroke="{col}" stroke-width="2.1" stroke-linejoin="round"/>')
        for x, v in pts:
            svg.append(f'<circle cx="{px(x):.1f}" cy="{py(v):.1f}" r="2.6" fill="{col}" '
                       f'opacity="0.85"><title>{lab} · batch {x}: {v:.4f}</title></circle>')
        lx, lv = pts[-1]
        svg.append(f'<text x="{px(lx)+7:.1f}" y="{py(lv)+4:.1f}" class="dl" fill="{col}">{lab.split(" · ")[1]}</text>')
        rw = [v for _, v in pts]
        h = len(rw) // 2
        d = statistics.mean(rw[h:]) - statistics.mean(rw[:h])
        sd = statistics.stdev(rw)
        rows_html.append(
            f'<tr><td><span class="sw" style="background:{col}"></span>{lab}</td>'
            f'<td class="n">{len(rw)}</td><td class="n">{rw[0]:.3f}</td>'
            f'<td class="n">{statistics.mean(rw[-5:]):.3f}</td>'
            f'<td class="n"><b>{d:+.4f}</b></td><td class="n">{sd:.4f}</td>'
            f'<td class="n">{abs(d)/sd:.2f}</td></tr>')

    legend = " ".join(
        f'<span class="lg"><span class="sw" style="background:{c}"></span>{l}</span>'
        for l, c, _ in data)

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Act-PRM Stage-1 EM — AdamW vs SGD</title><style>
:root{{--bg:#fbfbfa;--surface:#fff;--ink:#1b1b1a;--ink2:#4a4a48;--ink3:#77776f;--line:#e4e4e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16171a;--surface:#1e2024;--ink:#ececeb;--ink2:#b9b9b5;--ink3:#8b8b85;--line:#31333a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:920px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--ink3);font-size:13px;margin-bottom:16px}}
.lg{{font-size:12.5px;color:var(--ink2);margin-right:16px;white-space:nowrap}}
.sw{{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 4px;margin-top:10px}}
text.ax{{fill:var(--ink3);font-size:10.5px}} text.dl{{font-size:11.5px;font-weight:600}}
line.g{{stroke:var(--line);stroke-width:1}}
table{{border-collapse:collapse;width:100%;margin-top:18px;font-size:13px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:13px;color:var(--ink2);margin-top:16px;background:var(--surface);border:1px solid var(--line);
border-left:3px solid #2d6ca8;border-radius:8px;padding:12px 16px}}
</style></head><body><div class="wrap">
<h1>Act-PRM Stage-1 EM — the optimizer, not the learning rate</h1>
<div class="sub">All curves: r32/a32 LoRA, <code>advantage_mode action_probs</code>, group_size 4,
batch_size 4, Qwen3-4B-Instruct-2507. Retail appears with both optimizers at the same lr 4e-5,
so the optimizer is the only variable.</div>
<div>{legend}</div>
<div class="card"><svg viewBox="0 0 780 430" width="100%" role="img"
 aria-label="AdamW reward climbs from 0.4 to 0.78 and plateaus; all SGD runs stay flat">
{chr(10).join(svg)}
</svg></div>
<table><thead><tr><th>run</th><th>batches</th><th>start</th><th>last 5</th><th>delta</th><th>batch sd</th><th>|d|/sd</th></tr></thead>
<tbody>{''.join(rows_html)}</tbody></table>
<div class="note"><b>Reading:</b> at the <i>same</i> lr 4e-5, AdamW climbs 0.40 → ~0.78 and
plateaus by ~batch 20, while SGD is flat for 100 batches. Raising SGD's LR to 3e-3 (75×) moved
the adapter 50× further but left the reward flat too — so step size was never the constraint.
The cause is that <code>optim.get_optimizer</code> defaults to <code>name="sgd"</code> and
<code>main_pytorch</code> never passed one: SGD's update is <code>lr·grad</code> (with LoRA
gradients ~1e-3 that is ~4e-8/step), whereas Adam's is roughly <code>lr·sign(grad)</code> and
scale-invariant. This affected Stage-2 SFT identically, so the four-domain teacher-forced table
was produced under SGD and is a floor, not a tuned result.</div>
</div></body></html>"""
    open(args.out, "w").write(html)
    print(f"wrote {args.out} with {len(data)} series")
    for lab, _, pts in data:
        rw = [v for _, v in pts]
        print(f"  {lab:34} n={len(rw):3} {rw[0]:.3f} -> {statistics.mean(rw[-5:]):.3f}")


if __name__ == "__main__":
    main()
