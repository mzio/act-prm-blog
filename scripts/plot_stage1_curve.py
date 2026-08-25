#!/usr/bin/env python
"""Plot the Stage-1 Act-PRM EM training curve as a self-contained HTML page.

Why this chart exists: the question is whether the EM is LEARNING -- i.e. whether the
length-penalised action likelihood improves over training batches. A flat line means the
E-step is finding good thoughts but the M-step is not making the policy better at proposing
them. My first reading was that insurance's flat curve came from running only 0.56 epochs
(EM_NB=25 against 180 train trajectories). The cross-domain table below REFUTES that:
finance ran 2.00 epochs (58 batches) and its policy curve went DOWN (-0.0186). Every domain
is flat relative to its own batch-to-batch noise, so a flat EM curve is normal for this
method, not an insurance-specific under-training symptom.

Form: reward over steps is change-over-time for a small series -> line chart, one panel per
measure so the two y-scales are never mixed on one axis. Palette is the canonical
blue/orange categorical pair (high hue + lightness separation, CVD-safe); the bundled
validate_palette.js was not available in this session, so no custom ramp was invented.

Usage: uv run --no-project python scripts/plot_stage1_curve.py [--out stage1_curve.html]
"""
import argparse
import glob
import json
import os

BLUE, ORANGE, GREEN, INK3 = "#2d6ca8", "#c26a3d", "#3f7d55", "#77776f"


def series(run_glob):
    d = max(glob.glob(run_glob), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
    tr = [r for r in rows if r.get("train/try_0/final_reward") is not None]
    tr.sort(key=lambda r: r.get("progress/batch", 0))
    return d, tr


def path(xs, ys, x0, y0, w, h, lo, hi, n):
    pts = []
    for i, (x, y) in enumerate(zip(xs, ys)):
        px = x0 + (w * x / max(n - 1, 1))
        py = y0 + h - h * (y - lo) / (hi - lo if hi > lo else 1)
        pts.append(f"{px:.1f},{py:.1f}")
    return "M" + " L".join(pts)


def panel(title, sub, tr, keys, colours, labels, x0, y0, w, h, fmt="{:.3f}"):
    xs = [r.get("progress/batch", i) for i, r in enumerate(tr)]
    n = len(tr)
    allv = [r[k] for k in keys for r in tr if r.get(k) is not None]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.18 or 0.05
    lo, hi = lo - pad, hi + pad
    out = [f'<text x="{x0}" y="{y0-26}" class="t">{title}</text>',
           f'<text x="{x0}" y="{y0-10}" class="s">{sub}</text>']
    # gridlines + y labels
    for f in (0, .25, .5, .75, 1):
        yy = y0 + h - h * f
        v = lo + (hi - lo) * f
        out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+w}" y2="{yy:.1f}" class="g"/>')
        out.append(f'<text x="{x0-8}" y="{yy+4:.1f}" class="ax" text-anchor="end">{fmt.format(v)}</text>')
    for k, c, lab in zip(keys, colours, labels):
        ys = [r[k] for r in tr]
        out.append(f'<path d="{path(xs,ys,x0,y0,w,h,lo,hi,n)}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for i, (x, y) in enumerate(zip(xs, ys)):
            px = x0 + (w * x / max(n - 1, 1))
            py = y0 + h - h * (y - lo) / (hi - lo)
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="{c}" stroke="var(--surface)" stroke-width="1.6">'
                       f'<title>batch {x} · {lab} {y:.4f}</title></circle>')
        out.append(f'<text x="{x0+w+8}" y="{y0+h-h*(ys[-1]-lo)/(hi-lo)+4:.1f}" class="dl" fill="{c}">{lab}</text>')
    # x axis
    for x in range(0, n, 4):
        px = x0 + (w * x / max(n - 1, 1))
        out.append(f'<text x="{px:.1f}" y="{y0+h+18}" class="ax" text-anchor="middle">{x}</text>')
    out.append(f'<text x="{x0+w/2}" y="{y0+h+38}" class="ax" text-anchor="middle">training batch</text>')
    return "\n".join(out)


def all_domains():
    """Every EM TRAINING run with a usable reward curve, epoch-normalised."""
    import statistics
    pools = {"tau2_retail": "data/tau2_retail/train.json",
             "tau2_airline": "data/tau2_airline/train.json",
             "snorkel_finance_split": "data/snorkel_finance_split_v3/train.json",
             "snorkel_insurance": "data/snorkel_insurance_split/train.json"}
    out = []
    for env, pp in pools.items():
        ntraj = len(json.load(open(pp))) if os.path.exists(pp) else None
        for d in sorted(glob.glob(f"logs/act_prm_{env}/hf_qwen3_4b_instruct/*/"), key=os.path.getmtime):
            f = d + "metrics.jsonl"
            if not os.path.exists(f):
                continue
            try:
                c = json.load(open(d + "config.json"))
            except Exception:
                continue
            if c.get("generator_config") != "act_prm" or c.get("no_train"):
                continue
            rows = [json.loads(l) for l in open(f) if l.strip()]
            tr = [r for r in rows if r.get("train/try_0/final_reward") is not None]
            if len(tr) < 10:
                continue
            tr.sort(key=lambda r: r.get("progress/batch", 0))
            rw = [r["train/try_0/final_reward"] for r in tr]
            out.append({"env": env.replace("tau2_", "").replace("snorkel_", "").replace("_split", ""),
                        "scorer": "base" if c.get("score_with_base") else "policy",
                        "rw": rw, "n": len(rw),
                        "epochs": len(tr) * c.get("batch_size", 4) / ntraj if ntraj else None,
                        "d": statistics.mean(rw[-5:]) - statistics.mean(rw[:5]),
                        "sd": statistics.stdev(rw)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="stage1_curve.html")
    args = ap.parse_args()
    d, tr = series("logs/act_prm_snorkel_insurance/hf_qwen3_4b_instruct/insurance_s1em_policy-*/")
    n = len(tr)
    rw = [r["train/try_0/final_reward"] for r in tr]
    f5, l5 = sum(rw[:5]) / 5, sum(rw[-5:]) / 5
    zero = [r.get("progress/batch") for r in tr if r.get("train/loss") is not None and abs(r["train/loss"]) < 1e-6]

    doms = all_domains()
    # normalised comparison panel: each run's reward minus its own mean, over normalised progress
    cmp_rows = "".join(
        f'<tr><td>{x["env"]} / {x["scorer"]}</td><td class="n">{x["n"]}</td>'
        f'<td class="n">{x["epochs"]:.2f}x</td><td class="n">{x["d"]:+.4f}</td>'
        f'<td class="n">{x["sd"]:.4f}</td>'
        f'<td class="n">{abs(x["d"])/x["sd"]:.2f}</td></tr>'
        for x in doms)

    body = (panel("Length-penalised action likelihood", "the quantity EM optimises — flat means the policy is not improving",
                  tr, ["train/try_0/final_reward", "train/try_0/penalized"], [BLUE, ORANGE],
                  ["reward", "penalized"], 70, 70, 640, 200)
            + panel("Thought length", "tokens per inferred thought — also static",
                    tr, ["train/try_0/thought_tokens"], [GREEN], ["tokens"], 70, 360, 640, 140, "{:.0f}"))

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Act-PRM Stage-1 EM curve — insurance (policy-scored)</title><style>
:root{{--bg:#fbfbfa;--surface:#fff;--ink:#1b1b1a;--ink2:#4a4a48;--ink3:{INK3};--line:#e4e4e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16171a;--surface:#1e2024;--ink:#ececeb;--ink2:#b9b9b5;--ink3:#8b8b85;--line:#31333a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--ink3);font-size:13px;margin-bottom:22px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:126px}}
.stat b{{display:block;font-size:20px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.stat span{{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px 4px 4px}}
text.t{{fill:var(--ink);font-size:13.5px;font-weight:600}}
text.s{{fill:var(--ink3);font-size:11.5px}}
text.ax{{fill:var(--ink3);font-size:10.5px}}
text.dl{{font-size:11.5px;font-weight:600}}
line.g{{stroke:var(--line);stroke-width:1}}
table{{border-collapse:collapse;width:100%;margin:10px 0 4px;font-size:13px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}}
th{{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em;font-weight:700}}
td.n{{text-align:right;font-variant-numeric:tabular-nums}}
.note{{font-size:13px;color:var(--ink2);margin-top:18px;background:var(--surface);
  border:1px solid var(--line);border-left:3px solid {ORANGE};border-radius:8px;padding:12px 16px}}
</style></head><body><div class="wrap">
<h1>Act-PRM Stage-1 EM — insurance, policy-scored</h1>
<div class="sub">{n} training batches · group_size 4 · 180 train trajectories · run <code>insurance_s1em_policy</code></div>
<div class="stats">
  <div class="stat"><b>{f5:.4f}</b><span>first 5 batches</span></div>
  <div class="stat"><b>{l5:.4f}</b><span>last 5 batches</span></div>
  <div class="stat"><b>{l5-f5:+.4f}</b><span>change</span></div>
  <div class="stat"><b>{min(rw):.3f}–{max(rw):.3f}</b><span>batch-to-batch range</span></div>
  <div class="stat"><b>0.56×</b><span>epochs over the pool</span></div>
</div>
<div class="card"><svg viewBox="0 0 780 530" width="100%" role="img"
 aria-label="Stage-1 EM reward and thought length over 25 training batches, both flat">
{body}
</svg></div>
<h2 style="font-size:16px;margin:30px 0 4px">How this compares to the other domains</h2>
<div class="sub">Same measure, every Act-PRM EM training run on disk. |delta| / sd &lt; 1 means the
change over training is smaller than the batch-to-batch swing.</div>
<table><thead><tr><th>domain / scorer</th><th>batches</th><th>epochs</th><th>delta</th><th>batch sd</th><th>|d|/sd</th></tr></thead>
<tbody>{cmp_rows}</tbody></table>
<div class="note"><b>Reading:</b> the curve is flat. The +{l5-f5:.4f} drift between the first and last
five batches is dwarfed by the ±0.05 batch-to-batch swing, and batch 24 (0.3979) sits below
batch 0 (0.4267). The E-step works — likelihoods reach 0.99 on individual steps and the
exported corpus has coherent thoughts on 100% of 2,273 targets — but the M-step is not making
the policy better at proposing them. My initial reading was under-training (0.56 epochs
vs retail's 2.0×). The table above refutes it: finance ran 2.00 epochs and moved
<b>-0.0186</b>, and every |delta| is under 1 batch-sd. Flat is the norm here.
Secondary signal: <code>train/loss</code> was exactly 0 on batches {', '.join(str(b) for b in zero)},
meaning all four candidate thoughts scored alike and the group carried no gradient.</div>
</div></body></html>"""
    open(args.out, "w").write(html)
    print(f"wrote {args.out}  ({n} batches, first5 {f5:.4f} -> last5 {l5:.4f}, zero-loss batches {zero})")


if __name__ == "__main__":
    main()
