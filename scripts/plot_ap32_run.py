#!/usr/bin/env python
"""Chart the airline Stage-1 EM run (r32/a32, action_probs, lr 4e-5, 100 batches).

Three things worth seeing together:
  1. reward over batches -- the EM objective, length-normalised p(x|s,z)
  2. thought length -- watches for degenerate collapse/inflation
  3. ADAPTER MOVEMENT max|B@A| at the save_every checkpoints -- the measurement that decides
     whether the M-step is doing anything. lora_B is zero-initialised, so B@A IS the whole
     adapter contribution; against base weights of order 1e-2, <0.1% is a no-op.

The adapter panel is the point of the run: every previous Stage-1 EM stopped at 25 batches
(an arbitrary constant from the initial commit; pg.yaml's own default is 200) and reported a
no-op. Extending to 100 tests whether that was a real ceiling or just early stopping.

Usage: uv run --no-project python scripts/plot_ap32_run.py [--out stage1_airline_ap32.html]
"""
import argparse
import glob
import json
import os

BLUE, ORANGE, GREEN = "#2d6ca8", "#c26a3d", "#3f7d55"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="logs/act_prm_tau2_airline/hf_qwen3_4b_instruct/airline_s1em_policy_ap32-*/")
    ap.add_argument("--compare", default="", help="glob for a second run to overlay")
    ap.add_argument("--label", default="this run")
    ap.add_argument("--clabel", default="comparison")
    ap.add_argument("--nb", type=int, default=100)
    ap.add_argument("--out", default="stage1_airline_ap32.html")
    args = ap.parse_args()

    d = max(glob.glob(args.run), key=os.path.getmtime)
    rows = [json.loads(l) for l in open(d + "metrics.jsonl") if l.strip()]
    tr = sorted([r for r in rows if r.get("train/try_0/final_reward") is not None],
                key=lambda r: r.get("progress/batch", 0))
    xs = [r.get("progress/batch", i) for i, r in enumerate(tr)]
    rw = [r["train/try_0/final_reward"] for r in tr]
    tt = [r["train/try_0/thought_tokens"] for r in tr]
    ev = [r for r in rows if any("eval/" in k for k in r)]

    # adapter measurements taken during the run (batch, max|B@A|)
    growth = [(5, 1.228e-06), (15, 3.698e-06), (45, 1.127e-05)]

    def line(pts, x0, y0, w, h, xmax, lo, hi, colour, logy=False):
        import math
        def ty(v):
            if logy:
                return y0 + h - h * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            return y0 + h - h * (v - lo) / (hi - lo)
        pth = " L".join(f"{x0 + w*x/xmax:.1f},{ty(v):.1f}" for x, v in pts)
        dots = "".join(
            f'<circle cx="{x0 + w*x/xmax:.1f}" cy="{ty(v):.1f}" r="3.4" fill="{colour}" '
            f'stroke="var(--surface)" stroke-width="1.6"><title>batch {x}: {v:.4g}</title></circle>'
            for x, v in pts)
        return f'<path d="M{pth}" fill="none" stroke="{colour}" stroke-width="2"/>{dots}'

    def panel(title, sub, pts, colour, x0, y0, w, h, fmt="{:.3f}", logy=False, hlines=()):
        import math
        vals = [v for _, v in pts]
        lo, hi = min(vals), max(vals)
        if logy:
            lo, hi = lo / 3, max(hi * 3, 1.2e-4)
        else:
            pad = (hi - lo) * 0.2 or 0.05
            lo, hi = lo - pad, hi + pad
        out = [f'<text x="{x0}" y="{y0-26}" class="t">{title}</text>',
               f'<text x="{x0}" y="{y0-10}" class="s">{sub}</text>']
        for f in (0, .25, .5, .75, 1):
            yy = y0 + h - h * f
            v = (10 ** (math.log10(lo) + f * (math.log10(hi) - math.log10(lo)))) if logy else lo + (hi - lo) * f
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+w}" y2="{yy:.1f}" class="g"/>')
            out.append(f'<text x="{x0-8}" y="{yy+4:.1f}" class="ax" text-anchor="end">{fmt.format(v)}</text>')
        for hv, lab, col in hlines:
            yy = y0 + h - h * (math.log10(hv) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
            out.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+w}" y2="{yy:.1f}" '
                       f'stroke="{col}" stroke-width="1.4" stroke-dasharray="5 4"/>')
            out.append(f'<text x="{x0+w-4}" y="{yy-5:.1f}" class="ax" text-anchor="end" fill="{col}">{lab}</text>')
        out.append(line(pts, x0, y0, w, h, 100, lo, hi, colour, logy))
        for x in range(0, 101, 20):
            out.append(f'<text x="{x0+w*x/100:.1f}" y="{y0+h+18}" class="ax" text-anchor="middle">{x}</text>')
        out.append(f'<text x="{x0+w/2}" y="{y0+h+38}" class="ax" text-anchor="middle">training batch (of 100)</text>')
        return "\n".join(out)

    body = (panel("EM reward — length-normalised p(x | s, z)",
                  "the objective; flat means the M-step is not improving the policy",
                  list(zip(xs, rw)), BLUE, 78, 66, 620, 170)
            + panel("Thought length (tokens)", "watches for degenerate collapse or inflation",
                    list(zip(xs, tt)), GREEN, 78, 320, 620, 110, "{:.0f}")
            + panel("Adapter movement  max|B@A|  (log scale)",
                    "lora_B is zero-init, so B@A is the entire adapter. Base weights ~1e-2.",
                    growth, ORANGE, 78, 520, 620, 150, "{:.1e}", logy=True,
                    hlines=[(1e-5, "0.1% of base — no-op boundary", "#9b2c2c"),
                            (1e-4, "1% of base — 'marginal'", "#1f6f5c")]))

    import statistics
    h = len(rw) // 2
    delta = statistics.mean(rw[h:]) - statistics.mean(rw[:h])
    sd = statistics.stdev(rw)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage-1 EM — airline ap32</title><style>
:root{{--bg:#fbfbfa;--surface:#fff;--ink:#1b1b1a;--ink2:#4a4a48;--ink3:#77776f;--line:#e4e4e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16171a;--surface:#1e2024;--ink:#ececeb;--ink2:#b9b9b5;--ink3:#8b8b85;--line:#31333a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:900px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}}
.sub{{color:var(--ink3);font-size:13px;margin-bottom:20px}}
.stats{{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}}
.stat{{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:118px}}
.stat b{{display:block;font-size:19px;font-variant-numeric:tabular-nums;letter-spacing:-.02em}}
.stat span{{color:var(--ink3);font-size:11px;text-transform:uppercase;letter-spacing:.05em}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:6px 4px}}
text.t{{fill:var(--ink);font-size:13.5px;font-weight:600}} text.s{{fill:var(--ink3);font-size:11.5px}}
text.ax{{fill:var(--ink3);font-size:10.5px}} line.g{{stroke:var(--line);stroke-width:1}}
.note{{font-size:13px;color:var(--ink2);margin-top:16px;background:var(--surface);border:1px solid var(--line);
border-left:3px solid {ORANGE};border-radius:8px;padding:12px 16px}}
</style></head><body><div class="wrap">
<h1>Act-PRM Stage-1 EM — airline, <code>thoughts_policy</code></h1>
<div class="sub">r32/a32 · advantage_mode <code>action_probs</code> (raw, unnormalised) · length_penalty 0 ·
lr 4e-5 · group_size 4 · batch_size 4 · {len(tr)}/100 batches so far · eval runs only at batch 100
({len(ev)} eval rows yet)</div>
<div class="stats">
 <div class="stat"><b>{statistics.mean(rw[:h]):.4f}</b><span>reward, first half</span></div>
 <div class="stat"><b>{statistics.mean(rw[h:]):.4f}</b><span>reward, second half</span></div>
 <div class="stat"><b>{delta:+.4f}</b><span>change</span></div>
 <div class="stat"><b>{abs(delta)/sd:.2f}</b><span>|change| / batch sd</span></div>
 <div class="stat"><b>{growth[-1][1]:.2e}</b><span>max|B@A| @ b{growth[-1][0]}</span></div>
 <div class="stat"><b>0</b><span>zero-gradient batches</span></div>
</div>
<div class="card"><svg viewBox="0 0 760 700" width="100%" role="img"
 aria-label="EM reward flat, thought length stable, adapter movement growing linearly out of the no-op band">
{body}
</svg></div>
<div class="note"><b>Reading:</b> the adapter IS moving — max|B@A| grows 1.23e-06 → 3.70e-06 → 1.13e-05
across batches 5/15/45, i.e. ~3× per 3× the batches (linear), crossing out of the &lt;0.1%-of-base
no-op band that every previous Stage-1 run reported. Those runs all stopped at 25 batches, an
arbitrary constant from the initial commit, so they were measuring an adapter that had not yet
grown rather than one that could not. <b>But the EM reward is flat</b> ({delta:+.4f}, |Δ|/sd =
{abs(delta)/sd:.2f}): movement without improvement. Either the updates are too small to shift a
likelihood already near 0.44, or they are not in a direction that improves it. Linear
extrapolation reaches ~2.3e-05 at batch 100 — still below the 1%-of-base line. The unnormalised
<code>action_probs</code> advantage does fix one thing outright: <b>zero</b> zero-gradient batches,
versus 2/9 and 4/25 under group-normalised <code>em</code> weights.</div>
</div></body></html>"""
    open(args.out, "w").write(html)
    print(f"wrote {args.out} ({len(tr)} batches, delta {delta:+.4f}, |d|/sd {abs(delta)/sd:.2f})")


if __name__ == "__main__":
    main()
