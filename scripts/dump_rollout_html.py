#!/usr/bin/env python
"""Render rollout episodes from the Stage-2 checkpoints as a readable HTML transcript.

Reads the rollout evals' replay buffers (full_state = the complete conversation) and
lays each arm's episodes side by side, so the arms can be compared as BEHAVIOUR rather
than as a single completion rate. Highlights the reasoning prefix separately from the
tool call, because the central finding is about what the thoughts contain, not whether
the model emits any.
"""
import argparse
import glob
import html
import os

from datasets import load_from_disk

ap = argparse.ArgumentParser()
ap.add_argument("--domain", default="retail")
ap.add_argument("--episodes", type=int, default=2)
ap.add_argument("--max-turns", type=int, default=14)
ap.add_argument("--out", default="/tmp/aprm_plots/rollouts.html")
args = ap.parse_args()

ARMS = [("actions_only", "#D55E00"), ("thoughts_policy_adamw30", "#0072B2"),
        ("expert_thoughts", "#009E73")]
R = f"checkpoints_lora/tau2bench_{args.domain}_rlvr/hf_qwen3_4b_instruct"


def episodes(arm):
    ds = sorted(glob.glob(f"{R}/{args.domain}_rollout_{arm}_flat32-*/replay_buffer"))
    if not ds:
        return []
    d = load_from_disk(ds[0])
    by_task = {}
    for r in d:
        tp = r.get("task_prompt") or ""
        # keep the LAST row per task: its full_state is the complete conversation
        if r.get("is_complete") or tp not in by_task:
            by_task[tp] = r
    return list(by_task.items())


def render_turn(m):
    role = m.get("role", "?")
    content = m.get("content") or ""
    if role == "assistant":
        if "<tool_call>" in content:
            pre, tc = content.split("<tool_call>", 1)
            pre, tc = pre.strip(), "<tool_call>" + tc
        else:
            pre, tc = content.strip(), ""
        out = '<div class="t assistant">'
        if pre:
            out += f'<div class="think"><span class="lbl">thought</span>{html.escape(pre)}</div>'
        if tc:
            out += f'<div class="act"><span class="lbl">action</span><pre>{html.escape(tc.strip())}</pre></div>'
        if not pre and not tc:
            out += f'<div class="act">{html.escape(content[:600])}</div>'
        return out + "</div>"
    cls = {"user": "user", "tool": "tool", "system": "system"}.get(role, "other")
    body = html.escape(content[:700]) + ("…" if len(content) > 700 else "")
    return f'<div class="t {cls}"><span class="lbl">{role}</span>{body}</div>'


cols = []
for arm, colour in ARMS:
    eps = episodes(arm)
    blocks = []
    for tp, row in eps[: args.episodes]:
        msgs = [m for m in (row.get("full_state") or []) if m.get("role") != "system"]
        rew = row.get("reward")
        ok = "solved" if (rew or 0) > 0 else "not solved"
        turns = "".join(render_turn(m) for m in msgs[: args.max_turns])
        more = f'<div class="more">… {len(msgs) - args.max_turns} more messages</div>' if len(msgs) > args.max_turns else ""
        blocks.append(
            f'<div class="ep"><div class="ephead">task: {html.escape(tp[:150])}…'
            f'<span class="badge {"ok" if (rew or 0) > 0 else "no"}">{ok}</span></div>{turns}{more}</div>'
        )
    n_think = sum(
        1 for _, r in eps for m in (r.get("full_state") or [])
        if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")
        and len((m.get("content") or "").split("<tool_call>")[0].strip()) > 20
    )
    n_asst = sum(
        1 for _, r in eps for m in (r.get("full_state") or [])
        if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")
    )
    cols.append(
        f'<div class="col"><h2 style="border-color:{colour}">{arm}</h2>'
        f'<div class="stat">{n_think}/{n_asst} tool-calling turns carry a reasoning prefix '
        f'({100 * n_think / max(n_asst, 1):.0f}%)</div>{"".join(blocks)}</div>'
    )

doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Act-PRM Stage-2 rollouts — {args.domain}</title><style>
:root{{--bg:#fbfbfa;--surface:#fff;--ink:#1b1b1a;--ink2:#4a4a48;--ink3:#77776f;--line:#e4e4e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16171a;--surface:#1e2024;--ink:#ececeb;--ink2:#b9b9b5;--ink3:#8b8b85;--line:#31333a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1800px;margin:0 auto;padding:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--ink3);font-size:13px;margin-bottom:18px}}
.cols{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;align-items:start}}
.col h2{{font-size:15px;margin:0 0 8px;padding-left:9px;border-left:4px solid;font-family:ui-monospace,Menlo,monospace}}
.stat{{font-size:12px;color:var(--ink2);background:var(--surface);border:1px solid var(--line);
border-radius:7px;padding:7px 10px;margin-bottom:10px}}
.ep{{background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:10px;margin-bottom:14px}}
.ephead{{font-size:11.5px;color:var(--ink3);border-bottom:1px solid var(--line);padding-bottom:7px;margin-bottom:9px}}
.badge{{float:right;font-weight:600;padding:1px 7px;border-radius:20px;font-size:10.5px}}
.badge.ok{{background:#1f6f5c22;color:#1f6f5c}} .badge.no{{background:#9b2c2c22;color:#9b2c2c}}
.t{{margin-bottom:8px;font-size:12.5px}}
.lbl{{display:inline-block;font-size:9.5px;text-transform:uppercase;letter-spacing:.06em;
color:var(--ink3);margin-right:6px;vertical-align:top}}
.user{{color:var(--ink2)}} .tool{{color:var(--ink3);font-size:11.5px}}
.think{{background:#0072B215;border-left:3px solid #0072B2;padding:6px 9px;border-radius:0 5px 5px 0;margin-bottom:4px}}
.act pre{{margin:3px 0 0;font-family:ui-monospace,Menlo,monospace;font-size:11px;white-space:pre-wrap;
word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:5px;padding:6px}}
.more{{font-size:11px;color:var(--ink3);font-style:italic;margin-top:6px}}
</style></head><body><div class="wrap">
<h1>Act-PRM Stage-2 SFT — live rollouts on held-out {args.domain} tasks</h1>
<div class="sub">Checkpoints: <code>step_best</code> from the sft_flat sweep (AdamW lr 1e-3, steps_per_batch 32,
hide-observations). Rollouts in the real tau2 gym, Claude user simulator, no training —
generation only. First {args.max_turns} messages per episode.</div>
<div class="cols">{"".join(cols)}</div></div></body></html>"""
os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write(doc)
print(f"wrote {args.out} ({len(doc)//1024} KB)")
