#!/usr/bin/env python
"""Side-by-side rollout transcripts for the SAME retail task across four checkpoints.

Matching by task_prompt is the point: the four runs share an identical 42-task eval set
(verified set-equal), so laying the same task beside itself isolates behaviour from task
difficulty. Colour-codes each assistant turn by what it DOES -- talk to the user vs call a
tool -- because the measured difference between generations is how turns are spent
(OLD: 6.1 tool responses per 10.5 assistant turns; NEW: 10.5 per 10.5), not how many.
"""
import glob
import html
import os

from datasets import load_from_disk

RUNS = [
    ("OLD actions_only",    "retail_rollout_actions_only_lr3e_3-*",              "#8a6d3b"),
    ("OLD thoughts_policy", "retail_rollout_thoughts_policy_lr3e_3-*",           "#1f6f5c"),
    ("NEW actions_only",    "retail_rollout_actions_only_flat32-*",              "#D55E00"),
    ("NEW thoughts_policy", "retail_rollout_thoughts_policy_adamw30_flat32-*",   "#0072B2"),
]
ROOT = "checkpoints_lora/tau2bench_retail_rlvr/hf_qwen3_4b_instruct"
N_TASKS, MAX_MSG = 3, 10**6   # MAX_MSG effectively unlimited: show the whole trajectory


def load(pat, logpat):
    """Key episodes by TASK_ID, not task_prompt: the user simulator is an LLM, so it
    phrases the same task differently every run (41 vs 43 distinct openings, none
    matching). task_id is stable and identically ordered across all four runs."""
    import json as _json
    ds = sorted(glob.glob(f"{ROOT}/{pat}/replay_buffer"))
    lg = sorted(glob.glob(f"logs/tau2bench_retail_rlvr/hf_qwen3_4b_instruct/{logpat}/"),
                key=os.path.getmtime)
    if not ds or not lg:
        return {}
    per = [_json.loads(l) for l in open(lg[-1] + "rollouts_per_task.jsonl")]
    order = [r.get("task_id") for r in per]
    rew_by = {r.get("task_id"): r.get("correct") for r in per}
    d = load_from_disk(ds[0])
    # episodes appear in task order; take the longest full_state per distinct length
    eps = []
    seen = set()
    for row in d:
        ms = row.get("full_state") or []
        if len(ms) < 3:
            continue
        key = (len(ms), (row.get("task_prompt") or "")[:40])
        if key in seen:
            continue
        seen.add(key)
        eps.append(ms)
    return {order[i]: (eps[i], rew_by.get(order[i]))
            for i in range(min(len(eps), len(order)))}


data = {lab: load(pat, pat) for lab, pat, _ in RUNS}
common = set.intersection(*[set(v) for v in data.values() if v]) if all(data.values()) else set()
tasks = sorted(common, key=lambda t: int(t))[:N_TASKS]


def turn_html(m):
    role = m.get("role", "?")
    c = m.get("content") or ""
    if role == "assistant":
        tool = "<tool_call>" in c
        pre = c.split("<tool_call>")[0].strip() if tool else c.strip()
        call = ("<tool_call>" + c.split("<tool_call>", 1)[1]).strip() if tool else ""
        # a turn with NO tool call is a message to the user; respond_user is the other route
        kind = "to-user" if not tool else ("to-user" if "respond_user" in c else "tool")
        out = f'<div class="m a {kind}"><span class="k">assistant · {"→user" if kind=="to-user" else "tool"}</span>'
        if pre:
            out += f'<div class="prose">{html.escape(pre)}</div>'
        if call:
            out += f'<pre>{html.escape(call)}</pre>'
        return out + "</div>"
    # tool responses are the only thing worth trimming (retail order dumps run to
    # thousands of chars); user turns shown in full.
    lim = 700 if role == "tool" else 10**6
    body = html.escape(c[:lim]) + (f"… [{len(c)-lim} more chars]" if len(c) > lim else "")
    return f'<div class="m {role}"><span class="k">{role}</span>{body}</div>'


rows = []
for t in tasks:
    cells = []
    for lab, _, colour in RUNS:
        ms, rew = data[lab][t]
        nuser = sum(1 for m in ms if m.get("role") == "user")
        nasst = sum(1 for m in ms if m.get("role") == "assistant")
        body = "".join(turn_html(m) for m in ms[:MAX_MSG])
        more = ""
        cells.append(
            f'<td><div class="hd" style="border-color:{colour}">{lab}'
            f'<span class="badge {"ok" if (rew or 0)>0 else "no"}">{"solved" if (rew or 0)>0 else "failed"}</span></div>'
            f'<div class="sub">{nasst} assistant turns · <b>{nuser} user turns</b> · {len(ms)} messages total</div>{body}{more}</td>')
    rows.append(f'<tr><td class="task">task_id {html.escape(str(t))}</td></tr><tr>{"".join(cells)}</tr>')

doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Retail rollouts — 4-way</title><style>
:root{{--bg:#fbfbfa;--surface:#fff;--ink:#1b1b1a;--ink2:#4a4a48;--ink3:#77776f;--line:#e4e4e0}}
@media(prefers-color-scheme:dark){{:root{{--bg:#16171a;--surface:#1e2024;--ink:#ececeb;--ink2:#b9b9b5;--ink3:#8b8b85;--line:#31333a}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:2100px;margin:0 auto;padding:22px}}
h1{{font-size:19px;margin:0 0 4px}} .lede{{color:var(--ink3);font-size:12.5px;margin-bottom:16px;max-width:1100px}}
table{{width:100%;border-collapse:separate;border-spacing:10px 0}}
td{{vertical-align:top;width:25%;background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:9px}}
td.task{{width:auto;background:transparent;border:none;font-size:12px;color:var(--ink3);padding:16px 2px 5px}}
.hd{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;font-weight:600;border-left:4px solid;padding-left:8px;margin-bottom:3px}}
.sub{{font-size:11px;color:var(--ink3);margin-bottom:8px;padding-left:12px}}
.badge{{float:right;font-size:10px;font-weight:600;padding:1px 7px;border-radius:20px}}
.badge.ok{{background:#1f6f5c22;color:#1f6f5c}} .badge.no{{background:#9b2c2c22;color:#9b2c2c}}
.m{{margin-bottom:6px;font-size:11.5px}}
.k{{display:block;font-size:9px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink3);margin-bottom:2px}}
.m.user{{background:#0072B214;border-left:3px solid #0072B2;padding:5px 8px;border-radius:0 5px 5px 0}}
.m.tool{{color:var(--ink3);font-size:10.5px}}
.a.to-user{{background:#1f6f5c14;border-left:3px solid #1f6f5c;padding:5px 8px;border-radius:0 5px 5px 0}}
.a.tool{{border-left:3px solid var(--line);padding-left:8px}}
.prose{{color:var(--ink2)}}
pre{{margin:3px 0 0;font-family:ui-monospace,Menlo,monospace;font-size:10px;white-space:pre-wrap;word-break:break-word;
background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:5px}}
.more{{font-size:10.5px;color:var(--ink3);font-style:italic}}
</style></head><body><div class="wrap">
<h1>Retail rollouts — same task, four checkpoints</h1>
<div class="lede">Identical 42-task held-out eval set across all four runs (verified set-equal), identical harness
(max_turns 20, temperature 1.0, hide-observations). <b>Green = a turn that reaches the user</b>
(bare prose, or a <code>respond_user</code> call); grey = a tool call. Episodes are the same length
(~10.5 assistant turns) in every run — what changed is that the OLD checkpoints spend ~4 of those turns
talking to the user and the NEW ones spend ~0, so the user speaks 5.4 times per episode in OLD
<code>actions_only</code> versus 1.5 in NEW.</div>
<table>{"".join(rows)}</table></div></body></html>"""
os.makedirs("/tmp/aprm_plots", exist_ok=True)
open("/tmp/aprm_plots/rollout_compare.html", "w").write(doc)
print(f"wrote /tmp/aprm_plots/rollout_compare.html ({len(doc)//1024} KB, {len(tasks)} shared tasks)")
