#!/usr/bin/env python3
"""Emit the Result-2 thinking-trace HTML table (step / observation /
ground-truth action / generated thought) for the first held-out task at the
final iteration, with tool calls and dict-like observations pretty-printed and
verbose values truncated.

Usage:
  uv run --with datasets --with numpy --with "transformers>=4.51" \
         --with python-dotenv --with tinker --with jinja2 \
    python scripts/make_trace_table.py [runs/length_penalty_qwen3_8b_100.json]
"""
import ast
import html
import importlib.util
import json
import re
import sys
from pathlib import Path

LOG = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/length_penalty_qwen3_8b_100.json")
STR_TRUNC = 60          # max chars for any string value inside pretty-printed JSON
LIST_TRUNC = 6          # max list items shown
TOTAL_TRUNC = 700       # max chars for a rendered observation block

log = json.loads(LOG.read_text())
cfg = log["config"]

spec = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "act_prm_length_penalty.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)
trajs = lp.load_trajectories(cfg["num_trajectories"] + cfg["eval_trajectories"],
                             cfg["max_traj_timestep"])
traj = trajs[cfg["num_trajectories"]]           # first held-out task
final = log["iterations"][-1]
final_iter = final["iteration"]


def squeeze(value, depth=0):
    """Recursively truncate verbose values for display."""
    if isinstance(value, str):
        return value if len(value) <= STR_TRUNC else value[: STR_TRUNC - 1] + "…"
    if isinstance(value, dict):
        return {k: squeeze(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        out = [squeeze(v, depth + 1) for v in value[:LIST_TRUNC]]
        if len(value) > LIST_TRUNC:
            out.append(f"… (+{len(value) - LIST_TRUNC} more)")
        return out
    return value


def parse_structured(text):
    t = text.strip()
    if not t or t[0] not in "{[":
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(t)
        except Exception:
            continue
    return None


def pretty(text):
    """Pretty-print JSON-ish text with truncated values; fall back to plain trunc."""
    obj = parse_structured(text)
    if obj is None:
        s = " ".join(text.split())
        return None, (s if len(s) <= TOTAL_TRUNC else s[: TOTAL_TRUNC - 2] + " …")
    rendered = json.dumps(squeeze(obj), indent=2, ensure_ascii=False)
    if len(rendered) > TOTAL_TRUNC:
        rendered = rendered[: TOTAL_TRUNC - 4] + "\n(…)"
    return rendered, None


def pretty_action(action):
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", action, flags=re.DOTALL)
    if not m:
        return " ".join(action.split())[:TOTAL_TRUNC]
    rendered, plain = pretty(m.group(1))
    return rendered if rendered is not None else plain


msgs = traj["messages"]
a_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][: cfg["max_steps_per_traj"]]

rows = []
for s_i, idx in enumerate(a_idx):
    obs_text = msgs[idx - 1]["content"] if idx > 0 else msgs[0]["content"]
    if s_i == 0:
        obs_cell = html.escape(" ".join(obs_text.split())[:TOTAL_TRUNC])
    else:
        rendered, plain = pretty(obs_text)
        obs_cell = (f"<pre>{html.escape(rendered)}</pre>" if rendered is not None
                    else html.escape(plain))
    action_cell = f"<pre>{html.escape(pretty_action(msgs[idx]['content']))}</pre>"
    m = final["eval_metrics"][0][s_i]
    thought = m["thoughts"][m["best"]]
    rows.append(
        "        <tr>\n"
        f"          <td>{s_i + 1}</td>\n"
        f"          <td>{obs_cell}</td>\n"
        f"          <td>{action_cell}</td>\n"
        f"          <td>{html.escape(' '.join(thought.split()))}</td>\n"
        "        </tr>")

print(f"""  <figure>
    <div class="table-wrap">
      <table class="results wrap">
        <thead>
          <tr><th>Step</th><th>Observation</th><th>Ground-truth action</th><th>Generated thought (iteration {final_iter})</th></tr>
        </thead>
        <tbody>
{chr(10).join(rows)}
        </tbody>
      </table>
    </div>
    <figcaption><span class="figlabel">A relabelled held-out trajectory.</span> The meta
    lease-financing task from the eval set: only the observations and tool calls existed in the
    action-only log — every thought was <em>inferred</em> by the final Act-PRM checkpoint
    (iteration {final_iter}, λ=0.15 run). Long observation and argument values are truncated (…).</figcaption>
  </figure>""")
