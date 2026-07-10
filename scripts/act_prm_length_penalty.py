#!/usr/bin/env python3
r"""Act-PRM with a length-penalized thought reward, on Tinker.

Trains an LLM to infer the latent thoughts behind action-only demonstrations
(Snorkel Agent Finance traces), where each sampled thought z is rewarded by

    r_pen(z) = p(x | s, z) - lambda * (|z| / max_thought_tokens)
               \__________/   \_______________________________/
        length-normalized       length penalty: the fraction of the
        likelihood of the       thought budget the thought consumed
        logged action x

and the EM / policy-gradient weights are the group-normalized, non-negative
part of r_pen. Intuition: the likelihood term lives in (0, 1] and the penalty
in [0, lambda], so with lambda ~= 0.1-0.2 a thought that maxes out the token
budget must buy at least that much extra action-likelihood over a concise
alternative to win the group. Since only *relative* rewards matter after group
normalization, this shifts selection (and gradient mass) toward the shortest
thought that still explains the action.

Usage (any env with: tinker datasets python-dotenv transformers numpy):

  # small real training run
  uv run --with tinker --with datasets --with python-dotenv \
         --with "transformers>=4.51" --with numpy --with jinja2 \
    python scripts/act_prm_length_penalty.py train \
      --model Qwen/Qwen3-8B --group-size 4 --length-penalty 0.15 \
      --num-trajectories 2 --max-steps-per-traj 3 --em-iterations 2

  # demo: sample + score thoughts with a trained sampler (path printed by train,
  # also stored in the run log json)
  uv run ... python scripts/act_prm_length_penalty.py demo \
      --sampler-path "tinker://..." --model Qwen/Qwen3-8B

Expects TINKER_API_KEY in the environment or in a .env next to the repo root.
"""

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

import tinker
from tinker import types
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
DATASET = "mzio/aprm-snorkelai_agent_finance_reasoning"
THOUGHT_BOS, THOUGHT_EOS = "<thought>", "</thought>"
ACT_PRM_SYSTEM_PROMPT = (
    "You are a helpful assistant that infers reasoning thoughts behind your own observed actions."
)

# One compact worked example that seeds the action-then-thought (reversal) format.
FEWSHOT = [
    {"role": "user", "content": (
        "Here is the question : What was the company's total revenue growth in 2024?, "
        "Here are the companies name in the database to query for : acme"
    )},
    {"role": "assistant", "content": (
        '<tool_call>\n{"name": "get_descriptions", "arguments": {"company_name": "acme"}}\n</tool_call>\n\n'
        f"{THOUGHT_BOS}\nI need revenue figures for acme. First I should see which tables exist "
        f"for this company, so I'll list the available table descriptions.\n{THOUGHT_EOS}"
    )},
]
FEWSHOT_TRANSITION = "Great! Now do the same for the next task:\n\n## Next Task:\n\n"


# ---------------------------------------------------------------------------
# data: action-only trajectories from the Snorkel Agent Finance dataset
# ---------------------------------------------------------------------------
def extract_action(content: str) -> str | None:
    """Cut the narration out of an assistant message, keeping only the explicit
    action: the <tool_call>...</tool_call> block (or a Final Answer suffix)."""
    m = re.search(r"<tool_call>.*?</tool_call>", content, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    if "Final Answer:" in content:
        return ("Final Answer:" + content.split("Final Answer:", 1)[1]).strip()
    return None


def load_trajectories(n: int, max_timestep: int):
    """Stream the dataset; keep the first n successful (done, return_>0) rollouts
    that finished within max_timestep steps. Assistant turns are stripped to
    action-only content."""
    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", streaming=True)
    out, seen = [], set()
    for row in ds:
        if not (row["done"] and row["return_"] > 0):
            continue
        if row["max_timestep"] > max_timestep:
            continue
        uid = (row["unique_data_sample_id"], row["generation_id"])
        if uid in seen:
            continue
        seen.add(uid)

        messages = list(row["state"]) + [row["action"]]
        traj, ok = [], True
        for msg in messages:
            if msg["role"] == "assistant":
                action = extract_action(msg["content"])
                if action is None:          # unparseable action -> skip trajectory
                    ok = False
                    break
                traj.append({"role": "assistant", "content": action})
            else:
                traj.append({"role": msg["role"], "content": msg["content"]})
        if ok and sum(m["role"] == "assistant" for m in traj) >= 2:
            out.append({"messages": traj, "system_prompt": row["system_prompt"]})
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)}/{n} usable trajectories")
    return out


def compact_observations(messages, obs_max_chars: int, first_to_show: int = 2,
                         last_to_show: int = 1):
    """Cap observation lengths and hide middle observations (the codebase's
    hide_observations trick) so long tool outputs don't blow up the context."""
    out = [dict(m) for m in messages]
    obs_idx = [i for i, m in enumerate(out) if m["role"] in ("user", "tool")]
    for j, i in enumerate(obs_idx):
        if j >= first_to_show and j < len(obs_idx) - last_to_show:
            out[i]["content"] = "..."
        elif len(out[i]["content"]) > obs_max_chars:
            out[i]["content"] = out[i]["content"][:obs_max_chars] + " ...[truncated]"
    return out


# ---------------------------------------------------------------------------
# tokenization helpers
# ---------------------------------------------------------------------------
class Templater:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)

    def __call__(self, messages, continue_final_message=False, add_generation_prompt=False):
        out = self.tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            enable_thinking=False,
        )
        # newer transformers return a BatchEncoding instead of a token list
        if not isinstance(out, (list, tuple)):
            out = out["input_ids"]
            if out and isinstance(out[0], (list, tuple)):
                out = out[0]
        return list(out)

    def decode(self, tokens):
        return self.tok.decode(tokens, skip_special_tokens=True)


def build_thought_prompt(templater, state_messages, target_action, committed):
    """Reversal prompt: few-shot seed + prior turns as action-then-thought +
    the target action shown first, truncated right after `<thought>\\n`."""
    msgs = [{"role": "system", "content": ACT_PRM_SYSTEM_PROMPT}]
    msgs += [dict(m) for m in FEWSHOT]
    body = [dict(m) for m in state_messages if m["role"] != "system"]
    a_i = 0
    for m in body:
        if m["role"] == "assistant" and a_i < len(committed):
            m["content"] = (f"{m['content']}\n\n{THOUGHT_BOS}\n{committed[a_i]}\n{THOUGHT_EOS}")
            a_i += 1
    body[0]["content"] = FEWSHOT_TRANSITION + body[0]["content"]
    msgs += body
    msgs.append({"role": "assistant", "content": f"{target_action}\n\n{THOUGHT_BOS}\n"})
    return templater(msgs, continue_final_message=True)


# ---------------------------------------------------------------------------
# reward: length-penalized action likelihood
# ---------------------------------------------------------------------------
async def score_thought(sampling_client, templater, scoring_state, thought,
                        target_action, n_thought_tokens, cfg):
    """Returns (penalized_reward, raw_likelihood, len_frac, full_tokens,
    full_logprobs, prefix_len). Scoring uses the natural order — state, thought,
    action — under the task's original system prompt."""
    prefix_msgs = scoring_state + [{"role": "assistant", "content": thought}]
    prefix_tokens = templater(prefix_msgs, continue_final_message=True)
    full_msgs = scoring_state + [
        {"role": "assistant", "content": f"{thought}\n\n{target_action}"}
    ]
    full_tokens = templater(full_msgs)
    n_action = len(full_tokens) - len(prefix_tokens)

    full_logprobs = await sampling_client.compute_logprobs_async(
        types.ModelInput.from_ints(full_tokens)
    )
    action_lp = np.array([lp for lp in full_logprobs[-n_action:]], dtype=np.float64)
    likelihood = float(np.exp(action_lp.mean()))            # in (0, 1]

    len_frac = min(1.0, n_thought_tokens / cfg.max_thought_tokens)
    penalized = likelihood - cfg.length_penalty * len_frac  # may be negative

    prefix_len = len(templater(scoring_state, add_generation_prompt=True))
    return penalized, likelihood, len_frac, full_tokens, full_logprobs, prefix_len


def em_weights(penalized_rewards, likelihoods):
    """Group-normalized EM weights from possibly-negative penalized rewards:
    clamp at 0 and normalize by the sum. If the penalty pushed the whole group
    non-positive (common early in training, when likelihoods are still tiny),
    fall back to normalizing the raw likelihoods so the step still carries
    learning signal — the penalty then only affects *selection*, not weights."""
    clamped = np.maximum(np.array(penalized_rewards, dtype=np.float64), 0.0)
    total = clamped.sum()
    if total <= 0:
        lik = np.array(likelihoods, dtype=np.float64)
        return lik / lik.sum() if lik.sum() > 0 else np.full(len(lik), 1.0 / len(lik))
    return clamped / total


# ---------------------------------------------------------------------------
# E-step over one trajectory
# ---------------------------------------------------------------------------
async def rollout_trajectory(sampling_client, templater, traj, cfg, verbose=True):
    """Sample + score thoughts at each step; commit the best; return the datum
    ingredients for the M-step and per-step metrics."""
    messages = traj["messages"]
    system_prompt = traj["system_prompt"]
    action_indices = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    action_indices = action_indices[: cfg.max_steps_per_traj]

    committed, step_data, metrics = [], [], []
    for t, idx in enumerate(action_indices):
        state = compact_observations(messages[:idx], cfg.obs_max_chars)
        x_t = messages[idx]["content"]

        # --- sample G thoughts (action-prompted reversal) ---
        prompt_tokens = build_thought_prompt(templater, state, x_t, committed)
        res = await sampling_client.sample_async(
            types.ModelInput.from_ints(prompt_tokens),
            num_samples=cfg.group_size,
            sampling_params=types.SamplingParams(
                max_tokens=cfg.max_thought_tokens, temperature=1.0,
                stop=[THOUGHT_EOS],
            ),
        )
        thoughts, thought_lens = [], []
        for seq in res.sequences:
            text = templater.decode(seq.tokens).split(THOUGHT_EOS)[0].strip()
            thoughts.append(text)
            thought_lens.append(len(seq.tokens))

        # --- score: likelihood minus length penalty ---
        scoring_state = ([{"role": "system", "content": system_prompt}]
                         + [m for m in state if m["role"] != "system"])
        scored = await asyncio.gather(*[
            score_thought(sampling_client, templater, scoring_state, z, x_t, n, cfg)
            for z, n in zip(thoughts, thought_lens)
        ])
        penalized = [s[0] for s in scored]
        likelihoods = [s[1] for s in scored]
        weights = em_weights(penalized, likelihoods)
        best = int(np.argmax(penalized))

        committed.append(thoughts[best])
        for (_, _, _, full_tokens, full_logprobs, prefix_len), w in zip(scored, weights):
            step_data.append(dict(full_tokens=full_tokens, full_logprobs=full_logprobs,
                                  prefix_len=prefix_len, weight=float(w)))
        metrics.append(dict(
            step=t,
            likelihoods=likelihoods,
            penalized=penalized,
            weights=weights.tolist(),
            thought_tokens=thought_lens,
            best=best,
            thoughts=thoughts,
        ))
        if verbose:
            print(f"    step {t + 1}/{len(action_indices)}: "
                  f"best p(x|s,z)={likelihoods[best]:.4f}, "
                  f"penalized={penalized[best]:+.4f}, "
                  f"|z| tokens={thought_lens[best]} "
                  f"(group mean |z|={np.mean(thought_lens):.0f})")
    return step_data, metrics


# ---------------------------------------------------------------------------
# M-step
# ---------------------------------------------------------------------------
def make_datum(full_tokens, full_logprobs, prefix_len, weight):
    input_tokens = full_tokens[:-1]
    target_tokens = full_tokens[1:]
    n_prefix = prefix_len - 1                      # shift for next-token targets
    n_gen = len(target_tokens) - n_prefix          # thought + action tokens
    old_logprobs = [0.0] * n_prefix + [float(lp) for lp in full_logprobs[prefix_len:]]
    advantages = [0.0] * n_prefix + [weight] * n_gen
    return types.Datum(
        model_input=types.ModelInput.from_ints(input_tokens),
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_numpy(np.array(target_tokens, dtype=np.int64)),
            "logprobs": types.TensorData.from_numpy(np.array(old_logprobs, dtype=np.float64)),
            "advantages": types.TensorData.from_numpy(np.array(advantages, dtype=np.float64)),
        },
    )


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
async def train(cfg):
    print(f"loading {cfg.num_trajectories} trajectories from {DATASET} ...")
    trajs = load_trajectories(cfg.num_trajectories, cfg.max_traj_timestep)
    for i, tr in enumerate(trajs):
        n_act = sum(m["role"] == "assistant" for m in tr["messages"])
        print(f"  traj {i}: {len(tr['messages'])} messages, {n_act} actions "
              f"(training on first {min(n_act, cfg.max_steps_per_traj)})")

    templater = Templater(cfg.model)
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        cfg.model, rank=cfg.lora_rank
    )
    print(f"training client: {cfg.model} (LoRA r={cfg.lora_rank}), "
          f"lambda_len={cfg.length_penalty}, G={cfg.group_size}")

    log = dict(config=vars(cfg), iterations=[])
    sampler_path = None
    for n in range(cfg.em_iterations):
        print(f"\n=== EM iteration {n} ===")
        t0 = time.time()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        data, iter_metrics = [], []
        for i, traj in enumerate(trajs):
            print(f"  trajectory {i}:")
            step_data, metrics = await rollout_trajectory(
                sampling_client, templater, traj, cfg
            )
            data += [make_datum(**sd) for sd in step_data]
            iter_metrics.append(metrics)

        fb = await training_client.forward_backward_async(data, loss_fn="importance_sampling")
        opt = await training_client.optim_step_async(
            types.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        await fb.result_async()
        await opt.result_async()

        flat = [m for tm in iter_metrics for m in tm]
        mean_best_lik = float(np.mean([m["likelihoods"][m["best"]] for m in flat]))
        mean_lik = float(np.mean([lk for m in flat for lk in m["likelihoods"]]))
        mean_len = float(np.mean([tl for m in flat for tl in m["thought_tokens"]]))
        mean_best_len = float(np.mean([m["thought_tokens"][m["best"]] for m in flat]))
        print(f"  iter {n}: mean p(x|s,z)={mean_lik:.4f} (best-of-G {mean_best_lik:.4f}), "
              f"mean |z|={mean_len:.0f} tokens (selected {mean_best_len:.0f}) "
              f"[{time.time() - t0:.0f}s]")
        log["iterations"].append(dict(
            iteration=n, mean_likelihood=mean_lik, mean_best_likelihood=mean_best_lik,
            mean_thought_tokens=mean_len, mean_selected_thought_tokens=mean_best_len,
            trajectories=iter_metrics,
        ))

    # save final weights for the demo sampler
    resp = await (await training_client.save_weights_for_sampler_async(name="lenpen-final")).result_async()
    sampler_path = resp.path
    log["sampler_path"] = sampler_path
    print(f"\nfinal sampler weights: {sampler_path}")

    out = Path(cfg.log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(log, indent=1))
    print(f"run log -> {out}")

    # small demo with the trained sampler
    print("\n=== demo with trained sampler ===")
    await demo(cfg, sampler_path=sampler_path, trajs=trajs, templater=templater,
               service_client=service_client)


async def demo(cfg, sampler_path=None, trajs=None, templater=None, service_client=None):
    """Sample + score thoughts for the first step of one trajectory, and print
    the group with raw likelihoods, penalties, and the winner."""
    templater = templater or Templater(cfg.model)
    service_client = service_client or tinker.ServiceClient()
    sampler_path = sampler_path or cfg.sampler_path
    if sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=sampler_path)
        print(f"sampler: {sampler_path}")
    else:
        sampling_client = service_client.create_sampling_client(base_model=cfg.model)
        print(f"sampler: base {cfg.model} (no trained weights given)")

    trajs = trajs or load_trajectories(1, cfg.max_traj_timestep)
    traj = trajs[0]
    _, metrics = await rollout_trajectory(
        sampling_client, templater,
        {"messages": traj["messages"], "system_prompt": traj["system_prompt"]},
        argparse.Namespace(**{**vars(cfg), "max_steps_per_traj": 1}),
        verbose=False,
    )
    m = metrics[0]
    print("\nquestion:", traj["messages"][0]["content"][:220])
    print("logged action:", traj["messages"][
        [i for i, mm in enumerate(traj["messages"]) if mm["role"] == "assistant"][0]
    ]["content"][:200], "\n")
    order = np.argsort(m["penalized"])[::-1]
    for rank, g in enumerate(order):
        star = " <== selected" if g == m["best"] else ""
        print(f"[{rank + 1}] p(x|s,z)={m['likelihoods'][g]:.4f}  "
              f"penalized={m['penalized'][g]:+.4f}  "
              f"|z|={m['thought_tokens'][g]} tok  "
              f"weight={m['weights'][g]:.2f}{star}")
        text = m["thoughts"][g].replace("\n", " ")
        print(f"     {text[:300]}{'...' if len(text) > 300 else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["train", "demo"])
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--group-size", type=int, default=4, help="G thoughts per state")
    p.add_argument("--length-penalty", type=float, default=0.15,
                   help="lambda: reward = p(x|s,z) - lambda * |z|/max_thought_tokens")
    p.add_argument("--max-thought-tokens", type=int, default=200)
    p.add_argument("--num-trajectories", type=int, default=2)
    p.add_argument("--max-steps-per-traj", type=int, default=3)
    p.add_argument("--max-traj-timestep", type=int, default=12,
                   help="only use demonstrations that finished within this many steps")
    p.add_argument("--em-iterations", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--obs-max-chars", type=int, default=1500)
    p.add_argument("--log-path", default="runs/length_penalty_run.json")
    p.add_argument("--sampler-path", default=None, help="tinker:// path for demo mode")
    cfg = p.parse_args()

    if cfg.mode == "train":
        asyncio.run(train(cfg))
    else:
        asyncio.run(demo(cfg))


if __name__ == "__main__":
    main()
