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


def load_trajectories_split(split_file: str, dataset: str = DATASET):
    """Load ALL successful trajectories, partitioned by the canonical task split
    (runs/sftrl/split.json). Returns (train_pool, eval_pool); each trajectory dict
    carries its task uid."""
    from datasets import load_dataset

    split = json.loads(Path(split_file).read_text())
    train_uids, eval_uids = set(split["train_uids"]), set(split["eval_uids"])
    want = train_uids | eval_uids
    ds = load_dataset(split.get("dataset", dataset), split="train", streaming=True)
    train_pool, eval_pool, seen = [], [], set()
    for row in ds:
        if not (row["done"] and row["return_"] > 0):
            continue
        uid = row["unique_data_sample_id"]
        if uid not in want or uid in seen:
            continue
        seen.add(uid)
        messages = list(row["state"]) + [row["action"]]
        traj, ok = [], True
        for msg in messages:
            if msg["role"] == "assistant":
                action = extract_action(msg["content"])
                if action is None:
                    ok = False
                    break
                traj.append({"role": "assistant", "content": action})
            else:
                traj.append({"role": msg["role"], "content": msg["content"]})
        if not ok or sum(m["role"] == "assistant" for m in traj) < 2:
            continue
        entry = {"messages": traj,
                 "system_prompt": row.get("system_prompt") or "You are a helpful assistant.",
                 "uid": uid}
        (train_pool if uid in train_uids else eval_pool).append(entry)
    train_pool.sort(key=lambda t: t["uid"])
    eval_pool.sort(key=lambda t: t["uid"])
    return train_pool, eval_pool


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
                        target_action, n_thought_tokens, cfg, base_client=None):
    """Score one thought; returns a dict of reward ingredients. Scoring uses the
    natural order — state, thought, action — under the task's original system
    prompt."""
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
    sum_action_lp = float(action_lp.sum())                  # total log p(x | s, z)

    len_frac = min(1.0, n_thought_tokens / cfg.max_thought_tokens)
    penalized = likelihood - cfg.length_penalty * len_frac  # may be negative

    prefix_len = len(templater(scoring_state, add_generation_prompt=True))

    # per-token KL of the *thought* against the frozen base model (KL anchor)
    kl_mean = 0.0
    if base_client is not None:
        n_thought = max(1, len(prefix_tokens) - prefix_len)
        base_logprobs = await base_client.compute_logprobs_async(
            types.ModelInput.from_ints(full_tokens)
        )
        cur = np.array([lp or 0.0 for lp in full_logprobs[prefix_len: prefix_len + n_thought]])
        base = np.array([lp or 0.0 for lp in base_logprobs[prefix_len: prefix_len + n_thought]])
        kl_mean = float((cur - base).mean())

    return dict(penalized=penalized, likelihood=likelihood, len_frac=len_frac,
                sum_action_lp=sum_action_lp, kl_mean=kl_mean,
                full_tokens=full_tokens, full_logprobs=full_logprobs,
                prefix_len=prefix_len)


async def action_baseline(sampling_client, templater, scoring_state, target_action):
    """Total log p(x | s, ∅): the action's log-likelihood with NO thought."""
    prefix_tokens = templater(scoring_state, add_generation_prompt=True)
    full_tokens = templater(scoring_state + [{"role": "assistant", "content": target_action}])
    n_action = len(full_tokens) - len(prefix_tokens)
    logprobs = await sampling_client.compute_logprobs_async(
        types.ModelInput.from_ints(full_tokens))
    return float(np.array([lp for lp in logprobs[-n_action:]], dtype=np.float64).sum())


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
async def rollout_trajectory(sampling_client, templater, traj, cfg, verbose=True,
                             base_client=None):
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
            text = templater.decode(seq.tokens).split(THOUGHT_EOS)[0]
            # Qwen3's template treats <think> tags specially (it splits the message),
            # which breaks continue_final_message when they appear inside a thought —
            # scrub them, and never let a thought be empty.
            text = text.replace("<think>", "").replace("</think>", "").strip()
            thoughts.append(text if text else "(no thought)")
            thought_lens.append(len(seq.tokens))

        # --- score each candidate ---
        scoring_state = ([{"role": "system", "content": system_prompt}]
                         + [m for m in state if m["role"] != "system"])
        scored = await asyncio.gather(*[
            score_thought(sampling_client, templater, scoring_state, z, x_t, n, cfg,
                          base_client=base_client)
            for z, n in zip(thoughts, thought_lens)
        ])
        likelihoods = [s["likelihood"] for s in scored]
        kls = [s["kl_mean"] for s in scored]

        if cfg.reward_method == "lift":
            # (1) likelihood LIFT over the no-thought baseline, per thought token,
            # (3) minus a small per-token KL to the frozen base model
            baseline = await action_baseline(sampling_client, templater, scoring_state, x_t)
            rewards = [
                (s["sum_action_lp"] - baseline) / (n + cfg.lift_c) - cfg.kl_coef * s["kl_mean"]
                for s, n in zip(scored, thought_lens)
            ]
            # (2) lexicographic selection: shortest among near-best-likelihood candidates
            p_max = max(likelihoods)
            eligible = [g for g in range(len(thoughts))
                        if likelihoods[g] >= p_max * (1 - cfg.select_epsilon)]
            best = min(eligible, key=lambda g: thought_lens[g])
        else:
            rewards = [s["penalized"] for s in scored]
            best = int(np.argmax(rewards))

        weights = em_weights(rewards, likelihoods)

        committed.append(thoughts[best])
        for s, w in zip(scored, weights):
            step_data.append(dict(full_tokens=s["full_tokens"],
                                  full_logprobs=s["full_logprobs"],
                                  prefix_len=s["prefix_len"], weight=float(w)))
        metrics.append(dict(
            step=t,
            likelihoods=likelihoods,
            penalized=rewards,          # the reward actually used, whatever the method
            kl=kls,
            weights=weights.tolist(),
            thought_tokens=thought_lens,
            best=best,
            thoughts=thoughts,
        ))
        if verbose:
            print(f"    step {t + 1}/{len(action_indices)}: "
                  f"best p(x|s,z)={likelihoods[best]:.4f}, "
                  f"reward={rewards[best]:+.4f}, "
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
def summarize(all_metrics):
    """Aggregate a list of per-trajectory metric lists into iteration-level means."""
    flat = [m for tm in all_metrics for m in tm]
    if not flat:
        return {}
    return dict(
        mean_likelihood=float(np.mean([lk for m in flat for lk in m["likelihoods"]])),
        mean_best_likelihood=float(np.mean([m["likelihoods"][m["best"]] for m in flat])),
        mean_penalized=float(np.mean([p for m in flat for p in m["penalized"]])),
        mean_thought_tokens=float(np.mean([tl for m in flat for tl in m["thought_tokens"]])),
        mean_selected_thought_tokens=float(np.mean([m["thought_tokens"][m["best"]] for m in flat])),
        n_steps=len(flat),
    )


async def safe_rollout(sampling_client, templater, traj, cfg, base_client=None):
    """A rollout that degrades gracefully: one bad trajectory (transient API
    error, context overflow) shouldn't kill a long training run."""
    try:
        return await rollout_trajectory(sampling_client, templater, traj, cfg, verbose=False,
                                        base_client=base_client)
    except Exception as e:  # noqa: BLE001 — log and drop this trajectory this iteration
        print(f"  [warn] trajectory rollout failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
        return None


async def probe_generations(sampling_client, templater, traj, cfg, base_client=None):
    """Checkpoint snapshot: sample + score one group for the probe trajectory's
    first step, without training on it."""
    cfg1 = argparse.Namespace(**{**vars(cfg), "max_steps_per_traj": 1})
    _, metrics = await rollout_trajectory(
        sampling_client, templater,
        {"messages": traj["messages"], "system_prompt": traj["system_prompt"]},
        cfg1, verbose=False, base_client=base_client,
    )
    return metrics[0]


async def train(cfg):
    train_pool = None
    if cfg.split_file:
        print(f"loading trajectories via split {cfg.split_file} ...", flush=True)
        train_pool, eval_pool = load_trajectories_split(cfg.split_file)
        eval_trajs = eval_pool[: cfg.eval_trajectories]
        train_trajs = train_pool          # rebatched per iteration below
        print(f"  train pool: {len(train_pool)} tasks | eval pool: {len(eval_pool)} tasks "
              f"(metrics on first {len(eval_trajs)}) | {cfg.tasks_per_iter} tasks/iter", flush=True)
    else:
        total = cfg.num_trajectories + cfg.eval_trajectories
        print(f"loading {total} trajectories from {DATASET} "
              f"({cfg.num_trajectories} train / {cfg.eval_trajectories} held-out eval) ...", flush=True)
        trajs = load_trajectories(total, cfg.max_traj_timestep)
        train_trajs = trajs[: cfg.num_trajectories]
        eval_trajs = trajs[cfg.num_trajectories:]
        for name, group in [("train", train_trajs), ("eval", eval_trajs)]:
            for i, tr in enumerate(group):
                n_act = sum(m["role"] == "assistant" for m in tr["messages"])
                print(f"  {name} traj {i}: {len(tr['messages'])} messages, {n_act} actions "
                      f"(using first {min(n_act, cfg.max_steps_per_traj)})", flush=True)

    templater = Templater(cfg.model)
    service_client = tinker.ServiceClient()
    if cfg.resume_state:
        training_client = await service_client.create_training_client_from_state_async(cfg.resume_state)
        print(f"resumed training client from {cfg.resume_state}", flush=True)
    else:
        training_client = await service_client.create_lora_training_client_async(
            cfg.model, rank=cfg.lora_rank
        )
    base_client = None
    if cfg.reward_method == "lift" and cfg.kl_coef > 0:
        base_client = service_client.create_sampling_client(base_model=cfg.model)
    print(f"training client: {cfg.model} (LoRA r={cfg.lora_rank}), "
          f"reward={cfg.reward_method} "
          f"({'lift_c=' + str(cfg.lift_c) + ', eps=' + str(cfg.select_epsilon) + ', kl=' + str(cfg.kl_coef) if cfg.reward_method == 'lift' else 'lambda_len=' + str(cfg.length_penalty)}), "
          f"G={cfg.group_size}, iterations {cfg.start_iteration}..{cfg.em_iterations}", flush=True)

    out = Path(cfg.log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # resuming appends to an existing log so curves stay continuous across restarts
    if cfg.resume_state and out.exists():
        log = json.loads(out.read_text())
        log["config"] = vars(cfg)
    else:
        log = dict(config=vars(cfg), iterations=[], checkpoints=[])

    def flush_log():
        out.write_text(json.dumps(log, indent=1))

    async def snapshot(n, sampling_client):
        """Persist sampler weights + full trainer state + probe generations."""
        resp = await (await training_client.save_weights_for_sampler_async(
            name=f"lenpen-iter-{n}")).result_async()
        state = await (await training_client.save_state_async(
            name=f"lenpen-state-{n}", overwrite=True)).result_async()
        probe = await probe_generations(sampling_client, templater, eval_trajs[0], cfg,
                                        base_client=base_client)
        log["checkpoints"].append(dict(iteration=n, sampler_path=resp.path,
                                       state_path=state.path, probe=probe))
        flush_log()
        b = probe["best"]
        print(f"[ckpt @ iter {n}] sampler: {resp.path}", flush=True)
        print(f"[ckpt @ iter {n}] probe best-of-G: p(x|s,z)={probe['likelihoods'][b]:.4f}, "
              f"|z|={probe['thought_tokens'][b]} tok\n"
              f"    {probe['thoughts'][b][:220]}", flush=True)
        return resp.path

    sampler_path = None
    best_eval, best_iter, patience_left = float("-inf"), -1, cfg.patience
    if cfg.resume_state and log.get("best"):
        best_eval, best_iter = log["best"]["eval_likelihood"], log["best"]["iteration"]
    for n in range(cfg.start_iteration, cfg.em_iterations):
        t0 = time.time()
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        # periodic checkpoint BEFORE this iteration's update (n=0 == initial policy)
        if n % cfg.checkpoint_every == 0:
            sampler_path = await snapshot(n, sampling_client)

        # E-step: this iteration's train batch in parallel, then held-out eval
        if train_pool is not None:
            k = cfg.tasks_per_iter
            batch = [train_pool[(n * k + j) % len(train_pool)] for j in range(k)]
        else:
            batch = train_trajs
        results = [r for r in await asyncio.gather(*[
            safe_rollout(sampling_client, templater, tr, cfg, base_client) for tr in batch
        ]) if r is not None]
        if not results:
            raise RuntimeError("every trajectory rollout failed this iteration")
        eval_results = [r for r in await asyncio.gather(*[
            safe_rollout(sampling_client, templater, tr, cfg, base_client) for tr in eval_trajs
        ]) if r is not None]

        data = [make_datum(**sd) for step_data, _ in results for sd in step_data]
        train_metrics = [m for _, m in results]
        eval_metrics = [m for _, m in eval_results]

        # M-step: chunked forward/backward (gradients accumulate), one optim step
        fb_futures = []
        for i in range(0, len(data), cfg.fwd_bwd_chunk):
            fb_futures.append(await training_client.forward_backward_async(
                data[i: i + cfg.fwd_bwd_chunk], loss_fn="importance_sampling"))
        opt = await training_client.optim_step_async(
            types.AdamParams(learning_rate=cfg.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        )
        for f in fb_futures:
            await f.result_async()
        await opt.result_async()

        tr_s, ev_s = summarize(train_metrics), summarize(eval_metrics)
        secs = time.time() - t0
        print(f"iter {n:>2}/{cfg.em_iterations}: "
              f"train p={tr_s['mean_likelihood']:.4f} (best {tr_s['mean_best_likelihood']:.4f}) "
              f"|z|={tr_s['mean_thought_tokens']:.0f}/{tr_s['mean_selected_thought_tokens']:.0f}sel"
              f" || eval p={ev_s.get('mean_likelihood', float('nan')):.4f} "
              f"(best {ev_s.get('mean_best_likelihood', float('nan')):.4f}) "
              f"|z|={ev_s.get('mean_thought_tokens', float('nan')):.0f}"
              f" [{secs:.0f}s, {len(data)} datums]", flush=True)
        log["iterations"].append(dict(
            iteration=n, train=tr_s, eval=ev_s, seconds=secs,
            train_metrics=train_metrics, eval_metrics=eval_metrics,
        ))
        flush_log()

        # --- early stopping on held-out likelihood ---
        ev_lik = ev_s.get("mean_likelihood")
        if ev_lik is not None:
            if ev_lik > best_eval + 1e-4:
                best_eval, best_iter = ev_lik, n
                patience_left = cfg.patience
                resp = await (await training_client.save_weights_for_sampler_async(
                    name=f"lenpen-best-{n}")).result_async()
                log["best"] = dict(iteration=n, eval_likelihood=ev_lik, sampler_path=resp.path)
                flush_log()
                print(f"    new best eval p={ev_lik:.4f} -> {resp.path}", flush=True)
            else:
                patience_left -= 1
                degraded = n > cfg.patience and ev_lik < cfg.degrade_factor * best_eval
                if patience_left <= 0 or degraded:
                    why = "hard degradation" if degraded else f"no eval improvement for {cfg.patience} iters"
                    print(f"[early stop @ iter {n}] {why} (best p={best_eval:.4f} @ iter {best_iter})", flush=True)
                    break

    # final checkpoint + probe with the fully-trained weights
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    sampler_path = await snapshot(cfg.em_iterations, sampling_client)
    log["sampler_path"] = sampler_path
    flush_log()
    print(f"\nfinal sampler weights: {sampler_path}")
    print(f"run log -> {out}")

    # small demo with the trained sampler
    print("\n=== demo with trained sampler ===", flush=True)
    await demo(cfg, sampler_path=sampler_path, trajs=eval_trajs, templater=templater,
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
    p.add_argument("--reward-method", choices=["lenpen", "lift"], default="lenpen",
                   help="lenpen: additive length penalty; lift: likelihood lift over the "
                        "no-thought baseline per thought token, lexicographic selection, "
                        "small KL anchor to the base model")
    p.add_argument("--lift-c", type=float, default=15.0,
                   help="lift method: token-count smoothing constant in the denominator")
    p.add_argument("--select-epsilon", type=float, default=0.02,
                   help="lift method: commit the shortest thought within eps of best likelihood")
    p.add_argument("--kl-coef", type=float, default=0.05,
                   help="lift method: per-token KL-to-base-model penalty coefficient")
    p.add_argument("--max-thought-tokens", type=int, default=200)
    p.add_argument("--num-trajectories", type=int, default=2)
    p.add_argument("--eval-trajectories", type=int, default=1,
                   help="held-out trajectories for eval metrics + checkpoint probes")
    p.add_argument("--max-steps-per-traj", type=int, default=3)
    p.add_argument("--max-traj-timestep", type=int, default=12,
                   help="only use demonstrations that finished within this many steps")
    p.add_argument("--em-iterations", type=int, default=2)
    p.add_argument("--checkpoint-every", type=int, default=4,
                   help="save sampler weights + probe generations every N iterations")
    p.add_argument("--patience", type=int, default=12,
                   help="early-stop after this many iterations without eval improvement")
    p.add_argument("--degrade-factor", type=float, default=0.75,
                   help="early-stop immediately if eval likelihood falls below this fraction of best")
    p.add_argument("--fwd-bwd-chunk", type=int, default=128,
                   help="datums per forward_backward call (gradients accumulate)")
    p.add_argument("--learning-rate", type=float, default=4e-5)
    p.add_argument("--obs-max-chars", type=int, default=1500)
    p.add_argument("--log-path", default="runs/length_penalty_run.json")
    p.add_argument("--sampler-path", default=None, help="tinker:// path for demo mode")
    p.add_argument("--split-file", default=None,
                   help="canonical task split json (runs/sftrl/split.json); enables pooled "
                        "training with per-iteration round-robin batching")
    p.add_argument("--tasks-per-iter", type=int, default=16,
                   help="split mode: train tasks rolled out per iteration")
    p.add_argument("--resume-state", default=None,
                   help="tinker:// state path (from a checkpoint's state_path) to resume training")
    p.add_argument("--start-iteration", type=int, default=0,
                   help="iteration number to resume counting from")
    cfg = p.parse_args()

    if cfg.mode == "train":
        asyncio.run(train(cfg))
    else:
        asyncio.run(demo(cfg))


if __name__ == "__main__":
    main()
