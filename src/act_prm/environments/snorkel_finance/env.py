"""
Snorkel Agent Finance Reasoning Environment

Online evaluation environment where the agent queries financial data via tools
and answers questions about company 10-K filings.

Supports two benchmark tasks:
- finqa: Quantitative financial QA (290 questions, boxed numeric answers)
- finqa_reasoning: Qualitative financial reasoning (79 questions, paragraph answers)

Tools execute real SQL queries against local JSON table data, matching the
reference implementation at https://github.com/snorkel-ai/FinQABenchmark
"""

from __future__ import annotations

import logging
from copy import copy
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from pydantic import ConfigDict, InstanceOf

from ...graders.snorkel_finance import SnorkelFinanceGrader
from ...llm_handlers import ActionFromLLM
from ..base import BaseTool, Environment
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult
from .backend import DataBackend
from .prompts import SYSTEM_PROMPTS, render_prompt
from .tools import (
    TOOL_CLASSES,
)

logger = logging.getLogger(__name__)


class SnorkelFinanceState(EnvironmentStateWithAnswer):
    """State for Snorkel Finance tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    company: str
    tool_registry: dict[str, InstanceOf[BaseTool]]
    tools: list[dict[str, Any]]
    # backend: InstanceOf[DataBackend]


class SnorkelFinanceStepResult(EnvironmentStepResult):
    """Step result for Snorkel Finance tasks."""

    state: SnorkelFinanceState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class SnorkelFinanceEnv(Environment):
    """
    Online evaluation environment for Snorkel Agent Finance Reasoning.

    The agent is given a financial question about a company's 10-K filing and
    must use tools (get_descriptions, get_table_info, sql_query, calculator,
    respond_user) to gather data and answer the question.

    Supports two task types:
    - "finqa": 290 quantitative questions with boxed numeric answers
    - "finqa_reasoning": 79 qualitative reasoning questions with paragraph answers

    Tool responses are computed live against local financial table data
    via in-memory SQLite queries.
    """

    def __init__(
        self,
        data_path: str,
        benchmark_csv: str,
        task: str = "finqa_reasoning",
        grader_model_config: dict[str, Any] | None = None,
        grader_model_samples: int = 1,
        grader_model_verbose: bool = False,
        num_fewshot_prompts: int = 0,
        num_train_samples: int | None = None,
        num_val_samples: int | None = None,
        num_test_samples: int | None = None,
        frac_train: float = 0.6,
        frac_val: float = 0.2,
        frac_test: float = 0.2,
        eval_all_samples: bool = False,
        max_turns: int = 50,
        num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str | None = None,
        truncation_message: str = "Sorry, you have reached the maximum number of steps. Please try again.",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.benchmark_csv = benchmark_csv
        self.task = task

        self.num_fewshot_prompts = num_fewshot_prompts

        # LLM-as-a-judge for grading
        self.grader_model_config = grader_model_config
        self.grader_model_samples = grader_model_samples
        self.grader_model = SnorkelFinanceGrader(
            grader_model_config=grader_model_config,
            num_samples=grader_model_samples,
            verbose=grader_model_verbose,
        )

        # Split config
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.frac_train = frac_train
        self.frac_val = frac_val
        self.frac_test = frac_test
        # When True, every split (train/eval/test) holds the FULL dataset so an
        # eval run can score all samples (treat all samples as the eval set).
        self.eval_all_samples = eval_all_samples
        self.split = split

        self.max_turns = max_turns
        self.num_tries = num_tries
        self.seed = seed

        # System prompt: explicit > task default
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            template = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["finqa_reasoning"])
            self.system_prompt = template.format(max_turns=max_turns)
        self.truncation_message = truncation_message

        # Load questions from benchmark CSV
        self.datasets = self.init_data()

        # Initialize real data backend for tools
        self.backend = DataBackend(data_path)

        # Initialize tools
        self.tool_registry: dict[str, BaseTool] = {name: cls() for name, cls in TOOL_CLASSES.items()}
        self.tool_descriptions = [tool.get_tool_desc() for tool in self.tool_registry.values()]

    def __len__(self) -> int:
        return len(self.datasets[self.split])

    def init_data(self) -> DatasetDict:
        """Load questions from the benchmark CSV and split into train/eval/test."""
        df = pd.read_csv(self.benchmark_csv)

        # Normalize columns across finqa vs finqa_reasoning schemas
        records = []
        for _, row in df.iterrows():
            company = str(row.get("company", ""))
            question = str(row.get("question", ""))
            answer = str(row.get("answer", ""))
            query_id = int(row.get("id", 0))

            # Build user_query if not present
            user_query = row.get("user_query", None)
            if pd.isna(user_query) or user_query is None:
                user_query = question

            records.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "answer": answer,
                    "company": company,
                    "prompt": render_prompt(
                        user_query=user_query,
                        company=company,
                    ),
                }
            )

        ds = Dataset.from_list(records)
        return self._get_splits(ds)

    def _get_splits(self, dataset: Dataset) -> DatasetDict:
        """Split dataset into train/eval/test."""
        if self.eval_all_samples:
            # Evaluate over the entire dataset: all splits = full set.
            return DatasetDict({"train": dataset, "eval": dataset, "test": dataset})

        n = len(dataset)

        if self.num_test_samples is not None:
            n_test = self.num_test_samples
        else:
            n_test = max(1, int(n * self.frac_test))

        if self.num_val_samples is not None:
            n_val = self.num_val_samples
        else:
            n_val = max(1, int(n * self.frac_val))

        if self.num_train_samples is not None:
            n_train = self.num_train_samples
        else:
            n_train = n - n_test - n_val

        # Ensure splits don't exceed dataset size
        total_requested = n_train + n_val + n_test
        if total_requested > n:
            n_test = min(n_test, n - 2)  # leave at least 2 for train+val
            n_val = min(n_val, n - n_test - 1)  # leave at least 1 for train
            n_train = n - n_test - n_val

        trainval_test = dataset.train_test_split(
            test_size=n_test,
            shuffle=True,
            seed=self.seed,
        )
        train_val = trainval_test["train"].train_test_split(
            test_size=n_val,
            shuffle=True,
            seed=self.seed,
        )
        return DatasetDict(
            {
                "train": train_val["train"],
                "eval": train_val["test"],
                "test": trainval_test["test"],
            }
        )

    def shuffle(self, seed: int | None = None, split: str | None = None) -> None:
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        split = split or self.split
        indices = np.arange(len(self.datasets[split]))
        np.random.shuffle(indices)
        indices = indices.tolist()
        self.datasets[split] = self.datasets[split].select(indices)

    def reset(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
    ) -> SnorkelFinanceState:
        """Reset environment with a new financial question."""
        sample_id_adj = self.adjust_sample_id(sample_id)
        sample = self.datasets[self.split][sample_id_adj]

        messages = [{"role": "user", "content": sample["prompt"]}]

        return SnorkelFinanceState(
            system_prompt=self.system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tool_registry=self.tool_registry,
            tools=self.tool_descriptions,
            # backend=self.backend,
            # QA-specific fields
            question=str(sample["question"]),
            answer=str(sample["answer"]),
            company=str(sample["company"]),
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=0,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=len(messages) + 1,
        )

    def step(self, **kwargs: Any) -> SnorkelFinanceStepResult:
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: SnorkelFinanceState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SnorkelFinanceStepResult:
        """Process agent actions (tool calls and messages)."""
        question = str(current_state.question)
        answer = str(current_state.answer)
        backend = self.backend

        done = False
        truncated = False
        reward = 0.0
        updated_try_step = False

        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)

        env_messages = []
        available_tools = copy(current_state.tools)

        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments or {}

                # Check if this is the final answer tool
                if fc_name == "respond_user":
                    response_text = fc_args.get("text", "")
                    reward, grader_text = self.grader_model(
                        question=question,
                        correct_answer=answer,
                        response=response_text,
                        sample_id=current_state.sample_id,
                        generation_id=current_state.generation_id,
                        split=self.split,
                    )
                    reward = float(reward)
                    done = True
                    result_label = "CORRECT" if reward == 1 else "INCORRECT"
                    env_messages.append(
                        {
                            "role": "user",
                            "content": f"# RESULT: {result_label}!",
                        }
                    )
                    metadata["correct"] = reward
                    reward = (reward * 2.0 - 1.0) if self.negative_rewards else reward  # [-1,1] vs RLVR +1/0
                    continue

                # Execute other tool calls
                try:
                    tool = current_state.tool_registry[fc_name]
                    if fc_name == "calculator":
                        result = tool(
                            expression=fc_args.get("expression", ""),
                        )
                    else:
                        result = tool(**fc_args, backend=backend)
                except Exception as e:
                    _error_class = type(e).__name__
                    result = f"Tool call error:\n\n{action.text}\n\n{_error_class}: {e}"
                    logger.warning(f"Error during tool call: {_error_class}: {e}")

                env_messages.append(
                    {
                        "role": "tool",
                        "type": "function_call_output",
                        "call_id": action.call_id,
                        "output": result,
                    }
                )

            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if action.type == "message" and action_idx + 1 == len(parsed_actions) and "Final Answer:" in text:
                    # Final answer submitted as text message
                    reward, grader_text = self.grader_model(
                        question=question,
                        correct_answer=answer,
                        response=text,
                        sample_id=current_state.sample_id,
                        generation_id=current_state.generation_id,
                        split=self.split,
                    )
                    reward = float(reward)
                    done = True
                    result_label = "CORRECT" if reward == 1 else "INCORRECT"
                    env_messages.append(
                        {
                            "role": "user",
                            "content": f"# RESULT: {result_label}!",
                        }
                    )
                    metadata["correct"] = reward
                    reward = (reward * 2.0 - 1.0) if self.negative_rewards else reward  # [-1,1] vs RLVR +1/0

        # Update timesteps
        timestep += 1
        if timestep >= self.max_turns and not done:
            truncated = True
            done = True
            env_messages.append(
                {
                    "role": "user",
                    "content": self.truncation_message,
                }
            )
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        # Handle no response
        if len(env_messages) == 0:
            env_messages.append(
                {
                    # "role": "user",
                    # "content": (
                    #     "No tool calls or final answers were parsed. "
                    #     "Please call a tool or use respond_user to answer."
                    # ),
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": None,
                    "output": (
                        "Error: No tool calls or final answers were parsed. "
                        "Please call a tool or use respond_user to answer."
                    ),
                }
            )

        # Handle past observations
        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata.update({"reward": reward, "done": done, "truncated": truncated})
        new_state = SnorkelFinanceState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tool_registry=current_state.tool_registry,
            tools=available_tools,
            # backend=backend,
            question=question,
            answer=answer,
            company=current_state.company,
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return SnorkelFinanceStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )

    async def _step_impl_async(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: SnorkelFinanceState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SnorkelFinanceStepResult:
        """Async version of _step_impl - uses async grading."""
        question = str(current_state.question)
        answer = str(current_state.answer)
        backend = self.backend

        done = False
        truncated = False
        reward = 0.0
        updated_try_step = False

        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)

        env_messages = []
        available_tools = copy(current_state.tools)

        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments or {}

                # Check if this is the final answer tool
                if fc_name == "respond_user":
                    response_text = fc_args.get("text", "")
                    reward, grader_text = await self.grader_model.call_async(
                        question=question,
                        correct_answer=answer,
                        response=response_text,
                        sample_id=current_state.sample_id,
                        generation_id=current_state.generation_id,
                        split=self.split,
                    )
                    reward = float(reward)
                    done = True
                    result_label = "CORRECT" if reward == 1 else "INCORRECT"
                    env_messages.append(
                        {
                            "role": "user",
                            "content": f"# RESULT: {result_label}!",
                        }
                    )
                    metadata["correct"] = reward
                    reward = (reward * 2.0 - 1.0) if self.negative_rewards else reward  # [-1,1] vs RLVR +1/0
                    continue

                # Execute other tool calls
                try:
                    tool = current_state.tool_registry[fc_name]
                    if fc_name == "calculator":
                        result = tool(
                            expression=fc_args.get("expression", ""),
                        )
                    else:
                        result = tool(**fc_args, backend=backend)
                except Exception as e:
                    _error_class = type(e).__name__
                    result = f"Tool call error:\n\n{action.text}\n\n{_error_class}: {e}"
                    logger.warning(f"Error during tool call: {_error_class}: {e}")

                env_messages.append(
                    {
                        "role": "tool",
                        "type": "function_call_output",
                        "call_id": action.call_id,
                        "output": result,
                    }
                )

            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if action.type == "message" and action_idx + 1 == len(parsed_actions) and "Final Answer:" in text:
                    # Final answer submitted as text message
                    reward, grader_text = await self.grader_model.call_async(
                        question=question,
                        correct_answer=answer,
                        response=text,
                        sample_id=current_state.sample_id,
                        generation_id=current_state.generation_id,
                        split=self.split,
                    )
                    reward = float(reward)
                    done = True
                    result_label = "CORRECT" if reward == 1 else "INCORRECT"
                    env_messages.append(
                        {
                            "role": "user",
                            "content": f"# RESULT: {result_label}!",
                        }
                    )
                    metadata["correct"] = reward
                    reward = (reward * 2.0 - 1.0) if self.negative_rewards else reward  # [-1,1] vs RLVR +1/0

        # Update timesteps
        timestep += 1
        if timestep >= self.max_turns and not done:
            truncated = True
            done = True
            reward = -1.0 if self.negative_rewards else 0.0  # RLVR: truncation = failure = 0
            env_messages.append(
                {
                    "role": "user",
                    "content": self.truncation_message,
                }
            )
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        # Handle no response
        if len(env_messages) == 0:
            env_messages.append(
                {
                    # "role": "user",
                    # "content": (
                    #     "No tool calls or final answers were parsed. "
                    #     "Please call a tool or use respond_user to answer."
                    # ),
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": None,
                    "output": (
                        "Error: No tool calls or final answers were parsed. "
                        "Please call a tool or use respond_user to answer."
                    ),
                }
            )

        # Handle past observations
        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata.update({"reward": reward, "done": done, "truncated": truncated})
        new_state = SnorkelFinanceState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tool_registry=current_state.tool_registry,
            tools=available_tools,
            # backend=backend,
            question=question,
            answer=answer,
            company=current_state.company,
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return SnorkelFinanceStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )


class AsyncSnorkelFinanceEnv(SnorkelFinanceEnv):
    """Asynchronous Snorkel Finance environment."""

    async def reset_async(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
        **kwargs: Any,
    ) -> SnorkelFinanceState:
        return super().reset(
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> SnorkelFinanceStepResult:
        return await self._step_impl_async(**kwargs)
