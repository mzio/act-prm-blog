"""
Snorkel Agent Insurance Underwriting Environment

Online evaluation environment where the agent queries an insurance underwriting
database via tools and answers questions about commercial small-business insurance.

Supports six benchmark task types:
- appetite_determination: Is this LOB in appetite for the company?
- lob_recommendation: What LOBs should we offer?
- limits_recommendation: What policy limits to recommend?
- small_business_eligibility: Does this company qualify as a small business?
- deductible_recommendation: What deductible to recommend?
- naics_classification: What is this company's NAICS code?

Tools execute real SQL queries against a SQLite database built from
the reference implementation at
https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation
"""

from __future__ import annotations

import json
import logging
from copy import copy
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from pydantic import ConfigDict, InstanceOf

from ...graders.snorkel_insurance import SnorkelInsuranceGrader
from ...llm_handlers import ActionFromLLM
from ..base import BaseTool, Environment
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult
from .backend import DataBackend
from .prompts import SYSTEM_PROMPTS, render_prompt
from .tools import TOOL_CLASSES

logger = logging.getLogger(__name__)


class SnorkelInsuranceState(EnvironmentStateWithAnswer):
    """State for Snorkel Insurance tasks."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_type: str
    company_name: str
    tool_registry: dict[str, InstanceOf[BaseTool]]
    tools: list[dict[str, Any]]
    # backend: InstanceOf[DataBackend]


class SnorkelInsuranceStepResult(EnvironmentStepResult):
    """Step result for Snorkel Insurance tasks."""

    state: SnorkelInsuranceState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class SnorkelInsuranceEnv(Environment):
    """
    Online evaluation environment for Snorkel Agent Insurance Underwriting.

    The agent is given a company's information and an underwriting question,
    then must use tools (get_underwriting_guidelines, list_tables,
    get_table_descriptions, get_table_data_dictionary, get_table_schema,
    read_query, respond_user) to gather data and answer the question.

    Tool responses are computed live against an in-memory SQLite database
    built from parquet resource files.
    """

    def __init__(
        self,
        data_path: str,
        task_data_path: str,
        task: str = "default",
        task_types: list[str] | None = None,
        grader_model_config: dict[str, Any] | None = None,
        grader_model_samples: int = 1,
        grader_model_verbose: bool = False,
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
        self.task_data_path = task_data_path
        self.task = task
        self.task_types = task_types  # Optional filter for specific task types

        # LLM-as-a-judge for grading
        self.grader_model_config = grader_model_config
        self.grader_model_samples = grader_model_samples
        if grader_model_config is not None:
            self.grader_model = SnorkelInsuranceGrader(
                grader_model_config=grader_model_config,
                num_samples=grader_model_samples,
                verbose=grader_model_verbose,
            )
        else:
            # Grader will be set externally (e.g., in tests)
            self.grader_model = None  # type: ignore[assignment]

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
            template = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["default"])
            self.system_prompt = template.format(max_turns=max_turns)
        self.truncation_message = truncation_message

        # Load tasks from JSON
        self.datasets = self.init_data()

        # Initialize data backend (SQLite from parquet files)
        self.backend = DataBackend(data_path)

        # Initialize tools
        self.tool_registry: dict[str, BaseTool] = {name: cls() for name, cls in TOOL_CLASSES.items()}
        self.tool_descriptions = [tool.get_tool_desc() for tool in self.tool_registry.values()]

    def __len__(self) -> int:
        return len(self.datasets[self.split])

    def init_data(self) -> DatasetDict:
        """Load tasks from the task data JSON and split into train/eval/test."""
        with open(self.task_data_path) as f:
            tasks = json.load(f)

        # Filter by task types if specified
        if self.task_types is not None:
            tasks = [t for t in tasks if t.get("root_task") in self.task_types]

        records = []
        for task_entry in tasks:
            user_info = task_entry.get("user_information", {})
            task_type = str(task_entry.get("root_task", ""))
            user_task = str(task_entry.get("user_task", ""))
            task_id = task_entry.get("company_task_id", 0)

            company_name = str(user_info.get("company_name", ""))
            company_description = str(user_info.get("company_description", ""))
            naics_code = str(user_info.get("naics_code", ""))
            annual_revenue = str(user_info.get("annual_revenue", ""))
            number_of_employees = str(user_info.get("number_of_employees", ""))
            state = str(user_info.get("state", ""))
            lob = str(user_info.get("lob", ""))
            total_payroll = str(user_info.get("total_payroll", ""))
            number_of_vehicles = str(user_info.get("number_of_vehicles", ""))
            building_construction = str(user_info.get("building_construction", ""))
            total_insured_value_property = str(user_info.get("total_insured_value_property", ""))

            # Build reference answer from user_information fields
            reference_answer = self._build_reference_answer(task_type, user_info)

            prompt = render_prompt(
                user_task=user_task,
                company_name=company_name,
                company_description=company_description,
                naics_code=naics_code,
                annual_revenue=annual_revenue,
                number_of_employees=number_of_employees,
                state=state,
                lob=lob,
                total_payroll=total_payroll,
                number_of_vehicles=number_of_vehicles,
                building_construction=building_construction,
                total_insured_value_property=total_insured_value_property,
            )

            records.append(
                {
                    "query_id": task_id,
                    "question": user_task,
                    "answer": reference_answer,
                    "task_type": task_type,
                    "company_name": company_name,
                    "prompt": prompt,
                    "user_information": json.dumps(user_info),
                }
            )

        ds = Dataset.from_list(records)
        return self._get_splits(ds)

    def _build_reference_answer(self, task_type: str, user_info: dict[str, Any]) -> str:
        """Build a programmatic reference answer from the task's ground-truth fields."""
        if task_type == "appetite_determination":
            appetite = user_info.get("appetite", "Unknown")
            lob = user_info.get("lob", "")
            return f"The appetite for {lob} is: {appetite}."
        elif task_type == "lob_recommendation":
            # LOBs available: property, general liability, commercial auto,
            # workers compensation, cyber liability, business owners policy
            lob = user_info.get("lob", "")
            appetite = user_info.get("appetite", "")
            return f"Recommended LOBs based on company profile. Primary LOB ({lob}): appetite is {appetite}."
        elif task_type == "limits_recommendation":
            limits_per = user_info.get("policy_limits_per_incident_in_millions", "")
            limits_agg = user_info.get("policy_limits_aggregate_in_millions", "")
            lob = user_info.get("lob", "")
            return f"Recommended policy limits for {lob}: ${limits_per}M per occurrence / ${limits_agg}M aggregate."
        elif task_type == "deductible_recommendation":
            deductible = user_info.get("deductible", "")
            lob = user_info.get("lob", "")
            return f"Recommended deductible for {lob}: ${deductible}."
        elif task_type == "small_business_eligibility":
            is_small = user_info.get("small_business", "")
            annual_revenue = user_info.get("annual_revenue", "")
            number_of_employees = user_info.get("number_of_employees", "")
            return (
                f"Small business eligibility: {is_small}. Revenue: {annual_revenue}, Employees: {number_of_employees}."
            )
        elif task_type == "naics_classification":
            naics_code = user_info.get("naics_code", "")
            return f"NAICS code: {naics_code}."
        else:
            return json.dumps(user_info)

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
            n_test = min(n_test, n - 2)
            n_val = min(n_val, n - n_test - 1)
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
    ) -> SnorkelInsuranceState:
        """Reset environment with a new insurance underwriting question."""
        sample_id_adj = self.adjust_sample_id(sample_id)
        sample = self.datasets[self.split][sample_id_adj]

        messages = [{"role": "user", "content": sample["prompt"]}]

        return SnorkelInsuranceState(
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
            task_type=str(sample["task_type"]),
            company_name=str(sample["company_name"]),
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=0,
            metadata={"correct": 0, "total": 1},
            first_obs_to_show=len(messages) + 1,
        )

    def step(self, **kwargs: Any) -> SnorkelInsuranceStepResult:
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: SnorkelInsuranceState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SnorkelInsuranceStepResult:
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
                    reward = reward * 2.0 - 1.0  # convert to [-1, 1] scale
                    continue

                # Execute other tool calls via backend
                try:
                    tool = current_state.tool_registry[fc_name]
                    # Tools that take no args besides backend
                    no_arg_tools = {
                        "get_underwriting_guidelines",
                        "get_table_descriptions",
                        "list_tables",
                    }
                    if fc_name in no_arg_tools:
                        result = tool(backend=backend)
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
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    and "FINAL ANSWER" in text.upper()
                ):
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
                    reward = reward * 2.0 - 1.0  # convert to [-1, 1] scale

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
        new_state = SnorkelInsuranceState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tool_registry=current_state.tool_registry,
            tools=available_tools,
            # backend=backend,
            question=question,
            answer=answer,
            task_type=current_state.task_type,
            company_name=current_state.company_name,
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return SnorkelInsuranceStepResult(
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
        current_state: SnorkelInsuranceState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> SnorkelInsuranceStepResult:
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
                    reward = reward * 2.0 - 1.0
                    continue

                try:
                    tool = current_state.tool_registry[fc_name]
                    no_arg_tools = {
                        "get_underwriting_guidelines",
                        "get_table_descriptions",
                        "list_tables",
                    }
                    if fc_name in no_arg_tools:
                        result = tool(backend=backend)
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
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    and "FINAL ANSWER" in text.upper()
                ):
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
                    reward = reward * 2.0 - 1.0

        timestep += 1
        if timestep >= self.max_turns and not done:
            truncated = True
            done = True
            reward = -1.0
            env_messages.append(
                {
                    "role": "user",
                    "content": self.truncation_message,
                }
            )
            if not updated_try_step:
                try_step += 1
                updated_try_step = True

        if len(env_messages) == 0:
            env_messages.append(
                {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": None,
                    "output": (
                        "Error: No tool calls or final answers were parsed. "
                        "Please call a tool or use respond_user to answer."
                    ),
                }
            )

        current_messages = self.maybe_hide_observations(
            current_messages or [],
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )

        metadata.update({"reward": reward, "done": done, "truncated": truncated})
        new_state = SnorkelInsuranceState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages,
            tool_registry=current_state.tool_registry,
            tools=available_tools,
            # backend=backend,
            question=question,
            answer=answer,
            task_type=current_state.task_type,
            company_name=current_state.company_name,
            sample_id=current_state.sample_id,
            generation_id=current_state.generation_id,
            batch_id=current_state.batch_id,
            try_step=try_step,
            timestep=timestep,
            metadata=metadata,
            first_obs_to_show=current_state.first_obs_to_show,
            last_obs_to_show=current_state.last_obs_to_show,
        )
        return SnorkelInsuranceStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,
        )


class AsyncSnorkelInsuranceEnv(SnorkelInsuranceEnv):
    """Asynchronous Snorkel Insurance environment."""

    async def reset_async(
        self,
        sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        batch_id: int = 0,
        **kwargs: Any,
    ) -> SnorkelInsuranceState:
        return super().reset(
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            **kwargs,
        )

    async def step_async(self, **kwargs: Any) -> SnorkelInsuranceStepResult:
        return await self._step_impl_async(**kwargs)
