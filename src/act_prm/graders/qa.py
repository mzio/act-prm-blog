"""
LLM-based grader for QA tasks

Copied and modified from BrowseComp Eval:
https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py
"""

import re
from typing import Any

import numpy as np
from rich import print as rich_print

from ..llm_handlers import LLM, load_llm

# From: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.
```

If the extracted_final_answer is incorrect, provide a detailed explanation of why it is incorrect.
"""

CHOICE_STRINGS = ["yes", "no"]


class LLMGraderForQA:
    """
    LLM Grader for QA tasks
    - Copied and modified from BrowseComp Eval LLM Grader
    """

    def __init__(
        self,
        grader_model: LLM | None = None,
        grader_model_config: dict[str, Any] | None = None,
        max_new_tokens: int = 4096,
        num_samples: int = 1,
        grader_template: str | None = None,
        verbose: bool = False,
    ):
        if grader_model_config is not None:
            # Load LLM from config -> see load_llm in ../lm_handlers/__init__.py
            grader_model = load_llm(**grader_model_config)
        else:
            assert grader_model is not None, "grader_model or grader_model_config must be provided"
        self.grader_model = grader_model
        self.max_new_tokens = max_new_tokens
        self.num_samples = num_samples
        self.grader_template = grader_template if grader_template is not None else GRADER_TEMPLATE
        self.verbose = verbose

        # Track metrics over an evaluation or training batch
        self.metrics = {
            "correct": [],
            "sample_id": [],
            "generation_id": [],
            "split": [],
        }
        self.running_metrics = {}

    def grade_sample(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        """
        Grade a sample response to a question
        """
        grader_prompt = self.grader_template.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": grader_prompt}]
        sampler_response = self.grader_model.sample(
            system_prompt="",
            messages=prompt_messages,
            tools=None,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
            # temperature=0.0,
        )[0]
        grading_response = self.grader_model.get_actions(sampler_response)[-1].text
        match = re.search(r"correct:\s*(yes|no)\b", grading_response, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"  # -> 'yes' or 'no'
        return match, grading_response.strip()

    def __call__(
        self,
        question: str,
        correct_answer: str,
        response: str,
        sample_id: int = 0,
        generation_id: int = 0,
        split: str = "train",
    ) -> tuple[bool, str]:
        """
        Call grader on a single sample
        """
        # Do majority voting
        n_corrects = []
        n_messages = [[], []]  # incorrect and correct
        for _idx in range(self.num_samples):
            grade_result, grade_msg = self.grade_sample(question, correct_answer, response)
            # Metrics based on grading response
            is_correct = grade_result == "yes"
            n_corrects.append(is_correct)
            n_messages[is_correct].append(grade_msg)
            if self.verbose:
                rich_print(f"[green]is_correct vote {_idx + 1}: {is_correct}[/green]")
        is_correct = sum(n_corrects) / len(n_corrects) > 0.5
        grade_msg = n_messages[is_correct][0]  # just pick the first message
        rich_print(f"[bold green]Final Grade Result: is_correct is {is_correct}[/bold green]")
        rich_print(f"[yellow]Reasoning:\n{grade_msg}[/yellow]")

        # Track metrics over an evaluation or training batch
        self.metrics["correct"].append(is_correct)
        self.metrics["sample_id"].append(sample_id)
        self.metrics["generation_id"].append(generation_id)
        self.metrics["split"].append(split)

        for split in np.unique(self.metrics["split"]):
            _mask = np.array(self.metrics["split"]) == split
            _correct = np.array(self.metrics["correct"])[_mask].sum()
            _total = np.sum(_mask)
            _acc = _correct / _total * 100
            self.running_metrics[f"{split}/acc"] = _acc
            self.running_metrics[f"{split}/correct"] = _correct
            self.running_metrics[f"{split}/total"] = _total

        if self.verbose:
            for name, value in self.running_metrics.items():
                _prefix = "%" if name.endswith("/acc") else ""
                print(f"running {name}: {value:.2f}{_prefix}")

        return is_correct, grade_msg

    async def call_async(
        self,
        question: str,
        correct_answer: str,
        response: str,
        sample_id: int = 0,
        generation_id: int = 0,
        split: str = "train",
    ) -> tuple[bool, str]:
        """Async version of __call__ with majority voting."""
        n_corrects = []
        n_messages: list[list[str]] = [[], []]
        for _idx in range(self.num_samples):
            grade_result, grade_msg = await self.grade_sample_async(question, correct_answer, response)
            is_correct = grade_result == "yes"
            n_corrects.append(is_correct)
            n_messages[is_correct].append(grade_msg)
            if self.verbose:
                rich_print(f"[green]is_correct vote {_idx + 1}: {is_correct}[/green]")
        is_correct = sum(n_corrects) / len(n_corrects) > 0.5
        grade_msg = n_messages[is_correct][0] if n_messages[is_correct] else ""
        rich_print(f"[bold green]Final Grade Result: is_correct is {is_correct}[/bold green]")
        rich_print(f"[yellow]Reasoning:\n{grade_msg}[/yellow]")

        self.metrics["correct"].append(is_correct)
        self.metrics["sample_id"].append(sample_id)
        self.metrics["generation_id"].append(generation_id)
        self.metrics["split"].append(split)

        for _split in np.unique(self.metrics["split"]):
            _mask = np.array(self.metrics["split"]) == _split
            _correct = np.array(self.metrics["correct"])[_mask].sum()
            _total = np.sum(_mask)
            self.running_metrics[f"{_split}/acc"] = _correct / _total * 100
            self.running_metrics[f"{_split}/correct"] = _correct
            self.running_metrics[f"{_split}/total"] = _total

        return is_correct, grade_msg

    def reset_metrics(self) -> None:
        """
        Reset metrics, e.g., for new evaluation run
        """
        self.metrics = {
            "correct": [],
            "sample_id": [],
            "generation_id": [],
            "split": [],
        }
        self.running_metrics = {}
