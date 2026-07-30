"""
LLM-based grader for Snorkel Finance QA tasks.

Matches the grading logic from FinQABenchmark/src/llmj.py:
- System prompt: correctness evaluation instructions
- User prompt: question, model response, label
- Decimal matching rules specific to financial QA

Reference: https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/llmj.py
"""

import logging
import re

from .qa import LLMGraderForQA

logger = logging.getLogger(__name__)

# System prompt for correctness evaluation.
# From: https://github.com/snorkel-ai/FinQABenchmark/blob/main/src/llmj.py
SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT = """\
I am going to give you
- question
- model response
- label

Your task is to tell me if you think the model response matches the label.

The model response might be a larger piece of text, and the label might be in a different format, e.g.

- question : What is the age of dog Kamikaze?
- model response : The age of dog Kamikaze is 52
- label : \\boxed{{52}}

Your response : {{True, 'Yes they both match'}}

It's allowed to have model response as a fraction. You can compute the fraction and check if the prediction matches the fraction within two decimal places.

If model response OR label exceeds more than two decimal places, you will compare only uptill the first two decimal places for the model answer and the label, without rounding off.

Provide your judgement in the following format:
correct: Answer 'yes' if the model response matches the label, 'no' otherwise.
rationale: Brief explanation of why they match or don't match."""

# User prompt template (matches reference llmj.py get_correctness())
SNORKEL_FINANCE_GRADER_USER_TEMPLATE = """\
Question: {question}

Model Response: {response}

Label: {correct_answer}"""


class SnorkelFinanceGrader(LLMGraderForQA):
    """Grader for Snorkel Finance QA using the reference FinQABenchmark prompt.

    Follows the same system/user message structure as llmj.py:
    - System: correctness evaluation instructions with decimal matching rules
    - User: question, model response, and ground truth label

    Inherits majority voting, metrics tracking from LLMGraderForQA.
    Overrides grade_sample to use the Snorkel Finance-specific prompt.
    """

    def grade_sample(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        user_prompt = SNORKEL_FINANCE_GRADER_USER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": user_prompt}]
        sampler_response = self.grader_model.sample(
            system_prompt=SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT,
            messages=prompt_messages,
            tools=None,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=1,
        )[0]
        actions = self.grader_model.get_actions(sampler_response)
        if not actions:
            logger.warning("Grader returned no actions (model request may have failed)")
            return "no", "Grader error: no response from model"
        grading_response = actions[-1].text or ""
        match = re.search(r"correct:\s*(yes|no)\b", grading_response, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"
        return match, grading_response.strip()

    async def grade_sample_async(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        user_prompt = SNORKEL_FINANCE_GRADER_USER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": user_prompt}]
        sampler_response = (
            await self.grader_model.sample_async(
                system_prompt=SNORKEL_FINANCE_GRADER_SYSTEM_PROMPT,
                messages=prompt_messages,
                tools=None,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
            )
        )[0]
        actions = self.grader_model.get_actions(sampler_response)
        if not actions:
            logger.warning("Grader returned no actions (model request may have failed)")
            return "no", "Grader error: no response from model"
        grading_response = actions[-1].text or ""
        match = re.search(r"correct:\s*(yes|no)\b", grading_response, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"
        return match, grading_response.strip()
