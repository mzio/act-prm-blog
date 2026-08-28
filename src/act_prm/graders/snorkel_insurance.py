"""
LLM-based grader for Snorkel Insurance Underwriting tasks.

Uses LLM-as-judge to evaluate whether the agent's final answer matches
the reference answer for insurance underwriting questions.

Reference: https://github.com/snorkel-ai/multi-turn-insurance-underwriting-benchmark-generation
"""

import logging
import re

from .qa import LLMGraderForQA

logger = logging.getLogger(__name__)

# System prompt for insurance underwriting correctness evaluation
SNORKEL_INSURANCE_GRADER_SYSTEM_PROMPT = """\
I am going to give you:
- question: An insurance underwriting question
- model response: The AI assistant's answer
- label: The correct reference answer

Your task is to determine if the model response is correct by comparing it to the label.

The model response may contain additional rationale or explanation beyond the core answer.
Focus on whether the key factual content matches:
- For appetite questions: Does the response correctly identify Yes/No/Qualified?
- For LOB recommendations: Does the response recommend the correct lines of business?
- For limits/deductible questions: Does the response give the correct amounts?
- For small business eligibility: Does the response correctly identify eligibility?
- For NAICS classification: Does the response give the correct NAICS code?

Extra rationale or explanation is acceptable as long as the core answer is correct.
Missing key information from the label means the response is incorrect.

Provide your judgement in the following format:
correct: Answer 'yes' if the model response matches the label, 'no' otherwise.
rationale: Brief explanation of why they match or don't match."""

SNORKEL_INSURANCE_GRADER_USER_TEMPLATE = """\
Question: {question}

Model Response: {response}

Label: {correct_answer}"""


class SnorkelInsuranceGrader(LLMGraderForQA):
    """Grader for Snorkel Insurance Underwriting using LLM-as-judge.

    Evaluates whether the agent's answer matches the programmatic reference
    answer derived from the task's ground-truth fields.

    Inherits majority voting and metrics tracking from LLMGraderForQA.
    Overrides grade_sample to use insurance-specific evaluation prompt.
    """

    def grade_sample(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        user_prompt = SNORKEL_INSURANCE_GRADER_USER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": user_prompt}]
        sampler_response = self.grader_model.sample(
            system_prompt=SNORKEL_INSURANCE_GRADER_SYSTEM_PROMPT,
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
        _clean = re.sub(r"[*_`~]", "", grading_response)
        match = re.search(r"correct:\s*(yes|no)\b", _clean, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"
        return match, grading_response.strip()

    async def grade_sample_async(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[str, str]:
        user_prompt = SNORKEL_INSURANCE_GRADER_USER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        ).strip()
        prompt_messages = [{"role": "user", "content": user_prompt}]
        sampler_response = (
            await self.grader_model.sample_async(
                system_prompt=SNORKEL_INSURANCE_GRADER_SYSTEM_PROMPT,
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
        _clean = re.sub(r"[*_`~]", "", grading_response)
        match = re.search(r"correct:\s*(yes|no)\b", _clean, flags=re.IGNORECASE)
        match = match.group(1).lower() if match else "no"
        return match, grading_response.strip()
