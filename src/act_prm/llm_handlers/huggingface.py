"""
HuggingFace LLM class
"""

import logging
from copy import copy, deepcopy
from typing import Any

# import ast
# import json
# from json import JSONDecodeError
from rich import print as rich_print
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TextStreamer,
)

from .action_utils import get_actions, get_messages_from_text
from .base import LLM
from .types import ActionFromLLM

logger = logging.getLogger(__name__)


def load_hf_model_and_tokenizer(
    pretrained_model_name_or_path: str,
    chat_template_path: str | None = None,
    **kwargs: Any,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load model and tokenizer from Hugging Face Hub, accomodating for custom chat temmplates
    """
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)

    if pretrained_model_name_or_path == "Qwen/Qwen3-8B":  # hack but get Qwen2.5 tokenizer
        pretrained_model_name_or_path = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

    # Override chat template if provided
    if chat_template_path is not None:
        with open(chat_template_path, "r", encoding="utf-8") as f:
            tokenizer.chat_template = f.read()
            logger.info("-> Overriding chat template with %s", chat_template_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer


class HuggingFaceLLM(LLM):
    """
    HuggingFace LLM class
    """

    def __init__(
        self,
        model: Any | None = None,
        model_config: dict[str, Any] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        stream_generation: bool = False,
        # Maybe unnecessary, but specify how tool calls appear
        tool_call_bos: str = "<tool_call>",
        tool_call_eos: str = "</tool_call>",
        tool_call_argname: str = "arguments",  # e.g., llama3 uses "parameters"
        **kwargs: Any,
    ) -> None:
        # Load model and tokenizer from Hugging Face Hub
        if model is None or not isinstance(model, PreTrainedModel):
            assert model_config is not None, "model_config must be provided if model is not provided"
            model, tokenizer = load_hf_model_and_tokenizer(**model_config)
        else:
            assert tokenizer is not None, "tokenizer must be provided if model is provided"

        super().__init__(model=model, model_config=model_config, **kwargs)
        self.tokenizer = tokenizer

        # Parse tool calls
        self.tool_call_bos = tool_call_bos
        self.tool_call_eos = tool_call_eos
        self.tool_call_argname = tool_call_argname
        self.tool_call_parse_kwargs: dict[str, str] = {
            "tool_call_bos": self.tool_call_bos,
            "tool_call_eos": self.tool_call_eos,
            "tool_call_argname": self.tool_call_argname,
        }
        # Stream tokens as they generate
        if stream_generation:
            self.streamer = TextStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
        else:
            self.streamer = None

    def sample(
        self,
        system_prompt: str | None = None,
        messages: list[str] | list[list[dict[str, Any]]] | None = None,
        tools: list[dict[str, Any]] | list[list[dict[str, Any]]] | None = None,
        max_new_tokens: int = 1024,
        num_return_sequences: int = 1,
        verbose: bool = False,
        **generation_kwargs: Any,
    ) -> list[list[dict[str, Any]]]:
        """
        Generate text from a prompt
        """
        if isinstance(messages, list) and not isinstance(messages[0], list):
            messages = [messages]
        elif messages is None:
            messages = [[{"role": "user", "content": ""}]]

        if system_prompt is not None:
            messages = [[{"role": "system", "content": system_prompt}] + single_chat for single_chat in messages]
        # Get model inputs
        if (isinstance(tools, list) and not isinstance(tools[0], list)) or tools is None:
            model_inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                # enable_thinking=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            og_padding_side = copy(self.tokenizer.padding_side)
            self.tokenizer.padding_side = "left"
            # Apply chat template to each sample in batch, as tools may be different per sample
            input_texts = [
                self.tokenizer.apply_chat_template(
                    messages[_idx],
                    tools=_tools,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=False,
                )
                for _idx, _tools in enumerate(tools)
            ]
            # for _idx, _input_text in enumerate(input_texts):
            #     rich_print(f"[{_idx}]\n---\n{_input_text}\n============================\n")
            model_inputs = self.tokenizer(input_texts, return_tensors="pt", padding=True)
            self.tokenizer.padding_side = og_padding_side

        # Get input lengths
        input_len = model_inputs["input_ids"].shape[1]
        # Get generation config
        generation_config = generation_kwargs if generation_kwargs is not None else self.generation_config
        if generation_config.get("pad_token_id", None) is None:
            generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # Generate and decode
        outputs = self.model.generate(
            **model_inputs.to(self.model.device),
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            streamer=self.streamer,  # For silly visuals
            **generation_config,
        )
        # Track tokens hack
        outputs.prompt_tokens = input_len
        outputs.completion_tokens = outputs.shape[1] - input_len
        # # Alternatively, we want to ignore left-pads? (but what about right ones?)
        # input_lens = model_inputs["attention_mask"].sum(dim=1)  # (batch_size,)
        # outputs.prompt_tokens = input_lens.sum()
        # outputs.completion_tokens = outputs.shape[0] * outputs.shape[1] - outputs.prompt_tokens
        self._track_tokens(outputs)

        # Decode and convert tokens to messages
        decoded_texts = self.tokenizer.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # For now we only allow one tool call per response
        decoded_texts = [
            # text.split("</tool_call>")[0] + "</tool_call>"
            # if "<tool_call>" in text
            f"{text.split(self.tool_call_eos)[0]}{self.tool_call_eos}" if self.tool_call_eos in text else text
            for text in decoded_texts
        ]
        if verbose:
            for _text in decoded_texts:
                rich_print(f"{_text}\n{'-' * 100}")
        return [[{"role": "assistant", "content": message}] for message in decoded_texts]

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: list[dict[str, Any]] | None = None,
        prior_messages: list[dict[str, Any]] | None = None,
        interleave: bool = False,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Update the messages with the model response
        """
        if system_prompt is None:
            system_messages: list[dict[str, Any]] = []
        else:
            system_messages = [{"role": "system", "content": system_prompt}]

        if prior_messages is None:
            prior_messages: list[dict[str, Any]] = []

        # We build up new messages, then prepend
        # system prompt (optional) and prior messages
        if interleave or model_response is None:
            new_messages: list[dict[str, Any]] = []
        else:
            new_messages = deepcopy(model_response)

        for idx, message in enumerate(messages):
            if interleave and model_response is not None:
                new_messages.append(model_response[idx])
            if message.get("type", None) == "function_call_output":
                new_messages.append({"role": "tool", "content": message["output"]})
            else:
                new_messages.append(message)
        return system_messages + prior_messages + new_messages

    def get_actions(self, response: list[dict[str, Any]]) -> list[ActionFromLLM]:
        """
        Process response from HuggingFace LLM
        """
        return get_actions(response, **self.tool_call_parse_kwargs)

    def get_messages_from_text(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """
        Convert text to LLM chat messages
        """
        return get_messages_from_text(text, **self.tool_call_parse_kwargs)
