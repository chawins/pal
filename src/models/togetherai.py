import copy
import json
import logging
import os

import numpy as np
import requests
import torch
import transformers

from src.message import Message, Role
from src.models.base import BaseModel, LossOutput
from src.models.model_input import ModelInputIds, ModelInputs
from src.servers.request_server import (
    REQUEST_ERRORS,
    Queues,
    standalone_server,
)
from src.utils.suffix import SuffixManager, build_prompt
from src.utils.types import BatchTokenIds

logger = logging.getLogger(__name__)


class Response:
    def __init__(self, raw_response, stream: bool = False) -> None:
        self.response = raw_response
        self.stream = stream
        self.complete = False

        if self.stream:
            raise NotImplementedError(
                "Streaming is not implemented for TogetherAI API!"
            )
            # pylint: disable=unreachable
            self.response_iter = iter(self.response)
        else:
            if raw_response is None:
                self.response = ""
            else:
                self.response = raw_response.json()["choices"][0]["text"]

    def __iter__(self):
        return self

    def __next__(self):
        if self.complete:
            raise StopIteration

        if not self.stream:
            self.complete = True
            return self.response

        try:
            chunk = next(self.response_iter)
            raise NotImplementedError(
                "Streaming is not implemented for TogetherAI API!"
            )
            delta = chunk.choices[0].delta  # pylint: disable=unreachable
            return delta.content
        except StopIteration as e:
            self.complete = True
            raise e


class TogetherAIModel(BaseModel):
    """Model builder for TogetherAI API models.

    Call with a list of `Message` objects to generate a response.
    """

    supports_system_message = True

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        stream: bool = False,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        logit_bias: dict[int, float] | None = None,
        system_message: str | None = None,
        suffix_manager: SuffixManager | None = None,
        template_name: str = "helpful",
        num_api_processes: int = 8,
        **kwargs,
    ) -> None:
        logger.debug("Initializing TogetherAIModel...")
        _ = kwargs  # Unused
        self.model = model.replace("togetherai:", "")
        self.temperature = temperature
        self.stream = stream
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.logit_bias = logit_bias or {}
        self.template = template_name

        # kwargs for all OpenAI requests (not including logprobs, echo, and
        # logit_bias, max_tokens)
        self.request_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": ["</s>"],
            "logprobs": 1,
            "echo": True,
        }
        self.server_kwargs = {
            "display_progress": logging.root.level == logging.DEBUG,
            "num_processes": num_api_processes,
            "max_tokens": 0,
            **self.request_kwargs,
        }
        api_key = os.environ["TOGETHERAI_API_KEY"]
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        self.device = "cpu"
        self.num_fixed_tokens = 0
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True, use_fast=False
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.suffix_manager = suffix_manager
        self.system_message = system_message
        self._messages = None
        self._num_queries = 0
        self._num_tokens = 0
        self._num_input_tokens = 0
        self._num_output_tokens = 0
        self._queues: Queues | None = None

    def set_prefix_cache(self, messages: list[Message]) -> None:
        self._messages = messages

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> torch.Tensor:
        raise NotImplementedError("OpenAI does not support gradients.")

    def compute_suffix_loss(
        self,
        inputs: ModelInputIds | ModelInputs,
        batch_size: int | None = None,
        temperature: float = 1.0,
        max_target_len: int = 32,
        loss_func: str = "ce",
        cw_margin: float = 1e-3,
        **kwargs,
    ) -> LossOutput:
        _ = batch_size, cw_margin, temperature, max_target_len, kwargs  # unused
        if not isinstance(inputs, ModelInputs):
            raise NotImplementedError("Only ModelInputs is supported.")
        messages: list[Message] = inputs.messages
        suffixes: list[str] = inputs.suffixes
        assert len(inputs.targets) == 1, "Only support single target for now."
        assert loss_func == "ce-all", "Only ce-all loss is supported for now."
        target: str = inputs.targets[0]
        self._num_queries, self._num_tokens = 0, 0
        self._num_input_tokens, self._num_output_tokens = 0, 0
        messages = copy.deepcopy(messages)
        messages.append(Message(Role.ASSISTANT, target))
        goal = messages[1].content

        prompts = [""] * len(suffixes)
        for i, suffix in enumerate(suffixes):
            if suffix.startswith(" "):
                messages[1] = Message(Role.USER, f"{goal}{suffix}")
            else:
                messages[1] = Message(Role.USER, f"{goal} {suffix}")
            prompts[i] = build_prompt(messages, template_name=self.template)
        logger.debug("TogetherAI prompts[0]: %s", prompts[0])

        responses, self._queues = standalone_server(
            prompts,
            queues=self._queues,
            cleanup=False,
            url="https://api.together.xyz/v1/completions",
            headers=self.headers,
            **self.server_kwargs,
        )
        losses = np.zeros(len(responses)) + np.infty
        outputs = [""] * len(responses)
        for i, response in enumerate(responses):
            if response is None:
                continue
            try:
                response_json = response.json()
            except json.decoder.JSONDecodeError as e:
                logger.warning("Error when parsing response into json: %s", e)
                continue
            outputs[i] = response_json["choices"][0]["text"]
            tokens: list[str] = response_json["prompt"][0]["logprobs"]["tokens"]
            token_logprobs = response_json["prompt"][0]["logprobs"][
                "token_logprobs"
            ]

            # Find last "INST", "]"
            start_idx = None
            for j in range(len(tokens) - 1, 0, -1):
                if tokens[j] == "INST":
                    start_idx = j
                    break
            start_idx += 2

            loss = 0.0
            will_break = False
            for j in range(start_idx, len(tokens)):
                if will_break:
                    break
                if target.strip().endswith(tokens[j].strip()):
                    will_break = True
                loss -= token_logprobs[j]
            losses[i] = loss
            self._num_input_tokens += response_json["usage"]["prompt_tokens"]
            self._num_output_tokens += response_json["usage"][
                "completion_tokens"
            ]

        self._num_queries += len(prompts)
        self._num_tokens += self._num_input_tokens + self._num_output_tokens
        losses = torch.tensor(losses)
        logger.debug(
            "compute_suffix_loss done. Used queries: %d, tokens: %d",
            self._num_queries,
            self._num_tokens,
        )
        return LossOutput(
            losses=losses,
            texts=outputs,
            num_queries=self._num_queries,
            num_tokens=self._num_tokens,
        )

    def filter_suffixes(
        self,
        suffix_ids: BatchTokenIds | None = None,
        suffix: list[str] | None = None,
        skipped_suffixes: set | None = None,
    ) -> torch.Tensor:
        """Filter suffixes using all models.

        Args:
            suffix_ids: Suffix ids to filter. Defaults to None.
            suffix: Suffix strings to filter. Defaults to None.
            skipped_suffixes: Set of suffixes to skip. Defaults to None.

        Returns:
            Boolean filter of suffixes to keep.
        """
        assert (suffix_ids is not None) ^ (
            suffix is not None
        ), "Either suffix_ids OR suffix must be provided but not both!"
        if suffix is None:
            _, orig_len = suffix_ids.shape
            device = suffix_ids.device
            decoded = self.tokenizer.batch_decode(
                suffix_ids, skip_special_tokens=True
            )
            encoded = self.tokenizer(
                decoded,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)
            # Filter out suffixes that do not tokenize back to the same ids
            filter_cond = torch.all(encoded[:, :orig_len] == suffix_ids, dim=1)
        else:
            device = "cpu"
            encoded = self.tokenizer(
                suffix,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
            ).input_ids.to(device)
            decoded = self.tokenizer.batch_decode(
                encoded, skip_special_tokens=True
            )
            filter_cond = [s == d for s, d in zip(suffix, decoded)]
            filter_cond = torch.tensor(
                filter_cond, device=device, dtype=torch.bool
            )

        if skipped_suffixes is not None:
            # Skip seen/visited suffixes
            is_kept = [suffix not in skipped_suffixes for suffix in decoded]
            is_kept = torch.tensor(is_kept, device=device, dtype=torch.bool)
        else:
            # No skip
            is_kept = torch.ones(len(decoded), device=device, dtype=torch.bool)

        filter_cond &= is_kept
        return filter_cond

    def __call__(
        self,
        messages: list[Message],
        api_key: str | None = None,
        prompt: str | None = None,
        logit_bias: dict[int, float] | None = None,
        echo: bool = False,
        echo_len: int = 0,
        logprobs: int = 0,
    ) -> Response:
        _ = logit_bias, echo, echo_len, api_key, logprobs, prompt
        prompt = build_prompt(messages, template_name=self.template)
        logger.debug("TogetherAI prompt: %s", prompt)
        try:
            response = requests.post(
                "https://api.together.xyz/v1/completions",
                json={
                    "max_tokens": self.max_tokens,
                    "prompt": prompt,
                    **self.request_kwargs,
                },
                headers=self.headers,
                timeout=30,
            )
            response = Response(response)
        except REQUEST_ERRORS as e:
            logger.warning("Error found in requests.post: %s", str(e))
            response = Response(None)
        return response
