import copy
import json
import logging
from typing import Literal

import cohere
import numpy as np
import torch

from src.message import Message, Role
from src.models.base import BaseModel, Encoded, LossOutput
from src.models.model_input import ModelInputIds, ModelInputs
from src.servers.cohere_server import Queues, standalone_server
from src.utils.cohere_non_ascii import COHERE_NON_ASCII
from src.utils.suffix import SuffixManager, build_prompt
from src.utils.types import BatchTokenIds

logger = logging.getLogger(__name__)


class Response:
    def __init__(
        self, raw_response: cohere.Generation, stream: bool = False
    ) -> None:
        self.response = raw_response
        self.stream = stream
        self.complete = False
        if self.stream:
            self.response_iter = iter(self.response)
        else:
            self.response = raw_response.generations[0].text

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
            delta = chunk.choices[0].delta
            # return delta.get("content", "")
            return delta.content
        except StopIteration as e:
            self.complete = True
            raise e


class CohereTokenizer:
    non_ascii = COHERE_NON_ASCII

    def __init__(self, model_name: str):
        self._co = cohere.Client()
        # Assume command-light model
        self._model_name = model_name
        # 0: '<PAD>'
        # 1: '<UNK>'
        # 2: '<CLS>'
        # 3: '<SEP>'
        # 4: '<MASK_TOKEN>'
        # 5: '<BOS_TOKEN>'
        # 6: '<EOS_TOKEN>'
        # 7: '<EOP_TOKEN>'
        # Set interface to match HuggingFace
        self.vocab_size = 75500
        self.bos_token_id = 5
        self.eos_token_id = 6
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.eot_token = "<EOP_TOKEN>"  # Cohere uses EOP token instead of EOT

    def __len__(self):
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ):
        _ = kwargs  # unused
        if text is None:
            return Encoded([])

        def tokenize(t):
            try:
                ids = self._co.tokenize(text=t, model=self._model_name).tokens
            except json.decoder.JSONDecodeError as e:
                logger.warning("Error found in cohere.tokenize: %s", str(e))
                return []
            return ids

        if isinstance(text, list):
            # Encode all special tokens as normal text
            _ids = [tokenize(t) for t in text]
            max_len = max(len(i) for i in _ids)
            input_ids = np.zeros((len(_ids), max_len), dtype=np.int64)
            input_ids += self.pad_token_id
            for i, _id in enumerate(_ids):
                input_ids[i, : len(_id)] = _id
        else:
            input_ids = tokenize(text)
            input_ids = np.array(input_ids, dtype=np.int64)

        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
        elif return_tensors == "list":
            input_ids = input_ids.tolist()
        return Encoded(input_ids)

    def _parse_ids(self, ids):
        return ids

    def _detokenize(self, tokens: list[int]) -> str:
        try:
            text = self._co.detokenize(
                tokens=tokens, model=self._model_name
            ).text
        except json.decoder.JSONDecodeError as e:
            logger.warning("Error found in cohere.detokenize: %s", str(e))
            return ""
        return text

    def decode(self, ids, **kwargs) -> str:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        assert isinstance(ids, list) and isinstance(
            ids[0], int
        ), f"ids must be list or int, got {type(ids)} {ids}"
        decoded = self._detokenize(ids)
        return decoded

    def batch_decode(self, ids, **kwargs) -> list[str]:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [[ids]]
        if isinstance(ids, list) and isinstance(ids[0], int):
            ids = [ids]
        assert (
            isinstance(ids, list)
            and isinstance(ids[0], list)
            and isinstance(ids[0][0], int)
        ), f"ids must be list of list of int, got {type(ids)} {ids}"
        decoded_list = [self._detokenize(i) for i in ids]
        return decoded_list


class CohereModel(BaseModel):
    """Model builder for OpenAI API models.

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
        logger.debug("Initializing CohereModel...")
        _ = kwargs  # Unused
        self.model = model
        self.temperature = temperature
        self.stream = stream
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self._bias = 20  # NOTE: any harm setting it to 100?
        self.logit_bias = logit_bias or {}
        self.template = template_name
        self.client = cohere.Client()

        # kwargs for all OpenAI requests (not including logprobs, echo, and
        # logit_bias, max_tokens)
        self.request_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "p": self.top_p,
            "stop_sequences": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "return_likelihoods": "ALL",
            "raw_prompting": True,
            "seed": 0,  # Chat API is non-deterministic even with temperature 0
        }
        self.server_kwargs = {
            "display_progress": logging.root.level == logging.DEBUG,
            "num_processes": num_api_processes,
            "max_tokens": 0,
            **self.request_kwargs,
        }

        self.device = "cpu"
        self.num_fixed_tokens = 0
        self.tokenizer = CohereTokenizer(model)
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
        logger.debug("Cohere prompts[0]: %s", prompts[0])

        responses, self._queues = standalone_server(
            prompts,
            queues=self._queues,
            cleanup=False,
            **self.server_kwargs,
        )
        losses = np.zeros(len(responses)) - np.infty
        outputs = [""] * len(responses)
        for i, response in enumerate(responses):
            outputs[i] = response.generations[0].text
            token_likelihoods = response.generations[0].token_likelihoods
            is_target = False
            loss = 0.0
            for single_gen_token in token_likelihoods:
                if single_gen_token.token == "<EOP_TOKEN>":
                    is_target = True
                    continue
                if is_target:
                    loss -= single_gen_token.likelihood
            losses[i] = loss
            self._num_input_tokens += response.meta.billed_units.input_tokens
            self._num_output_tokens += response.meta.billed_units.output_tokens

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
        logger.debug("Cohere prompt: %s", prompt)
        response = self.client.generate(
            prompt=prompt,
            max_tokens=self.max_tokens,
            **self.request_kwargs,
        )
        response = Response(response)
        return response
