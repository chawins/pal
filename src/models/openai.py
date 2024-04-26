import ast
import copy
import dataclasses
import logging
import os
import time
from typing import Literal

import numpy as np
import openai
import tiktoken
import torch
from openai import OpenAI
from openai.types import Completion
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from tenacity import retry, retry_if_not_exception_type, wait_random_exponential

from src.message import Message, Role
from src.models.base import BaseModel, Encoded, LossOutput
from src.models.model_input import ModelInputIds, ModelInputs
from src.servers.openai_server import Queues, standalone_server
from src.utils.suffix import SuffixManager, build_prompt
from src.utils.types import BatchTokenIds, TokenIds

OPENAI_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-instruct-0914",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-1106-preview",  # GPT-4 Turbo
    "gpt-4-vision-preview",  # GPT-4 Turbo w/ vision
]
OPENAI_DEFAULT = "gpt-3.5-turbo-0613"

COMPLETION_MODELS = [
    "gpt-3.5-turbo-instruct-0914",  # Cannot use echo and logprobs at the same time
    "gpt-3.5-turbo-instruct",  # Cannot use echo and logprobs at the same time
    # "gpt-4-1106-preview",  # No
    # "gpt-3.5-turbo-1106",  # No
    "text-davinci-003",  # Ok to use both echo and logprobs at the same time
]

logger = logging.getLogger(__name__)


class GptTokenizer:
    def __init__(self, encoding: tiktoken.Encoding):
        # Get the tokeniser corresponding to a specific model in the OpenAI API
        self.encoding: tiktoken.Encoding = encoding
        # Set interface to match HuggingFace
        self.vocab_size = self.encoding.max_token_value + 1
        self.bos_token_id = self.encoding.eot_token
        self.eos_token_id = self.encoding.eot_token
        self.pad_token_id = self.encoding.eot_token
        self.unk_token_id = self.encoding.eot_token
        self.eot_token = "<|endoftext|>"

    def __len__(self):
        return self.vocab_size

    def __call__(
        self,
        text: str,
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ):
        _ = kwargs  # unused
        if text is None:
            return Encoded([])

        if isinstance(text, list):
            # Encode all special tokens as normal text
            _ids = self.encoding.encode_batch(text, disallowed_special=())
            max_len = max(len(i) for i in _ids)
            input_ids = np.zeros((len(_ids), max_len), dtype=np.int64)
            input_ids += self.pad_token_id
            for i, _id in enumerate(_ids):
                input_ids[i, : len(_id)] = _id
        else:
            input_ids = self.encoding.encode(text, disallowed_special=())
            input_ids = np.array(input_ids, dtype=np.int64)

        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
        elif return_tensors == "list":
            input_ids = input_ids.tolist()
        return Encoded(input_ids)

    def _parse_ids(self, ids):
        return ids

    def decode(self, ids, **kwargs) -> str:
        _ = kwargs  # unused
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        assert isinstance(ids, list) and isinstance(
            ids[0], int
        ), f"ids must be list or int, got {type(ids)} {ids}"
        decoded = self.encoding.decode(ids)
        return decoded.replace(self.eot_token, "")

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
        decoded_list = self.encoding.decode_batch(ids)
        decoded_list = [s.replace(self.eot_token, "") for s in decoded_list]
        return decoded_list


class Response:
    """Response wrapper class for OpenAI API response object.

    Implements the iterator interface to enable simple iteration over streamed
    response chunks, such that `"".join(response)` returns the full completion.
    """

    def __init__(self, response, stream=False):
        self.response = response
        self.stream = stream
        self.complete = False
        if self.stream:
            self.response_iter = iter(self.response)
        else:
            self.response = response.choices[0].message.content
            self.finish_reason = response.choices[0].finish_reason

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


class CompletionResponse:
    """Response wrapper class for OpenAI API completion response object.

    Implements the iterator interface to enable simple iteration over streamed
    response chunks, such that `"".join(response)` returns the full completion.
    """

    def __init__(self, response, stream: bool = False, echo_len: int = 0):
        self.echo_len = echo_len
        self.response = response
        self.stream = stream
        self.complete = False
        if self.stream:
            self.response_iter = iter(self.response)
        else:
            logprobs = response.choices[0].logprobs
            self.logprobs = logprobs.token_logprobs[echo_len:]
            self.response = self._decode(logprobs.tokens[echo_len:])
            text = response.choices[0].text
            all_response = self._decode(logprobs.tokens)
            assert text == all_response, (
                "Response text does not match logprobs tokens.\n"
                f"{text}\n\n{all_response}"
            )
            if logprobs.top_logprobs is None:
                self.top_tokens = None
            else:
                top_logprobs = logprobs.top_logprobs[echo_len:]
                top_tokens = [max(lp, key=lp.get) for lp in top_logprobs]
                self.top_tokens = self._decode(top_tokens)

    def _decode(self, tokens):
        """Decode tokens from logprobs into a string."""
        # tokens may contain split encoded bytes, e.g. ['bytes:\xe2\x80',
        # 'bytes:\x99'].
        new_tokens, cur_encoding = [], []
        for t in tokens:
            if t.startswith("bytes:"):
                cur_encoding.append(t[6:])
            else:
                if cur_encoding:
                    # Add current running encoding to new tokens
                    encoded = "".join(cur_encoding)
                    try:
                        decoded = ast.literal_eval(f"b'{encoded}'").decode()
                    except UnicodeDecodeError:
                        new_tokens.append(t)
                        break
                    new_tokens.append(decoded)
                    cur_encoding = []
                new_tokens.append(t)
        output = "".join(new_tokens)
        return output

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
            delta = chunk["choices"][0]["text"]
            return delta.get("content", "")
        except StopIteration as e:
            self.complete = True
            raise e


class OpenAIModel(BaseModel):
    """Model builder for OpenAI API models.

    Call with a list of `Message` objects to generate a response.
    """

    supports_system_message = True
    valid_score_modes = ("echo-logprobs", "logprobs", "hard")

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
    ):
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
        self.client = OpenAI()

        if self.model in COMPLETION_MODELS:
            logger.info("Using Completion API with %s model.", self.model)
            self.api_type = "completion"
            self.supports_system_message = False
            self._score_mode = "logprobs"
        else:
            logger.info("Using Chat API with %s model.", self.model)
            self.api_type = "chat"
            self._score_mode = "logprobs"
        # Echo mode is only available for text-davinci-003
        self._echo = self.model == "text-davinci-003"
        if self._echo:
            logger.warning("echo mode is used. This should not happen.")
            self._score_mode = "echo-logprobs"

        # kwargs for all OpenAI requests (not including logprobs, echo, and
        # logit_bias, max_tokens)
        self.request_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "stream": self.stream,
            "stop": self.stop,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "seed": 0,  # Chat API is non-deterministic even with temperature 0
        }
        server_kwargs = {
            "display_progress": logging.root.level == logging.DEBUG,
            "query_type": self.api_type,
            "max_tokens": 1,
            "num_processes": num_api_processes,
            **self.request_kwargs,
        }
        if self.api_type == "completion":
            server_kwargs["logprobs"] = 5
        else:
            server_kwargs["logprobs"] = True
            server_kwargs["top_logprobs"] = 5
        self.server_kwargs = server_kwargs

        self.device = "cpu"
        self.num_fixed_tokens = 0
        # self._batch_prefix_cache: dict[int, PrefixCache] = {}
        # self.prefix_cache: PrefixCache | None = None
        # self.default_eval_input: EvalInput | None = None
        self.tokenizer = GptTokenizer(tiktoken.encoding_for_model(self.model))
        self.suffix_manager = suffix_manager
        self.system_message = system_message
        self._messages = None
        self._num_queries = 0
        self._num_tokens = 0
        self._queues: Queues | None = None

    def set_prefix_cache(self, messages: list[Message]) -> None:
        self._messages = messages

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> torch.Tensor:
        raise NotImplementedError("OpenAI does not support gradients.")

    def _build_target_prefixes(
        self, target_ids: TokenIds, max_target_len: int
    ) -> tuple[list[str], list[str]]:
        target_prefixes: list[str] = ["" for _ in range(max_target_len + 1)]
        target_toks: list[str] = ["" for _ in range(max_target_len)]
        for i in range(max_target_len):
            target_prefixes[i + 1] = self.tokenizer.decode(
                target_ids[: i + 1], skip_special_tokens=True
            )
            # Get only the last token
            target_toks[i] = target_prefixes[i + 1].replace(
                target_prefixes[i], ""
            )
        target_prefixes = target_prefixes[:-1]
        logger.debug(
            "target_prefixes=%s, target_toks=%s",
            target_prefixes,
            target_toks,
        )
        # All target prefixes must start with a space except for the first one
        if self.api_type == "completion":
            # TODO(multitarget): remove/change this check
            assert all(
                p.startswith(" ") for p in target_prefixes[1:]
            ), target_prefixes
        return target_prefixes, target_toks

    def _get_target_logprob_from_2nd_response(
        self,
        response: Completion,
        target_tok: str,
        top_tok: str,
        top_logp: float,
        system_fingerprint: str,
    ) -> float:
        assert response is not None, "Response is None."
        assert top_tok != target_tok, (
            f"top_tok ({top_tok}) cannot be the same as target_tok "
            f"({target_tok}) because this function is called only when "
            "target_tok is not in top-k of the first call."
        )
        parsed_resp = _parse_response(response, self.api_type)
        biased_full_logprobs_i = parsed_resp.full_logprobs[0]
        biased_target_logp = biased_full_logprobs_i.get(target_tok)

        if biased_target_logp is None:
            logger.warning(
                "Target token ('%s') not in top-5. Have to increase logit bias "
                "(current: %f).",
                target_tok,
                self._bias,
            )
            # Skipping this candidate
            return -np.inf

        biased_top_logp = biased_full_logprobs_i.get(top_tok)
        if biased_top_logp is None:
            # Try getting top token with space difference
            biased_top_logp = biased_full_logprobs_i.get(f" {top_tok}")
        if biased_top_logp is None:
            # In some rare cases, the top token is no longer in top-5
            # maybe because of randomness in the API.
            logger.warning(
                'Top token ("%s") is somehow no longer in top-5 after '
                "applying logit bias: %s",
                top_tok,
                biased_full_logprobs_i,
            )
            logger.warning(
                "Old vs new system fingerprint: %s vs %s",
                system_fingerprint,
                parsed_resp.system_fingerprint,
            )
            logger.warning(
                "Setting new_top_logp to its upper bound (new_target_logp)."
            )
            biased_top_logp = biased_target_logp

        target_logp = (
            biased_target_logp + top_logp - biased_top_logp - self._bias
        )
        logger.debug(
            "target_logprob=%.4f, top_logprob=%.4f, biased_target_logprob=%.4f,"
            " biased_top_logprob=%.4f",
            target_logp,
            top_logp,
            biased_target_logp,
            biased_top_logp,
        )
        if not 0 >= top_logp > target_logp:
            logger.warning(
                "top_logp (%.5f) must be non-positive and larger than "
                "target_logprob (%.5f)! Something went wrong!",
                top_logp,
                target_logp,
            )
            # If this were true, we would not reach this function and target tok
            # would have already been top-1. So we set target_logp to just a bit
            # smaller than top_logp.
            # TODO: This almost always skips this sample. May have bad effect.
            # target_logp = top_logp - 1e-3
            target_logp = top_logp - self._bias
        if not 0 >= biased_target_logp > target_logp:
            logger.warning(
                "biased_target_logp (%.5f) must be larger than target_logprob "
                "(%.5f)!",
                biased_target_logp,
                target_logp,
            )
        if biased_top_logp != biased_target_logp and top_logp < biased_top_logp:
            logger.warning(
                "top_logp (%.5f) must be larger than biased_top_logp (%.5f)!",
                top_logp,
                biased_top_logp,
            )
        return target_logp

    def _get_first_response(
        self,
        prompts: list[str],
        target_toks: list[str],
    ):
        """Get response and logprobs without logit bias.

        Returns:
            target_logprobs: logprobs of target tokens.
            top_logprobs: logprobs of top tokens excluding target tokens.
            top_tokens: top tokens.
            full_logps: full logprobs of responses.
            system_fingerprints: system fingerprints of responses.
        """
        responses, self._queues = standalone_server(
            prompts,
            [{}] * len(prompts),
            queues=self._queues,
            cleanup=False,
            **self.server_kwargs,
        )
        self._num_queries += len(prompts)

        top_logps: list[float | None] = [None for _ in prompts]
        system_fingerprints: list[str | None] = [None for _ in prompts]
        top_toks: list[str | None] = ["" for _ in prompts]
        full_logps: list[dict[str, float] | None] = [None for _ in prompts]
        # Set default target logprobs to 1 which is invalid because logprob
        # should be negative or zero.
        target_logps = np.ones(len(prompts))

        for i, response in enumerate(responses):
            # Skip bad response
            if response is None:
                continue
            self._num_tokens += response.usage.total_tokens
            parsed_resp = _parse_response(response, self.api_type)
            system_fingerprints[i] = parsed_resp.system_fingerprint
            top_toks[i] = parsed_resp.top_tokens[0]
            full_logps[i] = parsed_resp.full_logprobs[0]
            logger.debug("prompt=%s, full_logps=%s", prompts[i], full_logps[i])

            # Check if any top-5 tokens match target
            for tok, lp in parsed_resp.full_logprobs[0].items():
                # We decided to allow for space diffeeence here as the API
                # does not seem to be consistent with space.
                if tok.strip() != target_toks[i].strip():
                    # Keep logprob of top token excluding target token
                    if top_logps[i] is None:
                        top_logps[i] = lp
                    continue

                logger.debug(
                    'Target token ("%s") in top-5 with logp=%.4f.',
                    target_toks[i],
                    lp,
                )
                if tok != target_toks[i]:
                    logger.debug(
                        'Tok and target have space difference ("%s" vs "%s")',
                        tok,
                        target_toks[i],
                    )
                if target_logps[i] == 1 or target_logps[i] < lp:
                    # Only set logprob if it is not set yet or if it is larger
                    # than the current target logprob.
                    target_logps[i] = lp

        return (
            target_logps,
            top_logps,
            top_toks,
            full_logps,
            system_fingerprints,
        )

    def _get_logprobs_score_echo(
        self,
        messages: list[Message],
        suffixes: list[str],
        target_ids: TokenIds,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
    ) -> tuple[list[str], list[float]]:
        num_suffixes = len(suffixes)
        max_target_len = min(max_target_len, len(target_ids))

        # (1) Build target prefixes
        target_prefixes, tmp_target_toks = self._build_target_prefixes(
            target_ids, max_target_len
        )

        # (2) Build prompts
        prompts: list[str] = ["" for _ in range(max_target_len * len(suffixes))]
        target_toks: list[str] = copy.deepcopy(prompts)
        logit_biases = [None for _ in enumerate(prompts)]
        goal = messages[1].content
        for i, suffix in enumerate(suffixes):
            # TODO(task): append suffix
            if suffix.startswith(" "):
                messages[1] = Message(Role.USER, f"{goal}{suffix}")
            else:
                messages[1] = Message(Role.USER, f"{goal} {suffix}")
            # Add target token one at a time
            for j in range(max_target_len):
                messages[2].content = target_prefixes[j]
                prompt = build_prompt(
                    messages if messages[2].content else messages[:2],
                    template_name=self.template,
                    return_openai_chat_format=self.api_type == "chat",
                )
                prompts[i * max_target_len + j] = prompt
                target_toks[i * max_target_len + j] = tmp_target_toks[j]
                logit_biases[i * max_target_len + j] = {
                    int(target_ids[j]): self._bias
                }

        # (3) Call API for the first time (no logit bias)
        out = self._get_first_response(prompts, target_toks)
        target_logps, top_logps, top_tokens, _, system_fingerprints = out

        # Concatenate top tokens at each position to get outputs
        outputs = ["" for _ in range(num_suffixes)]
        cur_output = ["" for _ in range(max_target_len)]
        for i, top_token in enumerate(top_tokens):
            suffix_idx = i // max_target_len
            # Concatenate output tokens for each suffix
            if top_token != "," and not top_token.startswith(" "):
                top_token = f" {top_token}"
            cur_output[i % max_target_len] = top_token
            if i % max_target_len == max_target_len - 1:
                outputs[suffix_idx] = "".join(cur_output)

        # target_logps is set to 1 if target token is NOT in top-5
        second_idx = [i for i, lp in enumerate(target_logps) if lp > 0]
        logger.debug(
            "%d/%d prompts already have target token in top-5.",
            len(prompts) - len(second_idx),
            len(prompts),
        )
        logger.debug("top_logps=%s, top_tokens=%s", top_logps, top_tokens)

        if second_idx:
            # (4) Call API again with remaining prompts and logit biases
            responses, self._queues = standalone_server(
                [prompts[i] for i in second_idx],
                [logit_biases[i] for i in second_idx],
                queues=self._queues,
                cleanup=False,
                **self.server_kwargs,
            )
            self._num_queries += len(second_idx)

            # (5) Calculate target_logprobs using the second call
            for i, idx in enumerate(second_idx):
                # Skip bad response
                if responses[i] is None or top_logps[idx] is None:
                    target_logps[idx] = -np.inf
                    continue

                assert (
                    top_tokens[idx].strip()
                    != target_toks[idx % max_target_len].strip()
                ), (top_tokens[idx], target_toks[idx % max_target_len])
                self._num_tokens += responses[i].usage.total_tokens
                target_logps[idx] = self._get_target_logprob_from_2nd_response(
                    responses[i],
                    target_toks[idx % max_target_len],
                    top_tokens[idx],
                    top_logps[idx],
                    system_fingerprints[idx],
                )
        logger.debug("target_logprobs: %s", target_logps)

        # (6) Sum over target_logprobs to get losses
        losses = [0.0 for _ in enumerate(suffixes)]
        for i, _ in enumerate(losses):
            begin, end = i * max_target_len, (i + 1) * max_target_len
            # Skip bad response
            if any(
                target_logps[j] == -np.inf or top_logps[j] is None
                for j in range(begin, end)
            ):
                losses[i] = np.inf
                continue

            if "ce" in loss_func:
                losses[i] = -np.mean(target_logps[begin:end])
            else:
                loss = 0.0
                for j in range(begin, end):
                    loss += max(top_logps[j] - target_logps[j], -cw_margin)
                losses[i] = loss / max_target_len

        self._assert_loss(losses, loss_func=loss_func, cw_margin=cw_margin)
        return outputs, losses

    def _get_logprobs_score(
        self,
        messages: list[Message],
        suffixes: list[str],
        target_ids: TokenIds,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
    ) -> tuple[list[str], list[float]]:
        # (1) Build target prefixes
        max_target_len = min(max_target_len, len(target_ids))
        target_prefixes, target_toks = self._build_target_prefixes(
            target_ids, max_target_len
        )

        goal = messages[1].content
        losses: list[float] = [np.inf] * len(suffixes)
        outputs: list[str] = ["" for _ in enumerate(suffixes)]
        suffix_idxs = list(range(len(suffixes)))
        best_target_len = 0

        for target_idx in range(max_target_len):
            tgt_tok = target_toks[target_idx]

            # (2) Build prompts
            prompts: list[str] = [""] * len(suffix_idxs)
            for i, idx in enumerate(suffix_idxs):
                suffix = suffixes[idx]
                # TODO(task): append suffix
                if suffix.startswith(" "):
                    messages[1] = Message(Role.USER, f"{goal}{suffix}")
                else:
                    messages[1] = Message(Role.USER, f"{goal} {suffix}")
                # Add target token one at a time
                messages[2].content = target_prefixes[target_idx]
                prompts[i] = build_prompt(
                    messages if messages[2].content else messages[:2],
                    template_name=self.template,
                    return_openai_chat_format=self.api_type == "chat",
                )

            # (3) Call API (no logit bias)
            logger.debug("Querying %d prompts", len(prompts))
            out = self._get_first_response(prompts, [tgt_tok] * len(prompts))
            target_logps, top_logps, top_toks, full_logps, _ = out

            # Get index of successful suffix for next step and concat outputs
            next_suffix_idxs, num_fails = [], 0
            for i, idx in enumerate(suffix_idxs):
                # Skip bad response
                if top_logps[i] is None:
                    continue
                if top_toks[i] == "," or top_toks[i].startswith(" "):
                    outputs[idx] = f"{outputs[idx]}{top_toks[i]}"
                else:
                    outputs[idx] = f"{outputs[idx]} {top_toks[i]}"

                if top_toks[i].strip() == tgt_tok.strip():
                    # Target token is top-1
                    next_suffix_idxs.append(idx)
                    assert 0 >= target_logps[i] > top_logps[i], (
                        f"target_logps[{i}]={target_logps[i]} must be "
                        f"non-positive and larger than top_logps[{i}]"
                        f"={top_logps[i]}!\n{full_logps[i]}"
                    )
                elif target_logps[i] > 0:
                    # Target token not in top-5
                    # Approximate target prob as 1 - sum of prob of other tokens
                    refuse_prob = sum(
                        np.exp(lp) for lp in full_logps[i].values()
                    )
                    approx_target_lprob = np.log(1 - refuse_prob)
                    target_logps[i] = approx_target_lprob
                    logger.debug(
                        "Approx. target prob: %.4f | full_logps: %s",
                        1 - refuse_prob,
                        full_logps[i],
                    )
                    num_fails += 1
                else:
                    # Target token is not top-1 but in top-5
                    assert 0 >= top_logps[i] > target_logps[i], (
                        f"top_logps[{i}]={top_logps[i]} must be "
                        f"non-positive and larger than target_logps[{i}]"
                        f"={target_logps[i]}!\n{full_logps[i]}"
                    )
                # Aggregate target logprobs for successful suffixes
                if losses[idx] == np.inf and target_idx == 0:
                    losses[idx] = 0.0
                if "ce" in loss_func:
                    losses[idx] -= target_logps[i]
                else:
                    losses[idx] += max(
                        top_logps[i] - target_logps[i], -cw_margin
                    )

            num_top1 = len(next_suffix_idxs)
            logger.debug(
                'Target token "%s" (%d/%d): top-1=%d, top-5=%d, others=%d',
                tgt_tok,
                target_idx + 1,
                max_target_len,
                num_top1,
                len(suffix_idxs) - num_top1 - num_fails,
                num_fails,
            )

            # Prioritize successful suffixes. Set loss of unsuccessful suffixes
            # to inf and ignore them in the next step.
            if num_top1 > 0:
                suffix_idxs = list(next_suffix_idxs)
                continue

            # We stop if there is at least one sample in top-5, assuming those
            # not in top-5 will have higher loss.
            if len(suffix_idxs) - num_top1 - num_fails > 0:
                best_target_len = target_idx + 1
                break

            # Reach here when no target token is in top-5 in any prompt
            best_target_len = target_idx
            break

        # NOTE: Add loss offset to indicate the best target length achieved
        offset = (max_target_len - best_target_len) * 1e3
        losses = [loss + offset for loss in losses]
        suffix_idxs = set(suffix_idxs)
        # Set loss of unsuccessful suffixes to inf
        for i, _ in enumerate(losses):
            if i not in suffix_idxs:
                losses[i] = np.inf
        self._assert_loss(losses, loss_func=loss_func, cw_margin=cw_margin)
        return outputs, losses

    def _get_logprobs_score_lb(
        self,
        messages: list[Message],
        suffixes: list[str],
        target_ids: TokenIds,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
    ) -> tuple[list[str], list[float]]:
        # DEPRECATED: This loss computation no longer works on OpenAI API
        # because they disable logit bias' effect on logprobs.
        # (1) Build target prefixes
        max_target_len = min(max_target_len, len(target_ids))
        target_prefixes, target_toks = self._build_target_prefixes(
            target_ids, max_target_len
        )

        goal = messages[1].content
        losses: list[float] = [np.inf] * len(suffixes)
        outputs: list[str] = ["" for _ in enumerate(suffixes)]
        suffix_idxs = list(range(len(suffixes)))
        best_target_len = 1

        for target_idx in range(max_target_len):
            tgt_tok = target_toks[target_idx]
            best_target_len = target_idx + 1  # Current target length

            # (2) Build prompts
            prompts: list[str] = [""] * len(suffix_idxs)
            for i, idx in enumerate(suffix_idxs):
                suffix = suffixes[idx]
                # TODO(task): append suffix
                if suffix.startswith(" "):
                    messages[1] = Message(Role.USER, f"{goal}{suffix}")
                else:
                    messages[1] = Message(Role.USER, f"{goal} {suffix}")
                # Add target token one at a time
                messages[2].content = target_prefixes[target_idx]
                prompts[i] = build_prompt(
                    messages if messages[2].content else messages[:2],
                    template_name=self.template,
                    return_openai_chat_format=self.api_type == "chat",
                )

            # (3) Call API for the first time (no logit bias)
            logger.debug("Querying %d prompts", len(prompts))
            out = self._get_first_response(prompts, [tgt_tok] * len(prompts))
            target_logps, top_logps, top_toks, _, system_fingerprints = out

            # Get index of successful suffix for next step and concat outputs
            next_suffix_idxs, second_call_idxs = [], []
            second_call_prompts = []
            for i, idx in enumerate(suffix_idxs):
                # Skip bad response
                if top_logps[i] is None:
                    continue
                if top_toks[i] == "," or top_toks[i].startswith(" "):
                    outputs[idx] = f"{outputs[idx]}{top_toks[i]}"
                else:
                    outputs[idx] = f"{outputs[idx]} {top_toks[i]}"

                # Target token not in top-5
                if target_logps[i] > 0:
                    second_call_idxs.append(idx)
                    second_call_prompts.append(prompts[i])
                    continue

                if top_toks[i].strip() == tgt_tok.strip():
                    # Target token is top-1
                    next_suffix_idxs.append(idx)
                    assert 0 >= target_logps[i] > top_logps[i], (
                        f"target_logps[{i}]={target_logps[i]} must be "
                        f"non-positive and larger than top_logps[{i}]"
                        f"={top_logps[i]}!"
                    )
                else:
                    # Target token is not top-1 but in top-5
                    assert 0 >= top_logps[i] > target_logps[i], (
                        f"top_logps[{i}]={top_logps[i]} must be "
                        f"non-positive and larger than target_logps[{i}]"
                        f"={target_logps[i]}!"
                    )
                # Aggregate target logprobs for successful suffixes
                if losses[idx] == np.inf and target_idx == 0:
                    losses[idx] = 0.0
                if "ce" in loss_func:
                    losses[idx] -= target_logps[i]
                else:
                    losses[idx] += max(
                        top_logps[i] - target_logps[i], -cw_margin
                    )

            num_top1, num_others = len(next_suffix_idxs), len(second_call_idxs)
            logger.debug(
                'Target token "%s" (%d/%d): top-1=%d, top-5=%d, others=%d',
                tgt_tok,
                target_idx + 1,
                max_target_len,
                num_top1,
                len(suffix_idxs) - num_top1 - num_others,
                num_others,
            )

            # Prioritize successful suffixes. Set loss of unsuccessful suffixes
            # to inf and ignore them in the next step.
            if num_top1 > 0:
                suffix_idxs = list(next_suffix_idxs)
                continue

            # We stop if there is at least one sample in top-5, assuming those
            # not in top-5 will have higher loss.
            if len(suffix_idxs) - num_top1 - num_others > 0:
                break

            # In case there's no successful prompts at this target length.
            # Compute target logprob using the second call.
            logit_bias = {int(target_ids[target_idx]): self._bias}
            responses, self._queues = standalone_server(
                second_call_prompts,
                [logit_bias] * len(second_call_prompts),
                queues=self._queues,
                cleanup=False,
                **self.server_kwargs,
            )
            self._num_queries += len(second_call_prompts)

            # Calculate target_logprobs using the second call
            for i, idx in enumerate(second_call_idxs):
                # Skip bad response (either first or second call fails)
                if responses[i] is None or top_logps[i] is None:
                    losses[idx] = np.inf
                    continue

                self._num_tokens += responses[i].usage.total_tokens
                target_logp = self._get_target_logprob_from_2nd_response(
                    responses[i],
                    tgt_tok,
                    top_toks[i],
                    top_logps[i],
                    system_fingerprints[i],
                )
                if losses[idx] == np.inf and target_idx == 0:
                    losses[idx] = 0.0
                if "ce" in loss_func:
                    losses[idx] -= target_logp
                else:
                    losses[idx] += max(top_logps[i] - target_logp, -cw_margin)
            break

        # NOTE: Add loss offset to indicate the best target length achieved
        offset = (max_target_len - best_target_len) * 1e3
        losses = [loss + offset for loss in losses]
        suffix_idxs = set(suffix_idxs)
        # Set loss of unsuccessful suffixes to inf
        for i, _ in enumerate(losses):
            if i not in suffix_idxs:
                losses[i] = np.inf
        self._assert_loss(losses, loss_func=loss_func, cw_margin=cw_margin)
        return outputs, losses

    def compute_suffix_loss(
        self,
        inputs: ModelInputIds | ModelInputs,
        batch_size: int | None = None,
        temperature: float = 1.0,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
        **kwargs,
    ) -> LossOutput:
        logger.debug("Computing message loss on OpenAI model...")
        _ = batch_size, kwargs  # unused
        if not isinstance(inputs, ModelInputs):
            raise NotImplementedError("Only ModelInputs is supported.")
        messages: list[Message] = inputs.messages
        suffixes: list[str] = inputs.suffixes
        assert len(inputs.targets) == 1, "Only support single target for now."
        target: str = inputs.targets[0]
        self._num_queries, self._num_tokens = 0, 0
        messages = copy.deepcopy(messages)
        default_temp = self.temperature
        self.temperature = temperature

        # TODO(multitarget): not remove space
        if self.api_type == "chat" and target.startswith(" "):
            target = target[1:]
            # max_target_len counts leading space by default for other
            # tokenizers but GPT tokenizer combines space with the next token
            max_target_len -= 1
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        logger.debug("target=%s, target_ids=%s", target, target_ids)

        messages.append(Message(Role.ASSISTANT, target))
        assert len(messages) == 3, messages  # TODO(task)

        if self._score_mode == "echo-logprobs":
            logger.warning("echo-logprobs mode is outdated.")
            raise NotImplementedError("echo-logprobs not implemented.")
        elif self._score_mode == "logprobs":
            if "one" in loss_func:
                func = self._get_logprobs_score
            elif "lb" in loss_func:
                # lb for logit bias (deprecated)
                func = self._get_logprobs_score_lb
            else:
                func = self._get_logprobs_score_echo
            outputs, losses = func(
                messages,
                suffixes,
                target_ids,
                max_target_len=max_target_len,
                loss_func=loss_func,
                cw_margin=cw_margin,
            )
        else:
            raise NotImplementedError("Hard-label mode not implemented.")

        self.temperature = default_temp
        losses = torch.tensor(losses)
        logger.debug(
            "compute_message_loss done. Used queries: %d, tokens: %d",
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
            # Count number of non-pad tokens
            # pad_token_id = self._tokenizer.pad_token_id
            # filter_cond = (encoded != pad_token_id).sum(1) == orig_len
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

    @staticmethod
    def _assert_loss(
        losses: list[float], loss_func: str = "ce-all", cw_margin: float = 1e-3
    ) -> None:
        lb = 0 if "ce" in loss_func else -cw_margin
        for i, loss in enumerate(losses):
            if loss >= lb:
                continue
            logger.warning("Loss %g is lower than lower bound %g", loss, lb)
            logger.warning("Assign inf loss, and this sample will be ignored.")
            logger.warning(
                "If this happens very rarely, then it is likely due to "
                "randomness in the API. Otherwise, there is likely a bug."
            )
            losses[i] = np.inf

    def encode(self, messages: list[Message]):
        return [
            {"role": m.role.name.lower(), "content": m.content}
            for m in messages
        ]

    def __call__(
        self,
        messages: list[Message],
        api_key: str | None = None,
        prompt: str | None = None,
        logit_bias: dict[int, float] | None = None,
        echo: bool = False,
        echo_len: int = 0,
        logprobs: int = 0,
    ) -> Response | CompletionResponse:
        logit_bias = logit_bias or self.logit_bias
        if api_key is not None:
            openai.api_key = api_key

        if self.api_type == "completion":
            if prompt is None:
                prompt = (
                    build_prompt(messages, template_name=self.template)
                    if len(messages) > 0
                    else messages[0].content
                )
            response = self.client.completions.create(
                prompt=prompt,
                echo=echo,
                logprobs=logprobs,
                logit_bias=logit_bias,
                **self.request_kwargs,
            )
            response = CompletionResponse(
                response, stream=self.stream, echo_len=echo_len
            )
        else:
            response = self.client.chat.completions.create(
                messages=self.encode(messages),
                logit_bias=logit_bias,
                **self.request_kwargs,
            )
            response = Response(response, stream=self.stream)

        if api_key is not None:
            openai.api_key = os.getenv("OPENAI_API_KEY", "")
        return response

    def _echo_logprobs_score(
        self, messages: list[Message], max_target_len: int = 32
    ) -> float:
        # DEPRECATED: Only works for text-davinci-003
        input_len = len(
            self.tokenizer(
                build_prompt(messages[:-1], template_name=self.template),
                add_special_tokens=False,
            ).input_ids
        )

        prompt = build_prompt(messages, template_name=self.template)
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        prompt_len = len(prompt_ids)

        max_prompt_len = max_target_len + input_len
        if max_prompt_len < prompt_len:
            # Update prompt according to max_target_len
            prompt = self.tokenizer.decode(
                prompt_ids[:max_prompt_len], skip_special_tokens=True
            )
            prompt_len = max_prompt_len

        # Call API
        orig_max_tokens = self.max_tokens
        self.max_tokens = prompt_len
        response = openai_call_with_retries(
            self,
            messages,
            echo=self._echo,
            logprobs=5,
            echo_len=input_len,
            prompt=prompt,
        )
        self.max_tokens = orig_max_tokens

        # Sum up logprobs over target
        score = -np.mean(response.logprobs)
        return score, response.top_tokens

    def _binary_search_logit_bias(
        self, messages: list[Message], target_id: int, index: int = 0
    ):
        # DEPRECATED: Only works for text-davinci-003
        logger.debug("Performing binary search on logit bias.")
        start_time = time.time()
        lo, hi = 0, 100
        target_id = int(target_id)
        orig_bias = self.logit_bias.get(target_id)
        num_requests = 0
        is_match, is_empty = False, False
        while hi - lo > 1e-3:
            mid = (lo + hi) / 2
            self.logit_bias[target_id] = mid

            response = openai_call_with_retries(self, messages)
            output = response.response
            # Content filter
            is_content_filter = response.finish_reason == "content_filter"
            if output:
                is_match = self.tokenizer(output).input_ids[index] == target_id
                is_empty = False
            else:
                is_match = False
                is_empty = True
            num_requests += 1
            # TODO(future): may check for output == "" also
            if (is_empty and is_content_filter) or is_match:
                hi = mid
            else:
                lo = mid
        if orig_bias is not None:
            self.logit_bias[target_id] = orig_bias
        logger.debug(
            "Binary search finished and used %d requests (%ds).",
            num_requests,
            time.time() - start_time,
        )
        return lo


@retry(
    retry=retry_if_not_exception_type(
        (
            openai.BadRequestError,
            openai.AuthenticationError,
            openai.NotFoundError,
            TypeError,
        )
    ),
    wait=wait_random_exponential(min=1, max=10),
)
def openai_call_with_retries(model: OpenAIModel, messages, **kwargs):
    return model(messages, **kwargs)


@dataclasses.dataclass
class ParsedResponse:
    text: str
    full_logprobs: list[dict[str, float]]
    top_logprobs: list[float]
    top_tokens: list[str]
    system_fingerprint: str = ""


def _parse_response(
    response: Completion | ChatCompletion,
    api_type: Literal["chat", "completion"],
) -> ParsedResponse:
    """Parse response from OpenAI API."""
    if api_type == "chat":
        contents = response.choices[0].logprobs.content
        full_logprobs = [None for _ in contents]
        top_logprobs = [0.0 for _ in contents]
        top_tokens = ["" for _ in contents]

        for i, content in enumerate(contents):
            top_lps: list[TopLogprob] = content.top_logprobs
            full_logprobs[i] = {
                top_lp.token: top_lp.logprob for top_lp in top_lps
            }
            top_logprobs[i] = top_lps[0].logprob
            top_tokens[i] = top_lps[0].token
        text = response.choices[0].message.content

    else:
        logprobs = response.choices[0].logprobs
        full_logprobs = logprobs.top_logprobs
        top_logprobs = logprobs.token_logprobs
        top_tokens = logprobs.tokens
        text = response.choices[0].text

    return ParsedResponse(
        text=text,
        full_logprobs=full_logprobs,
        top_logprobs=top_logprobs,
        top_tokens=top_tokens,
        system_fingerprint=response.system_fingerprint,
    )
