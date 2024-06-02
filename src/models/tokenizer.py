import logging
import os
from typing import Literal

import cohere
import numpy as np
import tiktoken
import torch
from llama.tokenizer import Tokenizer
from transformers import BatchEncoding

from src.servers.cohere_server import EXCEPT_COHERE_ERRORS
from src.utils.cohere_non_ascii import COHERE_NON_ASCII

logger = logging.getLogger(__name__)


class GptTokenizer:
    def __init__(self, encoding: tiktoken.Encoding) -> None:
        # Get the tokeniser corresponding to a specific model in the OpenAI API
        self.encoding: tiktoken.Encoding = encoding
        # Set interface to match HuggingFace
        self.vocab_size = self.encoding.max_token_value + 1
        self.bos_token_id = self.encoding.eot_token
        self.eos_token_id = self.encoding.eot_token
        self.pad_token_id = self.encoding.eot_token
        self.unk_token_id = self.encoding.eot_token
        self.eot_token = "<|endoftext|>"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ) -> BatchEncoding:
        _ = kwargs  # unused
        if text is None:
            return BatchEncoding()

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
        return BatchEncoding({"input_ids": input_ids})

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


class CohereTokenizer:
    non_ascii = COHERE_NON_ASCII

    def __init__(self, model_name: str) -> None:
        self._co = cohere.Client()
        # Assume command/command-light model
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

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str | list[str],
        return_tensors: Literal["list", "pt", "np"] = "list",
        **kwargs,
    ) -> BatchEncoding:
        _ = kwargs  # unused
        if text is None:
            return BatchEncoding()

        def tokenize(t):
            try:
                ids = self._co.tokenize(text=t, model=self._model_name).tokens
            except EXCEPT_COHERE_ERRORS as e:
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
        return BatchEncoding({"input_ids": input_ids})

    def _parse_ids(self, ids):
        return ids

    def _detokenize(self, tokens: list[int]) -> str:
        try:
            text = self._co.detokenize(
                tokens=tokens, model=self._model_name
            ).text
        except EXCEPT_COHERE_ERRORS as e:
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


class Llama3Tokenizer:
    def __init__(self) -> None:
        model_path = os.environ["LLAMA3_TOKENIZER_PATH"]
        self.encoder = Tokenizer(model_path)
        self.special_tokens = self.encoder.special_tokens

        # Set interface to match HuggingFace
        self.vocab_size = self.encoder.n_words
        self.bos_token_id = self.encoder.bos_id
        self.eos_token_id = self.encoder.eos_id
        self.unk_token_id = self.encoder.special_tokens["<|eot_id|>"]
        self.eot_token = "<|eot_id|>"
        # Padding Llama3: https://github.com/meta-llama/llama3/issues/42
        self.pad_token_id = self.encoder.eos_id  # "<|end_of_text|>"
        self.pad_token = "<|end_of_text|>"

    def __len__(self) -> int:
        return self.vocab_size

    def __call__(
        self,
        text: str,
        return_tensors: Literal["list", "pt", "np"] = "list",
        add_special_tokens: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        _ = kwargs  # unused
        if text is None:
            return BatchEncoding()

        def _encode(x):
            return self.encoder.encode(
                x,
                bos=add_special_tokens,
                eos=False,
                disallowed_special=(),
                allowed_special="all",
            )

        if isinstance(text, list):
            # Encode all special tokens as normal text
            _ids = [_encode(t) for t in text]
            max_len = max(len(i) for i in _ids)
            input_ids = np.zeros((len(_ids), max_len), dtype=np.int64)
            input_ids += self.pad_token_id
            for i, _id in enumerate(_ids):
                input_ids[i, : len(_id)] = _id
        else:
            input_ids = _encode(text)
            input_ids = np.array(input_ids, dtype=np.int64)

        attention_mask = None
        if return_tensors == "pt":
            input_ids = torch.from_numpy(input_ids)
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        elif return_tensors == "list":
            input_ids = input_ids.tolist()
        return BatchEncoding(
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )

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
        decoded = self.encoder.decode(ids)
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
        decoded_list = [self.encoder.decode(_id) for _id in ids]
        decoded_list = [s.replace(self.eot_token, "") for s in decoded_list]
        return decoded_list
