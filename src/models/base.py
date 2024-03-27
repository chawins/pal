import dataclasses
from typing import Iterable, List

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from src.message import Message, Role
from src.models.model_input import ModelInputIds, ModelInputs

BatchLoss = Float[Tensor, "batch_size"]
BatchLogits = Float[Tensor, "batch_size seq_len vocab_size"]


@dataclasses.dataclass
class LossOutput:
    """Loss output from model."""

    losses: BatchLoss
    logits: BatchLogits | None = None
    texts: List[str] | None = None
    num_queries: int = 0
    num_tokens: int = 0


@dataclasses.dataclass
class Encoded:
    input_ids: torch.Tensor | list[int] | np.ndarray


class BaseModel:
    supports_system_message = False

    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, messages: List[Message], api_key: str = None):
        raise NotImplementedError

    def set_prefix_cache(self, messages: list[Message]) -> None:
        """Set prefix KV-cache for model."""
        raise NotImplementedError

    def compute_suffix_loss(
        self, inputs: ModelInputIds | ModelInputs, **kwargs
    ) -> LossOutput:
        """Compute loss given EvalInput object.

        Args:
            inputs: Can be ModelInputIds or ModelInputs.

        Returns:
            LossOutput object.
        """
        raise NotImplementedError

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`."""
        raise NotImplementedError


class MockModel(BaseModel):
    """Testing model which returns the user's input as the response."""

    supports_system_message = False

    def __init__(self, **kwargs):
        pass

    def __call__(self, _, __):
        response = input("[Response]: ")
        return [response]

    def set_prefix_cache(self, messages: list[Message]) -> None:
        pass

    def compute_suffix_loss(
        self, inputs: ModelInputIds | ModelInputs, **kwargs
    ) -> LossOutput:
        if not isinstance(inputs, ModelInputIds):
            raise NotImplementedError("Only ModelInputIds is supported.")
        batch_size = len(inputs.suffix_ids)
        device = inputs.suffix_ids
        target_len = inputs.loss_slice.stop - inputs.loss_slice.start
        # Vocab size is hard-coded here for llama (can be wrong)
        mock_logits = torch.randn(
            (batch_size, target_len, 32000), device=device
        )
        mock_losses = torch.randn(batch_size, device=device)
        return LossOutput(
            losses=mock_losses, logits=mock_logits, num_queries=batch_size
        )

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`."""
        _ = kwargs  # unused
        optim_slice = eval_input.optim_slice
        input_ids = eval_input.dynamic_input_ids
        length = optim_slice.stop - optim_slice.start
        mock_grad = torch.randn(length, device=input_ids.device)
        return mock_grad


class UselessModel(BaseModel):
    supports_system_message = False

    def __init__(self, **kwargs):
        pass

    def __call__(self, messages: List[Message], api_key: str = None):
        return [f"I have ({len(messages)}) unread messages."]

    def set_prefix_cache(self, messages: list[Message]) -> None:
        pass

    def compute_suffix_loss(
        self, inputs: ModelInputIds, **kwargs
    ) -> LossOutput:
        """Compute loss given multiple suffixes."""
        batch_size = len(inputs.suffix_ids)
        device = inputs.suffix_ids
        target_len = inputs.loss_slice.stop - inputs.loss_slice.start
        # Vocab size is hard-coded here for llama (can be wrong)
        mock_logits = torch.randn(
            (batch_size, target_len, 32000), device=device
        )
        mock_losses = torch.zeros(batch_size, device=device)
        return LossOutput(
            losses=mock_losses, logits=mock_logits, num_queries=batch_size
        )

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`."""
        _ = kwargs  # unused
        optim_slice = eval_input.optim_slice
        input_ids = eval_input.dynamic_input_ids
        length = optim_slice.stop - optim_slice.start
        mock_grad = torch.zeros(length, device=input_ids.device)
        return mock_grad


def print_and_concat_stream(response: Iterable, role: Role = Role.ASSISTANT):
    chunks = []
    print(f"[{role.name.title()}]: ", end="", flush=True)
    for chunk in response:
        chunks.append(chunk)
        print(chunk, end="", flush=True)
    print("\n", end="", flush=True)
    return "".join(chunks)


def concat_stream(response: Iterable):
    return "".join(list(response))


class NaNLossError(Exception):
    """NaN loss occurs during fine-tuning."""
