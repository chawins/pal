"""GCG Attack."""

import logging

import numpy as np
import torch
from ml_collections import ConfigDict

from src.attacks.base import BaseAttack
from src.models.model_input import ModelInputIds
from src.utils.types import BatchTokenIds

logger = logging.getLogger(__name__)


def _rand_permute(size, device: str = "cuda", dim: int = -1):
    return torch.argsort(torch.rand(size, device=device), dim=dim)


class GCGAttack(BaseAttack):
    """GCG Attack (see https://llm-attacks.org/)."""

    name: str = "gcg"

    def __init__(self, config: ConfigDict, *args, **kwargs) -> None:
        """Initialize GCG attack."""
        self._topk = config.topk
        self._num_coords: tuple[int, int] = config.num_coords
        self._mu: float = config.mu
        self._sample_mode: str = config.sample_mode
        if (
            not isinstance(self._num_coords, tuple)
            or len(self._num_coords) != 2
        ):
            raise ValueError(
                f"num_coords must be tuple of two ints, got {self._num_coords}"
            )

        # Init base class after setting parameters because it will call
        # _get_name_tokens() which uses the parameters. See below.
        super().__init__(config, *args, **kwargs)
        self._momentum: torch.Tensor | None = None
        self._cur_num_coords: int = self._num_coords[0]

    def _get_name_tokens(self) -> list[str]:
        atk_tokens = super()._get_name_tokens()
        atk_tokens.append(f"k{self._topk}")
        atk_tokens.append(self._sample_mode)
        if any(c != 1 for c in self._num_coords):
            if self._num_coords[0] == self._num_coords[1]:
                atk_tokens.append(f"c{self._num_coords[0]}")
            else:
                atk_tokens.append(
                    f"c{self._num_coords[0]}-{self._num_coords[1]}"
                )
        if self._mu != 0:
            atk_tokens.append(f"m{self._mu}")
        return atk_tokens

    def _setup_run(self, *args, **kwargs) -> None:
        """Set up before each attack run."""
        super()._setup_run(*args, **kwargs)
        self._cur_num_coords = self._num_coords[0]

    def _on_step_begin(self, *args, **kwargs):
        """Executed at the beginning of each step."""
        self._cur_num_coords = round(
            self._num_coords[0]
            + (self._num_coords[1] - self._num_coords[0])
            * self._step
            / self._num_steps
        )

    @torch.no_grad()
    def _compute_grad(
        self, eval_input: ModelInputIds, **kwargs
    ) -> torch.Tensor:
        _ = kwargs  # unused
        grad = self._model.compute_grad(
            eval_input,
            max_target_len=self._cur_seq_len,
            temperature=self._loss_temperature,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
            **kwargs,
        )
        if self._mu == 0:
            return grad

        # Calculate momentum term
        if self._momentum is None:
            self._momentum = torch.zeros_like(grad)
        self._momentum.mul_(self._mu).add_(grad)
        return self._momentum

    @torch.no_grad()
    def _sample_updates(
        self,
        optim_ids: torch.LongTensor,
        *args,
        grad: torch.FloatTensor | None = None,
        **kwargs,
    ) -> BatchTokenIds:
        _ = args, kwargs  # unused
        assert isinstance(grad, torch.Tensor), "grad is required for GCG!"
        assert len(grad) == len(optim_ids), (
            f"grad and optim_ids must have the same length ({len(grad)} vs "
            f"{len(optim_ids)})!"
        )
        device = grad.device
        num_coords = min(self._cur_num_coords, len(optim_ids))
        if self._not_allowed_tokens is not None:
            grad[:, self._not_allowed_tokens.to(device)] = np.infty

        # pylint: disable=invalid-unary-operand-type
        top_indices = (-grad).topk(self._topk, dim=1).indices
        # Set a larger batch size as we lose some during filtering
        batch_size = max(int(self._batch_size * 1.5), 8)
        # old_token_ids: [batch_size, seq_len]
        old_token_ids = optim_ids.repeat(batch_size, 1)

        if num_coords == 1 and self._sample_mode == "orig":
            if batch_size < len(optim_ids):
                logger.warning(
                    "batch_size (%d) < len(optim_ids) (%d) in candidate "
                    "sampling with original mode!",
                    batch_size,
                    len(optim_ids),
                )
            # Each position will have `batch_size / len(optim_ids)` candidates
            new_token_pos = torch.arange(
                0,
                len(optim_ids),
                len(optim_ids) / batch_size,
                device=device,
            ).type(torch.int64)
            # Get random indices to select from topk
            # rand_idx: [seq_len, topk, 1]
            rand_idx = _rand_permute(
                (len(optim_ids), self._topk, 1), device=device, dim=1
            )
            # Get the first (roughly) batch_size / seq_len indices at each
            # position
            rand_idx = torch.cat(
                [
                    r[: (new_token_pos == i).sum()]
                    for i, r in enumerate(rand_idx)
                ],
                dim=0,
            )
            assert rand_idx.shape == (batch_size, 1), rand_idx.shape
            # top_indices: [seq_len, topk]
            # top_indices[new_token_pos]: [batch_size, topk]
            new_token_val = torch.gather(
                top_indices[new_token_pos], 1, rand_idx
            )
            # new_token_val: [batch_size, 1]
            new_token_ids = old_token_ids.scatter(
                1, new_token_pos.unsqueeze(-1), new_token_val
            )
        else:
            # Randomly choose positions to update
            new_token_pos = _rand_permute(
                (batch_size, len(optim_ids)), device=device, dim=1
            )[:, :num_coords]
            # Get random indices to select from topk
            rand_idx = torch.randint(
                0, self._topk, (batch_size, num_coords, 1), device=device
            )
            new_token_val = torch.gather(
                top_indices[new_token_pos], -1, rand_idx
            )
            new_token_ids = old_token_ids
            for i in range(num_coords):
                new_token_ids.scatter_(
                    1, new_token_pos[:, i].unsqueeze(-1), new_token_val[:, i]
                )

        assert new_token_ids.shape == (
            batch_size,
            len(optim_ids),
        ), new_token_ids.shape
        return new_token_ids
