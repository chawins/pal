"""GCG Attack."""

import logging

import torch
import torch.nn.functional as F
from ml_collections import ConfigDict
from torch.distributions.categorical import Categorical

from src.attacks.blackbox import BlackBoxAttack

logger = logging.getLogger(__name__)


def _rand_permute(size, device: str = "cuda", dim: int = -1):
    return torch.argsort(torch.rand(size, device=device), dim=dim)


class RalAttack(BlackBoxAttack):
    """RAL attack."""

    name: str = "ral"

    def __init__(
        self, config: ConfigDict, *args, lazy_init: bool = False, **kwargs
    ) -> None:
        """Initialize GCG-Random attack."""
        self._num_coords: tuple[int, int] = config.num_coords
        self._token_dist: str = config.token_dist
        self._token_probs_temp: float = config.token_probs_temp
        self._naive_mode: bool = (
            self._num_coords in (1, (1, 1)) and self._token_dist == "uniform"
        )
        self._sample_mode: str = config.sample_mode
        super().__init__(config, *args, lazy_init=lazy_init, **kwargs)

        # Assert params
        if (
            not isinstance(self._num_coords, tuple)
            or len(self._num_coords) != 2
        ):
            raise ValueError(
                f"num_coords must be tuple of two ints, got {self._num_coords}"
            )

        if not lazy_init and self._tokenizer is not None:
            self._token_probs = torch.ones(
                len(self._tokenizer), device=self._device
            )
            self._token_probs.scatter_(0, self._not_allowed_tokens, 0)
            self._num_valid_tokens = self._token_probs.sum().int().item()
            self._valid_token_ids = self._token_probs.nonzero().squeeze()
        self._cur_num_coords: int = self._num_coords[0]

    def _get_name_tokens(self) -> list[str]:
        atk_tokens = super()._get_name_tokens()
        atk_tokens.append(self._sample_mode)
        atk_tokens.append(self._token_dist)
        atk_tokens.append(f"t{self._token_probs_temp}")
        if self._num_coords[0] == self._num_coords[1]:
            atk_tokens.append(f"c{self._num_coords[0]}")
        else:
            atk_tokens.append(f"c{self._num_coords[0]}-{self._num_coords[1]}")
        return atk_tokens

    def _setup_run(self, *args, **kwargs) -> None:
        """Set up before each attack run."""
        super()._setup_run(*args, **kwargs)
        self._cur_num_coords = self._num_coords[0]

    def _on_step_begin(self, *args, **kwargs):
        """Exectued at the beginning of each step."""
        self._cur_num_coords = round(
            self._num_coords[0]
            + (self._num_coords[1] - self._num_coords[0])
            * self._step
            / self._num_steps
        )

    def _compute_token_probs(
        self, input_ids, optim_slice: slice
    ) -> torch.Tensor:
        if self._token_dist == "uniform":
            return self._token_probs

        if self._token_dist in ("rare-target", "common-target"):
            input_embeds = self._model.embed_layer(input_ids)
            input_embeds.unsqueeze_(0)
            # Forward pass
            logits = self._model(
                inputs_embeds=input_embeds,
                past_key_values=self._model.prefix_cache,
            ).logits
            logits = logits[:, optim_slice]
            logits.squeeze_(0)
            # logits: [optim_len, vocab_size]
            # DEPRECATED: Different way to prioritize "rare" tokens
            # if "rare" in self._token_dist:
            #     logits *= -1
            logits = logits.float()
            logits /= self._token_probs_temp
            logits *= self._token_probs[None, :]
            token_probs = F.softmax(logits, dim=-1)
            if "rare" in self._token_dist:
                token_probs = 1 - token_probs
            return token_probs

        raise ValueError(f"Unknown token_dist: {self._token_dist}!")

    def _sample_updates(
        self,
        optim_ids: torch.LongTensor,
        *args,
        optim_slice: slice | None = None,
        **kwargs,
    ) -> torch.Tensor:
        _ = args, kwargs  # unused
        device = optim_ids.device
        optim_len = len(optim_ids)
        num_coords = min(self._cur_num_coords, len(optim_ids))
        batch_size = max(int(self._batch_size * 1.5), 8)

        new_token_ids = optim_ids.repeat(batch_size, 1)

        if self._naive_mode and self._sample_mode == "orig":
            if batch_size < len(optim_ids):
                logger.warning(
                    "batch_size (%d) < len(optim_ids) (%d) in candidate "
                    "sampling with original mode!",
                    batch_size,
                    len(optim_ids),
                )
            # Fixed number of candidates per position
            # Each position will have `batch_size / len(optim_ids)` candidates
            new_token_pos = torch.arange(
                0,
                len(optim_ids),
                len(optim_ids) / batch_size,
                device=device,
            ).type(torch.int64)
            new_token_pos.unsqueeze_(-1)
            # Get random indices to select from topk
            # rand_idx: [seq_len, num_valid_tokens, 1]
            rand_idx = _rand_permute(
                (len(optim_ids), self._num_valid_tokens, 1),
                device=device,
                dim=1,
            )
            # Get the first (roughly) batch_size / seq_len indices at each position
            # rand_idx: [batch_size, 1]
            rand_idx = torch.cat(
                [
                    r[: (new_token_pos == i).sum()]
                    for i, r in enumerate(rand_idx)
                ],
                dim=0,
            )
            new_token_val = self._valid_token_ids[rand_idx]
            new_token_ids.scatter_(1, new_token_pos, new_token_val)
        else:
            # Random uniformly random positions to update
            new_token_pos = _rand_permute(
                (batch_size, optim_len), device=device, dim=1
            )[:, :num_coords]
            new_token_pos.unsqueeze_(-1)

            # Sample new tokens into batch: [batch_size, num_coords, 1]
            token_probs = self._compute_token_probs(optim_ids, optim_slice)
            if token_probs.ndim == 1:
                token_dist = Categorical(probs=token_probs)
                new_token_val = token_dist.sample((batch_size, num_coords, 1))
            else:
                token_probs = token_probs[new_token_pos]
                token_dist = Categorical(probs=token_probs)
                new_token_val = token_dist.sample()
            # new_token_pos, new_token_val: [batch_size, num_coords, 1]
            for i in range(num_coords):
                new_token_ids.scatter_(
                    1, new_token_pos[:, i], new_token_val[:, i]
                )

        assert new_token_ids.shape == (
            batch_size,
            len(optim_ids),
        ), new_token_ids.shape
        return new_token_ids
