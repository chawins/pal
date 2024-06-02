"""Base class for attacks."""

import gc
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import transformers
from ml_collections import ConfigDict

from src.message import Message
from src.models import BaseModel
from src.models.model_input import (
    LengthMismatchError,
    ModelInputIds,
    ModelInputs,
)
from src.utils.suffix import SuffixManager
from src.utils.types import BatchTokenIds

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Attack's output."""

    best_loss: float
    best_suffix: str
    num_queries: int


class BaseAttack:
    """Base class for attacks."""

    name: str = "base"  # Name of the attack
    valid_skip_modes = ("none", "seen", "visited")
    valid_loss_funcs = ("ce-one", "ce-all", "cw-one", "cw-all")

    def __init__(
        self,
        config: ConfigDict,
        wrapped_model: BaseModel,
        tokenizer: transformers.AutoTokenizer,
        suffix_manager: SuffixManager,
        eval_fn: Callable[[str], bool],
        not_allowed_tokens: torch.Tensor | None,
        lazy_init: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the attack."""
        _ = kwargs  # Unused
        self._num_steps: int = config.num_steps
        self._fixed_params: bool = config.fixed_params
        self._adv_suffix_init: str = config.adv_suffix_init
        self._init_suffix_len: int = config.init_suffix_len
        self._batch_size = config.batch_size
        if config.mini_batch_size <= 0:
            self._mini_batch_size = config.batch_size
        else:
            self._mini_batch_size = config.mini_batch_size
        self._log_freq: int = config.log_freq
        self._allow_non_ascii: bool = config.allow_non_ascii
        self._seed: int = config.seed
        self._seq_len: int = config.seq_len
        self._loss_temperature: float = config.loss_temperature
        self._max_queries: float = config.max_queries
        self._add_space: bool = config.add_space
        self._loss_func: str = config.loss_func
        self._cw_margin: float = config.cw_margin
        self._monotonic: bool = config.monotonic
        self._early_stop: bool = config.early_stop

        if self._loss_func not in self.valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func: {self._loss_func}! Must be one of "
                f"{self.valid_loss_funcs}."
            )
        if config.skip_mode not in self.valid_skip_modes:
            raise ValueError(
                f"Invalid skip_mode: {config.skip_mode}! Must be one of "
                f"{self.valid_skip_modes}."
            )
        self._skip_mode = config.skip_mode
        self._skip_seen = config.skip_mode == "seen"
        self._skip_visited = self._skip_seen or config.skip_mode == "visited"

        # Used for skipping expensive initialization
        if lazy_init:
            return

        self._model: BaseModel = wrapped_model
        self._device = wrapped_model.device
        self._not_allowed_tokens = not_allowed_tokens.to(self._device)
        self._tokenizer = tokenizer
        self._suffix_manager = suffix_manager
        self._eval_suffix: Callable[[str], tuple[bool, list[str]]] = eval_fn
        self._log_file: Path = self._setup_log_file(config)

        # Runtime variables
        self._start_time = None
        self._step = None
        self._best_loss = None
        self._best_suffix = None
        self._num_queries: int = 0
        self._num_tokens: int = 0
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0
        self._cur_seq_len: int = self._seq_len

    def _setup_log_file(self, config):
        atk_name = str(self).replace(f"{self.name}_", "")
        if config.custom_name:
            atk_name += f"_{config.custom_name}"
        log_dir = Path(config.log_dir) / self.name / atk_name
        logger.info("Logging to %s", log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{config.sample_name}.jsonl"
        # Delete log file if it exists
        log_file.unlink(missing_ok=True)
        return log_file

    def _get_name_tokens(self) -> list[str]:
        """Create a name for this attack based on its parameters."""
        if self._init_suffix_len <= 0:
            init_suffix_len = len(self._adv_suffix_init.split())
        else:
            init_suffix_len = self._init_suffix_len
        atk_tokens = [self.name, f"len{init_suffix_len}"]
        if self._max_queries > 0:
            atk_tokens.append(f"{self._max_queries:g}query")
        else:
            atk_tokens.append(f"{self._num_steps}step")
        atk_tokens.extend(
            [
                f"bs{self._batch_size}",
                f"seed{self._seed}",
                f"l{self._seq_len}",
            ]
        )
        if "ce" in self._loss_func:
            atk_tokens.append(f"{self._loss_func}-t{self._loss_temperature}")
        else:
            atk_tokens.append(f"{self._loss_func}-{self._cw_margin}")
        if self._fixed_params:
            atk_tokens.append("static")
        if self._allow_non_ascii:
            atk_tokens.append("nonascii")
        if self._skip_mode != "none":
            atk_tokens.append(self._skip_mode)
        if self._monotonic:
            atk_tokens.append("mono")
        if self._add_space:
            atk_tokens.append("space")
        return atk_tokens

    def __str__(self):
        return "_".join(self._get_name_tokens())

    def _sample_updates(self, optim_ids, *args, **kwargs):
        raise NotImplementedError("_sample_updates not implemented")

    def _setup_run(
        self,
        *args,
        messages: list[Message] | None = None,
        target: str = "",
        adv_suffix: str = "",
        **kwargs,
    ) -> None:
        """Set up before each attack run."""
        _ = args, kwargs  # Unused
        self._start_time = time.time()
        self._num_queries = 0
        self._num_tokens = 0
        self._step = None
        self._best_loss, self._best_suffix = float("inf"), adv_suffix
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0
        self._cur_seq_len = self._seq_len
        # Try running this to catch length mismatch error
        self._suffix_manager.gen_eval_inputs(
            messages,
            adv_suffix,
            target,
            num_fixed_tokens=0,
            max_target_len=self._seq_len,
        )
        if self._fixed_params:
            self._model.set_prefix_cache(messages)

    def _on_step_begin(self, *args, **kwargs):
        """Exectued at the beginning of each step."""

    def _on_step_end(
        self, eval_input: ModelInputIds, is_success: bool, *args, **kwargs
    ) -> bool:
        """Exectued at the end of each step."""
        _ = eval_input, args, kwargs  # Unused
        if self._early_stop and is_success:
            logger.info("Successful suffix found. Exit early.")
            return False
        if self._num_queries >= self._max_queries > 0:
            logger.info("Max queries reached! Finishing up...")
            return False
        return True

    def _save_best(self, current_loss: float, current_suffix: str) -> None:
        """Save the best loss and suffix so far."""
        if current_loss < self._best_loss:
            self._best_loss = current_loss
            self._best_suffix = current_suffix

    def cleanup(self):
        """Clean up memory after run."""

    def _compute_suffix_loss(
        self, inputs: ModelInputIds | ModelInputs
    ) -> torch.Tensor:
        """Compute loss given multiple suffixes.

        Args:
            eval_input: Input to evaluate. Must be EvalInput.

        Returns:
            Tuple of logits and loss.
        """
        output = self._model.compute_suffix_loss(
            inputs,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
            max_target_len=self._cur_seq_len,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
        )
        self._num_queries += output.num_queries
        return output.losses

    def _compute_grad(
        self, eval_input: ModelInputIds, **kwargs
    ) -> torch.Tensor | None:
        # Does not need grad by default
        _ = eval_input, kwargs
        return None

    def _filter_suffixes(
        self,
        adv_suffix_ids: BatchTokenIds | None = None,
        adv_suffixes: list[str] | None = None,
        filtering_models: list[BaseModel] | None = None,
    ) -> tuple[BatchTokenIds, int]:
        """Filter out invalid adversarial suffixes."""
        skipped_suffixes = None
        if self._skip_visited:
            skipped_suffixes = self._visited_suffixes
        elif self._skip_seen:
            skipped_suffixes = self._seen_suffixes

        if adv_suffixes is not None:
            # Prefer adv_suffixes (string) over adv_suffix_ids when target and
            # proxy models use different tokenizer
            _adv_suffix_ids = None
        else:
            _adv_suffix_ids = adv_suffix_ids
        assert (_adv_suffix_ids is not None) ^ (adv_suffixes is not None), (
            _adv_suffix_ids,
            adv_suffixes,
        )

        # Filter suffixes based on given tokenizers
        if filtering_models is not None:
            is_valid = None
            for model in filtering_models:
                _is_valid = model.filter_suffixes(
                    suffix_ids=_adv_suffix_ids,
                    suffix=adv_suffixes,
                    skipped_suffixes=skipped_suffixes,
                )
                if is_valid is None:
                    is_valid = _is_valid
                else:
                    is_valid &= _is_valid
        else:
            is_valid = self._model.filter_suffixes(
                suffix_ids=_adv_suffix_ids,
                suffix=adv_suffixes,
                skipped_suffixes=skipped_suffixes,
            )
        num_valid = is_valid.int().sum().item()
        adv_suffix_ids = adv_suffix_ids[is_valid]
        orig_len = adv_suffix_ids.shape[1]
        batch_size = self._batch_size

        adv_suffix_ids = adv_suffix_ids[:, :orig_len]
        if num_valid < batch_size:
            # Pad adv_suffix_ids to original batch size
            batch_pad = torch.zeros(
                (batch_size - num_valid, orig_len),
                dtype=adv_suffix_ids.dtype,
                device=adv_suffix_ids.device,
            )
            adv_suffix_ids = torch.cat([adv_suffix_ids, batch_pad], dim=0)
            logger.debug(
                "%.3f of suffixes are invalid", 1 - num_valid / batch_size
            )
        else:
            # We have more valid samples than the desired batch size
            num_valid = batch_size
        adv_suffix_ids = adv_suffix_ids[:batch_size]

        if num_valid == 0:
            raise RuntimeError("No valid suffixes found!")
        assert adv_suffix_ids.shape == (batch_size, orig_len)
        return adv_suffix_ids, num_valid

    def _init_adv_suffix(self, messages: list[Message], target: str) -> str:
        # Setting up init suffix
        num_failed = 0
        adv_suffix = self._adv_suffix_init
        adv_suffix_ids = self._tokenizer(
            adv_suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        adv_suffix_ids.squeeze_(0)
        while True:
            if num_failed >= len(adv_suffix_ids):
                # This should never be reached because "!" x N should be valid
                raise RuntimeError("Invalid init suffix!")
            try:
                self._setup_run(
                    messages=messages, target=target, adv_suffix=adv_suffix
                )
            except LengthMismatchError as e:
                logger.warning('Failing with suffix: "%s"', adv_suffix)
                logger.warning(str(e))
                logger.warning("Retrying with a new suffix...")
                # Replace the last N tokens with dummy where N is the number of
                # failed trials so far + 1
                dummy = self._tokenizer(
                    "!", add_special_tokens=False
                ).input_ids[0]
                adv_suffix_ids[-num_failed - 1 :] = dummy
                adv_suffix = self._tokenizer.decode(
                    adv_suffix_ids, skip_special_tokens=True
                )
                num_failed += 1
                continue
            break

        assert adv_suffix_ids.ndim == 1, adv_suffix_ids.shape
        logger.debug("Initialized suffix with %d tokens.", len(adv_suffix_ids))
        logger.debug(
            "adv_suffix=%s, adv_suffix_ids=%s", adv_suffix, adv_suffix_ids
        )
        return adv_suffix

    @torch.no_grad()
    def run(self, messages: list[Message], target: str) -> AttackResult:
        """Run the attack."""
        if self._add_space:
            # target = "▁" + target
            target = " " + target
        adv_suffix = self._init_adv_suffix(messages, target)
        num_fixed_tokens = self._model.num_fixed_tokens

        # =============== Prepare inputs and determine slices ================ #
        # NOTE: if we allow variable length messages, we need to update this
        # inside the loop
        # TODO(future): this upper bounds target length for all steps after
        # including fine-tuning.
        eval_input = self._suffix_manager.gen_eval_inputs(
            messages,
            adv_suffix,
            target,
            num_fixed_tokens=num_fixed_tokens,
            max_target_len=self._seq_len,
        )
        eval_input.to("cuda")
        optim_slice = eval_input.optim_slice

        for i in range(self._num_steps):
            self._step = i
            self._on_step_begin()

            dynamic_input_ids = self._suffix_manager.get_input_ids(
                messages, adv_suffix, target
            )[0][num_fixed_tokens:]
            dynamic_input_ids = dynamic_input_ids.to("cuda")
            optim_ids = dynamic_input_ids[optim_slice]
            eval_input.dynamic_input_ids = dynamic_input_ids
            eval_input.suffix_ids = optim_ids

            # Compute grad as needed (None if no-grad attack)
            # pylint: disable=assignment-from-none
            token_grads = self._compute_grad(eval_input)

            # Sample new candidate tokens
            adv_suffix_ids = self._sample_updates(
                optim_ids=optim_ids, grad=token_grads, optim_slice=optim_slice
            )
            # Filter out "invalid" adversarial suffixes
            adv_suffix_ids, num_valid = self._filter_suffixes(adv_suffix_ids)
            adv_suffixes = self._tokenizer.batch_decode(
                adv_suffix_ids, skip_special_tokens=True
            )
            self._seen_suffixes.update(adv_suffixes)
            eval_input.suffix_ids = adv_suffix_ids

            # Compute loss on model
            loss = self._compute_suffix_loss(eval_input)

            # ========== Update the suffix with the best candidate ========== #
            idx = loss[:num_valid].argmin()
            adv_suffix = adv_suffixes[idx]
            current_loss = loss[idx].item()
            self._save_best(current_loss, adv_suffix)
            if current_loss > self._best_loss and self._monotonic:
                adv_suffix = self._best_suffix
            self._visited_suffixes.add(adv_suffix)

            if i % self._log_freq == 0:
                self._num_queries += 1
                result = self._eval_suffix(adv_suffix)
                # TODO(feature): Count num tokens overall
                self.log(
                    log_dict={
                        "loss": current_loss,
                        "best_loss": self._best_loss,
                        "passed": result[0],
                        "suffix": adv_suffix,
                        "generated": result[1][-1],  # last message
                        "num_cands": adv_suffix_ids.shape[0],
                    },
                )
            del token_grads, loss, dynamic_input_ids
            gc.collect()

            if not self._on_step_end(eval_input, not result[0]):
                break

        # Evaluate last suffix on target model (this step is redundant on
        # attacks that do not use proxy model).
        eval_input.suffix_ids = adv_suffix_ids[idx : idx + 1]
        loss = self._model.compute_suffix_loss(
            eval_input,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
            max_target_len=self._cur_seq_len,
        ).losses
        self._save_best(loss.min().item(), adv_suffix)

        attack_result = AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
        )
        self._step += 1
        return attack_result

    def log(
        self, step: int | None = None, log_dict: dict[str, Any] | None = None
    ) -> None:
        """Log data using logger from a single step."""
        step = self._step if step is None else step
        log_dict["mem"] = torch.cuda.max_memory_allocated() / 1e9
        log_dict["time_per_step"] = (time.time() - self._start_time) / (
            step + 1
        )
        log_dict["queries"] = self._num_queries
        log_dict["tokens"] = self._num_tokens
        message = ""
        for key, value in log_dict.items():
            if "loss" in key:
                try:
                    value = f"{value:.4f}"
                except TypeError:
                    pass
            elif key == "mem":
                value = f"{value:.2f}GB"
            elif key == "time_per_step":
                value = f"{value:.2f}s"
            message += f"{key}={value}, "
        logger.info("[step: %4d/%4d] %s", step, self._num_steps, message)
        log_dict["step"] = step
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_dict) + "\n")
