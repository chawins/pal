import copy
import dataclasses
import gc
import hashlib
import heapq
import logging
import time
from enum import Enum
from typing import Any, Callable

import torch
from fastchat import conversation
from ml_collections import ConfigDict

from src.attacks.base import AttackResult
from src.attacks.gcg import GCGAttack
from src.message import Message
from src.models.base import NaNLossError
from src.models.ensemble import EnsembleModel
from src.models.huggingface import FineTuneConfig, TransformersModel
from src.models.model_input import (
    LengthMismatchError,
    ModelInputIds,
    ModelInputs,
    SuffixIds,
)
from src.models.mp_huggingface import MpTransformersModel
from src.models.tokenizer import GptTokenizer
from src.models.utils import get_nonascii_toks
from src.utils.suffix import SuffixManager

logger = logging.getLogger(__name__)

_BIG_NUM = 1e6


@dataclasses.dataclass
class HeapItem:
    neg_loss: float
    suffix_ids: torch.Tensor
    suffix: str

    def __post_init__(self):
        # Type check
        assert isinstance(self.neg_loss, float)
        assert isinstance(self.suffix, str)
        assert isinstance(self.suffix_ids, torch.Tensor)
        assert self.suffix_ids.ndim == 1, self.suffix_ids.shape

    def __lt__(self, other) -> bool:
        return self.neg_loss < other.neg_loss

    def __le__(self, other) -> bool:
        return self.neg_loss <= other.neg_loss

    def __gt__(self, other) -> bool:
        return self.neg_loss > other.neg_loss

    def __ge__(self, other) -> bool:
        return self.neg_loss >= other.neg_loss

    def __eq__(self, other) -> bool:
        return self.neg_loss == other.neg_loss


class AttackVer(Enum):
    ONE = "v1"
    TWO = "v2"
    THREE = "v3"
    FOUR = "v4"

    @classmethod
    def from_str(cls, ver: str) -> "AttackVer":
        """Convert string to AttackVer."""
        if ver == "v1":
            return cls.ONE
        if ver == "v2":
            return cls.TWO
        if ver == "v3":
            return cls.THREE
        if ver == "v4":
            return cls.FOUR
        raise ValueError(f"Invalid attack version: {ver}")


class PalAttack(GCGAttack):
    """PAL Attack."""

    name: str = "pal"

    def __init__(
        self,
        config: ConfigDict,
        build_messages_fn: Callable[[str], tuple[list[Message], str]],
        *args,
        lazy_init: bool = False,
        use_system_instructions: bool = False,
        **kwargs,
    ) -> None:
        """Initialize Proxy attack."""
        checkpoint_paths, model_names = [], []
        for model_name in config.proxy_model:
            _, checkpoint_path = model_name.split("@")
            checkpoint_paths.append(checkpoint_path)
            model_names.append(model_name)
        self._use_ensemble = len(checkpoint_paths) > 1
        self._use_mp = self._use_ensemble and config.use_mp
        if self._use_ensemble:
            checkpoint_paths.sort()
            checkpoint_paths = str(tuple(checkpoint_paths))
            self._proxy_model_name: str = _naming_hash(checkpoint_paths)
        else:
            self._proxy_model_name: str = checkpoint_path.split("/")[-1]
        self._build_messages_fn = build_messages_fn
        self._finetune: bool = config.finetune
        self._proxy_loss_func = config.proxy_loss_func or config.loss_func
        if self._proxy_loss_func not in self.valid_loss_funcs:
            raise ValueError(
                f"Invalid loss_func: {self._proxy_loss_func}! Must be one of "
                f"{self.valid_loss_funcs}."
            )

        # Options: "ce-target", "kl-target", "kl-all"
        self._proxy_ft_loss: str = config.proxy_loss
        self._proxy_peft: str = config.peft
        self._proxy_dtype: str = config.proxy_dtype
        self._proxy_optimizer: str = config.proxy_optimizer
        self._lr: float = config.proxy_lr
        self._wd: float = config.proxy_wd
        self._gradient_clipping: float | None = config.gradient_clipping
        self._lr_schedule: str = config.lr_schedule
        # Batch size for fine-tuning proxy model
        self._proxy_tune_bs: int = config.proxy_tune_bs
        self._tune_seq_len: int = config.tune_seq_len
        self._proxy_tune_temperature: float = config.tune_temperature

        # Number of steps to fine-tune proxy model. Default to -1 = tune on all
        # available samples.
        self._proxy_tune_steps: int | None
        if config.proxy_tune_steps > 0:
            self._proxy_tune_steps = config.proxy_tune_steps
        else:
            self._proxy_tune_steps = None

        # Tune proxy every N step (default: 1 = tune every step)
        self._proxy_tune_period: int = config.proxy_tune_period
        # How many samples to query target model with at each attack step
        # (default: -1 = all samples).
        self._num_target_query_per_step = config.num_target_query_per_step
        # If True, save all past target queries for fine-tuning proxy model
        self._proxy_tune_on_past_samples: bool = config.tune_on_past_samples
        # If True, log target loss in addition to proxy loss (does not count
        # towards best_loss, success, or number of queries).
        self._log_target_loss: bool = config.log_target_loss

        # Ablation studies
        # Use normal random noise as proxy grad to simulate random candidate
        # token selection.
        self._rand_grad: bool = config.use_rand_grad
        # Remove proxy filtering step
        self._no_proxy_filter: bool = config.no_proxy_filter

        super().__init__(config, *args, lazy_init=lazy_init, **kwargs)

        if lazy_init:
            return

        # Assume that all proxy models use the same fine-tuning config
        ft_config = None
        if self._finetune:
            ft_config = FineTuneConfig(
                batch_size=self._proxy_tune_bs,
                optimizer=self._proxy_optimizer,
                lr=self._lr,
                wd=self._wd,
                peft=self._proxy_peft,
                output_dir=str(self._log_file.parent),
                loss_name=self._proxy_ft_loss,
                max_target_len=self._tune_seq_len,
                gradient_clipping=self._gradient_clipping,
                lr_schedule=self._lr_schedule,
            )

        proxy_models = []
        for i, model_name in enumerate(model_names):
            proxy_model = _init_proxy_model(
                model_name,
                use_system_instructions=use_system_instructions,
                ft_config=ft_config,
                devices=f"cuda:{config.proxy_device[i]}",
                system_message=config.proxy_system_message[i],
                dtype=self._proxy_dtype,
            )
            proxy_model = MpTransformersModel(proxy_model, use_mp=self._use_mp)
            proxy_models.append(proxy_model)
        self._proxy_model = EnsembleModel(
            proxy_models, rendezvous_device=self._device
        )

        # Re-assign target model and proxy model
        self._tgt_model = self._model
        self._tgt_tokenizer = self._tokenizer
        self._tgt_suffix_manager = self._suffix_manager
        self._model = self._proxy_model
        self._rep_hf_model = self._proxy_model.models[0].hf_model
        self._tokenizer = self._rep_hf_model.tokenizer
        self._suffix_manager = self._rep_hf_model.suffix_manager
        # Set not allowed tokens according to proxy model instead of target
        self._not_allowed_tokens = get_nonascii_toks(
            self._rep_hf_model.tokenizer, device=self._device
        )

        # Run params
        self._proxy_loss = None  # Loss on proxy model
        self._tgt_loss = None  # Loss on target model
        # Cache for fine-tuning proxy model
        self._cached_samples: dict[str, Any] | None = None
        self._cur_loss_is_target: bool = False
        self._best_proxy_loss = _BIG_NUM
        self._best_proxy_suffix = None
        # Counter for proxy model steps. We can update attacks on proxy model
        # alone for a few steps before querying the target model.
        self._cur_num_proxy_steps = 0
        self._proxy_target = None  # Target string for proxy models
        self._tgt_target = None  # Target string for target model
        # Heap for keeping top-k candidates
        self._topk_heap: list[HeapItem] = []
        heapq.heapify(self._topk_heap)
        # Inputs for target model
        self._tgt_inputs: ModelInputs | None = None

    def cleanup(self):
        if isinstance(self._proxy_model, EnsembleModel):
            for model in self._proxy_model.models:
                model.stop()
                del model
        del self._proxy_model
        gc.collect()

    def _get_name_tokens(self) -> list[str]:
        atk_tokens = super()._get_name_tokens()
        atk_tokens.append(self._proxy_model_name)
        if self._proxy_loss_func != self._loss_func:
            atk_tokens.append(self._proxy_loss_func)
        if self._proxy_tune_period > 1:
            # Frequency of querying target model
            atk_tokens.append(f"f{self._proxy_tune_period}")
        if self._num_target_query_per_step > 0:
            # Number of samples to query target model per step
            atk_tokens.append(f"q{self._num_target_query_per_step}")

        if self._proxy_dtype == "int8":
            atk_tokens.append("int8")
        elif self._proxy_dtype == "float16":
            atk_tokens.append("fp16")
        elif self._proxy_dtype == "bfloat16":
            atk_tokens.append("bf16")

        if not self._finetune:
            atk_tokens.append("freeze")
        else:
            if self._proxy_optimizer != "adamw":
                atk_tokens.append(self._proxy_optimizer)
            atk_tokens.extend(
                [
                    f"lr{self._lr}",
                    f"wd{self._wd}",
                    f"bs{self._proxy_tune_bs}",
                    f"tl{self._tune_seq_len}",
                ]
            )
            if self._proxy_ft_loss != "ce":
                atk_tokens.append(self._proxy_ft_loss)
            if self._lr_schedule != "constant":
                atk_tokens.append(self._lr_schedule)
            if self._gradient_clipping is not None:
                atk_tokens.append(f"gc{self._gradient_clipping}")
            if self._proxy_tune_steps is not None:
                atk_tokens.append(f"{self._proxy_tune_steps}s")
            if self._proxy_peft != "none":
                atk_tokens.append(self._proxy_peft)
            if self._proxy_tune_on_past_samples:
                atk_tokens.append("past")
            if self._proxy_tune_temperature != 1.0:
                atk_tokens.append(f"ptt{self._proxy_tune_temperature}")

        if self._rand_grad:
            atk_tokens.append("rg")
        if self._no_proxy_filter:
            atk_tokens.append("np")
        return atk_tokens

    def _save_best(self, current_loss, current_suffix):
        # Use saved target loss/suffix instead
        _ = current_loss
        if not self._cur_loss_is_target:
            return
        if self._tgt_loss < self._best_loss:
            self._best_loss = self._tgt_loss
            self._best_suffix = current_suffix

    def _cache_samples(
        self,
        targets: list[str] | None = None,
        suffix_ids: list[SuffixIds] | None = None,
    ) -> None:
        """Save queried samples for fine-tuning proxy model."""
        if (
            self._proxy_tune_on_past_samples
            and self._cached_samples is not None
        ):
            # Keep concatenating samples for tuning
            self._cached_samples["suffix_ids"].extend(suffix_ids)
            self._cached_samples["targets"].extend(targets)
        else:
            self._cached_samples = {
                "suffix_ids": suffix_ids,
                "targets": targets,
            }

    def _setup_run(
        self,
        *args,
        messages: list[Message] | None = None,
        target: str = "",
        adv_suffix: str = "",
        **kwargs,
    ):
        _ = args, kwargs  # Unused
        assert target and adv_suffix, "target and adv_suffix must be set!"
        self._start_time = time.time()
        self._num_queries = 0
        self._step = None
        self._best_loss, self._best_suffix = float("inf"), adv_suffix
        self._seen_suffixes = set()
        self._visited_suffixes = set()
        self._num_repeated = 0
        self._proxy_loss, self._tgt_loss = None, None
        self._cached_samples = None
        self._best_proxy_loss, self._best_proxy_suffix = float("inf"), None
        self._cur_num_proxy_steps = 0

        self._proxy_target = target
        self._tgt_target = target
        if self._add_space:
            self._proxy_target = " " + self._proxy_target
            self._tgt_target = " " + self._tgt_target

        # Create a dummy suffix for now
        self._tgt_inputs = ModelInputs(
            messages=messages, targets=[self._tgt_target], suffixes=[]
        )
        self._topk_heap = []
        heapq.heapify(self._topk_heap)

        # Cache unchanged prefix for target model
        if self._fixed_params:
            self._tgt_model.set_prefix_cache(messages)

        # ========= Cache and get default eval input for proxy model ========= #
        num_opt_tokens = None
        for m in self._proxy_model.models:
            model: TransformersModel = m.hf_model
            # Set system message for proxy model
            proxy_messages, _ = self._build_messages_fn(model.system_message)
            if self._fixed_params:
                # Set prefix cache for proxy model
                model.set_prefix_cache(proxy_messages)
            eval_input = model.suffix_manager.gen_eval_inputs(
                proxy_messages,
                adv_suffix,
                self._proxy_target,
                num_fixed_tokens=model.num_fixed_tokens,
                max_target_len=max(self._seq_len, self._tune_seq_len),
            )
            eval_input.to(model.device)
            optim_slice = eval_input.optim_slice
            if num_opt_tokens is not None:
                if num_opt_tokens != optim_slice.stop - optim_slice.start:
                    raise ValueError(
                        "Number of optim tokens must be the same for all proxy "
                        f"models ({num_opt_tokens})!"
                    )
            num_opt_tokens = optim_slice.stop - optim_slice.start
            model.default_eval_input = eval_input

    def _update_suffix(
        self, model_input: ModelInputIds, num_valid: int
    ) -> tuple[str, float]:
        if self._no_proxy_filter:
            # Remove proxy filter step = set proxy loss to random number
            proxy_loss = torch.rand(num_valid, device=self._device)
        else:
            proxy_loss = self._compute_loss_proxy(model_input, num_valid)
        self._cur_num_proxy_steps += 1
        suffixes = to_string(self._tokenizer, model_input.suffix_ids)

        # Update the k lowest proxy losses
        for i, (loss, ids, suffix) in enumerate(
            zip(proxy_loss.tolist(), model_input.suffix_ids, suffixes)
        ):
            if i >= num_valid:
                break
            if not suffix:
                continue
            # Push to heap
            # NOTE: heappushpop() ends up comparing suffix ids, which is not
            # comparable. This should not happen because we already filter out
            # seen suffixes (chance of seeing exact same loss here should be
            # near zero), but perhaps, some self comparsion happens in pop().
            heap_item = HeapItem(-loss, ids, suffix)
            if len(self._topk_heap) < self._num_target_query_per_step:
                heapq.heappush(self._topk_heap, heap_item)
            else:
                heapq.heappushpop(self._topk_heap, heap_item)

        best_idx = proxy_loss.argmin().item()
        self._proxy_loss = proxy_loss[best_idx].item()
        self._best_proxy_loss = min(self._proxy_loss, self._best_proxy_loss)

        if self._cur_num_proxy_steps < self._proxy_tune_period:
            # Continue to query proxy model alone and not the target model yet
            self._cur_loss_is_target = False
            return suffixes[best_idx], self._proxy_loss

        # Get top-k suffixes for querying target model
        suffixes = [heap_item.suffix for heap_item in self._topk_heap]
        suffix_ids = [heap_item.suffix_ids for heap_item in self._topk_heap]
        # Reset heap and proxy step counter
        self._cur_num_proxy_steps = 0
        self._topk_heap = []
        heapq.heapify(self._topk_heap)
        # Add suffixes seen by proxy model
        self._seen_suffixes.update(suffixes)

        # =================== Compute loss on target model =================== #
        # Prepare queries for target model
        # Allows computing loss based on target STRING instead of token ids
        # in case not all strings tokenized to same length
        self._tgt_inputs.suffixes = suffixes
        tgt_outputs = self._tgt_model.compute_suffix_loss(
            self._tgt_inputs,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
            max_target_len=self._cur_seq_len,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
        )
        targets, tgt_loss = tgt_outputs.texts, tgt_outputs.losses
        tgt_logits = tgt_outputs.logits
        self._num_queries += tgt_outputs.num_queries
        self._num_tokens += tgt_outputs.num_tokens
        # Without this line, we can all zeros logits and losses. Not sure what
        # root cause is.
        if "cuda" in self._device:
            torch.cuda.synchronize()

        # Cache queried samples for fine-tuning proxy model
        if self._finetune:
            if tgt_logits is not None:
                # Convert logits to strings
                targets = to_string(self._tgt_tokenizer, tgt_logits.argmax(-1))
            assert len(targets) == len(suffixes), (len(targets), len(suffixes))
            logger.debug("Fine-tune target strings:\n%s", "\n".join(targets))
            self._cache_samples(targets=targets, suffix_ids=suffix_ids)

        # Save current losses for logging
        best_idx = tgt_loss.argmin().item()
        next_suffix = suffixes[best_idx]
        self._tgt_loss = tgt_loss[best_idx].item()
        self._visited_suffixes.add(next_suffix)
        self._cur_loss_is_target = True
        self._save_best(None, next_suffix)
        del tgt_logits, tgt_loss, tgt_outputs
        if self._tgt_loss > self._best_loss and self._monotonic:
            return self._best_suffix, self._tgt_loss
        return next_suffix, self._tgt_loss

    def _compute_loss_proxy(
        self, proxy_inputs: ModelInputIds, num_valid: int
    ) -> torch.Tensor:
        """Compute loss on proxy model."""
        proxy_loss = self._proxy_model.compute_suffix_loss(
            proxy_inputs,
            batch_size=self._mini_batch_size,
            max_target_len=self._cur_seq_len,
            loss_func=self._proxy_loss_func,
            temperature=self._loss_temperature,
            cw_margin=self._cw_margin,
        ).losses
        proxy_loss[num_valid:] = _BIG_NUM
        self._proxy_loss = proxy_loss.min().item()
        return proxy_loss

    @torch.no_grad()
    def run(self, messages: list[Message], target: str) -> AttackResult:
        """Run the attack."""
        # Swap tokenizer just for the initialization
        proxy_tokenizer = self._tokenizer
        self._tokenizer = self._tgt_tokenizer
        adv_suffix = self._init_adv_suffix(messages, target)
        self._tokenizer = proxy_tokenizer
        num_fixed_tokens = self._rep_hf_model.num_fixed_tokens

        # =============== Prepare inputs and determine slices ================ #
        # Need to get new messages for proxy model
        # TODO(proxy): Assume all proxy models use the same system message
        messages, _ = self._build_messages_fn(self._rep_hf_model.system_message)

        def gen_eval_input(_adv_suffix):
            _eval_input = self._suffix_manager.gen_eval_inputs(
                messages,
                _adv_suffix,
                self._proxy_target,
                num_fixed_tokens=num_fixed_tokens,
                max_target_len=self._seq_len,
            )
            _eval_input.to(self._device)
            return _eval_input

        eval_input = gen_eval_input(adv_suffix)
        optim_slice = eval_input.optim_slice
        prev_adv_suffix = adv_suffix

        eval_input.suffix_ids = eval_input.suffix_ids.unsqueeze(0)
        proxy_loss = self._compute_loss_proxy(eval_input, 1)
        self._proxy_loss = proxy_loss[0].item()
        self._tgt_inputs.suffixes = [adv_suffix]
        tgt_outputs = self._tgt_model.compute_suffix_loss(
            self._tgt_inputs,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
            max_target_len=self._cur_seq_len,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
        )
        self._num_queries += 1
        result = self._eval_suffix(adv_suffix)
        self._tgt_loss = tgt_outputs.losses[0].item()
        self._best_loss = self._tgt_loss
        self.log(
            step=0,
            log_dict={
                "loss": self._best_loss,
                "best_loss": self._best_loss,
                "passed": result[0],
                "suffix": adv_suffix,
                "generated": result[1][-1],  # last message
                "num_cands": 1,
            },
        )

        for i in range(self._num_steps):
            self._step = i
            self._on_step_begin()

            try:
                eval_input = gen_eval_input(adv_suffix)
                prev_adv_suffix = adv_suffix
            except LengthMismatchError:
                logger.warning(
                    "LengthMismatchError occurs on new adv_suffix. Reverting "
                    "to the previous adv_suffix..."
                )
                adv_suffix = prev_adv_suffix
                eval_input = gen_eval_input(prev_adv_suffix)
            dynamic_input_ids = eval_input.dynamic_input_ids
            optim_slice = eval_input.optim_slice
            optim_ids = dynamic_input_ids[optim_slice]

            # Compute grad as needed (None if no-grad attack)
            if self._rand_grad:
                token_grads = torch.randn(
                    (
                        optim_slice.stop - optim_slice.start,
                        len(self._tokenizer),
                    ),
                    device=self._device,
                )
            else:
                # pylint: disable=assignment-from-none
                token_grads = self._compute_grad(eval_input)

            # Sample new candidate tokens
            adv_suffix_ids = self._sample_updates(
                optim_ids=optim_ids, grad=token_grads, optim_slice=optim_slice
            )
            # DEPRECATED: Filter out "invalid" adversarial suffixes with both
            # proxy and target tokenizers
            # adv_suffixes = self._tokenizer.batch_decode(
            #     adv_suffix_ids, skip_special_tokens=True
            # )
            # Filter based on proxy model alone should be fine
            adv_suffix_ids, num_valid = self._filter_suffixes(
                adv_suffix_ids=adv_suffix_ids,
                # adv_suffixes=adv_suffixes,
                # filtering_models=[self._model, self._tgt_model],
            )
            eval_input.suffix_ids = adv_suffix_ids

            # Update suffix
            adv_suffix, current_loss = self._update_suffix(
                eval_input, num_valid
            )

            if i % self._log_freq == 0:
                self._num_queries += 1
                result = self._eval_suffix(adv_suffix)
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

            del token_grads, dynamic_input_ids, adv_suffix_ids, optim_ids
            gc.collect()

            if not self._on_step_end(eval_input, not result[0]):
                break

        attack_result = AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
        )
        self._step += 1
        return attack_result

    def _on_step_end(self, eval_input: ModelInputIds, *args, **kwargs) -> bool:
        return_val = super()._on_step_end(eval_input, *args, **kwargs)
        if not self._finetune or not self._cur_loss_is_target:
            return return_val
        try:
            self._proxy_model.finetune(
                suffix_ids=torch.vstack(self._cached_samples["suffix_ids"]),
                targets=self._cached_samples["targets"],
                ft_steps=self._proxy_tune_steps,
                temperature=self._proxy_tune_temperature,
            )
        except NaNLossError:
            logger.warning("NaN loss detected. Exiting...")
            return False
        return return_val

    def log(
        self, step: int | None = None, log_dict: dict[str, Any] | None = None
    ) -> None:
        new_log_dict = copy.deepcopy(log_dict)
        new_log_dict["proxy_loss"] = self._proxy_loss
        new_log_dict["target_loss"] = self._tgt_loss
        if "loss" in new_log_dict:
            new_log_dict.pop("loss")
        super().log(step=step, log_dict=new_log_dict)


def to_string(tokenizer, token_ids_or_logits: torch.Tensor) -> str | list[str]:
    # Convert to token ids
    if token_ids_or_logits.dtype != torch.long:
        token_ids = token_ids_or_logits.argmax(-1)
    else:
        token_ids = token_ids_or_logits
    assert token_ids.ndim in (1, 2), token_ids.shape
    # Convert to strings
    if token_ids.ndim == 1:
        strings = [tokenizer.decode(token_ids, skip_special_tokens=True)]
        first_toks = [token_ids[0]]
    else:
        strings = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
        first_toks = token_ids[:, 0]

    # Add metaspace if tokenizer dropped it
    new_strings = [""] * len(strings)
    num_spaces_added = 0
    for i, s in enumerate(strings):
        if tokenizer.decode(first_toks[i]) == "" and s[0] != " ":
            # Llama tokenizer removes the leading meta-space token; add it back
            new_strings[i] = "▁" + s
            num_spaces_added += 1
        elif isinstance(tokenizer, GptTokenizer) and s[0] == " ":
            new_strings[i] = s[1:]
        else:
            new_strings[i] = s

    if num_spaces_added > 0:
        logger.debug(
            "%d token_ids start with space. Adding space dropped by tokenizer.",
            num_spaces_added,
        )

    if len(new_strings) == 1:
        return new_strings[0]
    return new_strings


def _naming_hash(obj: str, length: int = 4) -> str:
    """Generate a hash from a string.

    Args:
        obj: Object to hash. Must be string.
        length: Length of hash to return. Defaults to 4.

    Returns:
        Hash string.
    """
    return hashlib.sha512(obj.encode("utf-8")).hexdigest()[:length]


def _init_proxy_model(
    model_name: str,
    use_system_instructions: bool = False,
    ft_config: FineTuneConfig | None = None,
    **kwargs,
) -> TransformersModel:
    """Initialize proxy model."""
    logger.info("Loading proxy model %s...", model_name)
    template_name, _ = model_name.split("@")

    # TODO(future): Proxy models are not used for generation so generation
    # params here are unused. Set max tokens if used.
    proxy_model = TransformersModel(
        model_name,
        temperature=0.0,
        max_tokens=32,
        suffix_manager=None,
        **kwargs,
    )
    if ft_config is None:
        for param in proxy_model.model.parameters():
            param.requires_grad = False

    # Creating suffix manager
    suffix_manager = SuffixManager(
        tokenizer=proxy_model.tokenizer,
        use_system_instructions=use_system_instructions,
        conv_template=conversation.get_conv_template(template_name),
    )
    proxy_model.suffix_manager = suffix_manager
    return proxy_model
