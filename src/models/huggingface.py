import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import List

# from accelerate import Accelerator
import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn.functional as F
import torch_optimizer
import transformers
from llama_recipes.policies import AnyPrecisionAdamW
from llama_recipes.utils.config_utils import generate_peft_config
from peft import get_peft_model, prepare_model_for_kbit_training
from torch.optim.lr_scheduler import StepLR

from src.message import Message
from src.models.base import BaseModel, LossOutput, NaNLossError
from src.models.llama2_train_config import train_config as TRAIN_CONFIG
from src.models.model_input import ModelInputIds, ModelInputs
from src.models.tokenizer import Llama3Tokenizer
from src.models.utils import batchify_kv_cache, get_prefix_cache
from src.utils.suffix import SuffixManager, build_prompt
from src.utils.types import BatchTokenIds, PrefixCache

logger = logging.getLogger(__name__)

Device = int | str | torch.device
Devices = list[Device] | tuple[Device]


@dataclass
class FineTuneConfig:
    """Fine-tuning config."""

    batch_size: int = 32
    optimizer: str = "adamw"
    lr: float = 1e-4
    wd: float = 0.0
    peft: str = "none"
    quantize: bool = False
    output_dir: str | Path = "./finetune/"
    loss_name: str = "ce"  # Loss config
    max_target_len: int = 32  # Sequence length for fine-tuning
    gradient_clipping: float | None = None
    lr_schedule: str = "cosine"


class TransformersModel(BaseModel):
    """Model builder for HuggingFace Transformers model.

    `model` should be in the format model_name@checkpoint_path.

    Call with a list of `Message` objects to generate a response.
    """

    supports_system_message = True
    available_peft = ("none", "noembed", "lora")

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        stream: bool = False,
        top_p: float = 1.0,
        max_tokens: int = 512,
        stop=None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        model: transformers.AutoModelForCausalLM | None = None,
        tokenizer: transformers.AutoTokenizer | None = None,
        suffix_manager: SuffixManager | None = None,
        ft_config: FineTuneConfig | None = None,
        devices: Device | Devices | None = None,
        system_message: str | None = None,
        dtype: str = "float32",
        **kwargs,
    ):
        model_name, checkpoint_path = model_name.split("@")
        self.model_name = model_name

        # Generation parameters
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.temperature = temperature
        self.stream = stream
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._stop = stop
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.suffix_manager = suffix_manager
        self.system_message = system_message
        self._dtype = dtype
        if self._dtype not in ("float32", "float16", "bfloat16", "int8"):
            raise ValueError(f"Unknown dtype: {self._dtype}!")

        quant = False
        if self._dtype == "int8":
            assert ft_config is not None, (
                "int8 dtype is only used when finetune is True. We recommend "
                "float16 instead when not fine-tuning."
            )
            quant = True

        # Parse devices
        if devices is None:
            devices = ["cuda"]
        elif isinstance(devices, Device):
            devices = [devices]
        self.device = str(model.device if model is not None else devices[0])

        # accelerator = Accelerator(mixed_precision="fp16")  # EDIT
        self._use_mixed_precision = False
        if ft_config is not None:
            self._use_mixed_precision = (
                self._dtype == "float16" and ft_config.optimizer == "adamw"
            )

        if model is not None:
            logger.info("Model is specified and already initialized.")
            self.model = model
            assert (
                tokenizer is not None
            ), "tokenizer must be provided if model is provided."
            self.tokenizer = tokenizer
        else:
            logger.info("Model is not specified.")
            logger.info("Initializing a new one from %s...", checkpoint_path)

            if self._dtype == "bfloat16":
                model_dtype = torch.bfloat16
            elif self._dtype == "float16" and not self._use_mixed_precision:
                model_dtype = torch.float16
            else:
                # If use_fp16 is True, autocast will take care of dtype.
                # Setting model_dtype to float16 manually will cause error.
                model_dtype = torch.float32

            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                self.checkpoint_path,
                torch_dtype=model_dtype,
                low_cpu_mem_usage=False,
                use_cache=True,
                load_in_8bit=True if quant else None,
                device_map="auto" if quant else None,
            )
            if not quant:
                self.model.to(self.device)
            else:
                assert len(devices) == 1, (
                    "4-bit and 8-bit bitsandbytes model is assigned a device "
                    "automatically. It does not support multi-GPU."
                )
            if "Meta-Llama-3" in self.checkpoint_path:
                # Llama-3's tokenizer on HuggingFace is not behaving correctly.
                # Encode and then decode "! ! !" removes all space in
                # transformers=4.42.
                self.tokenizer = Llama3Tokenizer()
            else:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    self.checkpoint_path
                )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # ==================== Deal with multi-GPU loading =================== #
        if len(devices) > 1:
            logger.info(
                "%d devices (%s) are specified. Using DataParallel...",
                len(devices),
                devices,
            )
            self.model = torch.nn.DataParallel(self.model, device_ids=devices)
            # Should be fine to have generate run on rank 0 only
            self.model.generate = self.model.module.generate
            embed_layer = self.model.module.get_input_embeddings()
            self.embed_layer = torch.nn.DataParallel(
                embed_layer, device_ids=devices
            )

            def get_input_embeddings():
                return self.embed_layer

            self.model.get_input_embeddings = get_input_embeddings
            self.embed_weights = self.embed_layer.module.weight.t().detach()
        else:
            self.embed_layer = self.model.get_input_embeddings()
            self.embed_weights = self.embed_layer.weight.t().detach()
        self.embed_layer.requires_grad_(False)

        # Dictionary containing batched prefix cache (key is batch size)
        self._batch_prefix_cache: dict[int, PrefixCache] = {}
        # Original unbatched prefix cache
        self.prefix_cache: PrefixCache | None = None
        self.num_fixed_tokens: int = 0
        self.default_eval_input: ModelInputIds | None = None

        # ============= Set up fine-tuning if config is provided ============= #
        self.cfg = ft_config
        if self.cfg is not None:
            # Save some fine-tuning params
            self._ft_batch_size: int = self.cfg.batch_size
            self._max_target_len: int = self.cfg.max_target_len
            self._grad_clip: float | None = self.cfg.gradient_clipping
            if self.cfg.peft not in self.available_peft:
                raise ValueError(
                    f"Unknown peft: {self.cfg.peft}. Available options are "
                    f"{self.available_peft}."
                )

            if quant:
                logger.info("Preparing proxy model for int8 training...")
                self.model = prepare_model_for_kbit_training(self.model)
                # Grad from quantized models is somehow float32 so we also need
                # embed weights to be float32 for compute_grad().
                self.embed_weights = self.embed_weights.float()
                assert (
                    "lora" in self.cfg.peft
                ), "LoRa is required for int8 training."

            if "lora" in self.cfg.peft:
                logger.info("Preparing proxy model for PEFT...")
                # Using LoRA by default
                # TODO(lora): Set up more general training script with FSDP
                train_config = TRAIN_CONFIG()
                train_config.model_name = checkpoint_path
                train_config.batch_size_training = self._ft_batch_size
                train_config.use_peft = True
                train_config.output_dir = str(self.cfg.output_dir)
                train_config.quantization = quant
                train_config.one_gpu = True  # TODO(lora): fix this

                peft_config = generate_peft_config(train_config, kwargs)
                self.model = get_peft_model(self.model, peft_config)
                self.model.print_trainable_parameters()

            if "noembed" in self.cfg.peft:
                self.model.model.embed_tokens.requires_grad_(False)

            # Set up optimizer
            # Parameters taken from Llama-2 paper
            adam_params = {
                "lr": self.cfg.lr,
                "betas": (0.9, 0.95),
                "eps": 1e-5,
                "weight_decay": self.cfg.wd,
            }
            if self._dtype == "bfloat16" and self.cfg.optimizer == "adamw":
                logger.info("pure_bf16 is True. Using AnyPrecisionAdamW...")
                self.optim = AnyPrecisionAdamW(
                    self.model.parameters(),
                    **adam_params,
                    momentum_dtype=torch.bfloat16,
                    variance_dtype=torch.bfloat16,
                    use_kahan_summation=False,
                )
            elif self.cfg.optimizer == "adamw8bit":
                logger.info("Using AdamW8bit...")
                orig_embed = self.model.model.embed_tokens
                new_embed = bnb.nn.StableEmbedding(
                    *orig_embed.weight.shape,
                    padding_idx=orig_embed.padding_idx,
                    device=orig_embed.weight.device,
                    max_norm=orig_embed.max_norm,
                )
                new_embed.weight.data = orig_embed.weight.clone()
                new_embed.weight.requires_grad_("noembed" not in self.cfg.peft)
                self.model.model.embed_tokens = new_embed
                del orig_embed
                self.optim = bnb.optim.AdamW8bit(
                    self.model.parameters(), **adam_params
                )
            elif self.cfg.optimizer == "adafactor":
                logger.info("Using default Adafactor...")
                self.optim = torch_optimizer.Adafactor(
                    self.model.parameters(),
                    lr=self.cfg.lr,
                    weight_decay=self.cfg.wd,
                )
            else:
                logger.info("Using default AdamW...")
                self.optim = torch.optim.AdamW(
                    self.model.parameters(), **adam_params
                )

            # Set up lr scheduler
            if self.cfg.lr_schedule == "constant":
                self.lr_scheduler = StepLR(self.optim, 1, gamma=1.0)
            else:
                raise NotImplementedError(
                    f"Unknown lr_schedule: {self.cfg.lr_schedule}"
                )

            # EDIT
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # Does not work. torch._dynamo.exc.TorchRuntimeError
            # self.model = torch.compile(self.model)
        #     self.model, self.optim = accelerator.prepare(self.model, self.optim)
        # else:
        #     self.model = accelerator.prepare(self.model)

        # EDIT
        # Turn off bfloat16
        # self.accelerator = accelerator
        # self.device = accelerator.device
        # self.device = self.model.device
        # self.embed_layer = self.embed_layer.to(self.device)
        # self.embed_weights = self.embed_weights.to(self.device)

        self.model.eval()

    def __call__(
        self,
        inputs: List[Message] | list[str] | torch.Tensor | None = None,
        api_key: str = None,
    ):
        # TODO(feature): This is not working yet. Need to add batch generation.
        if isinstance(inputs[0], Message):
            # Turn messages into strings
            inputs = [build_prompt(inputs, self.model_name)]
        if isinstance(inputs[0], str):
            # Turn strings to token ids
            model_inputs = self.tokenizer(
                inputs, return_tensors="pt", padding=True
            )
        else:
            # Assume inputs are token ids
            model_inputs = {
                "input_ids": inputs,
                "attention_mask": torch.ones_like(inputs, dtype=torch.long),
            }
        model_inputs["input_ids"] = model_inputs["input_ids"].to(self.device)
        model_inputs["attention_mask"] = model_inputs["attention_mask"].to(
            self.device
        )
        prompt_len = model_inputs["attention_mask"].sum(dim=1)
        output = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        response = self.tokenizer.decode(
            output[0][prompt_len:], skip_special_tokens=True
        )
        return [response]

    @torch.no_grad()
    def _get_batch_prefix_cache(self, batch_size: int) -> PrefixCache:
        if self.prefix_cache is None:
            return None
        if batch_size not in self._batch_prefix_cache:
            self._batch_prefix_cache[batch_size] = batchify_kv_cache(
                self.prefix_cache, batch_size
            )
        return self._batch_prefix_cache[batch_size]

    @torch.no_grad()
    def set_prefix_cache(self, messages: list[Message]) -> None:
        self.prefix_cache, self.num_fixed_tokens = get_prefix_cache(
            self.suffix_manager, self.model, self.tokenizer, messages
        )
        # Reset batched prefix cache
        self._batch_prefix_cache = {}

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
            device = suffix_ids.device
            _, orig_len = suffix_ids.shape
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

    def compute_suffix_loss(
        self,
        inputs: ModelInputIds | ModelInputs,
        loss_func: str = "ce-all",
        **kwargs,
    ) -> LossOutput:
        """Compute loss given multiple suffixes."""
        if "one" in loss_func:
            return self._compute_loss_one(inputs, loss_func=loss_func, **kwargs)
        if isinstance(inputs, ModelInputIds):
            return self._compute_loss_ids(inputs, loss_func=loss_func, **kwargs)
        return self._compute_loss_strings(inputs, loss_func=loss_func, **kwargs)

    def _compute_loss_strings(
        self,
        inputs: ModelInputs,
        batch_size: int | None = None,
        temperature: float = 1.0,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
        **kwargs,
    ) -> LossOutput:
        _ = kwargs  # Unused
        messages: list[Message] = inputs.messages
        suffixes: list[str] = inputs.suffixes
        assert len(inputs.targets) == 1, "Only support single target for now."
        target: str = inputs.targets[0]
        num_samples = len(suffixes)
        batch_size = batch_size or num_samples
        batch_size = min(batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))

        # Get input ids for each suffix which may have different lengths
        input_ids_list, loss_starts, loss_slice = [], [], None
        for suffix in suffixes:
            out = self.suffix_manager.get_input_ids(
                messages, suffix, target, static_only=False
            )
            input_ids, _, _, loss_slice = out
            loss_start = loss_slice.start - self.num_fixed_tokens
            loss_starts.append(loss_start)
            input_ids_list.append(input_ids[self.num_fixed_tokens :])

        # Pad batch same size
        input_ids_list.extend(
            [input_ids_list[-1]] * (num_batches * batch_size - num_samples)
        )
        # pylint: disable=not-callable
        input_ids = torch.nested.nested_tensor(input_ids_list)
        input_ids = torch.nested.to_padded_tensor(
            input_ids, self.tokenizer.pad_token_id
        )
        loss_len = min(max_target_len, loss_slice.stop - loss_slice.start)
        loss_slice = (
            torch.tensor(loss_starts).unsqueeze(-1) + torch.arange(loss_len)
        ).long()
        loss_slice.unsqueeze_(-1)
        loss_slice = loss_slice.expand(
            num_samples, loss_len, len(self.tokenizer)
        )

        target_ids = self.tokenizer(
            target, add_special_tokens=False, return_tensors="pt"
        ).input_ids[:, :max_target_len]
        target_ids = target_ids.repeat(num_samples, 1).to(self.device)
        input_ids = input_ids.to(self.device)
        loss_slice = loss_slice.to(self.device)

        loss_list, logits_list = [], []
        for i in range(num_batches):
            batch_targets = target_ids[i * batch_size : (i + 1) * batch_size]
            logits, loss = self._compute_loss(
                input_ids[i * batch_size : (i + 1) * batch_size],
                batch_targets,
                loss_slice[i * batch_size : (i + 1) * batch_size],
                num_samples=len(batch_targets),
                temperature=temperature,
                loss_func=loss_func,
                cw_margin=cw_margin,
            )
            loss_list.append(loss)
            logits_list.append(logits)

        loss = torch.cat(loss_list, dim=0)
        logits = torch.cat(logits_list, dim=0)
        assert loss.shape == (num_samples,), loss.shape
        logits_shape = (num_samples, loss_len, len(self.tokenizer))
        assert logits.shape == logits_shape, logits.shape
        return LossOutput(
            losses=loss, logits=logits, num_queries=num_samples * loss_len
        )

    def _compute_loss_ids(
        self,
        inputs: ModelInputIds,
        batch_size: int | None = None,
        temperature: float = 1.0,
        max_target_len: int | None = None,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
        **kwargs,
    ) -> LossOutput:
        """Compute loss given multiple suffixes.

        Args:
            eval_input: Input to evaluate. Must be EvalInput.
            batch_size: Optional batch size. Defaults to None (use all samples).

        Returns:
            LossOutput object.
        """
        _ = kwargs  # Unused
        suffix_ids = inputs.suffix_ids
        dynamic_input_ids = inputs.dynamic_input_ids
        targets = inputs.target_ids
        optim_slice = inputs.optim_slice
        loss_slice = inputs.loss_slice
        orig_device = suffix_ids.device
        device = self.device

        if max_target_len is not None:
            # Adjust loss_slice, targets, and input_ids according to
            # most max_target_len
            loss_slice = slice(
                loss_slice.start,
                min(loss_slice.stop, loss_slice.start + max_target_len),
            )
            if targets.ndim == 1:
                targets = targets[:max_target_len]
            else:
                targets = targets[:, :max_target_len]
            dynamic_input_ids = dynamic_input_ids[: loss_slice.stop + 1]

        # Determine batch size and number of batches
        num_samples = len(suffix_ids)
        if batch_size is None:
            batch_size = num_samples
        else:
            batch_size = min(batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))

        # Device placement BEFORE batch loop. This should be fine since inputs
        # don't take much memory anyway.
        dynamic_input_ids = dynamic_input_ids.to(device)
        batch_dynamic_input_ids = dynamic_input_ids.repeat(batch_size, 1)
        # Expand and repeat batch dimension
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        if targets.shape[0] == 1:
            targets = targets.repeat(num_samples, 1)
        assert targets.ndim in (2, 3), targets.shape
        assert targets.shape[0] == num_samples, targets.shape

        loss_list, logits_list = [], []
        for i in range(num_batches):
            # Update suffixes
            batch_suffix_ids = suffix_ids[i * batch_size : (i + 1) * batch_size]
            batch_targets = targets[i * batch_size : (i + 1) * batch_size]
            batch_suffix_ids = batch_suffix_ids.to(device)
            batch_targets = batch_targets.to(device)
            bs = len(batch_targets)
            batch_dynamic_input_ids[:bs, optim_slice] = batch_suffix_ids
            logits, loss = self._compute_loss(
                batch_dynamic_input_ids,
                batch_targets,
                loss_slice,
                num_samples=bs,
                temperature=temperature,
                loss_func=loss_func,
                cw_margin=cw_margin,
            )
            loss_list.append(loss)
            logits_list.append(logits)

        loss = torch.cat(loss_list, dim=0).to(orig_device)
        logits = torch.cat(logits_list, dim=0).to(orig_device)
        assert loss.shape == (num_samples,), loss.shape
        logits_shape = (
            num_samples,
            loss_slice.stop - loss_slice.start,
            len(self.tokenizer),
        )
        assert logits.shape == logits_shape, logits.shape
        return LossOutput(losses=loss, logits=logits, num_queries=num_samples)

    def _compute_loss_one(
        self,
        inputs: ModelInputs | ModelInputIds,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
        k: int = 5,
        **kwargs,
    ) -> LossOutput:
        """Compute loss 1 target token at a time similar to OpenAI Chat API."""
        if isinstance(inputs, ModelInputIds):
            func = self._compute_loss_ids
            num_samples = len(inputs.suffix_ids)
            target_ids = inputs.target_ids.to(self.device)
            if target_ids.ndim == 2:
                assert target_ids.shape[0] == 1, target_ids.shape
                target_ids = target_ids[0]
        else:
            func = self._compute_loss_strings
            num_samples = len(inputs.suffixes)
            targets = inputs.targets
            assert len(targets) == 1, "Only support single target for now."
            target_ids = self.tokenizer(
                targets, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)[0]
        assert target_ids.ndim == 1, target_ids.shape
        max_target_len = min(max_target_len, len(target_ids))

        # Compute loss and logits as usual
        out = func(
            inputs,
            loss_func=loss_func,
            max_target_len=max_target_len,
            cw_margin=cw_margin,
            **kwargs,
        )

        # Re-calculate number of queries as if we use OpenAI Chat API
        logits = out.logits.to(self.device)
        num_queries = 0
        cur_is_success = torch.ones(
            num_samples, dtype=torch.bool, device=self.device
        )
        output_strs = [""] * num_samples
        new_losses = torch.zeros_like(out.losses, device=self.device)
        batch_zeros = torch.zeros(
            (num_samples, 1), dtype=torch.long, device=self.device
        )

        for i in range(max_target_len):
            target_tok_id = target_ids[i]
            target_tok = self.tokenizer.decode(
                target_tok_id, skip_special_tokens=True
            )
            # Single space is ignored by non-GPT tokenizer
            if target_tok == "":
                if i == 0:
                    # First token is space
                    output_strs = [" "] * num_samples
                    continue
                target_tok = " "
            num_queries += cur_is_success.sum().item()
            # Find how many with target_tok in top-5
            topk_tok_ids = logits[:, i, :].topk(k, dim=-1).indices

            # Update output strings
            top_toks = self.tokenizer.batch_decode(
                topk_tok_ids[:, 0], skip_special_tokens=True
            )
            for j, top_tok in enumerate(top_toks):
                if cur_is_success[j]:
                    output_strs[j] = f"{output_strs[j]} {top_tok}"

            is_top1 = cur_is_success & (topk_tok_ids[:, 0] == target_tok_id)
            is_top5 = cur_is_success & (topk_tok_ids == target_tok_id).any(1)
            num_top1 = is_top1.sum().item()
            num_top5 = is_top5.sum().item() - num_top1
            logger.debug(
                'target_tok="%s" (%d/%d): total=%d, top1=%d, top5=%d, other=%d',
                target_tok,
                i + 1,
                max_target_len,
                num_samples,
                num_top1,
                num_top5,
                num_samples - num_top1 - num_top5,
            )

            # Update loss
            if is_top5.any():
                # If target_tok is in top-5, we can directly access logprob of
                # target_tok so we compute loss as usual
                best_target_len = i + 1
                if num_top1 == 0:
                    loss_idx_to_update = is_top5
                else:
                    # There's at least one prompt with target_tok in top-1
                    loss_idx_to_update = is_top1
                batch_target = batch_zeros + target_tok_id
                if "ce" in loss_func:
                    loss = F.cross_entropy(
                        logits[:, i], batch_target.squeeze(1), reduction="none"
                    )
                else:
                    loss = _cw_loss(
                        logits[:, i : i + 1], batch_target, cw_margin=cw_margin
                    )
                new_losses[loss_idx_to_update] += loss[loss_idx_to_update]
            else:
                # No prompt with target_tok in top-5: estimate target_tok prob
                best_target_len = i
                loss_idx_to_update = cur_is_success
                # probs: [batch_size, vocab_size]
                probs = logits[:, i].softmax(dim=-1)[loss_idx_to_update]
                top5_probs = probs.topk(5, dim=-1).values
                est_target_probs = 1 - top5_probs.sum(-1)
                if "ce" in loss_func:
                    new_losses[loss_idx_to_update] = -est_target_probs.log()
                else:
                    top5_lprobs = top5_probs.log()
                    new_losses[loss_idx_to_update] = (
                        top5_lprobs[:, 0] - est_target_probs.log()
                    ).clamp_min(-cw_margin)

            cur_is_success = loss_idx_to_update
            if num_top1 == 0:
                break

        # Set loss of unsuccessful queries to inf
        new_losses[~cur_is_success] = float("inf")
        new_losses[cur_is_success] += (max_target_len - best_target_len) * 1e3

        return LossOutput(
            losses=new_losses, num_queries=num_queries, texts=output_strs
        )

    @torch.no_grad()
    def _compute_loss(
        self,
        batch_input_ids: BatchTokenIds,
        batch_targets: torch.Tensor,
        loss_slice: slice | torch.Tensor,
        num_samples: int | None = None,
        temperature: float = 1.0,
        loss_func: str = "ce-all",
        cw_margin: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = num_samples or len(batch_input_ids)
        input_embeds = self.embed_layer(batch_input_ids)

        # logits: [batch_size, seq_len, vocab_size]
        logits = self.model(
            inputs_embeds=input_embeds,
            past_key_values=self._get_batch_prefix_cache(len(batch_input_ids)),
        ).logits[:num_samples]

        # loss_logits: [batch_size, loss_len, vocab_size]
        if isinstance(loss_slice, slice):
            loss_logits = logits[:, loss_slice]
        else:
            loss_logits = logits.gather(1, loss_slice)

        if batch_targets.dtype == torch.long:
            # Hard-label target usually used for computing loss on target
            if "ce" in loss_func:
                loss = F.cross_entropy(
                    loss_logits.permute(0, 2, 1) / temperature,
                    batch_targets,
                    reduction="none",
                ).mean(dim=1)
            elif "cw" in loss_func:
                loss = _cw_loss(loss_logits, batch_targets, cw_margin=cw_margin)
            else:
                raise ValueError(f"Unknown loss_func: {loss_func}!")
        else:
            # Soft-label target usually used for training proxy model
            loss = F.kl_div(
                (loss_logits / temperature).log_softmax(dim=-1),
                batch_targets / temperature,
                reduction="none",
            )
            loss = loss.sum(dim=-1).mean(dim=1)
        assert loss.shape == (num_samples,), loss.shape
        return loss_logits, loss

    @torch.no_grad()
    def compute_grad(
        self,
        eval_input: ModelInputIds,
        max_target_len: int = 32,
        loss_func: str = "ce-all",
        temperature: float = 1.0,
        cw_margin: float = 1e-3,
        **kwargs,
    ) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`.

        Assume a single input sequence (no mini-batch).
        """
        _ = kwargs  # Unused
        input_ids = eval_input.dynamic_input_ids
        target_ids = eval_input.target_ids
        optim_slice = eval_input.optim_slice
        loss_slice = eval_input.loss_slice

        # Update loss/target slice with max_target_len
        loss_slice = slice(
            loss_slice.start,
            min(loss_slice.start + max_target_len, loss_slice.stop),
        )
        target_ids = target_ids[..., :max_target_len]

        orig_device = input_ids.device
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        if target_ids.ndim == 2:
            target_ids.squeeze_(0)
        input_embeds = self.embed_layer(input_ids)
        input_embeds.unsqueeze_(0)
        input_embeds.requires_grad_(True)
        with torch.enable_grad():
            # Forward pass
            logits = self.model(
                inputs_embeds=input_embeds,
                past_key_values=self._get_batch_prefix_cache(len(input_embeds)),
                use_cache=True,
            ).logits
            # Compute loss and gradients
            loss_logits = logits[:, loss_slice].squeeze(0)
            if "ce" in loss_func:
                loss = F.cross_entropy(loss_logits / temperature, target_ids)
            else:
                loss = _cw_loss(loss_logits, target_ids, cw_margin=cw_margin)
            embed_grads = torch.autograd.grad(
                outputs=[loss], inputs=[input_embeds]
            )[0]
        embed_grads.detach_()
        embed_grads = embed_grads[0, optim_slice]
        token_grads = embed_grads @ self.embed_weights
        token_grads /= token_grads.norm(dim=-1, keepdim=True)
        token_grads = token_grads.to(orig_device)

        assert token_grads.shape == (
            optim_slice.stop - optim_slice.start,
            len(self.tokenizer),
        ), token_grads.shape
        return token_grads

    def finetune(
        self,
        eval_inputs: ModelInputIds,
        ft_steps: int | None = None,
        temperature: float = 1.0,
        **kwargs,
    ) -> None:
        _ = kwargs  # Unused
        if self.cfg is None:
            logger.warning("No fine-tuning config provided!")
            return

        is_training = self.model.training
        self.model.train()
        adv_suffix_ids = eval_inputs.suffix_ids
        dynamic_input_ids = eval_inputs.dynamic_input_ids.to(self.device)
        loss_slice = eval_inputs.loss_slice
        target_slice = eval_inputs.target_slice
        assert len(dynamic_input_ids) >= loss_slice.stop, loss_slice
        assert len(dynamic_input_ids) >= target_slice.stop, target_slice

        num_ft_samples = len(adv_suffix_ids)
        assert num_ft_samples == len(
            eval_inputs.target_ids
        ), f"{num_ft_samples} vs {len(eval_inputs.target_ids)}"
        num_epochs = 1
        if ft_steps is not None:
            num_epochs = int(np.ceil(num_ft_samples / ft_steps))

        # Get targets in correct format and update loss/target slice
        targets = eval_inputs.target_ids[:, : self._max_target_len]
        # Update target and loss slices due to max_target_len
        new_target_len = targets.shape[1]
        loss_slice = slice(loss_slice.start, loss_slice.start + new_target_len)
        target_slice = slice(
            target_slice.start, target_slice.start + new_target_len
        )
        dynamic_input_ids = dynamic_input_ids[: target_slice.stop]

        # target_ids can be either token ids or logits
        if "ce" in self.cfg.loss_name:
            if targets.dtype != torch.long:
                targets = targets.argmax(dim=-1)
            assert targets.ndim == 2
        elif "kl" in self.cfg.loss_name:
            targets = F.softmax(targets, dim=-1)
            assert targets.ndim == 3
        else:
            raise ValueError(f"Unknown proxy_loss: {self.cfg.loss_name}!")

        scaler = torch.cuda.amp.GradScaler()
        autocast = (
            torch.cuda.amp.autocast
            if self._use_mixed_precision
            else nullcontext
        )

        logger.debug("Tuning proxy model for %d epochs...", num_epochs)
        for _ in range(num_epochs):
            sample_idx = np.arange(num_ft_samples)
            np.random.shuffle(sample_idx)
            for i in range(0, num_ft_samples, self._ft_batch_size):
                self.model.zero_grad()
                batch_idx = sample_idx[i : i + self._ft_batch_size]

                # Get new batch inputs for training
                batch_input = ModelInputIds(
                    dynamic_input_ids=dynamic_input_ids,
                    suffix_ids=adv_suffix_ids[batch_idx],
                    target_ids=targets[batch_idx],
                    loss_slice=loss_slice,
                    target_slice=target_slice,
                    optim_slice=eval_inputs.optim_slice,
                )
                batch_input.to(self.device)
                logger.debug(batch_input.print())

                with torch.enable_grad():
                    with autocast():
                        loss = self.compute_suffix_loss(
                            batch_input, temperature=temperature
                        ).losses.mean()
                    if loss.isnan():
                        raise NaNLossError("Fine-tuning loss is NaN!")

                    if self._use_mixed_precision:
                        # Use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if self._grad_clip is not None:
                            scaler.unscale_(self.optim)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self._grad_clip
                            )
                        scaler.step(self.optim)
                        scaler.update()
                    else:
                        loss.backward()
                        # self.accelerator.backward(loss)  # EDIT
                        if self._grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), self._grad_clip
                            )
                        self.optim.step()

                    self.optim.zero_grad()

                logger.info(
                    "  [samples %4d/%4d] training loss: %.6f",
                    i + len(batch_idx),
                    num_ft_samples,
                    loss.item(),
                )
            self.lr_scheduler.step()

        self.model.train(is_training)


def _cw_loss(
    logits: torch.FloatTensor,
    target_ids: torch.LongTensor,
    cw_margin: float = 1e-3,
    dim: int = -1,
) -> torch.FloatTensor:
    """CW loss.

    Hinge loss on the difference between the largest and the target logits.
    """
    input_shape = target_ids.shape
    assert logits.shape[:-1] == input_shape, (logits.shape, input_shape)
    target_ids = target_ids.unsqueeze(-1)
    tgt_logits = logits.gather(dim, target_ids).squeeze(-1)
    # Set logits of target tok very low (-1e3) so it cannot be the largest
    tmp_logits = logits.clone()
    tmp_logits.scatter_(dim, target_ids, -1e3)
    largest_non_tgt_logits = tmp_logits.max(dim).values
    loss = largest_non_tgt_logits - tgt_logits
    loss = loss.clamp_min(-cw_margin).mean(-1)
    if len(input_shape) == 1:
        assert loss.ndim == 0, loss.shape
    else:
        assert loss.shape == input_shape[:1], loss.shape
    return loss
