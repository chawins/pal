import gc
import logging

import torch

from src.attacks.base import AttackResult, BaseAttack
from src.message import Message
from src.models.model_input import ModelInputs

logger = logging.getLogger(__name__)


class BlackBoxAttack(BaseAttack):
    """Base class for black-box attacks."""

    def _sample_updates(self, optim_ids, *args, **kwargs):
        raise NotImplementedError("_sample_updates not implemented")

    def _update_suffix(self, inputs: ModelInputs) -> tuple[str, float]:
        # Filter out seen or visited suffixes
        outputs = self._model.compute_suffix_loss(
            inputs,
            batch_size=self._mini_batch_size,
            temperature=self._loss_temperature,
            max_target_len=self._cur_seq_len,
            loss_func=self._loss_func,
            cw_margin=self._cw_margin,
        )
        self._seen_suffixes.update(inputs.suffixes)
        self._num_queries += outputs.num_queries
        self._num_tokens += outputs.num_tokens
        losses = outputs.losses

        # Save current losses for logging
        min_idx = losses.argmin().item()
        current_loss = losses[min_idx].item()
        next_suffix = inputs.suffixes[min_idx]
        self._visited_suffixes.add(next_suffix)
        self._save_best(current_loss, next_suffix)
        del losses, outputs
        if current_loss > self._best_loss and self._monotonic:
            return self._best_suffix, current_loss
        return next_suffix, current_loss

    @torch.no_grad()
    def run(self, messages: list[Message], target: str) -> AttackResult:
        """Run the attack."""
        if self._add_space:
            target = " " + target
        adv_suffix = self._init_adv_suffix(messages, target)
        num_fixed_tokens = self._model.num_fixed_tokens

        # =============== Prepare inputs and determine slices ================ #
        model_input_ids = self._suffix_manager.gen_eval_inputs(
            messages,
            adv_suffix,
            target,
            num_fixed_tokens=num_fixed_tokens,
            max_target_len=self._seq_len,
        )
        optim_slice = model_input_ids.optim_slice

        for i in range(self._num_steps):
            self._step = i
            self._on_step_begin()

            dynamic_input_ids = self._suffix_manager.get_input_ids(
                messages, adv_suffix, target
            )[0][num_fixed_tokens:]
            dynamic_input_ids = dynamic_input_ids.to(self._device)
            optim_ids = dynamic_input_ids[optim_slice]

            # Sample new candidate tokens
            adv_suffix_ids = self._sample_updates(
                optim_ids=optim_ids, optim_slice=optim_slice
            )
            # Filter out "invalid" adversarial suffixes
            adv_suffix_ids, num_valid = self._filter_suffixes(adv_suffix_ids)
            adv_suffixes = self._tokenizer.batch_decode(
                adv_suffix_ids[:num_valid], skip_special_tokens=True
            )

            # Update suffix
            model_inputs = ModelInputs(
                messages=messages, suffixes=adv_suffixes, targets=[target]
            )
            adv_suffix, current_loss = self._update_suffix(model_inputs)

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

            del dynamic_input_ids, adv_suffix_ids, optim_ids
            gc.collect()

            if not self._on_step_end(model_inputs, not result[0]):
                break

        attack_result = AttackResult(
            best_loss=self._best_loss,
            best_suffix=self._best_suffix,
            num_queries=self._num_queries,
        )
        self._step += 1
        return attack_result
