import dataclasses
import logging

import torch

from src.message import Message
from src.models.base import BaseModel, LossOutput
from src.models.model_input import ModelInputIds
from src.models.mp_huggingface import MpTransformersModel
from src.utils.types import BatchTokenIds

EnsembleEvalInput = list[ModelInputIds]

logger = logging.getLogger(__name__)


def _to_token_ids(
    worker: MpTransformersModel, strings: list[str], max_length: int
) -> BatchTokenIds:
    # TODO(tokenizer): Targets and suffixes may end up with padding. This may
    # lead to sub-optimal performance when target and proxy models use different
    # tokenizers.
    token_ids = worker.hf_model.tokenizer(
        strings,
        add_special_tokens=False,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    ).input_ids
    assert (
        token_ids.ndim == 2 and token_ids.shape[1] == max_length
    ), token_ids.shape
    return token_ids


class EnsembleModel(BaseModel):
    """Ensemble of MpTransformersModel models."""

    supports_system_message = True

    def __init__(
        self,
        models: list[MpTransformersModel] | None = None,
        rendezvous_device: str = "cuda:0",
        **kwargs,
    ) -> None:
        """Initialize ensemble of models.

        NOTE: We assume all models use the same tokenizer. Otherwise, averaging
        over logits and gradients will not work. Need optim slice to all have
        the same length.

        Args:
            models: List of models to ensemble.

        Raises:
            ValueError: If any model is not MpTransformersModel.
            ValueError: If no models are provided.
        """
        _ = kwargs  # unused
        assert models is not None
        self.models = models
        self._num_models = len(models)
        if not all(isinstance(m, MpTransformersModel) for m in models):
            raise ValueError("All models must be MpTransformersModel.")
        if self._num_models <= 0:
            raise ValueError("Must have at least 1 model.")
        self._rendezvous_device = rendezvous_device

    def __call__(
        self, messages: list[Message], api_key: str = None
    ) -> list[str]:
        for worker in self.models:
            worker(messages, api_key=api_key)
        out = []
        for worker in self.models:
            out.extend(worker.results.get())
        return out

    def _gen_proxy_eval_input(
        self,
        eval_input: ModelInputIds | None = None,
        suffix_ids: BatchTokenIds | None = None,
        suffixes: list[str] | None = None,
        targets: list[str] | None = None,
    ) -> EnsembleEvalInput:
        """Set eval_input for ensemble model."""
        assert not (
            suffix_ids is not None and suffixes is not None
        ), "Cannot provide both suffix_ids and suffixes!"
        if isinstance(suffixes, str):
            suffixes = [suffixes]
        if isinstance(targets, str):
            targets = [targets]

        proxy_eval_inputs = []
        for worker in self.models:
            assert worker.hf_model.default_eval_input is not None
            # Copy the defaul eval input
            if eval_input:
                _inpt = dataclasses.replace(eval_input)
            else:
                _inpt = dataclasses.replace(worker.hf_model.default_eval_input)
            # Replace suffix/target ids if strings are provided. This is used
            # because different models may have different tokenizers so we need
            # to convert from strings.
            if suffixes is not None:
                max_length = _inpt.optim_slice.stop - _inpt.optim_slice.start
                _inpt.suffix_ids = _to_token_ids(worker, suffixes, max_length)
            elif suffix_ids is not None:
                _inpt.suffix_ids = suffix_ids
            if targets is not None:
                max_length = _inpt.target_slice.stop - _inpt.target_slice.start
                _inpt.target_ids = _to_token_ids(worker, targets, max_length)
            proxy_eval_inputs.append(_inpt)

        return proxy_eval_inputs

    def set_prefix_cache(self, messages: list[Message]) -> None:
        """Set prefix KV-cache for model."""
        for worker in self.models:
            worker.set_prefix_cache(messages)

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
        return self.models[0].hf_model.filter_suffixes(
            suffix_ids=suffix_ids,
            suffix=suffix,
            skipped_suffixes=skipped_suffixes,
        )

    def compute_suffix_loss(
        self,
        inputs: ModelInputIds | EnsembleEvalInput,
        batch_size: int | None = None,
        suffixes: list[str] | None = None,
        targets: list[str] | None = None,
        **kwargs,
    ) -> LossOutput:
        """Compute loss given multiple suffixes.

        Args:
            inputs: Input to evaluate.
            batch_size: Optional batch size. Defaults to None (use all samples).

        Returns:
            Tuple of logits (over the entire input) and loss.
        """
        if not isinstance(inputs, list):
            inputs = self._gen_proxy_eval_input(
                eval_input=inputs, suffixes=suffixes, targets=targets
            )
        for inpt, worker in zip(inputs, self.models):
            logger.debug("Input to EnsembleModel.compute_suffix_loss:")
            logger.debug(inpt.print())
            worker.compute_suffix_loss(inpt, batch_size=batch_size, **kwargs)
        loss, num_queries = 0, 0
        for worker in self.models:
            outputs: LossOutput = worker.results.get()
            worker_loss = outputs.losses.to(self._rendezvous_device)
            num_queries += outputs.num_queries
            if loss is None:
                loss = worker_loss
            else:
                loss += worker_loss
        loss /= self._num_models
        # Logits can have different lengths because different models see
        # different input tokens. Just return None.
        return LossOutput(losses=loss, num_queries=num_queries)

    def compute_message_loss(
        self,
        messages: list[Message],
        suffixes: list[str],
        target: str,
        batch_size: int | None = None,
        **kwargs,
    ) -> LossOutput:
        raise NotImplementedError(
            "EnsembleModel does not implement compute_message_loss yet."
        )

    def compute_grad(
        self,
        eval_input: ModelInputIds | EnsembleEvalInput,
        suffixes: list[str] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute gradients w.r.t. `input_ids` tokens at `optim_slice`."""
        if not isinstance(eval_input, list):
            eval_input = self._gen_proxy_eval_input(
                eval_input=eval_input, suffixes=suffixes
            )
            # Gradients are computed over dynamic_input_ids so we have to also
            # update it here. There should be one suffix_ids here.
            for inpt in eval_input:
                inpt.dynamic_input_ids[inpt.optim_slice] = inpt.suffix_ids

        for inpt, worker in zip(eval_input, self.models):
            logger.debug("Input to EnsembleModel.compute_grad:")
            logger.debug(inpt.print())
            worker.compute_grad(inpt, **kwargs)
        out = sum(
            worker.results.get().to(self._rendezvous_device)
            for worker in self.models
        )
        out /= self._num_models
        return out

    def finetune(
        self,
        eval_input: ModelInputIds | EnsembleEvalInput | None = None,
        suffix_ids: BatchTokenIds | None = None,
        suffixes: list[str] | None = None,
        targets: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Fine-tune each proxy model independently."""
        if not isinstance(eval_input, list):
            eval_input = self._gen_proxy_eval_input(
                eval_input=eval_input,
                suffix_ids=suffix_ids,
                suffixes=suffixes,
                targets=targets,
            )
        for inpt, worker in zip(eval_input, self.models):
            logger.debug("Input to EnsembleModel.finetune:")
            logger.debug(inpt.print())
            worker.finetune(inpt, **kwargs)
