import logging
from typing import List

import torch
import torch.multiprocessing as mp

from src.message import Message
from src.models.base import BaseModel
from src.models.model_input import ModelInputIds

logger = logging.getLogger(__name__)


def _run(tasks, results):
    while True:
        task = tasks.get()
        if task is None:
            break
        model, fn, args, kwargs = task
        with torch.set_grad_enabled(fn == "compute_grad"):
            results.put(getattr(model, fn)(*args, **kwargs))
        tasks.task_done()


class DummyResults:
    """Dummy class to replace results queue when not using multiprocessing."""

    def __init__(self):
        self.val = None

    def put(self, val):
        self.val = val

    def get(self):
        if self.val is None:
            raise ValueError("No results to get!")
        val = self.val
        self.val = None
        return val

    def task_done(self):
        pass


class MpTransformersModel(BaseModel):
    """Wrapper around TransformersModel for multiprocessing."""

    supports_system_message = True

    def __init__(
        self,
        model,
        *args,
        use_mp: bool = True,
        **kwargs,
    ) -> None:
        _ = args, kwargs  # Unused
        logger.info("Initializing TransformersModel with multiprocessing...")
        self._use_mp = use_mp
        self.hf_model = model
        self.device = model.device
        self._tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue() if use_mp else DummyResults()
        self._process = None
        self.start()

    def start(self) -> None:
        if not self._use_mp:
            return
        self._process = mp.Process(
            target=_run, args=(self._tasks, self.results)
        )
        self._process.start()
        logger.info("Started worker %d", self._process.pid)

    def stop(self) -> None:
        if not self._use_mp:
            return
        self._tasks.put(None)
        if self._process is not None:
            self._process.join()
        torch.cuda.empty_cache()

    def __call__(self, messages: List[Message], api_key: str = None) -> None:
        if not self._use_mp:
            out = self.hf_model(messages, api_key=api_key)
            self.results.put(out)
            return
        self._tasks.put(
            (self.hf_model, "call", [messages], {"api_key": api_key})
        )

    def set_prefix_cache(self, messages: list[Message]) -> None:
        # Using queue here means that self.hf_model does not get updated, but we
        # need self.hf_model.prefix_cache to be updated.
        self.hf_model.set_prefix_cache(messages)

    def compute_suffix_loss(
        self, inputs: ModelInputIds, batch_size: int | None = None, **kwargs
    ) -> None:
        _ = kwargs  # Unused
        if not self._use_mp:
            out = self.hf_model.compute_suffix_loss(
                inputs, batch_size=batch_size, **kwargs
            )
            self.results.put(out)
            return

        self._tasks.put(
            (
                self.hf_model,
                "compute_suffix_loss",
                [inputs],
                {"batch_size": batch_size, **kwargs},
            )
        )

    def compute_grad(self, eval_input: ModelInputIds, **kwargs) -> None:
        if not self._use_mp:
            out = self.hf_model.compute_grad(eval_input, **kwargs)
            self.results.put(out)
            return
        self._tasks.put(
            (
                self.hf_model,
                "compute_grad",
                [eval_input],
                kwargs,
            )
        )

    def finetune(
        self, eval_inputs: ModelInputIds, ft_steps: int | None = None, **kwargs
    ) -> None:
        # self._tasks.put(
        #     (
        #         self.hf_model,
        #         "finetune",
        #         [eval_inputs],
        #         {"ft_steps": ft_steps},
        #     )
        # )
        self.hf_model.finetune(eval_inputs, ft_steps=ft_steps, **kwargs)
