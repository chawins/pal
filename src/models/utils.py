import logging
import os

import fastchat
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.message import Message
from src.models.base import BaseModel
from src.models.cohere import CohereModel, CohereTokenizer
from src.models.openai import OpenAIModel
from src.models.togetherai import TogetherAIModel
from src.utils.suffix import SuffixManager
from src.utils.types import PrefixCache

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(
    model_name: str,
    use_system_instructions: bool = False,
    num_api_processes: int = 8,
    **kwargs,
) -> tuple[BaseModel, AutoTokenizer, SuffixManager]:
    """Load model, tokenizer, and suffix manager."""
    if any(
        kw in model_name for kw in ("gpt", "davinci", "cohere", "togetherai")
    ):
        template_name, model_path = model_name.split("@")
        if "cohere" in model_name:
            model_class = CohereModel
        elif "togetherai" in model_name:
            model_class = TogetherAIModel
        else:
            model_class = OpenAIModel
        wrapped_model = model_class(
            model_path,
            template_name=template_name,
            num_api_processes=num_api_processes,
            **kwargs,
        )
        tokenizer = wrapped_model.tokenizer
        suffix_manager = SuffixManager(
            tokenizer=tokenizer,
            use_system_instructions=use_system_instructions,
            conv_template=fastchat.conversation.get_conv_template(
                template_name
            ),
        )
        wrapped_model.suffix_manager = suffix_manager
        return wrapped_model, tokenizer, suffix_manager
    return _load_huggingface_model_and_tokenizer(
        model_name, use_system_instructions=use_system_instructions, **kwargs
    )


def _load_huggingface_model_and_tokenizer(
    model_name: str,
    tokenizer_path: str | None = None,
    device: str = "cuda:0",
    load_in_8bit: bool | None = None,
    use_system_instructions: bool = False,
    system_message: str | None = None,
    max_tokens: int = 512,
    temperature: float = 1.0,
    **kwargs,
):
    from src.models.huggingface import TransformersModel

    template_name, model_path = model_name.split("@")
    model_path = os.path.expanduser(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        load_in_8bit=load_in_8bit,
        **kwargs,
    )
    if not load_in_8bit or load_in_8bit is None:
        model = model.to(device, non_blocking=True)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer_path = tokenizer_path or model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, use_fast=False
    )

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    suffix_manager = SuffixManager(
        tokenizer=tokenizer,
        use_system_instructions=use_system_instructions,
        conv_template=fastchat.conversation.get_conv_template(template_name),
    )

    # TODO(robust): Don't hard code max_tokens in case target is long
    # Longest target in Toxicity scenario is ~40 tokens
    wrapped_model = TransformersModel(
        model_name,
        suffix_manager=suffix_manager,
        model=model,
        tokenizer=tokenizer,
        system_message=system_message,
        max_tokens=max_tokens,
        temperature=temperature,
        devices=[device],
    )

    return wrapped_model, tokenizer, suffix_manager


def batchify_kv_cache(prefix_cache, batch_size):
    batch_prefix_cache = []
    for k, v in prefix_cache:
        batch_prefix_cache.append(
            (k.repeat(batch_size, 1, 1, 1), v.repeat(batch_size, 1, 1, 1))
        )
    return tuple(batch_prefix_cache)


def get_nonascii_toks(tokenizer, device="cpu") -> torch.Tensor:
    logger.debug("Gathering non-ascii tokens...")

    # Hack to get non-ascii from Cohere. Cohere uses API tokenizer which is
    # way too slow to rerun every time.
    if isinstance(tokenizer, CohereTokenizer):
        return torch.tensor(tokenizer.non_ascii, device=device)

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    non_ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        try:
            tok = tokenizer.decode([i])
        except:  # noqa: E722, pylint: disable=bare-except
            # GPT tokenizer throws an error for some tokens
            # pyo3_runtime.PanicException: no entry found for key
            non_ascii_toks.append(i)
            continue
        if not is_ascii(tok):
            non_ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        non_ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        non_ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        non_ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        non_ascii_toks.append(tokenizer.unk_token_id)
    non_ascii_toks = list(set(non_ascii_toks))

    logger.debug("Finished getting non-ascii tokens.")
    return torch.tensor(non_ascii_toks, device=device)


def get_prefix_cache(
    suffix_manager: SuffixManager,
    model,
    tokenizer,
    messages: list[Message],
) -> PrefixCache:
    static_input_ids = suffix_manager.get_input_ids(messages, static_only=True)
    static_input_str = tokenizer.decode(
        static_input_ids, skip_special_tokens=True
    )
    logger.info("Fixed prefix: %s", static_input_str)
    num_static_tokens = len(static_input_ids)
    logger.info("Fixing the first %d tokens as prefix", num_static_tokens)
    logger.info("Caching prefix...")
    device = model.device if hasattr(model, "device") else model.module.device
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        input_embeds = embed_layer(static_input_ids.to(device)).unsqueeze(0)
        outputs = model(inputs_embeds=input_embeds, use_cache=True)
        prefix_cache = outputs.past_key_values
    return prefix_cache, num_static_tokens
