"""Suffix manager for adversarial suffix generation."""

import dataclasses
import logging

import fastchat  # type: ignore
import torch
from fastchat.conversation import get_conv_template  # type: ignore

from src.message import Message, Role
from src.models.model_input import ModelInputIds

# Register new conv. pylint: disable=unused-import
from src.utils import cohere_conv, gpt_conv  # noqa: F401

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Llama3Conversation(fastchat.conversation.Conversation):
    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(
            system_message=self.system_message
        )
        # ret = "<|begin_of_text|>"  # This is added by tokenizer
        ret = ""
        if self.system_message:
            ret = system_prompt
        for role, message in self.messages:
            if message:
                ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                ret += f"{message.strip()}<|eot_id|>"
            else:
                ret += f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        return ret

    def copy(self):
        return Llama3Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )


fastchat.conversation.register_conv_template(
    Llama3Conversation(
        name="pal-llama-3",
        system_template="<|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|>",
        roles=("user", "assistant"),
        sep="",
        stop_str="<|eot_id|>",
        stop_token_ids=[128001, 128009],
    )
)


class SuffixManager:
    """Suffix manager for adversarial suffix generation."""

    valid_templates = (
        "llama-2",
        "pal-llama-3",
        "vicuna_v1.1",
        "mistral",
        "chatgpt",
        "completion",
        "raw",
        "openchat_3.5",
        "orca-2",
        "cohere",
        "mistral",
    )

    def __init__(self, *, tokenizer, use_system_instructions, conv_template):
        """Initialize suffix manager.

        Args:
            tokenizer: Tokenizer for model.
            use_system_instructions: Whether to use system instructions.
            conv_template: Conversation template.
        """
        self.tokenizer = tokenizer
        self.use_system_instructions = use_system_instructions
        self.conv_template = conv_template
        logger.info(
            "SuffixManager initialized with conv_template=%s, "
            "use_system_instructions=%s",
            self.conv_template.name,
            use_system_instructions,
        )

        self.num_tok_sep = len(
            self.tokenizer(
                self.conv_template.sep, add_special_tokens=False
            ).input_ids
        )
        if self.conv_template.name == "chatgpt":
            # Space is subsumed by following token in GPT tokenizer
            assert self.conv_template.sep == " ", self.conv_template.sep
            self.num_tok_sep = 0
        elif "llama-3" in self.conv_template.name:
            # FastChat adds <|eot_id|> after each message, but it's not sep.
            # Not exactly sure why, but not we need to manually set
            # self.num_tok_sep because sep is just "".
            # https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L167
            self.num_tok_sep = 1

        self.num_tok_sep2 = 0
        if self.conv_template.sep2 not in (None, ""):
            self.num_tok_sep2 = len(
                self.tokenizer(
                    self.conv_template.sep2, add_special_tokens=False
                ).input_ids
            )
        if self.conv_template.stop_str not in (None, ""):
            self.num_tok_sep2 += 1

    @torch.no_grad()
    def get_input_ids(
        self,
        messages: list[Message],
        adv_suffix: str | None = None,
        target: str | None = None,
        static_only: bool = False,
    ) -> tuple[torch.Tensor, slice, slice, slice]:
        """Turn messages into token ids.

        Compute token ids for given messages and target, along with slices
        tracking positions of important tokens.

        Args:
            messages: Messages in the conversation.
            adv_suffix: Current adversarial suffix.
            target: Current target output for model.
            static_only: If True, only return token ids for static tokens.

        Returns:
            input_ids: Token ids for messages and target.
            optim_slice: Slice of input_ids corresponding to tokens to optimize.
            target_slice: Slice of input_ids corresponding to target.
            loss_slice: Slice of input_ids corresponding to loss.
        """
        # This code was tested with llama-2 and vicuna_v1.1 templates but remove
        # this check to experiment with others.
        if self.conv_template.name not in self.valid_templates:
            raise NotImplementedError(
                f"{self.conv_template.name} is not implemented! Please use one "
                f"of {self.valid_templates}"
            )

        self.conv_template.messages = []

        if messages[0].content:
            self.conv_template.set_system_message(messages[0].content)

        user_msg = messages[1].content
        if len(messages) <= 2:
            # Toxicity scenario
            self.conv_template.append_message(
                self.conv_template.roles[0], messages[1].content
            )  # user rules
        else:
            if not self.use_system_instructions:
                self.conv_template.append_message(
                    self.conv_template.roles[0], messages[1].content
                )  # user rules
                self.conv_template.append_message(
                    self.conv_template.roles[1], messages[2].content
                )  # asst response
                user_msg = messages[3].content
            # user msg
            self.conv_template.append_message(
                self.conv_template.roles[0], user_msg
            )
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        num_static_tokens = len(toks)
        if user_msg:
            num_static_tokens -= self.num_tok_sep
        elif self.conv_template.name == "vicuna_v1.1":
            pass
        else:
            # NOTE: tested on openchat, llama-2
            num_static_tokens -= self.num_tok_sep2
        static_input_ids = torch.tensor(toks[:num_static_tokens])

        if static_only:
            return static_input_ids

        # user msg + adv suffix
        if user_msg:
            if adv_suffix.startswith(" "):
                # NOTE: space is part of token in tiktoken, i.e., " !" != "!".
                self.conv_template.update_last_message(
                    f"{user_msg}{adv_suffix}"
                )
            else:
                self.conv_template.update_last_message(
                    f"{user_msg} {adv_suffix}"
                )
        else:
            self.conv_template.update_last_message(adv_suffix)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        optim_slice = slice(num_static_tokens, len(toks) - self.num_tok_sep)

        self.conv_template.append_message(self.conv_template.roles[1], None)
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        assistant_role_slice = slice(optim_slice.stop, len(toks))

        self.conv_template.update_last_message(target)  # asst target
        toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
        target_slice = slice(
            assistant_role_slice.stop, len(toks) - self.num_tok_sep2
        )
        loss_slice = slice(
            assistant_role_slice.stop - 1, len(toks) - self.num_tok_sep2 - 1
        )

        # DEBUG
        # print("optim_slice:", self.tokenizer.decode(toks[optim_slice], clean_up_tokenization_spaces=False))
        # print("assistant_role_slice:", self.tokenizer.decode(toks[assistant_role_slice], clean_up_tokenization_spaces=False))
        # print("target_slice:", self.tokenizer.decode(toks[target_slice], clean_up_tokenization_spaces=False))
        # print("loss_slice:", self.tokenizer.decode(toks[loss_slice], clean_up_tokenization_spaces=False))
        # import pdb; pdb.set_trace()

        # Don't need final sep tokens
        input_ids = torch.tensor(toks[: target_slice.stop])

        return input_ids, optim_slice, target_slice, loss_slice

    @torch.no_grad()
    def gen_eval_inputs(
        self,
        messages: list[Message],
        suffix: str,
        target: str,
        num_fixed_tokens: int = 0,
        max_target_len: int | None = None,
    ) -> ModelInputIds:
        """Generate inputs for evaluation.

        Returns:
            eval_inputs: Inputs for evaluation.
        """
        suffix_ids = self.tokenizer(
            suffix, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        suffix_ids.requires_grad_(False)
        suffix_ids.squeeze_(0)

        out = self.get_input_ids(messages, suffix, target)
        orig_input_ids, optim_slice, target_slice, loss_slice = out

        if max_target_len is not None:
            # Adjust target slice to be at most max_target_len
            end = min(target_slice.stop, target_slice.start + max_target_len)
            target_slice = slice(target_slice.start, end)
            loss_slice = slice(loss_slice.start, end - 1)

        # Offset everything to ignore static tokens which are processed
        # separately
        orig_input_ids = orig_input_ids[num_fixed_tokens:]
        optim_slice = slice(
            optim_slice.start - num_fixed_tokens,
            optim_slice.stop - num_fixed_tokens,
        )
        target_slice = slice(
            target_slice.start - num_fixed_tokens,
            target_slice.stop - num_fixed_tokens,
        )
        loss_slice = slice(
            loss_slice.start - num_fixed_tokens,
            loss_slice.stop - num_fixed_tokens,
        )
        new_suffix_ids = orig_input_ids[optim_slice]
        if len(new_suffix_ids) != len(suffix_ids):
            logger.warning(
                "New suffix ids after get_input_ids() is different from the "
                "original\nsuffix_ids: %s\nnew_suffix_ids: %s",
                suffix_ids,
                new_suffix_ids,
            )
            new_suffix_ids = suffix_ids
        target_ids = orig_input_ids[target_slice]
        assert target_ids.ndim == 1, target_ids.shape
        target_ids.requires_grad_(False)

        eval_input = ModelInputIds(
            suffix_ids=new_suffix_ids,
            dynamic_input_ids=orig_input_ids,
            target_ids=target_ids,
            optim_slice=optim_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )
        return eval_input


def _simple_template(messages: list[Message]):
    texts = [
        "The following is a conversation between a user and an AI assistant. "
        "Please respond to the user as the assistant."
    ]
    for m in messages:
        texts.append(f"{m.role.name.title()}>{m.content}")
    texts.append(f"{Role.ASSISTANT.name.title()}>")
    return "\n".join(texts)


def build_prompt(
    messages: list[Message],
    template_name: str | None = None,
    return_openai_chat_format: bool = False,
):
    if template_name is None:
        return _simple_template(messages)

    conv = get_conv_template(template_name)
    for m in messages:
        if m.role == Role.SYSTEM and m.content:
            conv.set_system_message(m.content)
        elif m.role == Role.USER:
            conv.append_message(conv.roles[0], m.content)
        elif m.role == Role.ASSISTANT:
            conv.append_message(conv.roles[1], m.content)

    # Append assistant response if user message is the last message
    if messages[-1].role == Role.USER:
        conv.append_message(conv.roles[1], None)

    if return_openai_chat_format:
        return conv.to_openai_api_messages()
    return conv.get_prompt()
