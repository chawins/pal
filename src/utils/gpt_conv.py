import dataclasses
import logging

from fastchat.conversation import (
    Conversation,
    SeparatorStyle,
    register_conv_template,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CustomConversation(Conversation):
    """Custom conversation template for GPT tokenizer."""

    def __post_init__(self, *args, **kwargs) -> None:
        _ = args, kwargs  # Unused
        assert self.name == "chatgpt", self.name

    def get_prompt(self) -> str:
        system_prompt = self.system_template.format(
            system_message=self.system_message
        )
        if self.sep_style is SeparatorStyle.ADD_COLON_SINGLE:
            ret = system_prompt
            for role, message in self.messages:
                # Automatically add space only for user's messages
                space = " " if role == self.roles[0] else ""
                if message:
                    # Original always adds space after colon
                    # ret += role + ": " + message + self.sep
                    ret += f"{self.sep}{role}:{space}{message}"
                else:
                    ret += f"{self.sep}{role}:"
            return ret
        raise ValueError(f"Invalid style: {self.sep_style}")

    def copy(self):
        # Original copy() uses Conversation
        return CustomConversation(
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


GPT_TEMPLATE = CustomConversation(
    name="chatgpt",
    system_message="You are a helpful assistant.",
    roles=("user", "assistant"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep=" ",
    sep2="",
)

# This template is used for Completion API. This mimics a chat interface, though
# the official documentation (https://platform.openai.com/docs/guides/text-generation/chat-completions-vs-completions)
# allows just a direct instruction.
register_conv_template(GPT_TEMPLATE, override=True)
