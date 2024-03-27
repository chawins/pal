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
        assert self.name == "cohere", self.name

    def get_prompt(self) -> str:
        # <BOS_TOKEN>User: Hello!\nChatbot:<EOP_TOKEN> Hi!
        ret = "<BOS_TOKEN>"
        if self.sep_style is SeparatorStyle.ADD_COLON_SINGLE:
            for i, (role, message) in enumerate(self.messages):
                if i > 0:
                    # No sep after "<BOS_TOKEN>"
                    ret += self.sep
                if message:
                    space = " " if role == self.roles[0] else "<EOP_TOKEN> "
                    ret += f"{role}:{space}{message}"
                else:
                    space = "" if role == self.roles[0] else "<EOP_TOKEN>"
                    ret += f"{role}:{space}"
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


COHERE_TEMPLATE = CustomConversation(
    name="cohere",
    system_message="",
    roles=("User", "Chatbot"),
    sep_style=SeparatorStyle.ADD_COLON_SINGLE,
    sep="\n",
    sep2="",
)

# This template is used for Completion API. This mimics a chat interface, though
# the official documentation (https://platform.openai.com/docs/guides/text-generation/chat-completions-vs-completions)
# allows just a direct instruction.
register_conv_template(COHERE_TEMPLATE, override=True)
