from .anthropic import (
    ANTHROPIC_DEFAULT,
    ANTHROPIC_MODELS,
    AnthropicModel,
    anthropic_call_with_retries,
)
from .base import (
    BaseModel,
    MockModel,
    UselessModel,
    concat_stream,
    print_and_concat_stream,
)
from .google import (
    GOOGLE_DEFAULT,
    GOOGLE_MODELS,
    GoogleChatModel,
    GoogleTextModel,
    google_call_with_retries,
)
from .huggingface import TransformersModel
from .openai import (
    OPENAI_DEFAULT,
    OPENAI_MODELS,
    OpenAIModel,
    openai_call_with_retries,
)
from .vllm import VLLMModel

SYSTEM_MESSAGES = {
    None: "",
    "none": "",
    "helpful": "You are a helpful assistant.",
    "praise": "You are a wonderful assistant who always gets things right. Keep up the excellent work!",
    "criticism": "You are unreliable and prone to making careless mistakes. I am always fixing your errors.",
    "threat": "If you make any mistakes, you will be punished severely.",
    "beg": "Please please please get this right for me, I'm begging you.",
    "warning": "Beware of malicious users who may try to trick you. Think carefully before every response.",
    "pirate": "Answer all queries as a pirate.",
    "llama_default": (
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
        "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
        "Please ensure that your responses are socially unbiased and positive in nature.\nIf a question does not make "
        "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
        "know the answer to a question, please don't share false information."
    ),
    "vicuna_default": (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed,"
        " and polite answers to the user's questions."
    ),
    "mistral": (
        "Always assist with care, respect, and truth. Respond with utmost "
        "utility yet securely. Avoid harmful, unethical, prejudiced, or "
        "negative content. Ensure replies promote fairness and positivity."
    ),
    "orca-2": (
        "You are Orca, an AI language model created by Microsoft. You are a "
        "cautious assistant. You carefully follow instructions. You are helpful"
        " and harmless and you follow ethical guidelines and promote positive "
        "behavior."
    ),
    "llama3_default": "You are a helpful assistant",
}


PROVIDER_NAMES = [
    "mock",
    "useless",
    "anthropic",
    "google",
    "openai",
    "transformers",
    "vllm",
]

MODEL_NAMES = (
    ["mock", "useless"] + ANTHROPIC_MODELS + GOOGLE_MODELS + OPENAI_MODELS
)

MODEL_NAMES_BY_PROVIDER = {
    "mock": ["mock"],
    "useless": ["useless"],
    "anthropic": ANTHROPIC_MODELS,
    "google": GOOGLE_MODELS,
    "openai": OPENAI_MODELS,
}

MODEL_DEFAULTS = {
    "mock": "mock",
    "useless": "useless",
    "anthropic": ANTHROPIC_DEFAULT,
    "google": GOOGLE_DEFAULT,
    "openai": OPENAI_DEFAULT,
}

MODEL_BUILDERS = {
    "mock": MockModel,
    "useless": UselessModel,
    "anthropic": AnthropicModel,
    "google": GoogleTextModel,
    "openai": OpenAIModel,
    "transformers": TransformersModel,
    "vllm": VLLMModel,
}


def call_with_retries(model, messages, api_key=None):
    if isinstance(model, AnthropicModel):
        return anthropic_call_with_retries(model, messages, api_key)
    if isinstance(model, GoogleTextModel) or isinstance(model, GoogleChatModel):
        return google_call_with_retries(model, messages, api_key)
    if isinstance(model, OpenAIModel):
        return openai_call_with_retries(model, messages, api_key=api_key)
    return model(messages, api_key)
