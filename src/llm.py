"""OpenAI-compatible DeepSeek client wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LLMConfig, load_api_key


@dataclass(slots=True)
class ModelResponse:
    """Normalized result from one chat completion call."""

    text: str
    model: str
    usage: dict[str, Any]


class DeepSeekChatClient:
    """Thin wrapper around the OpenAI Python SDK configured for DeepSeek."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        api_key = load_api_key(config)

        # The import is intentionally local so unit tests that do not exercise
        # the remote model can run without importing the SDK eagerly.
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The 'openai' package is required to talk to DeepSeek. "
                "Install project dependencies with 'pip install openai PyYAML' first."
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=config.base_url)

    def generate(self, messages: list[dict[str, str]]) -> ModelResponse:
        """Generate one next-step completion from a full chat message list."""

        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            timeout=self.config.timeout_seconds,
            messages=messages,
        )

        message = response.choices[0].message.content or ""
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                "completion_tokens": getattr(response.usage, "completion_tokens", None),
                "total_tokens": getattr(response.usage, "total_tokens", None),
            }

        return ModelResponse(text=message, model=response.model, usage=usage)
