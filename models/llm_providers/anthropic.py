"""
Anthropic Claude Provider
=========================
Supports Claude 3.5/3.7/4 models (Opus, Sonnet, Haiku).
"""

from __future__ import annotations

import logging
from models.llm_providers.base import (
    BaseLLMProvider,
    _gen_cfg,
    _provider_cfg,
    _resolve_api_key,
    _resolve_model,
)

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude models (claude-opus, claude-sonnet, claude-haiku, ...)."""

    provider_name = "anthropic"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("anthropic", model_name or None)
        self.api_key     = _resolve_api_key("anthropic", "ANTHROPIC_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("anthropic").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 1024)
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install anthropic")

    def _call_api(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
