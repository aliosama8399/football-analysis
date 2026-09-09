"""
OpenAI Provider — GPT Models
============================
Supports GPT-4o, GPT-4o-mini, o1-mini, etc.
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


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT models (gpt-4o, gpt-4o-mini, o1-mini, ...)."""

    provider_name = "openai"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("openai", model_name or None)
        self.api_key     = _resolve_api_key("openai", "OPENAI_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("openai").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("pip install openai")

    def _call_api(self, prompt: str, json_mode: bool = True) -> str:
        """
        Call the OpenAI chat API. json_mode forces a JSON object reply.
        """
        kwargs = dict(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional football tactical analyst API designed to format inputs as JSON."
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            timeout=self.timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            kwargs["max_completion_tokens"] = self.max_tokens
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e)
            if "max_completion_tokens" in err:
                kwargs.pop("max_completion_tokens", None)
                kwargs["max_tokens"] = self.max_tokens
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        return response.choices[0].message.content
