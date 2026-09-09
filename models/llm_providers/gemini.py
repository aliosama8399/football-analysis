"""
Google Gemini Provider
======================
Supports Gemini 2.0 Flash, Gemini 1.5 Pro, etc.
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


class GeminiProvider(BaseLLMProvider):
    """Google Gemini models (gemini-2.0-flash, gemini-1.5-pro, ...)."""

    provider_name = "gemini"

    def __init__(self, model_name: str = "", api_key: str = "", **kwargs):
        self.model_name  = _resolve_model("gemini", model_name or None)
        self.api_key     = _resolve_api_key("gemini", "GEMINI_API_KEY", api_key or None)
        self.timeout     = _provider_cfg("gemini").get("timeout", 60)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)

        # Try stable SDK first, fall back to preview SDK
        self._sdk = None
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            self._sdk = "generativeai"
        except ImportError:
            try:
                from google import genai as genai_new
                self._client = genai_new.Client(api_key=self.api_key)
                self._sdk = "genai"
            except ImportError:
                raise ImportError(
                    "Install a Gemini SDK:\n"
                    "  pip install google-generativeai   (recommended)\n"
                    "  pip install google-genai          (preview)"
                )

    def _call_api(self, prompt: str) -> str:
        if self._sdk == "generativeai":
            response = self._client.generate_content(prompt)
            return response.text
        # Fallback for newer google.genai SDK
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": self.temperature,
                "max_output_tokens": self.max_tokens
            }
        )
        return response.text
