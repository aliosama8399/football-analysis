"""
TensorRT-LLM Provider
=====================
Local TensorRT-LLM server (OpenAI-compatible) hosting the fine-tuned model.
"""

from __future__ import annotations

import logging
import os
from models.llm_providers.base import (
    BaseLLMProvider,
    _gen_cfg,
    _provider_cfg,
)

logger = logging.getLogger(__name__)


class TrtLLMProvider(BaseLLMProvider):
    """Local TensorRT-LLM server (OpenAI-compatible) hosting the finetuned Qwen.

    Config (models/llm_config.yaml -> providers.trtllm):
      api_url:   http://localhost:8355/v1   (docker service) or override
      model_name: whatever trtllm-serve exposes (e.g. the engine alias)
      timeout:   120
    Requires NO api key (local inference).
    """

    provider_name = "trtllm"

    def __init__(self, model_name: str = "", api_url: str = "", **kwargs):
        cfg = _provider_cfg("trtllm")
        self.model_name  = (model_name or cfg.get("model_name", "")).strip() or \
            cfg.get("default_model", "football-analysisN-trtllm")
        self.api_url     = (
            os.environ.get("TRTLLM_API_URL")
            or api_url
            or cfg.get("api_url", "")
        ).strip().rstrip("/") or "http://localhost:8355/v1"
        self.timeout     = cfg.get("timeout", 120)
        self.temperature = _gen_cfg().get("temperature", 0.7)
        self.max_tokens  = _gen_cfg().get("max_tokens", 2048)
        if not self.api_url.endswith("/v1"):
            self.api_url += "/v1"
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.api_url, api_key="not-needed")
        except ImportError:
            raise ImportError("pip install openai")

    def _call_api(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a football tactical analyst serving a local Qwen engine."},
                {"role": "user",   "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return response.choices[0].message.content
