"""
Ollama Provider — Local SLM via Ollama
======================================
Supports local models like Llama, Mistral, Gemma, Phi, Qwen, etc.
"""

from __future__ import annotations

import logging
from models.llm_providers.base import BaseLLMProvider, _provider_cfg, _resolve_model

logger = logging.getLogger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Local SLM via Ollama (Llama, Mistral, Phi, Gemma, ...)."""

    provider_name = "ollama"

    def __init__(self, model_name: str = "", api_url: str = "", **kwargs):
        cfg = _provider_cfg("ollama")
        self.model_name = _resolve_model("ollama", model_name or None)
        base_url = api_url or cfg.get("api_url", "http://localhost:11434/api/chat")
        # Newer Ollama versions removed /api/generate (HTTP 410); normalise to /api/chat.
        if base_url.rstrip("/").endswith("/api/generate"):
            base_url = base_url[: base_url.rfind("/api/generate")] + "/api/chat"
        self.api_url = base_url
        self.timeout = cfg.get("timeout", 300)
        try:
            import requests
            self._requests = requests
        except ImportError:
            raise ImportError("pip install requests")

    def _call_api(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            r = self._requests.post(self.api_url, json=payload, timeout=self.timeout)
            if r.status_code in (404, 410):
                # Fall back to legacy generate endpoint if chat is unavailable
                legacy = self.api_url.replace("/api/chat", "/api/generate")
                if legacy != self.api_url:
                    r = self._requests.post(
                        legacy,
                        json={"model": self.model_name, "prompt": prompt, "stream": False},
                        timeout=self.timeout
                    )
            r.raise_for_status()
            data = r.json()
            return data.get("message", {}).get("content") or data.get("response", "")
        except self._requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.api_url}.\n"
                f"  Start it with: ollama serve"
            )
