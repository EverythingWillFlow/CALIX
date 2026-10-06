
from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

from .policy import CandidatePolicy, STRATEGY_ARCHETYPES, STRATEGY_DESCRIPTIONS

class OpenAICompatibleLLM:
    """Minimal client for OpenAI-compatible chat-completion endpoints.

    Enabled only when the environment variable ``CALIX_OPENAI_API_KEY`` is
    set; the base URL defaults to ``CALIX_OPENAI_BASE_URL`` or the official
    OpenAI endpoint.  Prompt templates are loaded from the ``prompts/``
    directory so that wording stays versioned with the repository.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.2,
                 top_p: float = 0.95, max_tokens: int = 4096,
                 prompt_dir: str | None = None):
        self.api_key = os.environ.get("CALIX_OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "CALIX_OPENAI_API_KEY is not set; "
                "configure an API key for real-benchmark reproduction."
            )
        self.base_url = os.environ.get(
            "CALIX_OPENAI_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        if prompt_dir is None:
            prompt_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prompts"
            )
        self.prompt_dir = prompt_dir

    def render_prompt(self, template_name: str, **fields) -> str:
        with open(os.path.join(self.prompt_dir, template_name), "r", encoding="utf-8") as f:
            template = f.read()
        return template.format(**fields)

    def chat(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
