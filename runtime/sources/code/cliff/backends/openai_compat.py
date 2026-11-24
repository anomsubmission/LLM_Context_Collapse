"""Any OpenAI-compatible chat-completions endpoint.

Covers vLLM, TGI (OpenAI router), Ollama, LM Studio, llama.cpp server, Together,
Groq, OpenRouter, Azure OpenAI and OpenAI itself. Only the standard library is
used, so there is no vendor SDK dependency.

    cliff sweep --backend openai_compat \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --base-url http://localhost:8000/v1

Set CLIFF_API_KEY (or pass --api-key) if the endpoint requires one.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Sequence


class OpenAICompatBackend:
    def __init__(self, model_id: str, base_url: str | None = None,
                 api_key: str | None = None, timeout: int = 120,
                 max_retries: int = 5, temperature: float = 0.0):
        self.name = f"openai_compat:{model_id}"
        self.model_id = model_id
        self.base_url = (base_url or os.environ.get("CLIFF_BASE_URL")
                         or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("CLIFF_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        payload = json.dumps({
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": self.temperature,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=payload, headers=headers)
        delay = 1.0
        last: Exception | None = None
        for _ in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read())
                return body["choices"][0]["message"]["content"].strip()
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError,
                    KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(delay)
                delay = min(delay * 2, 30)
        raise RuntimeError(f"{self.base_url} failed after {self.max_retries} "
                           f"attempts: {type(last).__name__}: {last}")

    def generate_batch(self, prompts: Sequence[str],
                       max_new_tokens: int = 10) -> list[str]:
        return [self.generate(p, max_new_tokens) for p in prompts]
