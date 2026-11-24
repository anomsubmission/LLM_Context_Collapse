"""Deterministic offline backend for tests and for smoke-testing the pipeline.

`--backend echo` exercises every code path (prompt construction, parsing, cell
aggregation, JSON output) with no model, no network and no GPU, in under a second.
It is not a model: it copies the label of the last demonstration, or emits the
first label at K=0. Use it to check a run configuration end to end before
committing GPU hours, and to verify the harness in CI.
"""
from __future__ import annotations

import re
import time
from typing import Sequence


class EchoBackend:
    name = "echo"

    def __init__(self, model_id: str = "echo", **_):
        self.model_id = model_id

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        labels = re.findall(r"^Label: (.+)$", prompt, re.M)
        if labels:
            return labels[-1]
        m = re.search(r"Choose exactly one label from: ([^\n.]+)", prompt)
        return m.group(1).split("|")[0].strip() if m else ""

    def generate_batch(self, prompts: Sequence[str],
                       max_new_tokens: int = 10) -> list[str]:
        return [self.generate(p, max_new_tokens) for p in prompts]

    def generate_record(self, prompt: str, max_new_tokens: int = 10) -> dict:
        t0 = time.perf_counter()
        return {"raw_text": self.generate(prompt, max_new_tokens),
                "measurement_status": "synthetic_offline_backend",
                "input_tokens_before": None, "input_tokens_after": None,
                "generation_tokens": None, "truncated": False, "rejected": False,
                "rejection_reason": None, "retained_instruction": True,
                "retained_query": True, "template": "raw",
                "latency_seconds": time.perf_counter() - t0}
