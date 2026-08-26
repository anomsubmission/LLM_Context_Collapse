"""Prompt construction and label parsing.

`parse_label` returns None when the generation matches no label. Callers score a
None as incorrect, which is what the published pipeline did; a cell in which every
generation is unparseable is a decode failure, not an accuracy measurement, and
`Cell.is_decode_failure` flags it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

from .data import Example, TASKS


def render_prompt(task_id: str, demos: Sequence[Example], query_text: str) -> str:
    cfg = TASKS[task_id]
    parts = [
        cfg["instruction"],
        f"Choose exactly one label from: {' | '.join(cfg['label_names'])}.",
        "Respond with only the label, nothing else.",
        "",
    ]
    for d in demos:
        parts += [f"Text: {d.text}", f"Label: {d.label}", ""]
    parts += [f"Text: {query_text}", "Label:"]
    return "\n".join(parts)


def parse_label(generation: str, label_names: Sequence[str]) -> str | None:
    """Constrained match, first-line, case-insensitive, prefix-tolerant."""
    pred = generation.split("\n")[0].strip().strip(".").lower()
    for name in label_names:
        if pred == name.lower() or pred.startswith(name.lower()):
            return name
    return None


@dataclass
class Cell:
    """One (task, K, seed) evaluation cell."""
    task: str
    k: int
    seed: int
    accuracy: float
    n_test: int
    n_unparseable: int
    n_truncated: int = 0

    @property
    def is_decode_failure(self) -> bool:
        return self.n_test > 0 and self.n_unparseable == self.n_test

    def to_dict(self) -> dict:
        return asdict(self)
