"""Prompt construction and label parsing.

`parse_label` returns None when the generation matches no label. Callers score a
None as incorrect, which is what the published pipeline did; a cell in which every
generation is unparseable is a decode failure, not an accuracy measurement, and
`Cell.is_decode_failure` flags it.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
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


def parse_label_strict(generation: str, label_names: Sequence[str]) -> str | None:
    """Exact complete output match for new runs; retain prefix parsing for archives."""
    pred = generation.strip().rstrip(".").strip().casefold()
    return next((name for name in label_names if pred == name.casefold()), None)


def contextual_calibrate(label_scores: dict[str, float],
                         content_free_scores: Sequence[dict[str, float]]) -> dict[str, float]:
    """Diagonal contextual calibration using a mean content-free label prior.

    Inputs are complete-label log likelihoods from identically formatted prompts.
    Normalize within labels, average the content-free probabilities, divide the
    query probabilities by that prior and renormalize. Content-free strings and
    their selection must be fixed independently of evaluation labels. The log
    domain calculation preserves extremely small likelihoods without clipping.
    """
    labels = list(label_scores)
    if not labels or not content_free_scores:
        raise ValueError("calibration requires labels and at least one content-free prompt")
    rows = [label_scores, *content_free_scores]
    if any(set(row) != set(labels) for row in rows):
        raise ValueError("query and content-free scores must use identical label sets")
    if any(not math.isfinite(row[label]) for row in rows for label in labels):
        raise ValueError("label log likelihoods must be finite")

    def logsumexp(values):
        maximum = max(values)
        return maximum + math.log(sum(math.exp(v - maximum) for v in values))

    normalized = [{label: row[label] - logsumexp(list(row.values())) for label in labels}
                  for row in rows]
    prior = {label: logsumexp([row[label] for row in normalized[1:]])
                    - math.log(len(content_free_scores)) for label in labels}
    corrected = {label: normalized[0][label] - prior[label] for label in labels}
    normalizer = logsumexp(list(corrected.values()))
    return {label: math.exp(corrected[label] - normalizer) for label in labels}


@dataclass
class Cell:
    """One (task, K, seed) evaluation cell."""
    task: str
    k: int
    seed: int
    accuracy: float | None
    n_test: int
    n_unparseable: int
    n_truncated: int = 0
    n_rejected: int = 0
    realized_k: int | None = None

    @property
    def is_decode_failure(self) -> bool:
        return self.n_test > 0 and self.n_unparseable == self.n_test

    def to_dict(self) -> dict:
        return asdict(self)
