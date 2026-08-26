"""Task definitions and deterministic data loading.

The task table and the sampling semantics are those used to produce the published
results. Two properties are load-bearing for reproducibility and are documented
rather than silently fixed, because changing them would change the numbers:

1.  ``class_balanced_sample`` allocates ``floor(K / n_classes)`` demonstrations per
    class, so the realized demonstration count equals K only when the class count
    divides K. At K=64 that is 64 for 4-class ag_news, 63 for 3-class MNLI and 60
    for 6-class TREC.
2.  Query sets are a fixed random draw and are NOT class balanced. Compare accuracy
    against ``majority_rate`` as well as ``chance_rate``.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Task table. `path`/`subset` are Hugging Face Hub coordinates.
# ---------------------------------------------------------------------------
TASKS: dict[str, dict[str, Any]] = {
    "ag_news": {
        "path": "fancyzhx/ag_news", "subset": None,
        "split_train": "train", "split_test": "test",
        "text_field": "text", "label_field": "label",
        "label_names": ["World", "Sports", "Business", "Sci/Tech"],
        "instruction": "Classify the news topic.",
    },
    "sst2": {
        "path": "glue", "subset": "sst2",
        "split_train": "train", "split_test": "validation",
        "text_field": "sentence", "label_field": "label",
        "label_names": ["negative", "positive"],
        "instruction": "Classify the sentence sentiment.",
    },
    "imdb": {
        "path": "stanfordnlp/imdb", "subset": None,
        "split_train": "train", "split_test": "test",
        "text_field": "text", "label_field": "label",
        "label_names": ["negative", "positive"],
        "instruction": "Classify the movie review sentiment.",
    },
    "mnli": {
        "path": "nyu-mll/multi_nli", "subset": None,
        "split_train": "train", "split_test": "validation_matched",
        "text_field": ["premise", "hypothesis"], "label_field": "label",
        "label_names": ["entailment", "neutral", "contradiction"],
        "instruction": "Given the premise and hypothesis, classify their relation.",
    },
    "trec": {
        "path": "SetFit/TREC-QC", "subset": None,
        "split_train": "train", "split_test": "test",
        "text_field": "text", "label_field": "label_coarse",
        "label_names": ["Description", "Entity", "Abbreviation",
                        "Person", "Number", "Location"],
        "instruction": "Classify what type of answer the question seeks.",
    },
}


@dataclass
class Example:
    text: str
    label_index: int
    label: str


@dataclass
class TaskData:
    task_id: str
    train: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)

    @property
    def label_names(self) -> list[str]:
        return TASKS[self.task_id]["label_names"]

    @property
    def chance_rate(self) -> float:
        return 1.0 / len(self.label_names)

    @property
    def majority_rate(self) -> float:
        if not self.test:
            return float("nan")
        counts = defaultdict(int)
        for e in self.test:
            counts[e.label_index] += 1
        return max(counts.values()) / len(self.test)


def _text_of(row: dict, field_spec) -> str:
    if isinstance(field_spec, str):
        return row[field_spec]
    return " ".join(row[f] for f in field_spec)


def load_task(task_id: str, n_train: int = 512, n_test: int = 200,
              shard_seed: int = 0) -> TaskData:
    """Load a task with the published deterministic shuffle-then-truncate draw.

    Rows whose label index is out of range are dropped *after* truncation, so the
    realized pool can be marginally smaller than requested. This mirrors the
    published behaviour.
    """
    from datasets import load_dataset  # imported lazily: figures/verify need no datasets

    cfg = TASKS[task_id]
    ds = load_dataset(cfg["path"], cfg["subset"])
    rng = random.Random(shard_seed)

    def extract(split, n) -> list[Example]:
        idx = list(range(len(split)))
        rng.shuffle(idx)
        out = []
        for i in idx[:n]:
            row = split[i]
            li = row[cfg["label_field"]]
            if li < 0 or li >= len(cfg["label_names"]):
                continue
            out.append(Example(_text_of(row, cfg["text_field"]), li,
                               cfg["label_names"][li]))
        return out

    return TaskData(task_id,
                    extract(ds[cfg["split_train"]], n_train),
                    extract(ds[cfg["split_test"]], n_test))


def class_balanced_sample(pool: Sequence[Example], k: int, seed: int) -> list[Example]:
    """Published demonstration sampler. See module docstring caveat (1)."""
    if k == 0:
        return []
    rng = random.Random(seed)
    by_class: dict[int, list[Example]] = defaultdict(list)
    for ex in pool:
        by_class[ex.label_index].append(ex)
    classes = sorted(by_class)
    per_class = max(1, k // len(classes))
    out: list[Example] = []
    for c in classes:
        bucket = list(by_class[c])
        rng.shuffle(bucket)
        out.extend(bucket[:per_class])
    rng.shuffle(out)
    return out[:k]
