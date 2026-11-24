"""Task definitions and deterministic data loading.

The original released sampler is retained for explicit historical reconstruction;
its presence cannot establish which code produced every archived result. New
controlled runs use ``exact_balanced_sample``. Historical caveats are:

1.  ``class_balanced_sample`` allocates ``max(1, floor(K / n_classes))`` per
    class, then takes at most K examples. With adequate class supplies and K at
    least the number of classes, this yields C*floor(K/C); at K=64 that is 64
    for 4-class ag_news, 63 for 3-class MNLI and 60 for 6-class TREC. For K<C,
    the final slice still returns K examples. Scarce classes can reduce the count.
2.  Query sets are a fixed random draw and are NOT class balanced. Compare accuracy
    against ``majority_rate`` as well as ``chance_rate``.
"""
from __future__ import annotations

import random
import os
import hashlib
import unicodedata
import re
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
    example_id: str = ""
    text_hash: str = ""

    def __post_init__(self):
        # Identity does not depend on a shuffle, label or experiment seed.
        actual_hash = normalized_text_hash(self.text)
        if self.text_hash and self.text_hash != actual_hash:
            raise ValueError("provided example hash does not match its normalized text")
        self.text_hash = actual_hash
        if not self.example_id:
            self.example_id = f"text:{self.text_hash}"


@dataclass
class TaskData:
    task_id: str
    train: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

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


def normalized_text_hash(text: str) -> str:
    """Conservative exact-duplicate key; near duplicates require a separate audit."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exclude_examples(pool: Sequence[Example], excluded: Sequence[Example]) -> list[Example]:
    """Exclude both source-row identities and normalized duplicate texts."""
    ids = {e.example_id for e in excluded}
    hashes = {e.text_hash for e in excluded}
    return [e for e in pool if e.example_id not in ids and e.text_hash not in hashes]


def load_task(task_id: str, n_train: int | None = 512, n_test: int | None = 200,
              shard_seed: int = 0, *, revision: str | None = None,
              train_split: str | None = None, test_split: str | None = None,
              train_indices: Sequence[int] | None = None,
              test_indices: Sequence[int] | None = None) -> TaskData:
    """Load a task with the published deterministic shuffle-then-truncate draw.

    Rows whose label index is out of range are dropped *after* truncation, so the
    realized pool can be marginally smaller than requested. This mirrors the
    published behaviour.
    """
    from datasets import load_dataset  # imported lazily: figures/verify need no datasets

    cfg = TASKS[task_id]
    if n_train is not None and n_train < 0 or n_test is not None and n_test < 0:
        raise ValueError("sample counts must be nonnegative or None for the full pool")
    dataset_kwargs = {"revision": revision} if revision else {}
    if os.environ.get("HF_DATASETS_CACHE"):
        dataset_kwargs["cache_dir"] = os.environ["HF_DATASETS_CACHE"]
    ds = load_dataset(cfg["path"], cfg["subset"], **dataset_kwargs)
    rng = random.Random(shard_seed)
    train_split = train_split or cfg["split_train"]
    test_split = test_split or cfg["split_test"]

    def extract(split_name, n, explicit_indices) -> list[Example]:
        split = ds[split_name]
        idx = list(range(len(split))) if explicit_indices is None else list(explicit_indices)
        if len(set(idx)) != len(idx) or any(i < 0 or i >= len(split) for i in idx):
            raise ValueError(f"invalid or repeated source indices in {split_name}")
        if explicit_indices is None:
            rng.shuffle(idx)
        out = []
        for i in idx[:n]:
            row = split[i]
            li = row[cfg["label_field"]]
            if li < 0 or li >= len(cfg["label_names"]):
                continue
            out.append(Example(_text_of(row, cfg["text_field"]), li,
                               cfg["label_names"][li],
                               f"{cfg['path']}:{cfg['subset'] or '-'}:"
                               f"{revision or 'unresolved'}:{split_name}:{i}"))
        return out

    return TaskData(task_id,
                    extract(train_split, n_train, train_indices),
                    extract(test_split, n_test, test_indices),
                    {"dataset_id": cfg["path"], "subset": cfg["subset"],
                     "requested_revision": revision,
                     "revision_pinned": bool(re.fullmatch(r"[0-9a-fA-F]{40}", revision or "")),
                     "train_split": train_split, "test_split": test_split,
                     "query_split_seed": shard_seed,
                     "train_fingerprint": getattr(ds[train_split], "_fingerprint", None),
                     "test_fingerprint": getattr(ds[test_split], "_fingerprint", None)})


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


def exact_balanced_sample(pool: Sequence[Example], k: int, seed: int, *,
                          exclude: Sequence[Example] = ()) -> list[Example]:
    """Exact K, without replacement, with seeded balanced remainder allocation.

    A scarce class fails explicitly instead of changing K or the class balance.
    Repeated source IDs and normalized duplicate texts are also removed. Use the
    separately named legacy sampler only for historical reconstruction.
    """
    if k < 0:
        raise ValueError("K must be nonnegative")
    if k == 0:
        return []
    classes = sorted({e.label_index for e in pool})
    if not classes:
        raise ValueError("cannot sample demonstrations from an empty pool")
    by_class: dict[int, list[Example]] = defaultdict(list)
    seen_ids, seen_hashes = set(), set()
    for ex in exclude_examples(pool, exclude):
        if ex.example_id in seen_ids or ex.text_hash in seen_hashes:
            continue
        seen_ids.add(ex.example_id)
        seen_hashes.add(ex.text_hash)
        by_class[ex.label_index].append(ex)
    rng = random.Random(seed)
    remainder_classes = list(classes)
    rng.shuffle(remainder_classes)
    extras = set(remainder_classes[:k % len(classes)])
    out = []
    for cls in classes:
        count = k // len(classes) + int(cls in extras)
        bucket = list(by_class[cls])
        if len(bucket) < count:
            raise ValueError(f"class {cls}: need {count} distinct examples, have {len(bucket)}")
        rng.shuffle(bucket)
        out.extend(bucket[:count])
    rng.shuffle(out)
    assert len(out) == k
    return out
