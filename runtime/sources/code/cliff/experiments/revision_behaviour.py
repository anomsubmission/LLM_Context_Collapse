"""W3 paired ordering/composition and complete-label contextual calibration.

Run with ``python -m cliff.experiments.revision_behaviour --config FILE``.
Configuration freezes the K locations, data roles, content-free strings, ordering,
and token-length matching criteria before outcomes are observed. Development
candidates remain descriptive; final-test confirmation requires a frozen,
independent development source. No checkpoint or dataset is fetched on import.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..backends import get_backend
from ..backends.base import ContextBudgetError
from ..data import (Example, TaskData, TASKS, exact_balanced_sample, exclude_examples,
                    load_task)
from ..prompting import parse_label_strict, render_prompt
from ..revision_records import RunWriter, canonical_json, content_hash


def _logsumexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def _log_probabilities(scores):
    normalizer = _logsumexp(list(scores.values()))
    return {label: score - normalizer for label, score in scores.items()}


def _probabilities(log_probabilities):
    return {label: math.exp(value) for label, value in log_probabilities.items()}


def _argmax(scores, labels):
    # Exact ties follow the fixed dataset label order, which is recorded.
    return max(labels, key=lambda label: scores[label])


def validate_config(config):
    if config.get("query_role") not in {"development", "final_test", "technical_pilot"}:
        raise ValueError("query_role must be development, final_test or technical_pilot")
    if not config.get("task") or not config.get("model", {}).get("id"):
        raise ValueError("task and model.id are required")
    selection = config.get("k_selection", {})
    ks = [0, config.get("healthy_k"), config.get("cliff_k"), config.get("next_k")]
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 0 for k in ks):
        raise ValueError("healthy_k, cliff_k and next_k must be explicit nonnegative integers")
    if not 0 < ks[1] < ks[2] < ks[3]:
        raise ValueError("require 0 < healthy_k < cliff_k < next_k")
    if config["query_role"] == "final_test" and not (
            selection.get("role") == "development" and selection.get("frozen") is True
            and selection.get("source_run_id") and selection.get("query_ids") and selection.get("query_hashes")):
        raise ValueError("final-test runs require a frozen, identifiable independent development selection")
    if config["query_role"] != "technical_pilot":
        require_pinned_model(config["model"])
        dataset = config.get("dataset", {})
        if not re.fullmatch(r"[0-9a-fA-F]{40}", dataset.get("revision") or ""):
            raise ValueError("scientific W3 requires an immutable dataset revision before model work")
        if not dataset.get("train_split") or not dataset.get("query_split"):
            raise ValueError("scientific W3 requires explicit demonstration/query splits")
        if config["query_role"] == "development" and dataset["query_split"] == TASKS[config["task"]]["split_test"]:
            raise ValueError("development queries must reserve the official final-test split")
    draws = config.get("demo_seeds", [])
    if not draws or any(not isinstance(seed, int) for seed in draws) or len(set(draws)) != len(draws):
        raise ValueError("demo_seeds must contain distinct integer draws")
    content_free = config.get("content_free_queries", ["N/A", "[MASK]", ""])
    if content_free != ["N/A", "[MASK]", ""]:
        raise ValueError("this registered W3 protocol fixes content-free queries to N/A, [MASK], empty")
    floor = config.get("calibration_floor", 1e-12)
    if not isinstance(floor, (int, float)) or not 0 <= floor < 1:
        raise ValueError("calibration_floor must be in [0, 1)")
    if config.get("last_examples_per_class", 2) < 1:
        raise ValueError("last_examples_per_class must be positive")
    majority = config.get("majority", {})
    if majority.get("enabled", True):
        if not .5 < majority.get("fraction", .625) < 1:
            raise ValueError("majority fraction must be strictly between .5 and 1")
        if majority.get("token_tolerance_fraction", .15) < 0:
            raise ValueError("majority token tolerance cannot be negative")
        if majority.get("max_length_match_attempts", 100) < 1:
            raise ValueError("positive majority length matching attempts are required")
        if not set(majority.get("ks", ks[2:])).issubset(ks[1:]):
            raise ValueError("majority K cells must be in the registered K grid")
    if config.get("max_new_tokens", 10) < 1:
        raise ValueError("max_new_tokens must be positive")
    dataset = config.get("dataset", {})
    for field in ("train_index_range", "query_index_range"):
        indices = dataset.get(field)
        if indices is not None and (len(indices) != 2 or any(not isinstance(i, int) for i in indices)
                                    or not 0 <= indices[0] < indices[1]):
            raise ValueError(f"{field} must be a nonempty [start, stop] source-row range")
    return ks


def require_pinned_model(model):
    if not all(re.fullmatch(r"[0-9a-fA-F]{40}", revision or "") for revision in (
            model.get("revision"), model.get("tokenizer_revision") or model.get("revision"))):
        raise ValueError("scientific runs require immutable model/tokenizer revisions before model work")


def verify_loaded_model(model_config, backend, role):
    """Verify resolved checkpoint and tokenizer snapshot identities before inference."""
    model = getattr(backend, "_model", None)
    tokenizer = getattr(backend, "_tok", None)
    actual_model = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_kwargs = getattr(tokenizer, "init_kwargs", {})
    actual_tokenizer = (getattr(backend, "resolved_tokenizer_revision", None)
                        or getattr(tokenizer, "_commit_hash", None) or tokenizer_kwargs.get("_commit_hash"))
    if not actual_tokenizer:
        for value in tokenizer_kwargs.values():
            if isinstance(value, str):
                match = re.search(r"[/\\]snapshots[/\\]([0-9a-fA-F]{40})(?:[/\\]|$)", value)
                if match:
                    actual_tokenizer = match.group(1)
                    break
    if role != "technical_pilot":
        if actual_model != model_config["revision"]:
            raise ValueError("loaded model commit differs from its immutable registered revision")
        if actual_tokenizer != (model_config.get("tokenizer_revision") or model_config["revision"]):
            raise ValueError("loaded tokenizer commit is missing or differs from its immutable registered revision")
    return {"resolved_model_revision": actual_model, "resolved_tokenizer_revision": actual_tokenizer}


def exclude_development_data(data, selection):
    ids, hashes = set(selection.get("query_ids", [])), set(selection.get("query_hashes", []))
    def retain(pool):
        return [example for example in pool if example.example_id not in ids and example.text_hash not in hashes]
    return TaskData(data.task_id, retain(data.train), retain(data.test),
                    {**data.metadata, "reserved_development_ids": sorted(ids),
                     "reserved_development_hashes": sorted(hashes),
                     "n_reserved_train_excluded": len(data.train) - len(retain(data.train)),
                     "n_reserved_queries_excluded": len(data.test) - len(retain(data.test))})


def _identity_list(examples):
    return [{"id": example.example_id, "hash": example.text_hash,
             "label": example.label, "label_index": example.label_index}
            for example in examples]


def _context(task, k, draw, demos, *, family, order, composition="balanced",
             last_class=None, last_repeat=None, reference=None, length_match=None):
    value = {"task": task, "requested_k": k, "realized_k": len(demos),
             "draw_id": draw, "comparison_family": family, "order": order,
             "composition": composition, "forced_last_class": last_class,
             "last_example_repeat": last_repeat,
             "ordered_demonstrations": _identity_list(demos),
             "demonstration_ids": [example.example_id for example in demos],
             "demonstration_hashes": [example.text_hash for example in demos],
             "demonstration_labels": [example.label for example in demos],
             "supported_final_classes": sorted({example.label for example in demos}),
             "label_counts": dict(Counter(example.label for example in demos)),
             "demonstration_multiset_id": content_hash(sorted(
                 (example.example_id, example.text_hash, example.label) for example in demos)),
             "composition_reference_context_id": reference,
             "length_matching": length_match}
    # Stable across draws/K for downstream crossed-query/draw aggregation.
    # Context IDs identify exact prompts; they are not inferential conditions.
    value["condition"] = "|".join((family, order, composition,
                                     f"last={last_class}", f"repeat={last_repeat}"))
    value["context_id"] = content_hash(value)
    return value


def _interleaved(demos, labels, seed):
    buckets = {label: [ex for ex in demos if ex.label == label] for label in labels}
    class_order = list(labels)
    random.Random(seed).shuffle(class_order)
    return [buckets[label][index] for index in range(max(map(len, buckets.values()), default=0))
            for label in class_order if index < len(buckets[label])]


def _blocked(demos, labels, seed):
    class_order = list(labels)
    random.Random(seed).shuffle(class_order)
    return [ex for label in class_order for ex in demos if ex.label == label]


def _forced_last_orders(demos, labels, repeats):
    present = [label for label in labels if any(ex.label == label for ex in demos)]
    # Every supported final class receives the same number of distinct last
    # examples. At K<C only present classes are supported and this is recorded.
    count = min([repeats] + [sum(ex.label == label for ex in demos) for label in present])
    for label in present:
        indices = [index for index, ex in enumerate(demos) if ex.label == label][:count]
        for repeat, index in enumerate(indices):
            yield label, repeat, demos[index + 1:] + demos[:index + 1]


def _demonstration_tokens(backend, task, demos):
    # The same empty query is used for every composition match. Counts are
    # measured on complete templates because tokenization is not additive.
    return (len(backend.encode_prompt(render_prompt(task, demos, "")))
            - len(backend.encode_prompt(render_prompt(task, [], ""))))


def _composition_sample(pool, labels, k, majority_label, fraction, seed):
    majority_count = math.ceil(k * fraction)
    others = [label for label in labels if label != majority_label]
    remaining = k - majority_count
    if remaining < len(others):
        raise ValueError("majority composition needs at least one demonstration of every class")
    rng = random.Random(seed)
    rng.shuffle(others)
    counts = {majority_label: majority_count}
    counts.update({label: remaining // len(others) + int(index < remaining % len(others))
                   for index, label in enumerate(others)})
    samples = []
    for label in labels:
        bucket = [example for example in pool if example.label == label]
        rng.shuffle(bucket)
        if len(bucket) < counts[label]:
            raise ValueError(f"class {label} cannot supply majority quota {counts[label]}")
        samples.extend(bucket[:counts[label]])
    rng.shuffle(samples)
    return samples


def build_contexts(config, data, backend):
    """Freeze the complete paired design using IDs and lengths, never outcomes."""
    ks = validate_config(config)
    labels, task = data.label_names, data.task_id
    # Exclude all query identities, then deduplicate the remaining pool.
    excluded_pool = exclude_examples(data.train, data.test)
    pool, seen_ids, seen_hashes = [], set(), set()
    for example in excluded_pool:
        if example.example_id not in seen_ids and example.text_hash not in seen_hashes:
            pool.append(example)
            seen_ids.add(example.example_id)
            seen_hashes.add(example.text_hash)
    by_id = {example.example_id: example for example in pool}
    contexts = [_context(task, 0, config["demo_seeds"][0], [], family="shared_zero_shot", order="zero_shot")]
    repeats = config.get("last_examples_per_class", 2)
    majority = config.get("majority", {})
    majority_ks = majority.get("ks", ks[2:]) if majority.get("enabled", True) else []
    for k in ks[1:]:
        for draw in config["demo_seeds"]:
            demos = exact_balanced_sample(pool, k, draw)
            base_contexts = [
                _context(task, k, draw, demos, family="order", order="random"),
                _context(task, k, draw, _interleaved(demos, labels, draw), family="order", order="interleaved"),
                _context(task, k, draw, _blocked(demos, labels, draw), family="order", order="blocked")]
            rotations = {}
            for last_class, repeat, order in _forced_last_orders(demos, labels, repeats):
                context = _context(task, k, draw, order, family="order", order="last_class_balanced",
                                   last_class=last_class, last_repeat=repeat)
                base_contexts.append(context)
                rotations[(last_class, repeat)] = context["context_id"]
            contexts.extend(base_contexts)
            if k not in majority_ks:
                continue
            reference_tokens = _demonstration_tokens(backend, task, demos)
            if reference_tokens <= 0:
                raise ValueError("positive measured demonstration length is required for composition matching")
            for majority_index, majority_label in enumerate(labels):
                matched = None
                attempts = majority.get("max_length_match_attempts", 100)
                for attempt in range(attempts):
                    candidate_seed = draw + 104729 * (majority_index + 1) + 1000003 * attempt
                    candidate = _composition_sample(pool, labels, k, majority_label,
                                                    majority.get("fraction", .625), candidate_seed)
                    tokens = _demonstration_tokens(backend, task, candidate)
                    difference = abs(tokens - reference_tokens) / reference_tokens
                    if difference <= majority.get("token_tolerance_fraction", .15):
                        matched = (candidate, {"reference_demo_tokens": reference_tokens,
                                   "actual_demo_tokens": tokens, "relative_difference": difference,
                                   "tolerance_fraction": majority.get("token_tolerance_fraction", .15),
                                   "attempts": attempt + 1, "selection_seed": candidate_seed,
                                   "selection_uses": "identities_and_token_lengths_only"})
                        break
                if matched is None:
                    raise ValueError(f"failed prospective length matching for K={k}, draw={draw}, "
                                     f"majority={majority_label}; no query outcomes were consulted")
                candidate, matching = matched
                for last_class, repeat, order in _forced_last_orders(candidate, labels, repeats):
                    # Balanced reference may support fewer last-example repeats
                    # at small K; omit unmatched repeats from both comparisons.
                    if (last_class, repeat) not in rotations:
                        continue
                    contexts.append(_context(task, k, draw, order, family="composition",
                        order="last_class_balanced", composition=f"majority:{majority_label}",
                        last_class=last_class, last_repeat=repeat,
                        reference=rotations[(last_class, repeat)], length_match=matching))
    return contexts, by_id


def _score(backend, prompt, labels):
    ids = backend.encode_prompt(prompt)
    if hasattr(backend, "label_encodings"):
        continuations = backend.label_encodings(prompt, labels)
    else:
        continuations = {label: backend.encode_completion(prompt, getattr(backend, "label_prefix", " ") + label) for label in labels}
        common_start = min(start for _, start in continuations.values())
        continuations = {label: (sequence, common_start) for label, (sequence, _) in continuations.items()}
    capacity = getattr(backend, "max_seq_len", None)
    if capacity is None:
        raise ValueError("W3 scoring requires a known context capacity")
    if len(ids) > capacity or any(len(sequence) > capacity for sequence, _ in continuations.values()):
        raise ContextBudgetError("complete-label scoring context exceeds checkpoint capacity")
    if any(not 0 < start < len(sequence) for sequence, start in continuations.values()):
        raise ValueError("complete labels require a nonempty conditioning prefix and continuation")
    scores = backend.score_labels(prompt, labels)
    if set(scores) != set(labels) or not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("backend must return finite complete-label scores for every label")
    metadata = {"measurement_status": "measured", "input_tokens_before": len(ids),
                "input_tokens_after": len(ids), "input_token_ids": ids,
                "input_token_ids_hash": content_hash(ids), "context_limit": capacity,
                "truncated": False, "rejected": False, "generation_tokens": 0,
                "retained_instruction": True, "retained_query": True,
                "template": getattr(backend, "template", "backend_defined"),
                "template_hash": backend._template_hash() if hasattr(backend, "_template_hash") else None,
                "label_scoring_policy": getattr(backend, "label_scoring_policy", "unspecified"),
                "label_sequence_token_counts": {label: len(sequence) for label, (sequence, _) in continuations.items()},
                "label_continuation_token_counts": {label: len(sequence) - start
                                                      for label, (sequence, start) in continuations.items()},
                "common_conditioning_tokens": min(start for _, start in continuations.values()),
                "standalone_prompt_tokens_beyond_conditioning": len(ids) - min(start for _, start in continuations.values()),
                "label_sequence_token_hashes": {label: content_hash(sequence)
                                               for label, (sequence, _) in continuations.items()}}
    return scores, metadata


def _calibration(backend, task, demos, labels, content_free, floor):
    rows, log_probs = [], []
    for text in content_free:
        scores, measured = _score(backend, render_prompt(task, demos, text), labels)
        row = {"content_free_query": text, "label_scores": scores, **measured}
        rows.append(row)
        log_probs.append(_log_probabilities(scores))
    log_prior = {label: _logsumexp([row[label] for row in log_probs]) - math.log(len(log_probs))
                 for label in labels}
    effective_log_prior = {label: max(value, math.log(floor)) if floor else value
                           for label, value in log_prior.items()}
    return {"content_free_records": rows, "label_prior": _probabilities(log_prior),
            "log_label_prior": log_prior, "effective_log_label_prior": effective_log_prior,
            "calibration_floor": floor, "combination": "arithmetic_mean_label_probabilities"}


def _scientific_gates(config, data, backend):
    sha = lambda value: bool(re.fullmatch(r"[0-9a-fA-F]{40}", value or ""))
    model, dataset = config["model"], config.get("dataset", {})
    duplicates = len({ex.example_id for ex in data.test}) != len(data.test) or \
        len({ex.text_hash for ex in data.test}) != len(data.test)
    expected_train = dataset.get("train_split")
    expected_test = dataset.get("query_split")
    gates = {"model_revision_pinned": sha(model.get("revision")),
             "tokenizer_revision_pinned": sha(model.get("tokenizer_revision") or model.get("revision")),
             "dataset_revision_pinned": sha(dataset.get("revision")),
             "explicit_splits_match_loaded_metadata": bool(expected_train and expected_test)
                and data.metadata.get("train_split") == expected_train
                and data.metadata.get("test_split") == expected_test,
             "dataset_revision_matches_loaded_metadata": data.metadata.get("requested_revision") == dataset.get("revision")
                and bool(dataset.get("revision")),
             "unique_nonempty_query_set": bool(data.test) and not duplicates,
             "backend_checkpoint_matches_config": getattr(backend, "model_id", None) == model["id"]
                and getattr(backend, "revision", None) == model.get("revision")
                and getattr(backend, "name", "").startswith("hf_local:"),
             "backend_template_matches_config": getattr(backend, "template", None) == model.get("template", "native"),
             "explicit_complete_label_scoring_policy": bool(getattr(backend, "label_scoring_policy", None)),
             "scientific_data_role": config["query_role"] in {"development", "final_test"}}
    if config["query_role"] == "final_test":
        selection = config.get("k_selection", {})
        gates["development_query_ids_independent"] = bool(selection.get("query_ids")) and not (
            {ex.example_id for ex in data.test} & set(selection.get("query_ids", [])))
        gates["development_query_hashes_independent"] = bool(selection.get("query_hashes")) and not (
            {ex.text_hash for ex in data.test} & set(selection.get("query_hashes", [])))
    return gates


def run(config, *, backend=None, data: TaskData | None = None):
    """Execute one immutable registered W3 design and return its artifact path."""
    validate_config(config)
    model_config = config["model"]
    writer = RunWriter(config.get("output_root", "outputs/revision"), "w3_order_calibration", config,
                       run_id=config.get("run_id"), scientific_evidence=False)
    try:
        if backend is None:
            from ..revision_runtime import configure_cache
            configure_cache(config.get("cache_dir", str(Path(__file__).resolve().parents[2] / ".cache" / "revision")))
            backend = get_backend("hf_local", model_config["id"],
                revision=model_config.get("revision"), tokenizer_revision=model_config.get("tokenizer_revision"),
                template=model_config.get("template", "native"), dtype=model_config.get("dtype", "bfloat16"),
                device=model_config.get("device", "auto"), max_seq_len=model_config.get("max_seq_len"),
                load_in_4bit=model_config.get("load_in_4bit", False), overflow_policy="reject",
                label_prefix=model_config.get("label_prefix", " "),
                label_score_implementation=model_config.get("label_score_implementation", "full_forward"),
                local_files_only=model_config.get("local_files_only", True))
        resolved_model = verify_loaded_model(model_config, backend, config["query_role"])
        if data is None:
            dataset = config.get("dataset", {})
            def indices(explicit, interval):
                if explicit is not None and interval is not None:
                    raise ValueError("specify explicit indices or an index range, not both")
                return list(range(*interval)) if interval is not None else explicit
            data = load_task(config["task"], n_train=dataset.get("n_train"),
                n_test=dataset.get("n_queries", 100), shard_seed=dataset.get("query_split_seed", 0),
                revision=dataset.get("revision"), train_split=dataset.get("train_split"),
                test_split=dataset.get("query_split"),
                train_indices=indices(dataset.get("train_indices"), dataset.get("train_index_range")),
                test_indices=indices(dataset.get("query_indices"), dataset.get("query_index_range")))
        if config["query_role"] != "technical_pilot":
            dataset = config["dataset"]
            if (data.metadata.get("requested_revision") != dataset["revision"] or
                    data.metadata.get("train_split") != dataset["train_split"] or
                    data.metadata.get("test_split") != dataset["query_split"]):
                raise ValueError("loaded dataset revision/splits differ from the registered scientific pools")
        if config["query_role"] == "final_test":
            data = exclude_development_data(data, config["k_selection"])
        if data.task_id != config["task"]:
            raise ValueError("loaded task differs from registered task")
        if not data.test or len({ex.text_hash for ex in data.test}) != len(data.test) or \
                len({ex.example_id for ex in data.test}) != len(data.test):
            raise ValueError("query pool must be nonempty with unique source IDs and normalized texts")
        labels = data.label_names
        contexts, pool = build_contexts(config, data, backend)
        design = {"query_role": config["query_role"], "query_examples": _identity_list(data.test),
                  "dataset_metadata": data.metadata, "label_tie_order": labels,
                  "contexts": contexts, "design_sha256": content_hash(contexts)}
        with (writer.path / "design.json").open("x") as handle:
            json.dump(design, handle, indent=2)
        gates = _scientific_gates(config, data, backend)
        measurements_valid = True
        aggregate = defaultdict(lambda: {"n": 0, "uncalibrated_correct": 0,
                                         "calibrated_correct": 0, "generated_correct": 0,
                                         "generated_unparseable": 0,
                                         "by_gold_class": defaultdict(lambda: {"n": 0, "uncalibrated_correct": 0,
                                                                              "calibrated_correct": 0})})
        content_free = config.get("content_free_queries", ["N/A", "[MASK]", ""])
        floor = config.get("calibration_floor", 1e-12)
        score_cache, calibration_cache, generation_cache = {}, {}, {}
        with (writer.path / "calibration.jsonl").open("x") as calibrations:
            for context in contexts:
                demos = [pool[identity] for identity in context["demonstration_ids"]]
                ordered_hash = content_hash(context["ordered_demonstrations"])
                if ordered_hash not in calibration_cache:
                    calibration_cache[ordered_hash] = _calibration(backend, data.task_id, demos, labels, content_free, floor)
                calibration = calibration_cache[ordered_hash]
                calibrations.write(canonical_json({"context_id": context["context_id"],
                                                   "ordered_demo_hash": ordered_hash, **calibration}) + "\n")
                calibrations.flush()
                for query in data.test:
                    prompt = render_prompt(data.task_id, demos, query.text)
                    key = content_hash([ordered_hash, query.example_id, query.text_hash])
                    if key not in score_cache:
                        score_cache[key] = _score(backend, prompt, labels)
                    scores, measured = score_cache[key]
                    uncalibrated = _probabilities(_log_probabilities(scores))
                    calibrated_log = _log_probabilities({label: scores[label] - calibration["effective_log_label_prior"][label]
                                                         for label in labels})
                    calibrated = _probabilities(calibrated_log)
                    raw_prediction = _argmax(scores, labels)
                    calibrated_prediction = _argmax(calibrated_log, labels)
                    generation, generated_prediction = None, None
                    if config.get("generate_labels", True):
                        if key not in generation_cache:
                            generation_cache[key] = backend.generate_record(prompt, config.get("max_new_tokens", 10))
                        generation = generation_cache[key]
                        if generation.get("truncated") or generation.get("rejected"):
                            measurements_valid = False
                        if generation.get("measurement_status") != "measured":
                            measurements_valid = False
                        if generation.get("input_tokens_before") != measured["input_tokens_before"]:
                            raise ValueError("generation and restricted scoring used different prompt tokenizations")
                        if generation.get("input_token_ids_hash") not in (None, measured["input_token_ids_hash"]):
                            raise ValueError("generation and restricted scoring used different prompt token IDs")
                        generated_prediction = parse_label_strict(generation.get("raw_text", ""), labels)
                    record = {**measured, "task": data.task_id, "query_role": config["query_role"],
                        "query_id": query.example_id, "query_hash": query.text_hash, "gold_label": query.label,
                        "gold_label_index": query.label_index, "context_id": context["context_id"],
                        "condition": context["condition"], "requested_k": context["requested_k"],
                        "realized_k": len(demos), "draw_id": context["draw_id"],
                        "comparison_family": context["comparison_family"], "order": context["order"],
                        "composition": context["composition"], "forced_last_class": context["forced_last_class"],
                        "last_example_repeat": context["last_example_repeat"],
                        "demonstration_ids": context["demonstration_ids"],
                        "demonstration_hashes": context["demonstration_hashes"],
                        "demonstration_labels": context["demonstration_labels"],
                        "demonstration_multiset_id": context["demonstration_multiset_id"],
                        "paired_query_draw_id": content_hash([data.task_id, query.example_id,
                                                              context["requested_k"], context["draw_id"]]),
                        "score_evaluation_id": key, "prediction": raw_prediction,
                        "parse_status": "restricted_complete_label", "correct": raw_prediction == query.label,
                        "label_scores": scores, "uncalibrated_probabilities": uncalibrated,
                        "calibrated_probabilities": calibrated, "calibrated_prediction": calibrated_prediction,
                        "calibrated_correct": calibrated_prediction == query.label,
                        "label_prior": calibration["label_prior"], "log_label_prior": calibration["log_label_prior"],
                        "calibration_floor": floor, "generated_prediction": generated_prediction,
                        "generated_parse_status": ("disabled" if generation is None else
                            "rejected" if generation.get("rejected") else
                            "parseable" if generated_prediction is not None else "unparseable"),
                        "raw_text": None if generation is None else generation.get("raw_text"),
                        "generation_tokens": 0 if generation is None else generation.get("generation_tokens"),
                        "generation_rejected": False if generation is None else generation.get("rejected", False),
                        "generation_truncated": False if generation is None else generation.get("truncated", False),
                        "generation": generation, "model_id": model_config["id"],
                        "composition_reference_context_id": context["composition_reference_context_id"]}
                    writer.write_prediction(record)
                    metric = aggregate[context["context_id"]]
                    metric["n"] += 1
                    metric["uncalibrated_correct"] += raw_prediction == query.label
                    metric["calibrated_correct"] += calibrated_prediction == query.label
                    metric["generated_correct"] += generated_prediction == query.label
                    metric["generated_unparseable"] += generation is not None and generated_prediction is None
                    group = metric["by_gold_class"][query.label]
                    group["n"] += 1
                    group["uncalibrated_correct"] += raw_prediction == query.label
                    group["calibrated_correct"] += calibrated_prediction == query.label
        gates["all_contexts_untruncated_and_measured"] = measurements_valid
        gates["development_selection_verified"] = config.get("k_selection", {}).get("frozen") is True
        scientific = all(gates.values())
        writer.manifest["scientific_evidence"] = scientific
        writer.manifest["data_metadata"] = data.metadata
        writer.manifest["scientific_gates"] = gates
        tokenizer = getattr(backend, "_tok", None)
        model_object = getattr(backend, "_model", None)
        writer.manifest["backend_provenance"] = {
            "model_id": getattr(backend, "model_id", None),
            "requested_model_revision": model_config.get("revision"),
            **resolved_model,
            "requested_tokenizer_revision": model_config.get("tokenizer_revision") or model_config.get("revision"),
            "template": getattr(backend, "template", "backend_defined"),
            "native_template_text": getattr(tokenizer, "chat_template", None),
            "template_hash": backend._template_hash() if hasattr(backend, "_template_hash") else None,
            "context_capacity": getattr(backend, "max_seq_len", None),
            "device": getattr(backend, "device", None),
            "label_scoring_policy": getattr(backend, "label_scoring_policy", None)}
        metrics = {"contexts": dict(aggregate), "n_contexts": len(contexts),
                   "n_queries": len(data.test), "unique_score_evaluations": len(score_cache),
                   "unique_generation_evaluations": len(generation_cache),
                   "unique_calibration_contexts": len(calibration_cache),
                   "scientific_evidence": scientific, "query_role": config["query_role"],
                   "inference_status": "descriptive paired design; crossed query/draw inference required",
                   "generated_readout_enabled": config.get("generate_labels", True)}
        validation = {"passed": measurements_valid, "scientific_gates": gates,
                      "paired_order_multisets_frozen": True, "query_demo_overlap": False,
                      "content_free_selection_uses_gold_labels": False,
                      "majority_match_uses_query_outcomes": False,
                      "k0_is_single_shared_reference": True,
                      "label_scoring_readout_is_separate_from_generated_readout": True}
        if not measurements_valid:
            raise RuntimeError("generation measurements failed context/format validation; records retained")
        writer.complete(metrics, validation)
        return writer.path
    except Exception as error:
        writer.fail(f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    if args.output_root:
        config["output_root"] = args.output_root
    if args.run_id:
        config["run_id"] = args.run_id
    path = run(config)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
