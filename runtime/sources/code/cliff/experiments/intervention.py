"""Audited region-restricted LoRA with independently initialized replicas.

Label-span supervision is the new-run default. Historical whole-prompt loss is
available explicitly; model-wide fallback and silent truncation are forbidden.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
import statistics as st
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ..backends import require_weight_access
from ..data import TASKS, exclude_examples, load_task, normalized_text_hash
from ..paths import outputs_dir
from ..prompting import Cell, parse_label, parse_label_strict, render_prompt
from ..revision_records import RunWriter, content_hash, file_hash

REGIONS = {"early": "first n blocks", "mid": "centred n blocks", "late": "last n blocks"}
_BLOCK = re.compile(r"(?:^|\.)(?:layers|h|blocks|block)\.(\d+)\.")
_ATTN = {"q_proj", "k_proj", "v_proj", "o_proj", "wqkv", "wq", "wk", "wv", "wo",
         "query", "key", "value", "out_proj", "c_attn", "query_key_value", "qkv_proj",
         "qkv", "Wqkv", "W_pack", "kv_proj"}
_MLP = {"gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "fc1", "fc2", "c_fc",
        "dense_h_to_4h", "dense_4h_to_h", "gate_up_proj"}


def seed_everything(seed: int) -> None:
    """Call before checkpoint/adapter construction, as well as training."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def region_window(n_layers: int, n_blocks: int, region: str,
                  block_start: int | None = None) -> range:
    if n_layers < 1 or n_blocks < 1 or n_blocks > n_layers:
        raise SystemExit("the requested positive block count must fit the model stack")
    if region not in REGIONS:
        raise SystemExit(f"region must be one of {sorted(REGIONS)}")
    start = block_start if block_start is not None else {
        "early": 0, "mid": (n_layers - n_blocks) // 2, "late": n_layers - n_blocks}[region]
    if not (0 <= start and start + n_blocks <= n_layers):
        raise SystemExit(f"window {start}..{start+n_blocks-1} outside a {n_layers}-block stack")
    return range(start, start + n_blocks)


def _component(name: str) -> str | None:
    leaf, context = name.rsplit(".", 1)[-1], name.lower().split(".")[:-1]
    if leaf in _ATTN:
        return "attn"
    if leaf in _MLP:
        return "mlp"
    if leaf in {"dense", "c_proj"}:
        if any("attn" in part or "attention" in part for part in context):
            return "attn"
        if any("mlp" in part or "ffn" in part or "feed_forward" in part for part in context):
            return "mlp"
    return None


def resolve_targets(model, window: range, component: str = "attn",
                    allow_fallback: bool = False) -> list[str]:
    """Require complete module coverage in every block, including fused QKV."""
    if allow_fallback:
        raise ValueError("model-wide fallback is disabled for revised experiments; use archived code for reconstruction")
    if component not in {"attn", "mlp", "both"} or not list(window):
        raise ValueError("a valid component and nonempty target window are required")
    requested = {"attn", "mlp"} if component == "both" else {component}
    coverage = {i: {part: set() for part in requested} for i in window}
    targets = []
    for name, module in model.named_modules():
        match, part = _BLOCK.search(name), _component(name)
        if (match and int(match.group(1)) in window and part in requested
                and getattr(module, "weight", None) is not None and not list(module.children())):
            targets.append(name)
            coverage[int(match.group(1))][part].add(name[match.end():])
    for part in requested:
        expected = set().union(*(coverage[i][part] for i in window))
        for i in window:
            if not expected or coverage[i][part] != expected:
                missing = sorted(expected - coverage[i][part])
                raise RuntimeError(f"incomplete {part} target coverage in block {i}; missing "
                                   f"{missing or 'all supported projections'}; refusing partial targeting")
    return sorted(targets)


def assert_targeting(peft_model, window: range, expected_targets: list[str] | None = None) -> dict:
    """Require exact A/B adapters, all requested blocks and frozen base weights."""
    adapters = {"A": set(), "B": set()}
    for name, _ in peft_model.named_modules():
        for side in adapters:
            marker = ".lora_" + side
            if marker in name:
                adapters[side].add(name.split(marker, 1)[0])
    if not adapters["A"] or adapters["A"] != adapters["B"]:
        raise RuntimeError("nonempty matching LoRA A/B module coverage is required")
    actual, seen = adapters["A"], set()
    for name in actual:
        match = _BLOCK.search(name)
        if not match:
            raise RuntimeError(f"cannot assign adapter to a transformer block: {name}")
        seen.add(int(match.group(1)))
    if seen != set(window):
        raise RuntimeError(f"adapter blocks {sorted(seen)} differ from requested {list(window)}")
    if expected_targets is not None:
        resolved = []
        for name in actual:
            matches = [target for target in expected_targets if name == target or name.endswith("." + target)]
            if len(matches) != 1:
                raise RuntimeError("instantiated adapter differs from registered target set")
            resolved.extend(matches)
        if set(resolved) != set(expected_targets) or len(actual) != len(expected_targets):
            raise RuntimeError("instantiated adapter modules differ from registered target set")
    trainable, frozen, covered, names = 0, 0, set(), []
    for name, parameter in peft_model.named_parameters():
        if parameter.requires_grad:
            match = re.search(r"\.lora_[AB]\.", name)
            prefix = name[:match.start()] if match else None
            if prefix not in actual:
                raise RuntimeError(f"unexpected trainable base/non-target parameter: {name}")
            covered.add((prefix, match.group(0)))
            names.append(name)
            trainable += parameter.numel()
        else:
            frozen += parameter.numel()
    expected_parameters = {(name, side) for name in actual for side in (".lora_A.", ".lora_B.")}
    if covered != expected_parameters or not trainable or not frozen:
        raise RuntimeError("every registered adapter must have trainable parameters")
    return {"adapted_blocks": sorted(seen), "requested": [window.start, window.stop - 1],
            "registered_targets": expected_targets or sorted(actual), "instantiated_targets": sorted(actual),
            "trainable_parameters": trainable, "frozen_parameters": frozen,
            "trainable_parameter_names": names, "localization_valid": True}


def _text_hash(text):
    return normalized_text_hash(text)


def _identity(example):
    return getattr(example, "example_id", None) or _text_hash(example.text)


def filter_excluded(pool, exclusions):
    """Remove registered development/reserved identities and normalized hashes."""
    ids = {row.get("id", row.get("query_id", row.get("example_id"))) for row in exclusions}
    hashes = {row.get("hash", row.get("query_hash", row.get("text_hash"))) for row in exclusions}
    return [example for example in pool if _identity(example) not in ids and _text_hash(example.text) not in hashes]


def read_exclusion_manifest(path, *, require_development=False):
    """Read explicit ID/hash lists, W2/W3 designs, preflights or prediction JSONL."""
    text = Path(path).read_text()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows, roles = [], set()
    def visit(value):
        if isinstance(value, list):
            for entry in value:
                visit(entry)
        elif isinstance(value, dict):
            if value.get("query_role"):
                roles.add(value["query_role"])
            identity = value.get("id", value.get("query_id", value.get("example_id")))
            text_hash = value.get("hash", value.get("query_hash", value.get("text_hash")))
            if identity or text_hash:
                rows.append({"id": identity, "hash": text_hash})
            if value.get("query_ids") or value.get("query_hashes"):
                rows.extend({"id": identity} for identity in value.get("query_ids", []))
                rows.extend({"hash": text_hash} for text_hash in value.get("query_hashes", []))
            for entry in value.values():
                if isinstance(entry, (list, dict)):
                    visit(entry)
    visit(payload)
    if require_development and "development" not in roles:
        raise ValueError("development exclusion manifest must explicitly identify its development query role")
    if not rows:
        raise ValueError("exclusion manifest contains no identifiable example IDs or text hashes")
    unique = {content_hash(row): row for row in rows}
    return list(unique.values()), {"path": str(path), "sha256": file_hash(path),
                                  "n_exclusions": len(unique), "roles": sorted(roles)}


def _pool_options(pool_spec, task_id, role):
    spec = (pool_spec or {}).get(task_id, pool_spec or {})
    if not spec:
        return {}
    options = {"n_train": spec.get("n_train"), "train_split": spec.get("train_split"),
               "test_split": spec.get("test_split", spec.get("query_split"))}
    fields = {"calibration": ("calibration_query", "train"),
              "demonstration": ("demonstration", "train"),
              "evaluation": ("evaluation_demo", "demonstration", "train")}[role]
    for prefix in fields:
        indices, interval = spec.get(prefix + "_indices"), spec.get(prefix + "_index_range")
        if indices is not None and interval is not None:
            raise ValueError("register source indices or an index range, not both")
        if indices is not None or interval is not None:
            options["train_indices"] = list(range(*interval)) if interval is not None else indices
            break
    for key in ("query_indices", "test_indices"):
        if key in spec:
            options["test_indices"] = spec[key]
    if spec.get("query_index_range") is not None:
        options["test_indices"] = list(range(*spec["query_index_range"]))
    return {key: value for key, value in options.items() if value is not None or key == "n_train"}


def _task_revision(pool_spec, task_id, fallback=None):
    spec = (pool_spec or {}).get(task_id, pool_spec or {})
    return spec.get("revision", fallback)


def shuffle_query_targets(rows, seed):
    """Random row-target permutation with exact marginals, never a class map.

    Fixed points are allowed. Reject structurally degenerate samples for which a
    rowwise shuffle cannot be distinguished from a consistent class permutation.
    The registered rejection rule uses target structure only, never outcomes.
    """
    labels = [row["target"] for row in rows]
    if len(set(labels)) < 2 or max(Counter(labels).values(), default=0) < 2:
        raise ValueError("query-target shuffle needs multiple labels and repeated source classes")
    rng = random.Random(seed)
    for attempt in range(1, 101):
        assigned = list(labels)
        rng.shuffle(assigned)
        mapping = {label: {target for original, target in zip(labels, assigned) if original == label}
                   for label in set(labels)}
        if any(len(values) > 1 for values in mapping.values()):
            break
    else:
        raise ValueError("could not construct a rowwise shuffle distinct from a class permutation")
    result, manifest = [], []
    for row, target in zip(rows, assigned):
        changed = {**row, "unshuffled_target": row["target"], "target": target,
                   "target_policy": "shuffle_query"}
        result.append(changed)
        manifest.append({"query_id": row["query_id"], "original_target": row["target"], "target": target})
    return result, {"policy": "shuffle_query", "seed": seed, "structural_rejection_attempts": attempt,
                    "target_distribution_before": dict(Counter(labels)),
                    "target_distribution_after": dict(Counter(assigned)),
                    "fraction_unchanged": sum(a == b for a, b in zip(labels, assigned)) / len(labels),
                    "consistent_class_permutation": False, "target_map": manifest,
                    "target_map_sha256": content_hash(manifest)}


def balanced_calibration_ks(n, ks, seed):
    """Assign a registered K mixture without consulting examples or outcomes."""
    values = list(ks)
    if not values or len(set(values)) != len(values) or min(values) < 0:
        raise ValueError("calibration K values must be distinct nonnegative integers")
    rng = random.Random(seed)
    remainder_order = list(values)
    rng.shuffle(remainder_order)
    assignment = values * (n // len(values)) + remainder_order[:n % len(values)]
    rng.shuffle(assignment)
    return assignment


def flatten_gold_presentations(rows, task_id):
    """Turn every K8 demo/query label presentation into its own K0 example.

    Multiplicities are preserved, including a source example appearing in several
    contexts. This matches example/target presentations, not instruction tokens,
    active tokens or supervised tokens.
    """
    if not rows or any(row["actual_k"] != 8 or row["target"] != row["gold_label"]
                       or row.get("target_policy", "gold") != "gold" for row in rows):
        raise ValueError("exposure-matched K0 requires unshuffled gold K8 source contexts")
    flattened, contexts, counts = [], [], Counter()
    for context_index, row in enumerate(rows):
        presentations = row["gold_presentations"]
        if len(presentations) != 9:
            raise ValueError("K8 source must contain exactly eight demos and one query")
        context_id = f"source_context_{context_index:06d}"
        contexts.append({"source_context_id": context_id, "query_id": row["query_id"],
                         "prompt_sha256": content_hash(row["prompt"]),
                         "presentations": presentations})
        for position, presentation in enumerate(presentations):
            counts[(presentation["example_id"], presentation["text_hash"], presentation["gold_label"])] += 1
            flattened.append({"prompt": render_prompt(task_id, [], presentation["text"]),
                "target": presentation["gold_label"], "gold_label": presentation["gold_label"],
                "query_id": presentation["example_id"], "query_text": presentation["text"],
                "query_text_hash": presentation["text_hash"], "demo_ids": [], "demo_text_hashes": [],
                "requested_k": 0, "actual_k": 0, "source_requested_k": 8,
                "presentation_id": f"{context_id}/presentation_{position:02d}",
                "source_context_id": context_id, "source_calibration_query_id": row["query_id"],
                "source_role": presentation["role"], "source_position": position,
                "demo_draw_seed": row["demo_draw_seed"], "target_policy": "gold",
                "zero_shot_generation": None, "zero_shot_record": None, "used_fallback": False})
    observed = Counter((row["query_id"], row["query_text_hash"], row["target"]) for row in flattened)
    if observed != counts:
        raise RuntimeError("flattening changed source example/target presentation multiplicities")
    manifest = [{"example_id": identity, "text_hash": text_hash, "target": target, "count": count}
                for (identity, text_hash, target), count in sorted(counts.items())]
    return flattened, {"source_context_count": len(rows), "training_row_count": len(flattened),
        "source_k": 8, "training_k": 0, "row_expansion_factor": 9,
        "source_context_manifest": contexts, "presentation_counts": manifest,
        "presentation_counts_sha256": content_hash(manifest), "exact_target_presentations_preserved": True,
        "unique_example_ids_preserved": True, "instruction_presentations_source": len(rows),
        "instruction_presentations_training": len(flattened),
        "matched": ["example_identity_multiset", "gold_target_presentation_multiset", "optimizer_update_count"],
        "unmatched": ["instruction_presentations", "active_tokens", "supervised_tokens", "within_context_dependence"]}


def exposure_optimizer_budget(source_rows, epochs, gradient_accumulation_steps):
    """Require complete effective batches; no epoch-tail rounding is accepted."""
    if source_rows < 1 or epochs < 1 or int(epochs) != epochs or gradient_accumulation_steps < 1:
        raise ValueError("positive integral rows, epochs and accumulation are required")
    if source_rows % gradient_accumulation_steps:
        raise ValueError("exposure matching requires source row count exactly divisible by gradient accumulation steps")
    return {"source_rows_per_epoch": source_rows, "training_rows_per_epoch": source_rows * 9,
            "source_gradient_accumulation_steps": gradient_accumulation_steps,
            "training_gradient_accumulation_steps": gradient_accumulation_steps * 9,
            "epochs": epochs, "optimizer_updates": source_rows // gradient_accumulation_steps * epochs,
            "exact_divisibility": True}


def build_calibration(backend, task_id: str, arm: str, n: int, k: int, seed: int = 42,
                      *, demo_seed: int | None = None, query_split_seed: int = 0,
                      invalid_target: str = "exclude", data_revision: str | None = None,
                      n_test: int = 200, parser: str = "strict", target_policy: str = "gold",
                      target_seed: int | None = None, excluded_examples=(), pool_spec=None,
                      intervention_ks=None, k_seed=None) -> tuple[list[dict], dict]:
    """Select unique queries first and remove ID/text duplicates from their demos.

    Invalid self-labels are excluded by default. The explicit first-label policy
    remains an audit comparison; invalid targets are never silently replaced.
    """
    from ..data import exact_balanced_sample
    if arm not in {"remove", "induce"} or invalid_target not in {"exclude", "error", "first-label"}:
        raise ValueError("unknown arm or invalid-target policy")
    if n < 1 or k < 0:
        raise ValueError("positive calibration count and nonnegative K required")
    if parser not in {"strict", "legacy_prefix"}:
        raise ValueError("unknown label parser")
    if target_policy not in {"gold", "shuffle_query"} or (target_policy != "gold" and arm != "remove"):
        raise ValueError("query-target shuffle is supported only for the supervised remove arm")
    k_seed = seed if k_seed is None else k_seed
    assigned_ks = balanced_calibration_ks(n, [k] if intervention_ks is None else intervention_ks, k_seed)
    parse = parse_label_strict if parser == "strict" else parse_label
    data_revision = _task_revision(pool_spec, task_id, data_revision)
    data = load_task(task_id, shard_seed=query_split_seed, revision=data_revision, n_test=n_test,
                     **_pool_options(pool_spec, task_id, "calibration"))
    pool = filter_excluded(exclude_examples(data.train, data.test), excluded_examples)
    demo_options = _pool_options(pool_spec, task_id, "demonstration")
    if demo_options and demo_options != _pool_options(pool_spec, task_id, "calibration"):
        demonstration_data = load_task(task_id, shard_seed=query_split_seed, revision=data_revision,
                                       n_test=n_test, **demo_options)
        demo_pool = filter_excluded(exclude_examples(demonstration_data.train, data.test), excluded_examples)
    else:
        demo_pool = pool
    labels = TASKS[task_id]["label_names"]
    rng, demo_rng = random.Random(seed), random.Random(seed if demo_seed is None else demo_seed)
    candidates = list(pool)
    rng.shuffle(candidates)
    seen_ids, seen_hashes, queries = set(), set(), []
    for query in candidates:
        identity, text_hash = _identity(query), _text_hash(query.text)
        if identity in seen_ids or text_hash in seen_hashes:
            continue
        seen_ids.add(identity)
        seen_hashes.add(text_hash)
        queries.append(query)
    if n > len(queries):
        raise ValueError(f"requested {n} unique calibration queries but pool has {len(queries)}")
    rows, excluded_targets, n_invalid, n_fallback = [], [], 0, 0
    for query, row_k in zip(queries[:n], assigned_ks):
        draw_seed = demo_rng.randint(0, 2 ** 31 - 1)
        demos = exact_balanced_sample(demo_pool, row_k, draw_seed, exclude=[query])
        target, generation, fallback, generation_record = query.label, None, False, None
        if arm == "induce":
            zero_shot_prompt = render_prompt(task_id, [], query.text)
            if hasattr(backend, "generate_record"):
                generation_record = backend.generate_record(zero_shot_prompt)
                if generation_record.get("rejected") or generation_record.get("truncated"):
                    raise RuntimeError("invalid context budget in zero-shot calibration target generation")
                generation = generation_record["raw_text"]
            else:
                generation = backend.generate(zero_shot_prompt)
            target = parse(generation, labels)
            if target is None:
                n_invalid += 1
                if invalid_target == "error":
                    raise ValueError(f"unparseable induce target for query {_identity(query)}")
                if invalid_target == "exclude":
                    excluded_targets.append({"query_id": _identity(query), "gold_label": query.label,
                                             "raw_text": generation, "generation_record": generation_record})
                    continue
                target, fallback = labels[0], True
                n_fallback += 1
        rows.append({"prompt": render_prompt(task_id, demos, query.text), "target": target,
                     "gold_label": query.label, "query_id": _identity(query), "query_text": query.text,
                     "query_text_hash": _text_hash(query.text), "demo_ids": [_identity(d) for d in demos],
                     "demo_text_hashes": [_text_hash(d.text) for d in demos], "demo_draw_seed": draw_seed,
                     "requested_k": row_k, "actual_k": len(demos), "zero_shot_generation": generation,
                     "zero_shot_record": generation_record, "used_fallback": fallback,
                     "target_policy": target_policy,
                     "gold_presentations": [{"example_id": _identity(example), "text_hash": _text_hash(example.text),
                         "text": example.text, "gold_label": example.label, "role": role}
                         for example, role in [(demo, "demonstration") for demo in demos] + [(query, "query")]]})
    if not rows:
        raise ValueError("no valid calibration targets remain; intervention cannot proceed")
    target_map = [{"query_id": row["query_id"], "original_target": row["target"], "target": row["target"]}
                  for row in rows]
    target_randomization = {"policy": "gold" if arm == "remove" else "base_model_self_label",
                            "target_map": target_map, "target_map_sha256": content_hash(target_map)}
    if target_policy == "shuffle_query":
        rows, target_randomization = shuffle_query_targets(rows, seed if target_seed is None else target_seed)
    stats = {"n": len(rows), "n_requested": n, "n_invalid_targets": n_invalid,
             "n_excluded_targets": n_invalid - n_fallback, "n_fallback": n_fallback,
             "fallback_rate": n_fallback / n, "invalid_target_rate": n_invalid / n,
             "invalid_target_policy": invalid_target,
             "target_gold_agreement": sum(r["target"] == r["gold_label"] for r in rows) / len(rows),
             "target_distribution": dict(Counter(r["target"] for r in rows)),
             "calibration_seed": seed, "demo_seed": seed if demo_seed is None else demo_seed,
             "query_split_seed": query_split_seed, "data_revision": data_revision,
             "dataset_metadata": getattr(data, "metadata", {}), "parser": parser,
             "n_heldout_excluded": len(data.train) - len(pool), "excluded_targets": excluded_targets,
             "target_policy": target_policy, "target_randomization": target_randomization,
             "intervention_ks": sorted(set(assigned_ks)), "calibration_k_seed": k_seed,
             "requested_k_distribution": dict(Counter(map(str, assigned_ks))),
             "realized_k_distribution": dict(Counter(str(row["actual_k"]) for row in rows)),
             "k_assignment_policy": "seeded_balanced_per_query_before_target_generation",
             "source_pool_spec": pool_spec, "reserved_exclusions_sha256": content_hash(list(excluded_examples)),
             "demonstration_pool_count": len(demo_pool)}
    return rows, stats


def encode_training_rows(backend, rows, label_only_loss=True):
    """Align complete-string targets; preserve EOS even when PAD equals EOS.

    The backend's longest-common-prefix boundary includes any token merging across
    prompt/label. No truncation is permitted. Padding is masked by position rather
    than token identity, so real EOS tokens remain supervised.
    """
    if not rows:
        raise ValueError("cannot train on an empty calibration set")
    tok = backend.tokenizer()
    if tok.eos_token_id is None or tok.pad_token_id is None:
        raise ValueError("training requires explicit EOS and PAD token IDs")
    encoded, starts = [], []
    for index, row in enumerate(rows):
        if not str(row["target"]).strip():
            raise ValueError(f"empty target at calibration row {index}")
        ids, start = backend.encode_completion(row["prompt"], getattr(backend, "label_prefix", " ") + row["target"])
        ids = list(ids)
        if not 0 < start < len(ids):
            raise ValueError(f"no supervised label token at calibration row {index}")
        if ids[-1] != tok.eos_token_id:
            ids.append(tok.eos_token_id)
        if len(ids) > backend.max_seq_len:
            raise ValueError(f"calibration row {index} needs {len(ids)} tokens including EOS; "
                             f"context budget is {backend.max_seq_len}; truncation is forbidden")
        encoded.append(ids)
        starts.append(start)
    width = max(map(len, encoded))
    data = {"input_ids": [], "attention_mask": [], "labels": []}
    left = getattr(tok, "padding_side", "right") == "left"
    for ids, start in zip(encoded, starts):
        pad = width - len(ids)
        mask = [1] * len(ids)
        labels = [-100] * start + ids[start:] if label_only_loss else list(ids)
        # The first real token has no preceding real token. With left padding,
        # supervising it would train a prediction from the final PAD position.
        labels[0] = -100
        data["input_ids"].append([tok.pad_token_id] * pad + ids if left else ids + [tok.pad_token_id] * pad)
        data["attention_mask"].append([0] * pad + mask if left else mask + [0] * pad)
        data["labels"].append([-100] * pad + labels if left else labels + [-100] * pad)
        if not any(label != -100 for label in data["labels"][-1][1:]):
            raise ValueError("no shifted supervised target tokens remain")
    return data, {"sequence_lengths": [len(ids) for ids in encoded], "completion_start_indices": starts,
                  "sequence_sha256": [hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                                      for ids in encoded],
                  "n_truncated": 0, "boundary_policy": "complete-string longest-common-prefix",
                  "eos_supervised": True, "padding_mask_source": "attention_mask",
                  "active_tokens_per_epoch": sum(sum(mask) for mask in data["attention_mask"]),
                  "supervised_tokens_per_epoch": sum(sum(label != -100 for label in labels[1:])
                                                       for labels in data["labels"]),
                  "prompt_tokens_per_epoch": sum(starts), "instruction_presentations_per_epoch": len(rows)}


def parameter_digest(model, *, trainable):
    """Streaming SHA256 of exact parameter bytes, including bf16 tensors."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if bool(parameter.requires_grad) != trainable:
            continue
        digest.update(json.dumps([name, list(parameter.shape), str(parameter.dtype)]).encode())
        raw = parameter.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    return digest.hexdigest()


def _new_run_dir(parent, prefix):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(parent) / f"{prefix}_{stamp}_{uuid4().hex[:10]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _artifact_hashes(directory):
    result = {}
    for path in sorted(Path(directory).rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.relative_to(directory))] = digest.hexdigest()
    return result


def consolidate(backend, rows, window, component, lora_r, lora_alpha, lr, epochs,
                allow_fallback=False, label_only_loss=True, *, train_seed=0, output_dir=None,
                untrained_adapter=False, gradient_accumulation_steps=8, expected_optimizer_updates=None,
                exposure_reference_rows=None, targeting_contract=None):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient accumulation steps must be positive")
    if expected_optimizer_updates is not None and (len(rows) % gradient_accumulation_steps
            or len(rows) // gradient_accumulation_steps * epochs != expected_optimizer_updates):
        raise ValueError("registered optimizer-update count requires exact accumulation divisibility")
    seed_everything(train_seed)
    model = backend.torch_model()
    if getattr(model, "peft_config", None):
        raise RuntimeError("a fresh base checkpoint is required for each training replica")
    targets = resolve_targets(model, window, component, allow_fallback)
    enc, encoding = encode_training_rows(backend, rows, label_only_loss)
    reference_encoding = (encode_training_rows(backend, exposure_reference_rows, label_only_loss)[1]
                          if exposure_reference_rows is not None else None)
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)
    previous_use_cache = getattr(model.config, "use_cache", True)
    model.config.use_cache = False
    # B=0 initialization must preserve the base prediction before optimization.
    # A short real token prefix avoids retaining a full long-context logit cube.
    probe_ids = [token for token, present in zip(enc["input_ids"][0], enc["attention_mask"][0]) if present][:32]
    probe_mask = [1] * len(probe_ids)
    input_device = model.get_input_embeddings().weight.device
    probe = {"input_ids": torch.tensor([probe_ids], device=input_device),
             "attention_mask": torch.tensor([probe_mask], device=input_device)}
    model.eval()
    with torch.inference_mode():
        base_logits = model(**probe).logits[:, -1].float().cpu()
    peft_model = get_peft_model(model, LoraConfig(task_type=TaskType.CAUSAL_LM,
        r=lora_r, lora_alpha=lora_alpha, target_modules=targets, lora_dropout=0.05, bias="none"))
    info = assert_targeting(peft_model, window, targets)
    if targeting_contract is not None:
        from ..revision_target_contract import audit_loaded_targeting
        info["registered_target_count_audit"] = audit_loaded_targeting(targeting_contract, backend, peft_model,
            window=window, component=component, rank=lora_r, alpha=lora_alpha)
    peft_model.eval()
    with torch.inference_mode():
        initial_logits = peft_model(**probe).logits[:, -1].float().cpu()
    if not torch.isfinite(initial_logits).all() or not torch.allclose(base_logits, initial_logits, rtol=1e-5, atol=1e-5):
        raise RuntimeError("zero-initialized adapter is not prediction-equivalent to its base")
    info["no_op_max_abs_logit_difference"] = float((base_logits - initial_logits).abs().max())
    initial_hash = parameter_digest(peft_model, trainable=True)
    frozen_before = parameter_digest(peft_model, trainable=False)
    destination = Path(output_dir) if output_dir else _new_run_dir(outputs_dir(), "training")
    destination.mkdir(parents=True, exist_ok=True)
    if targeting_contract is not None:
        contract_path = destination / "targeting_contract.json"
        with contract_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(targeting_contract, indent=2, allow_nan=False)+"\n")
        info["targeting_contract_file_sha256"] = file_hash(contract_path)
    trainer_dir, adapter_dir = destination / "trainer", destination / "adapter"
    trainer_dir.mkdir(exist_ok=False)
    if adapter_dir.exists():
        raise FileExistsError(f"refusing to overwrite adapter: {adapter_dir}")
    # Preserve the actual supervised sequences and masks, including their
    # padding positions, rather than relying on future tokenizer versions.
    with gzip.open(trainer_dir / "tokenized_calibration.jsonl.gz", "xt", encoding="utf-8") as stream:
        for index, row in enumerate(rows):
            payload = {"row_index": index, "query_id": row.get("query_id"),
                       **{key: values[index] for key, values in enc.items()}}
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    gradients = {"nonzero_observed": False, "finite": True}

    def audit_gradient(gradient):
        if not torch.isfinite(gradient).all():
            gradients["finite"] = False
            raise FloatingPointError("nonfinite adapter gradient")
        if torch.count_nonzero(gradient).item():
            gradients["nonzero_observed"] = True
        return gradient

    class AuditedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all():
                raise FloatingPointError("nonfinite intervention training loss")
            return result

    args = TrainingArguments(output_dir=str(trainer_dir), num_train_epochs=epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=gradient_accumulation_steps, learning_rate=lr,
        save_strategy="no", logging_steps=25, report_to="none", max_grad_norm=1.0,
        seed=train_seed, data_seed=train_seed,
        bf16=(backend.device == "cuda" and torch.cuda.is_bf16_supported()),
        use_cpu=(backend.device == "cpu"), remove_unused_columns=False, label_names=["labels"])
    if expected_optimizer_updates is not None and args.world_size != 1:
        raise ValueError("exposure-matched optimizer accounting currently requires one training process")
    (trainer_dir / "training_args.json").write_text(args.to_json_string(), encoding="utf-8")
    trainer, metrics, optimizer_updates = None, {}, 0
    started = time.perf_counter()
    if not untrained_adapter:
        hooks = [p.register_hook(audit_gradient) for p in peft_model.parameters() if p.requires_grad]
        trainer = AuditedTrainer(model=peft_model, args=args, train_dataset=Dataset.from_dict(enc))
        try:
            result = trainer.train()
        finally:
            for hook in hooks:
                hook.remove()
        metrics, optimizer_updates = result.metrics, trainer.state.global_step
    elapsed = time.perf_counter() - started
    if not untrained_adapter and not gradients["nonzero_observed"]:
        raise RuntimeError("training produced no nonzero adapter gradients")
    final_hash = parameter_digest(peft_model, trainable=True)
    if not untrained_adapter and initial_hash == final_hash:
        raise RuntimeError("training did not change any intended adapter parameter")
    if untrained_adapter and initial_hash != final_hash:
        raise RuntimeError("untrained control changed adapter parameters")
    if expected_optimizer_updates is not None and not untrained_adapter and optimizer_updates != expected_optimizer_updates:
        raise RuntimeError("actual optimizer-update count differs from registered exposure-matched count")
    frozen_after = parameter_digest(peft_model, trainable=False)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen base parameters changed during consolidation")
    info.update({"train_seed": train_seed, "encoding": encoding, "initial_adapter_sha256": initial_hash,
                 "adapter_sha256": final_hash, "frozen_sha256": frozen_after,
                 "parameter_hash_policy": "names_shapes_dtypes_and_exact_parameter_bytes_sha256",
                 "frozen_weights_unchanged": True, "gradient_audit": gradients,
                 "gradient_requirement": "explicitly_skipped_untrained_control" if untrained_adapter else "finite_nonzero_required",
                 "training_replica_kind": "adapter_initialization" if untrained_adapter else "trained_adapter",
                 "n_trained": 0 if untrained_adapter else 1, "optimizer_updates": optimizer_updates,
                 "gradient_accumulation_steps": gradient_accumulation_steps,
                 "training_elapsed_seconds": elapsed,
                 "training_active_tokens": 0 if untrained_adapter else encoding["active_tokens_per_epoch"] * epochs,
                 "training_supervised_tokens": 0 if untrained_adapter else encoding["supervised_tokens_per_epoch"] * epochs,
                 "training_metrics": metrics, "adapter_path": str(adapter_dir)})
    if reference_encoding is not None:
        info["exposure_budget_comparison"] = {"source": reference_encoding, "training": encoding,
            "epochs": epochs, "registered_optimizer_updates": expected_optimizer_updates,
            "optimizer_updates_matched": optimizer_updates == expected_optimizer_updates,
            "interpretation": "example/target presentations and optimizer updates matched; instruction, active and supervised tokens may differ"}
    adapter_dir.mkdir(exist_ok=False)
    peft_model.save_pretrained(adapter_dir)
    if trainer is not None:
        trainer.save_state()
    if trainer is not None and trainer.optimizer is not None:
        torch.save(trainer.optimizer.state_dict(), trainer_dir / "optimizer.pt")
    if trainer is not None and trainer.lr_scheduler is not None:
        torch.save(trainer.lr_scheduler.state_dict(), trainer_dir / "scheduler.pt")
    rng_state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    import numpy as np
    rng_state["numpy"] = np.random.get_state()
    if torch.cuda.is_available():
        rng_state["cuda"] = torch.cuda.get_rng_state_all()
    torch.save(rng_state, trainer_dir / "rng_state.pt")
    info["adapter_artifact_sha256"] = _artifact_hashes(adapter_dir)
    info["trainer_artifact_sha256"] = _artifact_hashes(trainer_dir)
    (destination / "training_audit.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    model.config.use_cache = previous_use_cache
    peft_model.eval()
    backend._model = peft_model
    return info


def evaluate(backend, tasks, ks, seeds, n_test, *, query_split_seed=0,
             record_sink=None, record_context=None, data_revision=None,
             parser="strict", allow_unmeasured=False, excluded_examples=(), pool_spec=None) -> list[Cell]:
    from ..data import exact_balanced_sample
    if parser not in {"strict", "legacy_prefix"}:
        raise ValueError("unknown label parser")
    if not hasattr(backend, "generate_record") and not allow_unmeasured:
        raise ValueError("scientific evaluation requires measured generate_record output")
    parse = parse_label_strict if parser == "strict" else parse_label
    cells = []
    for task_id in tasks:
        task_revision = _task_revision(pool_spec, task_id, data_revision)
        data = load_task(task_id, n_test=n_test, shard_seed=query_split_seed, revision=task_revision,
                         **_pool_options(pool_spec, task_id, "evaluation"))
        queries = filter_excluded(data.test, excluded_examples)
        pool = filter_excluded(exclude_examples(data.train, data.test), excluded_examples)
        if not queries:
            raise ValueError("no evaluation queries remain after registered development/reserved exclusions")
        for k in ks:
            for seed in (seeds[:1] if k == 0 else seeds):
                correct = unpar = 0
                demos = exact_balanced_sample(pool, k, seed)
                for ex in queries:
                    prompt = render_prompt(task_id, demos, ex.text)
                    generation = (backend.generate_record(prompt) if hasattr(backend, "generate_record") else
                                  {"raw_text": backend.generate(prompt), "rejected": False, "truncated": False,
                                   "measurement_status": "legacy_backend"})
                    pred = None if generation.get("rejected") else parse(generation["raw_text"], data.label_names)
                    record = {**(record_context or {}), **generation, "task": task_id, "k": k,
                              "demo_seed": seed, "query_id": _identity(ex), "query_text_hash": _text_hash(ex.text),
                              "gold_label": ex.label, "prediction": pred, "parsed_label": pred,
                              "correct": None if generation.get("rejected") else pred == ex.label,
                              "demo_ids": [_identity(d) for d in demos],
                              "actual_k": len(demos), "query_split_seed": query_split_seed,
                              "dataset_metadata": getattr(data, "metadata", {}),
                              "requested_k": k, "realized_k": len(demos), "query_hash": _text_hash(ex.text),
                              "demonstration_ids": [_identity(d) for d in demos],
                              "demonstration_hashes": [_text_hash(d.text) for d in demos],
                              "demonstration_labels": [d.label for d in demos], "draw_id": seed,
                              "condition": (record_context or {}).get("phase", "baseline"), "parser": parser,
                              "parse_status": "rejected" if generation.get("rejected") else
                                              ("parseable" if pred is not None else "unparseable")}
                    if record_sink:
                        record_sink(record)
                    if generation.get("rejected") or generation.get("truncated"):
                        raise RuntimeError(f"invalid context budget for {task_id} K={k}; use a feasible preregistered K grid")
                    if generation.get("measurement_status") != "measured" and not allow_unmeasured:
                        raise ValueError("scientific evaluation requires measured token-budget metadata")
                    unpar += pred is None
                    correct += pred == ex.label
                n = len(queries)
                cell = Cell(task_id, k, seed, correct / max(1, n), n, unpar)
                cells.append(cell)
                print(f"  {task_id:9s} K={k:<3d} draw={seed} acc={cell.accuracy:.3f} unpar={unpar}/{n}")
    return cells


def _paired_summary(before, after, tasks, ks):
    summary = {}
    for task in tasks:
        summary[task] = {}
        for k in ks:
            base = {c.seed: c for c in before if c.task == task and c.k == k}
            tuned = {c.seed: c for c in after if c.task == task and c.k == k}
            draws = sorted(set(base) & set(tuned))
            if draws:
                deltas = [tuned[d].accuracy - base[d].accuracy for d in draws]
                summary[task][str(k)] = {"before": st.mean(base[d].accuracy for d in draws),
                    "after": st.mean(tuned[d].accuracy for d in draws), "recovery": st.mean(deltas),
                    "per_draw": deltas, "after_is_decode_failure": all(tuned[d].is_decode_failure for d in draws)}
    return summary


def _csv_seeds(value):
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def main(argv=None) -> int:
    import argparse
    import gc
    from ..backends import get_backend
    ap = argparse.ArgumentParser(prog="cliff intervene")
    ap.add_argument("--model", required=True)
    ap.add_argument("--backend", default="hf_local")
    ap.add_argument("--arm", choices=("remove", "induce"), default="remove")
    ap.add_argument("--region", choices=sorted(REGIONS), default="early")
    ap.add_argument("--n-blocks", type=int, default=8)
    ap.add_argument("--block-start", type=int, default=None)
    ap.add_argument("--component", choices=("attn", "mlp", "both"), default="attn")
    ap.add_argument("--tasks", default="ag_news")
    ap.add_argument("--cal-task", default="ag_news")
    ap.add_argument("--ks", default="0,1,4,16")
    ap.add_argument("--cliff-k", type=int, default=16)
    ap.add_argument("--seeds", default="0,1,2,3,4", help="evaluation draws, not training replicas")
    ap.add_argument("--train-seeds", default="0,1,2,3,4")
    ap.add_argument("--calibration-seeds", default="42", help="one shared seed or one per replica")
    ap.add_argument("--demo-seeds", default="43", help="calibration demonstration seed(s)")
    ap.add_argument("--query-split-seed", type=int, default=0)
    ap.add_argument("--invalid-target", choices=("exclude", "error", "first-label"), default="exclude")
    ap.add_argument("--parser", choices=("strict", "legacy_prefix"), default="strict")
    ap.add_argument("--target-policy", choices=("gold", "shuffle_query"), default="gold")
    ap.add_argument("--target-seed", type=int, help="row-target permutation seed; defaults to calibration seed")
    ap.add_argument("--pairing-group-id", help="explicit registered pairing group across conditions/regions")
    ap.add_argument("--targeting-contract", help="frozen source-geometry and exact loaded LoRA count/target registration")
    ap.add_argument("--targeting-contract-sha256", help="SHA256 of the exact registered targeting contract file")
    ap.add_argument("--exclude-examples", action="append", default=[], help="JSON/JSONL reserved example ID/hash manifest")
    ap.add_argument("--development-manifest", action="append", default=[], help="W2/W3 development design or predictions; excluded from every pool")
    ap.add_argument("--pool-manifest", help="JSON per-task source calibration_query/demonstration/evaluation_demo index ranges")
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-calibration", type=int, default=500)
    ap.add_argument("--intervention-k", type=int, default=16)
    ap.add_argument("--intervention-ks", help="seeded balanced per-query calibration K mixture, e.g. 0,16")
    ap.add_argument("--calibration-k-seed", type=int, help="K assignment seed; defaults to each calibration seed")
    ap.add_argument("--calibration-presentation", choices=("contextual", "zero_shot_exposure_matched"), default="contextual")
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--untrained-adapter", action="store_true", help="save/evaluate zero-initialized adapters without optimization; n_trained=0")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--revision", help="immutable model checkpoint revision")
    ap.add_argument("--tokenizer-revision", help="immutable tokenizer revision (defaults to model revision)")
    ap.add_argument("--data-revision", help="immutable dataset revision, if shared across selected tasks")
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--max-seq-len", type=int, help="optional ceiling within checkpoint capacity")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--template", choices=("native", "raw", "llama", "hermes"), default="native")
    ap.add_argument("--allow-fallback", action="store_true", help="disabled; legacy invocations fail clearly")
    loss = ap.add_mutually_exclusive_group()
    loss.add_argument("--label-only-loss", dest="label_only_loss", action="store_true")
    loss.add_argument("--whole-prompt-loss", dest="label_only_loss", action="store_false")
    ap.set_defaults(label_only_loss=True)
    ap.add_argument("--out", help="new result file; artifacts use a unique sibling run directory")
    a = ap.parse_args(argv)
    targeting_contract = None
    if bool(a.targeting_contract) != bool(a.targeting_contract_sha256):
        ap.error("--targeting-contract and --targeting-contract-sha256 must be supplied together")
    if a.targeting_contract is not None:
        from ..revision_target_contract import read_contract
        targeting_contract = read_contract(a.targeting_contract, a.targeting_contract_sha256)
    if a.allow_fallback:
        ap.error("model-wide fallback is disabled; localization requires exact targeting")
    if a.arm != "remove" and a.target_policy != "gold":
        ap.error("query-target shuffle controls are defined only for the supervised remove arm")
    intervention_ks = _csv_seeds(a.intervention_ks) if a.intervention_ks else None
    if intervention_ks is not None and (len(set(intervention_ks)) != len(intervention_ks) or min(intervention_ks) < 0):
        ap.error("calibration K mixture must contain distinct nonnegative K values")
    exposure_budget = None
    if a.gradient_accumulation_steps < 1:
        ap.error("gradient accumulation steps must be positive")
    if a.calibration_presentation == "zero_shot_exposure_matched":
        if (a.arm != "remove" or a.target_policy != "gold" or a.intervention_k != 8
                or intervention_ks is not None or a.label_only_loss or a.untrained_adapter):
            ap.error("exposure-matched K0 requires remove/gold, --intervention-k 8, --whole-prompt-loss, without mixed K or untrained mode")
        try:
            exposure_budget = exposure_optimizer_budget(a.n_calibration, a.epochs, a.gradient_accumulation_steps)
        except ValueError as error:
            ap.error(str(error))
    excluded_examples, exclusion_sources = [], []
    for path, development in [(path, False) for path in a.exclude_examples] + [(path, True) for path in a.development_manifest]:
        entries, source = read_exclusion_manifest(path, require_development=development)
        excluded_examples.extend(entries)
        exclusion_sources.append(source)
    pool_spec = json.loads(Path(a.pool_manifest).read_text()) if a.pool_manifest else None
    tasks, ks, draws = a.tasks.split(","), _csv_seeds(a.ks), _csv_seeds(a.seeds)
    train_seeds, cal_seeds, demo_seeds = map(_csv_seeds, (a.train_seeds, a.calibration_seeds, a.demo_seeds))
    if len(ks) != len(set(ks)) or len(draws) != len(set(draws)):
        ap.error("K values and evaluation demonstration seeds must be unique")
    if len(set(train_seeds)) != len(train_seeds):
        ap.error("independent training replicas require distinct train seeds")
    for name, values in (("calibration", cal_seeds), ("demo", demo_seeds)):
        if len(values) not in {1, len(train_seeds)}:
            ap.error(f"{name} seeds must have length one or match train seeds")
    if 0 not in ks or a.cliff_k not in ks:
        ap.error("K grid must include K=0 and --cliff-k for specificity")
    if min(ks) < 0 or a.n_test < 1 or a.n_calibration < 1 or a.epochs < 1:
        ap.error("invalid K, sample count or epoch count")
    explicit_out = Path(a.out) if a.out else None
    if explicit_out and explicit_out.exists():
        raise FileExistsError(f"refusing to overwrite {explicit_out}")
    run_id = (f"{a.arm}_{a.region}_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
              + "_" + uuid4().hex[:10])
    config = {**vars(a), "train_seeds": train_seeds, "evaluation_demo_seeds": draws,
              "calibration_seeds": cal_seeds, "calibration_demo_seeds": demo_seeds,
              "objective": "label-span" if a.label_only_loss else "whole-prompt",
              "exclusion_sources": exclusion_sources, "reserved_exclusions": excluded_examples,
              "source_pool_spec": pool_spec,
              "run_id": run_id, "intervention_ks": intervention_ks,
              "exposure_optimizer_budget": exposure_budget,
              "training_replica_kind": "adapter_initialization" if a.untrained_adapter else "trained_adapter",
              "n_trained": 0 if a.untrained_adapter else len(train_seeds),
              "independent_unit": "adapter_initialization" if a.untrained_adapter else "trained_adapter"}
    run_writer = RunWriter(explicit_out.parent if explicit_out else outputs_dir(),
                           "intervention", config, run_id=run_id)
    run_dir = run_writer.path
    out = explicit_out if explicit_out else run_dir / "result.json"
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    replicas = []
    replica_writer = None
    try:
        for index, train_seed in enumerate(train_seeds):
            cal_seed = cal_seeds[index if len(cal_seeds) > 1 else 0]
            demo_seed = demo_seeds[index if len(demo_seeds) > 1 else 0]
            replica_id = f"replica_{index:02d}_train{train_seed}"
            context = {"run_id": replica_id, "parent_run_id": run_id,
                       "training_run_id": f"{run_id}/{replica_id}", "train_replica": index,
                       "train_seed": train_seed, "calibration_seed": cal_seed,
                       "calibration_demo_seed": demo_seed, "model_id": a.model,
                       "training_replica_kind": config["training_replica_kind"],
                       "n_trained": 0 if a.untrained_adapter else 1,
                       "region": a.region, "component": a.component}
            if a.pairing_group_id:
                context["paired_block_id"] = (f"{a.pairing_group_id}/train{train_seed}/cal{cal_seed}"
                                              f"/demo{demo_seed}/query{a.query_split_seed}")
            replica_writer = RunWriter(run_dir, "replicas", {**config, **context}, run_id=replica_id)
            replica_dir = replica_writer.path
            seed_everything(train_seed)
            backend = require_weight_access(get_backend(a.backend, a.model, device=a.device,
                        load_in_4bit=a.load_in_4bit or None, template=a.template,
                        revision=a.revision, tokenizer_revision=a.tokenizer_revision,
                        dtype=a.dtype, max_seq_len=a.max_seq_len, local_files_only=a.local_files_only,
                        trust_remote_code=a.trust_remote_code), "the intervene command")
            n_layers = backend.n_layers()
            window = region_window(n_layers, a.n_blocks, a.region, a.block_start)

            def write_record(record):
                replica_writer.write_prediction(record)
                run_writer.write_prediction({**record, "run_id": run_id})

            before = evaluate(backend, tasks, ks, draws, a.n_test,
                query_split_seed=a.query_split_seed, data_revision=a.data_revision,
                record_sink=write_record, record_context={**context, "phase": "before"}, parser=a.parser,
                excluded_examples=excluded_examples, pool_spec=pool_spec)
            rows, calibration = build_calibration(backend, a.cal_task, a.arm, a.n_calibration,
                a.intervention_k, seed=cal_seed, demo_seed=demo_seed,
                query_split_seed=a.query_split_seed, invalid_target=a.invalid_target, data_revision=a.data_revision,
                n_test=a.n_test, parser=a.parser, target_policy=a.target_policy, target_seed=a.target_seed,
                excluded_examples=excluded_examples, pool_spec=pool_spec,
                intervention_ks=intervention_ks, k_seed=a.calibration_k_seed)
            source_rows = None
            effective_accumulation = a.gradient_accumulation_steps
            if exposure_budget is not None:
                source_rows = rows
                rows, presentation = flatten_gold_presentations(rows, a.cal_task)
                target_map = [{"presentation_id": row["presentation_id"], "query_id": row["query_id"],
                               "original_target": row["target"], "target": row["target"]} for row in rows]
                calibration.update({"calibration_presentation": a.calibration_presentation,
                                    "presentation_audit": presentation, "exposure_optimizer_budget": exposure_budget,
                                    "source_realized_k_distribution": calibration["realized_k_distribution"],
                                    "source_requested_k_distribution": calibration["requested_k_distribution"],
                                    "source_intervention_ks": calibration["intervention_ks"], "intervention_ks": [0],
                                    "source_target_distribution": calibration["target_distribution"],
                                    "source_target_randomization": calibration["target_randomization"],
                                    "target_distribution": dict(Counter(row["target"] for row in rows)),
                                    "target_randomization": {"policy": "gold", "target_map": target_map,
                                                             "target_map_sha256": content_hash(target_map)},
                                    "requested_k_distribution": {"0": len(rows)},
                                    "realized_k_distribution": {"0": len(rows)}, "n": len(rows),
                                    "n_training_rows": len(rows)})
                effective_accumulation = exposure_budget["training_gradient_accumulation_steps"]
            else:
                calibration["calibration_presentation"] = "contextual"
            (replica_dir / "calibration.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            targeting = consolidate(backend, rows, window, a.component, a.lora_r, a.lora_alpha,
                a.lr, a.epochs, False, a.label_only_loss, train_seed=train_seed, output_dir=replica_dir,
                untrained_adapter=a.untrained_adapter, gradient_accumulation_steps=effective_accumulation,
                expected_optimizer_updates=exposure_budget["optimizer_updates"] if exposure_budget else None,
                exposure_reference_rows=source_rows, targeting_contract=targeting_contract)
            after = evaluate(backend, tasks, ks, draws, a.n_test,
                query_split_seed=a.query_split_seed, data_revision=a.data_revision,
                record_sink=write_record, record_context={**context, "phase": "after"}, parser=a.parser,
                excluded_examples=excluded_examples, pool_spec=pool_spec)
            if a.untrained_adapter and [cell.to_dict() for cell in before] != [cell.to_dict() for cell in after]:
                raise RuntimeError("untrained adapter changed measured full-curve evaluation cells")
            summary = _paired_summary(before, after, tasks, ks)
            for entries in summary.values():
                for entry in entries.values():
                    entry["specificity_vs_zero"] = entry["recovery"] - entries["0"]["recovery"]
            result = {**context, "window": [window.start, window.stop - 1], "n_layers": n_layers,
                      "targeting": targeting, "calibration": calibration,
                      "before_cells": [c.to_dict() for c in before], "after_cells": [c.to_dict() for c in after],
                      "summary": summary}
            with (replica_dir / "result.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            replica_writer.complete(result, {"passed": True, "independent_unit": config["independent_unit"],
                                              "n_trained": context["n_trained"],
                                              "train_seed": train_seed})
            replica_writer = None
            replicas.append(result)
            del backend
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        hashes = [replica["targeting"]["adapter_sha256"] for replica in replicas]
        if len(hashes) != len(set(hashes)):
            raise RuntimeError("duplicate adapter hashes: independent replicas were not established")
        frozen_hashes = {replica["targeting"]["frozen_sha256"] for replica in replicas}
        if len(frozen_hashes) != 1:
            raise RuntimeError("replicas did not use the same immutable base checkpoint parameters")
        result = {"experiment": "region_restricted_intervention", "arm": a.arm, "model_id": a.model,
                  "region": a.region, "component": a.component, "objective": config["objective"],
                  "run_id": run_id, "artifact_directory": str(run_dir),
                  "n_training_replicas": 0 if a.untrained_adapter else len(replicas),
                  "n_trained": 0 if a.untrained_adapter else len(replicas),
                  "n_adapter_initializations": len(replicas), "training_replica_kind": config["training_replica_kind"],
                  "n_evaluation_draws": len(draws),
                  "cliff_k": a.cliff_k, "replicas": replicas, "inference_status": "query-record analysis required"}
        with out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        run_writer.complete(result, {"passed": True, "distinct_adapter_hashes": True,
                                     "identical_base_checkpoint_hashes": True,
                                     "n_training_replicas": result["n_training_replicas"],
                                     "n_adapter_initializations": len(replicas), "n_trained": result["n_trained"]})
    except BaseException as error:
        if replica_writer is not None:
            replica_writer.fail(error)
        run_writer.fail(error)
        raise
    print(f"wrote {out}; {len(replicas)} distinct saved adapters in {run_dir}")
    return 0
