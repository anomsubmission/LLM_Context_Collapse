"""Pinned, credential-safe preflight and immutable W2 baseline execution.

Preflight resolves Hub metadata; payload downloads require --download. Run-pilot
is always technical evidence and never automatically launches confirmation.
Every execution requires immutable model, tokenizer and dataset commit hashes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from .data import TASKS, exclude_examples, load_task
from .revision_records import RunWriter, content_hash, environment_record, file_hash, scrub_secrets


ROOT = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-fA-F]{40}$")
HF_SECRET = re.compile(r"hf_[A-Za-z0-9]{8,}")
METADATA_PATTERNS = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt",
                     "merges.txt", "*.model", "chat_template*.jinja"]


def read_huggingface_token(path):
    """Read only explicitly Hugging Face-labeled secrets; never return others.

    Supports ``huggingface:`` followed by blank lines and the next nonblank
    token, same-line assignments, or structured Hugging Face keys. Multiple
    distinct labeled tokens are rejected rather than choosing an account.
    """
    path = Path(path)
    if not path.exists():
        return None
    text = path.read_text()
    candidates = set()
    normalized_labels = {"huggingface", "huggingfacetoken", "hftoken", "huggingfacehubapitoken"}
    try:
        structured = json.loads(text)
    except json.JSONDecodeError:
        structured = None
    if isinstance(structured, dict):
        for key, value in structured.items():
            if re.sub(r"[^a-z]", "", str(key).lower()) in normalized_labels and isinstance(value, str):
                candidates.update(HF_SECRET.findall(value))
    lines = text.splitlines()
    heading = re.compile(r"^\s*(?:hugging[ _-]*face(?:[ _-]*(?:hub[ _-]*)?(?:api[ _-]*)?token)?|hf[ _-]*token)\s*[:=]", re.I)
    for index, line in enumerate(lines):
        if not heading.search(line):
            continue
        found = HF_SECRET.findall(line)
        if not found:
            following = next((s.strip() for s in lines[index+1:] if s.strip()), "")
            if HF_SECRET.fullmatch(following):
                found = [following]
        candidates.update(found)
    if len(candidates) > 1:
        raise ValueError("multiple Hugging Face-labeled tokens found; select one credential file")
    return next(iter(candidates), None)


def configure_cache(path):
    cache = Path(path).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    # Set cache locations before lazy importing Hub/datasets. No authentication
    # value enters an environment variable or an experiment manifest.
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    return cache


def hardware_record():
    result = dict(system=platform.system(), machine=platform.machine(), processor=platform.processor())
    if platform.system() == "Darwin":
        try:
            result["memory_bytes"] = int(subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            result["memory_bytes"] = None
    try:
        import torch
        result.update(torch_version=torch.__version__, cuda_available=torch.cuda.is_available(),
                      mps_built=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built()),
                      mps_available=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
    except ImportError:
        result.update(torch_version=None, cuda_available=None, mps_built=None, mps_available=None)
    return result


def source_indices(dataset, role):
    """Return the registered source rows, accepting one explicit representation.

    Contiguous ranges are retained for historical pilots. New designs can use
    frozen randomly sampled indices without outcome-dependent row selection.
    """
    indices_key, range_key = f"{role}_indices", f"{role}_index_range"
    has_indices, has_range = indices_key in dataset, range_key in dataset
    if has_indices == has_range:
        raise ValueError(f"register exactly one of {indices_key} or {range_key}")
    if has_indices:
        indices = dataset[indices_key]
        if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError(f"{indices_key} requires distinct nonnegative integer source rows")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{indices_key} contains repeated source rows")
        return list(indices)
    interval = dataset[range_key]
    if not isinstance(interval, list) or len(interval) != 2 or any(type(i) is not int for i in interval) or not 0 <= interval[0] < interval[1]:
        raise ValueError(f"{range_key} requires an explicit nonempty [start, stop] source-row interval")
    return list(range(*interval))


def validate_config(config, *, require_revisions=True, scientific=False):
    if config.get("task") not in TASKS:
        raise ValueError("task must be a supported dataset identifier")
    model, dataset = config.get("model", {}), config.get("dataset", {})
    if not model.get("id"):
        raise ValueError("model.id is required")
    if dataset.get("id", TASKS[config["task"]]["path"]) != TASKS[config["task"]]["path"]:
        raise ValueError("dataset.id must match the task loader; aliases are not silently merged")
    if model.get("tokenizer_id", model["id"]) != model["id"]:
        raise ValueError("this local backend requires the tokenizer from the model repository")
    if require_revisions:
        for name, value in (("model", model.get("revision")), ("tokenizer", model.get("tokenizer_revision")),
                            ("dataset", dataset.get("revision"))):
            if not SHA.fullmatch(value or ""):
                raise ValueError(f"{name} revision must be an immutable 40-character Hub commit; run preflight first")
    ks, seeds = config.get("k_values", []), config.get("demo_seeds", [])
    if not ks or 0 not in ks or len(ks) != len(set(ks)) or any(type(k) is not int or k < 0 for k in ks):
        raise ValueError("k_values must be distinct nonnegative integers including K=0")
    if not seeds or len(seeds) != len(set(seeds)) or any(type(seed) is not int for seed in seeds):
        raise ValueError("demo_seeds must be distinct integers")
    if not dataset.get("train_split") or not dataset.get("query_split"):
        raise ValueError("explicit train_split and query_split are required")
    if type(dataset.get("n_queries")) is not int or dataset["n_queries"] < 1:
        raise ValueError("a positive n_queries is required")
    train_rows, query_rows = source_indices(dataset, "train"), source_indices(dataset, "query")
    if dataset["n_queries"] > len(query_rows):
        raise ValueError("registered query rows cannot supply requested n_queries")
    if dataset["train_split"] == dataset["query_split"]:
        if set(train_rows) & set(query_rows):
            raise ValueError("demonstration and query source rows overlap")
    role = config.get("query_role")
    if role not in {"development", "final_test", "technical_pilot"}:
        raise ValueError("explicit development/final_test/technical_pilot query_role is required")
    if role == "development" and dataset["query_split"] == TASKS[config["task"]]["split_test"]:
        raise ValueError("development queries must use a training/development split, leaving official final queries untouched")
    if role == "final_test":
        if dataset["query_split"] != TASKS[config["task"]]["split_test"]:
            raise ValueError("final queries must use the official task test/validation split")
        selection = config.get("development_selection", {})
        if require_revisions and not (selection.get("frozen") is True and selection.get("source_run_id")
                and selection.get("query_ids") and selection.get("query_hashes")
                and type(selection.get("k_cliff")) is int and type(selection.get("k_reference")) is int):
            raise ValueError("confirmation requires a frozen development run, identities/hashes and prespecified K contrast")
        if require_revisions and not (selection["k_cliff"] in ks and selection["k_reference"] in ks
                                      and selection["k_reference"] < selection["k_cliff"]):
            raise ValueError("the frozen reference/cliff contrast must be ordered and present in the registered K grid")
    if scientific and role == "technical_pilot":
        raise ValueError("technical pilot data cannot be marked scientific evidence")
    if model.get("overflow_policy", "reject") != "reject":
        raise ValueError("W2 rejects overflow; silent/adaptive truncation is not this registered condition")
    if config.get("max_new_tokens", 10) < 1:
        raise ValueError("max_new_tokens must be positive")


def _safe_failure(operation, exception):
    response = getattr(exception, "response", None)
    status = getattr(response, "status_code", None)
    return RuntimeError(f"{operation} failed ({type(exception).__name__}" +
                        (f", HTTP {status}" if isinstance(status, int) else "") + "); credential and response content withheld")


def _snapshot_files(info):
    rows = []
    for file in getattr(info, "siblings", []) or []:
        lfs = getattr(file, "lfs", None)
        rows.append(dict(name=getattr(file, "rfilename", ""), size=getattr(file, "size", None),
                         blob_id=getattr(file, "blob_id", None),
                         lfs_sha256=lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)))
    return rows


def preflight(config, *, token_file=None, download=None, api=None, snapshot=None, data_loader=None):
    """Resolve mutable labels to commits; download only explicitly selected payloads."""
    validate_config(config, require_revisions=False)
    if download not in {None, "metadata", "data", "weights", "all"}:
        raise ValueError("download must be metadata, data, weights or all")
    pinned = copy.deepcopy(config)
    cache = configure_cache(config.get("cache_dir", ROOT / ".cache" / "revision"))
    token = read_huggingface_token(token_file or ROOT.parents[1] / "token.txt")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    model, dataset = pinned["model"], pinned["dataset"]
    try:
        model_info = api.model_info(model["id"], revision=model.get("revision") or "main", token=token or False, files_metadata=True)
        model["revision"] = model_info.sha
        tokenizer_revision = model.get("tokenizer_revision") or model_info.sha
        tokenizer_info = model_info if tokenizer_revision == model_info.sha else api.model_info(
            model["id"], revision=tokenizer_revision, token=token or False, files_metadata=True)
        model["tokenizer_revision"] = tokenizer_info.sha
        dataset_id = dataset.get("id", TASKS[config["task"]]["path"])
        dataset_info = api.dataset_info(dataset_id, revision=dataset.get("revision") or "main", token=token or False, files_metadata=True)
        dataset.update(id=dataset_id, revision=dataset_info.sha)
    except Exception as exception:
        raise _safe_failure("Hub revision resolution", exception) from None
    for value in (model["revision"], model["tokenizer_revision"], dataset["revision"]):
        if not SHA.fullmatch(value or ""):
            raise ValueError("Hub returned a nonimmutable revision; refusing to continue")
    downloads = {}
    if download:
        if snapshot is None:
            from huggingface_hub import snapshot_download
            snapshot = snapshot_download
        patterns = list(METADATA_PATTERNS)
        if download in {"weights", "all"}:
            patterns += ["*.safetensors", "*.safetensors.index.json"]
        try:
            downloads["model_snapshot"] = snapshot(model["id"], revision=model["revision"], token=token or False,
                cache_dir=cache / "hub", allow_patterns=patterns, max_workers=4)
            if model["tokenizer_revision"] != model["revision"]:
                downloads["tokenizer_snapshot"] = snapshot(model["id"], revision=model["tokenizer_revision"],
                    token=token or False, cache_dir=cache / "hub", allow_patterns=METADATA_PATTERNS, max_workers=4)
            if download in {"data", "all"}:
                data = _load_data(pinned, loader=data_loader or load_task)
                downloads["dataset"] = dict(metadata=data.metadata, actual_demonstration_pool=len(data.train),
                                             actual_query_count=len(data.test), query_ids=[e.example_id for e in data.test],
                                             query_hashes=[e.text_hash for e in data.test])
        except Exception as exception:
            raise _safe_failure("Explicit payload download", exception) from None
    pinned["configuration_status"] = "immutable_revisions_resolved; pilot/development/final roles remain distinct"
    return dict(schema_version=1, status="revisions_resolved", payload_download_requested=download,
                scientific_evidence=False, pinned_config=pinned,
                credential_status="labeled_huggingface_token_read" if token else "no_labeled_token_public_access_only",
                model_files=_snapshot_files(model_info), dataset_files=_snapshot_files(dataset_info),
                downloads=downloads, hardware=hardware_record(), environment=environment_record(),
                input_config_sha256=content_hash(scrub_secrets(config)),
                next_step="run-pilot uses this pinned_config; confirmation is a separate explicit command")


def _load_data(config, *, loader=load_task):
    dataset = config["dataset"]
    return loader(config["task"], n_train=dataset.get("n_train"), n_test=dataset["n_queries"],
                  shard_seed=dataset.get("query_split_seed", 0), revision=dataset["revision"],
                  train_split=dataset["train_split"], test_split=dataset["query_split"],
                  train_indices=source_indices(dataset, "train"),
                  test_indices=source_indices(dataset, "query"))


def _continuation_inputs(directory, config, technical_pilot):
    """Read an unfinished source without changing or silently repairing it."""
    directory = Path(directory).resolve()
    manifest_path, predictions_path = directory / "manifest.json", directory / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text())
    expected_experiment = "w2_baseline_pilot" if technical_pilot else "w2_baseline"
    if manifest.get("experiment") != expected_experiment or manifest.get("status") not in {"running", "failed"}:
        raise ValueError("continuation requires an unfinished run of the same baseline role")
    normalize = lambda value: {key: item for key, item in value.items() if key != "run_id"}
    if normalize(manifest["configuration"]) != normalize(scrub_secrets(config)):
        raise ValueError("continuation configuration differs from the frozen source")
    if content_hash(manifest["configuration"]) != manifest.get("config_sha256"):
        raise ValueError("continuation source configuration hash is invalid")
    # Changes to unrelated experiment modules are harmless. Require exact source
    # identity for tokenization, prompting, sampling and the baseline analysis.
    snapshot = directory / "source_snapshot"
    for relative in ("cliff/backends/hf_local.py", "cliff/data.py", "cliff/prompting.py",
                     "cliff/experiments/behavioural.py", "cliff/revision_metrics.py"):
        if file_hash(snapshot / relative) != file_hash(ROOT / relative):
            raise ValueError(f"continuation executable source changed: {relative}")
    payload = predictions_path.read_bytes()
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("continuation source has an empty or incomplete JSONL record")
    index = {}
    for number, line in enumerate(payload.splitlines()):
        row = json.loads(line)
        if row.get("run_id") != manifest["run_id"]:
            raise ValueError("continuation prediction belongs to another source run")
        key = (row.get("requested_k"), row.get("draw_id"), row.get("query_id"))
        if key in index:
            raise ValueError("continuation source contains duplicate query identities")
        index[key] = (number, row)
    source_hash = hashlib.sha256(payload).hexdigest()
    if manifest.get("predictions_sha256", source_hash) != source_hash:
        raise ValueError("continuation prediction-file hash differs from source manifest")
    if file_hash(predictions_path) != source_hash:
        raise ValueError("source predictions are still changing; do not duplicate an active run")
    return index, dict(source_directory=str(directory), source_run_id=manifest["run_id"],
        source_manifest_sha256=file_hash(manifest_path), source_predictions_sha256=source_hash,
        source_status_at_read=manifest["status"], n_source_predictions=len(index),
        source_preserved=True, runtime_change="checked continuation support; source input tokens revalidated")


def _validate_continued_prediction(row, config, backend, query, demos, prompt):
    from .prompting import parse_label_strict
    model, dataset = config["model"], config["dataset"]
    expected = dict(task=config["task"], query_role=config["query_role"],
        query_id=query.example_id, query_hash=query.text_hash, gold_label=query.label,
        model_id=model["id"], model_revision=model["revision"],
        tokenizer_revision=model["tokenizer_revision"], dataset_revision=dataset["revision"],
        condition="baseline", realized_k=len(demos), retained_k=len(demos),
        demonstration_ids=[e.example_id for e in demos],
        demonstration_hashes=[e.text_hash for e in demos], demonstration_labels=[e.label for e in demos],
        parser="strict", sampling_protocol="exact_k_v1", max_new_tokens=config.get("max_new_tokens", 10),
        template=model.get("template", "native"), label_prefix=model.get("label_prefix", " "),
        label_score_implementation=model.get("label_score_implementation", "full_forward"))
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError("continued prediction differs from frozen query/model/demonstration/readout design")
    ids = backend.encode_prompt(prompt)
    if (row.get("input_token_ids") != ids or row.get("input_token_ids_hash") != content_hash(ids)
            or row.get("input_tokens_before") != len(ids) or row.get("input_tokens_after") != len(ids)
            or row.get("template_hash") != backend._template_hash()
            or row.get("label_scoring_policy") != backend.label_scoring_policy):
        raise ValueError("continued prediction has mismatched rendered input tokens or scoring policy")
    if row.get("rejected") or row.get("truncated") or row.get("measurement_status") != "measured":
        raise ValueError("rejected/truncated/unmeasured records cannot be reused as validated measurements")
    predicted = parse_label_strict(row["raw_text"], TASKS[config["task"]]["label_names"])
    if (row.get("prediction") != predicted or row.get("correct") != (predicted == query.label)
            or row.get("parse_status") != ("parseable" if predicted is not None else "unparseable")):
        raise ValueError("continued prediction's strict parse or correctness is inconsistent")
    scores = row.get("label_scores")
    if config.get("label_scores", True) != (scores is not None):
        raise ValueError("continued record has a different auxiliary-score protocol")


def run_baseline(config, *, technical_pilot=True, backend=None, data=None, token_file=None, resume_from=None):
    """Execute one pinned baseline; never starts a subsequent final experiment."""
    from .experiments.behavioural import summarize
    from .data import exact_balanced_sample
    from .prompting import Cell, parse_label_strict, render_prompt
    from .revision_metrics import summarize_predictions

    validate_config(config, scientific=not technical_pilot)
    configure_cache(config.get("cache_dir", ROOT / ".cache" / "revision"))
    writer = RunWriter(config.get("output_root", ROOT / "outputs" / "revision"),
                       "w2_baseline_pilot" if technical_pilot else "w2_baseline", config,
                       run_id=config.get("run_id"), scientific_evidence=False)
    started = time.perf_counter()
    try:
        mc = config["model"]
        if backend is None:
            from .backends.hf_local import HFLocalBackend
            backend = HFLocalBackend(mc["id"], revision=mc["revision"], tokenizer_revision=mc["tokenizer_revision"],
                device=mc.get("device", "auto"), dtype=mc.get("dtype", "bfloat16"),
                template=mc.get("template", "native"), overflow_policy="reject", max_seq_len=mc.get("max_seq_len"),
                template_marker_policy=mc.get("template_marker_policy", "native_tokens"),
                attn_implementation=mc.get("attn_implementation"),
                label_score_implementation=mc.get("label_score_implementation", "full_forward"),
                label_prefix=mc.get("label_prefix", " "),
                load_in_4bit=mc.get("load_in_4bit", False), local_files_only=mc.get("local_files_only", True),
                hf_token=read_huggingface_token(token_file or ROOT.parents[1] / "token.txt"))
        if getattr(backend, "model_id", None) != mc["id"] or getattr(backend, "revision", None) != mc["revision"]:
            raise ValueError("loaded backend checkpoint identity differs from pinned configuration")
        actual_commit = getattr(getattr(getattr(backend, "_model", None), "config", None), "_commit_hash", None)
        actual_tokenizer_commit = getattr(backend, "resolved_tokenizer_revision", None)
        if not technical_pilot and actual_commit != mc["revision"]:
            raise ValueError("scientific baseline requires a verified loaded model commit before evaluation")
        if not technical_pilot and actual_tokenizer_commit != mc["tokenizer_revision"]:
            raise ValueError("scientific baseline requires a verified loaded tokenizer commit before evaluation")
        data = data if data is not None else _load_data(config)
        dataset = config["dataset"]
        if data.task_id != config["task"] or len(data.test) != dataset["n_queries"]:
            raise ValueError("loaded task/query count differs from the registered configuration")
        if data.metadata.get("requested_revision") != dataset["revision"]:
            raise ValueError("loaded dataset revision differs from the pinned configuration")
        if data.metadata.get("train_split") != dataset["train_split"] or data.metadata.get("test_split") != dataset["query_split"]:
            raise ValueError("loaded source splits differ from registered roles")
        if len({e.example_id for e in data.test}) != len(data.test) or len({e.text_hash for e in data.test}) != len(data.test):
            raise ValueError("query source IDs and normalized texts must be unique")
        selection = config.get("development_selection", {})
        if config["query_role"] == "final_test" and (
                {e.example_id for e in data.test} & set(selection["query_ids"]) or
                {e.text_hash for e in data.test} & set(selection["query_hashes"])):
            raise ValueError("development and final query identities/texts overlap")
        pool = exclude_examples(data.train, data.test)
        continued, continuation = _continuation_inputs(resume_from, config, technical_pilot) if resume_from else ({}, None)
        expected_keys = {(k, draw, query.example_id) for k in config["k_values"]
                         for draw in (config["demo_seeds"][:1] if k == 0 else config["demo_seeds"])
                         for query in data.test}
        if set(continued) - expected_keys:
            raise ValueError("continuation includes predictions outside the frozen evaluation grid")
        if continuation:
            writer.manifest["continuation"] = continuation
        design = dict(query_role=config["query_role"], query_examples=[dict(id=e.example_id, hash=e.text_hash, gold=e.label) for e in data.test],
                      dataset_metadata=data.metadata, demonstration_pool_ids=[e.example_id for e in pool],
                      k_values=config["k_values"], demo_seeds=config["demo_seeds"],
                      query_label_counts=dict(Counter(e.label for e in data.test)),
                      demonstration_pool_label_counts=dict(Counter(e.label for e in pool)),
                      technical_pilot=technical_pilot, selection_uses="fixed_source_indices_and_seeds_only")
        _exclusive_json(writer.path / "design.json", design)
        predictions, cells, n_reused = [], [], 0
        for k in config["k_values"]:
            for draw in config["demo_seeds"][:1] if k == 0 else config["demo_seeds"]:
                demos = exact_balanced_sample(pool, k, draw)
                correct = invalid = rejected = truncated = 0
                for query in data.test:
                    prompt = render_prompt(data.task_id, demos, query.text)
                    query_started = time.perf_counter()
                    prior = continued.get((k, draw, query.example_id))
                    if prior is not None:
                        source_index, original = prior
                        _validate_continued_prediction(original, config, backend, query, demos, prompt)
                        generation = dict(original)
                        generation.update(reused_prediction=True,
                            continued_from_run_id=continuation["source_run_id"],
                            continued_from_prediction_index=source_index,
                            original_query_elapsed_seconds=original.get("total_query_elapsed_seconds"))
                        n_reused += 1
                    else:
                        generation = backend.generate_record(prompt, config.get("max_new_tokens", 10))
                    prediction = None if generation.get("rejected") else parse_label_strict(generation["raw_text"], data.label_names)
                    scores = None
                    scoring_started = time.perf_counter()
                    if config.get("label_scores", True) and not generation.get("rejected"):
                        scores = generation["label_scores"] if prior is not None else backend.score_labels(prompt, data.label_names)
                    row = dict(generation, run_id=writer.manifest["run_id"], task=data.task_id,
                        query_role=config["query_role"], query_id=query.example_id, query_hash=query.text_hash,
                        gold_label=query.label, gold_label_index=query.label_index, model_id=mc["id"],
                        model_revision=mc["revision"], tokenizer_revision=mc["tokenizer_revision"],
                        dataset_revision=dataset["revision"], requested_k=k, realized_k=len(demos),
                        retained_k=None if generation.get("truncated") else len(demos),
                        draw_id=draw, demo_seed=draw, training_run_id=None, condition="baseline",
                        demonstration_ids=[e.example_id for e in demos], demonstration_hashes=[e.text_hash for e in demos],
                        demonstration_labels=[e.label for e in demos], prediction=prediction,
                        parse_status="rejected" if generation.get("rejected") else "parseable" if prediction is not None else "unparseable",
                        correct=None if generation.get("rejected") else prediction == query.label, label_scores=scores,
                        scoring_elapsed_seconds=time.perf_counter()-scoring_started,
                        total_query_elapsed_seconds=time.perf_counter()-query_started,
                        sampling_protocol="exact_k_v1", parser="strict")
                    writer.write_prediction(row)
                    predictions.append(row)
                    correct += prediction == query.label
                    invalid += prediction is None and not generation.get("rejected")
                    rejected += bool(generation.get("rejected"))
                    truncated += bool(generation.get("truncated"))
                n = len(data.test)
                cells.append(Cell(data.task_id, k, draw, None if rejected else correct/n, n, invalid, truncated, rejected, len(demos)))
                print(f"K={k} draw={draw} queries={n} invalid={invalid} rejected={rejected} elapsed={time.perf_counter()-started:.1f}s", flush=True)
        measurements = all(p.get("measurement_status") == "measured" and not p.get("rejected") and not p.get("truncated") for p in predictions)
        identity = getattr(backend, "model_id", None) == mc["id"] and getattr(backend, "revision", None) == mc["revision"]
        gates = dict(measured_nontruncated_predictions=measurements, checkpoint_identity_matches=identity,
                     loaded_model_commit_matches=actual_commit == mc["revision"], immutable_revisions=True,
                     loaded_tokenizer_commit_matches=actual_tokenizer_commit == mc["tokenizer_revision"],
                     unique_query_ids=True, query_demo_overlap=False,
                     technical_pilot=technical_pilot, dataset_metadata_matches=True)
        scientific = (not technical_pilot and measurements and identity and actual_commit == mc["revision"]
                      and actual_tokenizer_commit == mc["tokenizer_revision"])
        writer.manifest.update(scientific_evidence=scientific, hardware=hardware_record(),
                               data_metadata=data.metadata, scientific_gates=gates,
                               elapsed_seconds=time.perf_counter()-started,
                               backend_provenance=dict(model_revision=mc["revision"], loaded_model_revision=actual_commit,
                                   tokenizer_revision=mc["tokenizer_revision"], template=mc.get("template", "native"),
                                   loaded_tokenizer_revision=actual_tokenizer_commit,
                                   tokenizer_provenance=getattr(backend, "tokenizer_provenance", None),
                                   label_scoring_policy=getattr(backend, "label_scoring_policy", None),
                                   label_prefix=getattr(backend, "label_prefix", None),
                                   label_score_implementation=getattr(backend, "label_score_implementation", None),
                                   native_template_text=getattr(getattr(backend, "_tok", None), "chat_template", None),
                                   template_hash=backend._template_hash() if hasattr(backend, "_template_hash") else None,
                                   actual_model_dtype=str(getattr(getattr(backend, "_model", None), "dtype", "unavailable")),
                                   context_limit=getattr(backend, "max_seq_len", None), device=getattr(backend, "device", None)))
        if continuation:
            continuation.update(n_reused_predictions=n_reused, n_new_predictions=len(predictions)-n_reused)
            if n_reused != len(continued):
                raise ValueError("continuation did not consume every registered source prediction")
        metrics = summarize(backend, [data.task_id], config["k_values"], config["demo_seeds"], dataset["n_queries"], cells,
                            regime_protocol="prospective_decline_v1")
        metrics.update(query_metrics=summarize_predictions(predictions, data.label_names), scientific_evidence=scientific,
                       technical_pilot=technical_pilot, query_role=config["query_role"],
                       elapsed_seconds=time.perf_counter()-started,
                       interpretation="technical throughput/correctness pilot only" if technical_pilot else "descriptive development curve" if config["query_role"] == "development" else "held-out prespecified final curve")
        if continuation:
            metrics["continuation"] = continuation
        if config["query_role"] == "final_test":
            # Never reselect a favourable peak or crossing on final queries.
            curve = metrics["curves"].get(data.task_id, {})
            curve.pop("development_candidate", None)
            curve.pop("regime", None)
            curve["inference_status"] = "prespecified_development_contrast_only"
            reference, cliff = selection["k_reference"], selection["k_cliff"]
            index = {(r["query_id"], r["requested_k"], r["draw_id"]): r for r in predictions}
            signed = []
            for query in data.test:
                for draw in config["demo_seeds"]:
                    ref_draw = config["demo_seeds"][0] if reference == 0 else draw
                    ref, low = index[(query.example_id, reference, ref_draw)], index[(query.example_id, cliff, draw)]
                    signed.append(dict(task=data.task_id, query_id=query.example_id, draw_id=draw,
                                       decline=float(ref["correct"])-float(low["correct"])))
            from .revision_metrics import crossed_bootstrap
            metrics["held_out_decline"] = crossed_bootstrap(signed, value_key="decline", seed=config.get("analysis_seed", 0)) if measurements else None
            metrics["development_selection"] = selection
        validation = dict(passed=bool(predictions) and measurements, scientific_gates=gates,
                          measured_valid=measurements, no_automatic_confirmation=True)
        if validation["passed"]:
            writer.complete(metrics, validation)
        else:
            # Preserve the attempted curve and every rejected record without
            # calling an over-budget pilot technically validated.
            _exclusive_json(writer.path / "metrics.json", metrics)
            _exclusive_json(writer.path / "validation.json", validation)
            writer.fail("prediction measurements failed: rejected/truncated/unmeasured inputs")
        return writer.path
    except Exception as exception:
        # Never serialize raw HTTP/library exception text, which can contain URLs
        # or authorization material. The original exception is not printed.
        writer.fail(f"{type(exception).__name__}; see safe CLI diagnostic")
        raise _safe_failure("W2 baseline", exception) from None


def _exclusive_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(scrub_secrets(document), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--config", type=Path, required=True)
    pre.add_argument("--out", type=Path, required=True)
    pre.add_argument("--token-file", type=Path)
    pre.add_argument("--download", nargs="?", const="metadata", choices=["metadata", "data", "weights", "all"])
    for name in ("run-pilot", "run-baseline"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--token-file", type=Path)
        command.add_argument("--resume-from", type=Path, help="checked continuation into a new directory; source files remain unchanged")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    config = config.get("pinned_config", config)
    if args.command == "preflight":
        if args.out.exists():
            raise FileExistsError("preflight output already exists; choose a new immutable path")
        report = preflight(config, token_file=args.token_file, download=args.download)
        _exclusive_json(args.out, report)
        print(json.dumps(dict(status=report["status"], output=str(args.out),
            model_revision=report["pinned_config"]["model"]["revision"],
            tokenizer_revision=report["pinned_config"]["model"]["tokenizer_revision"],
            dataset_revision=report["pinned_config"]["dataset"]["revision"],
            download=args.download, scientific_evidence=False), indent=2))
    else:
        path = run_baseline(config, technical_pilot=args.command == "run-pilot", token_file=args.token_file, resume_from=args.resume_from)
        print(json.dumps(dict(output=str(path), command=args.command), indent=2))


if __name__ == "__main__":
    main()
