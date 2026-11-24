"""Evaluate one saved audited adapter on task-specific feasible K grids.

Run ``python -m cliff.experiments.revision_adapter_eval --config FILE``.
The source must be a completed revised intervention replica, with explicit
manifest/audit/calibration hashes. Both the pinned base and its saved adapter
are evaluated; no optimizer, training, merging, or checkpoint download occurs.
This standalone command does not automate the training launcher's transfer
workflow. Scientific acceptance still requires independent adapter replication
and query/draw-aware analysis of its raw records.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import re
from pathlib import Path

from ..backends import get_backend, require_weight_access
from ..data import TASKS, exact_balanced_sample, exclude_examples, load_task, normalized_text_hash
from ..prompting import parse_label_strict, render_prompt
from ..revision_records import RunWriter, content_hash, file_hash
from .intervention import (_artifact_hashes, assert_targeting, encode_training_rows, filter_excluded,
                           parameter_digest, read_exclusion_manifest, resolve_targets)
from .revision_behaviour import _score, require_pinned_model, verify_loaded_model


def _read(path):
    return json.loads(Path(path).read_text())


def _checked(path, expected, description):
    # These inputs name local artifacts, never authenticated URLs. Error messages
    # intentionally omit the supplied path/content, including malformed inputs.
    if not isinstance(path, (str, Path)) or "://" in str(path):
        raise ValueError(description + " requires a local artifact path")
    path = Path(path).resolve()
    if (not path.is_file() or not re.fullmatch(r"[0-9a-fA-F]{64}", expected or "")
            or file_hash(path) != expected.lower()):
        raise ValueError(description + " hash is missing or changed")
    return path


def inspect_source(config):
    """Bind saved weights and exact targeting to a completed training replica."""
    model, source = config["model"], config["adapter"]
    require_pinned_model(model)
    manifest_path = _checked(source["source_manifest"], source.get("source_manifest_sha256"), "source replica manifest")
    manifest = _read(manifest_path)
    cfg = manifest.get("configuration", {})
    if manifest.get("status") != "complete" or manifest.get("experiment") != "replicas":
        raise ValueError("saved-adapter evaluation requires a completed revised intervention replica")
    if manifest.get("config_sha256") != content_hash(cfg):
        raise ValueError("source replica configuration hash is inconsistent")
    for filename, field in (("metrics.json", "metrics_sha256"), ("validation.json", "validation_sha256")):
        _checked(manifest_path.parent / filename, manifest.get(field), "source replica " + filename)
    if _read(manifest_path.parent / "validation.json").get("passed") is not True:
        raise ValueError("source replica validation did not pass")
    if (cfg.get("model") != model["id"] or cfg.get("revision") != model["revision"]
            or (cfg.get("tokenizer_revision") or cfg.get("revision")) != (model.get("tokenizer_revision") or model["revision"])
            or cfg.get("template", "native") != model.get("template", "native")
            or cfg.get("dtype", "bfloat16") != model.get("dtype", "bfloat16")
            or cfg.get("label_prefix", " ") != model.get("label_prefix", " ")
            or cfg.get("template_marker_policy", "native_tokens") != model.get("template_marker_policy", "native_tokens")
            or bool(cfg.get("load_in_4bit")) != bool(model.get("load_in_4bit"))):
        raise ValueError("evaluation base/tokenizer/template/precision/label-prefix differs from saved adapter source")
    audit_path = _checked(manifest_path.parent / "training_audit.json", source.get("training_audit_sha256"), "training audit")
    audit = _read(audit_path)
    metrics = _read(manifest_path.parent / "metrics.json")
    if metrics.get("targeting") != audit or audit.get("localization_valid") is not True or audit.get("frozen_weights_unchanged") is not True:
        raise ValueError("source targeting audit differs from its immutable replica metrics")
    directory = Path(source["directory"]).resolve()
    if directory != (manifest_path.parent / "adapter").resolve():
        raise ValueError("saved adapter must belong to the exact source replica")
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("saved adapter must contain local files belonging to its source replica")
    if not audit.get("adapter_artifact_sha256") or _artifact_hashes(directory) != audit["adapter_artifact_sha256"]:
        raise ValueError("saved adapter artifact hashes differ from the audited weights")
    calibration = _checked(manifest_path.parent / "calibration.json", source.get("calibration_sha256"), "source calibration identities")
    exclusions, calibration_source = read_exclusion_manifest(calibration)
    rows = _read(calibration)
    if not isinstance(rows, list) or not rows:
        raise ValueError("source calibration requires explicit query and demonstration identities")
    for row in rows:
        ids, hashes = row.get("demo_ids", []), row.get("demo_text_hashes", [])
        query_hash = row.get("query_text_hash")
        if (not row.get("query_id") or not isinstance(row.get("query_text"), str)
                or query_hash != normalized_text_hash(row["query_text"])
                or type(row.get("actual_k")) is not int or row["actual_k"] != len(ids) or len(ids) != len(hashes)
                or any(not isinstance(identity, str) or not identity for identity in ids)
                or any(not re.fullmatch(r"[0-9a-fA-F]{64}", h or "") for h in hashes)):
            raise ValueError("source calibration lacks complete validated query/demo identities and text hashes")
        exclusions.append({"id": row["query_id"], "hash": query_hash})
        exclusions.extend({"id": identity, "hash": h} for identity, h in zip(ids, hashes))
    requested = audit.get("requested", [])
    if (len(requested) != 2 or any(type(value) is not int for value in requested)
            or not 0 <= requested[0] <= requested[1] or not audit.get("registered_targets")):
        raise ValueError("source audit has no exact nonempty block/module target set")
    training_run_id = metrics.get("training_run_id") or cfg.get("training_run_id")
    if not training_run_id or "/" not in training_run_id:
        raise ValueError("source replica requires its globally unique training_run_id")
    return {"manifest": manifest, "manifest_path": manifest_path, "configuration": cfg,
            "audit": audit, "directory": directory, "calibration_source": calibration_source,
            "calibration_path": calibration,
            "calibration_exclusions": exclusions, "training_run_id": training_run_id,
            "window": range(requested[0], requested[1] + 1)}


def _indices(spec, name):
    explicit, interval = spec.get(name + "_indices"), spec.get(name + "_index_range")
    if (explicit is None) == (interval is None):
        raise ValueError("register exactly one explicit " + name + " indices/range")
    if interval is not None:
        if len(interval) != 2 or any(type(i) is not int for i in interval) or not 0 <= interval[0] < interval[1]:
            raise ValueError("invalid " + name + " source interval")
        explicit = list(range(*interval))
    if not explicit or any(type(i) is not int or i < 0 for i in explicit) or len(set(explicit)) != len(explicit):
        raise ValueError("source indices must be unique nonnegative integers: " + name)
    return explicit


def _source_ids(task, revision, split, indices):
    cfg = TASKS[task]
    return [f"{cfg['path']}:{cfg['subset'] or '-'}:{revision}:{split}:{index}" for index in indices]


def _development(spec, task, model, readout):
    entry = spec.get("development_selection", {})
    path = _checked(entry.get("path", "MISSING"), entry.get("sha256"), "development selection")
    value = _read(path)
    if (value.get("frozen") is not True or value.get("query_role") != "development"
            or not value.get("query_ids") or not value.get("query_hashes") or not value.get("source_run_id")
            or value.get("task") != task or value.get("model_id") != model["id"]
            or value.get("model_revision") != model["revision"]
            or value.get("tokenizer_revision") != (model.get("tokenizer_revision") or model["revision"])
            or value.get("dataset_id") != TASKS[task]["path"]
            or value.get("dataset_revision") != spec["source_pool"]["revision"]
            or value.get("template") != model.get("template", "native")
            or value.get("dtype") != model.get("dtype", "bfloat16")
            or value.get("label_prefix", " ") != model.get("label_prefix", " ")
            or value.get("readout") != readout or value.get("feasible_k_values") != spec["ks"]):
        raise ValueError("each final task needs its exact frozen development grid/model/template/readout: " + task)
    manifest_path = _checked(value.get("source_manifest", "MISSING"), value.get("source_manifest_sha256"), "development source manifest")
    manifest = _read(manifest_path)
    cfg = manifest.get("configuration", {})
    if (manifest.get("status") != "complete" or manifest.get("scientific_evidence") is not True
            or manifest.get("run_id") != value["source_run_id"] or cfg.get("query_role") != "development"
            or manifest.get("config_sha256") != content_hash(cfg)
            or cfg.get("task") != task or cfg.get("model", {}).get("id") != model["id"]
            or cfg.get("model", {}).get("revision") != model["revision"]
            or (cfg.get("model", {}).get("tokenizer_revision") or cfg.get("model", {}).get("revision")) != (model.get("tokenizer_revision") or model["revision"])
            or cfg.get("model", {}).get("template", "native") != model.get("template", "native")
            or cfg.get("model", {}).get("dtype", "bfloat16") != model.get("dtype", "bfloat16")
            or cfg.get("model", {}).get("label_prefix", " ") != model.get("label_prefix", " ")
            or cfg.get("dataset", {}).get("id", TASKS[task]["path"]) != TASKS[task]["path"]
            or not cfg.get("dataset", {}).get("query_split")
            or cfg["dataset"]["query_split"] == TASKS[task]["split_test"]
            or cfg.get("dataset", {}).get("revision") != spec["source_pool"]["revision"]):
        raise ValueError("technical/incomplete/nonmatching development runs cannot choose final grids")
    for filename, field in (("metrics.json", "metrics_sha256"), ("validation.json", "validation_sha256")):
        _checked(manifest_path.parent / filename, manifest.get(field), "development " + filename)
    if _read(manifest_path.parent / "validation.json").get("passed") is not True:
        raise ValueError("development source validation did not pass")
    predictions = _checked(manifest_path.parent / "predictions.jsonl", manifest.get("predictions_sha256"), "development source predictions")
    ids, hashes, grid_queries = set(), set(), {}
    for line in predictions.read_text().splitlines():
        row = json.loads(line)
        if (row.get("query_role") != "development" or row.get("run_id") != value["source_run_id"]
                or row.get("task") != task or row.get("model_id") != model["id"]
                or row.get("model_revision") != model["revision"]
                or row.get("dataset_revision") != spec["source_pool"]["revision"]
                or row.get("readout", "generated_label") != readout
                or row.get("rejected") or row.get("truncated")
                or row.get("measurement_status") != "measured"):
            raise ValueError("development predictions have different run/query roles")
        ids.add(row.get("query_id"))
        hashes.add(row.get("query_hash", row.get("query_text_hash")))
        if row.get("requested_k") != row.get("realized_k"):
            raise ValueError("development source contains a shortened K cell")
        grid_queries.setdefault(row.get("requested_k"), set()).add(row.get("query_id"))
    if None in ids or None in hashes or ids != set(value["query_ids"]) or hashes != set(value["query_hashes"]):
        raise ValueError("development exclusion identities differ from the actual source predictions")
    prefix = _source_ids(task, spec["source_pool"]["revision"], cfg["dataset"]["query_split"], [0])[0][:-1]
    if any(not identity.startswith(prefix) or not identity[len(prefix):].isdigit() for identity in ids):
        raise ValueError("development query identities differ from the registered nonfinal source split")
    if any(grid_queries.get(k) != ids for k in spec["ks"]):
        raise ValueError("frozen feasible K grid lacks a complete measured development curve")
    return ([{"id": identity} for identity in sorted(ids)] + [{"hash": h} for h in sorted(hashes)],
            {"path": str(path), "sha256": file_hash(path), "source_run_id": value["source_run_id"]})


def validate_config(config):
    if config.get("query_role") not in {"technical_pilot", "development", "final_test"}:
        raise ValueError("explicit query_role is required")
    if config.get("readout", "generated_label") not in {"generated_label", "complete_label_likelihood"}:
        raise ValueError("readout must be generated_label or complete_label_likelihood")
    source = inspect_source(config)
    exclusions, exclusion_sources = list(source["calibration_exclusions"]), [source["calibration_source"]]
    for entry in config.get("exclusion_manifests", []):
        path = _checked(entry["path"], entry.get("sha256"), "reserved exclusion manifest")
        rows, provenance = read_exclusion_manifest(path)
        exclusions.extend(rows)
        exclusion_sources.append(provenance)
    if not config.get("tasks"):
        raise ValueError("at least one task-specific grid is required")
    pools = {}
    for task, spec in config["tasks"].items():
        ks, seeds, pool = spec.get("ks", []), spec.get("demo_seeds", []), spec.get("source_pool", {})
        if (task not in TASKS or not ks or ks != sorted(set(ks)) or 0 not in ks
                or any(type(k) is not int or k < 0 for k in ks)
                or not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds)):
            raise ValueError("each known task requires a unique sorted feasible K grid including0 and explicit distinct draws")
        if (not re.fullmatch(r"[0-9a-fA-F]{40}", pool.get("revision") or "")
                or not pool.get("train_split") or not pool.get("query_split")
                or pool.get("id", TASKS[task]["path"]) != TASKS[task]["path"]
                or pool.get("subset", TASKS[task]["subset"]) != TASKS[task]["subset"]):
            raise ValueError("each task requires immutable dataset revision and explicit source splits")
        pools[task] = {"demonstration_indices": _indices(pool, "demonstration"), "query_indices": _indices(pool, "query")}
        if (pool["train_split"] == pool["query_split"]
                and set(pools[task]["demonstration_indices"]) & set(pools[task]["query_indices"])):
            raise ValueError("registered demonstration and query source indices overlap")
        if config["query_role"] == "final_test":
            if pool["query_split"] != TASKS[task]["split_test"]:
                raise ValueError("final queries must use the task's official held-out split")
            rows, provenance = _development(spec, task, config["model"], config.get("readout", "generated_label"))
            exclusions.extend(rows)
            exclusion_sources.append(provenance)
        elif config["query_role"] == "development" and pool["query_split"] == TASKS[task]["split_test"]:
            raise ValueError("development evaluation must reserve official final queries")
    return source, exclusions, exclusion_sources, pools


def load_saved_adapter(backend, source):
    """Load exact PEFT bytes, check training targeting/hashes, then freeze all."""
    import torch
    from peft import PeftModel
    from safetensors import safe_open
    base = backend.torch_model()
    if getattr(base, "peft_config", None):
        raise ValueError("saved-adapter evaluation requires a fresh unadapted base")
    targets = resolve_targets(base, source["window"], source["configuration"]["component"])
    if targets != sorted(source["audit"]["registered_targets"]):
        raise ValueError("loaded checkpoint module targets differ from the training audit")
    # A BF16 base initializes BF16 LoRA matrices. Loading an audited FP32 adapter
    # into those matrices with autocast=False rounds its values before any hash
    # check. Stage supported source tensors in FP32 BEFORE state assignment, then
    # restore each tensor's explicit recorded dtype. FP16/BF16 values are exactly
    # representable in FP32; unsupported source precisions fail closed.
    if _artifact_hashes(source["directory"]) != source["audit"]["adapter_artifact_sha256"]:
        raise ValueError("saved adapter artifacts changed before exact reload")
    saved_path = source["directory"] / "adapter_model.safetensors"
    if not saved_path.is_file():
        raise ValueError("exact saved reload requires the audited local safetensors adapter")
    supported = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
    layout = {}
    with safe_open(saved_path, framework="pt", device="cpu") as stream:
        for name in stream.keys():
            tensor = stream.get_slice(name)
            dtype = tensor.get_dtype()
            if not re.fullmatch(r".+\.lora_[AB]\.weight", name) or dtype not in supported:
                raise ValueError("unsupported saved adapter tensor name/precision; exact dtype preservation unavailable")
            layout[name] = {"shape": tensor.get_shape(), "dtype": dtype}
    if not layout:
        raise ValueError("saved adapter contains no exact A/B tensor layout")
    wrapped = PeftModel.from_pretrained(base, source["directory"], is_trainable=True,
                                        autocast_adapter_dtype=True, local_files_only=True)
    stored_config = _read(source["directory"] / "adapter_config.json")
    if set(wrapped.peft_config) != {"default"}:
        raise ValueError("saved reload requires the exact single default adapter")
    scaling_options = {}
    for field in ("r", "lora_alpha", "lora_dropout", "use_rslora", "use_dora", "bias", "fan_in_fan_out", "rank_pattern", "alpha_pattern"):
        if field in stored_config:
            actual = getattr(wrapped.peft_config["default"], field, None)
            if actual != stored_config[field]:
                raise ValueError("loaded adapter rank/scaling options differ from stored source")
            scaling_options[field] = actual
    observed = set()
    for name, parameter in wrapped.named_parameters():
        if not parameter.requires_grad:
            continue
        key = re.sub(r"(\.lora_[AB])\.default\.weight$", r"\1.weight", name)
        if key not in layout or key in observed or list(parameter.shape) != layout[key]["shape"]:
            raise ValueError("actual loaded adapter tensor layout differs from stored source")
        recorded_dtype = supported[layout[key]["dtype"]]
        if parameter.dtype != recorded_dtype:
            parameter.data = parameter.data.to(dtype=recorded_dtype)
        observed.add(key)
    if observed != set(layout):
        raise ValueError("actual loaded adapter tensor coverage differs from stored source")
    targeting = assert_targeting(wrapped, source["window"], targets)
    adapter_digest = parameter_digest(wrapped, trainable=True)
    frozen_digest = parameter_digest(wrapped, trainable=False)
    if (adapter_digest != source["audit"]["adapter_sha256"] or frozen_digest != source["audit"]["frozen_sha256"]):
        raise ValueError("loaded adapter or frozen checkpoint bytes differ from saved training hashes")
    targeting["saved_reload_audit"] = {"passed": True, "source_manifest_sha256": file_hash(source["manifest_path"]),
        "source_adapter_file_sha256": file_hash(saved_path), "source_adapter_config_sha256": file_hash(source["directory"]/"adapter_config.json"),
        "stored_tensor_layout_sha256": content_hash(layout), "n_stored_adapter_tensors": len(layout),
        "recorded_dtypes": sorted({item["dtype"] for item in layout.values()}), "stored_dtypes_preserved": True,
        "stored_scaling_options_preserved": scaling_options,
        "exact_adapter_parameter_sha256": adapter_digest, "exact_frozen_base_parameter_sha256": frozen_digest,
        "precision_policy": "FP32 staging before state assignment; restore each stored F32/F16/BF16 dtype; exact names/shapes/dtypes/bytes guard"}
    wrapped.requires_grad_(False)
    wrapped.eval()
    if any(p.requires_grad for p in wrapped.parameters()):
        raise RuntimeError("all parameters must be frozen for standalone evaluation")
    backend._model = wrapped
    return targeting


@contextmanager
def _frozen_phase(model, phase):
    """PEFT's disable context may re-enable adapter gradients on exit."""
    import torch
    try:
        with torch.inference_mode(), model.disable_adapter() if phase == "before" else nullcontext():
            yield
    finally:
        model.requires_grad_(False)


def run(config, *, backend_factory=None, data_by_task=None):
    source, exclusions, exclusion_sources, pools = validate_config(config)
    from ..revision_runtime import configure_cache
    configure_cache(config.get("cache_dir", str(Path(__file__).resolve().parents[2] / ".cache/revision")))
    writer = RunWriter(config.get("output_root", "outputs/revision"), "saved_adapter_evaluation",
                       {**config, "exclusion_sources": exclusion_sources}, run_id=config.get("run_id"), scientific_evidence=False)
    try:
        model = config["model"]
        backend = backend_factory() if backend_factory else require_weight_access(get_backend("hf_local", model["id"],
            revision=model["revision"], tokenizer_revision=model.get("tokenizer_revision") or model["revision"],
            template=model.get("template", "native"), dtype=model.get("dtype", "bfloat16"), device=model.get("device", "auto"),
            local_files_only=True, overflow_policy="reject", max_seq_len=model.get("max_seq_len"),
            load_in_4bit=model.get("load_in_4bit", False),
            template_marker_policy=model.get("template_marker_policy", "native_tokens"),
            attn_implementation=model.get("attn_implementation"),
            label_prefix=model.get("label_prefix", " "), label_score_implementation=model.get("label_score_implementation", "full_forward")),
            "saved-adapter evaluation")
        resolved = verify_loaded_model(model, backend, "final_test")
        if (backend.model_id != model["id"] or backend.revision != model["revision"]
                or backend.template != model.get("template", "native") or getattr(backend, "overflow_policy", None) != "reject"
                or getattr(backend, "label_prefix", " ") != model.get("label_prefix", " ")
                or getattr(backend, "label_score_implementation", "full_forward") != model.get("label_score_implementation", "full_forward")
                or getattr(backend, "template_marker_policy", "native_tokens") != model.get("template_marker_policy", "native_tokens")):
            raise ValueError("loaded base identity/template/budget policy differs from the registered evaluation")
        expected_encoding = source["audit"].get("encoding", {})
        if not expected_encoding.get("sequence_sha256"):
            raise ValueError("saved calibration requires its audited exact token-sequence hashes")
        _, calibration_encoding = encode_training_rows(backend, _read(source["calibration_path"]),
            source["configuration"].get("objective", "label-span") == "label-span")
        if calibration_encoding != expected_encoding:
            raise ValueError("source calibration identities/prompts/targets differ from audited training token sequences")
        targeting = load_saved_adapter(backend, source)
        evaluation_hash = parameter_digest(backend.torch_model(), trainable=False)
        datasets, designs, eligible_demo_pools = {}, {}, {}
        for task, spec in config["tasks"].items():
            pool, indices = spec["source_pool"], pools[task]
            data = data_by_task[task] if data_by_task is not None else load_task(task, n_train=None, n_test=None,
                revision=pool["revision"], train_split=pool["train_split"], test_split=pool["query_split"],
                train_indices=indices["demonstration_indices"], test_indices=indices["query_indices"])
            if (data.task_id != task or data.metadata.get("requested_revision") != pool["revision"]
                    or data.metadata.get("dataset_id") != TASKS[task]["path"]
                    or data.metadata.get("subset") != TASKS[task]["subset"]
                    or data.metadata.get("train_split") != pool["train_split"] or data.metadata.get("test_split") != pool["query_split"]
                    or [q.example_id for q in data.test] != _source_ids(task, pool["revision"], pool["query_split"], indices["query_indices"])
                    or [d.example_id for d in data.train] != _source_ids(task, pool["revision"], pool["train_split"], indices["demonstration_indices"])):
                raise ValueError("loaded dataset identity/splits/source counts differ from the exact requested pools")
            queries = filter_excluded(data.test, exclusions)
            if (not queries or len(queries) != len(data.test) or len({q.example_id for q in queries}) != len(queries)
                    or len({q.text_hash for q in queries}) != len(queries)):
                raise ValueError("registered evaluation queries overlap calibration/development/reserved examples or contain duplicates")
            demos = filter_excluded(exclude_examples(data.train, data.test), exclusions)
            eligible_demo_pools[task] = demos
            contexts = [(k, seed, exact_balanced_sample(demos, k, seed)) for k in spec["ks"]
                        for seed in (spec["demo_seeds"][:1] if k == 0 else spec["demo_seeds"])]
            datasets[task], designs[task] = (data, queries), contexts
        (writer.path / "design.json").write_text(json.dumps({task: {"source_pool": config["tasks"][task]["source_pool"],
            "loaded_dataset_metadata": datasets[task][0].metadata,
            "registered_demonstration_pool_ids": [d.example_id for d in datasets[task][0].train],
            "eligible_demonstration_pool": [{"id": d.example_id, "hash": d.text_hash} for d in eligible_demo_pools[task]],
            "n_demonstrations_excluded": len(datasets[task][0].train) - len(eligible_demo_pools[task]),
            "queries": [{"id": q.example_id, "hash": q.text_hash} for q in datasets[task][1]],
            "contexts": [{"k": k, "draw_id": seed, "demonstration_ids": [d.example_id for d in demos],
                          "demonstration_hashes": [d.text_hash for d in demos]} for k, seed, demos in contexts]}
            for task, contexts in designs.items()}, indent=2))
        readout, cells = config.get("readout", "generated_label"), []
        for phase in ("before", "after"):
            backend.torch_model().eval()
            for task, contexts in designs.items():
                data, queries = datasets[task]
                for k, seed, demos in contexts:
                    correct, invalid = 0, 0
                    for query in queries:
                        prompt = render_prompt(task, demos, query.text)
                        with _frozen_phase(backend.torch_model(), phase):
                            if readout == "generated_label":
                                measured = backend.generate_record(prompt, config.get("max_new_tokens", 10))
                                prediction = None if measured.get("rejected") else parse_label_strict(measured["raw_text"], data.label_names)
                            else:
                                scores, measured = _score(backend, prompt, data.label_names)
                                measured["label_scores"] = scores
                                prediction = max(data.label_names, key=scores.get)
                        record = {**measured, "model_id": model["id"], "model_revision": model["revision"],
                            "template": model.get("template", "native"), "dtype": model.get("dtype", "bfloat16"),
                            "label_prefix": model.get("label_prefix", " "), "label_score_implementation": model.get("label_score_implementation", "full_forward"),
                            "tokenizer_revision": model.get("tokenizer_revision") or model["revision"],
                            "dataset_id": TASKS[task]["path"], "dataset_revision": config["tasks"][task]["source_pool"]["revision"],
                            "training_run_id": source["training_run_id"], "train_seed": source["audit"].get("train_seed"),
                            "phase": phase, "condition": phase, "readout": readout, "task": task,
                            "query_role": config["query_role"], "query_id": query.example_id, "query_hash": query.text_hash,
                            "gold_label": query.label, "prediction": prediction, "parsed_label": prediction,
                            "correct": None if measured.get("rejected") else prediction == query.label,
                            "parse_status": "rejected" if measured.get("rejected") else "restricted_complete_label" if readout == "complete_label_likelihood" else "parseable" if prediction else "unparseable",
                            "requested_k": k, "realized_k": len(demos), "draw_id": seed,
                            "demonstration_ids": [d.example_id for d in demos], "demonstration_hashes": [d.text_hash for d in demos],
                            "demonstration_labels": [d.label for d in demos], "source_replica_manifest_sha256": config["adapter"]["source_manifest_sha256"],
                            "adapter_loaded": phase == "after", "n_new_training_runs": 0}
                        writer.write_prediction(record)
                        if measured.get("rejected") or measured.get("truncated") or measured.get("measurement_status") != "measured":
                            raise ValueError("evaluation rejected/truncated/unmeasured a registered cell; no reduced grid was substituted")
                        invalid += prediction is None
                        correct += prediction == query.label
                    cells.append({"task": task, "phase": phase, "k": k, "draw_id": seed, "n": len(queries),
                                  "accuracy": correct / len(queries), "n_invalid": invalid})
        if evaluation_hash != parameter_digest(backend.torch_model(), trainable=False):
            raise RuntimeError("evaluation changed frozen checkpoint/adapter bytes")
        summary = {"cells": cells, "targeting": targeting, "resolved_model": resolved, "training_run_id": source["training_run_id"],
                   "design_sha256": file_hash(writer.path / "design.json"),
                   "task_specific_k_grids": {task: spec["ks"] for task, spec in config["tasks"].items()},
                   "n_new_training_runs": 0, "adapter_parameter_sha256": source["audit"]["adapter_sha256"],
                   "evaluation_parameter_sha256": evaluation_hash, "scientific_evidence": False,
                   "inference_status": "one existing adapter; independent training/query/draw analysis required"}
        writer.complete(summary, {"passed": True, "no_training": True, "exact_saved_adapter_and_frozen_base_hashes": True,
            "same_query_and_demo_design_before_after": True, "k0_one_reference_per_task_and_phase": True,
            "all_prompts_measured_untruncated": True})
        return writer.path
    except BaseException as error:
        writer.fail(f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(f"wrote {run(_read(args.config))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
