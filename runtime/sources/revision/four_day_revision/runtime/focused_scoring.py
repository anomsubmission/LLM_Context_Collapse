#!/usr/bin/env python3
"""Prospective balanced-random calibration evaluator, outside frozen core.

STAGING STATUS: CPU fixtures validate orchestration and measurements. This is
not a frozen study, a completed device pilot, or permission to launch. Scientific
execution additionally needs a source-bound prospective registration, exact role
materialization, locally verified payloads, the shared accelerator descriptor,
and a validated full-length device pilot. No download or training is supported.

The existing saved-adapter final-grid API requires a completed development curve
to SELECT each grid. This new workflow instead requires an explicitly frozen
prospective grid, fixed from cited earlier evidence before these query outcomes.
It does not represent an adapter-specific development selection. All existing
inspect_source, calibration-token encoding, target geometry, exact reload and
parameter-digest checks remain mandatory. No frozen function is monkeypatched.

Explicit technical_pilot registrations use exactly eight train-source queries,
one draw and K0/16. They remain ineligible as scientific evidence. A technical
configuration or short pilot adapter cannot satisfy the final-test contract.

Base jobs use K0/16/32; prespecified whole-prompt arm-A jobs use K0/16. They share
the exact registered query/demo pool and draws. Each job computes its OWN three
content-free priors. K0 runs once per query/readout. Generation, raw likelihood
and calibrated likelihood are distinct rows, never independent query samples.

Run only through revision/resource_coordination/run_exclusive.py. Any wrapper
which launches this process must preserve its inherited accelerator descriptor.
CLI validate checks source/design metadata without loading weights or hashing
weight shards; run verifies every registered payload before model construction.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
CODE = PAPER / "code"
WORKSPACE = PAPER.parent
sys.path.insert(0, str(CODE))

from cliff.backends import get_backend, require_weight_access
from cliff.data import TASKS, exact_balanced_sample, load_task
from cliff.experiments.intervention import (encode_training_rows, parameter_digest,
                                           read_exclusion_manifest)
from cliff.experiments.revision_adapter_eval import inspect_source, load_saved_adapter
from cliff.experiments.revision_behaviour import (_argmax, _calibration, _context,
    _log_probabilities, _probabilities, _score, require_pinned_model,
    verify_loaded_model)
from cliff.prompting import parse_label_strict, render_prompt
from cliff.revision_records import canonical_json, content_hash, file_hash

READOUTS = ("generated_label", "raw_complete_label_likelihood",
            "calibrated_complete_label_likelihood")
POLICY = "complete_string_sum_ll_common_conditioning_prefix_v2"
CANONICAL_LOCK = WORKSPACE / "vlm_ijcv_cliff/revision/runs/accelerator.lock"


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return datetime.now(timezone.utc).isoformat()


def checked(reference):
    if not isinstance(reference, dict):
        raise ValueError("Artifact reference must contain path and sha256")
    path = Path(reference.get("path", "MISSING")).resolve()
    if (not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", reference.get("sha256", ""))
            or file_hash(path) != reference["sha256"]):
        raise ValueError("Required local artifact is missing or changed")
    return path


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("query_role") not in {"final_test", "technical_pilot"}:
        raise ValueError("Require explicit schema1 and final_test or technical_pilot role")
    technical = config["query_role"] == "technical_pilot"
    if technical and config.get("scientific_evidence") is not False:
        raise ValueError("Technical pilot must explicitly prohibit scientific evidence")
    if config.get("task") != "ag_news" or config.get("model_state") not in {"base", "adapter"}:
        raise ValueError("This bounded scorer supports AG News base or arm-A adapter only")
    require_pinned_model(config["model"])
    model = config["model"]
    if (model.get("template") != "native" or model.get("dtype") != "bfloat16"
            or model.get("load_in_4bit", False) or model.get("local_files_only") is not True
            or model.get("label_score_implementation") != "full_forward"
            or model.get("tokenizer_revision") != model.get("revision")):
        raise ValueError("Require pinned local native BF16 full-forward evaluation")
    expected_ks = [0, 16] if technical or config["model_state"] == "adapter" else [0, 16, 32]
    if config.get("k_values") != expected_ks:
        raise ValueError("Grid differs from the bounded prospective protocol")
    draws = config.get("demo_seeds", [])
    n_draws = 1 if technical else 5
    if len(draws) != n_draws or any(type(d) is not int for d in draws) or len(set(draws)) != n_draws:
        raise ValueError("Wrong number of explicitly registered distinct draws for this role")
    readouts = config.get("readouts", [])
    if (not readouts or len(readouts) != len(set(readouts)) or not set(readouts) <= set(READOUTS)
            or ((READOUTS[1] in readouts) != (READOUTS[2] in readouts))):
        raise ValueError("Raw/calibrated scores must be paired; generation is separate")
    if config.get("content_free_queries") != ["N/A", "[MASK]", ""]:
        raise ValueError("The three content-free strings are fixed prospectively")
    floor = config.get("calibration_floor")
    if not isinstance(floor, (int, float)) or not math.isfinite(floor) or not 0 <= floor < 1:
        raise ValueError("Require an explicit finite calibration floor")
    if type(config.get("max_new_tokens")) is not int or config["max_new_tokens"] < 1:
        raise ValueError("Require positive generation budget")
    dataset = config["dataset"]
    if (dataset.get("id") != TASKS[config["task"]]["path"]
            or not re.fullmatch(r"[0-9a-f]{40}", dataset.get("revision", ""))):
        raise ValueError("Require pinned official held-out queries and a separate demo split")
    if technical:
        if dataset.get("train_split") != "train" or dataset.get("query_split") != "train":
            raise ValueError("Technical timing queries must use the training split only")
    elif (dataset.get("query_split") != TASKS[config["task"]]["split_test"]
          or dataset.get("train_split") == dataset.get("query_split")):
        raise ValueError("Final queries must use the official held-out split")
    for name in ("train_indices", "query_indices"):
        values = dataset.get(name)
        if (not values or any(type(i) is not int or i < 0 for i in values)
                or len(values) != len(set(values))):
            raise ValueError("Require explicit unique nonnegative source indices")
    n_queries = 8 if technical else 200
    if len(dataset["query_indices"]) != n_queries:
        raise ValueError("Wrong fixed query count for the prospective study role")
    if technical and set(dataset["train_indices"]) & set(dataset["query_indices"]):
        raise ValueError("Technical query/demo source indices overlap")
    role = config["role_manifest"]
    roles = read(checked(role)).get("roles", {})
    if not role.get("query_role") or not role.get("demo_role"):
        raise ValueError("Require exact role names in the materialized manifest")
    for name, size in ((role["query_role"], n_queries), (role["demo_role"], len(dataset["train_indices"]))):
        rows = roles.get(name, [])
        if (len(rows) != size or any(not row.get("id") or not re.fullmatch(r"[0-9a-f]{64}", row.get("hash", "")) for row in rows)):
            raise ValueError("Materialized role has missing identities/hashes or wrong source count")
    if not config.get("exclusion_manifests"):
        raise ValueError("Require prior observed/development/training exclusions")
    for entry in config["exclusion_manifests"]:
        checked(entry)
    if config["model_state"] == "adapter":
        if config.get("adapter_arm") != "A":
            raise ValueError("Calibration applies only to prespecified arm A")
        source = inspect_source(config)
        if (source["configuration"].get("objective") != "whole-prompt"
                or source["configuration"].get("target_policy", "gold") != "gold"
                or source["configuration"].get("intervention_k") != 8
                or source["configuration"].get("component") != "attn"
                or source["configuration"].get("untrained_adapter", False)
                or source["audit"]["requested"] != [12, 19]
                or any(row.get("actual_k") != 8 or row.get("target_policy", "gold") != "gold"
                       for row in read(source["calibration_path"]))):
            raise ValueError("Arm A must be the audited whole-prompt gold protocol")
        expected_rows, expected_updates = (40, 20) if technical else (500, 250)
        if (len(read(source["calibration_path"])) != expected_rows
                or source["configuration"].get("n_calibration") != expected_rows
                or source["audit"].get("optimizer_updates") != expected_updates
                or source["configuration"].get("lora_r") != 16
                or source["configuration"].get("lora_alpha") != 32
                or source["audit"].get("trainable_parameters") != 3407872
                or source["audit"].get("gradient_accumulation_steps") != 4):
            raise ValueError("Saved A source has the wrong calibration/update budget or rank/geometry for this role")
        return source
    if config.get("adapter"):
        raise ValueError("Base job cannot contain an adapter source")
    return None


def verify_registration(config_path, registration_path, *, payloads=False):
    config_path, registration_path = Path(config_path).resolve(), Path(registration_path).resolve()
    config, registration = read(config_path), read(registration_path)
    expected_status = "frozen_technical_pilot" if config.get("query_role") == "technical_pilot" else "frozen_prospective"
    if (registration.get("status") != expected_status
            or registration.get("config_sha256") != file_hash(config_path)
            or not registration.get("study_id")
            or registration.get("new_query_outcomes_used") is not False):
        raise ValueError("An exact frozen prospective registration is required")
    if config.get("query_role") == "technical_pilot" and registration.get("scientific_evidence") is not False:
        raise ValueError("Technical registration must prohibit scientific evidence")
    fixed = registration.get("fixed_grid_selection", {})
    if fixed.get("uses_new_outcomes") is not False or not fixed.get("source_files"):
        raise ValueError("Grid must be fixed prospectively from source-bound earlier evidence")
    for ref in fixed["source_files"]:
        checked(ref)
    source = validate_config(config)
    runtime = registration.get("runtime_source_hashes", {})
    required = {str(p.resolve()) for p in (CODE / "cliff").rglob("*.py")} | {str(Path(__file__).resolve())}
    if not required <= set(runtime):
        raise ValueError("Registration must bind all frozen cliff sources and external driver")
    for name, digest in runtime.items():
        checked(dict(path=name, sha256=digest))
    artifacts = registration.get("artifact_hashes", {})
    for name, digest in artifacts.items():
        checked(dict(path=name, sha256=digest))
    for entry in [config["role_manifest"], *config["exclusion_manifests"]]:
        if artifacts.get(str(Path(entry["path"]).resolve())) != entry["sha256"]:
            raise ValueError("Role and exclusion files must also be bound in the registration")
    if source is not None and config["query_role"] == "final_test":
        validate_final_adapter_membership(config, registration, source)
    for category in ("model_payload_hashes", "dataset_payload_hashes"):
        hashes = registration.get(category, {})
        if not hashes or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in hashes.values()):
            raise ValueError("Require complete local model and dataset payload hash bindings")
        if payloads:
            for name, digest in hashes.items():
                checked(dict(path=name, sha256=digest))
    snapshot = Path(registration["model_snapshot"]).resolve()
    expected_snapshot = (Path(config["cache_dir"]) / "hub" /
                         ("models--" + config["model"]["id"].replace("/", "--")) /
                         "snapshots" / config["model"]["revision"]).resolve()
    if snapshot != expected_snapshot:
        raise ValueError("Local model snapshot differs from registered revision")
    index = read(snapshot / "model.safetensors.index.json")
    needed = {str((snapshot / name).resolve()) for name in set(index["weight_map"].values())}
    needed |= {str((snapshot / name).resolve()) for name in
               ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json")}
    if not needed <= {str(Path(p).resolve()) for p in registration["model_payload_hashes"]}:
        raise ValueError("Model binding omits a shard or required tokenizer/configuration metadata")
    if not all(Path(p).is_file() for p in needed):
        raise ValueError("All selected model shards must already be local")
    dataset_root = Path(config["cache_dir"]).resolve() / "datasets"
    arrows = [Path(p).resolve() for p in registration["dataset_payload_hashes"]]
    if (not arrows or any(not p.is_relative_to(dataset_root)
                          or config["dataset"]["revision"] not in p.parts for p in arrows)
            or not all(any(p.name.endswith("-" + split + ".arrow") for p in arrows)
                       for split in (config["dataset"]["train_split"], config["dataset"]["query_split"]))):
        raise ValueError("Dataset bindings must cover both pinned Arrow splits in the scoped cache")
    return config, registration, source


def validate_final_adapter_membership(config, registration, source, *, _test_validator=None):
    """Require the prespecified A block from this exact registered 12-job study.

    The internal validator injection is solely for tiny membership fixtures; the
    CLI never exposes it. Real execution reuses the external collection's full
    artifact-integrity validator and does not inspect scientific outcome values.
    """
    from run_training_collection import argv_config, validate_artifacts
    contract = registration.get("adapter_source_contract", {})
    block = contract.get("block")
    if (type(block) is not int or block not in (0, 1, 2)
            or config.get("training_block") != block
            or contract.get("job_id") != "A_block_" + str(block)):
        raise ValueError("Final adapter needs its prespecified A0/A1/A2 block membership")
    references = [contract.get("training_registration", {}), contract.get("completed_job_receipt", {})]
    for reference in references:
        checked(reference)
        if registration.get("artifact_hashes", {}).get(str(Path(reference["path"]).resolve())) != reference["sha256"]:
            raise ValueError("Training registration and completed-job receipt must be source-bound artifacts")
    collection, receipt = (read(reference["path"]) for reference in references)
    expected_jobs = {arm + "_block_" + str(i) for arm in "ABCD" for i in range(3)}
    if (collection.get("status") != "frozen_training_collection"
            or collection.get("study_id") != registration["study_id"]
            or len(collection.get("jobs", [])) != 12
            or {job.get("id") for job in collection.get("jobs", [])} != expected_jobs
            or collection.get("scientific_outcome_policy") != "quarantine_until_all12complete"
            or collection.get("expected_optimizer_updates") != 250
            or collection.get("expected_trainable_parameters") != 3407872
            or receipt.get("status") != "complete_and_integrity_validated"
            or receipt.get("job_id") != contract["job_id"]):
        raise ValueError("Saved adapter does not belong to this completed registered A block")
    jobs = [job for job in collection.get("jobs", []) if job.get("id") == contract["job_id"]]
    if len(jobs) != 1:
        raise ValueError("Registered A source job missing or ambiguous")
    job = jobs[0]
    declared = argv_config(job["argv"])
    cfg, audit = source["configuration"], source["audit"]
    for field, expected in (("train_seed", declared["train_seeds"][0]),
                            ("calibration_seed", declared["calibration_seeds"][0]),
                            ("calibration_demo_seed", declared["calibration_demo_seeds"][0])):
        if cfg.get(field) != expected:
            raise ValueError("Actual source seed differs from its prespecified A training block")
    if (audit.get("train_seed") != declared["train_seeds"][0]
            or audit.get("frozen_sha256") != collection.get("expected_frozen_base_sha256")
            or receipt.get("artifact_sha256", {}).get(str(source["manifest_path"])) != config["adapter"]["source_manifest_sha256"]):
        raise ValueError("A source identity or frozen base differs from its registered completed job")
    source_module = str((HERE / "run_training_collection.py").resolve())
    if registration.get("runtime_source_hashes", {}).get(source_module) != file_hash(source_module):
        raise ValueError("Collection integrity-validator source must also be bound")
    actual = (_test_validator or validate_artifacts)(job, collection)
    if actual != receipt:
        raise ValueError("Completed training job integrity receipt no longer reproduces")


def inherited_accelerator():
    descriptor = int(os.environ.get("LLM_SHARED_ACCELERATOR_LOCK_FD", "-1"))
    if descriptor < 0:
        raise ValueError("Model execution requires the shared accelerator launcher")
    opened, expected = os.fstat(descriptor), CANONICAL_LOCK.stat()
    if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
        raise ValueError("Inherited descriptor is not the canonical shared accelerator file")
    if not os.get_inheritable(descriptor):
        raise ValueError("Shared lock descriptor must remain inheritable")
    try:
        # Reasserting the same lock on an inherited open description is
        # idempotent. An unlocked descriptor must not borrow another owner's
        # blocked probe as purported evidence of its own ownership.
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise ValueError("Inherited descriptor does not own the accelerator lock") from error
    # A separate open must conflict with our inherited open description. Do not
    # LOCK_UN the inherited descriptor or mutate the canonical lock filename.
    with CANONICAL_LOCK.open("rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    raise ValueError("Canonical accelerator descriptor is not actively locked")


def validate_data(config, data, source=None):
    dataset, role = config["dataset"], config["role_manifest"]
    expected_meta = {"requested_revision": dataset["revision"], "dataset_id": dataset["id"],
                     "train_split": dataset["train_split"], "test_split": dataset["query_split"]}
    if data.task_id != config["task"] or any(data.metadata.get(k) != v for k, v in expected_meta.items()):
        raise ValueError("Loaded task/revision/splits differ from the registered source")
    manifest = read(checked(role))
    for pool, name in ((data.test, role["query_role"]), (data.train, role["demo_role"])):
        actual = [{"id": ex.example_id, "hash": ex.text_hash} for ex in pool]
        if actual != manifest["roles"][name]:
            raise ValueError("Loaded query/demo identities or order differ from frozen roles")
        if len({ex.example_id for ex in pool}) != len(pool) or len({ex.text_hash for ex in pool}) != len(pool):
            raise ValueError("Duplicate source identities or normalized texts in registered role")
    qids, qhashes = {q.example_id for q in data.test}, {q.text_hash for q in data.test}
    if qids & {d.example_id for d in data.train} or qhashes & {d.text_hash for d in data.train}:
        raise ValueError("Query/demo identity or text overlap")
    excluded = []
    for entry in config["exclusion_manifests"]:
        rows, _ = read_exclusion_manifest(checked(entry))
        excluded.extend(rows)
    if source:
        excluded.extend(source["calibration_exclusions"])
    forbidden_ids = {row.get("id") for row in excluded if row.get("id")}
    forbidden_hashes = {row.get("hash") for row in excluded if row.get("hash")}
    if qids & forbidden_ids or qhashes & forbidden_hashes:
        raise ValueError("Final query overlaps observed/development/training exclusions")
    if ({d.example_id for d in data.train} & forbidden_ids
            or {d.text_hash for d in data.train} & forbidden_hashes):
        raise ValueError("Evaluation demonstrations overlap reserved or adapter-calibration examples")
    # Preserve the complete common prospective pool, rather than silently
    # removing examples differently for each adapter. Materialization must make
    # all query/demo roles disjoint from all calibration roles before registration.
    return data


def random_contexts(config, data):
    contexts = []
    for k in config["k_values"]:
        for draw in config["demo_seeds"][:1] if k == 0 else config["demo_seeds"]:
            demos = exact_balanced_sample(data.train, k, draw, exclude=data.test)
            counts = Counter(d.label for d in demos)
            if len(demos) != k or (k and set(counts.values()) != {k // len(data.label_names)}):
                raise ValueError("Registered balanced-random context could not be realized exactly")
            contexts.append((_context(data.task_id, k, draw, demos, family="balanced_order",
                                      order="random", composition="balanced"), demos))
    return contexts


def checked_score(backend, prompt, labels):
    # Validate complete-token continuation alignment independently of argmax.
    encodings = backend.label_encodings(prompt, labels)
    if set(encodings) != set(labels):
        raise ValueError("Missing complete-label token encodings")
    starts = {start for sequence, start in encodings.values()}
    if len(starts) != 1 or not all(0 < start < len(sequence) for sequence, start in encodings.values()):
        raise ValueError("Complete labels do not have one valid conditioning boundary")
    if len({tuple(sequence[:start]) for sequence, start in encodings.values()}) != 1:
        raise ValueError("Labels use different conditioning prefixes")
    scores, measured = _score(backend, prompt, labels)
    if measured["label_scoring_policy"] != POLICY:
        raise ValueError("Require the validated complete-string scoring convention")
    return scores, measured


def generation_valid(measured, max_new_tokens):
    if (measured.get("measurement_status") != "measured" or measured.get("truncated")
            or measured.get("rejected") or measured.get("input_tokens_before") != measured.get("input_tokens_after")
            or not isinstance(measured.get("input_tokens_after"), int)
            or not isinstance(measured.get("context_limit"), int)
            or measured["input_tokens_after"] + max_new_tokens > measured["context_limit"]):
        raise ValueError("Generation is unmeasured/rejected/truncated or exceeds its own capacity")


def evaluate_grid(config, data, backend, write_prediction, write_calibration):
    """Pure injected-backend loop; tests use only synthetic CPU fixtures."""
    labels, counts = data.label_names, Counter()
    contexts = random_contexts(config, data)
    scoring = READOUTS[1] in config["readouts"]
    for context, demos in contexts:
        calibration = None
        if scoring:
            # Validate every content-free prompt with the same complete-label
            # boundary policy, then reuse frozen W3's actual prior computation.
            for text in config["content_free_queries"]:
                enc = backend.label_encodings(render_prompt(data.task_id, demos, text), labels)
                starts = {start for _, start in enc.values()}
                if (set(enc) != set(labels) or len(starts) != 1
                        or not all(0 < start < len(seq) for seq, start in enc.values())
                        or len({tuple(seq[:start]) for seq, start in enc.values()}) != 1):
                    raise ValueError("Invalid content-free complete-label conditioning boundary")
            calibration = _calibration(backend, data.task_id, demos, labels,
                                       config["content_free_queries"], config["calibration_floor"])
            if any(row["label_scoring_policy"] != POLICY for row in calibration["content_free_records"]):
                raise ValueError("Content-free scoring policy differs from query policy")
            write_calibration({"context_id": context["context_id"], "model_state": config["model_state"],
                               "model_id": config["model"]["id"], **calibration})
            counts["calibration_contexts"] += 1
        for query in data.test:
            prompt = render_prompt(data.task_id, demos, query.text)
            base = {"task": data.task_id, "query_role": config["query_role"],
                    "technical_pilot": config["query_role"] == "technical_pilot", "scientific_evidence": False,
                    "query_id": query.example_id, "query_hash": query.text_hash,
                    "gold_label": query.label, "context_id": context["context_id"],
                    "requested_k": context["requested_k"], "realized_k": len(demos), "draw_id": context["draw_id"],
                    "demonstration_ids": context["demonstration_ids"],
                    "demonstration_hashes": context["demonstration_hashes"],
                    "demonstration_labels": context["demonstration_labels"],
                    "model_id": config["model"]["id"], "model_revision": config["model"]["revision"],
                    "tokenizer_revision": config["model"]["tokenizer_revision"],
                    "dataset_revision": config["dataset"]["revision"], "model_state": config["model_state"],
                    "condition": config["model_state"], "paired_query_draw_id": content_hash(
                        [data.task_id, query.example_id, context["requested_k"], context["draw_id"]])}
            scores, measured = checked_score(backend, prompt, labels) if scoring else (None, None)
            if scoring:
                corrected = _log_probabilities({label: scores[label] - calibration["effective_log_label_prior"][label]
                                                for label in labels})
                for readout, distribution, prediction in (
                    (READOUTS[1], _probabilities(_log_probabilities(scores)), _argmax(scores, labels)),
                    (READOUTS[2], _probabilities(corrected), _argmax(corrected, labels))):
                    write_prediction({**base, **measured, "readout": readout,
                                      "label_scores": scores, "label_probabilities": distribution,
                                      "prediction": prediction, "correct": prediction == query.label,
                                      "parse_status": "restricted_complete_label", "calibration_context_id": context["context_id"],
                                      "score_evaluation_id": content_hash([context["context_id"], query.example_id]),
                                      "n_independent_queries_added_by_readout": 0})
                    counts[readout] += 1
            if READOUTS[0] in config["readouts"]:
                generation = backend.generate_record(prompt, config["max_new_tokens"])
                generation_valid(generation, config["max_new_tokens"])
                if measured and (generation["input_tokens_before"] != measured["input_tokens_before"]
                                 or generation.get("input_token_ids_hash") != measured["input_token_ids_hash"]):
                    raise ValueError("Generation and scoring used different prompt token IDs")
                prediction = parse_label_strict(generation.get("raw_text", ""), labels)
                write_prediction({**base, **generation, "readout": READOUTS[0], "prediction": prediction,
                                  "correct": prediction == query.label,
                                  "parse_status": "parseable" if prediction is not None else "unparseable",
                                  "parser": "strict", "n_independent_queries_added_by_readout": 0})
                counts[READOUTS[0]] += 1
                counts["invalid_generations"] += prediction is None
            counts["physical_query_contexts"] += 1
    expected = len(data.test) * (1 + (len(config["k_values"]) - 1) * len(config["demo_seeds"]))
    if counts["physical_query_contexts"] != expected or any(counts[r] != expected for r in config["readouts"]):
        raise ValueError("Registered query/context/readout grid is incomplete")
    return {"counts": dict(counts), "n_queries": len(data.test), "n_contexts": len(contexts),
            "k0_once_per_query_per_readout": True, "own_context_specific_priors": True,
            "readouts_are_paired_not_independent_queries": True}


def run(config_path, registration_path):
    inherited_accelerator()
    config, registration, source = verify_registration(config_path, registration_path, payloads=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:10]
    directory = PAPER / "revision/four_day_revision/runs" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"status": "running", "run_id": run_id, "started_at": utc(),
                "query_role": config["query_role"], "technical_pilot": config["query_role"] == "technical_pilot",
                "scientific_evidence": False, "configuration": config, "config_sha256": content_hash(config),
                "registration_sha256": file_hash(registration_path), "registration": registration,
                "python": sys.version, "source_policy": "prospective fixed grid; no adapter-specific grid selection"}
    def store(name, value):
        temporary = directory / ("." + name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, directory / name)
    store("manifest.json", manifest)
    try:
        shutil.copyfile(config_path, directory / "registered_config.json")
        shutil.copyfile(registration_path, directory / "registration.json")
        for name, digest in registration["runtime_source_hashes"].items():
            path = Path(name).resolve()
            relative = path.relative_to(WORKSPACE)
            target = directory / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if file_hash(target) != digest:
                raise ValueError("Runtime snapshot hash changed")
        dataset = config["dataset"]
        from cliff.revision_runtime import configure_cache
        configure_cache(config["cache_dir"])
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                          TOKENIZERS_PARALLELISM="false")
        data = load_task(config["task"], n_train=None, n_test=None, revision=dataset["revision"],
                         train_split=dataset["train_split"], test_split=dataset["query_split"],
                         train_indices=dataset["train_indices"], test_indices=dataset["query_indices"])
        validate_data(config, data, source)
        model = config["model"]
        backend = require_weight_access(get_backend("hf_local", model["id"], revision=model["revision"],
            tokenizer_revision=model["tokenizer_revision"], template=model["template"], dtype=model["dtype"],
            device=model.get("device", "mps"), local_files_only=True, overflow_policy="reject",
            label_prefix=model.get("label_prefix", " "), label_score_implementation="full_forward",
            template_marker_policy=model.get("template_marker_policy", "native_tokens"),
            max_seq_len=model.get("max_seq_len"), load_in_4bit=False), "focused prospective scoring")
        resolved = verify_loaded_model(model, backend, "final_test")
        if (backend.model_id != model["id"] or backend.revision != model["revision"]
                or backend.template != model["template"] or backend.overflow_policy != "reject"
                or backend.label_scoring_policy != POLICY
                or backend.label_prefix != model.get("label_prefix", " ")
                or backend.label_score_implementation != "full_forward"
                or backend.template_marker_policy != model.get("template_marker_policy", "native_tokens")):
            raise ValueError("Loaded backend identity/template/scoring policy differs")
        actual_dtype = str(next(backend.torch_model().parameters()).dtype)
        if actual_dtype != "torch.bfloat16":
            raise ValueError("Loaded base precision differs from registered BF16")
        resolved.update(actual_base_dtype=actual_dtype, actual_device=str(backend.device),
                        template_hash=backend._template_hash(),
                        label_scoring_policy=backend.label_scoring_policy)
        targeting, parameter_hash = None, None
        if source:
            _, encoding = encode_training_rows(backend, read(source["calibration_path"]),
                                               source["configuration"].get("objective", "label-span") == "label-span")
            if encoding != source["audit"].get("encoding"):
                raise ValueError("Saved calibration token sequences differ from original training audit")
            targeting = load_saved_adapter(backend, source)
            parameter_hash = parameter_digest(backend.torch_model(), trainable=False)
        else:
            backend.torch_model().requires_grad_(False)
            backend.torch_model().eval()
        contexts = random_contexts(config, data)
        store("design.json", {"queries": [{"id": q.example_id, "hash": q.text_hash, "gold": q.label} for q in data.test],
               "contexts": [context for context, _ in contexts], "data_metadata": data.metadata})
        with (directory / "predictions.jsonl").open("x") as predictions, (directory / "calibration.jsonl").open("x") as calibrations:
            def write_row(row):
                if source:
                    row.update(training_run_id=source["training_run_id"], source_manifest_sha256=config["adapter"]["source_manifest_sha256"])
                predictions.write(canonical_json({"run_id": run_id, **row}) + "\n")
                predictions.flush()
            def write_prior(row):
                calibrations.write(canonical_json({"run_id": run_id, **row}) + "\n")
                calibrations.flush()
            summary = evaluate_grid(config, data, backend, write_row, write_prior)
        if source and parameter_hash != parameter_digest(backend.torch_model(), trainable=False):
            raise ValueError("Evaluation changed the frozen base/adapter bytes")
        if (file_hash(registration_path) != manifest["registration_sha256"]
                or file_hash(config_path) != registration["config_sha256"]):
            raise ValueError("Initial prospective registration/config changed during evaluation")
        verify_registration(config_path, registration_path, payloads=True)
        summary.update(resolved_model=resolved, targeting=targeting, evaluation_parameter_sha256=parameter_hash,
                       model_context_capacity=backend.max_seq_len, scientific_evidence=False,
                       technical_pilot=config["query_role"] == "technical_pilot",
                       inference_status=("technical throughput/measurement validation only; not scientific evidence"
                         if config["query_role"] == "technical_pilot" else
                         "complete raw collection; independent validation/paired analysis required"))
        store("metrics.json", summary)
        store("validation.json", {"passed": True, "complete_registered_grid": True, "no_training": True,
                                  "all_measurements_untruncated": True, "source_hashes_rechecked": True,
                                  "exact_saved_adapter_checks": source is not None})
        manifest.update(status="complete", finished_at=utc(), elapsed_seconds=time.monotonic() - started,
                        data_metadata=data.metadata, backend_provenance=resolved,
                        scientific_evidence=False, independent_analysis_required=True,
                        artifact_sha256={name: file_hash(directory / name) for name in
                            ("registered_config.json", "registration.json", "design.json", "predictions.jsonl",
                             "calibration.jsonl", "metrics.json", "validation.json")})
        store("manifest.json", manifest)
        return directory
    except BaseException as error:
        manifest.update(status="failed", error_type=type(error).__name__, finished_at=utc(),
                        elapsed_seconds=time.monotonic() - started)
        store("manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["validate", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    if args.operation == "validate":
        config, _, _ = verify_registration(args.config, args.registration, payloads=False)
        print(json.dumps({"status": "metadata_contract_validated", "model_state": config["model_state"],
                          "weights_loaded": False, "weight_payloads_hashed": False,
                          "scientific_launch_ready": False, "device_pilot_required": True}))
    else:
        print(json.dumps({"status": "raw_collection_complete", "run_directory": str(run(args.config, args.registration))}))


if __name__ == "__main__":
    main()
