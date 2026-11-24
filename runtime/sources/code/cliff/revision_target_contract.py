"""Optional registered LoRA target/count gate for audited intervention runs.

``make_contract(geometry, choice, tokenizer_revision=..., protocol_id=...)``
returns a JSON registration, containing the independently source-verified header
geometry and its exact deterministic target choice. Save it before training;
pass its path and actual file SHA256 to ``cliff intervene --targeting-contract``
and ``--targeting-contract-sha256``. The gate runs after PEFT construction and
before Trainer/optimizer construction. It never loads a checkpoint or downloads.

Header geometry does not authenticate full checkpoint values. This gate retains
that limitation and checks the already-loaded immutable checkpoint/tokenizer
identities and exact realized A/B shapes/count; normal frozen-base, gradient,
calibration, payload-provenance and scientific-acceptance gates still apply.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re

from .revision_parameter_matching import (audit_realized_targeting, targeting_choice,
                                           verify_geometry)
from .revision_records import content_hash, file_hash

SHA40 = re.compile(r"[0-9a-fA-F]{40}")
SHA64 = re.compile(r"[0-9a-fA-F]{64}")


def choice_for_window(geometry, *, start, width, region, rank, component="attn", alpha_over_rank=2):
    """Create an explicit source-bound choice, including non-relative depth cuts."""
    base = targeting_choice(geometry, region=region, width=width, rank=rank,
        component=component, alpha_over_rank=alpha_over_rank)
    if type(start) is not int or not 0 <= start <= geometry["n_layers"]-width:
        raise ValueError("explicit registered window exceeds source geometry")
    if start == base["window"]["start"]:
        return base
    modules = [module for module in geometry["modules"] if start <= module["block"] < start+width
        and (component == "both" or module["component"] == component)]
    if not modules or any(rank > min(module["in_features"], module["out_features"]) for module in modules):
        raise ValueError("explicit window has no feasible source-bound low-rank targets")
    base["window"].update(start=start, stop_exclusive=start+width, inclusive=[start, start+width-1],
        normalized_start=start/geometry["n_layers"], normalized_stop=(start+width)/geometry["n_layers"],
        basis="new explicit source-verified window; not recovered historical targeting")
    base["expected_targets"] = sorted(module["name"] for module in modules)
    base["nominal_integer_matrices"] = [dict(module=module["name"], A_shape=[rank, module["in_features"]],
        B_shape=[module["out_features"], rank], n_trainable=rank*(module["in_features"]+module["out_features"])) for module in modules]
    base["exact_trainable_parameters"] = sum(row["n_trainable"] for row in base["nominal_integer_matrices"])
    base["choice_sha256"] = content_hash({key: value for key, value in base.items() if key != "choice_sha256"})
    return base


def validate_contract(contract):
    if (contract.get("schema_version") != 1 or contract.get("role") != "registered_loaded_target_count"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", contract.get("protocol_id", ""))
            or not SHA40.fullmatch(contract.get("tokenizer_revision") or "")):
        raise ValueError("target contract needs a registered protocol and immutable tokenizer revision")
    if content_hash({key: value for key, value in contract.items() if key != "contract_sha256"}) != contract.get("contract_sha256"):
        raise ValueError("registered target contract digest mismatch")
    geometry = verify_geometry(contract.get("geometry", {}))
    choice = contract.get("expected_targeting", {})
    window = choice.get("window", {})
    expected = choice_for_window(geometry, start=window.get("start"), region=window.get("region"), width=window.get("width"),
        rank=choice.get("lora_r"), component=choice.get("component"), alpha_over_rank=choice.get("alpha_over_rank"))
    if choice != expected:
        raise ValueError("target choice differs from the registered source geometry/window/rank/count")
    if contract.get("scientific_evidence") is not False:
        raise ValueError("a target/count registration cannot claim scientific evidence")
    return contract


def make_contract(geometry, choice, *, tokenizer_revision, protocol_id):
    contract = dict(schema_version=1, role="registered_loaded_target_count", protocol_id=protocol_id,
        tokenizer_revision=tokenizer_revision, geometry=copy.deepcopy(geometry),
        expected_targeting=copy.deepcopy(choice), scientific_evidence=False,
        full_checkpoint_values_authenticated_by_this_contract=False)
    contract["contract_sha256"] = content_hash(contract)
    return validate_contract(contract)


def read_contract(path, expected_sha256):
    if not isinstance(path, (str, Path)) or "://" in str(path):
        raise ValueError("registered target contract must be a local immutable artifact")
    path = Path(path)
    if not path.is_file() or not SHA64.fullmatch(expected_sha256 or "") or file_hash(path) != expected_sha256:
        raise ValueError("registered target contract file is missing or changed")
    return validate_contract(json.loads(path.read_text()))


def audit_loaded_targeting(contract, backend, model, *, window, component, rank, alpha):
    """Check the actual loaded adapter before constructing any optimizer."""
    validate_contract(contract)
    choice = contract["expected_targeting"]
    geometry = contract["geometry"]
    expected_window = choice["window"]
    if (list(window) != list(range(expected_window["start"], expected_window["stop_exclusive"]))
            or component != choice["component"] or rank != choice["lora_r"] or alpha != choice["lora_alpha"]):
        raise ValueError("actual intervention window/component/rank/alpha differs from registration")
    if (getattr(backend, "model_id", None) != choice["model_id"]
            or getattr(backend, "revision", None) != choice["revision"]
            or getattr(getattr(model, "config", None), "_commit_hash", None) != choice["revision"]
            or getattr(backend, "resolved_tokenizer_revision", None) != contract["tokenizer_revision"]):
        raise ValueError("actual loaded checkpoint/tokenizer identities differ from target registration")
    config = getattr(model, "config", None)
    config = getattr(config, "text_config", config)
    heads = getattr(config, "num_attention_heads", None)
    head_dim = getattr(config, "head_dim", None) or (getattr(config, "hidden_size", 0)//heads if heads else None)
    kv_heads = getattr(config, "num_key_value_heads", None) or getattr(config, "num_kv_heads", None) or heads
    if (getattr(config, "num_hidden_layers", None) != geometry["n_layers"]
            or getattr(config, "hidden_size", None) != geometry["hidden_size"]
            or getattr(config, "intermediate_size", None) != geometry["intermediate_size"]
            or getattr(config, "model_type", None) != geometry["model_type"]
            or heads != geometry["query_heads"] or head_dim != geometry["head_dim"]
            or kv_heads*head_dim not in geometry["actual_kv_widths"]):
        raise ValueError("actual loaded model configuration differs from source geometry")
    configurations = getattr(model, "peft_config", {})
    if set(configurations) != {"default"}:
        raise ValueError("registered targeting requires exactly one default adapter")
    peft = configurations["default"]
    if (getattr(peft, "r", None) != rank or getattr(peft, "lora_alpha", None) != alpha
            or getattr(peft, "bias", None) != "none" or getattr(peft, "use_dora", False)
            or getattr(peft, "use_rslora", False) or getattr(peft, "modules_to_save", None)):
        raise ValueError("actual adapter rank/scaling/bias options differ from registration")
    shapes = {name: list(parameter.shape) for name, parameter in model.named_parameters() if parameter.requires_grad}
    result = audit_realized_targeting(choice, shapes)
    return dict(result, contract_sha256=contract["contract_sha256"], protocol_id=contract["protocol_id"],
        geometry_sha256=geometry["geometry_sha256"], resolved_model_revision=choice["revision"],
        resolved_tokenizer_revision=contract["tokenizer_revision"], checked_before_optimizer_construction=True,
        scientific_evidence=False, full_checkpoint_values_authenticated_by_this_contract=False)
