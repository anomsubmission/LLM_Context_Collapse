"""Source-bound, exact nominal LoRA parameter matching; no model allocation.

Geometry comes from local pinned config and safetensors headers, not model-name
guesses or randomly initialized models. Counts are A[r,in] + B[out,r], with no
trainable bias, DoRA, rsLoRA, saved base modules, or heterogeneous rank patterns.
The matching grid and tolerance must be frozen independently of task outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
BLOCK = re.compile(r"^(?P<prefix>(?:.*\.)?(?:layers|h|blocks|block))\.(?P<block>\d+)\.(?P<suffix>.+)\.weight$")
SUPPORTED = {"llama", "mistral", "qwen2", "gemma", "gemma2", "gemma3_text", "internlm2"}
ATTN_LAYOUTS = [
    {"q_proj", "k_proj", "v_proj", "o_proj"}, {"wq", "wk", "wv", "wo"},
    {"wqkv", "wo"}, {"qkv_proj", "o_proj"}, {"W_pack", "o_proj"},
    {"Wqkv", "out_proj"}, {"q_proj", "kv_proj", "o_proj"},
]
MLP_LAYOUTS = [
    {"gate_proj", "up_proj", "down_proj"}, {"w1", "w2", "w3"},
    {"fc1", "fc2"}, {"gate_up_proj", "down_proj"},
    {"dense_h_to_4h", "dense_4h_to_h"},
]
FLOAT_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _path(value, root):
    return Path(value) if Path(value).is_absolute() else Path(root) / value


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} requires a positive explicit integer")
    return value


def _config_integer(config, names):
    values = [_integer(config[name], name) for name in names if name in config]
    if not values or len(set(values)) != 1:
        raise ValueError(f"Missing or conflicting configuration geometry: {names}")
    return values[0]


def _role(suffix):
    parts = suffix.split(".")[:-1]
    attention = any(part in {"attn", "self_attn", "attention", "self_attention"} for part in parts)
    mlp = any(part in {"mlp", "ffn", "feed_forward"} for part in parts)
    if attention and mlp:
        raise ValueError(f"Ambiguous attention/MLP target: {suffix}")
    return "attn" if attention else "mlp" if mlp else None


def _header(path):
    """Read only the safetensors JSON header, not any tensor values."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header prefix")
        length = struct.unpack("<Q", prefix)[0]
        if not 1 <= length <= 64 * 1024 * 1024:
            raise ValueError("Invalid or excessive safetensors header length")
        payload = stream.read(length)
    if len(payload) != length:
        raise ValueError("Truncated safetensors JSON header")
    header = json.loads(payload)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    size = Path(path).stat().st_size
    for name, tensor in tensors.items():
        if not isinstance(tensor, dict) or not isinstance(tensor.get("shape"), list):
            raise ValueError(f"Malformed tensor geometry: {name}")
        if any(type(dimension) is not int or dimension < 0 for dimension in tensor["shape"]):
            raise ValueError(f"Invalid tensor dimension: {name}")
        offsets = tensor.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(offset) is not int for offset in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - 8 - length):
            raise ValueError(f"Missing or invalid safetensors payload extent: {name}")
        if tensor.get("dtype") in FLOAT_BYTES and (
                math.prod(tensor["shape"]) * FLOAT_BYTES[tensor["dtype"]] != offsets[1] - offsets[0]):
            raise ValueError(f"Tensor extent does not match its matrix geometry: {name}")
    return tensors, {"path": str(Path(path).resolve()), "header_sha256": hashlib.sha256(prefix + payload).hexdigest(),
                     "header_bytes": 8 + length, "file_size_bytes": size,
                     "weight_values_read_or_verified": False}


def _source_entry(entry, root):
    if not entry.get("id") or not SHA40.fullmatch(entry.get("revision") or ""):
        raise ValueError("Geometry requires a named model and immutable 40-character revision")
    if entry.get("config_source_revision") != entry["revision"]:
        raise ValueError("Configuration provenance differs from the checkpoint revision")
    config_path = _path(entry.get("config_path") or "MISSING", root)
    snapshot = _path(entry.get("snapshot_path") or "MISSING", root)
    if not config_path.is_file() or file_hash(config_path) != entry.get("config_sha256"):
        raise ValueError("Missing or changed source-bound configuration")
    if not snapshot.is_dir():
        raise ValueError("Pinned local safetensors snapshot is required; config-only guesses are forbidden")
    if (snapshot.name != entry["revision"] or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name != "models--" + entry["id"].replace("/", "--")):
        raise ValueError("Snapshot path is not bound to the registered HF repository and commit")
    snapshot_config = snapshot / "config.json"
    if not snapshot_config.is_file() or file_hash(snapshot_config) != entry["config_sha256"]:
        raise ValueError("Snapshot configuration differs from the registered source")
    config = json.loads(config_path.read_text())
    if config.get("quantization_config"):
        raise ValueError("Packed/quantized source tensors have no supported nominal-matrix interpretation")
    reviews = []
    if config.get("auto_map"):
        review = entry.get("remote_code_review", {})
        if (entry.get("trust_remote_code_reviewed") is not True or review.get("revision") != entry["revision"]
                or not review.get("files")):
            raise ValueError("Remote-code geometry requires recorded reviewed implementation files")
        for item in review["files"]:
            path = _path(item["path"], root)
            if file_hash(path) != item["sha256"]:
                raise ValueError("Reviewed remote implementation changed")
            reviews.append({"path": str(path.resolve()), "sha256": item["sha256"]})
    source = {"id": entry["id"], "revision": entry["revision"],
              "config_source_revision": entry["config_source_revision"],
              "config_path": str(config_path.resolve()), "config_sha256": entry["config_sha256"],
              "snapshot_path": str(snapshot.absolute()), "trust_remote_code_reviewed": bool(reviews)}
    if reviews:
        source["remote_code_review"] = {"revision": entry["revision"], "files": reviews}
    return source, config, snapshot


def geometry_from_checkpoint(entry, *, root=Path(".")):
    source, outer, snapshot = _source_entry(entry, root)
    config = outer.get("text_config", outer)
    model_type = config.get("model_type", outer.get("model_type"))
    if model_type not in SUPPORTED or any(config.get(name) for name in ("num_local_experts", "num_experts", "moe_intermediate_size")):
        raise ValueError("Unsupported or expert architecture; no geometry is inferred from its name")
    depth = _config_integer(config, ("num_hidden_layers", "n_layer", "num_layers"))
    hidden = _config_integer(config, ("hidden_size",))
    intermediate = _config_integer(config, ("intermediate_size",))
    heads = _config_integer(config, ("num_attention_heads",))
    if "head_dim" in config:
        head_dim = _integer(config["head_dim"], "head_dim")
    elif hidden % heads == 0:
        head_dim = hidden // heads
    else:
        raise ValueError("Head dimension is not resolved by the pinned configuration")
    q_width = heads * head_dim
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("Missing safetensors index weight_map")
        files = sorted(set(mapping.values()))
        index_evidence = {"path": str(index_path.absolute()), "sha256": file_hash(index_path)}
    else:
        mapping, files, index_evidence = None, ["model.safetensors"], None
    tensors, evidence = {}, []
    for filename in files:
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Shard filename must be a single local path component")
        current, record = _header(snapshot / filename)
        if set(current) & set(tensors):
            raise ValueError("Duplicate tensors across safetensors shards")
        if mapping is not None and any(mapping.get(name) != filename for name in current):
            raise ValueError("Index and actual safetensors header names disagree")
        tensors.update(current)
        evidence.append(record)
    if mapping is not None and set(mapping) != set(tensors):
        raise ValueError("Index references tensors missing from the actual headers")
    if any(".lora_" in name or ".qweight" in name for name in tensors):
        raise ValueError("Adapter or packed-weight snapshot cannot define the base geometry")
    prefixes = defaultdict(list)
    excluded_norms = []
    known_leaves = set().union(*ATTN_LAYOUTS, *MLP_LAYOUTS)
    norm_leaves = {"q_norm", "k_norm", "q_layernorm", "k_layernorm", "query_layernorm", "key_layernorm"}
    for name, tensor in tensors.items():
        match = BLOCK.fullmatch(name)
        if match and _role(match["suffix"]) is not None:
            if len(tensor["shape"]) != 2:
                leaf = match["suffix"].rsplit(".", 1)[-1]
                if leaf in norm_leaves and len(tensor["shape"]) == 1:
                    excluded_norms.append({"name": name, "shape": tensor["shape"], "reason": "known frozen per-head normalization, not a LoRA projection"})
                    continue
                reason = "eligible projection" if leaf in known_leaves else "unsupported component weight"
                raise ValueError(f"Nonmatrix {reason}: {name}")
            if not all(tensor["shape"]):
                raise ValueError(f"Eligible projection has an empty nominal dimension: {name}")
            prefixes[match["prefix"]].append((name, match, tensor))
    complete = [prefix for prefix, values in prefixes.items()
                if {int(match["block"]) for _, match, _ in values} == set(range(depth))]
    if len(complete) != 1:
        raise ValueError("Missing or ambiguous complete text-block stack")
    prefix = complete[0]
    modules, suffix_sets, kv_widths = [], {}, set()
    for block in range(depth):
        block_rows = [(name, match, tensor) for name, match, tensor in prefixes[prefix] if int(match["block"]) == block]
        role_rows = {role: [(name, match, tensor) for name, match, tensor in block_rows if _role(match["suffix"]) == role] for role in ("attn", "mlp")}
        for role, rows in role_rows.items():
            suffixes = {match["suffix"] for _, match, _ in rows}
            containers = {suffix.rsplit(".", 1)[0] for suffix in suffixes}
            leaves = {suffix.rsplit(".", 1)[-1] for suffix in suffixes}
            layouts = ATTN_LAYOUTS if role == "attn" else MLP_LAYOUTS
            if len(containers) != 1 or leaves not in layouts:
                raise ValueError(f"Incomplete/ambiguous {role} projections in block {block}: {sorted(suffixes)}")
            if role in suffix_sets and suffixes != suffix_sets[role]:
                raise ValueError("Target projection names differ between blocks")
            suffix_sets[role] = suffixes
        shapes = {match["suffix"].rsplit(".", 1)[-1]: tensor["shape"] for _, match, tensor in role_rows["attn"]}
        key = next((leaf for leaf in ("k_proj", "wk") if leaf in shapes), None)
        if key is not None:
            kv_width = shapes[key][0]
        elif "kv_proj" in shapes:
            if shapes["kv_proj"][0] % 2:
                raise ValueError("Fused KV width is not divisible by two")
            kv_width = shapes["kv_proj"][0] // 2
        else:
            fused = next(leaf for leaf in ("wqkv", "qkv_proj", "W_pack", "Wqkv") if leaf in shapes)
            remaining = shapes[fused][0] - q_width
            if remaining <= 0 or remaining % 2:
                raise ValueError("Fused QKV shape cannot resolve grouped-query geometry")
            kv_width = remaining // 2
        if kv_width <= 0 or kv_width % head_dim or heads % (kv_width // head_dim):
            raise ValueError("Actual KV projection width is incompatible with grouped-query heads")
        kv_heads = kv_width // head_dim
        for field in ("num_key_value_heads", "num_kv_heads"):
            if field in config and _integer(config[field], field) != kv_heads:
                raise ValueError("Actual KV matrix geometry differs from the pinned configuration")
        kv_widths.add(kv_width)
        for name, match, tensor in block_rows:
            role = _role(match["suffix"])
            leaf = match["suffix"].rsplit(".", 1)[-1]
            expected = ((q_width, hidden) if leaf in {"q_proj", "wq"} else
                        (kv_width, hidden) if leaf in {"k_proj", "v_proj", "wk", "wv"} else
                        (q_width + 2 * kv_width, hidden) if leaf in {"wqkv", "qkv_proj", "W_pack", "Wqkv"} else
                        (2 * kv_width, hidden) if leaf == "kv_proj" else
                        (hidden, q_width) if role == "attn" else
                        (hidden, intermediate) if leaf in {"down_proj", "w2", "fc2", "dense_4h_to_h"} else
                        (2 * intermediate, hidden) if leaf == "gate_up_proj" else (intermediate, hidden))
            if tensor["dtype"] not in FLOAT_BYTES or tuple(tensor["shape"]) != expected:
                raise ValueError(f"Actual nominal projection matrix disagrees with supported geometry: {name}")
            modules.append({"name": name[:-7], "block": block, "component": role,
                            "out_features": expected[0], "in_features": expected[1],
                            "stored_shape": list(expected), "dtype": tensor["dtype"],
                            "linear_layout": "out_features,in_features"})
    geometry = {"schema_version": 1, "source": source, "model_type": model_type,
                "n_layers": depth, "hidden_size": hidden, "intermediate_size": intermediate,
                "query_heads": heads, "head_dim": head_dim,
                "actual_kv_widths": sorted(kv_widths), "text_block_prefix": prefix,
                "index_evidence": index_evidence, "header_evidence": evidence,
                "modules": sorted(modules, key=lambda module: module["name"]),
                "excluded_known_norms": sorted(excluded_norms, key=lambda item: item["name"]),
                "weight_values_verified": False, "geometry_source": "actual_local_safetensors_headers_and_pinned_config",
                "scientific_evidence": False}
    geometry["geometry_sha256"] = content_hash(geometry)
    return geometry


def verify_geometry(geometry):
    payload = {key: value for key, value in geometry.items() if key != "geometry_sha256"}
    if content_hash(payload) != geometry.get("geometry_sha256"):
        raise ValueError("Geometry artifact digest mismatch")
    current = geometry_from_checkpoint(geometry["source"])
    if current != geometry:
        raise ValueError("Source geometry or pinned header/config evidence changed")
    return geometry


def window_for(depth, width, region):
    _integer(depth, "depth"); _integer(width, "width")
    if width > depth or region not in {"early", "mid", "late"}:
        raise ValueError("Relative-depth window does not fit the verified stack")
    start = {"early": 0, "mid": (depth - width) // 2, "late": depth - width}[region]
    return {"region": region, "start": start, "width": width, "stop_exclusive": start + width,
            "inclusive": [start, start + width - 1],
            "normalized_start": start / depth, "normalized_stop": (start + width) / depth,
            "normalized_width_exact": f"{Fraction(width, depth)}",
            "basis": "new source-verified sensitivity, not recovered historical targeting"}


def targeting_choice(geometry, *, region, width, rank, component="attn", alpha_over_rank=2):
    _integer(rank, "rank"); _integer(alpha_over_rank, "alpha_over_rank")
    if component not in {"attn", "mlp", "both"}:
        raise ValueError("Component must be attn, mlp or both")
    window = window_for(geometry["n_layers"], width, region)
    targets = [module for module in geometry["modules"] if window["start"] <= module["block"] < window["stop_exclusive"]
               and (component == "both" or module["component"] == component)]
    if not targets or any(rank > min(module["in_features"], module["out_features"]) for module in targets):
        raise ValueError("Rank exceeds the low-rank resolution of an eligible projection")
    matrices = [{"module": module["name"], "A_shape": [rank, module["in_features"]],
                 "B_shape": [module["out_features"], rank],
                 "n_trainable": rank * (module["in_features"] + module["out_features"])} for module in targets]
    choice = {"model_id": geometry["source"]["id"], "revision": geometry["source"]["revision"],
              "geometry_sha256": geometry["geometry_sha256"], "window": window,
              "component": component, "lora_r": rank, "lora_alpha": rank * alpha_over_rank,
              "alpha_over_rank": alpha_over_rank, "bias": "none", "use_dora": False,
              "use_rslora": False, "modules_to_save": None, "adapter_name": "default",
              "expected_targets": sorted(module["name"] for module in targets),
              "nominal_integer_matrices": matrices,
              "exact_trainable_parameters": sum(matrix["n_trainable"] for matrix in matrices)}
    choice["choice_sha256"] = content_hash(choice)
    return choice


def plan_matching(policy, geometries):
    allowed = {"schema_version", "design_id", "selection_basis", "models", "reference", "component", "regions",
               "ranks", "widths", "max_relative_error", "alpha_over_rank"}
    if set(policy) - allowed or policy.get("selection_basis") != "geometry_only_no_outcome_selection":
        raise ValueError("Only an explicit geometry-only preregistration policy is supported")
    models = policy.get("models", [])
    if not models or len(set(models)) != len(models) or set(models) - set(geometries):
        raise ValueError("Every distinct registered model needs verified source geometry")
    for key in models:
        verify_geometry(geometries[key])
    ranks, widths, regions = (policy.get(name, []) for name in ("ranks", "widths", "regions"))
    for name, values in (("ranks", ranks), ("widths", widths)):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"Explicit unique {name} grid required")
        for value in values:
            _integer(value, name)
    if not regions or len(set(regions)) != len(regions) or set(regions) - {"early", "mid", "late"}:
        raise ValueError("Explicit distinct early/mid/late regions required")
    tolerance = Fraction(str(policy["max_relative_error"]))
    if not 0 <= tolerance < 1:
        raise ValueError("Relative tolerance must be explicit and between zero and one")
    reference = policy["reference"]
    if (set(reference) != {"model", "region", "width", "rank"} or reference["model"] not in models
            or reference["width"] not in widths or reference["rank"] not in ranks):
        raise ValueError("Reference must explicitly identify one registered window/rank")
    component, alpha = policy.get("component", "attn"), policy.get("alpha_over_rank", 2)
    ref_geometry = geometries[reference["model"]]
    ref_choice = targeting_choice(ref_geometry, region=reference["region"], width=reference["width"], rank=reference["rank"], component=component, alpha_over_rank=alpha)
    target = ref_choice["exact_trainable_parameters"]
    comparisons = []
    for key in models:
        geometry = geometries[key]
        for region in regions:
            candidates, rejected = [], []
            for width in sorted(widths):
                for rank in sorted(ranks):
                    try:
                        choice = targeting_choice(geometry, region=region, width=width, rank=rank, component=component, alpha_over_rank=alpha)
                    except ValueError as error:
                        rejected.append({"width": width, "rank": rank, "reason": str(error)})
                        continue
                    error = abs(choice["exact_trainable_parameters"] - target)
                    span_error = abs(Fraction(width, geometry["n_layers"]) - Fraction(reference["width"], ref_geometry["n_layers"]))
                    candidates.append((error, span_error, abs(rank - reference["rank"]), abs(width - reference["width"]), choice))
            candidates.sort(key=lambda candidate: (candidate[:4], candidate[4]["window"]["start"], candidate[4]["lora_r"]))
            best = candidates[0][4] if candidates else None
            within = bool(best is not None and Fraction(candidates[0][0], target) <= tolerance)
            comparisons.append({"model": key, "region": region, "status": "MATCHED" if within else "INSUFFICIENT_GRID_RESOLUTION",
                "chosen": best if within else None, "nearest_available": best,
                "relative_parameter_error_exact": str(Fraction(candidates[0][0], target)) if candidates else None,
                "n_feasible_candidates": len(candidates), "ineligible_grid_entries": rejected,
                "candidate_counts": [{"width": candidate[4]["window"]["width"], "rank": candidate[4]["lora_r"],
                    "exact_trainable_parameters": candidate[4]["exact_trainable_parameters"],
                    "absolute_parameter_error": candidate[0], "normalized_span_error_exact": str(candidate[1])} for candidate in candidates]})
    report = {"schema_version": 1, "policy": policy, "policy_sha256": content_hash(policy),
              "generator_sha256": file_hash(Path(__file__)), "status": "MATCHED" if all(row["status"] == "MATCHED" for row in comparisons) else "INFEASIBLE",
              "scientific_evidence": False, "new_training_runs": 0, "reference_choice": ref_choice,
              "target_trainable_parameters": target,
              "source_geometry_sha256": {key: geometries[key]["geometry_sha256"] for key in models},
              "selection_order": ["minimum exact absolute parameter error", "minimum normalized-width error from reference", "minimum rank distance from reference", "minimum absolute width distance", "start then rank"],
              "comparisons": comparisons,
              "limitations": ["Matching trainable matrix counts does not match FLOPs, activation memory, optimizer dynamics or expressivity.",
                "Changing width changes covered depth and changing rank changes factorization; report both sensitivities separately from fixed-width primary results.",
                "Geometry-only determinism does not prove prospective registration; freeze the policy hash before scientific outcome inspection.",
                "All existing development, source, task-pool and independent-training gates remain required. Runtime targets and trainable shapes must match the chosen matrices."]}
    report["plan_sha256"] = content_hash(report)
    return report


def targeting_override(plan, model, region):
    if content_hash({key: value for key, value in plan.items() if key != "plan_sha256"}) != plan.get("plan_sha256"):
        raise ValueError("Matching plan digest mismatch")
    if plan["status"] != "MATCHED":
        raise ValueError("No targeting override can be launched from an infeasible plan")
    found = [row for row in plan["comparisons"] if row["model"] == model and row["region"] == region]
    if len(found) != 1 or found[0]["chosen"] is None:
        raise ValueError("No unique matched targeting choice")
    choice = found[0]["chosen"]
    return {"window": {name: choice["window"][name] for name in ("region", "start", "width")},
            "component": choice["component"], "lora_r": choice["lora_r"], "lora_alpha": choice["lora_alpha"],
            "parameter_matching_plan_sha256": plan["plan_sha256"],
            "expected_targeting": choice,
            "required_payload_verification": "Training launcher must verify full checkpoint weight-content provenance and the loaded immutable model revision; header geometry alone cannot establish these.",
            "required_runtime_audit": "audit_realized_targeting(choice, actual_trainable_parameter_shapes) before optimizer construction"}


def audit_realized_targeting(choice, trainable_shapes):
    """Independent exact A/B shape guard for the already-loaded runner model."""
    if content_hash({key: value for key, value in choice.items() if key != "choice_sha256"}) != choice.get("choice_sha256"):
        raise ValueError("Targeting choice digest mismatch")
    expected = {(matrix["module"], side): matrix[side + "_shape"] for matrix in choice["nominal_integer_matrices"] for side in ("A", "B")}
    observed = {}
    for name, shape in trainable_shapes.items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.([^.]+)\.weight", name)
        if match is None or match[3] != choice["adapter_name"]:
            raise ValueError("Unexpected trainable non-target/base/bias or extra adapter parameter")
        names = [target for target in choice["expected_targets"] if match[1] == target or match[1].endswith("." + target)]
        if len(names) != 1 or (names[0], match[2]) in observed:
            raise ValueError("Ambiguous or repeated realized target")
        key = names[0], match[2]
        if list(shape) != expected[key]:
            raise ValueError("Actual LoRA matrix shape/rank differs from the matched choice")
        observed[key] = list(shape)
    if set(observed) != set(expected):
        raise ValueError("Actual target coverage differs from matched source geometry")
    count = sum(math.prod(shape) for shape in observed.values())
    if count != choice["exact_trainable_parameters"]:
        raise ValueError("Actual trainable parameter count differs from the matched choice")
    return {"passed": True, "exact_trainable_parameters": count, "n_target_modules": len(expected) // 2,
            "choice_sha256": choice["choice_sha256"], "trainable_shapes_sha256": content_hash(trainable_shapes)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    geometry_parser = sub.add_parser("geometry")
    geometry_parser.add_argument("--registry", type=Path, default=root / "configs/revision/w6_sources.json")
    geometry_parser.add_argument("--model", required=True)
    geometry_parser.add_argument("--out", type=Path, required=True)
    match_parser = sub.add_parser("match")
    match_parser.add_argument("--request", type=Path, required=True)
    match_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "geometry":
        registry = json.loads(args.registry.read_text())
        result = geometry_from_checkpoint(registry["models"][args.model], root=root)
    else:
        request = json.loads(args.request.read_text())
        geometries = {}
        for key, artifact in request["geometries"].items():
            path = _path(artifact["path"], args.request.parent)
            if file_hash(path) != artifact["sha256"]:
                raise ValueError("Registered geometry file hash changed")
            geometries[key] = json.loads(path.read_text())
        result = plan_matching(request["policy"], geometries)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.out), "sha256": file_hash(args.out),
                      "status": result.get("status", "SOURCE_GEOMETRY_VERIFIED"),
                      "scientific_evidence": False}))
    return 2 if result.get("status") == "INFEASIBLE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
