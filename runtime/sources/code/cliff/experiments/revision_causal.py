"""Training-free, explicitly scoped causal intervention infrastructure.

A successful technical audit is not evidence of localization. Query-state
transplants may transport a previously computed answer; short/long transplants
retain position/content confounds even when query token identities align.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
import random

from ..revision_records import RunWriter, canonical_json, content_hash, file_hash


def transformer_blocks(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers",
                 "language_model.model.layers", "model.language_model.layers"):
        current = model
        try:
            for part in path.split("."):
                current = getattr(current, part)
            if len(current):
                return current
        except (AttributeError, TypeError):
            continue
    raise ValueError("unsupported transformer block layout")


def component_module(block, component):
    if component in {"residual", "block"}:
        return block
    names = {"attn": ("self_attn", "attn", "attention", "self_attention"),
             "mlp": ("mlp", "feed_forward", "ffn")}
    if component not in names:
        raise ValueError("component must be residual, attn or mlp")
    found = [getattr(block, name) for name in names[component] if hasattr(block, name)]
    if len(found) != 1:
        raise ValueError(f"cannot uniquely identify {component} module")
    return found[0]


def prompt_token_roles(backend, task_id, demos, query_text):
    """Assign token roles by offsets in the exact canonical encoded prompt.

    Tokens overlapping a character span belong to that span; boundary-sharing
    tokens are exposed in ``overlapping_roles``. Unsupported offset mappings
    fail instead of claiming exact role alignment from approximate lengths.
    """
    from ..prompting import render_prompt
    from ..data import normalized_text_hash
    prompt = render_prompt(task_id, demos, query_text)
    wrapped = backend._wrap(prompt)
    offset = wrapped.find(prompt)
    if offset < 0 or wrapped.find(prompt, offset + 1) != -1:
        raise ValueError("canonical wrapper must contain the user prompt exactly once")
    tok = backend.tokenizer()
    try:
        enc = tok(wrapped, add_special_tokens=getattr(backend, "template", "native") == "raw",
                  truncation=False, return_offsets_mapping=True)
    except (NotImplementedError, TypeError) as error:
        raise ValueError("exact token roles require a tokenizer with offset mappings") from error
    ids, offsets = list(enc["input_ids"]), list(enc["offset_mapping"])
    if ids != backend.encode_prompt(prompt) or len(ids) != len(offsets):
        raise ValueError("offset tokenization differs from canonical encode_prompt")
    if not ids or len(ids) > backend.max_seq_len:
        raise ValueError("empty or over-budget canonical prompt")

    def positions(start, end):
        return [i for i, (a, b) in enumerate(offsets) if b > a and b > offset + start and a < offset + end]

    query_marker = f"Text: {query_text}\nLabel:"
    query_start = prompt.rfind(query_marker)
    if query_start < 0 or query_start + len(query_marker) != len(prompt):
        raise ValueError("cannot locate the exact final query")
    query_span = positions(query_start + len("Text: "), query_start + len("Text: ") + len(query_text))
    cursor, demonstration_spans, label_spans, input_spans = 0, [], [], []
    for demo in demos:
        segment = f"Text: {demo.text}\nLabel: {demo.label}\n\n"
        start = prompt.find(segment, cursor)
        if start < 0 or start >= query_start:
            raise ValueError("demonstration rendering does not match the canonical prompt")
        text_start = start + len("Text: ")
        label_start = text_start + len(demo.text) + len("\nLabel: ")
        demonstration_spans.append(positions(start, start + len(segment)))
        input_spans.append(positions(text_start, text_start + len(demo.text)))
        label_spans.append(positions(label_start, label_start + len(demo.label)))
        cursor = start + len(segment)
    if not query_span or any(not span for span in label_spans):
        raise ValueError("empty token span for a requested query or demonstration label")
    overlap = sorted(set(sum(input_spans, [])) & set(sum(label_spans, [])))
    return {"prompt": prompt, "input_token_ids": ids, "input_token_ids_hash": content_hash(ids),
            "query_positions": query_span, "query_text_hash": normalized_text_hash(query_text),
            "readout_positions": [len(ids) - 1], "demonstration_positions": demonstration_spans,
            "demonstration_input_positions": input_spans, "demonstration_label_positions": label_spans,
            "overlapping_roles": overlap, "span_policy": "canonical_token_offsets_overlap_v1",
            "template": getattr(backend, "template", "unknown"),
            "template_hash": backend._template_hash() if hasattr(backend, "_template_hash") else None}


def align_query_positions(donor_roles, recipient_roles, policy="query_span"):
    donor_ids, recipient_ids = donor_roles["input_token_ids"], recipient_roles["input_token_ids"]
    if donor_roles["query_text_hash"] != recipient_roles["query_text_hash"]:
        raise ValueError("query alignment requires the same query content")
    if donor_roles.get("template_hash") != recipient_roles.get("template_hash"):
        raise ValueError("donor/recipient templates differ; use an explicitly common format")
    if policy == "same_prompt":
        if donor_ids != recipient_ids:
            raise ValueError("same_prompt alignment requires token-identical prompts")
        donor = donor_roles["query_positions"]
        recipient = recipient_roles["query_positions"]
    elif policy == "query_span":
        donor, recipient = donor_roles["query_positions"], recipient_roles["query_positions"]
        if [donor_ids[i] for i in donor] != [recipient_ids[i] for i in recipient]:
            raise ValueError("query spans have incompatible tokenization; approximate alignment is forbidden")
    elif policy == "readout":
        donor, recipient = donor_roles["readout_positions"], recipient_roles["readout_positions"]
        if [donor_ids[i] for i in donor] != [recipient_ids[i] for i in recipient]:
            raise ValueError("readout boundary tokens differ")
    else:
        raise ValueError("alignment must be same_prompt, query_span or readout")
    return list(donor), list(recipient), {"policy": policy, "positions_equal": donor == recipient,
        "prompt_lengths_equal": len(donor_ids) == len(recipient_ids),
        "short_long_position_content_confound": donor_ids != recipient_ids,
        "interpretation": "state transport; does not identify where the state was computed"}


def _hidden(output):
    tensor = output[0] if isinstance(output, tuple) else output
    if not hasattr(tensor, "ndim") or tensor.ndim != 3:
        raise ValueError("expected a batch, sequence, hidden component output")
    return tensor


def _replace_hidden(output, tensor):
    return (tensor,) + output[1:] if isinstance(output, tuple) else tensor


class _PrefillGuard(AbstractContextManager):
    """Permit exactly one declared full forward, never subsequent decode steps."""
    def __init__(self, model, expected_input_ids):
        self.model, self.expected = model, list(expected_input_ids)
        if not self.expected:
            raise ValueError("expected prefill IDs must be nonempty")
        self.handles, self.active, self.consumed, self.calls = [], False, False, {}

    def _before(self, module, args, kwargs):
        inputs = kwargs.get("input_ids", args[0] if args else None)
        past = kwargs.get("past_key_values")
        cached = past is not None and (past.get_seq_length() > 0 if hasattr(past, "get_seq_length") else bool(past))
        self.active = bool(not self.consumed and not cached and inputs is not None
                           and inputs.ndim == 2 and inputs.shape[0] == 1
                           and inputs[0].detach().cpu().tolist() == self.expected)
        if self.active:
            self.consumed = True

    def _after(self, module, args, output):
        self.active = False

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self._before, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self._after, always_call=True))
        try:
            self._install()
        except BaseException:
            self._cleanup()
            raise
        return self

    def _cleanup(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False

    def __exit__(self, kind, error, traceback):
        self._cleanup()
        if error is None and (not self.consumed or any(count != 1 for count in self.calls.values())):
            raise RuntimeError("declared prefill intervention was not applied exactly once per target")
        return False


class PrefillPatch(_PrefillGuard):
    def __init__(self, model, states, positions, expected_input_ids, *, component="residual", operation="replace"):
        super().__init__(model, expected_input_ids)
        if operation not in {"replace", "add"} or not states:
            raise ValueError("nonempty states and replace/add operation required")
        self.states, self.positions, self.component, self.operation = states, list(positions), component, operation
        if not self.positions or len(set(self.positions)) != len(self.positions):
            raise ValueError("patch positions must be nonempty and unique")
        if min(self.positions) < 0 or max(self.positions) >= len(self.expected):
            raise ValueError("patch positions exceed the declared prefill")

    def _install(self):
        blocks = transformer_blocks(self.model)
        for layer, state in self.states.items():
            if not 0 <= layer < len(blocks):
                raise ValueError("patch layer outside model")
            self.calls[layer] = 0

            def hook(module, args, output, layer=layer, state=state):
                if not self.active:
                    return output
                h = _hidden(output)
                if h.shape[1] != len(self.expected):
                    raise ValueError("hooked output is not the declared prefill")
                donor = state.to(device=h.device, dtype=h.dtype)
                if tuple(donor.shape) != (1, len(self.positions), h.shape[-1]):
                    raise ValueError("donor shape does not match recipient positions and hidden width")
                replacement = h.clone()
                if self.operation == "replace":
                    replacement[:, self.positions, :] = donor
                else:
                    replacement[:, self.positions, :] += donor
                self.calls[layer] += 1
                return _replace_hidden(output, replacement)

            self.handles.append(component_module(blocks[layer], self.component).register_forward_hook(hook))


def _model_identity(backend):
    """Cache a byte identity only while parameter objects/versions stay unchanged."""
    import torch
    model = backend.torch_model()
    signature = tuple((name, id(p), p._version, tuple(p.shape), str(p.dtype)) for name, p in model.named_parameters())
    cached = getattr(backend, "_causal_model_identity", None)
    if cached is not None and cached[0] == signature:
        return cached[1]
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(canonical_json([name, list(tensor.shape), str(tensor.dtype)]).encode())
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(raw))
    identity = {"model_id": getattr(backend, "model_id", None),
                "revision": getattr(backend, "revision", None), "state_sha256": digest.hexdigest()}
    backend._causal_model_identity = (signature, identity)
    return identity


def capture_states(backend, prompt, layers, positions, *, component="residual", cache_dir=None):
    import torch
    model, ids = backend.torch_model(), backend.encode_prompt(prompt)
    layers, positions = list(layers), list(positions)
    if not ids or len(ids) > backend.max_seq_len or not layers or len(set(layers)) != len(layers):
        raise ValueError("nonempty unique layers and a valid untruncated prompt required")
    if not positions or min(positions) < 0 or max(positions) >= len(ids):
        raise ValueError("capture positions outside prompt")
    metadata = {"model": _model_identity(backend), "input_token_ids": ids, "layers": layers,
                "positions": positions, "component": component,
                "layer_convention": "zero_based_transformer_block",
                "output_site": "block_output" if component in {"residual", "block"} else component + "_module_output"}
    key = content_hash(metadata)
    pt = Path(cache_dir) / f"{key}.pt" if cache_dir else None
    manifest = Path(cache_dir) / f"{key}.json" if cache_dir else None
    if pt and pt.exists():
        if not manifest.exists():
            raise RuntimeError("incomplete donor-state cache")
        saved = json.loads(manifest.read_text())
        if saved["metadata"] != metadata or saved["states_sha256"] != file_hash(pt):
            raise RuntimeError("donor-state cache provenance/hash mismatch")
        return torch.load(pt, map_location="cpu", weights_only=True), {**metadata, "cache_key": key, "cache_hit": True}
    blocks, states, handles = transformer_blocks(model), {}, []
    for layer in layers:
        if not 0 <= layer < len(blocks):
            raise ValueError("capture layer outside model")
    try:
        for layer in layers:
            def hook(module, args, output, layer=layer):
                states[layer] = _hidden(output)[:, positions, :].detach().cpu().clone()
            handles.append(component_module(blocks[layer], component).register_forward_hook(hook))
        model.eval()
        inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(states) != set(layers):
        raise RuntimeError("not every requested donor component executed")
    if any(not torch.isfinite(state).all() for state in states.values()):
        raise FloatingPointError("nonfinite donor states cannot be cached or transplanted")
    if pt:
        pt.parent.mkdir(parents=True, exist_ok=True)
        with pt.open("xb") as stream:
            torch.save(states, stream)
        with manifest.open("x", encoding="utf-8") as stream:
            json.dump({"metadata": metadata, "states_sha256": file_hash(pt)}, stream, indent=2)
    return states, {**metadata, "cache_key": key, "cache_hit": False}


def score_labels_with_intervention(backend, prompt, labels, factory, *, required_prompt_positions=()):
    """Use one common conditioning prefix and one fresh hook per candidate.

    Joint tokenizations can diverge at different prompt/label boundaries. All
    candidates must re-score the suffix after the earliest boundary; otherwise
    their likelihood denominators differ and the label ranking can reverse.
    """
    import torch
    model, scores = backend.torch_model(), {}
    if hasattr(backend, "label_encodings"):
        encodings = backend.label_encodings(prompt, labels)
    else:
        encodings = {label: backend.encode_completion(prompt, getattr(backend, "label_prefix", " ") + label) for label in labels}
        if not encodings:
            raise ValueError("nonempty candidate labels required")
        common_start = min(start for _, start in encodings.values())
        encodings = {label: (ids, common_start) for label, (ids, _) in encodings.items()}
    if set(encodings) != set(labels) or len({start for _, start in encodings.values()}) != 1:
        raise ValueError("all label encodings must use the same declared conditioning boundary")
    prefixes = {tuple(ids[:start]) for ids, start in encodings.values()}
    if len(prefixes) != 1:
        raise ValueError("candidate labels do not share their declared conditioning prefix")
    for label in labels:
        ids, start = encodings[label]
        if start < 1 or start >= len(ids) or len(ids) > backend.max_seq_len:
            raise ValueError("invalid/over-budget complete-label continuation")
        if any(position >= start for position in required_prompt_positions):
            raise ValueError("label boundary retokenization overlaps a patched prompt position")
        inputs = torch.tensor([ids], device=model.get_input_embeddings().weight.device)
        with factory(ids), torch.inference_mode():
            logits = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False).logits
            logprobs = torch.log_softmax(logits[:, :-1].float(), -1)
            token_ll = logprobs.gather(-1, inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
            scores[label] = float(token_ll[0, start - 1:].sum())
            if not math.isfinite(scores[label]):
                raise FloatingPointError("nonfinite complete-label likelihood under intervention")
    return scores


def _text_config(model):
    cfg = getattr(model.config, "text_config", model.config)
    value = cfg.to_dict()
    ignored = {"_name_or_path", "_commit_hash", "transformers_version", "torch_dtype", "dtype",
               "_attn_implementation_autoset", "_attn_implementation_internal"}
    # JSON normalization makes integer config-map keys stable after a cache
    # manifest round-trip (e.g. id2label), without relaxing actual comparisons.
    return json.loads(canonical_json({key: val for key, val in value.items() if key not in ignored}))


def checkpoint_compatibility(recipient, donor):
    """Strict architecture/position/tokenizer/tensor compatibility, not alignment."""
    rmodel, dmodel = recipient.torch_model(), donor.torch_model()
    rcfg, dcfg = _text_config(rmodel), _text_config(dmodel)
    differing = sorted(key for key in set(rcfg) | set(dcfg) if rcfg.get(key) != dcfg.get(key))
    if differing:
        raise ValueError(f"checkpoint configurations differ: {differing}")
    rclasses = {name: type(module).__module__ + "." + type(module).__qualname__ for name, module in rmodel.named_modules()}
    dclasses = {name: type(module).__module__ + "." + type(module).__qualname__ for name, module in dmodel.named_modules()}
    if rclasses != dclasses:
        raise ValueError("checkpoint module layouts or attention implementation classes differ")
    rtok, dtok = recipient.tokenizer(), donor.tokenizer()
    rmap = {"vocab": rtok.get_vocab(), "specials": rtok.special_tokens_map}
    dmap = {"vocab": dtok.get_vocab(), "specials": dtok.special_tokens_map}
    if content_hash(rmap) != content_hash(dmap):
        raise ValueError("tokenizer vocabularies or special-token maps differ")
    rstate, dstate = rmodel.state_dict(), dmodel.state_dict()
    if set(rstate) != set(dstate) or any(rstate[key].shape != dstate[key].shape or
        rstate[key].dtype != dstate[key].dtype for key in rstate):
        raise ValueError("checkpoint tensor keys, shapes or dtypes differ")
    return {"configuration_sha256": content_hash(rcfg), "tokenizer_sha256": content_hash(rmap),
            "embedding_and_head": "retained from recipient", "coordinate_alignment": "assumed, not established"}


class BandSplice(AbstractContextManager):
    """Replace only selected blocks; original module objects are restored exactly."""
    def __init__(self, recipient, donor, layers):
        self.recipient, self.donor, self.layers = recipient, donor, list(layers)
        self.validation = checkpoint_compatibility(recipient, donor)
        self.original = {}
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("nonempty unique splice layers required")

    def __enter__(self):
        rblocks, dblocks = transformer_blocks(self.recipient.torch_model()), transformer_blocks(self.donor.torch_model())
        if min(self.layers) < 0 or max(self.layers) >= len(rblocks):
            raise ValueError("splice layer outside model")
        try:
            for layer in self.layers:
                self.original[layer] = rblocks[layer]
                reference = next(rblocks[layer].parameters())
                replacement = copy.deepcopy(dblocks[layer]).to(device=reference.device, dtype=reference.dtype)
                rblocks[layer] = replacement
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *args):
        rblocks = transformer_blocks(self.recipient.torch_model())
        for layer, module in self.original.items():
            rblocks[layer] = module
        self.original = {}
        return False


def weight_differences(base_model, descendant_model):
    """Both relative Frobenius distance and per-element RMS; no causal inference."""
    import torch
    base, descendant = base_model.state_dict(), descendant_model.state_dict()
    if set(base) != set(descendant):
        raise ValueError("weight-map tensor keys differ")
    rows, groups = [], {}
    from .intervention import _BLOCK, _component
    for name, reference in base.items():
        changed = descendant[name]
        if reference.shape != changed.shape:
            raise ValueError(f"weight-map shape differs: {name}")
        if not reference.is_floating_point():
            if not torch.equal(reference.cpu(), changed.cpu()):
                raise ValueError(f"non-floating checkpoint buffer differs: {name}")
            continue
        reference, changed = reference.detach().reshape(-1), changed.detach().reshape(-1)
        n, difference2, base2 = reference.numel(), 0.0, 0.0
        # Large embedding matrices must not create two full fp64 CPU copies.
        for start in range(0, n, 1024 * 1024):
            base_chunk = reference[start:start + 1024 * 1024].cpu().double()
            descendant_chunk = changed[start:start + 1024 * 1024].cpu().double()
            difference2 += float((descendant_chunk - base_chunk).square().sum())
            base2 += float(base_chunk.square().sum())
        block = _BLOCK.search(name)
        component = _component(name.rsplit(".", 1)[0]) or (
            "norm" if "norm" in name else "embedding_head" if "embed" in name or "lm_head" in name else "other")
        group = f"block_{block.group(1)}:{component}" if block else component
        row = {"tensor": name, "group": group, "n_elements": n, "difference_frobenius": math.sqrt(difference2),
               "base_frobenius": math.sqrt(base2), "difference_rms": math.sqrt(difference2 / n) if n else None,
               "base_rms": math.sqrt(base2 / n) if n else None,
               "relative_frobenius": math.sqrt(difference2 / base2) if base2 else None,
               "zero_base_norm": base2 == 0}
        rows.append(row)
        entry = groups.setdefault(group, {"difference2": 0.0, "base2": 0.0, "n_elements": 0})
        entry["difference2"] += difference2
        entry["base2"] += base2
        entry["n_elements"] += n
    for entry in groups.values():
        entry["relative_frobenius"] = math.sqrt(entry["difference2"] / entry["base2"]) if entry["base2"] else None
        entry["difference_rms"] = math.sqrt(entry["difference2"] / entry["n_elements"]) if entry["n_elements"] else None
    return {"tensors": rows, "groups": groups,
            "definitions": {"relative_frobenius": "norm(descendant-base)/norm(base); undefined at zero base norm",
                            "difference_rms": "norm(descendant-base)/sqrt(number of tensor elements)"},
            "interpretation": "descriptive parameter distance, not causal localization"}


class AttentionEdgeMask(_PrefillGuard):
    """Block declared causal edges in eager attention and inspect probabilities."""
    def __init__(self, model, layers, edges, expected_input_ids, *, tolerance=1e-7):
        super().__init__(model, expected_input_ids)
        self.layers, self.edges, self.tolerance, self.validation = list(layers), sorted(set(map(tuple, edges))), tolerance, {}
        if not self.layers or not self.edges or len(set(self.layers)) != len(self.layers):
            raise ValueError("nonempty edges and unique layers required")
        n = len(self.expected)
        if any(not (0 <= key <= query < n) for query, key in self.edges):
            raise ValueError("masked edges must be valid causal query,key positions")

    def _install(self):
        import torch
        cfg = getattr(self.model.config, "text_config", self.model.config)
        if getattr(cfg, "_attn_implementation", None) != "eager":
            raise ValueError("selective edge masking requires an explicitly eager attention implementation")
        blocks = transformer_blocks(self.model)
        for layer in self.layers:
            if not 0 <= layer < len(blocks):
                raise ValueError("attention mask layer outside model")
            module = component_module(blocks[layer], "attn")
            if any(tag in type(module).__name__.lower() for tag in ("flash", "sdpa")):
                raise ValueError("attention module class is not an eager implementation")
            signature = inspect.signature(module.forward)
            if "attention_mask" not in signature.parameters:
                raise ValueError("attention module exposes no supported additive mask interface")
            self.calls[layer] = 0

            def before(module, args, kwargs, signature=signature):
                if not self.active:
                    return args, kwargs
                bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
                hidden = bound.arguments.get("hidden_states")
                if hidden is None or hidden.ndim != 3 or hidden.shape[1] != len(self.expected):
                    raise ValueError("unsupported eager attention hidden-state signature")
                mask = bound.arguments.get("attention_mask")
                n = hidden.shape[1]
                if mask is None:
                    mask = torch.full((1, 1, n, n), -torch.inf, device=hidden.device, dtype=hidden.dtype).triu(1)
                if not mask.is_floating_point() or mask.ndim != 4 or mask.shape[-2:] != (n, n):
                    raise ValueError("unsupported eager attention mask shape")
                mask = mask.clone()
                for query, key in self.edges:
                    mask[:, :, query, key] = -torch.inf
                for query in {query for query, _ in self.edges}:
                    if not (torch.isfinite(mask[:, :, query, :]) & (mask[:, :, query, :] > torch.finfo(mask.dtype).min)).any():
                        raise ValueError("intervention would remove every available attention key")
                bound.arguments["attention_mask"] = mask
                if "output_attentions" in signature.parameters:
                    bound.arguments["output_attentions"] = True
                return bound.args, bound.kwargs

            def after(module, args, output, layer=layer):
                if not self.active:
                    return output
                attention = output[1] if isinstance(output, tuple) and len(output) > 1 else None
                if attention is None or attention.ndim != 4:
                    raise RuntimeError("eager attention did not expose actual probabilities; rejecting intervention")
                maximum = max(float(attention[:, :, query, key].abs().max()) for query, key in self.edges)
                if maximum > self.tolerance or not torch.isfinite(attention).all():
                    raise RuntimeError("requested attention edges were not actually masked")
                normalization_tolerance = max(1e-5, 2 * torch.finfo(attention.dtype).eps)
                normalization_error = float((attention.sum(-1).float() - 1).abs().max())
                if normalization_error > normalization_tolerance:
                    raise RuntimeError("masked attention probabilities are not normalized")
                self.validation[layer] = {"max_masked_probability": maximum, "n_edges": len(self.edges),
                    "max_normalization_error": normalization_error,
                    "normalization_tolerance": normalization_tolerance, "probability_dtype": str(attention.dtype)}
                self.calls[layer] += 1
                return output

            self.handles.append(module.register_forward_pre_hook(before, with_kwargs=True))
            self.handles.append(module.register_forward_hook(after))


def edge_set(roles, condition):
    """Keep query-to-label and input-to-label isolation as distinct operations."""
    if roles["overlapping_roles"]:
        raise ValueError("label/input roles share boundary tokens; exact edge isolation is ambiguous")
    if condition == "query_to_demo_labels":
        queries, keys = roles["query_positions"], sum(roles["demonstration_label_positions"], [])
        return [(q, k) for q in queries for k in keys if k <= q]
    if condition == "query_to_all_demos":
        queries, keys = roles["query_positions"], sum(roles["demonstration_positions"], [])
        return [(q, k) for q in queries for k in keys if k <= q]
    if condition == "demo_input_to_own_label":
        return [(label, source) for sources, labels in zip(roles["demonstration_input_positions"],
                   roles["demonstration_label_positions"]) for label in labels for source in sources if source <= label]
    raise ValueError("unknown edge condition")


def matched_count_edges(edges, n_tokens, seed):
    """Counterfactual edges preserve each query row's masked-edge count."""
    rng, result = random.Random(seed), []
    rows = {}
    for query, key in set(map(tuple, edges)):
        if not 0 <= key <= query < n_tokens:
            raise ValueError("control requires valid causal edges")
        rows.setdefault(query, set()).add(key)
    for query, target_keys in sorted(rows.items()):
        alternatives = [key for key in range(query + 1) if key not in target_keys]
        if len(alternatives) < len(target_keys):
            raise ValueError("insufficient distinct non-target keys for a matched-count edge control")
        result.extend((query, key) for key in rng.sample(alternatives, len(target_keys)))
    return result


def checkpoint_descriptor(backend):
    model, tok = backend.torch_model(), backend.tokenizer()
    return {"config": _text_config(model),
            "tokenizer": {"vocab": tok.get_vocab(), "specials": tok.special_tokens_map},
            "tensors": {name: [list(value.shape), str(value.dtype)] for name, value in model.state_dict().items()},
            "module_classes": {name: type(module).__module__ + "." + type(module).__qualname__
                               for name, module in model.named_modules()},
            "model_identity": _model_identity(backend)}


def validate_descriptor(recipient, donor_descriptor):
    current = checkpoint_descriptor(recipient)
    for key in ("config", "tokenizer", "tensors", "module_classes"):
        if current[key] != donor_descriptor[key]:
            raise ValueError(f"cached donor/recipient {key} compatibility failed")
    return {"compatible": True, "representation_coordinates": "assumed without learned alignment"}


def matched_norm_control(states, kind, seed):
    import torch
    if kind not in {"noise", "permuted"}:
        raise ValueError("control must be noise or permuted")
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for layer, state in states.items():
        value = state.float().cpu()
        candidate = (torch.randn(value.shape, generator=generator) if kind == "noise" else
                     value[..., torch.randperm(value.shape[-1], generator=generator)])
        norm = value.norm(dim=-1, keepdim=True)
        candidate = candidate * norm / candidate.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        result[layer] = candidate.to(state.dtype)
    return result


def norm_match_states(states, reference):
    """Match a wrong-query donor to the intended donor's norm at every position."""
    import torch
    if set(states) != set(reference):
        raise ValueError("norm matching requires the same layers")
    matched = {}
    for layer, state in states.items():
        if state.shape != reference[layer].shape:
            raise ValueError("norm matching requires aligned shapes")
        norm, target = state.float().norm(dim=-1, keepdim=True), reference[layer].float().norm(dim=-1, keepdim=True)
        if torch.any((norm == 0) & (target != 0)):
            raise ValueError("a zero donor vector cannot match a nonzero reference norm")
        matched[layer] = (state.float() * target / norm.clamp_min(1e-12)).to(state.dtype)
    return matched


def _backend(config):
    from ..backends import get_backend, require_weight_access
    options = {key: value for key, value in config.items() if key != "id"}
    if "attention_implementation" in options:
        attention = options.pop("attention_implementation")
        if "attn_implementation" in options and options["attn_implementation"] != attention:
            raise ValueError("conflicting attention implementation settings")
        options["attn_implementation"] = attention
    options.setdefault("overflow_policy", "reject")
    options.setdefault("local_files_only", True)
    return require_weight_access(get_backend("hf_local", config["id"], **options), "causal interventions")


def _dataset(config):
    from ..data import load_task
    cfg = config.get("dataset", {})
    return load_task(config["task"], n_train=cfg.get("n_train", 512), n_test=cfg.get("n_test", 200),
        shard_seed=cfg.get("query_split_seed", 0), revision=cfg.get("revision"),
        train_split=cfg.get("train_split"), test_split=cfg.get("query_split"),
        train_indices=cfg.get("train_indices"), test_indices=cfg.get("query_indices"))


def _nested_demos(pool, k_donor, k_recipient, seed):
    from ..data import exact_balanced_sample
    full = exact_balanced_sample(pool, max(k_donor, k_recipient), seed)
    def subset(k):
        selected = {example.example_id for example in exact_balanced_sample(full, k, seed)}
        return [example for example in full if example.example_id in selected]
    return subset(k_donor), subset(k_recipient)


def _cached_state_path(entry, manifest_path):
    return (Path(manifest_path).parent / entry["state_file"] if entry.get("state_file") else
            Path(entry["state_path"]))


def _read_json(path):
    return json.loads(Path(path).read_text())


def _w3_order_source(manifest_path):
    """Read exact completed W3 development artifacts, using one scored readout."""
    path = Path(manifest_path).resolve()
    manifest = _read_json(path)
    cfg = manifest.get("configuration", {})
    if (manifest.get("status") != "complete" or manifest.get("scientific_evidence") is not True
            or manifest.get("experiment") != "w3_order_calibration" or cfg.get("query_role") != "development"):
        raise ValueError("order selection requires a completed scientific W3 development run")
    files = {name: path.parent / filename for name, filename in (
        ("design", "design.json"), ("predictions", "predictions.jsonl"), ("validation", "validation.json"))}
    for name in ("predictions", "validation"):
        if not files[name].is_file() or file_hash(files[name]) != manifest.get(name + "_sha256"):
            raise ValueError("W3 source artifact hash mismatch: " + name)
    if _read_json(files["validation"]).get("passed") is not True:
        raise ValueError("W3 source technical validation did not pass")
    design = _read_json(files["design"])
    queries = design.get("query_examples", [])
    if (design.get("query_role") != "development" or not queries
            or len({q["id"] for q in queries}) != len(queries)
            or len({q["hash"] for q in queries}) != len(queries)
            or design.get("design_sha256") != content_hash(design.get("contexts", []))):
        raise ValueError("W3 design requires exact unique development query identities and contexts")
    import re
    model, dataset = cfg.get("model", {}), cfg.get("dataset", {})
    if (not all(re.fullmatch(r"[0-9a-fA-F]{40}", value or "") for value in
                (model.get("revision"), model.get("tokenizer_revision") or model.get("revision"), dataset.get("revision")))
            or dataset.get("query_split") is None or dataset.get("train_split") is None):
        raise ValueError("W3 selection requires pinned model/tokenizer/dataset revisions and source splits")
    from ..data import TASKS
    if dataset["query_split"] == TASKS[cfg["task"]]["split_test"]:
        raise ValueError("W3 development order selection cannot use official final queries")
    return path, manifest, design, files


def freeze_order_selection(source_manifest, output_path, *, ks, demo_seeds, min_accuracy_gap=.02):
    """Freeze aggregate healthy/failing orders on development queries only.

    All candidates use the uncalibrated complete-label likelihood readout. The
    same selected order applies to every later query; no final-query result is
    accepted by this selector. Selection is exploratory and requires independent
    confirmation, even when its observed development contrast is large.
    """
    if (not ks or not demo_seeds or len(set(ks)) != len(ks) or len(set(demo_seeds)) != len(demo_seeds)
            or any(type(k) is not int or k < 1 for k in ks)
            or any(type(seed) is not int for seed in demo_seeds) or not 0 < min_accuracy_gap <= 1):
        raise ValueError("positive unique Ks, unique draws and a positive prespecified selection gap are required")
    path, source, design, files = _w3_order_source(source_manifest)
    queries = {q["id"]: q for q in design["query_examples"]}
    from ..data import TASKS
    labels = TASKS[source["configuration"]["task"]]["label_names"]
    if design.get("label_tie_order") != labels:
        raise ValueError("W3 source requires the declared dataset label tie order")
    rows = {}
    for line in files["predictions"].read_text().splitlines():
        row = json.loads(line)
        if row.get("run_id") != source["run_id"] or row.get("query_role") != "development":
            raise ValueError("W3 source predictions have different run/query roles")
        rows.setdefault(row["context_id"], []).append(row)
    contexts = [c for c in design["contexts"] if c.get("comparison_family") == "order"]
    selected = []
    for k in ks:
        for draw in demo_seeds:
            candidates = []
            for context in contexts:
                if context.get("requested_k") != k or context.get("draw_id") != draw:
                    continue
                order = context.get("ordered_demonstrations", [])
                values = rows.get(context["context_id"], [])
                if (len(order) != k or context.get("realized_k") != k
                        or len({d["id"] for d in order}) != k or len({d["hash"] for d in order}) != k
                        or len(values) != len(queries) or {r["query_id"] for r in values} != set(queries)):
                    raise ValueError("W3 order candidate has incomplete counts, duplicates or query coverage")
                for row in values:
                    scores = row.get("label_scores", {})
                    if (row.get("query_hash") != queries[row["query_id"]]["hash"]
                            or row.get("measurement_status") != "measured" or row.get("rejected")
                            or row.get("truncated") or row.get("requested_k") != k or row.get("realized_k") != k
                            or row.get("draw_id") != draw or row.get("demonstration_ids") != [d["id"] for d in order]
                            or row.get("demonstration_hashes") != [d["hash"] for d in order]
                            or set(scores) != set(labels) or any(type(v) not in (float, int) or not math.isfinite(v) for v in scores.values())
                            or row.get("parse_status") != "restricted_complete_label"
                            or row.get("prediction") != max(labels, key=scores.get)
                            or row.get("gold_label") != queries[row["query_id"]].get("label")
                            or type(row.get("correct")) is not bool
                            or row["correct"] != (row["prediction"] == row.get("gold_label"))):
                        raise ValueError("W3 candidate requires valid fixed complete-label scores and exact query/order identities")
                multiset = sorted((d["id"], d["hash"], d["label"]) for d in order)
                if context.get("demonstration_multiset_id") != content_hash(multiset):
                    raise ValueError("W3 demonstration multiset fingerprint differs from its exact identities")
                candidates.append((sum(r["correct"] for r in values) / len(values), context, multiset))
            pairs = [(good[0] - bad[0], good, bad) for good in candidates for bad in candidates
                     if good[2] == bad[2] and good[1]["demonstration_ids"] != bad[1]["demonstration_ids"]]
            pairs.sort(key=lambda p: (-p[0], -p[1][0], p[1][1]["context_id"], p[2][1]["context_id"]))
            if not pairs or pairs[0][0] < min_accuracy_gap:
                raise ValueError(f"no development-verified same-multiset order contrast for K={k}, draw={draw}")
            gap, good, bad = pairs[0]
            selected.append({"k": k, "draw_id": draw, "healthy_context_id": good[1]["context_id"],
                "failing_context_id": bad[1]["context_id"], "healthy_order": good[1]["ordered_demonstrations"],
                "failing_order": bad[1]["ordered_demonstrations"],
                "demonstration_multiset_id": good[1]["demonstration_multiset_id"],
                "healthy_development_accuracy": good[0], "failing_development_accuracy": bad[0],
                "development_accuracy_gap": gap})
    cfg = source["configuration"]
    selection = {"schema_version": 1, "frozen": True, "query_role": "development",
        "selection_rule": "max_minus_min_complete_label_accuracy_same_multiset_v1",
        "readout": "complete_label_likelihood", "min_accuracy_gap": min_accuracy_gap,
        "source_run_id": source["run_id"], "source_manifest": str(path), "source_manifest_sha256": file_hash(path),
        "source_design_sha256": file_hash(files["design"]), "source_predictions_sha256": file_hash(files["predictions"]),
        "query_ids": [q["id"] for q in design["query_examples"]], "query_hashes": [q["hash"] for q in design["query_examples"]],
        "model": cfg["model"], "dataset": cfg["dataset"], "task": cfg["task"],
        "backend_provenance": source.get("backend_provenance", {}), "cells": selected,
        "inference_status": "development-selected order contrast; independent query confirmation required"}
    if output_path is not None:
        with Path(output_path).open("x", encoding="utf-8") as stream:
            json.dump(selection, stream, indent=2)
    return selection


def read_order_selection(path):
    """Verify a frozen selection by recomputing its exact development rule."""
    selection = _read_json(path)
    if (selection.get("frozen") is not True or selection.get("query_role") != "development"
            or selection.get("readout") != "complete_label_likelihood"
            or selection.get("selection_rule") != "max_minus_min_complete_label_accuracy_same_multiset_v1"):
        raise ValueError("frozen same-K order selection with a fixed scored readout is required")
    source_path = Path(selection.get("source_manifest", "MISSING"))
    if not source_path.is_absolute():
        source_path = Path(path).parent / source_path
    if not source_path.is_file() or file_hash(source_path) != selection.get("source_manifest_sha256"):
        raise ValueError("frozen order selection W3 source manifest hash changed")
    source, manifest, design, files = _w3_order_source(source_path)
    if (manifest["run_id"] != selection.get("source_run_id")
            or file_hash(files["design"]) != selection.get("source_design_sha256")
            or file_hash(files["predictions"]) != selection.get("source_predictions_sha256")
            or selection.get("query_ids") != [q["id"] for q in design["query_examples"]]
            or selection.get("query_hashes") != [q["hash"] for q in design["query_examples"]]
            or selection.get("model") != manifest["configuration"]["model"]
            or selection.get("dataset") != manifest["configuration"]["dataset"]
            or selection.get("backend_provenance") != manifest.get("backend_provenance", {})
            or selection.get("task") != manifest["configuration"]["task"]):
        raise ValueError("frozen order selection differs from its exact W3 source")
    contexts = {c["context_id"]: c for c in design["contexts"]}
    seen = set()
    for cell in selection.get("cells", []):
        key = (cell["k"], cell["draw_id"])
        if key in seen or type(key[0]) is not int or key[0] < 1 or type(key[1]) is not int:
            raise ValueError("frozen order cells must have unique positive K/draw identities")
        seen.add(key)
        for name in ("healthy", "failing"):
            ctx = contexts.get(cell.get(name + "_context_id"), {})
            order = cell.get(name + "_order", [])
            if (ctx.get("comparison_family") != "order" or (ctx.get("requested_k"), ctx.get("draw_id")) != key
                    or order != ctx.get("ordered_demonstrations") or len(order) != key[0]
                    or len({d["id"] for d in order}) != key[0] or len({d["hash"] for d in order}) != key[0]):
                raise ValueError("frozen order differs from its exact W3 context/counts")
        first = sorted((d["id"], d["hash"], d["label"]) for d in cell["healthy_order"])
        second = sorted((d["id"], d["hash"], d["label"]) for d in cell["failing_order"])
        if (first != second or content_hash(first) != cell.get("demonstration_multiset_id")
                or cell["healthy_order"] == cell["failing_order"]):
            raise ValueError("healthy/failing orders must use the exact same demonstration multiset")
        expected = freeze_order_selection(source, None, ks=[key[0]], demo_seeds=[key[1]],
                                         min_accuracy_gap=selection.get("min_accuracy_gap", .02))["cells"][0]
        if cell != expected:
            raise ValueError("frozen healthy/failing order selection differs from its development-only rule")
    if not seen:
        raise ValueError("frozen order selection contains no cells")
    return selection


def _ordered_demos(selection, k, seed, pool, direction):
    if direction not in {"healthy_to_failing", "failing_to_healthy"}:
        raise ValueError("same-K order direction must be healthy_to_failing or failing_to_healthy")
    cells = [cell for cell in selection["cells"] if (cell["k"], cell["draw_id"]) == (k, seed)]
    if len(cells) != 1:
        raise ValueError("frozen order selection is missing this exact K/draw")
    by_id = {example.example_id: example for example in pool}
    if len(by_id) != len(pool):
        raise ValueError("order demonstration source IDs are not unique")
    def resolve(name):
        order = cells[0][name + "_order"]
        out = []
        for row in order:
            example = by_id.get(row["id"])
            if example is None or example.text_hash != row["hash"] or example.label != row["label"]:
                raise ValueError("frozen order demonstration is missing, excluded, relabeled or changed")
            out.append(example)
        return out
    healthy, failing = resolve("healthy"), resolve("failing")
    return (healthy, failing) if direction == "healthy_to_failing" else (failing, healthy)


def run(config, *, recipient=None, donor=None, data=None):
    """Execute a registered technical/development condition with raw outcomes.

    ``cache_donor`` writes a portable donor manifest. A later ``patch`` condition
    can provide ``donor_cache_manifest`` and load only the recipient checkpoint.
    This avoids requiring two full checkpoints in accelerator memory.
    """
    from ..data import exclude_examples
    from ..revision_runtime import configure_cache
    configure_cache(config.get("cache_dir", Path(__file__).resolve().parents[2] / ".cache/revision"))
    operation = config.get("operation")
    allowed = {"patch", "edge_mask", "splice", "weight_difference", "cache_donor"}
    if operation not in allowed:
        raise ValueError(f"operation must be one of {sorted(allowed)}")
    if config.get("control") not in {None, "self", "noise", "permuted", "wrong_query"}:
        raise ValueError("unknown state control")
    if config.get("query_role", "technical_pilot") not in {"technical_pilot", "development", "final_test"}:
        raise ValueError("query_role must identify a technical pilot, development or final test")
    if config.get("alignment", "query_span") not in {"query_span", "readout", "same_prompt"}:
        raise ValueError("unknown token alignment policy")
    layers = config.get("layers", [])
    if operation != "weight_difference" and (not layers or len(set(layers)) != len(layers)
            or any(type(layer) is not int or layer < 0 for layer in layers)):
        raise ValueError("registered unique layers are required")
    if config.get("query_role", "technical_pilot") == "final_test" and not (
            config.get("selection", {}).get("frozen") and config.get("selection", {}).get("source_run_id")):
        raise ValueError("final queries require a frozen independent development selection")
    writer = RunWriter(config.get("output_root", "outputs/revision"), "causal_intervention", config,
                       run_id=config.get("run_id"), scientific_evidence=False)
    try:
        order_selection, order_fingerprint = None, None
        if config.get("order_selection_manifest"):
            if operation not in {"patch", "cache_donor"} or config.get("alignment", "query_span") != "query_span":
                raise ValueError("frozen same-K order donors require query-span patch/cache operations")
            order_fingerprint = file_hash(config["order_selection_manifest"])
            if config.get("order_selection_sha256") != order_fingerprint:
                raise ValueError("frozen order selection hash is missing or changed")
            order_selection = read_order_selection(config["order_selection_manifest"])
            if order_selection["task"] != config["task"]:
                raise ValueError("frozen order selection task differs")
        cache_manifest = None
        if config.get("donor_cache_manifest"):
            cache_path = Path(config["donor_cache_manifest"])
            cache_manifest = json.loads(cache_path.read_text())
            provenance = json.loads((cache_path.parent / "manifest.json").read_text())
            metrics_path = cache_path.parent / "metrics.json"
            if (provenance["status"] != "complete" or provenance["metrics_sha256"] != file_hash(metrics_path)
                    or json.loads(metrics_path.read_text()) != cache_manifest):
                raise ValueError("donor cache manifest does not match its completed immutable run")
            requested_donor = config.get("donor", {})
            cached_identity = cache_manifest["checkpoint_descriptor"]["model_identity"]
            if any(requested_donor.get(key) is not None and requested_donor[key] != cached_identity[field]
                   for key, field in (("id", "model_id"), ("revision", "revision"))):
                raise ValueError("cached donor checkpoint differs from the requested donor identity")
            if cache_manifest.get("configuration", {}).get("order_selection_sha256") != order_fingerprint:
                raise ValueError("cached donor frozen order selection differs")
        if operation != "cache_donor" and recipient is None:
            recipient = _backend(config["recipient"])
        if operation == "patch" and donor is None and not cache_manifest and (
                config.get("control") == "self" or config.get("donor") == config.get("recipient")):
            donor = recipient
        if operation in {"patch", "splice", "weight_difference", "cache_donor"} and not cache_manifest and donor is None:
            donor = _backend(config.get("donor", config.get("recipient")))
        for backend in (recipient, donor):
            if backend is not None:
                backend.torch_model().eval()
                if order_selection:
                    source_model = order_selection["model"]
                    if (getattr(backend, "model_id", None) != source_model["id"]
                            or getattr(backend, "revision", None) != source_model["revision"]
                            or getattr(backend, "template", None) != source_model.get("template", "native")
                            or getattr(backend, "label_prefix", " ") != source_model.get("label_prefix", " ")
                            or getattr(backend, "label_score_implementation", "full_forward")
                                != source_model.get("label_score_implementation", "full_forward")
                            or str(backend.torch_model().get_input_embeddings().weight.dtype).removeprefix("torch.")
                                != source_model.get("dtype", "bfloat16")):
                        raise ValueError("same-K order source requires the exact model/revision/template/precision/label-prefix/scoring-implementation")
                    provenance = order_selection.get("backend_provenance", {})
                    if (provenance.get("template_hash") and provenance["template_hash"] != backend._template_hash()
                            or provenance.get("label_scoring_policy") and provenance["label_scoring_policy"]
                                != getattr(backend, "label_scoring_policy", None)):
                        raise ValueError("same-K order source template/scoring policy differs")
        if operation == "weight_difference":
            compatibility = checkpoint_compatibility(recipient, donor)
            result = weight_differences(donor.torch_model(), recipient.torch_model())
            result.update({"base": _model_identity(donor), "descendant": _model_identity(recipient),
                           "compatibility": compatibility})
            writer.complete(result, {"passed": True, "descriptive_only": True})
            return writer.path
        if data is None:
            data = _dataset(config)
        if data.task_id != config["task"]:
            raise ValueError("loaded task differs from configured task")
        if (not data.test or len({q.text_hash for q in data.test}) != len(data.test)
                or len({q.example_id for q in data.test}) != len(data.test)):
            raise ValueError("unique nonempty final query set required")
        selection = config.get("selection", {})
        if order_selection:
            if config.get("dataset", {}).get("revision") != order_selection["dataset"]["revision"]:
                raise ValueError("same-K order source dataset revision differs")
            if (set(order_selection["query_ids"]) & {q.example_id for q in data.test}
                    or set(order_selection["query_hashes"]) & {q.text_hash for q in data.test}):
                raise ValueError("order-selection development and evaluation query IDs/hashes overlap")
        if set(selection.get("query_ids", [])) & {q.example_id for q in data.test} or \
                set(selection.get("query_hashes", [])) & {q.text_hash for q in data.test}:
            raise ValueError("development and final query sets overlap")
        if config.get("query_role") == "final_test" and not (selection.get("query_ids") and selection.get("query_hashes")):
            raise ValueError("final queries require development IDs and text hashes for independence checks")
        pool = exclude_examples(data.train, data.test)
        development_ids = set(selection.get("query_ids", []))
        development_hashes = set(selection.get("query_hashes", []))
        if order_selection:
            development_ids.update(order_selection["query_ids"])
            development_hashes.update(order_selection["query_hashes"])
        pool = [example for example in pool if example.example_id not in development_ids
                and example.text_hash not in development_hashes]
        ks = config.get("ks", [config.get("recipient_k", 16)])
        seeds = config.get("demo_seeds", [0])
        if (not ks or not seeds or len(set(ks)) != len(ks) or len(set(seeds)) != len(seeds)
                or any(type(k) is not int or k < 0 for k in ks) or any(type(seed) is not int for seed in seeds)):
            raise ValueError("unique nonempty K and demonstration seed lists required")
        component, alignment = config.get("component", "residual"), config.get("alignment", "query_span")
        intervention_condition = (operation + ":" + component + ":layers=" + ",".join(map(str, layers))
            + ":control=" + str(config.get("control") or config.get("edge_control") or "donor")
            + ":alignment=" + alignment + ":edge=" + str(config.get("edge_condition", "none")))
        donor_identity = (cache_manifest["checkpoint_descriptor"]["model_identity"] if cache_manifest else
                          {"model_id": getattr(donor, "model_id", None), "revision": getattr(donor, "revision", None)})
        intervention_condition += ":donor=" + str(donor_identity["model_id"]) + "@" + str(donor_identity["revision"])
        if order_selection:
            intervention_condition += ":order=" + config.get("order_direction", "healthy_to_failing") + ":selection=" + order_fingerprint
        def prediction_context(backend):
            return {"model_id": getattr(backend, "model_id", None),
                "model_revision": getattr(backend, "revision", None),
                "template": getattr(backend, "template", None),
                "template_hash": backend._template_hash() if hasattr(backend, "_template_hash") else None,
                "readout": "complete_label_likelihood",
                "label_scoring_policy": getattr(backend, "label_scoring_policy", "common_candidate_prefix_sum_LL"),
                "label_prefix": getattr(backend, "label_prefix", " "),
                "label_score_implementation": getattr(backend, "label_score_implementation", "full_forward"),
                "dtype": str(backend.torch_model().get_input_embeddings().weight.dtype)}
        cached_queries = {}
        if operation == "cache_donor":
            descriptor = checkpoint_descriptor(donor)
        elif cache_manifest:
            validate_descriptor(recipient, cache_manifest["checkpoint_descriptor"])
        elif operation in {"patch", "splice"}:
            checkpoint_compatibility(recipient, donor)
        recipient_identity = _model_identity(recipient) if operation != "cache_donor" else None
        summary = []
        for k in ks:
            donor_k = config.get("donor_k", k)
            # A K=0 recipient still varies across donor draws when donor K>0.
            for seed in (seeds[:1] if k == 0 and donor_k == 0 else seeds):
                if order_selection:
                    if donor_k != k:
                        raise ValueError("frozen same-K donor and recipient requested counts differ")
                    donor_demos, recipient_demos = _ordered_demos(order_selection, k, seed, pool,
                        config.get("order_direction", "healthy_to_failing"))
                else:
                    donor_demos, recipient_demos = _nested_demos(pool, donor_k, k, seed)
                for query in data.test:
                    key_fields = [query.example_id, query.text_hash, k, donor_k, seed]
                    key = content_hash(key_fields + [order_fingerprint] if order_fingerprint else key_fields)
                    if operation == "cache_donor":
                        roles = prompt_token_roles(donor, data.task_id, donor_demos, query.text)
                        positions = roles["readout_positions"] if alignment == "readout" else roles["query_positions"]
                        states, metadata = capture_states(donor, roles["prompt"], layers, positions,
                            component=component, cache_dir=writer.path / "donor_cache")
                        cached_queries[key] = {"roles": roles, "positions": positions, "metadata": metadata,
                            "state_file": "donor_cache/" + metadata["cache_key"] + ".pt",
                            "state_path": str((writer.path / "donor_cache" / (metadata["cache_key"] + ".pt")).resolve()),
                            "state_sha256": file_hash(writer.path / "donor_cache" / (metadata["cache_key"] + ".pt"))}
                        continue
                    roles = prompt_token_roles(recipient, data.task_id, recipient_demos, query.text)
                    order_donor_roles = (prompt_token_roles(recipient, data.task_id, donor_demos, query.text)
                                         if order_selection else None)
                    model = recipient.torch_model()
                    baseline = recipient.score_labels(roles["prompt"], data.label_names)
                    metadata, alignment_record, edge_audits = {}, {}, []
                    if operation == "patch":
                        if config.get("control") == "self":
                            donor_roles = roles
                            positions = roles["readout_positions"] if alignment == "readout" else roles["query_positions"]
                            states, metadata = capture_states(recipient, roles["prompt"], layers, positions,
                                component=component, cache_dir=writer.path / "self_cache")
                        elif cache_manifest:
                            import torch
                            entry = cache_manifest["queries"].get(key)
                            state_path = _cached_state_path(entry, config["donor_cache_manifest"]) if entry else None
                            if entry is None or file_hash(state_path) != entry["state_sha256"]:
                                raise ValueError("missing or corrupt donor cache for the exact query/K/draw")
                            donor_roles, metadata = entry["roles"], entry["metadata"]
                            if order_donor_roles and donor_roles["input_token_ids"] != order_donor_roles["input_token_ids"]:
                                raise ValueError("cached donor order differs from the exact frozen canonical prompt")
                            if metadata["layers"] != layers or metadata["component"] != component:
                                raise ValueError("cached donor component/layer configuration differs")
                            if (metadata["input_token_ids"] != donor_roles["input_token_ids"]
                                    or metadata["model"] != cache_manifest["checkpoint_descriptor"]["model_identity"]):
                                raise ValueError("cached donor state/query/checkpoint provenance differs")
                            states = torch.load(state_path, map_location="cpu", weights_only=True)
                        else:
                            donor_roles = prompt_token_roles(donor, data.task_id, donor_demos, query.text)
                            positions = donor_roles["readout_positions"] if alignment == "readout" else donor_roles["query_positions"]
                            states, metadata = capture_states(donor, donor_roles["prompt"], layers, positions,
                                component=component, cache_dir=writer.path / "donor_cache")
                        if config.get("control") == "wrong_query":
                            if alignment != "readout" or len(data.test) < 2:
                                raise ValueError("wrong-query norm control requires readout alignment and at least two queries")
                            offset = config.get("wrong_query_offset", 1)
                            if type(offset) is not int or offset % len(data.test) == 0:
                                raise ValueError("wrong_query_offset must identify a different query without using labels")
                            wrong = data.test[(data.test.index(query) + offset) % len(data.test)]
                            norm_reference, reference_key = states, metadata["cache_key"]
                            if cache_manifest:
                                import torch
                                wrong_fields = [wrong.example_id, wrong.text_hash, k, donor_k, seed]
                                wrong_key = content_hash(wrong_fields + [order_fingerprint] if order_fingerprint else wrong_fields)
                                wrong_entry = cache_manifest["queries"].get(wrong_key)
                                wrong_path = _cached_state_path(wrong_entry, config["donor_cache_manifest"]) if wrong_entry else None
                                if wrong_entry is None or file_hash(wrong_path) != wrong_entry["state_sha256"]:
                                    raise ValueError("wrong-query donor cache is missing or corrupt")
                                donor_roles, metadata = wrong_entry["roles"], wrong_entry["metadata"]
                                states = torch.load(wrong_path, map_location="cpu", weights_only=True)
                            else:
                                donor_roles = prompt_token_roles(donor, data.task_id, donor_demos, wrong.text)
                                states, metadata = capture_states(donor, donor_roles["prompt"], layers,
                                    donor_roles["readout_positions"], component=component,
                                    cache_dir=writer.path / "wrong_query_cache")
                            states = norm_match_states(states, norm_reference)
                            donor_positions, positions = donor_roles["readout_positions"], roles["readout_positions"]
                            alignment_record = {"policy": "wrong_query_readout", "query_content_equal": False,
                                "norm_reference_cache_key": reference_key, "wrong_query_id": wrong.example_id,
                                "selection_uses": "fixed cyclic query offset, no gold labels",
                                "short_long_position_content_confound": True}
                        else:
                            donor_positions, positions, alignment_record = align_query_positions(donor_roles, roles, alignment)
                        if metadata.get("positions") != donor_positions:
                            raise ValueError("donor states were captured at different token positions")
                        if config.get("control") in {"noise", "permuted"}:
                            states = matched_norm_control(states, config["control"], seed)
                        def factory(ids):
                            return PrefillPatch(model, states, positions, ids, component=component)
                        patched = score_labels_with_intervention(recipient, roles["prompt"], data.label_names,
                            factory, required_prompt_positions=positions)
                    elif operation == "edge_mask":
                        edges = edge_set(roles, config.get("edge_condition", "query_to_demo_labels"))
                        target_edges = list(edges)
                        if config.get("edge_control") == "matched_count":
                            edges = matched_count_edges(edges, len(roles["input_token_ids"]), seed)
                        if not edges and k != 0:
                            raise ValueError("declared edge intervention is empty at this K")
                        def factory(ids):
                            hook = AttentionEdgeMask(model, layers, edges, ids)
                            edge_audits.append(hook.validation)
                            return hook
                        patched = (score_labels_with_intervention(recipient, roles["prompt"], data.label_names,
                            factory, required_prompt_positions=sorted({p for edge in edges for p in edge}))
                            if edges else dict(baseline))
                        metadata = {"edges": edges, "target_edges": target_edges, "probability_audits": edge_audits,
                                    "edge_control": config.get("edge_control"), "zero_context_noop": not edges}
                    else:
                        with BandSplice(recipient, donor, layers) as splice:
                            patched = recipient.score_labels(roles["prompt"], data.label_names)
                            metadata = splice.validation
                        if config.get("reciprocal", True):
                            donor_roles = prompt_token_roles(donor, data.task_id, recipient_demos, query.text)
                            reciprocal_baseline = donor.score_labels(donor_roles["prompt"], data.label_names)
                            with BandSplice(donor, recipient, layers):
                                reciprocal_scores = donor.score_labels(donor_roles["prompt"], data.label_names)
                            for condition, scores in (("reciprocal_baseline", reciprocal_baseline),
                                                      ("reciprocal_splice", reciprocal_scores)):
                                prediction = max(data.label_names, key=scores.get)
                                writer.write_prediction({**prediction_context(donor),
                                    "condition": condition if condition == "reciprocal_baseline" else "reciprocal:" + intervention_condition,
                                    "phase": condition, "task": data.task_id,
                                    "query_id": query.example_id, "query_hash": query.text_hash, "gold_label": query.label,
                                    "prediction": prediction, "correct": prediction == query.label, "label_scores": scores,
                                    "requested_k": k, "realized_k": len(recipient_demos), "draw_id": seed,
                                    "prefill_only": False, "intervention_scope": "all_forward_calls_block_weights",
                                    "input_token_ids": donor_roles["input_token_ids"]})
                    generation_records = {}
                    if config.get("generation_check", False):
                        from ..prompting import parse_label_strict
                        budget = config.get("max_new_tokens", 10)
                        generation_records["baseline"] = recipient.generate_record(roles["prompt"], budget)
                        if (generation_records["baseline"].get("rejected")
                                or generation_records["baseline"].get("truncated")):
                            writer.write_prediction({"condition": "baseline_generation_rejected",
                                **prediction_context(recipient), "readout": "generated_label",
                                "query_id": query.example_id, **generation_records["baseline"], "correct": None})
                            raise RuntimeError("secondary baseline generation exceeded its untruncated context budget")
                        if operation == "splice":
                            with BandSplice(recipient, donor, layers):
                                generated = recipient.generate_record(roles["prompt"], budget)
                        elif operation == "edge_mask" and not edges:
                            generated = dict(generation_records["baseline"])
                        else:
                            with factory(roles["input_token_ids"]):
                                generated = recipient.generate_record(roles["prompt"], budget)
                        generation_records[operation] = generated
                        for condition, generation in generation_records.items():
                            if generation.get("rejected") or generation.get("truncated"):
                                writer.write_prediction({"condition": condition + "_generation_rejected",
                                    **prediction_context(recipient), "readout": "generated_label",
                                    "query_id": query.example_id, **generation, "correct": None})
                                raise RuntimeError("secondary generation check exceeded its untruncated context budget")
                            generation["parsed_label"] = parse_label_strict(generation["raw_text"], data.label_names)
                    conditions = [("baseline", baseline, roles, recipient_demos), (operation, patched, roles, recipient_demos)]
                    if order_selection:
                        conditions.append(("order_donor_baseline", recipient.score_labels(order_donor_roles["prompt"], data.label_names),
                                           order_donor_roles, donor_demos))
                    for condition, scores, recorded_roles, recorded_demos in conditions:
                        prediction = max(data.label_names, key=scores.get)
                        condition_id = condition if condition == "baseline" else intervention_condition
                        if order_selection and condition in {"baseline", "order_donor_baseline"}:
                            side = ("failing" if config.get("order_direction", "healthy_to_failing") == "healthy_to_failing" else "healthy")
                            if condition == "order_donor_baseline":
                                side = "healthy" if side == "failing" else "failing"
                            condition_id = "order_baseline:" + side + ":selection=" + order_fingerprint
                        record = {**prediction_context(recipient),
                            "condition": condition_id,
                            "phase": condition, "task": data.task_id, "query_id": query.example_id,
                            "query_hash": query.text_hash, "gold_label": query.label, "prediction": prediction,
                            "correct": prediction == query.label, "label_scores": scores, "requested_k": k,
                            "realized_k": len(recorded_demos), "donor_k": donor_k, "draw_id": seed,
                            "demonstration_ids": [d.example_id for d in recorded_demos],
                            "demonstration_hashes": [d.text_hash for d in recorded_demos],
                            "demonstration_labels": [d.label for d in recorded_demos],
                            "input_token_ids": recorded_roles["input_token_ids"], "input_token_ids_hash": recorded_roles["input_token_ids_hash"],
                            "token_roles": recorded_roles, "donor": metadata if condition == operation or not order_selection else {},
                            "alignment": alignment_record if condition == operation or not order_selection else {},
                            "prefill_only": operation != "splice", "recipient_model": recipient_identity,
                            "intervention_scope": "all_forward_calls_block_weights" if operation == "splice" else "prefill_activation_hooks",
                            "parse_status": "restricted_complete_label",
                            "score_forward_implementation": ("full_forward" if operation == "patch" and condition == operation
                                else getattr(recipient, "label_score_implementation", "full_forward")),
                            "rejected": False, "truncated": False, "measurement_status": "measured",
                            "query_role": config.get("query_role", "technical_pilot")}
                        if order_selection:
                            record["order_selection"] = {"sha256": order_fingerprint,
                                "source_run_id": order_selection["source_run_id"], "source_manifest_sha256": order_selection["source_manifest_sha256"],
                                "direction": config.get("order_direction", "healthy_to_failing"), "selection_uses": "aggregate development queries only",
                                "donor_demonstration_ids": [d.example_id for d in donor_demos],
                                "recipient_demonstration_ids": [d.example_id for d in recipient_demos],
                                "readout": "complete_label_likelihood"}
                        if condition in generation_records:
                            generation = generation_records[condition]
                            record.update({"generation": generation, "generated_prediction": generation["parsed_label"],
                                           "generated_correct": generation["parsed_label"] == query.label})
                        writer.write_prediction(record)
                        summary.append({"condition": condition, "correct": record["correct"], "k": k})
        if operation == "cache_donor":
            result = {"checkpoint_descriptor": descriptor, "queries": cached_queries,
                      "component": component, "layers": layers, "configuration": config,
                      "cache_manifest": str(writer.path / "donor_manifest.json")}
            with (writer.path / "donor_manifest.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
        else:
            result = {"operation": operation, "n_query_conditions": len(summary),
                      "query_conditions": summary, "inference_status": "paired query/draw analysis required",
                      "recipient": recipient_identity,
                      "interpretation": "technical intervention; sufficiency/necessity require valid controls"}
        writer.complete(result, {"passed": True, "no_training": True, "prefill_only": operation != "splice",
                                 "all_prompts_untruncated": True})
        return writer.path
    except BaseException as error:
        writer.fail(f"{type(error).__name__}: {error}")
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--freeze-order-source", help="completed scientific W3 development manifest; performs no model inference")
    parser.add_argument("--out", help="new frozen same-K order selection JSON")
    parser.add_argument("--ks", default="16", help="order-selection Ks, chosen before examining final queries")
    parser.add_argument("--demo-seeds", default="0,1,2")
    parser.add_argument("--min-order-gap", type=float, default=.02)
    parser.add_argument("--output-root")
    args = parser.parse_args(argv)
    if args.freeze_order_source:
        if not args.out:
            parser.error("--freeze-order-source requires --out")
        freeze_order_selection(args.freeze_order_source, args.out,
            ks=[int(k) for k in args.ks.split(",")], demo_seeds=[int(s) for s in args.demo_seeds.split(",")],
            min_accuracy_gap=args.min_order_gap)
        print(f"wrote frozen development order selection {args.out}")
        return 0
    config = json.loads(Path(args.config).read_text())
    if args.output_root:
        config["output_root"] = args.output_root
    print(f"wrote {run(config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
