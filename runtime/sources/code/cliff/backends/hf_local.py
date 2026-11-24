"""Local Hugging Face backend. Device is auto-selected: cuda > mps > cpu.

Scientific runs use explicit rendering and reject context overflow by default.
Historical left truncation remains available only through an explicit policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Sequence

from .base import ContextBudgetError


def pick_device(requested: str | None = None) -> str:
    import torch
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLocalBackend:
    """Wraps a local causal LM. Satisfies WeightAccessBackend."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "bfloat16",
                 load_in_4bit: bool = False, max_seq_len: int | None = None,
                 hf_token: str | None = None, *, template: str = "native",
                 overflow_policy: str = "reject", revision: str | None = None,
                 tokenizer_revision: str | None = None,
                 trust_remote_code: bool = False, local_files_only: bool = False,
                 attn_implementation: str | None = None,
                 template_marker_policy: str = "native_tokens",
                 label_score_implementation: str = "full_forward",
                 label_prefix: str = " "):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        if template not in {"native", "raw", "llama", "hermes"}:
            raise ValueError("template must be native, raw, llama or hermes")
        if overflow_policy not in {"reject", "legacy_left_truncate"}:
            raise ValueError("overflow_policy must be reject or legacy_left_truncate")
        if template_marker_policy not in {"native_tokens", "literal_markers"}:
            raise ValueError("unknown cross-template marker policy")
        if label_score_implementation not in {"full_forward", "prefix_cache"}:
            raise ValueError("unknown label likelihood implementation")
        if label_prefix not in {"", " ", "\n"}:
            raise ValueError("label_prefix must be empty, one space, or one newline")
        self.name = f"hf_local:{model_id}"
        self.model_id = model_id
        self.template = template
        self.template_marker_policy = template_marker_policy
        self.attn_implementation = attn_implementation
        self.label_score_implementation = label_score_implementation
        self.label_prefix = label_prefix
        self.overflow_policy = overflow_policy
        self.revision = revision
        self.device = pick_device(device)
        self.label_scoring_policy = "complete_string_sum_ll_common_conditioning_prefix_v2"
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]
        common = {"token": hf_token, "trust_remote_code": trust_remote_code,
                  "local_files_only": local_files_only}
        if os.environ.get("HF_HUB_CACHE"):
            common["cache_dir"] = os.environ["HF_HUB_CACHE"]
        self._tok = AutoTokenizer.from_pretrained(
            model_id, revision=tokenizer_revision or revision, **common)
        from transformers.utils.hub import cached_file, extract_commit_hash
        tokenizer_file = cached_file(model_id, "tokenizer_config.json",
            revision=tokenizer_revision or revision, cache_dir=common.get("cache_dir"),
            token=hf_token, local_files_only=local_files_only,
            _raise_exceptions_for_missing_entries=False)
        self.resolved_tokenizer_revision = extract_commit_hash(tokenizer_file, None) if tokenizer_file else None
        self.tokenizer_provenance = {
            "requested_revision": tokenizer_revision or revision,
            "resolved_revision": self.resolved_tokenizer_revision,
            "configuration_file": str(tokenizer_file) if tokenizer_file else None,
            "configuration_sha256": hashlib.sha256(open(tokenizer_file, "rb").read()).hexdigest() if tokenizer_file else None}
        if self._tok.pad_token is None:
            if self._tok.eos_token is None:
                raise ValueError("tokenizer has neither pad nor EOS token")
            self._tok.pad_token = self._tok.eos_token
        kwargs: dict = {**common, "revision": revision}
        if attn_implementation is not None:
            kwargs["attn_implementation"] = attn_implementation
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
            kwargs["device_map"] = "auto"
        config = AutoConfig.from_pretrained(model_id, revision=revision, **common)
        model_class = AutoModelForCausalLM
        if getattr(config, "model_type", "") == "gemma3":
            from transformers import AutoModelForImageTextToText
            model_class = AutoModelForImageTextToText
        self._model = model_class.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
        self.resolved_model_revision = getattr(self._model.config, "_commit_hash", None)
        if not load_in_4bit:
            self._model = self._model.to(self.device)
        self._model.eval()
        text_config = getattr(self._model.config, "text_config", self._model.config)
        limits = [getattr(text_config, attr, None) for attr in
                  ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")]
        limits.append(getattr(self._tok, "model_max_length", None))
        limits = [int(v) for v in limits if isinstance(v, (int, float)) and 0 < v < 10**9]
        if max_seq_len is not None:
            if max_seq_len <= 0:
                raise ValueError("max_seq_len must be positive")
            limits.append(max_seq_len)
        if not limits:
            raise ValueError("context capacity is unknown; specify max_seq_len explicitly")
        self.max_seq_len = min(limits)
        # Fail at construction rather than silently changing format during a run.
        self._wrap("")

    def _wrap(self, prompt: str) -> str:
        template = getattr(self, "template", "native")
        if template == "raw":
            return prompt
        if template == "native":
            if not getattr(self._tok, "chat_template", None):
                raise ValueError("native template requested but tokenizer has no chat_template")
            return self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        markers = ({"llama": ("<|begin_of_text|>", "<|start_header_id|>",
                              "<|end_header_id|>", "<|eot_id|>"),
                    "hermes": ("<|im_start|>", "<|im_end|>")})[template]
        vocab = self._tok.get_vocab()
        missing = [marker for marker in markers if marker not in vocab]
        if missing and getattr(self, "template_marker_policy", "native_tokens") == "native_tokens":
            raise ValueError(f"{template} template requires tokenizer markers: {missing}")
        if template == "llama":
            return ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                    + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
        return "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

    def _encode_rendered(self, text: str) -> list[int]:
        # Chat template strings already contain their own special tokens.
        encoded = self._tok(text, add_special_tokens=self.template == "raw",
                            truncation=False)["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        return list(encoded)

    def encode_prompt(self, prompt: str) -> list[int]:
        return self._encode_rendered(self._wrap(prompt))

    def encode_completion(self, prompt: str, completion: str) -> tuple[list[int], int]:
        """Tokenize the complete string and locate its first continuation token.

        A tokenizer can merge the final prompt characters with the label. The
        divergence token then belongs to the target, even if its offset overlaps
        the prompt boundary. Independently encoding the label would miss this.
        """
        wrapped = self._wrap(prompt)
        prefix = self._encode_rendered(wrapped)
        complete = self._encode_rendered(wrapped + completion)
        shared = 0
        for a, b in zip(prefix, complete):
            if a != b:
                break
            shared += 1
        if shared == 0:
            raise ValueError("continuation changes the first token; no conditioning prefix remains")
        return complete, shared

    @staticmethod
    def _ids_hash(ids: Sequence[int]) -> str:
        return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()

    def _template_hash(self) -> str:
        spec = getattr(self._tok, "chat_template", "") if self.template == "native" else self.template
        return hashlib.sha256((str(spec) + "|" + getattr(self, "template_marker_policy", "native_tokens")).encode()).hexdigest()

    def template_marker_record(self) -> dict:
        """Literal foreign markers are distinct from authentic special-token templates."""
        markers = {"llama": ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                   "hermes": ["<|im_start|>", "<|im_end|>"]}.get(self.template, [])
        vocab = self._tok.get_vocab() if markers else {}
        specials = set(getattr(self._tok, "all_special_tokens", []))
        return {"policy": getattr(self, "template_marker_policy", "native_tokens"),
                "markers": {marker: {"in_vocabulary": marker in vocab,
                                     "is_special_token": marker in specials,
                                     "token_ids": list(self._tok(marker, add_special_tokens=False)["input_ids"])}
                            for marker in markers},
                "vocabulary_resized": False}

    def generate_record(self, prompt: str, max_new_tokens: int = 10) -> dict:
        import torch
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        t0 = time.perf_counter()
        token_ids = self.encode_prompt(prompt)
        before = len(token_ids)
        budget = self.max_seq_len - max_new_tokens
        over = before > budget or budget < 1 or before == 0
        rejected = over and (self.overflow_policy == "reject" or budget < 1 or before == 0)
        truncated = over and not rejected
        before_hash = self._ids_hash(token_ids)
        if truncated:
            token_ids = token_ids[-budget:]
        record = {"raw_text": "", "input_tokens_before": before,
                  "input_tokens_after": len(token_ids), "generation_tokens": 0,
                  "input_budget": max(0, budget), "context_limit": self.max_seq_len,
                  "max_new_tokens": max_new_tokens, "truncated": truncated,
                  "rejected": rejected, "rejection_reason": "context_budget_exceeded" if rejected else None,
                  "overflow_policy": self.overflow_policy,
                  "retained_instruction": False if truncated else True,
                  "retained_query": None if truncated else True,
                  "template": self.template, "template_hash": self._template_hash(),
                  "template_markers": self.template_marker_record(),
                  "input_token_ids": token_ids, "input_token_ids_hash": self._ids_hash(token_ids),
                  "input_token_ids_before_hash": before_hash,
                  "measurement_status": "measured", "label_scoring_policy": self.label_scoring_policy}
        record["label_score_implementation"] = getattr(self, "label_score_implementation", "full_forward")
        record["label_prefix"] = getattr(self, "label_prefix", " ")
        if not rejected:
            inputs = torch.tensor([token_ids], dtype=torch.long, device=self._model.device)
            with torch.inference_mode():
                out = self._model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=self._tok.pad_token_id)
            generated = out[0][len(token_ids):]
            record["generation_tokens"] = len(generated)
            record["raw_text"] = self._tok.decode(generated, skip_special_tokens=True).strip()
        record["latency_seconds"] = time.perf_counter() - t0
        return record

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        record = self.generate_record(prompt, max_new_tokens)
        if record["rejected"]:
            raise ContextBudgetError(
                f"prompt has {record['input_tokens_before']} tokens plus {max_new_tokens} generation "
                f"tokens, context limit is {self.max_seq_len}")
        return record["raw_text"]

    def generate_batch(self, prompts: Sequence[str],
                       max_new_tokens: int = 10) -> list[str]:
        return [self.generate(p, max_new_tokens) for p in prompts]

    def token_length(self, prompt: str) -> int:
        return len(self.encode_prompt(prompt))

    def label_encodings(self, prompt: str, labels: Sequence[str]) -> dict[str, tuple[list[int], int]]:
        """Condition every candidate on the same token prefix.

        Candidate-specific boundary retokenization must not change the scoring
        denominator. Re-score any prompt suffix after the minimum shared prefix
        for every candidate, including those that did not retokenize it.
        """
        if not labels or len(set(labels)) != len(labels) or any(not label for label in labels):
            raise ValueError("labels must be distinct nonempty strings")
        encoded = {label: self.encode_completion(prompt, getattr(self, "label_prefix", " ") + label) for label in labels}
        common_start = min(start for _, start in encoded.values())
        if common_start < 1 or any(start >= len(ids) for ids, start in encoded.values()):
            raise ValueError("every label must have continuation tokens and a conditioning prefix")
        prefixes = {tuple(ids[:common_start]) for ids, _ in encoded.values()}
        if len(prefixes) != 1:
            raise ValueError("candidate label encodings do not share the declared conditioning prefix")
        return {label: (ids, common_start) for label, (ids, _) in encoded.items()}

    def score_labels(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        """Complete-label sum log likelihood; no length normalization or truncation.

        The explicit continuation is a space followed by the complete label. If
        boundary retokenization occurs, all labels use the same conditioning
        prefix and include any re-scored prompt suffix. This policy is recorded.
        """
        import torch
        if getattr(self, "label_score_implementation", "full_forward") == "prefix_cache":
            return self.score_labels_cached(prompt, labels)
        scores = {}
        for label, (ids, start) in self.label_encodings(prompt, labels).items():
            if len(ids) > self.max_seq_len:
                raise ContextBudgetError("complete label continuation exceeds context budget")
            inputs = torch.tensor([ids], dtype=torch.long, device=self._model.device)
            with torch.inference_mode():
                logits = self._model(input_ids=inputs, attention_mask=torch.ones_like(inputs)).logits
                logprobs = torch.log_softmax(logits[:, :-1].float(), dim=-1)
                target_ll = logprobs.gather(-1, inputs[:, 1:].unsqueeze(-1)).squeeze(-1)
            scores[label] = float(target_ll[0, start - 1:].sum().item())
        return scores

    def score_labels_cached(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        """Exact common-prefix likelihoods with one shared, immutable prefill.

        This is an explicit implementation option. It never reuses state across
        prompts or adapters. Legacy tuples are reconstructed into a fresh model
        cache per candidate; supported models must leave the shared tensors intact.
        """
        import inspect
        import torch
        encoded = self.label_encodings(prompt, labels)
        start = next(iter(encoded.values()))[1]
        if any(len(ids) > self.max_seq_len for ids, _ in encoded.values()):
            raise ContextBudgetError("complete label continuation exceeds context budget")
        prefix = next(iter(encoded.values()))[0][:start]
        device = self._model.device
        inputs = torch.tensor([prefix], dtype=torch.long, device=device)
        parameters = inspect.signature(self._model.forward).parameters
        last_only = {"num_logits_to_keep": 1} if "num_logits_to_keep" in parameters else {}
        with torch.inference_mode():
            prefill = self._model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                  use_cache=True, **last_only)
            cache = prefill.past_key_values
            legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
            if not isinstance(legacy, tuple) or not legacy:
                raise ValueError("prefix-cache scoring requires a supported immutable legacy KV tuple")
            first_logprobs = torch.log_softmax(prefill.logits[0, -1].float(), dim=-1)
            scores = {}
            for label, (ids, _) in encoded.items():
                target = ids[start:]
                score = first_logprobs[target[0]]
                if len(target) > 1:
                    branch = torch.tensor([target[:-1]], dtype=torch.long, device=device)
                    mask = torch.ones((1, start + len(target) - 1), dtype=torch.long, device=device)
                    if hasattr(type(cache), "from_legacy_cache"):
                        branch_cache = type(cache).from_legacy_cache(legacy)
                    elif isinstance(cache, tuple):
                        branch_cache = legacy
                    else:
                        raise ValueError("cache type does not support independent candidate reconstruction")
                    output = self._model(input_ids=branch, attention_mask=mask,
                                         past_key_values=branch_cache, use_cache=True)
                    logprobs = torch.log_softmax(output.logits[0].float(), dim=-1)
                    wanted = torch.tensor(target[1:], dtype=torch.long, device=device)
                    score = score + logprobs.gather(1, wanted[:, None]).sum()
                scores[label] = float(score.item())
        return scores

    # ---- WeightAccessBackend
    def torch_model(self):
        return self._model

    def tokenizer(self):
        return self._tok

    def n_layers(self) -> int:
        cfg = self._model.config
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(cfg, attr):
                return int(getattr(cfg, attr))
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            return int(text_cfg.num_hidden_layers)
        raise AttributeError(f"cannot determine layer count for {self.model_id}")
