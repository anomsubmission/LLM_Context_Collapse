"""Local Hugging Face backend. Device is auto-selected: cuda > mps > cpu.

This is the only backend with weight access, so every intervention and mechanistic
result in the paper is reproducible with it and needs no cloud account.
"""
from __future__ import annotations

from typing import Sequence


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
                 load_in_4bit: bool = False, max_seq_len: int = 8192,
                 hf_token: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf_local:{model_id}"
        self.model_id = model_id
        self.max_seq_len = max_seq_len
        self.device = pick_device(device)
        torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]

        self._tok = AutoTokenizer.from_pretrained(model_id, token=hf_token,
                                                  trust_remote_code=True)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        # Keep the tail (query + "Label:" cue) when a prompt exceeds the budget.
        self._tok.truncation_side = "left"

        kwargs: dict = {"token": hf_token, "trust_remote_code": True}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
            kwargs["device_map"] = "auto"
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, **kwargs)
        except (ValueError, KeyError, OSError):
            # Some multimodal checkpoints (e.g. Gemma-3) are not AutoModelForCausalLM.
            from transformers import AutoModelForImageTextToText
            self._model = AutoModelForImageTextToText.from_pretrained(
                model_id, dtype=torch_dtype, **kwargs)
        if not load_in_4bit:
            self._model = self._model.to(self.device)
        self._model.eval()

    # ---- LMBackend
    def _wrap(self, prompt: str) -> str:
        try:
            return self._tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        import torch
        text = self._wrap(prompt)
        enc = self._tok(text, return_tensors="pt", truncation=True,
                        max_length=self.max_seq_len).to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**enc, max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=self._tok.pad_token_id)
        return self._tok.decode(out[0][enc["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def generate_batch(self, prompts: Sequence[str],
                       max_new_tokens: int = 10) -> list[str]:
        return [self.generate(p, max_new_tokens) for p in prompts]

    def token_length(self, prompt: str) -> int:
        return len(self._tok(self._wrap(prompt), truncation=False)["input_ids"])

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
