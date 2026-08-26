"""Backend registry.

Adding your own infrastructure is a ~20-line class with a `generate` method plus
one line here; see README section "Bring your own backend".
"""
from __future__ import annotations

from .base import (BackendCapabilityError, LMBackend, WeightAccessBackend,
                   require_weight_access)

BACKENDS = {
    "hf_local": ("cliff.backends.hf_local", "HFLocalBackend"),
    "openai_compat": ("cliff.backends.openai_compat", "OpenAICompatBackend"),
    "echo": ("cliff.backends.echo", "EchoBackend"),
}


def get_backend(kind: str, model_id: str, **kwargs):
    """Instantiate a backend by name. Imports lazily so that `echo` and the
    verify/figures commands never import torch."""
    if kind not in BACKENDS:
        raise SystemExit(f"unknown backend {kind!r}; choose from {sorted(BACKENDS)}")
    module_name, cls_name = BACKENDS[kind]
    import importlib
    cls = getattr(importlib.import_module(module_name), cls_name)
    sig_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return cls(model_id, **sig_kwargs)


__all__ = ["BACKENDS", "get_backend", "LMBackend", "WeightAccessBackend",
           "require_weight_access", "BackendCapabilityError"]
