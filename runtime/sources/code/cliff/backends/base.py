"""Backend protocol.

A backend is anything that can turn a prompt into a short completion. That is the
only capability the behavioural sweeps need, which is why they are infrastructure
agnostic: local weights, a self-hosted server, a vendor API, or your own wrapper
all satisfy the same three-method interface.

Interventions and mechanistic analyses additionally need weights and hidden states,
so they require a backend that also implements `WeightAccessBackend`. Only
`hf_local` does. That split is deliberate and is checked at run time with a clear
error rather than an AttributeError deep in a loop.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LMBackend(Protocol):
    """Minimum capability: greedy short-completion generation."""

    name: str

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Greedy-decode a completion. Must be deterministic."""
        ...

    def generate_batch(self, prompts: Sequence[str],
                       max_new_tokens: int = 10) -> list[str]:
        """Default implementations may simply loop over `generate`."""
        ...


@runtime_checkable
class WeightAccessBackend(LMBackend, Protocol):
    """Additional capability needed for interventions and mechanistic probes."""

    def torch_model(self):
        """Return the underlying torch module."""
        ...

    def tokenizer(self):
        ...

    def n_layers(self) -> int:
        ...


class BackendCapabilityError(RuntimeError):
    """Raised when a command needs weight access and the backend cannot provide it."""


class ContextBudgetError(ValueError):
    """The complete prompt plus generation allowance exceeds context capacity."""


@runtime_checkable
class RecordedGenerationBackend(LMBackend, Protocol):
    """Optional measured generation capability; old text backends remain usable."""

    def generate_record(self, prompt: str, max_new_tokens: int = 10) -> dict:
        ...


@runtime_checkable
class LabelScoringBackend(LMBackend, Protocol):
    def score_labels(self, prompt: str, labels: Sequence[str]) -> dict[str, float]:
        """Sum conditional log probabilities over each complete label continuation."""
        ...


def require_weight_access(backend, what: str):
    if not isinstance(backend, WeightAccessBackend):
        raise BackendCapabilityError(
            f"{what} needs a backend with weight access (hidden states and "
            f"trainable parameters), but backend '{getattr(backend, 'name', backend)}' "
            f"only supports text generation.\n"
            f"Use --backend hf_local --model <hf-repo-id> for this command."
        )
    return backend
