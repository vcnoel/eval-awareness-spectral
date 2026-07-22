"""Public, CPU-safe API for the evaluation-cue causal research protocol.

Legacy model-running objects remain available through lazy attributes so importing
the package never imports a model, tokenizer, dataset client, or GPU framework.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .config import ProtocolConfig, load_config
from .schemas import (
    CacheConflictError,
    ClaimStatus,
    IncompleteRunError,
    ProtocolError,
    SpectralComparisonMode,
    ValidationError,
)

__version__ = "0.2.0"

__all__ = [
    "CacheConflictError",
    "ClaimStatus",
    "IncompleteRunError",
    "ProtocolConfig",
    "ProtocolError",
    "SpectralComparisonMode",
    "ValidationError",
    "load_config",
]

_LEGACY_EXPORTS = {
    "Condition": ("prompts", "Condition"),
    "Task": ("prompts", "Task"),
    "build_prompts": ("prompts", "build_prompts"),
    "load_framings": ("prompts", "load_framings"),
    "load_tasks": ("prompts", "load_tasks"),
    "RunConfig": ("runner", "RunConfig"),
    "run_experiment": ("runner", "run_experiment"),
    "run_model": ("runner", "run_model"),
}


def __getattr__(name: str) -> Any:
    """Load historical empirical APIs only when a caller requests one."""

    if name not in _LEGACY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LEGACY_EXPORTS[name]
    module = import_module(f"{__name__}.{module_name}")
    return getattr(module, attribute)
