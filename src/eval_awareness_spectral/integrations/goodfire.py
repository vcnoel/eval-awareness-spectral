"""Optional, protocol-based seams for external Goodfire ecosystem tools.

This module intentionally depends only on the public core and Python protocols.
External containers and callbacks are duck typed. Optional packages are imported
only by :func:`require_optional_package`, never while importing this module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np
import numpy.typing as npt

from ..schemas import ValidationError
from .standalone import ActivationRecord

T = TypeVar("T")


class OptionalIntegrationUnavailable(ImportError):
    """An optional integration was requested but its package is unavailable."""


@dataclass(frozen=True)
class IntegrationStatus:
    """Import-free availability result for one optional package."""

    package: str
    available: bool
    install_hint: str


_OPTIONAL_PACKAGES: dict[str, tuple[str, str]] = {
    "goodfire": (
        "goodfire_core",
        "Install goodfire-core in the calling environment; on Silico it is pre-provisioned.",
    ),
    "causalab": ("causalab", "Install causalab and configure its supported model runtime."),
    "sglang": ("sglang", "Install SGLang in the inference environment."),
    "lm_eval": ("lm_eval", "Install lm-evaluation-harness in the evaluation environment."),
}


def integration_status() -> dict[str, IntegrationStatus]:
    """Inspect optional package specs without executing or importing package code."""

    statuses: dict[str, IntegrationStatus] = {}
    for name, (module_name, hint) in _OPTIONAL_PACKAGES.items():
        try:
            available = find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        statuses[name] = IntegrationStatus(module_name, available, hint)
    return statuses


def require_optional_package(name: str) -> ModuleType:
    """Lazily import a known optional package or raise an actionable error."""

    try:
        module_name, hint = _OPTIONAL_PACKAGES[name]
    except KeyError as error:
        choices = ", ".join(sorted(_OPTIONAL_PACKAGES))
        raise ValueError(f"unknown optional integration {name!r}; choose one of: {choices}") from error
    try:
        return import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise OptionalIntegrationUnavailable(
            f"Optional integration {name!r} is unavailable ({module_name!r} could not be "
            f"imported). {hint} You may instead pass a duck-typed callback or container."
        ) from error


@runtime_checkable
class ActivationRowLike(Protocol):
    """Structural row supported by :func:`adapt_activation_container`."""

    sequence_id: str
    layer: int
    token_index: int
    activation: npt.ArrayLike


@runtime_checkable
class CausalInterchangeRunner(Protocol):
    """Callback seam for a causalab-like interchange runner."""

    def __call__(
        self,
        *,
        recipient: T,
        donor: T,
        target: object,
        endpoint: str,
    ) -> object: ...


@runtime_checkable
class SteeringCallback(Protocol):
    """Callback seam for an SGLang-like targeted steering request."""

    def __call__(
        self,
        *,
        prompt: str,
        vector: npt.ArrayLike,
        strength: float,
        target: object,
        endpoint: str,
    ) -> object: ...


@runtime_checkable
class EvaluationCallback(Protocol):
    """Callback seam for lm-eval or a gold scoring endpoint."""

    def __call__(
        self,
        *,
        samples: object,
        target: object,
        endpoint: str,
    ) -> object: ...


def _field(row: object, names: tuple[str, ...]) -> object:
    if isinstance(row, Mapping):
        for name in names:
            if name in row:
                return row[name]
    else:
        for name in names:
            if hasattr(row, name):
                return getattr(row, name)
    raise ValidationError(f"activation row is missing required field {names[0]!r}")


def adapt_activation_row(row: object) -> ActivationRecord:
    """Adapt one ActivationDataset-like row while retaining all provenance."""

    sequence_id = _field(row, ("sequence_id", "sequenceId", "id"))
    layer = _field(row, ("layer", "layer_index", "layer_idx"))
    token_index = _field(row, ("token_index", "token_idx", "position"))
    values = _field(row, ("activation", "activations", "values", "value"))
    if not isinstance(sequence_id, str):
        raise ValidationError("activation row sequence_id must be a string")
    if isinstance(layer, bool) or not isinstance(layer, (int, np.integer)):
        raise ValidationError("activation row layer must be an integer")
    if isinstance(token_index, bool) or not isinstance(token_index, (int, np.integer)):
        raise ValidationError("activation row token_index must be an integer")
    return ActivationRecord(sequence_id, int(layer), int(token_index), np.asarray(values))


def adapt_activation_container(container: Iterable[object]) -> list[ActivationRecord]:
    """Adapt an ActivationDataset/LocalSequenceDataset-like iterable.

    No concrete Goodfire class is imported or required. Each row must expose a
    sequence ID, layer, token position, and one activation vector as attributes
    or mapping keys; common ``*_idx`` aliases are accepted.
    """

    records = [adapt_activation_row(row) for row in container]
    if not records:
        raise ValidationError("activation container contains no rows")
    return records


def run_causal_interchange(
    runner: CausalInterchangeRunner[T],
    *,
    recipient: T,
    donor: T,
    target: object,
    endpoint: str,
) -> object:
    """Invoke an interchange runner without weakening target or endpoint scope."""

    if not endpoint:
        raise ValidationError("causal interchange endpoint must be explicit")
    return runner(recipient=recipient, donor=donor, target=target, endpoint=endpoint)


def run_steering_callback(
    callback: SteeringCallback,
    *,
    prompt: str,
    vector: npt.ArrayLike,
    strength: float,
    target: object,
    endpoint: str,
) -> object:
    """Invoke an SGLang-like callback with explicit target and endpoint."""

    if not endpoint:
        raise ValidationError("steering endpoint must be explicit")
    array = np.asarray(vector, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValidationError("steering vector must be a non-empty finite vector")
    if not np.isfinite(strength):
        raise ValidationError("steering strength must be finite")
    return callback(
        prompt=prompt,
        vector=array,
        strength=float(strength),
        target=target,
        endpoint=endpoint,
    )


def run_evaluation_callback(
    callback: EvaluationCallback,
    *,
    samples: object,
    target: object,
    endpoint: str,
) -> object:
    """Invoke lm-eval/gold scoring while preserving its target and endpoint."""

    if not endpoint:
        raise ValidationError("evaluation endpoint must be explicit")
    return callback(samples=samples, target=target, endpoint=endpoint)


# Domain-readable aliases for the two supported evaluation deployments.
run_lm_evaluation = run_evaluation_callback
score_gold_endpoint = run_evaluation_callback
