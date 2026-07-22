from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from eval_awareness_spectral.integrations import (
    OptionalIntegrationUnavailable,
    adapt_activation_container,
    integration_status,
    require_optional_package,
    run_causal_interchange,
    run_evaluation_callback,
    run_steering_callback,
)


@dataclass
class DuckActivationRow:
    sequence_id: str
    layer_idx: int
    token_idx: int
    activations: np.ndarray[tuple[int], np.dtype[np.float64]]


def test_duck_typed_activation_rows_retain_provenance() -> None:
    rows = [
        DuckActivationRow("sequence-a", 4, 8, np.array([1.0, 2.0])),
        DuckActivationRow("sequence-b", 9, 3, np.array([3.0, 4.0])),
    ]
    adapted = adapt_activation_container(rows)
    assert [(row.sequence_id, row.layer, row.token_index) for row in adapted] == [
        ("sequence-a", 4, 8),
        ("sequence-b", 9, 3),
    ]
    np.testing.assert_array_equal(adapted[1].values, [3.0, 4.0])


def test_missing_optional_integration_fails_actionably(monkeypatch: pytest.MonkeyPatch) -> None:
    import eval_awareness_spectral.integrations.goodfire as module

    def missing(_name: str) -> None:
        raise ModuleNotFoundError("not installed")

    monkeypatch.setattr(module, "import_module", missing)
    with pytest.raises(OptionalIntegrationUnavailable, match=r"Install.*duck-typed"):
        require_optional_package("causalab")


def test_status_inspection_does_not_import_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    import eval_awareness_spectral.integrations.goodfire as module

    inspected: list[str] = []

    def inspect(name: str) -> None:
        inspected.append(name)
        return None

    monkeypatch.setattr(module, "find_spec", inspect)
    statuses = integration_status()
    assert set(inspected) == {"goodfire", "causalab", "sglang", "lm_eval"}
    assert not any(status.available for status in statuses.values())


def test_callbacks_preserve_explicit_targets_and_endpoints() -> None:
    target = object()
    calls: list[dict[str, object]] = []

    def record(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return kwargs

    causal = run_causal_interchange(
        record,
        recipient="recipient",
        donor="donor",
        target=target,
        endpoint="causal-endpoint",
    )
    steering = run_steering_callback(
        record,
        prompt="prompt",
        vector=np.array([1.0, 0.0]),
        strength=2.0,
        target=target,
        endpoint="steering-endpoint",
    )
    evaluation = run_evaluation_callback(
        record,
        samples=["sample"],
        target=target,
        endpoint="gold-endpoint",
    )

    assert causal["target"] is target and causal["endpoint"] == "causal-endpoint"
    assert steering["target"] is target and steering["endpoint"] == "steering-endpoint"
    assert evaluation["target"] is target and evaluation["endpoint"] == "gold-endpoint"
    assert len(calls) == 3
