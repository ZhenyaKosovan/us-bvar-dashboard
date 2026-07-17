from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from us_bvar.config import ConvergencePolicy
from us_bvar.diagnostics import (
    array_diagnostic,
    chain_diagnostic,
    evaluate_convergence_gate,
)
from us_bvar.model import history_semantics_metadata


def test_history_semantics_metadata_is_explicit_and_non_draw_based() -> None:
    metadata = history_semantics_metadata()

    assert metadata == {
        "contract_version": 1,
        "mode": "fixed_published_history",
        "early_growth_anchors": "displayed_fixed_history",
        "terminal_state_pairing": "same_posterior_draw_as_forecast_parameters",
        "terminal_state_affects_forecast_dynamics": True,
        "paired_historical_draws": False,
    }


def _array_summary(
    dimensions: int,
    *,
    maximum_r_hat: float = 1.01,
    minimum_effective_sample_size: float = 30.0,
    r_hat_count: int = 0,
    ess_count: int = 0,
) -> dict[str, float | int]:
    return {
        "dimensions_checked": dimensions,
        "maximum_r_hat": maximum_r_hat,
        "r_hat_99th_percentile": 1.01,
        "r_hat_outside_nominal_count": r_hat_count,
        "r_hat_outside_nominal_fraction": r_hat_count / dimensions,
        "minimum_effective_sample_size": minimum_effective_sample_size,
        "effective_sample_size_1st_percentile": 30.0,
        "effective_sample_size_outside_nominal_count": ess_count,
        "effective_sample_size_outside_nominal_fraction": ess_count / dimensions,
    }


def _gate(array_summary: dict[str, float | int]) -> dict[str, object]:
    return evaluate_convergence_gate(
        {"scalar": {"r_hat": 1.0, "effective_sample_size": 100.0}},
        {"posterior": array_summary},
        ConvergencePolicy(),
    )


def test_rank_normalized_rhat_rejects_constant_disagreeing_chains() -> None:
    values = np.concatenate([np.zeros(8), np.ones(8)])
    chain_ids = np.repeat(np.arange(2), 8)

    diagnostic = chain_diagnostic(values, chain_ids)

    assert np.isinf(diagnostic["r_hat"])


def test_array_diagnostic_checks_every_parameter_dimension() -> None:
    rng = np.random.default_rng(31)
    values = rng.normal(size=(40, 3, 2))
    chain_ids = np.repeat(np.arange(2), 20)

    diagnostic = array_diagnostic(values, chain_ids)

    assert diagnostic["dimensions_checked"] == 6
    assert cast(float, diagnostic["maximum_r_hat"]) >= 1.0
    assert cast(float, diagnostic["minimum_effective_sample_size"]) > 0
    assert cast(int, diagnostic["r_hat_outside_nominal_count"]) >= 0
    assert cast(int, diagnostic["effective_sample_size_outside_nominal_count"]) >= 0


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("maximum_r_hat", 1.51, "R-hat absolute bound"),
        ("minimum_effective_sample_size", 9.0, "ESS absolute bound"),
    ],
)
def test_convergence_gate_rejects_absolute_extrema(field: str, value: float, failure: str) -> None:
    summary = _array_summary(100)
    summary[field] = value

    gate = _gate(summary)

    assert not gate["accepted"]
    failures = cast(list[str], gate["failures"])
    assert any(failure in reason for reason in failures)


@pytest.mark.parametrize(
    ("dimensions", "r_hat_count", "ess_count", "failure"),
    [
        (100, 26, 0, "R-hat nominal-tail bound"),
        (1_000, 0, 11, "ESS nominal-tail bound"),
    ],
)
def test_convergence_gate_rejects_tail_count_or_fraction(
    dimensions: int, r_hat_count: int, ess_count: int, failure: str
) -> None:
    gate = _gate(
        _array_summary(
            dimensions,
            r_hat_count=r_hat_count,
            ess_count=ess_count,
        )
    )

    assert not gate["accepted"]
    failures = cast(list[str], gate["failures"])
    assert any(failure in reason for reason in failures)
