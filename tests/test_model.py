from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_bvar.config import SERIES_BY_ID
from us_bvar.model import BVAR
from us_bvar.transforms import ScenarioConstraint, transform_path


def test_baseline_forecast_has_expected_shape_and_is_reproducible(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    first = model.forecast(horizon=12, draws=24, seed=7)
    second = model.forecast(horizon=12, draws=24, seed=7)

    assert first.median.shape == (12, 5)
    assert first.dates[0] > synthetic_levels.index[-1]
    assert np.allclose(first.median, second.median)
    assert (first.lower.to_numpy() <= first.upper.to_numpy()).all()


def test_conditional_forecast_hits_natural_unit_constraints(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    target_cpi = 275.25
    target_rate = 2.75
    scenario = model.forecast(
        horizon=6,
        draws=24,
        constraints={(2, "CPIAUCSL"): target_cpi, (5, "FEDFUNDS"): target_rate},
        seed=11,
    )

    assert scenario.is_scenario
    assert np.isclose(scenario.median.iloc[2]["CPIAUCSL"], target_cpi)
    assert np.isclose(scenario.lower.iloc[2]["CPIAUCSL"], target_cpi)
    assert np.isclose(scenario.upper.iloc[5]["FEDFUNDS"], target_rate)


def test_invalid_log_constraint_is_rejected(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)

    try:
        model.forecast(horizon=3, draws=20, constraints={(0, "INDPRO"): 0.0})
    except ValueError as exc:
        assert "greater than zero" in str(exc)
    else:
        raise AssertionError("Expected a non-positive logged scenario value to fail")


@pytest.mark.parametrize(
    ("transformation", "step"),
    [
        ("mom", 5),
        ("qoq", 5),
        ("yoy", 11),
        ("mom_annualized", 5),
        ("qoq_annualized", 5),
        ("yoy_annualized", 11),
    ],
)
@pytest.mark.parametrize(
    ("series_id", "target"),
    [("CPIAUCSL", 2.4), ("UNRATE", 1.2)],
)
def test_conditional_forecast_hits_transformed_constraints(
    synthetic_levels,
    transformation: str,
    step: int,
    series_id: str,
    target: float,
) -> None:
    model = BVAR().fit(synthetic_levels)
    constraint = ScenarioConstraint(target, transformation)
    scenario = model.forecast(
        horizon=12,
        draws=20,
        constraints={(step, series_id): constraint},
        seed=13,
    )
    path = pd.concat([synthetic_levels[series_id], scenario.median[series_id]])
    transformed = transform_path(path, SERIES_BY_ID[series_id], transformation)

    assert transformed.loc[scenario.dates[step]] == pytest.approx(target)
    assert scenario.constraints[(step, series_id)] == constraint


def test_log_growth_constraint_rejects_a_total_loss(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)

    with pytest.raises(ValueError, match="greater than -100 percent"):
        model.forecast(
            horizon=3,
            draws=20,
            constraints={(0, "CPIAUCSL"): ScenarioConstraint(-100.0, "mom")},
        )


def test_multiple_growth_constraints_form_an_exact_forecast_path(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    targets = {0: 0.2, 1: 0.3, 4: -0.1, 8: 0.4}
    scenario = model.forecast(
        horizon=12,
        draws=20,
        constraints={
            (step, "CPIAUCSL"): ScenarioConstraint(target, "mom")
            for step, target in targets.items()
        },
        seed=17,
    )
    path = pd.concat([synthetic_levels["CPIAUCSL"], scenario.median["CPIAUCSL"]])
    transformed = transform_path(path, SERIES_BY_ID["CPIAUCSL"], "mom")

    for step, target in targets.items():
        assert transformed.loc[scenario.dates[step]] == pytest.approx(target)


def test_joint_forecast_covariance_matches_direct_block_calculation(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    rng = np.random.default_rng(19)
    coefficients, sigma = model._draw_parameters(rng)
    horizon = 5
    _, actual = model._joint_forecast_moments(coefficients, sigma, horizon)

    n = len(model.variable_ids)
    p = model.config.lags
    autoregressive = [coefficients[1 + lag * n : 1 + (lag + 1) * n, :].T for lag in range(p)]
    responses = [np.eye(n)]
    for h in range(1, horizon):
        response = np.zeros((n, n))
        for lag in range(1, min(p, h) + 1):
            response += autoregressive[lag - 1] @ responses[h - lag]
        responses.append(response)

    expected = np.empty_like(actual)
    for h in range(horizon):
        for j in range(horizon):
            block = np.zeros((n, n))
            for shock_time in range(min(h, j) + 1):
                block += responses[h - shock_time] @ sigma @ responses[j - shock_time].T
            expected[h * n : (h + 1) * n, j * n : (j + 1) * n] = block

    assert np.allclose(actual, expected)


def test_gaussian_sampling_repairs_covariance_only_after_cholesky_failure(monkeypatch) -> None:
    rng = np.random.default_rng(23)
    mean = np.zeros(2)

    def unexpected_repair(matrix: np.ndarray) -> np.ndarray:
        raise AssertionError("A positive-definite covariance should not be repaired.")

    monkeypatch.setattr(BVAR, "_nearest_positive_definite", unexpected_repair)
    sample = BVAR._sample_gaussian(rng, mean, np.eye(2))

    assert sample.shape == (2,)
