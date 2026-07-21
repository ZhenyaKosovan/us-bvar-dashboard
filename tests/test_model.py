from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pytest

from us_bvar import model as model_module
from us_bvar.config import SERIES_BY_ID, SERIES_SPECS
from us_bvar.model import BVAR, BVARConfig
from us_bvar.state_space import observation_system
from us_bvar.transforms import PlotTransformation, ScenarioConstraint, transform_path


def test_default_and_quick_configs_make_estimation_intent_explicit() -> None:
    default = BVARConfig()
    quick = BVARConfig.quick()

    assert default.mcmc_iterations == 1200
    assert default.burn_in == 600
    assert default.mcmc_chains == 4
    assert default.innovation_prior_strength == 50.0
    assert not default.quick_mode
    assert quick.mcmc_iterations == 8
    assert quick.burn_in == 4
    assert quick.mcmc_chains == 1
    assert quick.quick_mode


def test_undersized_non_quick_fit_is_rejected_before_sampling(synthetic_levels) -> None:
    config = BVARConfig(mcmc_iterations=8, burn_in=4, thin=1, mcmc_chains=1)

    with pytest.raises(ValueError, match="BVARConfig.quick"):
        BVAR(config=config).fit(synthetic_levels)


def test_baseline_forecast_has_expected_shape_and_is_reproducible(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    first = model.forecast(horizon=12, draws=24, seed=7)
    second = model.forecast(horizon=12, draws=24, seed=7)

    variables = len(SERIES_SPECS)
    if first.median.shape != (12, variables):
        raise AssertionError(f"Unexpected median shape: {first.median.shape}")
    if first.samples.shape != (24, 12, variables):
        raise AssertionError(f"Unexpected sample shape: {first.samples.shape}")
    assert not first.samples.flags.writeable
    assert first.dates[0] > synthetic_levels.index[-1]
    assert np.allclose(first.median, second.median)
    assert (first.lower.to_numpy() <= first.upper.to_numpy()).all()
    assert model.history_levels is not None
    assert model.observed_levels is not None
    assert model.posterior_terminal_states is not None
    assert np.all(model.history_levels.notna().to_numpy())
    assert model.observed_levels["GDPC1"].count() == len(synthetic_levels) // 3
    expected_state_width = len(model.variable_ids) * model.config.lags
    if model.posterior_terminal_states.shape[1] != expected_state_width:
        raise AssertionError("The retained terminal companion state has the wrong width.")


def test_conditional_forecast_hits_natural_unit_constraints(synthetic_levels, fitted_model) -> None:
    model = fitted_model
    target_gdp = 105.0
    target_cpi = 275.25
    target_rate = 2.75
    scenario = model.forecast(
        horizon=6,
        draws=24,
        constraints={
            (1, "GDPC1"): target_gdp,
            (2, "CPIAUCSL"): target_cpi,
            (5, "FEDFUNDS"): target_rate,
        },
        seed=11,
    )

    assert scenario.is_scenario
    assert np.isclose(scenario.median.iloc[1]["GDPC1"], target_gdp)
    assert np.isclose(scenario.median.iloc[2]["CPIAUCSL"], target_cpi)
    assert np.isclose(scenario.lower.iloc[2]["CPIAUCSL"], target_cpi)
    assert np.isclose(scenario.upper.iloc[5]["FEDFUNDS"], target_rate)
    assert scenario.component_effective_sample_size is not None
    assert scenario.component_effective_sample_size > 0


def test_scenario_rejects_collapsed_posterior_component_weights(
    synthetic_levels, fitted_model, monkeypatch
) -> None:
    model = fitted_model
    assert model.posterior_coefficients is not None
    collapsed = np.zeros(len(model.posterior_coefficients))
    collapsed[0] = 1.0
    monkeypatch.setattr(
        model,
        "_constraint_component_probabilities",
        lambda components, constraint_matrix, targets: collapsed,
    )

    with pytest.raises(ValueError, match="too few posterior components"):
        model.forecast(horizon=3, draws=20, constraints={(0, "FEDFUNDS"): 3.0})


def test_invalid_log_constraint_is_rejected(synthetic_levels, fitted_model) -> None:
    model = fitted_model

    try:
        model.forecast(horizon=3, draws=20, constraints={(0, "GDPC1"): 0.0})
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
    fitted_model,
    transformation: PlotTransformation,
    step: int,
    series_id: str,
    target: float,
) -> None:
    model = fitted_model
    constraint = ScenarioConstraint(target, transformation)
    scenario = model.forecast(
        horizon=12,
        draws=20,
        constraints={(step, series_id): constraint},
        seed=13,
    )
    history = cast(pd.Series, synthetic_levels.loc[:, series_id])
    forecast = cast(pd.Series, scenario.median.loc[:, series_id])
    path = cast(pd.Series, pd.concat([history, forecast]))
    transformed = transform_path(path, SERIES_BY_ID[series_id], transformation)

    assert transformed.loc[scenario.dates[step]] == pytest.approx(target)
    if scenario.constraints[(step, series_id)] != constraint:
        raise AssertionError("The scenario did not preserve its constraint metadata.")


def test_log_growth_constraint_rejects_a_total_loss(synthetic_levels, fitted_model) -> None:
    model = fitted_model

    with pytest.raises(ValueError, match="greater than -100 percent"):
        model.forecast(
            horizon=3,
            draws=20,
            constraints={(0, "CPIAUCSL"): ScenarioConstraint(-100.0, "mom")},
        )


def test_multiple_growth_constraints_form_an_exact_forecast_path(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
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
    history = cast(pd.Series, synthetic_levels.loc[:, "CPIAUCSL"])
    forecast = cast(pd.Series, scenario.median.loc[:, "CPIAUCSL"])
    path = cast(pd.Series, pd.concat([history, forecast]))
    transformed = transform_path(path, SERIES_BY_ID["CPIAUCSL"], "mom")

    for step, target in targets.items():
        assert transformed.loc[scenario.dates[step]] == pytest.approx(target)


def test_joint_forecast_covariance_matches_direct_block_calculation(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    assert model.posterior_coefficients is not None
    assert model.posterior_sigmas is not None
    assert model.posterior_terminal_states is not None
    coefficients = model.posterior_coefficients[0]
    sigma = model.posterior_sigmas[0]
    horizon = 5
    component = model._forecast_component(
        coefficients, sigma, model.posterior_terminal_states[0], horizon
    )
    blocks = [response @ component.innovation_factor for response in component.responses]
    n = len(model.variable_ids)
    loading = np.zeros((horizon * n, horizon * n))
    for forecast_step in range(horizon):
        for shock_time in range(forecast_step + 1):
            loading[
                forecast_step * n : (forecast_step + 1) * n,
                shock_time * n : (shock_time + 1) * n,
            ] = blocks[forecast_step - shock_time]
    actual = loading @ loading.T

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


def test_state_regression_uses_every_transition_in_the_kalman_model() -> None:
    model = BVAR()
    periods = 7
    state_width = len(model.variable_ids) * model.config.lags
    states = np.arange(periods * state_width, dtype=float).reshape(periods, state_width)
    dates = pd.date_range("2019-01-01", periods=periods, freq="MS")

    design, targets = model._state_design_matrix(states, dates)

    assert len(design) == periods - 1
    assert np.array_equal(design[0, 1 : 1 + state_width], states[0])
    assert np.array_equal(targets[0], states[1, : len(model.variable_ids)])


def test_quarterly_measurement_aggregates_three_latent_gdp_months() -> None:
    row = np.full(len(SERIES_SPECS), np.nan)
    row[0] = 1.5
    observed, measurement, covariance = observation_system(
        row,
        tuple(SERIES_BY_ID.values()),
        lags=4,
        monthly_measurement_variance=1e-6,
        quarterly_measurement_variance=1e-4,
    )

    assert observed[0] == pytest.approx(1.5)
    variables = len(SERIES_SPECS)
    gdp_positions = [0, variables, variables * 2]
    assert np.allclose(measurement[0, gdp_positions], 1.0 / 3.0)
    assert np.count_nonzero(measurement[0]) == 3
    assert covariance[0, 0] == pytest.approx(1e-4)


def test_early_growth_anchor_uses_displayed_fixed_history(synthetic_levels, fitted_model) -> None:
    model = fitted_model
    assert model.history_levels is not None
    assert model.history_model is not None
    original_history_model = model.history_model
    model.history_model = original_history_model + 100.0
    try:
        scenario = model.forecast(
            horizon=6,
            draws=20,
            constraints={(2, "GDPC1"): ScenarioConstraint(1.5, "qoq")},
            seed=29,
        )
    finally:
        model.history_model = original_history_model

    history = cast(pd.Series, model.history_levels.loc[:, "GDPC1"])
    forecast = cast(pd.Series, scenario.median.loc[:, "GDPC1"])
    path = cast(pd.Series, pd.concat([history, forecast]))
    transformed = transform_path(path, SERIES_BY_ID["GDPC1"], "qoq")
    assert transformed.loc[scenario.dates[2]] == pytest.approx(1.5)


def test_latent_gdp_growth_constraint_uses_smoothed_monthly_history(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    scenario = model.forecast(
        horizon=6,
        draws=20,
        constraints={(2, "GDPC1"): ScenarioConstraint(1.5, "qoq")},
        seed=29,
    )
    assert model.history_levels is not None
    history = cast(pd.Series, model.history_levels.loc[:, "GDPC1"])
    forecast = cast(pd.Series, scenario.median.loc[:, "GDPC1"])
    path = cast(pd.Series, pd.concat([history, forecast]))
    transformed = transform_path(path, SERIES_BY_ID["GDPC1"], "qoq")

    assert transformed.loc[scenario.dates[2]] == pytest.approx(1.5)


def test_paired_terminal_states_drive_their_parameter_components(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    assert model.posterior_state_paths is not None
    assert model.posterior_coefficients is not None
    assert model.posterior_sigmas is not None
    assert model.posterior_terminal_states is not None
    variables = len(model.variable_ids)
    assert len(model.posterior_terminal_states) == len(model.posterior_state_paths)
    for index, terminal_state in enumerate(model.posterior_terminal_states):
        assert np.array_equal(terminal_state[:variables], model.posterior_state_paths[index, -1])

    terminal_state = model.posterior_terminal_states[0]
    shifted_terminal_state = terminal_state.copy()
    shifted_terminal_state[0] += 1.0
    original = model._forecast_component(
        model.posterior_coefficients[0], model.posterior_sigmas[0], terminal_state, horizon=2
    )
    shifted = model._forecast_component(
        model.posterior_coefficients[0],
        model.posterior_sigmas[0],
        shifted_terminal_state,
        horizon=2,
    )
    assert not np.allclose(original.mean, shifted.mean)


def test_invalid_state_space_configuration_is_rejected(synthetic_levels) -> None:
    model = BVAR(config=BVARConfig(quarterly_measurement_variance=-1.0))
    with pytest.raises(ValueError, match="variances must be positive"):
        model.fit(synthetic_levels)


def test_negative_innovation_prior_strength_is_rejected(synthetic_levels) -> None:
    model = BVAR(config=BVARConfig(innovation_prior_strength=-1.0))
    with pytest.raises(ValueError, match="covariance prior strength"):
        model.fit(synthetic_levels)


def test_invalid_unstable_draw_fraction_is_rejected(synthetic_levels) -> None:
    model = BVAR(config=BVARConfig(maximum_unstable_draw_fraction=1.0))
    with pytest.raises(ValueError, match="unstable-draw fraction"):
        model.fit(synthetic_levels)


def test_fit_handles_a_monthly_ragged_edge(synthetic_levels) -> None:
    ragged = synthetic_levels.copy()
    ragged.loc[ragged.index[-2:], "PCEC96"] = np.nan
    ragged.loc[ragged.index[-1:], "CPIAUCSL"] = np.nan

    model = BVAR(config=BVARConfig.quick()).fit(ragged)

    assert model.history_levels is not None
    assert model.observation_mask is not None
    assert model.history_levels.index[-1] == ragged.index[-1]
    assert not model.observation_mask.loc[ragged.index[-1], "PCEC96"]
    assert np.isfinite(model.history_levels.iloc[-1]).all()


def test_forecast_component_repairs_covariance_only_after_cholesky_failure(
    fitted_model, monkeypatch
) -> None:
    model = fitted_model
    assert model.posterior_coefficients is not None
    assert model.posterior_sigmas is not None
    assert model.posterior_terminal_states is not None

    def unexpected_repair(matrix: np.ndarray) -> np.ndarray:
        raise AssertionError("A positive-definite covariance should not be repaired.")

    monkeypatch.setattr(model_module, "nearest_positive_definite", unexpected_repair)
    component = model._forecast_component(
        model.posterior_coefficients[0],
        model.posterior_sigmas[0],
        model.posterior_terminal_states[0],
        horizon=2,
    )

    assert component.innovation_factor.shape == model.posterior_sigmas[0].shape


def test_compact_loading_conditioning_hits_every_target(fitted_model) -> None:
    model = fitted_model
    component = model._forecast_components(3)[0]
    variables = len(model.variable_ids)
    constraints = np.zeros((2, 3 * variables))
    constraints[0, 1] = 1.0
    constraints[1, 2 * variables + 4] = 1.0
    targets = constraints @ component.mean.reshape(-1) + np.asarray([0.5, -0.25])
    samples = component.mean[None, :, :] + model._apply_component_loading(
        component, np.random.default_rng(101).standard_normal((5, 3, variables))
    )

    conditioned = model._condition_component_samples(
        component,
        samples,
        constraints,
        targets,
        (np.eye(2), True),
    )

    assert np.allclose(conditioned.reshape(5, -1) @ constraints.T, targets)


def test_compact_component_weights_match_full_loading_calculation(fitted_model) -> None:
    model = fitted_model
    components = model._forecast_components(2)[:3]
    variables = len(model.variable_ids)
    constraints = np.zeros((2, 2 * variables))
    constraints[0, 0] = 1.0
    constraints[1, variables + 5] = 1.0
    targets = np.asarray([0.1, 0.2])

    actual = model._constraint_component_probabilities(components, constraints, targets)
    log_weights = []
    for component in components:
        loading = np.zeros((2 * variables, 2 * variables))
        blocks = [response @ component.innovation_factor for response in component.responses]
        loading[:variables, :variables] = blocks[0]
        loading[variables:, :variables] = blocks[1]
        loading[variables:, variables:] = blocks[0]
        projected = constraints @ loading
        covariance = projected @ projected.T
        difference = targets - constraints @ component.mean.reshape(-1)
        sign, log_determinant = np.linalg.slogdet(covariance)
        assert sign > 0
        log_weights.append(
            -0.5
            * (
                len(targets) * np.log(2.0 * np.pi)
                + log_determinant
                + difference @ np.linalg.solve(covariance, difference)
            )
        )
    expected = np.exp(np.asarray(log_weights) - np.max(log_weights))
    expected /= expected.sum()

    assert np.allclose(actual, expected)
