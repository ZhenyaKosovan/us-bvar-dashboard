from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, lstsq, solve_triangular
from scipy.stats import invwishart

from us_bvar.config import (
    PANDEMIC_CONTROL_MONTHS,
    SERIES_SPECS,
    ConvergencePolicy,
    SeriesSpec,
)
from us_bvar.transforms import (
    PLOT_TRANSFORMATIONS,
    LevelTransformer,
    ScenarioConstraint,
    transformation_spec,
)

from .state_space import (  # type: ignore[import-not-found]
    companion_radius,
    companion_transition,
    control_vector,
    kalman_filter,
    nearest_positive_definite,
    simulation_smoother,
)


def _lower_cholesky_factor(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    factor, _ = cho_factor(matrix, lower=True, check_finite=False)
    return np.asarray(factor), True


@dataclass(frozen=True)
class BVARConfig:
    lags: int = 4
    tightness: float = 0.02
    lag_decay: float = 1.0
    own_lag_prior_mean: float = 0.90
    innovation_prior_strength: float = 50.0
    pandemic_months: tuple[str, ...] = PANDEMIC_CONTROL_MONTHS
    interval: tuple[float, float] = (0.16, 0.84)
    mcmc_iterations: int = 1200
    burn_in: int = 600
    thin: int = 3
    mcmc_chains: int = 4
    estimation_seed: int = 202504
    monthly_measurement_variance: float = 1e-6
    quarterly_measurement_variance: float = 1e-4
    diffuse_state_variance: float = 10.0
    max_companion_radius: float = 1.10
    maximum_unstable_draw_fraction: float = 0.25
    maximum_mcmc_r_hat: float = 1.10
    minimum_mcmc_effective_sample_size: float = 20.0
    maximum_absolute_mcmc_r_hat: float = 1.50
    minimum_absolute_mcmc_effective_sample_size: float = 10.0
    maximum_mcmc_tail_count: int = 25
    maximum_mcmc_tail_fraction: float = 0.01
    minimum_component_effective_sample_size: float = 1.0
    minimum_component_effective_fraction: float = 0.05
    quick_mode: bool = False
    minimum_retained_draws_per_chain: int = 20

    @property
    def convergence_policy(self) -> ConvergencePolicy:
        """Return the centralized MCMC release policy for this configuration."""

        return ConvergencePolicy(
            nominal_r_hat=self.maximum_mcmc_r_hat,
            nominal_effective_sample_size=self.minimum_mcmc_effective_sample_size,
            absolute_maximum_r_hat=self.maximum_absolute_mcmc_r_hat,
            absolute_minimum_effective_sample_size=self.minimum_absolute_mcmc_effective_sample_size,
            maximum_tail_count=self.maximum_mcmc_tail_count,
            maximum_tail_fraction=self.maximum_mcmc_tail_fraction,
        )

    @classmethod
    def quick(cls, *, mcmc_chains: int = 1) -> BVARConfig:
        """Return the intentionally undersized configuration for deterministic tests."""

        return cls(
            mcmc_iterations=8,
            burn_in=4,
            thin=1,
            mcmc_chains=mcmc_chains,
            quick_mode=True,
        )


def history_semantics_metadata() -> dict[str, object]:
    """Describe the published-history contract without implying historical draws.

    The displayed history is a fixed posterior summary used by early growth anchors.
    Forecast components still retain their paired terminal state, coefficients, and
    innovation covariance, so terminal-state uncertainty continues into dynamics.
    """

    return {
        "contract_version": 1,
        "mode": "fixed_published_history",
        "early_growth_anchors": "displayed_fixed_history",
        "terminal_state_pairing": "same_posterior_draw_as_forecast_parameters",
        "terminal_state_affects_forecast_dynamics": True,
        "paired_historical_draws": False,
    }


@dataclass(frozen=True, slots=True)
class _ForecastComponent:
    mean: np.ndarray
    responses: np.ndarray
    innovation_factor: np.ndarray


@dataclass(frozen=True)
class ForecastResult:
    dates: pd.DatetimeIndex
    median: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    samples: np.ndarray
    constraints: Mapping[tuple[int, str], ScenarioConstraint]
    draws: int
    interval: tuple[float, float]
    component_effective_sample_size: float | None = None

    @property
    def is_scenario(self) -> bool:
        return bool(self.constraints)


class BVAR:
    """Bayesian mixed-frequency VAR with latent monthly GDP.

    Every variable evolves at a monthly frequency in a companion-state VAR.
    Monthly releases are direct measurements; quarterly GDP is a one-third
    aggregation of the current and preceding two latent monthly log levels.
    A Gibbs sampler alternates Kalman simulation smoothing and a conjugate
    matrix-normal/inverse-Wishart transition-system draw. The published history is
    deliberately fixed: early growth anchors use the displayed history, not paired
    historical posterior draws. Forecast components nevertheless pair each retained
    parameter draw with its terminal state, so terminal-state uncertainty affects
    forecast dynamics.
    Six pre-estimated pandemic-month nuisance effects remain fixed during sampling.
    """

    def __init__(
        self,
        specs: tuple[SeriesSpec, ...] = SERIES_SPECS,
        config: BVARConfig | None = None,
    ) -> None:
        self.specs = specs
        self.config = config or BVARConfig()
        self.variable_ids = tuple(spec.series_id for spec in specs)
        self.transformer: LevelTransformer | None = None
        self.observed_levels: pd.DataFrame | None = None
        self.observation_mask: pd.DataFrame | None = None
        self.history_levels: pd.DataFrame | None = None
        self.history_model: pd.DataFrame | None = None
        self.posterior_coefficients: np.ndarray | None = None
        self.posterior_sigmas: np.ndarray | None = None
        self.posterior_terminal_states: np.ndarray | None = None
        self.posterior_state_paths: np.ndarray | None = None
        self.posterior_mean: np.ndarray | None = None
        self.mcmc_log_likelihood: np.ndarray | None = None
        self.companion_radii: np.ndarray | None = None
        self.mcmc_chain_ids: np.ndarray | None = None
        self.convergence_diagnostics: dict[str, object] | None = None
        self.fixed_control_coefficients: np.ndarray | None = None
        self.unstable_draws_rejected = 0
        self.retention_attempts = 0
        self._n_controls = len(self.config.pandemic_months)
        self._forecast_component_cache: dict[int, tuple[_ForecastComponent, ...]] = {}

    def fit(self, levels: pd.DataFrame) -> BVAR:
        ids = list(self.variable_ids)
        if not isinstance(levels.index, pd.DatetimeIndex):
            raise TypeError("The history index must be a DatetimeIndex.")
        if missing := set(ids).difference(levels.columns):
            raise ValueError(f"Missing model variables: {sorted(missing)}")
        if self.config.lags < 3:
            raise ValueError("The mixed-frequency model requires at least three lags.")
        self._validate_config()
        self.unstable_draws_rejected = 0
        self.retention_attempts = 0
        if self.config.mcmc_iterations <= self.config.burn_in:
            raise ValueError("MCMC iterations must exceed burn-in.")

        observed = levels.loc[:, ids].sort_index().copy()
        if observed.index.has_duplicates:
            observed = observed.groupby(level=0).last()
        expected_index = pd.date_range(observed.index[0], observed.index[-1], freq="MS")
        observed = observed.reindex(expected_index)
        for spec in self.specs:
            if spec.frequency == "quarterly":
                non_quarter_end = observed.index.month % 3 != 0
                observed.loc[non_quarter_end, spec.series_id] = np.nan
        if len(observed) <= self.config.lags + len(ids) + 2:
            raise ValueError("Too few monthly calendar observations to estimate the BVAR.")
        if observed.notna().sum().min() < 8:
            raise ValueError("Every model variable needs at least eight observed releases.")

        self.transformer = LevelTransformer.fit(observed, self.specs)
        model_observations = self.transformer.encode_frame(observed)
        initial_path = model_observations.interpolate(method="linear", limit_direction="both")
        if initial_path.isna().to_numpy().any():
            raise ValueError("The mixed-frequency panel could not be initialized.")

        x, y = self._design_matrix(initial_path)
        variables = len(ids)
        regressors = x.shape[1]
        ridge_scale = np.sqrt(1e-8)
        augmented_x = np.vstack((x, ridge_scale * np.eye(regressors)))
        augmented_y = np.vstack((y, np.zeros((regressors, variables))))
        coefficients = lstsq(augmented_x, augmented_y, check_finite=False)[0]
        residuals = y - x @ coefficients
        sigma = nearest_positive_definite(
            residuals.T @ residuals / max(len(y) - regressors, variables + 2)
        )
        prior_mean = self._minnesota_prior_mean(regressors, variables)
        transition_regressors = 1 + variables * self.config.lags
        fixed_controls = coefficients[transition_regressors:].copy()
        self.fixed_control_coefficients = fixed_controls
        transition_prior_variance = self._conjugate_prior_variance(transition_regressors, variables)
        prior_df = variables + 2 + self.config.innovation_prior_strength
        prior_target = np.diag(np.diag(sigma))
        prior_scale = (prior_df - variables - 1) * prior_target

        first = initial_path.iloc[0].to_numpy(dtype=float)
        initial_mean = np.tile(first, self.config.lags)
        initial_covariance = (
            np.eye(variables * self.config.lags) * self.config.diffuse_state_variance
        )
        initial_coefficients = coefficients.copy()
        initial_sigma = sigma.copy()
        retained_coefficients: list[np.ndarray] = []
        retained_sigmas: list[np.ndarray] = []
        retained_terminal_states: list[np.ndarray] = []
        retained_paths: list[np.ndarray] = []
        retained_likelihoods: list[float] = []
        retained_radii: list[float] = []
        retained_chain_ids: list[int] = []

        for chain in range(self.config.mcmc_chains):
            coefficients = initial_coefficients.copy()
            sigma = initial_sigma.copy()
            pending_likelihood_index: int | None = None
            rng = np.random.default_rng(self.config.estimation_seed + 104_729 * chain)
            for iteration in range(self.config.mcmc_iterations):
                filtered = kalman_filter(
                    model_observations,
                    coefficients,
                    sigma,
                    self.specs,
                    self.config.lags,
                    self.config.pandemic_months,
                    initial_mean,
                    initial_covariance,
                    self.config.monthly_measurement_variance,
                    self.config.quarterly_measurement_variance,
                )
                if pending_likelihood_index is not None:
                    retained_likelihoods[pending_likelihood_index] = filtered.log_likelihood
                    pending_likelihood_index = None
                transition, _ = companion_transition(
                    coefficients, sigma, variables, self.config.lags
                )
                states = simulation_smoother(filtered, transition, variables, rng)
                x, y = self._state_design_matrix(states, pd.DatetimeIndex(model_observations.index))
                adjusted_y = y - x[:, transition_regressors:] @ fixed_controls
                transition_coefficients, sigma = self._sample_conjugate_parameters(
                    rng,
                    x[:, :transition_regressors],
                    adjusted_y,
                    prior_mean[:transition_regressors],
                    transition_prior_variance,
                    prior_scale,
                    prior_df,
                )
                coefficients = np.vstack([transition_coefficients, fixed_controls])

                if (
                    iteration >= self.config.burn_in
                    and (iteration - self.config.burn_in) % self.config.thin == 0
                ):
                    self.retention_attempts += 1
                    radius = companion_radius(coefficients, variables, self.config.lags)
                    if radius > self.config.max_companion_radius:
                        self.unstable_draws_rejected += 1
                        continue
                    retained_coefficients.append(coefficients.copy())
                    retained_sigmas.append(sigma.copy())
                    retained_terminal_states.append(states[-1].copy())
                    retained_paths.append(states[:, :variables].copy())
                    retained_likelihoods.append(np.nan)
                    pending_likelihood_index = len(retained_likelihoods) - 1
                    retained_radii.append(radius)
                    retained_chain_ids.append(chain)
            if pending_likelihood_index is not None:
                final_filter = kalman_filter(
                    model_observations,
                    coefficients,
                    sigma,
                    self.specs,
                    self.config.lags,
                    self.config.pandemic_months,
                    initial_mean,
                    initial_covariance,
                    self.config.monthly_measurement_variance,
                    self.config.quarterly_measurement_variance,
                )
                retained_likelihoods[pending_likelihood_index] = final_filter.log_likelihood

        if not retained_coefficients:
            raise RuntimeError("The Gibbs sampler retained no posterior draws.")
        if not self.config.quick_mode:
            retained_per_chain = np.bincount(
                np.asarray(retained_chain_ids, dtype=int), minlength=self.config.mcmc_chains
            )
            if np.min(retained_per_chain) < self.config.minimum_retained_draws_per_chain:
                raise RuntimeError(
                    "The fitted posterior is not forecast-ready: every production chain must "
                    f"retain at least {self.config.minimum_retained_draws_per_chain} draws. "
                    "Use BVARConfig.quick() only for explicitly fast test fits."
                )
        unstable_fraction = self.unstable_draws_rejected / self.retention_attempts
        if unstable_fraction > self.config.maximum_unstable_draw_fraction:
            raise RuntimeError(
                "The Gibbs sampler rejected too many unstable draws "
                f"({unstable_fraction:.1%}; maximum "
                f"{self.config.maximum_unstable_draw_fraction:.1%}). Increase shrinkage or "
                "review data."
            )
        self.posterior_coefficients = np.stack(retained_coefficients)
        self.posterior_sigmas = np.stack(retained_sigmas)
        self.posterior_terminal_states = np.stack(retained_terminal_states)
        self.posterior_state_paths = np.stack(retained_paths)
        self.posterior_mean = self.posterior_coefficients.mean(axis=0)
        self.mcmc_log_likelihood = np.asarray(retained_likelihoods)
        self.companion_radii = np.asarray(retained_radii)
        self.mcmc_chain_ids = np.asarray(retained_chain_ids, dtype=int)
        if not np.isfinite(self.mcmc_log_likelihood).all():
            raise RuntimeError("The Gibbs sampler produced non-finite likelihood diagnostics.")
        if np.max(self.companion_radii) > self.config.max_companion_radius:
            raise RuntimeError(
                "The Gibbs sampler retained an explosive VAR draw; increase shrinkage or "
                "review data."
            )

        smoothed_model = self.posterior_state_paths.mean(axis=0)
        for variable, spec in enumerate(self.specs):
            if spec.frequency == "monthly":
                available = model_observations.iloc[:, variable].notna().to_numpy()
                smoothed_model[available, variable] = model_observations.iloc[
                    available, variable
                ].to_numpy(dtype=float)
        self.history_model = pd.DataFrame(
            smoothed_model,
            index=observed.index,
            columns=pd.Index(ids, dtype="object"),
        )
        decoded = self.transformer.decode_array(smoothed_model)
        self.history_levels = pd.DataFrame(
            decoded,
            index=observed.index,
            columns=pd.Index(ids, dtype="object"),
        )
        self.observed_levels = observed
        self.observation_mask = observed.notna()
        return self

    def forecast(
        self,
        horizon: int = 12,
        draws: int = 500,
        constraints: Mapping[tuple[int, str], float | ScenarioConstraint] | None = None,
        seed: int = 202503,
    ) -> ForecastResult:
        self._require_fit()
        if horizon < 1 or draws < 20:
            raise ValueError(
                "Forecast horizon must be positive and at least 20 draws are required."
            )
        self._require_forecast_ready()
        normalized_constraints = self._normalize_constraints(constraints or {})
        constraint_matrix, targets = self._constraint_system(normalized_constraints, horizon)
        rng = np.random.default_rng(seed)
        variables = len(self.variable_ids)
        components = self._forecast_components(horizon)
        component_ess: float | None = None
        if len(targets):
            probabilities = self._constraint_component_probabilities(
                components, constraint_matrix, targets
            )
            effective_ess = cast(float, np.asarray(1.0 / np.sum(probabilities**2)).item())
            component_ess = effective_ess
            minimum_ess = max(
                self.config.minimum_component_effective_sample_size,
                self.config.minimum_component_effective_fraction * len(components),
            )
            supported_components = np.count_nonzero(probabilities > 1e-12)
            if effective_ess < minimum_ess or supported_components < 2:
                raise ValueError(
                    "Scenario restrictions concentrate on too few posterior components "
                    f"(effective components {effective_ess:.1f}; require {minimum_ess:.1f}). "
                    "Use less extreme assumptions or rebuild with more retained MCMC draws."
                )
            component_indices = rng.choice(
                len(components), size=draws, replace=True, p=probabilities
            )
        else:
            component_indices = rng.integers(0, len(components), size=draws)

        standard_normals = rng.standard_normal((draws, horizon, variables))
        simulations = np.empty((draws, horizon, variables))
        correction_factor: tuple[np.ndarray, bool] | None = None
        if len(targets):
            try:
                correction_factor = _lower_cholesky_factor(constraint_matrix @ constraint_matrix.T)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        for component_index in np.unique(component_indices):
            positions = np.flatnonzero(component_indices == component_index)
            component = components[component_index]
            samples = component.mean[None, :, :] + self._apply_component_loading(
                component, standard_normals[positions]
            )
            if len(targets):
                samples = self._condition_component_samples(
                    component,
                    samples,
                    constraint_matrix,
                    targets,
                    cast(tuple[np.ndarray, bool], correction_factor),
                )
            simulations[positions] = samples

        transformer = cast(LevelTransformer, self.transformer)
        history_levels = cast(pd.DataFrame, self.history_levels)
        natural_draws = transformer.decode_array(simulations)
        if not np.isfinite(natural_draws).all():
            raise ValueError(
                "The scenario produced non-finite forecast values. Use less extreme constraints."
            )
        natural_draws.setflags(write=False)
        lower_q, upper_q = self.config.interval
        history_end = cast(pd.Timestamp, pd.DatetimeIndex(history_levels.index)[-1])
        dates = pd.date_range(
            history_end + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        def frame(values: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(
                values,
                index=dates,
                columns=pd.Index(self.variable_ids, dtype="object"),
            )

        return ForecastResult(
            dates=dates,
            median=frame(np.median(natural_draws, axis=0)),
            lower=frame(np.quantile(natural_draws, lower_q, axis=0)),
            upper=frame(np.quantile(natural_draws, upper_q, axis=0)),
            samples=natural_draws,
            constraints=normalized_constraints,
            draws=draws,
            interval=self.config.interval,
            component_effective_sample_size=component_ess,
        )

    def _validate_config(self) -> None:
        numeric = np.asarray(
            [
                self.config.tightness,
                self.config.lag_decay,
                self.config.own_lag_prior_mean,
                self.config.innovation_prior_strength,
                self.config.monthly_measurement_variance,
                self.config.quarterly_measurement_variance,
                self.config.diffuse_state_variance,
                self.config.max_companion_radius,
                self.config.maximum_unstable_draw_fraction,
                self.config.maximum_mcmc_r_hat,
                self.config.minimum_mcmc_effective_sample_size,
                self.config.maximum_absolute_mcmc_r_hat,
                self.config.minimum_absolute_mcmc_effective_sample_size,
                self.config.maximum_mcmc_tail_fraction,
                self.config.minimum_component_effective_sample_size,
                self.config.minimum_component_effective_fraction,
                *self.config.interval,
            ],
            dtype=float,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("Every numerical BVAR setting must be finite.")
        if self.config.lags < 3:
            raise ValueError("The mixed-frequency model requires at least three lags.")
        if self.config.thin < 1 or self.config.mcmc_chains < 1 or self.config.burn_in < 0:
            raise ValueError("MCMC thinning and chains must be positive and burn-in non-negative.")
        if self.config.mcmc_iterations <= self.config.burn_in:
            raise ValueError("MCMC iterations must exceed burn-in.")
        scheduled_draws = (
            self.config.mcmc_iterations - self.config.burn_in + self.config.thin - 1
        ) // self.config.thin
        if self.config.minimum_retained_draws_per_chain < 1:
            raise ValueError("The minimum retained draws per chain must be positive.")
        if not self.config.quick_mode and (
            self.config.mcmc_chains < 2
            or scheduled_draws < self.config.minimum_retained_draws_per_chain
        ):
            raise ValueError(
                "Production BVAR fits require at least two chains and "
                f"{self.config.minimum_retained_draws_per_chain} scheduled retained draws per "
                "chain. Use BVARConfig.quick() explicitly for fast test fits."
            )
        if self.config.tightness <= 0:
            raise ValueError("Prior tightness must be positive.")
        if self.config.lag_decay < 0:
            raise ValueError("Minnesota prior lag decay must be non-negative.")
        if not -1 < self.config.own_lag_prior_mean <= 1:
            raise ValueError("The own-lag prior mean must be in (-1, 1].")
        if self.config.innovation_prior_strength < 0:
            raise ValueError("Innovation covariance prior strength must be non-negative.")
        if (
            self.config.monthly_measurement_variance <= 0
            or self.config.quarterly_measurement_variance <= 0
            or self.config.diffuse_state_variance <= 0
        ):
            raise ValueError("State-space variances must be positive.")
        lower, upper = self.config.interval
        if not 0 < lower < upper < 1:
            raise ValueError("Forecast interval probabilities must satisfy 0 < lower < upper < 1.")
        if self.config.max_companion_radius < 1:
            raise ValueError("The companion-radius acceptance threshold must be at least one.")
        if not 0 <= self.config.maximum_unstable_draw_fraction < 1:
            raise ValueError("The maximum unstable-draw fraction must be in [0, 1).")
        if self.config.maximum_mcmc_r_hat < 1:
            raise ValueError("The MCMC R-hat acceptance threshold must be at least one.")
        if self.config.maximum_absolute_mcmc_r_hat < self.config.maximum_mcmc_r_hat:
            raise ValueError(
                "The absolute MCMC R-hat bound must not be below its nominal threshold."
            )
        if (
            self.config.minimum_absolute_mcmc_effective_sample_size
            > self.config.minimum_mcmc_effective_sample_size
        ):
            raise ValueError("The absolute MCMC ESS bound must not exceed its nominal threshold.")
        if self.config.maximum_mcmc_tail_count < 0 or not (
            0 < self.config.maximum_mcmc_tail_fraction <= 1
        ):
            raise ValueError("The MCMC nominal-tail limits are invalid.")
        if (
            self.config.minimum_mcmc_effective_sample_size <= 0
            or self.config.minimum_absolute_mcmc_effective_sample_size <= 0
            or self.config.minimum_component_effective_sample_size <= 0
            or not 0 < self.config.minimum_component_effective_fraction <= 1
        ):
            raise ValueError("MCMC and scenario effective-sample thresholds are invalid.")

    def _state_design_matrix(
        self, states: np.ndarray, dates: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        variables = len(self.variable_ids)
        state_width = variables * self.config.lags
        if states.ndim != 2 or states.shape[1] != state_width or len(states) != len(dates):
            raise ValueError("Smoothed companion states have an unexpected shape.")
        rows: list[np.ndarray] = []
        for time in range(1, len(states)):
            controls = control_vector(dates[time], self.config.pandemic_months)
            rows.append(np.concatenate(([1.0], states[time - 1, :state_width], controls)))
        return np.vstack(rows), states[1:, :variables]

    def _design_matrix(self, model_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = model_frame.to_numpy(dtype=float)
        dates = model_frame.index
        p = self.config.lags
        rows: list[np.ndarray] = []
        for time in range(p, len(values)):
            lagged = np.concatenate([values[time - lag] for lag in range(1, p + 1)])
            controls = control_vector(dates[time], self.config.pandemic_months)
            rows.append(np.concatenate(([1.0], lagged, controls)))
        return np.vstack(rows), values[p:]

    def _minnesota_prior_mean(self, regressors: int, variables: int) -> np.ndarray:
        """Return the coefficient mean for the conjugate Minnesota-style prior."""

        mean = np.zeros((regressors, variables))
        for source in range(variables):
            row = 1 + source
            mean[row, source] = self.config.own_lag_prior_mean
        return mean

    def _conjugate_prior_variance(self, regressors: int, variables: int) -> np.ndarray:
        """Return the row covariance for the conjugate Minnesota-style prior."""

        variance = np.empty(regressors)
        variance[0] = 10.0**2
        for lag in range(1, self.config.lags + 1):
            start = 1 + (lag - 1) * variables
            stop = start + variables
            prior_sd = self.config.tightness / (lag**self.config.lag_decay)
            variance[start:stop] = prior_sd**2
        variance[1 + self.config.lags * variables :] = 10.0**2
        return variance

    @classmethod
    def _sample_conjugate_parameters(
        cls,
        rng: np.random.Generator,
        x: np.ndarray,
        y: np.ndarray,
        prior_mean: np.ndarray,
        prior_variance: np.ndarray,
        prior_scale: np.ndarray,
        prior_df: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw a complete VAR transition system from its conjugate posterior."""

        regressors, variables = prior_mean.shape
        if (
            x.shape[1] != regressors
            or y.shape[1] != variables
            or prior_variance.shape != (regressors,)
        ):
            raise ValueError("Conjugate sampler inputs have incompatible dimensions.")
        prior_precision = 1.0 / prior_variance
        precision = x.T @ x
        precision.flat[:: regressors + 1] += prior_precision
        rhs = x.T @ y + prior_precision[:, None] * prior_mean
        factor = cho_factor(precision, lower=True, check_finite=False)
        posterior_mean = cho_solve(factor, rhs, check_finite=False)
        residuals = y - x @ posterior_mean
        prior_deviation = posterior_mean - prior_mean
        posterior_scale = nearest_positive_definite(
            prior_scale
            + residuals.T @ residuals
            + prior_deviation.T @ (prior_precision[:, None] * prior_deviation)
        )
        sigma = np.atleast_2d(
            np.asarray(
                invwishart.rvs(
                    df=prior_df + len(y),
                    scale=posterior_scale,
                    random_state=rng,
                ),
                dtype=float,
            )
        )
        sigma = (sigma + sigma.T) / 2.0
        try:
            sigma_factor = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            del exc
            sigma_factor = np.linalg.cholesky(nearest_positive_definite(sigma))
        scaled_noise = rng.standard_normal((regressors, variables)) @ sigma_factor.T
        coefficients = posterior_mean + solve_triangular(
            factor[0].T,
            scaled_noise,
            lower=False,
            check_finite=False,
        )
        return coefficients, sigma

    def prepare_forecast_cache(self, horizon: int) -> None:
        """Build compact immutable forecast structures before serving concurrent requests."""

        if horizon < 1:
            raise ValueError("Forecast horizon must be positive.")
        self._forecast_components(horizon)

    def clear_derived_cache(self) -> None:
        """Drop immutable forecast structures that can be rebuilt from posterior draws."""

        self._forecast_component_cache = {}

    def _forecast_components(self, horizon: int) -> tuple[_ForecastComponent, ...]:
        self._require_fit()
        cache = getattr(self, "_forecast_component_cache", None)
        if cache is None:
            cache = {}
            self._forecast_component_cache = cache
        cached = cache.get(horizon)
        if cached is not None:
            return cached
        coefficients = cast(np.ndarray, self.posterior_coefficients)
        sigmas = cast(np.ndarray, self.posterior_sigmas)
        terminal_states = cast(np.ndarray, self.posterior_terminal_states)
        components = tuple(
            self._forecast_component(coefficients_draw, sigma, terminal_state, horizon)
            for coefficients_draw, sigma, terminal_state in zip(
                coefficients, sigmas, terminal_states, strict=True
            )
        )
        cache[horizon] = components
        return components

    def _forecast_component(
        self,
        coefficients: np.ndarray,
        sigma: np.ndarray,
        terminal_state: np.ndarray,
        horizon: int,
    ) -> _ForecastComponent:
        variables = len(self.variable_ids)
        lags = self.config.lags
        history = [
            terminal_state[lag * variables : (lag + 1) * variables].copy()
            for lag in range(lags - 1, -1, -1)
        ]
        means = np.empty((horizon, variables))
        for step in range(horizon):
            lagged = np.concatenate([history[-lag] for lag in range(1, lags + 1)])
            x = np.concatenate(([1.0], lagged, np.zeros(self._n_controls)))
            means[step] = x @ coefficients
            history.append(means[step])

        autoregressive = np.stack(
            [
                coefficients[1 + lag * variables : 1 + (lag + 1) * variables, :].T
                for lag in range(lags)
            ]
        )
        responses = np.empty((horizon, variables, variables))
        responses[0] = np.eye(variables)
        for forecast_step in range(1, horizon):
            responses[forecast_step] = 0.0
            for lag in range(1, min(lags, forecast_step) + 1):
                responses[forecast_step] += autoregressive[lag - 1] @ responses[forecast_step - lag]
        try:
            innovation_factor = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError as exc:
            del exc
            innovation_factor = np.linalg.cholesky(nearest_positive_definite(sigma))
        for array in (means, responses, innovation_factor):
            array.setflags(write=False)
        return _ForecastComponent(means, responses, innovation_factor)

    @staticmethod
    def _apply_component_loading(
        component: _ForecastComponent, standard_normals: np.ndarray
    ) -> np.ndarray:
        innovations = standard_normals @ component.innovation_factor.T
        result = np.zeros_like(innovations)
        horizon = innovations.shape[1]
        for response_lag, response in enumerate(component.responses):
            result[:, response_lag:, :] += innovations[:, : horizon - response_lag, :] @ response.T
        return result

    @staticmethod
    def _project_component_loading(
        component: _ForecastComponent, constraint_matrix: np.ndarray
    ) -> np.ndarray:
        horizon, variables = component.mean.shape
        constraint_blocks = constraint_matrix.reshape(len(constraint_matrix), horizon, variables)
        projected = np.zeros((len(constraint_matrix), horizon, variables))
        loading_blocks = component.responses @ component.innovation_factor
        for response_lag, loading_block in enumerate(loading_blocks):
            projected[:, : horizon - response_lag, :] += (
                constraint_blocks[:, response_lag:, :] @ loading_block
            )
        return projected.reshape(len(constraint_matrix), horizon * variables)

    @staticmethod
    def _component_cross_covariance(
        component: _ForecastComponent, projected_loading: np.ndarray
    ) -> np.ndarray:
        horizon, variables = component.mean.shape
        projected_blocks = projected_loading.reshape(-1, horizon, variables).transpose(1, 2, 0)
        result = np.zeros((horizon, variables, len(projected_loading)))
        loading_blocks = component.responses @ component.innovation_factor
        for response_lag, loading_block in enumerate(loading_blocks):
            result[response_lag:] += np.einsum(
                "ij,sjm->sim",
                loading_block,
                projected_blocks[: horizon - response_lag],
                optimize=True,
            )
        return result.reshape(horizon * variables, len(projected_loading))

    @staticmethod
    def _normalize_constraints(
        constraints: Mapping[tuple[int, str], float | ScenarioConstraint],
    ) -> dict[tuple[int, str], ScenarioConstraint]:
        result: dict[tuple[int, str], ScenarioConstraint] = {}
        for key, raw_constraint in constraints.items():
            try:
                constraint = (
                    raw_constraint
                    if isinstance(raw_constraint, ScenarioConstraint)
                    else ScenarioConstraint(value=float(raw_constraint))
                )
                value = float(constraint.value)
            except (TypeError, ValueError) as exc:
                raise ValueError("Scenario values must be numeric.") from exc
            if constraint.transformation not in PLOT_TRANSFORMATIONS:
                raise ValueError(f"Unknown scenario transformation: {constraint.transformation}")
            if not np.isfinite(value):
                raise ValueError("Scenario values must be finite numbers.")
            if abs(value) > 1_000_000_000:
                raise ValueError("Scenario values must be no larger than 1 billion in magnitude.")
            result[key] = ScenarioConstraint(value=value, transformation=constraint.transformation)
        return result

    def _constraint_system(
        self,
        constraints: Mapping[tuple[int, str], ScenarioConstraint],
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build exact future constraints against the fixed displayed history.

        A growth constraint whose reference predates the forecast uses the natural
        value in ``history_levels``—the same fixed history shown by the dashboard—
        and encodes it to model space. It intentionally does not draw a historical
        state or use a component-specific terminal state; terminal-state uncertainty
        enters only through the paired forecast dynamics.
        """

        transformer = cast(LevelTransformer, self.transformer)
        history_levels = cast(pd.DataFrame, self.history_levels)
        variables = len(self.variable_ids)
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for (step, variable_id), constraint in constraints.items():
            if not 0 <= step < horizon:
                raise ValueError(f"Scenario step {step} is outside the forecast horizon.")
            if variable_id not in self.variable_ids:
                raise ValueError(f"Unknown scenario variable: {variable_id}")
            variable_index = self.variable_ids.index(variable_id)
            row = np.zeros(horizon * variables)
            row[step * variables + variable_index] = 1.0

            if constraint.transformation == "level":
                target = transformer.encode_value(variable_index, constraint.value)
            else:
                spec = self.specs[variable_index]
                display_spec = transformation_spec(constraint.transformation)
                if spec.transform == "log":
                    if constraint.value <= -100.0:
                        raise ValueError(
                            f"{spec.short_label} growth must be greater than -100 percent."
                        )
                    latent_change = np.log1p(constraint.value / 100.0)
                else:
                    latent_change = constraint.value
                if display_spec.annualized:
                    latent_change /= display_spec.annualization_factor
                model_change = latent_change / transformer.scales[variable_index]
                lag_step = step - display_spec.periods
                if lag_step >= 0:
                    row[lag_step * variables + variable_index] = -1.0
                    target = model_change
                else:
                    if abs(lag_step) > len(history_levels):
                        raise ValueError(
                            f"Not enough history to apply {display_spec.label} to "
                            f"{spec.short_label}."
                        )
                    try:
                        anchor_level = float(history_levels.iloc[lag_step, variable_index])
                        target = (
                            transformer.encode_value(variable_index, anchor_level) + model_change
                        )
                    except (TypeError, ValueError) as exc:
                        raise ValueError("Scenario history anchor could not be encoded.") from exc
            rows.append(row)
            try:
                targets.append(float(target))
            except (TypeError, ValueError) as exc:
                raise ValueError("Scenario target could not be represented numerically.") from exc

        return (
            np.vstack(rows) if rows else np.empty((0, horizon * variables)),
            np.asarray(targets, dtype=float),
        )

    def _constraint_component_probabilities(
        self,
        components: tuple[_ForecastComponent, ...],
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        log_weights = np.empty(len(components))
        constant = len(targets) * np.log(2.0 * np.pi)
        for index, component in enumerate(components):
            projected_loading = self._project_component_loading(component, constraint_matrix)
            projected_covariance = projected_loading @ projected_loading.T
            difference = targets - constraint_matrix @ component.mean.reshape(-1)
            try:
                factor = _lower_cholesky_factor(projected_covariance)
                quadratic = difference @ cho_solve(factor, difference, check_finite=False)
                log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()
            except np.linalg.LinAlgError as exc:
                raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
            log_weights[index] = -0.5 * (constant + log_determinant + quadratic)
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            raise ValueError("The scenario is too improbable for the retained posterior draws.")
        return weights / total

    def _condition_component_samples(
        self,
        component: _ForecastComponent,
        samples: np.ndarray,
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
        correction_factor: tuple[np.ndarray, bool],
    ) -> np.ndarray:
        projected_loading = self._project_component_loading(component, constraint_matrix)
        conditioned_system = projected_loading @ projected_loading.T
        cross_covariance = self._component_cross_covariance(component, projected_loading)
        try:
            factor = _lower_cholesky_factor(conditioned_system)
            flat_samples = samples.reshape(len(samples), -1)
            differences = targets[None, :] - flat_samples @ constraint_matrix.T
            adjustments = cho_solve(factor, differences.T, check_finite=False).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        flat_samples += adjustments @ cross_covariance.T
        residuals = targets[None, :] - flat_samples @ constraint_matrix.T
        needs_correction = np.any(np.abs(residuals) > 1e-12, axis=1)
        if np.any(needs_correction):
            corrections = cho_solve(
                correction_factor,
                residuals[needs_correction].T,
                check_finite=False,
            ).T
            flat_samples[needs_correction] += corrections @ constraint_matrix
        return flat_samples.reshape(samples.shape)

    def _require_fit(self) -> None:
        if (
            self.posterior_coefficients is None
            or self.posterior_sigmas is None
            or self.posterior_terminal_states is None
            or self.history_levels is None
            or self.history_model is None
            or self.transformer is None
            or self.mcmc_chain_ids is None
        ):
            raise RuntimeError("Call fit() before forecasting.")

    def _require_forecast_ready(self) -> None:
        if getattr(self.config, "quick_mode", False):
            return
        chain_ids = cast(np.ndarray, self.mcmc_chain_ids)
        minimum_draws = getattr(self.config, "minimum_retained_draws_per_chain", 20)
        retained_per_chain = np.bincount(chain_ids, minlength=self.config.mcmc_chains)
        if len(retained_per_chain) < 2 or np.min(retained_per_chain) < minimum_draws:
            raise RuntimeError(
                "The fitted posterior is not forecast-ready: production forecasts require "
                f"{minimum_draws} retained draws per chain. "
                "Use BVARConfig.quick() only for explicitly fast test fits."
            )
