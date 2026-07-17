from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
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
    observation_system,
    simulation_smoother,
)


@dataclass(frozen=True)
class BVARConfig:
    lags: int = 4
    tightness: float = 0.02
    lag_decay: float = 1.0
    own_lag_prior_mean: float = 0.90
    pandemic_months: tuple[str, ...] = PANDEMIC_CONTROL_MONTHS
    interval: tuple[float, float] = (0.16, 0.84)
    mcmc_iterations: int = 600
    burn_in: int = 300
    thin: int = 3
    mcmc_chains: int = 2
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
        ridge = np.eye(regressors) * 1e-8
        coefficients = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        residuals = y - x @ coefficients
        sigma = self._nearest_positive_definite(
            residuals.T @ residuals / max(len(y) - regressors, variables + 2)
        )
        prior_mean, _prior_variance = self._minnesota_prior(regressors, variables)
        transition_regressors = 1 + variables * self.config.lags
        fixed_controls = coefficients[transition_regressors:].copy()
        self.fixed_control_coefficients = fixed_controls
        transition_prior_variance = self._conjugate_prior_variance(transition_regressors, variables)
        prior_scale = sigma.copy()
        prior_df = variables + 2

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
                transition, _ = companion_transition(
                    coefficients, sigma, variables, self.config.lags
                )
                states = simulation_smoother(filtered, transition, rng)
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
                    diagnostic_filter = kalman_filter(
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
                    retained_coefficients.append(coefficients.copy())
                    retained_sigmas.append(sigma.copy())
                    retained_terminal_states.append(states[-1].copy())
                    retained_paths.append(states[:, :variables].copy())
                    retained_likelihoods.append(diagnostic_filter.log_likelihood)
                    retained_radii.append(radius)
                    retained_chain_ids.append(chain)

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
        simulations = np.empty((draws, horizon, len(self.variable_ids)))
        component_structures = [
            self._joint_forecast_structure(
                coefficients,
                sigma,
                horizon,
                terminal_state=terminal_state,
            )
            for coefficients, sigma, terminal_state in self._paired_forecast_components()
        ]
        component_ess: float | None = None
        if len(targets):
            probabilities = self._constraint_component_probabilities_from_loadings(
                component_structures, constraint_matrix, targets
            )
            effective_ess = cast(float, np.asarray(1.0 / np.sum(probabilities**2)).item())
            component_ess = effective_ess
            minimum_ess = max(
                self.config.minimum_component_effective_sample_size,
                self.config.minimum_component_effective_fraction * len(component_structures),
            )
            supported_components = np.count_nonzero(probabilities > 1e-12)
            if effective_ess < minimum_ess or supported_components < 2:
                raise ValueError(
                    "Scenario restrictions concentrate on too few posterior components "
                    f"(effective components {effective_ess:.1f}; require {minimum_ess:.1f}). "
                    "Use less extreme assumptions or rebuild with more retained MCMC draws."
                )
            component_indices = rng.choice(
                len(component_structures), size=draws, replace=True, p=probabilities
            )
        else:
            component_indices = rng.integers(0, len(component_structures), size=draws)
        for draw, component in enumerate(component_indices):
            mean, loading = component_structures[component]
            if len(targets):
                sample = self._conditional_sample_from_loading(
                    rng, mean, loading, constraint_matrix, targets
                )
            else:
                sample = mean + loading @ rng.standard_normal(mean.size)
            simulations[draw] = sample.reshape(horizon, len(self.variable_ids))

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

    @staticmethod
    def _observation_system(
        row: np.ndarray,
        specs: tuple[SeriesSpec, ...],
        lags: int,
        monthly_measurement_variance: float,
        quarterly_measurement_variance: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return observation_system(
            row,
            specs,
            lags,
            monthly_measurement_variance,
            quarterly_measurement_variance,
        )

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

    def _minnesota_prior(self, regressors: int, variables: int) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((regressors, variables))
        variance = np.empty((regressors, variables))
        variance[0, :] = 10.0**2
        for lag in range(1, self.config.lags + 1):
            for source in range(variables):
                row = 1 + (lag - 1) * variables + source
                for equation in range(variables):
                    prior_sd = self.config.tightness / (lag**self.config.lag_decay)
                    variance[row, equation] = prior_sd**2
                    if lag == 1 and source == equation:
                        mean[row, equation] = self.config.own_lag_prior_mean
        variance[1 + self.config.lags * variables :, :] = 10.0**2
        return mean, variance.reshape(-1, order="F")

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
        prior_df: int,
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
        prior_quadratic = prior_mean.T @ (prior_precision[:, None] * prior_mean)
        posterior_quadratic = posterior_mean.T @ precision @ posterior_mean
        posterior_scale = cls._nearest_positive_definite(
            prior_scale + y.T @ y + prior_quadratic - posterior_quadratic
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
        row_factor = solve_triangular(
            factor[0].T,
            np.eye(regressors),
            lower=False,
            check_finite=False,
        )
        sigma_factor = np.linalg.cholesky(cls._nearest_positive_definite(sigma))
        coefficients = (
            posterior_mean
            + row_factor @ rng.standard_normal((regressors, variables)) @ sigma_factor.T
        )
        return coefficients, sigma

    @staticmethod
    def _sample_coefficients(
        rng: np.random.Generator,
        x: np.ndarray,
        y: np.ndarray,
        sigma: np.ndarray,
        prior_mean: np.ndarray,
        prior_variance: np.ndarray,
        current_coefficients: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw one coefficient block with an equation-wise Gibbs sweep.

        A direct draw under an independent Minnesota prior factors a dense matrix
        with ``(regressors * variables)`` rows. That is harmless for a small
        teaching VAR but needlessly cubic for a medium-size system. Conditional on
        the other equations, the multivariate-normal likelihood is an ordinary
        Gaussian regression with the Schur-complement innovation variance. Sweeping
        over equations is an exact Gibbs update for the same posterior while only
        factoring ``regressors``-square systems.
        """

        regressors, variables = prior_mean.shape
        if x.shape[1] != regressors or y.shape[1] != variables:
            raise ValueError("Coefficient sampler inputs have incompatible dimensions.")
        variance = prior_variance.reshape((regressors, variables), order="F")
        coefficients = (
            np.asarray(current_coefficients, dtype=float).copy()
            if current_coefficients is not None
            else np.linalg.lstsq(x, y, rcond=None)[0]
        )
        if coefficients.shape != (regressors, variables):
            raise ValueError("Current coefficients have an incompatible shape.")

        x_crossproduct = x.T @ x
        all_equations = np.arange(variables)
        for equation in rng.permutation(variables):
            others = all_equations[all_equations != equation]
            if len(others):
                other_covariance = sigma[np.ix_(others, others)]
                loadings = np.linalg.solve(other_covariance, sigma[others, equation])
                try:
                    conditional_variance = float(
                        sigma[equation, equation] - sigma[equation, others] @ loadings
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError("Conditional innovation variance is not numeric.") from exc
                other_residuals = y[:, others] - x @ coefficients[:, others]
                adjusted_target = y[:, equation] - other_residuals @ loadings
            else:
                try:
                    conditional_variance = float(sigma[equation, equation])
                except (TypeError, ValueError) as exc:
                    raise ValueError("Conditional innovation variance is not numeric.") from exc
                adjusted_target = y[:, equation]
            conditional_variance = max(conditional_variance, np.finfo(float).eps)
            prior_precision = 1.0 / variance[:, equation]
            precision = x_crossproduct / conditional_variance
            precision.flat[:: regressors + 1] += prior_precision
            rhs = (
                x.T @ adjusted_target / conditional_variance
                + prior_precision * prior_mean[:, equation]
            )
            factor = cho_factor(precision, lower=True, check_finite=False)
            posterior_mean = cho_solve(factor, rhs, check_finite=False)
            innovation = solve_triangular(
                factor[0].T,
                rng.standard_normal(regressors),
                lower=False,
                check_finite=False,
            )
            coefficients[:, equation] = posterior_mean + innovation
        return coefficients

    def _draw_parameters(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        self._require_fit()
        coefficients = cast(np.ndarray, self.posterior_coefficients)
        sigmas = cast(np.ndarray, self.posterior_sigmas)
        index = rng.integers(len(coefficients)).item()
        return coefficients[index].copy(), sigmas[index].copy()

    def _paired_forecast_components(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return retained parameters paired with their retained terminal states.

        The index of a terminal state is the same posterior component index as its
        coefficients and innovation covariance. This pairing is used for forecast
        dynamics; it does not create paired draws for the fixed published history.
        """

        self._require_fit()
        coefficients = cast(np.ndarray, self.posterior_coefficients)
        sigmas = cast(np.ndarray, self.posterior_sigmas)
        terminal_states = cast(np.ndarray, self.posterior_terminal_states)
        return list(zip(coefficients, sigmas, terminal_states, strict=True))

    def _joint_forecast_moments(
        self,
        coefficients: np.ndarray,
        sigma: np.ndarray,
        horizon: int,
        terminal_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        means, loading = self._joint_forecast_structure(
            coefficients, sigma, horizon, terminal_state
        )
        covariance = loading @ loading.T
        return means, (covariance + covariance.T) / 2.0

    def _joint_forecast_structure(
        self,
        coefficients: np.ndarray,
        sigma: np.ndarray,
        horizon: int,
        terminal_state: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return forecast dynamics conditional on one paired terminal state.

        ``terminal_state`` is a retained component-specific initial condition. If it
        is omitted, the posterior mean state is used for deterministic moment checks;
        routine forecast simulation passes the state paired with the parameters.
        """

        variables = len(self.variable_ids)
        lags = self.config.lags
        if terminal_state is None:
            posterior_terminal_states = cast(np.ndarray, self.posterior_terminal_states)
            terminal_state = posterior_terminal_states.mean(axis=0)
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

        autoregressive = [
            coefficients[1 + lag * variables : 1 + (lag + 1) * variables, :].T
            for lag in range(lags)
        ]
        responses: list[np.ndarray] = [np.eye(variables)]
        for forecast_step in range(1, horizon):
            response = np.zeros((variables, variables))
            for lag in range(1, min(lags, forecast_step) + 1):
                response += autoregressive[lag - 1] @ responses[forecast_step - lag]
            responses.append(response)

        innovation_cholesky = np.linalg.cholesky(self._nearest_positive_definite(sigma))
        loading = np.zeros((horizon * variables, horizon * variables))
        for forecast_step in range(horizon):
            for shock_time in range(forecast_step + 1):
                loading[
                    forecast_step * variables : (forecast_step + 1) * variables,
                    shock_time * variables : (shock_time + 1) * variables,
                ] = responses[forecast_step - shock_time] @ innovation_cholesky
        return means.reshape(-1), loading

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

    @classmethod
    def _constraint_component_probabilities(
        cls,
        component_moments: list[tuple[np.ndarray, np.ndarray]],
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        log_weights = np.empty(len(component_moments))
        constant = len(targets) * np.log(2.0 * np.pi)
        for index, (mean, covariance) in enumerate(component_moments):
            projected_mean = constraint_matrix @ mean
            projected_covariance = constraint_matrix @ covariance @ constraint_matrix.T
            try:
                factor = cho_factor(projected_covariance, lower=True, check_finite=False)
                difference = targets - projected_mean
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

    @classmethod
    def _constraint_component_probabilities_from_loadings(
        cls,
        component_structures: list[tuple[np.ndarray, np.ndarray]],
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        """Weight posterior components without materializing full forecast covariances."""

        projected: list[tuple[np.ndarray, np.ndarray]] = []
        for mean, loading in component_structures:
            projected_loading = constraint_matrix @ loading
            covariance = projected_loading @ projected_loading.T
            projected.append((constraint_matrix @ mean, covariance))
        identity = np.eye(len(targets))
        return cls._constraint_component_probabilities(
            [(mean, covariance) for mean, covariance in projected], identity, targets
        )

    @classmethod
    def _conditional_sample(
        cls,
        rng: np.random.Generator,
        mean: np.ndarray,
        covariance: np.ndarray,
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        sample = cls._sample_gaussian(rng, mean, covariance)
        cross_covariance = covariance @ constraint_matrix.T
        conditioned_system = constraint_matrix @ cross_covariance
        try:
            adjustment = np.linalg.solve(conditioned_system, targets - constraint_matrix @ sample)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        result = sample + cross_covariance @ adjustment
        residual = targets - constraint_matrix @ result
        if np.any(np.abs(residual) > 1e-12):
            correction_system = constraint_matrix @ constraint_matrix.T
            try:
                result += constraint_matrix.T @ np.linalg.solve(correction_system, residual)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        return result

    @classmethod
    def _conditional_sample_from_loading(
        cls,
        rng: np.random.Generator,
        mean: np.ndarray,
        loading: np.ndarray,
        constraint_matrix: np.ndarray,
        targets: np.ndarray,
    ) -> np.ndarray:
        sample = mean + loading @ rng.standard_normal(mean.size)
        projected_loading = constraint_matrix @ loading
        conditioned_system = projected_loading @ projected_loading.T
        cross_covariance = loading @ projected_loading.T
        try:
            adjustment = np.linalg.solve(conditioned_system, targets - constraint_matrix @ sample)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        result = sample + cross_covariance @ adjustment
        residual = targets - constraint_matrix @ result
        if np.any(np.abs(residual) > 1e-12):
            correction_system = constraint_matrix @ constraint_matrix.T
            try:
                result += constraint_matrix.T @ np.linalg.solve(correction_system, residual)
            except np.linalg.LinAlgError as exc:
                raise ValueError("Scenario constraints are redundant or inconsistent.") from exc
        return result

    @staticmethod
    def _sample_gaussian(
        rng: np.random.Generator, mean: np.ndarray, covariance: np.ndarray
    ) -> np.ndarray:
        symmetric = (covariance + covariance.T) / 2.0
        try:
            cholesky = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError as exc:
            del exc
            try:
                cholesky = np.linalg.cholesky(BVAR._nearest_positive_definite(symmetric))
            except np.linalg.LinAlgError as repair_exc:
                raise ValueError("Forecast covariance is not positive definite.") from repair_exc
        return mean + cholesky @ rng.standard_normal(mean.size)

    @staticmethod
    def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        symmetric = (matrix + matrix.T) / 2.0
        values, vectors = np.linalg.eigh(symmetric)
        floor = max(np.max(np.abs(values)).item() * 1e-10, 1e-10)
        return (vectors * np.maximum(values, floor)) @ vectors.T

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
