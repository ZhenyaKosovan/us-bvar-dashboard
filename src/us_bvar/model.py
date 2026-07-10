from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import invwishart

from us_bvar.config import PANDEMIC_CONTROL_MONTHS, SERIES_SPECS, SeriesSpec
from us_bvar.transforms import (
    PLOT_TRANSFORMATIONS,
    LevelTransformer,
    ScenarioConstraint,
    transformation_spec,
)


@dataclass(frozen=True)
class BVARConfig:
    lags: int = 4
    tightness: float = 0.20
    cross_variable_tightness: float = 0.50
    lag_decay: float = 1.0
    pandemic_months: tuple[str, ...] = PANDEMIC_CONTROL_MONTHS
    interval: tuple[float, float] = (0.16, 0.84)


@dataclass(frozen=True)
class ForecastResult:
    dates: pd.DatetimeIndex
    median: pd.DataFrame
    lower: pd.DataFrame
    upper: pd.DataFrame
    constraints: Mapping[tuple[int, str], ScenarioConstraint]
    draws: int

    @property
    def is_scenario(self) -> bool:
        return bool(self.constraints)


class BVAR:
    """Empirical-Bayes VAR with Minnesota priors and conditional forecasts.

    Coefficients have equation-specific Minnesota normal priors. The likelihood
    covariance is initialized from OLS and innovation covariance uncertainty is
    represented by an inverse-Wishart posterior. Forecast scenarios use the exact
    multivariate Gaussian conditional distribution at each parameter draw.
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
        self.history_levels: pd.DataFrame | None = None
        self.history_model: pd.DataFrame | None = None
        self.posterior_mean: np.ndarray | None = None
        self.posterior_cholesky: np.ndarray | None = None
        self.iw_scale: np.ndarray | None = None
        self.iw_df: int | None = None
        self._n_controls = len(self.config.pandemic_months)

    def fit(self, levels: pd.DataFrame) -> BVAR:
        ids = list(self.variable_ids)
        if not isinstance(levels.index, pd.DatetimeIndex):
            raise TypeError("The history index must be a DatetimeIndex.")
        if missing := set(ids).difference(levels.columns):
            raise ValueError(f"Missing model variables: {sorted(missing)}")
        levels = levels[ids].sort_index().dropna().copy()
        if len(levels) <= self.config.lags + len(ids) + 2:
            raise ValueError("Too few observations to estimate the BVAR.")

        self.transformer = LevelTransformer.fit(levels, self.specs)
        model_frame = self.transformer.encode_frame(levels)
        x, y = self._design_matrix(model_frame)
        n = len(ids)
        k = x.shape[1]

        xtx = x.T @ x
        ridge = np.eye(k) * 1e-8
        ols = np.linalg.solve(xtx + ridge, x.T @ y)
        residuals = y - x @ ols
        sigma = residuals.T @ residuals / max(len(y) - k, n + 2)
        sigma = self._nearest_positive_definite(sigma)

        prior_mean, prior_variance = self._minnesota_prior(k, n)
        likelihood_precision = np.kron(np.linalg.inv(sigma), xtx)
        prior_precision_diag = 1.0 / prior_variance
        posterior_precision = likelihood_precision.copy()
        posterior_precision.flat[:: posterior_precision.shape[0] + 1] += prior_precision_diag

        ols_vec = ols.reshape(-1, order="F")
        prior_vec = prior_mean.reshape(-1, order="F")
        rhs = likelihood_precision @ ols_vec + prior_precision_diag * prior_vec
        factor = cho_factor(posterior_precision, lower=True, check_finite=False)
        posterior_vec = cho_solve(factor, rhs, check_finite=False)
        posterior_covariance = cho_solve(
            factor, np.eye(posterior_precision.shape[0]), check_finite=False
        )

        posterior_b = posterior_vec.reshape((k, n), order="F")
        posterior_residuals = y - x @ posterior_b
        scale = posterior_residuals.T @ posterior_residuals + np.eye(n) * 1e-6

        self.history_levels = levels
        self.history_model = model_frame
        self.posterior_mean = posterior_b
        self.posterior_cholesky = np.linalg.cholesky(
            self._nearest_positive_definite(posterior_covariance)
        )
        self.iw_scale = self._nearest_positive_definite(scale)
        self.iw_df = len(y) + n + 2
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
        normalized_constraints = self._normalize_constraints(constraints or {})
        constraint_matrix, targets = self._constraint_system(normalized_constraints, horizon)

        rng = np.random.default_rng(seed)
        simulations = np.empty((draws, horizon, len(self.variable_ids)))
        for draw in range(draws):
            coefficients, sigma = self._draw_parameters(rng)
            mean, covariance = self._joint_forecast_moments(coefficients, sigma, horizon)
            if len(targets):
                sample = self._conditional_sample(rng, mean, covariance, constraint_matrix, targets)
            else:
                sample = self._sample_gaussian(rng, mean, covariance)
            simulations[draw] = sample.reshape(horizon, len(self.variable_ids))

        assert self.transformer is not None
        natural_draws = self.transformer.decode_array(simulations)
        lower_q, upper_q = self.config.interval
        dates = pd.date_range(
            pd.Timestamp(self.history_levels.index[-1]) + pd.offsets.MonthBegin(1),
            periods=horizon,
            freq="MS",
        )

        def frame(values: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame(values, index=dates, columns=self.variable_ids)

        return ForecastResult(
            dates=dates,
            median=frame(np.median(natural_draws, axis=0)),
            lower=frame(np.quantile(natural_draws, lower_q, axis=0)),
            upper=frame(np.quantile(natural_draws, upper_q, axis=0)),
            constraints=normalized_constraints,
            draws=draws,
        )

    def _design_matrix(self, model_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = model_frame.to_numpy()
        dates = model_frame.index
        p = self.config.lags
        rows: list[np.ndarray] = []
        for t in range(p, len(values)):
            lagged = np.concatenate([values[t - lag] for lag in range(1, p + 1)])
            controls = np.array(
                [float(dates[t] == pd.Timestamp(month)) for month in self.config.pandemic_months]
            )
            rows.append(np.concatenate(([1.0], lagged, controls)))
        return np.vstack(rows), values[p:]

    def _minnesota_prior(self, k: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        mean = np.zeros((k, n))
        variance = np.empty((k, n))
        variance[0, :] = 10.0**2
        for lag in range(1, self.config.lags + 1):
            for source in range(n):
                row = 1 + (lag - 1) * n + source
                for equation in range(n):
                    relative = 1.0 if source == equation else self.config.cross_variable_tightness
                    prior_sd = self.config.tightness * relative / (lag**self.config.lag_decay)
                    variance[row, equation] = prior_sd**2
                    if lag == 1 and source == equation:
                        mean[row, equation] = 1.0
        variance[1 + self.config.lags * n :, :] = 10.0**2
        return mean, variance.reshape(-1, order="F")

    def _draw_parameters(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        assert self.posterior_mean is not None
        assert self.posterior_cholesky is not None
        assert self.iw_scale is not None
        assert self.iw_df is not None
        mean_vec = self.posterior_mean.reshape(-1, order="F")
        coefficients = (
            mean_vec + self.posterior_cholesky @ rng.standard_normal(mean_vec.size)
        ).reshape(self.posterior_mean.shape, order="F")
        sigma = np.asarray(
            invwishart.rvs(df=self.iw_df, scale=self.iw_scale, random_state=rng), dtype=float
        )
        sigma = np.atleast_2d(sigma)
        return coefficients, (sigma + sigma.T) / 2

    def _joint_forecast_moments(
        self, coefficients: np.ndarray, sigma: np.ndarray, horizon: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.history_model is not None
        n = len(self.variable_ids)
        p = self.config.lags
        history = [row.copy() for row in self.history_model.to_numpy()]
        means = np.empty((horizon, n))
        for step in range(horizon):
            lagged = np.concatenate([history[-lag] for lag in range(1, p + 1)])
            x = np.concatenate(([1.0], lagged, np.zeros(self._n_controls)))
            means[step] = x @ coefficients
            history.append(means[step])

        autoregressive = [coefficients[1 + lag * n : 1 + (lag + 1) * n, :].T for lag in range(p)]
        responses: list[np.ndarray] = [np.eye(n)]
        for h in range(1, horizon):
            response = np.zeros((n, n))
            for lag in range(1, min(p, h) + 1):
                response += autoregressive[lag - 1] @ responses[h - lag]
            responses.append(response)

        response_matrix = np.zeros((horizon * n, horizon * n))
        for h in range(horizon):
            for shock_time in range(h + 1):
                response_matrix[
                    h * n : (h + 1) * n,
                    shock_time * n : (shock_time + 1) * n,
                ] = responses[h - shock_time]

        innovation_covariance = np.kron(np.eye(horizon), sigma)
        covariance = response_matrix @ innovation_covariance @ response_matrix.T
        return means.reshape(-1), (covariance + covariance.T) / 2

    @staticmethod
    def _normalize_constraints(
        constraints: Mapping[tuple[int, str], float | ScenarioConstraint],
    ) -> dict[tuple[int, str], ScenarioConstraint]:
        result: dict[tuple[int, str], ScenarioConstraint] = {}
        for key, raw_constraint in constraints.items():
            constraint = (
                raw_constraint
                if isinstance(raw_constraint, ScenarioConstraint)
                else ScenarioConstraint(value=float(raw_constraint))
            )
            if constraint.transformation not in PLOT_TRANSFORMATIONS:
                raise ValueError(f"Unknown scenario transformation: {constraint.transformation}")
            value = float(constraint.value)
            if not np.isfinite(value):
                raise ValueError("Scenario values must be finite numbers.")
            result[key] = ScenarioConstraint(value=value, transformation=constraint.transformation)
        return result

    def _constraint_system(
        self,
        constraints: Mapping[tuple[int, str], ScenarioConstraint],
        horizon: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert self.transformer is not None
        assert self.history_model is not None
        n = len(self.variable_ids)
        rows: list[np.ndarray] = []
        targets: list[float] = []
        for (step, variable_id), constraint in constraints.items():
            if not 0 <= step < horizon:
                raise ValueError(f"Scenario step {step} is outside the forecast horizon.")
            if variable_id not in self.variable_ids:
                raise ValueError(f"Unknown scenario variable: {variable_id}")
            variable_index = self.variable_ids.index(variable_id)
            row = np.zeros(horizon * n)
            row[step * n + variable_index] = 1.0

            if constraint.transformation == "level":
                target = self.transformer.encode_value(variable_index, constraint.value)
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
                model_change = latent_change / self.transformer.scales[variable_index]
                lag_step = step - display_spec.periods
                if lag_step >= 0:
                    row[lag_step * n + variable_index] = -1.0
                    target = model_change
                else:
                    if abs(lag_step) > len(self.history_model):
                        raise ValueError(
                            f"Not enough history to apply {display_spec.label} to "
                            f"{spec.short_label}."
                        )
                    target = float(self.history_model.iloc[lag_step, variable_index]) + model_change

            rows.append(row)
            targets.append(float(target))

        return (
            np.vstack(rows) if rows else np.empty((0, horizon * n)),
            np.asarray(targets, dtype=float),
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

        # Remove the last few floating-point bits of residual so exact constraints
        # remain exact after conversion back to natural units.
        residual = targets - constraint_matrix @ result
        if np.any(np.abs(residual) > 1e-12):
            correction_system = constraint_matrix @ constraint_matrix.T
            result += constraint_matrix.T @ np.linalg.solve(correction_system, residual)
        return result

    @staticmethod
    def _sample_gaussian(
        rng: np.random.Generator, mean: np.ndarray, covariance: np.ndarray
    ) -> np.ndarray:
        symmetric = (covariance + covariance.T) / 2
        try:
            cholesky = np.linalg.cholesky(symmetric)
        except np.linalg.LinAlgError:
            cholesky = np.linalg.cholesky(BVAR._nearest_positive_definite(symmetric))
        return mean + cholesky @ rng.standard_normal(mean.size)

    @staticmethod
    def _nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
        symmetric = (matrix + matrix.T) / 2
        values, vectors = np.linalg.eigh(symmetric)
        floor = max(float(np.max(np.abs(values))) * 1e-10, 1e-10)
        return (vectors * np.maximum(values, floor)) @ vectors.T

    def _require_fit(self) -> None:
        if self.posterior_mean is None or self.history_levels is None:
            raise RuntimeError("Call fit() before forecasting.")
