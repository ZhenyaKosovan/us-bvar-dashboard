from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve

from us_bvar.config import SeriesSpec


@dataclass(frozen=True)
class KalmanResult:
    """Filtering output needed by the simulation smoother."""

    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    log_likelihood: float


def companion_transition(
    coefficients: np.ndarray,
    sigma: np.ndarray,
    variables: int,
    lags: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the companion transition and its structurally singular covariance."""

    dimension = variables * lags
    transition = np.zeros((dimension, dimension))
    for lag in range(lags):
        transition[:variables, lag * variables : (lag + 1) * variables] = coefficients[
            1 + lag * variables : 1 + (lag + 1) * variables
        ].T
    if lags > 1:
        transition[variables:, :-variables] = np.eye(variables * (lags - 1))
    process_covariance = np.zeros((dimension, dimension))
    process_covariance[:variables, :variables] = sigma
    return transition, process_covariance


def observation_system(
    row: np.ndarray,
    specs: tuple[SeriesSpec, ...],
    lags: int,
    monthly_measurement_variance: float,
    quarterly_measurement_variance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the available-observation system for one month.

    Quarterly SAAR GDP is measured as the geometric-average approximation to
    the three latent monthly log levels. In standardized model space this is a
    linear row with one-third weight on each current-quarter month.
    """

    variables = len(specs)
    state_dimension = variables * lags
    observations: list[float] = []
    rows: list[np.ndarray] = []
    variances: list[float] = []
    for variable, spec in enumerate(specs):
        value = row[variable]
        if not np.isfinite(value):
            continue
        measurement = np.zeros(state_dimension)
        if spec.frequency == "quarterly":
            if lags < 3:
                raise ValueError("Quarterly aggregation requires at least three state lags.")
            for block in range(3):
                measurement[block * variables + variable] = 1.0 / 3.0
            variance = quarterly_measurement_variance
        else:
            measurement[variable] = 1.0
            variance = monthly_measurement_variance
        try:
            observations.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid observation for {spec.series_id}.") from exc
        rows.append(measurement)
        variances.append(variance)
    if not rows:
        return (
            np.empty(0),
            np.empty((0, state_dimension)),
            np.empty((0, 0)),
        )
    return np.asarray(observations), np.vstack(rows), np.diag(variances)


def control_vector(date: object, control_months: tuple[str, ...]) -> np.ndarray:
    """Return one fixed-effect indicator for each listed pandemic month."""

    if not control_months:
        return np.empty(0)
    try:
        timestamp = pd.DatetimeIndex([date])[0]
        active_months = pd.DatetimeIndex(control_months)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid pandemic control month configuration.") from exc
    return np.asarray(active_months == timestamp, dtype=float)


def kalman_filter(
    observations: pd.DataFrame,
    coefficients: np.ndarray,
    sigma: np.ndarray,
    specs: tuple[SeriesSpec, ...],
    lags: int,
    control_months: tuple[str, ...],
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    monthly_measurement_variance: float,
    quarterly_measurement_variance: float,
) -> KalmanResult:
    """Filter a ragged mixed-frequency panel with time-varying measurements."""

    variables = len(specs)
    transition, process_covariance = companion_transition(coefficients, sigma, variables, lags)
    periods = len(observations)
    dimension = variables * lags
    predicted_means = np.empty((periods, dimension))
    predicted_covariances = np.empty((periods, dimension, dimension))
    filtered_means = np.empty_like(predicted_means)
    filtered_covariances = np.empty_like(predicted_covariances)
    intercept = coefficients[0]
    control_coefficients = coefficients[1 + variables * lags :]
    values = observations.to_numpy(dtype=float)
    log_likelihood = 0.0

    previous_mean = np.asarray(initial_mean, dtype=float)
    previous_covariance = np.asarray(initial_covariance, dtype=float)
    for time, date in enumerate(observations.index):
        if time == 0:
            predicted_mean = previous_mean.copy()
            predicted_covariance = previous_covariance.copy()
        else:
            controls = control_vector(date, control_months)
            state_intercept = np.zeros(dimension)
            state_intercept[:variables] = intercept + controls @ control_coefficients
            predicted_mean = state_intercept + transition @ previous_mean
            predicted_covariance = (
                transition @ previous_covariance @ transition.T + process_covariance
            )
            predicted_covariance = _symmetrize(predicted_covariance)

        observed, measurement, measurement_covariance = observation_system(
            values[time],
            specs,
            lags,
            monthly_measurement_variance,
            quarterly_measurement_variance,
        )
        if observed.size:
            innovation = observed - measurement @ predicted_mean
            innovation_covariance = (
                measurement @ predicted_covariance @ measurement.T + measurement_covariance
            )
            innovation_covariance = _symmetrize(innovation_covariance)
            factor = _factor_positive_definite(innovation_covariance)
            solved_innovation = cho_solve(factor, innovation, check_finite=False)
            gain = cho_solve(
                factor,
                measurement @ predicted_covariance,
                check_finite=False,
            ).T
            filtered_mean = predicted_mean + gain @ innovation
            identity_update = np.eye(dimension) - gain @ measurement
            filtered_covariance = (
                identity_update @ predicted_covariance @ identity_update.T
                + gain @ measurement_covariance @ gain.T
            )
            filtered_covariance = _symmetrize(filtered_covariance)
            log_determinant = 2.0 * np.log(np.diag(factor[0])).sum()
            log_likelihood += -0.5 * (
                observed.size * np.log(2.0 * np.pi)
                + log_determinant
                + innovation @ solved_innovation
            )
        else:
            filtered_mean = predicted_mean
            filtered_covariance = predicted_covariance

        predicted_means[time] = predicted_mean
        predicted_covariances[time] = predicted_covariance
        filtered_means[time] = filtered_mean
        filtered_covariances[time] = filtered_covariance
        previous_mean = filtered_mean
        previous_covariance = filtered_covariance

    return KalmanResult(
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        log_likelihood=log_likelihood,
    )


def simulation_smoother(
    result: KalmanResult,
    transition: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one companion-state path using forward filtering/backward sampling."""

    periods, dimension = result.filtered_means.shape
    states = np.empty((periods, dimension))
    states[-1] = _sample_psd(rng, result.filtered_means[-1], result.filtered_covariances[-1])
    for time in range(periods - 2, -1, -1):
        predicted_covariance = result.predicted_covariances[time + 1]
        filtered_transition = result.filtered_covariances[time] @ transition.T
        smoothing_gain = cho_solve(
            _factor_positive_definite(predicted_covariance),
            filtered_transition.T,
            check_finite=False,
        ).T
        mean = result.filtered_means[time] + smoothing_gain @ (
            states[time + 1] - result.predicted_means[time + 1]
        )
        covariance = _symmetrize(
            result.filtered_covariances[time]
            - smoothing_gain @ predicted_covariance @ smoothing_gain.T
        )
        states[time] = _sample_psd(rng, mean, covariance)
    return states


def companion_radius(coefficients: np.ndarray, variables: int, lags: int) -> float:
    transition, _ = companion_transition(coefficients, np.eye(variables), variables, lags)
    try:
        return np.max(np.abs(np.linalg.eigvals(transition))).item()
    except np.linalg.LinAlgError as exc:
        raise ValueError("Could not evaluate VAR companion stability.") from exc


def _sample_psd(
    rng: np.random.Generator,
    mean: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray:
    symmetric = _symmetrize(covariance)
    scale = max(float(np.max(np.abs(np.diag(symmetric)))), 1.0)
    identity = np.eye(len(mean))
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            cholesky = np.linalg.cholesky(symmetric + relative_jitter * scale * identity)
            return mean + cholesky @ rng.standard_normal(mean.size)
        except np.linalg.LinAlgError:
            continue
    values, vectors = np.linalg.eigh(symmetric)
    return mean + (vectors * np.sqrt(np.maximum(values, 0.0))) @ rng.standard_normal(mean.size)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return (matrix + matrix.T) / 2.0


def _positive_definite(matrix: np.ndarray) -> np.ndarray:
    symmetric = _symmetrize(matrix)
    values, vectors = np.linalg.eigh(symmetric)
    floor = max(np.max(np.abs(values)).item() * 1e-10, 1e-10)
    return _symmetrize((vectors * np.maximum(values, floor)) @ vectors.T)


def _factor_positive_definite(matrix: np.ndarray) -> tuple[np.ndarray, bool]:
    """Factor a covariance with cheap jitter before an eigenvalue repair fallback."""

    symmetric = _symmetrize(matrix)
    scale = max(float(np.max(np.abs(np.diag(symmetric)))), 1.0)
    identity = np.eye(len(symmetric))
    for relative_jitter in (0.0, 1e-12, 1e-10, 1e-8):
        try:
            return cho_factor(
                symmetric + relative_jitter * scale * identity,
                lower=True,
                check_finite=False,
            )
        except np.linalg.LinAlgError:
            continue
    return cho_factor(_positive_definite(symmetric), lower=True, check_finite=False)
