from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
from scipy.stats import norm, rankdata

from us_bvar.config import ConvergencePolicy


def _as_float(value: object, label: str) -> float:
    try:
        return float(cast(float, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric.") from exc


def _as_int(value: object, label: str) -> int:
    try:
        return int(cast(int, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer.") from exc


def split_chain_matrix(values: np.ndarray, chain_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return equal-length raw chains and their split-chain representation."""

    chains = [values[chain_ids == chain] for chain in np.unique(chain_ids)]
    if len(chains) < 2 or min(map(len, chains)) < 8:
        raise RuntimeError(
            "Diagnostics require two chains with at least eight retained draws each."
        )
    retained = min(map(len, chains))
    retained -= retained % 2
    matrix = np.vstack([chain[:retained] for chain in chains])
    half = retained // 2
    split = np.vstack([matrix[:, :half], matrix[:, -half:]])
    return matrix, split


def rank_normalize(matrix: np.ndarray) -> np.ndarray:
    ranks = rankdata(matrix.reshape(-1), method="average")
    probabilities = (ranks - 0.375) / (ranks.size + 0.25)
    return norm.ppf(probabilities).reshape(matrix.shape)


def split_r_hat(matrix: np.ndarray) -> float:
    draws = matrix.shape[1]
    chain_means = matrix.mean(axis=1)
    within = np.mean(np.var(matrix, axis=1, ddof=1))
    if within <= np.finfo(float).eps:
        return 1.0 if np.allclose(chain_means, chain_means[0]) else np.inf
    between = draws * np.var(chain_means, ddof=1)
    variance = ((draws - 1) * within + between) / draws
    return max(1.0, np.sqrt(variance / within).item())


def effective_sample_size(matrix: np.ndarray) -> float:
    """Estimate split-chain ESS with Geyer's initial positive monotone sequence."""

    chains, draws = matrix.shape
    total_draws = matrix.size
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    within_variance = np.sum(centered**2, axis=1).mean() / (draws - 1)
    between_variance = draws * np.var(matrix.mean(axis=1), ddof=1)
    marginal_variance = ((draws - 1) * within_variance + between_variance) / draws
    if not np.isfinite(marginal_variance) or marginal_variance <= 0:
        return total_draws * 1.0

    autocorrelations = np.empty(draws, dtype=float)
    autocorrelations[0] = 1.0
    for lag in range(1, draws):
        autocovariance = np.sum(centered[:, :-lag] * centered[:, lag:], axis=1).mean() / draws
        autocorrelations[lag] = 1.0 - (within_variance - autocovariance) / marginal_variance

    paired_sums: list[float] = []
    for lag in range(0, draws - 1, 2):
        paired_sum = (autocorrelations[lag] + autocorrelations[lag + 1]).item()
        if not np.isfinite(paired_sum) or paired_sum <= 0:
            break
        if paired_sums:
            paired_sum = min(paired_sum, paired_sums[-1])
        paired_sums.append(paired_sum)

    integrated_autocorrelation = max(-1.0 + 2.0 * sum(paired_sums), 1.0)
    return min(total_draws, total_draws / integrated_autocorrelation)


def chain_diagnostic(values: np.ndarray, chain_ids: np.ndarray) -> dict[str, float]:
    """Compute folded rank-normalized split-R-hat and rank ESS for one scalar."""

    matrix, split = split_chain_matrix(values, chain_ids)
    ranked = rank_normalize(split)
    folded = rank_normalize(np.abs(split - np.median(split)))
    r_hat = max(split_r_hat(ranked), split_r_hat(folded))
    return {
        "r_hat": r_hat,
        "effective_sample_size": effective_sample_size(ranked),
        "minimum": np.min(matrix).item(),
        "median": np.median(matrix).item(),
        "maximum": np.max(matrix).item(),
    }


def _split_chain_array(values: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    chains = [values[chain_ids == chain] for chain in np.unique(chain_ids)]
    if len(chains) < 2 or min(map(len, chains)) < 8:
        raise RuntimeError(
            "Diagnostics require two chains with at least eight retained draws each."
        )
    retained = min(map(len, chains))
    retained -= retained % 2
    matrix = np.stack([chain[:retained] for chain in chains])
    half = retained // 2
    return np.concatenate((matrix[:, :half], matrix[:, -half:]), axis=0)


def _rank_normalize_columns(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1, values.shape[-1])
    ranks = rankdata(flattened, method="average", axis=0)
    probabilities = (ranks - 0.375) / (len(flattened) + 0.25)
    return norm.ppf(probabilities).reshape(values.shape)


def _split_r_hats(values: np.ndarray) -> np.ndarray:
    draws = values.shape[1]
    chain_means = values.mean(axis=1)
    within = np.mean(np.var(values, axis=1, ddof=1), axis=0)
    between = draws * np.var(chain_means, axis=0, ddof=1)
    variance = ((draws - 1) * within + between) / draws
    regular = np.maximum(1.0, np.sqrt(variance / np.maximum(within, np.finfo(float).eps)))
    degenerate = within <= np.finfo(float).eps
    equal_means = np.all(np.isclose(chain_means, chain_means[0]), axis=0)
    return np.where(degenerate, np.where(equal_means, 1.0, np.inf), regular)


def _effective_sample_sizes(values: np.ndarray) -> np.ndarray:
    chains, draws, _dimensions = values.shape
    total_draws = chains * draws
    centered = values - values.mean(axis=1, keepdims=True)
    within = np.sum(centered**2, axis=1).mean(axis=0) / (draws - 1)
    between = draws * np.var(values.mean(axis=1), axis=0, ddof=1)
    marginal = ((draws - 1) * within + between) / draws

    fft_size = 1 << (2 * draws - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size, axis=1)
    autocovariances = (
        np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size, axis=1)[:, :draws, :].real.mean(
            axis=0
        )
        / draws
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        autocorrelations = 1.0 - (within[None, :] - autocovariances) / marginal[None, :]
    autocorrelations[0] = 1.0

    paired = autocorrelations[0 : draws - 1 : 2] + autocorrelations[1:draws:2]
    valid = np.isfinite(paired) & (paired > 0)
    active = np.logical_and.accumulate(valid, axis=0)
    monotone = np.minimum.accumulate(paired, axis=0)
    paired_sum = np.sum(np.where(active, monotone, 0.0), axis=0)
    integrated = np.maximum(-1.0 + 2.0 * paired_sum, 1.0)
    effective = np.minimum(total_draws, total_draws / integrated)
    invalid = ~np.isfinite(marginal) | (marginal <= 0)
    effective[invalid] = total_draws
    return effective


def _array_chain_metrics(
    values: np.ndarray, chain_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    split = _split_chain_array(values, chain_ids)
    ranked = _rank_normalize_columns(split)
    median = np.median(split, axis=(0, 1), keepdims=True)
    folded = _rank_normalize_columns(np.abs(split - median))
    return np.maximum(_split_r_hats(ranked), _split_r_hats(folded)), _effective_sample_sizes(ranked)


def array_diagnostic(
    values: np.ndarray,
    chain_ids: np.ndarray,
    policy: ConvergencePolicy | None = None,
) -> dict[str, object]:
    """Report aggregate and explicitly bounded tails for every array dimension.

    The percentile summaries are a practical aggregate for large arrays. Tail
    counts/fractions and absolute extrema are retained separately so a few bad
    dimensions cannot be hidden by those percentiles.
    """

    policy = policy or ConvergencePolicy()
    flattened = values.reshape(len(values), -1)
    r_hat_chunks: list[np.ndarray] = []
    effective_size_chunks: list[np.ndarray] = []
    for start in range(0, flattened.shape[1], 512):
        r_hats, effective_sizes = _array_chain_metrics(flattened[:, start : start + 512], chain_ids)
        r_hat_chunks.append(r_hats)
        effective_size_chunks.append(effective_sizes)
    r_hats = np.concatenate(r_hat_chunks)
    effective_sizes = np.concatenate(effective_size_chunks)
    dimensions = flattened.shape[1]
    r_hat_outside = r_hats > policy.nominal_r_hat
    ess_outside = effective_sizes < policy.nominal_effective_sample_size
    return {
        "dimensions_checked": dimensions,
        "maximum_r_hat": np.max(r_hats).item(),
        "maximum_r_hat_flat_index": np.argmax(r_hats).item(),
        "r_hat_99th_percentile": np.quantile(r_hats, 0.99).item(),
        "r_hat_outside_nominal_count": _as_int(
            np.count_nonzero(r_hat_outside).item(), "R-hat tail count"
        ),
        "r_hat_outside_nominal_fraction": np.mean(r_hat_outside).item(),
        "minimum_effective_sample_size": np.min(effective_sizes).item(),
        "minimum_ess_flat_index": np.argmin(effective_sizes).item(),
        "effective_sample_size_1st_percentile": np.quantile(effective_sizes, 0.01).item(),
        "effective_sample_size_outside_nominal_count": _as_int(
            np.count_nonzero(ess_outside).item(), "ESS tail count"
        ),
        "effective_sample_size_outside_nominal_fraction": np.mean(ess_outside).item(),
    }


def evaluate_convergence_gate(
    scalar_diagnostics: Mapping[str, Mapping[str, float]],
    array_diagnostics: Mapping[str, Mapping[str, object]],
    policy: ConvergencePolicy,
) -> dict[str, object]:
    """Recompute the release decision from scalar and array diagnostic summaries."""

    scalar_r_hats = [
        _as_float(section["r_hat"], "Scalar R-hat") for section in scalar_diagnostics.values()
    ]
    scalar_effective_sizes = [
        _as_float(section["effective_sample_size"], "Scalar ESS")
        for section in scalar_diagnostics.values()
    ]
    failures: list[str] = []
    for name, section in scalar_diagnostics.items():
        r_hat = _as_float(section["r_hat"], f"{name} R-hat")
        effective_size = _as_float(section["effective_sample_size"], f"{name} ESS")
        if not np.isfinite(r_hat) or r_hat > policy.nominal_r_hat:
            failures.append(f"{name} R-hat")
        if not np.isfinite(effective_size) or effective_size < policy.nominal_effective_sample_size:
            failures.append(f"{name} ESS")

    total_dimensions = 0
    total_r_hat_outside = 0
    total_ess_outside = 0
    maximum_r_hat = max(scalar_r_hats, default=-np.inf)
    minimum_effective_size = min(scalar_effective_sizes, default=np.inf)
    maximum_absolute_r_hat = maximum_r_hat
    minimum_absolute_effective_size = minimum_effective_size
    for name, section in array_diagnostics.items():
        dimensions = _as_int(section["dimensions_checked"], "Diagnostic dimension count")
        r_hat_99 = _as_float(section["r_hat_99th_percentile"], "R-hat percentile")
        ess_01 = _as_float(section["effective_sample_size_1st_percentile"], "ESS percentile")
        maximum = _as_float(section["maximum_r_hat"], "Maximum R-hat")
        minimum = _as_float(section["minimum_effective_sample_size"], "Minimum ESS")
        r_hat_count = _as_int(section["r_hat_outside_nominal_count"], "R-hat tail count")
        ess_count = _as_int(
            section["effective_sample_size_outside_nominal_count"], "ESS tail count"
        )
        if dimensions <= 0:
            failures.append(f"{name} has no diagnostic dimensions")
            continue
        if not 0 <= r_hat_count <= dimensions:
            failures.append(f"{name} R-hat tail count is invalid")
        if not 0 <= ess_count <= dimensions:
            failures.append(f"{name} ESS tail count is invalid")
        total_dimensions += dimensions
        total_r_hat_outside += r_hat_count
        total_ess_outside += ess_count
        maximum_r_hat = max(maximum_r_hat, r_hat_99)
        minimum_effective_size = min(minimum_effective_size, ess_01)
        maximum_absolute_r_hat = max(maximum_absolute_r_hat, maximum)
        minimum_absolute_effective_size = min(minimum_absolute_effective_size, minimum)
        if not np.isfinite(r_hat_99) or r_hat_99 > policy.nominal_r_hat:
            failures.append(f"{name} R-hat aggregate")
        if not np.isfinite(ess_01) or ess_01 < policy.nominal_effective_sample_size:
            failures.append(f"{name} ESS aggregate")
        if not np.isfinite(maximum) or maximum > policy.absolute_maximum_r_hat:
            failures.append(f"{name} R-hat absolute bound")
        if not np.isfinite(minimum) or minimum < policy.absolute_minimum_effective_sample_size:
            failures.append(f"{name} ESS absolute bound")
        if r_hat_count > policy.maximum_tail_count or (
            r_hat_count / dimensions > policy.maximum_tail_fraction
        ):
            failures.append(f"{name} R-hat nominal-tail bound")
        if ess_count > policy.maximum_tail_count or (
            ess_count / dimensions > policy.maximum_tail_fraction
        ):
            failures.append(f"{name} ESS nominal-tail bound")

    return {
        "accepted": not failures,
        "maximum_r_hat": maximum_r_hat,
        "minimum_effective_sample_size": minimum_effective_size,
        "maximum_absolute_r_hat": maximum_absolute_r_hat,
        "minimum_absolute_effective_sample_size": minimum_absolute_effective_size,
        "r_hat_outside_nominal_count": total_r_hat_outside,
        "r_hat_outside_nominal_fraction": (
            total_r_hat_outside / total_dimensions if total_dimensions else 0.0
        ),
        "effective_sample_size_outside_nominal_count": total_ess_outside,
        "effective_sample_size_outside_nominal_fraction": (
            total_ess_outside / total_dimensions if total_dimensions else 0.0
        ),
        "failures": failures,
    }


def compute_posterior_diagnostics(
    *,
    coefficients: np.ndarray,
    innovation_covariances: np.ndarray,
    terminal_states: np.ndarray,
    latent_state_paths: np.ndarray,
    log_likelihood: np.ndarray,
    companion_radius: np.ndarray,
    chain_ids: np.ndarray,
    transition_rows: int,
    chains: int,
    policy: ConvergencePolicy,
) -> dict[str, object]:
    """Compute the complete convergence record used by publishing and validation."""

    scalar_diagnostics = {
        "log_likelihood": chain_diagnostic(log_likelihood, chain_ids),
        "companion_radius": chain_diagnostic(companion_radius, chain_ids),
    }
    array_diagnostics = {
        "transition_coefficients": array_diagnostic(
            coefficients[:, :transition_rows, :], chain_ids, policy
        ),
        "fixed_pandemic_control_coefficients": array_diagnostic(
            coefficients[:, transition_rows:, :], chain_ids, policy
        ),
        "innovation_covariances": array_diagnostic(innovation_covariances, chain_ids, policy),
        "terminal_states": array_diagnostic(terminal_states, chain_ids, policy),
        "latent_state_paths": array_diagnostic(latent_state_paths, chain_ids, policy),
    }
    gate = evaluate_convergence_gate(scalar_diagnostics, array_diagnostics, policy)
    retained_per_chain = np.bincount(chain_ids, minlength=chains)
    return {
        **gate,
        "chains": chains,
        "retained_draws_per_chain": retained_per_chain.tolist(),
        **scalar_diagnostics,
        **array_diagnostics,
    }
