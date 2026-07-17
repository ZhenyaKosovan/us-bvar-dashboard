from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from typing import cast

import numpy as np
import pandas as pd

from us_bvar.artifact import (
    RELEASE_MANIFEST_VERSION,
    ForecastArtifact,
    activate_release,
    artifact_sha256,
    create_release_manifest,
    load_artifact,
    save_artifact,
    validate_release_manifest,
)
from us_bvar.config import SERIES_SPECS
from us_bvar.data import FREDClient
from us_bvar.diagnostics import (  # type: ignore[import-not-found]
    array_diagnostic,
    chain_diagnostic,
    evaluate_convergence_gate,
)
from us_bvar.model import BVAR, BVARConfig, history_semantics_metadata

ROOT = Path(__file__).resolve().parents[1]


def load_environment(path: Path) -> None:
    """Load simple KEY=VALUE entries without overriding process environment."""

    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read environment file {path}.") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _new_release_id() -> str:
    timestamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{token_hex(6)}"


def _publish_release(
    panel: pd.DataFrame,
    artifact: ForecastArtifact,
    metadata: dict[str, object],
) -> tuple[Path, dict[str, object]]:
    """Stage and validate all release files before switching the active pointer."""

    release_id = _new_release_id()
    release_root = ROOT / "artifacts/releases"
    release_root.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=release_root))
    final_path = release_root / release_id
    try:
        panel_path = staging_path / "fred_panel.csv"
        artifact_path = staging_path / "bvar_forecast.pkl"
        metadata_path = staging_path / "metadata.json"
        panel.to_csv(panel_path, index_label="date")
        save_artifact(artifact, artifact_path)
        metadata["release_id"] = release_id
        metadata["release_manifest_version"] = RELEASE_MANIFEST_VERSION
        metadata["panel_sha256"] = sha256(panel_path.read_bytes()).hexdigest()
        metadata["artifact_sha256"] = artifact_sha256(artifact_path)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        files = {
            "panel": panel_path,
            "artifact": artifact_path,
            "checksum": artifact_path.with_suffix(f"{artifact_path.suffix}.sha256"),
            "metadata": metadata_path,
        }
        staged_manifest = create_release_manifest(ROOT, release_id, files, artifact.schema_version)
        validate_release_manifest(ROOT, staged_manifest)
        load_artifact(artifact_path)

        os.replace(staging_path, final_path)
        final_files = {key: final_path / path.name for key, path in files.items()}
        manifest = create_release_manifest(ROOT, release_id, final_files, artifact.schema_version)
        validate_release_manifest(ROOT, manifest)
        activate_release(ROOT, manifest)
        return final_path, manifest
    except Exception:
        if staging_path.exists():
            shutil.rmtree(staging_path)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FRED data and precompute the fitted BVAR and baseline forecast."
    )
    parser.add_argument("--draws", type=int, default=400, help="Posterior predictive draws.")
    parser.add_argument("--seed", type=int, default=202503, help="Baseline simulation seed.")
    parser.add_argument(
        "--mcmc-iterations",
        type=int,
        default=600,
        help="Mixed-frequency Gibbs iterations (default: 600).",
    )
    parser.add_argument(
        "--burn-in", type=int, default=300, help="Discarded Gibbs iterations (default: 300)."
    )
    parser.add_argument(
        "--thin", type=int, default=3, help="Retain every Nth post-burn-in draw (default: 3)."
    )
    parser.add_argument(
        "--mcmc-chains", type=int, default=2, help="Independent Gibbs chains (default: 2)."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the existing per-series FRED cache without making network calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.draws < 20:
        raise SystemExit("--draws must be at least 20")
    if (
        args.mcmc_iterations <= args.burn_in
        or args.burn_in < 0
        or args.thin < 1
        or args.mcmc_chains < 2
    ):
        raise SystemExit(
            "MCMC iterations must exceed burn-in, thinning must be positive, and production "
            "estimation requires at least two chains"
        )
    load_environment(ROOT / ".env")
    client = FREDClient(
        api_key="" if args.offline else None,
        cache_dir=ROOT / "data/cache",
    )
    panel = client.fetch_panel()
    print(
        f"Fitting {len(SERIES_SPECS)}-variable BVAR to {len(panel.values)} monthly rows...",
        flush=True,
    )
    model = BVAR(
        config=BVARConfig(
            mcmc_iterations=args.mcmc_iterations,
            burn_in=args.burn_in,
            thin=args.thin,
            mcmc_chains=args.mcmc_chains,
            quick_mode=False,
        )
    ).fit(panel.values)
    if (
        model.mcmc_log_likelihood is None
        or model.companion_radii is None
        or model.mcmc_chain_ids is None
        or model.posterior_coefficients is None
        or model.posterior_sigmas is None
        or model.posterior_terminal_states is None
        or model.posterior_state_paths is None
    ):
        raise RuntimeError("Mixed-frequency estimation did not produce diagnostics.")
    policy = model.config.convergence_policy
    likelihood_diagnostic = chain_diagnostic(model.mcmc_log_likelihood, model.mcmc_chain_ids)
    radius_diagnostic = chain_diagnostic(model.companion_radii, model.mcmc_chain_ids)
    transition_rows = 1 + len(model.variable_ids) * model.config.lags
    transition_coefficient_diagnostic = array_diagnostic(
        model.posterior_coefficients[:, :transition_rows, :], model.mcmc_chain_ids, policy
    )
    pandemic_control_diagnostic = array_diagnostic(
        model.posterior_coefficients[:, transition_rows:, :], model.mcmc_chain_ids, policy
    )
    covariance_diagnostic = array_diagnostic(model.posterior_sigmas, model.mcmc_chain_ids, policy)
    terminal_state_diagnostic = array_diagnostic(
        model.posterior_terminal_states, model.mcmc_chain_ids, policy
    )
    latent_path_diagnostic = array_diagnostic(
        model.posterior_state_paths, model.mcmc_chain_ids, policy
    )
    retained_per_chain = np.bincount(model.mcmc_chain_ids, minlength=model.config.mcmc_chains)
    scalar_diagnostics = {
        "log_likelihood": likelihood_diagnostic,
        "companion_radius": radius_diagnostic,
    }
    array_diagnostics = {
        "transition_coefficients": transition_coefficient_diagnostic,
        "fixed_pandemic_control_coefficients": pandemic_control_diagnostic,
        "innovation_covariances": covariance_diagnostic,
        "terminal_states": terminal_state_diagnostic,
        "latent_state_paths": latent_path_diagnostic,
    }
    gate = evaluate_convergence_gate(scalar_diagnostics, array_diagnostics, policy)
    maximum_r_hat = cast(float, gate["maximum_r_hat"])
    minimum_effective_size = cast(float, gate["minimum_effective_sample_size"])
    print(
        "Release diagnostics: "
        f"aggregate max R-hat={maximum_r_hat:.3f}, aggregate min ESS={minimum_effective_size:.1f}, "
        f"tail R-hat={gate['r_hat_outside_nominal_count']} "
        f"({gate['r_hat_outside_nominal_fraction']:.2%}), "
        f"tail ESS={gate['effective_sample_size_outside_nominal_count']} "
        f"({gate['effective_sample_size_outside_nominal_fraction']:.2%}), "
        f"retained={retained_per_chain.tolist()}, "
        f"unstable rejections={model.unstable_draws_rejected}/{model.retention_attempts}",
        flush=True,
    )
    if np.min(retained_per_chain) < model.config.minimum_retained_draws_per_chain:
        raise RuntimeError(
            "Production estimation requires at least "
            f"{model.config.minimum_retained_draws_per_chain} retained draws per chain."
        )
    if not cast(bool, gate["accepted"]):
        raise RuntimeError(
            "MCMC convergence release gate failed: " + ", ".join(cast(list[str], gate["failures"]))
        )
    model.convergence_diagnostics = {
        **gate,
        "chains": model.config.mcmc_chains,
        "retained_draws_per_chain": retained_per_chain.tolist(),
        "log_likelihood": likelihood_diagnostic,
        "companion_radius": radius_diagnostic,
        **array_diagnostics,
    }
    baseline = model.forecast(horizon=12, draws=args.draws, seed=args.seed)
    artifact = ForecastArtifact.create(model, baseline, created_at=panel.fetched_at)
    forecast_dates = pd.DatetimeIndex(artifact.baseline.dates)
    forecast_start = cast(pd.Timestamp, forecast_dates[0])
    forecast_end = cast(pd.Timestamp, forecast_dates[-1])

    if all(panel.cache_by_series.values()):
        source = "FRED API cache"
    elif any(panel.cache_by_series.values()):
        source = "mixed FRED API and cache"
    else:
        source = "FRED API"
    metadata = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at.isoformat(),
        "panel_start": artifact.panel_start.date().isoformat(),
        "panel_end": artifact.panel_end.date().isoformat(),
        "observation_count": artifact.observation_count,
        "variable_count": len(model.variable_ids),
        "forecast_start": forecast_start.date().isoformat(),
        "forecast_end": forecast_end.date().isoformat(),
        "posterior_draws": artifact.baseline.draws,
        "posterior_interval_quantiles": list(model.config.interval),
        "retained_parameter_draws": len(model.posterior_coefficients),
        "retention_attempts": model.retention_attempts,
        "unstable_draws_rejected": model.unstable_draws_rejected,
        "unstable_draw_fraction": (
            model.unstable_draws_rejected / model.retention_attempts
            if model.retention_attempts
            else 0.0
        ),
        "retained_draws_per_chain": retained_per_chain.tolist(),
        "mcmc_diagnostics": {
            **model.convergence_diagnostics,
            "release_thresholds": {
                "maximum_rank_normalized_split_r_hat": model.config.maximum_mcmc_r_hat,
                "minimum_effective_sample_size": model.config.minimum_mcmc_effective_sample_size,
                "maximum_absolute_r_hat": model.config.maximum_absolute_mcmc_r_hat,
                "minimum_absolute_effective_sample_size": (
                    model.config.minimum_absolute_mcmc_effective_sample_size
                ),
                "maximum_nominal_tail_count": model.config.maximum_mcmc_tail_count,
                "maximum_nominal_tail_fraction": model.config.maximum_mcmc_tail_fraction,
                "minimum_retained_draws_per_chain": 20,
                "maximum_companion_radius": model.config.max_companion_radius,
                "maximum_unstable_draw_fraction": (model.config.maximum_unstable_draw_fraction),
            },
        },
        "observed_counts": panel.observed_counts,
        "series": [
            {
                "series_id": spec.series_id,
                "label": spec.label,
                "group": spec.group,
                "frequency": spec.frequency,
                "model_transform": spec.transform,
            }
            for spec in SERIES_SPECS
        ],
        "last_observations": {
            series_id: date.date().isoformat()
            for series_id, date in panel.last_observations.items()
        },
        "mixed_frequency": {
            "latent_frequency": "monthly",
            "quarterly_series": "GDPC1",
            "aggregation": "one-third mean of three standardized monthly log levels",
            "filter": "Kalman",
            "smoother": "forward-filter backward-sample",
        },
        "history_semantics": history_semantics_metadata(),
        "model_config": asdict(model.config),
        "source": source,
        "source_by_series": {
            series_id: "cache" if used_cache else "api"
            for series_id, used_cache in panel.cache_by_series.items()
        },
    }
    release_path, _manifest = _publish_release(panel.values, artifact, metadata)
    print(
        f"Published release {release_path.name}: {artifact.observation_count} observations through "
        f"{artifact.panel_end:%Y-%m} and a {len(baseline.dates)}-month baseline."
    )


if __name__ == "__main__":
    main()
