from __future__ import annotations

import json
from copy import deepcopy
from typing import cast

import numpy as np
import pandas as pd
import pytest

from us_bvar.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    ForecastArtifact,
    activate_release,
    artifact_sha256,
    create_release_manifest,
    load_artifact,
    load_published_release,
    save_artifact,
)
from us_bvar.diagnostics import array_diagnostic, chain_diagnostic, evaluate_convergence_gate
from us_bvar.model import BVAR, BVARConfig


def _matched_draws(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Create two short, identically distributed chains for artifact tests."""

    center = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), np.maximum(np.abs(center) * 1e-5, 1e-5))
    shape = (len(template),) + (1,) * (values.ndim - 1)
    chain = center + 0.001 * scale * template.reshape(shape)
    return np.concatenate((chain, chain.copy()))


@pytest.fixture(scope="module")
def production_model_base(synthetic_levels) -> BVAR:
    model = BVAR(config=BVARConfig.quick(mcmc_chains=2)).fit(synthetic_levels)
    assert model.mcmc_chain_ids is not None
    rng = np.random.default_rng(17)
    retained_per_chain = 20
    template = rng.normal(size=retained_per_chain)
    template = (template - template.mean()) / template.std(ddof=1)

    assert model.posterior_coefficients is not None
    assert model.fixed_control_coefficients is not None
    assert model.posterior_sigmas is not None
    assert model.posterior_terminal_states is not None
    assert model.posterior_state_paths is not None
    assert model.mcmc_log_likelihood is not None
    assert model.companion_radii is not None
    model.posterior_coefficients = _matched_draws(model.posterior_coefficients, template)
    controls = len(model.config.pandemic_months)
    model.posterior_coefficients[:, -controls:, :] = model.fixed_control_coefficients[None, :, :]
    sigma_center = model.posterior_sigmas.mean(axis=0)
    model.posterior_sigmas = np.concatenate(
        [sigma_center[None, :, :] * np.exp(0.01 * template[:, None, None])] * 2
    )
    model.posterior_terminal_states = _matched_draws(model.posterior_terminal_states, template)
    model.posterior_state_paths = _matched_draws(model.posterior_state_paths, template)
    model.mcmc_log_likelihood = _matched_draws(model.mcmc_log_likelihood, template)
    model.companion_radii = model.companion_radii.mean() + 0.001 * np.tile(template, 2)
    model.mcmc_chain_ids = np.repeat(np.arange(2), retained_per_chain)
    model.retention_attempts = 2 * retained_per_chain
    model.unstable_draws_rejected = 0
    model.posterior_mean = model.posterior_coefficients.mean(axis=0)

    policy = model.config.convergence_policy
    transition_rows = 1 + len(model.variable_ids) * model.config.lags
    scalar_diagnostics = {
        "log_likelihood": chain_diagnostic(model.mcmc_log_likelihood, model.mcmc_chain_ids),
        "companion_radius": chain_diagnostic(model.companion_radii, model.mcmc_chain_ids),
    }
    array_diagnostics = {
        "transition_coefficients": array_diagnostic(
            model.posterior_coefficients[:, :transition_rows, :],
            model.mcmc_chain_ids,
            policy,
        ),
        "fixed_pandemic_control_coefficients": array_diagnostic(
            model.posterior_coefficients[:, transition_rows:, :],
            model.mcmc_chain_ids,
            policy,
        ),
        "innovation_covariances": array_diagnostic(
            model.posterior_sigmas, model.mcmc_chain_ids, policy
        ),
        "terminal_states": array_diagnostic(
            model.posterior_terminal_states, model.mcmc_chain_ids, policy
        ),
        "latent_state_paths": array_diagnostic(
            model.posterior_state_paths, model.mcmc_chain_ids, policy
        ),
    }
    gate = evaluate_convergence_gate(scalar_diagnostics, array_diagnostics, policy)
    assert gate["accepted"]
    model.convergence_diagnostics = {
        **gate,
        "chains": 2,
        "retained_draws_per_chain": [retained_per_chain, retained_per_chain],
        **scalar_diagnostics,
        **array_diagnostics,
    }
    return model


@pytest.fixture
def production_model(production_model_base: BVAR) -> BVAR:
    return deepcopy(production_model_base)


def test_artifact_round_trip(tmp_path, synthetic_levels, production_model: BVAR) -> None:
    model = production_model
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(
        model,
        baseline,
        created_at=cast(pd.Timestamp, pd.Timestamp("2026-07-10T08:00:00Z")),
    )
    path = tmp_path / "forecast.pkl"

    save_artifact(artifact, path)
    restored = load_artifact(path)

    assert restored.schema_version == ARTIFACT_SCHEMA_VERSION
    assert restored.observation_count == len(synthetic_levels)
    assert restored.panel_end == synthetic_levels.index[-1]
    assert restored.baseline.median.equals(baseline.median)
    assert path.with_suffix(".pkl.sha256").is_file()


def test_active_release_loads_one_validated_file_set(
    tmp_path, synthetic_levels, production_model: BVAR
) -> None:
    release_dir = tmp_path / "artifacts/releases/release-one"
    release_dir.mkdir(parents=True)
    artifact_path = release_dir / "bvar_forecast.pkl"
    panel_path = release_dir / "fred_panel.csv"
    metadata_path = release_dir / "metadata.json"
    save_artifact(
        ForecastArtifact.create(production_model, production_model.forecast(12, 20, seed=19)),
        artifact_path,
    )
    panel_path.write_text(synthetic_levels.to_csv(index_label="date"), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "release_id": "release-one",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_sha256": artifact_sha256(artifact_path),
                "panel_sha256": artifact_sha256(panel_path),
            }
        ),
        encoding="utf-8",
    )
    manifest = create_release_manifest(
        tmp_path,
        "release-one",
        {
            "panel": panel_path,
            "artifact": artifact_path,
            "checksum": artifact_path.with_suffix(".pkl.sha256"),
            "metadata": metadata_path,
        },
        ARTIFACT_SCHEMA_VERSION,
    )
    activate_release(tmp_path, manifest)

    release = load_published_release(tmp_path)

    assert release.release_id == "release-one"
    assert not release.legacy
    assert release.artifact_path == artifact_path.resolve()
    assert release.metadata_path == metadata_path.resolve()


def test_active_release_rejects_file_changed_after_activation(
    tmp_path, synthetic_levels, production_model: BVAR
) -> None:
    release_dir = tmp_path / "artifacts/releases/release-one"
    release_dir.mkdir(parents=True)
    artifact_path = release_dir / "bvar_forecast.pkl"
    panel_path = release_dir / "fred_panel.csv"
    metadata_path = release_dir / "metadata.json"
    save_artifact(
        ForecastArtifact.create(production_model, production_model.forecast(12, 20, seed=23)),
        artifact_path,
    )
    panel_path.write_text(synthetic_levels.to_csv(index_label="date"), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            {
                "release_id": "release-one",
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "artifact_sha256": artifact_sha256(artifact_path),
                "panel_sha256": artifact_sha256(panel_path),
            }
        ),
        encoding="utf-8",
    )
    manifest = create_release_manifest(
        tmp_path,
        "release-one",
        {
            "panel": panel_path,
            "artifact": artifact_path,
            "checksum": artifact_path.with_suffix(".pkl.sha256"),
            "metadata": metadata_path,
        },
        ARTIFACT_SCHEMA_VERSION,
    )
    activate_release(tmp_path, manifest)
    panel_path.write_text(panel_path.read_text(encoding="utf-8") + "# changed\\n", encoding="utf-8")

    with pytest.raises(ValueError, match="active release panel"):
        load_published_release(tmp_path)


def test_artifact_rejects_failed_convergence_gate(tmp_path, production_model: BVAR) -> None:
    model = production_model
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(model, baseline)
    assert model.convergence_diagnostics is not None
    model.convergence_diagnostics["maximum_r_hat"] = 1.5
    path = tmp_path / "unconverged.pkl"
    save_artifact(artifact, path)

    with pytest.raises(ValueError, match="convergence release gate"):
        load_artifact(path)


def test_artifact_rejects_malformed_mixed_frequency_posterior(
    tmp_path, production_model: BVAR
) -> None:
    model = production_model
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(model, baseline)
    assert model.posterior_coefficients is not None
    model.posterior_coefficients = model.posterior_coefficients[:, :-1, :]
    path = tmp_path / "malformed.pkl"
    save_artifact(artifact, path)

    with pytest.raises(ValueError, match="coefficient posterior"):
        load_artifact(path)


def test_artifact_corruption_is_rejected_before_unpickling(
    tmp_path, production_model: BVAR
) -> None:
    model = production_model
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(model, baseline)
    path = tmp_path / "forecast.pkl"
    save_artifact(artifact, path)
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="integrity check"):
        load_artifact(path)
