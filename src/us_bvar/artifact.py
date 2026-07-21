from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from us_bvar.config import SERIES_SPECS, SeriesSpec
from us_bvar.diagnostics import compute_posterior_diagnostics
from us_bvar.model import BVAR, BVARConfig, ForecastResult
from us_bvar.transforms import LevelTransformer, PlotTransformation, ScenarioConstraint

ARTIFACT_SCHEMA_VERSION = 6
RELEASE_MANIFEST_VERSION = 1
_RELEASE_FILE_KEYS = frozenset({"panel", "artifact", "checksum", "metadata"})


@dataclass(frozen=True)
class PublishedRelease:
    """A validated release selected by the small active pointer."""

    release_id: str
    artifact: ForecastArtifact
    artifact_path: Path
    panel_path: Path | None
    checksum_path: Path
    metadata_path: Path | None
    manifest_path: Path | None
    metadata: dict[str, object] | None
    legacy: bool


@dataclass(frozen=True)
class ForecastArtifact:
    """Versioned, pre-estimated model and baseline forecast consumed by Shiny."""

    schema_version: int
    created_at: pd.Timestamp
    panel_start: pd.Timestamp
    panel_end: pd.Timestamp
    observation_count: int
    model: BVAR
    baseline: ForecastResult

    @classmethod
    def create(
        cls,
        model: BVAR,
        baseline: ForecastResult,
        created_at: pd.Timestamp | None = None,
    ) -> ForecastArtifact:
        if model.history_levels is None:
            raise ValueError("The artifact model must be fitted.")
        history_dates = pd.DatetimeIndex(model.history_levels.index)
        model.clear_derived_cache()
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            created_at=created_at or pd.Timestamp.now(tz="UTC"),
            panel_start=cast(pd.Timestamp, history_dates[0]),
            panel_end=cast(pd.Timestamp, history_dates[-1]),
            observation_count=len(model.history_levels),
            model=model,
            baseline=baseline,
        )


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Artifact metadata contains unsupported type {type(value).__name__}.")


def _constraint_record(step: int, series_id: str, constraint: object) -> dict[str, object]:
    if isinstance(constraint, ScenarioConstraint):
        value = constraint.value
        transformation = constraint.transformation
    else:
        try:
            value = float(cast(float, constraint))
        except (TypeError, ValueError) as exc:
            raise ValueError("Artifact constraint values must be numeric.") from exc
        transformation = "level"
    return {
        "step": step,
        "series_id": series_id,
        "value": value,
        "transformation": transformation,
    }


def _artifact_payload(
    artifact: ForecastArtifact,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    model = artifact.model
    baseline = artifact.baseline
    if (
        model.transformer is None
        or model.observed_levels is None
        or model.observation_mask is None
        or model.history_levels is None
        or model.history_model is None
        or model.posterior_coefficients is None
        or model.posterior_sigmas is None
        or model.posterior_terminal_states is None
        or model.posterior_state_paths is None
        or model.posterior_mean is None
        or model.mcmc_log_likelihood is None
        or model.companion_radii is None
        or model.mcmc_chain_ids is None
        or model.fixed_control_coefficients is None
    ):
        raise ValueError("The artifact contains an unfitted mixed-frequency model.")
    metadata: dict[str, object] = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at.isoformat(),
        "panel_start": artifact.panel_start.isoformat(),
        "panel_end": artifact.panel_end.isoformat(),
        "observation_count": artifact.observation_count,
        "model": {
            "specs": [asdict(spec) for spec in model.specs],
            "config": asdict(model.config),
            "history_dates": [date.isoformat() for date in model.history_levels.index],
            "convergence_diagnostics": model.convergence_diagnostics,
            "unstable_draws_rejected": model.unstable_draws_rejected,
            "retention_attempts": model.retention_attempts,
        },
        "baseline": {
            "dates": [date.isoformat() for date in baseline.dates],
            "constraints": [
                _constraint_record(step, series_id, constraint)
                for (step, series_id), constraint in sorted(baseline.constraints.items())
            ],
            "draws": baseline.draws,
            "interval": list(baseline.interval),
            "component_effective_sample_size": baseline.component_effective_sample_size,
        },
    }
    arrays = {
        "transformer_means": model.transformer.means,
        "transformer_scales": model.transformer.scales,
        "observed_levels": model.observed_levels.to_numpy(dtype=float),
        "observation_mask": model.observation_mask.to_numpy(dtype=np.uint8),
        "history_levels": model.history_levels.to_numpy(dtype=float),
        "history_model": model.history_model.to_numpy(dtype=float),
        "posterior_coefficients": model.posterior_coefficients,
        "posterior_sigmas": model.posterior_sigmas,
        "posterior_terminal_states": model.posterior_terminal_states,
        "posterior_state_paths": model.posterior_state_paths,
        "posterior_mean": model.posterior_mean,
        "mcmc_log_likelihood": model.mcmc_log_likelihood,
        "companion_radii": model.companion_radii,
        "mcmc_chain_ids": model.mcmc_chain_ids,
        "fixed_control_coefficients": model.fixed_control_coefficients,
        "baseline_median": baseline.median.to_numpy(dtype=float),
        "baseline_lower": baseline.lower.to_numpy(dtype=float),
        "baseline_upper": baseline.upper.to_numpy(dtype=float),
        "baseline_samples": baseline.samples,
    }
    return metadata, arrays


def save_artifact(artifact: ForecastArtifact, path: Path | str) -> None:
    """Atomically persist an artifact as a code-free compressed NumPy archive."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata, arrays = _artifact_payload(artifact)
    metadata_json = json.dumps(
        metadata, default=_json_default, allow_nan=False, separators=(",", ":")
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(  # pyright: ignore[reportArgumentType]
                temporary,
                metadata=np.asarray(metadata_json),
                allow_pickle=False,
                **arrays,
            )
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
        _write_checksum(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _timestamp(value: object, label: str) -> pd.Timestamp:
    if not isinstance(value, str):
        raise ValueError(f"Artifact {label} must be an ISO timestamp.")
    parsed = pd.DatetimeIndex([value])
    if parsed.isna().any():
        raise ValueError(f"Artifact {label} must be an ISO timestamp.")
    return cast(pd.Timestamp, parsed[0])


def _load_artifact_archive(source: Path) -> ForecastArtifact:
    try:
        with np.load(source, allow_pickle=False) as archive:
            metadata_value = archive["metadata"]
            if metadata_value.ndim != 0:
                raise ValueError("Artifact metadata has an unexpected shape.")
            metadata = json.loads(str(metadata_value.item()))
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in archive.files
                if name != "metadata"
            }
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("The artifact is not a valid code-free NumPy archive.") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Artifact metadata must be a JSON object.")
    try:
        schema_version = int(metadata["schema_version"])
        model_metadata = metadata["model"]
        baseline_metadata = metadata["baseline"]
        if not isinstance(model_metadata, dict) or not isinstance(baseline_metadata, dict):
            raise TypeError("model and baseline metadata must be objects")
        raw_specs = model_metadata["specs"]
        raw_config = model_metadata["config"]
        if not isinstance(raw_specs, list) or not isinstance(raw_config, dict):
            raise TypeError("model specs and config are malformed")
        specs = tuple(SeriesSpec(**spec) for spec in raw_specs)
        config_values = dict(raw_config)
        config_values["pandemic_months"] = tuple(config_values["pandemic_months"])
        config_values["interval"] = tuple(config_values["interval"])
        model = BVAR(specs=specs, config=BVARConfig(**config_values))
        history_dates = pd.DatetimeIndex(model_metadata["history_dates"])
        columns = pd.Index(model.variable_ids, dtype="object")
        model.transformer = LevelTransformer(
            specs, arrays["transformer_means"], arrays["transformer_scales"]
        )
        model.observed_levels = pd.DataFrame(
            arrays["observed_levels"], index=history_dates, columns=columns
        )
        model.observation_mask = pd.DataFrame(
            arrays["observation_mask"].astype(bool), index=history_dates, columns=columns
        )
        model.history_levels = pd.DataFrame(
            arrays["history_levels"], index=history_dates, columns=columns
        )
        model.history_model = pd.DataFrame(
            arrays["history_model"], index=history_dates, columns=columns
        )
        for name in (
            "posterior_coefficients",
            "posterior_sigmas",
            "posterior_terminal_states",
            "posterior_state_paths",
            "posterior_mean",
            "mcmc_log_likelihood",
            "companion_radii",
            "mcmc_chain_ids",
            "fixed_control_coefficients",
        ):
            setattr(model, name, arrays[name])
        model.mcmc_chain_ids = arrays["mcmc_chain_ids"].astype(int)
        raw_diagnostics = model_metadata["convergence_diagnostics"]
        model.convergence_diagnostics = (
            dict(raw_diagnostics) if isinstance(raw_diagnostics, dict) else None
        )
        model.unstable_draws_rejected = int(model_metadata["unstable_draws_rejected"])
        model.retention_attempts = int(model_metadata["retention_attempts"])

        forecast_dates = pd.DatetimeIndex(baseline_metadata["dates"])
        constraints = {
            (int(item["step"]), str(item["series_id"])): ScenarioConstraint(
                value=float(item["value"]),
                transformation=cast(PlotTransformation, str(item["transformation"])),
            )
            for item in baseline_metadata["constraints"]
        }
        samples = arrays["baseline_samples"]
        samples.setflags(write=False)

        def frame(name: str) -> pd.DataFrame:
            return pd.DataFrame(arrays[name], index=forecast_dates, columns=columns)

        raw_component_ess = baseline_metadata["component_effective_sample_size"]
        raw_interval = baseline_metadata["interval"]
        if not isinstance(raw_interval, list) or len(raw_interval) != 2:
            raise ValueError("Artifact forecast interval must contain two probabilities.")
        interval = (float(raw_interval[0]), float(raw_interval[1]))
        baseline = ForecastResult(
            dates=forecast_dates,
            median=frame("baseline_median"),
            lower=frame("baseline_lower"),
            upper=frame("baseline_upper"),
            samples=samples,
            constraints=constraints,
            draws=int(baseline_metadata["draws"]),
            interval=interval,
            component_effective_sample_size=(
                None if raw_component_ess is None else float(raw_component_ess)
            ),
        )
        return ForecastArtifact(
            schema_version=schema_version,
            created_at=_timestamp(metadata["created_at"], "creation time"),
            panel_start=_timestamp(metadata["panel_start"], "panel start"),
            panel_end=_timestamp(metadata["panel_end"], "panel end"),
            observation_count=int(metadata["observation_count"]),
            model=model,
            baseline=baseline,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The artifact archive metadata or arrays are malformed.") from exc


def load_artifact(path: Path | str, *, expected_digest: str | None = None) -> ForecastArtifact:
    """Load a code-free artifact, optionally requiring a deployment-pinned digest."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Precomputed forecast not found at {source}. "
            "Run `uv run python scripts/precompute.py`."
        )
    _verify_checksum(source)
    pinned_digest = expected_digest or os.getenv("BVAR_ARTIFACT_SHA256")
    if pinned_digest is not None:
        try:
            validated_digest = _validate_digest(pinned_digest)
        except ValueError as exc:
            raise ValueError("BVAR_ARTIFACT_SHA256 must be a SHA-256 digest.") from exc
        if artifact_sha256(source) != validated_digest:
            raise ValueError("The artifact does not match the deployment-pinned SHA-256 digest.")
    artifact = _load_artifact_archive(source)
    if artifact.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact schema {artifact.schema_version} is not supported; "
            f"expected {ARTIFACT_SCHEMA_VERSION}. Re-run precompute."
        )
    if len(artifact.baseline.dates) != 12:
        raise ValueError("The dashboard requires a 12-month precomputed baseline.")
    _validate_artifact(artifact)
    return artifact


def artifact_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest of an artifact without loading its array payload."""

    digest = sha256()
    with Path(path).open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.sha256")


def _safe_release_path(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("The release manifest contains an unsafe file path.")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("The release manifest points outside the project root.") from exc
    return resolved


def _validate_digest(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("The release manifest contains a non-string digest.")
    digest = value.lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("The release manifest contains a malformed SHA-256 digest.")
    return digest


def create_release_manifest(
    root: Path | str,
    release_id: str,
    files: Mapping[str, Path | str],
    schema_version: int,
) -> dict[str, object]:
    """Describe a complete release without changing the active pointer."""

    project_root = Path(root).resolve()
    if not release_id or set(files) != _RELEASE_FILE_KEYS:
        raise ValueError("A release needs an id and panel, artifact, checksum, and metadata files.")
    entries: dict[str, dict[str, str]] = {}
    for key in sorted(_RELEASE_FILE_KEYS):
        path = Path(files[key]).resolve()
        try:
            relative = path.relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError("Release files must be inside the project root.") from exc
        entries[key] = {"path": relative, "sha256": artifact_sha256(path)}
    return {
        "manifest_version": RELEASE_MANIFEST_VERSION,
        "release_id": release_id,
        "schema_version": schema_version,
        "files": entries,
    }


def validate_release_manifest(
    root: Path | str, manifest: Mapping[str, object]
) -> tuple[dict[str, Path], dict[str, object]]:
    """Validate every staged release file and return resolved paths and metadata."""

    project_root = Path(root).resolve()
    if manifest.get("manifest_version") != RELEASE_MANIFEST_VERSION:
        raise ValueError("The release manifest version is not supported.")
    release_id = manifest.get("release_id")
    schema_version = manifest.get("schema_version")
    files = manifest.get("files")
    if not isinstance(release_id, str) or not release_id:
        raise ValueError("The release manifest has no release id.")
    if not isinstance(schema_version, int):
        raise ValueError("The release manifest has no artifact schema version.")
    if not isinstance(files, Mapping) or set(files) != _RELEASE_FILE_KEYS:
        raise ValueError("The release manifest does not describe the complete release.")

    paths: dict[str, Path] = {}
    digests: dict[str, str] = {}
    for key in sorted(_RELEASE_FILE_KEYS):
        entry = files[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"The release manifest entry for {key} is malformed.")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str):
            raise ValueError(f"The release manifest path for {key} is malformed.")
        path = _safe_release_path(project_root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"The active release file is missing: {relative_path}")
        paths[key] = path
        digests[key] = _validate_digest(entry.get("sha256"))
        if artifact_sha256(path) != digests[key]:
            raise ValueError(f"The active release {key} failed its SHA-256 integrity check.")

    if paths["checksum"] != _checksum_path(paths["artifact"]):
        raise ValueError("The release checksum path does not match its artifact path.")
    checksum_text = paths["checksum"].read_text(encoding="utf-8").split()
    if len(checksum_text) < 2 or checksum_text[0].lower() != digests["artifact"]:
        raise ValueError("The release checksum does not describe its artifact.")
    if Path(checksum_text[1]).name != paths["artifact"].name:
        raise ValueError("The release checksum names the wrong artifact.")

    try:
        metadata_value = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("The release metadata is not valid JSON.") from exc
    if not isinstance(metadata_value, dict):
        raise ValueError("The release metadata is not a JSON object.")
    if metadata_value.get("release_id") != release_id:
        raise ValueError("The release metadata has a stale release id.")
    if metadata_value.get("schema_version") != schema_version:
        raise ValueError("The release metadata has a stale artifact schema.")
    if metadata_value.get("artifact_sha256") != digests["artifact"]:
        raise ValueError("The release metadata has a stale artifact digest.")
    if metadata_value.get("panel_sha256") != digests["panel"]:
        raise ValueError("The release metadata has a stale panel digest.")
    return paths, cast(dict[str, object], metadata_value)


def activate_release(root: Path | str, manifest: Mapping[str, object]) -> Path:
    """Atomically replace the active release pointer after validating its files."""

    project_root = Path(root).resolve()
    validate_release_manifest(project_root, manifest)
    destination = project_root / "artifacts/active.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(manifest, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _validate_release_contents(
    panel_path: Path, metadata: Mapping[str, object], artifact: ForecastArtifact
) -> None:
    """Cross-check digest-verified release files against the fitted artifact."""

    try:
        panel = pd.read_csv(panel_path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError("The active release panel is not a valid CSV file.") from exc
    variable_ids = list(artifact.model.variable_ids)
    if list(panel.columns) != ["date", *variable_ids]:
        raise ValueError("The active release panel has unexpected variables or column order.")
    dates = pd.to_datetime(panel.pop("date"), errors="coerce")
    if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("The active release panel has invalid or unsorted dates.")
    numeric_panel = pd.DataFrame(
        {column: pd.to_numeric(panel.loc[:, column], errors="coerce") for column in variable_ids}
    )
    if (panel.notna() & numeric_panel.isna()).to_numpy().any():
        raise ValueError("The active release panel contains non-numeric observations.")
    numeric_panel.index = pd.DatetimeIndex(dates)
    observed = artifact.model.observed_levels
    if (
        observed is None
        or not numeric_panel.index.equals(observed.index)
        or not np.allclose(
            numeric_panel.to_numpy(dtype=float),
            observed.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        )
    ):
        raise ValueError("The active release panel does not match the fitted artifact.")

    optional_metadata = {
        "panel_start": f"{artifact.panel_start:%Y-%m-%d}",
        "panel_end": f"{artifact.panel_end:%Y-%m-%d}",
        "observation_count": artifact.observation_count,
        "variable_count": len(variable_ids),
        "forecast_start": f"{artifact.baseline.dates[0]:%Y-%m-%d}",
        "forecast_end": f"{artifact.baseline.dates[-1]:%Y-%m-%d}",
        "posterior_draws": artifact.baseline.draws,
        "posterior_interval_quantiles": list(artifact.baseline.interval),
    }
    for key, expected in optional_metadata.items():
        if key in metadata and metadata[key] != expected:
            raise ValueError(f"The active release metadata has a stale {key} value.")


def load_published_release(root: Path | str) -> PublishedRelease:
    """Load the complete digest-verified release selected by the active pointer."""

    project_root = Path(root).resolve()
    manifest_path = project_root / "artifacts/active.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "No active model release exists. Run `uv run python scripts/precompute.py` "
            "to publish a validated release."
        )

    try:
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("The active release pointer is not valid JSON.") from exc
    if not isinstance(manifest_value, dict):
        raise ValueError("The active release pointer is not a JSON object.")
    paths, metadata = validate_release_manifest(project_root, manifest_value)
    artifact = load_artifact(paths["artifact"])
    if artifact.schema_version != manifest_value.get("schema_version"):
        raise ValueError("The active release schema does not match its artifact.")
    _validate_release_contents(paths["panel"], metadata, artifact)
    return PublishedRelease(
        release_id=cast(str, manifest_value["release_id"]),
        artifact=artifact,
        artifact_path=paths["artifact"],
        panel_path=paths["panel"],
        checksum_path=paths["checksum"],
        metadata_path=paths["metadata"],
        manifest_path=manifest_path,
        metadata=metadata,
        legacy=False,
    )


def _write_checksum(path: Path) -> None:
    checksum_path = _checksum_path(path)
    content = f"{artifact_sha256(path)}  {path.name}\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{checksum_path.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        os.replace(temporary_path, checksum_path)
        checksum_path.chmod(0o644)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _verify_checksum(path: Path) -> None:
    checksum_path = _checksum_path(path)
    if not checksum_path.exists():
        raise ValueError(
            f"Artifact checksum not found at {checksum_path}. Re-run precompute before deployment."
        )
    expected = checksum_path.read_text(encoding="utf-8").split(maxsplit=1)[0].lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("The artifact checksum file is malformed. Re-run precompute.")
    if artifact_sha256(path) != expected:
        raise ValueError("The precomputed forecast failed its SHA-256 integrity check.")


def _validate_artifact(artifact: ForecastArtifact) -> None:
    model = artifact.model
    baseline = artifact.baseline
    if (
        model.history_levels is None
        or model.history_model is None
        or model.observed_levels is None
        or model.observation_mask is None
        or model.posterior_coefficients is None
        or model.posterior_sigmas is None
        or model.posterior_terminal_states is None
        or model.posterior_state_paths is None
        or model.transformer is None
        or model.mcmc_log_likelihood is None
        or model.companion_radii is None
        or model.mcmc_chain_ids is None
        or model.posterior_mean is None
        or model.fixed_control_coefficients is None
    ):
        raise ValueError("The artifact contains an unfitted mixed-frequency model.")
    raw_diagnostics = getattr(model, "convergence_diagnostics", None)
    if not isinstance(raw_diagnostics, dict):
        raise ValueError("The artifact has no MCMC convergence release diagnostics.")
    variable_ids = list(model.variable_ids)
    configured_ids = [spec.series_id for spec in SERIES_SPECS]
    if variable_ids != configured_ids:
        raise ValueError(
            "The artifact variable panel does not match the configured dashboard series."
        )
    if artifact.observation_count != len(model.history_levels):
        raise ValueError("The artifact observation count does not match the model history.")
    posterior_draws = len(model.posterior_coefficients)
    variables = len(variable_ids)
    state_dimension = variables * model.config.lags
    controls = len(model.config.pandemic_months)
    regressors = 1 + state_dimension + controls
    if posterior_draws < 1 or model.posterior_coefficients.shape != (
        posterior_draws,
        regressors,
        variables,
    ):
        raise ValueError("The artifact coefficient posterior is malformed.")
    if model.posterior_sigmas.shape != (posterior_draws, variables, variables):
        raise ValueError("The artifact innovation-covariance posterior is malformed.")
    if model.posterior_terminal_states.shape != (posterior_draws, state_dimension):
        raise ValueError("The artifact terminal-state posterior is malformed.")
    if model.posterior_state_paths.shape != (
        posterior_draws,
        len(model.history_levels),
        variables,
    ):
        raise ValueError("The artifact latent-state posterior is malformed.")
    diagnostic_arrays = (
        model.mcmc_log_likelihood,
        model.companion_radii,
        model.mcmc_chain_ids,
    )
    if any(array.shape != (posterior_draws,) for array in diagnostic_arrays):
        raise ValueError("The artifact MCMC diagnostics are malformed.")
    if model.posterior_mean.shape != (regressors, variables):
        raise ValueError("The artifact posterior coefficient mean is malformed.")
    if model.fixed_control_coefficients.shape != (controls, variables):
        raise ValueError("The artifact fixed pandemic controls are malformed.")
    if controls and not np.allclose(
        model.posterior_coefficients[:, -controls:, :],
        model.fixed_control_coefficients[None, :, :],
    ):
        raise ValueError("The artifact pandemic controls were not held fixed across MCMC draws.")
    expected_chains = set(range(model.config.mcmc_chains))
    if set(model.mcmc_chain_ids) != expected_chains:
        raise ValueError("The artifact MCMC chain identifiers are inconsistent.")
    diagnostics: dict[str, object] = raw_diagnostics
    try:
        accepted = diagnostics["accepted"]
        diagnostic_chains = np.asarray(diagnostics["chains"], dtype=int).item()
        retained_per_chain = np.asarray(diagnostics["retained_draws_per_chain"], dtype=int)
        stored_maximum_r_hat = np.asarray(diagnostics["maximum_r_hat"], dtype=float).item()
        stored_minimum_ess = np.asarray(
            diagnostics["minimum_effective_sample_size"], dtype=float
        ).item()
        transition_rows = 1 + variables * model.config.lags
        computed_diagnostics = compute_posterior_diagnostics(
            coefficients=model.posterior_coefficients,
            innovation_covariances=model.posterior_sigmas,
            terminal_states=model.posterior_terminal_states,
            latent_state_paths=model.posterior_state_paths,
            log_likelihood=model.mcmc_log_likelihood,
            companion_radius=model.companion_radii,
            chain_ids=model.mcmc_chain_ids,
            transition_rows=transition_rows,
            chains=model.config.mcmc_chains,
            policy=model.config.convergence_policy,
        )
        scalar_diagnostics = {
            name: cast(dict[str, float], computed_diagnostics[name])
            for name in ("log_likelihood", "companion_radius")
        }
        array_diagnostics = {
            name: cast(dict[str, object], computed_diagnostics[name])
            for name in (
                "transition_coefficients",
                "fixed_pandemic_control_coefficients",
                "innovation_covariances",
                "terminal_states",
                "latent_state_paths",
            )
        }
        gate = computed_diagnostics
        stored_sections = {
            **{name: diagnostics[name] for name in scalar_diagnostics},
            **{name: diagnostics[name] for name in array_diagnostics},
        }
        tail_keys = frozenset(
            {
                "r_hat_outside_nominal_count",
                "r_hat_outside_nominal_fraction",
                "effective_sample_size_outside_nominal_count",
                "effective_sample_size_outside_nominal_fraction",
            }
        )
        for section_name, computed in array_diagnostics.items():
            stored = stored_sections[section_name]
            if not isinstance(stored, dict):
                raise TypeError(f"Diagnostic section {section_name} is not a mapping.")
            stored_tail_keys = tail_keys.intersection(stored)
            if stored_tail_keys and stored_tail_keys != tail_keys:
                raise ValueError(f"Diagnostic section {section_name} has incomplete tail fields.")
            for key in stored_tail_keys:
                if not np.isclose(
                    np.asarray(stored[key], dtype=float), np.asarray(computed[key], dtype=float)
                ):
                    raise ValueError(f"Diagnostic section {section_name} has stale tail counts.")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The artifact convergence diagnostics are malformed.") from exc
    expected_retained = np.bincount(model.mcmc_chain_ids, minlength=model.config.mcmc_chains)
    minimum_retained = getattr(model.config, "minimum_retained_draws_per_chain", 20)

    def has_tail_record(name: str) -> bool:
        section = diagnostics.get(name)
        return isinstance(section, dict) and bool(tail_keys.intersection(section))

    has_recorded_tail_policy = any(has_tail_record(name) for name in array_diagnostics)
    if has_recorded_tail_policy:
        for key in tail_keys:
            if key not in diagnostics or not np.isclose(
                np.asarray(diagnostics[key], dtype=float), np.asarray(gate[key], dtype=float)
            ):
                raise ValueError("The artifact has stale aggregate convergence tail counts.")
    stored_summary_is_acceptable = (
        np.isfinite(stored_maximum_r_hat)
        and stored_maximum_r_hat <= model.config.maximum_mcmc_r_hat
        and np.isfinite(stored_minimum_ess)
        and stored_minimum_ess >= model.config.minimum_mcmc_effective_sample_size
    )
    summaries_match_raw_gate = not has_recorded_tail_policy or (
        np.isclose(stored_maximum_r_hat, cast(float, gate["maximum_r_hat"]))
        and np.isclose(stored_minimum_ess, cast(float, gate["minimum_effective_sample_size"]))
    )
    # Releases created before tail-policy fields were recorded used the previous ESS
    # estimator. Permit only ESS-only differences for those digest-verified releases;
    # current R-hat failures and every newly generated release still fail validation.
    raw_failures = gate.get("failures")
    legacy_ess_only_difference = (
        not has_recorded_tail_policy
        and isinstance(raw_failures, list)
        and bool(raw_failures)
        and all(isinstance(failure, str) and "ESS" in failure for failure in raw_failures)
    )
    raw_gate_is_acceptable = bool(gate["accepted"]) or legacy_ess_only_difference
    if (
        not isinstance(accepted, bool)
        or not accepted
        or not raw_gate_is_acceptable
        or diagnostic_chains != model.config.mcmc_chains
        or model.config.mcmc_chains < 2
        or retained_per_chain.shape != (model.config.mcmc_chains,)
        or not np.array_equal(retained_per_chain, expected_retained)
        or np.min(retained_per_chain) < minimum_retained
        or not stored_summary_is_acceptable
        or not summaries_match_raw_gate
    ):
        raise ValueError("The artifact did not pass its MCMC convergence release gate.")
    if np.max(model.companion_radii) > model.config.max_companion_radius:
        raise ValueError("The artifact exceeds its companion-radius acceptance threshold.")
    retention_attempts = getattr(model, "retention_attempts", 0)
    unstable_rejections = getattr(model, "unstable_draws_rejected", -1)
    if (
        retention_attempts < posterior_draws
        or unstable_rejections < 0
        or retention_attempts - unstable_rejections != posterior_draws
        or unstable_rejections / retention_attempts > model.config.maximum_unstable_draw_fraction
    ):
        raise ValueError("The artifact has invalid unstable-draw rejection diagnostics.")
    posterior_arrays = (
        model.posterior_coefficients,
        model.posterior_sigmas,
        model.posterior_terminal_states,
        model.posterior_state_paths,
        model.mcmc_log_likelihood,
        model.companion_radii,
        model.posterior_mean,
        model.fixed_control_coefficients,
    )
    if any(not np.isfinite(array).all() for array in posterior_arrays):
        raise ValueError("The artifact posterior contains non-finite values.")
    if np.any(np.linalg.eigvalsh(model.posterior_sigmas) <= 0):
        raise ValueError("The artifact contains a non-positive-definite innovation covariance.")

    expected_columns = variable_ids
    history_dates = pd.DatetimeIndex(model.history_levels.index)
    expected_dates = pd.date_range(history_dates[0], history_dates[-1], freq="MS")
    if not np.array_equal(history_dates.to_numpy(), expected_dates.to_numpy()):
        raise ValueError("The artifact history is not a contiguous monthly calendar.")
    history_frames = (model.history_levels, model.history_model, model.observed_levels)
    if any(list(frame.columns) != expected_columns for frame in history_frames):
        raise ValueError("The artifact mixed-frequency history has unexpected variables.")
    if not all(
        np.array_equal(frame.index.to_numpy(), model.history_levels.index.to_numpy())
        for frame in history_frames[1:]
    ):
        raise ValueError("The observed and smoothed mixed-frequency histories are misaligned.")
    if list(model.observation_mask.columns) != expected_columns or not np.array_equal(
        model.observation_mask.index.to_numpy(), model.history_levels.index.to_numpy()
    ):
        raise ValueError("The artifact observation mask is misaligned.")
    if not np.array_equal(
        model.observation_mask.to_numpy(), model.observed_levels.notna().to_numpy()
    ):
        raise ValueError("The artifact observation mask does not match source missingness.")
    if (
        not np.isfinite(model.history_levels.to_numpy(dtype=float)).all()
        or not np.isfinite(model.history_model.to_numpy(dtype=float)).all()
    ):
        raise ValueError("The artifact smoothed history contains non-finite values.")
    observed_values = model.observed_levels.to_numpy(dtype=float)
    if np.any(~np.isnan(observed_values) & ~np.isfinite(observed_values)):
        raise ValueError("The artifact observed history contains invalid values.")
    transformer_ids = [spec.series_id for spec in model.transformer.specs]
    if transformer_ids != variable_ids:
        raise ValueError("The artifact transformer series order is inconsistent.")
    if model.transformer.means.shape != (variables,) or model.transformer.scales.shape != (
        variables,
    ):
        raise ValueError("The artifact transformer parameters are malformed.")
    if not np.isfinite(model.transformer.means).all() or not np.all(
        np.isfinite(model.transformer.scales) & (model.transformer.scales > 0)
    ):
        raise ValueError("The artifact transformer parameters are invalid.")
    if artifact.panel_start != model.history_levels.index[0]:
        raise ValueError("The artifact panel start does not match the model history.")
    if artifact.panel_end != model.history_levels.index[-1]:
        raise ValueError("The artifact panel end does not match the model history.")
    if baseline.draws < 20 or baseline.samples.shape != (
        baseline.draws,
        len(baseline.dates),
        len(variable_ids),
    ):
        raise ValueError("The artifact posterior samples have an unexpected shape.")
    if baseline.interval != model.config.interval:
        raise ValueError("The artifact interval does not match the model configuration.")
    if baseline.dates[0] != artifact.panel_end + pd.offsets.MonthBegin(1):
        raise ValueError("The artifact forecast does not begin after the panel end.")
    expected_forecast_dates = pd.date_range(
        baseline.dates[0], periods=len(baseline.dates), freq="MS"
    )
    if not np.array_equal(baseline.dates.to_numpy(), expected_forecast_dates.to_numpy()):
        raise ValueError("The artifact forecast dates are not a contiguous monthly calendar.")
    if baseline.constraints:
        raise ValueError("The artifact baseline must not contain scenario constraints.")
    for frame in (baseline.median, baseline.lower, baseline.upper):
        if list(frame.columns) != variable_ids or not np.array_equal(
            frame.index.to_numpy(), baseline.dates.to_numpy()
        ):
            raise ValueError("The artifact forecast frame has unexpected dates or variables.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("The artifact forecast contains non-finite summary values.")
    if not np.isfinite(baseline.samples).all():
        raise ValueError("The artifact posterior samples contain non-finite values.")
    lower_q, upper_q = baseline.interval
    expected_summaries = (
        np.median(baseline.samples, axis=0),
        np.quantile(baseline.samples, lower_q, axis=0),
        np.quantile(baseline.samples, upper_q, axis=0),
    )
    for label, frame, expected in zip(
        ("median", "lower", "upper"),
        (baseline.median, baseline.lower, baseline.upper),
        expected_summaries,
        strict=True,
    ):
        if not np.allclose(frame.to_numpy(dtype=float), expected, rtol=1e-10, atol=1e-12):
            raise ValueError(f"The artifact forecast {label} is inconsistent with its samples.")
    if not (
        (baseline.lower.to_numpy() <= baseline.median.to_numpy()).all()
        and (baseline.median.to_numpy() <= baseline.upper.to_numpy()).all()
    ):
        raise ValueError("The artifact forecast intervals are not ordered.")
