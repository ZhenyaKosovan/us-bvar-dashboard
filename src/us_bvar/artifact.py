from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from us_bvar.model import BVAR, ForecastResult

ARTIFACT_SCHEMA_VERSION = 3


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
        return cls(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            created_at=created_at or pd.Timestamp.now(tz="UTC"),
            panel_start=pd.Timestamp(model.history_levels.index[0]),
            panel_end=pd.Timestamp(model.history_levels.index[-1]),
            observation_count=len(model.history_levels),
            model=model,
            baseline=baseline,
        )


def save_artifact(artifact: ForecastArtifact, path: Path | str) -> None:
    """Atomically persist a trusted local artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            pickle.dump(artifact, temporary, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, destination)
        destination.chmod(0o644)
        _write_checksum(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_artifact(path: Path | str) -> ForecastArtifact:
    """Load an artifact produced locally by :mod:`scripts.precompute`."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Precomputed forecast not found at {source}. "
            "Run `uv run python scripts/precompute.py`."
        )
    _verify_checksum(source)
    with source.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)  # noqa: S301 - trusted, locally generated artifact
    if not isinstance(artifact, ForecastArtifact):
        raise TypeError("The precomputed forecast has an unexpected object type.")
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
    """Return the SHA-256 digest of an artifact without loading its pickle payload."""

    digest = sha256()
    with Path(path).open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.sha256")


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
    if model.history_levels is None or model.history_model is None:
        raise ValueError("The artifact contains an unfitted model.")
    variable_ids = list(model.variable_ids)
    if artifact.observation_count != len(model.history_levels):
        raise ValueError("The artifact observation count does not match the model history.")
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
    for frame in (baseline.median, baseline.lower, baseline.upper):
        if list(frame.columns) != variable_ids or not frame.index.equals(baseline.dates):
            raise ValueError("The artifact forecast frame has unexpected dates or variables.")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("The artifact forecast contains non-finite summary values.")
    if not np.isfinite(baseline.samples).all():
        raise ValueError("The artifact posterior samples contain non-finite values.")
