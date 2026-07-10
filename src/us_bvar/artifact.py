from __future__ import annotations

import os
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from us_bvar.model import BVAR, ForecastResult

ARTIFACT_SCHEMA_VERSION = 2


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
    return artifact
