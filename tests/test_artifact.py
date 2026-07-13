from __future__ import annotations

import pandas as pd
import pytest

from us_bvar.artifact import ARTIFACT_SCHEMA_VERSION, ForecastArtifact, load_artifact, save_artifact
from us_bvar.model import BVAR


def test_artifact_round_trip(tmp_path, synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(
        model,
        baseline,
        created_at=pd.Timestamp("2026-07-10T08:00:00Z"),
    )
    path = tmp_path / "forecast.pkl"

    save_artifact(artifact, path)
    restored = load_artifact(path)

    assert restored.schema_version == ARTIFACT_SCHEMA_VERSION
    assert restored.observation_count == len(synthetic_levels)
    assert restored.panel_end == synthetic_levels.index[-1]
    assert restored.baseline.median.equals(baseline.median)
    assert path.with_suffix(".pkl.sha256").is_file()


def test_artifact_corruption_is_rejected_before_unpickling(tmp_path, synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=17)
    artifact = ForecastArtifact.create(model, baseline)
    path = tmp_path / "forecast.pkl"
    save_artifact(artifact, path)
    path.write_bytes(path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="integrity check"):
        load_artifact(path)
