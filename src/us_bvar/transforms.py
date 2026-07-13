from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from us_bvar.config import SeriesSpec

PlotTransformation = Literal[
    "level",
    "mom",
    "qoq",
    "yoy",
    "mom_annualized",
    "qoq_annualized",
    "yoy_annualized",
]


@dataclass(frozen=True)
class TransformationSpec:
    label: str
    periods: int
    annualization_factor: float
    annualized: bool = False


PLOT_TRANSFORMATIONS: dict[PlotTransformation, TransformationSpec] = {
    "level": TransformationSpec("Level", periods=0, annualization_factor=1.0),
    "mom": TransformationSpec("MoM", periods=1, annualization_factor=1.0),
    "qoq": TransformationSpec("QoQ", periods=3, annualization_factor=1.0),
    "yoy": TransformationSpec("YoY", periods=12, annualization_factor=1.0),
    "mom_annualized": TransformationSpec(
        "MoM · annualized", periods=1, annualization_factor=12.0, annualized=True
    ),
    "qoq_annualized": TransformationSpec(
        "QoQ · annualized", periods=3, annualization_factor=4.0, annualized=True
    ),
    "yoy_annualized": TransformationSpec(
        "YoY · annualized", periods=12, annualization_factor=1.0, annualized=True
    ),
}


@dataclass(frozen=True)
class ScenarioConstraint:
    """One user-entered scenario value and the scale on which it was entered."""

    value: float
    transformation: PlotTransformation = "level"


def transformation_spec(transformation: PlotTransformation) -> TransformationSpec:
    try:
        return PLOT_TRANSFORMATIONS[transformation]
    except KeyError as exc:
        raise ValueError(f"Unknown transformation: {transformation}") from exc


def transform_path(
    values: pd.Series,
    series_spec: SeriesSpec,
    transformation: PlotTransformation,
) -> pd.Series:
    """Convert a natural-unit path to one of the dashboard display scales."""

    spec = transformation_spec(transformation)
    if transformation == "level":
        return values.astype(float)

    lagged = values.shift(spec.periods)
    if series_spec.transform == "log":
        ratio = values / lagged
        if spec.annualized:
            return (ratio.pow(spec.annualization_factor) - 1.0) * 100.0
        return (ratio - 1.0) * 100.0

    change = values - lagged
    if spec.annualized:
        change *= spec.annualization_factor
    return change


def transform_forecast_samples(
    history: pd.Series,
    samples: np.ndarray,
    series_spec: SeriesSpec,
    transformation: PlotTransformation,
) -> np.ndarray:
    """Transform forecast draws while preserving their cross-horizon dependence."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 2:
        raise ValueError("Forecast samples must have shape (draws, horizon).")
    if transformation == "level":
        return values.copy()

    spec = transformation_spec(transformation)
    references = np.empty_like(values)
    historical = history.to_numpy(dtype=float)
    for step in range(values.shape[1]):
        lag_step = step - spec.periods
        references[:, step] = values[:, lag_step] if lag_step >= 0 else historical[lag_step]

    if series_spec.transform == "log":
        ratios = values / references
        exponent = spec.annualization_factor if spec.annualized else 1.0
        return (np.power(ratios, exponent) - 1.0) * 100.0

    changes = values - references
    if spec.annualized:
        changes *= spec.annualization_factor
    return changes


@dataclass(frozen=True)
class LevelTransformer:
    """Log selected levels and standardize every model variable."""

    specs: tuple[SeriesSpec, ...]
    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, levels: pd.DataFrame, specs: tuple[SeriesSpec, ...]) -> LevelTransformer:
        latent = cls._latent(levels.to_numpy(dtype=float), specs)
        scales = latent.std(axis=0, ddof=1)
        if np.any(~np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("Every model variable must have positive finite sample variance.")
        return cls(specs=specs, means=latent.mean(axis=0), scales=scales)

    def encode_frame(self, levels: pd.DataFrame) -> pd.DataFrame:
        ids = [spec.series_id for spec in self.specs]
        latent = self._latent(levels[ids].to_numpy(dtype=float), self.specs)
        values = (latent - self.means) / self.scales
        return pd.DataFrame(values, index=levels.index, columns=ids)

    def encode_value(self, variable_index: int, natural_value: float) -> float:
        if not np.isfinite(natural_value):
            raise ValueError("Scenario values must be finite numbers.")
        spec = self.specs[variable_index]
        if spec.transform == "log":
            if natural_value <= 0:
                raise ValueError(f"{spec.short_label} must be greater than zero.")
            latent = np.log(natural_value)
        else:
            latent = natural_value
        return float((latent - self.means[variable_index]) / self.scales[variable_index])

    def decode_array(self, model_values: np.ndarray) -> np.ndarray:
        latent = model_values * self.scales + self.means
        result = latent.copy()
        for index, spec in enumerate(self.specs):
            if spec.transform == "log":
                result[..., index] = np.exp(latent[..., index])
        return result

    @staticmethod
    def _latent(values: np.ndarray, specs: tuple[SeriesSpec, ...]) -> np.ndarray:
        result = values.copy()
        for index, spec in enumerate(specs):
            if spec.transform == "log":
                if np.any(values[..., index] <= 0):
                    raise ValueError(f"{spec.short_label} contains a non-positive value.")
                result[..., index] = np.log(values[..., index])
        return result
