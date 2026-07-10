from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from us_bvar.config import SeriesSpec


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
