from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR, BVARConfig


@pytest.fixture(scope="session")
def synthetic_levels() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    observations = 180
    variables = len(SERIES_SPECS)
    latent = np.zeros((observations, variables))
    shocks = rng.multivariate_normal(
        np.zeros(variables),
        0.015 * np.eye(variables) + 0.004 * np.ones((variables, variables)),
        size=observations,
    )
    for t in range(1, observations):
        latent[t] = 0.88 * latent[t - 1] + shocks[t]

    natural = np.empty_like(latent)
    monthly_gdp = np.exp(np.log(100.0) + latent[:, 0] * 0.12)
    natural[:, 0] = np.nan
    for quarter_end in range(2, observations, 3):
        natural[quarter_end, 0] = np.exp(
            np.mean(np.log(monthly_gdp[quarter_end - 2 : quarter_end + 1]))
        )
    bases = {
        "PCEC96": 14_000.0,
        "CPIAUCSL": 250.0,
        "UNRATE": 4.5,
        "FEDFUNDS": 3.0,
    }
    for variable, spec in enumerate(SERIES_SPECS[1:], start=1):
        base = bases.get(spec.series_id, 100.0 + 10.0 * variable)
        if spec.transform == "log":
            natural[:, variable] = np.exp(np.log(base) + latent[:, variable] * 0.10)
        else:
            natural[:, variable] = base + latent[:, variable]
    return pd.DataFrame(
        natural,
        index=pd.date_range("2010-01-01", periods=observations, freq="MS"),
        columns=pd.Index([spec.series_id for spec in SERIES_SPECS], dtype="object"),
    )


@pytest.fixture(scope="session")
def fitted_model(synthetic_levels) -> BVAR:
    """Share the comparatively expensive 22-variable smoke fit across read-only tests."""

    return BVAR(config=BVARConfig.quick()).fit(synthetic_levels)
