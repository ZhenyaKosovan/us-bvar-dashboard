from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from us_bvar.config import SERIES_SPECS


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
    natural[:, 0] = np.exp(np.log(100.0) + latent[:, 0] * 0.12)
    natural[:, 1] = np.exp(np.log(14_000.0) + latent[:, 1] * 0.15)
    natural[:, 2] = np.exp(np.log(250.0) + latent[:, 2] * 0.10)
    natural[:, 3] = 4.5 + latent[:, 3]
    natural[:, 4] = 3.0 + latent[:, 4]
    return pd.DataFrame(
        natural,
        index=pd.date_range("2010-01-01", periods=observations, freq="MS"),
        columns=[spec.series_id for spec in SERIES_SPECS],
    )
