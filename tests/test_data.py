from __future__ import annotations

import pandas as pd

from us_bvar.config import SERIES_SPECS
from us_bvar.data import FREDClient


def test_client_uses_existing_cache_without_key(tmp_path) -> None:
    spec = SERIES_SPECS[0]
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    pd.DataFrame({"date": dates, "value": [99.0, 100.0, 101.0, 102.0]}).to_csv(
        tmp_path / f"{spec.series_id}.csv", index=False
    )

    values, used_cache = FREDClient(api_key="", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache is True
    assert values.name == spec.series_id
    assert values.index.equals(dates)
    assert values.iloc[-1] == 102.0
