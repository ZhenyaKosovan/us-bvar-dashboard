from __future__ import annotations

import pandas as pd
import pytest

from us_bvar import data as data_module
from us_bvar.config import SERIES_SPECS
from us_bvar.data import FREDClient


def test_model_panel_contains_22_unique_series_across_four_macro_blocks() -> None:
    assert len(SERIES_SPECS) == 22
    assert len({spec.series_id for spec in SERIES_SPECS}) == 22
    assert {spec.group for spec in SERIES_SPECS} == {
        "Activity",
        "Labor",
        "Prices",
        "Financial",
    }


def test_client_uses_existing_cache_without_key(tmp_path) -> None:
    spec = SERIES_SPECS[1]
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    pd.DataFrame({"date": dates, "value": [99.0, 100.0, 101.0, 102.0]}).to_csv(
        tmp_path / f"{spec.series_id}.csv", index=False
    )

    values, used_cache = FREDClient(api_key="", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache
    assert values.name == spec.series_id
    assert values.index.equals(dates)
    assert values.iloc[-1] == 102.0


def test_quarterly_gdp_cache_dates_are_mapped_to_quarter_end(tmp_path) -> None:
    spec = SERIES_SPECS[0]
    pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"],
            "value": [22_000.0, 22_100.0, 22_250.0, 22_400.0],
        }
    ).to_csv(tmp_path / f"{spec.series_id}.csv", index=False)

    values, used_cache = FREDClient(api_key="", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache
    expected = pd.DatetimeIndex(["2023-03-01", "2023-06-01", "2023-09-01", "2023-12-01"])
    assert values.index.equals(expected)


def test_panel_preserves_quarterly_gdp_and_monthly_ragged_edge(tmp_path) -> None:
    dates = pd.date_range("2010-01-01", periods=24, freq="MS")
    for index, spec in enumerate(SERIES_SPECS):
        if spec.frequency == "quarterly":
            source_dates = dates[::3]
            values = [20_000.0 + 25.0 * step for step in range(len(source_dates))]
        else:
            length = 24 if spec.series_id == "PCEC96" else 24 - index
            source_dates = dates[:length]
            values = [100.0 + step for step in range(len(source_dates))]
        pd.DataFrame({"date": source_dates, "value": values}).to_csv(
            tmp_path / f"{spec.series_id}.csv", index=False
        )

    panel = FREDClient(api_key="", cache_dir=tmp_path).fetch_panel(
        observation_start="2010-01-01", minimum_observations=12
    )

    assert len(panel.values) == 24
    assert panel.values["GDPC1"].count() == 8
    quarter_ends = pd.date_range("2010-03-01", periods=8, freq="3MS")
    non_quarter_dates = panel.values.index.difference(quarter_ends)
    assert panel.values.loc[non_quarter_dates, "GDPC1"].isna().all()
    assert panel.values["FEDFUNDS"].isna().sum() == 4
    assert panel.last_observation == dates[-1]


def test_cached_series_respects_observation_start(tmp_path) -> None:
    spec = SERIES_SPECS[1]
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    pd.DataFrame({"date": dates, "value": range(6)}).to_csv(
        tmp_path / f"{spec.series_id}.csv", index=False
    )

    values, _ = FREDClient(api_key="", cache_dir=tmp_path).fetch_series(
        spec, observation_start="2020-04-01"
    )

    assert values.index.equals(dates[3:])


def _mock_fred_observations(monkeypatch, observations: list[dict[str, str]]) -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"observations": observations}

    monkeypatch.setattr(data_module.httpx, "get", lambda *args, **kwargs: Response())


def test_nonempty_partial_api_response_does_not_replace_complete_cache(
    tmp_path, monkeypatch
) -> None:
    spec = SERIES_SPECS[1]
    cache_path = tmp_path / f"{spec.series_id}.csv"
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    pd.DataFrame({"date": dates, "value": [100.0, 101.0, 102.0, 103.0]}).to_csv(
        cache_path, index=False
    )
    original = cache_path.read_bytes()
    _mock_fred_observations(
        monkeypatch,
        [
            {"date": "2020-01-01", "value": "200"},
            {"date": "2020-02-01", "value": "201"},
        ],
    )

    values, used_cache = FREDClient(api_key="test", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache
    assert values.tolist() == [100.0, 101.0, 102.0, 103.0]
    assert cache_path.read_bytes() == original


def test_endpoint_regression_does_not_replace_complete_cache(tmp_path, monkeypatch) -> None:
    spec = SERIES_SPECS[1]
    cache_path = tmp_path / f"{spec.series_id}.csv"
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    pd.DataFrame({"date": dates, "value": [100.0, 101.0, 102.0, 103.0]}).to_csv(
        cache_path, index=False
    )
    _mock_fred_observations(
        monkeypatch,
        [
            {"date": "2019-12-01", "value": "99"},
            {"date": "2020-01-01", "value": "100"},
            {"date": "2020-02-01", "value": "101"},
            {"date": "2020-03-01", "value": "102"},
        ],
    )

    values, used_cache = FREDClient(api_key="test", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache
    assert values.index.equals(dates)
    assert values.iloc[-1] == 103.0


def test_legitimate_revision_and_extension_replace_cache(tmp_path, monkeypatch) -> None:
    spec = SERIES_SPECS[1]
    cache_path = tmp_path / f"{spec.series_id}.csv"
    dates = pd.date_range("2020-01-01", periods=4, freq="MS")
    pd.DataFrame({"date": dates, "value": [100.0, 101.0, 102.0, 103.0]}).to_csv(
        cache_path, index=False
    )
    _mock_fred_observations(
        monkeypatch,
        [
            {"date": "2020-01-01", "value": "100"},
            {"date": "2020-02-01", "value": "111"},
            {"date": "2020-03-01", "value": "102"},
            {"date": "2020-04-01", "value": "103"},
            {"date": "2020-05-01", "value": "104"},
        ],
    )

    values, used_cache = FREDClient(api_key="test", cache_dir=tmp_path).fetch_series(spec)

    assert not used_cache
    assert values.index.equals(pd.date_range("2020-01-01", periods=5, freq="MS"))
    assert values.iloc[1] == 111.0
    cached = pd.read_csv(cache_path)
    assert cached["value"].tolist() == [100.0, 111.0, 102.0, 103.0, 104.0]


def test_invalid_api_payload_does_not_replace_valid_cache(tmp_path, monkeypatch) -> None:
    spec = SERIES_SPECS[1]
    cache_path = tmp_path / f"{spec.series_id}.csv"
    pd.DataFrame({"date": ["2020-01-01"], "value": [100.0]}).to_csv(cache_path, index=False)
    original = cache_path.read_bytes()

    class InvalidResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {"observations": [{"date": "2020-02-01", "value": "."}]}

    monkeypatch.setattr(data_module.httpx, "get", lambda *args, **kwargs: InvalidResponse())

    values, used_cache = FREDClient(api_key="test", cache_dir=tmp_path).fetch_series(spec)

    assert used_cache
    assert values.iloc[0] == 100.0
    assert cache_path.read_bytes() == original


def test_request_error_message_does_not_expose_api_key_or_url(tmp_path, monkeypatch) -> None:
    spec = SERIES_SPECS[1]
    sentinel = "fred-api-key-sentinel"
    request = data_module.httpx.Request(
        "GET",
        data_module.FRED_OBSERVATIONS_URL,
        params={"series_id": spec.series_id, "api_key": sentinel},
    )
    request_error = data_module.httpx.ConnectError(
        f"request failed: {request.url}", request=request
    )

    def raise_request_error(*args: object, **kwargs: object) -> None:
        raise request_error

    monkeypatch.setattr(data_module.httpx, "get", raise_request_error)

    with pytest.raises(data_module.FREDDataError) as raised:
        FREDClient(api_key=sentinel, cache_dir=tmp_path).fetch_series(spec)

    message = str(raised.value)
    assert sentinel not in message
    assert str(request.url) not in message
    assert spec.series_id in message
    assert "status=unavailable" in message
    assert "error_class=ConnectError" in message


def test_http_error_message_keeps_status_without_exposing_api_key_or_url(
    tmp_path, monkeypatch
) -> None:
    spec = SERIES_SPECS[1]
    sentinel = "fred-api-key-sentinel"
    request = data_module.httpx.Request(
        "GET",
        data_module.FRED_OBSERVATIONS_URL,
        params={"series_id": spec.series_id, "api_key": sentinel},
    )
    response = data_module.httpx.Response(503, request=request)
    http_error = data_module.httpx.HTTPStatusError(
        f"server error for {request.url}", request=request, response=response
    )

    def raise_http_error(*args: object, **kwargs: object) -> None:
        raise http_error

    monkeypatch.setattr(data_module.httpx, "get", raise_http_error)

    with pytest.raises(data_module.FREDDataError) as raised:
        FREDClient(api_key=sentinel, cache_dir=tmp_path).fetch_series(spec)

    message = str(raised.value)
    assert sentinel not in message
    assert str(request.url) not in message
    assert spec.series_id in message
    assert "status=503" in message
    assert "error_class=HTTPStatusError" in message
