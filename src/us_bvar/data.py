from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import pandas as pd

from us_bvar.config import SERIES_SPECS, SeriesSpec

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDDataError(RuntimeError):
    """Raised when FRED data cannot be loaded or validated."""


@dataclass(frozen=True)
class PanelData:
    values: pd.DataFrame
    fetched_at: pd.Timestamp
    from_cache: bool

    @property
    def last_observation(self) -> pd.Timestamp:
        return pd.Timestamp(self.values.index[-1])


class FREDClient:
    """Small FRED API client with a per-series CSV fallback cache."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path | str = Path("data/cache"),
        timeout: float = 20.0,
    ) -> None:
        self.api_key = (os.getenv("FRED_API_KEY", "") if api_key is None else api_key).strip()
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout

    def fetch_series(
        self,
        spec: SeriesSpec,
        observation_start: str = "1985-01-01",
    ) -> tuple[pd.Series, bool]:
        cache_file = self.cache_dir / f"{spec.series_id}.csv"
        if not self.api_key:
            cached = self._read_cache(cache_file, spec.series_id)
            if cached is not None:
                return cached, True
            raise FREDDataError(
                "A FRED API key is required. Set FRED_API_KEY or enter one in the dashboard."
            )

        params = {
            "series_id": spec.series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "sort_order": "asc",
        }
        try:
            response = httpx.get(FRED_OBSERVATIONS_URL, params=params, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
            if "error_message" in payload:
                raise FREDDataError(str(payload["error_message"]))
            observations = payload.get("observations", [])
            frame = pd.DataFrame(observations, columns=["date", "value"])
            if frame.empty:
                raise FREDDataError(f"FRED returned no observations for {spec.series_id}.")
            frame["date"] = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            series = frame.dropna().set_index("date")["value"].rename(spec.series_id)
            self._write_cache(series, cache_file)
            return series, False
        except (httpx.HTTPError, ValueError, KeyError, FREDDataError) as exc:
            cached = self._read_cache(cache_file, spec.series_id)
            if cached is not None:
                return cached, True
            if isinstance(exc, FREDDataError):
                raise
            raise FREDDataError(f"Could not fetch {spec.series_id} from FRED: {exc}") from exc

    def fetch_panel(
        self,
        specs: tuple[SeriesSpec, ...] = SERIES_SPECS,
        observation_start: str = "1985-01-01",
        minimum_observations: int = 120,
    ) -> PanelData:
        series: list[pd.Series] = []
        cache_flags: list[bool] = []
        for spec in specs:
            values, used_cache = self.fetch_series(spec, observation_start)
            series.append(values)
            cache_flags.append(used_cache)

        # A balanced panel deliberately ends at the latest month available for every
        # series. This avoids silently filling the ragged release edge.
        panel = pd.concat(series, axis=1, join="inner").sort_index().dropna()
        if len(panel) < minimum_observations:
            raise FREDDataError(
                f"Only {len(panel)} complete monthly observations are available; "
                f"at least {minimum_observations} are required."
            )
        if (panel <= 0)[[s.series_id for s in specs if s.transform == "log"]].any().any():
            raise FREDDataError("Log-transformed FRED series contain a non-positive value.")
        return PanelData(
            values=panel,
            fetched_at=pd.Timestamp.now(tz="UTC"),
            from_cache=any(cache_flags),
        )

    def _write_cache(self, series: pd.Series, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        series.rename("value").to_csv(path, index_label="date")

    @staticmethod
    def _read_cache(path: Path, series_id: str) -> pd.Series | None:
        if not path.exists():
            return None
        frame = pd.read_csv(path, parse_dates=["date"])
        if "value" not in frame:
            return None
        values = pd.to_numeric(frame["value"], errors="coerce")
        index = pd.to_datetime(frame["date"]).dt.to_period("M").dt.to_timestamp()
        return pd.Series(values.to_numpy(), index=index, name=series_id).dropna().sort_index()
