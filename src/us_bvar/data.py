from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
import pandas as pd

from us_bvar.config import SERIES_SPECS, SeriesSpec

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FREDDataError(RuntimeError):
    """Raised when FRED data cannot be loaded or validated."""


@dataclass(frozen=True)
class PanelData:
    """Ragged monthly calendar containing monthly releases and quarterly GDP."""

    values: pd.DataFrame
    fetched_at: pd.Timestamp
    from_cache: bool
    cache_by_series: dict[str, bool]
    last_observations: dict[str, pd.Timestamp]

    @property
    def last_observation(self) -> pd.Timestamp:
        return cast(pd.Timestamp, pd.DatetimeIndex(self.values.index)[-1])

    @property
    def observed_counts(self) -> dict[str, int]:
        counts = cast(pd.Series, self.values.count())
        try:
            return {str(column): int(value) for column, value in counts.items()}
        except (TypeError, ValueError) as exc:
            raise FREDDataError("Could not count panel observations.") from exc


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
            cached = self._read_cache(cache_file, spec)
            if cached is not None:
                return self._apply_observation_start(cached, observation_start), True
            raise FREDDataError(
                "A FRED API key is required. Set FRED_API_KEY or run with a valid local cache."
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
            date_values = cast(pd.Series, frame.loc[:, "date"])
            raw_values = cast(pd.Series, frame.loc[:, "value"])
            normalized_dates = self._normalize_dates(date_values, spec)
            numeric_values = cast(pd.Series, pd.to_numeric(raw_values, errors="coerce"))
            frame = pd.DataFrame(
                {"date": normalized_dates.to_numpy(), "value": numeric_values.to_numpy()}
            )
            series = cast(pd.Series, frame.dropna().set_index("date").loc[:, "value"]).copy()
            series = series.loc[series.index.notna()].sort_index().groupby(level=0).last()
            series.name = spec.series_id
            series = self._apply_observation_start(series, observation_start)
            if not self._has_expected_cadence(series, spec):
                raise FREDDataError(
                    f"FRED returned incomplete observation cadence for {spec.series_id}."
                )
            cached = self._read_cache(cache_file, spec)
            cached_window = (
                None if cached is None else self._apply_observation_start(cached, observation_start)
            )
            if cached_window is not None and not self._can_replace_cache(
                series, cached_window, spec
            ):
                return cached_window, True
            self._write_cache(series, cache_file)
            return series, False
        except (httpx.HTTPError, ValueError, KeyError, FREDDataError) as exc:
            cached = self._read_cache(cache_file, spec)
            if cached is not None:
                return self._apply_observation_start(cached, observation_start), True
            if isinstance(exc, FREDDataError):
                raise
            response = getattr(exc, "response", None)
            status = str(getattr(response, "status_code", "unavailable"))
            raise FREDDataError(
                f"Could not fetch {spec.series_id} from FRED "
                f"(status={status}; error_class={type(exc).__name__})."
            ) from None

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

        raw_panel = pd.concat(series, axis=1, join="outer", sort=False).sort_index()
        monthly_specs = [spec for spec in specs if spec.frequency == "monthly"]
        starts = [raw_panel[spec.series_id].first_valid_index() for spec in monthly_specs]
        ends = [raw_panel[spec.series_id].last_valid_index() for spec in monthly_specs]
        if any(date is None for date in starts + ends):
            raise FREDDataError("At least one monthly FRED series contains no observations.")
        try:
            start = pd.DatetimeIndex(starts).max()
            end = pd.DatetimeIndex(ends).max()
        except (TypeError, ValueError) as exc:
            raise FREDDataError("FRED returned invalid monthly observation dates.") from exc
        calendar = pd.date_range(start, end, freq="MS")
        panel = cast(
            pd.DataFrame,
            raw_panel.reindex(calendar).loc[:, [spec.series_id for spec in specs]],
        )
        if len(panel) < minimum_observations:
            raise FREDDataError(
                f"Only {len(panel)} monthly calendar observations are available; "
                f"at least {minimum_observations} are required."
            )
        log_columns = [spec.series_id for spec in specs if spec.transform == "log"]
        if (panel[log_columns] <= 0).any().any():
            raise FREDDataError("Log-transformed FRED series contain a non-positive value.")
        last_observations: dict[str, pd.Timestamp] = {}
        for spec in specs:
            column = cast(pd.Series, panel.loc[:, spec.series_id])
            last = column.last_valid_index()
            if last is not None:
                last_observations[spec.series_id] = cast(pd.Timestamp, pd.DatetimeIndex([last])[0])
        return PanelData(
            values=panel,
            fetched_at=pd.Timestamp.now(tz="UTC"),
            from_cache=any(cache_flags),
            cache_by_series={
                spec.series_id: used_cache
                for spec, used_cache in zip(specs, cache_flags, strict=True)
            },
            last_observations=last_observations,
        )

    @staticmethod
    def _apply_observation_start(series: pd.Series, observation_start: str) -> pd.Series:
        try:
            parsed = pd.DatetimeIndex([observation_start])
            cutoff_timestamp = cast(pd.Timestamp, parsed[0])
            if pd.isna(cutoff_timestamp):
                raise ValueError("observation start is not a date")
            cutoff = cutoff_timestamp.to_period("M").to_timestamp()
        except (TypeError, ValueError) as exc:
            raise FREDDataError(f"Invalid observation start: {observation_start}") from exc
        result = series.loc[series.index >= cutoff]
        if result.empty:
            raise FREDDataError(
                "No valid FRED observations remain at or after the requested start."
            )
        return result

    def _write_cache(self, series: pd.Series, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                series.to_frame(name="value").to_csv(temporary, index_label="date")
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _normalize_dates(values: pd.Series, spec: SeriesSpec) -> pd.Series:
        parsed = pd.to_datetime(values, errors="coerce")
        if spec.frequency == "quarterly":
            return parsed.dt.to_period("Q").dt.asfreq("M", how="end").dt.to_timestamp()
        return parsed.dt.to_period("M").dt.to_timestamp()

    @staticmethod
    def _has_expected_cadence(series: pd.Series, spec: SeriesSpec) -> bool:
        if series.empty:
            return False
        index = pd.DatetimeIndex(series.index)
        if not index.is_monotonic_increasing or not index.is_unique:
            return False
        frequency = "MS" if spec.frequency == "monthly" else "3MS"
        expected = pd.date_range(index[0], index[-1], freq=frequency)
        return index.equals(expected)

    @classmethod
    def _can_replace_cache(cls, candidate: pd.Series, cached: pd.Series, spec: SeriesSpec) -> bool:
        if not cls._has_expected_cadence(candidate, spec):
            return False
        if not cls._has_expected_cadence(cached, spec):
            return False
        candidate_start = cast(pd.Timestamp, candidate.index[0])
        cached_start = cast(pd.Timestamp, cached.index[0])
        candidate_end = cast(pd.Timestamp, candidate.index[-1])
        cached_end = cast(pd.Timestamp, cached.index[-1])
        return (
            candidate_start <= cached_start
            and candidate_end >= cached_end
            and len(candidate) >= len(cached)
        )

    @classmethod
    def _read_cache(cls, path: Path, spec: SeriesSpec) -> pd.Series | None:
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeError):
            return None
        if not {"date", "value"}.issubset(frame.columns):
            return None
        raw_values = cast(pd.Series, frame.loc[:, "value"])
        raw_dates = cast(pd.Series, frame.loc[:, "date"])
        values = cast(pd.Series, pd.to_numeric(raw_values, errors="coerce"))
        index = cls._normalize_dates(raw_dates, spec)
        series = (
            pd.Series(values.to_numpy(), index=index, name=spec.series_id).dropna().sort_index()
        )
        series = series.loc[series.index.notna()]
        if series.empty:
            return None
        series = series.groupby(level=0).last()
        if not cls._has_expected_cadence(series, spec):
            return None
        return series
