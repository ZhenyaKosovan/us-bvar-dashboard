from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock

from us_bvar.model import BVAR, ForecastResult
from us_bvar.transforms import PlotTransformation, ScenarioConstraint

ConstraintKey = tuple[tuple[int, str, float, PlotTransformation], ...]


@dataclass(frozen=True, slots=True)
class ScenarioCacheInfo:
    hits: int
    misses: int
    size: int
    max_size: int
    in_flight: int


class ScenarioForecastService:
    """Thread-safe LRU cache that coalesces concurrent forecasts for the same key."""

    def __init__(
        self,
        model: BVAR,
        *,
        horizon: int,
        draws: int,
        seed: int,
        max_size: int,
        max_concurrency: int,
    ) -> None:
        if max_size < 1:
            raise ValueError("Scenario cache size must be positive.")
        if max_concurrency < 1:
            raise ValueError("Scenario concurrency must be positive.")
        self._model = model
        self._horizon = horizon
        self._draws = draws
        self._seed = seed
        self._max_size = max_size
        self._cache: OrderedDict[ConstraintKey, ForecastResult] = OrderedDict()
        self._in_flight: dict[ConstraintKey, Future[ForecastResult]] = {}
        self._lock = Lock()
        self._semaphore = BoundedSemaphore(max_concurrency)
        self._hits = 0
        self._misses = 0

    def get_cached(self, key: ConstraintKey) -> ForecastResult | None:
        with self._lock:
            result = self._cache.get(key)
            if result is None:
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return result

    def forecast(self, key: ConstraintKey) -> ForecastResult:
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._hits += 1
                return cached
            pending = self._in_flight.get(key)
            owner = pending is None
            if owner:
                pending = Future()
                self._in_flight[key] = pending
                self._misses += 1
        if pending is None:
            raise RuntimeError("Scenario single-flight state is inconsistent.")
        if not owner:
            return pending.result()

        constraints = {
            (step, series_id): ScenarioConstraint(value, transformation)
            for step, series_id, value, transformation in key
        }
        try:
            with self._semaphore:
                result = self._model.forecast(
                    horizon=self._horizon,
                    draws=self._draws,
                    constraints=constraints,
                    seed=self._seed,
                )
        except BaseException as exc:
            with self._lock:
                self._in_flight.pop(key, None)
                pending.set_exception(exc)
            raise
        with self._lock:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            self._in_flight.pop(key, None)
            pending.set_result(result)
        return result

    def cache_info(self) -> ScenarioCacheInfo:
        with self._lock:
            return ScenarioCacheInfo(
                hits=self._hits,
                misses=self._misses,
                size=len(self._cache),
                max_size=self._max_size,
                in_flight=len(self._in_flight),
            )
