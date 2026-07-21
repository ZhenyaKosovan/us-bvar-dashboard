from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep
from types import SimpleNamespace
from typing import cast

from us_bvar.model import BVAR, ForecastResult
from us_bvar.scenario_service import (  # pyright: ignore[reportMissingImports]
    ScenarioForecastService,
)


class CountingModel:
    def __init__(self) -> None:
        self.calls = 0
        self._lock = Lock()

    def forecast(self, **kwargs) -> ForecastResult:
        del kwargs
        with self._lock:
            self.calls += 1
        sleep(0.05)
        return cast(ForecastResult, SimpleNamespace(constraints={}))


def test_concurrent_identical_requests_share_one_forecast() -> None:
    model = CountingModel()
    service = ScenarioForecastService(
        cast(BVAR, model),
        horizon=12,
        draws=20,
        seed=1,
        max_size=4,
        max_concurrency=2,
    )
    barrier = Barrier(4)

    def request() -> ForecastResult:
        barrier.wait()
        return service.forecast(())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _index: request(), range(4)))

    assert model.calls == 1
    assert all(result is results[0] for result in results)
    info = service.cache_info()
    assert info.misses == 1
    assert info.size == 1
    assert info.in_flight == 0
