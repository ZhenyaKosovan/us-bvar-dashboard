from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import httpx  # type: ignore[import-not-found]
import pytest

from us_bvar.artifact import ForecastArtifact
from us_bvar.config import SERIES_SPECS
from us_bvar.dashboard import (
    DashboardRuntime,
    _cached_scenario_forecast,
    chart_cards,
    scenario_modal,
)
from us_bvar.transforms import ScenarioConstraint


class RuntimeForTest(DashboardRuntime):
    test_events: list[tuple[str, dict[str, object]]]


@pytest.fixture
def dashboard_runtime(fitted_model) -> RuntimeForTest:
    events: list[tuple[str, dict[str, object]]] = []
    baseline = fitted_model.forecast(horizon=12, draws=20, seed=41)
    runtime = RuntimeForTest(
        ForecastArtifact.create(fitted_model, baseline),
        release_id="synthetic-test",
        static_root=Path(__file__).parents[1] / "www",
        max_scenario_concurrency=1,
        telemetry=lambda name, **fields: events.append((name, fields)),
    )
    runtime.test_events = events
    return runtime


def test_generated_ui_loads_local_browser_assets(dashboard_runtime: DashboardRuntime) -> None:
    rendered = str(dashboard_runtime.ui)

    assert '<script src="vendor/echarts/echarts.min.js"></script>' in rendered
    assert '<script src="app.js"></script>' in rendered
    assert 'href="app.css"' in rendered
    assert 'href="https://' not in rendered


def test_browser_bootstrap_is_served_as_a_local_static_module(
    dashboard_runtime: DashboardRuntime,
) -> None:
    async def fetch_assets() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=dashboard_runtime.app.starlette_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/app.js"), await client.get("/vendor/echarts/echarts.min.js")

    browser_script, echarts_bundle = asyncio.run(fetch_assets())

    assert browser_script.status_code == 200
    assert browser_script.headers["content-type"].startswith("text/javascript")
    assert "DOMContentLoaded" in browser_script.text
    assert "MutationObserver" in browser_script.text
    assert "us-bvar-workspace-v1" in browser_script.text
    assert "dragstart" in browser_script.text
    assert "localStorage" in browser_script.text
    assert echarts_bundle.status_code == 200
    assert len(echarts_bundle.content) > 100_000


def test_generated_ui_contains_accessible_workspace_builder(
    dashboard_runtime: DashboardRuntime,
) -> None:
    rendered = str(dashboard_runtime.ui)
    cards_html = "".join(str(card) for card in chart_cards(dashboard_runtime, SERIES_SPECS[:2]))

    assert 'id="variable-library-search"' in rendered
    assert 'id="variable-library-list"' in rendered
    assert rendered.count('class="variable-library-item"') == len(SERIES_SPECS)
    assert rendered.count('class="variable-add-button"') == len(SERIES_SPECS)
    assert 'data-workspace-preset="Overview"' in rendered
    assert 'id="workspace-announcer"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'class="workspace-selection-control"' in rendered
    assert cards_html.count('class="chart-card"') == 2
    assert cards_html.count('class="chart-action chart-drag-handle"') == 2
    assert cards_html.count('class="chart-menu"') == 2
    assert cards_html.count('class="chart-menu-option chart-remove-button"') == 2
    assert cards_html.count('role="menu"') == 2
    assert cards_html.count('aria-checked="true"') == 2


def test_scenario_modal_generated_output_is_accessible(dashboard_runtime: DashboardRuntime) -> None:
    modal_html = str(scenario_modal(dashboard_runtime, None))

    assert 'aria-label="Conditional scenario path"' in modal_html
    assert 'aria-modal="true"' in modal_html
    assert 'role="dialog"' in modal_html
    assert modal_html.count("HISTORY / EST.") == 3
    assert ">ACTUAL<" not in modal_html
    for spec in SERIES_SPECS:
        assert f'id="sc_transform_{spec.series_id}"' in modal_html
        assert f'id="sc_{spec.series_id}_0"' in modal_html
        assert f'data-scenario-row="{spec.series_id}"' in modal_html
    assert modal_html.count('class="form-control scenario-value"') == 12 * len(SERIES_SPECS)
    assert modal_html.count('class="scenario-grid-row"') == len(SERIES_SPECS)
    assert 'id="scenario-variable-search"' in modal_html
    assert 'id="scenario-group-filter"' in modal_html

    existing = dashboard_runtime.model.forecast(
        horizon=12,
        draws=20,
        constraints={(0, "GDPC1"): ScenarioConstraint(1.25, "qoq")},
        seed=5,
    )
    existing_html = str(scenario_modal(dashboard_runtime, existing))
    assert '<option value="qoq" selected="">QoQ</option>' in existing_html
    assert 'value="1.25"' in existing_html
    expected_name = (
        f"{SERIES_SPECS[0].label}, {dashboard_runtime.baseline.dates[0]:%B %Y}, Percent change"
    )
    assert f'aria-label="{expected_name}"' in existing_html


def test_runtime_scenario_cache_reuses_generated_forecast(
    dashboard_runtime: DashboardRuntime,
) -> None:
    key = ()
    before = _cached_scenario_forecast.cache_info()
    first = _cached_scenario_forecast(dashboard_runtime.model, key)
    second = _cached_scenario_forecast(dashboard_runtime.model, key)
    after = _cached_scenario_forecast.cache_info()

    assert first is second
    assert after.hits >= before.hits + 1
    assert len(first.constraints) == 0


def test_scenario_flow_and_progress_are_present_in_generated_ui(
    dashboard_runtime: DashboardRuntime,
) -> None:
    rendered = str(dashboard_runtime.ui)

    assert 'id="scenario_progress"' in rendered
    assert 'id="chart_grid"' in rendered
    assert "scenario-progress-overlay" in (Path(__file__).parents[1] / "www/app.css").read_text()


def test_health_and_security_middleware_are_http_behaviors(
    dashboard_runtime: DashboardRuntime,
) -> None:
    async def fetch_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=dashboard_runtime.app.starlette_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/healthz")

    response = asyncio.run(fetch_health())
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["release_id"] == "synthetic-test"
    assert payload["artifact_schema"] == dashboard_runtime.artifact.schema_version
    assert payload["variable_count"] == len(SERIES_SPECS)
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "same-origin"
    assert response.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert dashboard_runtime.app.sanitize_errors == True  # noqa: E712


def test_runtime_startup_telemetry_is_privacy_conscious(
    dashboard_runtime: DashboardRuntime,
) -> None:
    recorded_events = cast(RuntimeForTest, dashboard_runtime).test_events
    events = [name for name, _fields in recorded_events]

    assert events == ["application_started"]
    fields = recorded_events[0][1]
    assert fields["release_id"] == "synthetic-test"
    assert "scenario_values" not in json.dumps(fields)
