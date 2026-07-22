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
    MAX_SAVED_SCENARIOS,
    DashboardRuntime,
    NamedScenario,
    build_ui,
    chart_cards,
    default_scenario_name,
    duplicate_scenario_name,
    normalize_scenario_name,
    parse_scenario_value,
    save_named_scenario,
    scenario_modal,
    scenario_switcher,
    validate_scenario_name,
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
    assert browser_script.headers.get("content-type", "").startswith("text/javascript")
    assert "DOMContentLoaded" in browser_script.text
    assert "MutationObserver" in browser_script.text
    assert "us-bvar-workspace-v1" in browser_script.text
    assert "dragstart" in browser_script.text
    assert "localStorage" in browser_script.text
    assert "resetWorkspace" in browser_script.text
    assert "button.disabled = !isSelected && atLimit" in browser_script.text
    assert "Keep entered numbers and reinterpret" in browser_script.text
    assert "Discard the name and assumptions" in browser_script.text
    assert "requestScenarioEditorClose" in browser_script.text
    assert "validateScenarioEditor" in browser_script.text
    assert "updateAssumptionSummary" in browser_script.text
    assert "applyScenarioRowAction" in browser_script.text
    assert "applyScenarioStarter" in browser_script.text
    assert "initializeScenarioGuards" in browser_script.text
    assert "maxConstraints" in browser_script.text
    assert "run_scenario_request" in browser_script.text
    assert "reset_modal_request" in browser_script.text
    assert 'event.key !== "Escape"' in browser_script.text
    assert 'field.dispatchEvent(new Event("input", { bubbles: true }))' in browser_script.text
    assert "forecastData.open = true" in browser_script.text
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
    assert 'class="workspace-preset workspace-reset"' in rendered
    assert 'href="#analysis-workspace"' in rendered
    assert 'href="#forecast-data"' in rendered
    assert 'id="workspace-feedback"' in rendered
    assert 'aria-live="polite"' in rendered
    assert 'class="workspace-selection-control"' in rendered
    assert 'id="workspace_legend"' in rendered
    assert cards_html.count('class="chart-card"') == 2
    assert cards_html.count('class="chart-action chart-drag-handle"') == 2
    assert cards_html.count('class="chart-menu"') == 2
    assert cards_html.count('class="chart-menu-option chart-remove-button"') == 2
    assert cards_html.count('role="menu"') == 2
    assert cards_html.count('aria-checked="true"') == 2
    assert cards_html.count('class="chart-menu-option chart-size-option"') == 4
    assert cards_html.count('disabled=""') == 2
    assert 'data-default-transform="qoq"' in cards_html


def test_scenario_modal_generated_output_is_accessible(dashboard_runtime: DashboardRuntime) -> None:
    modal_html = str(scenario_modal(dashboard_runtime, None))

    assert 'aria-label="Conditional scenario path"' in modal_html
    assert 'data-max-constraints="60"' in modal_html
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
    assert 'data-bs-backdrop="static"' in modal_html
    assert 'data-bs-keyboard="false"' in modal_html
    assert 'class="btn-clear scenario-exit"' in modal_html
    assert ">Exit editor</button>" in modal_html
    assert 'id="scenario_name"' in modal_html
    assert 'maxlength="60"' in modal_html
    assert 'value="Scenario 1"' in modal_html
    assert "Used to identify this scenario" not in modal_html
    assert "Calculate scenario" in modal_html
    assert "Show assumptions only" in modal_html
    assert "No assumptions entered yet." in modal_html
    assert "Policy rate +1 pp" in modal_html
    assert "CPI inflation −1 pp" in modal_html
    assert 'id="scenario_validation"' in modal_html
    assert 'id="scenario-transform-dialog"' in modal_html
    assert modal_html.count('class="scenario-transform-shell"') == len(SERIES_SPECS)
    assert modal_html.count("form-select scenario-transform") == len(SERIES_SPECS)
    assert modal_html.count('class="scenario-row-tools"') == len(SERIES_SPECS)
    assert modal_html.count('data-row-action="hold"') == len(SERIES_SPECS)
    assert modal_html.count('data-row-action="interpolate"') == len(SERIES_SPECS)
    assert modal_html.count('data-row-action="clear"') == len(SERIES_SPECS)
    run_button = modal_html.split('id="run_scenario"', maxsplit=1)[1].split(">", maxsplit=1)[0]
    assert "disabled" not in run_button

    existing = dashboard_runtime.model.forecast(
        horizon=12,
        draws=20,
        constraints={(0, "GDPC1"): ScenarioConstraint(1.25, "qoq")},
        seed=5,
    )
    existing_html = str(scenario_modal(dashboard_runtime, existing, scenario_name="Soft landing"))
    assert 'value="Soft landing"' in existing_html
    assert "Edit scenario" in existing_html
    assert "Update scenario" in existing_html
    assert '<option value="qoq" selected="">QoQ</option>' in existing_html
    assert 'value="1.25"' in existing_html
    expected_name = (
        f"{SERIES_SPECS[0].label}, {dashboard_runtime.baseline.dates[0]:%B %Y}, Percent change"
    )
    assert f'aria-label="{expected_name}"' in existing_html


def test_named_scenarios_enforce_compact_unique_names(
    dashboard_runtime: DashboardRuntime,
) -> None:
    first = NamedScenario("first", "  Soft   landing  ", dashboard_runtime.baseline)
    scenarios = save_named_scenario({}, first)

    assert scenarios["first"].name == "Soft landing"
    assert normalize_scenario_name("Oil\nshock") == "Oil shock"
    assert validate_scenario_name(scenarios, "SOFT LANDING", scenario_id="first") == (
        "SOFT LANDING"
    )
    with pytest.raises(ValueError, match="already exists"):
        validate_scenario_name(scenarios, "soft landing", scenario_id="second")
    with pytest.raises(ValueError, match="Give this scenario a name"):
        normalize_scenario_name("  ")

    sequential = {
        "one": NamedScenario("one", "Scenario 1", dashboard_runtime.baseline),
        "two": NamedScenario("two", "Scenario 2", dashboard_runtime.baseline),
    }
    name, number = default_scenario_name(sequential, 1)
    assert name == "Scenario 3"
    assert number == 3
    name, number = default_scenario_name({}, 0)
    assert name == "Scenario 1"
    assert number == 1

    long_name = "A" * 60
    long_scenarios = {
        "original": NamedScenario("original", long_name, dashboard_runtime.baseline),
        "copy": NamedScenario("copy", f"{'A' * 55} copy", dashboard_runtime.baseline),
    }
    duplicate_name = duplicate_scenario_name(long_scenarios, long_name)
    assert duplicate_name == f"{'A' * 53} copy 2"
    assert len(duplicate_name) == 60

    at_capacity = {
        str(index): NamedScenario(str(index), f"Scenario {index + 1}", dashboard_runtime.baseline)
        for index in range(MAX_SAVED_SCENARIOS)
    }
    with pytest.raises(ValueError, match="at most 4 scenarios"):
        save_named_scenario(
            at_capacity,
            NamedScenario("fifth", "Scenario 5", dashboard_runtime.baseline),
        )
    replacement = save_named_scenario(
        at_capacity,
        NamedScenario("0", "Renamed scenario", dashboard_runtime.baseline),
    )
    assert len(replacement) == MAX_SAVED_SCENARIOS
    assert replacement["0"].name == "Renamed scenario"


@pytest.mark.parametrize(
    ("entered", "expected"),
    [
        ("1000", 1_000.0),
        ("1,000.25", 1_000.25),
        ("-.5", -0.5),
        ("1e3", 1_000.0),
    ],
)
def test_scenario_value_parser_accepts_unambiguous_numbers(entered: str, expected: float) -> None:
    assert parse_scenario_value(entered) == expected


@pytest.mark.parametrize("entered", ["1,2", "1,23", "12,34", "NaN", "inf", "1e999", "1_000"])
def test_scenario_value_parser_rejects_ambiguous_or_nonfinite_numbers(entered: str) -> None:
    with pytest.raises(ValueError, match="Scenario values"):
        parse_scenario_value(entered)


def test_scenario_switcher_lists_saved_names(dashboard_runtime: DashboardRuntime) -> None:
    forecast = dashboard_runtime.baseline
    scenarios = {
        "first": NamedScenario("first", "Soft landing", forecast),
        "second": NamedScenario("second", "Oil shock", forecast),
    }

    rendered = str(
        scenario_switcher(
            scenarios,
            "second",
            ("first",),
            show_intervals=True,
        )
    )

    assert 'id="active_scenario"' in rendered
    assert '<option value="first">Soft landing</option>' in rendered
    assert '<option value="second" selected="">Oil shock</option>' in rendered
    assert 'id="comparison_scenario"' not in rendered
    assert 'id="visible_scenarios"' in rendered
    assert 'value="first" checked="checked"' in rendered
    assert 'value="second" checked="checked"' not in rendered
    assert 'id="show_intervals"' in rendered
    assert 'id="show_intervals" type="checkbox" checked="checked"' in rendered
    assert 'class="scenario-export"' in rendered
    assert 'data-scenario-name="Oil shock"' in rendered
    assert "schema_version" in rendered


def test_runtime_scenario_cache_reuses_generated_forecast(
    dashboard_runtime: DashboardRuntime,
) -> None:
    key = ()
    before = dashboard_runtime.scenario_forecasts.cache_info()
    first = dashboard_runtime.scenario_forecasts.forecast(key)
    second = dashboard_runtime.scenario_forecasts.forecast(key)
    after = dashboard_runtime.scenario_forecasts.cache_info()

    assert first is second
    assert after.hits >= before.hits + 1
    assert len(first.constraints) == 0


def test_scenario_flow_and_progress_are_present_in_generated_ui(
    dashboard_runtime: DashboardRuntime,
) -> None:
    rendered = str(dashboard_runtime.ui)

    assert 'id="scenario_progress"' in rendered
    assert 'id="chart_grid"' in rendered
    assert 'id="open_scenario"' in rendered
    assert 'id="edit_scenario"' in rendered
    assert 'id="duplicate_scenario"' in rendered
    assert 'id="delete_scenario"' in rendered
    stylesheet = (Path(__file__).parents[1] / "www/app.css").read_text()
    assert "scenario-progress-overlay" in stylesheet
    assert "height: 4.5rem" in stylesheet
    assert "flex-wrap: nowrap" in stylesheet
    assert "height: calc(100dvh - 2rem)" in stylesheet
    assert ".scenario-assumption-summary" in stylesheet
    assert ".scenario-validation-message" in stylesheet


def test_refresh_required_release_is_visible_in_ui_and_health(
    dashboard_runtime: DashboardRuntime,
) -> None:
    dashboard_runtime.release_warning = "Rebuild with corrected monthly averages."
    rendered = str(build_ui(dashboard_runtime))

    async def fetch_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=dashboard_runtime.app.starlette_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/healthz")

    payload = asyncio.run(fetch_health()).json()
    dashboard_runtime.release_warning = None

    assert "MODEL REFRESH REQUIRED" in rendered
    assert "corrected monthly averages" in rendered
    assert payload["data_status"] == "refresh_required"
    assert payload["release_warning"] is not None


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
    assert response.headers.get("cache-control") == "no-store"
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("referrer-policy") == "same-origin"
    assert response.headers.get("permissions-policy") == "camera=(), microphone=(), geolocation=()"
    assert dashboard_runtime.app.sanitize_errors


def test_runtime_startup_telemetry_is_privacy_conscious(
    dashboard_runtime: DashboardRuntime,
) -> None:
    recorded_events = cast(RuntimeForTest, dashboard_runtime).test_events
    events = [name for name, _fields in recorded_events]

    assert events == ["application_started"]
    fields = recorded_events[0][1]
    assert fields["release_id"] == "synthetic-test"
    assert "scenario_values" not in json.dumps(fields)
