from __future__ import annotations

import runpy
from pathlib import Path

from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR
from us_bvar.transforms import ScenarioConstraint


def test_chart_bootstrap_renders_a_chart_inserted_as_the_mutation_root() -> None:
    script = (Path(__file__).parents[1] / "app.py").read_text()

    assert "root.matches?.('.bvar-echart')" in script
    assert "nodes.push(root)" in script


def test_chart_bootstrap_destroys_charts_removed_by_shiny() -> None:
    script = (Path(__file__).parents[1] / "app.py").read_text()

    assert "node.bvarChart?.dispose()" in script
    assert "node.bvarResizeObserver?.disconnect()" in script
    assert "mutation.removedNodes.forEach" in script
    assert "destroyCharts(node)" in script


def test_scenario_modal_has_accessible_dialog_and_input_names(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    app_module = runpy.run_path(str(Path(__file__).parents[1] / "app.py"))
    scenario_modal = app_module["scenario_modal"]
    production_baseline = app_module["BASELINE"]

    modal_html = str(scenario_modal(None))

    assert 'aria-label="Conditional scenario path"' in modal_html
    assert 'aria-modal="true"' in modal_html
    assert 'role="dialog"' in modal_html
    for spec in SERIES_SPECS:
        assert f'id="sc_transform_{spec.series_id}"' in modal_html
        assert f'id="sc_{spec.series_id}_0"' in modal_html
        assert f'data-scenario-row="{spec.series_id}"' in modal_html
    assert modal_html.count('class="form-control scenario-value"') == 12 * len(SERIES_SPECS)
    assert modal_html.count('class="scenario-grid-row"') == len(SERIES_SPECS)
    assert "data-placeholders-by-scale=" in modal_html
    assert modal_html.index("scenario-variable-header") < modal_html.index(
        "scenario-transformation-header"
    )
    first_row = modal_html[modal_html.index('class="scenario-grid-row"') :]
    assert (
        first_row.index("scenario-variable-cell")
        < first_row.index("scenario-transformation-cell")
        < first_row.index("history-value")
    )
    long_name = SERIES_SPECS[1]
    assert f'title="{long_name.label}">{long_name.short_label}</div>' in modal_html
    stylesheet = (Path(__file__).parents[1] / "www" / "app.css").read_text()
    assert "-webkit-line-clamp: 2" in stylesheet

    existing = model.forecast(
        horizon=12,
        draws=20,
        constraints={(0, "INDPRO"): ScenarioConstraint(1.25, "qoq")},
        seed=5,
    )
    existing_html = str(scenario_modal(existing))
    assert '<option value="qoq" selected="">QoQ</option>' in existing_html
    assert 'value="1.25"' in existing_html
    expected_name = f"{SERIES_SPECS[0].label}, {production_baseline.dates[0]:%B %Y}, Percent change"
    assert f'aria-label="{expected_name}"' in existing_html
    app_source = (Path(__file__).parents[1] / "app.py").read_text()
    assert "editor.querySelectorAll('[id^=\"sc_transform_\"]')" in app_source


def test_echarts_is_served_from_local_static_assets() -> None:
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text()

    assert 'src="vendor/echarts/echarts.min.js"' in app_source
    assert (root / "www/vendor/echarts/echarts.min.js").is_file()
    assert (root / "www/vendor/echarts/LICENSE").is_file()


def test_scenario_forecast_runs_as_an_extended_background_task() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text()

    assert "@reactive.extended_task" in app_source
    assert "async with SCENARIO_SEMAPHORE" in app_source
    assert "await asyncio.to_thread(_cached_scenario_forecast, key)" in app_source
    assert "@lru_cache(maxsize=SCENARIO_CACHE_SIZE)" in app_source
    assert '"run_scenario"' in app_source
    assert "@ui.bind_task_button" not in app_source


def test_scenario_flow_closes_editor_and_shows_full_page_progress() -> None:
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text()
    stylesheet = (root / "www/app.css").read_text()

    modal_remove = app_source.index("ui.modal_remove()", app_source.index("def _run_scenario"))
    task_start = app_source.index("_scenario_task(key)", modal_remove)

    assert modal_remove < task_start
    assert 'ui.output_ui("scenario_progress")' in app_source
    assert '_scenario_task.status() != "running"' in app_source
    assert "scenario-progress-overlay" in stylesheet


def test_production_app_has_health_route_and_sanitized_errors() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text()

    assert 'Route("/healthz"' in app_source
    assert "app.sanitize_errors = True" in app_source
    assert "x-content-type-options" in app_source


def test_app_registers_privacy_conscious_usage_events() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text()

    for event_name in (
        "application_started",
        "session_started",
        "session_ended",
        "scenario_run_clicked",
        "scenario_requested",
        "scenario_completed",
        "scenario_failed",
    ):
        assert f'"{event_name}"' in app_source
    assert "scenario_values=" not in app_source
