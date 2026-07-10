from __future__ import annotations

import runpy
from pathlib import Path

from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR


def test_chart_bootstrap_renders_a_chart_inserted_as_the_mutation_root() -> None:
    script = (Path(__file__).parents[1] / "app.py").read_text()

    assert "root.matches?.('.bvar-highchart')" in script
    assert "nodes.push(root)" in script


def test_chart_bootstrap_destroys_charts_removed_by_shiny() -> None:
    script = (Path(__file__).parents[1] / "app.py").read_text()

    assert "node.bvarChart?.destroy()" in script
    assert "mutation.removedNodes.forEach" in script
    assert "destroyCharts(node)" in script


def test_scenario_modal_has_accessible_dialog_and_input_names(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    scenario_modal = runpy.run_path(str(Path(__file__).parents[1] / "app.py"))["scenario_modal"]

    modal_html = str(scenario_modal(model, baseline, None))

    assert 'aria-label="Conditional scenario path"' in modal_html
    assert 'aria-modal="true"' in modal_html
    assert 'role="dialog"' in modal_html
    assert modal_html.count('aria-label="') == 1 + 12 * len(SERIES_SPECS)
    for spec in SERIES_SPECS:
        expected_name = f"{spec.label}, {baseline.dates[0]:%B %Y}, {spec.units}"
        assert f'aria-label="{expected_name}"' in modal_html


def test_highcharts_is_served_from_local_static_assets() -> None:
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text()

    assert 'src="vendor/highcharts/highcharts.js"' in app_source
    assert 'src="vendor/highcharts/highcharts-more.js"' in app_source
    assert (root / "www/vendor/highcharts/highcharts.js").is_file()
    assert (root / "www/vendor/highcharts/highcharts-more.js").is_file()


def test_scenario_forecast_runs_as_an_extended_background_task() -> None:
    app_source = (Path(__file__).parents[1] / "app.py").read_text()

    assert "@reactive.extended_task" in app_source
    assert "await asyncio.to_thread(" in app_source
    assert 'ui.input_action_button(\n                "run_scenario"' in app_source
    assert "@ui.bind_task_button" not in app_source


def test_scenario_flow_closes_editor_and_shows_full_page_progress() -> None:
    root = Path(__file__).parents[1]
    app_source = (root / "app.py").read_text()
    stylesheet = (root / "www/app.css").read_text()

    modal_remove = app_source.index("ui.modal_remove()", app_source.index("def _run_scenario"))
    task_start = app_source.index("_scenario_task(model, constraints)")

    assert modal_remove < task_start
    assert 'ui.output_ui("scenario_progress")' in app_source
    assert '_scenario_task.status() != "running"' in app_source
    assert "scenario-progress-overlay" in stylesheet
