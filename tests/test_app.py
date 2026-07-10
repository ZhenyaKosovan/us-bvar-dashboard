from __future__ import annotations

import runpy
from pathlib import Path

from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR
from us_bvar.transforms import ScenarioConstraint


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
    app_module = runpy.run_path(str(Path(__file__).parents[1] / "app.py"))
    scenario_modal = app_module["scenario_modal"]
    scenario_variable_cells = app_module["scenario_variable_cells"]

    modal_html = str(scenario_modal(model, baseline, None))

    assert 'aria-label="Conditional scenario path"' in modal_html
    assert 'aria-modal="true"' in modal_html
    assert 'role="dialog"' in modal_html
    for spec in SERIES_SPECS:
        assert f'id="sc_transform_{spec.series_id}"' in modal_html
        assert f'id="scenario_cells_{spec.series_id}"' in modal_html

    existing = model.forecast(
        horizon=12,
        draws=20,
        constraints={(0, "INDPRO"): ScenarioConstraint(1.25, "qoq")},
        seed=5,
    )
    existing_html = str(scenario_modal(model, baseline, existing))
    assert '<option value="qoq" selected="">QoQ</option>' in existing_html

    spec = SERIES_SPECS[0]
    cells_html = str(
        scenario_variable_cells(
            model,
            baseline,
            {(0, spec.series_id): ScenarioConstraint(1.25, "qoq")},
            spec,
            "qoq",
        )
    )
    assert cells_html.count('aria-label="') == 12
    assert "Values in Percent change" in cells_html
    assert 'value="1.25"' in cells_html
    expected_name = f"{SERIES_SPECS[0].label}, {baseline.dates[0]:%B %Y}, Percent change"
    assert f'aria-label="{expected_name}"' in cells_html


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
