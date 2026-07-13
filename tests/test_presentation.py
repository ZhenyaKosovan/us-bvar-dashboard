from __future__ import annotations

import numpy as np
import pytest

from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR
from us_bvar.presentation import echarts_options, forecast_display_frame, forecast_gt, plot_units


def test_display_frame_and_chart_use_six_plus_twelve_months(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    frame = forecast_display_frame(synthetic_levels, baseline, SERIES_SPECS)
    options = echarts_options(synthetic_levels, baseline, SERIES_SPECS[0])

    assert len(frame) == 18
    assert frame["Observation"].value_counts().to_dict() == {"Forecast": 12, "Historical": 6}
    assert len(options["series"][0]["data"]) == 6
    assert len(options["series"][1]["data"]) == 13
    assert options["bvarBands"][0]["data"][0] == [
        options["series"][0]["data"][-1][0],
        options["series"][0]["data"][-1][1],
        options["series"][0]["data"][-1][1],
    ]
    assert options["xAxis"]["type"] == "time"
    assert model.config.interval == (0.16, 0.84)


def test_forecast_table_centers_and_formats_values_with_intervals(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)

    table_html = forecast_gt(synthetic_levels, baseline, SERIES_SPECS).as_raw_html()

    assert table_html.count('class="forecast-interval"') == 12 * len(SERIES_SPECS)
    assert "16–84:" in table_html
    assert "16th–84th percentile interval" in table_html
    assert "Billions · chained 2017 $, SAAR" in table_html
    assert 'class="gt_row gt_center"' in table_html
    assert "$" in table_html
    assert "%</span>" in table_html


@pytest.mark.parametrize(
    ("transformation", "periods", "annualization"),
    [
        ("mom", 1, 1),
        ("qoq", 3, 1),
        ("yoy", 12, 1),
        ("mom_annualized", 1, 12),
        ("qoq_annualized", 3, 4),
        ("yoy_annualized", 12, 1),
    ],
)
def test_chart_transforms_growth_series(
    synthetic_levels, transformation: str, periods: int, annualization: int
) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    spec = SERIES_SPECS[0]
    options = echarts_options(synthetic_levels, baseline, spec, transformation=transformation)

    history_ratio = (
        synthetic_levels[spec.series_id].iloc[-1]
        / synthetic_levels[spec.series_id].iloc[-1 - periods]
    )
    expected_history = (history_ratio**annualization - 1) * 100

    assert np.isclose(options["series"][0]["data"][-1][1], expected_history)
    expected_draw_median = np.median(
        (baseline.samples[:, 0, 0] / synthetic_levels[spec.series_id].iloc[-periods])
        ** annualization
        * 100
        - 100
    )
    assert np.isclose(options["series"][1]["data"][1][1], expected_draw_median)
    assert options["bvarBands"][0]["data"][0][1:] == pytest.approx(
        [expected_history, expected_history]
    )
    assert options["yAxis"]["name"] == (
        "Annualized percent change" if transformation.endswith("_annualized") else "Percent change"
    )


def test_chart_uses_percentage_point_changes_for_rate_series(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    spec = SERIES_SPECS[3]
    options = echarts_options(synthetic_levels, baseline, spec, transformation="mom_annualized")

    expected_history = (
        synthetic_levels[spec.series_id].iloc[-1] - synthetic_levels[spec.series_id].iloc[-2]
    ) * 12
    assert np.isclose(options["series"][0]["data"][-1][1], expected_history)
    expected_draw_median = np.median(
        (baseline.samples[:, 0, 3] - synthetic_levels[spec.series_id].iloc[-1]) * 12
    )
    assert np.isclose(options["series"][1]["data"][1][1], expected_draw_median)
    assert plot_units(spec, "mom") == "Percentage-point change"
    assert plot_units(spec, "mom_annualized") == "Annualized percentage-point change"


def test_scenario_chart_uses_high_contrast_color(synthetic_levels) -> None:
    model = BVAR().fit(synthetic_levels)
    baseline = model.forecast(horizon=12, draws=20, seed=3)

    options = echarts_options(
        synthetic_levels,
        baseline,
        SERIES_SPECS[0],
        scenario=baseline,
    )

    assert options["series"][-1]["name"] == "Scenario"
    assert options["series"][-1]["lineStyle"]["color"] == "#ff8f5c"
    assert len(options["bvarBands"]) == 2
