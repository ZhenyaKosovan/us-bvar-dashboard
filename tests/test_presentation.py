from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from us_bvar.config import SERIES_SPECS
from us_bvar.presentation import (
    echarts_options,
    forecast_gt,
    interval_label,
    interval_range_label,
    plot_units,
)
from us_bvar.transforms import PlotTransformation, transform_forecast_samples


def test_chart_uses_six_history_and_twelve_forecast_months(synthetic_levels, fitted_model) -> None:
    model = fitted_model
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    assert model.history_levels is not None
    options = echarts_options(model.history_levels, baseline, SERIES_SPECS[0])

    assert len(options["series"][0]["data"]) == 6
    assert len(options["series"][1]["data"]) == 13
    assert options["bvarBands"][0]["data"][0] == [
        options["series"][0]["data"][-1][0],
        options["series"][0]["data"][-1][1],
        options["series"][0]["data"][-1][1],
    ]
    assert options["xAxis"]["type"] == "time"
    assert options["legend"] == {"show": False}
    assert options["grid"]["top"] == 20
    if model.config.interval != (0.16, 0.84):
        raise AssertionError(f"Unexpected interval: {model.config.interval}")


def test_forecast_table_centers_and_formats_values_with_intervals(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    assert model.history_levels is not None

    table_html = forecast_gt(model.history_levels, baseline, SERIES_SPECS).as_raw_html()

    assert table_html.count('class="forecast-interval"') == 12 * len(SERIES_SPECS)
    assert "16–84:" in table_html
    assert "16th–84th percentile interval" in table_html
    assert "Billions · chained 2017 $, SAAR" in table_html
    assert 'class="gt_row gt_center"' in table_html
    assert "154px" in table_html
    assert "$" in table_html
    assert "%</span>" in table_html


def test_presentation_labels_follow_a_non_default_forecast_interval(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    baseline = replace(model.forecast(horizon=12, draws=20, seed=3), interval=(0.10, 0.90))
    assert model.history_levels is not None

    assert interval_label(baseline) == "10th–90th percentile interval"
    assert interval_range_label(baseline) == "10–90"
    table_html = forecast_gt(model.history_levels, baseline, SERIES_SPECS).as_raw_html()
    options = echarts_options(model.history_levels, baseline, SERIES_SPECS[0])

    assert "10–90:" in table_html
    assert 'title="10th–90th percentile interval"' in table_html
    assert options["bvarBands"][0]["name"] == "Baseline 10th–90th percentile interval"


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
    synthetic_levels,
    fitted_model,
    transformation: PlotTransformation,
    periods: int,
    annualization: int,
) -> None:
    model = fitted_model
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    assert model.history_levels is not None
    spec = SERIES_SPECS[0]
    history = model.history_levels
    options = echarts_options(history, baseline, spec, transformation=transformation)

    history_ratio = history[spec.series_id].iloc[-1] / history[spec.series_id].iloc[-1 - periods]
    expected_history = (history_ratio**annualization - 1) * 100

    assert np.isclose(options["series"][0]["data"][-1][1], expected_history, atol=1e-6)
    expected_draw_median = np.median(
        (baseline.samples[:, 0, 0] / history[spec.series_id].iloc[-periods]) ** annualization * 100
        - 100
    )
    assert np.isclose(options["series"][1]["data"][1][1], expected_draw_median, atol=1e-6)
    assert options["bvarBands"][0]["data"][0][1:] == pytest.approx(
        [expected_history, expected_history], abs=1e-6
    )
    assert options["yAxis"]["name"] == (
        "Annualized percent change" if transformation.endswith("_annualized") else "Percent change"
    )


def test_transform_forecast_samples_uses_fixed_history_for_early_anchor() -> None:
    history = pd.Series([100.0, 110.0, 121.0])
    samples = np.asarray([[133.1, 146.41], [121.0, 133.1]])

    transformed = transform_forecast_samples(history, samples, SERIES_SPECS[0], "qoq")

    assert transformed[:, 0] == pytest.approx([33.1, 21.0])
    assert transformed[:, 1] == pytest.approx([33.1, 21.0])


def test_chart_uses_percentage_point_changes_for_rate_series(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    assert model.history_levels is not None
    spec = SERIES_SPECS[3]
    history = model.history_levels
    options = echarts_options(history, baseline, spec, transformation="mom_annualized")

    expected_history = (history[spec.series_id].iloc[-1] - history[spec.series_id].iloc[-2]) * 12
    assert np.isclose(options["series"][0]["data"][-1][1], expected_history)
    expected_draw_median = np.median(
        (baseline.samples[:, 0, 3] - synthetic_levels[spec.series_id].iloc[-1]) * 12
    )
    assert np.isclose(options["series"][1]["data"][1][1], expected_draw_median)
    assert plot_units(spec, "mom") == "Percentage-point change"
    assert plot_units(spec, "mom_annualized") == "Annualized percentage-point change"


def test_four_scenarios_use_distinct_colors_and_optional_intervals(
    synthetic_levels, fitted_model
) -> None:
    model = fitted_model
    baseline = model.forecast(horizon=12, draws=20, seed=3)
    scenarios = tuple(
        (name, baseline, index)
        for index, name in enumerate(("Soft landing", "Oil shock", "Fiscal boost", "Credit event"))
    )

    options = echarts_options(
        synthetic_levels,
        baseline,
        SERIES_SPECS[0],
        scenarios,
        show_intervals=False,
    )
    assert model.history_levels is not None
    table_html = forecast_gt(
        model.history_levels,
        baseline,
        SERIES_SPECS[:1],
        scenarios,
        show_intervals=False,
    ).as_raw_html()

    assert [series["name"] for series in options["series"][-4:]] == [
        name for name, _forecast, _color in scenarios
    ]
    assert [series["lineStyle"]["color"] for series in options["series"][-4:]] == [
        "#ff8f5c",
        "#a78bfa",
        "#fbbf24",
        "#60a5fa",
    ]
    third_only = echarts_options(
        synthetic_levels,
        baseline,
        SERIES_SPECS[0],
        (scenarios[2],),
        show_intervals=False,
    )
    assert third_only["series"][-1]["lineStyle"]["color"] == "#fbbf24"
    assert options["bvarBands"] == []
    assert 'class="forecast-interval"' not in table_html
    for name, _forecast, _color in scenarios:
        assert name in table_html

    options_with_intervals = echarts_options(
        synthetic_levels,
        baseline,
        SERIES_SPECS[0],
        scenarios,
        show_intervals=True,
    )
    table_with_intervals = forecast_gt(
        model.history_levels,
        baseline,
        SERIES_SPECS[:1],
        scenarios,
        show_intervals=True,
    ).as_raw_html()
    assert len(options_with_intervals["bvarBands"]) == 5
    assert table_with_intervals.count('class="forecast-interval"') == 5 * 12
