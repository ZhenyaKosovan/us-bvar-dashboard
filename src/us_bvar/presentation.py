from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from great_tables import GT, html, md

from us_bvar.config import SeriesSpec
from us_bvar.model import ForecastResult
from us_bvar.transforms import (
    PlotTransformation,
    transform_path,
    transformation_spec,
)


def _transformed_interval(
    history: pd.Series,
    baseline: ForecastResult,
    series_spec: SeriesSpec,
    transformation: PlotTransformation,
) -> tuple[pd.Series, pd.Series]:
    series_id = series_spec.series_id
    lower_path = pd.concat([history, baseline.lower[series_id]]).astype(float)
    upper_path = pd.concat([history, baseline.upper[series_id]]).astype(float)
    if transformation == "level":
        return lower_path.loc[baseline.dates], upper_path.loc[baseline.dates]

    transform_spec = transformation_spec(transformation)
    lower_lag = lower_path.shift(transform_spec.periods)
    upper_lag = upper_path.shift(transform_spec.periods)
    if series_spec.transform == "log":
        lower_ratio = lower_path / upper_lag
        upper_ratio = upper_path / lower_lag
        if transform_spec.annualized:
            lower_values = (lower_ratio.pow(transform_spec.annualization_factor) - 1.0) * 100.0
            upper_values = (upper_ratio.pow(transform_spec.annualization_factor) - 1.0) * 100.0
        else:
            lower_values = (lower_ratio - 1.0) * 100.0
            upper_values = (upper_ratio - 1.0) * 100.0
    else:
        lower_values = lower_path - upper_lag
        upper_values = upper_path - lower_lag
        if transform_spec.annualized:
            lower_values *= transform_spec.annualization_factor
            upper_values *= transform_spec.annualization_factor
    return lower_values.loc[baseline.dates], upper_values.loc[baseline.dates]


def plot_units(series_spec: SeriesSpec, transformation: PlotTransformation) -> str:
    """Return the chart-axis unit appropriate for a display transformation."""

    if transformation == "level":
        return series_spec.units
    transform_spec = transformation_spec(transformation)
    if series_spec.transform == "log":
        return "Annualized percent change" if transform_spec.annualized else "Percent change"
    return (
        "Annualized percentage-point change"
        if transform_spec.annualized
        else "Percentage-point change"
    )


def forecast_display_frame(
    history: pd.DataFrame,
    baseline: ForecastResult,
    specs: tuple[SeriesSpec, ...],
    scenario: ForecastResult | None = None,
    history_months: int = 6,
) -> pd.DataFrame:
    """Build the exact 6-history + 12-forecast rows shown in the dashboard."""

    recent = history.tail(history_months)
    dates = recent.index.append(baseline.dates)
    result = pd.DataFrame(
        {
            "Month": dates.strftime("%b %Y"),
            "Observation": ["Historical"] * len(recent) + ["Forecast"] * len(baseline.dates),
        }
    )
    for spec in specs:
        baseline_values = np.concatenate(
            [recent[spec.series_id].to_numpy(), baseline.median[spec.series_id].to_numpy()]
        )
        result[f"{spec.short_label} · Baseline"] = baseline_values
        if scenario is not None:
            scenario_values = np.concatenate(
                [recent[spec.series_id].to_numpy(), scenario.median[spec.series_id].to_numpy()]
            )
            result[f"{spec.short_label} · Scenario"] = scenario_values
    return result


def _format_table_value(value: float, spec: SeriesSpec) -> str:
    number = f"{value:,.{spec.decimals}f}"
    return f"{spec.value_prefix}{number}{spec.value_suffix}"


def _forecast_table_frame(
    history: pd.DataFrame,
    baseline: ForecastResult,
    specs: tuple[SeriesSpec, ...],
    scenario: ForecastResult | None = None,
    history_months: int = 6,
) -> pd.DataFrame:
    """Build display-ready cells while keeping the public data frame numeric."""

    recent = history.tail(history_months)
    result = pd.DataFrame(
        {
            "Month": recent.index.append(baseline.dates).strftime("%b %Y"),
            "Period": (
                ['<span class="period-chip period-actual">Actual</span>'] * len(recent)
                + ['<span class="period-chip period-forecast">Forecast</span>']
                * len(baseline.dates)
            ),
        }
    )

    def value_cell(value: float, spec: SeriesSpec) -> str:
        return f'<span class="table-value">{_format_table_value(value, spec)}</span>'

    def forecast_cell(value: float, lower: float, upper: float, spec: SeriesSpec) -> str:
        interval = f"{_format_table_value(lower, spec)} – {_format_table_value(upper, spec)}"
        return (
            f'<span class="table-value forecast-median">'
            f"{_format_table_value(value, spec)}</span>"
            f'<span class="forecast-interval" title="16th–84th percentile interval">'
            f"16–84: {interval}</span>"
        )

    for spec in specs:
        historical_cells = [value_cell(value, spec) for value in recent[spec.series_id]]
        baseline_cells = [
            forecast_cell(value, lower, upper, spec)
            for value, lower, upper in zip(
                baseline.median[spec.series_id],
                baseline.lower[spec.series_id],
                baseline.upper[spec.series_id],
                strict=True,
            )
        ]
        result[f"{spec.series_id}_baseline"] = historical_cells + baseline_cells

        if scenario is not None:
            scenario_cells = [
                forecast_cell(value, lower, upper, spec)
                for value, lower, upper in zip(
                    scenario.median[spec.series_id],
                    scenario.lower[spec.series_id],
                    scenario.upper[spec.series_id],
                    strict=True,
                )
            ]
            result[f"{spec.series_id}_scenario"] = historical_cells + scenario_cells

    return result


def forecast_gt(
    history: pd.DataFrame,
    baseline: ForecastResult,
    specs: tuple[SeriesSpec, ...],
    scenario: ForecastResult | None = None,
) -> GT:
    frame = _forecast_table_frame(history, baseline, specs, scenario)
    value_columns = [column for column in frame if column not in {"Month", "Period"}]
    table = (
        GT(frame)
        .tab_header(
            title="Monthly history and forecast",
            subtitle=(
                "Natural units · forecast cells show the median and 16th–84th percentile interval"
            ),
        )
        .tab_source_note(
            source_note=md(
                "**Source:** FRED. Forecasts are model output and are not "
                "Federal Reserve forecasts."
            )
        )
        .cols_align(align="center")
        .cols_width({"Month": "100px", "Period": "88px"})
        .cols_label(Month="Month", Period="Period")
        .tab_options(
            table_layout="auto",
            data_row_padding="10px",
            data_row_padding_horizontal="12px",
            column_labels_padding="10px",
        )
        .opt_row_striping()
    )
    for spec in specs:
        columns = [f"{spec.series_id}_baseline"]
        scenario_column = f"{spec.series_id}_scenario"
        if scenario_column in frame:
            columns.append(scenario_column)
        labels = {columns[0]: "Baseline"}
        if scenario_column in frame:
            labels[scenario_column] = "Scenario"
        table = table.cols_label(labels).tab_spanner(
            label=html(
                f'<span class="variable-name">{spec.short_label}</span>'
                f'<span class="variable-units">{spec.table_units}</span>'
            ),
            columns=columns,
            id=spec.series_id,
        )
    table = table.cols_width({column: "178px" for column in value_columns})
    return table


def highcharts_options(
    history: pd.DataFrame,
    baseline: ForecastResult,
    spec: SeriesSpec,
    scenario: ForecastResult | None = None,
    transformation: PlotTransformation = "level",
) -> dict[str, Any]:
    series_id = spec.series_id
    history_series = history[series_id].astype(float)
    baseline_path = pd.concat([history_series, baseline.median[series_id]]).astype(float)
    transformed_history = transform_path(history_series, spec, transformation)
    transformed_baseline = transform_path(baseline_path, spec, transformation)
    recent = transformed_history.tail(6)
    lower, upper = _transformed_interval(history_series, baseline, spec, transformation)
    units = plot_units(spec, transformation)

    def point(date: pd.Timestamp, value: float) -> list[float]:
        return [float(pd.Timestamp(date).timestamp() * 1000), round(float(value), 6)]

    historical = [point(date, value) for date, value in recent.items()]
    last_point = point(recent.index[-1], recent.iloc[-1])
    baseline_line = [last_point] + [
        point(date, transformed_baseline.loc[date]) for date in baseline.dates
    ]
    interval = [
        [
            float(pd.Timestamp(recent.index[-1]).timestamp() * 1000),
            round(float(recent.iloc[-1]), 6),
            round(float(recent.iloc[-1]), 6),
        ]
    ] + [
        [
            float(pd.Timestamp(date).timestamp() * 1000),
            round(float(lower.loc[date]), 6),
            round(float(upper.loc[date]), 6),
        ]
        for date in baseline.dates
    ]
    series: list[dict[str, Any]] = [
        {
            "name": "Historical",
            "type": "line",
            "data": historical,
            "color": "#f6f6f6",
            "lineWidth": 2.5,
            "marker": {"enabled": True, "radius": 2.5},
            "zIndex": 4,
        },
        {
            "name": "Baseline 16th–84th percentile interval",
            "type": "arearange",
            "data": interval,
            "color": "#4fcdb0",
            "fillOpacity": 0.16,
            "lineWidth": 0,
            "marker": {"enabled": False},
            "zIndex": 1,
        },
        {
            "name": "Baseline",
            "type": "line",
            "data": baseline_line,
            "color": "#4fcdb0",
            "dashStyle": "ShortDash",
            "lineWidth": 2.5,
            "marker": {"enabled": False},
            "zIndex": 3,
        },
    ]
    if scenario is not None:
        scenario_path = pd.concat([history_series, scenario.median[series_id]]).astype(float)
        transformed_scenario = transform_path(scenario_path, spec, transformation)
        scenario_line = [last_point] + [
            point(date, transformed_scenario.loc[date]) for date in scenario.dates
        ]
        series.append(
            {
                "name": "Scenario",
                "type": "line",
                "data": scenario_line,
                "color": "#ff7a45",
                "lineWidth": 3,
                "marker": {"enabled": False},
                "zIndex": 5,
            }
        )

    return {
        "chart": {
            "backgroundColor": "transparent",
            "height": 310,
            "spacing": [8, 8, 8, 8],
            "style": {"fontFamily": "Inter, system-ui, sans-serif"},
        },
        "title": {"text": None},
        "credits": {"enabled": False},
        "legend": {
            "align": "left",
            "verticalAlign": "top",
            "itemStyle": {"color": "#bfb29e", "fontSize": "11px", "fontWeight": "600"},
        },
        "xAxis": {
            "type": "datetime",
            "labels": {"style": {"color": "#bfb29e"}},
            "lineColor": "rgba(191, 178, 158, 0.5)",
            "tickColor": "rgba(191, 178, 158, 0.5)",
        },
        "yAxis": {
            "labels": {"style": {"color": "#bfb29e"}},
            "title": {
                "text": units,
                "style": {"color": "#bfb29e", "fontSize": "10px"},
            },
            "gridLineColor": "rgba(191, 178, 158, 0.18)",
        },
        "tooltip": {
            "backgroundColor": "#141413",
            "borderColor": "rgba(191, 178, 158, 0.5)",
            "shared": True,
            "style": {"color": "#f6f6f6"},
            "xDateFormat": "%B %Y",
            "valueDecimals": spec.decimals if transformation == "level" else 2,
        },
        "plotOptions": {"series": {"animation": False}},
        "series": series,
    }
