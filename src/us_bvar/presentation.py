from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from great_tables import GT, html, md

from us_bvar.config import SeriesSpec
from us_bvar.model import ForecastResult
from us_bvar.transforms import (
    PlotTransformation,
    transform_forecast_samples,
    transform_path,
    transformation_spec,
)


def _percentile_text(probability: float, *, ordinal: bool) -> str:
    percentage = probability * 100
    rendered = f"{percentage:g}"
    if not ordinal or not percentage.is_integer():
        return rendered
    integer = int(percentage)
    if 10 <= integer % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(integer % 10, "th")
    return f"{rendered}{suffix}"


def interval_label(forecast: ForecastResult) -> str:
    """Describe the configured forecast interval for user-facing text."""

    lower, upper = forecast.interval
    return (
        f"{_percentile_text(lower, ordinal=True)}–"
        f"{_percentile_text(upper, ordinal=True)} percentile interval"
    )


def interval_range_label(forecast: ForecastResult) -> str:
    """Return the compact configured percentile range used inside forecast cells."""

    lower, upper = forecast.interval
    return f"{_percentile_text(lower, ordinal=False)}–{_percentile_text(upper, ordinal=False)}"


def forecast_summary_on_scale(
    history: pd.Series,
    forecast: ForecastResult,
    series_spec: SeriesSpec,
    transformation: PlotTransformation,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return forecast quantiles after transforming complete posterior paths."""

    series_index = forecast.median.columns.get_loc(series_spec.series_id)
    transformed = transform_forecast_samples(
        history,
        forecast.samples[:, :, series_index],
        series_spec,
        transformation,
    )
    lower_q, upper_q = forecast.interval

    def quantile(probability: float) -> pd.Series:
        return pd.Series(np.quantile(transformed, probability, axis=0), index=forecast.dates)

    return quantile(0.5), quantile(lower_q), quantile(upper_q)


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
    """Build the exact history and forecast rows shown in the dashboard."""

    recent = history.tail(history_months)
    dates = pd.DatetimeIndex(recent.index.append(baseline.dates))
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
    display_dates = pd.DatetimeIndex(recent.index.append(baseline.dates))
    result = pd.DataFrame(
        {
            "Month": display_dates.strftime("%b %Y"),
            "Period": (
                ['<span class="period-chip period-actual">History</span>'] * len(recent)
                + ['<span class="period-chip period-forecast">Forecast</span>']
                * len(baseline.dates)
            ),
        }
    )

    def value_cell(value: float, spec: SeriesSpec) -> str:
        return f'<span class="table-value">{_format_table_value(value, spec)}</span>'

    def forecast_cell(
        value: float,
        lower: float,
        upper: float,
        spec: SeriesSpec,
        forecast: ForecastResult,
    ) -> str:
        interval = f"{_format_table_value(lower, spec)} – {_format_table_value(upper, spec)}"
        return (
            f'<span class="table-value forecast-median">'
            f"{_format_table_value(value, spec)}</span>"
            f'<span class="forecast-interval" title="{interval_label(forecast)}">'
            f"{interval_range_label(forecast)}: {interval}</span>"
        )

    for spec in specs:
        historical_cells = [value_cell(value, spec) for value in recent[spec.series_id]]
        baseline_cells = [
            forecast_cell(value, lower, upper, spec, baseline)
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
                forecast_cell(value, lower, upper, spec, scenario)
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
                f"Natural units · forecast cells show the median and {interval_label(baseline)}"
            ),
        )
        .tab_source_note(
            source_note=md(
                "**Source:** FRED and model estimates. Monthly GDP history is a latent estimate; "
                "forecasts are model output and are not Federal Reserve forecasts."
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
        table = table.cols_label(cases=cast(dict[str, Any], labels)).tab_spanner(
            label=html(
                f'<span class="variable-name">{spec.short_label}</span>'
                f'<span class="variable-units">{spec.table_units}</span>'
            ),
            columns=columns,
            id=spec.series_id,
        )
    return table.cols_width({column: "178px" for column in value_columns})


def _timestamp_milliseconds(value: object) -> float:
    try:
        parsed = pd.DatetimeIndex([value])
        timestamp = cast(pd.Timestamp, parsed[0])
        if pd.isna(timestamp):
            raise ValueError("chart date is missing")
        return timestamp.timestamp() * 1000.0
    except (TypeError, ValueError, OSError, OverflowError) as exc:
        raise ValueError("Chart data contains an invalid date.") from exc


def _rounded_chart_value(value: object) -> float:
    try:
        return round(float(cast(Any, value)), 6)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Chart data contains an invalid value.") from exc


def echarts_options(
    history: pd.DataFrame,
    baseline: ForecastResult,
    spec: SeriesSpec,
    scenario: ForecastResult | None = None,
    transformation: PlotTransformation = "level",
) -> dict[str, Any]:
    """Build a local Apache ECharts configuration for one model variable."""

    history_series = cast(pd.Series, history.loc[:, spec.series_id]).astype(float)
    transformed_history = transform_path(history_series, spec, transformation)
    recent = transformed_history.tail(6)
    baseline_median, lower, upper = forecast_summary_on_scale(
        history_series, baseline, spec, transformation
    )
    units = plot_units(spec, transformation)

    def point(date: object, value: object) -> list[float]:
        return [_timestamp_milliseconds(date), _rounded_chart_value(value)]

    historical = [point(date, value) for date, value in recent.items()]
    last_point = point(recent.index[-1], recent.iloc[-1])
    baseline_line = [last_point] + [
        point(date, baseline_median.loc[date]) for date in baseline.dates
    ]
    interval = [
        [last_point[0], last_point[1], last_point[1]],
        *[
            [
                _timestamp_milliseconds(date),
                _rounded_chart_value(lower.loc[date]),
                _rounded_chart_value(upper.loc[date]),
            ]
            for date in baseline.dates
        ],
    ]
    series: list[dict[str, Any]] = [
        {
            "name": "History / estimate",
            "type": "line",
            "data": historical,
            "showSymbol": True,
            "symbolSize": 5,
            "lineStyle": {"color": "#e9eef5", "width": 2.5},
            "itemStyle": {"color": "#e9eef5"},
            "z": 4,
        },
        {
            "name": "Baseline",
            "type": "line",
            "data": baseline_line,
            "showSymbol": False,
            "lineStyle": {"color": "#36d6bd", "type": "dashed", "width": 2.5},
            "itemStyle": {"color": "#36d6bd"},
            "z": 3,
        },
    ]
    bands = [
        {
            "name": f"Baseline {interval_label(baseline)}",
            "data": interval,
            "color": "rgba(54, 214, 189, 0.16)",
        }
    ]
    if scenario is not None:
        scenario_median, scenario_lower, scenario_upper = forecast_summary_on_scale(
            history_series, scenario, spec, transformation
        )
        scenario_line = [last_point] + [
            point(date, scenario_median.loc[date]) for date in scenario.dates
        ]
        bands.append(
            {
                "name": f"Scenario {interval_label(scenario)}",
                "data": [
                    [last_point[0], last_point[1], last_point[1]],
                    *[
                        [
                            _timestamp_milliseconds(date),
                            _rounded_chart_value(scenario_lower.loc[date]),
                            _rounded_chart_value(scenario_upper.loc[date]),
                        ]
                        for date in scenario.dates
                    ],
                ],
                "color": "rgba(255, 143, 92, 0.12)",
            }
        )
        series.append(
            {
                "name": "Scenario",
                "type": "line",
                "data": scenario_line,
                "showSymbol": False,
                "lineStyle": {"color": "#ff8f5c", "width": 3},
                "itemStyle": {"color": "#ff8f5c"},
                "z": 5,
            }
        )

    return {
        "animation": False,
        "backgroundColor": "transparent",
        "aria": {"enabled": True, "description": f"{spec.label} history and forecast chart."},
        "grid": {"left": 62, "right": 22, "top": 54, "bottom": 42},
        "legend": {"top": 4, "left": 4, "textStyle": {"color": "#9eacbd", "fontSize": 11}},
        "bvarBands": bands,
        "bvarValueDecimals": spec.decimals if transformation == "level" else 2,
        "bvarUnits": units,
        "xAxis": {
            "type": "time",
            "axisLabel": {"color": "#9eacbd", "hideOverlap": True},
            "axisLine": {"lineStyle": {"color": "rgba(158, 172, 189, 0.34)"}},
            "axisTick": {"show": False},
            "splitLine": {"show": False},
        },
        "yAxis": {
            "type": "value",
            "name": units,
            "nameLocation": "middle",
            "nameGap": 46,
            "nameTextStyle": {"color": "#9eacbd", "fontSize": 10},
            "axisLabel": {"color": "#9eacbd"},
            "axisLine": {"show": False},
            "axisTick": {"show": False},
            "splitLine": {"lineStyle": {"color": "rgba(158, 172, 189, 0.13)"}},
            "scale": True,
        },
        "tooltip": {"trigger": "axis", "confine": True, "backgroundColor": "#0b1220"},
        "series": series,
    }
