from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd
from great_tables import GT, html, md  # type: ignore[import-not-found]

from us_bvar.config import SeriesSpec
from us_bvar.model import ForecastResult
from us_bvar.transforms import (
    PlotTransformation,
    transform_forecast_samples,
    transform_path,
    transformation_spec,
)

ScenarioForecast = tuple[str, ForecastResult, int]
SCENARIO_STYLES: tuple[tuple[str, str], ...] = (
    ("#ff8f5c", "rgba(255, 143, 92, 0.12)"),
    ("#a78bfa", "rgba(167, 139, 250, 0.10)"),
    ("#fbbf24", "rgba(251, 191, 36, 0.10)"),
    ("#60a5fa", "rgba(96, 165, 250, 0.10)"),
)


def _percentile_text(probability: float, *, ordinal: bool) -> str:
    percentage = probability * 100
    rendered = f"{percentage:g}"
    if not ordinal or not percentage.is_integer():
        return rendered
    integer = round(percentage)
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


def _format_table_value(value: float, spec: SeriesSpec) -> str:
    number = f"{value:,.{spec.decimals}f}"
    return f"{spec.value_prefix}{number}{spec.value_suffix}"


def _forecast_table_frame(
    history: pd.DataFrame,
    baseline: ForecastResult,
    specs: tuple[SeriesSpec, ...],
    scenarios: Sequence[ScenarioForecast] = (),
    *,
    show_intervals: bool = True,
    history_months: int = 6,
) -> pd.DataFrame:
    """Build display-ready cells for the baseline and every visible scenario."""

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
        median = (
            f'<span class="table-value forecast-median">{_format_table_value(value, spec)}</span>'
        )
        if not show_intervals:
            return median
        interval = f"{_format_table_value(lower, spec)} – {_format_table_value(upper, spec)}"
        return (
            median + f'<span class="forecast-interval" title="{interval_label(forecast)}">'
            f"{interval_range_label(forecast)}: {interval}</span>"
        )

    forecasts: tuple[tuple[str, ForecastResult], ...] = (
        ("baseline", baseline),
        *(
            (f"scenario_{index}", forecast)
            for index, (_name, forecast, _color) in enumerate(scenarios)
        ),
    )
    for spec in specs:
        historical_cells = [value_cell(value, spec) for value in recent[spec.series_id]]
        for suffix, forecast in forecasts:
            forecast_cells = [
                forecast_cell(value, lower, upper, spec, forecast)
                for value, lower, upper in zip(
                    forecast.median[spec.series_id],
                    forecast.lower[spec.series_id],
                    forecast.upper[spec.series_id],
                    strict=True,
                )
            ]
            result[f"{spec.series_id}_{suffix}"] = historical_cells + forecast_cells

    return result


def forecast_gt(
    history: pd.DataFrame,
    baseline: ForecastResult,
    specs: tuple[SeriesSpec, ...],
    scenarios: Sequence[ScenarioForecast] = (),
    *,
    show_intervals: bool = True,
) -> GT:
    frame = _forecast_table_frame(
        history,
        baseline,
        specs,
        scenarios,
        show_intervals=show_intervals,
    )
    value_columns = [column for column in frame if column not in {"Month", "Period"}]
    subtitle = "Natural units · forecast cells show medians"
    if show_intervals:
        subtitle += f" and {interval_label(baseline)}"
    table = (
        GT(frame)
        .tab_header(title="Monthly history and forecast", subtitle=subtitle)
        .tab_source_note(
            source_note=md(
                "**Source:** FRED and model estimates. Monthly GDP history is a latent estimate; "
                "forecasts are model output and are not Federal Reserve forecasts."
            )
        )
        .cols_align(align="center")
        .cols_width({"Month": "92px", "Period": "82px"})
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
        labels = {columns[0]: "Baseline"}
        for index, (name, _forecast, _color) in enumerate(scenarios):
            column = f"{spec.series_id}_scenario_{index}"
            columns.append(column)
            labels[column] = name
        table = table.cols_label(cases=cast(dict[str, Any], labels)).tab_spanner(
            label=html(
                f'<span class="variable-name">{spec.short_label}</span>'
                f'<span class="variable-units">{spec.table_units}</span>'
            ),
            columns=columns,
            id=spec.series_id,
        )
    return table.cols_width({column: "154px" for column in value_columns})


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
    scenarios: Sequence[ScenarioForecast] = (),
    transformation: PlotTransformation = "level",
    *,
    show_intervals: bool = True,
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

    def band_data(
        forecast: ForecastResult,
        forecast_lower: pd.Series,
        forecast_upper: pd.Series,
    ) -> list[list[float]]:
        return [
            [last_point[0], last_point[1], last_point[1]],
            *[
                [
                    _timestamp_milliseconds(date),
                    _rounded_chart_value(forecast_lower.loc[date]),
                    _rounded_chart_value(forecast_upper.loc[date]),
                ]
                for date in forecast.dates
            ],
        ]

    historical = [point(date, value) for date, value in recent.items()]
    last_point = point(recent.index[-1], recent.iloc[-1])
    baseline_line = [last_point] + [
        point(date, baseline_median.loc[date]) for date in baseline.dates
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
    bands: list[dict[str, Any]] = []
    if show_intervals:
        bands.append(
            {
                "name": f"Baseline {interval_label(baseline)}",
                "data": band_data(baseline, lower, upper),
                "color": "rgba(54, 214, 189, 0.16)",
            }
        )

    for index, (label, forecast, color_index) in enumerate(scenarios[: len(SCENARIO_STYLES)]):
        line_color, band_color = SCENARIO_STYLES[color_index % len(SCENARIO_STYLES)]
        median, forecast_lower, forecast_upper = forecast_summary_on_scale(
            history_series, forecast, spec, transformation
        )
        line = [last_point] + [point(date, median.loc[date]) for date in forecast.dates]
        if show_intervals:
            bands.append(
                {
                    "name": f"{label} {interval_label(forecast)}",
                    "data": band_data(forecast, forecast_lower, forecast_upper),
                    "color": band_color,
                }
            )
        series.append(
            {
                "name": label,
                "type": "line",
                "data": line,
                "showSymbol": False,
                "lineStyle": {"color": line_color, "width": 3},
                "itemStyle": {"color": line_color},
                "z": 5 + index,
            }
        )

    return {
        "animation": False,
        "backgroundColor": "transparent",
        "aria": {"enabled": True, "description": f"{spec.label} history and forecast chart."},
        "grid": {"left": 62, "right": 22, "top": 20, "bottom": 42},
        "legend": {"show": False},
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
