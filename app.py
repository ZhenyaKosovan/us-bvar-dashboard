from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from secrets import token_hex
from time import perf_counter
from typing import cast

import pandas as pd
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from us_bvar.artifact import ForecastArtifact, load_artifact
from us_bvar.config import SERIES_SPECS, SeriesSpec
from us_bvar.model import ForecastResult
from us_bvar.presentation import (
    echarts_options,
    forecast_gt,
    forecast_summary_on_scale,
    plot_units,
)
from us_bvar.telemetry import event as telemetry_event
from us_bvar.transforms import (
    PLOT_TRANSFORMATIONS,
    PlotTransformation,
    ScenarioConstraint,
    transform_path,
)

ROOT = Path(__file__).resolve().parent
FORECAST_HORIZON = 12
POSTERIOR_DRAWS = 400
ARTIFACT_PATH = ROOT / "artifacts/bvar_forecast.pkl"
SCENARIO_CACHE_SIZE = 32

LOG_LEVEL_NAME = os.getenv("BVAR_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(
    logging, LOG_LEVEL_NAME, logging.DEBUG if LOG_LEVEL_NAME == "TRACE" else logging.INFO
)
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("us_bvar.dashboard")
if LOG_LEVEL_NAME not in logging.getLevelNamesMapping() and LOG_LEVEL_NAME != "TRACE":
    logger.warning("Ignoring invalid BVAR_LOG_LEVEL=%r; using INFO", LOG_LEVEL_NAME)


def _positive_int_environment(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        logger.warning("Ignoring invalid %s value; using %d", name, default)
        return default


def _default_scenario_concurrency() -> int:
    millicores = _positive_int_environment("CDSW_CPU_MILLICORES", 2000)
    return min(4, max(1, millicores // 1000))


MAX_SCENARIO_CONCURRENCY = _positive_int_environment(
    "BVAR_MAX_CONCURRENT_SCENARIOS", _default_scenario_concurrency()
)
ARTIFACT: ForecastArtifact = load_artifact(ARTIFACT_PATH)
MODEL = ARTIFACT.model
BASELINE = ARTIFACT.baseline
if MODEL.history_levels is None:
    raise RuntimeError("The production artifact does not contain model history.")
HISTORY = MODEL.history_levels
SCENARIO_SEMAPHORE = asyncio.Semaphore(MAX_SCENARIO_CONCURRENCY)
telemetry_event(
    "application_started",
    artifact_schema=ARTIFACT.schema_version,
    panel_end=ARTIFACT.panel_end.date().isoformat(),
    posterior_draws=BASELINE.draws,
    scenario_cache_size=SCENARIO_CACHE_SIZE,
    scenario_concurrency=MAX_SCENARIO_CONCURRENCY,
)


ConstraintKey = tuple[tuple[int, str, float, PlotTransformation], ...]


def _constraint_key(constraints: dict[tuple[int, str], ScenarioConstraint]) -> ConstraintKey:
    return tuple(
        sorted(
            (step, series_id, constraint.value, constraint.transformation)
            for (step, series_id), constraint in constraints.items()
        )
    )


@lru_cache(maxsize=SCENARIO_CACHE_SIZE)
def _cached_scenario_forecast(key: ConstraintKey) -> ForecastResult:
    constraints = {
        (step, series_id): ScenarioConstraint(value, transformation)
        for step, series_id, value, transformation in key
    }
    return MODEL.forecast(
        horizon=FORECAST_HORIZON,
        draws=POSTERIOR_DRAWS,
        constraints=constraints,
        seed=202507,
    )


def browser_bootstrap() -> ui.Tag:
    """Initialize responsive ECharts and the client-side scenario editor."""

    return ui.tags.script(
        """
        (() => {
          const chartNodes = (root) => {
            const nodes = [];
            if (root.matches?.('.bvar-echart')) nodes.push(root);
            root.querySelectorAll?.('.bvar-echart').forEach((node) => nodes.push(node));
            return nodes;
          };

          const destroyCharts = (root) => {
            chartNodes(root).forEach((node) => {
              node.bvarResizeObserver?.disconnect();
              node.bvarChart?.dispose();
              node.bvarChart = null;
            });
          };

          const escapeHtml = (value) => String(value).replace(
            /[&<>'"]/g,
            (character) => ({
              '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
            }[character]),
          );

          const prepareChartOption = (raw) => {
            const bands = raw.bvarBands || [];
            const decimals = raw.bvarValueDecimals ?? 2;
            const units = raw.bvarUnits || '';
            delete raw.bvarBands;
            delete raw.bvarValueDecimals;
            delete raw.bvarUnits;
            const bandSeries = bands.map((band) => ({
              name: band.name,
              type: 'custom',
              data: band.data,
              dimensions: ['month', 'lower', 'upper'],
              encode: {x: 0, y: [1, 2]},
              silent: true,
              tooltip: {show: false},
              z: 1,
              renderItem: (params, api) => {
                const current = band.data[params.dataIndex];
                const next = band.data[params.dataIndex + 1];
                if (!next) return null;
                return {
                  type: 'polygon',
                  shape: {
                    points: [
                      api.coord([current[0], current[1]]),
                      api.coord([next[0], next[1]]),
                      api.coord([next[0], next[2]]),
                      api.coord([current[0], current[2]]),
                    ],
                  },
                  style: {fill: band.color, stroke: 'none'},
                };
              },
            }));
            raw.series = [...bandSeries, ...raw.series];
            raw.xAxis.axisLabel.formatter = (value) => new Intl.DateTimeFormat(
              undefined,
              {month: 'short', year: '2-digit', timeZone: 'UTC'},
            ).format(new Date(value));
            raw.tooltip.formatter = (parameters) => {
              const visible = parameters.filter((item) => item.seriesType !== 'custom');
              if (!visible.length) return '';
              const month = new Intl.DateTimeFormat(
                undefined,
                {month: 'long', year: 'numeric', timeZone: 'UTC'},
              ).format(new Date(visible[0].value[0]));
              const rows = visible.map((item) => (
                `${item.marker}${escapeHtml(item.seriesName)}: ` +
                `<strong>${Number(item.value[1]).toLocaleString(undefined, {
                  minimumFractionDigits: decimals,
                  maximumFractionDigits: decimals,
                })}</strong>`
              ));
              return `<strong>${month}</strong><br>${rows.join('<br>')}<br>` +
                `<span class="chart-tooltip-units">${escapeHtml(units)}</span>`;
            };
            return raw;
          };

          const renderCharts = (root = document) => {
            chartNodes(root).forEach((node) => {
              const configNode = node.querySelector('script.chart-config');
              const target = node.querySelector('.chart-target');
              if (!configNode || !target || typeof echarts === 'undefined') return;
              const signature = configNode.textContent;
              if (node.dataset.signature === signature) return;
              node.bvarResizeObserver?.disconnect();
              node.bvarChart?.dispose();
              node.bvarChart = echarts.init(target, null, {renderer: 'svg'});
              node.bvarChart.setOption(prepareChartOption(JSON.parse(signature)));
              let lastWidth = target.clientWidth;
              let lastHeight = target.clientHeight;
              node.bvarResizeObserver = new ResizeObserver((entries) => {
                const rectangle = entries[0]?.contentRect;
                if (!rectangle || (
                  rectangle.width === lastWidth && rectangle.height === lastHeight
                )) {
                  return;
                }
                lastWidth = rectangle.width;
                lastHeight = rectangle.height;
                requestAnimationFrame(() => node.bvarChart?.resize());
              });
              node.bvarResizeObserver.observe(target);
              node.dataset.signature = signature;
            });
          };

          const updateConstraintCount = () => {
            const editor = document.querySelector('.scenario-modal');
            if (!editor) return;
            const dialog = editor.closest('.modal-content') || editor;
            const count = [...editor.querySelectorAll('.scenario-value')]
              .filter((field) => field.value.trim()).length;
            const badge = editor.querySelector('.constraint-count');
            const runButton = dialog.querySelector('#run_scenario');
            if (badge) badge.textContent = count === 1 ? '1 constraint' : `${count} constraints`;
            if (runButton) runButton.disabled = count === 0;
          };

          const updateScenarioRow = (select) => {
            const row = select.closest('[data-scenario-row]');
            if (!row) return;
            const unitsByScale = JSON.parse(row.dataset.unitsByScale);
            const transformation = Object.hasOwn(unitsByScale, select.value)
              ? select.value
              : 'level';
            row.querySelector('.scenario-row-units').textContent = unitsByScale[transformation];
            row.querySelectorAll('[data-values-by-scale]').forEach((cell) => {
              const values = JSON.parse(cell.dataset.valuesByScale);
              cell.textContent = values[transformation];
            });
            row.querySelectorAll('.scenario-value').forEach((field) => {
              const placeholders = JSON.parse(field.dataset.placeholdersByScale);
              field.placeholder = placeholders[transformation] ?? placeholders.level;
              field.setAttribute(
                'aria-label',
                `${field.dataset.seriesLabel}, ${field.dataset.month}, ` +
                  unitsByScale[transformation],
              );
            });
          };

          const initializeScenarioEditors = (root = document) => {
            const editors = [];
            if (root.matches?.('.scenario-modal')) editors.push(root);
            root.querySelectorAll?.('.scenario-modal').forEach((node) => editors.push(node));
            editors.forEach((editor) => {
              editor.querySelectorAll('[id^="sc_transform_"]').forEach(updateScenarioRow);
              updateConstraintCount();
            });
          };

          document.addEventListener('change', (event) => {
            if (event.target.matches('[id^="sc_transform_"]')) updateScenarioRow(event.target);
            if (event.target.matches('.scenario-value')) updateConstraintCount();
          });
          document.addEventListener('input', (event) => {
            if (event.target.matches('.scenario-value')) updateConstraintCount();
          });
          document.addEventListener('DOMContentLoaded', () => {
            renderCharts();
            initializeScenarioEditors();
            new MutationObserver((mutations) => {
              mutations.forEach((mutation) => {
                mutation.removedNodes.forEach((node) => {
                  if (node.nodeType === 1) destroyCharts(node);
                });
                mutation.addedNodes.forEach((node) => {
                  if (node.nodeType === 1) {
                    renderCharts(node);
                    initializeScenarioEditors(node);
                  }
                });
              });
            }).observe(document.body, {childList: true, subtree: true});
          });
        })();
        """
    )


def chart_cards() -> list[ui.Tag]:
    cards: list[ui.Tag] = []
    for spec in SERIES_SPECS:
        cards.append(
            ui.tags.article(
                ui.div(
                    ui.div(
                        ui.div(spec.short_label.upper(), class_="chart-kicker"), ui.h2(spec.label)
                    ),
                    ui.div(
                        ui.input_select(
                            f"plot_transform_{spec.series_id}",
                            "View as",
                            choices={
                                key: transform_spec.label
                                for key, transform_spec in PLOT_TRANSFORMATIONS.items()
                            },
                            selected="level",
                            width="190px",
                        ),
                        class_="chart-transform",
                    ),
                    class_="chart-heading",
                ),
                ui.output_ui(f"chart_{spec.series_id}"),
                class_="chart-card",
            )
        )
    return cards


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.meta(name="theme-color", content="#08111f"),
        ui.tags.link(rel="icon", href="favicon.svg", type="image/svg+xml"),
        ui.tags.link(rel="stylesheet", href="app.css"),
        ui.tags.script(src="vendor/echarts/echarts.min.js"),
        browser_bootstrap(),
    ),
    ui.output_ui("scenario_progress"),
    ui.tags.header(
        ui.div(
            ui.div("US MACRO · MONTHLY", class_="eyebrow"),
            ui.h1("Macro scenario studio", class_="app-title"),
            ui.p(
                "Explore a transparent Bayesian baseline, then shape conditional paths "
                "across output, demand, prices, labor, and policy.",
                class_="app-subtitle",
            ),
            class_="brand-block",
        ),
        ui.div(
            ui.div("FORECAST VINTAGE", class_="artifact-kicker"),
            ui.div(f"Data through {ARTIFACT.panel_end:%B %Y}", class_="artifact-date"),
            ui.div(f"Built {ARTIFACT.created_at:%d %b %Y · %H:%M UTC}", class_="artifact-built"),
            class_="artifact-badge",
        ),
        class_="masthead",
    ),
    ui.tags.section(
        ui.div(
            ui.div("MODEL READY", class_="status-label"),
            ui.div(
                f"{ARTIFACT.observation_count:,} observations · "
                f"{BASELINE.draws:,} posterior paths · "
                f"{FORECAST_HORIZON}-month horizon",
                class_="status-detail",
            ),
            class_="status-copy",
        ),
        ui.div(
            ui.output_ui("scenario_summary"),
            ui.input_action_button("open_scenario", "Build a scenario", class_="btn-scenario"),
            ui.input_action_button("clear_scenario", "Clear", class_="btn-clear", disabled=True),
            class_="scenario-actions",
        ),
        class_="status-bar",
    ),
    ui.div(
        ui.span("DISPLAY", class_="guidance-label"),
        "Each chart can show levels or a growth/change rate. Scenario inputs use the same choices.",
        class_="transform-guidance",
    ),
    ui.tags.main(*chart_cards(), class_="chart-grid"),
    ui.tags.section(ui.output_ui("forecast_table"), class_="table-shell"),
    ui.tags.footer(
        "4 monthly lags · Minnesota prior · pandemic controls · "
        "pointwise 16th–84th percentile posterior interval · not a causal forecast",
        class_="method-note",
    ),
)


def _selected_transformation(
    existing_constraints: dict[tuple[int, str], ScenarioConstraint], spec: SeriesSpec
) -> PlotTransformation:
    for (_step, variable_id), constraint in existing_constraints.items():
        if variable_id == spec.series_id:
            return constraint.transformation
    return "level"


def _scenario_row(
    spec: SeriesSpec,
    existing_constraints: dict[tuple[int, str], ScenarioConstraint],
) -> ui.Tag:
    history = cast(pd.Series, HISTORY[spec.series_id]).astype(float)
    selected = _selected_transformation(existing_constraints, spec)
    units_by_scale: dict[str, str] = {}
    history_by_scale: dict[str, list[str]] = {}
    baseline_by_scale: dict[str, list[str]] = {}
    for transformation in PLOT_TRANSFORMATIONS:
        decimals = spec.decimals if transformation == "level" else 2
        units_by_scale[transformation] = plot_units(spec, transformation)
        transformed_history = transform_path(history, spec, transformation)
        median, _lower, _upper = forecast_summary_on_scale(history, BASELINE, spec, transformation)
        history_by_scale[transformation] = [
            f"{value:,.{decimals}f}" for value in transformed_history.tail(3)
        ]
        baseline_by_scale[transformation] = [f"{value:.{decimals}f}" for value in median]

    transformation_input = ui.input_select(
        f"sc_transform_{spec.series_id}",
        f"Transformation for {spec.label}",
        choices={key: value.label for key, value in PLOT_TRANSFORMATIONS.items()},
        selected=selected,
        width="100%",
    )
    cast(ui.Tag, transformation_input.children[1]).attrs.update(
        {
            "class": "form-select scenario-transform",
            "data-series-id": spec.series_id,
        }
    )
    variable_cell = ui.div(
        ui.div(spec.short_label, class_="scenario-row-label", title=spec.label),
        class_="scenario-variable-cell",
    )
    transformation_cell = ui.div(
        transformation_input,
        ui.div(units_by_scale[selected], class_="scenario-row-units"),
        class_="scenario-transformation-cell",
    )
    cells: list[ui.Tag] = [variable_cell, transformation_cell]
    for history_index in range(3):
        values_by_scale = {
            transformation: values[history_index]
            for transformation, values in history_by_scale.items()
        }
        cell = ui.div(values_by_scale[selected], class_="scenario-grid-cell history-value")
        cell.attrs["data-values-by-scale"] = json.dumps(values_by_scale, separators=(",", ":"))
        cells.append(cell)

    for step, date in enumerate(BASELINE.dates):
        constraint = existing_constraints.get((step, spec.series_id))
        value = ""
        if constraint is not None and constraint.transformation == selected:
            value = str(constraint.value)
        placeholders = {
            transformation: values[step] for transformation, values in baseline_by_scale.items()
        }
        scenario_input = ui.input_text(
            f"sc_{spec.series_id}_{step}",
            None,
            value=value,
            placeholder=placeholders[selected],
        )
        input_tag = cast(ui.Tag, scenario_input.children[1])
        input_tag.attrs.update(
            {
                "aria-label": f"{spec.label}, {date:%B %Y}, {units_by_scale[selected]}",
                "class": "form-control scenario-value",
                "data-month": f"{date:%B %Y}",
                "data-placeholders-by-scale": json.dumps(placeholders, separators=(",", ":")),
                "data-series-label": spec.label,
                "inputmode": "decimal",
                "autocomplete": "off",
            }
        )
        cells.append(ui.div(scenario_input, class_="scenario-grid-cell forecast-value"))
    row = ui.div(*cells, class_="scenario-grid-row")
    row.attrs.update(
        {
            "data-scenario-row": spec.series_id,
            "data-units-by-scale": json.dumps(units_by_scale, separators=(",", ":")),
        }
    )
    return row


def scenario_modal(existing: ForecastResult | None) -> ui.Tag:
    existing_constraints = dict(existing.constraints) if existing else {}
    recent_dates = HISTORY.index[-3:]
    header_cells: list[ui.Tag] = [
        ui.div(
            ui.div("VARIABLE", class_="scenario-header-primary"),
            class_="scenario-grid-corner scenario-variable-header",
        ),
        ui.div(
            ui.div("TRANSFORMATION", class_="scenario-header-primary"),
            ui.div("Choose a scale", class_="scenario-header-secondary"),
            class_="scenario-grid-corner scenario-transformation-header",
        ),
    ]
    for date in recent_dates:
        header_cells.append(
            ui.div(
                ui.span("ACTUAL", class_="month-kind actual-kind"),
                ui.span(date.strftime("%b %y"), class_="month-label"),
                class_="scenario-month-header actual-month",
            )
        )
    for date in BASELINE.dates:
        header_cells.append(
            ui.div(
                ui.span("FORECAST", class_="month-kind forecast-kind"),
                ui.span(date.strftime("%b %y"), class_="month-label"),
                class_="scenario-month-header forecast-month",
            )
        )

    grid_cells = header_cells.copy()
    for spec in SERIES_SPECS:
        grid_cells.append(_scenario_row(spec, existing_constraints))

    modal = ui.modal(
        ui.div(
            ui.div(
                ui.div(
                    "Enter only the values you want to constrain. Empty forecast cells stay "
                    "model-driven; muted placeholders are the current baseline medians.",
                    class_="modal-instructions",
                ),
                ui.div(
                    f"{len(existing_constraints)} constraints",
                    class_="constraint-count",
                    aria_live="polite",
                ),
                class_="scenario-intro",
            ),
            ui.div(ui.div(*grid_cells, class_="scenario-grid"), class_="scenario-grid-scroll"),
            class_="scenario-modal",
        ),
        title="Build a conditional path",
        size="xl",
        easy_close=True,
        footer=ui.div(
            ui.div(
                ui.span("Tip", class_="footer-tip-label"),
                " Start with one or two assumptions; add detail only where it matters.",
                class_="modal-tip",
            ),
            ui.div(
                ui.input_action_button("reset_modal", "Reset fields", class_="btn-clear"),
                ui.input_action_button(
                    "run_scenario",
                    "Run scenario",
                    class_="btn-scenario",
                    disabled=not existing_constraints,
                ),
                class_="modal-actions",
            ),
            class_="modal-footer-layout",
        ),
    )
    modal.attrs.update(
        {"aria-label": "Conditional scenario path", "aria-modal": "true", "role": "dialog"}
    )
    return modal


def server(input: Inputs, output: Outputs, session: Session) -> None:
    scenario_state: reactive.Value[ForecastResult | None] = reactive.Value(None)
    session_id = token_hex(8)
    session_started_at = perf_counter()
    scenario_requested_at: float | None = None
    scenario_constraint_count = 0
    scenario_variables: list[str] = []
    telemetry_event("session_started", session_id=session_id)

    def _session_ended() -> None:
        telemetry_event(
            "session_ended",
            session_id=session_id,
            duration_ms=round((perf_counter() - session_started_at) * 1000),
        )

    session.on_ended(_session_ended)

    @reactive.extended_task
    async def _scenario_task(key: ConstraintKey) -> ForecastResult:
        async with SCENARIO_SEMAPHORE:
            return await asyncio.to_thread(_cached_scenario_forecast, key)

    @output
    @render.ui
    def scenario_progress() -> ui.Tag | None:
        if _scenario_task.status() != "running":
            return None
        return ui.div(
            ui.div(
                ui.div(class_="scenario-progress-spinner", aria_hidden="true"),
                ui.div("Calculating your conditional path", class_="scenario-progress-title"),
                ui.p(
                    "Updating 400 posterior paths and their joint uncertainty.",
                    class_="scenario-progress-copy",
                ),
                class_="scenario-progress-card",
            ),
            class_="scenario-progress-overlay",
            role="status",
            aria_live="polite",
            aria_busy="true",
        )

    @output
    @render.ui
    def scenario_summary() -> ui.Tag:
        scenario = scenario_state.get()
        if scenario is None:
            return ui.div(ui.span(class_="scenario-dot"), "Baseline", class_="scenario-chip")
        return ui.div(
            ui.span(class_="scenario-dot scenario-dot-active"),
            f"Scenario · {len(scenario.constraints)}",
            class_="scenario-chip scenario-chip-active",
        )

    def _register_chart(chart_spec: SeriesSpec) -> None:
        @output(id=f"chart_{chart_spec.series_id}", suspend_when_hidden=False)
        @render.ui
        def _chart():
            raw_transformation = input[f"plot_transform_{chart_spec.series_id}"]()
            transformation: PlotTransformation = (
                raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
            )
            options = json.dumps(
                echarts_options(
                    HISTORY,
                    BASELINE,
                    chart_spec,
                    scenario_state.get(),
                    transformation,
                ),
                separators=(",", ":"),
            ).replace("</", "<\\/")
            return ui.HTML(
                f'<div class="bvar-echart" role="img" aria-label="{chart_spec.label} forecast">'
                '<div class="chart-target"></div>'
                f'<script type="application/json" class="chart-config">{options}</script>'
                "</div>"
            )

    for chart_spec in SERIES_SPECS:
        _register_chart(chart_spec)

    @output
    @render.ui
    def forecast_table():
        table = forecast_gt(HISTORY, BASELINE, SERIES_SPECS, scenario_state.get())
        return ui.HTML(table.as_raw_html())

    @reactive.effect
    @reactive.event(input.open_scenario)
    def _open_scenario() -> None:
        ui.modal_show(scenario_modal(scenario_state.get()))

    @reactive.effect
    @reactive.event(input.run_scenario)
    def _run_scenario() -> None:
        nonlocal scenario_constraint_count, scenario_requested_at, scenario_variables
        telemetry_event("scenario_run_clicked", session_id=session_id)
        constraints: dict[tuple[int, str], ScenarioConstraint] = {}
        try:
            for spec in SERIES_SPECS:
                raw_transformation = input[f"sc_transform_{spec.series_id}"]()
                transformation: PlotTransformation = (
                    raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
                )
                for step in range(FORECAST_HORIZON):
                    raw = input[f"sc_{spec.series_id}_{step}"]()
                    if raw is not None and str(raw).strip():
                        normalized = str(raw).strip().replace(",", "")
                        try:
                            value = float(normalized)
                        except ValueError as exc:
                            raise ValueError(
                                f"{spec.short_label}, {BASELINE.dates[step]:%b %Y} "
                                "must be a valid number."
                            ) from exc
                        constraints[(step, spec.series_id)] = ScenarioConstraint(
                            value=value,
                            transformation=transformation,
                        )
            if not constraints:
                raise ValueError("Enter at least one scenario value.")
            key = _constraint_key(constraints)
            scenario_requested_at = perf_counter()
            scenario_constraint_count = len(constraints)
            scenario_variables = sorted({series_id for _step, series_id in constraints})
            telemetry_event(
                "scenario_requested",
                session_id=session_id,
                constraint_count=scenario_constraint_count,
                variables=scenario_variables,
            )
            ui.modal_remove()
            _ = _scenario_task(key)
        except ValueError as exc:
            telemetry_event(
                "scenario_rejected",
                session_id=session_id,
                reason="validation_error",
            )
            ui.notification_show(str(exc), type="error", duration=8)

    @reactive.effect
    def _handle_scenario_result() -> None:
        nonlocal scenario_requested_at
        task_status = _scenario_task.status()
        if task_status == "success":
            scenario = _scenario_task.result()
            if scenario_requested_at is not None:
                telemetry_event(
                    "scenario_completed",
                    session_id=session_id,
                    constraint_count=scenario_constraint_count,
                    variables=scenario_variables,
                    duration_ms=round((perf_counter() - scenario_requested_at) * 1000),
                )
                scenario_requested_at = None
            scenario_state.set(scenario)
            ui.update_action_button("clear_scenario", disabled=False)
            ui.notification_show(
                f"Scenario applied with {len(scenario.constraints)} constrained values.",
                type="message",
                duration=4,
            )
        elif task_status == "error":
            error = _scenario_task.error.get()
            if scenario_requested_at is not None:
                telemetry_event(
                    "scenario_failed",
                    session_id=session_id,
                    constraint_count=scenario_constraint_count,
                    variables=scenario_variables,
                    duration_ms=round((perf_counter() - scenario_requested_at) * 1000),
                    error_type=type(error).__name__,
                )
                scenario_requested_at = None
            logger.error(
                "Scenario calculation failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            message = (
                str(error) if isinstance(error, ValueError) else "Scenario calculation failed."
            )
            ui.notification_show(message, type="error", duration=8)

    @reactive.effect
    @reactive.event(input.clear_scenario)
    def _clear_scenario() -> None:
        scenario_state.set(None)
        ui.update_action_button("clear_scenario", disabled=True)

    @reactive.effect
    @reactive.event(input.reset_modal)
    def _reset_modal() -> None:
        ui.modal_remove()
        ui.modal_show(scenario_modal(None))


async def _health(_request: Request) -> JSONResponse:
    cache = _cached_scenario_forecast.cache_info()
    return JSONResponse(
        {
            "status": "ok",
            "artifact_schema": ARTIFACT.schema_version,
            "panel_end": ARTIFACT.panel_end.date().isoformat(),
            "forecast_end": cast(pd.Timestamp, BASELINE.dates[-1]).date().isoformat(),
            "scenario_cache": {"size": cache.currsize, "max_size": cache.maxsize},
        },
        headers={"Cache-Control": "no-store"},
    )


class SecurityHeadersMiddleware:
    """Set safe response headers without interfering with CML iframe proxying."""

    def __init__(self, application) -> None:
        self.application = application

    async def __call__(self, scope, receive, send) -> None:
        async def send_with_headers(message) -> None:
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"referrer-policy", b"same-origin"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
            await send(message)

        await self.application(scope, receive, send_with_headers)


app = App(app_ui, server, static_assets=ROOT / "www", debug=False)
app.sanitize_errors = True
app.starlette_app.routes.insert(0, Route("/healthz", endpoint=_health, methods=["GET"]))
app.starlette_app.add_middleware(SecurityHeadersMiddleware)
