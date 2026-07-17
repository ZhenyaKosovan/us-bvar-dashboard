from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from secrets import token_hex
from time import perf_counter
from typing import Any, cast

import pandas as pd
from shiny import (  # type: ignore[import-not-found]
    App,
    Inputs,
    Outputs,
    Session,
    reactive,
    render,
    ui,
)
from starlette.requests import Request  # type: ignore[import-not-found]
from starlette.responses import JSONResponse  # type: ignore[import-not-found]
from starlette.routing import Route  # type: ignore[import-not-found]

from us_bvar.artifact import ForecastArtifact, PublishedRelease, load_published_release
from us_bvar.config import (
    DEFAULT_DASHBOARD_SERIES,
    SERIES_BY_ID,
    SERIES_GROUPS,
    SERIES_SPECS,
    SeriesSpec,
)
from us_bvar.model import ForecastResult
from us_bvar.presentation import (
    echarts_options,
    forecast_gt,
    forecast_summary_on_scale,
    interval_label,
    plot_units,
)
from us_bvar.telemetry import event as telemetry_event
from us_bvar.transforms import (
    PLOT_TRANSFORMATIONS,
    PlotTransformation,
    ScenarioConstraint,
    transform_path,
)

ROOT = Path(__file__).resolve().parents[2]
FORECAST_HORIZON = 12
POSTERIOR_DRAWS = 400
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


ConstraintKey = tuple[tuple[int, str, float, PlotTransformation], ...]


def _constraint_key(constraints: dict[tuple[int, str], ScenarioConstraint]) -> ConstraintKey:
    return tuple(
        sorted(
            (step, series_id, constraint.value, constraint.transformation)
            for (step, series_id), constraint in constraints.items()
        )
    )


@lru_cache(maxsize=SCENARIO_CACHE_SIZE)
def _cached_scenario_forecast(model, key: ConstraintKey) -> ForecastResult:
    constraints = {
        (step, series_id): ScenarioConstraint(value, transformation)
        for step, series_id, value, transformation in key
    }
    return model.forecast(
        horizon=FORECAST_HORIZON,
        draws=POSTERIOR_DRAWS,
        constraints=constraints,
        seed=202507,
    )


def browser_bootstrap() -> ui.Tag:
    """Load the local browser integration module."""

    return ui.tags.script(src="app.js")


WORKSPACE_PRESETS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Overview", "A balanced view of activity, prices, labor, and rates", DEFAULT_DASHBOARD_SERIES),
    (
        "Inflation & policy",
        "Headline and core inflation alongside policy and market rates",
        ("CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "FEDFUNDS", "GS10"),
    ),
    (
        "Growth & labor",
        "Output, spending, production, payrolls, and unemployment",
        ("GDPC1", "PCEC96", "INDPRO", "RRSFS", "PAYEMS", "UNRATE"),
    ),
)


def _sparkline(runtime, spec: SeriesSpec) -> Any:
    history = cast(pd.Series, runtime.history[spec.series_id])
    numeric_history = cast(pd.Series, pd.to_numeric(history, errors="coerce"))
    values = numeric_history.dropna().tail(18).tolist()
    if len(values) < 2:
        return ui.span(class_="library-sparkline-placeholder", aria_hidden="true")
    low, high = min(values), max(values)
    spread = high - low or 1.0
    points = " ".join(
        f"{index * 88 / (len(values) - 1):.1f},{22 - ((value - low) / spread) * 18:.1f}"
        for index, value in enumerate(values)
    )
    return ui.HTML(
        '<svg class="library-sparkline" viewBox="0 0 88 26" aria-hidden="true" '
        'focusable="false"><polyline points="'
        f"{points}"
        '" vector-effect="non-scaling-stroke"></polyline></svg>'
    )


def variable_library(runtime) -> ui.Tag:
    items: list[ui.Tag] = []
    for spec in SERIES_SPECS:
        add_button = ui.tags.button(
            ui.span("+", aria_hidden="true"),
            ui.span(f"Add {spec.short_label}", class_="visually-hidden"),
            type="button",
            class_="variable-add-button",
            title=f"Add {spec.short_label} to the canvas",
        )
        add_button.attrs["data-series-id"] = spec.series_id
        item = ui.div(
            ui.div(
                ui.div(spec.short_label, class_="library-item-label", title=spec.label),
                ui.div(f"{spec.series_id} · {spec.group}", class_="library-item-meta"),
                class_="library-item-copy",
            ),
            _sparkline(runtime, spec),
            add_button,
            class_="variable-library-item",
        )
        item.attrs.update(
            {
                "draggable": "true",
                "data-series-id": spec.series_id,
                "data-series-group": spec.group,
                "data-series-search": (
                    f"{spec.series_id} {spec.short_label} {spec.label} {spec.group}".lower()
                ),
            }
        )
        items.append(item)
    return ui.div(*items, class_="variable-library-list", id="variable-library-list")


def _card_action(
    symbol: str, label: str, class_name: str, spec: SeriesSpec, *, pressed: bool | None = None
) -> ui.Tag:
    button = ui.tags.button(
        ui.span(symbol, aria_hidden="true"),
        ui.span(f"{label} {spec.short_label}", class_="visually-hidden"),
        type="button",
        class_=f"chart-action {class_name}",
        title=f"{label} {spec.short_label}",
    )
    if pressed is not None:
        button.attrs["aria-pressed"] = str(pressed).lower()
    button.attrs["data-series-id"] = spec.series_id
    return button


def _chart_menu_option(
    label: str, class_name: str, spec: SeriesSpec, *, size: str | None = None
) -> ui.Tag:
    button = ui.tags.button(
        label,
        type="button",
        class_=f"chart-menu-option {class_name}",
        role="menuitemradio" if size else "menuitem",
    )
    button.attrs["data-series-id"] = spec.series_id
    if size:
        button.attrs.update(
            {
                "data-card-size": size,
                "aria-checked": str(size == "standard").lower(),
            }
        )
    return button


def chart_cards(runtime, specs: tuple[SeriesSpec, ...] = SERIES_SPECS) -> list[ui.Tag]:
    cards: list[ui.Tag] = []
    for spec in specs:
        drag_handle = _card_action("⠿", "Drag", "chart-drag-handle", spec)
        drag_handle.attrs["draggable"] = "true"
        chart_menu = ui.tags.details(
            ui.tags.summary(
                ui.span("•••", aria_hidden="true"),
                ui.span(f"Options for {spec.short_label}", class_="visually-hidden"),
                class_="chart-menu-trigger",
                title=f"Options for {spec.short_label}",
            ),
            ui.div(
                _chart_menu_option("Move earlier", "chart-move-earlier", spec),
                _chart_menu_option("Move later", "chart-move-later", spec),
                ui.div("CARD SIZE", class_="chart-menu-label"),
                ui.div(
                    _chart_menu_option("Standard", "chart-size-option", spec, size="standard"),
                    _chart_menu_option("Wide", "chart-size-option", spec, size="wide"),
                    _chart_menu_option("Tall", "chart-size-option", spec, size="tall"),
                    _chart_menu_option("Focus", "chart-size-option", spec, size="focus"),
                    class_="chart-size-options",
                ),
                ui.div(class_="chart-menu-divider"),
                _chart_menu_option("Remove chart", "chart-remove-button", spec),
                class_="chart-menu-panel",
                role="menu",
            ),
            class_="chart-menu",
        )
        card = ui.tags.article(
            ui.div(
                ui.div(
                    drag_handle,
                    ui.div(
                        ui.div(spec.short_label.upper(), class_="chart-kicker"),
                        ui.h2(spec.label),
                    ),
                    class_="chart-title-group",
                ),
                ui.div(
                    ui.div(
                        ui.input_select(
                            f"plot_transform_{spec.series_id}",
                            "View as",
                            choices={
                                key: transform_spec.label
                                for key, transform_spec in PLOT_TRANSFORMATIONS.items()
                            },
                            selected=spec.default_plot_transform,
                            width="150px",
                        ),
                        class_="chart-transform",
                    ),
                    chart_menu,
                    class_="chart-heading-controls",
                ),
                class_="chart-heading",
            ),
            ui.output_ui(f"chart_{spec.series_id}"),
            class_="chart-card",
        )
        card.attrs.update({"data-series-id": spec.series_id, "data-card-size": "standard"})
        cards.append(card)
    return cards


def _preset_buttons() -> list[ui.Tag]:
    buttons: list[ui.Tag] = []
    for name, description, series_ids in WORKSPACE_PRESETS:
        button = ui.tags.button(name, type="button", class_="workspace-preset", title=description)
        button.attrs.update(
            {"data-workspace-preset": name, "data-series-ids": json.dumps(series_ids)}
        )
        buttons.append(button)
    return buttons


def build_ui(runtime) -> ui.Tag:
    return ui.page_fluid(
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
                ui.div(
                    ui.div(
                        ui.span(class_="app-mark-bar"),
                        ui.span(class_="app-mark-bar"),
                        ui.span(class_="app-mark-bar"),
                        class_="app-mark",
                        aria_hidden="true",
                    ),
                    ui.div("US MACRO · MIXED FREQUENCY", class_="eyebrow"),
                    ui.div(
                        ui.span(class_="live-dot", aria_hidden="true"),
                        "Model online",
                        class_="live-badge",
                    ),
                    class_="brand-meta",
                ),
                ui.h1(
                    "Macro scenario ", ui.span("studio", class_="title-accent"), class_="app-title"
                ),
                ui.p(
                    "Compose a forecast view, then test conditional paths across the US economy.",
                    class_="app-subtitle",
                ),
                class_="brand-block",
            ),
            ui.div(
                ui.div(
                    ui.span("↗", class_="artifact-icon", aria_hidden="true"),
                    ui.div(
                        ui.div("FORECAST VINTAGE", class_="artifact-kicker"),
                        ui.div(
                            f"Data through {runtime.artifact.panel_end:%B %Y}",
                            class_="artifact-date",
                        ),
                        ui.div(
                            f"Built {runtime.artifact.created_at:%d %b %Y · %H:%M UTC}",
                            class_="artifact-built",
                        ),
                    ),
                    class_="artifact-badge-content",
                ),
                ui.div("Latest published baseline", class_="artifact-status"),
                class_="artifact-badge",
            ),
            class_="masthead",
        ),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.span(class_="status-pulse", aria_hidden="true"),
                    ui.div("MODEL READY", class_="status-label"),
                    class_="status-heading",
                ),
                ui.div(
                    ui.div(
                        ui.strong(f"{runtime.artifact.observation_count:,}"),
                        ui.span("months"),
                        class_="model-stat",
                    ),
                    ui.div(
                        ui.strong(str(len(SERIES_SPECS))),
                        ui.span("variables"),
                        class_="model-stat",
                    ),
                    ui.div(
                        ui.strong(f"{runtime.baseline.draws:,}"),
                        ui.span("paths"),
                        class_="model-stat",
                    ),
                    ui.div(
                        ui.strong(str(FORECAST_HORIZON)),
                        ui.span("mo horizon"),
                        class_="model-stat",
                    ),
                    class_="model-stats",
                ),
                class_="status-copy",
            ),
            ui.div(
                ui.output_ui("scenario_summary"),
                ui.input_action_button("open_scenario", "Build scenario", class_="btn-scenario"),
                ui.input_action_button(
                    "clear_scenario", "Clear", class_="btn-clear", disabled=True
                ),
                class_="scenario-actions",
            ),
            class_="status-bar",
        ),
        ui.tags.details(
            ui.tags.summary("How to read these forecasts"),
            ui.p(
                "Charts can show levels or growth/change rates. GDP history and monthly scenarios "
                "refer to the latent monthly estimate; official GDPC1 remains quarterly. Early "
                "growth anchors use displayed fixed history, while paired terminal-state "
                "uncertainty still drives forecast dynamics. This is not a causal forecast."
            ),
            class_="method-disclosure",
        ),
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.div("ANALYSIS CANVAS", class_="guidance-label"),
                    ui.h2("Build your forecast view", class_="workspace-title"),
                    ui.p(
                        "Drag or add up to eight indicators. Reorder and resize cards to match "
                        "the question you are exploring.",
                        class_="workspace-subtitle",
                    ),
                    class_="workspace-heading",
                ),
                ui.div(
                    ui.span("START FROM", class_="preset-label"),
                    ui.div(*_preset_buttons(), class_="workspace-presets"),
                    class_="preset-group",
                ),
                class_="workspace-toolbar",
            ),
            ui.div(
                ui.tags.aside(
                    ui.div(
                        ui.div("CHART LIBRARY", class_="guidance-label"),
                        ui.output_ui("variable_selection_summary"),
                        class_="library-heading",
                    ),
                    ui.tags.label(
                        "Find an indicator",
                        ui.tags.input(
                            id="variable-library-search",
                            type="search",
                            placeholder="Search name or FRED ID",
                            autocomplete="off",
                        ),
                        class_="library-search-label",
                    ),
                    ui.div(
                        ui.tags.button(
                            "All",
                            type="button",
                            class_="library-filter is-active",
                            data_group="All",
                            aria_pressed="true",
                        ),
                        *[
                            ui.tags.button(
                                group,
                                type="button",
                                class_="library-filter",
                                data_group=group,
                                aria_pressed="false",
                            )
                            for group in SERIES_GROUPS
                        ],
                        class_="library-filters",
                        aria_label="Filter chart library by group",
                    ),
                    variable_library(runtime),
                    ui.div("No indicators match this filter.", class_="library-empty", hidden=True),
                    class_="variable-library",
                    aria_label="Available forecast charts",
                ),
                ui.div(
                    ui.div(
                        ui.div(
                            ui.span("YOUR MATRIX", class_="canvas-kicker"),
                            ui.span("Drag cards to rearrange", class_="canvas-hint"),
                        ),
                        ui.div(
                            "Workspace changes are saved in this browser.",
                            class_="canvas-save-status",
                        ),
                        class_="canvas-header",
                    ),
                    ui.output_ui("chart_grid"),
                    class_="workspace-canvas-shell",
                ),
                class_="workspace-shell",
            ),
            ui.div(
                ui.input_selectize(
                    "visible_variables",
                    "Series shown",
                    choices={
                        spec.series_id: f"{spec.group} · {spec.short_label} ({spec.series_id})"
                        for spec in SERIES_SPECS
                    },
                    selected=list(DEFAULT_DASHBOARD_SERIES),
                    multiple=True,
                    options={"maxItems": 8},
                ),
                class_="workspace-selection-control",
                aria_hidden="true",
            ),
            ui.div(
                "",
                id="workspace-announcer",
                class_="visually-hidden",
                aria_live="polite",
                aria_atomic="true",
            ),
            class_="workspace",
        ),
        ui.tags.details(
            ui.tags.summary(
                ui.span("Forecast data", class_="table-disclosure-title"),
                ui.span(
                    "History, medians, and uncertainty intervals",
                    class_="table-disclosure-copy",
                ),
            ),
            ui.tags.section(ui.output_ui("forecast_table"), class_="table-shell"),
            class_="table-disclosure",
        ),
        ui.tags.footer(
            "Latent monthly GDP from quarterly GDPC1 · Kalman simulation smoother · "
            "22-variable BVAR · 4 monthly lags · Minnesota prior · "
            f"pointwise {runtime.forecast_interval_label} · not a causal forecast",
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
    runtime,
    spec: SeriesSpec,
    existing_constraints: dict[tuple[int, str], ScenarioConstraint],
) -> ui.Tag:
    history = cast(pd.Series, runtime.history[spec.series_id]).astype(float)
    selected = _selected_transformation(existing_constraints, spec)
    units_by_scale: dict[str, str] = {}
    history_by_scale: dict[str, list[str]] = {}
    baseline_by_scale: dict[str, list[str]] = {}
    for transformation in PLOT_TRANSFORMATIONS:
        decimals = spec.decimals if transformation == "level" else 2
        units_by_scale[transformation] = plot_units(spec, transformation)
        transformed_history = transform_path(history, spec, transformation)
        median, _lower, _upper = forecast_summary_on_scale(
            history, runtime.baseline, spec, transformation
        )
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
        ui.div(spec.group.upper(), class_="scenario-row-group"),
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

    for step, date in enumerate(runtime.baseline.dates):
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
            "data-series-group": spec.group,
            "data-series-search": (
                f"{spec.series_id} {spec.short_label} {spec.label} {spec.group}".lower()
            ),
            "data-units-by-scale": json.dumps(units_by_scale, separators=(",", ":")),
        }
    )
    return row


def scenario_modal(runtime, existing: ForecastResult | None) -> ui.Tag:
    existing_constraints = dict(existing.constraints) if existing else {}
    recent_dates = runtime.history.index[-3:]
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
                ui.span("HISTORY / EST.", class_="month-kind actual-kind"),
                ui.span(date.strftime("%b %y"), class_="month-label"),
                class_="scenario-month-header actual-month",
            )
        )
    for date in runtime.baseline.dates:
        header_cells.append(
            ui.div(
                ui.span("FORECAST", class_="month-kind forecast-kind"),
                ui.span(date.strftime("%b %y"), class_="month-label"),
                class_="scenario-month-header forecast-month",
            )
        )

    grid_cells = header_cells.copy()
    for spec in SERIES_SPECS:
        grid_cells.append(_scenario_row(runtime, spec, existing_constraints))

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
            ui.div(
                ui.tags.input(
                    id="scenario-variable-search",
                    type="search",
                    placeholder="Find a variable or FRED ID",
                    aria_label="Find a scenario variable",
                ),
                ui.tags.select(
                    ui.tags.option("All groups", value="All"),
                    *[ui.tags.option(group, value=group) for group in SERIES_GROUPS],
                    id="scenario-group-filter",
                    aria_label="Filter scenario variables by group",
                ),
                class_="scenario-filters",
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


def server(runtime, input: Inputs, output: Outputs, session: Session) -> None:
    scenario_state: reactive.Value[ForecastResult | None] = reactive.Value(None)
    session_id = token_hex(8)
    session_started_at = perf_counter()
    scenario_requested_at: float | None = None
    scenario_constraint_count = 0
    scenario_variables: list[str] = []
    runtime.telemetry_event("session_started", session_id=session_id)

    def _session_ended() -> None:
        runtime.telemetry_event(
            "session_ended",
            session_id=session_id,
            duration_ms=round((perf_counter() - session_started_at) * 1000),
        )

    session.on_ended(_session_ended)

    @reactive.extended_task
    async def _scenario_task(key: ConstraintKey) -> ForecastResult:
        async with runtime.scenario_semaphore:
            return await asyncio.to_thread(_cached_scenario_forecast, runtime.model, key)

    scenario_task = cast(Any, _scenario_task)

    @reactive.calc
    def _visible_specs() -> tuple[SeriesSpec, ...]:
        raw = input.visible_variables()
        selected = [raw] if isinstance(raw, str) else list(raw or ())
        valid_ids: list[str] = []
        for series_id in selected:
            if series_id in SERIES_BY_ID and series_id not in valid_ids:
                valid_ids.append(series_id)
            if len(valid_ids) == 8:
                break
        if not valid_ids:
            valid_ids = [DEFAULT_DASHBOARD_SERIES[0]]
        return tuple(SERIES_BY_ID[series_id] for series_id in valid_ids)

    @output
    @render.ui
    def variable_selection_summary() -> ui.Tag:
        count = len(_visible_specs())
        return ui.div(
            f"Showing {count} of {len(SERIES_SPECS)} variables",
            class_="variable-selection-summary",
            aria_live="polite",
        )

    @output
    @render.ui
    def chart_grid() -> ui.Tag:
        grid = ui.div(*chart_cards(runtime, _visible_specs()), class_="chart-grid")
        grid.attrs.update(
            {
                "id": "workspace-canvas",
                "aria-label": "Arrangeable forecast chart matrix",
            }
        )
        return grid

    @output
    @render.ui
    def scenario_progress() -> ui.Tag | None:
        if scenario_task.status() != "running":
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
        @output(id=f"chart_{chart_spec.series_id}", suspend_when_hidden=True)
        @render.ui
        def _chart():
            raw_transformation = input[f"plot_transform_{chart_spec.series_id}"]()
            transformation: PlotTransformation = (
                raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
            )
            options = json.dumps(
                echarts_options(
                    runtime.history,
                    runtime.baseline,
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
        table = forecast_gt(
            runtime.history, runtime.baseline, _visible_specs(), scenario_state.get()
        )
        return ui.HTML(table.as_raw_html())

    @reactive.effect
    @reactive.event(input.open_scenario)
    def _open_scenario() -> None:
        ui.modal_show(scenario_modal(runtime, scenario_state.get()))

    @reactive.effect
    @reactive.event(input.run_scenario)
    def _run_scenario() -> None:
        nonlocal scenario_constraint_count, scenario_requested_at, scenario_variables
        runtime.telemetry_event("scenario_run_clicked", session_id=session_id)
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
                                f"{spec.short_label}, {runtime.baseline.dates[step]:%b %Y} "
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
            runtime.telemetry_event(
                "scenario_requested",
                session_id=session_id,
                constraint_count=scenario_constraint_count,
                variables=scenario_variables,
            )
            ui.modal_remove()
            _ = _scenario_task(key)
        except ValueError as exc:
            runtime.telemetry_event(
                "scenario_rejected",
                session_id=session_id,
                reason="validation_error",
            )
            ui.notification_show(str(exc), type="error", duration=8)

    @reactive.effect
    def _handle_scenario_result() -> None:
        nonlocal scenario_requested_at
        task_status = scenario_task.status()
        if task_status == "success":
            scenario = scenario_task.result()
            if scenario_requested_at is not None:
                runtime.telemetry_event(
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
            error = scenario_task.error.get()
            if scenario_requested_at is not None:
                runtime.telemetry_event(
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
        ui.modal_show(scenario_modal(runtime, None))


async def _health(runtime, _request: Request) -> JSONResponse:
    cache = _cached_scenario_forecast.cache_info()
    return JSONResponse(
        {
            "status": "ok",
            "artifact_schema": runtime.artifact.schema_version,
            "release_id": runtime.release_id,
            "variable_count": len(runtime.model.variable_ids),
            "panel_end": runtime.artifact.panel_end.date().isoformat(),
            "forecast_end": cast(pd.Timestamp, runtime.baseline.dates[-1]).date().isoformat(),
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


class DashboardRuntime:
    """Own one validated model release and the Shiny application built from it.

    Published history is fixed and displayed as one posterior summary: early growth
    anchors use that history. Retained terminal states remain paired with their
    forecast parameter draws, so terminal-state uncertainty still affects dynamics.
    The runtime is intentionally injectable: tests can provide a synthetic artifact,
    while the production entrypoint supplies the published release loaded from disk.
    """

    def __init__(
        self,
        artifact: ForecastArtifact,
        *,
        release_id: str = "injected",
        static_root: Path = ROOT / "www",
        max_scenario_concurrency: int = MAX_SCENARIO_CONCURRENCY,
        telemetry=telemetry_event,
    ) -> None:
        self.artifact = artifact
        self.release_id = release_id
        self.model = artifact.model
        self.baseline = artifact.baseline
        if self.model.history_levels is None:
            raise RuntimeError("The production artifact does not contain model history.")
        self.history = self.model.history_levels
        self.forecast_interval_label = interval_label(self.baseline)
        self.scenario_semaphore = asyncio.Semaphore(max(1, max_scenario_concurrency))
        self.telemetry_event = telemetry
        self.static_root = static_root
        self.ui = build_ui(self)
        self.app = App(
            self.ui,
            self.server,
            static_assets=static_root,
            debug=False,
        )
        self.app.sanitize_errors = True
        self.app.starlette_app.routes.insert(
            0,
            Route("/healthz", endpoint=self.health, methods=["GET"]),
        )
        self.app.starlette_app.add_middleware(SecurityHeadersMiddleware)
        self.telemetry_event(
            "application_started",
            artifact_schema=artifact.schema_version,
            release_id=release_id,
            variable_count=len(self.model.variable_ids),
            panel_end=artifact.panel_end.date().isoformat(),
            posterior_draws=self.baseline.draws,
            scenario_cache_size=SCENARIO_CACHE_SIZE,
            scenario_concurrency=max(1, max_scenario_concurrency),
        )

    def server(self, input: Inputs, output: Outputs, session: Session) -> None:
        return server(self, input, output, session)

    async def health(self, request: Request) -> JSONResponse:
        return await _health(self, request)


def create_runtime(
    *,
    root: Path = ROOT,
    artifact: ForecastArtifact | None = None,
    release: PublishedRelease | None = None,
    max_scenario_concurrency: int = MAX_SCENARIO_CONCURRENCY,
    telemetry=telemetry_event,
) -> DashboardRuntime:
    """Create a runtime from an injected artifact or the published local release."""

    if artifact is not None and release is not None:
        raise ValueError("Pass either artifact or release, not both.")
    if release is None and artifact is None:
        release = load_published_release(root)
    if release is not None:
        artifact = release.artifact
        release_id = release.release_id
    else:
        release_id = "injected"
    assert artifact is not None
    return DashboardRuntime(
        artifact,
        release_id=release_id,
        static_root=root / "www",
        max_scenario_concurrency=max_scenario_concurrency,
        telemetry=telemetry,
    )


def create_app(**kwargs) -> App:
    """Build a Shiny app, primarily for embedding and HTTP tests."""

    return create_runtime(**kwargs).app
