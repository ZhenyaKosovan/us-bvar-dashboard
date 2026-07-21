from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
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
from us_bvar.scenario_service import (  # pyright: ignore[reportMissingImports]
    ConstraintKey,
    ScenarioForecastService,
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
MAX_SCENARIO_NAME_LENGTH = 60
MAX_SCENARIO_CONSTRAINTS = 60
MAX_SCENARIO_VALUE = 1_000_000_000.0
SCENARIO_NUMBER_PATTERN = re.compile(
    r"^[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$"
)


@dataclass(frozen=True, slots=True)
class NamedScenario:
    """A session-local saved scenario with a stable UI identifier."""

    scenario_id: str
    name: str
    forecast: ForecastResult


def normalize_scenario_name(raw_name: object) -> str:
    """Return a compact display name or raise a user-facing validation error."""

    name = " ".join(str(raw_name or "").split())
    if not name:
        raise ValueError("Give this scenario a name.")
    if len(name) > MAX_SCENARIO_NAME_LENGTH:
        raise ValueError(f"Scenario names must be {MAX_SCENARIO_NAME_LENGTH} characters or fewer.")
    if not all(character.isprintable() for character in name):
        raise ValueError("Scenario names cannot contain control characters.")
    return name


def parse_scenario_value(raw_value: object) -> float:
    """Parse an unambiguous finite scenario number with optional grouped thousands."""

    text = str(raw_value).strip()
    if not SCENARIO_NUMBER_PATTERN.fullmatch(text):
        raise ValueError("Scenario values must be valid numbers.")
    try:
        value = float(text.replace(",", ""))
    except (OverflowError, ValueError) as exc:
        raise ValueError("Scenario values must be valid numbers.") from exc
    if not math.isfinite(value):
        raise ValueError("Scenario values must be finite.")
    if abs(value) > MAX_SCENARIO_VALUE:
        raise ValueError("Scenario values cannot exceed one billion in absolute magnitude.")
    return value


def default_scenario_name(
    scenarios: Mapping[str, NamedScenario], start_number: int
) -> tuple[str, int]:
    """Return the next available sequential scenario name and its number."""

    number = max(1, start_number)
    existing_names = {scenario.name.casefold() for scenario in scenarios.values()}
    while f"Scenario {number}".casefold() in existing_names:
        number += 1
    return f"Scenario {number}", number


def duplicate_scenario_name(scenarios: Mapping[str, NamedScenario], original_name: str) -> str:
    """Return a unique copy name that stays within the UI and server length limit."""

    existing_names = {scenario.name.casefold() for scenario in scenarios.values()}
    copy_number = 1
    while True:
        suffix = " copy" if copy_number == 1 else f" copy {copy_number}"
        prefix = original_name[: MAX_SCENARIO_NAME_LENGTH - len(suffix)].rstrip()
        candidate = f"{prefix}{suffix}"
        if candidate.casefold() not in existing_names:
            return candidate
        copy_number += 1


def validate_scenario_name(
    scenarios: Mapping[str, NamedScenario],
    raw_name: object,
    *,
    scenario_id: str | None = None,
) -> str:
    """Normalize a name and ensure another saved scenario does not use it."""

    normalized_name = normalize_scenario_name(raw_name)
    for existing_id, existing in scenarios.items():
        if existing_id != scenario_id and existing.name.casefold() == normalized_name.casefold():
            raise ValueError(f'A scenario named "{normalized_name}" already exists.')
    return normalized_name


def save_named_scenario(
    scenarios: Mapping[str, NamedScenario], scenario: NamedScenario
) -> dict[str, NamedScenario]:
    """Add or replace one scenario while enforcing case-insensitive unique names."""

    normalized_name = validate_scenario_name(
        scenarios, scenario.name, scenario_id=scenario.scenario_id
    )
    updated = dict(scenarios)
    updated[scenario.scenario_id] = NamedScenario(
        scenario_id=scenario.scenario_id,
        name=normalized_name,
        forecast=scenario.forecast,
    )
    return updated


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


def _constraint_key(constraints: dict[tuple[int, str], ScenarioConstraint]) -> ConstraintKey:
    return tuple(
        sorted(
            (step, series_id, constraint.value, constraint.transformation)
            for (step, series_id), constraint in constraints.items()
        )
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
    label: str,
    class_name: str,
    spec: SeriesSpec,
    *,
    size: str | None = None,
    disabled: bool = False,
) -> ui.Tag:
    button = ui.tags.button(
        label,
        type="button",
        class_=f"chart-menu-option {class_name}",
        role="menuitemradio" if size else "menuitem",
        disabled=disabled,
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
    for index, spec in enumerate(specs):
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
                _chart_menu_option("Move earlier", "chart-move-earlier", spec, disabled=index == 0),
                _chart_menu_option(
                    "Move later",
                    "chart-move-later",
                    spec,
                    disabled=index == len(specs) - 1,
                ),
                ui.div("CARD WIDTH", class_="chart-menu-label"),
                ui.div(
                    _chart_menu_option("Standard", "chart-size-option", spec, size="standard"),
                    _chart_menu_option("Wide", "chart-size-option", spec, size="wide"),
                    class_="chart-size-options",
                ),
                ui.div(class_="chart-menu-divider"),
                _chart_menu_option(
                    "Remove chart", "chart-remove-button", spec, disabled=len(specs) == 1
                ),
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
                            width="135px",
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
        card.attrs.update(
            {
                "data-series-id": spec.series_id,
                "data-card-size": "standard",
                "data-default-transform": spec.default_plot_transform,
            }
        )
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
        ui.tags.section(
            ui.div(
                ui.div(
                    ui.span(class_="status-pulse", aria_hidden="true"),
                    ui.div("MODEL READY", class_="status-label"),
                    class_="status-heading",
                ),
                ui.div(
                    ui.div(
                        ui.strong(f"{runtime.artifact.panel_end:%b %Y}"),
                        ui.span("data through"),
                        class_="model-stat model-vintage",
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
                ui.tags.nav(
                    ui.tags.a("Series", href="#analysis-workspace"),
                    ui.tags.a("Forecast data", href="#forecast-data"),
                    class_="section-navigation",
                    aria_label="Dashboard sections",
                ),
                ui.div(
                    ui.output_ui("scenario_summary"),
                    ui.input_action_button("open_scenario", "New scenario", class_="btn-scenario"),
                    ui.input_action_button(
                        "edit_scenario", "Edit", class_="btn-clear", disabled=True
                    ),
                    ui.input_action_button(
                        "duplicate_scenario", "Duplicate", class_="btn-clear", disabled=True
                    ),
                    ui.input_action_button(
                        "delete_scenario", "Delete", class_="btn-clear", disabled=True
                    ),
                    class_="scenario-actions",
                ),
                class_="status-controls",
            ),
            class_="status-bar",
        ),
        (
            ui.div(
                ui.strong("MODEL REFRESH REQUIRED"),
                ui.span(runtime.release_warning),
                class_="release-warning",
                role="alert",
            )
            if runtime.release_warning
            else None
        ),
        ui.tags.section(
            ui.div(
                ui.span("QUICK START", class_="preset-label"),
                ui.div(
                    *_preset_buttons(),
                    ui.tags.button(
                        "Reset",
                        type="button",
                        class_="workspace-preset workspace-reset",
                        data_series_ids=json.dumps(DEFAULT_DASHBOARD_SERIES),
                        title="Restore the default charts, order, widths, and transformations",
                    ),
                    class_="workspace-presets",
                ),
                class_="preset-group workspace-quick-start",
            ),
            ui.div(
                ui.tags.aside(
                    ui.div(
                        ui.div("EXPLORE SERIES", class_="guidance-label"),
                        ui.output_ui("variable_selection_summary"),
                        class_="library-heading",
                    ),
                    ui.tags.label(
                        "Search and add indicators",
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
                            ui.span("SELECTED SERIES", class_="canvas-kicker"),
                            ui.span("Drag to rearrange", class_="canvas-hint"),
                        ),
                        ui.div(
                            "Workspace changes are saved in this browser.",
                            class_="canvas-save-status",
                        ),
                        class_="canvas-header",
                    ),
                    ui.output_ui("workspace_legend"),
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
                id="workspace-feedback",
                class_="workspace-feedback",
                aria_live="polite",
                aria_atomic="true",
            ),
            class_="workspace",
            id="analysis-workspace",
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
            id="forecast-data",
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
        baseline_by_scale[transformation] = [f"{value:,.{decimals}f}" for value in median]

    transformation_input = ui.input_select(
        f"sc_transform_{spec.series_id}",
        f"Transformation for {spec.label}",
        choices={key: value.label for key, value in PLOT_TRANSFORMATIONS.items()},
        selected=selected,
        width="100%",
    )
    transformation_shell = cast(ui.Tag, transformation_input.children[1])
    transformation_shell.attrs["class"] = "scenario-transform-shell"
    transformation_select = cast(ui.Tag, transformation_shell.children[0])
    transformation_select.attrs.update(
        {
            "class": "shiny-input-select form-select scenario-transform",
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
        ui.div(
            ui.tags.button(
                "Hold →",
                type="button",
                class_="scenario-row-action",
                data_row_action="hold",
                title="Hold the first entered value through later months",
                aria_label=f"Hold {spec.short_label} from first entered value",
            ),
            ui.tags.button(
                "Ramp ↗",
                type="button",
                class_="scenario-row-action",
                data_row_action="interpolate",
                title="Interpolate a straight path between two entered endpoints",
                aria_label=f"Interpolate {spec.short_label} endpoints",
            ),
            ui.tags.button(
                "Clear ×",
                type="button",
                class_="scenario-row-action scenario-row-clear",
                data_row_action="clear",
                title="Clear all assumptions in this row",
                aria_label=f"Clear {spec.short_label} assumptions",
            ),
            class_="scenario-row-tools",
            aria_label=f"Path tools for {spec.short_label}",
        ),
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
                "data-series-short-label": spec.short_label,
                "data-step": str(step),
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
            "data-model-transform": spec.transform,
        }
    )
    return row


def scenario_modal(
    runtime,
    existing: ForecastResult | None,
    *,
    scenario_name: str = "Scenario 1",
    editing: bool = False,
) -> ui.Tag:
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
                    "Enter only the values you want to constrain, up to "
                    f"{MAX_SCENARIO_CONSTRAINTS} assumptions. Empty forecast cells stay "
                    "model-driven; muted placeholders are the current baseline medians.",
                    class_="modal-instructions",
                ),
                ui.div(
                    ui.div(
                        f"{len(existing_constraints)} constraints",
                        class_="constraint-count",
                        aria_live="polite",
                    ),
                    ui.tags.button(
                        "Show assumptions only",
                        type="button",
                        class_="btn-clear scenario-selected-only",
                        aria_pressed="false",
                        disabled=not existing_constraints,
                    ),
                    ui.tags.button(
                        "Exit editor",
                        type="button",
                        class_="btn-clear scenario-exit",
                        title="Close the scenario editor (Esc)",
                    ),
                    class_="scenario-intro-actions",
                ),
                class_="scenario-intro",
            ),
            ui.div(
                ui.tags.label(
                    "Scenario name",
                    ui.tags.input(
                        id="scenario_name",
                        type="text",
                        value=scenario_name,
                        maxlength="60",
                        placeholder="Scenario name",
                        autocomplete="off",
                        class_="shiny-input-text form-control",
                        required=True,
                    ),
                    class_="scenario-name-field",
                ),
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
            ui.div(
                ui.div(
                    ui.span("STARTER PATHS", class_="scenario-starter-label"),
                    ui.tags.button(
                        "Policy rate +1 pp",
                        type="button",
                        class_="scenario-starter",
                        data_series_id="FEDFUNDS",
                        data_transformation="level",
                        data_adjustment="1",
                        data_scenario_name="Policy rate +1 pp",
                    ),
                    ui.tags.button(
                        "CPI inflation −1 pp",
                        type="button",
                        class_="scenario-starter",
                        data_series_id="CPIAUCSL",
                        data_transformation="yoy",
                        data_adjustment="-1",
                        data_scenario_name="Faster disinflation",
                    ),
                    class_="scenario-starters",
                ),
                ui.span(
                    "Starter paths adjust all 12 baseline months and can be edited.",
                    class_="scenario-starter-help",
                ),
                class_="scenario-starter-bar",
            ),
            ui.div(
                ui.div(
                    "No assumptions entered yet.",
                    class_="scenario-assumption-empty",
                ),
                ui.div(class_="scenario-assumption-list"),
                class_="scenario-assumption-summary",
                aria_live="polite",
                aria_label="Entered scenario assumptions",
            ),
            ui.output_ui("scenario_validation", class_="scenario-validation-output"),
            ui.div(ui.div(*grid_cells, class_="scenario-grid"), class_="scenario-grid-scroll"),
            ui.tags.dialog(
                ui.h3("Change transformation?"),
                ui.p(id="scenario-transform-dialog-copy"),
                ui.div(
                    ui.tags.button(
                        "Keep and reinterpret",
                        type="button",
                        class_="btn-scenario",
                        data_transform_action="keep",
                    ),
                    ui.tags.button(
                        "Clear entered values",
                        type="button",
                        class_="btn-clear",
                        data_transform_action="clear",
                    ),
                    ui.tags.button(
                        "Cancel",
                        type="button",
                        class_="btn-clear",
                        data_transform_action="cancel",
                    ),
                    class_="scenario-transform-dialog-actions",
                ),
                id="scenario-transform-dialog",
                class_="scenario-transform-dialog",
            ),
            class_="scenario-modal",
            data_max_constraints=str(MAX_SCENARIO_CONSTRAINTS),
        ),
        title="Edit scenario" if existing or editing else "Create a scenario",
        size="xl",
        easy_close=False,
        footer=ui.div(
            ui.div(
                ui.span("Tip", class_="footer-tip-label"),
                " Start with one or two assumptions; add detail only where it matters.",
                class_="modal-tip",
            ),
            ui.div(
                ui.input_action_button("reset_modal", "Reset fields", class_="btn-clear"),
                ui.div(
                    "Enter at least one valid assumption.",
                    class_="scenario-action-help",
                    id="scenario-action-help",
                    aria_live="polite",
                ),
                ui.input_action_button(
                    "run_scenario",
                    "Update scenario" if existing or editing else "Calculate scenario",
                    class_="btn-scenario",
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


def scenario_switcher(
    scenarios: dict[str, NamedScenario],
    active_scenario_id: str | None,
    comparison_scenario_id: str | None = None,
) -> ui.Tag:
    if not scenarios:
        return ui.div(ui.span(class_="scenario-dot"), "Baseline", class_="scenario-chip")

    choices = {"": "Baseline"}
    choices.update({scenario_id: scenario.name for scenario_id, scenario in scenarios.items()})
    saved_label = (
        "1 scenario in this session"
        if len(scenarios) == 1
        else f"{len(scenarios)} scenarios in this session"
    )
    active = scenarios.get(active_scenario_id or "")
    comparison_choices = {
        scenario_id: scenario.name
        for scenario_id, scenario in scenarios.items()
        if scenario_id != active_scenario_id
    }
    export_payload = None
    if active is not None:
        export_payload = json.dumps(
            {
                "schema_version": 1,
                "name": active.name,
                "constraints": [
                    {
                        "date": str(active.forecast.dates[step])[:10],
                        "series_id": series_id,
                        "value": constraint.value,
                        "transformation": constraint.transformation,
                    }
                    for (step, series_id), constraint in sorted(active.forecast.constraints.items())
                ],
            },
            separators=(",", ":"),
        )
    return ui.div(
        ui.input_select(
            "active_scenario",
            None,
            choices=choices,
            selected=active_scenario_id or "",
        ),
        ui.span(
            saved_label,
            class_="scenario-saved-count",
            title="Scenarios reset when this session ends",
        ),
        (
            ui.input_select(
                "comparison_scenario",
                None,
                choices={"": "Compare with…", **comparison_choices},
                selected=(
                    comparison_scenario_id if comparison_scenario_id in comparison_choices else ""
                ),
            )
            if active is not None and comparison_choices
            else None
        ),
        (
            ui.tags.button(
                "Export",
                type="button",
                class_="scenario-export",
                data_scenario_export=export_payload,
                data_scenario_name=active.name,
                title="Download this scenario's assumptions as JSON",
            )
            if active is not None
            else None
        ),
        class_="scenario-switcher",
    )


def server(runtime, input: Inputs, output: Outputs, session: Session) -> None:
    scenarios_state: reactive.Value[dict[str, NamedScenario]] = reactive.Value({})
    active_scenario_id: reactive.Value[str | None] = reactive.Value(None)
    comparison_scenario_id: reactive.Value[str | None] = reactive.Value(None)
    scenario_validation_message: reactive.Value[str] = reactive.Value("")
    session_id = token_hex(8)
    session_started_at = perf_counter()
    scenario_requested_at: float | None = None
    scenario_constraint_count = 0
    scenario_variables: list[str] = []
    editor_scenario_id: str | None = None
    pending_scenario_id: str | None = None
    pending_scenario_name: str | None = None
    pending_creates_scenario = False
    next_scenario_number = 1
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
        cached = runtime.scenario_forecasts.get_cached(key)
        if cached is not None:
            return cached
        return await asyncio.to_thread(runtime.scenario_forecasts.forecast, key)

    scenario_task = cast(Any, _scenario_task)

    def _active_named_scenario() -> NamedScenario | None:
        scenario_id = active_scenario_id.get()
        return scenarios_state.get().get(scenario_id) if scenario_id else None

    def _comparison_named_scenario() -> NamedScenario | None:
        scenario_id = comparison_scenario_id.get()
        return scenarios_state.get().get(scenario_id) if scenario_id else None

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
        return scenario_switcher(
            scenarios_state.get(),
            active_scenario_id.get(),
            comparison_scenario_id.get(),
        )

    @output
    @render.ui
    def scenario_validation() -> ui.Tag:
        message = scenario_validation_message.get()
        return ui.div(
            message,
            class_="scenario-validation-message",
            role="alert",
            aria_live="assertive",
            hidden=not message,
        )

    @reactive.effect
    @reactive.event(input.active_scenario)
    def _select_scenario() -> None:
        selected = str(input.active_scenario() or "")
        active_scenario_id.set(selected if selected in scenarios_state.get() else None)
        if selected == comparison_scenario_id.get() or not selected:
            comparison_scenario_id.set(None)

    @reactive.effect
    @reactive.event(input.comparison_scenario)
    def _select_comparison_scenario() -> None:
        selected = str(input.comparison_scenario() or "")
        valid = selected in scenarios_state.get() and selected != active_scenario_id.get()
        comparison_scenario_id.set(selected if valid else None)

    @reactive.effect
    def _sync_scenario_actions() -> None:
        has_active_scenario = _active_named_scenario() is not None
        ui.update_action_button("edit_scenario", disabled=not has_active_scenario)
        ui.update_action_button("duplicate_scenario", disabled=not has_active_scenario)
        ui.update_action_button("delete_scenario", disabled=not has_active_scenario)

    @output
    @render.ui
    def workspace_legend() -> ui.Tag:
        items = [
            ("history", "History / estimate"),
            ("baseline", "Baseline median"),
            ("baseline-band", "Baseline interval"),
        ]
        active = _active_named_scenario()
        comparison = _comparison_named_scenario()
        if active is not None:
            items.extend(
                [
                    ("scenario", f"{active.name} median"),
                    ("scenario-band", f"{active.name} interval"),
                ]
            )
        if comparison is not None:
            items.extend(
                [
                    ("comparison", f"{comparison.name} median"),
                    ("comparison-band", f"{comparison.name} interval"),
                ]
            )
        return ui.div(
            *[
                ui.span(
                    ui.span(class_=f"workspace-legend-mark legend-{kind}"),
                    label,
                    class_="workspace-legend-item",
                )
                for kind, label in items
            ],
            class_="workspace-legend",
            aria_label="Chart legend",
        )

    def _register_chart(chart_spec: SeriesSpec) -> None:
        @output(id=f"chart_{chart_spec.series_id}", suspend_when_hidden=True)
        @render.ui
        def _chart():
            active = _active_named_scenario()
            comparison = _comparison_named_scenario()
            raw_transformation = input[f"plot_transform_{chart_spec.series_id}"]()
            transformation: PlotTransformation = (
                raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
            )
            options = json.dumps(
                echarts_options(
                    runtime.history,
                    runtime.baseline,
                    chart_spec,
                    active.forecast if active else None,
                    transformation,
                    scenario_name=active.name if active else None,
                    comparison=comparison.forecast if comparison else None,
                    comparison_name=comparison.name if comparison else None,
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
        active = _active_named_scenario()
        comparison = _comparison_named_scenario()
        table = forecast_gt(
            runtime.history,
            runtime.baseline,
            _visible_specs(),
            active.forecast if active else None,
            scenario_name=active.name if active else None,
            comparison=comparison.forecast if comparison else None,
            comparison_name=comparison.name if comparison else None,
        )
        return ui.HTML(table.as_raw_html())

    @reactive.effect
    @reactive.event(input.open_scenario)
    def _open_scenario() -> None:
        nonlocal editor_scenario_id, next_scenario_number
        editor_scenario_id = None
        scenario_validation_message.set("")
        scenario_name, next_scenario_number = default_scenario_name(
            scenarios_state.get(), next_scenario_number
        )
        ui.modal_show(scenario_modal(runtime, None, scenario_name=scenario_name))

    @reactive.effect
    @reactive.event(input.edit_scenario)
    def _edit_scenario() -> None:
        nonlocal editor_scenario_id
        active = _active_named_scenario()
        if active is None:
            return
        editor_scenario_id = active.scenario_id
        scenario_validation_message.set("")
        ui.modal_show(scenario_modal(runtime, active.forecast, scenario_name=active.name))

    @reactive.effect
    @reactive.event(input.duplicate_scenario)
    def _duplicate_scenario() -> None:
        nonlocal editor_scenario_id
        active = _active_named_scenario()
        if active is None:
            return
        editor_scenario_id = None
        scenario_validation_message.set("")
        candidate = duplicate_scenario_name(scenarios_state.get(), active.name)
        ui.modal_show(scenario_modal(runtime, active.forecast, scenario_name=candidate))

    @reactive.effect
    @reactive.event(input.run_scenario_request)
    def _run_scenario() -> None:
        nonlocal pending_creates_scenario, pending_scenario_id, pending_scenario_name
        nonlocal scenario_constraint_count, scenario_requested_at, scenario_variables
        runtime.telemetry_event("scenario_run_clicked", session_id=session_id)
        scenario_validation_message.set("")
        if pending_scenario_id is not None:
            message = "A scenario calculation is already in progress."
            runtime.telemetry_event(
                "scenario_rejected",
                session_id=session_id,
                reason="scenario_in_progress",
            )
            scenario_validation_message.set(message)
            ui.notification_show(message, type="warning", duration=5)
            return
        pending_creates_scenario = False
        constraints: dict[tuple[int, str], ScenarioConstraint] = {}
        try:
            scenario_name = validate_scenario_name(
                scenarios_state.get(),
                input.scenario_name(),
                scenario_id=editor_scenario_id,
            )
            for spec in SERIES_SPECS:
                raw_transformation = input[f"sc_transform_{spec.series_id}"]()
                transformation: PlotTransformation = (
                    raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
                )
                for step in range(FORECAST_HORIZON):
                    raw = input[f"sc_{spec.series_id}_{step}"]()
                    if raw is not None and str(raw).strip():
                        try:
                            value = parse_scenario_value(raw)
                        except ValueError as exc:
                            raise ValueError(
                                f"{spec.short_label}, {runtime.baseline.dates[step]:%b %Y}: {exc}"
                            ) from exc
                        constraints[(step, spec.series_id)] = ScenarioConstraint(
                            value=value,
                            transformation=transformation,
                        )
            if not constraints:
                raise ValueError("Enter at least one scenario value.")
            if len(constraints) > MAX_SCENARIO_CONSTRAINTS:
                raise ValueError(
                    f"Use at most {MAX_SCENARIO_CONSTRAINTS} scenario assumptions per run."
                )
            key = _constraint_key(constraints)
            pending_scenario_id = editor_scenario_id or token_hex(6)
            pending_scenario_name = scenario_name
            pending_creates_scenario = editor_scenario_id is None
            scenario_requested_at = perf_counter()
            scenario_constraint_count = len(constraints)
            scenario_variables = sorted({series_id for _step, series_id in constraints})
            runtime.telemetry_event(
                "scenario_requested",
                session_id=session_id,
                constraint_count=scenario_constraint_count,
                variables=scenario_variables,
            )
            _ = _scenario_task(key)
        except ValueError as exc:
            runtime.telemetry_event(
                "scenario_rejected",
                session_id=session_id,
                reason="validation_error",
            )
            scenario_validation_message.set(str(exc))
            ui.notification_show(str(exc), type="error", duration=8)

    @reactive.effect
    def _handle_scenario_result() -> None:
        nonlocal next_scenario_number, pending_creates_scenario
        nonlocal pending_scenario_id, pending_scenario_name, scenario_requested_at
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
            if pending_scenario_id is None or pending_scenario_name is None:
                logger.error("Scenario calculation completed without pending scenario metadata")
                scenario_validation_message.set("Scenario could not be saved.")
                ui.notification_show("Scenario could not be saved.", type="error", duration=8)
                return
            named_scenario = NamedScenario(
                scenario_id=pending_scenario_id,
                name=pending_scenario_name,
                forecast=scenario,
            )
            with reactive.isolate():
                saved_scenarios = save_named_scenario(scenarios_state.get(), named_scenario)
            scenarios_state.set(saved_scenarios)
            active_scenario_id.set(named_scenario.scenario_id)
            if pending_creates_scenario:
                next_scenario_number += 1
            pending_creates_scenario = False
            pending_scenario_id = None
            pending_scenario_name = None
            ui.modal_remove()
            ui.notification_show(
                f'"{named_scenario.name}" saved with '
                f"{len(scenario.constraints)} constrained values.",
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
            pending_creates_scenario = False
            pending_scenario_id = None
            pending_scenario_name = None
            logger.error(
                "Scenario calculation failed: %s",
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
            message = (
                str(error) if isinstance(error, ValueError) else "Scenario calculation failed."
            )
            scenario_validation_message.set(message)
            ui.notification_show(message, type="error", duration=8)

    @reactive.effect
    @reactive.event(input.delete_scenario)
    def _delete_scenario() -> None:
        scenario_id = active_scenario_id.get()
        if scenario_id is None:
            return
        scenarios = dict(scenarios_state.get())
        deleted = scenarios.pop(scenario_id, None)
        scenarios_state.set(scenarios)
        active_scenario_id.set(None)
        comparison_scenario_id.set(None)
        if deleted is not None:
            ui.notification_show(f'"{deleted.name}" deleted.', type="message", duration=4)

    @reactive.effect
    @reactive.event(input.reset_modal_request)
    def _reset_modal() -> None:
        scenario_name = str(input.scenario_name() or "")
        scenario_validation_message.set("")
        ui.modal_remove()
        ui.modal_show(
            scenario_modal(
                runtime,
                None,
                scenario_name=scenario_name,
                editing=editor_scenario_id is not None,
            )
        )


async def _health(runtime, _request: Request) -> JSONResponse:
    cache = runtime.scenario_forecasts.cache_info()
    return JSONResponse(
        {
            "status": "ok",
            "artifact_schema": runtime.artifact.schema_version,
            "release_id": runtime.release_id,
            "data_status": "refresh_required" if runtime.release_warning else "current",
            "release_warning": runtime.release_warning,
            "variable_count": len(runtime.model.variable_ids),
            "panel_end": runtime.artifact.panel_end.date().isoformat(),
            "forecast_end": cast(pd.Timestamp, runtime.baseline.dates[-1]).date().isoformat(),
            "scenario_cache": {
                "size": cache.size,
                "max_size": cache.max_size,
                "in_flight": cache.in_flight,
            },
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
        release_warning: str | None = None,
        static_root: Path = ROOT / "www",
        max_scenario_concurrency: int = MAX_SCENARIO_CONCURRENCY,
        telemetry=telemetry_event,
    ) -> None:
        self.artifact = artifact
        self.release_id = release_id
        self.release_warning = release_warning
        self.model = artifact.model
        self.baseline = artifact.baseline
        if self.model.history_levels is None:
            raise RuntimeError("The production artifact does not contain model history.")
        self.history = self.model.history_levels
        self.forecast_interval_label = interval_label(self.baseline)
        self.model.prepare_forecast_cache(FORECAST_HORIZON)
        self.scenario_forecasts = ScenarioForecastService(
            self.model,
            horizon=FORECAST_HORIZON,
            draws=POSTERIOR_DRAWS,
            seed=202507,
            max_size=SCENARIO_CACHE_SIZE,
            max_concurrency=max(1, max_scenario_concurrency),
        )
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
            refresh_required=release_warning is not None,
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
    release_warning: str | None = None
    if release is not None:
        artifact = release.artifact
        release_id = release.release_id
        if release.metadata is not None:
            raw_warning = release.metadata.get("refresh_required_reason")
            if isinstance(raw_warning, str) and raw_warning.strip():
                release_warning = raw_warning.strip()[:500]
    else:
        release_id = "injected"
    if artifact is None:
        raise RuntimeError("No forecast artifact was loaded.")
    return DashboardRuntime(
        artifact,
        release_id=release_id,
        release_warning=release_warning,
        static_root=root / "www",
        max_scenario_concurrency=max_scenario_concurrency,
        telemetry=telemetry,
    )


def create_app(**kwargs) -> App:
    """Build a Shiny app, primarily for embedding and HTTP tests."""

    return create_runtime(**kwargs).app
