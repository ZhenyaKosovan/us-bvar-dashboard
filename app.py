from __future__ import annotations

import asyncio
import json
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, reactive, render, ui

from us_bvar.artifact import load_artifact
from us_bvar.config import SERIES_SPECS
from us_bvar.model import BVAR, ForecastResult
from us_bvar.presentation import (
    PLOT_TRANSFORMATIONS,
    PlotTransformation,
    forecast_gt,
    highcharts_options,
)

ROOT = Path(__file__).parent
FORECAST_HORIZON = 12
POSTERIOR_DRAWS = 400
ARTIFACT_PATH = ROOT / "artifacts/bvar_forecast.pkl"


def chart_bootstrap() -> ui.Tag:
    return ui.tags.script(
        """
        (() => {
          const chartNodes = (root) => {
            const nodes = [];
            if (root.matches?.('.bvar-highchart')) nodes.push(root);
            root.querySelectorAll?.('.bvar-highchart').forEach((node) => nodes.push(node));
            return nodes;
          };

          const destroyCharts = (root) => {
            chartNodes(root).forEach((node) => {
              node.bvarChart?.destroy();
              node.bvarChart = null;
            });
          };

          const renderCharts = (root = document) => {
            chartNodes(root).forEach((node) => {
              const configNode = node.querySelector('script.chart-config');
              if (!configNode || typeof Highcharts === 'undefined') return;
              const signature = configNode.textContent;
              if (node.dataset.signature === signature) return;
              node.bvarChart?.destroy();
              node.bvarChart = Highcharts.chart(
                node.querySelector('.chart-target'),
                JSON.parse(signature),
              );
              node.dataset.signature = signature;
            });
          };
          document.addEventListener('DOMContentLoaded', () => {
            renderCharts();
            new MutationObserver((mutations) => {
              mutations.forEach((mutation) => {
                mutation.removedNodes.forEach((node) => {
                  if (node.nodeType === 1) destroyCharts(node);
                });
                mutation.addedNodes.forEach((node) => {
                  if (node.nodeType === 1) renderCharts(node);
                });
              });
            }).observe(document.body, {childList: true, subtree: true});
          });
        })();
        """
    )


app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="stylesheet", href="app.css"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap",
        ),
        ui.tags.script(src="vendor/highcharts/highcharts.js"),
        ui.tags.script(src="vendor/highcharts/highcharts-more.js"),
        chart_bootstrap(),
    ),
    ui.output_ui("scenario_progress"),
    ui.div(
        ui.div(
            ui.div("US MACRO LAB", class_="eyebrow"),
            ui.h1("Bayesian outlook", class_="app-title"),
            ui.p(
                "A compact monthly BVAR for exploring baseline projections "
                "and conditional macro scenarios.",
                class_="app-subtitle",
            ),
            class_="brand-block",
        ),
        ui.div(
            ui.div("PRECOMPUTED", class_="artifact-kicker"),
            ui.output_text("artifact_header"),
            class_="artifact-badge",
        ),
        class_="masthead",
    ),
    ui.div(
        ui.div(
            ui.div("MODEL STATUS", class_="status-label"),
            ui.output_text("status"),
            class_="status-copy",
        ),
        ui.div(
            ui.input_action_button(
                "open_scenario", "Build a scenario", class_="btn-scenario", disabled=True
            ),
            ui.input_action_button("clear_scenario", "Clear scenario", class_="btn-clear"),
            class_="scenario-actions",
        ),
        class_="status-bar",
    ),
    ui.p(
        "Choose a transformation independently on each chart. QoQ compares each month "
        "with three months earlier; rate charts use percentage-point changes.",
        class_="transform-guidance",
    ),
    ui.output_ui("charts"),
    ui.div(ui.output_ui("forecast_table"), class_="table-shell"),
    ui.div(
        "Model: 4 monthly lags · Minnesota prior · pandemic controls · "
        "16th–84th percentile posterior interval",
        class_="method-note",
    ),
)


def scenario_modal(
    model: BVAR,
    baseline: ForecastResult,
    existing: ForecastResult | None,
) -> ui.Tag:
    assert model.history_levels is not None
    recent = model.history_levels.tail(6)
    existing_constraints = dict(existing.constraints) if existing else {}
    variable_sections: list[ui.Tag] = []
    for spec in SERIES_SPECS:
        cells: list[ui.Tag] = []
        for date, value in recent[spec.series_id].items():
            cells.append(
                ui.div(
                    ui.span(date.strftime("%b %y"), class_="scenario-date"),
                    ui.span(f"{value:,.{spec.decimals}f}", class_="history-value"),
                    class_="scenario-cell historical-cell",
                )
            )
        for step, date in enumerate(baseline.dates):
            constraint = existing_constraints.get((step, spec.series_id))
            scenario_input = ui.input_text(
                f"sc_{spec.series_id}_{step}",
                None,
                value="" if constraint is None else str(constraint),
                placeholder=f"{baseline.median.loc[date, spec.series_id]:.{spec.decimals}f}",
            )
            scenario_input.children[1].attrs["aria-label"] = (
                f"{spec.label}, {date:%B %Y}, {spec.units}"
            )
            cells.append(
                ui.div(
                    ui.span(date.strftime("%b %y"), class_="scenario-date"),
                    scenario_input,
                    class_="scenario-cell forecast-cell",
                )
            )
        variable_sections.append(
            ui.div(
                ui.div(
                    ui.h4(spec.label),
                    ui.span(spec.units),
                    class_="scenario-variable-title",
                ),
                ui.div(*cells, class_="scenario-strip"),
                class_="scenario-variable",
            )
        )
    modal = ui.modal(
        ui.p(
            "Enter natural-unit values in any forecast cells. Blank cells remain "
            "unconstrained; placeholders show the baseline median.",
            class_="modal-instructions",
        ),
        *variable_sections,
        title="Conditional scenario path",
        size="xl",
        easy_close=True,
        footer=ui.div(
            ui.input_action_button("reset_modal", "Reset", class_="btn-clear"),
            ui.input_action_button(
                "run_scenario",
                "Run Conditional Analysis",
                class_="btn-scenario",
            ),
            class_="modal-actions",
        ),
    )
    modal.attrs["aria-label"] = "Conditional scenario path"
    modal.attrs["aria-modal"] = "true"
    modal.attrs["role"] = "dialog"
    return modal


def server(input: Inputs, output: Outputs, session: Session) -> None:
    model_state: reactive.Value[BVAR | None] = reactive.Value(None)
    baseline_state: reactive.Value[ForecastResult | None] = reactive.Value(None)
    scenario_state: reactive.Value[ForecastResult | None] = reactive.Value(None)
    status_state = reactive.Value("Loading the precomputed forecast …")
    artifact_header_state = reactive.Value("Forecast artifact")
    initial_attempted = reactive.Value(False)

    @reactive.extended_task
    async def _scenario_task(
        model: BVAR, constraints: dict[tuple[int, str], float]
    ) -> ForecastResult:
        return await asyncio.to_thread(
            model.forecast,
            horizon=FORECAST_HORIZON,
            draws=POSTERIOR_DRAWS,
            constraints=constraints,
            seed=202507,
        )

    @reactive.effect
    def _initial_load() -> None:
        if initial_attempted.get():
            return
        initial_attempted.set(True)
        try:
            artifact = load_artifact(ARTIFACT_PATH)
            model_state.set(artifact.model)
            baseline_state.set(artifact.baseline)
            status_state.set(
                f"Data through {artifact.panel_end:%B %Y} · "
                f"{artifact.observation_count:,} monthly observations · "
                f"{artifact.baseline.draws:,} baseline draws"
            )
            artifact_header_state.set(f"Built {artifact.created_at:%d %b %Y, %H:%M UTC}")
            ui.update_action_button("open_scenario", disabled=False)
        except Exception as exc:
            status_state.set(f"Could not load precomputed forecast: {exc}")

    @output
    @render.text
    def artifact_header() -> str:
        return artifact_header_state.get()

    @output
    @render.text
    def status() -> str:
        return status_state.get()

    @output
    @render.ui
    def scenario_progress() -> ui.Tag | None:
        if _scenario_task.status() != "running":
            return None
        return ui.div(
            ui.div(
                ui.div(class_="scenario-progress-spinner", aria_hidden="true"),
                ui.div("Running conditional analysis", class_="scenario-progress-title"),
                ui.p(
                    "Estimating the scenario forecast. Results will appear here automatically.",
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
    def charts() -> ui.Tag:
        model = model_state.get()
        baseline = baseline_state.get()
        if model is None or baseline is None or model.history_levels is None:
            return ui.div(
                ui.div("NO FORECAST YET", class_="empty-kicker"),
                ui.h2("Precompute the forecast artifact"),
                ui.p("Run `uv run python scripts/precompute.py`, then reload this page."),
                class_="empty-state",
            )
        cards: list[ui.Tag] = []
        for spec in SERIES_SPECS:
            cards.append(
                ui.div(
                    ui.div(
                        ui.h3(spec.label),
                        ui.div(
                            ui.input_select(
                                f"plot_transform_{spec.series_id}",
                                "Transformation",
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
        return ui.div(*cards, class_="chart-grid")

    def _register_chart(chart_spec) -> None:
        @output(id=f"chart_{chart_spec.series_id}", suspend_when_hidden=False)
        @render.ui
        def _chart() -> ui.Tag:
            model = model_state.get()
            baseline = baseline_state.get()
            if model is None or baseline is None or model.history_levels is None:
                return ui.div()

            raw_transformation = input[f"plot_transform_{chart_spec.series_id}"]()
            transformation: PlotTransformation = (
                raw_transformation if raw_transformation in PLOT_TRANSFORMATIONS else "level"
            )
            options = json.dumps(
                highcharts_options(
                    model.history_levels,
                    baseline,
                    chart_spec,
                    scenario_state.get(),
                    transformation,
                ),
                separators=(",", ":"),
            ).replace("</", "<\\/")
            return ui.HTML(
                '<div class="bvar-highchart">'
                '<div class="chart-target"></div>'
                f'<script type="application/json" class="chart-config">{options}</script>'
                "</div>"
            )

    for chart_spec in SERIES_SPECS:
        _register_chart(chart_spec)

    @output
    @render.ui
    def forecast_table() -> ui.Tag:
        model = model_state.get()
        baseline = baseline_state.get()
        if model is None or baseline is None or model.history_levels is None:
            return ui.div()
        table = forecast_gt(model.history_levels, baseline, SERIES_SPECS, scenario_state.get())
        return ui.HTML(table.as_raw_html())

    @reactive.effect
    @reactive.event(input.open_scenario)
    def _open_scenario() -> None:
        model = model_state.get()
        baseline = baseline_state.get()
        if model is None or baseline is None:
            return
        ui.modal_show(scenario_modal(model, baseline, scenario_state.get()))

    @reactive.effect
    @reactive.event(input.run_scenario)
    def _run_scenario() -> None:
        model = model_state.get()
        if model is None or baseline_state.get() is None:
            return
        constraints: dict[tuple[int, str], float] = {}
        try:
            for spec in SERIES_SPECS:
                for step in range(FORECAST_HORIZON):
                    raw = input[f"sc_{spec.series_id}_{step}"]()
                    if raw is not None and str(raw).strip():
                        constraints[(step, spec.series_id)] = float(str(raw).replace(",", ""))
            if not constraints:
                raise ValueError("Enter at least one scenario value.")
            ui.modal_remove()
            _scenario_task(model, constraints)
        except Exception as exc:
            ui.notification_show(str(exc), type="error", duration=8)

    @reactive.effect
    def _handle_scenario_result() -> None:
        task_status = _scenario_task.status()
        if task_status == "success":
            scenario = _scenario_task.result()
            scenario_state.set(scenario)
            ui.modal_remove()
            ui.notification_show(
                f"Scenario applied with {len(scenario.constraints)} constrained values.",
                type="message",
                duration=5,
            )
        elif task_status == "error":
            ui.notification_show(str(_scenario_task.error.get()), type="error", duration=8)

    @reactive.effect
    @reactive.event(input.clear_scenario)
    def _clear_scenario() -> None:
        scenario_state.set(None)

    @reactive.effect
    @reactive.event(input.reset_modal)
    def _reset_modal() -> None:
        model = model_state.get()
        baseline = baseline_state.get()
        if model is None or baseline is None:
            return
        ui.modal_remove()
        ui.modal_show(scenario_modal(model, baseline, None))


app = App(app_ui, server, static_assets=ROOT / "www")
