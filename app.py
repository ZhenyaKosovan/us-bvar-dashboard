"""Production entrypoint for the US BVAR dashboard.

The application runtime lives in :mod:`us_bvar.dashboard`; this module remains the
stable ``shiny run app.py`` target used by local development and CML.
"""

from __future__ import annotations

from pathlib import Path

from us_bvar.dashboard import DashboardRuntime, create_app, create_runtime

ROOT = Path(__file__).resolve().parent
RUNTIME: DashboardRuntime = create_runtime(root=ROOT)
app = RUNTIME.app

# Compatibility aliases for local integrations that imported these entrypoint names.
app_ui = RUNTIME.ui
server = RUNTIME.server
ARTIFACT = RUNTIME.artifact
BASELINE = RUNTIME.baseline
HISTORY = RUNTIME.history

__all__ = [
    "ARTIFACT",
    "BASELINE",
    "DashboardRuntime",
    "HISTORY",
    "RUNTIME",
    "app",
    "app_ui",
    "create_app",
    "create_runtime",
    "server",
]
