"""Cloudera Machine Learning application launcher.

Select this file as the CML Analytical Application script after running
``python scripts/setup_cml.py`` once in a Session or dependency-setup Job.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".cml-venv/bin/python"
VALID_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


def _application_port() -> int:
    raw_port = os.getenv("CDSW_APP_PORT") or os.getenv("PORT") or "8000"
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise SystemExit(f"Invalid CML application port: {raw_port!r}") from exc
    if not 1 <= port <= 65535:
        raise SystemExit(f"CML application port is outside 1..65535: {port}")
    return port


def main() -> None:
    if not VENV_PYTHON.is_file():
        raise SystemExit(
            "CML environment not found. Run `python scripts/setup_cml.py` in this project first."
        )
    required_files = (
        ROOT / "artifacts/bvar_forecast.pkl",
        ROOT / "artifacts/bvar_forecast.pkl.sha256",
        ROOT / "www/vendor/echarts/echarts.min.js",
    )
    missing = [str(path.relative_to(ROOT)) for path in required_files if not path.is_file()]
    if missing:
        raise SystemExit(f"Deployment assets are missing: {', '.join(missing)}")

    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("PYTHONHASHSEED", "0")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(ROOT / "src"), environment.get("PYTHONPATH", "")))
    )
    log_level = environment.get("BVAR_LOG_LEVEL", "info").lower()
    if log_level not in VALID_LOG_LEVELS:
        log_level = "info"

    arguments = [
        str(VENV_PYTHON),
        "-m",
        "shiny",
        "run",
        "--host",
        "127.0.0.1",
        "--port",
        str(_application_port()),
        "--ws-max-size",
        str(4 * 1024 * 1024),
        "--no-dev-mode",
        "--log-level",
        log_level,
        str(ROOT / "app.py"),
    ]
    os.chdir(ROOT)
    os.execve(VENV_PYTHON, arguments, environment)


if __name__ == "__main__":
    main()
