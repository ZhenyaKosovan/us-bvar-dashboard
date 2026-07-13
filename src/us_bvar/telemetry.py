from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import TypeAlias

TelemetryValue: TypeAlias = str | int | float | bool | None | list[str]

_LOGGER = logging.getLogger("us_bvar.telemetry")
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False
if not _LOGGER.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)


def telemetry_enabled() -> bool:
    """Return whether structured telemetry is enabled for this process."""

    return os.getenv("BVAR_TELEMETRY_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def event(event_name: str, **fields: TelemetryValue) -> dict[str, TelemetryValue] | None:
    """Write one privacy-conscious JSON event to stdout for CML log collection."""

    if not telemetry_enabled():
        return None
    payload: dict[str, TelemetryValue] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "level": "INFO",
        "service": "us-bvar-dashboard",
        "event": event_name,
        "runtime_id": os.getenv("CDSW_ENGINE_ID", "local"),
        "process_id": os.getpid(),
    }
    payload.update(fields)
    _LOGGER.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return payload
