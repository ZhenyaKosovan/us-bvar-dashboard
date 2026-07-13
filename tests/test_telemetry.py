from __future__ import annotations

from us_bvar.telemetry import event


def test_telemetry_event_is_structured_and_omits_scenario_values() -> None:
    payload = event(
        "scenario_requested",
        session_id="session-token",
        constraint_count=2,
        variables=["CPIAUCSL", "FEDFUNDS"],
    )

    assert payload is not None
    assert payload["event"] == "scenario_requested"
    assert payload["service"] == "us-bvar-dashboard"
    assert payload["session_id"] == "session-token"
    assert payload["constraint_count"] == 2
    assert payload["variables"] == ["CPIAUCSL", "FEDFUNDS"]
    assert "scenario_values" not in payload
    assert payload["timestamp"].endswith("Z")


def test_telemetry_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("BVAR_TELEMETRY_ENABLED", "false")

    assert event("session_started", session_id="not-logged") is None
