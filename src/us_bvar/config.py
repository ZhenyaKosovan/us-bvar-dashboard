from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SeriesSpec:
    """Metadata and model transformation for one FRED series."""

    series_id: str
    label: str
    short_label: str
    units: str
    transform: Literal["log", "level"]
    decimals: int
    table_units: str
    value_prefix: str = ""
    value_suffix: str = ""


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        "INDPRO",
        "Industrial production",
        "Industrial production",
        "Index (2017=100)",
        "log",
        2,
        "Index · 2017=100",
    ),
    SeriesSpec(
        "PCEC96",
        "Real personal consumption expenditures",
        "Real PCE",
        "Billions of chained 2017 dollars, SAAR",
        "log",
        1,
        "Billions · chained 2017 $, SAAR",
        "$",
        "B",
    ),
    SeriesSpec(
        "CPIAUCSL",
        "Consumer Price Index",
        "CPI",
        "Index (1982–84=100)",
        "log",
        2,
        "Index · 1982–84=100",
    ),
    SeriesSpec(
        "UNRATE",
        "Unemployment rate",
        "Unemployment",
        "Percent",
        "level",
        2,
        "Percent",
        value_suffix="%",
    ),
    SeriesSpec(
        "FEDFUNDS",
        "Effective federal funds rate",
        "Fed funds rate",
        "Percent",
        "level",
        2,
        "Percent",
        value_suffix="%",
    ),
)

SERIES_BY_ID = {spec.series_id: spec for spec in SERIES_SPECS}

# Separate controls absorb the extraordinary monthly observations highlighted in
# Cascaldi-Garcia (2022), while being zero throughout the forecast horizon.
PANDEMIC_CONTROL_MONTHS = (
    "2020-03-01",
    "2020-04-01",
    "2020-05-01",
    "2020-06-01",
    "2020-07-01",
    "2020-08-01",
)
