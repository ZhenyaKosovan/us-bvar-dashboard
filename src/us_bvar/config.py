from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConvergencePolicy:
    """Release policy for scalar and high-dimensional MCMC diagnostics.

    Aggregate quantiles make a large posterior practical to review, while the
    tail limits ensure that a small number of bad dimensions is still visible
    and bounded. Scalar diagnostics use the nominal thresholds directly.
    """

    nominal_r_hat: float = 1.10
    nominal_effective_sample_size: float = 20.0
    absolute_maximum_r_hat: float = 1.50
    absolute_minimum_effective_sample_size: float = 10.0
    maximum_tail_count: int = 25
    maximum_tail_fraction: float = 0.01


DEFAULT_CONVERGENCE_POLICY = ConvergencePolicy()

SeriesGroup = Literal["Activity", "Labor", "Prices", "Financial"]


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
    frequency: Literal["monthly", "quarterly"] = "monthly"
    group: SeriesGroup = "Activity"
    default_plot_transform: Literal["level", "mom", "qoq", "yoy"] = "level"


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(
        "GDPC1",
        "Real GDP (latent monthly estimate)",
        "Real GDP",
        "Billions of chained 2017 dollars, SAAR · monthly latent estimate",
        "log",
        1,
        "Billions · chained 2017 $, SAAR · estimated monthly",
        "$",
        "B",
        "quarterly",
        "Activity",
        "qoq",
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
        group="Activity",
        default_plot_transform="qoq",
    ),
    SeriesSpec(
        "CPIAUCSL",
        "Consumer Price Index",
        "CPI",
        "Index (1982–84=100)",
        "log",
        2,
        "Index · 1982–84=100",
        group="Prices",
        default_plot_transform="yoy",
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
        group="Labor",
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
        group="Financial",
    ),
    SeriesSpec(
        "INDPRO",
        "Industrial production",
        "Industrial production",
        "Index (2017=100)",
        "log",
        2,
        "Index · 2017=100",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "RRSFS",
        "Real retail and food services sales",
        "Real retail sales",
        "Millions of chained 2017 dollars, SA",
        "log",
        0,
        "Millions · chained 2017 $, SA",
        "$",
        "M",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "CMRMTSPL",
        "Real manufacturing and trade industries sales",
        "Real business sales",
        "Millions of chained 2017 dollars, SA",
        "log",
        0,
        "Millions · chained 2017 $, SA",
        "$",
        "M",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "HOUST",
        "Housing starts",
        "Housing starts",
        "Thousands of units, SAAR",
        "log",
        0,
        "Thousands · SAAR",
        value_suffix="K",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "PERMIT",
        "Building permits",
        "Building permits",
        "Thousands of units, SAAR",
        "log",
        0,
        "Thousands · SAAR",
        value_suffix="K",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "DGORDER",
        "Manufacturers' new orders: durable goods",
        "Durable goods orders",
        "Millions of dollars, SA",
        "log",
        0,
        "Millions $, SA",
        "$",
        "M",
        group="Activity",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "PAYEMS",
        "All employees: total nonfarm payrolls",
        "Nonfarm payrolls",
        "Thousands of persons, SA",
        "log",
        0,
        "Thousands of persons, SA",
        value_suffix="K",
        group="Labor",
        default_plot_transform="mom",
    ),
    SeriesSpec(
        "CIVPART",
        "Labor force participation rate",
        "Participation rate",
        "Percent",
        "level",
        2,
        "Percent",
        value_suffix="%",
        group="Labor",
    ),
    SeriesSpec(
        "AWHAETP",
        "Average weekly hours of all private employees",
        "Average weekly hours",
        "Hours per week, SA",
        "level",
        1,
        "Hours per week, SA",
        group="Labor",
    ),
    SeriesSpec(
        "CES0500000003",
        "Average hourly earnings of all private employees",
        "Average hourly earnings",
        "Dollars per hour, SA",
        "log",
        2,
        "Dollars per hour, SA",
        "$",
        group="Labor",
        default_plot_transform="yoy",
    ),
    SeriesSpec(
        "CPILFESL",
        "Consumer Price Index excluding food and energy",
        "Core CPI",
        "Index (1982–84=100)",
        "log",
        2,
        "Index · 1982–84=100",
        group="Prices",
        default_plot_transform="yoy",
    ),
    SeriesSpec(
        "PCEPI",
        "Personal consumption expenditures price index",
        "PCE price index",
        "Index (2017=100)",
        "log",
        2,
        "Index · 2017=100",
        group="Prices",
        default_plot_transform="yoy",
    ),
    SeriesSpec(
        "PCEPILFE",
        "PCE price index excluding food and energy",
        "Core PCE prices",
        "Index (2017=100)",
        "log",
        2,
        "Index · 2017=100",
        group="Prices",
        default_plot_transform="yoy",
    ),
    SeriesSpec(
        "PPIFIS",
        "Producer Price Index: final demand",
        "PPI final demand",
        "Index (November 2009=100)",
        "log",
        2,
        "Index · Nov 2009=100",
        group="Prices",
        default_plot_transform="yoy",
    ),
    SeriesSpec(
        "GS10",
        "10-year Treasury constant maturity rate",
        "10-year Treasury",
        "Percent",
        "level",
        2,
        "Percent",
        value_suffix="%",
        group="Financial",
    ),
    SeriesSpec(
        "BAA10Y",
        "Moody's Baa corporate bond yield relative to 10-year Treasury",
        "Baa credit spread",
        "Percentage points",
        "level",
        2,
        "Percentage points",
        value_suffix=" pp",
        group="Financial",
    ),
    SeriesSpec(
        "M2SL",
        "M2 money stock",
        "M2 money stock",
        "Billions of dollars, SA",
        "log",
        1,
        "Billions $, SA",
        "$",
        "B",
        group="Financial",
        default_plot_transform="yoy",
    ),
)

SERIES_BY_ID = {spec.series_id: spec for spec in SERIES_SPECS}
SERIES_GROUPS: tuple[SeriesGroup, ...] = ("Activity", "Labor", "Prices", "Financial")
DEFAULT_DASHBOARD_SERIES: tuple[str, ...] = (
    "GDPC1",
    "INDPRO",
    "CPIAUCSL",
    "UNRATE",
    "FEDFUNDS",
    "GS10",
)

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
