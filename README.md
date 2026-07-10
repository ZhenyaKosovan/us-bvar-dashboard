# US Bayesian VAR scenario dashboard

A Shiny for Python application for a compact monthly US macro BVAR. A separate precompute
command downloads a balanced panel from the FRED API, estimates a four-lag
Minnesota-prior VAR, and saves a 12-month baseline forecast. The dashboard only loads that
artifact and lets a user condition the estimated posterior on paths for one or more variables.

## Model specification

The requested observation window begins in January 1985 and is truncated to the dates
observed for all five series. The current balanced panel begins in January 2007 because that
is the first common month available from these exact FRED series:

| FRED series | Model variable | Model scale | Dashboard scale |
|---|---|---|---|
| `INDPRO` | Industrial production | standardized log level | index, 2017=100 |
| `PCEC96` | Real personal consumption expenditures | standardized log level | billions chained 2017 dollars, SAAR |
| `CPIAUCSL` | CPI, all items | standardized log level | index, 1982–84=100 |
| `UNRATE` | Unemployment rate | standardized level | percent |
| `FEDFUNDS` | Effective federal funds rate | standardized level | percent |

Real GDP is quarterly, so it cannot provide twelve genuinely monthly observations in the
requested interface without a mixed-frequency model or interpolation. Industrial production
is the standard monthly output proxy; real PCE adds broad monthly household demand. The
four core series—industrial production, CPI, unemployment, and the federal funds rate—match
the monthly forecast variables used in the New York Fed forecasting literature.

The VAR uses four monthly lags. First own lags have random-walk prior means; other prior
means are zero. Prior standard deviations use overall tightness `0.20`, cross-variable
tightness `0.50`, and harmonic lag decay. Six separate March–August 2020 controls absorb
the exceptional pandemic observations and are zero in forecasts. Coefficient uncertainty is
normal and innovation covariance uncertainty is inverse-Wishart. Reported bands are pointwise
16th–84th percentiles across posterior predictive draws. The dashboard table shows those
intervals beneath each forecast median in the variable's natural units.

Relevant primary sources:

- The New York Fed's [A Large Bayesian VAR of the United States Economy](https://www.newyorkfed.org/research/staff_reports/sr976) motivates shrinkage BVARs and conditional counterfactual scenarios.
- The New York Fed paper [Forecasting with Bayesian Vector Autoregressions with Time Variation in the Mean](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr327.pdf) uses monthly CPI inflation, industrial production, unemployment, and the effective federal funds rate as its four forecast variables.
- The Federal Reserve's [Averaging Forecasts from VARs with Uncertain Instabilities](https://www.federalreserve.gov/pubs/feds/2007/200742/index.html) documents a four-lag BVAR and the conventional Minnesota settings `0.20`, `0.50`, and lag decay `1`.
- The Federal Reserve's [Pandemic Priors](https://www.federalreserve.gov/econres/ifdp/pandemic-priors.htm) shows why a few extreme pandemic observations can distort persistence and forecasts, and motivates time controls in the Minnesota-prior setup.
- FRED's official [`series/observations` API documentation](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) describes the data endpoint and required API key.

This is a deliberately small reduced-form forecasting model. Its scenarios are conditional
projections, not identified causal interventions, and its output is not a Federal Reserve
forecast.

## Conditional scenarios

For every posterior parameter draw, the code constructs the joint Gaussian distribution of
all five variables over all twelve future months. User-entered cells are exact linear
conditions on that 60-dimensional path. Standard Gaussian conditioning then updates every
unconstrained variable and horizon. This matters: a CPI or policy-rate path affects the whole
joint projection instead of merely replacing values after forecasting.

The modal shows six actual months and twelve forecast months for every variable. Blank cells
are unconstrained and their placeholders show baseline medians. Values are entered in the
natural units shown beside each variable.

## Precompute, then run

Install [uv](https://docs.astral.sh/uv/), obtain a free FRED API key, and create `.env`:

```bash
cd /home/zhenya/local-repos/us-bvar-dashboard
cp .env.example .env
# Edit .env and set FRED_API_KEY.
uv sync
uv run python scripts/precompute.py
```

Precompute writes:

- `data/fred_panel.csv`: the balanced downloaded panel;
- `artifacts/bvar_forecast.pkl`: the fitted posterior and baseline forecast consumed by Shiny;
- `artifacts/metadata.json`: human-readable vintage, sample, horizon, and draw metadata.

Successful individual FRED downloads are also cached under `data/cache/`. To rebuild from
that cache without a network call, run `uv run python scripts/precompute.py --offline`.

Start the dashboard separately:

```bash
uv run shiny run --reload app.py
```

The dashboard process does not read `.env`, contact FRED, or estimate the BVAR. Open the local
URL printed by Shiny. Highcharts is loaded from its official CDN, so deployment needs outbound
browser access and an appropriate Highcharts license for the intended use.

## Quality checks

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
