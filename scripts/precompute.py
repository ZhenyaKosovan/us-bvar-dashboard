from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from us_bvar.artifact import ForecastArtifact, save_artifact
from us_bvar.data import FREDClient
from us_bvar.model import BVAR

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FRED data and precompute the fitted BVAR and baseline forecast."
    )
    parser.add_argument("--draws", type=int, default=400, help="Posterior predictive draws.")
    parser.add_argument("--seed", type=int, default=202503, help="Baseline simulation seed.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the existing per-series FRED cache without making network calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    client = FREDClient(
        api_key="" if args.offline else None,
        cache_dir=ROOT / "data/cache",
    )
    panel = client.fetch_panel()
    model = BVAR().fit(panel.values)
    baseline = model.forecast(horizon=12, draws=args.draws, seed=args.seed)
    artifact = ForecastArtifact.create(model, baseline, created_at=panel.fetched_at)

    panel_path = ROOT / "data/fred_panel.csv"
    panel.values.to_csv(panel_path, index_label="date")
    artifact_path = ROOT / "artifacts/bvar_forecast.pkl"
    save_artifact(artifact, artifact_path)
    metadata = {
        "schema_version": artifact.schema_version,
        "created_at": artifact.created_at.isoformat(),
        "panel_start": artifact.panel_start.date().isoformat(),
        "panel_end": artifact.panel_end.date().isoformat(),
        "observation_count": artifact.observation_count,
        "forecast_start": artifact.baseline.dates[0].date().isoformat(),
        "forecast_end": artifact.baseline.dates[-1].date().isoformat(),
        "posterior_draws": artifact.baseline.draws,
        "posterior_interval_quantiles": list(model.config.interval),
        "source": "FRED API cache" if panel.from_cache else "FRED API",
    }
    (ROOT / "artifacts/metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Saved {artifact.observation_count} observations through "
        f"{artifact.panel_end:%Y-%m} and a {len(baseline.dates)}-month baseline "
        f"to {artifact_path.relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
