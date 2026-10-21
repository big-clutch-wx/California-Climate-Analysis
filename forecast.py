"""WeatherNext 2 medium-range precipitation, averaged over hydrologic regions.

Source: Open-Meteo ensemble API, model google_weathernext2_ensemble
(64 members, 0.25 degree grid, 15 days, runs at 00 and 12 UTC).

This module does not import compare.py. The app passes in the region polygon
and point-in-polygon function it already loaded, so boundaries are defined once.
"""
import time

import numpy as np
import pandas as pd
import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL = "google_weathernext2_ensemble"
MAX_POINTS_PER_CALL = 50
PAUSE_BETWEEN_CALLS_S = 2.0
MAX_ATTEMPTS = 4
MM_PER_INCH = 25.4


def grid_points(polygon, in_polygon, step=0.5):
    """Grid centers inside a polygon. polygon is a list of (lat, lon) pairs."""
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    points = []
    for lat in np.arange(min(lats), max(lats) + step, step):
        for lon in np.arange(min(lons), max(lons) + step, step):
            if in_polygon((lat, lon), polygon):
                points.append((round(float(lat), 4), round(float(lon), 4)))
    return points


def _get_with_retry(params):
    """GET the ensemble endpoint, backing off on 429 responses."""
    resp = None
    for attempt in range(MAX_ATTEMPTS):
        resp = requests.get(ENSEMBLE_URL, params=params, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        wait = float(resp.headers.get("Retry-After", 5 * 2 ** attempt))
        time.sleep(min(wait, 60))
    raise RuntimeError(
        f"Open-Meteo rate limit still in place after {MAX_ATTEMPTS} attempts. "
        f"Wait a few minutes, or use a coarser grid. Server said: {resp.text[:200]}"
    )


def fetch_daily_precip(points, forecast_days=15, timezone="America/Los_Angeles"):
    """Long-format DataFrame: member, date, grid_index, precip_mm."""
    rows = []
    chunks = [points[i:i + MAX_POINTS_PER_CALL]
              for i in range(0, len(points), MAX_POINTS_PER_CALL)]
    for n, chunk in enumerate(chunks):
        if n > 0:
            time.sleep(PAUSE_BETWEEN_CALLS_S)
        start = n * MAX_POINTS_PER_CALL
        payload = _get_with_retry({
            "latitude": ",".join(str(p[0]) for p in chunk),
            "longitude": ",".join(str(p[1]) for p in chunk),
            "daily": "precipitation_sum",
            "models": MODEL,
            "forecast_days": forecast_days,
            "timezone": timezone,
        })
        locations = payload if isinstance(payload, list) else [payload]
        for offset, loc in enumerate(locations):
            daily = loc["daily"]
            for key, values in daily.items():
                if not key.startswith("precipitation_sum"):
                    continue
                for date, value in zip(daily["time"], values):
                    rows.append({
                        "member": key,
                        "date": pd.Timestamp(date),
                        "grid_index": start + offset,
                        "precip_mm": value,
                    })
    return pd.DataFrame(rows)


def member_region_means(raw):
    """Average over grid cells, giving one value per member per day (mm)."""
    per_member = (
        raw.groupby(["member", "date"], as_index=False)["precip_mm"].mean()
        .rename(columns={"precip_mm": "region_precip_mm"})
    )
    # The last forecast day can be incomplete, so drop days with no data.
    return per_member.dropna(subset=["region_precip_mm"])


def cumulative_summary(per_member):
    """Cumulative precipitation in inches, rounded to 0.01.

    Each member is accumulated on its own first, then the mean and percentiles
    are taken across members. Accumulating the daily percentiles instead would
    give a different and misleading total.
    """
    df = per_member.sort_values(["member", "date"]).copy()
    df["cum_mm"] = df.groupby("member")["region_precip_mm"].cumsum()
    df["cum_in"] = df["cum_mm"] / MM_PER_INCH
    summary = (
        df.groupby("date")["cum_in"]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .rename(columns={"10%": "p10", "50%": "median", "90%": "p90"})
        [["mean", "p10", "median", "p90"]]
        .reset_index()
    )
    for col in ["mean", "p10", "median", "p90"]:
        summary[col] = summary[col].round(2)
    return summary
