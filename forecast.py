"""WeatherNext 2 medium-range precipitation, averaged over hydrologic regions.

Source: Open-Meteo ensemble API, model google_weathernext2_ensemble
(64 members, 0.25 degree grid, 15 days, runs at 00 and 12 UTC).

This module does not import compare.py. The app passes in the region polygon
and point-in-polygon function it already loaded, so boundaries are defined once.
"""
import numpy as np
import pandas as pd
import requests

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL = "google_weathernext2_ensemble"
MAX_POINTS_PER_CALL = 50
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


def fetch_daily_precip(points, forecast_days=15, timezone="America/Los_Angeles"):
    """Long-format DataFrame: member, date, grid_index, precip_mm."""
    rows = []
    for start in range(0, len(points), MAX_POINTS_PER_CALL):
        chunk = points[start:start + MAX_POINTS_PER_CALL]
        resp = requests.get(
            ENSEMBLE_URL,
            params={
                "latitude": ",".join(str(p[0]) for p in chunk),
                "longitude": ",".join(str(p[1]) for p in chunk),
                "daily": "precipitation_sum",
                "models": MODEL,
                "forecast_days": forecast_days,
                "timezone": timezone,
            },
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
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
    """Average over grid cells, giving one value per member per day."""
    per_member = (
        raw.groupby(["member", "date"], as_index=False)["precip_mm"].mean()
        .rename(columns={"precip_mm": "region_precip_mm"})
    )
    # The last forecast day can be incomplete, so drop days with no data.
    return per_member.dropna(subset=["region_precip_mm"])


def ensemble_summary(per_member):
    """Mean, p10, median, p90 across members for each day, in millimeters."""
    summary = (
        per_member.groupby("date")["region_precip_mm"]
        .describe(percentiles=[0.1, 0.5, 0.9])
        .rename(columns={"10%": "p10", "50%": "median", "90%": "p90"})
        [["mean", "p10", "median", "p90"]]
        .reset_index()
    )
    return summary
