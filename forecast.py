"""WeatherNext 2 medium-range precipitation, averaged over hydrologic regions.

Source: Open-Meteo ensemble API, model google_weathernext2_ensemble
(64 members, 0.25 degree grid, 15 days, runs at 00 and 12 UTC).

Daily totals are built from the hourly values. The API's daily precipitation_sum
is blank for some days that the hourly data covers, so summing the 24 hours of
each local day gives a complete window.

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
    """GET the ensemble endpoint. One retry after a capped wait on a 429."""
    resp = None
    for attempt in range(2):
        resp = requests.get(ENSEMBLE_URL, params=params, timeout=60)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        if attempt == 0:
            wait = min(float(resp.headers.get("Retry-After", 10)), 20)
            time.sleep(wait)
    raise RuntimeError(
        "Open-Meteo is rate limiting this app right now. Wait a few minutes and "
        "reload, or use a coarser grid. Server said: " + resp.text[:200]
    )


def fetch_daily_precip(points, forecast_days=16, timezone="America/Los_Angeles"):
    """Hourly member precipitation for each point, as a long DataFrame.

    Columns: member, time (local), grid_index, precip_mm. The name is kept from
    earlier versions; the values are hourly. Member columns are matched on the
    'precipitation' prefix and 'member' in the name, and should be checked on the
    first real response.
    """
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
            "hourly": "precipitation",
            "models": MODEL,
            "forecast_days": forecast_days,
            "timezone": timezone,
        })
        locations = payload if isinstance(payload, list) else [payload]
        for offset, loc in enumerate(locations):
            hourly = loc["hourly"]
            times = pd.to_datetime(hourly["time"])
            for key, values in hourly.items():
                if key == "time" or not key.startswith("precipitation"):
                    continue
                # The plain 'precipitation' field is the control run, the 64th
                # ensemble member. Numbered fields are members 1 to 63.
                member = "precipitation_control" if key == "precipitation" else key
                for t, value in zip(times, values):
                    rows.append({
                        "member": member,
                        "time": t,
                        "grid_index": start + offset,
                        "precip_mm": value,
                    })
    return pd.DataFrame(rows)


def member_region_means(raw, max_days=15):
    """Regional daily totals per member (mm), from hourly values.

    1. Sum each grid cell's hourly values by local day.
    2. Keep only days with all 24 hours present, so partial days are dropped.
    3. Average the cells inside the region, per member and day.
    4. Keep the first max_days complete days.
    """
    if raw.empty:
        return pd.DataFrame(columns=["member", "date", "region_precip_mm"])

    df = raw.dropna(subset=["precip_mm"]).copy()
    df["date"] = df["time"].dt.normalize()

    cell_daily = (
        df.groupby(["member", "grid_index", "date"])["precip_mm"]
        .agg(total="sum", hours="count")
        .reset_index()
    )
    cell_daily = cell_daily[cell_daily["hours"] == 24]

    region = (
        cell_daily.groupby(["member", "date"], as_index=False)["total"].mean()
        .rename(columns={"total": "region_precip_mm"})
    )
    keep_dates = sorted(region["date"].unique())[:max_days]
    return region[region["date"].isin(keep_dates)].reset_index(drop=True)


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
