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


def daily_matrix(region_members, metric="chance", threshold_in=0.10):
    """Region-by-day table from per-member regional daily totals.

    region_members: {region_name: DataFrame[member, date, region_precip_mm]}
    metric "chance": percent of members whose regional-average daily total is at
        or above threshold_in inches.
    metric "median": median across members of the regional daily total, inches.
    Values are for the region-wide average, not for any single point in it.
    """
    rows = {}
    for region, pm in region_members.items():
        inches = pm["region_precip_mm"] / MM_PER_INCH
        by_day = pm["date"]
        if metric == "chance":
            rows[region] = (inches >= threshold_in).groupby(by_day).mean() * 100
        else:
            rows[region] = inches.groupby(by_day).median()
    return pd.DataFrame(rows).T.sort_index(axis=1)


def member_matrix(per_member, cumulative=False):
    """Members-by-day table in inches, rounded to 0.01, with a Mean row last.

    per_member: DataFrame[member, date, region_precip_mm] for one region.
    cumulative: accumulate each member over the days first, then average, so
        the Mean row is the mean of the members' own running totals.
    Rows are members (Control first, then M01...), columns are dates.
    """
    df = per_member.copy()
    df["inches"] = df["region_precip_mm"] / MM_PER_INCH
    wide = df.pivot(index="member", columns="date", values="inches").sort_index(axis=1)
    if cumulative:
        wide = wide.cumsum(axis=1)

    def label(m):
        text = str(m)
        if "control" in text.lower():
            return "Control"
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"M{int(digits):02d}" if digits else text

    order = sorted(wide.index, key=lambda m: (0, 0) if "control" in str(m).lower()
                   else (1, int("".join(ch for ch in str(m) if ch.isdigit()) or 0)))
    wide = wide.loc[order]
    wide.index = [label(m) for m in wide.index]
    wide.loc["Mean"] = wide.mean(axis=0)
    return wide.round(2)


# Plume-matrix color bands (inches), modeled on the weathermodels.com EPS plume.
PLUME_EDGES = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,
               0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4,
               4.5, 5, 5.5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30]
PLUME_COLORS = (
    ["#ffffff"]
    + ["#e6e6e6", "#cfcfcf", "#b0b0b0", "#8c8c8c"]
    + ["#9be39b", "#66cc66", "#33b533", "#1fa01f", "#0d8a0d", "#0a6e0a"]
    + ["#bfe3ff", "#90cdf5", "#62b4ec", "#3b9ae0", "#2480d0"]
    + ["#ffffb3", "#fff58a", "#ffeb5c", "#ffdd33", "#ffc800"]
    + ["#ffa500", "#ff8000", "#ff5a00", "#f03000", "#d81800"]
    + ["#b00000", "#8b0000", "#6e0020"]
    + ["#800080", "#9a30b0", "#b060c8", "#c890e0"]
    + ["#f0b8f8", "#f5a0f0", "#f888e0", "#fa70d0", "#fc58c0", "#fe40b0"]
    + ["#ff20a0"]
)
assert len(PLUME_COLORS) == len(PLUME_EDGES) + 1


def plume_bin(values):
    """Index of the color band for each value (0 means under 0.01 in)."""
    import numpy as np
    return np.searchsorted(PLUME_EDGES, np.asarray(values, dtype=float), side="right")


def plume_colorscale():
    """Step colorscale for bin indices 0..len(PLUME_COLORS)-1 (use zmin=-0.5, zmax=n-0.5)."""
    n = len(PLUME_COLORS)
    scale = []
    for i, c in enumerate(PLUME_COLORS):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


def plume_text_is_light(bins):
    """True where the band is dark enough that white text reads better."""
    import numpy as np
    dark = np.array([
        (0.299 * int(c[1:3], 16) + 0.587 * int(c[3:5], 16) + 0.114 * int(c[5:7], 16)) < 140
        for c in PLUME_COLORS
    ])
    return dark[np.asarray(bins, dtype=int)]
