"""WeatherNext 2 medium-range precipitation, averaged over hydrologic regions.

Source: Open-Meteo ensemble API, model google_weathernext2_ensemble
(64 members, 0.25 degree grid, 15 days, runs at 00 and 12 UTC).

Daily totals are built from the hourly values. The API's daily precipitation_sum
is blank for some days that the hourly data covers, so summing the 24 hours of
each local day gives a complete window.

This module does not import compare.py. The app passes in the region polygon
and point-in-polygon function it already loaded, so boundaries are defined once.
"""
import re
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


# ---------------------------------------------------------------------------
# Single-point model blend: WeatherNext 2 + ECMWF EPS + AIFS-ENS
# ---------------------------------------------------------------------------
# The single-run AIFS is left out on purpose: one deterministic run has no
# spread to add to the ensemble percentiles.
BLEND_MODELS = [
    ("WN2", "google_weathernext2_ensemble"),
    ("EPS", "ecmwf_ifs025_ensemble"),
    ("AIFS-ENS", "ecmwf_aifs025_ensemble"),
]


def fetch_point_members(lat, lon, model_id, forecast_days=16,
                        timezone="America/Los_Angeles"):
    """Hourly member precipitation (mm) for one model at one point.

    Returns columns member, time, grid_index, precip_mm in the same shape as
    fetch_daily_precip, so member_region_means works on it unchanged. The plain
    'precipitation' field is the control run and 'precipitation_memberNN' are
    the perturbed members. A trailing '_<model id>' suffix is tolerated in case
    the API adds one.
    """
    payload = _get_with_retry({
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "models": model_id,
        "forecast_days": forecast_days,
        "timezone": timezone,
    })
    loc = payload[0] if isinstance(payload, list) else payload
    hourly = loc["hourly"]
    times = pd.to_datetime(hourly["time"])
    pattern = re.compile(r"^precipitation(?:_member(\d+))?(?:_" + re.escape(model_id) + r")?$")
    columns = {}
    for key, values in hourly.items():
        match = pattern.match(key)
        if key == "time" or not match:
            continue
        if match.group(1) is None:
            name = "precipitation_control"
        else:
            name = f"precipitation_member{int(match.group(1)):02d}"
        columns[name] = values
    if not columns:
        raise ValueError(f"No member precipitation fields returned for {model_id}.")
    wide = pd.DataFrame(columns, index=times)
    long = wide.reset_index(names="time").melt(
        id_vars="time", var_name="member", value_name="precip_mm"
    )
    long["grid_index"] = 0
    return long[["member", "time", "grid_index", "precip_mm"]]


def fetch_point_blend(lat, lon, forecast_days=15):
    """Daily member totals for every blend model. Returns (per_model, errors).

    A model that fails is reported in errors and left out, so one outage does
    not blank the whole page.
    """
    per_model, errors = {}, {}
    for i, (name, model_id) in enumerate(BLEND_MODELS):
        if i:
            time.sleep(1.0)
        try:
            raw = fetch_point_members(lat, lon, model_id, forecast_days=forecast_days + 1)
            daily = member_region_means(raw, max_days=forecast_days)
            if daily.empty:
                raise ValueError("no complete forecast days returned")
            per_model[name] = daily
        except Exception as exc:  # noqa: BLE001 - shown to the user per model
            errors[name] = str(exc)
    return per_model, errors


def blend_members(per_model):
    """Stack the models into one long frame with equal weight per model.

    Columns: model, member, date, region_precip_mm, weight. Only dates present
    in every model are kept. Each model's weights sum to 1/n_models, so a model
    with more members does not outvote one with fewer.
    """
    common = None
    for df in per_model.values():
        dates = set(df["date"].unique())
        common = dates if common is None else common & dates
    n_models = len(per_model)
    parts = []
    for name, df in per_model.items():
        sub = df[df["date"].isin(common)].copy()
        sub["model"] = name
        sub["weight"] = 1.0 / (n_models * sub["member"].nunique())
        parts.append(sub[["model", "member", "date", "region_precip_mm", "weight"]])
    return pd.concat(parts, ignore_index=True)


def _wquantile(values, weights, qs):
    """Weighted quantiles by interpolating the weighted empirical CDF."""
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    pos = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(qs, pos, v)


def blend_cumulative_summary(blend):
    """Weighted mean and 10/50/90th percentiles of cumulative inches by date.

    Each member is accumulated on its own first, then combined with the model
    weights, the same order of operations as cumulative_summary.
    """
    df = blend.sort_values(["model", "member", "date"]).copy()
    df["cum_in"] = df.groupby(["model", "member"])["region_precip_mm"].cumsum() / MM_PER_INCH
    rows = []
    for date, g in df.groupby("date"):
        v = g["cum_in"].to_numpy()
        w = g["weight"].to_numpy()
        p10, med, p90 = _wquantile(v, w, [0.1, 0.5, 0.9])
        rows.append({"date": date, "mean": float(np.average(v, weights=w)),
                     "p10": p10, "median": med, "p90": p90})
    out = pd.DataFrame(rows)
    for col in ["mean", "p10", "median", "p90"]:
        out[col] = out[col].round(2)
    return out


def blend_daily_matrix(blend, metric="chance", threshold_in=0.10):
    """Rows Blend, then each model; columns dates.

    chance: weighted percent of members at or above threshold_in inches.
    median: weighted median daily total in inches.
    """
    def one(sub):
        out = {}
        for date, g in sub.groupby("date"):
            v = (g["region_precip_mm"] / MM_PER_INCH).to_numpy()
            w = g["weight"].to_numpy()
            if metric == "chance":
                out[date] = w[v >= threshold_in].sum() / w.sum() * 100
            else:
                out[date] = _wquantile(v, w, [0.5])[0]
        return pd.Series(out)

    rows = {"Blend": one(blend)}
    for name in blend["model"].unique():
        rows[name] = one(blend[blend["model"] == name])
    return pd.DataFrame(rows).T.sort_index(axis=1)


def blend_member_matrix(blend, cumulative=False):
    """(blend_mean, by_model) for the plume matrices.

    blend_mean: one-row DataFrame labeled BLEND, the equal-weight average of the
        model means (computed before rounding).
    by_model: {model: (members, mean_row)}. members has one row per member
        (CONT, M01, ...); mean_row has one row labeled MEAN.
    """
    by_model, model_means = {}, {}
    for name in blend["model"].unique():
        sub = blend[blend["model"] == name][["member", "date", "region_precip_mm"]]
        mm = member_matrix(sub, cumulative=cumulative)
        body = mm.iloc[:-1].copy()
        body.index = ["CONT" if i == "Control" else i for i in body.index]
        by_model[name] = (body, mm.loc[["Mean"]].rename(index={"Mean": "MEAN"}))

        inches = sub.assign(inches=sub["region_precip_mm"] / MM_PER_INCH)
        wide = inches.pivot(index="member", columns="date", values="inches").sort_index(axis=1)
        if cumulative:
            wide = wide.cumsum(axis=1)
        model_means[name] = wide.mean(axis=0)
    blend_mean = pd.DataFrame(model_means).T.mean(axis=0).round(2).to_frame("BLEND").T
    return blend_mean, by_model


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Text sizes are arguments with defaults. The app passes its own values from the
# TEXT SIZES block in app.py, so change them there.

def fig_cumulative(summary, mean_label="Mean", font_size=12):
    """Cumulative line chart: 10th-90th band, median and mean."""
    import plotly.graph_objects as go
    fig = go.Figure()
    # Each trace gets an explicit hover label, so none shows as "trace N".
    fig.add_trace(go.Scatter(
        x=summary["date"], y=summary["p90"], mode="lines",
        line={"width": 0}, showlegend=False,
        hovertemplate="90th pct: %{y:.2f} in<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=summary["date"], y=summary["p10"], mode="lines",
        line={"width": 0}, fill="tonexty",
        fillcolor="rgba(70,130,180,0.25)", name="10th–90th percentile",
        hovertemplate="10th pct: %{y:.2f} in<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=summary["date"], y=summary["median"], mode="lines",
        name="Median", line={"width": 2.5},
        hovertemplate="Median: %{y:.2f} in<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=summary["date"], y=summary["mean"], mode="lines",
        name=mean_label, line={"width": 2, "dash": "dash"},
        hovertemplate=mean_label + ": %{y:.2f} in<extra></extra>",
    ))
    fig.update_layout(
        yaxis_title="Cumulative precipitation (in)",
        xaxis_title="Date",
        hovermode="x unified",
        height=420,
        margin={"t": 30},
        font={"size": font_size},
    )
    return fig


def fig_daily_matrix(matrix, use_chance, threshold_in, font_size=12):
    """Heatmap of chance of rain (percent) or median daily total (inches)."""
    import plotly.graph_objects as go
    day_labels = [d.strftime("%a %b %d") for d in matrix.columns]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=day_labels, y=list(matrix.index),
        colorscale="Blues", zmin=0, zmax=100 if use_chance else None,
        text=matrix.round(0 if use_chance else 2).values,
        texttemplate="%{text}%" if use_chance else "%{text}",
        textfont={"size": font_size},
        hovertemplate=(
            "%{y}<br>%{x}<br>"
            + ("Chance of ≥ " + f"{threshold_in:.2f}" + " in: %{z:.0f}%"
               if use_chance else "Median: %{z:.2f} in")
            + "<extra></extra>"
        ),
        colorbar={"title": "%" if use_chance else "in"},
    ))
    fig.update_layout(
        height=max(190, 46 * len(matrix.index) + 90),
        margin={"t": 10, "b": 10},
        xaxis={"side": "top"}, yaxis={"autorange": "reversed"},
        font={"size": font_size},
    )
    return fig


def _plume_fmt(v):
    if v < 0.01:
        return ""
    return f"{v:.2f}".rstrip("0").rstrip(".") if v < 10 else f"{v:.1f}"


def _plume_traces(frame, days, yaxis_id, show_scale, cell_font, colorbar_font):
    """Heatmap plus a white-text overlay for dark cells, for one panel."""
    import plotly.graph_objects as go
    nb = len(PLUME_COLORS)
    vals = frame.values
    bins = plume_bin(vals)
    light = plume_text_is_light(bins)
    txt = np.vectorize(_plume_fmt)(vals).astype(object)
    heat = go.Heatmap(
        z=bins, x=days, y=list(frame.index),
        colorscale=plume_colorscale(), zmin=-0.5, zmax=nb - 0.5,
        xgap=1, ygap=1, yaxis=yaxis_id,
        customdata=vals, text=np.where(light, "", txt), texttemplate="%{text}",
        textfont={"size": cell_font, "color": "black"},
        hovertemplate="%{y}<br>%{x}<br>%{customdata:.2f} in<extra></extra>",
        showscale=show_scale,
        colorbar=dict(
            tickvals=[i - 0.5 for i in range(1, nb)],
            ticktext=[("%g" % e) for e in PLUME_EDGES],
            tickfont={"size": colorbar_font}, len=0.9, y=0.55, thickness=12, title="in",
        ) if show_scale else None,
    )
    xs, ys = np.meshgrid(days, list(frame.index))
    over = go.Scatter(
        x=xs.ravel(), y=ys.ravel(), mode="text", text=np.where(light, txt, "").ravel(),
        textfont={"size": cell_font, "color": "white"}, hoverinfo="skip",
        showlegend=False, xaxis="x", yaxis=yaxis_id,
    )
    return heat, over


# Same left and right margins on every plume figure, so the date columns line up
# from one figure to the next.
_PLUME_MARGIN = {"l": 90, "r": 90}


def fig_member_plume(members, means, member_font=9, mean_font=14, label_font=9,
                     date_font=12, colorbar_font=8, show_scale=True):
    """Plume matrix: one row per member and a mean panel underneath.

    members, means: DataFrames of inches with date columns; row labels are
    shown as given. Cells print their value; colors follow PLUME_EDGES.
    """
    import plotly.graph_objects as go
    days = [d.strftime("%a %b %d") for d in members.columns]
    fig = go.Figure()
    for tr in _plume_traces(members, days, "y", show_scale, member_font, colorbar_font):
        fig.add_trace(tr)
    for tr in _plume_traces(means, days, "y2", False, mean_font, colorbar_font):
        fig.add_trace(tr)

    main_px = 15 * len(members)
    mean_px = (56 if len(means) == 1 else 36) * len(means)
    gap_px, top_px, bottom_px = 50, 40, 20
    plot_h = main_px + gap_px + mean_px
    fig.update_layout(
        height=plot_h + top_px + bottom_px,
        margin={"t": top_px, "b": bottom_px, **_PLUME_MARGIN},
        plot_bgcolor="white",
        xaxis={"side": "top", "type": "category", "tickfont": {"size": date_font}},
        yaxis={"domain": [(mean_px + gap_px) / plot_h, 1.0], "autorange": "reversed",
               "type": "category", "tickfont": {"size": label_font},
               "tickmode": "array", "tickvals": list(members.index),
               "ticktext": list(members.index)},
        yaxis2={"domain": [0.0, mean_px / plot_h], "autorange": "reversed",
                "type": "category", "anchor": "x", "tickmode": "array",
                "tickfont": {"size": label_font},
                "tickvals": list(means.index), "ticktext": list(means.index)},
    )
    return fig


def fig_mean_strip(means, mean_font=14, label_font=9, date_font=12, colorbar_font=8):
    """A mean-only plume strip (for the blend mean above the member matrices)."""
    import plotly.graph_objects as go
    days = [d.strftime("%a %b %d") for d in means.columns]
    fig = go.Figure()
    for tr in _plume_traces(means, days, "y", False, mean_font, colorbar_font):
        fig.add_trace(tr)
    row_px = 56
    top_px, bottom_px = 40, 10
    fig.update_layout(
        height=row_px * len(means) + top_px + bottom_px,
        margin={"t": top_px, "b": bottom_px, **_PLUME_MARGIN},
        plot_bgcolor="white",
        xaxis={"side": "top", "type": "category", "tickfont": {"size": date_font}},
        yaxis={"autorange": "reversed", "type": "category", "tickmode": "array",
               "tickfont": {"size": label_font},
               "tickvals": list(means.index), "ticktext": list(means.index)},
    )
    return fig
