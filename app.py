#!/usr/bin/env python3
"""
California Precipitation Analysis - Streamlit Web App

Water Year (WY) mode implemented to match compare(3).py:
- July 1 through June 30 water years
- same station-validity rules
- same California hydrological regions and WY baselines
- same common-station filtering across all selected WYs
- regional comparison summary table
- optional detailed station breakdown
- statewide average + side-by-side matrix when multiple regions are selected
"""

import os
import json
import importlib.util
import glob
import calendar
from pathlib import Path

import duckdb

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="California Precipitation Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Locate the data / comparison engine
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

# Local development can use E:\ACIS; deployment can set ACIS_DATA_DIR.
_default_data_dir = Path(r"E:\ACIS") if Path(r"E:\ACIS").exists() else APP_DIR
DATA_DIR = Path(os.environ.get("ACIS_DATA_DIR", str(_default_data_dir))).resolve()

ENGINE_PATH = APP_DIR / "compare.py"

if not ENGINE_PATH.exists():
    st.error(
        "compare.py was not found next to app.py. "
        "The web app uses its calculation engine so the results stay consistent "
        "with the command-line analysis."
    )
    st.stop()

spec = importlib.util.spec_from_file_location("compare_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


# compare(3).py's DuckDB functions look for daily_*.parquet in the current
# working directory. Keep that behavior exactly, while allowing the web app
# to point at a different data directory through ACIS_DATA_DIR.
os.chdir(DATA_DIR)


# ---------------------------------------------------------------------------
# Cached metadata / region mapping
# ---------------------------------------------------------------------------

@st.cache_data
def load_metadata(data_dir: str):
    metadata_path = Path(data_dir) / "stations_meta.jsonl"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"stations_meta.jsonl was not found in {metadata_path.parent}"
        )

    metadata = {}

    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta = json.loads(line)
            metadata[meta["id"]] = meta

    return metadata


@st.cache_data
def build_region_map(metadata_items, selected_regions):
    """
    Match compare(3).py's station-to-region logic:
    test every station's lat/lon against the selected official polygons,
    stopping at the first matching region.
    """
    metadata = dict(metadata_items)
    regions = engine.get_official_regions()

    station_region_map = {}

    for sid, meta in metadata.items():
        point = (meta["lat"], meta["lon"])

        for region in selected_regions:
            polygon = regions[region]["polygon"]

            if engine.point_in_polygon(point, polygon):
                station_region_map[sid] = region
                break

    return station_region_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_years_web(value: str):
    """Use compare(3).py's year parser and provide a friendly error."""
    try:
        years = engine.parse_years_input(value)
    except Exception as exc:
        raise ValueError(f"Invalid water-year input: {exc}") from exc

    return years


def relabel_water_year_output(df):
    """
    Convert the engine's internal starting-year water-year labels to the
    standard ending-year convention used by the web UI.

    Example:
      engine WY 1997 = 1997-07-01 through 1998-06-30
      web UI         = WY 1998
    """
    if df is None or df.empty:
        return df

    out = df.copy()
    out["period_id"] = pd.to_numeric(out["period_id"], errors="coerce") + 1
    out["period_id"] = out["period_id"].astype("Int64")
    out["period_label"] = out["period_id"].map(
        lambda y: f"WY {int(y)}" if pd.notna(y) else "WY"
    )
    return out


def format_wy_columns(columns):
    """
    Convert compare(3).py period labels such as 1994-95 into the
    short detailed-table headers used by the CLI (1994).
    """
    result = []

    for period in columns:
        text = str(period)
        parts = text.split("-")

        if len(parts) == 2 and len(parts[0]) == 4 and len(parts[1]) == 2:
            result.append(parts[0])
        else:
            result.append(text)

    return result


def make_detail_table(df_res, metadata, station_region_map):
    """
    Reproduce compare(3).py's detailed station-table behavior:
    Station Name, Region, then one precipitation column per selected WY,
    sorted by average precipitation across the selected periods.
    """
    detail = df_res.copy()

    pivot = detail.pivot(
        index="station_id",
        columns="period_label",
        values="total_precip",
    )

    pivot["station_name"] = [
        metadata.get(sid, {}).get("name", sid)
        for sid in pivot.index
    ]

    pivot["region"] = [
        station_region_map.get(sid, "Unknown")
        for sid in pivot.index
    ]

    precip_cols = [
        col for col in pivot.columns
        if col not in ["station_name", "region"]
    ]

    if precip_cols:
        pivot["avg_precip"] = pivot[precip_cols].mean(axis=1)
        pivot = pivot.sort_values("avg_precip", ascending=False)
        pivot = pivot.drop(columns=["avg_precip"])

    display_cols = ["station_name", "region"] + precip_cols
    display = pivot[display_cols].copy()

    # compare(3).py explicitly drops N/A rows in the displayed detail table.
    display = display.dropna(subset=precip_cols)

    rename_map = {
        old: new
        for old, new in zip(precip_cols, format_wy_columns(precip_cols))
    }

    display = display.rename(columns=rename_map).reset_index(drop=True)

    # Match the CLI's displayed precision.
    for col in format_wy_columns(precip_cols):
        display[col] = display[col].round(2)

    return display


def build_summary(df_res, regions, station_region_map):
    """
    Reproduce compare(3).py's regional summary calculation.
    """
    summary = (
        df_res.groupby(["region", "period_id", "period_label"])
        .agg(
            avg_precip=("total_precip", "mean"),
            avg_wet_days=("wet_days", "mean"),
            avg_max_daily=("max_daily_precip", "mean"),
            avg_consec_wet=("max_consec_wet", "mean"),
            avg_consec_dry=("max_consec_dry", "mean"),
            station_count=("station_id", "count"),
        )
        .reset_index()
    )

    summary["regional_base"] = summary["region"].map(
        lambda r: regions[r]["avg_precip"]
    )
    summary["pct_base"] = (
        summary["avg_precip"] / summary["regional_base"]
    ) * 100

    return summary


def build_statewide_summary(df_res):
    """True statewide-style mean across all selected stations."""
    statewide = (
        df_res.groupby("period_label")["total_precip"]
        .mean()
        .reset_index(name="avg_precip")
    )

    statewide["pct_base"] = (statewide["avg_precip"] / 23.5) * 100
    return statewide


def intersect_common_stations(df, expected_periods):
    """Keep only stations with valid data in every expected period.

    This is the single station-selection rule used by every comparison mode,
    for regional rows and for the California statewide rows alike.
    """
    if df is None or df.empty:
        return df
    counts = df.groupby("station_id")["period_label"].nunique()
    keep = counts[counts == expected_periods].index
    return df[df["station_id"].isin(keep)].copy()


def query_daily_comparison_series(station_ids, period_specs):
    """Daily series for the same station set the comparison table averages.

    Every calendar day is averaged over the full fixed station set. A station
    that did not report on a given day counts as 0 for that day, so the
    cumulative line ends at the mean of each station's observed period total,
    which is the table's average. The station_count column is the number of
    stations that actually reported that day.
    """
    if not station_ids or not period_specs:
        return pd.DataFrame()

    ids = sorted(set(station_ids))
    ids_str = ", ".join([f"'{sid}'" for sid in ids])
    union_queries = []

    for idx, spec in enumerate(period_specs):
        union_queries.append(f"""
            SELECT
                {idx} AS period_index,
                CAST(date AS DATE) AS date,
                station_id,
                CASE
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS DOUBLE)
                END AS precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '{spec['start']}'
              AND CAST(date AS DATE) <= DATE '{spec['end']}'
        """)

    sql = (
        "WITH raw AS (" + " UNION ALL ".join(union_queries) + ") "
        "SELECT period_index, date, station_id, precip "
        "FROM raw WHERE precip IS NOT NULL"
    )

    con = duckdb.connect()
    try:
        raw = con.execute(sql).df()
    finally:
        con.close()

    n_stations = len(ids)
    frames = []
    for idx, spec in enumerate(period_specs):
        calendar = pd.date_range(
            pd.Timestamp(spec["start"]), pd.Timestamp(spec["end"]), freq="D"
        )
        period_rows = raw[raw["period_index"] == idx]
        if period_rows.empty:
            wide = pd.DataFrame(index=calendar)
        else:
            wide = (
                period_rows.assign(date=pd.to_datetime(period_rows["date"]))
                .pivot_table(index="date", columns="station_id",
                             values="precip", aggfunc="first")
                .reindex(calendar)
            )
        frames.append(pd.DataFrame({
            "period_index": idx,
            "period_label": spec["label"],
            "date": calendar,
            "daily_precip": wide.fillna(0.0).sum(axis=1).to_numpy() / n_stations,
            "station_count": wide.notna().sum(axis=1).to_numpy(),
        }))

    return pd.concat(frames, ignore_index=True)

CANONICAL_1991_2020_NORMALS = {
    "California": 25.73,
    "Central Coast": 21.61,
    "Colorado River": 4.35,
    "North Coast": 42.88,
    "North Lahontan": 33.31,
    "Sacramento River": 41.52,
    "San Francisco Bay": 26.73,
    "San Joaquin River": 26.17,
    "South Coast": 16.99,
    "South Lahontan": 8.98,
    "Tulare Lake": 14.98,
}


def canonical_annual_normal(scope_key):
    return CANONICAL_1991_2020_NORMALS.get(scope_key)


@st.cache_data(show_spinner=False)
def query_daily_normal(station_ids):
    """Build a station-based 1991-2020 Water Year daily normal.

    The normal uses WY 1991 through WY 2020 (1990-07-01 through
    2020-06-30), matching the app's Water Year convention.  For each
    station, only water years with at least 200 valid precipitation days are
    included.  Each station is first averaged across its valid WYs, then the
    station normals are averaged across the region.  This keeps stations from
    having more complete records disproportionately weighted.

    Returns one row per calendar month/day plus the regional mean WY total.
    """
    if not station_ids:
        return pd.DataFrame(), None

    ids_str = ", ".join([f"'{str(sid).replace(chr(39), chr(39)+chr(39))}'" for sid in station_ids])

    sql = f"""
        WITH raw AS (
            SELECT
                station_id,
                CAST(date AS DATE) AS date,
                CASE
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(
                        TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', ''))
                        AS DOUBLE
                    )
                END AS precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '1990-07-01'
              AND CAST(date AS DATE) <= DATE '2020-06-30'
        ),
        tagged AS (
            SELECT
                station_id,
                date,
                precip,
                CASE
                    WHEN EXTRACT(MONTH FROM date) >= 7
                        THEN CAST(EXTRACT(YEAR FROM date) AS INTEGER) + 1
                    ELSE CAST(EXTRACT(YEAR FROM date) AS INTEGER)
                END AS wy_end_year
            FROM raw
            WHERE precip IS NOT NULL
        ),
        station_wy AS (
            SELECT
                station_id,
                wy_end_year,
                SUM(precip) AS wy_total,
                COUNT(*) AS valid_days
            FROM tagged
            WHERE wy_end_year BETWEEN 1991 AND 2020
            GROUP BY station_id, wy_end_year
            HAVING COUNT(*) >= 200
        ),
        station_normal AS (
            SELECT
                station_id,
                AVG(wy_total) AS station_wy_normal
            FROM station_wy
            GROUP BY station_id
            HAVING COUNT(*) >= 20
        ),
        daily_station AS (
            SELECT
                t.station_id,
                EXTRACT(MONTH FROM t.date)::INTEGER AS month,
                EXTRACT(DAY FROM t.date)::INTEGER AS day,
                AVG(t.precip) AS station_daily_normal
            FROM tagged t
            INNER JOIN station_wy sw
                ON t.station_id = sw.station_id
               AND t.wy_end_year = sw.wy_end_year
            GROUP BY
                t.station_id,
                EXTRACT(MONTH FROM t.date)::INTEGER,
                EXTRACT(DAY FROM t.date)::INTEGER
        ),
        regional_daily AS (
            SELECT
                month,
                day,
                AVG(station_daily_normal) AS normal_daily_precip
            FROM daily_station ds
            INNER JOIN station_normal sn
                ON ds.station_id = sn.station_id
            GROUP BY month, day
        ),
        regional_annual AS (
            SELECT AVG(station_wy_normal) AS normal_annual_inches
            FROM station_normal
        )
        SELECT
            rd.month,
            rd.day,
            rd.normal_daily_precip,
            ra.normal_annual_inches
        FROM regional_daily rd
        CROSS JOIN regional_annual ra
        ORDER BY rd.month, rd.day
    """

    con = duckdb.connect()
    try:
        result = con.execute(sql).df()
    finally:
        con.close()

    if result.empty:
        return result, None

    annual = float(result["normal_annual_inches"].iloc[0])
    return result[["month", "day", "normal_daily_precip"]], annual


def normal_total_for_dates(normal_df, start_date, end_date):
    """Sum the 1991-2020 daily normal over an arbitrary calendar span."""
    if normal_df is None or normal_df.empty:
        return None
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if start > end:
        return None
    lookup = {
        (int(row.month), int(row.day)): float(row.normal_daily_precip)
        for row in normal_df.itertuples(index=False)
    }
    return sum(
        lookup.get((int(d.month), int(d.day)), 0.0)
        for d in pd.date_range(start, end, freq="D")
    )


@st.cache_data(show_spinner=False)
def get_period_normal_map(station_ids, period_specs, scope_key=None):
    """Return canonical 1991-2020 normal totals for each period."""
    if not station_ids or not period_specs:
        return {}, None
    normal_df, computed_annual = query_daily_normal(tuple(station_ids))
    annual_normal = canonical_annual_normal(scope_key) if scope_key else computed_annual
    if normal_df.empty:
        return {}, annual_normal
    period_normals = {}
    for spec in period_specs:
        start = pd.Timestamp(spec["start"])
        end = pd.Timestamp(spec["end"])

        # A full Water Year must use the exact same annual 1991-2020
        # station-network normal used by Extremes / Records.  Summing the
        # daily normal can differ slightly because the daily normal is built
        # from calendar-day means and may have different day-level coverage.
        if (
            start.month == 7
            and start.day == 1
            and end.month == 6
            and end.day == 30
            and end.year == start.year + 1
        ):
            period_normals[spec["label"]] = annual_normal
        else:
            period_normals[spec["label"]] = normal_total_for_dates(
                normal_df, spec["start"], spec["end"]
            )

    return period_normals, annual_normal


def render_xmacis_style_chart(daily_df, station_ids, chart_title, show_normal=True, normal_station_ids=None, scope_key=None):
    """Render an XMACIS-style cumulative precipitation chart with hover data."""
    if daily_df is None or daily_df.empty:
        st.info("Daily precipitation data are not available for this chart.")
        return

    import plotly.graph_objects as go

    fig = go.Figure()

    # Map month/day onto a synthetic numeric calendar.  Numeric x values
    # avoid invalid dates when a leap-year comparison contains Feb 29 while
    # another comparison is mapped onto a non-leap synthetic year.  The
    # calendar is anchored to the first month/day of each comparison period
    # and wraps at the end of the calendar, so a Jul-Jun water year remains
    # Jul -> Aug -> ... -> Dec -> Jan -> ... -> Jun.
    def synthetic_positions(dates):
        dates = pd.to_datetime(dates)
        if dates.empty:
            return [], None
        start_month = int(dates.iloc[0].month)
        start_day = int(dates.iloc[0].day)
        anchor = pd.Timestamp(2000, start_month, start_day)
        positions = []
        for d in dates:
            canonical = pd.Timestamp(2000, int(d.month), int(d.day))
            pos = (canonical - anchor).days
            if pos < 0:
                pos += 366
            positions.append(pos)
        return positions, (start_month, start_day)

    def calendar_tick_positions(start_month, start_day):
        anchor = pd.Timestamp(2000, start_month, start_day)
        ticks = []
        labels = []
        for month in range(1, 13):
            day = 1
            canonical = pd.Timestamp(2000, month, day)
            pos = (canonical - anchor).days
            if pos < 0:
                pos += 366
            ticks.append(pos)
            labels.append(canonical.strftime('%b'))
        return ticks, labels

    for period_index, group in daily_df.groupby("period_index", sort=False):
        group = group.sort_values("date").copy()
        group["date"] = pd.to_datetime(group["date"])
        group["cumulative"] = group["daily_precip"].fillna(0.0).cumsum()
        label = str(group["period_label"].iloc[0])
        x_values, tick_anchor = synthetic_positions(group["date"])

        customdata = list(zip(
            group["date"].dt.strftime("%b %d, %Y"),
            group["daily_precip"].round(2),
            group["cumulative"].round(2),
        ))

        fig.add_trace(go.Scatter(
            x=x_values,
            y=group["cumulative"],
            mode="lines",
            name=label,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Daily: %{customdata[1]:.2f}\"<br>"
                "Accumulation: %{customdata[2]:.2f}\"<extra></extra>"
            ),
            line={"width": 2.5},
        ))

    # Add the daily normal as a cumulative reference line. The normal uses
    # the fixed full-scope station network, independent of the comparison
    # station intersection.
    if show_normal and (normal_station_ids or station_ids):
        first_index = daily_df["period_index"].min()
        first = daily_df[daily_df["period_index"] == first_index].sort_values("date").copy()
        first["date"] = pd.to_datetime(first["date"])
        if not first.empty:
            normal_ids = tuple(normal_station_ids) if normal_station_ids else tuple(station_ids)
            normal_df, normal_annual_inches = query_daily_normal(normal_ids)
            if not normal_df.empty:
                normal_lookup = {
                    (int(r.month), int(r.day)): float(r.normal_daily_precip)
                    for r in normal_df.itertuples(index=False)
                }
                raw_cumulative = 0.0
                raw_y = []
                normal_x = []
                # Use the same month/day axis as the first comparison period.
                normal_positions, _ = synthetic_positions(first["date"])
                for d, xpos in zip(first["date"], normal_positions):
                    raw_cumulative += normal_lookup.get((int(d.month), int(d.day)), 0.0)
                    normal_x.append(xpos)
                    raw_y.append(raw_cumulative)

                # For full Water Year charts, force the plotted normal to
                # terminate at the same canonical 1991-2020 annual normal
                # used everywhere else in the app.  query_daily_normal() can
                # produce a slightly different endpoint because its daily
                # curve has its own day-level coverage rules.
                normal_y = raw_y
                is_full_water_year = (
                    len(first["date"]) > 0
                    and int(first["date"].iloc[0].month) == 7
                    and int(first["date"].iloc[0].day) == 1
                    and int(first["date"].iloc[-1].month) == 6
                    and int(first["date"].iloc[-1].day) == 30
                )
                canonical = canonical_annual_normal(scope_key) if is_full_water_year and scope_key else None
                if canonical is not None and raw_y and raw_y[-1] > 0:
                    scale = canonical / raw_y[-1]
                    normal_y = [value * scale for value in raw_y]

                fig.add_trace(go.Scatter(
                    x=normal_x,
                    y=normal_y,
                    mode="lines",
                    name="Normal (1991-2020)",
                    line={"width": 2},
                    hovertemplate="<b>Normal (1991-2020)</b><br>Accumulation: %{y:.2f}\"<extra></extra>",
                ))

    fig.update_layout(
        title=chart_title,
        xaxis_title=None,
        yaxis_title="Precipitation (inches)",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "top", "y": -0.16, "xanchor": "center", "x": 0.5},
        margin={"l": 55, "r": 30, "t": 55, "b": 75},
        height=500,
    )
    # Use clean monthly calendar ticks.  Plotly receives numeric day positions,
    # so leap days never create invalid synthetic timestamps.
    first_period = daily_df.groupby("period_index", sort=False).first().iloc[0]
    axis_dates = pd.to_datetime(daily_df.loc[
        daily_df["period_index"] == first_period.name, "date"
    ])
    if not axis_dates.empty:
        start_month = int(axis_dates.iloc[0].month)
        start_day = int(axis_dates.iloc[0].day)
        tickvals, ticktext = calendar_tick_positions(start_month, start_day)
        fig.update_xaxes(
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            showgrid=True,
            range=[min(tickvals), max(tickvals)],
        )
    fig.update_yaxes(showgrid=True)

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("California Climate")
st.sidebar.caption("California precipitation analysis (1890-2026)")

st.sidebar.markdown("---")
st.sidebar.markdown("### Hydrological Regions")

regions = engine.get_official_regions()
region_list = sorted(regions.keys())

selected_regions = st.sidebar.multiselect(
    "Select region(s):",
    region_list,
    default=["Sacramento River"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Analysis Mode")

mode_choice = st.sidebar.radio(
    "Mode:",
    [
        "Comparison Mode",
        "Extremes / Records",
        "Regional Forecast (WN2)",
        "Single-Point Forecast (Model Blend)",
    ],
    index=0,
)

# The two forecast choices share one code path below; forecast_mode tells it which.
if mode_choice == "Regional Forecast (WN2)":
    analysis_mode = "Medium-Range Forecast"
    forecast_mode = "Regional (WN2)"
elif mode_choice == "Single-Point Forecast (Model Blend)":
    analysis_mode = "Medium-Range Forecast"
    forecast_mode = "Single-Point (Model Blend)"
else:
    analysis_mode = mode_choice
    forecast_mode = None

if analysis_mode == "Comparison Mode":
    comparison_mode = st.sidebar.radio(
        "Comparison type:",
        [
            "Full Water Years",
            "Recurring Seasonal Stretch",
            "Distinct Custom Date Ranges",
        ],
        index=0,
    )
else:
    comparison_mode = None

st.sidebar.markdown("---")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("California Precipitation Analysis")

if not selected_regions and forecast_mode != "Single-Point (Model Blend)":
    st.warning("Select at least one hydrological region.")
    st.stop()

# ---------------------------------------------------------------------------
# Regional precipitation baselines
# ---------------------------------------------------------------------------

# Show the long-term annual precipitation baseline for every selected
# hydrological region. California is a separate statewide scope and uses the
# statewide 23.5-inch baseline.
hydrological_regions_selected = [
    r for r in selected_regions if r != "California"
]

if hydrological_regions_selected or "California" in selected_regions:
    st.subheader("Average Annual Precipitation")

    info_regions = hydrological_regions_selected.copy()
    if "California" in selected_regions:
        info_regions.insert(0, "California")

    info_cols = st.columns(min(len(info_regions), 4))

    for i, region in enumerate(info_regions):
        with info_cols[i % len(info_cols)]:
            if region == "California":
                cdwr_avg = 23.5
                our_avg = canonical_annual_normal("California")
            else:
                cdwr_avg = regions[region]["avg_precip"]
                our_avg = canonical_annual_normal(region)

            st.info(
                f"**{region}**\n\n"
                f"**Our 1991–2020 station-network average:** {our_avg:.2f}\"\n\n"
                f"**CDWR average:** {cdwr_avg:.1f}\""
            )

# ---------------------------------------------------------------------------
# Memory-efficient Extremes / Records helpers
# ---------------------------------------------------------------------------

def _sql_ids(matching_ids):
    return ", ".join(
        "'" + str(sid).replace("'", "''") + "'"
        for sid in matching_ids
    )


def run_lightweight_water_year_records(matching_ids, min_valid_days=100, limit=5):
    """
    Find WY records entirely inside DuckDB.

    The comparison-mode engine returns station-level rows plus streak metrics.
    Extremes only need the station WY total and valid-day count, so reducing
    immediately to one row per WY keeps the pandas result tiny.
    """
    if not glob.glob("daily_*.parquet") or not matching_ids:
        return None

    ids = _sql_ids(matching_ids)

    sql = f"""
    WITH daily AS (
        SELECT
            station_id,
            CAST(date AS DATE) AS date,
            CASE
                WHEN precip IS NULL
                     OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M')
                    THEN NULL
                WHEN TRIM(CAST(precip AS VARCHAR)) = 'T'
                    THEN 0.0
                ELSE TRY_CAST(
                    TRIM(
                        REGEXP_REPLACE(
                            CAST(precip AS VARCHAR), '[AS]$', ''
                        )
                    ) AS DOUBLE
                )
            END AS p
        FROM read_parquet('daily_*.parquet', union_by_name=true)
        WHERE station_id IN ({ids})
          AND CAST(date AS DATE) >= DATE '1889-07-01'
          AND CAST(date AS DATE) <= DATE '2026-06-30'
    ),
    station_wy AS (
        SELECT
            station_id,
            CASE
                WHEN EXTRACT(MONTH FROM date) >= 7
                    THEN CAST(EXTRACT(YEAR FROM date) AS INTEGER) + 1
                ELSE CAST(EXTRACT(YEAR FROM date) AS INTEGER)
            END AS wy_end_year,
            SUM(p) AS total_precip,
            COUNT(p) AS valid_days
        FROM daily
        WHERE p IS NOT NULL
        GROUP BY station_id, wy_end_year
        HAVING COUNT(p) >= {int(min_valid_days)}
    ),
    period_averages AS (
        SELECT
            wy_end_year,
            AVG(total_precip) AS precip,
            COUNT(*) AS station_count
        FROM station_wy
        WHERE wy_end_year BETWEEN 1890 AND 2025
        GROUP BY wy_end_year
    )
    SELECT
        'Lowest' AS record_type,
        wy_end_year,
        precip,
        station_count
    FROM period_averages
    QUALIFY ROW_NUMBER() OVER (
        ORDER BY precip ASC, wy_end_year ASC
    ) <= {int(limit)}

    UNION ALL

    SELECT
        'Highest' AS record_type,
        wy_end_year,
        precip,
        station_count
    FROM period_averages
    QUALIFY ROW_NUMBER() OVER (
        ORDER BY precip DESC, wy_end_year ASC
    ) <= {int(limit)}

    ORDER BY record_type, precip, wy_end_year
    """

    with duckdb.connect() as con:
        return con.execute(sql).df()


def run_lightweight_seasonal_records(
    matching_ids,
    start_mmdd,
    end_mmdd,
    min_valid_ratio=0.50,
    limit=5,
):
    """
    Find recurring seasonal records inside DuckDB.

    This preserves compare.py's validity rule (a station must have the
    required number of valid observations for the occurrence), but returns
    only one aggregated row per occurrence instead of every station row.
    """
    if not glob.glob("daily_*.parquet") or not matching_ids:
        return None

    ids = _sql_ids(matching_ids)
    sm, sd = map(int, start_mmdd.split("-"))
    em, ed = map(int, end_mmdd.split("-"))
    cross_year = (sm, sd) > (em, ed)

    if cross_year:
        date_where = f"""
            (
                EXTRACT(MONTH FROM CAST(date AS DATE)) > {sm}
                OR (
                    EXTRACT(MONTH FROM CAST(date AS DATE)) = {sm}
                    AND EXTRACT(DAY FROM CAST(date AS DATE)) >= {sd}
                )
                OR EXTRACT(MONTH FROM CAST(date AS DATE)) < {em}
                OR (
                    EXTRACT(MONTH FROM CAST(date AS DATE)) = {em}
                    AND EXTRACT(DAY FROM CAST(date AS DATE)) <= {ed}
                )
            )
        """
        occurrence_year = f"""
            CASE
                WHEN (
                    EXTRACT(MONTH FROM CAST(date AS DATE)) > {sm}
                    OR (
                        EXTRACT(MONTH FROM CAST(date AS DATE)) = {sm}
                        AND EXTRACT(DAY FROM CAST(date AS DATE)) >= {sd}
                    )
                )
                    THEN CAST(EXTRACT(YEAR FROM CAST(date AS DATE)) AS INTEGER)
                ELSE CAST(EXTRACT(YEAR FROM CAST(date AS DATE)) AS INTEGER) - 1
            END
        """
        first_date = f"DATE '1890-{sm:02d}-{sd:02d}'"
        last_date = f"DATE '2026-{em:02d}-{ed:02d}'"
        # Start years 1890-2025 for a cross-year period.
        occurrence_range = "BETWEEN 1890 AND 2025"
        from_year = 1889
        to_year = 2026
    else:
        date_where = f"""
            (
                EXTRACT(MONTH FROM CAST(date AS DATE)) > {sm}
                OR (
                    EXTRACT(MONTH FROM CAST(date AS DATE)) = {sm}
                    AND EXTRACT(DAY FROM CAST(date AS DATE)) >= {sd}
                )
            )
            AND
            (
                EXTRACT(MONTH FROM CAST(date AS DATE)) < {em}
                OR (
                    EXTRACT(MONTH FROM CAST(date AS DATE)) = {em}
                    AND EXTRACT(DAY FROM CAST(date AS DATE)) <= {ed}
                )
            )
        """
        occurrence_year = "CAST(EXTRACT(YEAR FROM CAST(date AS DATE)) AS INTEGER)"
        first_date = f"DATE '1890-{sm:02d}-{sd:02d}'"
        last_date = f"DATE '2026-{em:02d}-{ed:02d}'"
        occurrence_range = "BETWEEN 1890 AND 2026"
        from_year = 1890
        to_year = 2026

    # Match compare.py's day-count calculation, including leap-day effects
    # when the selected span crosses February.
    from datetime import datetime

    d1 = datetime(2001, sm, sd)
    d2 = datetime(2002 if cross_year else 2001, em, ed)
    total_days = (d2 - d1).days + 1
    required = max(1, int(total_days * min_valid_ratio))

    sql = f"""
    WITH daily AS (
        SELECT
            station_id,
            CAST(date AS DATE) AS date,
            CASE
                WHEN precip IS NULL
                     OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M')
                    THEN NULL
                WHEN TRIM(CAST(precip AS VARCHAR)) = 'T'
                    THEN 0.0
                ELSE TRY_CAST(
                    TRIM(
                        REGEXP_REPLACE(
                            CAST(precip AS VARCHAR), '[AS]$', ''
                        )
                    ) AS DOUBLE
                )
            END AS p
        FROM read_parquet('daily_*.parquet', union_by_name=true)
        WHERE station_id IN ({ids})
          AND CAST(date AS DATE) >= DATE '{from_year}-01-01'
          AND CAST(date AS DATE) <= DATE '{to_year}-12-31'
          AND {date_where}
    ),
    station_occurrence AS (
        SELECT
            station_id,
            {occurrence_year} AS occurrence_year,
            SUM(p) AS total_precip,
            COUNT(p) AS valid_days
        FROM daily
        WHERE p IS NOT NULL
        GROUP BY station_id, occurrence_year
        HAVING COUNT(p) >= {required}
    ),
    period_averages AS (
        SELECT
            occurrence_year,
            AVG(total_precip) AS precip,
            COUNT(*) AS station_count
        FROM station_occurrence
        WHERE occurrence_year {occurrence_range}
        GROUP BY occurrence_year
    )
    SELECT
        'Lowest' AS record_type,
        occurrence_year,
        precip,
        station_count
    FROM period_averages
    QUALIFY ROW_NUMBER() OVER (
        ORDER BY precip ASC, occurrence_year ASC
    ) <= {int(limit)}

    UNION ALL

    SELECT
        'Highest' AS record_type,
        occurrence_year,
        precip,
        station_count
    FROM period_averages
    QUALIFY ROW_NUMBER() OVER (
        ORDER BY precip DESC, occurrence_year ASC
    ) <= {int(limit)}

    ORDER BY record_type, precip, occurrence_year
    """

    with duckdb.connect() as con:
        return con.execute(sql).df()


def run_lightweight_rolling_records(
    matching_ids,
    window_days,
    min_valid_ratio=0.70,
    limit=5,
):
    """
    Find top-N lowest and highest rolling-N-day periods.

    Rolling windows are selected greedily so ranked periods do not overlap.
    This prevents the results from being dominated by several one-day-shifted
    versions of the same wet/dry event.
    """
    if not glob.glob("daily_*.parquet") or not matching_ids:
        return None

    ids = _sql_ids(matching_ids)
    n = int(window_days)
    required = max(1, int(n * min_valid_ratio))

    sql = f"""
    WITH daily AS (
        SELECT
            station_id,
            CAST(date AS DATE) AS date,
            CASE
                WHEN precip IS NULL
                     OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M')
                    THEN NULL
                WHEN TRIM(CAST(precip AS VARCHAR)) = 'T'
                    THEN 0.0
                ELSE TRY_CAST(
                    TRIM(
                        REGEXP_REPLACE(
                            CAST(precip AS VARCHAR), '[AS]$', ''
                        )
                    ) AS DOUBLE
                )
            END AS p
        FROM read_parquet('daily_*.parquet', union_by_name=true)
        WHERE station_id IN ({ids})
          AND CAST(date AS DATE) >= DATE '1890-01-01'
          AND CAST(date AS DATE) <= DATE '2026-12-31'
          AND precip IS NOT NULL
    ),
    rolling AS (
        SELECT
            station_id,
            date - INTERVAL '{n - 1} days' AS period_start,
            date AS period_end,
            SUM(p) OVER (
                PARTITION BY station_id
                ORDER BY date
                RANGE BETWEEN INTERVAL '{n - 1} days'
                    PRECEDING AND CURRENT ROW
            ) AS total_precip,
            COUNT(p) OVER (
                PARTITION BY station_id
                ORDER BY date
                RANGE BETWEEN INTERVAL '{n - 1} days'
                    PRECEDING AND CURRENT ROW
            ) AS valid_days
        FROM daily
        WHERE p IS NOT NULL
    ),
    valid_windows AS (
        SELECT
            station_id,
            period_start,
            period_end,
            total_precip
        FROM rolling
        WHERE valid_days >= {required}
    )
    SELECT
        period_start,
        period_end,
        AVG(total_precip) AS precip,
        COUNT(*) AS station_count
    FROM valid_windows
    GROUP BY period_start, period_end
    ORDER BY period_end
    """

    with duckdb.connect() as con:
        candidates = con.execute(sql).df()

    if candidates.empty:
        return None

    # Greedily choose the strongest remaining period whose start is strictly
    # after the end of the previously selected period. This guarantees that
    # selected windows do not overlap.
    candidates["period_start"] = pd.to_datetime(candidates["period_start"])
    candidates["period_end"] = pd.to_datetime(candidates["period_end"])

    def select_distinct(frame, ascending):
        ordered = frame.sort_values(
            ["precip", "period_end"],
            ascending=[ascending, True],
        )

        selected = []

        # Select the strongest remaining candidate that does not overlap ANY
        # already-selected window. The old implementation only tracked the
        # most recently selected window, which could discard valid historical
        # candidates because the candidates are sorted by precipitation, not
        # chronologically. For N=1, this incorrectly turned the tail of the
        # rankings into a chronological list and could return fewer records
        # than requested.
        for _, row in ordered.iterrows():
            overlaps = any(
                row["period_start"] <= selected_row["period_end"]
                and row["period_end"] >= selected_row["period_start"]
                for selected_row in selected
            )

            if not overlaps:
                selected.append(row)

            if len(selected) >= int(limit):
                break

        if not selected:
            return pd.DataFrame(
                columns=["record_type", "period_start", "period_end",
                         "precip", "station_count"]
            )

        result = pd.DataFrame(selected).reset_index(drop=True)
        result.insert(
            0,
            "record_type",
            "Lowest" if ascending else "Highest",
        )
        return result[
            ["record_type", "period_start", "period_end", "precip", "station_count"]
        ]

    lowest = select_distinct(candidates, ascending=True)
    highest = select_distinct(candidates, ascending=False)

    return pd.concat([lowest, highest], ignore_index=True)



# ---------------------------------------------------------------------------
# Comparison Mode
# ---------------------------------------------------------------------------

if analysis_mode == "Comparison Mode":

    st.header("Comparison Mode")

    if comparison_mode == "Full Water Years":
        st.markdown(
            "**Full Water Years** — July 1 through June 30. "
            "Enter the year in which the water year ends."
        )

        st.info(
            "Water-year identifiers use the standard ending-year convention: "
            "**WY 1998 = July 1, 1997 through June 30, 1998**, displayed as "
            "**1997-98**."
        )

        years_input = st.text_input(
            "Enter water years",
            value="1998, 1995",
            help=(
                "Enter WY ending years. Examples: 1983, 1998, 2016 "
                "or 2015-2020."
            ),
        )

        min_valid_days = st.number_input(
            "Minimum valid days per WY",
            min_value=1,
            max_value=366,
            value=200,
            step=1,
            help=(
                "Modern data uses the selected threshold. Historical "
                "pre-1950 WYs automatically use 100 valid days."
            ),
        )

        show_detail = st.checkbox(
            "Show detailed station breakdown",
            value=True,
        )

        run_analysis = st.button(
            "🔍 Run Water Year Comparison",
            type="primary",
            use_container_width=True,
        )

    elif comparison_mode == "Recurring Seasonal Stretch":
        st.markdown(
            "**Recurring Seasonal Stretch** — the same MM-DD period is "
            "evaluated in each selected calendar year. If the end date is "
            "earlier than the start date, the stretch crosses New Year's."
        )

        col1, col2 = st.columns(2)
        with col1:
            start_mmdd = st.text_input(
                "Start date (MM-DD)",
                value="11-01",
                help="Example: 11-01",
            )
        with col2:
            end_mmdd = st.text_input(
                "End date (MM-DD)",
                value="11-30",
                help="Example: 02-18 for a cross-year stretch.",
            )

        years_input = st.text_input(
            "Enter years to compare",
            value="1981, 1982",
            help="Examples: 1980, 1995, 2010 or 2015-2020.",
        )

        show_detail = st.checkbox(
            "Show detailed station breakdown",
            value=True,
        )

        run_analysis = st.button(
            "🔍 Run Seasonal Comparison",
            type="primary",
            use_container_width=True,
        )

    else:  # Distinct Custom Date Ranges
        st.markdown(
            "**Distinct Custom Date Ranges** — compare two or more "
            "individually specified date ranges and include consecutive "
            "wet/dry spell metrics in the station-level results."
        )
        st.caption("Dates available: January 1, 1890 through December 31, 2026.")

        range_count = st.number_input(
            "Number of date ranges",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
        )

        date_ranges = []

        for i in range(int(range_count)):
            st.markdown(f"**Range #{i + 1}**")
            def date_selector(prefix, default_date):
                years = list(range(1890, 2027))
                months = list(range(1, 13))
                month_names = [
                    "January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December",
                ]

                y_col, m_col, d_col = st.columns(3)

                with y_col:
                    year = st.selectbox(
                        "Year",
                        years,
                        index=years.index(default_date.year),
                        key=f"{prefix}_year",
                    )

                with m_col:
                    month = st.selectbox(
                        "Month",
                        months,
                        index=default_date.month - 1,
                        format_func=lambda m: month_names[m - 1],
                        key=f"{prefix}_month",
                    )

                max_day = calendar.monthrange(year, month)[1]
                day_options = list(range(1, max_day + 1))
                day_default = min(default_date.day, max_day)

                with d_col:
                    day = st.selectbox(
                        "Day",
                        day_options,
                        index=day_default - 1,
                        key=f"{prefix}_day",
                    )

                return __import__("datetime").date(year, month, day)

            c1, c2 = st.columns(2)

            with c1:
                start_date = date_selector(
                    f"custom_start_{i}",
                    __import__("datetime").date(1981 + i, 1, 1),
                )

            with c2:
                end_date = date_selector(
                    f"custom_end_{i}",
                    __import__("datetime").date(1981 + i, 3, 31),
                )

            if start_date > end_date:
                st.error(f"Range #{i + 1}: start date must be on or before end date.")
            else:
                total_days = (end_date - start_date).days + 1
                min_valid = max(1, int(total_days * 0.70))
                date_ranges.append(
                    {
                        "start": start_date.strftime("%Y-%m-%d"),
                        "end": end_date.strftime("%Y-%m-%d"),
                        "label": (
                            f"{start_date.strftime('%b %d, %Y')} - "
                            f"{end_date.strftime('%b %d, %Y')}"
                        ),
                        "min_valid_days": min_valid,
                        "total_days": total_days,
                    }
                )
                st.caption(
                    f"{total_days} total days; requiring at least "
                    f"{min_valid} valid days (70%)."
                )

        show_detail = st.checkbox(
            "Show detailed station breakdown",
            value=True,
        )

        run_analysis = st.button(
            "🔍 Run Custom-Range Comparison",
            type="primary",
            use_container_width=True,
        )

    if run_analysis:
        try:
            metadata = load_metadata(str(DATA_DIR))

            # California is a statewide scope, not a hydrological region.
            # Exclude it from station-to-region assignment so its polygon
            # cannot claim every station before the actual hydrological
            # regions are tested.
            hydrological_regions = [
                r for r in selected_regions if r != "California"
            ]

            station_region_map = build_region_map(
                tuple(metadata.items()),
                tuple(hydrological_regions),
            )
            matching_ids = set(station_region_map.keys())

            # True California statewide station universe.
            california_polygon = regions["California"]["polygon"]
            statewide_ids = {
                sid
                for sid, meta in metadata.items()
                if engine.point_in_polygon(
                    (meta["lat"], meta["lon"]),
                    california_polygon,
                )
            }
            california_selected = "California" in selected_regions

            if not matching_ids and not california_selected:
                st.error("No stations were found inside the selected hydrological region(s).")
                st.stop()

            if california_selected and not statewide_ids:
                st.error("No stations were found inside the California state polygon.")
                st.stop()

            # ---------------------------------------------------------------
            # Run the selected comparison engine for hydrological regions
            # ---------------------------------------------------------------
            # Keep comparison inputs available below as well, because
            # California is queried separately after the regional query.
            engine_years = None
            effective_min_valid = None
            wy_end_years = None
            years = None
            min_ratio = None

            if not matching_ids:
                # California can be selected by itself, so parse the
                # comparison inputs even when there are no hydrological
                # region stations to query.
                if comparison_mode == "Full Water Years":
                    wy_end_years = parse_years_web(years_input)
                    if not wy_end_years:
                        st.error("No valid water years were entered.")
                        st.stop()
                    engine_years = [y - 1 for y in wy_end_years]
                    effective_min_valid = (
                        100 if any(y < 1950 for y in wy_end_years)
                        else int(min_valid_days)
                    )
                elif comparison_mode == "Recurring Seasonal Stretch":
                    years = parse_years_web(years_input)
                    if not years:
                        st.error("No valid years were entered.")
                        st.stop()
                    try:
                        from datetime import datetime
                        sm, sd = map(int, start_mmdd.strip().split("-"))
                        em, ed = map(int, end_mmdd.strip().split("-"))
                        datetime(2001, sm, sd)
                        datetime(2001, em, ed)
                        start_mmdd = f"{sm:02d}-{sd:02d}"
                        end_mmdd = f"{em:02d}-{ed:02d}"
                    except ValueError:
                        st.error("Invalid MM-DD date. Use the format MM-DD.")
                        st.stop()
                    has_pre_1950 = any(y < 1950 for y in years)
                    min_ratio = 0.50 if has_pre_1950 else 0.70

                df_res = pd.DataFrame()
            elif comparison_mode == "Full Water Years":
                wy_end_years = parse_years_web(years_input)

                if not wy_end_years:
                    st.error("No valid water years were entered.")
                    st.stop()

                # compare.py expects the starting year:
                # 1997-07-01 through 1998-06-30 is passed as 1997.
                engine_years = [y - 1 for y in wy_end_years]
                
                effective_min_valid = (
                    100
                    if any(y < 1950 for y in wy_end_years)
                    else int(min_valid_days)
                )

                with st.spinner(
                    f"Querying {len(matching_ids)} stations across "
                    f"{len(wy_end_years)} water year(s)..."
                ):
                    df_res = engine.run_duckdb_water_years(
                        matching_ids,
                        engine_years,
                        min_valid_days=effective_min_valid,
                    )
                    df_res = relabel_water_year_output(df_res)

            elif comparison_mode == "Recurring Seasonal Stretch":
                years = parse_years_web(years_input)

                if not years:
                    st.error("No valid years were entered.")
                    st.stop()

                # Validate MM-DD using the same calendar logic as compare.py.
                try:
                    from datetime import datetime
                    sm, sd = map(int, start_mmdd.strip().split("-"))
                    em, ed = map(int, end_mmdd.strip().split("-"))
                    datetime(2001, sm, sd)
                    datetime(2001, em, ed)
                    start_mmdd = f"{sm:02d}-{sd:02d}"
                    end_mmdd = f"{em:02d}-{ed:02d}"
                except ValueError:
                    st.error("Invalid MM-DD date. Use the format MM-DD.")
                    st.stop()

                has_pre_1950 = any(y < 1950 for y in years)
                min_ratio = 0.50 if has_pre_1950 else 0.70

                with st.spinner(
                    f"Querying {len(matching_ids)} stations across "
                    f"{len(years)} seasonal occurrence(s)..."
                ):
                    # Strict consistency is deliberately ALWAYS enabled.
                    df_res = engine.run_duckdb_custom_stretches(
                        matching_ids,
                        start_mmdd,
                        end_mmdd,
                        years,
                        min_valid_ratio=min_ratio,
                        strict_consistency=True,
                    )

            else:
                if len(date_ranges) < 2:
                    st.error("At least two valid custom date ranges are required.")
                    st.stop()

                with st.spinner(
                    f"Querying {len(matching_ids)} stations across "
                    f"{len(date_ranges)} custom ranges..."
                ):
                    df_res = engine.run_duckdb_distinct_ranges(
                        matching_ids,
                        date_ranges,
                    )

            # California is queried separately using the true state polygon.
            # This keeps statewide results independent of whichever
            # hydrological regions were selected.
            if comparison_mode == "Full Water Years":
                expected_n_periods = len(wy_end_years)
            elif comparison_mode == "Recurring Seasonal Stretch":
                expected_n_periods = len(years)
            else:
                expected_n_periods = len(date_ranges)

            statewide_df = None
            if california_selected:
                if comparison_mode == "Full Water Years":
                    with st.spinner(
                        f"Querying {len(statewide_ids)} California stations..."
                    ):
                        statewide_df = engine.run_duckdb_water_years(
                            statewide_ids,
                            engine_years,
                            min_valid_days=effective_min_valid,
                        )
                        statewide_df = relabel_water_year_output(statewide_df)
                elif comparison_mode == "Recurring Seasonal Stretch":
                    with st.spinner(
                        f"Querying {len(statewide_ids)} California stations..."
                    ):
                        statewide_df = engine.run_duckdb_custom_stretches(
                            statewide_ids,
                            start_mmdd,
                            end_mmdd,
                            years,
                            min_valid_ratio=min_ratio,
                            strict_consistency=True,
                        )
                else:
                    with st.spinner(
                        f"Querying {len(statewide_ids)} California stations..."
                    ):
                        statewide_df = engine.run_duckdb_distinct_ranges(
                            statewide_ids,
                            date_ranges,
                        )

            if california_selected:
                statewide_df = intersect_common_stations(
                    statewide_df, expected_n_periods
                )

            if df_res is None:
                df_res = pd.DataFrame()

            if df_res.empty and not california_selected:
                if comparison_mode == "Distinct Custom Date Ranges":
                    st.warning("No stations have sufficient valid observations for the specified date ranges.")
                else:
                    st.warning("No consistent station data found across all specified periods.")
                st.stop()

            # Map stations to hydrological regions. ALL comparison modes
            # use the same strict station intersection: a station must have
            # sufficient valid data in every selected period/range to be
            # included. This makes every comparison genuinely apples-to-apples.
            if not df_res.empty:
                df_res["region"] = df_res["station_id"].map(station_region_map)

                df_res = intersect_common_stations(df_res, expected_n_periods)

                if df_res.empty and not california_selected:
                    st.warning(
                        "No stations have valid data in every selected period."
                    )
                    st.stop()

            # Build the actual date spans used by the daily chart.
            period_specs = []
            if comparison_mode == "Full Water Years":
                for ending_year in wy_end_years:
                    period_specs.append({
                        "label": f"WY {ending_year}",
                        "start": f"{ending_year - 1}-07-01",
                        "end": f"{ending_year}-06-30",
                    })
            elif comparison_mode == "Recurring Seasonal Stretch":
                crosses_year = (start_mmdd > end_mmdd)
                for y in years:
                    period_specs.append({
                        "label": (
                            f"{start_mmdd} ({y}) to {end_mmdd} ({y + 1})"
                            if crosses_year
                            else f"{start_mmdd} to {end_mmdd} ({y})"
                        ),
                        "start": f"{y}-{start_mmdd}",
                        "end": f"{y + 1}-{end_mmdd}" if crosses_year else f"{y}-{end_mmdd}",
                    })
            else:
                for r in date_ranges:
                    period_specs.append({
                        "label": r["label"],
                        "start": r["start"],
                        "end": r["end"],
                    })

            # ---------------------------------------------------------------
            # Regional summary
            # ---------------------------------------------------------------

            summary = (
                build_summary(df_res, regions, station_region_map)
                if not df_res.empty
                else pd.DataFrame(
                    columns=[
                        "region", "period_id", "period_label",
                        "avg_precip", "pct_base", "station_count",
                    ]
                )
            )

            # % Avg uses one FIXED 1991–2020 station-network normal for each
            # scope. The baseline comes from the full station network belonging
            # to the region, not from the subset that survives the selected
            # comparison periods. This keeps % Avg identical between
            # Comparison Mode and Extremes / Records.
            regional_scope_ids = {
                region: tuple(
                    sid for sid, mapped_region in station_region_map.items()
                    if mapped_region == region
                )
                for region in hydrological_regions
            }

            regional_period_normals = {}
            for region, scope_ids in regional_scope_ids.items():
                if scope_ids:
                    regional_period_normals[region], _ = get_period_normal_map(
                        scope_ids, period_specs, scope_key=region
                    )

            # The comparison tables must use the same 1991-2020
            # station-network normal as Extremes / Records.  build_summary()
            # initializes pct_base from the legacy CDWR-style annual baseline,
            # so replace it here with the period-specific station normal for
            # every hydrological region.
            for region, normal_map in regional_period_normals.items():
                mask = summary["region"] == region
                summary.loc[mask, "pct_base"] = summary.loc[mask, "period_label"].map(
                    lambda label: (
                        summary.loc[mask & (summary["period_label"] == label), "avg_precip"].iloc[0]
                        / normal_map[label] * 100
                        if normal_map.get(label) not in (None, 0)
                        else float("nan")
                    )
                )

            statewide_period_normals = {}
            if california_selected and statewide_df is not None and not statewide_df.empty:
                statewide_period_normals, _ = get_period_normal_map(
                    tuple(statewide_ids),
                    period_specs,
                )

            regional_station_count = (
                df_res["station_id"].nunique() if not df_res.empty else 0
            )
            st.success(
                f"Analysis complete — {regional_station_count} stations "
                "have valid data in every selected period/range for the "
                "hydrological-region analysis."
            )

            st.header("Regional Comparison Summary Table")
            st.caption(
                "% Avg compares each period with its 1991–2020 station-network normal "
                "for the same calendar span; Water Years use the full WY normal."
            )

            # California is a separate statewide scope and is always shown
            # first when selected. It does not consume stations from the
            # hydrological-region summaries below.
            if california_selected:
                ca_summary = (
                    build_statewide_summary(statewide_df)
                    if statewide_df is not None and not statewide_df.empty
                    else pd.DataFrame()
                )
                if not ca_summary.empty and statewide_period_normals:
                    ca_summary["pct_base"] = ca_summary["period_label"].map(
                        lambda label: (
                            ca_summary.loc[ca_summary["period_label"] == label, "avg_precip"].iloc[0]
                            / statewide_period_normals[label] * 100
                            if statewide_period_normals.get(label) not in (None, 0)
                            else float("nan")
                        )
                    )
                if ca_summary.empty:
                    st.subheader("California (0 stations)")
                    st.info("No valid station data for California.")
                else:
                    ca_station_count = int(statewide_df["station_id"].nunique())
                    st.subheader(f"California ({ca_station_count} stations)")
                    ca_table = ca_summary[
                        ["period_label", "avg_precip", "pct_base"]
                    ].copy()
                    ca_table.columns = ["Period", "Avg Precip", "% Avg"]
                    ca_table["Avg Precip"] = ca_table["Avg Precip"].map(
                        lambda x: f'{x:.2f}"'
                    )
                    ca_table["% Avg"] = ca_table["% Avg"].map(
                        lambda x: f"{x:.1f}%"
                    )
                    st.dataframe(
                        ca_table,
                        use_container_width=True,
                        hide_index=True,
                    )

            for region in hydrological_regions:
                region_data = (
                    summary[summary["region"] == region]
                    .sort_values("period_id")
                    .copy()
                )

                if region_data.empty:
                    st.subheader(f"{region} (0 stations)")
                    st.info("No valid station data for this region.")
                    continue

                if comparison_mode == "Distinct Custom Date Ranges":
                    st.subheader(region)
                    table = region_data[
                        ["period_label", "avg_precip", "pct_base", "station_count"]
                    ].copy()
                    table.columns = ["Period", "Avg Precip", "% Avg", "Stations"]
                else:
                    station_count = int(region_data["station_count"].iloc[0])
                    st.subheader(f"{region} ({station_count} stations)")

                    table = region_data[
                        ["period_label", "avg_precip", "pct_base"]
                    ].copy()
                    table.columns = ["Period", "Avg Precip", "% Avg"]
                table["Avg Precip"] = table["Avg Precip"].map(
                    lambda x: f'{x:.2f}"'
                )
                table["% Avg"] = table["% Avg"].map(
                    lambda x: f"{x:.1f}%"
                )

                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                )

            # ---------------------------------------------------------------
            # XMACIS-style daily precipitation / accumulation chart
            # ---------------------------------------------------------------

            st.header("Daily Precipitation & Accumulation")
            st.caption(
                "Every day uses the same stations as the comparison table. "
                "The daily value is the average across that fixed set, with a "
                "station that did not report counted as 0, so the cumulative line "
                "ends at the table's average total."
                "The Normal line uses the fixed full-scope 1991–2020 station-network daily average."
            )

            chart_regions = []
            if california_selected and statewide_df is not None and not statewide_df.empty:
                chart_regions.append((
                    "California",
                    tuple(statewide_df["station_id"].drop_duplicates().tolist()),
                    tuple(statewide_ids),
                ))

            for region in hydrological_regions:
                # Comparison lines remain apples-to-apples using stations valid
                # in every selected period; the normal line uses the fixed full
                # regional station network.
                region_ids = tuple(
                    df_res.loc[df_res["region"] == region, "station_id"]
                    .drop_duplicates()
                    .tolist()
                )
                if region_ids:
                    chart_regions.append((
                        region,
                        region_ids,
                        regional_scope_ids.get(region, region_ids),
                    ))

            for region, region_ids, normal_ids in chart_regions:
                st.subheader(region)
                daily_chart_df = query_daily_comparison_series(
                    region_ids, period_specs
                )
                render_xmacis_style_chart(
                    daily_chart_df,
                    region_ids,
                    f"{region} — Daily Precipitation Accumulation",
                    show_normal=True,
                    normal_station_ids=normal_ids,
                    scope_key=region,
                )

            # ---------------------------------------------------------------
            # Side-by-side regional matrix
            # ---------------------------------------------------------------

            if len(selected_regions) > 1:
                st.header("Side-by-Side Regional Precipitation Matrix (Inches)")

                matrix = summary.pivot(
                    index="region",
                    columns="period_label",
                    values="avg_precip",
                )

                # California is a separate statewide scope. Include it in the
                # matrix when selected, but do not calculate a second
                # "statewide average" from the hydrological regions.
                if california_selected and statewide_df is not None and not statewide_df.empty:
                    california_avg = statewide_df.groupby("period_label")[
                        "total_precip"
                    ].mean()
                    matrix.loc["California"] = california_avg

                # Keep California first when it is present.
                if "California" in matrix.index:
                    ordered = ["California"] + [
                        r for r in matrix.index if r != "California"
                    ]
                    matrix = matrix.reindex(ordered)

                st.dataframe(
                    matrix.round(2),
                    use_container_width=True,
                )

                pct_matrix = summary.pivot(
                    index="region",
                    columns="period_label",
                    values="pct_base",
                )

                if california_selected and statewide_df is not None and not statewide_df.empty:
                    ca_avg = statewide_df.groupby("period_label")["total_precip"].mean()
                    pct_matrix.loc["California"] = pd.Series({
                        label: (
                            ca_avg[label] / statewide_period_normals[label] * 100
                            if statewide_period_normals.get(label) not in (None, 0)
                            else float("nan")
                        )
                        for label in ca_avg.index
                    })

                if "California" in pct_matrix.index:
                    ordered = ["California"] + [
                        r for r in pct_matrix.index if r != "California"
                    ]
                    pct_matrix = pct_matrix.reindex(ordered)

                st.subheader("Side-by-Side % Avg Matrix")
                st.dataframe(
                    pct_matrix.round(1),
                    use_container_width=True,
                    column_config={
                        col: st.column_config.NumberColumn(col, format="%.1f%%")
                        for col in pct_matrix.columns
                    },
                )

            # ---------------------------------------------------------------
            # Detailed station breakdown
            # ---------------------------------------------------------------

            if show_detail:
                st.header("Detailed Station Breakdown")

                # California is a separate statewide scope, so when it is
                # selected by itself df_res can legitimately be empty. Use
                # the California query for the detail table in that case.
                # When hydrological regions are also selected, retain the
                # regional station table here to avoid duplicating the same
                # stations (California contains the regional stations).
                detail_source = df_res
                detail_region_override = {}
                if detail_source.empty and california_selected:
                    detail_source = statewide_df.copy() if statewide_df is not None else pd.DataFrame()
                    if not detail_source.empty:
                        detail_region_override = {
                            sid: "California"
                            for sid in detail_source["station_id"].drop_duplicates()
                        }

                if detail_source.empty:
                    st.info("No station-level data are available for the selected scope.")
                else:
                    detail = detail_source.pivot(
                        index="station_id",
                        columns="period_label",
                        values="total_precip",
                    )

                    detail["station_name"] = [
                        metadata.get(sid, {}).get("name", sid)
                        for sid in detail.index
                    ]
                    detail["region"] = [
                        detail_region_override.get(
                            sid, station_region_map.get(sid, "Unknown")
                        )
                        for sid in detail.index
                    ]

                    precip_cols = [
                        c for c in detail.columns
                        if c not in ["station_name", "region"]
                    ]
    
                    if precip_cols:
                        detail["avg_precip"] = detail[precip_cols].mean(axis=1)
                        detail = detail.sort_values("avg_precip", ascending=False)
                        detail = detail.drop(columns=["avg_precip"])
    
                    display = detail[
                        ["station_name", "region"] + precip_cols
                    ].copy()
    
                    if comparison_mode != "Distinct Custom Date Ranges":
                        display = display.dropna(subset=precip_cols).copy()
    
                    # Human-friendly headers matching the CLI's intent:
                    # seasonal/custom periods use the year(s) or compact dates;
                    # WY columns use the ending-year identifier.
                    rename_map = {}
                    for col in precip_cols:
                        s = str(col)
                        import re
                        years_found = re.findall(r"\d{4}", s)
    
                        if comparison_mode == "Full Water Years":
                            if len(years_found) >= 2:
                                rename_map[col] = years_found[-1]
                            elif years_found:
                                rename_map[col] = years_found[0]
                            else:
                                rename_map[col] = s
                        elif comparison_mode == "Recurring Seasonal Stretch":
                            if years_found:
                                rename_map[col] = years_found[-1]
                            else:
                                rename_map[col] = s
                        else:
                            rename_map[col] = s
    
                    display = display.rename(columns=rename_map)
    
                    numeric_period_cols = [
                        rename_map.get(c, c) for c in precip_cols
                    ]
                    for col in numeric_period_cols:
                        if col in display.columns:
                            display[col] = display[col].round(2)
    
                    if comparison_mode == "Distinct Custom Date Ranges":
                        st.caption(
                            f"{len(display)} stations with observations in at least one "
                            "selected range, sorted by average precipitation. Missing "
                            "station-period observations are shown as blank."
                        )
                    else:
                        st.caption(
                            f"All {len(display)} stations with valid data in every "
                            f"selected period, sorted by average precipitation."
                        )
    
                    st.dataframe(
                        display.reset_index(drop=True),
                        use_container_width=True,
                        hide_index=True,
                    )
    
                    csv = display.to_csv(index=False)
                    st.download_button(
                        "📥 Download Station Breakdown (CSV)",
                        data=csv,
                        file_name="california_comparison_station_breakdown.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )


        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Analysis error: {exc}")
            with st.expander("Technical error details"):
                st.exception(exc)

# ---------------------------------------------------------------------------
# Extremes / Records
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Medium-Range Forecast (WeatherNext 2 via Open-Meteo)
# ---------------------------------------------------------------------------

elif analysis_mode == "Medium-Range Forecast":

    try:
        import forecast
    except ImportError:
        st.error("forecast.py is missing from the app folder.")
        st.stop()

    # The local date is part of the cache key, so a new forecast day triggers a
    # fresh fetch and the chart's date range moves forward.
    # FORECAST_CACHE_VERSION changes whenever the fetch logic changes, so results
    # cached by an older version of forecast.py are not reused from disk.
    FORECAST_CACHE_VERSION = "hourly-v4"
    forecast_day_key = (
        FORECAST_CACHE_VERSION + "|"
        + pd.Timestamp.now(tz="America/Los_Angeles").strftime("%Y-%m-%d")
    )

    # ======================================================================
    # TEXT SIZES for the forecast charts. Edit these numbers (points).
    # ======================================================================
    FONT_MEMBER_CELLS = 9       # numbers inside the member matrix cells
    FONT_MEMBER_LABELS = 9      # row labels: CONT, M01, M02, ...
    FONT_MEAN_CELLS = 24        # numbers inside the MEAN / BLEND boxes
    FONT_DATE_LABELS = 14       # date labels across the top of the member matrices
    FONT_COLORBAR = 16           # numbers on the color bar beside the member matrix
    FONT_DAILY_MATRIX = 24      # text in the daily precipitation matrix
    FONT_CUMULATIVE_CHART = 12  # text on the cumulative line chart
    # ======================================================================

    def member_plume_fig(members, means):
        return forecast.fig_member_plume(
            members, means,
            member_font=FONT_MEMBER_CELLS, mean_font=FONT_MEAN_CELLS,
            label_font=FONT_MEMBER_LABELS, date_font=FONT_DATE_LABELS,
            colorbar_font=FONT_COLORBAR,
        )

    def cumulative_fig(summary, mean_label="Mean"):
        return forecast.fig_cumulative(summary, mean_label=mean_label,
                                       font_size=FONT_CUMULATIVE_CHART)

    def daily_matrix_fig(matrix, use_chance, threshold_in):
        return forecast.fig_daily_matrix(matrix, use_chance, threshold_in,
                                         font_size=FONT_DAILY_MATRIX)

    def inferred_init(end_local):
        """Model run time (UTC), worked back 360 hours from the last data hour.

        The API does not report the run time. All three models run 15 days
        (360 h) from their 00z and 12z runs, so end time minus 360 h gives it.
        """
        end_utc = (pd.Timestamp(end_local)
                   .tz_localize("America/Los_Angeles", ambiguous=True,
                                nonexistent="shift_forward")
                   .tz_convert("UTC"))
        init = end_utc - pd.Timedelta(hours=360)
        return f"{init:%b} {init.day} {init:%H}z"

    def show_forecast_table(summary):
        table = summary.copy()
        table["date"] = table["date"].dt.strftime("%a %b %d")
        table = table.rename(columns={
            "date": "Day", "mean": "Cumulative mean (in)", "p10": "Cumulative 10th pct (in)",
            "median": "Cumulative median (in)", "p90": "Cumulative 90th pct (in)",
        })
        st.dataframe(table.round(2), use_container_width=True, hide_index=True)

    def daily_matrix_controls(key):
        mcol1, mcol2 = st.columns(2)
        metric = mcol1.radio(
            "Daily matrix shows",
            ["Chance of rain", "Median daily total"],
            horizontal=True,
            key=f"fc_matrix_metric_{key}",
        )
        threshold = mcol2.selectbox(
            "Rain threshold (in/day)",
            [0.01, 0.10, 0.25, 0.50, 1.00],
            index=1,
            disabled=(metric != "Chance of rain"),
            key=f"fc_matrix_threshold_{key}",
            help="A member counts as 'rain' when the day's total is at or above this amount.",
        )
        return metric == "Chance of rain", threshold

    if forecast_mode == "Regional (WN2)":

        st.header("Medium-Range Forecast (WeatherNext 2)")
        st.caption(
            "64-member ensemble, 15 days, averaged over the grid cells inside each "
            "selected hydrological region. The line shows cumulative precipitation "
            "from the forecast start, in inches. The shaded band is the 10th to 90th "
            "percentile of the cumulative totals across members."
        )

        grid_step = st.selectbox(
            "Grid spacing (degrees)",
            [0.5, 0.25],
            index=0,
            help="0.5 uses about a quarter of the grid points of 0.25 and is the "
                 "default because the API counts each location toward your quota.",
        )

        @st.cache_data(ttl=6 * 3600, show_spinner="Fetching WeatherNext 2 ensemble...")
        def cached_member_means(points, forecast_days, day_key):
            # Request one extra day, because the API leaves the last day empty.
            # member_region_means keeps the first forecast_days days with data.
            raw = forecast.fetch_daily_precip(list(points), forecast_days=forecast_days + 1)
            fetched_at = pd.Timestamp.now(tz="UTC")
            daily = forecast.member_region_means(raw, max_days=forecast_days)
            # Last hour with data (local time), used to work out the model run.
            daily.attrs["data_end"] = raw.dropna(subset=["precip_mm"])["time"].max()
            return daily, fetched_at

        # TEMPORARY: California forecast is paused to avoid spending API quota
        # until the WeatherNext data arrives. Remove this set to re-enable it.
        FORECAST_PAUSED_REGIONS = {"California"}

        for region_name in selected_regions:
            if region_name in FORECAST_PAUSED_REGIONS:
                st.info(f"{region_name} forecast is paused until the WeatherNext data is available.")
                continue
            polygon = regions[region_name]["polygon"]
            points = forecast.grid_points(
                polygon, engine.point_in_polygon, step=grid_step
            )
            if not points:
                st.warning(f"No grid points fall inside {region_name} at this spacing.")
                continue

            try:
                per_member, fetched_at = cached_member_means(
                    tuple(points), 15, forecast_day_key
                )
            except Exception as exc:
                st.error(f"{region_name}: forecast request failed: {exc}")
                continue

            summary = forecast.cumulative_summary(per_member)

            st.subheader(region_name)
            first_day = per_member["date"].min().strftime("%b %d")
            last_day = per_member["date"].max().strftime("%b %d")
            data_end = per_member.attrs.get("data_end")
            run_text = f" (init {inferred_init(data_end)})" if data_end is not None else ""
            st.caption(
                f"Source: Open-Meteo, WeatherNext 2 ensemble · "
                f"{len(points)} grid points · {per_member['member'].nunique()} members · "
                f"Complete days available: {first_day} to {last_day}{run_text} · "
                f"Retrieved {fetched_at.strftime('%b %d, %Y %H:%M')} UTC"
            )

            st.plotly_chart(cumulative_fig(summary), use_container_width=True)
            show_forecast_table(summary)

            # Daily precipitation matrix for this region (chance of rain or median).
            st.markdown("**Daily precipitation matrix**")
            use_chance, threshold_in = daily_matrix_controls(region_name)
            matrix = forecast.daily_matrix(
                {region_name: per_member},
                metric="chance" if use_chance else "median",
                threshold_in=threshold_in,
            )
            st.plotly_chart(
                daily_matrix_fig(matrix, use_chance, threshold_in),
                use_container_width=True,
            )
            st.caption(
                "Uses the 64 WeatherNext 2 members. Values describe the region-wide "
                "average for the day, so a cell can read low even when a single "
                "location in the region gets rain."
            )

            # Member-by-day plume matrix: x = day, y = member, MEAN panel at the bottom.
            st.markdown("**Member matrix**")
            mm_mode = st.radio(
                "Matrix values",
                ["Cumulative", "Daily"],
                horizontal=True,
                key=f"fc_member_mode_{region_name}",
            )
            mm = forecast.member_matrix(per_member, cumulative=(mm_mode == "Cumulative"))
            members_only = mm.iloc[:-1].copy()
            members_only.index = ["CONT" if i == "Control" else i for i in members_only.index]
            mean_row = mm.loc[["Mean"]].rename(index={"Mean": "MEAN"})
            st.plotly_chart(member_plume_fig(members_only, mean_row),
                            use_container_width=True)
            st.caption(
                mm_mode
                + " region-average precipitation in inches for each of the 64 members "
                "(CONT is the control run). The MEAN row averages the members."
            )
            st.download_button(
                "Download member matrix as CSV",
                mm.rename(columns=lambda d: d.strftime("%Y-%m-%d")).to_csv(index_label="Member"),
                file_name=f"wn2_members_{region_name.replace(' ', '_')}_{mm_mode.lower()}.csv",
                mime="text/csv",
                key=f"fc_member_csv_{region_name}",
            )

        st.caption(
            "Skill declines with lead time. Treat later days as a range of possibilities, "
            "not a single expected total. Data: Open-Meteo, Google WeatherNext 2."
        )

    else:
        # ------------------------------------------------------------------
        # Single-point model blend: WN2 + ECMWF EPS + AIFS-ENS, equal weight
        # per model. The single-run AIFS is not used.
        # ------------------------------------------------------------------
        st.header("Single-Point Forecast (Model Blend)")
        st.caption(
            "Blend of three ensembles at one location: Google WeatherNext 2 (64 members), "
            "ECMWF EPS and ECMWF AIFS-ENS. Each model counts equally in the blend, "
            "whatever its member count. Totals are accumulated per member first, then "
            "averaged; the band is the weighted 10th to 90th percentile."
        )

        FORECAST_PRESETS = {
            "Bakersfield": (35.37, -119.02),
            "Bear Valley": (38.49, -120.04),
            "Big Bear City": (34.26, -116.84),
            "Bishop": (37.36, -118.40),
            "Blue Canyon": (39.28, -120.71),
            "Brawley": (32.98, -115.53),
            "Brookings (OR)": (42.05, -124.28),
            "Carson City (NV)": (39.16, -119.77),
            "Crescent City": (41.76, -124.20),
            "Eureka": (40.80, -124.16),
            "Fresno": (36.74, -119.79),
            "Lodgepole": (36.60, -118.72),
            "Los Angeles": (34.05, -118.24),
            "Mammoth Lakes": (37.65, -118.97),
            "Modesto": (37.64, -121.00),
            "Monterey": (36.60, -121.89),
            "Mt Shasta": (41.31, -122.31),
            "Needles": (34.85, -114.61),
            "Oroville": (39.51, -121.56),
            "Oxnard": (34.20, -119.18),
            "Palm Springs": (33.83, -116.55),
            "Pollock Pines": (38.76, -120.59),
            "Redding": (40.59, -122.39),
            "Reno (NV)": (39.53, -119.81),
            "Ridgecrest": (35.62, -117.67),
            "Sacramento": (38.58, -121.49),
            "San Bernardino": (34.11, -117.29),
            "San Diego": (32.72, -117.16),
            "San Francisco": (37.77, -122.42),
            "San Jose": (37.34, -121.89),
            "San Luis Obispo": (35.28, -120.66),
            "Santa Barbara": (34.42, -119.70),
            "Santa Rosa": (38.44, -122.71),
            "Soda Springs": (39.32, -120.38),
            "Sonora": (37.98, -120.38),
            "South Lake Tahoe": (38.94, -119.98),
            "Susanville": (40.42, -120.65),
            "Ukiah": (39.15, -123.21),
            "Victorville": (34.54, -117.29),
            "Yosemite": (37.75, -119.58),
        }
        choices = list(FORECAST_PRESETS) + ["Custom lat/lon"]
        pc1, pc2, pc3 = st.columns([2, 1, 1])
        choice = pc1.selectbox("Location", choices, index=choices.index("Sacramento"), key="sp_location")
        if choice == "Custom lat/lon":
            lat = pc2.number_input("Latitude", min_value=32.0, max_value=42.5,
                                   value=38.58, step=0.01, format="%.2f", key="sp_lat")
            lon = pc3.number_input("Longitude", min_value=-125.0, max_value=-114.0,
                                   value=-121.49, step=0.01, format="%.2f", key="sp_lon")
        else:
            lat, lon = FORECAST_PRESETS[choice]
            pc2.text_input("Latitude", f"{lat:.2f}", disabled=True)
            pc3.text_input("Longitude", f"{lon:.2f}", disabled=True)
        point_name = choice if choice != "Custom lat/lon" else f"{lat:.2f}, {lon:.2f}"

        @st.cache_data(ttl=6 * 3600, show_spinner="Fetching WeatherNext 2, EPS and AIFS-ENS...")
        def cached_point_blend(lat, lon, day_key):
            per_model, errors = forecast.fetch_point_blend(lat, lon, forecast_days=15)
            return per_model, errors, pd.Timestamp.now(tz="UTC")

        try:
            per_model, model_errors, fetched_at = cached_point_blend(
                round(float(lat), 2), round(float(lon), 2), forecast_day_key
            )
        except Exception as exc:
            st.error(f"Forecast request failed: {exc}")
            st.stop()

        for name, msg in model_errors.items():
            st.warning(f"{name} unavailable ({msg}). The blend uses the remaining models.")
        if not per_model:
            st.error("No model returned data for this point. Try again in a few minutes.")
            st.stop()

        blend = forecast.blend_members(per_model)
        summary = forecast.blend_cumulative_summary(blend)
        member_counts = ", ".join(
            f"{n} ({df['member'].nunique()} members)" for n, df in per_model.items()
        )

        # Last complete Pacific-time day each model returned. The blend only keeps
        # days that every model has, so the shortest model sets its length.
        model_last_day = {n: df["date"].max() for n, df in per_model.items()}
        coverage_text = ", ".join(
            f"{n} to {d.strftime('%b %d')}"
            + (f" (init {inferred_init(df.attrs['data_end'])})"
               if df.attrs.get("data_end") is not None else "")
            for (n, d), df in zip(model_last_day.items(), per_model.values())
        )

        st.subheader(point_name)
        st.caption(
            f"Source: Open-Meteo · {member_counts} · "
            f"Forecast covers {blend['date'].min().strftime('%b %d, %Y')} to "
            f"{blend['date'].max().strftime('%b %d, %Y')} "
            f"({blend['date'].nunique()} days) · "
            f"Complete days available: {coverage_text} · "
            f"Retrieved {fetched_at.strftime('%b %d, %Y %H:%M')} UTC"
        )
        if blend["date"].max() < max(model_last_day.values()):
            limiting = [n for n, d in model_last_day.items() if d == blend["date"].max()]
            st.info(
                f"The blend ends {blend['date'].max().strftime('%b %d')} because "
                f"{', '.join(limiting)} has no complete Pacific-time day after that "
                "yet. It should reach the next day when that model's next run is "
                "published."
            )

        st.plotly_chart(cumulative_fig(summary, mean_label="Blend mean"),
                        use_container_width=True)
        show_forecast_table(summary)

        st.markdown("**Daily precipitation matrix**")
        use_chance, threshold_in = daily_matrix_controls("point")
        matrix = forecast.blend_daily_matrix(
            blend,
            metric="chance" if use_chance else "median",
            threshold_in=threshold_in,
        ).loc[["Blend"]]
        st.plotly_chart(daily_matrix_fig(matrix, use_chance, threshold_in),
                        use_container_width=True)
        st.caption(
            "Blend weights each model equally. Each model is read at its nearest "
            "0.25° grid cell, so a point in the mountains reflects the cell average "
            "rather than the exact spot."
        )

        st.markdown("**Member matrix**")
        mm_mode = st.radio(
            "Matrix values",
            ["Cumulative", "Daily"],
            horizontal=True,
            key="fc_member_mode_point",
        )
        blend_mean, by_model = forecast.blend_member_matrix(
            blend, cumulative=(mm_mode == "Cumulative")
        )

        st.markdown("**Blend mean**")
        st.plotly_chart(
            forecast.fig_mean_strip(
                blend_mean, mean_font=FONT_MEAN_CELLS, label_font=FONT_MEMBER_LABELS,
                date_font=FONT_DATE_LABELS, colorbar_font=FONT_COLORBAR,
            ),
            use_container_width=True,
        )
        for model_name, (m_members, m_mean) in by_model.items():
            st.markdown(f"**{model_name}** ({len(m_members)} members)")
            st.plotly_chart(member_plume_fig(m_members, m_mean), use_container_width=True)

        st.caption(
            mm_mode + " precipitation in inches for every member of each model "
            "(CONT is each model's control run). MEAN under each matrix is that "
            "model's own mean; the Blend mean at the top is the equal-weight "
            "average of the three model means."
        )
        csv_frames = [blend_mean.rename(index={"BLEND": "BLEND MEAN"})]
        for model_name, (m_members, m_mean) in by_model.items():
            csv_frames.append(m_members.rename(index=lambda i, n=model_name: f"{n} {i}"))
            csv_frames.append(m_mean.rename(index=lambda i, n=model_name: f"{n} {i}"))
        csv_df = pd.concat(csv_frames)
        st.download_button(
            "Download member matrix as CSV",
            csv_df.rename(columns=lambda d: d.strftime("%Y-%m-%d")).to_csv(index_label="Member"),
            file_name=f"blend_members_{point_name.replace(' ', '_').replace(',', '')}_{mm_mode.lower()}.csv",
            mime="text/csv",
            key="fc_member_csv_point",
        )

        st.caption(
            "Skill declines with lead time. Treat later days as a range of possibilities, "
            "not a single expected total. Data: Open-Meteo (Google WeatherNext 2, "
            "ECMWF IFS ensemble, ECMWF AIFS ensemble)."
        )

elif analysis_mode == "Extremes / Records":
    st.header("Extremes / Records")

    extreme_type = st.radio(
        "Period type:",
        [
            "Water Years",
            "Recurring Seasonal / Custom Calendar Stretch",
            "Rolling N-Day Window",
        ],
        index=0,
    )

    records_to_show = st.number_input(
        "Number of records to show",
        min_value=1,
        max_value=136,
        value=5,
        step=1,
        help="Shows the N lowest and N highest periods. 136 covers all complete years from WY 1890 through WY 2025.",
    )

    if extreme_type == "Water Years":
        run_extremes = st.button(
            "🏆 Search Water-Year Records",
            type="primary",
            use_container_width=True,
        )

    elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
        c1, c2 = st.columns(2)

        with c1:
            extreme_start_mmdd = st.text_input(
                "Start date (MM-DD)",
                value="11-01",
            )

        with c2:
            extreme_end_mmdd = st.text_input(
                "End date (MM-DD)",
                value="02-18",
            )

        run_extremes = st.button(
            "🏆 Search Seasonal Records",
            type="primary",
            use_container_width=True,
        )

    else:
        extreme_window_days = st.number_input(
            "Rolling window length (days)",
            min_value=1,
            max_value=3650,
            value=30,
            step=1,
            help="Examples: 1, 7, 30, 90, 365.",
        )

        run_extremes = st.button(
            "🏆 Search Rolling Records",
            type="primary",
            use_container_width=True,
        )

    if run_extremes:
        try:
            metadata = load_metadata(str(DATA_DIR))

            hydrological_regions = [
                r for r in selected_regions if r != "California"
            ]

            station_region_map = build_region_map(
                tuple(metadata.items()),
                tuple(hydrological_regions),
            )

            regional_ids = set(station_region_map.keys())

            # True California statewide scope: filter stations by the
            # California state polygon rather than assuming every station in
            # stations_meta.jsonl is inside California.
            california_polygon = regions["California"]["polygon"]
            statewide_ids = {
                sid
                for sid, meta in metadata.items()
                if engine.point_in_polygon(
                    (meta["lat"], meta["lon"]),
                    california_polygon,
                )
            }

            if not regional_ids and "California" not in selected_regions:
                st.error("No stations were found inside the selected hydrological region(s).")
                st.stop()

            # Search each selected hydrological region independently.
            # This prevents multiple selected regions from being combined
            # into one regional record.
            region_record_results = {}

            if extreme_type == "Water Years":
                with st.spinner(
                    f"Searching {len(hydrological_regions)} region(s) and "
                    f"{len(statewide_ids)} statewide stations across "
                    "WY 1890–2025..."
                ):
                    for region in hydrological_regions:
                        region_ids = {
                            sid for sid, mapped_region in station_region_map.items()
                            if mapped_region == region
                        }
                        region_record_results[region] = (
                            run_lightweight_water_year_records(
                                region_ids,
                                min_valid_days=100,
                                limit=int(records_to_show),
                            )
                            if region_ids else None
                        )

                    statewide_records = run_lightweight_water_year_records(
                        statewide_ids,
                        min_valid_days=100,
                        limit=int(records_to_show),
                    )

            elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
                try:
                    from datetime import datetime

                    sm, sd = map(int, extreme_start_mmdd.strip().split("-"))
                    em, ed = map(int, extreme_end_mmdd.strip().split("-"))
                    datetime(2001, sm, sd)
                    datetime(2001, em, ed)

                    extreme_start_mmdd = f"{sm:02d}-{sd:02d}"
                    extreme_end_mmdd = f"{em:02d}-{ed:02d}"
                except (ValueError, TypeError):
                    st.error("Invalid MM-DD date. Use the format MM-DD.")
                    st.stop()

                with st.spinner(
                    f"Searching {len(hydrological_regions)} region(s) and "
                    f"{len(statewide_ids)} statewide stations..."
                ):
                    for region in hydrological_regions:
                        region_ids = {
                            sid for sid, mapped_region in station_region_map.items()
                            if mapped_region == region
                        }
                        region_record_results[region] = (
                            run_lightweight_seasonal_records(
                                region_ids,
                                extreme_start_mmdd,
                                extreme_end_mmdd,
                                min_valid_ratio=0.50,
                                limit=int(records_to_show),
                            )
                            if region_ids else None
                        )

                    statewide_records = run_lightweight_seasonal_records(
                        statewide_ids,
                        extreme_start_mmdd,
                        extreme_end_mmdd,
                        min_valid_ratio=0.50,
                        limit=int(records_to_show),
                    )

            else:
                window_days = int(extreme_window_days)

                with st.spinner(
                    f"Searching {window_days}-day rolling records across "
                    "1890–2026..."
                ):
                    for region in hydrological_regions:
                        region_ids = {
                            sid for sid, mapped_region in station_region_map.items()
                            if mapped_region == region
                        }
                        region_record_results[region] = (
                            run_lightweight_rolling_records(
                                region_ids,
                                window_days,
                                min_valid_ratio=0.70,
                                limit=int(records_to_show),
                            )
                            if region_ids else None
                        )

                    statewide_records = run_lightweight_rolling_records(
                        statewide_ids,
                        window_days,
                        min_valid_ratio=0.70,
                        limit=int(records_to_show),
                    )

            extreme_normal_cache = {}

            def extreme_normal_total(scope_key, station_ids, period_start, period_end):
                cache_key = (scope_key, tuple(sorted(station_ids)))
                if cache_key not in extreme_normal_cache:
                    extreme_normal_cache[cache_key] = query_daily_normal(tuple(station_ids))
                normal_df, computed_annual = extreme_normal_cache[cache_key]
                if extreme_type == "Water Years":
                    canonical = canonical_annual_normal(scope_key)
                    return canonical if canonical is not None else computed_annual
                return normal_total_for_dates(normal_df, period_start, period_end)

            def get_ranked_records(records, scope_key, station_ids):
                if records is None or records.empty:
                    return None

                result = records.copy()

                if extreme_type == "Water Years":
                    result["period"] = result["wy_end_year"].apply(
                        lambda y: f"WY {int(y)}"
                    )
                elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
                    # For a cross-year seasonal stretch (e.g. Nov 1-Feb 18),
                    # occurrence_year is the year in which the stretch starts.
                    # Display the full seasonal span so 1997 means 1997-98,
                    # 1913 means 1913-14, etc. Non-cross-year stretches keep
                    # the single calendar year.
                    sm, sd = map(int, extreme_start_mmdd.split("-"))
                    em, ed = map(int, extreme_end_mmdd.split("-"))
                    cross_year = (sm, sd) > (em, ed)

                    if cross_year:
                        result["period"] = result["occurrence_year"].apply(
                            lambda y: f"{int(y)}-{str(int(y) + 1)[-2:]}"
                        )
                    else:
                        result["period"] = result["occurrence_year"].apply(
                            lambda y: str(int(y))
                        )
                else:
                    result["period"] = result.apply(
                        lambda row: (
                            f"{pd.Timestamp(row['period_start']).strftime('%Y-%m-%d')} "
                            f"to "
                            f"{pd.Timestamp(row['period_end']).strftime('%Y-%m-%d')}"
                        ),
                        axis=1,
                    )

                pct_values = []
                for _, row in result.iterrows():
                    if extreme_type == "Water Years":
                        start = f"{int(row['wy_end_year']) - 1}-07-01"
                        end = f"{int(row['wy_end_year'])}-06-30"
                    elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
                        start_year = int(row["occurrence_year"])
                        sm, sd = map(int, extreme_start_mmdd.split("-"))
                        em, ed = map(int, extreme_end_mmdd.split("-"))
                        start = f"{start_year}-{sm:02d}-{sd:02d}"
                        end_year = start_year + 1 if (sm, sd) > (em, ed) else start_year
                        end = f"{end_year}-{em:02d}-{ed:02d}"
                    else:
                        start = row["period_start"]
                        end = row["period_end"]

                    normal_total = extreme_normal_total(
                        scope_key, station_ids, start, end
                    )
                    pct_values.append(
                        float(row["precip"]) / normal_total * 100
                        if normal_total not in (None, 0) and pd.notna(normal_total)
                        else float("nan")
                    )
                result["pct_avg"] = pct_values

                return {
                    "lowest": (
                        result[result["record_type"] == "Lowest"]
                        .sort_values(["precip", "period"])
                        .head(int(records_to_show))
                        .reset_index(drop=True)
                    ),
                    "highest": (
                        result[result["record_type"] == "Highest"]
                        .sort_values(["precip", "period"], ascending=[False, True])
                        .head(int(records_to_show))
                        .reset_index(drop=True)
                    ),
                }

            def show_records(title, ranked):
                st.subheader(title)

                if ranked is None:
                    st.info("No valid periods found.")
                    return

                left, right = st.columns(2)

                def render_table(frame):
                    if frame.empty:
                        st.info("No valid periods found.")
                        return

                    display = pd.DataFrame({
                        "Rank": range(1, len(frame) + 1),
                        "Precipitation": frame["precip"].map(
                            lambda x: f'{float(x):.2f}"'
                        ),
                        "% Avg": frame["pct_avg"].map(
                            lambda x: f"{float(x):.1f}%" if pd.notna(x) else "N/A"
                        ),
                        "Period": frame["period"],
                        "Stations": frame["station_count"].astype(int),
                    })
                    st.dataframe(
                        display,
                        use_container_width=True,
                        hide_index=True,
                    )

                with left:
                    st.markdown("**Lowest**")
                    render_table(ranked["lowest"])

                with right:
                    st.markdown("**Highest**")
                    render_table(ranked["highest"])

            st.header("Regional Records")

            for region in hydrological_regions:
                show_records(
                    region,
                    get_ranked_records(
                        region_record_results.get(region),
                        region,
                        {sid for sid, mapped_region in station_region_map.items() if mapped_region == region},
                    ),
                )

            st.header("Statewide California Records")
            show_records(
                "All California Stations",
                get_ranked_records(
                    statewide_records,
                    "All California Stations",
                    statewide_ids,
                ),
            )

            rows = []

            for region in hydrological_regions:
                ranked = get_ranked_records(
                    region_record_results.get(region),
                    region,
                    {sid for sid, mapped_region in station_region_map.items() if mapped_region == region},
                )
                if ranked is None:
                    continue

                for rank, (_, row) in enumerate(ranked["lowest"].iterrows(), start=1):
                    rows.append({
                        "Scope": region,
                        "Record": f"Lowest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "% Avg": f'{float(row["pct_avg"]):.1f}%' if pd.notna(row["pct_avg"]) else "N/A",
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

                for rank, (_, row) in enumerate(ranked["highest"].iterrows(), start=1):
                    rows.append({
                        "Scope": region,
                        "Record": f"Highest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "% Avg": f'{float(row["pct_avg"]):.1f}%' if pd.notna(row["pct_avg"]) else "N/A",
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

            ranked = get_ranked_records(
                statewide_records,
                "All California Stations",
                statewide_ids,
            )
            if ranked is not None:
                for rank, (_, row) in enumerate(ranked["lowest"].iterrows(), start=1):
                    rows.append({
                        "Scope": "All California Stations",
                        "Record": f"Lowest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "% Avg": f'{float(row["pct_avg"]):.1f}%' if pd.notna(row["pct_avg"]) else "N/A",
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

                for rank, (_, row) in enumerate(ranked["highest"].iterrows(), start=1):
                    rows.append({
                        "Scope": "All California Stations",
                        "Record": f"Highest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "% Avg": f'{float(row["pct_avg"]):.1f}%' if pd.notna(row["pct_avg"]) else "N/A",
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

            if rows:
                record_table = pd.DataFrame(rows)

                with st.expander("View record summary data"):
                    st.dataframe(
                        record_table,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "📥 Download Record Summary (CSV)",
                        data=record_table.to_csv(index=False),
                        file_name="california_extreme_records.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

        except FileNotFoundError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Extremes / Records error: {exc}")
            with st.expander("Technical error details"):
                st.exception(exc)


# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------

st.markdown("---")

with st.expander("ℹ️ About This Tool"):
    st.markdown(
        """
        ### California Precipitation Analysis

        **Web-app modes:** Comparison Mode and Extremes / Records

        The Water Year calculations are performed by the same DuckDB
        calculation engine used by `compare.py`.

        The results include:

        - Regional Comparison Summary Table
        - Average precipitation
        - Percent of regional WY baseline
        - Common-station filtering across all selected WYs
        - Detailed station breakdown
        - Statewide average when multiple regions are selected
        - Side-by-side regional precipitation matrix
        - CSV export of the station table

        **Data source:** ACIS daily precipitation data.
        """
    )

st.caption("Built with Streamlit + DuckDB")
