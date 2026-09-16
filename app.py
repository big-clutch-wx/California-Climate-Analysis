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


def query_daily_comparison_series(station_ids, period_specs):
    """Return daily regional-mean precipitation for comparison charting.

    Each period is evaluated using the same station set that survived the
    comparison's strict common-station filter. The chart therefore shows the
    same regional population as the summary tables.
    """
    if not station_ids or not period_specs:
        return pd.DataFrame()

    ids_str = ", ".join([f"'{sid}'" for sid in station_ids])
    union_queries = []

    for idx, spec in enumerate(period_specs):
        union_queries.append(f"""
            SELECT
                {idx} AS period_index,
                '{spec['label'].replace("'", "''")}' AS period_label,
                CAST(date AS DATE) AS date,
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

    sql = """
        WITH raw AS (
            {union}
        ),
        daily AS (
            SELECT
                period_index,
                period_label,
                date,
                AVG(precip) AS daily_precip,
                COUNT(precip) AS station_count
            FROM raw
            WHERE precip IS NOT NULL
            GROUP BY period_index, period_label, date
        )
        SELECT * FROM daily
        ORDER BY period_index, date
    """.format(union=" UNION ALL ".join(union_queries))

    con = duckdb.connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def query_daily_normal(station_ids, start_date, end_date):
    """Calculate a 1991-2020 daily precipitation normal for the chart.

    The normal is computed from the same station population, using the
    calendar-day mean for each month/day and then mapped onto the selected
    comparison period.
    """
    if not station_ids:
        return pd.DataFrame()

    ids_str = ", ".join([f"'{sid}'" for sid in station_ids])
    sql = f"""
        WITH raw AS (
            SELECT
                CAST(date AS DATE) AS date,
                CASE
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS DOUBLE)
                END AS precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '1991-01-01'
              AND CAST(date AS DATE) <= DATE '2020-12-31'
        ),
        daily_station AS (
            SELECT date, AVG(precip) AS daily_value
            FROM raw
            WHERE precip IS NOT NULL
            GROUP BY date
        )
        SELECT
            EXTRACT(MONTH FROM date)::INTEGER AS month,
            EXTRACT(DAY FROM date)::INTEGER AS day,
            AVG(daily_value) AS normal_daily_precip
        FROM daily_station
        GROUP BY 1, 2
        ORDER BY 1, 2
    """

    con = duckdb.connect()
    try:
        return con.execute(sql).df()
    finally:
        con.close()


def render_xmacis_style_chart(daily_df, station_ids, chart_title, show_normal=True):
    """Render an XMACIS-style cumulative precipitation chart with hover data."""
    if daily_df is None or daily_df.empty:
        st.info("Daily precipitation data are not available for this chart.")
        return

    import plotly.graph_objects as go

    fig = go.Figure()

    # Put every comparison period onto the same synthetic calendar while
    # preserving the actual chronological order.  This is important for
    # water years such as Jul-Jun: January must come AFTER December, not
    # jump backward to the beginning of the x-axis.
    base_year = 2000  # leap year, so Feb 29 can be represented naturally

    def synthetic_dates(dates):
        dates = pd.to_datetime(dates)
        first_date = dates.iloc[0]
        first_year = int(first_date.year)
        result = []
        for d in dates:
            year_offset = int(d.year) - first_year
            result.append(pd.Timestamp(base_year + year_offset, int(d.month), int(d.day)))
        return result

    for period_index, group in daily_df.groupby("period_index", sort=False):
        group = group.sort_values("date").copy()
        group["date"] = pd.to_datetime(group["date"])
        group["cumulative"] = group["daily_precip"].fillna(0.0).cumsum()
        label = str(group["period_label"].iloc[0])
        x_values = synthetic_dates(group["date"])

        customdata = list(zip(
            group["date"].dt.strftime("%b %d, %Y"),
            group["daily_precip"].round(2),
            group["cumulative"].round(2),
            group["station_count"].astype(int),
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
                "Accumulation: %{customdata[2]:.2f}\"<br>"
                "Stations reporting: %{customdata[3]}<extra></extra>"
            ),
            line={"width": 2.5},
        ))

    # Add the daily normal as a cumulative reference line.
    if show_normal and station_ids:
        first_index = daily_df["period_index"].min()
        first = daily_df[daily_df["period_index"] == first_index].sort_values("date").copy()
        first["date"] = pd.to_datetime(first["date"])
        if not first.empty:
            normal_df = query_daily_normal(
                tuple(station_ids),
                first["date"].min().strftime("%Y-%m-%d"),
                first["date"].max().strftime("%Y-%m-%d"),
            )
            if not normal_df.empty:
                normal_lookup = {
                    (int(r.month), int(r.day)): float(r.normal_daily_precip)
                    for r in normal_df.itertuples(index=False)
                }
                cumulative = 0.0
                normal_x = []
                normal_y = []
                first_year = int(first["date"].iloc[0].year)
                for d in first["date"]:
                    cumulative += normal_lookup.get((int(d.month), int(d.day)), 0.0)
                    year_offset = int(d.year) - first_year
                    normal_x.append(pd.Timestamp(
                        base_year + year_offset, int(d.month), int(d.day)
                    ))
                    normal_y.append(cumulative)

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
    fig.update_xaxes(
        tickformat="%b %-d",
        dtick="D7",
        showgrid=True,
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

analysis_mode = st.sidebar.radio(
    "Mode:",
    ["Comparison Mode", "Extremes / Records"],
    index=0,
)

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

if not selected_regions:
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
                avg = 23.5
                label = "Statewide California baseline"
            else:
                avg = regions[region]["avg_precip"]
                label = "Hydrological region baseline"

            st.info(
                f"**{region}**\n\n"
                f"**{avg:.1f}\"** average annual precipitation\n\n"
                f"{label}"
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

            if not matching_ids:
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

                all_periods = df_res["period_label"].unique()
                stations_in_all_periods = (
                    df_res.groupby("station_id")["period_label"]
                    .nunique()
                    .loc[lambda x: x == len(all_periods)]
                    .index
                )

                df_res = df_res[
                    df_res["station_id"].isin(stations_in_all_periods)
                ].copy()

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
                            f"{start_mmdd}–{end_mmdd} ({y}-{y + 1})"
                            if crosses_year
                            else f"{start_mmdd}–{end_mmdd} ({y})"
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

            regional_station_count = (
                len(stations_in_all_periods) if not df_res.empty else 0
            )
            st.success(
                f"Analysis complete — {regional_station_count} stations "
                "have valid data in every selected period/range for the "
                "hydrological-region analysis."
            )

            st.header("Regional Comparison Summary Table")

            # California is a separate statewide scope and is always shown
            # first when selected. It does not consume stations from the
            # hydrological-region summaries below.
            if california_selected:
                ca_summary = (
                    build_statewide_summary(statewide_df)
                    if statewide_df is not None and not statewide_df.empty
                    else pd.DataFrame()
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
                    ca_table.columns = ["Period", "Avg Precip", "% WY Avg"]
                    ca_table["Avg Precip"] = ca_table["Avg Precip"].map(
                        lambda x: f'{x:.2f}"'
                    )
                    ca_table["% WY Avg"] = ca_table["% WY Avg"].map(
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
                    table.columns = ["Period", "Avg Precip", "% WY Avg", "Stations"]
                else:
                    station_count = int(region_data["station_count"].iloc[0])
                    st.subheader(f"{region} ({station_count} stations)")

                    table = region_data[
                        ["period_label", "avg_precip", "pct_base"]
                    ].copy()
                    table.columns = ["Period", "Avg Precip", "% WY Avg"]
                table["Avg Precip"] = table["Avg Precip"].map(
                    lambda x: f'{x:.2f}"'
                )
                table["% WY Avg"] = table["% WY Avg"].map(
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
                "Interactive daily precipitation chart. Hover over the lines "
                "for the daily value, cumulative accumulation, and stations "
                "reporting. The normal line uses 1991–2020 daily climatology."
            )

            chart_regions = []
            if california_selected and statewide_df is not None and not statewide_df.empty:
                chart_regions.append((
                    "California",
                    tuple(statewide_df["station_id"].drop_duplicates().tolist()),
                ))

            for region in hydrological_regions:
                region_ids = tuple(
                    df_res.loc[df_res["region"] == region, "station_id"]
                    .drop_duplicates()
                    .tolist()
                )
                if region_ids:
                    chart_regions.append((region, region_ids))

            for region, region_ids in chart_regions:
                st.subheader(region)
                daily_chart_df = query_daily_comparison_series(
                    region_ids, period_specs
                )
                render_xmacis_style_chart(
                    daily_chart_df,
                    region_ids,
                    f"{region} — Daily Precipitation Accumulation",
                    show_normal=True,
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

            # ---------------------------------------------------------------
            # Detailed station breakdown
            # ---------------------------------------------------------------

            if show_detail:
                st.header("Detailed Station Breakdown")

                detail = df_res.pivot(
                    index="station_id",
                    columns="period_label",
                    values="total_precip",
                )

                detail["station_name"] = [
                    metadata.get(sid, {}).get("name", sid)
                    for sid in detail.index
                ]
                detail["region"] = [
                    station_region_map.get(sid, "Unknown")
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

else:
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

            def get_ranked_records(records):
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
                    get_ranked_records(region_record_results.get(region)),
                )

            st.header("Statewide California Records")
            show_records(
                "All California Stations",
                get_ranked_records(statewide_records),
            )

            rows = []

            for region in hydrological_regions:
                ranked = get_ranked_records(region_record_results.get(region))
                if ranked is None:
                    continue

                for rank, (_, row) in enumerate(ranked["lowest"].iterrows(), start=1):
                    rows.append({
                        "Scope": region,
                        "Record": f"Lowest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

                for rank, (_, row) in enumerate(ranked["highest"].iterrows(), start=1):
                    rows.append({
                        "Scope": region,
                        "Record": f"Highest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

            ranked = get_ranked_records(statewide_records)
            if ranked is not None:
                for rank, (_, row) in enumerate(ranked["lowest"].iterrows(), start=1):
                    rows.append({
                        "Scope": "All California Stations",
                        "Record": f"Lowest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
                        "Period": row["period"],
                        "Stations": int(row["station_count"]),
                    })

                for rank, (_, row) in enumerate(ranked["highest"].iterrows(), start=1):
                    rows.append({
                        "Scope": "All California Stations",
                        "Record": f"Highest #{rank}",
                        "Precip": f'{float(row["precip"]):.2f}"',
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
