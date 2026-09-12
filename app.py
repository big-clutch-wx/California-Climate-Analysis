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
from pathlib import Path

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

    statewide["pct_base"] = (statewide["avg_precip"] / 26.0) * 100
    return statewide


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("🌧️ California Climate")
st.sidebar.caption("California precipitation analysis")

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
st.sidebar.radio(
    "Mode:",
    ["Water Years (WY)"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Data directory: `{DATA_DIR}`")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("🌧️ California Precipitation Analysis")
st.markdown(
    "**Water Year (WY) comparison** — July 1 through June 30"
)

st.info(
    "WY labels use the water-year convention: WY 1994 is displayed as "
    "**1994-95**."
)

if not selected_regions:
    st.warning("Select at least one hydrological region.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    years_input = st.text_input(
        "Water years",
        value="1997, 1994",
        help="Examples: 1982, 1997, 2015 or 2015-2020",
    )

with col2:
    min_valid_days = st.number_input(
        "Minimum valid days per WY",
        min_value=1,
        max_value=366,
        value=200,
        step=1,
        help=(
            "For modern data this defaults to the 200-day threshold used "
            "by compare(3).py. Historical WYs can automatically use 100."
        ),
    )

show_detail = st.checkbox(
    "Show detailed station breakdown",
    value=True,
)

run_analysis = st.button(
    "🔍 Run Water Year Analysis",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# Run WY analysis
# ---------------------------------------------------------------------------

if run_analysis:

    try:
        metadata = load_metadata(str(DATA_DIR))

        years = parse_years_web(years_input)

        if not years:
            st.error("No valid water years were entered.")
            st.stop()

        if any(y > 2025 for y in years):
            st.warning(
                "WY 2026 is incomplete in the current dataset. "
                "The CLI extremes mode excludes it, but WY comparison mode "
                "can still query it if requested."
            )

        station_region_map = build_region_map(
            tuple(metadata.items()),
            tuple(selected_regions),
        )

        matching_ids = set(station_region_map.keys())

        if not matching_ids:
            st.error("No stations were found inside the selected region(s).")
            st.stop()

        # Match compare(3).py's threshold behavior:
        # historical selections (<1950) use 100 valid days; otherwise 200.
        effective_min_valid = (
            100 if any(y < 1950 for y in years) else int(min_valid_days)
        )

        with st.spinner(
            f"Querying {len(matching_ids)} stations across "
            f"{len(years)} water year(s)..."
        ):
            # This calls the exact DuckDB WY engine from compare(3).py.
            df_res = engine.run_duckdb_water_years(
                matching_ids,
                years,
                min_valid_days=effective_min_valid,
            )

        if df_res is None or df_res.empty:
            st.warning(
                "No station data met the minimum valid-day threshold "
                "for the selected water years."
            )
            st.stop()

        # Map every station to its selected hydrological region.
        df_res["region"] = df_res["station_id"].map(station_region_map)

        # -------------------------------------------------------------------
        # Exact compare(3).py common-station filtering
        # -------------------------------------------------------------------
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

        if df_res.empty:
            st.warning(
                "No stations have valid data in every selected water year."
            )
            st.stop()

        # -------------------------------------------------------------------
        # Regional Comparison Summary Table
        # -------------------------------------------------------------------

        summary = build_summary(
            df_res,
            regions,
            station_region_map,
        )

        st.success(
            f"Analysis complete — {len(stations_in_all_periods)} stations "
            f"have data in every selected WY."
        )

        st.header("Regional Comparison Summary Table")

        for region in selected_regions:

            region_data = (
                summary[summary["region"] == region]
                .sort_values("period_id")
                .copy()
            )

            if region_data.empty:
                st.subheader(f"{region} (0 stations)")
                st.info("No valid station data for this region.")
                continue

            # In WY comparison mode, compare(3).py uses the station count
            # from the first period after the global common-station filter.
            station_count = int(region_data["station_count"].iloc[0])

            st.subheader(f"{region} ({station_count} stations)")

            table = region_data[
                ["period_label", "avg_precip", "pct_base"]
            ].copy()

            table.columns = [
                "Period",
                "Avg Precip",
                "% WY Base",
            ]

            table["Avg Precip"] = table["Avg Precip"].map(
                lambda x: f'{x:.2f}"'
            )
            table["% WY Base"] = table["% WY Base"].map(
                lambda x: f"{x:.1f}%"
            )

            st.dataframe(
                table,
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------------------
        # Statewide average and side-by-side matrix for multiple regions
        # -------------------------------------------------------------------

        if len(selected_regions) > 1:

            st.header("Statewide Average (All Stations)")

            statewide = build_statewide_summary(df_res)

            statewide_table = statewide[
                ["period_label", "avg_precip", "pct_base"]
            ].copy()

            statewide_table.columns = [
                "Period",
                "Avg Precip",
                "% WY Base",
            ]

            statewide_table["Avg Precip"] = statewide_table[
                "Avg Precip"
            ].map(lambda x: f'{x:.2f}"')

            statewide_table["% WY Base"] = statewide_table[
                "% WY Base"
            ].map(lambda x: f"{x:.1f}%")

            st.dataframe(
                statewide_table,
                use_container_width=True,
                hide_index=True,
            )

            st.header(
                "Side-by-Side Regional Precipitation Matrix (Inches)"
            )

            matrix = summary.pivot(
                index="region",
                columns="period_label",
                values="avg_precip",
            )

            # Match compare(3).py: statewide average is calculated from
            # individual stations, not from regional means.
            statewide_avg = (
                df_res.groupby("period_label")["total_precip"]
                .mean()
            )

            matrix.loc["STATEWIDE AVERAGE"] = statewide_avg

            matrix = matrix.round(2)
            matrix.index.name = "Region"

            st.dataframe(
                matrix,
                use_container_width=True,
            )

        # -------------------------------------------------------------------
        # Detailed station breakdown
        # -------------------------------------------------------------------

        if show_detail:

            st.header("Detailed Station Breakdown")

            detail_table = make_detail_table(
                df_res,
                metadata,
                station_region_map,
            )

            st.caption(
                f"All {len(detail_table)} stations with valid data in "
                f"every selected WY, sorted by average precipitation "
                f"across the selected water years."
            )

            st.dataframe(
                detail_table,
                use_container_width=True,
                hide_index=True,
            )

            # Download exactly the displayed station-level precipitation
            # table as CSV.
            csv = detail_table.to_csv(index=False)

            st.download_button(
                "📥 Download Station Breakdown (CSV)",
                data=csv,
                file_name="california_wy_station_breakdown.csv",
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
# About
# ---------------------------------------------------------------------------

st.markdown("---")

with st.expander("ℹ️ About This Tool"):
    st.markdown(
        """
        ### California Precipitation Analysis

        **Current web-app mode:** Full Water Years (July 1–June 30)

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
