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
st.sidebar.caption(f"Data directory: `{DATA_DIR}`")


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.title("🌧️ California Precipitation Analysis")

if not selected_regions:
    st.warning("Select at least one hydrological region.")
    st.stop()

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
            c1, c2 = st.columns(2)

            with c1:
                start_date = st.date_input(
                    "Start date",
                    value=__import__("datetime").date(1981 + i, 1, 1),
                    key=f"custom_start_{i}",
                )

            with c2:
                end_date = st.date_input(
                    "End date",
                    value=__import__("datetime").date(1981 + i, 3, 31),
                    key=f"custom_end_{i}",
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

            station_region_map = build_region_map(
                tuple(metadata.items()),
                tuple(selected_regions),
            )
            matching_ids = set(station_region_map.keys())

            if not matching_ids:
                st.error("No stations were found inside the selected region(s).")
                st.stop()

            # ---------------------------------------------------------------
            # Run the selected comparison engine
            # ---------------------------------------------------------------

            if comparison_mode == "Full Water Years":
                wy_end_years = parse_years_web(years_input)

                if not wy_end_years:
                    st.error("No valid water years were entered.")
                    st.stop()

                # compare.py expects the starting year:
                # 1997-07-01 through 1998-06-30 is passed as 1997.
                engine_years = [y - 1 for y in wy_end_years]

                if any(y > 2025 for y in wy_end_years):
                    st.warning(
                        "WY 2026 is incomplete in the current dataset."
                    )

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

            if df_res is None or df_res.empty:
                st.warning(
                    "No consistent station data found across all specified periods."
                )
                st.stop()

            # Map station to hydrological region.
            df_res["region"] = df_res["station_id"].map(station_region_map)

            # Always use the intersection of stations across ALL periods.
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
                    "No stations have valid data in every selected period."
                )
                st.stop()

            # ---------------------------------------------------------------
            # Regional summary
            # ---------------------------------------------------------------

            summary = build_summary(
                df_res,
                regions,
                station_region_map,
            )

            st.success(
                f"Analysis complete — {len(stations_in_all_periods)} stations "
                f"have valid data in every selected period."
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

                station_count = int(region_data["station_count"].iloc[0])
                st.subheader(f"{region} ({station_count} stations)")

                table = region_data[
                    ["period_label", "avg_precip", "pct_base"]
                ].copy()
                table.columns = ["Period", "Avg Precip", "% WY Base"]

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

            # ---------------------------------------------------------------
            # Statewide + matrix
            # ---------------------------------------------------------------

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

                st.header("Side-by-Side Regional Precipitation Matrix (Inches)")

                matrix = summary.pivot(
                    index="region",
                    columns="period_label",
                    values="avg_precip",
                )

                statewide_avg = df_res.groupby("period_label")[
                    "total_precip"
                ].mean()
                matrix.loc["STATEWIDE AVERAGE"] = statewide_avg

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
                ].dropna(subset=precip_cols).copy()

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

    st.markdown(
        "**Extremes / Records** searches the complete available "
        "1890–2026 precipitation dataset for the lowest and highest "
        "average precipitation totals for the selected period definition."
    )

    extreme_type = st.radio(
        "Period type:",
        [
            "Water Years",
            "Recurring Seasonal / Custom Calendar Stretch",
            "Rolling N-Day Window",
        ],
        index=0,
        help=(
            "These correspond directly to the three period types in "
            "compare.py's Extremes / Records mode."
        ),
    )

    if extreme_type == "Water Years":
        st.info(
            "Records use complete water years only: **WY 1890 through WY 2025**. "
            "WY 2026 is excluded because it is incomplete."
        )

        run_extremes = st.button(
            "🏆 Search Water-Year Records",
            type="primary",
            use_container_width=True,
        )

    elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
        st.markdown(
            "Search every occurrence of the same calendar stretch from "
            "**1890 through 2026**. Cross-year stretches (for example "
            "**11-12 to 02-18**) run through the following calendar year, "
            "so their final start year is 2025."
        )

        c1, c2 = st.columns(2)

        with c1:
            extreme_start_mmdd = st.text_input(
                "Start date (MM-DD)",
                value="11-01",
                help="Example: 11-12",
            )

        with c2:
            extreme_end_mmdd = st.text_input(
                "End date (MM-DD)",
                value="02-18",
                help="Example: 02-18 for a cross-year stretch.",
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

        st.info(
            "Searches every rolling window from **1890-01-01 through "
            "2026-12-31**, requiring at least 70% of the window's days "
            "to contain valid precipitation observations."
        )

        run_extremes = st.button(
            "🏆 Search Rolling Records",
            type="primary",
            use_container_width=True,
        )

    if run_extremes:
        try:
            metadata = load_metadata(str(DATA_DIR))

            station_region_map = build_region_map(
                tuple(metadata.items()),
                tuple(selected_regions),
            )

            regional_ids = set(station_region_map.keys())
            statewide_ids = set(metadata.keys())

            if not regional_ids:
                st.error("No stations were found inside the selected region(s).")
                st.stop()

            # ---------------------------------------------------------------
            # Run the same three extremes engines used by compare.py
            # ---------------------------------------------------------------

            if extreme_type == "Water Years":
                extreme_years = list(range(1890, 2026))

                with st.spinner(
                    f"Searching {len(regional_ids)} regional stations and "
                    f"{len(statewide_ids)} statewide stations across "
                    "WY 1890–2025..."
                ):
                    regional_df = engine.run_duckdb_water_years(
                        regional_ids,
                        extreme_years,
                        min_valid_days=100,
                    )
                    statewide_df = engine.run_duckdb_water_years(
                        statewide_ids,
                        extreme_years,
                        min_valid_days=100,
                    )

            elif extreme_type == "Recurring Seasonal / Custom Calendar Stretch":
                # Validate MM-DD exactly as in compare.py.
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

                cross_year = (sm, sd) > (em, ed)
                last_start_year = 2025 if cross_year else 2026
                extreme_years = list(range(1890, last_start_year + 1))

                with st.spinner(
                    f"Searching {len(regional_ids)} regional stations and "
                    f"{len(statewide_ids)} statewide stations across "
                    f"{len(extreme_years)} occurrences..."
                ):
                    # Extremes mode deliberately matches compare.py:
                    # 50% minimum valid data and no strict all-year
                    # consistency requirement.
                    regional_df = engine.run_duckdb_custom_stretches(
                        regional_ids,
                        extreme_start_mmdd,
                        extreme_end_mmdd,
                        extreme_years,
                        min_valid_ratio=0.50,
                        strict_consistency=False,
                    )
                    statewide_df = engine.run_duckdb_custom_stretches(
                        statewide_ids,
                        extreme_start_mmdd,
                        extreme_end_mmdd,
                        extreme_years,
                        min_valid_ratio=0.50,
                        strict_consistency=False,
                    )

            else:
                window_days = int(extreme_window_days)

                with st.spinner(
                    f"Searching every {window_days}-day window across "
                    "1890-01-01 through 2026-12-31..."
                ):
                    regional_df = engine.run_duckdb_rolling_extremes(
                        regional_ids,
                        window_days,
                        min_valid_ratio=0.70,
                    )
                    statewide_df = engine.run_duckdb_rolling_extremes(
                        statewide_ids,
                        window_days,
                        min_valid_ratio=0.70,
                    )

                # Make rolling results compatible with the same record
                # summarization used for the other two period types.
                for df in (regional_df, statewide_df):
                    if df is not None and not df.empty:
                        df["period_label"] = (
                            df["period_start"].dt.strftime("%Y-%m-%d")
                            + " to "
                            + df["period_end"].dt.strftime("%Y-%m-%d")
                        )

            # ---------------------------------------------------------------
            # Display helpers
            # ---------------------------------------------------------------

            def extreme_stats(df):
                if df is None or df.empty:
                    return None

                stats = (
                    df.groupby("period_label")
                    .agg(
                        precip=("total_precip", "mean"),
                        station_count=("station_id", "count"),
                    )
                    .reset_index()
                )

                if stats.empty:
                    return None

                low = stats.loc[stats["precip"].idxmin()]
                high = stats.loc[stats["precip"].idxmax()]

                return {
                    "lowest_period": str(low["period_label"]),
                    "lowest_precip": float(low["precip"]),
                    "lowest_stations": int(low["station_count"]),
                    "highest_period": str(high["period_label"]),
                    "highest_precip": float(high["precip"]),
                    "highest_stations": int(high["station_count"]),
                }

            def display_record_card(title, stats):
                st.subheader(title)

                if stats is None:
                    st.info("No valid periods found.")
                    return

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Lowest",
                        f'{stats["lowest_precip"]:.2f}"',
                    )
                    st.caption(
                        f'{stats["lowest_period"]} '
                        f'({stats["lowest_stations"]} stations)'
                    )

                with c2:
                    st.metric(
                        "Highest",
                        f'{stats["highest_precip"]:.2f}"',
                    )
                    st.caption(
                        f'{stats["highest_period"]} '
                        f'({stats["highest_stations"]} stations)'
                    )

            # ---------------------------------------------------------------
            # Regional records — exactly the same grouping concept as
            # print_extreme_table() in compare.py.
            # ---------------------------------------------------------------

            if regional_df is None or regional_df.empty:
                st.warning("No valid regional periods were found.")
            else:
                regional_df = regional_df.copy()
                regional_df["region"] = regional_df["station_id"].map(
                    station_region_map
                )

                st.header("Regional Records")

                for region in selected_regions:
                    reg = regional_df[
                        regional_df["region"] == region
                    ]

                    stats = extreme_stats(reg)

                    if stats is None:
                        st.subheader(f"{region}")
                        st.info("No valid periods found.")
                    else:
                        display_record_card(region, stats)

            # ---------------------------------------------------------------
            # Statewide records — same metadata-wide station universe as
            # compare.py.
            # ---------------------------------------------------------------

            st.header("Statewide California Records")
            display_record_card(
                "All California Stations",
                extreme_stats(statewide_df),
            )

            # ---------------------------------------------------------------
            # Optional record data tables
            # ---------------------------------------------------------------

            if regional_df is not None and not regional_df.empty:
                with st.expander("View regional record summary data"):
                    regional_summary_rows = []

                    for region in selected_regions:
                        reg = regional_df[
                            regional_df["region"] == region
                        ]
                        stats = extreme_stats(reg)

                        if stats is not None:
                            regional_summary_rows.append(
                                {
                                    "Region": region,
                                    "Lowest Precip": (
                                        f'{stats["lowest_precip"]:.2f}"'
                                    ),
                                    "Lowest Period": stats["lowest_period"],
                                    "Lowest Stations": stats["lowest_stations"],
                                    "Highest Precip": (
                                        f'{stats["highest_precip"]:.2f}"'
                                    ),
                                    "Highest Period": stats["highest_period"],
                                    "Highest Stations": stats["highest_stations"],
                                }
                            )

                    if regional_summary_rows:
                        st.dataframe(
                            pd.DataFrame(regional_summary_rows),
                            use_container_width=True,
                            hide_index=True,
                        )

            if statewide_df is not None and not statewide_df.empty:
                with st.expander("View statewide record summary data"):
                    statewide_stats = extreme_stats(statewide_df)

                    if statewide_stats is not None:
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "Record": "Lowest",
                                        "Precip": (
                                            f'{statewide_stats["lowest_precip"]:.2f}"'
                                        ),
                                        "Period": statewide_stats["lowest_period"],
                                        "Stations": statewide_stats["lowest_stations"],
                                    },
                                    {
                                        "Record": "Highest",
                                        "Precip": (
                                            f'{statewide_stats["highest_precip"]:.2f}"'
                                        ),
                                        "Period": statewide_stats["highest_period"],
                                        "Stations": statewide_stats["highest_stations"],
                                    },
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

            st.success(
                "Extremes / Records search complete using the same DuckDB "
                "analysis functions as compare.py."
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
        calculation engine used by `compare(3).py`.

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
