#!/usr/bin/env python3
"""
California Precipitation & Snowfall Analysis - Streamlit Web App
"""

import streamlit as st
import pandas as pd
import duckdb
import glob
from datetime import datetime

st.set_page_config(
    page_title="California Climate Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🌧️ ❄️ California Climate")
st.sidebar.markdown("Precipitation & Snowfall Analysis (1950-2026)")
st.sidebar.markdown("---")

dataset = st.sidebar.radio("Select Dataset:", ["Precipitation", "Snowfall"], key="dataset_select")
st.sidebar.markdown("---")

if dataset == "Precipitation":
    st.sidebar.markdown("### Hydrological Regions")
    all_regions = ['North Coast', 'Sacramento River', 'North Lahontan', 'San Francisco Bay', 
                   'San Joaquin River', 'Tulare Lake', 'Central Coast', 'South Lahontan', 'South Coast', 'Colorado River']
    default_region = 'Sacramento River'
else:
    st.sidebar.markdown("### Sierra Watersheds")
    all_regions = ['Northern Central Valley Drainage', 'Mid-Central Valley Drainage', 'Southern Central Valley Drainage Area']
    default_region = 'Northern Central Valley Drainage'

selected_regions = st.sidebar.multiselect("Select regions:", all_regions, default=[default_region])
st.sidebar.markdown("---")

mode = st.sidebar.radio("Select mode:", ["Full Water Years", "Recurring Seasonal Stretch", "Custom Date Ranges"], key="mode_select")

st.title("🌧️ ❄️ California Climate Analysis")
st.markdown(f"**Dataset:** {dataset} | **Mode:** {mode}")
st.markdown("---")
st.markdown("### Configure Analysis")

if mode == "Full Water Years":
    col1, col2 = st.columns(2)
    with col1:
        years_input = st.text_input("Water years (e.g., '1980, 1995, 2010' or '2015-2020'):", "2000, 2010, 2020", key="years_input")
    with col2:
        min_valid = st.number_input("Min valid days per year:", min_value=1, value=200 if dataset == "Precipitation" else 50)

elif mode == "Recurring Seasonal Stretch":
    col1, col2, col3 = st.columns(3)
    with col1:
        start_date = st.text_input("Start (MM-DD):", "11-01", key="start_date")
    with col2:
        end_date = st.text_input("End (MM-DD):", "03-31", key="end_date")
    with col3:
        years_input = st.text_input("Years:", "2000, 2010, 2020", key="years_seasonal")

else:
    st.info("📅 Enter date ranges (one per line, format: YYYY-MM-DD to YYYY-MM-DD)")
    date_ranges = st.text_area("Date ranges:", "2000-01-01 to 2000-12-31\n2010-01-01 to 2010-12-31", height=100)

st.markdown("---")
run_analysis = st.button("🔍 Run Analysis", use_container_width=True, key="run_btn")

if run_analysis:
    if not selected_regions:
        st.error("❌ Please select at least one region/watershed!")
    else:
        with st.spinner("Running analysis... please wait"):
            try:
                def parse_years(years_str):
                    years = set()
                    for part in years_str.split(','):
                        part = part.strip()
                        if '-' in part:
                            start, end = part.split('-')
                            years.update(range(int(start), int(end) + 1))
                        else:
                            if part:
                                years.add(int(part))
                    return sorted([y for y in years if 1950 <= y <= 2026])
                
                con = duckdb.connect()
                
                parquet_pattern = "daily_*.parquet" if dataset == "Precipitation" else "snowfall_*.parquet"
                col_name = "precip" if dataset == "Precipitation" else "snow"
                
                parquet_files = glob.glob(parquet_pattern)
                
                if not parquet_files:
                    st.error(f"❌ No parquet files found! Make sure {parquet_pattern} files are in the repo.")
                else:
                    if mode == "Full Water Years":
                        years = parse_years(years_input)
                        
                        if not years:
                            st.error("❌ Invalid year format")
                        else:
                            union_queries = []
                            for wy in years:
                                start_date = f"{wy}-07-01"
                                end_date = f"{wy + 1}-06-30"
                                
                                union_queries.append(f"""
                                    SELECT 
                                        '{wy}' as water_year,
                                        station_id,
                                        station_name,
                                        COUNT(*) as total_days,
                                        SUM(CASE WHEN "{col_name}" IS NOT NULL THEN 1 ELSE 0 END) as valid_days,
                                        ROUND(AVG(CAST(COALESCE("{col_name}", 0) AS FLOAT)), 2) as avg_value,
                                        ROUND(MAX(CAST(COALESCE("{col_name}", 0) AS FLOAT)), 2) as max_value
                                    FROM read_parquet('{parquet_pattern}', union_by_name=true)
                                    WHERE CAST(date AS DATE) >= DATE '{start_date}' 
                                      AND CAST(date AS DATE) <= DATE '{end_date}'
                                    GROUP BY station_id, station_name
                                """)
                            
                            full_query = " UNION ALL ".join(union_queries)
                            
                            try:
                                df_res = con.execute(full_query).df()
                                
                                if len(df_res) > 0:
                                    st.success("✅ Analysis Complete!")
                                    st.markdown("### Results")
                                    st.dataframe(df_res, use_container_width=True, hide_index=True)
                                    
                                    csv = df_res.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results (CSV)",
                                        data=csv,
                                        file_name=f"analysis_{dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    st.warning("⚠️ No data found for selected years and regions")
                            
                            except Exception as e:
                                st.error(f"❌ Query error: {str(e)}")
                    
                    else:
                        st.info("Seasonal and custom date range modes coming soon!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.markdown("---")
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    ### California Climate Analysis Tool
    - **Precipitation:** 328 stations, 10 regions, 1950-2026
    - **Snowfall:** 262 stations, 3 Sierra watersheds, 1950-2026
    - **Data Source:** ACIS (Applied Climate Information System)
    """)

st.markdown("**Built with:** Streamlit + DuckDB")
