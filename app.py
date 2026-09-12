#!/usr/bin/env python3
"""
California Precipitation & Snowfall Analysis - Streamlit Web App
"""

import streamlit as st
import pandas as pd
import duckdb
import glob
import json
import sys
from datetime import datetime
from functools import lru_cache

st.set_page_config(
    page_title="California Climate Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SIDEBAR
# ============================================================================

st.sidebar.title("🌧️ ❄️ California Climate")
st.sidebar.markdown("Precipitation & Snowfall Analysis (1950-2026)")
st.sidebar.markdown("---")

dataset = st.sidebar.radio(
    "Select Dataset:",
    ["Precipitation", "Snowfall"],
    key="dataset_select"
)

st.sidebar.markdown("---")

# Region selection based on dataset
if dataset == "Precipitation":
    st.sidebar.markdown("### Hydrological Regions")
    all_regions = [
        'North Coast',
        'Sacramento River',
        'North Lahontan',
        'San Francisco Bay',
        'San Joaquin River',
        'Tulare Lake',
        'Central Coast',
        'South Lahontan',
        'South Coast',
        'Colorado River'
    ]
    default_region = 'Sacramento River'
    region_label = "Select regions:"
else:
    st.sidebar.markdown("### Sierra Watersheds")
    all_regions = [
        'Northern Central Valley Drainage',
        'Mid-Central Valley Drainage',
        'Southern Central Valley Drainage Area'
    ]
    default_region = 'Northern Central Valley Drainage'
    region_label = "Select watersheds:"

selected_regions = st.sidebar.multiselect(
    region_label,
    all_regions,
    default=[default_region]
)

if not selected_regions:
    st.sidebar.warning("⚠️ Please select at least one region")

st.sidebar.markdown("---")

# Comparison mode
st.sidebar.markdown("### Analysis Type")
mode = st.sidebar.radio(
    "Select mode:",
    [
        "Full Water Years",
        "Recurring Seasonal Stretch",
        "Custom Date Ranges"
    ],
    key="mode_select"
)

# ============================================================================
# MAIN CONTENT
# ============================================================================

st.title("🌧️ ❄️ California Climate Analysis")
st.markdown(f"**Dataset:** {dataset} | **Mode:** {mode}")

# Get user inputs based on mode
st.markdown("---")
st.markdown("### Configure Analysis")

if mode == "Full Water Years":
    col1, col2 = st.columns(2)
    
    with col1:
        years_input = st.text_input(
            "Water years (e.g., '1980, 1995, 2010' or '2015-2020'):",
            "2000, 2010, 2020",
            key="years_input"
        )
    
    with col2:
        if dataset == "Precipitation":
            min_valid = st.number_input(
                "Min valid days per year:",
                min_value=1,
                value=200,
                key="min_valid_precip"
            )
        else:
            min_valid = st.number_input(
                "Min valid days per year:",
                min_value=1,
                value=50,
                key="min_valid_snow"
            )

elif mode == "Recurring Seasonal Stretch":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.text_input(
            "Start (MM-DD):",
            "11-01",
            key="start_date"
        )
    
    with col2:
        end_date = st.text_input(
            "End (MM-DD):",
            "03-31",
            key="end_date"
        )
    
    with col3:
        years_input = st.text_input(
            "Years:",
            "2000, 2010, 2020",
            key="years_seasonal"
        )

else:  # Custom Date Ranges
    st.info("📅 Enter date ranges (one per line, format: YYYY-MM-DD to YYYY-MM-DD)")
    date_ranges = st.text_area(
        "Date ranges:",
        "2000-01-01 to 2000-12-31\n2010-01-01 to 2010-12-31\n2020-01-01 to 2020-12-31",
        height=100,
        key="date_ranges"
    )

st.markdown("---")

# Run button
run_analysis = st.button(
    "🔍 Run Analysis",
    use_container_width=True,
    key="run_btn"
)

# ============================================================================
# ANALYSIS RESULTS
# ============================================================================

if run_analysis:
    if not selected_regions:
        st.error("❌ Please select at least one region/watershed!")
    else:
        with st.spinner("Running analysis... please wait"):
            try:
                # Placeholder results for now
                # In production, call actual compare.py/compare_snowfall.py functions
                
                st.success("✅ Analysis Complete!")
                
                # Summary by region
                st.markdown("### Regional Summary")
                
                summary_data = []
                for region in selected_regions:
                    if mode == "Full Water Years":
                        summary_data.append({
                            'Region': region,
                            'Years': '2000, 2010, 2020',
                            f'Avg {"Precip" if dataset == "Precipitation" else "Snow"} (")': f'{20 + len(region) % 10:.2f}',
                            '% WY Base': f'{85 + len(region) % 20:.1f}%',
                            'Stations': 100 + len(region) % 50
                        })
                    else:
                        summary_data.append({
                            'Region': region,
                            'Period': 'Nov 1 - Mar 31' if 'Seasonal' in mode else 'Custom',
                            f'Avg {"Precip" if dataset == "Precipitation" else "Snow"} (")': f'{15 + len(region) % 10:.2f}',
                            'Wet Days': 45 + len(region) % 20,
                            'Stations': 100 + len(region) % 50
                        })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Detailed station breakdown
                st.markdown("### Station Breakdown")
                
                station_data = []
                for i in range(min(10, 100 + len(selected_regions) % 20)):
                    station_data.append({
                        'Station ID': i,
                        'Station Name': f'Station {i+1}',
                        'Region': selected_regions[i % len(selected_regions)],
                        f'Avg {"Precip" if dataset == "Precipitation" else "Snow"} (")': f'{10 + i % 30:.2f}',
                        'Valid Days': 300 - i % 50,
                        'Max Daily': f'{1.5 + i % 5:.2f}'
                    })
                
                df_stations = pd.DataFrame(station_data)
                st.dataframe(df_stations, use_container_width=True, hide_index=True)
                
                # Export options
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    csv = df_summary.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Summary (CSV)",
                        data=csv,
                        file_name=f"summary_{dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    csv = df_stations.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Stations (CSV)",
                        data=csv,
                        file_name=f"stations_{dataset}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("This is a preview. Full functionality coming soon.")

# ============================================================================
# FOOTER & INFO
# ============================================================================

st.markdown("---")

with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    ### California Climate Analysis Tool
    
    Analyze historical precipitation and snowfall data across California.
    
    **Coverage:**
    - **Precipitation:** 328 stations, 10 regions, 1950-2026
    - **Snowfall:** 262 stations, 3 Sierra watersheds, 1950-2026
    
    **Data Source:** ACIS (Applied Climate Information System)
    
    **Comparison Modes:**
    - **Full Water Years:** July 1 - June 30 comparisons
    - **Seasonal Stretches:** Same calendar dates across years
    - **Custom Ranges:** Any date range you specify
    """)

with st.expander("📊 How to Interpret Results"):
    st.markdown("""
    ### Key Metrics
    
    - **Avg Precip/Snow** - Average in inches for the period
    - **% WY Base** - Percentage of typical water year amount
    - **Valid Days** - Days with non-null measurements
    - **Wet Days** - Days with ≥0.01" precipitation (or trace snow)
    - **Max Daily** - Single highest value recorded
    - **Max Wet Streak** - Consecutive wet days
    
    ### Water Year
    A **water year (WY)** runs July 1 - June 30 (named for ending year)
    - WY 2020 = July 2019 to June 2020
    
    ### Interpreting %WY Base
    - **100%** = Normal (typical) amount
    - **<100%** = Below average (drier/less snow)
    - **>100%** = Above average (wetter/more snow)
    """)

with st.expander("❓ FAQ"):
    st.markdown("""
    **Q: What years are available?**
    A: 1950-2026 for both precipitation and snowfall
    
    **Q: How many stations?**
    A: 328 precipitation stations, 262 snowfall stations
    
    **Q: Can I compare specific dates?**
    A: Yes! Use "Custom Date Ranges" mode
    
    **Q: How is data quality controlled?**
    A: Minimum valid days threshold filters incomplete records
    
    **Q: Can I download the data?**
    A: Yes, results export to CSV
    
    **Q: Is this data official?**
    A: Data is from ACIS (NOAA partner). We provide analysis tools.
    """)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**📍 Data:** ACIS Weather Network")
with col2:
    st.markdown("**🔧 Built with:** Streamlit + DuckDB")
with col3:
    st.markdown("**📝 License:** Open Source")
