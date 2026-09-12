# California Precipitation & Snowfall Analysis

A comprehensive web tool for analyzing California precipitation and snowfall data from 1950-2026 across multiple regions and time periods.

🌐 **Live Demo:** [Visit App](https://share.streamlit.io) (deploying soon)

## Features

### Precipitation Analysis
- **10 California Hydrological Regions** (North Coast, Sacramento River, San Joaquin River, etc.)
- **1950-2026 data** from 328 weather stations
- **3 comparison modes:**
  1. Full Water Years (July 1 - June 30)
  2. Recurring Seasonal Stretches (e.g., Nov-Feb across years)
  3. Custom Date Ranges

### Snowfall Analysis
- **3 Sierra Watershed Regions** (Northern, Mid-Central, Southern Central Valley Drainage)
- **1950-2026 snowfall & snow depth data** from 262 stations
- Same 3 comparison modes as precipitation

### Output
- Regional summaries with average precipitation/snowfall
- Percentage of typical water year
- Detailed station-by-station breakdowns
- Wet days, max daily totals, consecutive day streaks
- Multi-region statewide averages

## Tech Stack

- **Backend:** Python 3.11, DuckDB, Pandas
- **Frontend:** Streamlit
- **Data:** Parquet files (decade-based, indexed by station)
- **Hosting:** Streamlit Cloud / Railway

## Local Installation

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/california-climate.git
cd california-climate

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app_streamlit.py
```

Opens at `http://localhost:8501`

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete setup instructions.

**Quick deploy to Streamlit Cloud (FREE):**
1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo and deploy
4. Share the URL!

## Data Structure

```
Station Metadata (stations_meta.jsonl):
{
  "id": 0,
  "name": "STATION NAME",
  "lat": 39.5,
  "lon": -120.5,
  "sids": ["US1CAMD0015 6", ...]
}

Precipitation Parquets (daily_1950_1959.parquet, etc.):
Columns: id, date, precip

Snowfall Parquets (snowfall_1950_1959.parquet, etc.):
Columns: id, date, snow, snwd
```

## File Organization

```
├── app_streamlit.py           # Web UI
├── compare.py                 # Precipitation comparison logic
├── compare_snowfall.py        # Snowfall comparison logic
├── requirements.txt
├── stations_meta.jsonl        # Station metadata
├── daily_*.parquet            # Precipitation (all decades)
├── snowfall_*.parquet         # Snowfall (all decades)
└── README.md
```

## Data Source

- **ACIS** (Applied Climate Information System)
- California weather station network
- Official NOAA partner data

## Features Coming Soon

- 📈 Interactive charts and visualizations
- 📊 Export to CSV/Excel
- 🗺️ Interactive map with station locations
- 📅 Monthly and seasonal normals
- 🌡️ Temperature analysis (mean, max, min)
- 🔄 Auto-updating data (latest year additions)

## License

Open source - use freely for research, education, or analysis.

## Author

Max - California Climate Analysis Project

## Questions?

Open an issue or reach out!
