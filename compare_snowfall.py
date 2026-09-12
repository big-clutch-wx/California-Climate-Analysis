#!/usr/bin/env python3
"""
California Snowfall - Multi-Year & Custom Stretch Regional Comparison
Uses DuckDB for high-performance direct querying over daily Parquet files.
Supports full water years, recurring date stretches, arbitrary custom date ranges,
and consecutive wet/dry day spell tracking.
"""

import json
import os
import glob
import re
from datetime import datetime
import duckdb
import pandas as pd

def point_in_polygon(point, polygon):
    """Determines whether a given point (lat, lon) is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_official_regions():
    """Returns Sierra watershed region polygons."""
    return {
        'Northern Central Valley Drainage': {
            'avg_snowfall': 0.0,
            'polygon': [
                (41.7220427, -120.5327942), (41.7236618, -120.2760954), (41.1999754, -120.1753769),
                (41.0466504, -120.3417544), (41.1333552, -120.5575579), (40.9791679, -120.5957818),
                (40.9012347, -120.7862927), (40.7952578, -120.8566352), (40.6155697, -121.0604821),
                (40.467983, -121.2280584), (40.0788443, -120.2206377), (39.8058323, -120.0970956),
                (39.5314918, -120.0784299), (39.4949991, -120.4329238), (39.4402623, -120.4594848),
                (39.3127721, -120.3131157), (39.3384141, -120.1351486), (38.6612759, -119.9847154),
                (38.6255118, -120.1937641), (38.7342502, -120.4906176), (38.6322507, -121.1397622),
                (38.6643839, -121.2320114), (38.8187602, -121.1137765), (38.8440636, -121.1449594),
                (38.7647496, -121.2510676), (38.7799442, -121.2532331), (38.8840364, -121.1284778),
                (38.8965494, -121.0658277), (38.9140271, -121.0684282), (39.0422838, -120.9774989),
                (39.1382383, -120.9371663), (39.3037458, -120.6721664), (39.3156566, -120.8916306),
                (39.2320794, -121.1643202), (39.2413618, -121.282221), (39.3914766, -121.1633685),
                (39.5114019, -121.223411), (39.5012586, -121.5014363), (39.8206585, -121.5859011),
                (40.3003236, -121.34864), (40.3839801, -121.4907623), (40.8015687, -121.943934),
                (40.7229342, -122.2336741), (40.71199, -122.4600044), (40.746814, -122.4899528),
                (40.7717867, -122.6039138), (41.3350802, -122.4414034), (41.6374171, -121.5254881),
                (41.7205733, -120.5698811), (41.7220427, -120.5327942),
            ]
        },
        'Mid-Central Valley Drainage': {
            'avg_snowfall': 0.0,
            'polygon': [
                (38.6637249, -119.986802), (38.6255519, -120.1929264), (38.7344846, -120.4932588),
                (38.6650407, -120.9312537), (38.5035691, -121.0631543), (38.476483, -120.826162),
                (38.5137913, -120.7544444), (38.5098534, -120.4849875), (38.4306921, -120.5388915),
                (38.3145335, -120.7320138), (38.2873699, -120.7015706), (38.3153416, -120.2614287),
                (38.2516675, -120.3490905), (38.0969271, -120.5338376), (38.0351015, -120.6743248),
                (37.8732551, -120.6145429), (37.6903387, -120.4214081), (37.5394369, -120.2794053),
                (37.4822077, -119.8612905), (37.3631884, -119.5865394), (36.9935696, -119.7326244),
                (37.058275, -119.3774053), (37.1046621, -119.1985174), (37.1954255, -118.8868385),
                (37.1412605, -118.6541993), (37.1674538, -118.672838), (37.2102491, -118.67901),
                (37.3271624, -118.7168841), (37.4891811, -118.7991363), (37.5945017, -119.0270266),
                (37.631569, -119.0344916), (37.6551557, -119.0609136), (37.7340341, -119.1272184),
                (37.7272421, -119.2592845), (37.8978597, -119.2172554), (37.9664532, -119.322985),
                (38.1086992, -119.4123202), (38.2007412, -119.6315563), (38.2614345, -119.6152692),
                (38.3198903, -119.633805), (38.3496947, -119.628687), (38.5491391, -119.8057182),
                (38.6638296, -119.9615656), (38.6637249, -119.986802),
            ]
        },
        'Southern Central Valley Drainage Area': {
            'avg_snowfall': 0.0,
            'polygon': [
                (37.0638673, -119.358151), (36.8307367, -119.3344488), (36.7109426, -118.9287961),
                (36.3949068, -119.0102048), (36.3769361, -118.9234384), (36.0598677, -118.939178),
                (35.9443746, -118.603998), (35.6862295, -118.655455), (35.6103815, -118.360623),
                (35.4655667, -118.4051917), (35.4415963, -118.2910743), (35.4814556, -118.1626809),
                (35.5552501, -118.1240312), (35.6073595, -118.1117993), (35.6966231, -117.9903956),
                (35.8597567, -118.0078388), (35.938036, -117.9939311), (36.0985167, -118.0687092),
                (36.3006555, -118.1283628), (36.4290781, -118.1536274), (36.4310345, -118.2131997),
                (36.5232533, -118.2407569), (36.5623236, -118.2930346), (36.6693399, -118.3331547),
                (36.6925539, -118.3682278), (36.7281569, -118.3379286), (36.8361957, -118.3935463),
                (36.849751, -118.3619661), (37.0676512, -118.4437324), (37.1411478, -118.6516326),
                (37.19539, -118.8861403), (37.1047212, -119.1986121), (37.0638673, -119.358151),
            ]
        }
    }


def parse_years_input(user_input):
    """Parses year ranges or comma-separated lists (e.g. '1980, 1995-2000')."""
    years = set()
    parts = [p.strip() for p in user_input.split(',')]
    for part in parts:
        if '-' in part:
            start, end = part.split('-')
            years.update(range(int(start), int(end) + 1))
        else:
            if part:
                years.add(int(part))
    return sorted([y for y in years if 1850 <= y <= 2026])

def parse_date_string(date_str):
    """Parses a date string in YYYY-MM-DD format."""
    try:
        dt = datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected YYYY-MM-DD.")

def run_duckdb_distinct_ranges(matching_ids, date_ranges):
    """Queries distinct custom date ranges using DuckDB."""
    con = duckdb.connect()

    parquet_files = glob.glob('snowfall_*.parquet')

    if not parquet_files:
        print("No snowfall_*.parquet files found in current directory.")
        con.close()
        return None

    ids_str = ", ".join([str(int(s)) for s in matching_ids])

    union_queries = []

    for idx, r in enumerate(date_ranges):

        min_valid = r['min_valid_days']

        union_queries.append(f"""
            SELECT
                '{idx}' AS period_id,
                '{r['label']}' AS period_label,
                '{min_valid}' AS min_valid_days,

                id,
                station_id,

                CAST(date AS DATE) AS date,

                CASE
                    WHEN snow IS NULL
                         OR TRIM(snow) = ''
                         OR UPPER(TRIM(snow)) = 'M'
                         OR UPPER(TRIM(snow)) = 'T'
                    THEN 0.0
                    ELSE TRY_CAST(snow AS DOUBLE)
                END AS daily_snow,

                CASE
                    WHEN snwd IS NULL
                         OR TRIM(snwd) = ''
                         OR UPPER(TRIM(snwd)) = 'M'
                    THEN NULL
                    ELSE TRY_CAST(snwd AS DOUBLE)
                END AS snow_depth

            FROM read_parquet(
                'snowfall_*.parquet',
                union_by_name=true
            )

            WHERE id IN ({ids_str})

              AND CAST(date AS DATE)
                  >= DATE '{r['start']}'

              AND CAST(date AS DATE)
                  <= DATE '{r['end']}'
        """)

    full_query = " UNION ALL ".join(union_queries)

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),

    flagged AS (
        SELECT
            *,
            CASE
                WHEN daily_snow >= 0.01
                THEN 1
                ELSE 0
            END AS is_wet

        FROM raw_data

        WHERE daily_snow IS NOT NULL
    ),

    grouped_streaks AS (
        SELECT
            *,

            SUM(
                CASE
                    WHEN is_wet = 0 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS wet_grp,

            SUM(
                CASE
                    WHEN is_wet = 1 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS dry_grp

        FROM flagged
    ),

    streak_lengths AS (
        SELECT
            period_id,
            id,
            is_wet,
            COUNT(*) AS streak_len

        FROM grouped_streaks

        GROUP BY
            period_id,
            id,
            is_wet,
            CASE
                WHEN is_wet = 1
                THEN wet_grp
                ELSE dry_grp
            END
    ),

    max_streaks AS (
        SELECT
            period_id,
            id,

            COALESCE(
                MAX(
                    CASE
                        WHEN is_wet = 1
                        THEN streak_len
                    END
                ),
                0
            ) AS max_consec_wet,

            COALESCE(
                MAX(
                    CASE
                        WHEN is_wet = 0
                        THEN streak_len
                    END
                ),
                0
            ) AS max_consec_dry

        FROM streak_lengths

        GROUP BY
            period_id,
            id
    ),

    station_totals AS (
        SELECT
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,

            MAX(
                CAST(fd.min_valid_days AS INTEGER)
            ) AS min_valid_days,

            SUM(fd.daily_snow) AS total_snow,

            MAX(fd.daily_snow) AS max_daily_snow,

            COUNT(fd.daily_snow) AS valid_days,

            SUM(
                CASE
                    WHEN fd.daily_snow >= 0.01
                    THEN 1
                    ELSE 0
                END
            ) AS wet_days,

            COALESCE(
                ms.max_consec_wet,
                0
            ) AS max_consec_wet,

            COALESCE(
                ms.max_consec_dry,
                0
            ) AS max_consec_dry

        FROM raw_data fd

        LEFT JOIN max_streaks ms
            ON fd.period_id = ms.period_id
           AND fd.id = ms.id

        GROUP BY
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,
            ms.max_consec_wet,
            ms.max_consec_dry
    )

    SELECT
        st.period_id,
        st.period_label,
        st.id,
        st.station_id,
        st.total_snow,
        st.max_daily_snow,
        st.valid_days,
        st.wet_days,
        st.max_consec_wet,
        st.max_consec_dry

    FROM station_totals st

    WHERE st.valid_days >= st.min_valid_days;
    """

    df_res = con.execute(sql).df()

    con.close()

    return df_res


def run_duckdb_custom_stretches(
    matching_ids,
    start_mmdd,
    end_mmdd,
    years,
    min_valid_ratio=0.70,
    strict_consistency=True
):
    """Queries recurring seasonal stretches across multiple years using DuckDB."""

    con = duckdb.connect()

    parquet_files = glob.glob('snowfall_*.parquet')

    if not parquet_files:
        print("No snowfall_*.parquet files found in current directory.")
        con.close()
        return None

    ids_str = ", ".join([str(int(s)) for s in matching_ids])

    union_queries = []

    start_month, start_day = map(
        int,
        start_mmdd.split('-')
    )

    end_month, end_day = map(
        int,
        end_mmdd.split('-')
    )

    # Calculate total days in the stretch
    if (
        start_month > end_month
        or (
            start_month == end_month
            and start_day > end_day
        )
    ):
        d1 = datetime(
            2001,
            start_month,
            start_day
        )

        d2 = datetime(
            2002,
            end_month,
            end_day
        )

    else:
        d1 = datetime(
            2001,
            start_month,
            start_day
        )

        d2 = datetime(
            2001,
            end_month,
            end_day
        )

    total_days = (
        d2 - d1
    ).days + 1

    req_valid_days = max(
        1,
        int(
            total_days
            * min_valid_ratio
        )
    )

    for y in years:

        if (
            start_month > end_month
            or (
                start_month == end_month
                and start_day > end_day
            )
        ):
            s_date = f"{y}-{start_mmdd}"
            e_date = f"{y + 1}-{end_mmdd}"

            lbl = (
                f"{start_mmdd} ({y}) "
                f"to {end_mmdd} ({y + 1})"
            )

        else:
            s_date = f"{y}-{start_mmdd}"
            e_date = f"{y}-{end_mmdd}"

            lbl = (
                f"{start_mmdd} to "
                f"{end_mmdd} ({y})"
            )

        union_queries.append(f"""
            SELECT

                '{y}' AS period_id,

                '{lbl}' AS period_label,

                id,

                station_id,

                CAST(date AS DATE) AS date,

                CASE
                    WHEN snow IS NULL
                         OR TRIM(snow) = ''
                         OR UPPER(TRIM(snow)) = 'M'
                         OR UPPER(TRIM(snow)) = 'T'
                    THEN 0.0
                    ELSE TRY_CAST(snow AS DOUBLE)
                END AS daily_snow,

                CASE
                    WHEN snwd IS NULL
                         OR TRIM(snwd) = ''
                         OR UPPER(TRIM(snwd)) = 'M'
                    THEN NULL
                    ELSE TRY_CAST(snwd AS DOUBLE)
                END AS snow_depth

            FROM read_parquet(
                'snowfall_*.parquet',
                union_by_name=true
            )

            WHERE id IN ({ids_str})

              AND CAST(date AS DATE)
                  >= DATE '{s_date}'

              AND CAST(date AS DATE)
                  <= DATE '{e_date}'
        """)

    full_query = " UNION ALL ".join(
        union_queries
    )

    if strict_consistency:

        consistency_clause = f"""
        ,
        consistent_stations AS (
            SELECT
                station_id

            FROM station_totals

            GROUP BY station_id

            HAVING COUNT(
                DISTINCT period_id
            ) = {len(years)}
        )
        """

        join_clause = """
        JOIN consistent_stations cs
          ON st.station_id = cs.station_id
        """

    else:

        consistency_clause = ""

        join_clause = ""

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),

    flagged AS (
        SELECT
            *,

            CASE
                WHEN daily_snow >= 0.01
                THEN 1
                ELSE 0
            END AS is_wet

        FROM raw_data

        WHERE daily_snow IS NOT NULL
    ),

    grouped_streaks AS (
        SELECT
            *,

            SUM(
                CASE
                    WHEN is_wet = 0 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS wet_grp,

            SUM(
                CASE
                    WHEN is_wet = 1 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS dry_grp

        FROM flagged
    ),

    streak_lengths AS (
        SELECT
            period_id,
            id,
            is_wet,
            COUNT(*) AS streak_len

        FROM grouped_streaks

        GROUP BY
            period_id,
            id,
            is_wet,
            CASE
                WHEN is_wet = 1
                THEN wet_grp
                ELSE dry_grp
            END
    ),

    max_streaks AS (
        SELECT
            period_id,
            id,

            COALESCE(
                MAX(
                    CASE
                        WHEN is_wet = 1
                        THEN streak_len
                    END
                ),
                0
            ) AS max_consec_wet,

            COALESCE(
                MAX(
                    CASE
                        WHEN is_wet = 0
                        THEN streak_len
                    END
                ),
                0
            ) AS max_consec_dry

        FROM streak_lengths

        GROUP BY
            period_id,
            id
    ),

    station_totals AS (
        SELECT
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,

            SUM(fd.daily_snow)
                AS total_snow,

            MAX(fd.daily_snow)
                AS max_daily_snow,

            COUNT(fd.daily_snow)
                AS valid_days,

            SUM(
                CASE
                    WHEN fd.daily_snow >= 0.01
                    THEN 1
                    ELSE 0
                END
            ) AS wet_days,

            COALESCE(
                ms.max_consec_wet,
                0
            ) AS max_consec_wet,

            COALESCE(
                ms.max_consec_dry,
                0
            ) AS max_consec_dry

        FROM raw_data fd

        LEFT JOIN max_streaks ms
            ON fd.period_id = ms.period_id
           AND fd.id = ms.id

        GROUP BY
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,
            ms.max_consec_wet,
            ms.max_consec_dry

        HAVING COUNT(fd.daily_snow)
            >= {req_valid_days}
    )

    {consistency_clause}

    SELECT
        st.period_id,
        st.period_label,
        st.id,
        st.station_id,
        st.total_snow,
        st.max_daily_snow,
        st.valid_days,
        st.wet_days,
        st.max_consec_wet,
        st.max_consec_dry

    FROM station_totals st

    {join_clause};
    """

    df_res = con.execute(sql).df()

    con.close()

    return df_res


def run_duckdb_water_years(
    matching_ids,
    years,
    min_valid_days=200
):
    """Queries full water years (July 1 - June 30) using DuckDB."""

    con = duckdb.connect()

    parquet_files = glob.glob(
        'snowfall_*.parquet'
    )

    if not parquet_files:

        print(
            "No snowfall_*.parquet files "
            "found in current directory."
        )

        con.close()

        return None

    ids_str = ", ".join(
        [str(int(s)) for s in matching_ids]
    )

    union_queries = []

    for wy in years:

        start_date = (
            f"{wy}-07-01"
        )

        end_date = (
            f"{wy + 1}-06-30"
        )

        union_queries.append(f"""
            SELECT

                '{wy}' AS period_id,

                'WY {wy}' AS period_label,

                id,

                station_id,

                CAST(date AS DATE) AS date,

                CASE
                    WHEN snow IS NULL
                         OR TRIM(snow) = ''
                         OR UPPER(TRIM(snow)) = 'M'
                         OR UPPER(TRIM(snow)) = 'T'
                    THEN 0.0
                    ELSE TRY_CAST(snow AS DOUBLE)
                END AS daily_snow,

                CASE
                    WHEN snwd IS NULL
                         OR TRIM(snwd) = ''
                         OR UPPER(TRIM(snwd)) = 'M'
                    THEN NULL
                    ELSE TRY_CAST(snwd AS DOUBLE)
                END AS snow_depth

            FROM read_parquet(
                'snowfall_*.parquet',
                union_by_name=true
            )

            WHERE id IN ({ids_str})

              AND CAST(date AS DATE)
                  >= DATE '{start_date}'

              AND CAST(date AS DATE)
                  <= DATE '{end_date}'
        """)

    full_query = " UNION ALL ".join(
        union_queries
    )

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),

    flagged AS (
        SELECT
            *,

            CASE
                WHEN daily_snow >= 0.01
                THEN 1
                ELSE 0
            END AS is_wet

        FROM raw_data

        WHERE daily_snow IS NOT NULL
    ),

    grouped_streaks AS (
        SELECT
            *,

            SUM(
                CASE
                    WHEN is_wet = 0 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS wet_grp,

            SUM(
                CASE
                    WHEN is_wet = 1 THEN 1
                    ELSE 0
                END
            ) OVER (
                PARTITION BY period_id, id
                ORDER BY date
            ) AS dry_grp

        FROM flagged
    ),

    streak_lengths AS (
        SELECT
            period_id,
            id,
            is_wet,
            COUNT(*) AS streak_len

        FROM grouped_streaks

        GROUP BY
            period_id,
            id,
            is_wet,
            CASE
                WHEN is_wet = 1
                THEN wet_grp
                ELSE dry_grp
            END
    ),

    max_streaks AS (
        SELECT
            period_id,
            id,

            MAX(
                CASE
                    WHEN is_wet = 1
                    THEN streak_len
                    ELSE 0
                END
            ) AS max_consec_wet,

            MAX(
                CASE
                    WHEN is_wet = 0
                    THEN streak_len
                    ELSE 0
                END
            ) AS max_consec_dry

        FROM streak_lengths

        GROUP BY
            period_id,
            id
    ),

    station_totals AS (
        SELECT
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,

            SUM(fd.daily_snow)
                AS total_snow,

            MAX(fd.daily_snow)
                AS max_daily_snow,

            COUNT(fd.daily_snow)
                AS valid_days,

            SUM(
                CASE
                    WHEN fd.daily_snow >= 0.01
                    THEN 1
                    ELSE 0
                END
            ) AS wet_days,

            COALESCE(
                ms.max_consec_wet,
                0
            ) AS max_consec_wet,

            COALESCE(
                ms.max_consec_dry,
                0
            ) AS max_consec_dry

        FROM raw_data fd

        LEFT JOIN max_streaks ms
            ON fd.period_id = ms.period_id
           AND fd.id = ms.id

        GROUP BY
            fd.period_id,
            fd.period_label,
            fd.id,
            fd.station_id,
            ms.max_consec_wet,
            ms.max_consec_dry
    )

    SELECT
        st.period_id,
        st.period_label,
        st.id,
        st.station_id,
        st.total_snow,
        st.max_daily_snow,
        st.valid_days,
        st.wet_days,
        st.max_consec_wet,
        st.max_consec_dry

    FROM station_totals st

    WHERE st.valid_days >= {min_valid_days};
    """

    df_res = con.execute(sql).df()

    con.close()

    return df_res




def main():
    print("="*95)
    print("California Snowfall - Multi-Region & Multi-Period Comparison (DuckDB)")
    print("="*95)
    
    regions = get_official_regions()
    region_list = sorted(regions.keys())
    
    print("\nSierra Watershed Selection:")
    print("  0. ALL Sierra Watersheds (Full Multi-Region Comparison)")
    for i, region in enumerate(region_list, 1):
        print(f"  {i:2}. {region} (WY Base: {regions[region]['avg_snowfall']:.2f}\")")
    
    selected_regions = []
    while True:
        choice_raw = input(f"\nSelect region(s) (0 for ALL, e.g. '2', or comma-separated e.g. '1, 4, 6'): ").strip()
        if choice_raw == '0' or choice_raw.lower() == 'all':
            selected_regions = region_list
            break
        try:
            indices = [int(p.strip()) for p in choice_raw.split(',')]
            if all(1 <= idx <= len(region_list) for idx in indices):
                selected_regions = [region_list[idx - 1] for idx in indices]
                break
            print(f"Invalid selection. Please choose numbers between 1 and {len(region_list)} or 0 for ALL.")
        except ValueError:
            print("Please enter valid numbers.")

    metadata = {}
    if os.path.exists('stations_meta.jsonl'):
        with open('stations_meta.jsonl', 'r') as f:
            for line in f:
                meta = json.loads(line)
                metadata[meta['id']] = meta
    else:
        print("stations_meta.jsonl not found!")
        return

    # Map each station ID to its region
    station_region_map = {}
    for sid, meta in metadata.items():
        pt = (meta['lat'], meta['lon'])
        for reg in selected_regions:
            polygon = regions[reg]['polygon']
            if point_in_polygon(pt, polygon):
                station_region_map[sid] = reg
                break

    matching_ids = set(station_region_map.keys())
    
    print(f"\nFound {len(matching_ids)} potential stations across {len(selected_regions)} region(s):")
    for reg in selected_regions:
        count = sum(1 for r in station_region_map.values() if r == reg)
        print(f"  - {reg:<20}: {count} stations")
    
    print("\nComparison Mode:")
    print("  1. Full Water Years (July 1 - June 30)")
    print("  2. Recurring Seasonal Stretch Across Selected Years (e.g., Nov 12 - Feb 18)")
    print("  3. Distinct Custom Date Ranges with Consecutive Spell Analysis")
    
    mode_choice = input("Select mode (1, 2, or 3): ").strip()

    if mode_choice == '3':
        print("\n--- Compare Distinct Custom Date Ranges & Spell Metrics ---")
        date_ranges = []
        range_count = 1
        while True:
            print(f"\nRange #{range_count}:")
            start_raw = input("  Enter start date (e.g., 1983-01-15): ").strip()
            if not start_raw:
                break
            end_raw = input("  Enter end date   (e.g., 1983-03-31): ").strip()
            
            try:
                start_iso = parse_date_string(start_raw)
                end_iso = parse_date_string(end_raw)
                
                dt_start = datetime.strptime(start_iso, '%Y-%m-%d')
                dt_end = datetime.strptime(end_iso, '%Y-%m-%d')
                
                if dt_start > dt_end:
                    print("  Error: Start date must be on or before end date. Try again.")
                    continue

                total_days = (dt_end - dt_start).days + 1
                min_valid_days = max(1, int(total_days * 0.70))
                
                lbl = f"{dt_start.strftime('%b %d, %Y')} - {dt_end.strftime('%b %d, %Y')}"
                date_ranges.append({
                    'start': start_iso,
                    'end': end_iso,
                    'label': lbl,
                    'min_valid_days': min_valid_days,
                    'total_days': total_days
                })
                print(f"  Added range: {lbl} ({total_days} total days, requiring >={min_valid_days} valid days)")
                range_count += 1
                
                if len(date_ranges) >= 2:
                    more = input("\nAdd another date range to compare? (y/n): ").lower().strip()
                    if more not in ['y', 'yes']:
                        break
            except ValueError as e:
                print(f"  {e}")
        
        if len(date_ranges) < 2:
            print("\nAt least 2 custom date ranges are required for comparison.")
            return

        print("\nQuerying daily Parquet files with DuckDB across distinct date ranges...")
        df_res = run_duckdb_distinct_ranges(matching_ids, date_ranges)

    elif mode_choice == '2':
        start_mmdd = input("Enter start date (MM-DD, e.g., 11-12): ").strip()
        end_mmdd = input("Enter end date (MM-DD, e.g., 02-18): ").strip()
        raw_years = input("Enter years to compare (e.g., '1980, 1995, 2010' or '2015-2020'): ")
        years = parse_years_input(raw_years)
        
        strict_ans = input("Require strict same stations across ALL years? (y/n, default=y): ").strip().lower()
        strict_consistency = False if strict_ans in ['n', 'no'] else True

        print("\nQuerying daily Parquet files with DuckDB...")
        # Use lower threshold for pre-1950 historical data
        has_pre_1950 = any(y < 1950 for y in years)
        min_ratio = 0.50 if has_pre_1950 else 0.70
        print(f"Using min_valid_ratio={min_ratio} ({'historical data' if has_pre_1950 else 'modern data'})")
        df_res = run_duckdb_custom_stretches(matching_ids, start_mmdd, end_mmdd, years, min_valid_ratio=min_ratio, strict_consistency=strict_consistency)
    else:
        raw_years = input("\nEnter water years (e.g., '1980, 1995, 2010' or '2015-2020'): ")
        years = parse_years_input(raw_years)
        
        print("\nQuerying daily Parquet files with DuckDB...")
        # Use lower threshold for pre-1950 historical data (less complete records)
        has_pre_1950 = any(y < 1950 for y in years)
        min_valid = 100 if has_pre_1950 else 200
        print(f"Using min_valid_days={min_valid} ({'historical data' if has_pre_1950 else 'modern data'})")
        df_res = run_duckdb_water_years(matching_ids, years, min_valid_days=min_valid)

    if df_res is None or df_res.empty:
        print("\nNo consistent station data found across all specified periods.")
        return

    # Map station to region
    df_res['region'] = df_res['station_id'].map(station_region_map)

    # Aggregate by region and period
    summary = df_res.groupby(['region', 'period_id', 'period_label']).agg(
        avg_snowfall=('total_snow', 'mean'),
        avg_wet_days=('wet_days', 'mean'),
        avg_max_daily=('max_daily_snow', 'mean'),
        avg_consec_wet=('max_consec_wet', 'mean'),
        avg_consec_dry=('max_consec_dry', 'mean'),
        station_count=('station_id', 'count')
    ).reset_index()

    # Calculate baseline % per region
    summary['regional_base'] = summary['region'].map(lambda r: regions[r]['avg_snowfall'])
    summary['pct_base'] = (summary['avg_snowfall'] / summary['regional_base']) * 100

    print("\n" + "="*70)
    print("REGIONAL COMPARISON SUMMARY TABLE")
    print("="*70)
    
    # Group by region and print with station count in header
    for region in summary['region'].unique():
        region_data = summary[summary['region'] == region].sort_values('period_id')
        
        # Calculate stations present across ALL periods in this region
        region_df = df_res[df_res['station_id'].map(lambda s: station_region_map.get(s) == region)]
        stations_by_period = [
            set(region_df[region_df['period_label'] == period]['station_id'])
            for period in region_data['period_label'].unique()
        ]
        
        common_stations = set.intersection(*stations_by_period) if stations_by_period else set()
        station_count = len(common_stations)
        
        print(f"\n{region} ({int(station_count)} stations)")
        print(f"{'Period':<30} {'Avg Snowfall':>11}")
        print("-" * 55)
        
    for _, row in region_data.iterrows():
        if pd.isna(row['pct_base']) or row['pct_base'] == float('inf'):
            pct_str = " " * 11  # Keeps column alignment clean
        else:
            pct_str = f"{row['pct_base']:>9.1f}%"
    
        print(f"{row['period_label']:<30} {row['avg_snowfall']:>10.2f}\"{pct_str}")
    
    # Add statewide average if multiple regions selected
    if len(selected_regions) > 1:
        print("\n" + "="*70)
        print("STATEWIDE AVERAGE (All Stations)")
        print("=" * 70)
        
        # Calculate TRUE statewide average from ALL individual stations in df_res
        # Group by period and calculate mean snowfall across ALL stations
        statewide_periods = df_res.groupby('period_label').agg({
            'total_snow': 'mean'
        }).reset_index()
        statewide_periods.columns = ['period_label', 'avg_snowfall']
        
        # Calculate % of statewide WY base
        statewide_wy_base = 26.0  # CA statewide average for full water year
        statewide_periods['pct_base'] = (statewide_periods['avg_snowfall'] / statewide_wy_base) * 100
        
        print(f"{'Period':<30} {'Avg Snowfall':>11}")
        print("-" * 55)
        
        for _, row in statewide_periods.iterrows():
            print(f"{row['period_label']:<30} {row['avg_snowfall']:>10.2f}\" {row['pct_base']:>9.1f}%")
    
    print("\n" + "="*70)

    # Multi-Region side-by-side view if multiple regions selected
    if len(selected_regions) > 1:
        print("\n" + "="*95)
        print("SIDE-BY-SIDE REGIONAL PRECIPITATION MATRIX (Inches)")
        print("="*95)
        piv_snowfall = summary.pivot(index='region', columns='period_label', values='avg_snowfall')
        
        # Add statewide average row (calculated from ALL stations, not region means)
        statewide_avg = df_res.groupby('period_label')['total_snow'].mean()
        piv_snowfall.loc['STATEWIDE AVERAGE'] = statewide_avg
        
        # Format to 2 decimal places
        piv_snowfall_formatted = piv_snowfall.round(2)
        print(piv_snowfall_formatted.to_string())

    show_detail = input("\nShow detailed station breakdown? (yes/no): ").lower().strip()
    if show_detail in ['yes', 'y']:
        # Filter to stations that have valid data present in ALL requested periods
        all_periods = df_res['period_label'].unique()
        stations_with_all_periods = (
            df_res.groupby('station_id')['period_label']
            .nunique()
            .loc[lambda x: x == len(all_periods)]
            .index
        )
        df_detail = df_res[df_res['station_id'].isin(stations_with_all_periods)]
        
        # Get snowfall pivot
        piv_snowfall = df_detail.pivot(index='station_id', columns='period_label', values='total_snow')
        piv_snowfall['station_name'] = piv_snowfall.index.map(lambda sid: metadata.get(sid, {}).get('name', sid))
        piv_snowfall['region'] = piv_snowfall.index.map(lambda sid: station_region_map.get(sid, 'Unknown'))
        
        # Sort by average snowfall across periods
        snowfall_cols = [col for col in piv_snowfall.columns if col not in ['station_name', 'region']]
        if snowfall_cols:
            piv_snowfall['avg_snowfall'] = piv_snowfall[snowfall_cols].mean(axis=1)
            piv_snowfall = piv_snowfall.sort_values('avg_snowfall', ascending=False)
            piv_snowfall = piv_snowfall.drop('avg_snowfall', axis=1)
        
        # Reorder columns: station_name, region, then periods
        cols_to_display = ['station_name', 'region'] + snowfall_cols
        piv_display = piv_snowfall[cols_to_display].copy()
        
        print(f"\n--- All {len(piv_display)} Stations Detail Table ---")
        # Extract year range from period label (e.g., "Dec 01, 1955 - Jan 31, 1956" -> "1955-56")
        year_headers = []
        for p in snowfall_cols:
            # Extract years from period label like "Dec 01, 1955 - Jan 31, 1956"
            import re
            years = re.findall(r'\d{4}', p)
            if len(years) >= 2:
                year_range = f"{years[0][-2:]}-{years[1][-2:]}"  # e.g., "1955-1956" -> "55-56"
            elif len(years) == 1:
                year_range = years[0]
            else:
                year_range = p[:10]
            year_headers.append(year_range.rjust(10))
        
        print(f"{'Station Name':<40} {'Region':<20} {' '.join(year_headers)}")
        print("-" * 150)
        
        for idx, row in piv_display.iterrows():
            snowfall_str = ' '.join([f"{row[p]:>10.2f}" if pd.notna(row[p]) else f"{'N/A':>10}" for p in snowfall_cols])
            print(f"{row['station_name']:<40} {row['region']:<20} {snowfall_str}")

if __name__ == "__main__":
    main()