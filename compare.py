#!/usr/bin/env python3
"""
California Precipitation - Multi-Year & Custom Stretch Regional Comparison
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
    """Returns official California hydrological region polygons and long-term averages."""
    return {
        'North Coast': {
            'avg_precip': 53.21,
            'polygon': [(38.2702032, -122.9776379), (38.3369471, -122.5722272), (38.6094044, -122.6339736), (38.6501083, -122.589511), (39.0353191, -123.0571648), (39.3028288, -123.0097332), (39.2626524, -122.7156631), (40.5471052, -123.0040884), (40.6944964, -122.7657947), (40.9650472, -122.5695359), (40.9694436, -122.5287788), (41.3386684, -122.5149875), (41.4005144, -122.3154672), (41.474002, -122.2469363), (41.6203526, -121.5527067), (41.7581061, -120.531301), (41.9931194, -120.5300127), (41.9956353, -124.2870452), (40.3916193, -124.4964902), (38.2697248, -123.4310139), (38.2701968, -122.9776361)]
        },
        'Sacramento River': {
            'avg_precip': 36.53,
            'polygon': [(41.9928373, -120.5159566), (41.6890215, -120.5859052), (41.4288328, -122.2494046), (41.2844566, -122.4085743), (40.6241342, -122.7383887), (40.1673285, -122.9292018), (39.1520226, -122.7128925), (39.251362, -122.9441957), (39.0329315, -123.0663705), (38.5062378, -122.2980625), (38.4216746, -122.1642765), (38.2882388, -121.9961007), (38.2842684, -121.85033), (38.0209465, -121.8245015), (38.0389557, -121.7070678), (38.256322, -121.4699102), (38.4944265, -121.2253158), (38.6492623, -121.0579403), (38.7215974, -120.7520674), (38.7501726, -120.5531845), (38.6365225, -120.143228), (38.6599081, -119.9701808), (39.0836447, -120.23352), (39.3422579, -120.3168465), (39.4837152, -120.4305762), (39.5066559, -120.4154684), (39.5320049, -120.104523), (40.1028792, -120.326557), (40.4208631, -120.832061), (40.5767865, -121.0976132), (41.08895, -120.5326133), (41.1343096, -120.1627876), (41.7040094, -120.2745954), (41.9923109, -120.2029238), (41.9929928, -120.4012211), (41.9928373, -120.5159566)]
        },
        'North Lahontan': {
            'avg_precip': 21.91,
            'polygon': [(41.9937554, -120.2452047), (41.1323446, -120.2120748), (41.0663628, -120.373286), (40.4929284, -121.2488904), (39.9678333, -120.1227146), (39.5098563, -120.1578947), (39.4904443, -120.4272834), (39.3023438, -120.2897524), (38.8265007, -120.1080995), (38.648032, -119.9644667), (38.6016855, -119.8704507), (38.3276032, -119.6372236), (38.2858988, -119.6511891), (38.2351016, -119.6051498), (38.1958106, -119.633329), (38.1355858, -119.5083085), (38.158683, -119.5035458), (38.0990522, -119.463063), (38.0956165, -119.4420278), (38.1162284, -119.4289305), (38.029975, -119.3071973), (38.1614526, -118.8070832), (39.0039356, -120.0051101), (41.9952167, -119.9996092), (41.9937554, -120.2452047)]
        },
        'San Francisco Bay': {
            'avg_precip': 26.13,
            'polygon': [(38.2704047, -122.9779651), (38.326118, -122.5960843), (38.4669563, -122.5167764), (38.5853798, -122.6582783), (38.6356651, -122.5398115), (38.4796835, -122.2126941), (38.29083, -121.988599), (38.2853313, -121.8603422), (38.0243947, -121.8342063), (37.8282113, -121.9162432), (37.7065351, -121.6501886), (37.2095492, -121.3646549), (37.061077, -121.3916277), (37.1614522, -121.6331366), (37.0739736, -121.7591515), (37.1280391, -121.9773634), (37.2565128, -122.1229452), (37.2127431, -122.156453), (37.2271692, -122.4099784), (37.9956126, -123.0374841), (38.232812, -123.032822), (38.2704047, -122.9779651)]
        },
        'San Joaquin River': {
            'avg_precip': 27.08,
            'polygon': [(36.649354, -120.828199), (38.025366, -121.836623), (38.660766, -119.958766), (37.258328, -118.661045)]
        },
        'Central Coast': {
            'avg_precip': 20.18,
            'polygon': [(37.2455629, -122.4184386), (37.2145472, -122.2423714), (37.2016389, -122.1595981), (37.2579119, -122.1219365), (37.1446709, -121.9846484), (37.0933468, -121.8393121), (37.1731675, -121.7888054), (37.1651094, -121.6309091), (37.0023438, -121.4694994), (37.0860504, -121.2350113), (36.7271981, -121.0167671), (36.6141704, -121.0951125), (36.3269432, -120.5967628), (36.272844, -120.6793971), (36.2014039, -120.6231013), (36.1056446, -120.6538763), (35.8742509, -120.241041), (35.7846053, -120.1950196), (34.8114939, -119.1993905), (34.3775921, -119.4808839), (34.3527657, -121.0781982), (37.2455629, -122.4184386)]
        },
        'Tulare Lake': {
            'avg_precip': 15.81,
            'polygon': [(36.6525036, -120.7645184), (36.7336391, -120.907832), (36.6922928, -121.0692567), (36.5753898, -121.0318159), (36.3726469, -120.7030478), (36.3322647, -120.6091489), (36.209343, -120.6425521), (36.1251178, -120.6682129), (35.8781315, -120.2432014), (35.7887609, -120.2162418), (35.6136878, -120.1950771), (35.0333814, -119.4701225), (34.8572686, -119.2436342), (34.8131625, -119.1570481), (34.8014119, -118.8185102), (35.1371605, -118.3596924), (35.2004322, -118.3692078), (35.2090039, -118.2170963), (35.4144535, -118.3351139), (35.4403265, -118.1219212), (35.7880397, -118.0085525), (36.2621052, -118.1138062), (36.6896617, -118.3587374), (36.9945173, -118.4056432), (37.2121732, -118.6828138), (36.7614617, -120.4065215), (36.6525036, -120.7645184)]
        },
        'South Lahontan': {
            'avg_precip': 7.43,
            'polygon': [(38.1951919, -118.846444), (38.0241651, -119.3062255), (37.9673083, -119.3229534), (37.8847705, -119.2023756), (37.77786, -119.2226926), (37.7242797, -119.2613065), (37.7258851, -119.1109356), (37.6219595, -119.0306759), (37.5847064, -119.0236925), (37.5005257, -118.8612527), (37.4761566, -118.854677), (37.4905134, -118.7981067), (37.4395444, -118.7565709), (37.3388329, -118.7866874), (37.2599111, -118.664874), (37.1410999, -118.656065), (36.064722, -118.4423168), (36.8438566, -118.3588297), (36.6910407, -118.3657248), (36.4813379, -118.2475701), (36.3282616, -118.0979098), (36.0906495, -118.0670259), (35.8686394, -117.983301), (35.7860265, -118.0074298), (35.4247542, -118.098036), (35.36334, -118.318171), (35.0127476, -118.438009), (34.8012065, -118.8041222), (34.6653573, -118.3631704), (34.6176357, -118.3650588), (34.4912146, -118.0853744), (34.3450581, -117.9867532), (34.3721906, -117.7874582), (34.3296411, -117.5768922), (34.3235865, -117.4047953), (34.2214066, -117.2914946), (34.201513, -117.1007601), (34.2834939, -116.9421849), (34.7118611, -117.0467489), (34.8213287, -115.6127995), (35.4398066, -115.177839), (37.8017072, -118.296658), (38.1951919, -118.846444)]
        },
        'South Coast': {
            'avg_precip': 17.46,
            'polygon': [(34.3868003, -119.4983978), (34.6447985, -119.4221001), (34.6666238, -119.1627123), (34.8174766, -119.1681972), (34.8032319, -118.7525271), (34.6456287, -118.3167689), (34.6327268, -118.3859439), (34.4921369, -118.0608696), (34.3463846, -117.9855303), (34.3415524, -117.6010506), (34.3786954, -117.5738064), (34.2221094, -117.279909), (34.2002322, -117.0796451), (34.2877388, -116.9398086), (34.296139, -116.7989238), (34.1707188, -116.6912068), (34.0992638, -116.8330984), (34.0596029, -116.9122667), (33.8757089, -116.9246715), (33.8160931, -116.6732156), (33.561693, -116.55315), (33.4948028, -116.6747532), (33.2099988, -116.482266), (33.2064689, -116.6345239), (33.0570605, -116.5786469), (32.869087, -116.4068149), (32.5982001, -116.3464382), (33.4188615, -117.9023061), (34.2266896, -119.4797679), (34.3868003, -119.4983978)]
        },
        'Colorado River': {
            'avg_precip': 5.34,
            'polygon': [(33.4073534, -114.7125564), (33.5613144, -114.5232322), (33.93216, -114.5265703), (34.1106375, -114.418076), (34.2978507, -114.1280103), (34.4775174, -114.4076641), (34.8319974, -114.5847724), (34.8764914, -114.6379122), (35.005689, -114.6252817), (35.3944509, -115.1219644), (35.2332269, -115.4522383), (34.8570146, -115.6234406), (34.8350224, -115.9188037), (34.7815284, -116.0703595), (34.6846658, -116.0108026), (34.6774736, -116.2702705), (34.7888029, -116.2952263), (34.7336012, -116.5512818), (34.7765124, -116.9988864), (34.4817115, -117.0769542), (34.3849935, -117.0018596), (34.3023356, -116.8339109), (34.296139, -116.7989238), (34.1707188, -116.6912068), (34.0992638, -116.8330984), (34.0596029, -116.9122667), (33.8757089, -116.9246715), (33.8160931, -116.6732156), (33.561693, -116.55315), (33.4948028, -116.6747532), (33.2099988, -116.482266), (33.2064689, -116.6345239), (33.0570605, -116.5786469), (32.869087, -116.4068149), (32.5982001, -116.3464382), (32.7032734, -114.7260471), (32.7466288, -114.7029812), (32.7586209, -114.527175), (33.0117032, -114.4917616), (33.0597579, -114.6817855), (33.4073534, -114.7125564)]
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
    parquet_files = glob.glob('daily_*.parquet')
    if not parquet_files:
        print("No daily_*.parquet files found in current directory.")
        return None

    ids_str = ", ".join([f"'{s}'" for s in matching_ids])
    union_queries = []

    for idx, r in enumerate(date_ranges):
        min_valid = r['min_valid_days']
        union_queries.append(f"""
            SELECT 
                '{idx}' as period_id,
                '{r['label']}' as period_label,
                '{min_valid}' as min_valid_days,
                station_id,
                CAST(date AS DATE) as date,
                CASE 
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS FLOAT)
                END as parsed_precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '{r['start']}'
              AND CAST(date AS DATE) <= DATE '{r['end']}'
        """)

    full_query = " UNION ALL ".join(union_queries)

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),
    flagged AS (
        SELECT *,
            CASE WHEN parsed_precip >= 0.01 THEN 1 ELSE 0 END AS is_wet
        FROM raw_data
        WHERE parsed_precip IS NOT NULL
    ),
    grouped_streaks AS (
        SELECT *,
            SUM(CASE WHEN is_wet = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as wet_grp,
            SUM(CASE WHEN is_wet = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as dry_grp
        FROM flagged
    ),
    streak_lengths AS (
        SELECT 
            period_id, station_id, is_wet, COUNT(*) as streak_len
        FROM grouped_streaks
        GROUP BY period_id, station_id, is_wet, CASE WHEN is_wet = 1 THEN wet_grp ELSE dry_grp END
    ),
    max_streaks AS (
        SELECT 
            period_id, station_id,
            COALESCE(MAX(CASE WHEN is_wet = 1 THEN streak_len END), 0) as max_consec_wet,
            COALESCE(MAX(CASE WHEN is_wet = 0 THEN streak_len END), 0) as max_consec_dry
        FROM streak_lengths
        GROUP BY period_id, station_id
    ),
    station_totals AS (
        SELECT 
            fd.period_id,
            fd.period_label,
            fd.station_id,
            MAX(CAST(fd.min_valid_days AS INT)) as min_valid_days,
            SUM(fd.parsed_precip) as total_precip,
            MAX(fd.parsed_precip) as max_daily_precip,
            COUNT(fd.parsed_precip) as valid_days,
            SUM(CASE WHEN fd.parsed_precip >= 0.01 THEN 1 ELSE 0 END) as wet_days,
            COALESCE(ms.max_consec_wet, 0) as max_consec_wet,
            COALESCE(ms.max_consec_dry, 0) as max_consec_dry
        FROM raw_data fd
        LEFT JOIN max_streaks ms ON fd.period_id = ms.period_id AND fd.station_id = ms.station_id
        GROUP BY fd.period_id, fd.period_label, fd.station_id, ms.max_consec_wet, ms.max_consec_dry
    )
    SELECT 
        st.period_id, st.period_label, st.station_id, st.total_precip, st.max_daily_precip,
        st.valid_days, st.wet_days, st.max_consec_wet, st.max_consec_dry
    FROM station_totals st
    WHERE st.valid_days >= st.min_valid_days;
    """

    df_res = con.execute(sql).df()
    con.close()
    return df_res

def run_duckdb_custom_stretches(matching_ids, start_mmdd, end_mmdd, years, min_valid_ratio=0.70, strict_consistency=True):
    """Queries recurring seasonal stretches across multiple years using DuckDB."""
    con = duckdb.connect()
    parquet_files = glob.glob('daily_*.parquet')
    if not parquet_files:
        print("No daily_*.parquet files found in current directory.")
        return None

    ids_str = ", ".join([f"'{s}'" for s in matching_ids])

    union_queries = []
    start_month, start_day = map(int, start_mmdd.split('-'))
    end_month, end_day = map(int, end_mmdd.split('-'))

    # Calculate total days in the defined stretch to set dynamic min_valid_days
    if (start_month > end_month) or (start_month == end_month and start_day > end_day):
        d1 = datetime(2001, start_month, start_day)
        d2 = datetime(2002, end_month, end_day)
    else:
        d1 = datetime(2001, start_month, start_day)
        d2 = datetime(2001, end_month, end_day)
    
    total_days = (d2 - d1).days + 1
    req_valid_days = max(1, int(total_days * min_valid_ratio))

    for y in years:
        if (start_month > end_month) or (start_month == end_month and start_day > end_day):
            s_date = f"{y}-{start_mmdd}"
            e_date = f"{y + 1}-{end_mmdd}"
            lbl = f"{start_mmdd} ({y}) to {end_mmdd} ({y+1})"
        else:
            s_date = f"{y}-{start_mmdd}"
            e_date = f"{y}-{end_mmdd}"
            lbl = f"{start_mmdd} to {end_mmdd} ({y})"

        union_queries.append(f"""
            SELECT 
                '{y}' as period_id,
                '{lbl}' as period_label,
                station_id,
                CAST(date AS DATE) as date,
                CASE 
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS FLOAT)
                END as parsed_precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '{s_date}'
              AND CAST(date AS DATE) <= DATE '{e_date}'
        """)

    full_query = " UNION ALL ".join(union_queries)

    consistency_clause = f"""
    , consistent_stations AS (
        SELECT station_id
        FROM station_totals
        GROUP BY station_id
        HAVING COUNT(DISTINCT period_id) = {len(years)}
    )
    """ if strict_consistency else ""

    join_clause = "JOIN consistent_stations cs ON st.station_id = cs.station_id" if strict_consistency else ""

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),
    flagged AS (
        SELECT *,
            CASE WHEN parsed_precip >= 0.01 THEN 1 ELSE 0 END AS is_wet
        FROM raw_data
        WHERE parsed_precip IS NOT NULL
    ),
    grouped_streaks AS (
        SELECT *,
            SUM(CASE WHEN is_wet = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as wet_grp,
            SUM(CASE WHEN is_wet = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as dry_grp
        FROM flagged
    ),
    streak_lengths AS (
        SELECT 
            period_id, station_id, is_wet, COUNT(*) as streak_len
        FROM grouped_streaks
        GROUP BY period_id, station_id, is_wet, CASE WHEN is_wet = 1 THEN wet_grp ELSE dry_grp END
    ),
    max_streaks AS (
        SELECT 
            period_id, station_id,
            COALESCE(MAX(CASE WHEN is_wet = 1 THEN streak_len END), 0) as max_consec_wet,
            COALESCE(MAX(CASE WHEN is_wet = 0 THEN streak_len END), 0) as max_consec_dry
        FROM streak_lengths
        GROUP BY period_id, station_id
    ),
    station_totals AS (
        SELECT 
            fd.period_id,
            fd.period_label,
            fd.station_id,
            SUM(fd.parsed_precip) as total_precip,
            MAX(fd.parsed_precip) as max_daily_precip,
            COUNT(fd.parsed_precip) as valid_days,
            SUM(CASE WHEN fd.parsed_precip >= 0.01 THEN 1 ELSE 0 END) as wet_days,
            COALESCE(ms.max_consec_wet, 0) as max_consec_wet,
            COALESCE(ms.max_consec_dry, 0) as max_consec_dry
        FROM raw_data fd
        LEFT JOIN max_streaks ms ON fd.period_id = ms.period_id AND fd.station_id = ms.station_id
        GROUP BY fd.period_id, fd.period_label, fd.station_id, ms.max_consec_wet, ms.max_consec_dry
        HAVING COUNT(fd.parsed_precip) >= {req_valid_days}
    )
    {consistency_clause}
    SELECT 
        st.period_id, st.period_label, st.station_id, st.total_precip, st.max_daily_precip,
        st.valid_days, st.wet_days, st.max_consec_wet, st.max_consec_dry
    FROM station_totals st
    {join_clause};
    """

    df_res = con.execute(sql).df()
    con.close()
    return df_res

def run_duckdb_water_years(matching_ids, years, min_valid_days=200):
    """Queries full water years (July 1 - June 30) using DuckDB directly over Parquet files."""
    con = duckdb.connect()
    parquet_files = glob.glob('daily_*.parquet')
    if not parquet_files:
        print("No daily_*.parquet files found in current directory.")
        return None

    ids_str = ", ".join([f"'{s}'" for s in matching_ids])

    union_queries = []
    for wy in years:
        start_date = f"{wy}-07-01"
        end_date = f"{wy + 1}-06-30"
        union_queries.append(f"""
            SELECT 
                '{wy}' as period_id,
                '{wy}-{str(wy + 1)[-2:]}' as period_label,
                station_id,
                CAST(date AS DATE) as date,
                CASE 
                    WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                    WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                    ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS FLOAT)
                END as parsed_precip
            FROM read_parquet('daily_*.parquet', union_by_name=true)
            WHERE station_id IN ({ids_str})
              AND CAST(date AS DATE) >= DATE '{start_date}'
              AND CAST(date AS DATE) <= DATE '{end_date}'
        """)

    full_query = " UNION ALL ".join(union_queries)

    sql = f"""
    WITH raw_data AS (
        {full_query}
    ),
    flagged AS (
        SELECT *,
            CASE WHEN parsed_precip >= 0.01 THEN 1 ELSE 0 END AS is_wet
        FROM raw_data
        WHERE parsed_precip IS NOT NULL
    ),
    grouped_streaks AS (
        SELECT *,
            SUM(CASE WHEN is_wet = 0 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as wet_grp,
            SUM(CASE WHEN is_wet = 1 THEN 1 ELSE 0 END) OVER (PARTITION BY period_id, station_id ORDER BY date) as dry_grp
        FROM flagged
    ),
    streak_lengths AS (
        SELECT 
            period_id, station_id, is_wet, COUNT(*) as streak_len
        FROM grouped_streaks
        GROUP BY period_id, station_id, is_wet, (CASE WHEN is_wet = 1 THEN wet_grp ELSE dry_grp END)
    ),
    max_streaks AS (
        SELECT 
            period_id,
            station_id,
            MAX(CASE WHEN is_wet = 1 THEN streak_len ELSE 0 END) as max_consec_wet,
            MAX(CASE WHEN is_wet = 0 THEN streak_len ELSE 0 END) as max_consec_dry
        FROM streak_lengths
        GROUP BY period_id, station_id
    ),
    station_totals AS (
        SELECT 
            fd.period_id,
            fd.period_label,
            fd.station_id,
            SUM(fd.parsed_precip) as total_precip,
            MAX(fd.parsed_precip) as max_daily_precip,
            COUNT(fd.parsed_precip) as valid_days,
            SUM(CASE WHEN fd.parsed_precip >= 0.01 THEN 1 ELSE 0 END) as wet_days,
            COALESCE(ms.max_consec_wet, 0) as max_consec_wet,
            COALESCE(ms.max_consec_dry, 0) as max_consec_dry
        FROM raw_data fd
        LEFT JOIN max_streaks ms ON fd.period_id = ms.period_id AND fd.station_id = ms.station_id
        GROUP BY fd.period_id, fd.period_label, fd.station_id, ms.max_consec_wet, ms.max_consec_dry
    )
    SELECT 
        st.period_id, st.period_label, st.station_id, st.total_precip, st.max_daily_precip,
        st.valid_days, st.wet_days, st.max_consec_wet, st.max_consec_dry
    FROM station_totals st
    WHERE st.valid_days >= {min_valid_days};
    """

    df_res = con.execute(sql).df()
    con.close()
    return df_res


def run_duckdb_rolling_extremes(matching_ids, window_days, min_valid_ratio=0.70):
    """Find every rolling N-day precipitation total in the dataset."""
    con = duckdb.connect()
    if not glob.glob('daily_*.parquet'):
        print("No daily_*.parquet files found in current directory.")
        return None

    ids_str = ", ".join([f"'{s}'" for s in matching_ids])

    sql = f"""
    WITH daily AS (
        SELECT
            station_id,
            CAST(date AS DATE) AS date,
            CASE
                WHEN precip IS NULL OR TRIM(CAST(precip AS VARCHAR)) IN ('', 'M') THEN NULL
                WHEN TRIM(CAST(precip AS VARCHAR)) = 'T' THEN 0.0
                ELSE TRY_CAST(TRIM(REGEXP_REPLACE(CAST(precip AS VARCHAR), '[AS]$', '')) AS DOUBLE)
            END AS parsed_precip
        FROM read_parquet('daily_*.parquet', union_by_name=true)
        WHERE station_id IN ({ids_str})
          AND CAST(date AS DATE) >= DATE '1890-01-01'
          AND CAST(date AS DATE) <= DATE '2026-12-31'
    ),
    rolling AS (
        SELECT
            station_id,
            date - INTERVAL '{window_days - 1} days' AS period_start,
            date AS period_end,
            SUM(parsed_precip) OVER (
                PARTITION BY station_id
                ORDER BY date
                RANGE BETWEEN INTERVAL '{window_days - 1} days' PRECEDING AND CURRENT ROW
            ) AS total_precip,
            COUNT(parsed_precip) OVER (
                PARTITION BY station_id
                ORDER BY date
                RANGE BETWEEN INTERVAL '{window_days - 1} days' PRECEDING AND CURRENT ROW
            ) AS valid_days
        FROM daily
        WHERE parsed_precip IS NOT NULL
    )
    SELECT *
    FROM rolling
    WHERE valid_days >= CEIL({window_days} * {min_valid_ratio})
    """

    df = con.execute(sql).df()
    con.close()
    return df


def print_extreme_table(df, title, selected_regions=None, station_region_map=None):
    """Print lowest/highest period totals, optionally split by region."""
    if df is None or df.empty:
        print(f"\n{title}: no valid periods found.")
        return

    print("\n" + "=" * 95)
    print(title)
    print("=" * 95)

    if selected_regions is not None:
        df = df.copy()
        df['region'] = df['station_id'].map(station_region_map)

        for region in selected_regions:
            reg = df[df['region'] == region]
            if reg.empty:
                print(f"\n{region}: no valid periods found.")
                continue

            stats = reg.groupby('period_label').agg(
                precip=('total_precip', 'mean'),
                station_count=('station_id', 'count')
            ).reset_index()

            low = stats.loc[stats['precip'].idxmin()]
            high = stats.loc[stats['precip'].idxmax()]

            print(f"\n{region}")
            print(
                f"  LOWEST:  {low['precip']:.2f}\" — "
                f"{low['period_label']} ({int(low['station_count'])} stations)"
            )
            print(
                f"  HIGHEST: {high['precip']:.2f}\" — "
                f"{high['period_label']} ({int(high['station_count'])} stations)"
            )
    else:
        stats = df.groupby('period_label').agg(
            precip=('total_precip', 'mean'),
            station_count=('station_id', 'count')
        ).reset_index()

        low = stats.loc[stats['precip'].idxmin()]
        high = stats.loc[stats['precip'].idxmax()]

        print(
            f"  LOWEST:  {low['precip']:.2f}\" — "
            f"{low['period_label']} ({int(low['station_count'])} stations)"
        )
        print(
            f"  HIGHEST: {high['precip']:.2f}\" — "
            f"{high['period_label']} ({int(high['station_count'])} stations)"
        )


def extreme_mode(selected_regions, regions, station_region_map, metadata):
    """Interactive all-records/extremes mode."""
    statewide_ids = set(metadata.keys())
    regional_ids = {
        sid for sid, region in station_region_map.items()
        if region in selected_regions
    }

    print("\nExtremes / Records Mode")
    print("Searches the complete 1890-2026 dataset for the lowest and highest")
    print("precipitation totals for the selected period definition.")
    print("\nPeriod type:")
    print("  1. Water Years (July 1 - June 30)")
    print("  2. Recurring Seasonal / Custom Calendar Stretch (e.g. Nov 12 - Feb 18)")
    print("  3. Rolling N-Day Window (e.g. 1, 7, 30, 90, 365 days)")

    choice = input("Select period type (1, 2, or 3): ").strip()

    if choice == '1':
        # WY 2026 is incomplete, so records use complete WYs through WY 2025.
        years = list(range(1890, 2026))
        print("\nSearching complete water years WY 1890 through WY 2025...")

        regional_df = run_duckdb_water_years(
            regional_ids, years, min_valid_days=100
        )
        statewide_df = run_duckdb_water_years(
            statewide_ids, years, min_valid_days=100
        )

        print_extreme_table(
            regional_df,
            "REGIONAL WATER-YEAR RECORDS",
            selected_regions,
            station_region_map
        )
        print_extreme_table(
            statewide_df,
            "STATEWIDE CALIFORNIA WATER-YEAR RECORDS"
        )

    elif choice == '2':
        start_mmdd = input(
            "Enter start date (MM-DD, e.g., 11-12): "
        ).strip()
        end_mmdd = input(
            "Enter end date (MM-DD, e.g., 02-18): "
        ).strip()

        try:
            sm, sd = map(int, start_mmdd.split('-'))
            em, ed = map(int, end_mmdd.split('-'))

            datetime(2001, sm, sd)
            datetime(2001, em, ed)
        except ValueError:
            print("Invalid MM-DD date.")
            return

        # Cross-year stretches ending in the following calendar year
        # cannot use a 2026 start because their ending portion would
        # extend beyond the dataset.
        cross_year = (sm, sd) > (em, ed)
        last_start_year = 2025 if cross_year else 2026
        years = list(range(1890, last_start_year + 1))

        print(
            f"\nSearching every occurrence from 1890 through "
            f"{last_start_year}..."
        )

        regional_df = run_duckdb_custom_stretches(
            regional_ids,
            start_mmdd,
            end_mmdd,
            years,
            min_valid_ratio=0.50,
            strict_consistency=False
        )
        statewide_df = run_duckdb_custom_stretches(
            statewide_ids,
            start_mmdd,
            end_mmdd,
            years,
            min_valid_ratio=0.50,
            strict_consistency=False
        )

        print_extreme_table(
            regional_df,
            "REGIONAL SEASONAL / CUSTOM-STRETCH RECORDS",
            selected_regions,
            station_region_map
        )
        print_extreme_table(
            statewide_df,
            "STATEWIDE CALIFORNIA SEASONAL / CUSTOM-STRETCH RECORDS"
        )

    elif choice == '3':
        raw_days = input(
            "Enter rolling window length in days (e.g. 1, 7, 30, 90, 365): "
        ).strip()

        try:
            window_days = int(raw_days)
            if not 1 <= window_days <= 3650:
                raise ValueError
        except ValueError:
            print("Window length must be an integer from 1 to 3650.")
            return

        print(
            f"\nSearching every {window_days}-day window from "
            "1890-01-01 through 2026-12-31..."
        )

        regional_df = run_duckdb_rolling_extremes(
            regional_ids,
            window_days,
            min_valid_ratio=0.70
        )
        statewide_df = run_duckdb_rolling_extremes(
            statewide_ids,
            window_days,
            min_valid_ratio=0.70
        )

        # Convert rolling result into the same shape used by the
        # normal extreme table.
        for df in (regional_df, statewide_df):
            if df is not None and not df.empty:
                df['period_label'] = (
                    df['period_start'].dt.strftime('%Y-%m-%d')
                    + " to "
                    + df['period_end'].dt.strftime('%Y-%m-%d')
                )

        print_extreme_table(
            regional_df,
            f"REGIONAL {window_days}-DAY ROLLING RECORDS",
            selected_regions,
            station_region_map
        )
        print_extreme_table(
            statewide_df,
            f"STATEWIDE CALIFORNIA {window_days}-DAY ROLLING RECORDS"
        )

    else:
        print("Invalid extreme-mode selection.")


def main():
    print("="*95)
    print("California Precipitation - Multi-Region & Multi-Period Comparison (DuckDB)")
    print("="*95)
    
    regions = get_official_regions()
    region_list = sorted(regions.keys())
    
    print("\nHydrological Region Selection:")
    print("  0. ALL Hydrological Regions (Full Multi-Region Comparison)")
    for i, region in enumerate(region_list, 1):
        print(f"  {i:2}. {region} (WY Base: {regions[region]['avg_precip']:.2f}\")")
    
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
    
    print("\nAnalysis Mode:")
    print("  1. Comparison Mode")
    print("  2. Extremes / Records")
    analysis_mode = input("Select mode (1 or 2): ").strip()

    if analysis_mode == '2':
        extreme_mode(selected_regions, regions, station_region_map, metadata)
        return

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
        raw_years = input("\nEnter water years (e.g., '1982, 1997, 2015' or '2015-2020'): ")
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

    # --- OPTION A: Filter to stations present across ALL selected periods ---
    all_periods = df_res['period_label'].unique()
    stations_in_all_periods = (
        df_res.groupby('station_id')['period_label']
        .nunique()
        .loc[lambda x: x == len(all_periods)]
        .index
    )
    df_res = df_res[df_res['station_id'].isin(stations_in_all_periods)]
    # ------------------------------------------------------------------------

    # Aggregate by region and period
    summary = df_res.groupby(['region', 'period_id', 'period_label']).agg(
        avg_precip=('total_precip', 'mean'),
        avg_wet_days=('wet_days', 'mean'),
        avg_max_daily=('max_daily_precip', 'mean'),
        avg_consec_wet=('max_consec_wet', 'mean'),
        avg_consec_dry=('max_consec_dry', 'mean'),
        station_count=('station_id', 'count')
    ).reset_index()

    # Calculate baseline % per region
    summary['regional_base'] = summary['region'].map(lambda r: regions[r]['avg_precip'])
    summary['pct_base'] = (summary['avg_precip'] / summary['regional_base']) * 100

    print("\n" + "="*70)
    print("REGIONAL COMPARISON SUMMARY TABLE")
    print("="*70)
    
    # Group by region and print with station count in header
    for region in summary['region'].unique():
        region_data = summary[summary['region'] == region].sort_values('period_id')
        
        # For modes 2 & 3, count stations that appear in ALL periods; for mode 1, use first period count
        if mode_choice in ['2', '3']:
            # Get stations for this region in each period
            region_df = df_res[df_res['station_id'].map(lambda s: station_region_map.get(s) == region)]
            stations_by_period = []
            for period in region_data['period_label'].unique():
                period_stations = set(region_df[region_df['period_label'] == period]['station_id'])
                stations_by_period.append(period_stations)
            # Intersection: stations in all periods
            if stations_by_period:
                common_stations = set.intersection(*stations_by_period) if len(stations_by_period) > 1 else stations_by_period[0]
                station_count = len(common_stations)
            else:
                station_count = 0
        else:
            station_count = region_data['station_count'].iloc[0]
        
        print(f"\n{region} ({int(station_count)} stations)")
        print(f"{'Period':<30} {'Avg Precip':>11} {'% WY Base':>10}")
        print("-" * 55)
        
        for _, row in region_data.iterrows():
            print(f"{row['period_label']:<30} {row['avg_precip']:>10.2f}\" {row['pct_base']:>9.1f}%")
    
    # Add statewide average if multiple regions selected
    if len(selected_regions) > 1:
        print("\n" + "="*70)
        print("STATEWIDE AVERAGE (All Stations)")
        print("=" * 70)
        
        # Calculate TRUE statewide average from ALL individual stations in df_res
        # Group by period and calculate mean precip across ALL stations
        statewide_periods = df_res.groupby('period_label').agg({
            'total_precip': 'mean'
        }).reset_index()
        statewide_periods.columns = ['period_label', 'avg_precip']
        
        # Calculate % of statewide WY base
        statewide_wy_base = 26.0  # CA statewide average for full water year
        statewide_periods['pct_base'] = (statewide_periods['avg_precip'] / statewide_wy_base) * 100
        
        print(f"{'Period':<30} {'Avg Precip':>11} {'% WY Base':>10}")
        print("-" * 55)
        
        for _, row in statewide_periods.iterrows():
            print(f"{row['period_label']:<30} {row['avg_precip']:>10.2f}\" {row['pct_base']:>9.1f}%")
    
    print("\n" + "="*70)

    # Multi-Region side-by-side view if multiple regions selected
    if len(selected_regions) > 1:
        print("\n" + "="*95)
        print("SIDE-BY-SIDE REGIONAL PRECIPITATION MATRIX (Inches)")
        print("="*95)
        piv_precip = summary.pivot(index='region', columns='period_label', values='avg_precip')
        
        # Add statewide average row (calculated from ALL stations, not region means)
        statewide_avg = df_res.groupby('period_label')['total_precip'].mean()
        piv_precip.loc['STATEWIDE AVERAGE'] = statewide_avg
        
        # Format to 2 decimal places
        piv_precip_formatted = piv_precip.round(2)
        print(piv_precip_formatted.to_string())

    show_detail = input("\nShow detailed station breakdown? (yes/no): ").lower().strip()
    if show_detail in ['yes', 'y']:
        # For modes 2 & 3, filter to stations in ALL periods (intersection)
        if mode_choice in ['2', '3']:
            all_stations = set(df_res['station_id'])
            for period in df_res['period_label'].unique():
                period_stations = set(df_res[df_res['period_label'] == period]['station_id'])
                all_stations = all_stations.intersection(period_stations)
            df_detail = df_res[df_res['station_id'].isin(all_stations)]
        else:
            df_detail = df_res
        
        # Get precipitation pivot
        piv_precip = df_detail.pivot(index='station_id', columns='period_label', values='total_precip')
        piv_precip['station_name'] = piv_precip.index.map(lambda sid: metadata.get(sid, {}).get('name', sid))
        piv_precip['region'] = piv_precip.index.map(lambda sid: station_region_map.get(sid, 'Unknown'))
        
        # Sort by average precipitation across periods
        precip_cols = [col for col in piv_precip.columns if col not in ['station_name', 'region']]
        if precip_cols:
            piv_precip['avg_precip'] = piv_precip[precip_cols].mean(axis=1)
            piv_precip = piv_precip.sort_values('avg_precip', ascending=False)
            piv_precip = piv_precip.drop('avg_precip', axis=1)
        
        # Reorder columns: station_name, region, then periods
        cols_to_display = ['station_name', 'region'] + precip_cols
        piv_display = piv_precip[cols_to_display].copy()
        
        # Drop rows with N/A / NaN values in any period column
        piv_display = piv_display.dropna(subset=precip_cols)
        
        print(f"\n--- All {len(piv_display)} Stations Detail Table ---")
        
        # Extract year range from period label
        year_headers = []
        for p in precip_cols:
            years = re.findall(r'\d{4}', p)
            if len(years) >= 2:
                year_range = f"{years[0][-2:]}-{years[1][-2:]}"
            elif len(years) == 1:
                year_range = years[0]
            else:
                year_range = p[:10]
            year_headers.append(year_range.rjust(10))
        
        print(f"{'Station Name':<40} {'Region':<20} {' '.join(year_headers)}")
        print("-" * 150)
        
        for idx, row in piv_display.iterrows():
            precip_str = ' '.join([f"{row[p]:>10.2f}" for p in precip_cols])
            print(f"{row['station_name']:<40} {row['region']:<20} {precip_str}")

if __name__ == "__main__":
    main()