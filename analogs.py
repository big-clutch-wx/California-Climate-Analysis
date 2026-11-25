"""
analogs.py - 500 hPa analog search for the California Climate Analysis app.

Finds the dates in the ERA5 record whose 500 hPa height anomaly pattern best
matches a target pattern (an AIFS forecast field, or a past date for testing),
then labels each analog with the MJO state and, in app.py, what California got.

Nothing here imports Streamlit, so it can be tested on its own.

DATA
  The ERA5 record is not fetched while the app runs. build_z500_archive.py makes
  one small file (daily 12 UTC 500 hPa height on a 2.5 degree grid, 1940 to
  now) and this module reads it. See that script for how to build and host it.

  Archive file (.npz):
      dates  int32   days since 1970-01-01
      lat    float32 ascending
      lon    float32 ascending, degrees east (-180..180)
      z      int16   [time, lat, lon] 500 hPa geopotential height in metres;
                     -32768 marks missing

METHOD
  1. Anomaly = height minus a smooth day-of-year climatology (3 harmonics fitted
     to 1991-2020), so matching does not just pick the same season.
  2. Optional per-gridpoint linear detrend, because 500 hPa heights have risen
     over the record and would otherwise favour recent dates.
  3. Score every day: cos(latitude)-weighted anomaly pattern correlation, RMSE
     and amplitude ratio over the chosen domain.
  4. Pick the best days, skipping days within min_sep_days of an already picked
     day, so the list is not five consecutive days of one storm.
"""

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

APP_DIR = Path(__file__).resolve().parent
ARCHIVE_NAME = "z500_12z_daily.npz"
ARCHIVE_URL = "https://storage.googleapis.com/cawn2/analogs/" + ARCHIVE_NAME
MISSING_Z = -32768

BASE_PERIOD = (1991, 2020)
HARMONICS = 3

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
AIFS_MODEL = "ecmwf_aifs025_single"
Z500_VAR = "geopotential_height_500hPa"

# BoM publishes the Wheeler-Hendon RMM index back to 1974 as a text file.
MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"

# Named analysis domains: (lat_min, lat_max, lon_min, lon_max), degrees east.
DOMAINS = {
    "Eastern Pacific / West Coast (25-55N, 170-110W)": (25, 55, -170, -110),
    "North Pacific to North America (20-65N, 180-90W)": (20, 65, -180, -90),
    "Tight California region (30-50N, 140-110W)": (30, 50, -140, -110),
}


# ---------------------------------------------------------------------------
# Loading the archive
# ---------------------------------------------------------------------------

def archive_candidates():
    """Places to look for the archive, in order: env var, working dir, app dir, URL."""
    out = []
    env = os.environ.get("ACIS_Z500_ARCHIVE")
    if env:
        out.append(env)
    out.append(str(Path.cwd() / ARCHIVE_NAME))
    out.append(str(APP_DIR / ARCHIVE_NAME))
    out.append(ARCHIVE_URL)
    return out


def check_archive(arch):
    """Refuse an archive whose fields are flat east-west or north-south (a build bug)."""
    z = arch["z"]
    sample = z[:: max(1, len(z) // 200)].astype(float)
    sample[sample == MISSING_Z] = np.nan
    across_lon = float(np.nanmean(np.nanstd(sample, axis=2)))
    across_lat = float(np.nanmean(np.nanstd(sample, axis=1)))
    if across_lon < 1.0 or across_lat < 1.0:
        raise ValueError(
            "The 500 hPa archive is not usable: its fields do not vary like real heights "
            f"(spread across longitude {across_lon:.2f} m, across latitude {across_lat:.2f} m). "
            "It was probably built incorrectly. Rebuild it with build_z500_archive_arco.py "
            "(current version) or build_z500_archive.py."
        )


def load_archive(source=None):
    """Return (archive dict, where it came from). Raises FileNotFoundError if none works."""
    errors = []
    for src in ([source] if source else archive_candidates()):
        try:
            if str(src).startswith("http"):
                resp = requests.get(src, timeout=120)
                resp.raise_for_status()
                handle = io.BytesIO(resp.content)
            elif Path(src).exists():
                handle = src
            else:
                errors.append(f"{src}: not found")
                continue
            with np.load(handle) as z:
                arch = {k: z[k] for k in ("dates", "lat", "lon", "z")}
            check_archive(arch)
            return arch, str(src)
        except Exception as exc:  # noqa: BLE001 - try the next place
            errors.append(f"{src}: {exc}")
    raise FileNotFoundError(
        "No 500 hPa archive found. Build it with build_z500_archive.py and put "
        + ARCHIVE_NAME + " next to app.py, or upload it to the cawn2 bucket under "
        "analogs/. Tried: " + "; ".join(errors)
    )


# ---------------------------------------------------------------------------
# Climatology, anomalies, trend
# ---------------------------------------------------------------------------

def _float_doy(dates):
    """Day of year on a 365-day calendar; Feb 29 sits halfway between Feb 28 and Mar 1."""
    idx = pd.DatetimeIndex(dates)
    doy = idx.dayofyear.to_numpy().astype(float)
    leap = idx.is_leap_year
    feb29 = leap & (idx.month == 2) & (idx.day == 29)
    doy = np.where(leap & (idx.month > 2), doy - 1.0, doy)
    doy = np.where(feb29, 59.5, doy)
    return doy


def _design(dates, harmonics=HARMONICS):
    theta = 2.0 * np.pi * (_float_doy(dates) - 0.5) / 365.0
    cols = [np.ones(len(theta))]
    for k in range(1, harmonics + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    return np.column_stack(cols)


def _year_frac(dates):
    d = np.asarray(dates).astype("datetime64[D]")
    years = d.astype("datetime64[Y]")
    frac = (d - years.astype("datetime64[D]")).astype(float) / 365.25
    return years.astype(int) + 1970 + frac


@dataclass
class Prepared:
    dates: np.ndarray        # datetime64[D], [t]
    lat: np.ndarray          # [ny]
    lon: np.ndarray          # [nx]
    anom: np.ndarray         # float32 [t, ny, nx], NaN where missing
    coef: np.ndarray         # climatology harmonics, [1+2K, ny*nx]
    trend: tuple             # (intercept, slope) per point, each [ny*nx]; or None
    stride: int
    detrended: bool


def prepare(arch, stride=1, detrend=True, base=BASE_PERIOD, harmonics=HARMONICS):
    """Turn the archive into anomalies on the (optionally thinned) grid."""
    dates = arch["dates"].astype("int64").astype("datetime64[D]")
    lat = np.asarray(arch["lat"], dtype=float)[::stride]
    lon = np.asarray(arch["lon"], dtype=float)[::stride]
    z = arch["z"][:, ::stride, ::stride]
    nt, ny, nx = z.shape
    flat = z.reshape(nt, ny * nx)
    zf = np.where(flat == MISSING_Z, np.nan, flat.astype(np.float32))

    years = pd.DatetimeIndex(dates).year.to_numpy()
    in_base = (years >= base[0]) & (years <= base[1])
    complete = ~np.isnan(zf).any(axis=1)
    fit_rows = in_base & complete
    if fit_rows.sum() < 365:
        raise ValueError(
            f"Only {int(fit_rows.sum())} complete days in {base[0]}-{base[1]}; "
            "cannot build a climatology."
        )
    X = _design(dates, harmonics)
    coef, *_ = np.linalg.lstsq(X[fit_rows], zf[fit_rows].astype(np.float64), rcond=None)

    anom = np.empty_like(zf)
    for s in range(0, nt, 2000):
        e = min(s + 2000, nt)
        anom[s:e] = zf[s:e] - (X[s:e] @ coef).astype(np.float32)

    trend = None
    if detrend:
        t = _year_frac(dates)
        ok = ~np.isnan(anom)
        # Closed-form least squares per gridpoint, ignoring missing days.
        w = ok.astype(np.float64)
        n = w.sum(axis=0)
        tm = (w * t[:, None]).sum(axis=0) / n
        a0 = np.where(ok, anom, 0.0).astype(np.float64)
        ym = a0.sum(axis=0) / n
        dt = (t[:, None] - tm) * w
        slope = (dt * (a0 - ym)).sum(axis=0) / (dt * (t[:, None] - tm)).sum(axis=0)
        intercept = ym - slope * tm
        trend = (intercept, slope)
        for s in range(0, nt, 2000):
            e = min(s + 2000, nt)
            anom[s:e] -= (intercept + slope * t[s:e, None]).astype(np.float32)

    return Prepared(dates, lat, lon, anom.reshape(nt, ny, nx), coef, trend,
                    stride, bool(detrend))


def target_from_field(prep, z_field, date):
    """Anomaly of a forecast height field [ny, nx] (metres) valid on `date`."""
    d = np.array([np.datetime64(pd.Timestamp(date).date(), "D")])
    clim = (_design(d, (prep.coef.shape[0] - 1) // 2) @ prep.coef).reshape(len(prep.lat), len(prep.lon))
    out = np.asarray(z_field, dtype=float) - clim
    if prep.trend is not None:
        t = _year_frac(d)[0]
        icpt, slope = prep.trend
        out = out - (icpt + slope * t).reshape(out.shape)
    return out


def target_from_history(prep, date):
    """Anomaly of an archive date, exactly as it is stored. Raises KeyError if absent."""
    d = np.datetime64(pd.Timestamp(date).date(), "D")
    hit = np.nonzero(prep.dates == d)[0]
    if len(hit) == 0:
        raise KeyError(f"{pd.Timestamp(date).date()} is not in the archive")
    return prep.anom[hit[0]].astype(float)


# ---------------------------------------------------------------------------
# Fetching the AIFS target
# ---------------------------------------------------------------------------

def _open_meteo_get(params):
    resp = None
    for attempt in range(2):
        try:
            resp = requests.get(OPEN_METEO_URL, params=params, timeout=60)
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Open-Meteo ({type(exc).__name__}).") from None
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        if attempt == 0:
            time.sleep(min(float(resp.headers.get("Retry-After", 10)), 20))
    raise RuntimeError("Open-Meteo is rate limiting right now. Wait a few minutes and retry.")


def fetch_aifs_z500(lat, lon, valid_date, hour_utc=12, model=AIFS_MODEL, batch=100):
    """AIFS 500 hPa height (m) on the lat x lon grid for `valid_date` at `hour_utc`.

    Open-Meteo returns the model's own 0.25 degree grid cell nearest each point.
    Returns an array [len(lat), len(lon)]. Raises if any point has no value, which
    is what happens when the date is beyond the model's range.
    """
    day = pd.Timestamp(valid_date).strftime("%Y-%m-%d")
    stamp = f"{day}T{hour_utc:02d}:00"
    pts = [(float(la), float(lo)) for la in lat for lo in lon]
    vals = np.full(len(pts), np.nan)
    for s in range(0, len(pts), batch):
        chunk = pts[s:s + batch]
        data = _open_meteo_get({
            "latitude": ",".join(f"{p[0]:.2f}" for p in chunk),
            "longitude": ",".join(f"{p[1]:.2f}" for p in chunk),
            "hourly": Z500_VAR,
            "models": model,
            "start_date": day,
            "end_date": day,
            "timezone": "GMT",
        })
        if isinstance(data, dict):
            data = [data]
        for k, item in enumerate(data):
            hourly = item.get("hourly", {})
            times = hourly.get("time", [])
            series = hourly.get(Z500_VAR)
            if series is None or stamp not in times:
                continue
            v = series[times.index(stamp)]
            if v is not None:
                vals[s + k] = v
    if np.isnan(vals).any():
        raise ValueError(
            f"{model} returned no 500 hPa height for {int(np.isnan(vals).sum())} of "
            f"{len(pts)} points on {stamp}Z. The date may be past the model's range, "
            "or Open-Meteo does not serve this variable for this model."
        )
    return vals.reshape(len(lat), len(lon))


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def domain_slices(prep, domain):
    """Index arrays for the lat/lon box (lat_min, lat_max, lon_min, lon_max)."""
    la0, la1, lo0, lo1 = domain
    iy = np.nonzero((prep.lat >= la0) & (prep.lat <= la1))[0]
    ix = np.nonzero((prep.lon >= lo0) & (prep.lon <= lo1))[0]
    if len(iy) < 2 or len(ix) < 2:
        raise ValueError("The domain is too small for this grid.")
    return iy, ix


def rank_analogs(prep, target, domain, n=10, window_days=183, target_date=None,
                 min_sep_days=5, by="corr", exclude_date=None):
    """Best-matching days. Returns a DataFrame sorted best first.

    window_days: keep only days within +/- this many days of the target's date of
                 year (183 or more keeps the whole record).
    exclude_date: drop days within min_sep_days of this date (use when the target
                 is itself a date in the archive).
    by: "corr" (higher is better) or "rmse" (lower is better).
    """
    iy, ix = domain_slices(prep, domain)
    sub = prep.anom[:, iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
    tgt = np.asarray(target, dtype=float)[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
    if np.isnan(tgt).any():
        raise ValueError("The target field has missing values inside the domain.")

    w = np.cos(np.deg2rad(prep.lat[iy[0]:iy[-1] + 1]))[:, None] * np.ones((1, sub.shape[2]))
    w = (w / w.sum()).reshape(-1)
    A = sub.reshape(sub.shape[0], -1)
    t = tgt.reshape(-1)

    valid = ~np.isnan(A).any(axis=1)
    A0 = np.where(valid[:, None], A, 0.0)
    cross = (A0 * (w * t)).sum(axis=1)
    a2 = (A0 ** 2 * w).sum(axis=1)
    t2 = float((t ** 2 * w).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cross / np.sqrt(a2 * t2)
        rmse = np.sqrt(np.maximum(a2 + t2 - 2.0 * cross, 0.0))
        amp = np.sqrt(a2 / t2)
    corr[~valid] = np.nan

    keep = valid.copy()
    if target_date is not None and window_days < 183:
        d0 = _float_doy([pd.Timestamp(target_date)])[0]
        diff = np.abs(_float_doy(prep.dates) - d0)
        diff = np.minimum(diff, 365.0 - diff)
        keep &= diff <= window_days

    day_num = prep.dates.astype("int64")
    if exclude_date is not None:
        ex = int(np.datetime64(pd.Timestamp(exclude_date).date(), "D").astype("int64"))
        keep &= np.abs(day_num - ex) > min_sep_days

    score = corr if by == "corr" else -rmse
    cand = np.nonzero(keep)[0]
    cand = cand[np.argsort(-score[cand])]

    picked = []
    for i in cand:
        if all(abs(day_num[i] - day_num[j]) > min_sep_days for j in picked):
            picked.append(i)
            if len(picked) >= n:
                break

    out = pd.DataFrame({
        "rank": np.arange(1, len(picked) + 1),
        "date": pd.to_datetime(prep.dates[picked]),
        "corr": corr[picked],
        "rmse_m": rmse[picked],
        "amp_ratio": amp[picked],
        "index": picked,
    })
    return out.reset_index(drop=True)


def analog_field(prep, index):
    return prep.anom[int(index)].astype(float)


def composite(prep, indices):
    """Mean anomaly field of several analogs."""
    return np.nanmean(prep.anom[np.asarray(indices, dtype=int)], axis=0).astype(float)


# ---------------------------------------------------------------------------
# MJO (Wheeler-Hendon RMM, from the Bureau of Meteorology)
# ---------------------------------------------------------------------------

def parse_rmm(text):
    """Parse the BoM RMM text file into a DataFrame indexed by date."""
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 7:
            continue
        try:
            y, m, d = int(f[0]), int(f[1]), int(f[2])
            phase = int(float(f[5]))
            amp = float(f[6])
        except ValueError:
            continue
        if amp >= 1e30 or phase == 999 or amp >= 999:
            continue
        rows.append((pd.Timestamp(y, m, d), phase, amp))
    if not rows:
        raise ValueError("No RMM rows found in the file.")
    df = pd.DataFrame(rows, columns=["date", "phase", "amp"]).set_index("date")
    return df[~df.index.duplicated(keep="last")]


def load_mjo(url=MJO_URL):
    resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    return parse_rmm(resp.text)


def mjo_label(mjo, date):
    """'P6 1.4' for an active MJO (amplitude >= 1), 'weak 0.6' otherwise, '' if unknown."""
    if mjo is None:
        return ""
    try:
        row = mjo.loc[pd.Timestamp(date).normalize()]
    except KeyError:
        return ""
    if row["amp"] < 1.0:
        return f"weak {row['amp']:.1f}"
    return f"P{int(row['phase'])} {row['amp']:.1f}"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_pattern_maps(panels, lat, lon, outline=None, zmax=None, ncols=3, height_per_row=300,
                     scale=1.0, unit="m"):
    """Small multiples of 500 hPa anomaly maps.

    panels: list of (title, 2-D anomaly field [ny, nx]). All share one symmetric
    colour scale. outline: optional list of (lat, lon) drawn on every map.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n = len(panels)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    # scale: divide the fields (metres) by this to show other units, e.g. scale=10, unit="dam".
    panels = [(t, np.asarray(f, dtype=float) / scale) for t, f in panels]
    if zmax is None:
        step, floor = 20.0 / scale, 60.0 / scale
        zmax = max(float(np.nanmax(np.abs(f))) for _, f in panels)
        zmax = max(floor, float(np.ceil(zmax / step) * step))
    hover_fmt = ".0f" if scale == 1 else ".1f"

    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[t for t, _ in panels],
                        horizontal_spacing=0.04, vertical_spacing=0.10)
    for k, (_, field) in enumerate(panels):
        r, c = divmod(k, ncols)
        fig.add_trace(
            go.Heatmap(
                z=field, x=lon, y=lat, zmin=-zmax, zmax=zmax, zmid=0,
                colorscale="RdBu_r", zsmooth="best",
                showscale=(k == 0),
                colorbar=dict(title=unit, len=0.6, thickness=12) if k == 0 else None,
                hovertemplate="lon %{x}<br>lat %{y}<br>%{z:" + hover_fmt + "} " + unit + "<extra></extra>",
            ),
            row=r + 1, col=c + 1,
        )
        if outline:
            fig.add_trace(
                go.Scatter(x=[p[1] for p in outline], y=[p[0] for p in outline],
                           mode="lines", line=dict(color="black", width=1.2),
                           hoverinfo="skip", showlegend=False),
                row=r + 1, col=c + 1,
            )
        fig.update_xaxes(range=[float(lon[0]), float(lon[-1])], showticklabels=(r == nrows - 1),
                         row=r + 1, col=c + 1)
        fig.update_yaxes(range=[float(lat[0]), float(lat[-1])], showticklabels=(c == 0),
                         row=r + 1, col=c + 1)
    fig.update_layout(height=height_per_row * nrows + 40, margin=dict(l=10, r=10, t=40, b=10))
    return fig
