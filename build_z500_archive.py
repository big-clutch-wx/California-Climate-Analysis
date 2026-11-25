"""
build_z500_archive.py - one-time (then occasional) build of the 500 hPa archive
that analogs.py reads.

Downloads ERA5 500 hPa geopotential at 12 UTC, one file per year, from the
Copernicus Climate Data Store, converts it to height in metres, and packs the
lot into z500_12z_daily.npz (about 30-45 MB).

You run this on your own computer, not inside the Streamlit app.

SETUP (once)
  1. Free account at https://cds.climate.copernicus.eu and accept the ERA5
     licence on the dataset page ("ERA5 hourly data on pressure levels").
  2. Put your key in ~/.cdsapirc as described on the CDS "how to API" page.
  3. pip install cdsapi xarray netCDF4 numpy

RUN
  python build_z500_archive.py                    # 1940 to this year
  python build_z500_archive.py --start 1979       # shorter record, much faster
  python build_z500_archive.py --refresh-current  # top up the newest year later

  Each year is a separate request and is saved in ./era5_z500_cache/, so an
  interrupted run resumes where it stopped. The CDS queue decides the speed;
  expect hours for the full record.

PUBLISH
  Put z500_12z_daily.npz next to app.py, or upload it to the cawn2 bucket as
  analogs/z500_12z_daily.npz (analogs.py checks both). Re-run with
  --refresh-current now and then to extend the record. ERA5 lags real time by
  about five days.

Grid: 20-65N, 180-90W at 2.5 degrees, which covers the domains in analogs.py.
"""

import argparse
import datetime as dt
from pathlib import Path

import numpy as np

AREA = [65, -180, 20, -90]       # north, west, south, east
GRID = 2.5
G = 9.80665
MISSING_Z = -32768


def pack(years_data):
    """Combine per-year arrays into the archive dict.

    years_data: list of (dates datetime64[D] [t], lat [ny], lon [nx], z metres [t, ny, nx]).
    Latitudes and longitudes are put in ascending order; every year must share a grid.
    """
    lat0, lon0 = years_data[0][1], years_data[0][2]
    dates, zs = [], []
    for d, la, lo, z in years_data:
        if not (np.allclose(la, lat0) and np.allclose(lo, lon0)):
            raise ValueError("A year has a different grid from the first year.")
        dates.append(np.asarray(d).astype("datetime64[D]"))
        zs.append(z)
    dates = np.concatenate(dates)
    z = np.concatenate(zs, axis=0)
    order = np.argsort(dates)
    dates, z = dates[order], z[order]
    keep = np.concatenate([[True], np.diff(dates.astype("int64")) > 0])   # drop duplicate days
    dates, z = dates[keep], z[keep]
    z16 = np.where(np.isfinite(z), np.round(z), MISSING_Z).astype(np.int16)
    return {
        "dates": dates.astype("int64").astype(np.int32),
        "lat": np.asarray(lat0, dtype=np.float32),
        "lon": np.asarray(lon0, dtype=np.float32),
        "z": z16,
    }


def read_year(path):
    """Read one CDS NetCDF file: returns (dates, lat asc, lon asc in -180..180, z metres)."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        tname = "valid_time" if "valid_time" in ds.coords else "time"
        var = "z" if "z" in ds.data_vars else list(ds.data_vars)[0]
        da = ds[var]
        for extra in ("pressure_level", "level", "number", "expver"):
            if extra in da.dims:
                da = da.isel({extra: 0})
        da = da.load()
        times = da[tname].values
        lat = da["latitude"].values
        lon = da["longitude"].values
        z = da.values.astype(np.float64) / G
    lon = np.where(lon >= 180, lon - 360, lon)      # 180E and -180 are the same meridian
    ilat = np.argsort(lat)
    ilon = np.argsort(lon)
    z = z[:, ilat][:, :, ilon]
    return times.astype("datetime64[D]"), lat[ilat], lon[ilon], z


def download_year(year, target):
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "pressure_level": ["500"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": ["12:00"],
            "area": AREA,
            "grid": [GRID, GRID],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1940)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--cache", default="era5_z500_cache")
    ap.add_argument("--out", default="z500_12z_daily.npz")
    ap.add_argument("--refresh-current", action="store_true",
                    help="re-download the current year even if it is cached")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    this_year = dt.date.today().year
    data = []
    for year in range(args.start, args.end + 1):
        f = cache / f"era5_z500_{year}.nc"
        if args.refresh_current and year == this_year and f.exists():
            f.unlink()
        if not f.exists():
            print(f"{year}: requesting from CDS ...", flush=True)
            download_year(year, f)
        data.append(read_year(f))
        print(f"{year}: {len(data[-1][0])} days", flush=True)

    arch = pack(data)
    np.savez_compressed(args.out, **arch)
    d = arch["dates"].astype("int64").astype("datetime64[D]")
    print(f"Wrote {args.out}: {len(d)} days, {d[0]} to {d[-1]}, grid {arch['z'].shape[1]}x{arch['z'].shape[2]}")


if __name__ == "__main__":
    main()
