"""
build_z500_archive.py - one-time (then occasional) build of the 500 hPa archive
that analogs.py reads.

Downloads ERA5 500 hPa geopotential at 12 UTC from the Copernicus Climate Data
Store, a block of years per request, converts it to height in metres, and packs
the lot into z500_12z_daily.npz (about 30-45 MB).

You run this on your own computer, not inside the Streamlit app.

SETUP (once)
  1. Free account at https://cds.climate.copernicus.eu and accept the ERA5
     licence on the dataset page ("ERA5 hourly data on pressure levels").
  2. Put your key in ~/.cdsapirc as described on the CDS "how to API" page.
  3. pip install cdsapi xarray netCDF4 numpy

RUN
  python build_z500_archive.py                    # 1940 through last year
  python build_z500_archive.py --start 1979       # shorter record, faster
  python build_z500_archive.py --chunk 5          # smaller requests if one is refused

  Years are requested --chunk at a time (default 10), so there are only a handful
  of waits in the CDS queue. Every download is saved in ./era5_z500_cache/ and
  years already there are skipped, so an interrupted run resumes where it
  stopped. The queue decides the speed.

PUBLISH
  Put z500_12z_daily.npz next to app.py, or upload it to the cawn2 bucket as
  analogs/z500_12z_daily.npz (analogs.py checks both). The archive stops at the
  end of last year; the current year is not requested because ERA5 trails real
  time by several days and a request for dates that do not exist yet may fail.

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


def download_years(years, target):
    import cdsapi

    client = cdsapi.Client()
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "pressure_level": ["500"],
            "year": [str(y) for y in years],
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


def cached_files(cache):
    """{(first_year, last_year): path} for files named era5_z500_<y>.nc or era5_z500_<y0>_<y1>.nc."""
    found = {}
    for f in Path(cache).glob("era5_z500_*.nc"):
        parts = f.stem.split("_")[2:]
        try:
            found[(int(parts[0]), int(parts[-1]))] = f
        except (ValueError, IndexError):
            continue
    return found


def plan_chunks(missing_years, size):
    """Group sorted missing years into runs of consecutive years, at most `size` long."""
    chunks, run = [], []
    for y in sorted(missing_years):
        if run and (y != run[-1] + 1 or len(run) >= size):
            chunks.append(run)
            run = []
        run.append(y)
    if run:
        chunks.append(run)
    return chunks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1940)
    ap.add_argument("--end", type=int, default=dt.date.today().year - 1)
    ap.add_argument("--chunk", type=int, default=10, help="years per CDS request")
    ap.add_argument("--cache", default="era5_z500_cache")
    ap.add_argument("--out", default="z500_12z_daily.npz")
    args = ap.parse_args()

    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)
    wanted = range(args.start, args.end + 1)

    have = set()
    for (y0, y1) in cached_files(cache):
        have.update(range(y0, y1 + 1))
    missing = [y for y in wanted if y not in have]
    chunks = plan_chunks(missing, args.chunk)
    print(f"{len(have & set(wanted))} years cached, {len(missing)} to request in {len(chunks)} request(s).",
          flush=True)
    for run in chunks:
        f = cache / (f"era5_z500_{run[0]}.nc" if len(run) == 1 else f"era5_z500_{run[0]}_{run[-1]}.nc")
        print(f"Requesting {run[0]}-{run[-1]} from CDS (this waits in the queue) ...", flush=True)
        download_years(run, f)
        print(f"  saved {f.name}", flush=True)

    data = []
    for (y0, y1), f in sorted(cached_files(cache).items()):
        if y1 < args.start or y0 > args.end:
            continue
        data.append(read_year(f))
        print(f"read {f.name}: {len(data[-1][0])} days", flush=True)
    if not data:
        raise SystemExit("Nothing downloaded.")

    arch = pack(data)
    np.savez_compressed(args.out, **arch)
    d = arch["dates"].astype("int64").astype("datetime64[D]")
    print(f"Wrote {args.out}: {len(d)} days, {d[0]} to {d[-1]}, grid {arch['z'].shape[1]}x{arch['z'].shape[2]}")


if __name__ == "__main__":
    main()
