"""
build_z500_archive_arco.py - build the 500 hPa archive from ECMWF's ARCO ERA5
pressure-level Zarr store instead of the Climate Data Store request queue.

Writes the same z500_12z_daily.npz as build_z500_archive.py, so use whichever
finishes first. This one reads the data directly, so there is no queue.

STATUS: written against ECMWF's published description of the store, not run
against the real thing. The first run prints what the store actually contains
(dimension names, coordinate ranges) before it downloads anything, and stops
with a clear message if something is not as expected. Use --inspect to print
that and exit.

SETUP
  1. A CDS account and API key (the same ~/.cdsapirc the other builder uses, or
     pass --key / set CDSAPI_KEY). If the store answers 401 or 403, accept the
     ERA5 licence on the dataset page in the CDS website and try again.
  2. pip install xarray zarr fsspec aiohttp requests numpy pandas truststore
  3. Keep this file in the same folder as build_z500_archive.py (it reuses it).

RUN
  python build_z500_archive_arco.py --inspect     # look at the store, download nothing
  python build_z500_archive_arco.py               # build the archive
  python build_z500_archive_arco.py --start 1979  # shorter record

  Completed years are saved in ./era5_z500_arco_cache/ and skipped on the next
  run, so it can be stopped and resumed. The newest (partial) year is always
  read again, so running it later extends the record to about five days
  behind real time.

What it keeps: 500 hPa geopotential height at 12 UTC, every 2.5 degrees over
20-65N, 180-90W, taken from the store's 0.25 degree grid (points are selected,
not averaged, so values differ slightly from the CDS 2.5 degree download).
"""

import argparse
import os
from pathlib import Path

# Check HTTPS certificates with the operating system's trust store (as browsers and
# curl do) instead of Python's own chain building. Certificate checking stays on.
# Fixes "certificate has expired" errors when Python picks an outdated chain that
# Windows does not. Must run before xarray/fsspec/aiohttp are imported.
try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import pandas as pd

import build_z500_archive as B

URL = ("https://arco.datastores.ecmwf.int/cadl-arco-geo-048/arco/"
       "reanalysis_era5_pressure_levels/pl/geoChunked.zarr")

LAT_NAMES = ("latitude", "lat")
LON_NAMES = ("longitude", "lon")
TIME_NAMES = ("time", "valid_time")
LEVEL_NAMES = ("pressureLevel", "pressureLevels", "pressure_level", "level", "isobaricInhPa", "plev")
VAR_NAMES = ("geopotential", "z")

WANT_LAT = np.arange(20.0, 65.0 + 1e-6, 2.5)
WANT_LON = np.arange(-180.0, -90.0 + 1e-6, 2.5)


def find_key(cli_key=None):
    """API key from --key, CDSAPI_KEY, or the 'key:' line of ~/.cdsapirc."""
    if cli_key:
        return cli_key
    if os.environ.get("CDSAPI_KEY"):
        return os.environ["CDSAPI_KEY"]
    rc = Path.home() / ".cdsapirc"
    if rc.exists():
        for line in rc.read_text().splitlines():
            if line.strip().startswith("key:"):
                return line.split(":", 1)[1].strip()
    raise SystemExit("No CDS API key. Put it in ~/.cdsapirc, set CDSAPI_KEY, or pass --key.")


def open_store(url, key):
    import xarray as xr

    kw = {}
    if url.startswith("http"):
        kw["storage_options"] = {"headers": {"Authorization": f"Bearer {key}"}}
    last = None
    for consolidated in (True, None):
        try:
            return xr.open_zarr(url, consolidated=consolidated, chunks=None, **kw)
        except Exception as exc:  # noqa: BLE001 - try the other metadata mode
            last = exc
    hint = ""
    if any(code in str(last) for code in ("401", "403")):
        hint = " The store refused the key: check it, and accept the ERA5 licence on the CDS website."
    raise SystemExit(f"Could not open the store: {last}.{hint}")


def pick(names, available, what):
    for n in names:
        if n in available:
            return n
    raise SystemExit(f"Cannot find the {what} in the store. Looked for {names}; it has {sorted(available)}.")


def describe(ds):
    names = set(ds.coords) | set(ds.dims)
    tdim = pick(TIME_NAMES, names, "time coordinate")
    ldim = pick(LEVEL_NAMES, names, "pressure-level coordinate")
    latd = pick(LAT_NAMES, names, "latitude")
    lond = pick(LON_NAMES, names, "longitude")
    var = pick(VAR_NAMES, set(ds.data_vars), "geopotential variable")
    t = pd.DatetimeIndex(ds[tdim].values)
    print(f"variable {var!r}, dims {dict(ds[var].sizes)}")
    print(f"time {tdim!r}: {t[0]} to {t[-1]}, {len(t)} steps, hours present {sorted(set(t.hour))}")
    print(f"levels {ldim!r}: {ds[ldim].values.tolist()}")
    la, lo = ds[latd].values, ds[lond].values
    print(f"latitude {latd!r}: {la[0]} .. {la[-1]} ({len(la)} points)")
    print(f"longitude {lond!r}: {lo[0]} .. {lo[-1]} ({len(lo)} points)")
    return tdim, ldim, latd, lond, var, t


def nearest_index(values, want, circular=False):
    values = np.asarray(values, dtype=float)
    diff = np.abs(values[:, None] - want[None, :])
    if circular:
        diff = np.minimum(diff, 360.0 - diff)
    idx = diff.argmin(axis=0)
    worst = diff[idx, np.arange(len(want))].max()
    if worst > 0.01:
        raise SystemExit(f"Grid does not contain the 2.5 degree points (off by up to {worst:.3f} degrees).")
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--key")
    ap.add_argument("--start", type=int, default=1940)
    ap.add_argument("--end", type=int, default=9999)
    ap.add_argument("--cache", default="era5_z500_arco_cache")
    ap.add_argument("--out", default="z500_12z_daily.npz")
    ap.add_argument("--inspect", action="store_true", help="print the store's layout and exit")
    args = ap.parse_args()

    key = find_key(args.key) if args.url.startswith("http") else None
    ds = open_store(args.url, key)
    tdim, ldim, latd, lond, var, times = describe(ds)
    if args.inspect:
        return
    if 12 not in set(times.hour):
        raise SystemExit("The store has no 12 UTC time steps.")

    levels = np.asarray(ds[ldim].values, dtype=float)
    if 500 not in levels:
        raise SystemExit("The store has no 500 hPa level.")
    ilat = nearest_index(ds[latd].values, WANT_LAT)
    ilon = nearest_index(ds[lond].values, WANT_LON % 360.0, circular=True)
    lev500 = ds[ldim].values[int(np.nonzero(levels == 500)[0][0])]

    da = ds[var].sel({ldim: lev500})
    mask12 = np.asarray(times.hour == 12)
    years = np.asarray(times.year)
    last_year = int(years.max())
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)

    data = []
    for year in range(max(args.start, int(years.min())), min(args.end, last_year) + 1):
        f = cache / f"z_{year}.npz"
        if year < last_year and f.exists():
            with np.load(f) as z:
                data.append((z["dates"].astype("datetime64[D]"), WANT_LAT, WANT_LON, z["z"]))
            print(f"{year}: from cache", flush=True)
            continue
        it = np.nonzero(mask12 & (years == year))[0]
        if len(it) == 0:
            continue
        print(f"{year}: reading {len(it)} days ...", flush=True)
        sub = da.isel({tdim: it, latd: ilat, lond: ilon}).transpose(tdim, latd, lond).load()
        z = sub.values.astype(np.float64) / B.G
        dates = times[it].values.astype("datetime64[D]")
        if year < last_year:                       # only completed years are cached
            np.savez_compressed(f, dates=dates.astype("int64"), z=z)
        data.append((dates, WANT_LAT, WANT_LON, z))

    if not data:
        raise SystemExit("Nothing was read.")
    arch = B.pack(data)
    np.savez_compressed(args.out, **arch)
    d = arch["dates"].astype("int64").astype("datetime64[D]")
    print(f"Wrote {args.out}: {len(d)} days, {d[0]} to {d[-1]}, grid {arch['z'].shape[1]}x{arch['z'].shape[2]}")


if __name__ == "__main__":
    main()
