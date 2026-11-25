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
  python build_z500_archive_arco.py --diagnose    # read one day three ways, check east-west variation
  python build_z500_archive_arco.py               # build the archive
  python build_z500_archive_arco.py --start 1979  # shorter record

  Completed years are saved in ./era5_z500_arco_cache_v2/ and skipped on the next
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
    delta = values[:, None] - want[None, :]
    if circular:
        diff = np.abs((delta + 180.0) % 360.0 - 180.0)      # shortest way round the globe, always >= 0
    else:
        diff = np.abs(delta)
    idx = diff.argmin(axis=0)
    worst = diff[idx, np.arange(len(want))].max()
    if worst > 0.01:
        raise SystemExit(f"Grid does not contain the 2.5 degree points (off by up to {worst:.3f} degrees).")
    return idx


def axis_plan(coord_values, want, circular=False):
    """How to read the wanted points with plain strided slicing: (slice, flip).

    Plain slices are the most dependable way to read a Zarr array; fancy
    (list) indexing across several axes at once is avoided on purpose.
    """
    idx = nearest_index(coord_values, want, circular)
    steps = np.diff(idx)
    if len(steps) == 0 or steps[0] == 0 or not np.all(steps == steps[0]):
        raise SystemExit("The wanted grid points are not evenly spaced in the store "
                         f"(index steps {sorted(set(steps.tolist()))[:5]}).")
    s = int(steps[0])
    if s > 0:
        return slice(int(idx[0]), int(idx[-1]) + 1, s), False
    return slice(int(idx[-1]), int(idx[0]) + 1, -s), True


def check_variation(z, label):
    """Real 500 hPa fields vary both north-south and east-west. Refuse anything flat."""
    across_lon = float(np.nanmean(np.nanstd(z, axis=2)))
    across_lat = float(np.nanmean(np.nanstd(z, axis=1)))
    if across_lon < 1.0 or across_lat < 1.0:
        raise SystemExit(
            f"{label}: the fields do not vary like real 500 hPa heights (mean spread across "
            f"longitude {across_lon:.2f} m, across latitude {across_lat:.2f} m). Nothing was written. "
            "Run with --diagnose and send me the output."
        )


def diagnose(da, tdim, latd, lond, times, ds):
    """Read one day three different ways and report east-west spread, to find where data goes wrong."""
    ts = pd.DatetimeIndex(times)
    hit = np.nonzero((ts.year == 2017) & (ts.month == 2) & (ts.day == 7) & (ts.hour == 12))[0]
    if len(hit) == 0:                              # store does not cover Feb 2017: use a 12 UTC step mid-record
        twelve = np.nonzero(ts.hour == 12)[0]
        hit = twelve[len(twelve) // 2: len(twelve) // 2 + 1]
    k = int(hit[0])
    lat, lon = ds[latd].values, ds[lond].values
    la0, la1 = nearest_index(lat, np.array([20.0]))[0], nearest_index(lat, np.array([65.0]))[0]
    lo0 = nearest_index(lon, np.array([-180.0 % 360.0]), circular=True)[0]
    lo1 = nearest_index(lon, np.array([-90.0 % 360.0]), circular=True)[0]
    la_lo, la_hi = sorted((int(la0), int(la1)))
    j_lo, j_hi = sorted((int(lo0), int(lo1)))
    i45 = nearest_index(lat, np.array([45.0]))[0] - la_lo
    print(f"day {ts[k]}  lat rows {la_lo}..{la_hi}  lon cols {j_lo}..{j_hi}  (row for 45N = {i45})")

    def report(name, arr):
        arr = np.asarray(arr, dtype=float)
        row = arr[min(i45, arr.shape[0] - 1)] if arr.ndim == 2 else None
        print(f"  {name}: shape {arr.shape}, spread across longitude {np.nanstd(arr, axis=1).mean():.1f}, "
              f"spread across latitude {np.nanstd(arr, axis=0).mean():.1f}")
        if row is not None:
            print(f"    45N row, 8 evenly spaced columns: {np.round(row[::max(1, len(row)//8)][:8], 0).tolist()}")

    one = da.isel({tdim: k})
    box = one.isel({latd: slice(la_lo, la_hi + 1), lond: slice(j_lo, j_hi + 1)}).transpose(latd, lond)
    report("A  plain box, every 0.25 point", box.values)
    st = max(1, int(round(2.5 / abs(float(lat[1] - lat[0])))))
    report(f"C  plain box, every {st}th point", box.values[::st, ::st])
    il = np.arange(la_lo, la_hi + 1, st)
    jl = np.arange(j_lo, j_hi + 1, st)
    fancy = one.isel({latd: il, lond: jl}).transpose(latd, lond)
    report("B  list indexing (what the first version used)", fancy.values)
    print("If A varies with longitude but B does not, the first version's list indexing was the bug.")
    print("If A also does not vary, the store itself (or how it is opened) is the problem.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=URL)
    ap.add_argument("--key")
    ap.add_argument("--start", type=int, default=1940)
    ap.add_argument("--end", type=int, default=9999)
    ap.add_argument("--cache", default="era5_z500_arco_cache_v2")
    ap.add_argument("--out", default="z500_12z_daily.npz")
    ap.add_argument("--inspect", action="store_true", help="print the store's layout and exit")
    ap.add_argument("--diagnose", action="store_true",
                    help="read one day three ways, print the east-west spread of each, and exit")
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
    lev500 = ds[ldim].values[int(np.nonzero(levels == 500)[0][0])]
    da = ds[var].sel({ldim: lev500})
    if args.diagnose:
        diagnose(da, tdim, latd, lond, times, ds)
        return

    lat_sl, lat_flip = axis_plan(ds[latd].values, WANT_LAT)
    lon_sl, lon_flip = axis_plan(ds[lond].values, WANT_LON % 360.0, circular=True)
    mask12 = np.asarray(times.hour == 12)
    years = np.asarray(times.year)
    last_year = int(years.max())
    cache = Path(args.cache)
    cache.mkdir(exist_ok=True)

    data = []
    for year in range(max(args.start, int(years.min())), min(args.end, last_year) + 1):
        f = cache / f"z_{year}.npz"
        if year < last_year and f.exists():
            with np.load(f) as zf:
                zc, dc = zf["z"], zf["dates"].astype("datetime64[D]")
            check_variation(zc, f"{year} (cached)")
            data.append((dc, WANT_LAT, WANT_LON, zc))
            print(f"{year}: from cache", flush=True)
            continue
        it = np.nonzero(mask12 & (years == year))[0]
        if len(it) == 0:
            continue
        print(f"{year}: reading {len(it)} days ...", flush=True)
        a, b = int(it[0]), int(it[-1]) + 1
        # One contiguous block of time steps with strided lat/lon slices, then pick the
        # 12 UTC rows in memory. No list indexing anywhere.
        block = da.isel({tdim: slice(a, b), latd: lat_sl, lond: lon_sl}).transpose(tdim, latd, lond).values
        z = block[(it - a)].astype(np.float64) / B.G
        if lat_flip:
            z = z[:, ::-1]
        if lon_flip:
            z = z[:, :, ::-1]
        mean_z = float(np.nanmean(z))
        if not 4800 <= mean_z <= 5900:             # 500 hPa over 20-65N is roughly 5000-5800 m
            raise SystemExit(
                f"{year}: the mean 500 hPa height came out as {mean_z:.0f} m, which is not "
                f"plausible. The store's units are probably not m2/s2 (units attribute "
                f"{da.attrs.get('units')!r}). Nothing was written."
            )
        check_variation(z, str(year))
        dates = times[it].values.astype("datetime64[D]")
        if year < last_year:                       # only completed years are cached
            np.savez_compressed(f, dates=dates.astype("int64"), z=z)
        data.append((dates, WANT_LAT, WANT_LON, z))

    if not data:
        raise SystemExit("Nothing was read.")
    arch = B.pack(data)
    check_variation(arch["z"][:: max(1, len(arch["z"]) // 200)].astype(float), "final archive")
    np.savez_compressed(args.out, **arch)
    d = arch["dates"].astype("int64").astype("datetime64[D]")
    print(f"Wrote {args.out}: {len(d)} days, {d[0]} to {d[-1]}, grid {arch['z'].shape[1]}x{arch['z'].shape[2]}")


if __name__ == "__main__":
    main()
